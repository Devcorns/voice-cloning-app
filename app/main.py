"""
Voice Cloning Web App — Stateless, local-only, no persistence.
Uses Coqui TTS XTTS-v2 for zero-shot voice cloning.

Voice-similarity tuning (v2):
  • Minimal preprocessing — mono conversion only; NO trimming, NO normalisation
  • XTTS-v2 handles its own internal resampling to 22 050 Hz
  • Reference audio up to 30 s used via gpt_cond_len=30 & max_ref_len=30
  • Low temperature (0.15) for deterministic, voice-faithful output
  • Language selectable from frontend (critical for non-English voices)
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

# Auto-accept Coqui XTTS license (non-commercial CPML)
os.environ["COQUI_TOS_AGREED"] = "1"
from pathlib import Path
import re
from typing import Optional

import librosa
import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pydub import AudioSegment

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger("voice-clone")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB (larger samples = better cloning)
ALLOWED_CONTENT_TYPES = {"audio/wav", "audio/x-wav", "audio/wave", "audio/mpeg", "audio/mp3"}
XTTS_SR = 22050  # write pre-processed WAV at this rate; XTTS loads at 22050 internally
REF_MIN_SEC = 3  # minimum seconds of speech
REF_MAX_SEC = 30  # XTTS can condition on up to 30 s of reference
STATIC_DIR = Path(__file__).resolve().parent / "static"

# Languages supported by XTTS-v2
SUPPORTED_LANGS = {
    "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru",
    "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi",
}

# ---------------------------------------------------------------------------
# Global model holder (loaded once at startup)
# ---------------------------------------------------------------------------
_tts_model: Optional[object] = None
_synth_executor = ThreadPoolExecutor(max_workers=1)  # serialise GPU/CPU work


def _get_device() -> str:
    if torch.cuda.is_available():
        log.info("CUDA GPU detected — using GPU acceleration.")
        return "cuda"
    log.info("No CUDA GPU — falling back to CPU (slower).")
    return "cpu"


def _load_model():
    """Load XTTS-v2 once into memory."""
    global _tts_model
    from TTS.api import TTS

    device = _get_device()
    log.info("Loading XTTS-v2 model (this may take a minute on first run)…")
    _tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    log.info("Model loaded successfully on %s.", device)


# ---------------------------------------------------------------------------
# Lifespan — load model on startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield
    log.info("Shutting down.")


# ---------------------------------------------------------------------------
# Pydantic models (for API v1 JSON endpoints)
# ---------------------------------------------------------------------------
class CloneRequest(BaseModel):
    """JSON body for /api/v1/clone."""
    audio_base64: str = Field(..., description="Base64-encoded WAV or MP3 voice sample (max 10 MB decoded)")
    audio_format: str = Field("wav", description="Format of the encoded audio: 'wav' or 'mp3'")
    text: str = Field(..., min_length=1, max_length=1000, description="Text to synthesize in the cloned voice")
    language: str = Field("en", description="Language code — must match the text language")


class CloneResponse(BaseModel):
    """JSON response for /api/v1/clone."""
    audio_base64: str = Field(..., description="Base64-encoded WAV of the cloned voice")
    sample_rate: int = Field(..., description="Sample rate of the output WAV (Hz)")
    duration_sec: float = Field(..., description="Duration of the generated audio (seconds)")
    language: str = Field(..., description="Language used for synthesis")
    processing_time_sec: float = Field(..., description="Server-side processing time (seconds)")


class LanguageInfo(BaseModel):
    code: str
    name: str


class LanguagesResponse(BaseModel):
    languages: list[LanguageInfo]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    model_name: str
    max_upload_bytes: int
    max_text_length: int
    supported_languages: list[str]


class ErrorResponse(BaseModel):
    detail: str


# ---------------------------------------------------------------------------
# Language registry
# ---------------------------------------------------------------------------
_LANG_NAMES: dict[str, str] = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German",
    "it": "Italian", "pt": "Portuguese", "pl": "Polish", "tr": "Turkish",
    "ru": "Russian", "nl": "Dutch", "cs": "Czech", "ar": "Arabic",
    "zh-cn": "Chinese", "ja": "Japanese", "hu": "Hungarian", "ko": "Korean",
    "hi": "Hindi",
}

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Voice Cloning API",
    version="2.0.0",
    description=(
        "Zero-shot voice cloning powered by XTTS-v2. "
        "Supports 17 languages, runs 100 %% local — no data leaves the server.\n\n"
        "**Web UI** → `GET /`\n\n"
        "**Integration APIs** → all under `/api/v1/`\n\n"
        "Use the JSON-based `/api/v1/clone` endpoint for tool integration "
        "(accepts & returns base64-encoded audio)."
    ),
    lifespan=lifespan,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    openapi_tags=[
        {"name": "Web UI", "description": "HTML frontend"},
        {"name": "Health", "description": "Server & model status"},
        {"name": "Voice Cloning", "description": "Clone a voice from a reference sample"},
        {"name": "Languages", "description": "Supported language metadata"},
    ],
)

# --- CORS — allow any origin so external tools can call the API ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _validate_upload(file: UploadFile, raw_bytes: bytes) -> None:
    """Raise 400 on bad content-type or oversize file."""
    ct = (file.content_type or "").lower()
    if ct not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ct}'. Upload a WAV or MP3 file.",
        )
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({len(raw_bytes)} bytes). Max is {MAX_UPLOAD_BYTES} bytes (10 MB).",
        )


def _preprocess_speaker(raw_bytes: bytes, content_type: str) -> np.ndarray:
    """
    Prepare speaker reference audio for XTTS-v2 embedding extraction.

    MINIMAL preprocessing for best voice similarity:
      1. Convert to mono (required by XTTS).
      2. Resample to 22 050 Hz (matching XTTS internal load_sr).
      3. NO silence trimming — preserves natural speech rhythm/dynamics.
      4. NO normalisation — XTTS expects natural loudness.
      5. Cap at REF_MAX_SEC (30 s) — XTTS uses gpt_cond_len internally.
      6. Ensure at least REF_MIN_SEC seconds of audio.
    """
    # --- Decode with pydub (handles both WAV and MP3) ---
    fmt = "mp3" if "mp3" in content_type or "mpeg" in content_type else "wav"
    seg = AudioSegment.from_file(io.BytesIO(raw_bytes), format=fmt)

    # → mono
    seg = seg.set_channels(1)
    buf = io.BytesIO()
    seg.export(buf, format="wav")
    buf.seek(0)

    # --- Resample with librosa (mono, 22050 Hz) ---
    y, sr = librosa.load(buf, sr=XTTS_SR, mono=True)

    duration = len(y) / XTTS_SR

    # --- Cap to max reference length (XTTS handles sub-selection internally) ---
    if duration > REF_MAX_SEC:
        y = y[: int(REF_MAX_SEC * XTTS_SR)]
        duration = REF_MAX_SEC

    log.info(
        "Speaker ref: %.1fs @ %d Hz (no trim, max %ds)",
        duration, XTTS_SR, REF_MAX_SEC,
    )
    return y


# ---------------------------------------------------------------------------
# Text helpers — handle num2words limitations for unsupported languages
# ---------------------------------------------------------------------------
try:
    from num2words import num2words as _num2words
except ImportError:
    _num2words = None

# Languages where num2words is known to crash (NotImplementedError)
_NUM2WORDS_UNSUPPORTED: set[str] = set()


def _safe_expand_numbers(text: str, lang: str) -> str:
    """Replace digit sequences with spoken words *before* XTTS tokenises.

    For languages supported by num2words we do the conversion here so XTTS
    doesn't have to.  For unsupported languages we simply remove the digits
    (XTTS can't pronounce raw '24' correctly anyway) or transliterate them.
    """
    if _num2words is None:
        # num2words not installed — just strip digits
        return re.sub(r"\d+", "", text).strip()

    def _replace(m: re.Match) -> str:
        number_str = m.group(0)
        if lang in _NUM2WORDS_UNSUPPORTED:
            # Already known unsupported — spell out digit-by-digit
            return " ".join(_digit_word(d, lang) for d in number_str)
        try:
            return _num2words(int(number_str), lang=lang)
        except (NotImplementedError, OverflowError):
            # Mark language as unsupported for future calls
            _NUM2WORDS_UNSUPPORTED.add(lang)
            log.warning("num2words does not support lang '%s'; using digit spelling", lang)
            return " ".join(_digit_word(d, lang) for d in number_str)

    return re.sub(r"\d+", _replace, text)


# Digit-to-word maps for languages without num2words support
_HINDI_DIGITS = {
    "0": "शून्य", "1": "एक", "2": "दो", "3": "तीन", "4": "चार",
    "5": "पाँच", "6": "छह", "7": "सात", "8": "आठ", "9": "नौ",
}
_ARABIC_DIGITS = {
    "0": "صفر", "1": "واحد", "2": "اثنان", "3": "ثلاثة", "4": "أربعة",
    "5": "خمسة", "6": "ستة", "7": "سبعة", "8": "ثمانية", "9": "تسعة",
}


def _digit_word(digit: str, lang: str) -> str:
    """Return the spoken word for a single digit character."""
    if lang == "hi":
        return _HINDI_DIGITS.get(digit, digit)
    if lang == "ar":
        return _ARABIC_DIGITS.get(digit, digit)
    # Fallback: just keep the raw digit character
    return digit


def _write_temp_wav(audio: np.ndarray, sr: int) -> str:
    """Write numpy audio to a named temp WAV file and return its path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, sr)
    tmp.close()
    return tmp.name


def _cleanup(*paths: str) -> None:
    """Delete temp files silently."""
    for p in paths:
        try:
            Path(p).unlink(missing_ok=True)
            log.debug("Cleaned up %s", p)
        except Exception:
            log.warning("Failed to clean up %s", p, exc_info=True)


# ---------------------------------------------------------------------------
# Shared synthesis logic
# ---------------------------------------------------------------------------
async def _synthesize(
    raw_bytes: bytes,
    content_type: str,
    text: str,
    lang: str,
    background_tasks: BackgroundTasks,
) -> tuple[io.BytesIO, int, float, float]:
    """
    Core synthesis pipeline — shared by form-based /clone and JSON-based /api/v1/clone.

    Returns (wav_buffer, sample_rate, duration_sec, processing_time_sec).
    """
    t0 = time.perf_counter()

    # 1. Preprocess speaker reference
    try:
        processed = _preprocess_speaker(raw_bytes, content_type)
    except Exception as exc:
        log.error("Audio preprocessing failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=422, detail=f"Could not process audio: {exc}")

    duration = len(processed) / XTTS_SR
    if duration < REF_MIN_SEC:
        raise HTTPException(
            status_code=400,
            detail=f"Voice sample too short ({duration:.1f}s). Provide at least {REF_MIN_SEC}s of clear speech.",
        )

    # 2. CPU text-length guard
    device = str(next(_tts_model.synthesizer.tts_model.parameters()).device)
    if device == "cpu" and len(text) > 500:
        raise HTTPException(
            status_code=400,
            detail=f"Text too long for CPU mode ({len(text)} chars). "
                   f"Keep it under 500 characters, or use a CUDA GPU for longer text.",
        )

    # 3. Pre-expand numbers
    text = _safe_expand_numbers(text, lang)
    log.info("Text after number expansion (%d chars): %s", len(text), text[:120])

    # 4. Write speaker audio to temp file (XTTS needs a path)
    speaker_path = _write_temp_wav(processed, XTTS_SR)
    output_path: Optional[str] = None

    def _run_synthesis(speaker: str, out: str, txt: str, lng: str) -> None:
        _tts_model.tts_to_file(
            text=txt,
            speaker_wav=speaker,
            language=lng,
            file_path=out,
            gpt_cond_len=30,
            gpt_cond_chunk_len=6,
            max_ref_len=30,
            temperature=0.15,
            repetition_penalty=5.0,
            top_p=0.8,
            top_k=50,
            length_penalty=1.0,
        )

    try:
        log.info("Starting voice cloning synthesis (lang=%s)…", lang)
        output_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            _synth_executor, _run_synthesis, speaker_path, output_path, text, lang
        )
        log.info("Synthesis complete → %s", output_path)

        gen_audio, gen_sr = sf.read(output_path)
        out_buf = io.BytesIO()
        sf.write(out_buf, gen_audio, gen_sr, format="WAV")
        out_buf.seek(0)

        gen_duration = len(gen_audio) / gen_sr
        elapsed = time.perf_counter() - t0

        return out_buf, gen_sr, gen_duration, elapsed

    except HTTPException:
        raise
    except Exception as exc:
        log.error("Synthesis failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Voice synthesis failed: {exc}")
    finally:
        paths_to_clean = [speaker_path]
        if output_path:
            paths_to_clean.append(output_path)
        background_tasks.add_task(_cleanup, *paths_to_clean)


# ---------------------------------------------------------------------------
# Web UI routes
# ---------------------------------------------------------------------------
@app.get("/", tags=["Web UI"], include_in_schema=False)
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/clone", tags=["Web UI"], include_in_schema=False)
async def clone_voice(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="WAV or MP3 voice sample (max 10 MB)"),
    text: str = Form(..., min_length=1, max_length=1000, description="Text to synthesize"),
    language: str = Form("en", description="Language code (e.g. en, hi, es, fr, de)"),
):
    """
    Accept a voice sample + text + language, clone the voice, return generated audio.
    No files are persisted after the response is sent.
    """

    # Validate
    lang = language.strip().lower()
    if lang not in SUPPORTED_LANGS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language '{lang}'. Supported: {', '.join(sorted(SUPPORTED_LANGS))}",
        )
    raw_bytes = await file.read()
    _validate_upload(file, raw_bytes)
    log.info("[UI] upload=%s (%d B), text=%d chars, lang=%s", file.filename, len(raw_bytes), len(text), lang)

    out_buf, gen_sr, gen_dur, elapsed = await _synthesize(
        raw_bytes, file.content_type or "", text, lang, background_tasks,
    )
    log.info("[UI] Done in %.1fs — %.1fs audio @ %d Hz", elapsed, gen_dur, gen_sr)

    return StreamingResponse(
        out_buf,
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="cloned_voice.wav"'},
    )


# ===========================================================================
#  API v1 — JSON endpoints for external tool integration
# ===========================================================================

@app.get(
    "/api/v1/health",
    tags=["Health"],
    response_model=HealthResponse,
    summary="Server & model status",
)
async def api_health():
    """Returns model status, device, and server capabilities."""
    device = (
        str(next(_tts_model.synthesizer.tts_model.parameters()).device)
        if _tts_model
        else "n/a"
    )
    return HealthResponse(
        status="ok" if _tts_model else "loading",
        model_loaded=_tts_model is not None,
        device=device,
        model_name="tts_models/multilingual/multi-dataset/xtts_v2",
        max_upload_bytes=MAX_UPLOAD_BYTES,
        max_text_length=1000,
        supported_languages=sorted(SUPPORTED_LANGS),
    )


@app.get(
    "/api/v1/languages",
    tags=["Languages"],
    response_model=LanguagesResponse,
    summary="List supported languages",
)
async def api_languages():
    """Returns every language code and human-readable name supported by XTTS-v2."""
    langs = [
        LanguageInfo(code=code, name=_LANG_NAMES.get(code, code))
        for code in sorted(SUPPORTED_LANGS)
    ]
    return LanguagesResponse(languages=langs)


@app.post(
    "/api/v1/clone",
    tags=["Voice Cloning"],
    response_model=CloneResponse,
    summary="Clone a voice (JSON / base64)",
    responses={
        200: {"description": "Cloned audio returned as base64 WAV"},
        400: {"model": ErrorResponse, "description": "Validation error"},
        422: {"model": ErrorResponse, "description": "Audio processing error"},
        500: {"model": ErrorResponse, "description": "Synthesis failure"},
    },
)
async def api_clone(body: CloneRequest, background_tasks: BackgroundTasks):
    """
    **JSON-based voice cloning** — designed for tool / service integration.

    Send the reference audio as a base64-encoded string and get the cloned
    audio back as a base64 WAV.

    ### Example request
    ```json
    {
      "audio_base64": "<base64 WAV or MP3>",
      "audio_format": "wav",
      "text": "Hello, this is a cloned voice.",
      "language": "en"
    }
    ```
    """
    # 1. Validate language
    lang = body.language.strip().lower()
    if lang not in SUPPORTED_LANGS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language '{lang}'. Supported: {', '.join(sorted(SUPPORTED_LANGS))}",
        )

    # 2. Decode base64 audio
    try:
        raw_bytes = base64.b64decode(body.audio_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 in 'audio_base64'.")

    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"Decoded audio too large ({len(raw_bytes)} bytes). Max is {MAX_UPLOAD_BYTES} bytes.",
        )

    # Map format to content-type for internal processing
    fmt = body.audio_format.strip().lower()
    ct_map = {"wav": "audio/wav", "mp3": "audio/mpeg"}
    content_type = ct_map.get(fmt)
    if not content_type:
        raise HTTPException(status_code=400, detail=f"Unsupported audio_format '{fmt}'. Use 'wav' or 'mp3'.")

    log.info("[API] base64 audio (%d B decoded), text=%d chars, lang=%s", len(raw_bytes), len(body.text), lang)

    # 3. Synthesize
    out_buf, gen_sr, gen_dur, elapsed = await _synthesize(
        raw_bytes, content_type, body.text, lang, background_tasks,
    )

    # 4. Encode result as base64
    audio_b64 = base64.b64encode(out_buf.read()).decode("ascii")
    log.info("[API] Done in %.1fs — %.1fs audio @ %d Hz", elapsed, gen_dur, gen_sr)

    return CloneResponse(
        audio_base64=audio_b64,
        sample_rate=gen_sr,
        duration_sec=round(gen_dur, 3),
        language=lang,
        processing_time_sec=round(elapsed, 2),
    )


@app.post(
    "/api/v1/clone/file",
    tags=["Voice Cloning"],
    summary="Clone a voice (multipart form → WAV file)",
    responses={
        200: {"content": {"audio/wav": {}}, "description": "Cloned WAV audio"},
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def api_clone_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="WAV or MP3 voice sample (max 10 MB)"),
    text: str = Form(..., min_length=1, max_length=1000, description="Text to synthesize"),
    language: str = Form("en", description="Language code"),
):
    """
    **Form-based voice cloning** — upload a file, get a WAV stream back.

    Same as the web-UI `/clone` but with proper API tags and documented
    error models. Useful for tools that prefer multipart uploads over base64.
    """
    lang = language.strip().lower()
    if lang not in SUPPORTED_LANGS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language '{lang}'. Supported: {', '.join(sorted(SUPPORTED_LANGS))}",
        )
    raw_bytes = await file.read()
    _validate_upload(file, raw_bytes)
    log.info("[API/file] upload=%s (%d B), text=%d chars, lang=%s", file.filename, len(raw_bytes), len(text), lang)

    out_buf, gen_sr, gen_dur, elapsed = await _synthesize(
        raw_bytes, file.content_type or "", text, lang, background_tasks,
    )
    log.info("[API/file] Done in %.1fs — %.1fs audio @ %d Hz", elapsed, gen_dur, gen_sr)

    return StreamingResponse(
        out_buf,
        media_type="audio/wav",
        headers={
            "Content-Disposition": 'attachment; filename="cloned_voice.wav"',
            "X-Processing-Time": str(round(elapsed, 2)),
            "X-Audio-Duration": str(round(gen_dur, 3)),
            "X-Sample-Rate": str(gen_sr),
        },
    )


# Legacy /health kept for backward compatibility
@app.get("/health", tags=["Health"], include_in_schema=False)
async def health_legacy():
    return await api_health()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
