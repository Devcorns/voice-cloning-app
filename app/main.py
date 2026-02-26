"""
Voice Cloning Web App — Stateless, local-only, no persistence.
Uses Coqui TTS XTTS-v2 for zero-shot voice cloning.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

# Auto-accept Coqui XTTS license (non-commercial CPML)
os.environ["COQUI_TOS_AGREED"] = "1"
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
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
MAX_UPLOAD_BYTES = 2 * 1024 * 1024  # 2 MB
ALLOWED_CONTENT_TYPES = {"audio/wav", "audio/x-wav", "audio/wave", "audio/mpeg", "audio/mp3"}
TARGET_SR = 22050
STATIC_DIR = Path(__file__).resolve().parent / "static"

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
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Voice Cloning API",
    version="1.0.0",
    lifespan=lifespan,
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
            detail=f"File too large ({len(raw_bytes)} bytes). Max is {MAX_UPLOAD_BYTES} bytes (2 MB).",
        )


def _preprocess_audio(raw_bytes: bytes, content_type: str) -> np.ndarray:
    """
    Convert uploaded audio to 22 050 Hz mono float32 numpy array.
    Steps: decode → mono → resample → normalize → trim silence.
    """
    # --- Decode with pydub (handles both WAV and MP3) ---
    fmt = "mp3" if "mp3" in content_type or "mpeg" in content_type else "wav"
    seg = AudioSegment.from_file(io.BytesIO(raw_bytes), format=fmt)

    # → mono, export as raw PCM in a WAV container so librosa can read it
    seg = seg.set_channels(1)
    buf = io.BytesIO()
    seg.export(buf, format="wav")
    buf.seek(0)

    # --- Load with librosa for resampling + analysis ---
    y, sr = librosa.load(buf, sr=TARGET_SR, mono=True)

    # --- Peak-normalize ---
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak

    # --- Trim silence ---
    y_trimmed, _ = librosa.effects.trim(y, top_db=25)

    log.info(
        "Preprocessed audio: %.2fs → %.2fs @ %d Hz",
        len(y) / TARGET_SR,
        len(y_trimmed) / TARGET_SR,
        TARGET_SR,
    )
    return y_trimmed


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
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/clone")
async def clone_voice(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="WAV or MP3 voice sample (max 2 MB)"),
    text: str = Form(..., min_length=1, max_length=1000, description="Text to synthesize"),
):
    """
    Accept a voice sample + text, clone the voice, return generated audio.
    No files are persisted after the response is sent.
    """

    # 1. Read & validate -------------------------------------------------------
    raw_bytes = await file.read()
    _validate_upload(file, raw_bytes)
    log.info("Received upload: %s (%d bytes), text length: %d", file.filename, len(raw_bytes), len(text))

    # 2. Preprocess ------------------------------------------------------------
    try:
        processed = _preprocess_audio(raw_bytes, file.content_type or "")
    except Exception as exc:
        log.error("Audio preprocessing failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=422, detail=f"Could not process audio: {exc}")

    # Minimum duration check (~1 second)
    if len(processed) / TARGET_SR < 1.0:
        raise HTTPException(status_code=400, detail="Voice sample too short. Provide at least 1 second of speech.")

    # Limit text length on CPU to avoid multi-minute hangs
    device = str(next(_tts_model.synthesizer.tts_model.parameters()).device)
    if device == "cpu" and len(text) > 300:
        raise HTTPException(
            status_code=400,
            detail=f"Text too long for CPU mode ({len(text)} chars). "
                   f"Keep it under 300 characters, or use a CUDA GPU for longer text.",
        )

    # 3. Write preprocessed speaker audio to a temp file (XTTS needs a path) ---
    speaker_path = _write_temp_wav(processed, TARGET_SR)
    output_path: Optional[str] = None

    def _run_synthesis(speaker: str, out: str, txt: str) -> None:
        """Run the blocking TTS call (called from thread pool)."""
        _tts_model.tts_to_file(
            text=txt,
            speaker_wav=speaker,
            language="en",
            file_path=out,
        )

    try:
        # 4. Synthesize with XTTS-v2 (in thread pool to avoid blocking) -------
        log.info("Starting voice cloning synthesis…")
        output_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            _synth_executor, _run_synthesis, speaker_path, output_path, text
        )
        log.info("Synthesis complete → %s", output_path)

        # 5. Read generated audio into memory ----------------------------------
        gen_audio, gen_sr = sf.read(output_path)
        out_buf = io.BytesIO()
        sf.write(out_buf, gen_audio, gen_sr, format="WAV")
        out_buf.seek(0)

    except HTTPException:
        raise
    except Exception as exc:
        log.error("Synthesis failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Voice synthesis failed: {exc}")
    finally:
        # 6. Schedule cleanup of ALL temp files --------------------------------
        paths_to_clean = [speaker_path]
        if output_path:
            paths_to_clean.append(output_path)
        background_tasks.add_task(_cleanup, *paths_to_clean)

    # 7. Stream the WAV back ---------------------------------------------------
    return StreamingResponse(
        out_buf,
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="cloned_voice.wav"'},
    )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": _tts_model is not None,
        "device": str(next(_tts_model.synthesizer.tts_model.parameters()).device) if _tts_model else "n/a",
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
