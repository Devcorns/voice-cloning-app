"""
Lip-Sync Video Generator — Stateless, local-only, no persistence.
Uses Wav2Lip (Prajwal et al.) for audio-driven lip synchronisation.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import cv2
import librosa
import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from wav2lip import audio as wav2lip_audio
from wav2lip.inference import (
    get_device,
    load_model,
    extract_frames_from_video,
    image_to_frames,
    run_inference,
    write_video,
)
from wav2lip.model import Wav2Lip

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger("lip-sync")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
STATIC_DIR = Path(__file__).resolve().parent / "static"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
CHECKPOINT = MODELS_DIR / "wav2lip_gan.pth"

MAX_VIDEO_BYTES = 10 * 1024 * 1024   # 10 MB
MAX_IMAGE_BYTES = 5 * 1024 * 1024    # 5 MB
MAX_AUDIO_BYTES = 5 * 1024 * 1024    # 5 MB

IMAGE_TYPES = {"image/jpeg", "image/png", "image/jpg"}
VIDEO_TYPES = {
    "video/mp4", "video/avi", "video/quicktime",
    "video/x-msvideo", "video/webm",
}
AUDIO_TYPES = {"audio/wav", "audio/x-wav", "audio/wave", "audio/mpeg", "audio/mp3"}
FACE_TYPES = IMAGE_TYPES | VIDEO_TYPES

FPS = 25.0

# ─────────────────────────────────────────────────────────────────────────────
# Global model
# ─────────────────────────────────────────────────────────────────────────────
_model: Optional[Wav2Lip] = None
_device: Optional[torch.device] = None
_executor = ThreadPoolExecutor(max_workers=1)


def _load_wav2lip():
    global _model, _device
    if not CHECKPOINT.exists():
        log.error(
            "Wav2Lip checkpoint NOT FOUND at %s.  "
            "Run: python scripts/download_models.py  "
            "or download wav2lip_gan.pth manually.",
            CHECKPOINT,
        )
        raise FileNotFoundError(
            f"Wav2Lip model not found at {CHECKPOINT}. "
            "See README.md for download instructions."
        )
    _device = get_device()
    _model = load_model(str(CHECKPOINT), _device)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_wav2lip()
    yield
    log.info("Shutting down.")


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    checkpoint: str
    max_video_mb: int = 10
    max_image_mb: int = 5
    max_audio_mb: int = 5


class ErrorResponse(BaseModel):
    detail: str


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Lip-Sync Video API",
    version="1.0.0",
    description=(
        "Generate lip-synced videos from a face image/video + audio file.  "
        "Powered by Wav2Lip — runs 100 % local.\n\n"
        "**Web UI** → `GET /`\n\n"
        "**API**    → `POST /api/v1/generate`"
    ),
    lifespan=lifespan,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _cleanup(*paths: str):
    for p in paths:
        try:
            Path(p).unlink(missing_ok=True)
        except Exception:
            pass


def _is_image(ct: str) -> bool:
    return ct in IMAGE_TYPES


def _save_upload(data: bytes, suffix: str) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(data)
    tmp.close()
    return tmp.name


def _convert_audio_16k(audio_path: str) -> str:
    """Convert any audio file to 16 kHz mono WAV (required by Wav2Lip)."""
    wav, _ = librosa.load(audio_path, sr=wav2lip_audio.SAMPLE_RATE, mono=True)
    out_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    sf.write(out_path, wav, wav2lip_audio.SAMPLE_RATE)
    return out_path


def _run_pipeline(face_path: str, audio_path: str, is_img: bool) -> str:
    """Synchronous pipeline: face + audio → lip-synced MP4.  Returns output path."""
    wav_path = _convert_audio_16k(audio_path)

    try:
        # Load audio & mel
        wav, _ = librosa.load(wav_path, sr=wav2lip_audio.SAMPLE_RATE, mono=True)
        mel = wav2lip_audio.melspectrogram(wav)
        duration = len(wav) / wav2lip_audio.SAMPLE_RATE

        # Get face frames
        if is_img:
            frames = image_to_frames(face_path, duration, FPS)
            fps = FPS
        else:
            frames, fps = extract_frames_from_video(face_path)
            # Loop video if shorter than audio
            target = int(duration * fps)
            if len(frames) < target:
                reps = (target // len(frames)) + 1
                frames = (frames * reps)[:target]
                log.info("Looped video frames to %d to match audio.", target)

        # Run Wav2Lip
        output_frames = run_inference(
            _model, _device, frames, mel, fps, is_static=is_img,
        )

        # Write MP4
        output_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        write_video(output_frames, wav_path, output_path, fps)
        return output_path

    finally:
        Path(wav_path).unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Routes — Web UI
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(STATIC_DIR / "index.html")


# ─────────────────────────────────────────────────────────────────────────────
# Routes — API v1
# ─────────────────────────────────────────────────────────────────────────────
@app.post(
    "/api/v1/generate",
    tags=["Lip Sync"],
    summary="Generate a lip-synced video",
    responses={
        200: {"content": {"video/mp4": {}}, "description": "Lip-synced MP4 video"},
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def generate_video(
    background_tasks: BackgroundTasks,
    face: UploadFile = File(
        ..., description="Face image (JPG/PNG, ≤ 5 MB) or video (MP4, ≤ 10 MB)"
    ),
    audio: UploadFile = File(
        ..., description="Voice audio file (WAV/MP3, ≤ 5 MB)"
    ),
):
    """
    Upload a face image or short video **+ an audio file**.
    The server runs Wav2Lip inference and returns a lip-synced MP4.

    - For **images**: the face is replicated to match audio duration at 25 fps.
    - For **videos**: the video is looped if shorter than the audio.
    - All temp files are deleted after the response.
    """
    t0 = time.perf_counter()

    # ── Validate face ────────────────────────────────────────────────────
    face_ct = (face.content_type or "").lower()
    if face_ct not in FACE_TYPES:
        raise HTTPException(
            400,
            f"Unsupported face file type '{face_ct}'. "
            f"Accepted: {', '.join(sorted(FACE_TYPES))}",
        )
    face_bytes = await face.read()
    is_img = _is_image(face_ct)
    max_face = MAX_IMAGE_BYTES if is_img else MAX_VIDEO_BYTES
    if len(face_bytes) > max_face:
        raise HTTPException(
            400,
            f"Face file too large ({len(face_bytes)} B). "
            f"Max: {max_face // (1024*1024)} MB.",
        )

    # ── Validate audio ───────────────────────────────────────────────────
    audio_ct = (audio.content_type or "").lower()
    if audio_ct not in AUDIO_TYPES:
        raise HTTPException(
            400,
            f"Unsupported audio type '{audio_ct}'. "
            f"Accepted: {', '.join(sorted(AUDIO_TYPES))}",
        )
    audio_bytes = await audio.read()
    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise HTTPException(
            400,
            f"Audio too large ({len(audio_bytes)} B). Max: {MAX_AUDIO_BYTES // (1024*1024)} MB.",
        )

    # ── Save to temp files ───────────────────────────────────────────────
    face_suffix = ".png" if is_img else ".mp4"
    face_path = _save_upload(face_bytes, face_suffix)

    audio_suffix = ".mp3" if ("mp3" in audio_ct or "mpeg" in audio_ct) else ".wav"
    audio_path = _save_upload(audio_bytes, audio_suffix)

    output_path: Optional[str] = None

    try:
        loop = asyncio.get_running_loop()
        output_path = await loop.run_in_executor(
            _executor, _run_pipeline, face_path, audio_path, is_img,
        )
        elapsed = time.perf_counter() - t0
        log.info("Video generated in %.1fs → %s", elapsed, output_path)

        # Read into memory buffer so temp file can be cleaned up
        with open(output_path, "rb") as fh:
            video_data = fh.read()

        return StreamingResponse(
            io.BytesIO(video_data),
            media_type="video/mp4",
            headers={
                "Content-Disposition": 'attachment; filename="lip_synced.mp4"',
                "X-Processing-Time": str(round(elapsed, 2)),
            },
        )

    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(500, str(exc))
    except RuntimeError as exc:
        log.error("Pipeline error: %s", exc, exc_info=True)
        raise HTTPException(500, str(exc))
    except Exception as exc:
        log.error("Unexpected error: %s", exc, exc_info=True)
        raise HTTPException(500, f"Video generation failed: {exc}")
    finally:
        paths = [face_path, audio_path]
        if output_path:
            paths.append(output_path)
        background_tasks.add_task(_cleanup, *paths)


# ── /generate form endpoint (same as API, for web UI) ────────────────────────
@app.post("/generate", include_in_schema=False)
async def generate_video_form(
    background_tasks: BackgroundTasks,
    face: UploadFile = File(...),
    audio: UploadFile = File(...),
):
    return await generate_video(background_tasks, face, audio)


# ─────────────────────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/v1/health", tags=["Health"], response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok" if _model else "model_not_loaded",
        model_loaded=_model is not None,
        device=str(_device) if _device else "n/a",
        checkpoint=str(CHECKPOINT),
    )


@app.get("/health", include_in_schema=False)
async def health_legacy():
    return await health()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=False, log_level="info")
