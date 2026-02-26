"""
Wav2Lip inference pipeline
──────────────────────────
Face detection → frame extraction → mel alignment → model inference → video write + audio merge.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import librosa
import numpy as np
import soundfile as sf
import torch

from .model import Wav2Lip
from . import audio as wav2lip_audio

log = logging.getLogger("wav2lip-infer")

# ── Constants ────────────────────────────────────────────────────────────────
IMG_SIZE = 96                 # Wav2Lip face crop resolution
WAV2LIP_BATCH = 128          # frames per GPU batch — reduce if OOM
FACE_PAD = [0, 10, 0, 0]     # top, bottom, left, right padding (px)


# ── Device helpers ───────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        log.info("CUDA GPU detected → using GPU acceleration.")
        return torch.device("cuda")
    log.info("No CUDA GPU → falling back to CPU (slower).")
    return torch.device("cpu")


# ── Model loading ────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device) -> Wav2Lip:
    """Load Wav2Lip generator weights from a ``.pth`` checkpoint."""
    model = Wav2Lip()
    log.info("Loading Wav2Lip checkpoint from %s …", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    state_dict = checkpoint
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

    # Strip 'module.' prefix added by DataParallel
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    log.info("Wav2Lip model loaded on %s.", device)
    return model


# ── Face detection (OpenCV Haar cascade — zero extra downloads) ──────────────

_cascade = None


def _get_cascade():
    global _cascade
    if _cascade is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _cascade = cv2.CascadeClassifier(cascade_path)
    return _cascade


def detect_face(image: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return ``(x1, y1, x2, y2)`` for the largest face, or *None*."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = _get_cascade()
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        # retry with relaxed parameters
        faces = cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20))
    if len(faces) == 0:
        return None
    areas = [w * h for (x, y, w, h) in faces]
    idx = int(np.argmax(areas))
    x, y, w, h = faces[idx]
    return (x, y, x + w, y + h)


def detect_faces_in_frames(
    frames: list[np.ndarray],
    *,
    is_static: bool = False,
) -> list[tuple[int, int, int, int]]:
    """Detect face in every frame.  For static images, runs detection once."""
    if is_static:
        rect = detect_face(frames[0])
        if rect is None:
            raise RuntimeError(
                "No face detected in the image. "
                "Use a clear, front-facing photo with good lighting."
            )
        return [rect] * len(frames)

    rects: list[tuple[int, int, int, int]] = []
    last_rect = None
    missed = 0
    for frame in frames:
        rect = detect_face(frame)
        if rect is not None:
            last_rect = rect
        else:
            missed += 1
        if last_rect is None:
            raise RuntimeError("No face detected in the first frame of the video.")
        rects.append(last_rect)

    if missed:
        log.warning("Face detection missed %d / %d frames (used fallback).", missed, len(frames))
    return rects


# ── Frame extraction ─────────────────────────────────────────────────────────

def extract_frames_from_video(video_path: str) -> tuple[list[np.ndarray], float]:
    """Read all frames from a video file.  Returns ``(frames, fps)``."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames: list[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError("Video contains no frames.")
    log.info("Extracted %d frames at %.1f fps.", len(frames), fps)
    return frames, fps


def image_to_frames(
    image_path: str,
    audio_duration_sec: float,
    fps: float = 25.0,
) -> list[np.ndarray]:
    """Replicate a static image into *N* frames to match audio duration."""
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Cannot read image: {image_path}")
    num_frames = max(1, int(audio_duration_sec * fps))
    log.info("Replicating image → %d frames at %.0f fps (%.1fs audio).", num_frames, fps, audio_duration_sec)
    return [img.copy() for _ in range(num_frames)]


# ── Core inference ───────────────────────────────────────────────────────────

def _pad_and_crop(
    frame: np.ndarray,
    rect: tuple[int, int, int, int],
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop face region with padding.  Returns ``(crop, (y1, y2, x1, x2))``."""
    x1, y1, x2, y2 = rect
    h, w = frame.shape[:2]
    py1 = max(0, y1 - FACE_PAD[0])
    py2 = min(h, y2 + FACE_PAD[1])
    px1 = max(0, x1 - FACE_PAD[2])
    px2 = min(w, x2 + FACE_PAD[3])
    return frame[py1:py2, px1:px2], (py1, py2, px1, px2)


def run_inference(
    model: Wav2Lip,
    device: torch.device,
    frames: list[np.ndarray],
    mel: np.ndarray,
    fps: float = 25.0,
    *,
    is_static: bool = False,
) -> list[np.ndarray]:
    """Run Wav2Lip lip-sync on *frames* with *mel* spectrogram.

    Returns a new list of frames with lip-synced faces pasted back in.
    """
    # 1. Detect faces
    log.info("Detecting faces in %d frames …", len(frames))
    rects = detect_faces_in_frames(frames, is_static=is_static)

    # 2. Create mel chunks (one per frame)
    mel_chunks = wav2lip_audio.get_mel_chunks(mel, len(frames))

    # Trim to shortest
    n = min(len(frames), len(mel_chunks))
    frames = frames[:n]
    rects = rects[:n]
    mel_chunks = mel_chunks[:n]

    log.info("Running Wav2Lip inference on %d frames (batch=%d) …", n, WAV2LIP_BATCH)
    output_frames: list[np.ndarray] = [f.copy() for f in frames]

    for batch_start in range(0, n, WAV2LIP_BATCH):
        batch_end = min(batch_start + WAV2LIP_BATCH, n)

        face_crops: list[np.ndarray] = []
        crop_coords: list[tuple[int, int, int, int]] = []

        for i in range(batch_start, batch_end):
            crop, coords = _pad_and_crop(frames[i], rects[i])
            face_crops.append(cv2.resize(crop, (IMG_SIZE, IMG_SIZE)))
            crop_coords.append(coords)

        # Build input tensors
        img_batch: list[np.ndarray] = []
        mel_batch: list[np.ndarray] = []

        for idx, (face, mel_chunk) in enumerate(
            zip(face_crops, mel_chunks[batch_start:batch_end])
        ):
            # Mask lower half of face (Wav2Lip convention)
            face_masked = face.copy()
            face_masked[IMG_SIZE // 2 :] = 0

            # 6-channel input: [masked_face | original_face]
            img_input = np.concatenate((face_masked, face_crops[idx]), axis=2) / 255.0
            img_batch.append(img_input)
            mel_batch.append(mel_chunk)  # (80, 16)

        img_np = np.array(img_batch, dtype=np.float32)   # (B, 96, 96, 6)
        mel_np = np.array(mel_batch, dtype=np.float32)    # (B, 80, 16)

        # (B, H, W, 6) → (B, 6, H, W)
        img_tensor = torch.from_numpy(img_np).permute(0, 3, 1, 2).to(device)
        # (B, 80, 16) → (B, 1, 80, 16)
        mel_tensor = torch.from_numpy(mel_np).unsqueeze(1).to(device)

        with torch.no_grad():
            pred = model(mel_tensor, img_tensor)           # (B, 3, 96, 96)

        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0  # → (B, H, W, 3)

        # Paste predicted face back into each output frame
        for j in range(pred.shape[0]):
            frame_idx = batch_start + j
            py1, py2, px1, px2 = crop_coords[j]
            face_out = pred[j].astype(np.uint8)
            face_out = cv2.resize(face_out, (px2 - px1, py2 - py1))
            output_frames[frame_idx][py1:py2, px1:px2] = face_out

    log.info("Wav2Lip inference complete.")
    return output_frames


# ── FFmpeg locator ────────────────────────────────────────────────────────────

def _find_ffmpeg() -> str:
    """Return the absolute path to ffmpeg, searching PATH + common Windows locations."""
    # 1. shutil.which checks PATH
    found = shutil.which("ffmpeg")
    if found:
        return found
    # 2. Common Windows install paths (winget, choco, manual)
    candidates = [
        Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Links" / "ffmpeg.exe",
        Path("C:/ProgramData/chocolatey/bin/ffmpeg.exe"),
        Path("C:/ffmpeg/bin/ffmpeg.exe"),
        Path("C:/tools/ffmpeg/bin/ffmpeg.exe"),
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        "ffmpeg not found on PATH or common locations. "
        "Install it: winget install Gyan.FFmpeg  (then restart terminal)"
    )


# ── Video writing + audio merge ──────────────────────────────────────────────

def write_video(
    frames: list[np.ndarray],
    audio_path: str,
    output_path: str,
    fps: float = 25.0,
) -> None:
    """Write *frames* as a silent video, then merge with *audio_path* via ffmpeg."""
    h, w = frames[0].shape[:2]

    tmp_video = tempfile.NamedTemporaryFile(suffix=".avi", delete=False).name
    try:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(tmp_video, fourcc, fps, (w, h))
        for f in frames:
            writer.write(f)
        writer.release()
        log.info("Wrote %d frames to temp AVI.", len(frames))

        ffmpeg_bin = _find_ffmpeg()
        cmd = [
            ffmpeg_bin, "-y",
            "-i", tmp_video,
            "-i", audio_path,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            "-shortest",
            "-movflags", "+faststart",
            output_path,
        ]
        log.info("Merging audio + video with ffmpeg …")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg error: {result.stderr[-500:]}")
        log.info("Final MP4 → %s", output_path)
    finally:
        Path(tmp_video).unlink(missing_ok=True)
