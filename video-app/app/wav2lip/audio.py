"""
Audio preprocessing — mel-spectrogram extraction matching Wav2Lip training params.

These hyper-parameters MUST match the values used during Wav2Lip training.
Changing any of them will produce garbage output.
"""

from __future__ import annotations

import numpy as np
import librosa

# ── Wav2Lip audio hyper-parameters ───────────────────────────────────────────
SAMPLE_RATE = 16_000          # Hz
N_FFT = 800
HOP_SIZE = 200                # 12.5 ms
WIN_SIZE = 800                # 50 ms
NUM_MELS = 80
FMIN = 55
FMAX = 7600
REF_LEVEL_DB = 20
MIN_LEVEL_DB = -100
MAX_ABS_VALUE = 4.0
PREEMPHASIS = 0.97

# Inference constants
MEL_STEP_SIZE = 16            # mel columns per video-frame group
FPS = 25                      # assumed video fps

# ── Internal state ───────────────────────────────────────────────────────────
_mel_basis: np.ndarray | None = None


def _build_mel_basis() -> np.ndarray:
    return librosa.filters.mel(
        sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=NUM_MELS, fmin=FMIN, fmax=FMAX,
    )


def _preemphasis(wav: np.ndarray) -> np.ndarray:
    return np.append(wav[0], wav[1:] - PREEMPHASIS * wav[:-1])


def _amp_to_db(x: np.ndarray) -> np.ndarray:
    min_level = np.exp(MIN_LEVEL_DB / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))


def _normalize(S: np.ndarray) -> np.ndarray:
    return np.clip(
        (2 * MAX_ABS_VALUE) * ((S - MIN_LEVEL_DB) / (-MIN_LEVEL_DB)) - MAX_ABS_VALUE,
        -MAX_ABS_VALUE,
        MAX_ABS_VALUE,
    )


# ── Public API ───────────────────────────────────────────────────────────────

def melspectrogram(wav: np.ndarray) -> np.ndarray:
    """Compute mel spectrogram matching Wav2Lip training parameters.

    Args:
        wav: Audio waveform at 16 kHz, float32.

    Returns:
        Mel spectrogram of shape ``(80, T)`` normalised to ``[−4, 4]``.
    """
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()

    D = librosa.stft(
        y=_preemphasis(wav),
        n_fft=N_FFT,
        hop_length=HOP_SIZE,
        win_length=WIN_SIZE,
    )
    S = _amp_to_db(np.dot(_mel_basis, np.abs(D))) - REF_LEVEL_DB
    return _normalize(S)


def get_mel_chunks(mel: np.ndarray, num_frames: int) -> list[np.ndarray]:
    """Split mel spectrogram into chunks aligned with video frames at 25 fps.

    Each chunk is ``(80, 16)`` — one per video frame.  The mapping is::

        start_col = int(80 * frame_idx / fps)

    which gives ~3.2 mel columns per video frame.  If the audio is shorter
    than the video, the last valid window is repeated.
    """
    chunks: list[np.ndarray] = []
    for i in range(num_frames):
        start = int(80.0 * (i / float(FPS)))
        if start + MEL_STEP_SIZE > mel.shape[1]:
            chunks.append(mel[:, mel.shape[1] - MEL_STEP_SIZE :])
        else:
            chunks.append(mel[:, start : start + MEL_STEP_SIZE])
    return chunks
