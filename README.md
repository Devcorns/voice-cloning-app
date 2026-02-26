# Voice Cloning Web App — XTTS v2

Minimal, stateless, local-only voice cloning powered by [Coqui TTS XTTS-v2](https://github.com/coqui-ai/TTS).

---

## Folder Structure

```
audio-clonning-2/
├── requirements.txt
├── README.md
└── app/
    ├── main.py            # FastAPI backend (single file)
    └── static/
        └── index.html     # Frontend UI
```

---

## 1 · Setup Instructions

### Prerequisites
| Requirement | Min version |
|---|---|
| Python | 3.10+ |
| pip | 23+ |
| FFmpeg | any (required by pydub) |

**Install FFmpeg** (needed for MP3 decoding):

```bash
# Windows (winget)
winget install Gyan.FFmpeg

# Windows (choco)
choco install ffmpeg

# Linux
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

### Create virtual environment & install

```bash
cd audio-clonning-2

python -m venv venv

# Activate
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

> **First run** will auto-download the XTTS-v2 model (~1.8 GB). This happens once and is cached under `~/.local/share/tts/` (Linux) or `%LOCALAPPDATA%\tts\` (Windows).

---

## 2 · Run Instructions

```bash
cd app
python main.py
```

Open **http://localhost:8000** in your browser.

That's it.

---

## 3 · GPU Setup Notes

| Scenario | What happens |
|---|---|
| **NVIDIA GPU + CUDA toolkit installed** | Auto-detected. Model runs on GPU. ~5–15 s per synthesis. |
| **No GPU / CPU only** | Auto-fallback to CPU. ~60–120 s per synthesis. |

### Installing PyTorch with CUDA (optional, for GPU)

Uninstall the CPU-only torch first, then reinstall with CUDA:

```bash
pip uninstall torch torchaudio -y
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Replace `cu121` with your CUDA version (`cu118`, `cu124`, etc.).  
Verify:

```python
import torch; print(torch.cuda.is_available())  # should print True
```

---

## 4 · Cloning Pipeline Explained

```
User Upload (WAV/MP3, ≤ 2 MB)
        │
        ▼
  ┌─────────────┐
  │  Validate    │  content-type + size check
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │  Decode      │  pydub reads WAV or MP3 → raw PCM
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │  Mono        │  stereo → mono
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │  Resample    │  librosa resamples to 22 050 Hz
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │  Normalize   │  peak normalization to [-1, 1]
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │  Trim        │  librosa.effects.trim removes silence (25 dB threshold)
  └──────┬──────┘
         ▼
  ┌─────────────────────────────────────────────┐
  │  XTTS-v2 tts_to_file()                      │
  │  • Extracts speaker embedding from sample    │
  │  • Generates mel-spectrogram conditioned     │
  │    on text + embedding                       │
  │  • Vocoder synthesizes waveform              │
  └──────┬──────────────────────────────────────┘
         ▼
  ┌─────────────┐
  │  BytesIO     │  read output WAV into memory buffer
  └──────┬──────┘
         ▼
  StreamingResponse → browser auto-downloads
         │
  BackgroundTasks → deletes ALL temp files
```

---

## 5 · Performance Notes

| Metric | GPU (RTX 3060+) | CPU |
|---|---|---|
| First request (model load) | ~30 s | ~60 s |
| Subsequent synthesis (short text) | 5–15 s | 60–120 s |
| Memory (VRAM / RAM) | ~2 GB VRAM | ~4 GB RAM |

**Tips:**
- Keep voice samples 3–10 seconds of clean speech for best quality.
- Shorter text → faster synthesis.
- Model is loaded once at startup and reused for all requests.
- No files persist — temp files are created via `tempfile` and deleted in `BackgroundTasks`.

---

## 6 · Common Errors and Fixes

| Error | Cause | Fix |
|---|---|---|
| `No module named 'TTS'` | TTS not installed | `pip install TTS` |
| `ffmpeg not found` or pydub errors | FFmpeg missing from PATH | Install FFmpeg (see above) |
| `CUDA out of memory` | GPU VRAM too low | Use a shorter voice sample, or force CPU: set env `CUDA_VISIBLE_DEVICES=""` |
| `Could not process audio` (422) | Corrupt or unsupported file | Re-export as 16-bit WAV, 22–48 kHz |
| `File too large` (400) | Upload > 2 MB | Trim or compress the voice sample |
| `Voice sample too short` (400) | < 1 s after silence trimming | Record a longer sample with speech |
| Model download stalls | Network / firewall | Manually download from [Coqui releases](https://github.com/coqui-ai/TTS/releases) and place in the TTS cache dir |
| Slow on CPU | Expected | Use a CUDA GPU, or be patient (~1–2 min) |
| `RuntimeError: DataLoader worker… exited unexpectedly` | Windows multiprocessing issue | Add `if __name__ == "__main__":` guard (already present in `main.py`) |
| Port 8000 in use | Another process on :8000 | Run with `uvicorn main:app --port 8001` |

---

## API Reference

### `POST /clone`

| Param | Type | Description |
|---|---|---|
| `file` | `UploadFile` | WAV or MP3, max 2 MB |
| `text` | `string` | Text to synthesize (1–1000 chars) |

**Response:** `audio/wav` stream (`Content-Disposition: attachment`).

### `GET /health`

Returns JSON with model status and device info.
