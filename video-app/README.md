# 🎬 Lip-Sync Video Generator (Wav2Lip)

Generate lip-synced talking-head videos from a **face image or short video** + a **voice audio file**.  
Powered by [Wav2Lip](https://github.com/Rudrabha/Wav2Lip) — runs **100 % local**, no paid APIs.

---

## 1. Folder Structure

```
video-app/
├── app/
│   ├── __init__.py
│   ├── main.py                 ← FastAPI backend (port 8001)
│   ├── wav2lip/
│   │   ├── __init__.py
│   │   ├── model.py            ← Wav2Lip neural-network architecture
│   │   ├── audio.py            ← Mel-spectrogram extraction
│   │   └── inference.py        ← Face detection + inference pipeline
│   └── static/
│       └── index.html          ← Dark-themed frontend
├── models/
│   ├── .gitkeep
│   └── wav2lip_gan.pth         ← Download separately (~400 MB)
├── scripts/
│   └── download_models.py      ← Auto-download helper
├── requirements.txt
├── .gitignore
└── README.md                   ← You are here
```

---

## 2. Backend Code (`app/main.py`)

- **FastAPI** with CORS enabled for external tool integration.
- `POST /generate` & `POST /api/v1/generate` — accepts `face` (image/video) + `audio` (WAV/MP3) as multipart form uploads.
- All synthesis runs in a `ThreadPoolExecutor` so the async event loop stays responsive.
- Temporary files are created with `tempfile` and cleaned up via `BackgroundTasks` after the response is sent (no file persistence).
- Returns the final **MP4** as a `StreamingResponse` with `Content-Disposition: attachment`.
- `GET /api/v1/health` — model status, device, limits.
- Swagger docs auto-generated at `/docs`.

---

## 3. Frontend (`app/static/index.html`)

- Dark-themed single-page UI.
- Two drag-and-drop zones: face file + audio file.
- Submits to `/generate`, shows a spinner, then renders the returned MP4 in a `<video>` player and auto-downloads it.
- 10-minute AbortController timeout.

---

## 4. Requirements (`requirements.txt`)

```
fastapi==0.115.6
uvicorn[standard]==0.34.0
python-multipart==0.0.20
torch>=2.1
torchaudio>=2.1
opencv-python-headless>=4.8
librosa>=0.10.2
soundfile>=0.12
numpy>=1.26
pydantic>=2.0
```

---

## 5. Wav2Lip Model Download

### Option A — Automatic (recommended)

```bash
cd video-app
python scripts/download_models.py
```

Downloads `wav2lip_gan.pth` (~400 MB) from HuggingFace into `models/`.

### Option B — Manual

1. Download **wav2lip_gan.pth** (GAN-trained, better quality):
   - [Official SharePoint link](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z8VTmoxp55YNDcIA?e=n9ljGW)
   - Or search HuggingFace for `wav2lip_gan.pth`

2. Place the file at: `video-app/models/wav2lip_gan.pth`

> **Note:** `wav2lip.pth` (L1-only, lower quality) also works — no code changes needed.

---

## 6. FFmpeg Installation

FFmpeg is required for merging audio + video into MP4.

### Windows

```powershell
winget install Gyan.FFmpeg
# Then restart terminal or refresh PATH:
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
ffmpeg -version
```

### Linux (Ubuntu/Debian)

```bash
sudo apt update && sudo apt install -y ffmpeg
ffmpeg -version
```

### macOS

```bash
brew install ffmpeg
```

---

## 7. Setup & Run

### Windows

```powershell
cd video-app

# Create virtual environment
py -3.11 -m venv venv          # Python 3.10–3.12 recommended
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Download Wav2Lip model
python scripts\download_models.py

# Run server
cd app
python main.py
# → http://localhost:8001
```

### Linux / macOS

```bash
cd video-app

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Download Wav2Lip model
python scripts/download_models.py

# Run server
cd app
python main.py
# → http://localhost:8001
```

> The server runs on **port 8001** (voice-cloning runs on 8000).

---

## 8. GPU Setup Guide

### Check CUDA availability

```python
import torch
print(torch.cuda.is_available())   # True = GPU ready
print(torch.cuda.get_device_name(0))
```

### Install PyTorch with CUDA

Visit [pytorch.org/get-started](https://pytorch.org/get-started/locally/) and pick your CUDA version. Example for CUDA 12.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Key GPU considerations

| Setting | CPU | GPU (8 GB VRAM) |
|---------|-----|------------------|
| `WAV2LIP_BATCH` in `inference.py` | 16 | 128 (default) |
| 10 s audio processing time | ~2–5 min | ~15–30 s |
| Max practical audio length | ~15 s | ~60 s |

If you get **CUDA OOM**, reduce `WAV2LIP_BATCH` in `inference.py` (line 11).

---

## 9. Performance Optimization Notes

1. **GPU vs CPU**: GPU is 10–20× faster. Even a GTX 1060 is dramatically better than CPU.
2. **Static image optimization**: Face detection runs only once when the input is an image (all frames are identical).
3. **Batch size**: `WAV2LIP_BATCH=128` for GPU, reduce to `16–32` for CPU or limited VRAM.
4. **Frame resolution**: Wav2Lip processes faces at 96×96 regardless of input resolution. Larger input frames only affect the paste-back step, not the neural network.
5. **Audio length**: Shorter audio = faster. Each second of audio at 25 fps = 25 frames through the model.
6. **Video looping**: If the input video is shorter than the audio, frames are looped automatically (no re-inference needed for the loop detection).
7. **FFmpeg preset**: Uses `-preset fast` for a good speed/quality balance. Change to `ultrafast` for development.
8. **Half precision**: For GPUs with Tensor Cores (RTX 20xx+), you can add `model.half()` and use `torch.float16` tensors for ~2× speedup.

---

## 10. Common Errors & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `Wav2Lip model not found` | `wav2lip_gan.pth` missing | Run `python scripts/download_models.py` or download manually to `models/` |
| `No face detected` | Face not visible or too small | Use a clear front-facing photo; ensure face fills ≥ 30 % of the image |
| `ffmpeg: command not found` | FFmpeg not installed or not in PATH | Install FFmpeg (see § 6) and restart terminal |
| `CUDA out of memory` | GPU VRAM exhausted | Reduce `WAV2LIP_BATCH` in `inference.py` from 128 to 32 or 16 |
| `Cannot open video` | Corrupted or unsupported codec | Re-encode with: `ffmpeg -i input.mov -c:v libx264 output.mp4` |
| `RuntimeError: state_dict mismatch` | Wrong checkpoint file | Ensure you downloaded `wav2lip_gan.pth`, not a fine-tuned variant |
| `librosa / soundfile error` | Audio codec not supported | Convert to WAV first: `ffmpeg -i input.opus -ar 16000 -ac 1 output.wav` |
| Port 8001 in use | Another process on port | Change port in `main.py`: `uvicorn.run(..., port=8002)` |

---

## 11. Technical Explanation

### 11.1 Face Detection

The system uses **OpenCV's Haar cascade** (`haarcascade_frontalface_default.xml`) for face detection:

- Converts each frame to grayscale.
- Runs multi-scale sliding-window detection.
- Selects the **largest** detected face (by area).
- If a frame misses detection, the **last known** bounding box is reused.
- For static images, detection runs **once** and the result is reused for all frames.

The detected bounding box `(x1, y1, x2, y2)` is padded (default: +10 px bottom) to include the chin and jaw area essential for lip rendering.

### 11.2 Lip Landmark Alignment

Wav2Lip doesn't use explicit lip landmarks at inference time. Instead:

- The face crop is resized to **96 × 96** pixels.
- The **lower half** of the face (rows 48–96) is **zeroed out** (masked).
- The model learns an implicit alignment during training via:
  - A **SyncNet discriminator** that checks audio-visual sync.
  - Reconstruction loss on the full face.
- The 6-channel input `[masked_face | original_face]` gives the model both the "what to generate" (lower half) and "reference appearance" (full face) signals.

This implicit approach is more robust than explicit landmark-based methods because it handles varied face shapes, expressions, and poses.

### 11.3 Mel Spectrogram Syncing

The audio-to-video sync is driven by **mel spectrograms**:

```
Audio (16 kHz) → Preemphasis → STFT (n_fft=800, hop=200) → Mel filter bank (80 bins) → dB → Normalize
```

Key sync parameters:
- **80 mel bins** × window of **16 time-steps** = one model input (shape: `1 × 80 × 16`).
- Each video frame at 25 fps maps to a mel window starting at column `int(80 × frame_idx / 25)`.
- This gives ~3.2 mel columns per video frame — the overlapping windows ensure smooth sync.
- The mel values are normalized to `[−4, +4]` matching the training distribution.

### 11.4 Frame Synthesis

The Wav2Lip generator is a **U-Net** with:

1. **Face encoder** (7 blocks): Compresses the 6-channel 96×96 face to a 512-d vector, storing multi-scale features at each level.
2. **Audio encoder**: Compresses the `1×80×16` mel window to a 512-d vector.
3. **Face decoder** (7 blocks): Takes the audio embedding and progressively upsamples back to 96×96, fusing with face encoder features via **skip connections** at each scale.
4. **Output head**: Projects to 3-channel RGB with sigmoid activation (values in `[0, 1]`).

The audio embedding acts as a "lip shape command" — it tells the decoder what mouth shape to generate, while the face encoder skip connections preserve identity, skin tone, and non-lip facial features.

### 11.5 Audio-Video Merging

The final step combines the lip-synced frames with the original audio:

1. **OpenCV VideoWriter** writes all frames to a temp `.avi` file (XVID codec).
2. **FFmpeg** merges the silent video with the audio:
   ```
   ffmpeg -y -i temp.avi -i audio.wav \
     -c:v libx264 -preset fast -crf 23 \
     -c:a aac -b:a 128k \
     -shortest -movflags +faststart \
     output.mp4
   ```
3. `-shortest` trims to the shorter of audio/video.
4. `-movflags +faststart` moves the MP4 moov atom to the beginning for faster streaming.
5. All temp files are deleted after the response is sent.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/` | Web UI |
| `POST` | `/generate` | Form upload → MP4 (web UI) |
| `POST` | `/api/v1/generate` | Form upload → MP4 (API, documented) |
| `GET`  | `/api/v1/health` | Model status, device, limits |
| `GET`  | `/docs` | Swagger UI |

---

## License

- Wav2Lip model weights: [CPML license](https://github.com/Rudrabha/Wav2Lip/blob/master/LICENSE) (non-commercial research).
- This wrapper code: MIT.
