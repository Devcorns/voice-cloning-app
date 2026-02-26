"""Download Wav2Lip model weights.

Tries HuggingFace first, then falls back to manual instructions.
Run from the video-app directory:

    python scripts/download_models.py
"""

from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
CHECKPOINT = MODELS_DIR / "wav2lip_gan.pth"

# Multiple mirrors — tried in order until one succeeds
MIRROR_URLS = [
    "https://huggingface.co/numz/wav2lip_studio/resolve/main/Wav2Lip/wav2lip_gan.pth",
    "https://huggingface.co/camenduru/Wav2Lip/resolve/main/checkpoints/wav2lip_gan.pth",
    "https://huggingface.co/Ian09/Wav2Lip/resolve/main/wav2lip_gan.pth",
    "https://huggingface.co/sashahdo/Wav2Lip/resolve/main/wav2lip_gan.pth",
    "https://huggingface.co/Klaas2000/Wav2Lip/resolve/main/wav2lip_gan.pth",
]

OFFICIAL_LINKS = """
If the automatic download fails, download the model manually:

1. wav2lip_gan.pth (recommended — GAN-trained, better quality):
   https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z8VTmoxp55YNDcIA?e=n9ljGW

2. wav2lip.pth (L1-only, faster but lower quality):
   https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/Eb3LEKfCAzpuhR2BXQ1QCwEB5niZkBYOCNaAHMMAKmNJLA?e=n9ljGW

Place the downloaded file at:
   {checkpoint}
""".strip()


def download() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if CHECKPOINT.exists():
        size_mb = CHECKPOINT.stat().st_size / (1024 * 1024)
        if size_mb > 100:  # valid checkpoint should be ~415 MB
            print(f"✔ Model already exists at {CHECKPOINT} ({size_mb:.0f} MB)")
            return
        else:
            print(f"⚠ Existing file too small ({size_mb:.1f} MB) — likely incomplete. Re-downloading…")
            CHECKPOINT.unlink()

    for i, url in enumerate(MIRROR_URLS, 1):
        print(f"\n[{i}/{len(MIRROR_URLS)}] Trying: {url}")
        print(f"  Dest → {CHECKPOINT}")
        try:
            urllib.request.urlretrieve(url, str(CHECKPOINT), _progress)
            print()
            size_mb = CHECKPOINT.stat().st_size / (1024 * 1024)
            if size_mb < 100:   # too small = partial or HTML error page
                print(f"  ✖ File too small ({size_mb:.1f} MB) — likely an error page. Skipping.")
                CHECKPOINT.unlink(missing_ok=True)
                continue
            print(f"✔ Downloaded successfully ({size_mb:.0f} MB)")
            return
        except Exception as exc:
            print(f"  ✖ Failed: {exc}")
            CHECKPOINT.unlink(missing_ok=True)

    print()
    print("All mirrors failed.")
    print(OFFICIAL_LINKS.format(checkpoint=CHECKPOINT))
    sys.exit(1)


def _progress(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        print(f"\r  Progress: {pct:3d}%  ({mb:.1f} / {total_mb:.1f} MB)", end="", flush=True)
    else:
        mb = downloaded / (1024 * 1024)
        print(f"\r  Downloaded: {mb:.1f} MB", end="", flush=True)


if __name__ == "__main__":
    download()
