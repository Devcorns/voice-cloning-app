"""
Wav2Lip generator architecture — inference only.

This is the exact model architecture from the Wav2Lip paper/repo
(Prajwal et al., ACM Multimedia 2020).  Compatible with both
``wav2lip.pth`` and ``wav2lip_gan.pth`` checkpoints.

Architecture overview
─────────────────────
• **Face encoder** (7 blocks): takes a 6-channel input (masked lower-half
  face ∥ original face) at 96×96 and compresses it to a 512-d vector.
• **Audio encoder**: takes a mel-spectrogram window (1×80×16) and
  compresses it to a 512-d vector.
• **Face decoder** (7 blocks): fuses the audio embedding with multi-scale
  face features via U-Net skip connections, producing a 3-channel 96×96
  lip-synced face.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ── Building blocks ──────────────────────────────────────────────────────────

class Conv2d(nn.Module):
    """Conv → BatchNorm → ReLU, with optional residual connection."""

    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size, stride, padding),
            nn.BatchNorm2d(cout),
        )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)


class Conv2dTranspose(nn.Module):
    """ConvTranspose → BatchNorm → ReLU."""

    def __init__(self, cin, cout, kernel_size, stride, padding, output_padding=0):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ConvTranspose2d(cin, cout, kernel_size, stride, padding, output_padding),
            nn.BatchNorm2d(cout),
        )
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.conv_block(x))


# ── Wav2Lip generator ───────────────────────────────────────────────────────

class Wav2Lip(nn.Module):
    """Wav2Lip generator network.

    Input
    ─────
    audio_sequences : (B, 1, 80, 16)  — mel-spectrogram window
    face_sequences  : (B, 6, 96, 96)  — [masked_face ∥ original_face]

    Output
    ──────
    (B, 3, 96, 96) — lip-synced face, pixel values in [0, 1]
    """

    def __init__(self):
        super().__init__()

        # ── Face encoder ─────────────────────────────────────────────────
        self.face_encoder_blocks = nn.ModuleList([
            # 96×96
            nn.Sequential(
                Conv2d(6, 16, kernel_size=7, stride=1, padding=3),
            ),
            # → 48×48
            nn.Sequential(
                Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            ),
            # → 24×24
            nn.Sequential(
                Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            ),
            # → 12×12
            nn.Sequential(
                Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            ),
            # → 6×6
            nn.Sequential(
                Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            ),
            # → 3×3
            nn.Sequential(
                Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            ),
            # → 1×1
            nn.Sequential(
                Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
                Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            ),
        ])

        # ── Audio encoder ────────────────────────────────────────────────
        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
        )

        # ── Face decoder (U-Net with skip connections) ───────────────────
        self.face_decoder_blocks = nn.ModuleList([
            # 1×1
            nn.Sequential(
                Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            ),
            # → 3×3  (cat with encoder 512@3×3 = 1024)
            nn.Sequential(
                Conv2dTranspose(1024, 512, kernel_size=3, stride=1, padding=0),
                Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            ),
            # → 6×6  (cat with encoder 256@6×6 = 768)
            nn.Sequential(
                Conv2dTranspose(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
                Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            ),
            # → 12×12 (cat with encoder 128@12×12 = 512)
            nn.Sequential(
                Conv2dTranspose(768, 384, kernel_size=3, stride=2, padding=1, output_padding=1),
                Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),
            ),
            # → 24×24 (cat with encoder 64@24×24 = 320)
            nn.Sequential(
                Conv2dTranspose(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
                Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            ),
            # → 48×48 (cat with encoder 32@48×48 = 160)
            nn.Sequential(
                Conv2dTranspose(320, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            ),
            # → 96×96 (cat with encoder 16@96×96 = 80)
            nn.Sequential(
                Conv2dTranspose(160, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            ),
        ])

        # ── Output head ──────────────────────────────────────────────────
        self.output_block = nn.Sequential(
            Conv2d(80, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, audio_sequences, face_sequences):
        """
        Parameters
        ----------
        audio_sequences : Tensor (B, 1, 80, 16)  or  (B, T, 1, 80, 16)
        face_sequences  : Tensor (B, 6, 96, 96)  or  (B, T, 6, 96, 96)
        """
        B = audio_sequences.size(0)
        input_dim_size = len(face_sequences.size())

        if input_dim_size > 4:
            audio_sequences = torch.cat(
                [audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0,
            )
            face_sequences = torch.cat(
                [face_sequences[:, i] for i in range(face_sequences.size(1))], dim=0,
            )

        audio_embedding = self.audio_encoder(audio_sequences)  # → (B, 512, 1, 1)

        # Encode face at multiple scales (for skip connections)
        feats = []
        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)
            feats.append(x)

        # Decode: fuse audio embedding with face features
        x = audio_embedding
        for f in self.face_decoder_blocks:
            x = f(x)
            try:
                x = torch.cat((x, feats[-1]), dim=1)
            except Exception:
                raise
            feats.pop()

        x = self.output_block(x)  # (B, 3, 96, 96) in [0, 1]

        if input_dim_size > 4:
            x = torch.split(x, B, dim=0)
            outputs = torch.stack(x, dim=2)
        else:
            outputs = x

        return outputs
