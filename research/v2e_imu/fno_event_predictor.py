"""
Fourier Neural Operator for Event Prediction.

Drop-in replacement for EventPredictor (UNet).  Select via MODEL_TYPE="fno" in train.py.

Architecture:
  1. CNN encoder     — 2-channel frame pair  → local features (H/2, W/2)
  2. IMU encoder     — bidirectional LSTM    → conditioning vector
  3. FiLM + FNO loop — FiLM modulates each FNO layer; global frequency mixing
  4. CNN decoder     — transposed conv       → (B, 2, H, W) sigmoid probs

Advantages over UNet:
  - Global receptive field via FFT (captures long-range motion patterns)
  - O(N log N) complexity vs O(N²) for attention
  - Resolution-invariant: same weights work at any spatial scale
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierLayer(nn.Module):  # type: ignore[misc]
    """Fourier neural operator layer — global mixing in the frequency domain.

    Learns separate complex weights per low-frequency mode:
        out_fft[m,n] = W[m,n] * x_fft[m,n]   for m,n < modes
    All other frequency coefficients are zeroed (low-pass filter + learned mixing).
    """

    def __init__(self, channels: int, modes: int = 8) -> None:
        super().__init__()
        self.modes = modes
        scale = 1.0 / (channels * channels)
        # (C_out, C_in, modes_h, modes_w) stored as real + imag separately
        self.weight_real = nn.Parameter(scale * torch.randn(channels, channels, modes, modes))
        self.weight_imag = nn.Parameter(scale * torch.randn(channels, channels, modes, modes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        dtype = x.dtype
        autocast_device = "cuda" if x.device.type == "cuda" else "cpu"

        # FFT in full float32 is much more robust than ComplexHalf under CUDA autocast,
        # and clamping the active modes avoids shape crashes on smaller feature maps.
        with torch.amp.autocast(device_type=autocast_device, enabled=False):
            x_fft = torch.fft.rfft2(x.float())  # (B, C, H, W//2+1) complex64
            modes_h = min(self.modes, x_fft.shape[-2])
            modes_w = min(self.modes, x_fft.shape[-1])
            if modes_h == 0 or modes_w == 0:
                return torch.zeros_like(x)

            x_r = x_fft[:, :, :modes_h, :modes_w].real
            x_i = x_fft[:, :, :modes_h, :modes_w].imag
            weight_real = self.weight_real[:, :, :modes_h, :modes_w].float()
            weight_imag = self.weight_imag[:, :, :modes_h, :modes_w].float()

            # Complex matrix multiply: (W_r + iW_i)(x_r + ix_i)
            out_r = torch.einsum("oimn,bimn->bomn", weight_real, x_r) - torch.einsum(
                "oimn,bimn->bomn", weight_imag, x_i
            )
            out_i = torch.einsum("oimn,bimn->bomn", weight_real, x_i) + torch.einsum(
                "oimn,bimn->bomn", weight_imag, x_r
            )

            out_fft = torch.zeros((B, C, H, x_fft.shape[-1]), dtype=x_fft.dtype, device=x.device)
            out_fft[:, :, :modes_h, :modes_w] = torch.complex(out_r.float(), out_i.float())
            out = torch.fft.irfft2(out_fft, s=(H, W))

        return out.to(dtype=dtype)


class FNOEventPredictor(nn.Module):  # type: ignore[misc]
    """FNO-based RGB+IMU → ON/OFF event probability maps.

    Same interface as EventPredictor:
        forward(image: Tensor[B,2,H,W], imu_seq: Tensor[B,T,6]) → Tensor[B,2,H,W]
    Output is sigmoid probabilities in [0, 1].

    Args:
        modes:          Number of Fourier modes to keep per spatial dim (default 8).
        fno_layers:     Number of FNO + FiLM blocks (default 4).
        channels:       Feature channels in FNO trunk (default 128).
        imu_hidden_dim: IMU LSTM hidden size (default 128).
    """

    def __init__(
        self,
        modes: int = 8,
        fno_layers: int = 4,
        channels: int = 128,
        imu_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        # ----- CNN encoder: 2-channel frame pair → spatial features -----
        half = channels // 2
        self.encoder = nn.Sequential(
            nn.Conv2d(2, half, 3, padding=1),
            nn.GroupNorm(8, half),
            nn.SiLU(inplace=True),
            nn.Conv2d(half, channels, 3, stride=2, padding=1),
            nn.GroupNorm(16, channels),
            nn.SiLU(inplace=True),
        )

        # ----- IMU encoder: bidirectional LSTM (matches UNet IMUEncoder) -----
        self.imu_lstm = nn.LSTM(
            6, imu_hidden_dim, num_layers=2, batch_first=True, bidirectional=True
        )
        self.imu_fc = nn.Linear(imu_hidden_dim * 2, imu_hidden_dim)
        self.imu_norm = nn.LayerNorm(imu_hidden_dim)

        # ----- FiLM scale/bias for each FNO block -----
        # Scale initialised to identity (weight=0, bias=1): no IMU effect at step 0.
        self.film_scale = nn.ModuleList(
            [nn.Linear(imu_hidden_dim, channels) for _ in range(fno_layers)]
        )
        self.film_bias = nn.ModuleList(
            [nn.Linear(imu_hidden_dim, channels) for _ in range(fno_layers)]
        )
        for linear in self.film_scale:
            nn.init.zeros_(linear.weight)
            nn.init.ones_(linear.bias)

        # ----- FNO layers + GroupNorm -----
        self.fno = nn.ModuleList([FourierLayer(channels, modes=modes) for _ in range(fno_layers)])
        self.fno_norm = nn.ModuleList(
            [nn.GroupNorm(min(8, channels), channels) for _ in range(fno_layers)]
        )

        # ----- CNN decoder: features → (B, 2, H, W) -----
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(channels, half, 4, stride=2, padding=1),
            nn.GroupNorm(8, half),
            nn.SiLU(inplace=True),
            nn.Conv2d(half, 2, 3, padding=1),
        )

    def forward(self, image: torch.Tensor, imu_seq: torch.Tensor) -> torch.Tensor:
        B, _, H, W = image.shape

        # Encode image
        x = self.encoder(image)  # (B, channels, H/2, W/2)

        # Encode IMU — bidirectional: use last-layer forward+backward hidden states
        _, (h_n, _) = self.imu_lstm(imu_seq)
        imu_feat = self.imu_norm(
            self.imu_fc(torch.cat([h_n[-2], h_n[-1]], dim=-1))
        )  # (B, imu_hidden_dim)

        # FNO blocks with FiLM conditioning
        for fno_layer, norm, scale_fn, bias_fn in zip(
            self.fno, self.fno_norm, self.film_scale, self.film_bias, strict=True
        ):
            scale = scale_fn(imu_feat).view(B, -1, 1, 1)  # (B, C, 1, 1)
            bias = bias_fn(imu_feat).view(B, -1, 1, 1)
            # FiLM modulation + FNO residual + normalisation
            x = norm(x * scale + bias + fno_layer(x))

        # Decode to logits
        logit = self.decoder(x)  # ≈ (B, 2, H, W) — may differ by 1px due to stride
        if logit.shape[2:] != (H, W):
            logit = F.interpolate(logit, size=(H, W), mode="bilinear", align_corners=False)

        return torch.sigmoid(logit)
