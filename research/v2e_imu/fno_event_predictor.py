#!/usr/bin/env python
"""
Fourier Neural Operator for Event Prediction.

Global receptive field via Fourier transform.
O(n log n) complexity, resolution-invariant.

Usage:
    from fno_event_predictor import FNOEventPredictor
    model = FNOEventPredictor()  # modes=4 default
"""

import torch
import torch.nn as nn


class FourierLayer(nn.Module):
    """Fourier neural operator layer (global mixing in frequency domain).

    Uses separate learnable complex weights per frequency bin, which is the
    core FNO design: each low-frequency mode gets its own (C_in → C_out) mixing.
    """

    def __init__(self, channels: int, modes: int = 16) -> None:
        super().__init__()
        self.modes = modes
        self.channels = channels

        # Per-frequency complex weights stored as real/imag pairs: (C_out, C_in, modes, modes)
        scale = 1.0 / (channels * channels)
        self.weight_real = nn.Parameter(scale * torch.randn(channels, channels, modes, modes))
        self.weight_imag = nn.Parameter(scale * torch.randn(channels, channels, modes, modes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) spatial features
        Returns:
            x_out: (B, C, H, W) features after global frequency mixing
        """
        B, C, H, W = x.shape

        x_fft = torch.fft.rfft2(x)  # (B, C, H, W//2+1), complex

        # Extract low-frequency block
        x_r = x_fft[:, :, :self.modes, :self.modes].real  # (B, C_in, modes, modes)
        x_i = x_fft[:, :, :self.modes, :self.modes].imag

        # Complex multiply per frequency: out = W * x  (W = W_r + i*W_i)
        # out_r = W_r @ x_r - W_i @ x_i
        # out_i = W_r @ x_i + W_i @ x_r
        out_r = (
            torch.einsum("oimn,bimn->bomn", self.weight_real, x_r)
            - torch.einsum("oimn,bimn->bomn", self.weight_imag, x_i)
        )
        out_i = (
            torch.einsum("oimn,bimn->bomn", self.weight_real, x_i)
            + torch.einsum("oimn,bimn->bomn", self.weight_imag, x_r)
        )

        # Embed back into full FFT buffer
        out_fft = torch.zeros_like(x_fft)
        out_fft[:, :, :self.modes, :self.modes] = out_r + 1j * out_i

        return torch.fft.irfft2(out_fft, s=(H, W))


class FNOEventPredictor(nn.Module):
    """Fourier Neural Operator for event prediction.
    
    Architecture:
    1. CNN encoder (local features)
    2. IMU encoder + fusion
    3. FNO layers (global mixing)
    4. CNN decoder (event prediction)
    """
    
    def __init__(self, modes: int = 4, fno_layers: int = 2, imu_hidden_dim: int = 128) -> None:
        super().__init__()
        self.modes = modes
        self.imu_hidden_dim = imu_hidden_dim
        
        # Encoder (local features)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(16, 128),
            nn.SiLU(inplace=True),
        )
        
        # IMU encoder (temporal features)
        self.imu_encoder = nn.LSTM(6, imu_hidden_dim, batch_first=True)
        self.imu_fusion = nn.Linear(imu_hidden_dim, 128)
        
        # FNO layers (global mixing)
        self.fno_layers = nn.ModuleList([
            FourierLayer(128, modes=modes)
            for _ in range(fno_layers)
        ])
        
        # Decoder (event prediction)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 2, 3, padding=1),
        )
        
    def forward(self, rgb: torch.Tensor, imu_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb: (B, 1, H, W) grayscale images
            imu_seq: (B, T, 6) IMU sequences (accelerometer + gyroscope)
        Returns:
            events: (B, 2, H, W) event rate prediction (positive, for Poisson)
        """
        B, _, H, W = rgb.shape
        
        # Encode RGB (local features)
        x = self.encoder(rgb)  # (B, 128, H/2, W/2)
        
        # Encode IMU (temporal features)
        _, (imu_hidden, _) = self.imu_encoder(imu_seq)  # imu_hidden: (1, B, 128)
        imu_features = self.imu_fusion(imu_hidden[0])  # (B, 128)
        
        # Fuse IMU with RGB features (broadcast IMU to spatial dimensions)
        imu_map = imu_features.view(B, -1, 1, 1).expand(-1, -1, x.shape[2], x.shape[3])
        x = x + imu_map  # Additive fusion (IMU modulates RGB features)
        
        # FNO global mixing (with residual connections)
        for fno_layer in self.fno_layers:
            x = x + fno_layer(x)
        
        # Decode to event rate
        events = self.decoder(x)
        
        # Ensure positive rate (for Poisson likelihood)
        events = torch.exp(events)
        
        return events


def test_fno() -> None:
    """Test FNO event predictor with IMU fusion."""
    B, H, W = 2, 260, 346
    T = 50
    
    model = FNOEventPredictor(fno_layers=2, imu_hidden_dim=128)  # uses default modes=4
    rgb = torch.randn(B, 1, H, W)
    imu = torch.randn(B, T, 6)
    
    events = model(rgb, imu)
    
    print(f"RGB Input: {rgb.shape}")
    print(f"IMU Input: {imu.shape}")
    print(f"Output: {events.shape}")
    print(f"Output range: [{events.min():.4f}, {events.max():.4f}] (should be positive for Poisson)")
    print("✅ FNO with IMU fusion test passed!")


if __name__ == "__main__":
    test_fno()
