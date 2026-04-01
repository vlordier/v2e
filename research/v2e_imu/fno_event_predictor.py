#!/usr/bin/env python
"""
Fourier Neural Operator for Event Prediction.

Global receptive field via Fourier transform.
O(n log n) complexity, resolution-invariant.

Usage:
    from fno_event_predictor import FNOEventPredictor
    model = FNOEventPredictor(modes=16)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierLayer(nn.Module):
    """Fourier neural operator layer (global mixing in frequency domain)."""
    
    def __init__(self, channels: int, modes: int = 16) -> None:
        super().__init__()
        self.modes = modes
        self.channels = channels
        
        # Learnable weights in Fourier space
        self.scale = nn.Linear(channels, channels, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) spatial features
        Returns:
            x_out: (B, C, H, W) features after global mixing
        """
        B, C, H, W = x.shape
        
        # FFT to frequency domain
        x_fft = torch.fft.rfft2(x)
        
        # Filter low frequencies only (learnable)
        # Create frequency mask
        mask = torch.zeros_like(x_fft)
        mask[:, :, :self.modes, :self.modes] = 1
        
        # Apply learned filter in Fourier space (handle complex dtype)
        # Split into real and imaginary parts for linear layer
        x_real = x_fft.real
        x_imag = x_fft.imag
        
        # Apply scale to both real and imaginary
        x_real_scaled = self.scale(x_real.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_imag_scaled = self.scale(x_imag.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        # Apply mask
        x_filtered = (x_real_scaled + 1j * x_imag_scaled) * mask
        
        # IFFT back to spatial domain
        x_out = torch.fft.irfft2(x_filtered, s=(H, W))
        
        return x_out


class FNOEventPredictor(nn.Module):
    """Fourier Neural Operator for event prediction.
    
    Architecture:
    1. CNN encoder (local features)
    2. FNO layers (global mixing)
    3. CNN decoder (event prediction)
    """
    
    def __init__(self, modes: int = 16, fno_layers: int = 2) -> None:
        super().__init__()
        self.modes = modes
        
        # Encoder (local features)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(16, 128),
            nn.SiLU(inplace=True),
        )
        
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
        
    def forward(self, rgb: torch.Tensor, imu_seq: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            rgb: (B, 1, H, W) grayscale images
            imu_seq: (B, T, 6) IMU sequences (optional, not used in FNO yet)
        Returns:
            events: (B, 2, H, W) event rate prediction
        """
        # Encode
        x = self.encoder(rgb)
        
        # FNO global mixing
        for fno_layer in self.fno_layers:
            x = x + fno_layer(x)  # Residual connection
        
        # Decode
        events = self.decoder(x)
        
        # Ensure positive rate (for Poisson)
        events = torch.exp(events)
        
        return events


def test_fno() -> None:
    """Test FNO event predictor."""
    B, H, W = 2, 260, 346
    
    model = FNOEventPredictor(modes=16, fno_layers=2)
    rgb = torch.randn(B, 1, H, W)
    imu = torch.randn(B, 50, 6)
    
    events = model(rgb, imu)
    
    print(f"Input: {rgb.shape}")
    print(f"Output: {events.shape}")
    print(f"Output range: [{events.min():.4f}, {events.max():.4f}]")
    print(f"✅ FNO test passed!")


if __name__ == "__main__":
    test_fno()
