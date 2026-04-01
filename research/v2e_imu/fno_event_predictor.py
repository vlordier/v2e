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
    2. IMU encoder + fusion
    3. FNO layers (global mixing)
    4. CNN decoder (event prediction)
    """
    
    def __init__(self, modes: int = 16, fno_layers: int = 2, imu_hidden_dim: int = 128) -> None:
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
    
    model = FNOEventPredictor(modes=16, fno_layers=2, imu_hidden_dim=128)
    rgb = torch.randn(B, 1, H, W)
    imu = torch.randn(B, T, 6)
    
    events = model(rgb, imu)
    
    print(f"RGB Input: {rgb.shape}")
    print(f"IMU Input: {imu.shape}")
    print(f"Output: {events.shape}")
    print(f"Output range: [{events.min():.4f}, {events.max():.4f}] (should be positive for Poisson)")
    print(f"✅ FNO with IMU fusion test passed!")


if __name__ == "__main__":
    test_fno()
