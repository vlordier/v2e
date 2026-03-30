"""
IMU-Enhanced Event Camera Model.

Architecture:
- RGB encoder: CNN that extracts features from grayscale images
- IMU encoder: LSTM that processes IMU sequences (acc + gyro)
- Fusion: IMU-conditioned RGB feature modulation
- Output heads:
  1. Primary: Next-frame prediction (or depth estimation)
  2. Auxiliary: Event prediction (multi-task learning)

IMU acts as extra input that conditions/modulates RGB features.
Events serve as auxiliary ground truth during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class IMUEncoder(nn.Module):  # type: ignore[misc]
    """Encode IMU sequences (6D: acc_xyz + gyro_xyz) into feature vectors."""

    def __init__(self, imu_dim: int = 6, hidden_dim: int = 128, num_layers: int = 2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            imu_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)  # *2 for bidirectional
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, imu_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            imu_seq: (B, seq_len, 6) - IMU measurements over time

        Returns:
            imu_features: (B, hidden_dim) - Encoded IMU features
        """
        # LSTM encoding
        lstm_out, (h_n, _) = self.lstm(imu_seq)

        # Use last hidden state from both directions
        # h_n: (num_layers*2, B, hidden_dim)
        h_forward = h_n[-2]  # Last layer, forward
        h_backward = h_n[-1]  # Last layer, backward
        h_cat = torch.cat([h_forward, h_backward], dim=-1)  # (B, hidden_dim*2)

        # Project and normalize
        imu_features = self.fc(h_cat)
        imu_features = self.norm(imu_features)

        return imu_features


class RGBEncoder(nn.Module):  # type: ignore[misc]
    """Encode grayscale RGB images into feature maps."""

    def __init__(self, in_channels: int = 1, base_channels: int = 64) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            # Block 1: (B, 1, H, W) -> (B, 64, H/2, W/2)
            nn.Conv2d(in_channels, base_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            # Block 2: (B, 64, H/2, W/2) -> (B, 128, H/4, W/4)
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            # Block 3: (B, 128, H/4, W/4) -> (B, 256, H/8, W/8)
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            # Block 4: (B, 256, H/8, W/8) -> (B, 512, H/16, W/16)
            nn.Conv2d(base_channels * 4, base_channels * 8, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
        )

        self.out_channels = base_channels * 8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, H, W) - Grayscale image

        Returns:
            features: (B, 512, H/16, W/16) - Feature maps
        """
        return self.encoder(x)


class IMUConditionedRGB(nn.Module):  # type: ignore[misc]
    """
    Modulate RGB features using IMU information.

    IMU predicts per-channel scale and bias for RGB feature maps.
    This allows the model to adapt RGB processing based on motion.
    """

    def __init__(self, rgb_channels: int, imu_dim: int = 128) -> None:
        super().__init__()

        # IMU to feature modulation parameters
        self.imu_to_scale = nn.Sequential(
            nn.Linear(imu_dim, rgb_channels),
            nn.Sigmoid(),  # Scale between 0 and 1
        )

        self.imu_to_bias = nn.Sequential(
            nn.Linear(imu_dim, rgb_channels),
            nn.Tanh(),  # Bias between -1 and 1
        )

    def forward(self, rgb_features: torch.Tensor, imu_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb_features: (B, C, H, W) - RGB feature maps
            imu_features: (B, imu_dim) - IMU features

        Returns:
            modulated: (B, C, H, W) - IMU-modulated RGB features
        """
        B, C, H, W = rgb_features.shape

        # Predict modulation parameters from IMU
        scale = self.imu_to_scale(imu_features).view(B, C, 1, 1)
        bias = self.imu_to_bias(imu_features).view(B, C, 1, 1)

        # Apply modulation
        modulated = rgb_features * scale + bias

        return modulated


class EventPredictionHead(nn.Module):  # type: ignore[misc]
    """Predict events (2-channel: positive/negative) from features."""

    def __init__(self, in_channels: int, out_size: tuple[int, int] = (480, 640)) -> None:
        super().__init__()

        self.out_size = out_size

        self.decoder = nn.Sequential(
            # Upsample: (B, 512, H/16, W/16) -> (B, 256, H/8, W/8)
            nn.ConvTranspose2d(in_channels, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # Upsample: (B, 256, H/8, W/8) -> (B, 128, H/4, W/4)
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Upsample: (B, 128, H/4, W/4) -> (B, 64, H/2, W/2)
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Upsample: (B, 64, H/2, W/2) -> (B, 32, H, W)
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # Final: (B, 32, H, W) -> (B, 2, H, W)
            nn.Conv2d(32, 2, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) - Feature maps

        Returns:
            events: (B, 2, H_out, W_out) - Predicted event maps [positive, negative]
        """
        events = self.decoder(x)

        # Upsample to target size if needed
        if events.shape[2:] != self.out_size:
            events = F.interpolate(events, size=self.out_size, mode="bilinear", align_corners=False)

        return events


class DepthPredictionHead(nn.Module):  # type: ignore[misc]
    """Predict depth from features (primary task)."""

    def __init__(self, in_channels: int, out_size: tuple[int, int] = (480, 640)) -> None:
        super().__init__()

        self.out_size = out_size

        self.decoder = nn.Sequential(
            # Upsample layers
            nn.ConvTranspose2d(in_channels, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # Single channel depth
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid(),  # Normalize depth to [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) - Feature maps

        Returns:
            depth: (B, 1, H_out, W_out) - Predicted depth map
        """
        depth = self.decoder(x)

        if depth.shape[2:] != self.out_size:
            depth = F.interpolate(depth, size=self.out_size, mode="bilinear", align_corners=False)

        return depth


class IMUEnhancedEventModel(nn.Module):  # type: ignore[misc]
    """
    Main model: RGB + IMU → Depth + Events

    IMU serves as extra input that conditions RGB processing.
    Events serve as auxiliary ground truth during training.
    """

    def __init__(
        self,
        imu_seq_len: int = 50,
        imu_hidden_dim: int = 128,
        image_size: tuple[int, int] = (480, 640),
    ):
        super().__init__()

        self.image_size = image_size

        # Encoders
        self.imu_encoder = IMUEncoder(imu_dim=6, hidden_dim=imu_hidden_dim)
        self.rgb_encoder = RGBEncoder(in_channels=1, base_channels=64)

        # IMU-conditioned RGB modulation
        self.imu_conditioning = IMUConditionedRGB(
            rgb_channels=self.rgb_encoder.out_channels, imu_dim=imu_hidden_dim
        )

        # Prediction heads
        self.depth_head = DepthPredictionHead(
            in_channels=self.rgb_encoder.out_channels, out_size=image_size
        )

        self.event_head = EventPredictionHead(
            in_channels=self.rgb_encoder.out_channels, out_size=image_size
        )

    def forward(
        self, rgb: torch.Tensor, imu_seq: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            rgb: (B, 1, H, W) - Grayscale image
            imu_seq: (B, seq_len, 6) - IMU sequence

        Returns:
            depth: (B, 1, H, W) - Predicted depth (primary task)
            events: (B, 2, H, W) - Predicted events (auxiliary GT)
        """
        # Encode IMU
        imu_features = self.imu_encoder(imu_seq)  # (B, 128)

        # Encode RGB
        rgb_features = self.rgb_encoder(rgb)  # (B, 512, H/16, W/16)

        # Modulate RGB features with IMU
        conditioned_features = self.imu_conditioning(rgb_features, imu_features)

        # Predict depth (primary task)
        depth = self.depth_head(conditioned_features)

        # Predict events (auxiliary task - events as extra GT)
        events = self.event_head(conditioned_features)

        return depth, events


class MultiTaskLoss(nn.Module):  # type: ignore[misc]
    """
    Multi-task loss combining depth and event prediction.

    Automatically balances losses using uncertainty weighting.
    """

    def __init__(self, task_weights: dict[str, float] | None = None) -> None:
        super().__init__()

        # Learnable log variance for uncertainty weighting
        self.log_var_depth = nn.Parameter(torch.tensor(0.0))
        self.log_var_events = nn.Parameter(torch.tensor(0.0))

        # Task weights (manual weighting)
        self.task_weights = task_weights or {"depth": 1.0, "events": 1.0}

    def forward(
        self,
        pred_depth: torch.Tensor,
        pred_events: torch.Tensor,
        gt_depth: torch.Tensor | None = None,
        gt_events: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            pred_depth: (B, 1, H, W) - Predicted depth
            pred_events: (B, 2, H, W) - Predicted events
            gt_depth: (B, 1, H, W) - Ground truth depth (optional)
            gt_events: (B, 2, H, W) - Ground truth events (optional)

        Returns:
            total_loss: Scalar loss
            loss_dict: Dictionary of individual losses
        """
        losses = {}
        total_loss = torch.tensor(0.0, device=pred_depth.device)

        # Depth loss (MSE)
        if gt_depth is not None:
            depth_loss = F.mse_loss(pred_depth, gt_depth)
            # Uncertainty weighting: L = (1/2σ²) * L_task + log(σ)
            depth_weighted = (
                1.0 / (2.0 * torch.exp(self.log_var_depth))
            ) * depth_loss + self.log_var_depth / 2.0
            total_loss += self.task_weights["depth"] * depth_weighted
            losses["depth_loss"] = depth_loss.item()

        # Event loss (MSE on event counts)
        if gt_events is not None:
            # L2 loss on positive and negative event channels
            event_loss = F.mse_loss(pred_events, gt_events)
            events_weighted = (
                1.0 / (2.0 * torch.exp(self.log_var_events))
            ) * event_loss + self.log_var_events / 2.0
            total_loss += self.task_weights["events"] * events_weighted
            losses["event_loss"] = event_loss.item()

        losses["total_loss"] = total_loss.item()
        losses["depth_weight"] = torch.exp(-self.log_var_depth).item()
        losses["events_weight"] = torch.exp(-self.log_var_events).item()

        return total_loss, losses


def create_model(
    imu_seq_len: int = 50,
    image_size: tuple[int, int] = (480, 640),
    device: str = "cpu",
) -> tuple[IMUEnhancedEventModel, MultiTaskLoss]:
    """Create model and loss function."""
    model = IMUEnhancedEventModel(
        imu_seq_len=imu_seq_len,
        image_size=image_size,
    ).to(device)

    criterion = MultiTaskLoss().to(device)

    return model, criterion


if __name__ == "__main__":
    # Test model creation
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    model, criterion = create_model(device=device)

    # Test forward pass
    batch_size = 2
    rgb = torch.randn(batch_size, 1, 480, 640).to(device)
    imu_seq = torch.randn(batch_size, 50, 6).to(device)

    depth, events = model(rgb, imu_seq)
    print(f"Depth shape: {depth.shape}")
    print(f"Events shape: {events.shape}")

    # Test loss computation
    gt_depth = torch.rand_like(depth)
    gt_events = torch.rand_like(events)

    loss, loss_dict = criterion(depth, events, gt_depth, gt_events)
    print(f"Loss: {loss.item():.4f}")
    print(f"Loss dict: {loss_dict}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
