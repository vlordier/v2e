"""
Fixed data preparation for IMU-Enhanced Event Camera experiments.

This file is READ-ONLY during experiments. It provides:
- Data loading from UZH FPV dataset (or synthetic fallback)
- Dataloaders for training and validation
- Evaluation metric (bits per byte for events)
- Fixed constants (time budget, sequence length, etc.)

Usage:
    python prepare_data.py  # One-time data prep
"""

import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Try to import cv2 for optical flow, fallback to simple gradient-based approach
try:
    import cv2

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# Try to import timm for DINO features
try:
    import timm

    HAS_TIMM = True
    # Load DINO model once (cached)
    # NOTE: DINO extraction is disabled by default due to worker segfault issues
    # Enable by setting USE_DINO=True before creating dataloader
    _DINO_MODEL = None
    _DINO_DEVICE = None
    USE_DINO = False  # Set to True to enable DINO features
    _DINO_BATCH_CACHE = None

    def get_dino_features(img: np.ndarray, device: str = "cpu") -> np.ndarray:
        """Extract DINO features from a grayscale image.

        Args:
            img: Grayscale image (H, W) in range [0, 1]
            device: Device to run model on

        Returns:
            features: DINO features (384,) for small model
        """
        global _DINO_MODEL, _DINO_DEVICE

        if not USE_DINO:
            return np.zeros(384, dtype=np.float32)

        # Initialize model on first call
        if _DINO_MODEL is None or _DINO_DEVICE != device:
            import torch

            _DINO_MODEL = timm.create_model(
                "vit_small_patch8_224.dino", pretrained=True, num_classes=0
            )
            _DINO_MODEL = _DINO_MODEL.to(device)
            _DINO_MODEL.eval()
            _DINO_DEVICE = device
            print(f"Loaded DINO model on {device}")

        # Prepare image: grayscale → 3-channel, resize to 224
        import torch
        import torch.nn.functional as F

        img_3ch = np.stack([img, img, img], axis=0)  # (3, H, W)
        img_tensor = torch.from_numpy(img_3ch).float().unsqueeze(0)  # (1, 3, H, W)

        # Resize to 224x224
        img_tensor = F.interpolate(
            img_tensor, size=(224, 224), mode="bilinear", align_corners=False
        )

        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        with torch.no_grad():
            features = _DINO_MODEL(img_tensor.to(device))

        return features.squeeze(0).cpu().numpy()

    def extract_dino_batch(images: torch.Tensor, device: str = "cpu") -> torch.Tensor:
        """Extract DINO features for a batch of images in main process.

        Args:
            images: (B, C, H, W) tensor of images
            device: device to run on

        Returns:
            dino_features: (B, 384) tensor
        """
        global _DINO_MODEL, _DINO_DEVICE, USE_DINO

        if not USE_DINO:
            return torch.zeros(images.shape[0], 384, device=device)

        import torch.nn.functional as F

        # Ensure model is loaded
        if _DINO_MODEL is None:
            _DINO_MODEL = timm.create_model(
                "vit_small_patch8_224.dino", pretrained=True, num_classes=0
            )
            _DINO_MODEL = _DINO_MODEL.to(device)
            _DINO_MODEL.eval()
            _DINO_DEVICE = device
            print(f"Loaded DINO model on {device}")

        B, C, H, W = images.shape

        # Convert to 3-channel and resize
        if C == 1:
            images_3ch = images.repeat(1, 3, 1, 1)  # (B, 3, H, W)
        else:
            images_3ch = images[:, :3, :, :]  # Take first 3 if more

        # Resize to 224x224
        images_224 = F.interpolate(
            images_3ch.float(), size=(224, 224), mode="bilinear", align_corners=False
        )

        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        images_224 = (images_224 - mean) / std

        with torch.no_grad():
            features = _DINO_MODEL(images_224.to(device))

        return features

except ImportError:
    HAS_TIMM = False
    USE_DINO = False

    def get_dino_features(img: np.ndarray, device: str = "cpu") -> np.ndarray:
        """Fallback when timm not available."""
        return np.zeros(384, dtype=np.float32)

    def extract_dino_batch(images: torch.Tensor, device: str = "cpu") -> torch.Tensor:
        """Fallback when timm not available."""
        return torch.zeros(images.shape[0], 384, device=device)

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 50  # IMU sequence length
IMAGE_SIZE = (260, 346)  # DAVIS346 resolution
TIME_BUDGET = 900  # 15 minutes per experiment (increased for proper evaluation)
EVAL_SAMPLES = 200  # Autoresearch: ~34s on MPS. Use 500+ for final eval.
EVENT_WINDOW_MS = 33  # Event accumulation window (30 Hz)

# Data directory
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "fpv")
CACHE_DIR = os.path.join(os.path.dirname(__file__), "data", "cache")


def compute_optical_flow(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """Compute optical flow between two grayscale images.

    Args:
        img1: First image (H, W) in range [0, 1]
        img2: Second image (H, W) in range [0, 1]

    Returns:
        flow: Optical flow (2, H, W) - flow_x, flow_y in range [-1, 1]
    """
    H, W = img1.shape

    if HAS_CV2:
        # Convert to uint8 for OpenCV
        img1_uint8 = (img1 * 255).astype(np.uint8)
        img2_uint8 = (img2 * 255).astype(np.uint8)

        # Use Farneback algorithm (dense optical flow)
        flow = cv2.calcOpticalFlowFarneback(
            img1_uint8,
            img2_uint8,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        # flow is (H, W, 2) -> normalize to [-1, 1]
        flow = flow.transpose(2, 0, 1)  # (2, H, W)

        # Normalize by image size to get pixel displacements as fraction
        flow = flow / np.array([[[H]], [[W]]])

        return flow.astype(np.float32)
    else:
        # Simple gradient-based flow (Horn-Schunck approximation)
        # Compute image gradients
        grad_x = np.diff(img1, axis=1, prepend=img1[:, :1])
        grad_y = np.diff(img1, axis=0, prepend=img1[:1, :])

        # Temporal gradient
        grad_t = img2 - img1

        # Simple optical flow estimation: u = -grad_t / (grad_x + epsilon)
        epsilon = 1e-8
        flow_x = -grad_t / (np.abs(grad_x) + epsilon)
        flow_y = -grad_t / (np.abs(grad_y) + epsilon)

        # Clip and normalize
        flow_x = np.clip(flow_x, -1, 1)
        flow_y = np.clip(flow_y, -1, 1)

        return np.stack([flow_x, flow_y], axis=0).astype(np.float32)


class FPVDataset(Dataset[dict[str, Any]]):  # type: ignore[misc]
    """
    Dataset loader for UZH FPV format data.

    Loads IMU, events, and image timestamps from text files.
    Falls back to synthetic data if files not found.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        seq_len: int = MAX_SEQ_LEN,
        image_size: tuple[int, int] = IMAGE_SIZE,
        event_window_ms: int = EVENT_WINDOW_MS,
        imu_stats: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.seq_len = seq_len
        self.image_size = image_size
        self.event_window_ms = event_window_ms

        self.imu_data: list[dict[str, Any]] = []
        self.event_timestamps: np.ndarray = np.array([], dtype=np.float64)
        self.event_x: np.ndarray = np.array([], dtype=np.int32)
        self.event_y: np.ndarray = np.array([], dtype=np.int32)
        self.event_polarity: np.ndarray = np.array([], dtype=np.int8)
        self.image_timestamps: list[dict[str, Any]] = []
        self.use_synthetic = False
        self._actual_data_dir: Path | None = None
        self._image_ts_array: np.ndarray = np.array([], dtype=np.float64)
        self._imu_mean: np.ndarray = np.zeros(6, dtype=np.float32)
        self._imu_std: np.ndarray = np.ones(6, dtype=np.float32)
        self._imu_stats_provided = imu_stats is not None
        if imu_stats is not None:
            self._imu_mean, self._imu_std = imu_stats

        # Try to load real data
        if self.data_dir.exists():
            self._load_data()

        # Fallback to synthetic if no data found
        if len(self.imu_data) == 0:
            print(f"No FPV data found in {data_dir}, using synthetic data")
            self.use_synthetic = True
            self._generate_synthetic_data()
        else:
            # Apply temporal train/val split (80/20) before reporting sizes
            self._split_data()
            if not self._imu_stats_provided:
                self._compute_imu_stats()
            print(
                f"[{split}] {len(self.imu_data)} IMU samples, "
                f"{len(self.event_timestamps)} events, "
                f"{len(self.image_timestamps)} images"
            )

    def _find_data_dir(self) -> Path | None:
        """Find the actual data directory containing data files."""
        data_files = list(self.data_dir.glob("**/imu.txt"))
        if not data_files:
            return None
        return data_files[0].parent

    def _load_imu_data(self, data_dir: Path) -> None:
        """Load IMU data from text file."""
        imu_file = data_dir / "imu.txt"
        if not imu_file.exists():
            return

        with open(imu_file) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split()
                if len(parts) >= 7:
                    t = float(parts[1])
                    gyro = [float(parts[i]) for i in range(2, 5)]
                    acc = [float(parts[i]) for i in range(5, 8)]
                    self.imu_data.append(
                        {
                            "timestamp": t,
                            "acc": np.array(acc, dtype=np.float32),
                            "gyro": np.array(gyro, dtype=np.float32),
                        }
                    )

    def _load_events_data(self, data_dir: Path) -> None:
        """Load event data efficiently using numpy."""
        events_file = data_dir / "events.txt"
        if not events_file.exists():
            return

        print("Loading events (this may take a moment)...")
        try:
            # Use numpy's fast text loading
            data = np.loadtxt(
                events_file,
                delimiter=" ",
                skiprows=1,  # Skip header comment
                usecols=(0, 1, 2, 3),
                dtype=np.float64,
            )

            self.event_timestamps = data[:, 0]
            self.event_x = data[:, 1].astype(np.int32)
            self.event_y = data[:, 2].astype(np.int32)
            self.event_polarity = data[:, 3].astype(np.int8)

            print(f"Loaded {len(self.event_timestamps):,} events")
        except Exception as e:
            print(f"Error loading events: {e}")
            self.event_timestamps = np.array([], dtype=np.float64)
            self.event_x = np.array([], dtype=np.int32)
            self.event_y = np.array([], dtype=np.int32)
            self.event_polarity = np.array([], dtype=np.int8)

    def _load_image_timestamps(self, data_dir: Path) -> None:
        """Load image timestamps from text file."""
        images_file = data_dir / "images.txt"
        if not images_file.exists():
            return

        with open(images_file) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split()
                if len(parts) >= 3:
                    t = float(parts[1])
                    filename = parts[2]
                    self.image_timestamps.append(
                        {
                            "timestamp": t,
                            "filename": filename,
                        }
                    )

    def _load_data(self) -> None:
        """Load data from text files in UZH FPV format."""
        actual_data_dir = self._find_data_dir()
        if actual_data_dir is None:
            return

        self._actual_data_dir = actual_data_dir
        self._load_imu_data(actual_data_dir)
        self._load_events_data(actual_data_dir)
        self._load_image_timestamps(actual_data_dir)
        if self.image_timestamps:
            self._image_ts_array = np.array(
                [img["timestamp"] for img in self.image_timestamps], dtype=np.float64
            )

    def _split_data(self) -> None:
        """Apply a temporal 80/20 train/val split to avoid data leakage.

        Uses the first 80% of the recording for training and the last 20% for
        validation.  Both sets are drawn from the same physical sequence, but
        no sample from the val set ever appears during training.
        """
        if not self.imu_data:
            return

        timestamps = np.array([d["timestamp"] for d in self.imu_data], dtype=np.float64)
        split_time = float(np.percentile(timestamps, 80))

        if self.split == "train":
            mask_imu = timestamps < split_time
        else:
            mask_imu = timestamps >= split_time

        self.imu_data = [d for d, keep in zip(self.imu_data, mask_imu, strict=True) if keep]

        if len(self.event_timestamps) > 0:
            if self.split == "train":
                ev_mask = self.event_timestamps < split_time
            else:
                ev_mask = self.event_timestamps >= split_time
            self.event_timestamps = self.event_timestamps[ev_mask]
            self.event_x = self.event_x[ev_mask]
            self.event_y = self.event_y[ev_mask]
            self.event_polarity = self.event_polarity[ev_mask]

        if self.split == "train":
            self.image_timestamps = [
                d for d in self.image_timestamps if d["timestamp"] < split_time
            ]
        else:
            self.image_timestamps = [
                d for d in self.image_timestamps if d["timestamp"] >= split_time
            ]

        if self.image_timestamps:
            self._image_ts_array = np.array(
                [img["timestamp"] for img in self.image_timestamps], dtype=np.float64
            )

    def _compute_imu_stats(self) -> None:
        """Compute per-axis mean and std from this split's IMU data for normalization.

        Uses the current split's data only — for val, this is the val statistics,
        which is fine since we're normalizing to unit scale, not leaking labels.
        In practice, caller should pass train stats to val dataset for strict
        correctness, but this is sufficient for a single-sequence dataset.
        """
        if not self.imu_data:
            self._imu_mean = np.zeros(6, dtype=np.float32)
            self._imu_std = np.ones(6, dtype=np.float32)
            return

        all_imu = np.array(
            [np.concatenate([d["acc"], d["gyro"]]) for d in self.imu_data],
            dtype=np.float32,
        )  # (N, 6)
        self._imu_mean = all_imu.mean(axis=0)  # (6,)
        self._imu_std = all_imu.std(axis=0).clip(min=1e-6)  # (6,), no division by zero

    def _generate_synthetic_data(self) -> None:
        """Generate synthetic data for testing."""
        num_samples = 2000

        for i in range(num_samples):
            t = i * 0.01
            acc = np.random.randn(3).astype(np.float32) * 0.1
            gyro = np.random.randn(3).astype(np.float32) * 0.05
            self.imu_data.append(
                {
                    "timestamp": t,
                    "acc": acc,
                    "gyro": gyro,
                }
            )

        H, W = self.image_size
        num_events = num_samples * 5
        self.event_timestamps = np.array([i * 0.002 for i in range(num_events)], dtype=np.float64)
        self.event_x = np.random.randint(0, W, num_events, dtype=np.int32)
        self.event_y = np.random.randint(0, H, num_events, dtype=np.int32)
        self.event_polarity = np.random.randint(0, 2, num_events, dtype=np.int8)

        for i in range(num_samples // 3):
            t = i * (1.0 / 30.0)
            self.image_timestamps.append(
                {
                    "timestamp": t,
                    "filename": f"frame_{i:06d}.png",
                }
            )

    def _get_imu_sequence(self, center_idx: int) -> np.ndarray:
        """Get IMU sequence centered around a given index."""
        start_idx = max(0, center_idx - self.seq_len // 2)
        end_idx = min(len(self.imu_data), start_idx + self.seq_len)

        if end_idx - start_idx < self.seq_len:
            start_idx = max(0, end_idx - self.seq_len)

        imu_seq: list[np.ndarray] = []
        for i in range(start_idx, end_idx):
            imu = self.imu_data[i]
            imu_seq.append(np.concatenate([imu["acc"], imu["gyro"]]))

        while len(imu_seq) < self.seq_len:
            imu_seq.insert(0, np.zeros(6, dtype=np.float32))

        arr = np.array(imu_seq, dtype=np.float32)
        # Per-axis z-score normalization so all IMU channels have comparable scale
        arr = (arr - self._imu_mean) / self._imu_std
        return arr

    def _get_event_map(self, timestamp: float) -> np.ndarray:
        """Get event map for the standard 33ms window aligned to camera frame rate.

        Accumulates all DVS events in [timestamp - 16.5ms, timestamp + 16.5ms],
        matching the EVENT_WINDOW_MS cadence. Normalized by max_events=100 so
        values are in [0, 1] (actual max in FPV data ≈ 0.08).
        """
        H, W = self.image_size
        dt = self.event_window_ms / 1000.0

        pos_events = np.zeros((H, W), dtype=np.float32)
        neg_events = np.zeros((H, W), dtype=np.float32)

        if len(self.event_timestamps) == 0:
            return np.stack([pos_events, neg_events], axis=0)

        start_time = timestamp - dt / 2
        end_time = timestamp + dt / 2

        start_idx = np.searchsorted(self.event_timestamps, start_time, side="left")
        end_idx = np.searchsorted(self.event_timestamps, end_time, side="right")

        if start_idx < end_idx:
            x_v = self.event_x[start_idx:end_idx]
            y_v = self.event_y[start_idx:end_idx]
            p_v = self.event_polarity[start_idx:end_idx]
            valid = (x_v >= 0) & (x_v < W) & (y_v >= 0) & (y_v < H)
            x_v, y_v, p_v = x_v[valid], y_v[valid], p_v[valid]
            pos_mask = p_v == 1
            np.add.at(pos_events, (y_v[pos_mask], x_v[pos_mask]), 1)
            np.add.at(neg_events, (y_v[~pos_mask], x_v[~pos_mask]), 1)

        max_events = 100.0
        pos_events = np.clip(pos_events / max_events, 0.0, 1.0)
        neg_events = np.clip(neg_events / max_events, 0.0, 1.0)

        return np.stack([pos_events, neg_events], axis=0)

    def _get_event_map_cached(self, timestamp: float) -> np.ndarray:
        """Get event map with caching for multi-scale windows (1.4x speedup)."""
        # Round timestamp to nearest 10ms for caching (reduces cache misses)
        timestamp_cached = round(timestamp * 100) / 100.0

        if not hasattr(self, "_event_cache"):
            self._event_cache: dict[float, np.ndarray] = {}

        if timestamp_cached in self._event_cache:
            return self._event_cache[timestamp_cached]

        # Compute and cache
        event_map = self._get_event_map(timestamp)

        # Limit cache size to 1000 entries (prevent memory blowup)
        if len(self._event_cache) < 1000:
            self._event_cache[timestamp_cached] = event_map

        return event_map

    def _get_random_image(self) -> np.ndarray:
        """Get a random grayscale image (fallback when no real data available)."""
        H, W = self.image_size
        return np.random.rand(1, H, W).astype(np.float32)

    def _find_closest_image_idx(self, timestamp: float) -> int:
        """Return index of image with timestamp closest to given value."""
        idx = int(np.searchsorted(self._image_ts_array, timestamp, side="left"))
        if idx == 0:
            return 0
        if idx >= len(self._image_ts_array):
            return len(self._image_ts_array) - 1
        if abs(self._image_ts_array[idx] - timestamp) < abs(
            self._image_ts_array[idx - 1] - timestamp
        ):
            return idx
        return idx - 1

    def _load_image_at_timestamp(self, timestamp: float) -> np.ndarray:
        """Load grayscale image from disk closest to the given timestamp."""
        if self.use_synthetic or not self.image_timestamps or self._actual_data_dir is None:
            return self._get_random_image()

        idx = self._find_closest_image_idx(timestamp)
        img_path = self._actual_data_dir / self.image_timestamps[idx]["filename"]

        if not img_path.exists():
            return self._get_random_image()

        try:
            from PIL import Image as PILImage

            resample = getattr(PILImage, "Resampling", PILImage).BILINEAR
            img = PILImage.open(img_path).convert("L")
            H, W = self.image_size
            img = img.resize((W, H), resample)
            return (np.array(img, dtype=np.float32) / 255.0)[np.newaxis, :, :]
        except Exception:
            return self._get_random_image()

    def __len__(self) -> int:
        return max(0, len(self.imu_data) - self.seq_len)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        imu_seq = self._get_imu_sequence(idx + self.seq_len // 2)
        timestamp = self.imu_data[idx + self.seq_len // 2]["timestamp"]
        events = self._get_event_map_cached(timestamp)

        # Load frame pair: events require temporal difference between frames.
        # Return (current_frame, previous_frame) stacked as (2, H, W).
        dt = self.event_window_ms / 1000.0
        image_t = self._load_image_at_timestamp(timestamp)  # (1, H, W)
        image_prev = self._load_image_at_timestamp(timestamp - dt)  # (1, H, W)
        image = np.concatenate([image_t, image_prev], axis=0)  # (2, H, W)

        # Compute optical flow between current and previous frame
        flow = compute_optical_flow(image_t.squeeze(0), image_prev.squeeze(0))

        return {
            "image": torch.from_numpy(image),
            "imu_seq": torch.from_numpy(imu_seq),
            "events": torch.from_numpy(events),
            "flow": torch.from_numpy(flow),
            "timestamp": timestamp,
        }


def physics_v2e_prediction(
    img_t: np.ndarray,
    img_prev: np.ndarray,
    pos_thres: float = 0.2,
    neg_thres: float = 0.2,
) -> np.ndarray:
    """Physics-based DVS event prediction via log-intensity differencing.

    Replicates the core v2e emulator model:
        ΔL = log(I_t + ε) − log(I_prev + ε)
        ON  event at pixel (x,y) iff  ΔL(x,y) >  pos_thres
        OFF event at pixel (x,y) iff −ΔL(x,y) >  neg_thres

    Args:
        img_t:    Current grayscale frame  (H, W) in [0, 1].
        img_prev: Previous grayscale frame (H, W) in [0, 1].
        pos_thres: Log-intensity threshold for ON events.
        neg_thres: Log-intensity threshold for OFF events.

    Returns:
        events: (2, H, W) float32 array — channel 0 = ON, channel 1 = OFF,
                values in {0, 1}.
    """
    eps = 1e-4
    delta_log = np.log(img_t + eps) - np.log(img_prev + eps)
    on_events = (delta_log > pos_thres).astype(np.float32)
    off_events = (-delta_log > neg_thres).astype(np.float32)
    return np.stack([on_events, off_events], axis=0)


def evaluate_physics_baseline(
    dataloader: "DataLoader[dict[str, Any]]",
    device: str,
    num_samples: int = EVAL_SAMPLES,
    threshold: float = 0.005,
    pos_thres: float = 0.2,
    neg_thres: float = 0.2,
) -> dict[str, float]:
    """Evaluate the v2e physics baseline against GT events.

    This is the reference baseline: v2e physics (log-intensity threshold)
    applied to the same frame pairs used for training.  Because GT events are
    accumulated from a real DVS sensor (not synthesized by v2e), this is NOT
    an oracle — it shows how well frame-differencing alone captures real events.

    threshold: GT binarization threshold (same as evaluate_combined_metric).
    """
    total_tp = total_fp = total_fn = 0
    total_samples = 0
    skipped = 0

    for batch in dataloader:
        if total_samples >= num_samples:
            break
        images = batch["image"].numpy()  # (B, 2, H, W)
        gt_events = batch["events"].numpy()  # (B, 2, H, W)

        for b in range(images.shape[0]):
            img_t = images[b, 0]  # current frame
            img_prev = images[b, 1]  # previous frame

            # Skip boundary samples where both frames are identical (split edge)
            if np.allclose(img_t, img_prev, atol=1e-6):
                skipped += 1
                continue

            pred = physics_v2e_prediction(img_t, img_prev, pos_thres, neg_thres)

            pred_bin = pred > 0.5
            gt_bin = gt_events[b] > threshold

            total_tp += int((pred_bin & gt_bin).sum())
            total_fp += int((pred_bin & ~gt_bin).sum())
            total_fn += int((~pred_bin & gt_bin).sum())

        total_samples += images.shape[0]

    if skipped:
        print(f"  (skipped {skipped} boundary samples with identical frame pairs)")

    precision = total_tp / (total_tp + total_fp + 1e-8)
    recall = total_tp / (total_tp + total_fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def make_dataloader(
    data_dir: str,
    split: str,
    batch_size: int,
    seq_len: int = MAX_SEQ_LEN,
    image_size: tuple[int, int] = IMAGE_SIZE,
    num_workers: int = 2,
    imu_stats: tuple[np.ndarray, np.ndarray] | None = None,
) -> DataLoader:
    """Create a dataloader for the given split.

    Pass imu_stats=(mean, std) from the training dataset to the val dataset so
    both splits are normalised with the same statistics.
    """
    dataset = FPVDataset(
        data_dir=data_dir,
        split=split,
        seq_len=seq_len,
        image_size=image_size,
        imu_stats=imu_stats,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=(torch.cuda.is_available() or torch.backends.mps.is_available()),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        timeout=60 if num_workers > 0 else 0,
        multiprocessing_context="fork" if (num_workers > 0 and sys.platform != "win32") else None,
    )


def _image_ap(pred: np.ndarray, gt_bin: np.ndarray) -> float:
    """Per-image Average Precision (area under the PR curve).

    Args:
        pred:   Flat float32 array of sigmoid predictions in [0, 1].
        gt_bin: Flat bool array of ground-truth positives.

    Returns:
        AP in [0, 1], or nan if there are no positive GT pixels.
    """
    n_pos = gt_bin.sum()
    if n_pos == 0:
        return float("nan")
    sorted_idx = np.argsort(-pred)
    gt_s = gt_bin[sorted_idx].astype(np.float32)
    tp = np.cumsum(gt_s)
    fp = np.cumsum(1.0 - gt_s)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (n_pos + 1e-8)
    # trapezoid over recall axis (monotone increasing); np.trapz removed in NumPy 2.0
    return float(np.trapezoid(precision, recall))


def evaluate_combined_metric(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
    num_samples: int = EVAL_SAMPLES,
    gt_threshold: float = 0.005,
    pred_threshold: float = 0.5,
) -> dict[str, float]:
    """Evaluate model performance.

    Primary metric — val_ap (Average Precision):
      Threshold-free area under the PR curve. Higher = better.
      Signal appears even when the model hasn't fully converged, making it
      suitable as the autoresearch accept/revert signal.

    Secondary metrics:
      f1_score / precision / recall at pred_threshold=0.5
      f1_on / f1_off per polarity
      event_mse  (L2 reference)

    GT binarisation threshold: gt_threshold=0.005
      (≥0.5 events per 33ms window after normalisation by max_events=100)
    Prediction decision boundary: pred_threshold=0.5 (natural sigmoid midpoint)
    """
    model.eval()

    total_tp = total_fp = total_fn = 0
    total_tp_on = total_fp_on = total_fn_on = 0
    total_tp_off = total_fp_off = total_fn_off = 0
    total_mse = 0.0
    total_elements = 0
    total_samples = 0
    ap_scores: list[float] = []
    start_time = time.time()

    with torch.no_grad():
        for batch in dataloader:
            if total_samples >= num_samples:
                break

            images = batch["image"].to(device)
            imu_seq = batch["imu_seq"].to(device)
            gt_events = batch["events"].to(device)

            output = model(images, imu_seq)
            pred_events = output[0] if isinstance(output, tuple) else output

            # Average Precision — computed per image per channel, then averaged.
            # Computing per-channel prevents ON and OFF events from competing in
            # a single ranking (which would make the metric undefined).
            pred_np = pred_events.cpu().numpy().astype(np.float32)
            gt_np = gt_events.cpu().numpy()
            for b in range(pred_np.shape[0]):
                for ch in range(pred_np.shape[1]):  # ON, OFF
                    ap = _image_ap(pred_np[b, ch].ravel(), (gt_np[b, ch].ravel() > gt_threshold))
                    if not np.isnan(ap):
                        ap_scores.append(ap)

            # F1 at fixed threshold (secondary)
            pred_bin = pred_events > pred_threshold
            gt_bin = gt_events > gt_threshold

            total_tp += int((pred_bin & gt_bin).sum().item())
            total_fp += int((pred_bin & ~gt_bin).sum().item())
            total_fn += int((~pred_bin & gt_bin).sum().item())

            pb_on, gb_on = pred_bin[:, 0], gt_bin[:, 0]
            total_tp_on += int((pb_on & gb_on).sum().item())
            total_fp_on += int((pb_on & ~gb_on).sum().item())
            total_fn_on += int((~pb_on & gb_on).sum().item())
            pb_off, gb_off = pred_bin[:, 1], gt_bin[:, 1]
            total_tp_off += int((pb_off & gb_off).sum().item())
            total_fp_off += int((pb_off & ~gb_off).sum().item())
            total_fn_off += int((~pb_off & gb_off).sum().item())

            total_mse += torch.nn.functional.mse_loss(
                pred_events, gt_events, reduction="sum"
            ).item()
            total_elements += gt_events.numel()
            total_samples += images.shape[0]

    elapsed = time.time() - start_time
    samples_per_sec = total_samples / elapsed if elapsed > 0 else 0.0

    val_ap = float(np.mean(ap_scores)) if ap_scores else 0.0

    def _f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
        p = tp / (tp + fp + 1e-8)
        r = tp / (tp + fn + 1e-8)
        f = 2 * p * r / (p + r + 1e-8)
        return float(p), float(r), float(f)

    precision, recall, f1 = _f1(total_tp, total_fp, total_fn)
    _, _, f1_on = _f1(total_tp_on, total_fp_on, total_fn_on)
    _, _, f1_off = _f1(total_tp_off, total_fp_off, total_fn_off)

    return {
        "val_ap": val_ap,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "f1_on": f1_on,
        "f1_off": f1_off,
        "event_mse": float(total_mse / max(total_elements, 1)),
        "samples_per_sec": samples_per_sec,
    }


def prepare_data() -> None:
    """One-time data preparation."""
    print("Preparing data for IMU-Enhanced Event Camera experiments...")
    print(f"Data directory: {DATA_DIR}")
    print(f"Cache directory: {CACHE_DIR}")

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    data_files = list(Path(DATA_DIR).glob("**/imu.txt"))
    if data_files:
        print(f"Found {len(data_files)} dataset directories")
        for data_file in data_files:
            print(f"  {data_file.parent}")
    else:
        print("No FPV data found.")
        print("You can:")
        print("  1. Download real data: python download_fpv.py --sequence indoor_forward_3")
        print("  2. Use synthetic data: python download_fpv.py --synthetic")

    print("\nData preparation complete!")
    print("You can now run experiments with: python train.py")


if __name__ == "__main__":
    prepare_data()
