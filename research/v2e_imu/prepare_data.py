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
EVAL_SAMPLES = 100  # Number of samples for evaluation (reduced for speed)
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

        # Try to load real data
        if self.data_dir.exists():
            self._load_data()

        # Fallback to synthetic if no data found
        if len(self.imu_data) == 0:
            print(f"No FPV data found in {data_dir}, using synthetic data")
            self.use_synthetic = True
            self._generate_synthetic_data()
        else:
            print(
                f"Loaded {len(self.imu_data)} IMU samples, "
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

        return np.array(imu_seq, dtype=np.float32)

    def _get_event_map(self, timestamp: float) -> np.ndarray:
        """Get event map around a given timestamp using multi-scale windows."""
        H, W = self.image_size

        # Multi-scale temporal windows (captures fast + slow events)
        windows_ms = [10, 33, 100]  # 10ms, 33ms, 100ms

        all_pos_maps = []
        all_neg_maps = []

        for dt_ms in windows_ms:
            dt = dt_ms / 1000.0

            pos_events = np.zeros((H, W), dtype=np.float32)
            neg_events = np.zeros((H, W), dtype=np.float32)

            if len(self.event_timestamps) == 0:
                all_pos_maps.append(pos_events)
                all_neg_maps.append(neg_events)
                continue

            start_time = timestamp - dt / 2
            end_time = timestamp + dt / 2

            start_idx = np.searchsorted(self.event_timestamps, start_time, side="left")
            end_idx = np.searchsorted(self.event_timestamps, end_time, side="right")

            if start_idx < end_idx:
                x_v = self.event_x[start_idx:end_idx]
                y_v = self.event_y[start_idx:end_idx]
                p_v = self.event_polarity[start_idx:end_idx]
                valid = (x_v >= 0) & (x_v < W) & (y_v >= 0) & (y_v < H)
                x_v = x_v[valid]
                y_v = y_v[valid]
                p_v = p_v[valid]
                pos_mask = p_v == 1
                np.add.at(pos_events, (y_v[pos_mask], x_v[pos_mask]), 1)
                np.add.at(neg_events, (y_v[~pos_mask], x_v[~pos_mask]), 1)

            # Normalize for this window size
            max_events = 100.0 * (dt_ms / 33.0)  # Scale max with window size
            pos_events = np.clip(pos_events / max_events, 0.0, 1.0)
            neg_events = np.clip(neg_events / max_events, 0.0, 1.0)

            all_pos_maps.append(pos_events)
            all_neg_maps.append(neg_events)

        # Average across scales (simple fusion)
        pos_events = np.mean(np.stack(all_pos_maps), axis=0)
        neg_events = np.mean(np.stack(all_neg_maps), axis=0)

        return np.stack([pos_events, neg_events], axis=0)

    def _get_event_map_cached(self, timestamp: float) -> np.ndarray:
        """Get event map with caching for multi-scale windows (1.4x speedup)."""
        # Round timestamp to nearest 10ms for caching (reduces cache misses)
        timestamp_cached = round(timestamp * 100) / 100.0

        if not hasattr(self, "_event_cache"):
            self._event_cache = {}

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

        # Extract DINO features from current frame
        if HAS_TIMM:
            dino_features = get_dino_features(image_t.squeeze(0), "cpu")
        else:
            dino_features = np.zeros(384, dtype=np.float32)  # fallback

        return {
            "image": torch.from_numpy(image),
            "imu_seq": torch.from_numpy(imu_seq),
            "events": torch.from_numpy(events),
            "flow": torch.from_numpy(flow),
            "dino": torch.from_numpy(dino_features),
            "timestamp": timestamp,
        }


def make_dataloader(
    data_dir: str,
    split: str,
    batch_size: int,
    seq_len: int = MAX_SEQ_LEN,
    image_size: tuple[int, int] = IMAGE_SIZE,
    num_workers: int = 2,  # Use 2 workers (safer for macOS, avoids semaphore issues)
) -> DataLoader:
    """Create a dataloader for the given split."""
    dataset = FPVDataset(
        data_dir=data_dir,
        split=split,
        seq_len=seq_len,
        image_size=image_size,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        timeout=60 if num_workers > 0 else 0,
        multiprocessing_context="fork" if (num_workers > 0 and sys.platform != "win32") else None,
    )


def evaluate_combined_metric(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
    num_samples: int = EVAL_SAMPLES,
) -> dict[str, float]:
    """Evaluate model with improved Multimodal spatiotemporal metrics.

    Metrics:
    - event_bpb: Bits per byte for event prediction (lower is better)
    - event_mse: Mean squared error on events
    - event_rate_error: How well event rate matches motion magnitude
    - depth_motion_corr: Correlation between predicted depth and IMU motion
    """
    model.eval()

    total_poisson_nll = 0.0
    total_mse = 0.0
    total_elements = 0
    total_rate_error = 0.0
    total_depth_motion_error = 0.0
    total_samples = 0
    start_time = time.time()
    pred_depth = None

    with torch.no_grad():
        for batch in dataloader:
            if total_samples >= num_samples:
                break

            images = batch["image"].to(device)
            imu_seq = batch["imu_seq"].to(device)
            gt_events = batch["events"].to(device)

            output = model(images, imu_seq)
            if isinstance(output, tuple):
                pred_events, pred_depth = output
            else:
                pred_events = output
                pred_depth = None

            # Poisson NLL (consistent with training loss)
            gt_counts = gt_events * 100.0
            pred_counts = pred_events * 100.0
            poisson_nll = pred_counts - gt_counts * torch.log(pred_counts + 1e-6)
            total_poisson_nll += poisson_nll.sum().item()

            # MSE
            total_mse += torch.nn.functional.mse_loss(
                pred_events, gt_events, reduction="sum"
            ).item()
            total_elements += gt_events.numel()

            # Event rate error
            pred_event_rate = pred_events.abs().sum(dim=(1, 2, 3))
            gt_event_rate = gt_events.abs().sum(dim=(1, 2, 3))
            total_rate_error += torch.nn.functional.mse_loss(pred_event_rate, gt_event_rate).item()

            # Depth-motion consistency
            if pred_depth is not None:
                imu_motion = imu_seq[:, :, :3].norm(dim=-1).mean(dim=1)
                total_depth_motion_error += torch.nn.functional.mse_loss(
                    pred_depth.squeeze(1), imu_motion
                ).item()

            total_samples += images.shape[0]

    elapsed = time.time() - start_time
    samples_per_sec = total_samples / elapsed if elapsed > 0 else 0

    # event_bpb: Poisson NLL per element in bits (consistent with training)
    event_bpb = (total_poisson_nll / total_elements) / np.log(2)
    avg_event_mse = total_mse / total_elements
    avg_rate_error = total_rate_error / total_samples
    avg_depth_motion_error = (
        total_depth_motion_error / total_samples if pred_depth is not None else 0.0
    )

    return {
        "event_bpb": float(event_bpb),
        "event_mse": float(avg_event_mse),
        "event_rate_error": float(avg_rate_error),
        "depth_motion_error": float(avg_depth_motion_error),
        "samples_per_sec": float(samples_per_sec),
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
