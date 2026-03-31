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

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 50  # IMU sequence length
IMAGE_SIZE = (260, 346)  # DAVIS346 resolution
TIME_BUDGET = 600  # 10 minutes per experiment (in seconds)
EVAL_SAMPLES = 100  # Number of samples for evaluation (reduced for speed)
EVENT_WINDOW_MS = 33  # Event accumulation window (30 Hz)

# Data directory
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "fpv")
CACHE_DIR = os.path.join(os.path.dirname(__file__), "data", "cache")


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

        self._load_imu_data(actual_data_dir)
        self._load_events_data(actual_data_dir)
        self._load_image_timestamps(actual_data_dir)

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
        """Get event map around a given timestamp using binary search."""
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

        for i in range(start_idx, end_idx):
            x = self.event_x[i]
            y = self.event_y[i]
            if 0 <= x < W and 0 <= y < H:
                if self.event_polarity[i] == 1:
                    pos_events[y, x] += 1
                else:
                    neg_events[y, x] += 1

        return np.stack([pos_events, neg_events], axis=0)

    def _get_random_image(self) -> np.ndarray:
        """Get a random grayscale image (placeholder)."""
        H, W = self.image_size
        return np.random.rand(1, H, W).astype(np.float32)

    def __len__(self) -> int:
        return max(0, len(self.imu_data) - self.seq_len)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        imu_seq = self._get_imu_sequence(idx + self.seq_len // 2)
        timestamp = self.imu_data[idx + self.seq_len // 2]["timestamp"]
        events = self._get_event_map(timestamp)
        image = self._get_random_image()

        return {
            "image": torch.from_numpy(image),
            "imu_seq": torch.from_numpy(imu_seq),
            "events": torch.from_numpy(events),
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
        persistent_workers=(num_workers > 0),  # Keep workers alive between epochs
        prefetch_factor=2 if num_workers > 0 else None,  # Pre-fetch batches
        timeout=60,  # Timeout for data loading
        multiprocessing_context="fork" if sys.platform != "win32" else None,  # Better for macOS
    )


def evaluate_event_bpb(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
    num_samples: int = EVAL_SAMPLES,
) -> float:
    """Evaluate model using bits per byte (BPB) metric for event prediction."""
    model.eval()

    total_loss = 0.0
    total_bytes = 0
    samples_processed = 0

    with torch.no_grad():
        for batch in dataloader:
            if samples_processed >= num_samples:
                break

            images = batch["image"].to(device)
            imu_seq = batch["imu_seq"].to(device)
            gt_events = batch["events"].to(device)

            pred_events = model(images, imu_seq)
            loss = torch.nn.functional.mse_loss(pred_events, gt_events, reduction="sum")

            batch_bytes = gt_events.numel()
            total_loss += loss.item()
            total_bytes += batch_bytes
            samples_processed += images.shape[0]

    if total_bytes == 0:
        return float("inf")

    bpb = (total_loss / total_bytes) / np.log(2)
    return float(bpb)


def evaluate_combined_metric(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
    num_samples: int = EVAL_SAMPLES,
) -> dict[str, float]:
    """Evaluate model with improved 3D-aware metrics.

    Metrics:
    - event_bpb: Bits per byte for event prediction (lower is better)
    - event_mse: Mean squared error on events
    - event_rate_error: How well event rate matches motion magnitude
    - depth_motion_corr: Correlation between predicted depth and IMU motion
    """
    model.eval()

    total_event_loss = 0.0
    total_rate_error = 0.0
    total_depth_motion_error = 0.0
    total_samples = 0
    start_time = time.time()

    with torch.no_grad():
        for batch in dataloader:
            if total_samples >= num_samples:
                break

            images = batch["image"].to(device)
            imu_seq = batch["imu_seq"].to(device)
            gt_events = batch["events"].to(device)

            # Handle both old (events only) and new (events, depth) model outputs
            output = model(images, imu_seq)
            if isinstance(output, tuple):
                pred_events, pred_depth = output
            else:
                pred_events = output
                pred_depth = None

            # Event prediction loss
            event_loss = torch.nn.functional.mse_loss(pred_events, gt_events, reduction="sum")
            total_event_loss += event_loss.item()

            # Event rate error: do events correlate with motion?
            pred_event_rate = pred_events.abs().sum(dim=(1, 2, 3))  # (B,)
            gt_event_rate = gt_events.abs().sum(dim=(1, 2, 3))  # (B,)
            imu_motion = imu_seq[:, :, :3].norm(dim=-1).mean(dim=1)  # (B,)

            rate_error = torch.nn.functional.mse_loss(pred_event_rate, gt_event_rate)
            total_rate_error += rate_error.item()

            # Depth-motion consistency (if model predicts depth)
            if pred_depth is not None:
                depth_motion_error = torch.nn.functional.mse_loss(pred_depth.squeeze(1), imu_motion)
                total_depth_motion_error += depth_motion_error.item()

            total_samples += images.shape[0]

    elapsed = time.time() - start_time
    samples_per_sec = total_samples / elapsed if elapsed > 0 else 0

    # Primary metrics
    avg_event_mse = total_event_loss / (total_samples * gt_events[0].numel())
    event_bpb = avg_event_mse / np.log(2)
    avg_rate_error = total_rate_error / total_samples

    # Secondary metrics (3D-awareness)
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
