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
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 50  # IMU sequence length
IMAGE_SIZE = (260, 346)  # DAVIS346 resolution
TIME_BUDGET = 600  # 10 minutes per experiment (in seconds)
EVAL_SAMPLES = 1000  # Number of samples for evaluation
EVENT_WINDOW_MS = 33  # Event accumulation window (30 Hz)

# Data directory
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "fpv")
CACHE_DIR = os.path.join(os.path.dirname(__file__), "data", "cache")


class FPVDataset(Dataset):
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
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.seq_len = seq_len
        self.image_size = image_size
        self.event_window_ms = event_window_ms

        self.imu_data = []
        self.event_data = []
        self.image_timestamps = []
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
                f"{len(self.event_data)} events, "
                f"{len(self.image_timestamps)} images"
            )

    def _load_data(self):
        """Load data from text files in UZH FPV format."""
        # Find the actual data directory (might be in a subdirectory)
        data_files = list(self.data_dir.glob("**/imu.txt"))
        if not data_files:
            return

        # Use the first directory that has imu.txt
        actual_data_dir = data_files[0].parent

        # Load IMU data
        # Format: timestamp ang_vel_x ang_vel_y ang_vel_z lin_acc_x lin_acc_y lin_acc_z
        imu_file = actual_data_dir / "imu.txt"
        if imu_file.exists():
            with open(imu_file) as f:
                for line in f:
                    if line.startswith("#"):
                        continue
                    parts = line.strip().split()
                    if len(parts) >= 7:
                        # Skip the first column (index), use second as timestamp
                        t = float(parts[1])  # timestamp
                        gyro = [
                            float(parts[i]) for i in range(2, 5)
                        ]  # ang_vel_x, ang_vel_y, ang_vel_z
                        acc = [
                            float(parts[i]) for i in range(5, 8)
                        ]  # lin_acc_x, lin_acc_y, lin_acc_z
                        self.imu_data.append(
                            {
                                "timestamp": t,
                                "acc": np.array(acc, dtype=np.float32),
                                "gyro": np.array(gyro, dtype=np.float32),
                            }
                        )

        # Load event data
        # Format: timestamp x y polarity
        events_file = actual_data_dir / "events.txt"
        if events_file.exists():
            with open(events_file) as f:
                for line in f:
                    if line.startswith("#"):
                        continue
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        t = float(parts[0])
                        x = int(parts[1])
                        y = int(parts[2])
                        p = int(parts[3])
                        self.event_data.append(
                            {
                                "timestamp": t,
                                "x": x,
                                "y": y,
                                "polarity": p,
                            }
                        )

        # Load image timestamps
        # Format: id timestamp image_name
        images_file = actual_data_dir / "images.txt"
        if images_file.exists():
            with open(images_file) as f:
                for line in f:
                    if line.startswith("#"):
                        continue
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        # Skip the first column (id), use second as timestamp
                        t = float(parts[1])  # timestamp
                        filename = parts[2]  # image_name
                        self.image_timestamps.append(
                            {
                                "timestamp": t,
                                "filename": filename,
                            }
                        )

    def _generate_synthetic_data(self):
        """Generate synthetic data for testing."""
        num_samples = 2000

        # Generate IMU data (100 Hz)
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

        # Generate event data
        H, W = self.image_size
        for i in range(num_samples * 5):
            t = i * 0.002  # ~500 Hz
            x = np.random.randint(0, W)
            y = np.random.randint(0, H)
            p = np.random.randint(0, 2)
            self.event_data.append(
                {
                    "timestamp": t,
                    "x": x,
                    "y": y,
                    "polarity": p,
                }
            )

        # Generate image timestamps (30 Hz)
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

        imu_seq = []
        for i in range(start_idx, end_idx):
            imu = self.imu_data[i]
            imu_seq.append(np.concatenate([imu["acc"], imu["gyro"]]))

        # Pad if necessary
        while len(imu_seq) < self.seq_len:
            imu_seq.insert(0, np.zeros(6, dtype=np.float32))

        return np.array(imu_seq, dtype=np.float32)

    def _get_event_map(self, timestamp: float) -> np.ndarray:
        """Get event map around a given timestamp."""
        H, W = self.image_size
        dt = self.event_window_ms / 1000.0

        pos_events = np.zeros((H, W), dtype=np.float32)
        neg_events = np.zeros((H, W), dtype=np.float32)

        for event in self.event_data:
            t = event["timestamp"]
            if timestamp - dt / 2 <= t <= timestamp + dt / 2:
                x, y = event["x"], event["y"]
                if 0 <= x < W and 0 <= y < H:
                    if event["polarity"] == 1:
                        pos_events[y, x] += 1
                    else:
                        neg_events[y, x] += 1

        return np.stack([pos_events, neg_events], axis=0)

    def _get_random_image(self) -> np.ndarray:
        """Get a random grayscale image (placeholder)."""
        H, W = self.image_size
        # In real usage, this would load actual images
        # For now, generate random grayscale
        return np.random.rand(1, H, W).astype(np.float32)

    def __len__(self):
        return max(0, len(self.imu_data) - self.seq_len)

    def __getitem__(self, idx):
        # Get IMU sequence
        imu_seq = self._get_imu_sequence(idx + self.seq_len // 2)

        # Get event map
        timestamp = self.imu_data[idx + self.seq_len // 2]["timestamp"]
        events = self._get_event_map(timestamp)

        # Get image
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
    num_workers: int = 0,
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
        pin_memory=True if torch.cuda.is_available() else False,
    )


def evaluate_event_bpb(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
    num_samples: int = EVAL_SAMPLES,
) -> float:
    """
    Evaluate model using bits per byte (BPB) metric for event prediction.

    Lower is better. This is analogous to val_bpb in autoresearch-mlx.
    """
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

            # Forward pass
            pred_events = model(images, imu_seq)

            # Compute loss (MSE on event counts)
            loss = torch.nn.functional.mse_loss(pred_events, gt_events, reduction="sum")

            # Count "bytes" (pixels * channels)
            batch_bytes = gt_events.numel()

            total_loss += loss.item()
            total_bytes += batch_bytes
            samples_processed += images.shape[0]

    if total_bytes == 0:
        return float("inf")

    # Convert to bits per byte
    # MSE loss is in nats, convert to bits
    bpb = (total_loss / total_bytes) / np.log(2)
    return bpb


def evaluate_combined_metric(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
    num_samples: int = EVAL_SAMPLES,
) -> dict[str, float]:
    """
    Evaluate model with combined metrics.

    Returns:
        Dictionary with:
        - 'event_bpb': Bits per byte for event prediction
        - 'event_mse': Mean squared error for events
        - 'samples_per_sec': Throughput
    """
    model.eval()

    total_event_loss = 0.0
    total_samples = 0
    start_time = time.time()

    with torch.no_grad():
        for batch in dataloader:
            if total_samples >= num_samples:
                break

            images = batch["image"].to(device)
            imu_seq = batch["imu_seq"].to(device)
            gt_events = batch["events"].to(device)

            pred_events = model(images, imu_seq)

            event_loss = torch.nn.functional.mse_loss(pred_events, gt_events, reduction="sum")

            total_event_loss += event_loss.item()
            total_samples += images.shape[0]

    elapsed = time.time() - start_time
    samples_per_sec = total_samples / elapsed if elapsed > 0 else 0

    avg_event_mse = total_event_loss / (total_samples * gt_events[0].numel())
    event_bpb = avg_event_mse / np.log(2)

    return {
        "event_bpb": event_bpb,
        "event_mse": avg_event_mse,
        "samples_per_sec": samples_per_sec,
    }


def prepare_data():
    """One-time data preparation."""
    print("Preparing data for IMU-Enhanced Event Camera experiments...")
    print(f"Data directory: {DATA_DIR}")
    print(f"Cache directory: {CACHE_DIR}")

    # Create directories
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Check for existing data
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
