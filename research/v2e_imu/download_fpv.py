"""
Download UZH FPV Drone Racing Dataset sequences.

Dataset: https://fpv.ifi.uzh.ch/datasets/
Format: ZIP files containing text-based sensor data

Usage:
    python download_fpv.py --sequence indoor_forward_3 --output data/
    python download_fpv.py --list
    python download_fpv.py --synthetic  # Create synthetic data for testing
"""

import argparse
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

# UZH FPV Dataset base URLs
BASE_URL_V3 = "http://rpg.ifi.uzh.ch/datasets/uzh-fpv-newer-versions/v3"
BASE_URL_RAW = "http://rpg.ifi.uzh.ch/datasets/uzh-fpv-newer-versions/raw"

# Sequences with ground truth (recommended for training)
SEQUENCES_WITH_GT = {
    # Indoor forward facing
    "indoor_forward_3": {
        "description": "Easy, 54s, GT available",
        "davis_zip": f"{BASE_URL_V3}/indoor_forward_3_davis_with_gt.zip",
        "size_mb": 413.9,
    },
    "indoor_forward_5": {
        "description": "Medium, 50s, GT available",
        "davis_zip": f"{BASE_URL_V3}/indoor_forward_5_davis_with_gt.zip",
        "size_mb": 655.3,
    },
    "indoor_forward_6": {
        "description": "Medium, 33s, GT available",
        "davis_zip": f"{BASE_URL_V3}/indoor_forward_6_davis_with_gt.zip",
        "size_mb": 365.1,
    },
    "indoor_forward_7": {
        "description": "Hard, 73s, GT available",
        "davis_zip": f"{BASE_URL_V3}/indoor_forward_7_davis_with_gt.zip",
        "size_mb": 559.5,
    },
    "indoor_forward_9": {
        "description": "Easy, 34s, GT available",
        "davis_zip": f"{BASE_URL_V3}/indoor_forward_9_davis_with_gt.zip",
        "size_mb": 365.6,
    },
    "indoor_forward_10": {
        "description": "Easy, 33s, GT available",
        "davis_zip": f"{BASE_URL_V3}/indoor_forward_10_davis_with_gt.zip",
        "size_mb": 327.9,
    },
    # Indoor 45 degree downward facing
    "indoor_45_2": {
        "description": "Easy, 56s, GT available",
        "davis_zip": f"{BASE_URL_V3}/indoor_45_2_davis_with_gt.zip",
        "size_mb": 602.5,
    },
    "indoor_45_4": {
        "description": "Easy, 47s, GT available",
        "davis_zip": f"{BASE_URL_V3}/indoor_45_4_davis_with_gt.zip",
        "size_mb": 496.3,
    },
    "indoor_45_9": {
        "description": "Medium, 40s, GT available",
        "davis_zip": f"{BASE_URL_V3}/indoor_45_9_davis_with_gt.zip",
        "size_mb": 370.8,
    },
    # Outdoor forward facing
    "outdoor_forward_1": {
        "description": "Easy, 49s, GT available",
        "davis_zip": f"{BASE_URL_V3}/outdoor_forward_1_davis_with_gt.zip",
        "size_mb": 358.6,
    },
    "outdoor_forward_3": {
        "description": "Medium, 93s, GT available",
        "davis_zip": f"{BASE_URL_V3}/outdoor_forward_3_davis_with_gt.zip",
        "size_mb": 637.2,
    },
    "outdoor_forward_5": {
        "description": "Hard, 22s, GT available",
        "davis_zip": f"{BASE_URL_V3}/outdoor_forward_5_davis_with_gt.zip",
        "size_mb": 342.7,
    },
}

# Small sequences for quick experiments
SMALL_SEQUENCES = [
    "indoor_forward_3",  # 413.9 MB
    "indoor_forward_9",  # 365.6 MB
    "indoor_forward_10",  # 327.9 MB
]


class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: str) -> bool:
    """Download a file with progress bar."""
    try:
        with DownloadProgressBar(
            unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
        ) as t:
            urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
        return True
    except Exception as e:
        print(f"  Error downloading {url}: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return False


def extract_zip(zip_path: str, extract_dir: str):
    """Extract a ZIP file."""
    print(f"  Extracting {zip_path} to {extract_dir}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)
    os.remove(zip_path)  # Remove ZIP after extraction


def download_sequence(sequence_name: str, output_dir: str):
    """
    Download a UZH FPV sequence.

    Args:
        sequence_name: Sequence name (e.g., 'indoor_forward_3')
        output_dir: Output directory
    """
    if sequence_name not in SEQUENCES_WITH_GT:
        print(f"Unknown sequence: {sequence_name}")
        print("Available sequences:")
        for name, info in SEQUENCES_WITH_GT.items():
            print(f"  {name}: {info['description']}")
        return False

    seq_info = SEQUENCES_WITH_GT[sequence_name]
    seq_dir = Path(output_dir) / sequence_name
    seq_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading sequence: {sequence_name}")
    print(f"  Description: {seq_info['description']}")
    print(f"  Estimated size: {seq_info['size_mb']:.1f} MB")
    print(f"  Output directory: {seq_dir}")

    # Check if already downloaded
    if any(seq_dir.glob("*.txt")) or any(seq_dir.glob("*.bag")):
        print(f"  Data already exists in {seq_dir}")
        return True

    # Download DAVIS data (events + images + IMU + GT)
    davis_zip = seq_dir / f"{sequence_name}_davis_with_gt.zip"
    davis_url = seq_info["davis_zip"]
    print(f"  Downloading DAVIS data from {davis_url}")

    if not download_file(davis_url, str(davis_zip)):
        print("  Failed to download DAVIS data")
        return False

    extract_zip(str(davis_zip), str(seq_dir))
    print("  Download complete!")
    return True


def create_synthetic_fpv_data(output_dir: str, num_samples: int = 1000):
    """
    Create synthetic FPV-like data for testing when download is not available.

    Generates:
    - IMU data (acc_xyz, gyro_xyz)
    - Event data (x, y, timestamp, polarity)
    - Image timestamps
    """
    import numpy as np

    sample_dir = Path(output_dir) / "synthetic_fpv"
    sample_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating synthetic FPV data with {num_samples} samples...")

    # Generate IMU data (100 Hz)
    imu_file = sample_dir / "imu.txt"
    imu_data = []
    for i in range(num_samples):
        t = i * 0.01  # 100 Hz
        acc = np.random.randn(3) * 0.1  # Small accelerations
        gyro = np.random.randn(3) * 0.05  # Small rotations
        imu_data.append(
            f"{t:.6f} {acc[0]:.6f} {acc[1]:.6f} {acc[2]:.6f} {gyro[0]:.6f} {gyro[1]:.6f} {gyro[2]:.6f}"
        )

    with open(imu_file, "w") as f:
        f.write("# timestamp ax ay az wx wy wz\n")
        f.write("\n".join(imu_data))
    print(f"  Created IMU data: {imu_file}")

    # Generate event data (simulated events)
    events_file = sample_dir / "events.txt"
    events = []
    width, height = 346, 260  # DAVIS346 resolution
    for i in range(num_samples * 10):  # More events than IMU samples
        t = i * 0.001  # Events at ~kHz rate
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        p = np.random.randint(0, 2)  # Polarity
        events.append(f"{t:.6f} {x} {y} {p}")

    with open(events_file, "w") as f:
        f.write("# timestamp x y polarity\n")
        f.write("\n".join(events))
    print(f"  Created event data: {events_file}")

    # Generate image timestamps (30 Hz)
    images_file = sample_dir / "images.txt"
    images = []
    for i in range(num_samples // 3):  # 30 Hz
        t = i * (1.0 / 30.0)
        images.append(f"{t:.6f} frame_{i:06d}.png")

    with open(images_file, "w") as f:
        f.write("# timestamp filename\n")
        f.write("\n".join(images))
    print(f"  Created image timestamps: {images_file}")

    print(f"  Synthetic data created at {sample_dir}")
    return str(sample_dir)


def list_sequences():
    """List available sequences."""
    print("Available UZH FPV sequences for download:")
    print()
    for name, info in SEQUENCES_WITH_GT.items():
        print(f"  {name}")
        print(f"    Description: {info['description']}")
        print(f"    Size: {info['size_mb']:.1f} MB")
        print()


def main():
    parser = argparse.ArgumentParser(description="Download UZH FPV dataset")
    parser.add_argument(
        "--sequence", type=str, default="indoor_forward_3", help="Sequence name to download"
    )
    parser.add_argument("--output", type=str, default="data/fpv/", help="Output directory")
    parser.add_argument("--list", action="store_true", help="List available sequences")
    parser.add_argument(
        "--synthetic", action="store_true", help="Create synthetic data instead of downloading"
    )

    args = parser.parse_args()

    if args.list:
        list_sequences()
        return

    if args.synthetic:
        create_synthetic_fpv_data(args.output)
        return

    # Download sequence
    success = download_sequence(
        args.sequence,
        args.output,
    )

    if success:
        print(f"\nDownload complete! Data saved to {args.output}/{args.sequence}")
        print("You can now use this data for training.")
    else:
        print("\nDownload failed. You can try:")
        print("  1. Use --synthetic to create synthetic data for testing")
        print("  2. Manually download from https://fpv.ifi.uzh.ch/datasets/")
        sys.exit(1)


if __name__ == "__main__":
    main()
