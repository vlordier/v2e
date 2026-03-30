"""
Download DSEC dataset sequences.

DSEC provides events in HDF5 format, images, and IMU data.
This script downloads a small subset for research.

Usage:
    python download_dsec.py --sequence thun_00_a --output data/
"""

import argparse
import urllib.request
from pathlib import Path

from tqdm import tqdm

# Small sequences for quick experiments
SMALL_SEQUENCES = [
    "thun_00_a",  # 92s, with optical flow GT
    "interlaken_00_c",  # 113s
    "zurich_city_00_a",  # 127s
]

BASE_URL = "https://download.ifi.uzh.ch/rpg/DSEC/train/"


class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: str):
    """Download a file with progress bar."""
    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download_sequence(sequence: str, output_dir: str, components: list = None):
    """
    Download a DSEC sequence.

    Args:
        sequence: Sequence name (e.g., 'thun_00_a')
        output_dir: Output directory
        components: List of components to download. Options:
            - 'events_left': Left event camera events (HDF5)
            - 'events_right': Right event camera events (HDF5)
            - 'images_rectified_left': Rectified RGB images
            - 'image_timestamps': Image timestamps
            - 'imu': IMU data (if available)
    """
    if components is None:
        # Default: download events and timestamps only (smaller)
        components = ["events_left", "image_timestamps"]

    seq_dir = Path(output_dir) / sequence
    seq_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading sequence: {sequence}")
    print(f"Output directory: {seq_dir}")

    for component in components:
        component_dir = seq_dir / component
        component_dir.mkdir(exist_ok=True)

        # Check if already downloaded
        if any(component_dir.iterdir()):
            print(f"  {component} already exists, skipping...")
            continue

        # Construct download URL
        if component == "events_left":
            url = f"{BASE_URL}{sequence}/{sequence}_events_left.zip"
        elif component == "events_right":
            url = f"{BASE_URL}{sequence}/{sequence}_events_right.zip"
        elif component == "images_rectified_left":
            url = f"{BASE_URL}{sequence}/{sequence}_images_rectified_left.zip"
        elif component == "image_timestamps":
            url = f"{BASE_URL}{sequence}/{sequence}_image_timestamps.txt"
        elif component == "imu":
            # IMU data might be in a different format
            print("  IMU data: Check DSEC website for format details")
            continue
        else:
            print(f"  Unknown component: {component}")
            continue

        # Download
        output_path = component_dir / url.split("/")[-1]
        print(f"  Downloading {component} from {url}")
        try:
            download_file(url, str(output_path))
            print(f"  Saved to {output_path}")
        except Exception as e:
            print(f"  Error downloading {component}: {e}")
            print(f"  You may need to download manually from: {url}")


def create_sample_event_data(output_dir: str, num_events: int = 100000):
    """
    Create synthetic event data for testing when download is not available.

    This creates a simple HDF5 file with events in DSEC format.
    """
    import h5py
    import numpy as np

    sample_dir = Path(output_dir) / "sample"
    sample_dir.mkdir(parents=True, exist_ok=True)

    h5_path = sample_dir / "events.h5"

    if h5_path.exists():
        print(f"Sample data already exists at {h5_path}")
        return str(h5_path)

    print(f"Creating synthetic event data with {num_events} events...")

    # Generate synthetic events
    width, height = 640, 480
    x = np.random.randint(0, width, num_events, dtype=np.uint16)
    y = np.random.randint(0, height, num_events, dtype=np.uint16)
    t = np.sort(np.random.randint(0, 10_000_000, num_events, dtype=np.uint64))  # 10 seconds
    p = np.random.randint(0, 2, num_events, dtype=np.uint8)

    # Create HDF5 file in DSEC format
    with h5py.File(h5_path, "w") as f:
        events_group = f.create_group("events")
        events_group.create_dataset("x", data=x, compression="gzip")
        events_group.create_dataset("y", data=y, compression="gzip")
        events_group.create_dataset("t", data=t, compression="gzip")
        events_group.create_dataset("p", data=p, compression="gzip")

        # Create ms_to_idx mapping
        ms_to_idx = np.zeros(10_000, dtype=np.uint64)
        for ms in range(10_000):
            idx = np.searchsorted(t, ms * 1000)
            ms_to_idx[ms] = idx
        f.create_dataset("ms_to_idx", data=ms_to_idx)

        f.attrs["t_offset"] = 0

    print(f"Created synthetic events at {h5_path}")
    return str(h5_path)


def main():
    parser = argparse.ArgumentParser(description="Download DSEC dataset")
    parser.add_argument(
        "--sequence", type=str, default="thun_00_a", help="Sequence name to download"
    )
    parser.add_argument("--output", type=str, default="data/", help="Output directory")
    parser.add_argument("--all", action="store_true", help="Download all small sequences")
    parser.add_argument(
        "--synthetic", action="store_true", help="Create synthetic data instead of downloading"
    )
    parser.add_argument(
        "--components",
        nargs="+",
        default=["events_left", "image_timestamps"],
        help="Components to download",
    )

    args = parser.parse_args()

    if args.synthetic:
        create_sample_event_data(args.output)
    elif args.all:
        for seq in SMALL_SEQUENCES:
            download_sequence(seq, args.output, args.components)
    else:
        download_sequence(args.sequence, args.output, args.components)


if __name__ == "__main__":
    main()
