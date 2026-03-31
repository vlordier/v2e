#!/usr/bin/env python
"""
Create mini subsets of FPV data for rapid iteration.

Usage:
    python create_mini_fpv.py --input data/fpv/indoor_forward_3 --output data/fpv/mini_fpv
    python create_mini_fpv.py --max-events 100000 --max-imu 1000
"""

import argparse
import shutil
from pathlib import Path


def create_mini_subset(
    input_dir: str,
    output_dir: str,
    max_events: int = 100000,
    max_imu: int = 1000,
    max_images: int = 200,
) -> None:
    """Create a mini subset of FPV data."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return

    output_path.mkdir(parents=True, exist_ok=True)

    # Copy IMU data (subset)
    imu_file = input_path / "imu.txt"
    if imu_file.exists():
        print(f"Creating mini IMU file (max {max_imu} samples)...")
        with open(imu_file) as f_in:
            lines = f_in.readlines()

        # Keep header + max_imu data lines
        header = [line for line in lines if line.startswith("#")]
        data = [line for line in lines if not line.startswith("#")][:max_imu]

        with open(output_path / "imu.txt", "w") as f_out:
            f_out.writelines(header)
            f_out.writelines(data)

        print(f"  Created imu.txt with {len(data)} samples")

    # Copy events data (subset)
    events_file = input_path / "events.txt"
    if events_file.exists():
        print(f"Creating mini events file (max {max_events} events)...")
        with open(events_file) as f_in:
            lines = f_in.readlines()

        # Keep header + max_events data lines
        header = [line for line in lines if line.startswith("#")]
        data = [line for line in lines if not line.startswith("#")][:max_events]

        with open(output_path / "events.txt", "w") as f_out:
            f_out.writelines(header)
            f_out.writelines(data)

        print(f"  Created events.txt with {len(data)} events")

    # Copy images timestamps (subset)
    images_file = input_path / "images.txt"
    if images_file.exists():
        print(f"Creating mini images file (max {max_images} images)...")
        with open(images_file) as f_in:
            lines = f_in.readlines()

        # Keep header + max_images data lines
        header = [line for line in lines if line.startswith("#")]
        data = [line for line in lines if not line.startswith("#")][:max_images]

        with open(output_path / "images.txt", "w") as f_out:
            f_out.writelines(header)
            f_out.writelines(data)

        print(f"  Created images.txt with {len(data)} images")

    # Copy README
    readme_file = input_path / "README.md"
    if readme_file.exists():
        shutil.copy(readme_file, output_path / "README.md")

    print(f"\nMini FPV subset created at {output_dir}")
    print(f"  - IMU samples: {max_imu}")
    print(f"  - Events: {max_events}")
    print(f"  - Images: {max_images}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create mini FPV data subset")
    parser.add_argument(
        "--input",
        type=str,
        default="data/fpv/indoor_forward_3",
        help="Input directory with FPV data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/fpv/mini_fpv",
        help="Output directory for mini subset",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=100000,
        help="Maximum number of events to keep",
    )
    parser.add_argument(
        "--max-imu",
        type=int,
        default=1000,
        help="Maximum number of IMU samples to keep",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=200,
        help="Maximum number of image timestamps to keep",
    )

    args = parser.parse_args()

    create_mini_subset(
        args.input,
        args.output,
        args.max_events,
        args.max_imu,
        args.max_images,
    )


if __name__ == "__main__":
    main()
