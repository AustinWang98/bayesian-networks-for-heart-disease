"""CLI helper: download and cache the UCI Heart Disease dataset.

Usage
-----
    python scripts/download_data.py [--force]

The dataset is cached to ``data/raw/heart.csv`` so that subsequent
notebook runs are deterministic and offline-friendly.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow `python scripts/download_data.py` to run from repo root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import DEFAULT_RAW_PATH, load_heart_disease  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Download UCI Heart Disease dataset.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if a cached copy exists.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_RAW_PATH),
        help=f"Destination path (default: {DEFAULT_RAW_PATH}).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    df = load_heart_disease(cache_path=Path(args.output), force_refresh=args.force)
    print(f"Loaded {len(df)} rows x {df.shape[1]} columns -> {args.output}")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
