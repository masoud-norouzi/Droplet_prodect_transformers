from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.canonical_dataset.canonical_dataset_builder import CanonicalDatasetBuilder


INPUT_CSV = REPO_ROOT / "outputs" / "processed" / "2" / "tracked_features.csv"
OUTPUT_NPZ = REPO_ROOT / "outputs" / "processed" / "2" / "canonical_dataset.npz"
INLET_Y_MAX_PX = 100.0


def main() -> None:
    args = parse_args()
    builder = CanonicalDatasetBuilder(
        input_csv=args.input_csv,
        output_npz=args.output_npz,
        inlet_y_max_px=args.inlet_y_max_px,
    )
    builder.run()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the canonical droplet dataset tensor.")
    parser.add_argument("--input-csv", type=Path, default=INPUT_CSV)
    parser.add_argument("--output-npz", type=Path, default=OUTPUT_NPZ)
    parser.add_argument(
        "--inlet-y-max-px",
        type=float,
        default=INLET_Y_MAX_PX,
        help="Incoming inlet region criterion: patch first-visible missing velocities only when y <= this value.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
