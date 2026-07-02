from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.canonical_dataset.canonical_dataset_builder import CanonicalDatasetBuilder


INPUT_CSV = REPO_ROOT / "outputs" / "processed" / "2" / "tracked_features.csv"
OUTPUT_NPZ = REPO_ROOT / "outputs" / "processed" / "2" / "canonical_dataset.npz"


def main() -> None:
    builder = CanonicalDatasetBuilder(
        input_csv=INPUT_CSV,
        output_npz=OUTPUT_NPZ,
    )
    builder.run()


if __name__ == "__main__":
    main()
