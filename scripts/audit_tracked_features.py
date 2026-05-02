from __future__ import annotations

import argparse
from pathlib import Path
import sys
sys.path.append(r"C:\Droplet_prodect_transformers")

import pandas as pd

from configs.constants import EXPERIMENT_NAME
from configs.paths import PROCESSED_DIR


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit tracked droplet features before model training."
    )
    parser.add_argument(
        "--experiment-name",
        default=EXPERIMENT_NAME,
        help=(
            "Experiment name used under outputs/processed/. "
            "If omitted, uses configs.constants.EXPERIMENT_NAME."
        ),
    )
    parser.add_argument(
        "--tracked-csv",
        type=Path,
        default=None,
        help=(
            "Optional path to tracked_features.csv. If provided, this path is used "
            "instead of outputs/processed/<experiment_name>/tracked_features.csv."
        ),
    )
    return parser.parse_args(args)


def load_tracked_features(tracked_csv_path: Path) -> pd.DataFrame:
    if not tracked_csv_path.exists():
        raise FileNotFoundError(f"Tracked features file not found: {tracked_csv_path}")
    return pd.read_csv(tracked_csv_path)


def print_summary(df: pd.DataFrame) -> None:
    print("=== tracked_features audit ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst 5 rows:")
    print(df.head(5).to_string(index=False))
    print("\nMissing values per column:")
    print(df.isna().sum().to_string())

    if "frame" in df.columns:
        frame_counts = df["frame"].value_counts().sort_index()
        print(f"\nUnique frames: {int(df['frame'].nunique())}")
        print(f"Min frame: {int(df['frame'].min())}")
        print(f"Max frame: {int(df['frame'].max())}")
        print("Detections per frame:")
        print(f"  Min: {int(frame_counts.min())}")
        print(f"  Median: {int(frame_counts.median())}")
        print(f"  Max: {int(frame_counts.max())}")
    else:
        print("\nNo 'frame' column found. Skipping frame-level statistics.")
        frame_counts = pd.Series(dtype=int)

    if "track_id" in df.columns:
        track_counts = df["track_id"].dropna().astype(int).value_counts().sort_index()
        print(f"\nUnique track_id: {int(track_counts.size)}")
        if not track_counts.empty:
            print("Track length distribution:")
            print(f"  Min: {int(track_counts.min())}")
            print(f"  Median: {int(track_counts.median())}")
            print(f"  Max: {int(track_counts.max())}")
        else:
            print("No valid track_id values available to compute track lengths.")
    else:
        print("\nNo 'track_id' column found. Skipping track-length statistics.")
        track_counts = pd.Series(dtype=int)

    for coord in ("centroid_x", "centroid_y"):
        if coord in df.columns:
            print(
                f"{coord} range: {df[coord].min():.3f} to {df[coord].max():.3f}"
            )
        else:
            print(f"No '{coord}' column found. Skipping {coord} range.")

    print("=== end audit ===")
    return frame_counts, track_counts


def save_summary_csvs(
    output_dir: Path,
    frame_counts: pd.Series,
    track_counts: pd.Series,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    detections_per_frame_path = output_dir / "detections_per_frame.csv"
    track_lengths_path = output_dir / "track_lengths.csv"

    detections_df = pd.DataFrame(
        {
            "frame": frame_counts.index.astype(int),
            "detection_count": frame_counts.values.astype(int),
        }
    )
    track_lengths_df = pd.DataFrame(
        {
            "track_id": track_counts.index.astype(int),
            "track_length": track_counts.values.astype(int),
        }
    )

    detections_df.to_csv(detections_per_frame_path, index=False)
    track_lengths_df.to_csv(track_lengths_path, index=False)

    return track_lengths_path, detections_per_frame_path


def main() -> None:
    args = parse_args()

    if args.tracked_csv is not None:
        tracked_csv_path = args.tracked_csv
        output_dir = args.tracked_csv.parent
    else:
        if not args.experiment_name:
            raise ValueError(
                "Experiment name is required when --tracked-csv is not provided."
            )
        output_dir = PROCESSED_DIR / args.experiment_name
        tracked_csv_path = output_dir / "tracked_features.csv"

    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory does not exist: {output_dir}")

    df = load_tracked_features(tracked_csv_path)
    frame_counts, track_counts = print_summary(df)
    track_lengths_path, detections_path = save_summary_csvs(
        output_dir, frame_counts, track_counts
    )

    print(f"\nSaved summary files:")
    print(f"  {track_lengths_path}")
    print(f"  {detections_path}")


if __name__ == "__main__":
    main()
