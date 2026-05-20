from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd

from configs.constants import EXPERIMENT_NAME
from configs.paths import PROCESSED_DIR, PROJECT_ROOT
from src.geometry import ChannelGeometry


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate droplet trajectories with channel geometry features."
    )
    parser.add_argument(
        "--experiment-name",
        default=EXPERIMENT_NAME,
        help=(
            "Experiment name under outputs/processed/. "
            "If omitted, uses configs.constants.EXPERIMENT_NAME."
        ),
    )
    parser.add_argument(
        "--centerlines-csv",
        type=Path,
        default=PROJECT_ROOT / "geometry" / "centerlines.csv",
        help="Path to centerlines.csv with channel_id, x, and y columns.",
    )
    parser.add_argument(
        "--trajectories-csv",
        type=Path,
        default=None,
        help=(
            "Optional path to trajectories_clean.csv. If provided, this path is used "
            "instead of outputs/processed/<experiment_name>/trajectories_clean.csv."
        ),
    )
    return parser.parse_args(args)


def load_trajectories(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Trajectories file not found: {path}")
    return pd.read_csv(path)


def print_summary(df: pd.DataFrame) -> None:
    d_stats = df["d_centerline"].describe()

    print("=== geometry annotation audit ===")
    print(f"Rows annotated: {len(df)}")
    print("\nChannel counts:")
    print(df["channel_id"].value_counts(dropna=False).to_string())
    print("\nd_centerline statistics:")
    print(f"  min: {d_stats['min']:.4f}")
    print(f"  median: {df['d_centerline'].median():.4f}")
    print(f"  max: {d_stats['max']:.4f}")
    print("=== end geometry annotation audit ===")


def warn_large_negative_s_jumps(
    df: pd.DataFrame, threshold: float = -20.0
) -> None:
    if "track_id" not in df.columns or "frame" not in df.columns:
        print("\nWARNING: cannot check s_coord jumps; track_id/frame columns are missing.")
        return

    warnings = []
    for track_id, track_df in df.sort_values(["track_id", "frame"]).groupby("track_id"):
        s_diff = track_df["s_coord"].diff()
        jump_positions = [i for i, value in enumerate(s_diff.to_numpy()) if value < threshold]
        for position in jump_positions:
            row = track_df.iloc[position]
            previous_row = track_df.iloc[position - 1]
            warnings.append(
                (
                    track_id,
                    int(row["frame"]),
                    float(s_diff.iloc[position]),
                    str(previous_row["channel_id"]),
                    str(row["channel_id"]),
                    float(previous_row["s_coord"]),
                    float(row["s_coord"]),
                )
            )

    if not warnings:
        print("\nNo large negative global s_coord jumps found.")
        return

    print(f"\nWARNING: found {len(warnings)} large negative global s_coord jumps.")
    for (
        track_id,
        frame,
        jump,
        previous_channel,
        channel,
        previous_s,
        s_coord,
    ) in warnings[:10]:
        print(
            f"  track_id={track_id} frame={frame} diff={jump:.2f} "
            f"{previous_channel}->{channel} s={previous_s:.2f}->{s_coord:.2f}"
        )
    if len(warnings) > 10:
        print(f"  ... {len(warnings) - 10} more")


def main() -> None:
    args = parse_args()

    if args.trajectories_csv is not None:
        trajectories_csv_path = args.trajectories_csv
        output_dir = args.trajectories_csv.parent
    else:
        if not args.experiment_name:
            raise ValueError(
                "Experiment name is required when --trajectories-csv is not provided."
            )
        output_dir = PROCESSED_DIR / args.experiment_name
        trajectories_csv_path = output_dir / "trajectories_clean.csv"

    trajectories = load_trajectories(trajectories_csv_path)
    geometry = ChannelGeometry(args.centerlines_csv)
    annotated = geometry.annotate_dataframe(trajectories)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "trajectories_geometry.csv"
    annotated.to_csv(output_path, index=False)

    print_summary(annotated)
    warn_large_negative_s_jumps(annotated)
    print(f"\nSaved geometry-annotated trajectories file: {output_path}")


if __name__ == "__main__":
    main()
