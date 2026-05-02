from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd

from configs.constants import EXPERIMENT_NAME
from configs.paths import PROCESSED_DIR


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build clean trajectory dataset from tracked droplet features."
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
        "--tracked-csv",
        type=Path,
        default=None,
        help=(
            "Optional path to tracked_features.csv. If provided, this path is used "
            "instead of outputs/processed/<experiment_name>/tracked_features.csv."
        ),
    )
    return parser.parse_args(args)


def load_tracked_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Tracked features file not found: {path}")
    return pd.read_csv(path)


def build_trajectories(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"frame", "track_id", "centroid_x", "centroid_y"}
    missing = required_columns - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    df = df.loc[:, ["frame", "track_id", "centroid_x", "centroid_y"]].copy()
    df["frame"] = df["frame"].astype(int)
    df["track_id"] = df["track_id"].astype(int)
    df.rename(columns={"centroid_x": "x", "centroid_y": "y"}, inplace=True)

    track_lengths = df.groupby("track_id", sort=False)["frame"].count()
    long_tracks = track_lengths[track_lengths >= 20].index
    df = df[df["track_id"].isin(long_tracks)].copy()
    df.sort_values(["track_id", "frame"], inplace=True)

    def compute_velocity(track: pd.DataFrame) -> pd.DataFrame:
        track = track.copy()
        track["vx"] = track["x"].diff()
        track["vy"] = track["y"].diff()

        if len(track) >= 2:
            track.iloc[0, track.columns.get_loc("vx")] = track.iloc[1]["vx"]
            track.iloc[0, track.columns.get_loc("vy")] = track.iloc[1]["vy"]
        else:
            track["vx"] = 0.0
            track["vy"] = 0.0

        return track

    output_df = (
        df.groupby("track_id", sort=False)
        .apply(compute_velocity)
        .reset_index(level=0)
        .reset_index(drop=True)
    )
    output_df = output_df[["frame", "track_id", "x", "y", "vx", "vy"]]

    return output_df


def print_summary(df: pd.DataFrame) -> None:
    track_lengths = df.groupby("track_id").size()
    print("=== trajectories audit ===")
    print(f"Tracks after filtering: {df['track_id'].nunique()}")
    print(f"Average track length: {track_lengths.mean():.2f}")

    for axis in ("vx", "vy"):
        if axis in df.columns:
            stats = df[axis].describe()
            print(f"\n{axis} statistics:")
            print(f"  count: {int(stats['count'])}")
            print(f"  mean: {stats['mean']:.4f}")
            print(f"  std: {stats['std']:.4f}")
            print(f"  min: {stats['min']:.4f}")
            print(f"  50%: {stats['50%']:.4f}")
            print(f"  max: {stats['max']:.4f}")
        else:
            print(f"No '{axis}' column found.")

    print("=== end trajectories audit ===")


def save_trajectories(df: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "trajectories_clean.csv"
    df.to_csv(path, index=False)
    return path


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

    df = load_tracked_features(tracked_csv_path)
    trajectories = build_trajectories(df)
    output_path = save_trajectories(trajectories, output_dir)
    print_summary(trajectories)
    print(f"\nSaved trajectories file: {output_path}")


if __name__ == "__main__":
    main()
