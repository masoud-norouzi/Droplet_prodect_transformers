from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from configs.constants import EXPERIMENT_NAME
from configs.paths import PROCESSED_DIR

T_HISTORY = 20
T_FUTURE = 10
N_MAX = 16
STRIDE = 5


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build trajectory windows for attention-model training."
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
        "--trajectory-csv",
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

    df = pd.read_csv(path)
    required_columns = {"frame", "track_id", "x", "y", "vx", "vy"}
    missing = required_columns - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    return df


def build_windows(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = df.copy()
    df["frame"] = df["frame"].astype(int)
    df["track_id"] = df["track_id"].astype(int)

    min_frame = int(df["frame"].min())
    max_frame = int(df["frame"].max())
    sample_starts = list(range(min_frame, max_frame - T_HISTORY - T_FUTURE + 1, STRIDE))

    if not sample_starts:
        raise ValueError(
            "Not enough frames to build any training windows with the configured history and future lengths."
        )

    x_windows: list[np.ndarray] = []
    y_windows: list[np.ndarray] = []
    input_masks: list[np.ndarray] = []
    target_masks: list[np.ndarray] = []
    start_frames: list[int] = []
    track_ids_list: list[np.ndarray] = []

    grouped = {track_id: track_df.set_index("frame", drop=False)
               for track_id, track_df in df.groupby("track_id", sort=True)}

    for start_frame in sample_starts:
        last_input_frame = start_frame + T_HISTORY - 1
        future_frame_end = start_frame + T_HISTORY + T_FUTURE - 1

        tokens = df.loc[df["frame"] == last_input_frame, "track_id"].sort_values().unique()
        tokens = tokens[:N_MAX]
        num_tokens = len(tokens)

        track_ids = np.full((N_MAX,), -1, dtype=int)
        track_ids[:num_tokens] = tokens.astype(int)
        track_ids_list.append(track_ids)

        x_window = np.zeros((T_HISTORY, N_MAX, 4), dtype=float)
        y_window = np.zeros((T_FUTURE, N_MAX, 2), dtype=float)
        input_mask = np.zeros((T_HISTORY, N_MAX), dtype=bool)
        target_mask = np.zeros((T_FUTURE, N_MAX), dtype=bool)

        for token_index, track_id in enumerate(tokens):
            track = grouped.get(int(track_id))
            if track is None:
                continue

            for t in range(T_HISTORY):
                frame = start_frame + t
                if frame in track.index:
                    row = track.loc[frame]
                    x_window[t, token_index] = [row["x"], row["y"], row["vx"], row["vy"]]
                    input_mask[t, token_index] = True

            for t in range(T_FUTURE):
                frame = start_frame + T_HISTORY + t
                if frame in track.index:
                    row = track.loc[frame]
                    y_window[t, token_index] = [row["vx"], row["vy"]]
                    target_mask[t, token_index] = True

        x_windows.append(x_window)
        y_windows.append(y_window)
        input_masks.append(input_mask)
        target_masks.append(target_mask)
        start_frames.append(start_frame)

    X = np.stack(x_windows, axis=0)
    Y = np.stack(y_windows, axis=0)
    input_mask = np.stack(input_masks, axis=0)
    target_mask = np.stack(target_masks, axis=0)
    track_ids = np.stack(track_ids_list, axis=0)
    start_frames_arr = np.array(start_frames, dtype=int)

    return X, Y, input_mask, target_mask, track_ids, start_frames_arr


def save_windows(
    output_dir: Path,
    X: np.ndarray,
    Y: np.ndarray,
    input_mask: np.ndarray,
    target_mask: np.ndarray,
    track_ids: np.ndarray,
    start_frames: np.ndarray,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "trajectory_windows.npz"
    np.savez_compressed(
        output_path,
        X=X,
        Y=Y,
        input_mask=input_mask,
        target_mask=target_mask,
        track_ids=track_ids,
        start_frames=start_frames,
        target_type="velocity",
    )
    return output_path


def print_summary(X: np.ndarray, Y: np.ndarray, target_mask: np.ndarray, track_ids: np.ndarray) -> None:
    num_samples = X.shape[0]

    valid_droplets = np.sum(track_ids != -1, axis=1)
    avg_valid_droplets = float(valid_droplets.mean())

    total_future_values = target_mask.sum()
    valid_future_targets = float(total_future_values) / (num_samples * T_FUTURE * N_MAX)

    print("=== trajectory window summary ===")
    print(f"Number of samples: {num_samples}")
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")
    print(f"Average valid droplets per sample: {avg_valid_droplets:.2f}")
    print(f"Fraction of valid future targets: {valid_future_targets:.4f}")
    print("=== end summary ===")


def main() -> None:
    args = parse_args()

    if args.trajectory_csv is not None:
        trajectory_csv = args.trajectory_csv
        output_dir = args.trajectory_csv.parent
    else:
        if not args.experiment_name:
            raise ValueError(
                "Experiment name is required when --trajectory-csv is not provided."
            )
        output_dir = PROCESSED_DIR / args.experiment_name
        trajectory_csv = output_dir / "trajectories_clean.csv"

    df = load_trajectories(trajectory_csv)
    X, Y, input_mask, target_mask, track_ids, start_frames = build_windows(df)
    output_path = save_windows(
        output_dir, X, Y, input_mask, target_mask, track_ids, start_frames
    )
    print_summary(X, Y, target_mask, track_ids)
    print(f"\nSaved window dataset: {output_path}")


if __name__ == "__main__":
    main()
