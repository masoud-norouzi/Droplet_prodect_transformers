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

NUMERIC_FEATURES = ["s_coord", "d_centerline", "v_s"]
FEATURE_COLUMNS = NUMERIC_FEATURES + ["channel_id_int"]
TARGET_COLUMNS = ["v_s"]
DIAGNOSTIC_VELOCITY_COLUMNS = ["v_s", "v_d"]


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build geometry-native trajectory windows with centerline velocity targets."
        )
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
            "Optional path to trajectories_geometry.csv. If provided, this path is used "
            "instead of outputs/processed/<experiment_name>/trajectories_geometry.csv."
        ),
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help=(
            "Optional output NPZ path. If omitted, saves "
            "trajectory_windows_geometry_velocity.npz next to the input trajectories."
        ),
    )
    return parser.parse_args(args)


def load_trajectories(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Trajectories file not found: {path}")

    df = pd.read_csv(path)
    required_columns = {
        "frame",
        "track_id",
        "s_coord",
        "d_centerline",
        "channel_id",
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    return df


def add_centerline_velocities(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["frame"] = df["frame"].astype(int)
    df["track_id"] = df["track_id"].astype(int)
    df.sort_values(["track_id", "frame"], inplace=True)

    df["v_s"] = 0.0
    df["v_d"] = 0.0

    for _, track_df in df.groupby("track_id", sort=False):
        track_index = track_df.index
        if len(track_index) == 1:
            df.loc[track_index[0], ["v_s", "v_d"]] = 0.0
            continue

        frame_diff = np.diff(track_df["frame"].to_numpy(dtype=np.float32))
        valid_diff = np.isfinite(frame_diff) & (frame_diff != 0.0)

        v_s_tail = np.zeros(len(track_index) - 1, dtype=np.float32)
        v_d_tail = np.zeros(len(track_index) - 1, dtype=np.float32)
        v_s_tail[valid_diff] = (
            np.diff(track_df["s_coord"].to_numpy(dtype=np.float32))[valid_diff]
            / frame_diff[valid_diff]
        )
        v_d_tail[valid_diff] = (
            np.diff(track_df["d_centerline"].to_numpy(dtype=np.float32))[valid_diff]
            / frame_diff[valid_diff]
        )

        v_s = np.zeros(len(track_index), dtype=np.float32)
        v_d = np.zeros(len(track_index), dtype=np.float32)
        v_s[1:] = v_s_tail
        v_d[1:] = v_d_tail

        first_valid = np.flatnonzero(valid_diff)
        if len(first_valid) > 0:
            v_s[0] = v_s_tail[first_valid[0]]
            v_d[0] = v_d_tail[first_valid[0]]

        df.loc[track_index, "v_s"] = v_s
        df.loc[track_index, "v_d"] = v_d

    df[["v_s", "v_d"]] = df[["v_s", "v_d"]].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return df


def encode_channels(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    df = df.copy()
    channel_names = np.array(sorted(df["channel_id"].astype(str).unique()), dtype=str)
    channel_ids = np.arange(len(channel_names), dtype=np.int64)
    channel_lookup = {
        channel_name: int(channel_id)
        for channel_name, channel_id in zip(channel_names, channel_ids)
    }
    df["channel_id_int"] = df["channel_id"].astype(str).map(channel_lookup).astype(int)
    return df, channel_names, channel_ids


def normalize_numeric_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    df = df.copy()
    feature_mean = df[NUMERIC_FEATURES].mean().to_numpy(dtype=np.float32)
    feature_std = df[NUMERIC_FEATURES].std(ddof=0).to_numpy(dtype=np.float32)
    feature_std = np.where(feature_std == 0.0, 1.0, feature_std).astype(np.float32)
    df.loc[:, NUMERIC_FEATURES] = (
        df[NUMERIC_FEATURES].to_numpy(dtype=np.float32) - feature_mean
    ) / feature_std
    return df, feature_mean, feature_std


def compute_target_stats(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    target_mean = df[TARGET_COLUMNS].mean().to_numpy(dtype=np.float32)
    target_std = df[TARGET_COLUMNS].std(ddof=0).to_numpy(dtype=np.float32)
    target_std = np.where(target_std == 0.0, 1.0, target_std).astype(np.float32)
    return target_mean, target_std


def build_windows(
    df: pd.DataFrame,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
]:
    df = add_centerline_velocities(df)
    raw_feature_df = df.copy()
    target_mean, target_std = compute_target_stats(raw_feature_df)
    df, channel_names, channel_ids = encode_channels(df)
    df, feature_mean, feature_std = normalize_numeric_features(df)
    df.loc[:, TARGET_COLUMNS] = (
        raw_feature_df[TARGET_COLUMNS].to_numpy(dtype=np.float32) - target_mean
    ) / target_std

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

    grouped = {}
    for track_id, track_df in df.groupby("track_id", sort=True):
        track_df = track_df.sort_values("frame")
        grouped[int(track_id)] = {
            "frames": track_df["frame"].to_numpy(dtype=int),
            "features": track_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32),
            "targets": track_df[TARGET_COLUMNS].to_numpy(dtype=np.float32),
        }

    for start_frame in sample_starts:
        last_input_frame = start_frame + T_HISTORY - 1

        tokens = df.loc[df["frame"] == last_input_frame, "track_id"].sort_values().unique()
        tokens = tokens[:N_MAX]
        num_tokens = len(tokens)

        track_ids = np.full((N_MAX,), -1, dtype=int)
        track_ids[:num_tokens] = tokens.astype(int)
        track_ids_list.append(track_ids)

        x_window = np.zeros((T_HISTORY, N_MAX, len(FEATURE_COLUMNS)), dtype=np.float32)
        y_window = np.zeros((T_FUTURE, N_MAX, len(TARGET_COLUMNS)), dtype=np.float32)
        input_mask = np.zeros((T_HISTORY, N_MAX), dtype=bool)
        target_mask = np.zeros((T_FUTURE, N_MAX), dtype=bool)

        for token_index, track_id in enumerate(tokens):
            track_data = grouped.get(int(track_id))
            if track_data is None:
                continue

            frames = track_data["frames"]
            history_frames = np.arange(start_frame, start_frame + T_HISTORY, dtype=int)
            history_indices = np.searchsorted(frames, history_frames)
            valid_history = (
                (history_indices < len(frames))
                & (frames[np.minimum(history_indices, len(frames) - 1)] == history_frames)
            )
            x_window[valid_history, token_index] = track_data["features"][
                history_indices[valid_history]
            ]
            input_mask[valid_history, token_index] = True

            future_frames = np.arange(
                start_frame + T_HISTORY,
                start_frame + T_HISTORY + T_FUTURE,
                dtype=int,
            )
            future_indices = np.searchsorted(frames, future_frames)
            valid_future = (
                (future_indices < len(frames))
                & (frames[np.minimum(future_indices, len(frames) - 1)] == future_frames)
            )
            y_window[valid_future, token_index] = track_data["targets"][
                future_indices[valid_future]
            ]
            target_mask[valid_future, token_index] = True

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

    return (
        X,
        Y,
        input_mask,
        target_mask,
        track_ids,
        start_frames_arr,
        channel_names,
        channel_ids,
        feature_mean,
        feature_std,
        target_mean,
        target_std,
        raw_feature_df,
    )


def save_windows(
    output_path: Path,
    X: np.ndarray,
    Y: np.ndarray,
    input_mask: np.ndarray,
    target_mask: np.ndarray,
    track_ids: np.ndarray,
    start_frames: np.ndarray,
    channel_names: np.ndarray,
    channel_ids: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X=X,
        Y=Y,
        input_mask=input_mask,
        target_mask=target_mask,
        track_ids=track_ids,
        start_frames=start_frames,
        feature_columns=np.array(FEATURE_COLUMNS, dtype=str),
        numeric_features=np.array(NUMERIC_FEATURES, dtype=str),
        target_columns=np.array(TARGET_COLUMNS, dtype=str),
        feature_names=np.array(FEATURE_COLUMNS, dtype=str),
        numeric_feature_names=np.array(NUMERIC_FEATURES, dtype=str),
        target_names=np.array(TARGET_COLUMNS, dtype=str),
        numeric_feature_mean=feature_mean,
        numeric_feature_std=feature_std,
        target_mean=target_mean,
        target_std=target_std,
        channel_names=channel_names,
        channel_ids=channel_ids,
        target_type="geometry_velocity_vs_only",
    )
    return output_path


def print_summary(
    X: np.ndarray,
    Y: np.ndarray,
    target_mask: np.ndarray,
    raw_feature_df: pd.DataFrame,
) -> None:
    num_samples = X.shape[0]
    valid_future_targets = float(target_mask.sum()) / (num_samples * T_FUTURE * N_MAX)

    print("=== geometry velocity window summary ===")
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")
    print(f"Feature order: {FEATURE_COLUMNS}")
    print(f"Numeric features: {NUMERIC_FEATURES}")
    print(f"Target columns: {TARGET_COLUMNS}")
    print("\nRaw centerline velocity diagnostics:")
    for column in DIAGNOSTIC_VELOCITY_COLUMNS:
        values = raw_feature_df[column]
        print(f"  {column}:")
        print(f"    mean: {values.mean():.6f}")
        print(f"    std: {values.std(ddof=0):.6f}")
        print(f"    min: {values.min():.6f}")
        print(f"    max: {values.max():.6f}")
    print(f"\nModel target columns: {TARGET_COLUMNS} (v_d is diagnostic only)")
    print("\nNormalized Y statistics over valid targets:")
    valid_y = Y[target_mask]
    for target_index, column in enumerate(TARGET_COLUMNS):
        values = valid_y[:, target_index] if len(valid_y) else np.array([], dtype=np.float32)
        print(f"  {column}:")
        if len(values):
            print(f"    mean: {values.mean():.6f}")
            print(f"    std: {values.std(ddof=0):.6f}")
        else:
            print("    mean: nan")
            print("    std: nan")
    print(f"\nFraction of valid future targets: {valid_future_targets:.4f}")
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
        trajectory_csv = output_dir / "trajectories_geometry.csv"

    output_path = (
        args.output_file
        if args.output_file is not None
        else output_dir / "trajectory_windows_geometry_velocity.npz"
    )

    df = load_trajectories(trajectory_csv)
    (
        X,
        Y,
        input_mask,
        target_mask,
        track_ids,
        start_frames,
        channel_names,
        channel_ids,
        feature_mean,
        feature_std,
        target_mean,
        target_std,
        raw_feature_df,
    ) = build_windows(df)

    save_windows(
        output_path,
        X,
        Y,
        input_mask,
        target_mask,
        track_ids,
        start_frames,
        channel_names,
        channel_ids,
        feature_mean,
        feature_std,
        target_mean,
        target_std,
    )

    print_summary(X, Y, target_mask, raw_feature_df)
    print("\nChannel mapping:")
    for channel_name, channel_id in zip(channel_names, channel_ids):
        print(f"  {int(channel_id)}: {channel_name}")
    print(f"\nSaved geometry velocity window dataset: {output_path}")


if __name__ == "__main__":
    main()
