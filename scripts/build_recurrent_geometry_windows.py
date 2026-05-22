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
T_TOTAL = T_HISTORY + T_FUTURE
N_MAX = 16
STRIDE = 5

NUMERIC_FEATURES = ["s_coord", "v_s", "d_centerline"]
FEATURE_COLUMNS = NUMERIC_FEATURES + ["channel_id_int"]
TARGET_COLUMNS = ["v_s"]


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build recurrent geometry-native rollout windows."
    )
    parser.add_argument(
        "--experiment-name",
        default=EXPERIMENT_NAME,
        help="Experiment name under outputs/processed/.",
    )
    parser.add_argument(
        "--trajectory-csv",
        type=Path,
        default=None,
        help=(
            "Optional trajectories_geometry.csv path. If omitted, uses "
            "outputs/processed/<experiment_name>/trajectories_geometry.csv."
        ),
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help=(
            "Optional output NPZ path. If omitted, saves "
            "recurrent_geometry_windows.npz next to the input trajectories."
        ),
    )
    return parser.parse_args(args)


def load_trajectories(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Trajectories file not found: {path}")

    df = pd.read_csv(path)
    required_columns = {"frame", "track_id", "s_coord", "d_centerline", "channel_id"}
    missing = required_columns - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    return df


def add_centerline_velocity(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["frame"] = df["frame"].astype(int)
    df["track_id"] = df["track_id"].astype(int)
    df.sort_values(["track_id", "frame"], inplace=True)
    df["v_s"] = 0.0

    for _, track_df in df.groupby("track_id", sort=False):
        track_index = track_df.index
        if len(track_index) == 1:
            df.loc[track_index[0], "v_s"] = 0.0
            continue

        frame_diff = np.diff(track_df["frame"].to_numpy(dtype=np.float32))
        valid_diff = np.isfinite(frame_diff) & (frame_diff != 0.0)
        v_tail = np.zeros(len(track_index) - 1, dtype=np.float32)
        s_diff = np.diff(track_df["s_coord"].to_numpy(dtype=np.float32))
        v_tail[valid_diff] = s_diff[valid_diff] / frame_diff[valid_diff]

        v_s = np.zeros(len(track_index), dtype=np.float32)
        v_s[1:] = v_tail
        first_valid = np.flatnonzero(valid_diff)
        if len(first_valid) > 0:
            v_s[0] = v_tail[first_valid[0]]

        df.loc[track_index, "v_s"] = v_s

    df["v_s"] = df["v_s"].replace([np.inf, -np.inf], 0.0).fillna(0.0)
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


def compute_normalization_stats(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feature_mean = df[NUMERIC_FEATURES].mean().to_numpy(dtype=np.float32)
    feature_std = df[NUMERIC_FEATURES].std(ddof=0).to_numpy(dtype=np.float32)
    feature_std = np.where(feature_std == 0.0, 1.0, feature_std).astype(np.float32)

    target_mean = df[TARGET_COLUMNS].mean().to_numpy(dtype=np.float32)
    target_std = df[TARGET_COLUMNS].std(ddof=0).to_numpy(dtype=np.float32)
    target_std = np.where(target_std == 0.0, 1.0, target_std).astype(np.float32)
    return feature_mean, feature_std, target_mean, target_std


def normalize_dataframe(
    df: pd.DataFrame,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_df = df.copy()
    feature_df.loc[:, NUMERIC_FEATURES] = (
        feature_df[NUMERIC_FEATURES].to_numpy(dtype=np.float32) - feature_mean
    ) / feature_std

    target_df = df.copy()
    target_df.loc[:, TARGET_COLUMNS] = (
        target_df[TARGET_COLUMNS].to_numpy(dtype=np.float32) - target_mean
    ) / target_std
    return feature_df, target_df


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
    df = add_centerline_velocity(df)
    raw_df = df.copy()
    df, channel_names, channel_ids = encode_channels(df)
    feature_mean, feature_std, target_mean, target_std = compute_normalization_stats(df)
    feature_df, target_df = normalize_dataframe(
        df, feature_mean, feature_std, target_mean, target_std
    )

    min_frame = int(feature_df["frame"].min())
    max_frame = int(feature_df["frame"].max())
    sample_starts = list(range(min_frame, max_frame - T_TOTAL + 1, STRIDE))
    if not sample_starts:
        raise ValueError(
            "Not enough frames to build recurrent windows with the configured lengths."
        )

    grouped: dict[int, dict[str, np.ndarray]] = {}
    for track_id, track_df in feature_df.groupby("track_id", sort=True):
        track_df = track_df.sort_values("frame")
        target_track_df = target_df.loc[track_df.index]
        grouped[int(track_id)] = {
            "frames": track_df["frame"].to_numpy(dtype=int),
            "features": track_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32),
            "targets": target_track_df[TARGET_COLUMNS].to_numpy(dtype=np.float32),
        }

    z_windows: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    target_windows: list[np.ndarray] = []
    target_masks: list[np.ndarray] = []
    start_frames: list[int] = []
    track_ids_list: list[np.ndarray] = []

    for start_frame in sample_starts:
        last_history_frame = start_frame + T_HISTORY - 1
        tokens = (
            feature_df.loc[feature_df["frame"] == last_history_frame, "track_id"]
            .sort_values()
            .unique()
        )
        tokens = tokens[:N_MAX]
        num_tokens = len(tokens)

        track_ids = np.full((N_MAX,), -1, dtype=int)
        track_ids[:num_tokens] = tokens.astype(int)

        z_window = np.zeros((T_TOTAL, N_MAX, len(FEATURE_COLUMNS)), dtype=np.float32)
        mask = np.zeros((T_TOTAL, N_MAX), dtype=bool)
        target_v_s = np.zeros((T_TOTAL - 1, N_MAX, len(TARGET_COLUMNS)), dtype=np.float32)
        target_mask = np.zeros((T_TOTAL - 1, N_MAX), dtype=bool)

        window_frames = np.arange(start_frame, start_frame + T_TOTAL, dtype=int)

        for token_index, track_id in enumerate(tokens):
            track_data = grouped.get(int(track_id))
            if track_data is None:
                continue

            frames = track_data["frames"]
            frame_indices = np.searchsorted(frames, window_frames)
            valid_frames = (
                (frame_indices < len(frames))
                & (frames[np.minimum(frame_indices, len(frames) - 1)] == window_frames)
            )

            z_window[valid_frames, token_index] = track_data["features"][
                frame_indices[valid_frames]
            ]
            mask[valid_frames, token_index] = True

            next_valid = valid_frames[1:]
            current_valid = valid_frames[:-1]
            valid_targets = current_valid & next_valid
            next_indices = frame_indices[1:]
            target_v_s[valid_targets, token_index] = track_data["targets"][
                next_indices[valid_targets]
            ]
            target_mask[valid_targets, token_index] = True

        z_windows.append(z_window)
        masks.append(mask)
        target_windows.append(target_v_s)
        target_masks.append(target_mask)
        start_frames.append(start_frame)
        track_ids_list.append(track_ids)

    return (
        np.stack(z_windows, axis=0),
        np.stack(masks, axis=0),
        np.stack(target_windows, axis=0),
        np.stack(target_masks, axis=0),
        np.stack(track_ids_list, axis=0),
        np.array(start_frames, dtype=int),
        channel_names,
        channel_ids,
        feature_mean,
        feature_std,
        target_mean,
        target_std,
        raw_df,
    )


def save_windows(
    output_path: Path,
    Z: np.ndarray,
    mask: np.ndarray,
    target_v_s: np.ndarray,
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
        Z=Z,
        mask=mask,
        target_v_s=target_v_s,
        target_mask=target_mask,
        track_ids=track_ids,
        start_frames=start_frames,
        feature_columns=np.array(FEATURE_COLUMNS, dtype=str),
        numeric_features=np.array(NUMERIC_FEATURES, dtype=str),
        target_columns=np.array(TARGET_COLUMNS, dtype=str),
        numeric_feature_mean=feature_mean,
        numeric_feature_std=feature_std,
        target_mean=target_mean,
        target_std=target_std,
        channel_names=channel_names,
        channel_ids=channel_ids,
        t_history=np.array(T_HISTORY, dtype=np.int64),
        t_future=np.array(T_FUTURE, dtype=np.int64),
        t_total=np.array(T_TOTAL, dtype=np.int64),
        n_max=np.array(N_MAX, dtype=np.int64),
        stride=np.array(STRIDE, dtype=np.int64),
        target_type="next_step_geometry_velocity_vs",
    )
    return output_path


def print_summary(
    Z: np.ndarray,
    mask: np.ndarray,
    target_v_s: np.ndarray,
    target_mask: np.ndarray,
    raw_df: pd.DataFrame,
) -> None:
    valid_targets = target_v_s[target_mask]
    print("=== recurrent geometry window summary ===")
    print(f"Z shape: {Z.shape}")
    print(f"target_v_s shape: {target_v_s.shape}")
    print(f"mask fraction: {mask.mean():.4f}")
    print(f"target valid fraction: {target_mask.mean():.4f}")
    print(f"Feature order: {FEATURE_COLUMNS}")
    print(f"Numeric features: {NUMERIC_FEATURES}")
    print(f"Target columns: {TARGET_COLUMNS}")
    print("\nv_s raw stats:")
    print(f"  mean: {raw_df['v_s'].mean():.6f}")
    print(f"  std: {raw_df['v_s'].std(ddof=0):.6f}")
    print(f"  min: {raw_df['v_s'].min():.6f}")
    print(f"  max: {raw_df['v_s'].max():.6f}")
    print("\nNormalized target stats over valid targets:")
    if len(valid_targets):
        print(f"  mean: {valid_targets[:, 0].mean():.6f}")
        print(f"  std: {valid_targets[:, 0].std(ddof=0):.6f}")
    else:
        print("  mean: nan")
        print("  std: nan")
    print("=== end recurrent geometry window summary ===")


def main() -> None:
    args = parse_args()

    if args.trajectory_csv is not None:
        trajectory_csv = args.trajectory_csv
        output_dir = args.trajectory_csv.parent
    else:
        output_dir = PROCESSED_DIR / args.experiment_name
        trajectory_csv = output_dir / "trajectories_geometry.csv"

    output_path = (
        args.output_file
        if args.output_file is not None
        else output_dir / "recurrent_geometry_windows.npz"
    )

    df = load_trajectories(trajectory_csv)
    (
        Z,
        mask,
        target_v_s,
        target_mask,
        track_ids,
        start_frames,
        channel_names,
        channel_ids,
        feature_mean,
        feature_std,
        target_mean,
        target_std,
        raw_df,
    ) = build_windows(df)

    save_windows(
        output_path,
        Z,
        mask,
        target_v_s,
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
    print_summary(Z, mask, target_v_s, target_mask, raw_df)
    print("\nChannel mapping:")
    for channel_name, channel_id in zip(channel_names, channel_ids):
        print(f"  {int(channel_id)}: {channel_name}")
    print(f"\nSaved recurrent geometry dataset: {output_path}")


if __name__ == "__main__":
    main()
