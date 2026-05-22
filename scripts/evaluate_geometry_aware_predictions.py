from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from configs.constants import EXPERIMENT_NAME
from configs.paths import PROCESSED_DIR, PROJECT_ROOT
from src.geometry import ChannelGeometry
from src.trajectory_model import TrajectoryAttentionModel


class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, input_mask: np.ndarray) -> None:
        self.X = torch.from_numpy(X).float()
        self.input_mask = torch.from_numpy(input_mask).bool()

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.input_mask[idx]


@dataclass
class CartesianTrajectories:
    pred_xy: np.ndarray
    true_xy: np.ndarray
    history_xy: np.ndarray
    target_mask: np.ndarray
    input_mask: np.ndarray
    channel_labels: np.ndarray


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a geometry-aware trajectory model in Cartesian pixels."
    )
    parser.add_argument("--experiment-name", default=EXPERIMENT_NAME)
    parser.add_argument(
        "--windows-file",
        type=Path,
        default=None,
        help=(
            "Geometry-aware windows NPZ. Defaults to "
            "outputs/processed/<experiment_name>/trajectory_windows_geometry_velocity.npz."
        ),
    )
    parser.add_argument(
        "--model-file",
        type=Path,
        default=None,
        help="Geometry-aware checkpoint. Defaults to attention_model.pt next to windows.",
    )
    parser.add_argument(
        "--trajectories-csv",
        type=Path,
        default=None,
        help="trajectories_geometry.csv with true x,y,s,d,channel columns.",
    )
    parser.add_argument(
        "--centerlines-csv",
        type=Path,
        default=PROJECT_ROOT / "geometry" / "centerlines.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to outputs/evaluation/<experiment_name>/.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(args)


def split_indices(
    num_samples: int, train_frac: float = 0.8, seed: int = 42
) -> dict[str, np.ndarray]:
    indices = np.arange(num_samples)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    split = int(num_samples * train_frac)
    return {"train": indices[:split], "val": indices[split:]}


def npz_array(data: np.lib.npyio.NpzFile, names: list[str]) -> np.ndarray | None:
    for name in names:
        if name in data.files:
            return data[name]
    return None


def load_windows(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Windows file not found: {path}")
    data = np.load(path)
    required = {"X", "Y", "input_mask", "target_mask", "track_ids", "start_frames"}
    missing = required - set(data.files)
    if missing:
        raise KeyError(f"Missing required NPZ arrays: {sorted(missing)}")

    feature_names = npz_array(data, ["feature_columns", "feature_names"])
    numeric_feature_names = npz_array(data, ["numeric_features", "numeric_feature_names"])
    target_names = npz_array(data, ["target_columns", "target_names"])
    if feature_names is None or numeric_feature_names is None or target_names is None:
        raise KeyError("Windows file is missing feature/target metadata.")

    return {
        "X": data["X"],
        "Y": data["Y"],
        "input_mask": data["input_mask"].astype(bool),
        "target_mask": data["target_mask"].astype(bool),
        "track_ids": data["track_ids"],
        "start_frames": data["start_frames"],
        "feature_names": feature_names.astype(str),
        "numeric_feature_names": numeric_feature_names.astype(str),
        "target_names": target_names.astype(str),
        "channel_names": data["channel_names"].astype(str),
        "channel_ids": data["channel_ids"],
        "numeric_feature_mean": npz_array(data, ["numeric_feature_mean"]),
        "numeric_feature_std": npz_array(data, ["numeric_feature_std"]),
        "target_mean": npz_array(data, ["target_mean"]),
        "target_std": npz_array(data, ["target_std"]),
        "target_type": str(data["target_type"]) if "target_type" in data.files else "",
    }


def unnormalize_features(X: np.ndarray, windows: dict[str, Any]) -> np.ndarray:
    mean = windows["numeric_feature_mean"]
    std = windows["numeric_feature_std"]
    if mean is None or std is None:
        return X.copy()

    X_out = X.copy()
    feature_names = list(windows["feature_names"])
    for numeric_index, name in enumerate(windows["numeric_feature_names"]):
        feature_index = feature_names.index(str(name))
        X_out[..., feature_index] = X_out[..., feature_index] * std[numeric_index] + mean[numeric_index]
    return X_out


def unnormalize_targets(Y: np.ndarray, windows: dict[str, Any]) -> np.ndarray:
    mean = windows["target_mean"]
    std = windows["target_std"]
    if mean is None or std is None:
        return Y
    return Y * std.reshape(1, 1, 1, -1) + mean.reshape(1, 1, 1, -1)


def load_model(
    model_path: Path, windows: dict[str, Any], device: torch.device
) -> TrajectoryAttentionModel:
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    state = torch.load(model_path, map_location=device)

    expected_numeric_dim = len(windows["numeric_feature_names"])
    expected_target_dim = windows["Y"].shape[-1]
    checkpoint_numeric_dim = int(state["numeric_embed.0.weight"].shape[1])
    checkpoint_target_dim = int(state["predictor.3.weight"].shape[0] // windows["Y"].shape[1])
    if checkpoint_numeric_dim != expected_numeric_dim or checkpoint_target_dim != expected_target_dim:
        raise ValueError(
            "Checkpoint does not match geometry-aware windows: "
            f"checkpoint numeric_dim={checkpoint_numeric_dim}, target_dim={checkpoint_target_dim}; "
            f"windows numeric_dim={expected_numeric_dim}, target_dim={expected_target_dim}."
        )

    model = TrajectoryAttentionModel(
        input_dim=windows["X"].shape[-1],
        numeric_input_dim=expected_numeric_dim,
        num_channel_embeddings=int(windows["channel_ids"].max()) + 1,
        target_dim=expected_target_dim,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def predict(model: TrajectoryAttentionModel, X: np.ndarray, input_mask: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    loader = DataLoader(WindowDataset(X, input_mask), batch_size=batch_size, shuffle=False)
    batches: list[np.ndarray] = []
    with torch.no_grad():
        for X_batch, mask_batch in loader:
            batches.append(model(X_batch.to(device), mask_batch.to(device)).cpu().numpy())
    return np.concatenate(batches, axis=0)


def load_trajectory_lookup(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Geometry trajectories file not found: {path}")
    df = pd.read_csv(path)
    required = {"frame", "track_id", "x", "y", "s_coord", "d_centerline", "channel_id"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required trajectory columns: {sorted(missing)}")
    return {
        (int(row.track_id), int(row.frame)): {
            "x": float(row.x),
            "y": float(row.y),
            "s_coord": float(row.s_coord),
            "d_centerline": float(row.d_centerline),
            "channel_id": str(row.channel_id),
        }
        for row in df.itertuples(index=False)
    }


class CenterlineLookup:
    def __init__(self, centerlines_csv: Path) -> None:
        self.geometry = ChannelGeometry(centerlines_csv)
        self.channels = self.geometry.channels
        self.offsets = self.geometry.channel_offsets
        self.inlet_id = "inlet" if "inlet" in self.channels else None
        self.outlet_id = "outlet" if "outlet" in self.channels else None
        self.branch_ids = [
            str(channel_id)
            for channel_id in self.channels
            if str(channel_id) not in {"inlet", "outlet"}
        ]

    def infer_channel(self, s_coord: float, previous_channel: str) -> str:
        if self.inlet_id is not None:
            inlet_end = self.offsets[self.inlet_id] + self.channels[self.inlet_id].length
            if s_coord <= inlet_end:
                return self.inlet_id
        if self.outlet_id is not None and s_coord >= self.offsets[self.outlet_id]:
            return self.outlet_id
        if previous_channel in self.branch_ids:
            return previous_channel
        if self.branch_ids:
            return self.branch_ids[0]
        return previous_channel

    def geometry_to_xy(self, s_coord: float, d_centerline: float, channel_id: str) -> np.ndarray:
        if channel_id not in self.channels:
            channel_id = self.infer_channel(s_coord, str(channel_id))
        channel = self.channels[channel_id]
        local_s = float(np.clip(s_coord - self.offsets[channel_id], 0.0, channel.length))
        segment_end_s = channel.cumulative_s + channel.segment_lengths
        segment_index = int(np.searchsorted(segment_end_s, local_s, side="left"))
        segment_index = min(segment_index, len(channel.segment_lengths) - 1)

        segment_start_s = channel.cumulative_s[segment_index]
        segment_length = channel.segment_lengths[segment_index]
        t = 0.0 if segment_length == 0 else (local_s - segment_start_s) / segment_length
        t = float(np.clip(t, 0.0, 1.0))

        start = channel.segment_starts[segment_index]
        vector = channel.segment_vectors[segment_index]
        tangent = vector / segment_length
        normal = np.array([-tangent[1], tangent[0]], dtype=float)
        centerline_xy = start + t * vector

        # d_centerline is unsigned in preprocessing, so this applies one consistent
        # normal side. True targets are evaluated from measured x/y, not unsigned d.
        return centerline_xy + float(d_centerline) * normal

    def plot_centerlines(self) -> None:
        for channel in self.channels.values():
            plt.plot(
                channel.points[:, 0],
                channel.points[:, 1],
                "-",
                color="0.85",
                linewidth=1,
                zorder=0,
            )


def build_cartesian_trajectories(
    windows: dict[str, Any],
    val_indices: np.ndarray,
    X_physical: np.ndarray,
    pred_targets: np.ndarray,
    trajectories_csv: Path,
    centerlines_csv: Path,
) -> CartesianTrajectories:
    target_names = list(windows["target_names"])
    if target_names != ["v_s"]:
        raise ValueError(
            "This script is for the geometry-aware v_s model. "
            f"Found target columns {target_names}."
        )

    feature_names = list(windows["feature_names"])
    s_index = feature_names.index("s_coord")
    d_index = feature_names.index("d_centerline")
    channel_index = feature_names.index("channel_id_int")
    lookup = load_trajectory_lookup(trajectories_csv)
    centerlines = CenterlineLookup(centerlines_csv)

    target_mask = windows["target_mask"][val_indices].copy()
    input_mask = windows["input_mask"][val_indices].copy()
    track_ids = windows["track_ids"][val_indices]
    start_frames = windows["start_frames"][val_indices]
    channel_names = list(windows["channel_names"])

    n_samples, t_history, n_max, _ = X_physical.shape
    t_future = pred_targets.shape[1]
    pred_xy = np.zeros((n_samples, t_future, n_max, 2), dtype=np.float32)
    true_xy = np.zeros_like(pred_xy)
    history_xy = np.zeros((n_samples, t_history, n_max, 2), dtype=np.float32)
    channel_labels = np.full((n_samples, t_future, n_max), "", dtype=object)

    last_s = X_physical[:, -1, :, s_index]
    last_d = X_physical[:, -1, :, d_index]
    pred_s = last_s[:, None, :] + np.cumsum(pred_targets[..., 0], axis=1)

    for sample in range(n_samples):
        for token in range(n_max):
            track_id = int(track_ids[sample, token])
            if track_id < 0:
                target_mask[sample, :, token] = False
                input_mask[sample, :, token] = False
                continue

            last_channel_int = int(round(X_physical[sample, -1, token, channel_index]))
            last_channel_int = int(np.clip(last_channel_int, 0, len(channel_names) - 1))
            pred_channel = channel_names[last_channel_int]

            for hist_step in range(t_history):
                frame = int(start_frames[sample]) + hist_step
                row = lookup.get((track_id, frame))
                if row is None:
                    input_mask[sample, hist_step, token] = False
                else:
                    history_xy[sample, hist_step, token] = [row["x"], row["y"]]

            for future_step in range(t_future):
                frame = int(start_frames[sample]) + t_history + future_step
                row = lookup.get((track_id, frame))
                if row is None:
                    target_mask[sample, future_step, token] = False
                    continue

                true_xy[sample, future_step, token] = [row["x"], row["y"]]
                channel_labels[sample, future_step, token] = str(row["channel_id"])
                pred_channel = centerlines.infer_channel(
                    float(pred_s[sample, future_step, token]), pred_channel
                )
                pred_xy[sample, future_step, token] = centerlines.geometry_to_xy(
                    float(pred_s[sample, future_step, token]),
                    float(last_d[sample, token]),
                    pred_channel,
                )

    return CartesianTrajectories(
        pred_xy, true_xy, history_xy, target_mask, input_mask, channel_labels
    )


def rmse_components(errors: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    if not mask.any():
        raise ValueError("No valid targets available for metrics.")
    ex = errors[..., 0][mask]
    ey = errors[..., 1][mask]
    return (
        float(np.sqrt(np.mean(ex**2))),
        float(np.sqrt(np.mean(ey**2))),
        float(np.sqrt(np.mean(ex**2 + ey**2))),
    )


def compute_position_metrics(pred_xy: np.ndarray, true_xy: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    errors = pred_xy - true_xy
    rmse_x, rmse_y, rmse_xy = rmse_components(errors, mask)
    by_horizon = {"RMSE_x": [], "RMSE_y": [], "RMSE_xy_euclidean": []}
    for step in range(errors.shape[1]):
        step_mask = mask[:, step]
        if step_mask.any():
            x, y, xy = rmse_components(errors[:, step], step_mask)
        else:
            x = y = xy = float("nan")
        by_horizon["RMSE_x"].append(x)
        by_horizon["RMSE_y"].append(y)
        by_horizon["RMSE_xy_euclidean"].append(xy)
    return {"RMSE_x": rmse_x, "RMSE_y": rmse_y, "RMSE_xy_euclidean": rmse_xy, "by_horizon": by_horizon}


def compute_velocity_metrics(pred_xy: np.ndarray, true_xy: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    pred_v = pred_xy[:, 1:] - pred_xy[:, :-1]
    true_v = true_xy[:, 1:] - true_xy[:, :-1]
    vel_mask = mask[:, 1:] & mask[:, :-1]
    errors = pred_v - true_v
    rmse_vx, rmse_vy, rmse_v = rmse_components(errors, vel_mask)
    by_horizon = {"RMSE_vx": [], "RMSE_vy": [], "RMSE_v_euclidean": []}
    for step in range(errors.shape[1]):
        step_mask = vel_mask[:, step]
        if step_mask.any():
            vx, vy, v = rmse_components(errors[:, step], step_mask)
        else:
            vx = vy = v = float("nan")
        by_horizon["RMSE_vx"].append(vx)
        by_horizon["RMSE_vy"].append(vy)
        by_horizon["RMSE_v_euclidean"].append(v)
    return {"RMSE_vx": rmse_vx, "RMSE_vy": rmse_vy, "RMSE_v_euclidean": rmse_v, "by_horizon": by_horizon}


def safe_rmse_components(errors: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    if not mask.any():
        return float("nan"), float("nan"), float("nan")
    return rmse_components(errors, mask)


def compute_channel_metrics(cart: CartesianTrajectories) -> pd.DataFrame:
    position_errors = cart.pred_xy - cart.true_xy
    pred_v = cart.pred_xy[:, 1:] - cart.pred_xy[:, :-1]
    true_v = cart.true_xy[:, 1:] - cart.true_xy[:, :-1]
    velocity_errors = pred_v - true_v
    velocity_mask = cart.target_mask[:, 1:] & cart.target_mask[:, :-1]
    velocity_channel_labels = cart.channel_labels[:, 1:]

    valid_labels = sorted(
        str(label)
        for label in np.unique(cart.channel_labels[cart.target_mask])
        if str(label)
    )
    rows: list[dict[str, Any]] = []
    for channel_id in valid_labels:
        channel_position_mask = cart.target_mask & (cart.channel_labels == channel_id)
        count = int(channel_position_mask.sum())
        rmse_x, rmse_y, rmse_xy = safe_rmse_components(
            position_errors, channel_position_mask
        )

        channel_velocity_mask = velocity_mask & (velocity_channel_labels == channel_id)
        velocity_count = int(channel_velocity_mask.sum())
        rmse_vx, rmse_vy, rmse_v = safe_rmse_components(
            velocity_errors, channel_velocity_mask
        )
        rows.append(
            {
                "channel_id": channel_id,
                "count": count,
                "velocity_count": velocity_count,
                "RMSE_x": rmse_x,
                "RMSE_y": rmse_y,
                "RMSE_xy_euclidean": rmse_xy,
                "RMSE_vx": rmse_vx,
                "RMSE_vy": rmse_vy,
                "RMSE_v_euclidean": rmse_v,
            }
        )

    metrics_df = pd.DataFrame(rows)
    per_channel_count = int(metrics_df["count"].sum()) if not metrics_df.empty else 0
    total_valid_count = int(cart.target_mask.sum())
    if per_channel_count != total_valid_count:
        raise ValueError(
            "Per-channel valid count sanity check failed: "
            f"sum={per_channel_count}, total={total_valid_count}."
        )
    return metrics_df


def print_channel_metrics(metrics_df: pd.DataFrame) -> None:
    print("\nChannel-wise Cartesian validation metrics:")
    print(
        "channel_id | count | RMSE x | RMSE y | RMSE xy | "
        "RMSE vx | RMSE vy | RMSE v"
    )
    for row in metrics_df.itertuples(index=False):
        print(
            f"{row.channel_id} | {int(row.count)} | "
            f"{row.RMSE_x:.6f} | {row.RMSE_y:.6f} | "
            f"{row.RMSE_xy_euclidean:.6f} | "
            f"{row.RMSE_vx:.6f} | {row.RMSE_vy:.6f} | "
            f"{row.RMSE_v_euclidean:.6f}"
        )


def first_valid_sample(mask: np.ndarray) -> int:
    counts = mask.sum(axis=(1, 2))
    if not counts.any():
        raise ValueError("No validation samples contain valid targets.")
    return int(np.argmax(counts))


def plot_single(
    path: Path,
    cart: CartesianTrajectories,
    sample: int,
    centerlines: CenterlineLookup,
) -> None:
    droplet = int(np.argmax(cart.target_mask[sample].sum(axis=0)))
    plt.figure(figsize=(7, 7))
    centerlines.plot_centerlines()

    hist_mask = cart.input_mask[sample, :, droplet]
    future_mask = cart.target_mask[sample, :, droplet]
    if hist_mask.any():
        plt.plot(cart.history_xy[sample, hist_mask, droplet, 0], cart.history_xy[sample, hist_mask, droplet, 1], "-o", color="0.45", label="history")
    plt.plot(cart.true_xy[sample, future_mask, droplet, 0], cart.true_xy[sample, future_mask, droplet, 1], "-o", label="true future")
    plt.plot(cart.pred_xy[sample, future_mask, droplet, 0], cart.pred_xy[sample, future_mask, droplet, 1], "--x", label="pred future")
    plt.axis("equal")
    plt.gca().invert_yaxis()
    plt.xlabel("x pixels")
    plt.ylabel("y pixels")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_all(path: Path, cart: CartesianTrajectories, sample: int, centerlines: CenterlineLookup) -> None:
    plt.figure(figsize=(8, 8))
    centerlines.plot_centerlines()
    for droplet in np.where(cart.target_mask[sample].any(axis=0))[0]:
        hist_mask = cart.input_mask[sample, :, droplet]
        future_mask = cart.target_mask[sample, :, droplet]
        if hist_mask.any():
            plt.plot(cart.history_xy[sample, hist_mask, droplet, 0], cart.history_xy[sample, hist_mask, droplet, 1], "-", color="0.75", linewidth=1)
        plt.plot(cart.true_xy[sample, future_mask, droplet, 0], cart.true_xy[sample, future_mask, droplet, 1], "-o", markersize=3, alpha=0.8)
        plt.plot(cart.pred_xy[sample, future_mask, droplet, 0], cart.pred_xy[sample, future_mask, droplet, 1], "--x", markersize=3, alpha=0.8)
    plt.axis("equal")
    plt.gca().invert_yaxis()
    plt.xlabel("x pixels")
    plt.ylabel("y pixels")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_horizon(path: Path, title: str, metrics: dict[str, list[float]], x_start: int) -> None:
    plt.figure(figsize=(8, 5))
    for name, values in metrics.items():
        steps = np.arange(x_start, x_start + len(values))
        plt.plot(steps, values, "-o", label=name)
    plt.title(title)
    plt.xlabel("future step")
    plt.ylabel("RMSE pixels/frame" if "velocity" in title.lower() else "RMSE pixels")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_metrics(
    path: Path,
    model_path: Path,
    n_val: int,
    pos: dict[str, Any],
    vel: dict[str, Any],
    channel_metrics: pd.DataFrame,
) -> dict[str, Any]:
    summary = {
        "best_checkpoint_path": str(model_path),
        "model_type": "geometry-aware",
        "coordinate_system_used_for_final_metrics": "Cartesian x,y pixels",
        "number_of_validation_samples": int(n_val),
        "RMSE_x": pos["RMSE_x"],
        "RMSE_y": pos["RMSE_y"],
        "RMSE_xy_euclidean": pos["RMSE_xy_euclidean"],
        "RMSE_vx": vel["RMSE_vx"],
        "RMSE_vy": vel["RMSE_vy"],
        "RMSE_v_euclidean": vel["RMSE_v_euclidean"],
        "position_rmse_by_horizon": pos["by_horizon"],
        "velocity_rmse_by_horizon": vel["by_horizon"],
        "channel_metrics_csv": "geometry_validation_metrics_by_channel.csv",
        "channel_metrics": channel_metrics.to_dict(orient="records"),
        "notes": [
            "Metrics are computed after unnormalizing model outputs and converting geometry predictions to Cartesian pixels.",
            "Current v_s-only model does not predict d_centerline or channel_id; predictions use last observed d_centerline and deterministic channel inference from global s_coord.",
            "d_centerline is unsigned in preprocessing, so geometry_to_xy applies one consistent normal side; true targets use measured x,y from trajectories_geometry.csv.",
        ],
    }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    windows_path = args.windows_file or (
        PROCESSED_DIR / args.experiment_name / "trajectory_windows_geometry_velocity.npz"
    )
    model_path = args.model_file or windows_path.parent / "attention_model.pt"
    trajectories_csv = args.trajectories_csv or windows_path.parent / "trajectories_geometry.csv"
    output_dir = args.output_dir or PROJECT_ROOT / "outputs" / "evaluation" / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    windows = load_windows(windows_path)
    val_indices = split_indices(windows["X"].shape[0], seed=args.seed)["val"]
    if len(val_indices) == 0:
        raise ValueError("No validation samples available.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, windows, device)
    pred_normalized = predict(
        model,
        windows["X"][val_indices],
        windows["input_mask"][val_indices],
        args.batch_size,
        device,
    )
    pred_targets = unnormalize_targets(pred_normalized, windows)
    X_physical = unnormalize_features(windows["X"][val_indices], windows)

    cart = build_cartesian_trajectories(
        windows,
        val_indices,
        X_physical,
        pred_targets,
        trajectories_csv,
        args.centerlines_csv,
    )

    pos = compute_position_metrics(cart.pred_xy, cart.true_xy, cart.target_mask)
    vel = compute_velocity_metrics(cart.pred_xy, cart.true_xy, cart.target_mask)
    channel_metrics = compute_channel_metrics(cart)
    channel_metrics_path = windows_path.parent / "geometry_validation_metrics_by_channel.csv"
    channel_metrics.to_csv(channel_metrics_path, index=False)
    centerlines = CenterlineLookup(args.centerlines_csv)
    sample = first_valid_sample(cart.target_mask)

    plot_single(output_dir / "single_droplet_overlay_sample_000.png", cart, sample, centerlines)
    plot_all(output_dir / "all_droplets_overlay_sample_000.png", cart, sample, centerlines)
    plot_horizon(output_dir / "rmse_position_by_horizon.png", "Position RMSE by horizon", pos["by_horizon"], 1)
    plot_horizon(output_dir / "rmse_velocity_by_horizon.png", "Velocity RMSE by horizon", vel["by_horizon"], 1)
    summary = save_metrics(
        output_dir / "metrics_summary.json",
        model_path,
        len(val_indices),
        pos,
        vel,
        channel_metrics,
    )

    print("Geometry-aware validation metrics in Cartesian coordinates:")
    print(f"RMSE x: {summary['RMSE_x']:.6f}")
    print(f"RMSE y: {summary['RMSE_y']:.6f}")
    print(f"RMSE xy euclidean: {summary['RMSE_xy_euclidean']:.6f}")
    print(f"RMSE vx: {summary['RMSE_vx']:.6f}")
    print(f"RMSE vy: {summary['RMSE_vy']:.6f}")
    print(f"RMSE velocity euclidean: {summary['RMSE_v_euclidean']:.6f}")
    print_channel_metrics(channel_metrics)
    print(
        "\nPer-channel valid count sanity check: "
        f"{int(channel_metrics['count'].sum())} == {int(cart.target_mask.sum())}"
    )
    print(f"Saved per-channel metrics to: {channel_metrics_path}")
    print(f"\nSaved evaluation outputs to: {output_dir}")


if __name__ == "__main__":
    main()
