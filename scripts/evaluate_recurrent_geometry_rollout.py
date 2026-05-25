from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from configs.constants import EXPERIMENT_NAME
from configs.paths import PROCESSED_DIR
from scripts.train_recurrent_geometry_model import RecurrentGeometryModel

T_HISTORY = 20
T_FUTURE = 10


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate recurrent geometry model with autoregressive rollout."
    )
    parser.add_argument("--experiment-name", default=EXPERIMENT_NAME)
    parser.add_argument(
        "--windows-file",
        type=Path,
        default=None,
        help=(
            "Optional recurrent geometry NPZ. Defaults to "
            "outputs/processed/<experiment_name>/recurrent_geometry_windows.npz."
        ),
    )
    parser.add_argument(
        "--model-file",
        type=Path,
        default=None,
        help=(
            "Optional checkpoint. Defaults to "
            "outputs/processed/<experiment_name>/recurrent_geometry_rollout_best.pt."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--num-inter-op-threads", type=int, default=1)
    parser.add_argument(
        "--diagnostics-dir",
        type=Path,
        default=None,
        help="Defaults to outputs/diagnostics/step10_blowup/.",
    )
    return parser.parse_args(args)


def npz_array(data: np.lib.npyio.NpzFile, names: list[str]) -> np.ndarray | None:
    for name in names:
        if name in data.files:
            return data[name]
    return None


def load_windows(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Recurrent geometry windows file not found: {path}")
    data = np.load(path)
    required = {
        "Z",
        "mask",
        "target_v_s",
        "target_mask",
        "feature_columns",
        "numeric_features",
        "target_columns",
        "numeric_feature_mean",
        "numeric_feature_std",
        "target_mean",
        "target_std",
        "channel_ids",
    }
    missing = required - set(data.files)
    if missing:
        raise KeyError(f"Missing required NPZ arrays: {sorted(missing)}")
    return {
        "Z": data["Z"],
        "mask": data["mask"].astype(bool),
        "target_v_s": data["target_v_s"],
        "target_mask": data["target_mask"].astype(bool),
        "feature_columns": data["feature_columns"].astype(str),
        "numeric_features": data["numeric_features"].astype(str),
        "target_columns": data["target_columns"].astype(str),
        "numeric_feature_mean": data["numeric_feature_mean"],
        "numeric_feature_std": data["numeric_feature_std"],
        "target_mean": data["target_mean"],
        "target_std": data["target_std"],
        "channel_names": npz_array(data, ["channel_names"]),
        "channel_ids": data["channel_ids"],
        "track_ids": npz_array(data, ["track_ids"]),
    }


def split_indices(
    num_samples: int, train_frac: float = 0.8, seed: int = 42
) -> dict[str, np.ndarray]:
    indices = np.arange(num_samples)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    split = int(num_samples * train_frac)
    return {"train": indices[:split], "val": indices[split:]}


def load_model(model_path: Path, windows: dict[str, Any], device: torch.device) -> RecurrentGeometryModel:
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})

    model = RecurrentGeometryModel(
        numeric_input_dim=int(config.get("numeric_input_dim", len(windows["numeric_features"]))),
        num_channel_embeddings=int(
            config.get("num_channel_embeddings", int(windows["channel_ids"].max()) + 1)
        ),
        use_recurrent=True,
        channel_embedding_dim=int(config.get("channel_embedding_dim", 16)),
        hidden_dim=int(config.get("hidden_dim", 64)),
        num_heads=int(config.get("num_heads", 4)),
        num_layers=int(config.get("num_layers", 2)),
        dropout=float(config.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def denormalize(value: np.ndarray, mean: float, std: float) -> np.ndarray:
    return value * std + mean


def normalize(value: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (value - mean) / std


def transformer_context(
    model: RecurrentGeometryModel,
    z_step: torch.Tensor,
    mask_step: torch.Tensor,
) -> torch.Tensor:
    numeric = z_step[..., : model.numeric_input_dim]
    channel_ids = z_step[..., model.numeric_input_dim].long()
    channel_ids = channel_ids.clamp(min=0, max=model.num_channel_embeddings - 1)
    channel_embedding = model.channel_embed(channel_ids)
    hidden = model.state_encoder(torch.cat([numeric, channel_embedding], dim=-1))
    return model.transformer(hidden, src_key_padding_mask=~mask_step)


def update_hidden(
    model: RecurrentGeometryModel,
    h: torch.Tensor,
    context: torch.Tensor,
    mask_step: torch.Tensor,
) -> torch.Tensor:
    batch_size, n_max, hidden_dim = h.shape
    h_candidate = model.gru_cell(
        context.reshape(batch_size * n_max, hidden_dim),
        h.reshape(batch_size * n_max, hidden_dim),
    ).reshape(batch_size, n_max, hidden_dim)
    return torch.where(mask_step.unsqueeze(-1), h_candidate, h)


def rollout_batch(
    model: RecurrentGeometryModel,
    Z: np.ndarray,
    mask: np.ndarray,
    stats: dict[str, float],
    feature_indices: dict[str, int],
    device: torch.device,
) -> dict[str, np.ndarray]:
    z_batch = torch.from_numpy(Z).float().to(device)
    mask_batch = torch.from_numpy(mask).bool().to(device)
    batch_size, _, n_max, _ = z_batch.shape

    h = torch.zeros(
        batch_size,
        n_max,
        model.hidden_dim,
        dtype=z_batch.dtype,
        device=device,
    )

    for step in range(T_HISTORY):
        context = transformer_context(model, z_batch[:, step], mask_batch[:, step])
        h = update_hidden(model, h, context, mask_batch[:, step])

    current_z = z_batch[:, T_HISTORY - 1].clone()
    pred_vs_norm_steps: list[torch.Tensor] = []
    pred_s_norm_steps: list[torch.Tensor] = []
    raw_s_before_steps: list[torch.Tensor] = []
    raw_v_s_steps: list[torch.Tensor] = []
    raw_s_after_steps: list[torch.Tensor] = []
    pred_s_returned_steps: list[torch.Tensor] = []

    s_index = feature_indices["s_coord"]
    v_s_index = feature_indices["v_s"]

    for step in range(T_FUTURE):
        future_mask_step = mask_batch[:, T_HISTORY + step]
        context = transformer_context(model, current_z, future_mask_step)
        h = update_hidden(model, h, context, future_mask_step)

        pred_v_s_norm = model.velocity_head(h)
        pred_vs_norm_steps.append(pred_v_s_norm)

        current_s_norm = current_z[..., s_index : s_index + 1]
        raw_s = denormalize(current_s_norm, stats["s_mean"], stats["s_std"])
        raw_v_s = denormalize(pred_v_s_norm, stats["target_mean"], stats["target_std"])
        next_raw_s = raw_s + raw_v_s
        next_s_norm = normalize(next_raw_s, stats["s_mean"], stats["s_std"])
        pred_s_norm_steps.append(next_s_norm)
        pred_s_returned_raw = denormalize(next_s_norm, stats["s_mean"], stats["s_std"])
        raw_s_before_steps.append(raw_s)
        raw_v_s_steps.append(raw_v_s)
        raw_s_after_steps.append(next_raw_s)
        pred_s_returned_steps.append(pred_s_returned_raw)

        next_z = current_z.clone()
        next_z[..., s_index : s_index + 1] = next_s_norm
        next_z[..., v_s_index : v_s_index + 1] = pred_v_s_norm
        current_z = next_z

    pred_v_s_norm = torch.stack(pred_vs_norm_steps, dim=1)
    pred_s_norm = torch.stack(pred_s_norm_steps, dim=1)
    raw_s_before = torch.stack(raw_s_before_steps, dim=1)
    raw_v_s = torch.stack(raw_v_s_steps, dim=1)
    raw_s_after = torch.stack(raw_s_after_steps, dim=1)
    pred_s_returned = torch.stack(pred_s_returned_steps, dim=1)

    direct_residual = torch.abs(pred_s_returned - raw_s_after).amax(dim=(0, 2, 3))
    chain_residuals: list[torch.Tensor] = []
    for step in range(T_FUTURE):
        if step == 0:
            expected_s = raw_s_before[:, step] + raw_v_s[:, step]
        else:
            expected_s = pred_s_returned[:, step - 1] + raw_v_s[:, step]
        chain_residuals.append(
            torch.abs(pred_s_returned[:, step] - expected_s).amax()
        )
    chain_residual = torch.stack(chain_residuals)

    return {
        "pred_v_s_norm": pred_v_s_norm.cpu().numpy(),
        "pred_s_norm": pred_s_norm.cpu().numpy(),
        "direct_integration_residual": direct_residual.cpu().numpy(),
        "chain_integration_residual": chain_residual.cpu().numpy(),
    }


def teacher_forced_batch(
    model: RecurrentGeometryModel,
    Z: np.ndarray,
    mask: np.ndarray,
    stats: dict[str, float],
    feature_indices: dict[str, int],
    device: torch.device,
) -> dict[str, np.ndarray]:
    z_batch = torch.from_numpy(Z).float().to(device)
    mask_batch = torch.from_numpy(mask).bool().to(device)
    batch_size, _, n_max, _ = z_batch.shape

    h = torch.zeros(
        batch_size,
        n_max,
        model.hidden_dim,
        dtype=z_batch.dtype,
        device=device,
    )

    for step in range(T_HISTORY):
        context = transformer_context(model, z_batch[:, step], mask_batch[:, step])
        h = update_hidden(model, h, context, mask_batch[:, step])

    pred_vs_norm_steps: list[torch.Tensor] = []
    pred_s_norm_steps: list[torch.Tensor] = []
    s_index = feature_indices["s_coord"]

    for step in range(T_FUTURE):
        previous_frame = T_HISTORY + step - 1
        current_z = z_batch[:, previous_frame]
        current_mask = mask_batch[:, previous_frame]
        context = transformer_context(model, current_z, current_mask)
        h = update_hidden(model, h, context, current_mask)

        pred_v_s_norm = model.velocity_head(h)
        pred_vs_norm_steps.append(pred_v_s_norm)

        current_s_norm = current_z[..., s_index : s_index + 1]
        raw_s = denormalize(current_s_norm, stats["s_mean"], stats["s_std"])
        raw_v_s = denormalize(pred_v_s_norm, stats["target_mean"], stats["target_std"])
        next_raw_s = raw_s + raw_v_s
        pred_s_norm_steps.append(normalize(next_raw_s, stats["s_mean"], stats["s_std"]))

    pred_v_s_norm = torch.stack(pred_vs_norm_steps, dim=1)
    pred_s_norm = torch.stack(pred_s_norm_steps, dim=1)
    return {
        "pred_v_s_norm": pred_v_s_norm.cpu().numpy(),
        "pred_s_norm": pred_s_norm.cpu().numpy(),
    }


def compute_metrics(
    pred_s_norm: np.ndarray,
    pred_v_s_norm: np.ndarray,
    Z: np.ndarray,
    target_v_s: np.ndarray,
    target_mask: np.ndarray,
    stats: dict[str, float],
    feature_indices: dict[str, int],
) -> tuple[pd.DataFrame, dict[str, float]]:
    future_slice = slice(T_HISTORY, T_HISTORY + T_FUTURE)
    target_slice = slice(T_HISTORY - 1, T_HISTORY + T_FUTURE - 1)
    true_s_norm = Z[:, future_slice, :, feature_indices["s_coord"]][..., None]
    true_v_s_norm = target_v_s[:, target_slice]
    eval_mask = target_mask[:, target_slice]

    pred_s_raw = denormalize(pred_s_norm, stats["s_mean"], stats["s_std"])
    true_s_raw = denormalize(true_s_norm, stats["s_mean"], stats["s_std"])
    pred_v_s_raw = denormalize(pred_v_s_norm, stats["target_mean"], stats["target_std"])
    true_v_s_raw = denormalize(true_v_s_norm, stats["target_mean"], stats["target_std"])

    rows: list[dict[str, float]] = []
    for step in range(T_FUTURE):
        step_mask = eval_mask[:, step]
        if step_mask.any():
            s_error = pred_s_raw[:, step, :, 0] - true_s_raw[:, step, :, 0]
            v_error = pred_v_s_raw[:, step, :, 0] - true_v_s_raw[:, step, :, 0]
            norm_error = pred_v_s_norm[:, step, :, 0] - true_v_s_norm[:, step, :, 0]
            rows.append(
                {
                    "horizon_step": step + 1,
                    "count": int(step_mask.sum()),
                    "rmse_s_coord": float(np.sqrt(np.mean(s_error[step_mask] ** 2))),
                    "rmse_v_s": float(np.sqrt(np.mean(v_error[step_mask] ** 2))),
                    "normalized_mse": float(np.mean(norm_error[step_mask] ** 2)),
                }
            )
        else:
            rows.append(
                {
                    "horizon_step": step + 1,
                    "count": 0,
                    "rmse_s_coord": float("nan"),
                    "rmse_v_s": float("nan"),
                    "normalized_mse": float("nan"),
                }
            )

    all_s_error = pred_s_raw[..., 0] - true_s_raw[..., 0]
    all_v_error = pred_v_s_raw[..., 0] - true_v_s_raw[..., 0]
    all_norm_error = pred_v_s_norm[..., 0] - true_v_s_norm[..., 0]
    summary = {
        "rmse_s_coord": float(np.sqrt(np.mean(all_s_error[eval_mask] ** 2))),
        "rmse_v_s": float(np.sqrt(np.mean(all_v_error[eval_mask] ** 2))),
        "normalized_mse": float(np.mean(all_norm_error[eval_mask] ** 2)),
        "count": int(eval_mask.sum()),
    }
    return pd.DataFrame(rows), summary


def print_horizon_metrics(horizon_metrics: pd.DataFrame) -> None:
    print("\nPer-step rollout metrics:")
    print("step | count | RMSE s_coord | RMSE v_s | normalized MSE")
    for row in horizon_metrics.itertuples(index=False):
        print(
            f"{int(row.horizon_step):>4} | "
            f"{int(row.count):>5} | "
            f"{row.rmse_s_coord:>12.6f} | "
            f"{row.rmse_v_s:>8.6f} | "
            f"{row.normalized_mse:>14.6f}"
        )


def channel_label(channel_value: float, channel_names: np.ndarray | None) -> str:
    channel_index = int(round(float(channel_value)))
    if channel_names is not None and 0 <= channel_index < len(channel_names):
        return str(channel_names[channel_index])
    return str(channel_index)


def channel_ranges_from_dataset(
    Z: np.ndarray,
    mask: np.ndarray,
    stats: dict[str, float],
    feature_indices: dict[str, int],
    channel_names: np.ndarray | None,
) -> dict[str, tuple[float, float]]:
    s_raw = denormalize(
        Z[..., feature_indices["s_coord"]], stats["s_mean"], stats["s_std"]
    )
    channel_values = Z[..., feature_indices["channel_id_int"]]
    ranges: dict[str, tuple[float, float]] = {}
    for channel_value in sorted(np.unique(channel_values[mask])):
        label = channel_label(float(channel_value), channel_names)
        channel_mask = mask & (np.rint(channel_values).astype(int) == int(round(channel_value)))
        if channel_mask.any():
            ranges[label] = (
                float(np.min(s_raw[channel_mask])),
                float(np.max(s_raw[channel_mask])),
            )
    return ranges


def save_mode_comparison(
    diagnostics_dir: Path,
    autoregressive_metrics: pd.DataFrame,
    teacher_forced_metrics: pd.DataFrame,
) -> None:
    ar = autoregressive_metrics.copy()
    tf = teacher_forced_metrics.copy()
    ar["mode"] = "autoregressive"
    tf["mode"] = "teacher_forced"
    comparison = pd.concat([ar, tf], ignore_index=True)
    comparison.to_csv(diagnostics_dir / "teacher_forced_vs_autoregressive.csv", index=False)

    print("\nTeacher-forced vs autoregressive per-step RMSE:")
    print("step | AR RMSE s | TF RMSE s | AR RMSE v_s | TF RMSE v_s")
    for step in range(1, T_FUTURE + 1):
        ar_row = ar.loc[ar["horizon_step"] == step].iloc[0]
        tf_row = tf.loc[tf["horizon_step"] == step].iloc[0]
        print(
            f"{step:>4} | "
            f"{ar_row['rmse_s_coord']:>9.6f} | "
            f"{tf_row['rmse_s_coord']:>9.6f} | "
            f"{ar_row['rmse_v_s']:>11.6f} | "
            f"{tf_row['rmse_v_s']:>11.6f}"
        )


def save_worst_per_step(
    diagnostics_dir: Path,
    val_indices: np.ndarray,
    pred_s_norm: np.ndarray,
    pred_v_s_norm: np.ndarray,
    Z_val: np.ndarray,
    target_v_s_val: np.ndarray,
    target_mask_val: np.ndarray,
    track_ids_val: np.ndarray | None,
    stats: dict[str, float],
    feature_indices: dict[str, int],
    worst_n: int = 20,
) -> pd.DataFrame:
    target_slice = slice(T_HISTORY - 1, T_HISTORY + T_FUTURE - 1)
    eval_mask = target_mask_val[:, target_slice]
    pred_s_raw = denormalize(pred_s_norm[..., 0], stats["s_mean"], stats["s_std"])
    true_s_raw = denormalize(
        Z_val[:, T_HISTORY : T_HISTORY + T_FUTURE, :, feature_indices["s_coord"]],
        stats["s_mean"],
        stats["s_std"],
    )
    pred_v_raw = denormalize(
        pred_v_s_norm[..., 0], stats["target_mean"], stats["target_std"]
    )
    true_v_raw = denormalize(
        target_v_s_val[:, target_slice, :, 0],
        stats["target_mean"],
        stats["target_std"],
    )
    abs_s_error = np.abs(pred_s_raw - true_s_raw)
    rows: list[dict[str, Any]] = []
    for step in range(T_FUTURE):
        valid_positions = np.argwhere(eval_mask[:, step])
        step_rows: list[dict[str, Any]] = []
        for sample, droplet in valid_positions:
            track_id = (
                int(track_ids_val[sample, droplet])
                if track_ids_val is not None
                else -1
            )
            step_rows.append(
                {
                    "horizon_step": step + 1,
                    "sample_index": int(val_indices[sample]),
                    "validation_sample_index": int(sample),
                    "droplet_index": int(droplet),
                    "track_id": track_id,
                    "abs_s_error": float(abs_s_error[sample, step, droplet]),
                    "pred_s": float(pred_s_raw[sample, step, droplet]),
                    "true_s": float(true_s_raw[sample, step, droplet]),
                    "pred_v_s": float(pred_v_raw[sample, step, droplet]),
                    "true_v_s": float(true_v_raw[sample, step, droplet]),
                }
            )
        rows.extend(
            sorted(step_rows, key=lambda row: row["abs_s_error"], reverse=True)[:worst_n]
        )
    worst_df = pd.DataFrame(rows)
    worst_df.to_csv(diagnostics_dir / "rollout_worst_windows_by_step.csv", index=False)
    return worst_df


def save_worst_trajectory_details(
    diagnostics_dir: Path,
    worst_step10: pd.DataFrame,
    pred_s_norm: np.ndarray,
    Z_val: np.ndarray,
    mask_val: np.ndarray,
    target_mask_val: np.ndarray,
    stats: dict[str, float],
    feature_indices: dict[str, int],
    max_plots: int = 5,
) -> None:
    pred_s_raw = denormalize(pred_s_norm[..., 0], stats["s_mean"], stats["s_std"])
    true_s_raw = denormalize(
        Z_val[..., feature_indices["s_coord"]],
        stats["s_mean"],
        stats["s_std"],
    )
    rows: list[dict[str, Any]] = []
    for rank, row in enumerate(worst_step10.head(max_plots).itertuples(index=False), start=1):
        sample = int(row.validation_sample_index)
        droplet = int(row.droplet_index)
        history_steps = np.arange(T_HISTORY)
        future_steps = np.arange(1, T_FUTURE + 1)
        history_s = true_s_raw[sample, :T_HISTORY, droplet]
        gt_future_s = true_s_raw[sample, T_HISTORY : T_HISTORY + T_FUTURE, droplet]
        pred_future_s = pred_s_raw[sample, :, droplet]

        for step in range(T_HISTORY):
            rows.append(
                {
                    "rank": rank,
                    "sample_index": int(row.sample_index),
                    "droplet_index": droplet,
                    "phase": "history",
                    "step": int(step),
                    "s_coord": float(history_s[step]),
                    "mask": bool(mask_val[sample, step, droplet]),
                }
            )
        for step in range(T_FUTURE):
            rows.append(
                {
                    "rank": rank,
                    "sample_index": int(row.sample_index),
                    "droplet_index": droplet,
                    "phase": "future",
                    "step": int(step + 1),
                    "gt_s_coord": float(gt_future_s[step]),
                    "pred_s_coord": float(pred_future_s[step]),
                    "abs_s_error": float(abs(pred_future_s[step] - gt_future_s[step])),
                    "target_mask": bool(
                        target_mask_val[sample, T_HISTORY - 1 + step, droplet]
                    ),
                }
            )

        plt.figure(figsize=(8, 4.8))
        plt.plot(history_steps - T_HISTORY + 1, history_s, "-o", label="history")
        plt.plot(future_steps, gt_future_s, "-o", label="GT future")
        plt.plot(future_steps, pred_future_s, "--x", label="pred future")
        plt.scatter([10], [pred_future_s[-1]], s=90, facecolors="none", edgecolors="red", label="step 10")
        plt.xlabel("rollout step")
        plt.ylabel("raw s_coord")
        plt.title(
            f"Worst step-10 rollout rank {rank}: sample {row.sample_index}, droplet {droplet}"
        )
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(diagnostics_dir / f"worst_step10_rank_{rank:02d}.png", dpi=160)
        plt.close()

    pd.DataFrame(rows).to_csv(
        diagnostics_dir / "rollout_worst_step10_trajectories.csv", index=False
    )


def print_and_save_step10_diagnostics(
    output_dir: Path,
    diagnostics_dir: Path,
    val_indices: np.ndarray,
    pred_s_norm: np.ndarray,
    pred_v_s_norm: np.ndarray,
    Z_val: np.ndarray,
    target_v_s_val: np.ndarray,
    target_mask_val: np.ndarray,
    track_ids_val: np.ndarray | None,
    windows: dict[str, Any],
    stats: dict[str, float],
    feature_indices: dict[str, int],
    worst_n: int = 20,
) -> None:
    target_slice = slice(T_HISTORY - 1, T_HISTORY + T_FUTURE - 1)
    eval_mask = target_mask_val[:, target_slice]
    step10_mask = eval_mask[:, 9]

    pred_s_raw = denormalize(pred_s_norm[..., 0], stats["s_mean"], stats["s_std"])
    true_s_raw = denormalize(
        Z_val[:, T_HISTORY : T_HISTORY + T_FUTURE, :, feature_indices["s_coord"]],
        stats["s_mean"],
        stats["s_std"],
    )
    pred_v_raw = denormalize(
        pred_v_s_norm[..., 0], stats["target_mean"], stats["target_std"]
    )
    true_v_raw = denormalize(
        target_v_s_val[:, target_slice, :, 0],
        stats["target_mean"],
        stats["target_std"],
    )
    abs_s_error = np.abs(pred_s_raw - true_s_raw)

    channel_names = windows["channel_names"]
    channel_index = feature_indices["channel_id_int"]
    channel_start_values = Z_val[:, T_HISTORY - 1, :, channel_index]
    channel_step10_values = Z_val[:, T_HISTORY + 9, :, channel_index]
    changed_channel = (
        np.rint(channel_start_values).astype(int)
        != np.rint(channel_step10_values).astype(int)
    )

    rows: list[dict[str, Any]] = []
    for sample in range(Z_val.shape[0]):
        original_sample_index = int(val_indices[sample])
        for droplet in range(Z_val.shape[2]):
            valid = bool(step10_mask[sample, droplet])
            track_id = (
                int(track_ids_val[sample, droplet])
                if track_ids_val is not None
                else -1
            )
            rows.append(
                {
                    "sample_index": original_sample_index,
                    "validation_sample_index": sample,
                    "droplet_index": droplet,
                    "track_id": track_id,
                    "abs_s_error": float(abs_s_error[sample, 9, droplet]) if valid else np.nan,
                    "pred_s": float(pred_s_raw[sample, 9, droplet]) if valid else np.nan,
                    "true_s": float(true_s_raw[sample, 9, droplet]) if valid else np.nan,
                    "pred_v_s": float(pred_v_raw[sample, 9, droplet]) if valid else np.nan,
                    "true_v_s": float(true_v_raw[sample, 9, droplet]) if valid else np.nan,
                    "channel_start": channel_label(
                        channel_start_values[sample, droplet], channel_names
                    ),
                    "channel_step10": channel_label(
                        channel_step10_values[sample, droplet], channel_names
                    ),
                    "changed_channel": bool(changed_channel[sample, droplet]),
                    "valid_step10": valid,
                }
            )

    diagnostics = pd.DataFrame(rows)
    diagnostics_path = output_dir / "rollout_step10_diagnostics.csv"
    diagnostics.to_csv(diagnostics_path, index=False)

    step_counts = eval_mask.sum(axis=(0, 2))
    step1_count = max(int(step_counts[0]), 1)
    print("\nValid target counts by rollout horizon:")
    for step, count in enumerate(step_counts, start=1):
        print(
            f"  step {step:>2}: {int(count):>6} "
            f"({float(count) / step1_count:.4f} of step 1)"
        )
    disappeared_9_to_10 = int((eval_mask[:, 8] & ~eval_mask[:, 9]).sum())
    print(f"Invalidated from step 9 to step 10: {disappeared_9_to_10}")

    step10_error = pred_s_raw[:, 9] - true_s_raw[:, 9]
    same_mask = step10_mask & ~changed_channel
    changed_mask = step10_mask & changed_channel
    print("\nStep-10 channel transition split:")
    if same_mask.any():
        same_rmse = float(np.sqrt(np.mean(step10_error[same_mask] ** 2)))
        print(
            f"Step-10 RMSE s_coord (same-channel): {same_rmse:.6f} "
            f"count={int(same_mask.sum())}"
        )
    else:
        print("Step-10 RMSE s_coord (same-channel): nan count=0")
    if changed_mask.any():
        changed_rmse = float(np.sqrt(np.mean(step10_error[changed_mask] ** 2)))
        print(
            f"Step-10 RMSE s_coord (changed-channel): {changed_rmse:.6f} "
            f"count={int(changed_mask.sum())}"
        )
    else:
        print("Step-10 RMSE s_coord (changed-channel): nan count=0")

    print("\nStep-10 RMSE by true channel:")
    for label in ["inlet", "left", "right", "outlet"]:
        label_mask = np.array(
            [
                [
                    channel_label(value, channel_names) == label
                    for value in row
                ]
                for row in channel_step10_values
            ],
            dtype=bool,
        )
        channel_mask = step10_mask & label_mask
        if channel_mask.any():
            rmse = float(np.sqrt(np.mean(step10_error[channel_mask] ** 2)))
            print(f"  {label}: RMSE={rmse:.6f} count={int(channel_mask.sum())}")
        else:
            print(f"  {label}: RMSE=nan count=0")

    channel_ranges = channel_ranges_from_dataset(
        windows["Z"], windows["mask"], stats, feature_indices, channel_names
    )
    worst = diagnostics.loc[diagnostics["valid_step10"]].sort_values(
        "abs_s_error", ascending=False
    ).head(worst_n)
    print(f"\nWorst {len(worst)} step-10 absolute s_coord errors:")
    for row in worst.itertuples(index=False):
        channel_range = channel_ranges.get(row.channel_start)
        if channel_range is None:
            crossed_boundary = "unknown"
        else:
            crossed_boundary = not (channel_range[0] <= row.pred_s <= channel_range[1])
        print(
            f"sample={row.sample_index} val_sample={row.validation_sample_index} "
            f"droplet={row.droplet_index} track_id={row.track_id} "
            f"pred_s={row.pred_s:.3f} true_s={row.true_s:.3f} "
            f"abs_err={row.abs_s_error:.3f} "
            f"pred_v_s={row.pred_v_s:.3f} true_v_s={row.true_v_s:.3f} "
            f"channel_start={row.channel_start} channel_step10={row.channel_step10} "
            f"changed_channel={row.changed_channel} "
            f"pred_crossed_start_channel_range={crossed_boundary}"
        )
    print(f"\nSaved step-10 diagnostics: {diagnostics_path}")
    save_worst_trajectory_details(
        diagnostics_dir,
        worst,
        pred_s_norm,
        Z_val,
        windows["mask"][val_indices],
        target_mask_val,
        stats,
        feature_indices,
    )


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.num_threads)
    torch.set_num_interop_threads(args.num_inter_op_threads)

    output_dir = PROCESSED_DIR / args.experiment_name
    windows_path = args.windows_file or output_dir / "recurrent_geometry_windows.npz"
    model_path = args.model_file or output_dir / "recurrent_geometry_rollout_best.pt"
    metrics_path = output_dir / "recurrent_geometry_rollout_metrics.csv"
    diagnostics_dir = args.diagnostics_dir or (
        output_dir.parent.parent / "diagnostics" / "step10_blowup"
    )
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    windows = load_windows(windows_path)
    val_indices = split_indices(windows["Z"].shape[0])["val"]
    if len(val_indices) == 0:
        raise ValueError("No validation samples available.")

    feature_columns = list(windows["feature_columns"])
    numeric_features = list(windows["numeric_features"])
    target_columns = list(windows["target_columns"])
    feature_indices = {
        "s_coord": feature_columns.index("s_coord"),
        "v_s": feature_columns.index("v_s"),
        "channel_id_int": feature_columns.index("channel_id_int"),
    }
    s_numeric_index = numeric_features.index("s_coord")
    target_index = target_columns.index("v_s")
    stats = {
        "s_mean": float(windows["numeric_feature_mean"][s_numeric_index]),
        "s_std": float(windows["numeric_feature_std"][s_numeric_index]),
        "target_mean": float(windows["target_mean"][target_index]),
        "target_std": float(windows["target_std"][target_index]),
    }

    device = torch.device("cpu")
    model = load_model(model_path, windows, device)

    pred_s_batches: list[np.ndarray] = []
    pred_v_batches: list[np.ndarray] = []
    tf_pred_s_batches: list[np.ndarray] = []
    tf_pred_v_batches: list[np.ndarray] = []
    direct_residual_max = np.zeros((T_FUTURE,), dtype=np.float64)
    chain_residual_max = np.zeros((T_FUTURE,), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(val_indices), args.batch_size):
            batch_indices = val_indices[start : start + args.batch_size]
            batch = rollout_batch(
                model,
                windows["Z"][batch_indices],
                windows["mask"][batch_indices],
                stats,
                feature_indices,
                device,
            )
            pred_s_batches.append(batch["pred_s_norm"])
            pred_v_batches.append(batch["pred_v_s_norm"])
            direct_residual_max = np.maximum(
                direct_residual_max, batch["direct_integration_residual"]
            )
            chain_residual_max = np.maximum(
                chain_residual_max, batch["chain_integration_residual"]
            )
            teacher_batch = teacher_forced_batch(
                model,
                windows["Z"][batch_indices],
                windows["mask"][batch_indices],
                stats,
                feature_indices,
                device,
            )
            tf_pred_s_batches.append(teacher_batch["pred_s_norm"])
            tf_pred_v_batches.append(teacher_batch["pred_v_s_norm"])

    pred_s_norm = np.concatenate(pred_s_batches, axis=0)
    pred_v_s_norm = np.concatenate(pred_v_batches, axis=0)
    tf_pred_s_norm = np.concatenate(tf_pred_s_batches, axis=0)
    tf_pred_v_s_norm = np.concatenate(tf_pred_v_batches, axis=0)
    horizon_metrics, summary = compute_metrics(
        pred_s_norm,
        pred_v_s_norm,
        windows["Z"][val_indices],
        windows["target_v_s"][val_indices],
        windows["target_mask"][val_indices],
        stats,
        feature_indices,
    )
    tf_horizon_metrics, tf_summary = compute_metrics(
        tf_pred_s_norm,
        tf_pred_v_s_norm,
        windows["Z"][val_indices],
        windows["target_v_s"][val_indices],
        windows["target_mask"][val_indices],
        stats,
        feature_indices,
    )

    global_row = pd.DataFrame(
        [
            {
                "horizon_step": "global",
                "count": summary["count"],
                "rmse_s_coord": summary["rmse_s_coord"],
                "rmse_v_s": summary["rmse_v_s"],
                "normalized_mse": summary["normalized_mse"],
            }
        ]
    )
    pd.concat([global_row, horizon_metrics], ignore_index=True).to_csv(
        metrics_path, index=False
    )

    print("Recurrent geometry autoregressive rollout metrics:")
    print(f"Validation samples: {len(val_indices)}")
    print(f"RMSE s_coord: {summary['rmse_s_coord']:.6f}")
    print(f"RMSE v_s: {summary['rmse_v_s']:.6f}")
    print(f"Global normalized MSE: {summary['normalized_mse']:.6f}")
    print("\nRollout integration consistency diagnostics:")
    print("step | max |pred_s_returned - (raw_s_before + raw_v_s)| | max chain residual")
    for step in range(T_FUTURE):
        print(
            f"{step + 1:>4} | "
            f"{direct_residual_max[step]:>45.9f} | "
            f"{chain_residual_max[step]:>18.9f}"
        )
    print_horizon_metrics(horizon_metrics)
    print(
        "\nTeacher-forced global summary: "
        f"RMSE s_coord={tf_summary['rmse_s_coord']:.6f}, "
        f"RMSE v_s={tf_summary['rmse_v_s']:.6f}, "
        f"normalized MSE={tf_summary['normalized_mse']:.6f}"
    )
    save_mode_comparison(diagnostics_dir, horizon_metrics, tf_horizon_metrics)
    worst_by_step = save_worst_per_step(
        diagnostics_dir,
        val_indices,
        pred_s_norm,
        pred_v_s_norm,
        windows["Z"][val_indices],
        windows["target_v_s"][val_indices],
        windows["target_mask"][val_indices],
        windows["track_ids"][val_indices] if windows["track_ids"] is not None else None,
        stats,
        feature_indices,
    )
    print_and_save_step10_diagnostics(
        output_dir,
        diagnostics_dir,
        val_indices,
        pred_s_norm,
        pred_v_s_norm,
        windows["Z"][val_indices],
        windows["target_v_s"][val_indices],
        windows["target_mask"][val_indices],
        windows["track_ids"][val_indices] if windows["track_ids"] is not None else None,
        windows,
        stats,
        feature_indices,
    )
    print(f"Saved worst-window per-step diagnostics: {diagnostics_dir / 'rollout_worst_windows_by_step.csv'}")
    print(f"Saved teacher-forced comparison: {diagnostics_dir / 'teacher_forced_vs_autoregressive.csv'}")
    if not horizon_metrics.empty:
        step_1 = horizon_metrics.loc[horizon_metrics["horizon_step"] == 1].iloc[0]
        step_10 = horizon_metrics.loc[horizon_metrics["horizon_step"] == 10].iloc[0]
        print(f"\nStep 1 RMSE s_coord: {step_1['rmse_s_coord']:.6f}")
        print(f"Step 10 RMSE s_coord: {step_10['rmse_s_coord']:.6f}")
    print(f"Saved rollout metrics: {metrics_path}")


if __name__ == "__main__":
    main()
