from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from models.canonical_rollout_transformer import CanonicalRolloutTransformer
from train_canonical_rollout_transformer import (
    LOSS_ALPHA,
    ROLLOUT_HORIZON,
    boundary_conditioned_rollout,
    move_batch_to_device,
    rollout_weights,
)
from utils.canonical_dataset.canonical_window_dataset import create_train_val_test_datasets


CHECKPOINT_PATH = "outputs/models/train_canonical_rollout_transformer/canonical_rollout_transformer_best.pt"
NPZ_PATH = "outputs/processed/2/canonical_dataset.npz"
OUTPUT_CSV = "outputs/post_processing/rollout_metrics.csv"
OUTPUT_ANIMATION = "outputs/post_processing/rollout_sample0.mp4"

STRIDE = 5
BATCH_SIZE = 4
NUM_WORKERS = 0


def main() -> None:
    args = parse_args()
    device, model, datasets, normalization_stats, weights = load_model_and_datasets(args)

    split_names = ["val", "test"] if args.split == "both" else [args.split]
    all_rows = []
    first_animation_payload = None

    for split_name in split_names:
        dataset = datasets[split_name]
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
        )
        rows, animation_payload = evaluate_split(
            model=model,
            loader=loader,
            dataset=dataset,
            normalization_stats=normalization_stats,
            weights=weights,
            device=device,
            split_name=split_name,
        )
        all_rows.extend(rows)
        if first_animation_payload is None:
            first_animation_payload = animation_payload

    save_metrics(all_rows, args.output_csv)
    print_summary(all_rows)

    if first_animation_payload is not None:
        make_animation(first_animation_payload, args.output_animation)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the boundary-conditioned rollout Transformer.")
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    parser.add_argument("--npz-path", default=NPZ_PATH)
    parser.add_argument("--output-csv", default=OUTPUT_CSV)
    parser.add_argument("--output-animation", default=OUTPUT_ANIMATION)
    parser.add_argument("--split", choices=("val", "test", "both"), default="val")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--stride", type=int, default=STRIDE)
    return parser.parse_args()


def load_model_and_datasets(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_config = checkpoint["model_config"]
    normalization_stats = checkpoint["normalization_stats"]
    horizon = int(checkpoint.get("rollout_horizon", ROLLOUT_HORIZON))
    loss_alpha = float(checkpoint.get("loss_alpha", LOSS_ALPHA))
    stride = int(checkpoint.get("stride", args.stride))

    train_ds, val_ds, test_ds, _ = create_train_val_test_datasets(
        npz_path=args.npz_path,
        stride=stride,
        T_history=model_config["T_history"],
        T_future=horizon,
        max_droplets=model_config["max_droplets"],
        target_features=tuple(str(name) for name in normalization_stats["target_features"]),
    )
    train_ds.normalization_stats = normalization_stats
    val_ds.normalization_stats = normalization_stats
    test_ds.normalization_stats = normalization_stats

    model = CanonicalRolloutTransformer(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    weights = rollout_weights(horizon, loss_alpha, device)

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Validation windows: {len(val_ds)}")
    print(f"Test windows: {len(test_ds)}")

    return device, model, {"val": val_ds, "test": test_ds}, normalization_stats, weights


def evaluate_split(model, loader, dataset, normalization_stats, weights, device, split_name):
    accumulators = create_accumulators(weights.numel())
    weighted_loss_sum = 0.0
    total_boundary = 0
    total_valid = 0
    total_rollouts = 0
    num_batches = 0
    animation_payload = None

    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            batch = move_batch_to_device(batch, device)
            rollout = boundary_conditioned_rollout(
                model=model,
                batch=batch,
                dataset=dataset,
                normalization_stats=normalization_stats,
                weights=weights,
            )
            weighted_loss_sum += float(rollout["weighted_loss_internal_only"].detach().cpu())
            total_boundary += int(rollout["boundary_mask"].sum().detach().cpu())
            total_valid += int(rollout["mask"].sum().detach().cpu())
            total_rollouts += int(batch["history_x"].shape[0])
            num_batches += 1
            update_metric_accumulators(accumulators, rollout)

            if animation_payload is None:
                animation_payload = extract_sample_animation_payload(rollout, batch, split_name)

            if batch_index % 25 == 0 or batch_index == len(loader):
                print(f"Evaluated {split_name} {batch_index}/{len(loader)} batches")

    diagnostics = {
        "weighted_loss_internal_only": weighted_loss_sum / max(num_batches, 1),
        "boundary_injected_total": total_boundary,
        "valid_future_samples_total": total_valid,
        "boundary_injected_per_rollout": total_boundary / max(total_rollouts, 1),
        "boundary_fraction": total_boundary / max(total_valid, 1),
    }
    rows = build_metric_rows(accumulators, diagnostics, split_name)
    return rows, animation_payload


def create_accumulators(num_steps):
    return {
        "overall": new_accumulator(),
        "steps": [new_accumulator() for _ in range(num_steps)],
    }


def new_accumulator():
    return {
        "count": 0,
        "sum_sq_vx": 0.0,
        "sum_sq_vy": 0.0,
        "sum_sq_speed": 0.0,
        "sum_abs_vx": 0.0,
        "sum_abs_vy": 0.0,
        "sum_abs_speed": 0.0,
        "position_count": 0,
        "sum_sq_x": 0.0,
        "sum_sq_y": 0.0,
        "sum_sq_position": 0.0,
        "sum_abs_x": 0.0,
        "sum_abs_y": 0.0,
        "sum_abs_position": 0.0,
    }


def update_metric_accumulators(accumulators, rollout):
    velocity_error = rollout["pred_velocity"] - rollout["true_velocity"]
    speed_error = torch.sqrt(velocity_error[..., 0] ** 2 + velocity_error[..., 1] ** 2)
    position_error = rollout["pred_position"] - rollout["true_position"]
    position_error_norm = torch.sqrt(position_error[..., 0] ** 2 + position_error[..., 1] ** 2)
    position_finite = torch.isfinite(position_error).all(dim=-1)

    update_one_accumulator(
        accumulators["overall"],
        velocity_error,
        speed_error,
        rollout["mask"],
        position_error,
        position_error_norm,
        rollout["mask"] & position_finite,
    )
    for step_index in range(rollout["mask"].shape[1]):
        update_one_accumulator(
            accumulators["steps"][step_index],
            velocity_error[:, step_index, :, :],
            speed_error[:, step_index, :],
            rollout["mask"][:, step_index, :],
            position_error[:, step_index, :, :],
            position_error_norm[:, step_index, :],
            rollout["mask"][:, step_index, :] & position_finite[:, step_index, :],
        )


def update_one_accumulator(
    accumulator,
    velocity_error,
    speed_error,
    velocity_mask,
    position_error,
    position_error_norm,
    position_mask,
):
    valid = velocity_mask.bool()
    if valid.sum().item() > 0:
        vx_error = velocity_error[..., 0][valid]
        vy_error = velocity_error[..., 1][valid]
        speed = speed_error[valid]
        accumulator["count"] += int(valid.sum().item())
        accumulator["sum_sq_vx"] += float((vx_error**2).sum().detach().cpu())
        accumulator["sum_sq_vy"] += float((vy_error**2).sum().detach().cpu())
        accumulator["sum_sq_speed"] += float((speed**2).sum().detach().cpu())
        accumulator["sum_abs_vx"] += float(vx_error.abs().sum().detach().cpu())
        accumulator["sum_abs_vy"] += float(vy_error.abs().sum().detach().cpu())
        accumulator["sum_abs_speed"] += float(speed.abs().sum().detach().cpu())

    valid_position = position_mask.bool()
    if valid_position.sum().item() > 0:
        x_error = position_error[..., 0][valid_position]
        y_error = position_error[..., 1][valid_position]
        position = position_error_norm[valid_position]
        accumulator["position_count"] += int(valid_position.sum().item())
        accumulator["sum_sq_x"] += float((x_error**2).sum().detach().cpu())
        accumulator["sum_sq_y"] += float((y_error**2).sum().detach().cpu())
        accumulator["sum_sq_position"] += float((position**2).sum().detach().cpu())
        accumulator["sum_abs_x"] += float(x_error.abs().sum().detach().cpu())
        accumulator["sum_abs_y"] += float(y_error.abs().sum().detach().cpu())
        accumulator["sum_abs_position"] += float(position.abs().sum().detach().cpu())


def build_metric_rows(accumulators, diagnostics, split_name):
    rows = []
    for step_index, accumulator in enumerate(accumulators["steps"], start=1):
        row = metrics_from_accumulator(accumulator)
        row["split"] = split_name
        row["horizon"] = f"Step {step_index}"
        row["weighted_loss_internal_only"] = np.nan
        row["boundary_injected_total"] = np.nan
        row["valid_future_samples_total"] = np.nan
        row["boundary_injected_per_rollout"] = np.nan
        row["boundary_fraction"] = np.nan
        rows.append(row)

    overall = metrics_from_accumulator(accumulators["overall"])
    overall["split"] = split_name
    overall["horizon"] = "Overall"
    overall.update(diagnostics)
    rows.append(overall)
    return rows


def metrics_from_accumulator(accumulator):
    count = accumulator["count"]
    position_count = accumulator["position_count"]
    return {
        "valid_samples": count,
        "valid_position_samples": position_count,
        "rmse_vx": safe_rmse(accumulator["sum_sq_vx"], count),
        "rmse_vy": safe_rmse(accumulator["sum_sq_vy"], count),
        "rmse_speed": safe_rmse(accumulator["sum_sq_speed"], count),
        "mae_vx": safe_mae(accumulator["sum_abs_vx"], count),
        "mae_vy": safe_mae(accumulator["sum_abs_vy"], count),
        "mae_speed": safe_mae(accumulator["sum_abs_speed"], count),
        "rmse_x": safe_rmse(accumulator["sum_sq_x"], position_count),
        "rmse_y": safe_rmse(accumulator["sum_sq_y"], position_count),
        "rmse_position": safe_rmse(accumulator["sum_sq_position"], position_count),
        "mae_x": safe_mae(accumulator["sum_abs_x"], position_count),
        "mae_y": safe_mae(accumulator["sum_abs_y"], position_count),
        "mae_position": safe_mae(accumulator["sum_abs_position"], position_count),
    }


def safe_rmse(sum_sq, count):
    return np.sqrt(sum_sq / count) if count else np.nan


def safe_mae(sum_abs, count):
    return sum_abs / count if count else np.nan


def save_metrics(rows, output_csv):
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Saved metrics CSV: {output_path}")


def print_summary(rows):
    print("----------------------------------------------------")
    print("Boundary-Conditioned Rollout Metrics")
    print()
    for row in rows:
        if row["horizon"] != "Overall":
            continue
        print(f"Split            : {row['split']}")
        print(f"Valid velocity   : {row['valid_samples']}")
        print(f"Valid position   : {row['valid_position_samples']}")
        print(f"Weighted loss internal only : {row['weighted_loss_internal_only']:.6f}")
        print(f"Boundary injected per rollout : {row['boundary_injected_per_rollout']:.3f}")
        print(f"Boundary fraction of valid future : {row['boundary_fraction']:.6f}")
        print(f"RMSE vx          : {row['rmse_vx']:.6f}")
        print(f"RMSE vy          : {row['rmse_vy']:.6f}")
        print(f"RMSE speed       : {row['rmse_speed']:.6f}")
        print(f"RMSE position    : {row['rmse_position']:.6f}")
        print()
    print("Stepwise Position RMSE")
    for row in rows:
        if row["horizon"] == "Overall":
            continue
        print(
            f"{row['split']:<4} {row['horizon']:<7} "
            f"valid_xy={row['valid_position_samples']:<8d} "
            f"RMSE(pos)={row['rmse_position']:.6f}"
        )
    print("----------------------------------------------------")


def extract_sample_animation_payload(rollout, batch, split_name):
    sample_index = 0
    return {
        "split": split_name,
        "pred_position": rollout["pred_position"][sample_index].detach().cpu().numpy(),
        "true_position": rollout["true_position"][sample_index].detach().cpu().numpy(),
        "mask": rollout["mask"][sample_index].detach().cpu().numpy(),
        "droplet_ids": batch["droplet_ids"][sample_index].detach().cpu().numpy(),
    }


def make_animation(payload, output_video):
    try:
        import imageio_ffmpeg
    except ImportError:
        print("Skipping animation because imageio-ffmpeg is not installed.")
        return

    pred_position = payload["pred_position"]
    true_position = payload["true_position"]
    mask = payload["mask"].astype(bool)
    droplet_ids = payload["droplet_ids"]

    valid_positions = np.concatenate([pred_position[mask], true_position[mask]], axis=0)
    if valid_positions.size == 0:
        print("Skipping animation because no valid rollout positions are available.")
        return

    x_min, y_min = np.nanmin(valid_positions, axis=0)
    x_max, y_max = np.nanmax(valid_positions, axis=0)
    x_pad = max((x_max - x_min) * 0.05, 1.0)
    y_pad = max((y_max - y_min) * 0.05, 1.0)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_max + y_pad, y_min - y_pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    gt_scatter = ax.scatter([], [], s=35, c="black", label="ground truth", alpha=0.8)
    pred_scatter = ax.scatter([], [], s=35, c="red", marker="x", label="prediction", alpha=0.8)
    ax.legend(loc="upper right")
    id_artists = []

    def draw(step_index):
        nonlocal id_artists
        for artist in id_artists:
            artist.remove()
        id_artists = []

        valid = mask[step_index]
        gt = true_position[step_index, valid]
        pred = pred_position[step_index, valid]
        ids = droplet_ids[valid]

        gt_scatter.set_offsets(gt if gt.size else np.empty((0, 2)))
        pred_scatter.set_offsets(pred if pred.size else np.empty((0, 2)))
        ax.set_title(f"{payload['split']} rollout - step {step_index + 1}")

        for point, track_id in zip(pred, ids):
            id_artists.append(ax.text(point[0] + 2, point[1] + 2, str(track_id), fontsize=7, color="red"))

        return [gt_scatter, pred_scatter, *id_artists]

    plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    animation = FuncAnimation(fig, draw, frames=np.arange(mask.shape[0]), interval=200, blit=False)

    output_path = Path(output_video)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=5)
    animation.save(output_path, writer=writer)
    plt.close(fig)
    print(f"Saved rollout animation: {output_path}")


if __name__ == "__main__":
    main()
