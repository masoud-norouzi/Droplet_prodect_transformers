from __future__ import annotations

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

from models.canonical_window_transformer import CanonicalWindowTransformer
from utils.canonical_dataset.canonical_window_dataset import create_train_val_test_datasets


CHECKPOINT_PATH = "outputs/models/canonical_window_transformer_best.pt"
NPZ_PATH = "outputs/processed/2/canonical_dataset.npz"
OUTPUT_CSV = "outputs/post_processing/teacher_forced_rollout_metrics.csv"
OUTPUT_ANIMATION = "outputs/post_processing/teacher_forced_rollout_sample0.mp4"

STRIDE = 5
BATCH_SIZE = 8
NUM_WORKERS = 0
ROLLOUT_STEPS = 10
PRINT_FIRST_SAMPLE_DIAGNOSTIC = True


def main() -> None:
    device, model, val_dataset, val_loader, normalization_stats = load_model_and_dataset()
    accumulators = create_accumulators(ROLLOUT_STEPS, include_position=True)

    sample_animation_payload = None
    printed_first_sample_diagnostic = False
    with torch.no_grad():
        for batch_index, batch in enumerate(val_loader, start=1):
            batch = move_batch_to_device(batch, device)
            if PRINT_FIRST_SAMPLE_DIAGNOSTIC and not printed_first_sample_diagnostic:
                print_first_sample_entry_diagnostic(
                    batch=batch,
                    val_dataset=val_dataset,
                    normalization_stats=normalization_stats,
                    device=device,
                )
                printed_first_sample_diagnostic = True

            rollout = rollout_batch(
                model=model,
                batch=batch,
                val_dataset=val_dataset,
                normalization_stats=normalization_stats,
            )
            update_metric_accumulators(accumulators, rollout)

            if sample_animation_payload is None:
                sample_animation_payload = extract_sample_animation_payload(rollout, batch)

            if batch_index % 25 == 0 or batch_index == len(val_loader):
                print(f"Evaluated {batch_index}/{len(val_loader)} validation batches")

    rows = build_metric_rows(accumulators)
    save_metrics(rows)
    print_summary(rows)

    if sample_animation_payload is None:
        print("Skipping animation because future ground-truth positions are not available from the current dataset object.")
    else:
        make_animation(sample_animation_payload, OUTPUT_ANIMATION)


def load_model_and_dataset():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model_config = checkpoint["model_config"]
    normalization_stats = checkpoint["normalization_stats"]

    _, val_dataset, _, _ = create_train_val_test_datasets(
        npz_path=NPZ_PATH,
        stride=STRIDE,
        T_history=model_config["T_history"],
        T_future=model_config["T_future"],
        max_droplets=model_config["max_droplets"],
        target_features=tuple(str(name) for name in normalization_stats["target_features"]),
    )
    val_dataset.normalization_stats = normalization_stats

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    model = CanonicalWindowTransformer(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Device: {device}")
    print(f"Validation windows: {len(val_dataset)}")
    return device, model, val_dataset, val_loader, normalization_stats


def move_batch_to_device(batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def denormalize_features(features, normalization_stats, device):
    mean = torch.as_tensor(normalization_stats["input_mean"], dtype=torch.float32, device=device)
    std = torch.as_tensor(normalization_stats["input_std"], dtype=torch.float32, device=device)
    return features * mean.new_tensor(std).view(1, 1, 1, -1) + mean.view(1, 1, 1, -1)


def normalize_features(features, normalization_stats, device):
    mean = torch.as_tensor(normalization_stats["input_mean"], dtype=torch.float32, device=device)
    std = torch.as_tensor(normalization_stats["input_std"], dtype=torch.float32, device=device)
    return (features - mean.view(1, 1, -1)) / std.view(1, 1, -1)


def denormalize_targets(targets, normalization_stats, device):
    mean = torch.as_tensor(normalization_stats["target_mean"], dtype=torch.float32, device=device)
    std = torch.as_tensor(normalization_stats["target_std"], dtype=torch.float32, device=device)
    return targets * std.view(1, 1, 1, -1) + mean.view(1, 1, 1, -1)


def rollout_batch(model, batch, val_dataset, normalization_stats):
    device = batch["history_x"].device
    rollout_history = batch["history_x"].clone()
    history_mask = batch["history_mask"].clone()

    pred_velocities = []
    true_velocities = []
    pred_positions = []
    true_positions = []
    step_masks = []

    feature_index = val_dataset.feature_indices
    true_future_features = get_true_future_features(batch, val_dataset, device)
    true_future_xy = true_future_features[:, :, :, [
        feature_index["x"],
        feature_index["y"],
    ]]

    for step_index in range(ROLLOUT_STEPS):
        previous_last_mask = history_mask[:, -1, :]
        pred = model(rollout_history, history_mask)
        pred_step_norm = pred[:, 0, :, :]
        pred_step_phys = denormalize_targets(
            pred_step_norm[:, None, :, :],
            normalization_stats,
            device,
        )[:, 0, :, :]

        true_step_norm = batch["future_y"][:, step_index, :, :]
        true_step_phys = denormalize_targets(
            true_step_norm[:, None, :, :],
            normalization_stats,
            device,
        )[:, 0, :, :]

        history_phys = denormalize_features(rollout_history, normalization_stats, device)
        last_frame = history_phys[:, -1, :, :]
        x_new = last_frame[:, :, feature_index["x"]] + pred_step_phys[:, :, 0]
        y_new = last_frame[:, :, feature_index["y"]] + pred_step_phys[:, :, 1]

        new_frame_phys = last_frame.clone()
        new_frame_phys[:, :, feature_index["x"]] = x_new
        new_frame_phys[:, :, feature_index["y"]] = y_new
        new_frame_phys[:, :, feature_index["vx"]] = pred_step_phys[:, :, 0]
        new_frame_phys[:, :, feature_index["vy"]] = pred_step_phys[:, :, 1]

        new_mask = batch["future_mask"][:, step_index, :]
        entering_mask = new_mask & ~previous_last_mask
        true_step_features = true_future_features[:, step_index, :, :]
        true_step_features_finite = torch.isfinite(true_step_features).all(dim=-1)
        boundary_mask = entering_mask & true_step_features_finite
        new_frame_phys[boundary_mask] = true_step_features[boundary_mask]
        pred_step_phys = pred_step_phys.clone()
        pred_step_phys[boundary_mask] = true_step_phys[boundary_mask]
        x_new = new_frame_phys[:, :, feature_index["x"]]
        y_new = new_frame_phys[:, :, feature_index["y"]]

        new_frame_norm = normalize_features(new_frame_phys, normalization_stats, device)

        rollout_history = torch.cat(
            [rollout_history[:, 1:, :, :], new_frame_norm[:, None, :, :]],
            dim=1,
        )
        history_mask = torch.cat(
            [history_mask[:, 1:, :], new_mask[:, None, :]],
            dim=1,
        )

        pred_velocities.append(pred_step_phys)
        true_velocities.append(true_step_phys)
        pred_positions.append(torch.stack([x_new, y_new], dim=-1))
        true_positions.append(true_future_xy[:, step_index, :, :])
        step_masks.append(batch["future_mask"][:, step_index, :])

    return {
        "pred_velocity": torch.stack(pred_velocities, dim=1),
        "true_velocity": torch.stack(true_velocities, dim=1),
        "pred_position": torch.stack(pred_positions, dim=1),
        "true_position": torch.stack(true_positions, dim=1),
        "mask": torch.stack(step_masks, dim=1),
    }


def get_true_future_features(batch, val_dataset, device):
    droplet_ids = batch["droplet_ids"].detach().cpu().numpy()
    frame_starts = batch["frame_start"].detach().cpu().numpy()
    track_id_to_index = {int(track_id): index for index, track_id in enumerate(val_dataset.track_ids)}

    B, M = droplet_ids.shape
    true_features = np.full((B, ROLLOUT_STEPS, M, len(val_dataset.feature_names)), np.nan, dtype=np.float32)

    for batch_index in range(B):
        for slot_index in range(M):
            track_id = int(droplet_ids[batch_index, slot_index])
            if track_id < 0:
                continue
            droplet_index = track_id_to_index.get(track_id)
            if droplet_index is None:
                continue

            start = int(frame_starts[batch_index]) + val_dataset.T_history
            end = start + ROLLOUT_STEPS
            true_features[batch_index, :, slot_index, :] = val_dataset.Z[droplet_index, start:end, :]

    return torch.as_tensor(true_features, dtype=torch.float32, device=device)


def print_first_sample_entry_diagnostic(batch, val_dataset, normalization_stats, device):
    sample_index = 0
    droplet_ids = batch["droplet_ids"][sample_index].detach().cpu().numpy()
    initial_last_mask = batch["history_mask"][sample_index, -1, :].detach().cpu().numpy().astype(bool)
    future_mask = batch["future_mask"][sample_index].detach().cpu().numpy().astype(bool)

    history_phys = denormalize_features(batch["history_x"], normalization_stats, device)
    last_history_xy = history_phys[sample_index, -1, :, :][:, [
        val_dataset.feature_indices["x"],
        val_dataset.feature_indices["y"],
    ]].detach().cpu().numpy()
    true_future_features = get_true_future_features(batch, val_dataset, device)
    true_future_xy = true_future_features[sample_index, :, :, [
        val_dataset.feature_indices["x"],
        val_dataset.feature_indices["y"],
    ]].detach().cpu().numpy()

    print("----------------------------------------------------")
    print("First Animation Sample Entry Diagnostic")
    print()
    print(
        "slot | droplet_id | history_last_valid | future_valid | "
        "first_future_step | initial_last_xy | true_future_xy_at_first_valid"
    )

    new_entry_slots = []
    for slot_index, droplet_id in enumerate(droplet_ids):
        if int(droplet_id) < 0:
            continue

        future_valid_steps = np.flatnonzero(future_mask[:, slot_index])
        becomes_valid = future_valid_steps.size > 0
        first_future_step = int(future_valid_steps[0] + 1) if becomes_valid else None

        initial_xy_text = "None"
        if initial_last_mask[slot_index]:
            initial_xy = last_history_xy[slot_index]
            initial_xy_text = f"({initial_xy[0]:.3f}, {initial_xy[1]:.3f})"

        future_xy_text = "None"
        if becomes_valid:
            future_xy = true_future_xy[future_valid_steps[0], slot_index]
            future_xy_text = f"({future_xy[0]:.3f}, {future_xy[1]:.3f})"

        print(
            f"{slot_index:>4d} | "
            f"{int(droplet_id):>10d} | "
            f"{str(bool(initial_last_mask[slot_index])):>18s} | "
            f"{str(bool(becomes_valid)):>12s} | "
            f"{str(first_future_step):>17s} | "
            f"{initial_xy_text:>20s} | "
            f"{future_xy_text}"
        )

        if not initial_last_mask[slot_index] and becomes_valid:
            new_entry_slots.append((slot_index, int(droplet_id), first_future_step, future_xy_text))

    print()
    print("New-entry slots where history_mask[0, -1, slot] is False and future_mask becomes True:")
    if not new_entry_slots:
        print("  None")
    else:
        for slot_index, droplet_id, first_future_step, future_xy_text in new_entry_slots:
            print(
                f"  slot={slot_index} droplet_id={droplet_id} "
                f"first_future_step={first_future_step} true_future_xy={future_xy_text}"
            )
    print("----------------------------------------------------")


def create_accumulators(num_steps, include_position):
    return {
        "overall": new_accumulator(include_position),
        "steps": [new_accumulator(include_position) for _ in range(num_steps)],
    }


def new_accumulator(include_position):
    accumulator = {
        "count": 0,
        "sum_sq_vx": 0.0,
        "sum_sq_vy": 0.0,
        "sum_sq_speed": 0.0,
        "sum_abs_vx": 0.0,
        "sum_abs_vy": 0.0,
        "sum_abs_speed": 0.0,
    }
    if include_position:
        accumulator.update(
            {
                "position_count": 0,
                "sum_sq_x": 0.0,
                "sum_sq_y": 0.0,
                "sum_sq_position": 0.0,
                "sum_abs_x": 0.0,
                "sum_abs_y": 0.0,
                "sum_abs_position": 0.0,
            }
        )
    return accumulator


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

    for step_index in range(ROLLOUT_STEPS):
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
    if "position_count" not in accumulator or valid_position.sum().item() == 0:
        return

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


def build_metric_rows(accumulators):
    rows = []
    for step_index, accumulator in enumerate(accumulators["steps"], start=1):
        row = metrics_from_accumulator(accumulator)
        row["horizon"] = f"Step {step_index}"
        rows.append(row)

    overall = metrics_from_accumulator(accumulators["overall"])
    overall["horizon"] = "Overall"
    rows.append(overall)
    return rows


def metrics_from_accumulator(accumulator):
    count = accumulator["count"]
    row = {
        "valid_samples": count,
        "rmse_vx": safe_rmse(accumulator["sum_sq_vx"], count),
        "rmse_vy": safe_rmse(accumulator["sum_sq_vy"], count),
        "rmse_speed": safe_rmse(accumulator["sum_sq_speed"], count),
        "mae_vx": safe_mae(accumulator["sum_abs_vx"], count),
        "mae_vy": safe_mae(accumulator["sum_abs_vy"], count),
        "mae_speed": safe_mae(accumulator["sum_abs_speed"], count),
    }

    position_count = accumulator.get("position_count", 0)
    row.update(
        {
            "valid_position_samples": position_count,
            "rmse_x": safe_rmse(accumulator.get("sum_sq_x", 0.0), position_count),
            "rmse_y": safe_rmse(accumulator.get("sum_sq_y", 0.0), position_count),
            "rmse_position": safe_rmse(accumulator.get("sum_sq_position", 0.0), position_count),
            "mae_x": safe_mae(accumulator.get("sum_abs_x", 0.0), position_count),
            "mae_y": safe_mae(accumulator.get("sum_abs_y", 0.0), position_count),
            "mae_position": safe_mae(accumulator.get("sum_abs_position", 0.0), position_count),
        }
    )
    return row


def safe_rmse(sum_sq, count):
    return np.sqrt(sum_sq / count) if count else np.nan


def safe_mae(sum_abs, count):
    return sum_abs / count if count else np.nan


def save_metrics(rows):
    output_csv = Path(OUTPUT_CSV)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"Saved metrics CSV: {output_csv}")


def print_summary(rows):
    overall = rows[-1]
    print("----------------------------------------------------")
    print("Teacher-Forced Model: Autoregressive Rollout Metrics")
    print()
    print(f"Overall valid velocity samples : {overall['valid_samples']}")
    print(f"Overall valid position samples : {overall['valid_position_samples']}")
    print(f"RMSE vx        : {overall['rmse_vx']:.6f}")
    print(f"RMSE vy        : {overall['rmse_vy']:.6f}")
    print(f"RMSE speed     : {overall['rmse_speed']:.6f}")
    print(f"MAE vx         : {overall['mae_vx']:.6f}")
    print(f"MAE vy         : {overall['mae_vy']:.6f}")
    print(f"MAE speed      : {overall['mae_speed']:.6f}")
    print(f"RMSE x         : {overall['rmse_x']:.6f}")
    print(f"RMSE y         : {overall['rmse_y']:.6f}")
    print(f"RMSE position  : {overall['rmse_position']:.6f}")
    print(f"MAE x          : {overall['mae_x']:.6f}")
    print(f"MAE y          : {overall['mae_y']:.6f}")
    print(f"MAE position   : {overall['mae_position']:.6f}")
    print()
    print("Per-step Metrics")
    print()

    for row in rows[:-1]:
        print(
            f"{row['horizon']:<7} "
            f"valid_v={row['valid_samples']:<8d} "
            f"valid_xy={row['valid_position_samples']:<8d} "
            f"RMSE(vx)={row['rmse_vx']:.6f} "
            f"RMSE(vy)={row['rmse_vy']:.6f} "
            f"RMSE(speed)={row['rmse_speed']:.6f} "
            f"RMSE(pos)={row['rmse_position']:.6f}"
        )

    print("----------------------------------------------------")


def extract_sample_animation_payload(rollout, batch):
    sample_index = 0
    return {
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

    valid_positions = np.concatenate(
        [
            pred_position[mask],
            true_position[mask],
        ],
        axis=0,
    )
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
        ax.set_title(f"Teacher-forced rollout diagnostic - step {step_index + 1}")

        for point, track_id in zip(pred, ids):
            id_artists.append(ax.text(point[0] + 2, point[1] + 2, str(track_id), fontsize=7, color="red"))

        return [gt_scatter, pred_scatter, *id_artists]

    plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    animation = FuncAnimation(fig, draw, frames=np.arange(ROLLOUT_STEPS), interval=300, blit=False)

    output_path = Path(output_video)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=3)
    animation.save(output_path, writer=writer)
    plt.close(fig)
    print(f"Saved rollout animation: {output_path}")


if __name__ == "__main__":
    main()
