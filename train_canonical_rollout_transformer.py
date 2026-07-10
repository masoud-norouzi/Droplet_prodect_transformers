from __future__ import annotations

import argparse
import csv
from pathlib import Path
import random

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, default_collate

from models.canonical_rollout_transformer import CanonicalRolloutTransformer
from utils.canonical_dataset.canonical_window_dataset import create_train_val_test_datasets


NPZ_PATH = "outputs/processed/2/canonical_dataset.npz"
OUTPUT_DIR = "outputs/models/train_canonical_rollout_transformer"
BEST_CHECKPOINT_PATH = f"{OUTPUT_DIR}/canonical_rollout_transformer_best.pt"
LATEST_CHECKPOINT_PATH = f"{OUTPUT_DIR}/canonical_rollout_transformer_latest.pt"
CURVES_CSV_PATH = f"{OUTPUT_DIR}/canonical_rollout_transformer_training_curves.csv"
ANIMATION_DIR = f"{OUTPUT_DIR}/rollout_training_animations"

T_HISTORY = 20
ROLLOUT_HORIZON = 50
MAX_DROPLETS = 64
INPUT_DIM = 5
TARGET_DIM = 2
STRIDE = 5
LOSS_ALPHA = 2.0

BATCH_SIZE = 4
EPOCHS = 50
LR = 1e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
NUM_WORKERS = 0
LOG_EVERY_N_BATCHES = 25
ANIMATE_EVERY_N_EPOCHS = 10
DIAGNOSTIC_STEPS = (1, 5, 10, 20, 30, 40, 50)

MODEL_CONFIG = {
    "input_dim": INPUT_DIM,
    "target_dim": TARGET_DIM,
    "T_history": T_HISTORY,
    "max_droplets": MAX_DROPLETS,
    "d_model": 128,
    "n_heads": 4,
    "num_layers": 4,
    "dim_feedforward": 512,
    "dropout": 0.1,
}


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds, val_ds, test_ds, normalization_stats = create_train_val_test_datasets(
        npz_path=args.npz_path,
        stride=args.stride,
        T_history=T_HISTORY,
        T_future=ROLLOUT_HORIZON,
        max_droplets=MAX_DROPLETS,
    )
    print(f"Train windows: {len(train_ds)}")
    print(f"Val windows: {len(val_ds)}")
    print(f"Test windows: {len(test_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    model = CanonicalRolloutTransformer(**MODEL_CONFIG).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    weights = rollout_weights(ROLLOUT_HORIZON, LOSS_ALPHA, device)

    run_shape_test(model, train_loader, train_ds, normalization_stats, weights, device)
    if args.shape_test_only:
        return

    best_val_loss = float("inf")
    best_checkpoint_path = Path(args.best_checkpoint)
    latest_checkpoint_path = Path(args.latest_checkpoint)
    curves_csv_path = Path(args.curves_csv)
    best_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    curves_csv_path.parent.mkdir(parents=True, exist_ok=True)
    initialize_curves_csv(curves_csv_path)

    for epoch in range(1, args.epochs + 1):
        train_summary = train_one_epoch(
            model=model,
            loader=train_loader,
            dataset=train_ds,
            optimizer=optimizer,
            normalization_stats=normalization_stats,
            weights=weights,
            device=device,
            log_every=args.log_every,
        )
        val_summary = evaluate(
            model=model,
            loader=val_loader,
            dataset=val_ds,
            normalization_stats=normalization_stats,
            weights=weights,
            device=device,
            log_every=args.log_every,
        )

        print_epoch_summary(epoch, train_summary, val_summary)
        append_curves_csv(curves_csv_path, epoch, train_summary, val_summary)

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": val_summary["weighted_loss_internal_only"],
            "normalization_stats": normalization_stats,
            "model_config": MODEL_CONFIG,
            "rollout_horizon": ROLLOUT_HORIZON,
            "loss_alpha": LOSS_ALPHA,
            "stride": args.stride,
        }
        torch.save(checkpoint, latest_checkpoint_path)

        if val_summary["weighted_loss_internal_only"] < best_val_loss:
            best_val_loss = val_summary["weighted_loss_internal_only"]
            torch.save(checkpoint, best_checkpoint_path)
            print(f"Saved best checkpoint: {best_checkpoint_path}")

        if args.animate_every > 0 and epoch % args.animate_every == 0:
            make_random_validation_animation(
                model=model,
                dataset=val_ds,
                normalization_stats=normalization_stats,
                device=device,
                output_path=Path(args.animation_dir) / f"rollout_epoch_{epoch:03d}.mp4",
            )

    make_random_validation_animation(
        model=model,
        dataset=val_ds,
        normalization_stats=normalization_stats,
        device=device,
        output_path=Path(args.animation_dir) / "rollout_after_training.mp4",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the boundary-conditioned rollout Transformer.")
    parser.add_argument("--npz-path", default=NPZ_PATH)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--stride", type=int, default=STRIDE)
    parser.add_argument("--log-every", type=int, default=LOG_EVERY_N_BATCHES)
    parser.add_argument("--animate-every", type=int, default=ANIMATE_EVERY_N_EPOCHS)
    parser.add_argument("--best-checkpoint", default=BEST_CHECKPOINT_PATH)
    parser.add_argument("--latest-checkpoint", default=LATEST_CHECKPOINT_PATH)
    parser.add_argument("--curves-csv", default=CURVES_CSV_PATH)
    parser.add_argument("--animation-dir", default=ANIMATION_DIR)
    parser.add_argument("--shape-test-only", action="store_true")
    return parser.parse_args()


def rollout_weights(horizon, alpha, device):
    if horizon == 1:
        return torch.ones(1, dtype=torch.float32, device=device)
    step_ids = torch.arange(horizon, dtype=torch.float32, device=device)
    return 1.0 + alpha * step_ids / float(horizon - 1)


def run_shape_test(model, train_loader, dataset, normalization_stats, weights, device) -> None:
    model.eval()
    batch = move_batch_to_device(next(iter(train_loader)), device)
    with torch.no_grad():
        rollout = boundary_conditioned_rollout(
            model=model,
            batch=batch,
            dataset=dataset,
            normalization_stats=normalization_stats,
            weights=weights,
        )

    print(f"history_x:       {tuple(batch['history_x'].shape)}")
    print(f"history_mask:    {tuple(batch['history_mask'].shape)}")
    print(f"future_y:        {tuple(batch['future_y'].shape)}")
    print(f"future_mask:     {tuple(batch['future_mask'].shape)}")
    print(f"pred_velocity:   {tuple(rollout['pred_velocity'].shape)}")
    print(f"pred_position:   {tuple(rollout['pred_position'].shape)}")
    print(f"weighted_loss_internal_only: {float(rollout['weighted_loss_internal_only']):.6f}")
    print(f"boundary_injected:           {int(rollout['boundary_mask'].sum().detach().cpu())}")
    print(f"valid_future_samples:        {int(rollout['mask'].sum().detach().cpu())}")

    assert rollout["pred_velocity"].shape == batch["future_y"].shape
    assert rollout["mask"].shape == batch["future_mask"].shape


def train_one_epoch(model, loader, dataset, optimizer, normalization_stats, weights, device, log_every):
    model.train()
    total_loss = 0.0
    total_boundary = 0
    total_valid = 0
    total_rollouts = 0
    num_batches = 0
    total_batches = len(loader)

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)
        rollout = boundary_conditioned_rollout(
            model=model,
            batch=batch,
            dataset=dataset,
            normalization_stats=normalization_stats,
            weights=weights,
        )
        loss = rollout["weighted_loss_internal_only"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        total_loss += float(loss.detach().cpu())
        total_boundary += int(rollout["boundary_mask"].sum().detach().cpu())
        total_valid += int(rollout["mask"].sum().detach().cpu())
        total_rollouts += int(batch["history_x"].shape[0])
        num_batches += 1
        if log_every > 0 and (num_batches % log_every == 0 or num_batches == total_batches):
            print_progress("train", num_batches, total_batches, total_loss / num_batches)

    return {
        "weighted_loss_internal_only": total_loss / max(num_batches, 1),
        "boundary_injected_per_rollout": total_boundary / max(total_rollouts, 1),
        "boundary_fraction": total_boundary / max(total_valid, 1),
    }


def evaluate(model, loader, dataset, normalization_stats, weights, device, log_every=0):
    model.eval()
    total_loss = 0.0
    total_boundary = 0
    total_valid = 0
    total_rollouts = 0
    num_batches = 0
    total_batches = len(loader)
    accumulators = create_accumulators(ROLLOUT_HORIZON)

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            rollout = boundary_conditioned_rollout(
                model=model,
                batch=batch,
                dataset=dataset,
                normalization_stats=normalization_stats,
                weights=weights,
            )
            total_loss += float(rollout["weighted_loss_internal_only"].detach().cpu())
            total_boundary += int(rollout["boundary_mask"].sum().detach().cpu())
            total_valid += int(rollout["mask"].sum().detach().cpu())
            total_rollouts += int(batch["history_x"].shape[0])
            num_batches += 1
            update_metric_accumulators(accumulators, rollout)
            if log_every > 0 and (num_batches % log_every == 0 or num_batches == total_batches):
                print_progress("val", num_batches, total_batches, total_loss / num_batches)

    summary = metrics_from_accumulator(accumulators["overall"])
    summary["weighted_loss_internal_only"] = total_loss / max(num_batches, 1)
    summary["boundary_injected_per_rollout"] = total_boundary / max(total_rollouts, 1)
    summary["boundary_fraction"] = total_boundary / max(total_valid, 1)
    summary["boundary_injected_total"] = total_boundary
    summary["valid_future_samples_total"] = total_valid
    summary["step_rmse_position"] = [
        metrics_from_accumulator(accumulator)["rmse_position"]
        for accumulator in accumulators["steps"]
    ]
    return summary


def boundary_conditioned_rollout(model, batch, dataset, normalization_stats, weights):
    device = batch["history_x"].device
    rollout_history = batch["history_x"].clone()
    history_mask = batch["history_mask"].clone()

    pred_velocities_norm = []
    true_velocities_norm = []
    pred_velocities_phys = []
    true_velocities_phys = []
    pred_positions = []
    true_positions = []
    step_masks = []
    boundary_masks = []
    internal_loss_masks = []
    step_losses = []

    feature_index = dataset.feature_indices
    true_future_features = get_true_future_features(batch, dataset, device, weights.numel())
    true_future_xy = true_future_features[:, :, :, [
        feature_index["x"],
        feature_index["y"],
    ]]

    for step_index in range(weights.numel()):
        previous_last_mask = history_mask[:, -1, :]
        pred_step_norm_raw = model(rollout_history, history_mask)
        pred_step_phys_raw = denormalize_targets(
            pred_step_norm_raw[:, None, :, :],
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
        x_next = last_frame[:, :, feature_index["x"]] + pred_step_phys_raw[:, :, 0]
        y_next = last_frame[:, :, feature_index["y"]] + pred_step_phys_raw[:, :, 1]

        new_frame_phys = last_frame.clone()
        new_frame_phys[:, :, feature_index["x"]] = x_next
        new_frame_phys[:, :, feature_index["y"]] = y_next
        new_frame_phys[:, :, feature_index["vx"]] = pred_step_phys_raw[:, :, 0]
        new_frame_phys[:, :, feature_index["vy"]] = pred_step_phys_raw[:, :, 1]

        new_mask = batch["future_mask"][:, step_index, :]
        entering_mask = new_mask & ~previous_last_mask
        true_step_features = true_future_features[:, step_index, :, :]
        true_step_features_finite = torch.isfinite(true_step_features).all(dim=-1)
        boundary_mask = entering_mask & true_step_features_finite
        new_frame_phys[boundary_mask] = true_step_features[boundary_mask]

        circularity_index = feature_index.get("circularity")
        if circularity_index is not None:
            true_circularity = true_step_features[:, :, circularity_index]
            circularity_mask = new_mask & torch.isfinite(true_circularity)
            new_frame_phys[:, :, circularity_index] = torch.where(
                circularity_mask,
                true_circularity,
                new_frame_phys[:, :, circularity_index],
            )

        pred_step_norm = pred_step_norm_raw.clone()
        pred_step_phys = pred_step_phys_raw.clone()
        pred_step_norm[boundary_mask] = true_step_norm[boundary_mask]
        pred_step_phys[boundary_mask] = true_step_phys[boundary_mask]

        loss_mask = new_mask & ~boundary_mask
        step_loss = masked_velocity_mse(pred_step_norm, true_step_norm, loss_mask)
        step_losses.append(step_loss)

        new_frame_norm = normalize_features(new_frame_phys, normalization_stats, device)
        new_frame_norm = torch.where(
            new_mask[:, :, None],
            new_frame_norm,
            torch.zeros_like(new_frame_norm),
        )
        rollout_history = torch.cat(
            [rollout_history[:, 1:, :, :], new_frame_norm[:, None, :, :]],
            dim=1,
        )
        history_mask = torch.cat(
            [history_mask[:, 1:, :], new_mask[:, None, :]],
            dim=1,
        )

        pred_velocities_norm.append(pred_step_norm)
        true_velocities_norm.append(true_step_norm)
        pred_velocities_phys.append(pred_step_phys)
        true_velocities_phys.append(true_step_phys)
        pred_positions.append(new_frame_phys[:, :, [feature_index["x"], feature_index["y"]]])
        true_positions.append(true_future_xy[:, step_index, :, :])
        step_masks.append(new_mask)
        boundary_masks.append(boundary_mask)
        internal_loss_masks.append(loss_mask)

    step_loss_tensor = torch.stack(step_losses)
    weighted_loss_internal_only = (step_loss_tensor * weights).sum() / weights.sum()

    return {
        "weighted_loss": weighted_loss_internal_only,
        "weighted_loss_internal_only": weighted_loss_internal_only,
        "step_losses": step_loss_tensor,
        "pred_velocity_norm": torch.stack(pred_velocities_norm, dim=1),
        "true_velocity_norm": torch.stack(true_velocities_norm, dim=1),
        "pred_velocity": torch.stack(pred_velocities_phys, dim=1),
        "true_velocity": torch.stack(true_velocities_phys, dim=1),
        "pred_position": torch.stack(pred_positions, dim=1),
        "true_position": torch.stack(true_positions, dim=1),
        "mask": torch.stack(step_masks, dim=1),
        "boundary_mask": torch.stack(boundary_masks, dim=1),
        "internal_loss_mask": torch.stack(internal_loss_masks, dim=1),
    }


def masked_velocity_mse(prediction, target, mask):
    expanded_mask = mask.unsqueeze(-1).expand_as(target)
    squared_error = (prediction - target) ** 2
    valid_error = squared_error[expanded_mask]
    if valid_error.numel() == 0:
        return squared_error.sum() * 0.0
    return valid_error.mean()


def get_true_future_features(batch, dataset, device, horizon):
    droplet_ids = batch["droplet_ids"].detach().cpu().numpy()
    frame_starts = batch["frame_start"].detach().cpu().numpy()
    track_id_to_index = {int(track_id): index for index, track_id in enumerate(dataset.track_ids)}

    B, M = droplet_ids.shape
    true_features = np.full((B, horizon, M, len(dataset.feature_names)), np.nan, dtype=np.float32)

    for batch_index in range(B):
        start = int(frame_starts[batch_index]) + dataset.T_history
        end = start + horizon
        for slot_index in range(M):
            track_id = int(droplet_ids[batch_index, slot_index])
            if track_id < 0:
                continue
            droplet_index = track_id_to_index.get(track_id)
            if droplet_index is None:
                continue
            true_features[batch_index, :, slot_index, :] = dataset.Z[droplet_index, start:end, :]

    return torch.as_tensor(true_features, dtype=torch.float32, device=device)


def move_batch_to_device(batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def denormalize_features(features, normalization_stats, device):
    mean = torch.as_tensor(normalization_stats["input_mean"], dtype=torch.float32, device=device)
    std = torch.as_tensor(normalization_stats["input_std"], dtype=torch.float32, device=device)
    return features * std.view(1, 1, 1, -1) + mean.view(1, 1, 1, -1)


def normalize_features(features, normalization_stats, device):
    mean = torch.as_tensor(normalization_stats["input_mean"], dtype=torch.float32, device=device)
    std = torch.as_tensor(normalization_stats["input_std"], dtype=torch.float32, device=device)
    return (features - mean.view(1, 1, -1)) / std.view(1, 1, -1)


def denormalize_targets(targets, normalization_stats, device):
    mean = torch.as_tensor(normalization_stats["target_mean"], dtype=torch.float32, device=device)
    std = torch.as_tensor(normalization_stats["target_std"], dtype=torch.float32, device=device)
    return targets * std.view(1, 1, 1, -1) + mean.view(1, 1, 1, -1)


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
        "position_count": 0,
        "sum_sq_position": 0.0,
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
        position_error_norm,
        rollout["mask"] & position_finite,
    )
    for step_index in range(rollout["mask"].shape[1]):
        update_one_accumulator(
            accumulators["steps"][step_index],
            velocity_error[:, step_index, :, :],
            speed_error[:, step_index, :],
            rollout["mask"][:, step_index, :],
            position_error_norm[:, step_index, :],
            rollout["mask"][:, step_index, :] & position_finite[:, step_index, :],
        )


def update_one_accumulator(accumulator, velocity_error, speed_error, velocity_mask, position_error_norm, position_mask):
    valid = velocity_mask.bool()
    if valid.sum().item() > 0:
        vx_error = velocity_error[..., 0][valid]
        vy_error = velocity_error[..., 1][valid]
        speed = speed_error[valid]
        accumulator["count"] += int(valid.sum().item())
        accumulator["sum_sq_vx"] += float((vx_error**2).sum().detach().cpu())
        accumulator["sum_sq_vy"] += float((vy_error**2).sum().detach().cpu())
        accumulator["sum_sq_speed"] += float((speed**2).sum().detach().cpu())

    valid_position = position_mask.bool()
    if valid_position.sum().item() > 0:
        position = position_error_norm[valid_position]
        accumulator["position_count"] += int(valid_position.sum().item())
        accumulator["sum_sq_position"] += float((position**2).sum().detach().cpu())


def metrics_from_accumulator(accumulator):
    count = accumulator["count"]
    position_count = accumulator["position_count"]
    return {
        "valid_samples": count,
        "valid_position_samples": position_count,
        "rmse_vx": safe_rmse(accumulator["sum_sq_vx"], count),
        "rmse_vy": safe_rmse(accumulator["sum_sq_vy"], count),
        "rmse_speed": safe_rmse(accumulator["sum_sq_speed"], count),
        "rmse_position": safe_rmse(accumulator["sum_sq_position"], position_count),
    }


def safe_rmse(sum_sq, count):
    return np.sqrt(sum_sq / count) if count else np.nan


def print_progress(label, num_batches, total_batches, running_loss):
    percent = 100.0 * num_batches / max(total_batches, 1)
    print(
        f"  {label:<5} batch {num_batches:04d}/{total_batches:04d} "
        f"({percent:5.1f}%) weighted_loss={running_loss:.6f}"
    )


def print_epoch_summary(epoch, train_summary, val_summary):
    step_text = " ".join(
        f"s{step}={val_summary['step_rmse_position'][step - 1]:.6f}"
        for step in DIAGNOSTIC_STEPS
    )
    print(
        f"epoch {epoch:03d} "
        f"train_weighted_loss_internal_only={train_summary['weighted_loss_internal_only']:.6f} "
        f"val_weighted_loss_internal_only={val_summary['weighted_loss_internal_only']:.6f} "
        f"val_rmse_vx={val_summary['rmse_vx']:.6f} "
        f"val_rmse_vy={val_summary['rmse_vy']:.6f} "
        f"val_rmse_speed={val_summary['rmse_speed']:.6f} "
        f"val_rmse_position={val_summary['rmse_position']:.6f}"
    )
    print(
        "  boundary_diagnostics "
        f"train_boundary_injected_per_rollout={train_summary['boundary_injected_per_rollout']:.3f} "
        f"train_boundary_fraction={train_summary['boundary_fraction']:.6f} "
        f"val_boundary_injected_per_rollout={val_summary['boundary_injected_per_rollout']:.3f} "
        f"val_boundary_fraction={val_summary['boundary_fraction']:.6f}"
    )
    print(f"  stepwise_val_rmse_position {step_text}")


def initialize_curves_csv(path):
    if path.exists():
        return
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "epoch",
                "train_weighted_loss_internal_only",
                "val_weighted_loss_internal_only",
                "train_boundary_injected_per_rollout",
                "train_boundary_fraction",
                "val_boundary_injected_per_rollout",
                "val_boundary_fraction",
                "val_rmse_vx",
                "val_rmse_vy",
                "val_rmse_speed",
                "val_rmse_position",
                *[f"val_rmse_position_step_{step}" for step in DIAGNOSTIC_STEPS],
            ]
        )


def append_curves_csv(path, epoch, train_summary, val_summary):
    with path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                epoch,
                train_summary["weighted_loss_internal_only"],
                val_summary["weighted_loss_internal_only"],
                train_summary["boundary_injected_per_rollout"],
                train_summary["boundary_fraction"],
                val_summary["boundary_injected_per_rollout"],
                val_summary["boundary_fraction"],
                val_summary["rmse_vx"],
                val_summary["rmse_vy"],
                val_summary["rmse_speed"],
                val_summary["rmse_position"],
                *[val_summary["step_rmse_position"][step - 1] for step in DIAGNOSTIC_STEPS],
            ]
        )


def make_random_validation_animation(model, dataset, normalization_stats, device, output_path):
    if len(dataset) == 0:
        print("Skipping animation because the validation dataset is empty.")
        return
    try:
        import imageio_ffmpeg
    except ImportError:
        print("Skipping animation because imageio-ffmpeg is not installed.")
        return

    model.eval()
    sample_index = random.randrange(len(dataset))
    batch = default_collate([dataset[sample_index]])
    batch = move_batch_to_device(batch, device)
    weights = rollout_weights(ROLLOUT_HORIZON, LOSS_ALPHA, device)
    with torch.no_grad():
        rollout = boundary_conditioned_rollout(model, batch, dataset, normalization_stats, weights)

    payload = {
        "pred_position": rollout["pred_position"][0].detach().cpu().numpy(),
        "true_position": rollout["true_position"][0].detach().cpu().numpy(),
        "mask": rollout["mask"][0].detach().cpu().numpy(),
        "droplet_ids": batch["droplet_ids"][0].detach().cpu().numpy(),
    }
    save_rollout_animation(payload, output_path, imageio_ffmpeg.get_ffmpeg_exe())


def save_rollout_animation(payload, output_path, ffmpeg_path):
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

    def draw(step_index):
        valid = mask[step_index]
        gt = true_position[step_index, valid]
        pred = pred_position[step_index, valid]
        gt_scatter.set_offsets(gt if gt.size else np.empty((0, 2)))
        pred_scatter.set_offsets(pred if pred.size else np.empty((0, 2)))
        ax.set_title(f"Boundary-conditioned rollout - step {step_index + 1}")
        return [gt_scatter, pred_scatter]

    plt.rcParams["animation.ffmpeg_path"] = ffmpeg_path
    animation = FuncAnimation(fig, draw, frames=np.arange(ROLLOUT_HORIZON), interval=200, blit=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=5)
    animation.save(output_path, writer=writer)
    plt.close(fig)
    print(f"Saved rollout animation: {output_path}")


if __name__ == "__main__":
    main()
