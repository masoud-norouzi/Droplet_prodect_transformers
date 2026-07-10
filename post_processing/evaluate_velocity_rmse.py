from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from models.canonical_window_transformer import CanonicalWindowTransformer
from utils.canonical_dataset.canonical_window_dataset import (
    create_train_val_test_datasets,
    masked_future_velocity_mse_loss,
)


CHECKPOINT_PATH = "outputs/models/train_canonical_window_transformer/canonical_window_transformer_best.pt"
NPZ_PATH = "outputs/processed/2/canonical_dataset.npz"
OUTPUT_CSV = "outputs/post_processing/velocity_rmse.csv"

STRIDE = 5
BATCH_SIZE = 8
NUM_WORKERS = 0


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model_config = checkpoint["model_config"]
    normalization_stats = checkpoint["normalization_stats"]

    _, val_ds, _, _ = create_train_val_test_datasets(
        npz_path=NPZ_PATH,
        stride=STRIDE,
        T_history=model_config["T_history"],
        T_future=model_config["T_future"],
        max_droplets=model_config["max_droplets"],
        target_features=tuple(str(name) for name in normalization_stats["target_features"]),
    )
    val_ds.normalization_stats = normalization_stats

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    model = CanonicalWindowTransformer(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    target_mean = torch.as_tensor(
        normalization_stats["target_mean"],
        dtype=torch.float32,
        device=device,
    )
    target_std = torch.as_tensor(
        normalization_stats["target_std"],
        dtype=torch.float32,
        device=device,
    )

    accumulators = create_accumulators(model_config["T_future"])
    normalized_loss_sum = 0.0
    normalized_loss_batches = 0

    with torch.no_grad():
        for batch_index, batch in enumerate(val_loader, start=1):
            history_x = batch["history_x"].to(device)
            history_mask = batch["history_mask"].to(device)
            future_y = batch["future_y"].to(device)
            future_mask = batch["future_mask"].to(device)

            pred_v = model(history_x, history_mask)
            normalized_loss = masked_future_velocity_mse_loss(
                pred_v,
                future_y,
                future_mask,
            )
            normalized_loss_sum += float(normalized_loss.detach().cpu())
            normalized_loss_batches += 1

            pred_physical = denormalize_velocity(pred_v, target_mean, target_std)
            target_physical = denormalize_velocity(future_y, target_mean, target_std)
            update_accumulators(
                accumulators,
                pred_physical,
                target_physical,
                future_mask,
            )

            if batch_index % 25 == 0 or batch_index == len(val_loader):
                print(f"Evaluated {batch_index}/{len(val_loader)} validation batches")

    normalized_mse = normalized_loss_sum / max(normalized_loss_batches, 1)
    rows = build_metric_rows(accumulators, normalized_mse)
    save_metrics(rows)
    print_summary(rows, normalized_mse)


def denormalize_velocity(values, target_mean, target_std):
    return values * target_std.view(1, 1, 1, -1) + target_mean.view(1, 1, 1, -1)


def create_accumulators(T_future):
    return {
        "overall": new_accumulator(),
        "steps": [new_accumulator() for _ in range(T_future)],
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
    }


def update_accumulators(accumulators, pred, target, mask):
    error = pred - target
    speed_error = torch.sqrt(error[..., 0] ** 2 + error[..., 1] ** 2)

    update_one_accumulator(accumulators["overall"], error, speed_error, mask)

    for step_index in range(pred.shape[1]):
        update_one_accumulator(
            accumulators["steps"][step_index],
            error[:, step_index, :, :],
            speed_error[:, step_index, :],
            mask[:, step_index, :],
        )


def update_one_accumulator(accumulator, error, speed_error, mask):
    valid = mask.bool()
    if valid.sum().item() == 0:
        return

    vx_error = error[..., 0][valid]
    vy_error = error[..., 1][valid]
    speed = speed_error[valid]

    accumulator["count"] += int(valid.sum().item())
    accumulator["sum_sq_vx"] += float((vx_error**2).sum().detach().cpu())
    accumulator["sum_sq_vy"] += float((vy_error**2).sum().detach().cpu())
    accumulator["sum_sq_speed"] += float((speed**2).sum().detach().cpu())
    accumulator["sum_abs_vx"] += float(vx_error.abs().sum().detach().cpu())
    accumulator["sum_abs_vy"] += float(vy_error.abs().sum().detach().cpu())
    accumulator["sum_abs_speed"] += float(speed.abs().sum().detach().cpu())


def build_metric_rows(accumulators, normalized_mse):
    rows = []
    for step_index, accumulator in enumerate(accumulators["steps"], start=1):
        row = metrics_from_accumulator(accumulator)
        row["horizon"] = f"Step {step_index}"
        row["normalized_mse"] = np.nan
        rows.append(row)

    overall = metrics_from_accumulator(accumulators["overall"])
    overall["horizon"] = "Overall"
    overall["normalized_mse"] = normalized_mse
    rows.append(overall)
    return rows


def metrics_from_accumulator(accumulator):
    count = accumulator["count"]
    if count == 0:
        return {
            "valid_samples": 0,
            "rmse_vx": np.nan,
            "rmse_vy": np.nan,
            "rmse_speed": np.nan,
            "mae_vx": np.nan,
            "mae_vy": np.nan,
            "mae_speed": np.nan,
        }

    return {
        "valid_samples": count,
        "rmse_vx": np.sqrt(accumulator["sum_sq_vx"] / count),
        "rmse_vy": np.sqrt(accumulator["sum_sq_vy"] / count),
        "rmse_speed": np.sqrt(accumulator["sum_sq_speed"] / count),
        "mae_vx": accumulator["sum_abs_vx"] / count,
        "mae_vy": accumulator["sum_abs_vy"] / count,
        "mae_speed": accumulator["sum_abs_speed"] / count,
    }


def save_metrics(rows):
    output_csv = Path(OUTPUT_CSV)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"Saved metrics CSV: {output_csv}")


def print_summary(rows, normalized_mse):
    overall = rows[-1]

    print("----------------------------------------------------")
    print("Overall Validation Metrics")
    print()
    print(f"Valid samples  : {overall['valid_samples']}")
    print(f"Normalized MSE : {normalized_mse:.6f}")
    print(f"RMSE vx        : {overall['rmse_vx']:.6f}")
    print(f"RMSE vy        : {overall['rmse_vy']:.6f}")
    print(f"RMSE speed     : {overall['rmse_speed']:.6f}")
    print(f"MAE vx         : {overall['mae_vx']:.6f}")
    print(f"MAE vy         : {overall['mae_vy']:.6f}")
    print(f"MAE speed      : {overall['mae_speed']:.6f}")
    print()
    print("Per-step Metrics")
    print()

    for row in rows[:-1]:
        print(
            f"{row['horizon']:<7} "
            f"valid={row['valid_samples']:<8d} "
            f"RMSE(vx)={row['rmse_vx']:.6f} "
            f"RMSE(vy)={row['rmse_vy']:.6f} "
            f"RMSE(speed)={row['rmse_speed']:.6f} "
            f"MAE(vx)={row['mae_vx']:.6f} "
            f"MAE(vy)={row['mae_vy']:.6f} "
            f"MAE(speed)={row['mae_speed']:.6f}"
        )

    print("----------------------------------------------------")


if __name__ == "__main__":
    main()
