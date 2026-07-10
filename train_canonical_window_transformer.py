from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from models.canonical_window_transformer import CanonicalWindowTransformer
from utils.canonical_dataset.canonical_window_dataset import (
    create_train_val_test_datasets,
    masked_future_velocity_mse_loss,
)


NPZ_PATH = "outputs/processed/2/canonical_dataset.npz"
OUTPUT_DIR = "outputs/models/train_canonical_window_transformer"
CHECKPOINT_PATH = f"{OUTPUT_DIR}/canonical_window_transformer_best.pt"

T_HISTORY = 20
T_FUTURE = 10
MAX_DROPLETS = 64
INPUT_DIM = 5
TARGET_DIM = 2
STRIDE = 5

BATCH_SIZE = 8
EPOCHS = 50
LR = 1e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
NUM_WORKERS = 0
LOG_EVERY_N_BATCHES = 25

MODEL_CONFIG = {
    "input_dim": INPUT_DIM,
    "target_dim": TARGET_DIM,
    "T_history": T_HISTORY,
    "T_future": T_FUTURE,
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
        npz_path=NPZ_PATH,
        stride=STRIDE,
        T_history=T_HISTORY,
        T_future=T_FUTURE,
        max_droplets=MAX_DROPLETS,
    )
    print(f"Train windows: {len(train_ds)}")
    print(f"Val windows: {len(val_ds)}")
    print(f"Test windows: {len(test_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    model = CanonicalWindowTransformer(**MODEL_CONFIG).to(device)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    run_shape_test(model, train_loader, device)
    if args.shape_test_only:
        return

    best_val_loss = float("inf")
    checkpoint_path = Path(CHECKPOINT_PATH)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, args.log_every)
        val_loss = evaluate(model, val_loader, device, args.log_every)

        print(f"epoch {epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "normalization_stats": normalization_stats,
                    "model_config": MODEL_CONFIG,
                },
                checkpoint_path,
            )
            print(f"Saved best checkpoint: {checkpoint_path}")

    test_loss = evaluate(model, test_loader, device)
    print(f"test_loss={test_loss:.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the canonical window Transformer.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--log-every", type=int, default=LOG_EVERY_N_BATCHES)
    parser.add_argument("--shape-test-only", action="store_true")
    return parser.parse_args()


def run_shape_test(model, train_loader, device) -> None:
    model.eval()
    batch = next(iter(train_loader))
    history_x = batch["history_x"].to(device)
    history_mask = batch["history_mask"].to(device)
    future_y = batch["future_y"].to(device)
    future_mask = batch["future_mask"].to(device)

    with torch.no_grad():
        pred_v = model(history_x, history_mask)
        loss = masked_future_velocity_mse_loss(pred_v, future_y, future_mask)

    print(f"history_x:    {tuple(history_x.shape)}")
    print(f"history_mask: {tuple(history_mask.shape)}")
    print(f"future_y:     {tuple(future_y.shape)}")
    print(f"future_mask:  {tuple(future_mask.shape)}")
    print(f"pred_v:       {tuple(pred_v.shape)}")
    print(f"loss:         {float(loss):.6f}")

    assert pred_v.shape == future_y.shape


def train_one_epoch(model, train_loader, optimizer, device, log_every) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0
    total_batches = len(train_loader)

    for batch in train_loader:
        history_x = batch["history_x"].to(device)
        history_mask = batch["history_mask"].to(device)
        future_y = batch["future_y"].to(device)
        future_mask = batch["future_mask"].to(device)

        optimizer.zero_grad(set_to_none=True)
        pred_v = model(history_x, history_mask)
        loss = masked_future_velocity_mse_loss(pred_v, future_y, future_mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        total_loss += float(loss.detach().cpu())
        num_batches += 1
        if log_every > 0 and (num_batches % log_every == 0 or num_batches == total_batches):
            running_loss = total_loss / num_batches
            percent = 100.0 * num_batches / max(total_batches, 1)
            print(
                f"  train batch {num_batches:04d}/{total_batches:04d} "
                f"({percent:5.1f}%) loss={running_loss:.6f}"
            )

    return total_loss / max(num_batches, 1)


def evaluate(model, loader, device, log_every=0) -> float:
    model.eval()
    total_loss = 0.0
    num_batches = 0
    total_batches = len(loader)

    with torch.no_grad():
        for batch in loader:
            history_x = batch["history_x"].to(device)
            history_mask = batch["history_mask"].to(device)
            future_y = batch["future_y"].to(device)
            future_mask = batch["future_mask"].to(device)

            pred_v = model(history_x, history_mask)
            loss = masked_future_velocity_mse_loss(pred_v, future_y, future_mask)
            total_loss += float(loss.detach().cpu())
            num_batches += 1
            if log_every > 0 and (num_batches % log_every == 0 or num_batches == total_batches):
                running_loss = total_loss / num_batches
                percent = 100.0 * num_batches / max(total_batches, 1)
                print(
                    f"  val batch   {num_batches:04d}/{total_batches:04d} "
                    f"({percent:5.1f}%) loss={running_loss:.6f}"
                )

    return total_loss / max(num_batches, 1)


if __name__ == "__main__":
    main()
