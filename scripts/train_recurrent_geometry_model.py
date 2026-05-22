from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from configs.constants import EXPERIMENT_NAME
from configs.paths import PROCESSED_DIR


def log(message: str) -> None:
    print(message, flush=True)


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a one-step recurrent geometry-native droplet model."
    )
    parser.add_argument(
        "--experiment-name",
        default=EXPERIMENT_NAME,
        help="Experiment name under outputs/processed/.",
    )
    parser.add_argument(
        "--windows-file",
        type=Path,
        default=None,
        help=(
            "Optional recurrent geometry NPZ. Defaults to "
            "outputs/processed/<experiment_name>/trajectory_windows_recurrent_geometry.npz."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(args)


class RecurrentGeometryDataset(Dataset):
    def __init__(
        self,
        Z: np.ndarray,
        mask: np.ndarray,
        target_v_s: np.ndarray,
        target_mask: np.ndarray,
    ) -> None:
        self.Z = torch.from_numpy(Z).float()
        self.mask = torch.from_numpy(mask).bool()
        self.target_v_s = torch.from_numpy(target_v_s).float()
        self.target_mask = torch.from_numpy(target_mask).bool()

    def __len__(self) -> int:
        return self.Z.shape[0]

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.Z[index],
            self.mask[index],
            self.target_v_s[index],
            self.target_mask[index],
        )


class RecurrentGeometryModel(nn.Module):
    def __init__(
        self,
        numeric_input_dim: int,
        num_channel_embeddings: int,
        channel_embedding_dim: int = 16,
        hidden_dim: int = 96,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.numeric_input_dim = numeric_input_dim
        self.num_channel_embeddings = num_channel_embeddings

        self.channel_embed = nn.Embedding(num_channel_embeddings, channel_embedding_dim)
        self.input_mlp = nn.Sequential(
            nn.Linear(numeric_input_dim + channel_embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.velocity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # z: [B, T, N, 4], feature order: s_coord, v_s, d_centerline, channel_id_int
        # mask: [B, T, N], True for valid droplets.
        batch_size, time_steps, n_max, _ = z.shape
        numeric = z[..., : self.numeric_input_dim]
        channel_ids = z[..., self.numeric_input_dim].long()
        channel_ids = channel_ids.clamp(min=0, max=self.num_channel_embeddings - 1)

        channel_embedding = self.channel_embed(channel_ids)
        hidden = self.input_mlp(torch.cat([numeric, channel_embedding], dim=-1))
        hidden = hidden.reshape(batch_size * time_steps, n_max, -1)

        padding_mask = ~mask.reshape(batch_size * time_steps, n_max)
        hidden = self.transformer(hidden, src_key_padding_mask=padding_mask)
        hidden = hidden.reshape(batch_size, time_steps, n_max, -1)

        pred_v_s = self.velocity_head(hidden[:, :-1])
        return pred_v_s


def load_windows(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Recurrent geometry windows file not found: {path}")

    data = np.load(path)
    required = {"Z", "mask", "target_v_s", "target_mask", "channel_ids"}
    missing = required - set(data.files)
    if missing:
        raise KeyError(f"Missing required NPZ arrays: {sorted(missing)}")

    return {
        "Z": data["Z"],
        "mask": data["mask"],
        "target_v_s": data["target_v_s"],
        "target_mask": data["target_mask"],
        "feature_columns": data["feature_columns"].astype(str),
        "numeric_features": data["numeric_features"].astype(str),
        "target_columns": data["target_columns"].astype(str),
        "numeric_feature_mean": data["numeric_feature_mean"],
        "numeric_feature_std": data["numeric_feature_std"],
        "target_mean": data["target_mean"],
        "target_std": data["target_std"],
        "channel_names": data["channel_names"].astype(str),
        "channel_ids": data["channel_ids"],
    }


def split_dataset(
    Z: np.ndarray,
    mask: np.ndarray,
    target_v_s: np.ndarray,
    target_mask: np.ndarray,
    train_frac: float = 0.8,
    seed: int = 42,
) -> tuple[RecurrentGeometryDataset, RecurrentGeometryDataset]:
    num_samples = Z.shape[0]
    indices = np.arange(num_samples)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    split = int(num_samples * train_frac)
    train_idx = indices[:split]
    val_idx = indices[split:]

    return (
        RecurrentGeometryDataset(
            Z[train_idx], mask[train_idx], target_v_s[train_idx], target_mask[train_idx]
        ),
        RecurrentGeometryDataset(
            Z[val_idx], mask[val_idx], target_v_s[val_idx], target_mask[val_idx]
        ),
    )


def masked_mse_loss(
    pred: torch.Tensor, target: torch.Tensor, target_mask: torch.Tensor
) -> torch.Tensor:
    mask = target_mask.unsqueeze(-1).expand_as(pred).float()
    squared = (pred - target) ** 2
    total = (squared * mask).sum()
    count = mask.sum()
    if count == 0:
        return torch.tensor(0.0, device=pred.device)
    return total / count


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_every: int = 25,
) -> float:
    model.train()
    total_loss = 0.0
    count = 0

    for batch_index, (Z, mask, target_v_s, target_mask) in enumerate(loader, start=1):
        Z = Z.to(device)
        mask = mask.to(device)
        target_v_s = target_v_s.to(device)
        target_mask = target_mask.to(device)

        optimizer.zero_grad()
        pred = model(Z, mask)
        loss = masked_mse_loss(pred, target_v_s, target_mask)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * Z.size(0)
        count += Z.size(0)
        if batch_index == 1 or batch_index % log_every == 0 or batch_index == len(loader):
            log(
                f"Epoch {epoch:02d} train batch {batch_index}/{len(loader)} "
                f"| running_loss={total_loss / max(count, 1):.6f}"
            )

    return total_loss / max(count, 1)


def eval_epoch(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for Z, mask, target_v_s, target_mask in loader:
            Z = Z.to(device)
            mask = mask.to(device)
            target_v_s = target_v_s.to(device)
            target_mask = target_mask.to(device)
            pred = model(Z, mask)
            loss = masked_mse_loss(pred, target_v_s, target_mask)
            total_loss += loss.item() * Z.size(0)
            count += Z.size(0)
    return total_loss / max(count, 1)


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    windows: dict[str, Any],
    config: dict[str, Any],
    epoch: int,
    val_loss: float,
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "feature_columns": windows["feature_columns"],
        "numeric_features": windows["numeric_features"],
        "target_columns": windows["target_columns"],
        "numeric_feature_mean": windows["numeric_feature_mean"],
        "numeric_feature_std": windows["numeric_feature_std"],
        "target_mean": windows["target_mean"],
        "target_std": windows["target_std"],
        "channel_names": windows["channel_names"],
        "channel_ids": windows["channel_ids"],
        "config": config,
        "best_epoch": epoch,
        "best_val_loss": val_loss,
    }


def main() -> None:
    args = parse_args()
    output_dir = PROCESSED_DIR / args.experiment_name
    windows_path = (
        args.windows_file
        if args.windows_file is not None
        else output_dir / "trajectory_windows_recurrent_geometry.npz"
    )

    windows = load_windows(windows_path)
    Z = windows["Z"]
    mask = windows["mask"]
    target_v_s = windows["target_v_s"]
    target_mask = windows["target_mask"]

    train_ds, val_ds = split_dataset(
        Z, mask, target_v_s, target_mask, seed=args.seed
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    config = {
        "model_type": "recurrent_geometry_one_step",
        "numeric_input_dim": int(len(windows["numeric_features"])),
        "num_channel_embeddings": int(windows["channel_ids"].max()) + 1,
        "channel_embedding_dim": 16,
        "hidden_dim": 96,
        "num_heads": 4,
        "num_layers": 2,
        "dropout": 0.1,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "patience": args.patience,
        "seed": args.seed,
        "input_shape": tuple(int(value) for value in Z.shape),
        "target_shape": tuple(int(value) for value in target_v_s.shape),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RecurrentGeometryModel(
        numeric_input_dim=config["numeric_input_dim"],
        num_channel_embeddings=config["num_channel_embeddings"],
        channel_embedding_dim=config["channel_embedding_dim"],
        hidden_dim=config["hidden_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    best_model_path = output_dir / "recurrent_geometry_model_best.pt"
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0

    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"Using device: {device}")
    log(f"Windows file: {windows_path}")
    log(f"Dataset shapes: Z={Z.shape}, target_v_s={target_v_s.shape}")
    log(f"Mask shapes: mask={mask.shape}, target_mask={target_mask.shape}")
    log(f"Feature order: {list(windows['feature_columns'])}")
    log(f"Numeric features: {list(windows['numeric_features'])}")
    log(f"Target columns: {list(windows['target_columns'])}")
    log(f"Channel embeddings: {len(windows['channel_names'])}")
    log(
        f"Training for up to {args.epochs} epochs with "
        f"early stopping patience={args.patience}."
    )

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        val_loss = eval_epoch(model, val_loader, device)
        improved = val_loss < best_val_loss

        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(
                checkpoint_payload(
                    model, optimizer, windows, config, best_epoch, best_val_loss
                ),
                best_model_path,
            )
        else:
            epochs_no_improve += 1

        log(
            f"Epoch {epoch:02d} | train_loss={train_loss:.6f} "
            f"| val_loss={val_loss:.6f}" + (" | best" if improved else "")
        )

        if epochs_no_improve >= args.patience:
            log(f"Early stopping after {epoch:02d} epochs.")
            break

    log(f"Best epoch: {best_epoch:02d} | best val_loss={best_val_loss:.6f}")
    log(f"Saved best model to: {best_model_path}")


if __name__ == "__main__":
    main()
