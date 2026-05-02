from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from configs.constants import EXPERIMENT_NAME
from configs.paths import PROCESSED_DIR
from src.trajectory_model import TrajectoryAttentionModel


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an attention model for trajectory velocity prediction."
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
        "--windows-file",
        type=Path,
        default=None,
        help=(
            "Optional path to trajectory_windows.npz. If provided, this file is used "
            "instead of outputs/processed/<experiment_name>/trajectory_windows.npz."
        ),
    )
    return parser.parse_args(args)


class TrajectoryWindowDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        input_mask: np.ndarray,
        target_mask: np.ndarray,
    ) -> None:
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()
        self.input_mask = torch.from_numpy(input_mask).bool()
        self.target_mask = torch.from_numpy(target_mask).bool()

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.X[idx],
            self.Y[idx],
            self.input_mask[idx],
            self.target_mask[idx],
        )


def load_windows(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Trajectory windows file not found: {path}")
    data = np.load(path)
    return data["X"], data["Y"], data["input_mask"], data["target_mask"]


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.unsqueeze(-1).expand_as(pred).float()
    squared = (pred - target) ** 2
    total = (squared * mask).sum()
    count = mask.sum()
    if count == 0:
        return torch.tensor(0.0, device=pred.device)
    return total / count


def split_dataset(
    X: np.ndarray,
    Y: np.ndarray,
    input_mask: np.ndarray,
    target_mask: np.ndarray,
    train_frac: float = 0.8,
    seed: int = 42,
) -> tuple[TrajectoryWindowDataset, TrajectoryWindowDataset]:
    num_samples = X.shape[0]
    indices = np.arange(num_samples)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    split = int(num_samples * train_frac)
    train_idx = indices[:split]
    val_idx = indices[split:]

    train_ds = TrajectoryWindowDataset(
        X[train_idx], Y[train_idx], input_mask[train_idx], target_mask[train_idx]
    )
    val_ds = TrajectoryWindowDataset(
        X[val_idx], Y[val_idx], input_mask[val_idx], target_mask[val_idx]
    )
    return train_ds, val_ds


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    count = 0

    for X_batch, Y_batch, input_mask, target_mask in loader:
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)
        input_mask = input_mask.to(device)
        target_mask = target_mask.to(device)

        optimizer.zero_grad()
        pred = model(X_batch, input_mask)
        loss = masked_mse_loss(pred, Y_batch, target_mask)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        count += X_batch.size(0)

    return total_loss / max(count, 1)


def eval_epoch(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for X_batch, Y_batch, input_mask, target_mask in loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            input_mask = input_mask.to(device)
            target_mask = target_mask.to(device)

            pred = model(X_batch, input_mask)
            loss = masked_mse_loss(pred, Y_batch, target_mask)
            total_loss += loss.item() * X_batch.size(0)
            count += X_batch.size(0)

    return total_loss / max(count, 1)


def main() -> None:
    args = parse_args()

    if args.windows_file is not None:
        windows_path = args.windows_file
        output_dir = args.windows_file.parent
    else:
        if not args.experiment_name:
            raise ValueError(
                "Experiment name is required when --windows-file is not provided."
            )
        output_dir = PROCESSED_DIR / args.experiment_name
        windows_path = output_dir / "trajectory_windows.npz"

    X, Y, input_mask, target_mask = load_windows(windows_path)
    train_ds, val_ds = split_dataset(X, Y, input_mask, target_mask)

    batch_size = 64
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrajectoryAttentionModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 20
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = eval_epoch(model, val_loader, device)
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "attention_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    main()
