from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
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
from scripts.train_recurrent_geometry_model import RecurrentGeometryModel

T_HISTORY = 20
T_FUTURE = 10
ALIGNMENT_PRINTED = False


def log(message: str) -> None:
    print(message, flush=True)


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train recurrent geometry model with 10-step autoregressive rollout."
    )
    parser.add_argument("--experiment-name", default=EXPERIMENT_NAME)
    parser.add_argument(
        "--windows-file",
        type=Path,
        default=None,
        help="Defaults to outputs/processed/<experiment_name>/recurrent_geometry_windows.npz.",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help=(
            "Defaults to "
            "outputs/processed/<experiment_name>/recurrent_geometry_model_gru_best.pt."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda-s", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--num-inter-op-threads", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args(args)


class RolloutDataset(Dataset):
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
        "channel_names",
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
) -> tuple[RolloutDataset, RolloutDataset]:
    indices = np.arange(Z.shape[0])
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    split = int(Z.shape[0] * train_frac)
    train_idx = indices[:split]
    val_idx = indices[split:]
    return (
        RolloutDataset(
            Z[train_idx], mask[train_idx], target_v_s[train_idx], target_mask[train_idx]
        ),
        RolloutDataset(
            Z[val_idx], mask[val_idx], target_v_s[val_idx], target_mask[val_idx]
        ),
    )


def build_model_from_checkpoint(
    checkpoint_path: Path,
    windows: dict[str, Any],
    device: torch.device,
) -> tuple[RecurrentGeometryModel, dict[str, Any]]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Initial checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = dict(checkpoint.get("config", {}))
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
    return model, config


def transformer_context(
    model: RecurrentGeometryModel,
    z_step: torch.Tensor,
    mask_step: torch.Tensor,
) -> torch.Tensor:
    numeric = z_step[..., : model.numeric_input_dim]
    channel_ids = z_step[..., model.numeric_input_dim].long()
    channel_ids = channel_ids.clamp(min=0, max=model.num_channel_embeddings - 1)
    channel_embedding = model.channel_embed(channel_ids)
    encoded = model.state_encoder(torch.cat([numeric, channel_embedding], dim=-1))
    return model.transformer(encoded, src_key_padding_mask=~mask_step)


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


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded_mask = mask.unsqueeze(-1).expand_as(pred)
    if not expanded_mask.any():
        return torch.tensor(0.0, device=pred.device)
    return ((pred - target) ** 2)[expanded_mask].mean()


def future_target_slice(target_v_s: torch.Tensor) -> slice:
    target_slice = slice(T_HISTORY - 1, T_HISTORY + T_FUTURE - 1)
    requested_slice = slice(T_HISTORY, T_HISTORY + T_FUTURE)
    if target_v_s[:, requested_slice].shape[1] != T_FUTURE:
        # target_v_s[t] stores v_s at frame t+1, so frame T_HISTORY maps to
        # target index T_HISTORY - 1. With T_TOTAL=30, slice(20, 30) has 9 values.
        return target_slice
    return requested_slice


def assert_rollout_alignment(
    pred_v_s: torch.Tensor,
    pred_s: torch.Tensor,
    true_v_s: torch.Tensor,
    true_s: torch.Tensor,
    valid: torch.Tensor,
    target_slice: slice,
) -> None:
    global ALIGNMENT_PRINTED
    assert pred_v_s.shape == true_v_s.shape, (
        f"pred_v_s shape {pred_v_s.shape} != true_v_s shape {true_v_s.shape}"
    )
    assert pred_s.shape == true_s.shape, (
        f"pred_s shape {pred_s.shape} != true_s shape {true_s.shape}"
    )
    assert pred_v_s.shape[:3] == valid.shape, (
        f"prediction prefix {pred_v_s.shape[:3]} != valid mask shape {valid.shape}"
    )
    if not ALIGNMENT_PRINTED:
        print(
            "Rollout alignment check: "
            f"pred_v_s[:, k] -> target_v_s[:, {target_slice.start}+k] "
            f"and pred_s[:, k] -> Z[:, {T_HISTORY}+k, s_coord].",
            flush=True,
        )
        print(
            "For this dataset target_v_s[t] is v_s at frame t+1; "
            "therefore target_v_s index 19 aligns with future frame 20.",
            flush=True,
        )
        ALIGNMENT_PRINTED = True


def rollout_forward(
    model: RecurrentGeometryModel,
    Z: torch.Tensor,
    mask: torch.Tensor,
    stats: dict[str, torch.Tensor],
    feature_indices: dict[str, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, _, n_max, _ = Z.shape
    h = torch.zeros(
        batch_size,
        n_max,
        model.hidden_dim,
        dtype=Z.dtype,
        device=Z.device,
    )

    for step in range(T_HISTORY):
        context = transformer_context(model, Z[:, step], mask[:, step])
        h = update_hidden(model, h, context, mask[:, step])

    current_z = Z[:, T_HISTORY - 1].clone()
    pred_v_steps: list[torch.Tensor] = []
    pred_s_steps: list[torch.Tensor] = []
    s_index = feature_indices["s_coord"]
    v_s_index = feature_indices["v_s"]

    for step in range(T_FUTURE):
        future_mask_step = mask[:, T_HISTORY + step]
        context = transformer_context(model, current_z, future_mask_step)
        h = update_hidden(model, h, context, future_mask_step)
        pred_v_s_norm = model.velocity_head(h)

        current_s_norm = current_z[..., s_index : s_index + 1]
        raw_s = current_s_norm * stats["s_std"] + stats["s_mean"]
        raw_v_s = pred_v_s_norm * stats["target_std"] + stats["target_mean"]
        next_s_norm = (raw_s + raw_v_s - stats["s_mean"]) / stats["s_std"]

        pred_v_steps.append(pred_v_s_norm)
        pred_s_steps.append(next_s_norm)

        next_z = current_z.clone()
        next_z[..., s_index : s_index + 1] = next_s_norm
        next_z[..., v_s_index : v_s_index + 1] = pred_v_s_norm
        current_z = next_z

    return torch.stack(pred_v_steps, dim=1), torch.stack(pred_s_steps, dim=1)


def batch_losses(
    model: RecurrentGeometryModel,
    Z: torch.Tensor,
    mask: torch.Tensor,
    target_v_s: torch.Tensor,
    target_mask: torch.Tensor,
    stats: dict[str, torch.Tensor],
    feature_indices: dict[str, int],
    lambda_s: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pred_v_s, pred_s = rollout_forward(model, Z, mask, stats, feature_indices)
    target_slice = future_target_slice(target_v_s)
    true_v_s = target_v_s[:, target_slice]
    valid = target_mask[:, target_slice]
    true_s = Z[:, T_HISTORY : T_HISTORY + T_FUTURE, :, feature_indices["s_coord"]]
    true_s = true_s.unsqueeze(-1)
    assert_rollout_alignment(pred_v_s, pred_s, true_v_s, true_s, valid, target_slice)

    v_loss = masked_mse(pred_v_s, true_v_s, valid)
    s_loss = masked_mse(pred_s, true_s, valid)
    loss = v_loss + lambda_s * s_loss
    return loss, v_loss, s_loss, pred_v_s.detach(), pred_s.detach()


def make_stats(windows: dict[str, Any], device: torch.device) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    feature_columns = list(windows["feature_columns"])
    numeric_features = list(windows["numeric_features"])
    target_columns = list(windows["target_columns"])
    feature_indices = {
        "s_coord": feature_columns.index("s_coord"),
        "v_s": feature_columns.index("v_s"),
    }
    s_numeric_index = numeric_features.index("s_coord")
    target_index = target_columns.index("v_s")
    stats = {
        "s_mean": torch.tensor(float(windows["numeric_feature_mean"][s_numeric_index]), device=device),
        "s_std": torch.tensor(float(windows["numeric_feature_std"][s_numeric_index]), device=device),
        "target_mean": torch.tensor(float(windows["target_mean"][target_index]), device=device),
        "target_std": torch.tensor(float(windows["target_std"][target_index]), device=device),
    }
    return stats, feature_indices


def train_epoch(
    model: RecurrentGeometryModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    stats: dict[str, torch.Tensor],
    feature_indices: dict[str, int],
    lambda_s: float,
) -> tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    total_v_loss = 0.0
    total_s_loss = 0.0
    count = 0
    for Z, mask, target_v_s, target_mask in loader:
        Z = Z.to(device)
        mask = mask.to(device)
        target_v_s = target_v_s.to(device)
        target_mask = target_mask.to(device)

        optimizer.zero_grad()
        loss, v_loss, s_loss, _, _ = batch_losses(
            model, Z, mask, target_v_s, target_mask, stats, feature_indices, lambda_s
        )
        loss.backward()
        optimizer.step()

        batch_size = Z.size(0)
        total_loss += loss.item() * batch_size
        total_v_loss += v_loss.item() * batch_size
        total_s_loss += s_loss.item() * batch_size
        count += batch_size
    return total_loss / count, total_v_loss / count, total_s_loss / count


def eval_epoch(
    model: RecurrentGeometryModel,
    loader: DataLoader,
    device: torch.device,
    stats: dict[str, torch.Tensor],
    feature_indices: dict[str, int],
    lambda_s: float,
) -> tuple[float, float, float, list[float], list[float]]:
    model.eval()
    total_loss = 0.0
    total_v_loss = 0.0
    total_s_loss = 0.0
    count = 0
    s_sq_sum = np.zeros((T_FUTURE,), dtype=np.float64)
    v_sq_sum = np.zeros((T_FUTURE,), dtype=np.float64)
    step_count = np.zeros((T_FUTURE,), dtype=np.float64)

    with torch.no_grad():
        for Z, mask, target_v_s, target_mask in loader:
            Z = Z.to(device)
            mask = mask.to(device)
            target_v_s = target_v_s.to(device)
            target_mask = target_mask.to(device)
            loss, v_loss, s_loss, pred_v_s, pred_s = batch_losses(
                model, Z, mask, target_v_s, target_mask, stats, feature_indices, lambda_s
            )

            batch_size = Z.size(0)
            total_loss += loss.item() * batch_size
            total_v_loss += v_loss.item() * batch_size
            total_s_loss += s_loss.item() * batch_size
            count += batch_size

            target_slice = future_target_slice(target_v_s)
            valid = target_mask[:, target_slice].cpu().numpy()
            true_v_s = target_v_s[:, target_slice].cpu().numpy()
            true_s = Z[:, T_HISTORY : T_HISTORY + T_FUTURE, :, feature_indices["s_coord"]]
            true_s = true_s.unsqueeze(-1).cpu().numpy()

            pred_v_raw = (
                pred_v_s.cpu().numpy() * float(stats["target_std"].cpu())
                + float(stats["target_mean"].cpu())
            )
            true_v_raw = (
                true_v_s * float(stats["target_std"].cpu())
                + float(stats["target_mean"].cpu())
            )
            pred_s_raw = (
                pred_s.cpu().numpy() * float(stats["s_std"].cpu())
                + float(stats["s_mean"].cpu())
            )
            true_s_raw = (
                true_s * float(stats["s_std"].cpu())
                + float(stats["s_mean"].cpu())
            )

            for step in range(T_FUTURE):
                step_mask = valid[:, step]
                if step_mask.any():
                    s_err = pred_s_raw[:, step, :, 0] - true_s_raw[:, step, :, 0]
                    v_err = pred_v_raw[:, step, :, 0] - true_v_raw[:, step, :, 0]
                    s_sq_sum[step] += float(np.sum(s_err[step_mask] ** 2))
                    v_sq_sum[step] += float(np.sum(v_err[step_mask] ** 2))
                    step_count[step] += float(step_mask.sum())

    s_rmse = [
        float(np.sqrt(s_sq_sum[i] / step_count[i])) if step_count[i] else float("nan")
        for i in range(T_FUTURE)
    ]
    v_rmse = [
        float(np.sqrt(v_sq_sum[i] / step_count[i])) if step_count[i] else float("nan")
        for i in range(T_FUTURE)
    ]
    return (
        total_loss / count,
        total_v_loss / count,
        total_s_loss / count,
        s_rmse,
        v_rmse,
    )


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
    torch.set_num_threads(args.num_threads)
    torch.set_num_interop_threads(args.num_inter_op_threads)

    output_dir = PROCESSED_DIR / args.experiment_name
    windows_path = args.windows_file or output_dir / "recurrent_geometry_windows.npz"
    init_checkpoint = args.init_checkpoint or output_dir / "recurrent_geometry_model_gru_best.pt"
    best_model_path = output_dir / "recurrent_geometry_rollout_best.pt"

    windows = load_windows(windows_path)
    train_ds, val_ds = split_dataset(
        windows["Z"],
        windows["mask"],
        windows["target_v_s"],
        windows["target_mask"],
        seed=args.seed,
    )
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": False,
        "persistent_workers": False,
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    device = torch.device("cpu")
    model, init_config = build_model_from_checkpoint(init_checkpoint, windows, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    stats, feature_indices = make_stats(windows, device)
    config = {
        **init_config,
        "model_type": "recurrent_geometry_rollout",
        "init_checkpoint": str(init_checkpoint),
        "lambda_s": args.lambda_s,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "num_threads": args.num_threads,
        "num_inter_op_threads": args.num_inter_op_threads,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "t_history": T_HISTORY,
        "t_future": T_FUTURE,
    }

    log(f"Using device: {device}")
    log(f"Windows file: {windows_path}")
    log(f"Initial checkpoint: {init_checkpoint}")
    log(f"Saving best rollout checkpoint to: {best_model_path}")
    log(f"Batch size: {args.batch_size}")
    log(f"lambda_s: {args.lambda_s}")
    log(f"Learning rate: {args.lr}")
    log(f"Torch threads: {torch.get_num_threads()}")
    log(f"Torch inter-op threads: {torch.get_num_interop_threads()}")
    log(f"DataLoader workers: {args.num_workers}")

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, train_v_loss, train_s_loss = train_epoch(
            model, train_loader, optimizer, device, stats, feature_indices, args.lambda_s
        )
        val_loss, val_v_loss, val_s_loss, s_rmse, v_rmse = eval_epoch(
            model, val_loader, device, stats, feature_indices, args.lambda_s
        )
        epoch_duration = time.perf_counter() - epoch_start
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
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.6f} train_v_loss={train_v_loss:.6f} "
            f"train_s_loss={train_s_loss:.6f} | "
            f"val_loss={val_loss:.6f} val_v_loss={val_v_loss:.6f} "
            f"val_s_loss={val_s_loss:.6f}"
            + (" | best" if improved else "")
        )
        log(
            "  val RMSE s_coord by step: "
            + ", ".join(f"{value:.3f}" for value in s_rmse)
        )
        log(
            "  val RMSE v_s by step: "
            + ", ".join(f"{value:.3f}" for value in v_rmse)
        )
        log(f"Epoch {epoch:02d} completed in {epoch_duration:.1f} sec")

        if epochs_no_improve >= args.patience:
            log(f"Early stopping after {epoch:02d} epochs.")
            break

    log(f"Best epoch: {best_epoch:02d} | best val_loss={best_val_loss:.6f}")
    log(f"Saved best model to: {best_model_path}")


if __name__ == "__main__":
    main()
