from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from configs.constants import EXPERIMENT_NAME
from configs.paths import PROCESSED_DIR
from src.trajectory_model import TrajectoryAttentionModel


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot true vs predicted future trajectories from a trained attention model."
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
    parser.add_argument(
        "--model-file",
        type=Path,
        default=None,
        help=(
            "Optional path to attention_model.pt. If provided, this path is used "
            "instead of outputs/processed/<experiment_name>/attention_model.pt."
        ),
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Number of random validation samples to plot.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for validation sample selection.",
    )
    return parser.parse_args(args)


def load_windows(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Trajectory windows file not found: {path}")
    data = np.load(path)
    return data["X"], data["Y"], data["input_mask"], data["target_mask"]


def split_indices(num_samples: int, train_frac: float = 0.8, seed: int = 42) -> dict[str, np.ndarray]:
    indices = np.arange(num_samples)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    split = int(num_samples * train_frac)
    return {
        "train": indices[:split],
        "val": indices[split:],
    }


def load_model(model_path: Path, device: torch.device) -> nn.Module:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = TrajectoryAttentionModel().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def integrate_positions(last_xy: torch.Tensor, velocities: torch.Tensor) -> torch.Tensor:
    # last_xy: [N_MAX, 2], velocities: [T_FUTURE, N_MAX, 2]
    displacements = velocities.cumsum(dim=0)
    return last_xy.unsqueeze(0) + displacements


def plot_sample(
    sample_idx: int,
    X: np.ndarray,
    Y: np.ndarray,
    input_mask: np.ndarray,
    target_mask: np.ndarray,
    predictions: np.ndarray,
    output_dir: Path,
) -> None:
    x_history = X[sample_idx]  # [T_HISTORY, N_MAX, 4]
    y_true = Y[sample_idx]  # [T_FUTURE, N_MAX, 2]
    mask = target_mask[sample_idx]  # [T_FUTURE, N_MAX]
    y_pred = predictions  # [T_FUTURE, N_MAX, 2]

    last_xy = x_history[-1, :, :2]
    true_positions = np.zeros((y_true.shape[0] + 1, y_true.shape[1], 2), dtype=float)
    pred_positions = np.zeros_like(true_positions)
    true_positions[0] = last_xy
    pred_positions[0] = last_xy
    true_positions[1:] = last_xy[np.newaxis] + np.cumsum(y_true, axis=0)
    pred_positions[1:] = last_xy[np.newaxis] + np.cumsum(y_pred, axis=0)

    valid_droplets = np.where(mask.any(axis=0))[0]
    if valid_droplets.size == 0:
        print(f"Sample {sample_idx} has no valid future targets; skipping plot.")
        return

    plt.figure(figsize=(10, 8))
    for droplet in valid_droplets:
        valid_steps = np.concatenate([[True], mask[:, droplet]])
        if not valid_steps.any():
            continue

        plt.plot(
            true_positions[valid_steps, droplet, 0],
            true_positions[valid_steps, droplet, 1],
            "-o",
            label=f"true droplet {droplet}",
            alpha=0.7,
        )
        plt.plot(
            pred_positions[valid_steps, droplet, 0],
            pred_positions[valid_steps, droplet, 1],
            "--x",
            label=f"pred droplet {droplet}",
            alpha=0.7,
        )

    plt.title(f"Sample {sample_idx}: true vs predicted future trajectories")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc="best", fontsize="small")
    plt.grid(True)
    output_path = output_dir / f"trajectory_plot_sample_{sample_idx}.png"
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot: {output_path}")


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

    if args.model_file is not None:
        model_path = args.model_file
    else:
        model_path = output_dir / "attention_model.pt"

    X, Y, input_mask, target_mask = load_windows(windows_path)
    indices = split_indices(X.shape[0], seed=args.seed)["val"]
    if len(indices) == 0:
        raise ValueError("No validation samples available to plot.")

    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(indices, size=min(args.samples, len(indices)), replace=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)

    X_tensor = torch.from_numpy(X[chosen]).float().to(device)
    input_mask_tensor = torch.from_numpy(input_mask[chosen]).bool().to(device)
    with torch.no_grad():
        y_pred = model(X_tensor, input_mask_tensor).cpu().numpy()

    output_dir.mkdir(parents=True, exist_ok=True)
    for rel_idx, sample_idx in enumerate(chosen):
        plot_sample(sample_idx, X, Y, input_mask, target_mask, y_pred[rel_idx], output_dir)


if __name__ == "__main__":
    main()
