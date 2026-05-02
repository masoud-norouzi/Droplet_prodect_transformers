from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from configs.constants import EXPERIMENT_NAME
from configs.paths import PROCESSED_DIR


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a constant velocity baseline on trajectory windows."
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
            "Optional path to trajectory_windows.npz. If provided, this path is used "
            "instead of outputs/processed/<experiment_name>/trajectory_windows.npz."
        ),
    )
    return parser.parse_args(args)


def load_windows(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Trajectory windows file not found: {path}")
    data = np.load(path)
    X = data["X"]
    Y = data["Y"]
    target_mask = data["target_mask"]
    return X, Y, target_mask


def evaluate_baseline(X: np.ndarray, Y: np.ndarray, target_mask: np.ndarray) -> dict[str, float]:
    last_v = X[:, -1, :, 2:4]
    Y_pred = np.broadcast_to(last_v[:, np.newaxis, :, :], Y.shape)

    if target_mask.dtype != bool:
        target_mask = target_mask.astype(bool)

    errors = (Y_pred - Y) ** 2
    masked_errors = errors[target_mask]
    if masked_errors.size == 0:
        raise ValueError("No valid target values found in target_mask.")

    mse = float(masked_errors.mean())
    rmse = float(np.sqrt(mse))

    vx_errors = errors[..., 0][target_mask]
    vy_errors = errors[..., 1][target_mask]
    rmse_vx = float(np.sqrt(vx_errors.mean()))
    rmse_vy = float(np.sqrt(vy_errors.mean()))

    return {
        "masked_mse": mse,
        "masked_rmse": rmse,
        "rmse_vx": rmse_vx,
        "rmse_vy": rmse_vy,
    }


def main() -> None:
    args = parse_args()

    if args.windows_file is not None:
        windows_path = args.windows_file
    else:
        if not args.experiment_name:
            raise ValueError(
                "Experiment name is required when --windows-file is not provided."
            )
        windows_path = PROCESSED_DIR / args.experiment_name / "trajectory_windows.npz"

    X, Y, target_mask = load_windows(windows_path)
    stats = evaluate_baseline(X, Y, target_mask)

    print("=== constant velocity baseline ===")
    print(f"masked MSE: {stats['masked_mse']:.6f}")
    print(f"masked RMSE: {stats['masked_rmse']:.6f}")
    print(f"RMSE vx: {stats['rmse_vx']:.6f}")
    print(f"RMSE vy: {stats['rmse_vy']:.6f}")
    print("=== end baseline ===")


if __name__ == "__main__":
    main()
