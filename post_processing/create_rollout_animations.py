from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
import numpy as np
import torch
from torch.utils.data import default_collate

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.canonical_rollout_transformer import CanonicalRolloutTransformer
from train_canonical_rollout_transformer import (
    LOSS_ALPHA,
    MAX_DROPLETS,
    T_HISTORY,
    boundary_conditioned_rollout,
    move_batch_to_device,
    rollout_weights,
)
from utils.canonical_dataset.canonical_window_dataset import CanonicalWindowDataset


DEFAULT_NPZ = Path("outputs/processed/2/canonical_dataset.npz")
DEFAULT_OUTPUT_DIR = Path("outputs/post_processing/rollout_100_frame_animations")


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    checkpoint_path = args.checkpoint or newest_rollout_checkpoint(Path("outputs/models"))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = CanonicalRolloutTransformer(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    normalization_stats = checkpoint["normalization_stats"]
    dataset = build_dataset(
        npz_path=args.npz_path,
        horizon=args.length,
        stride=args.stride,
        normalization_stats=normalization_stats,
    )
    if len(dataset) < args.count:
        raise ValueError(f"Requested {args.count} animations, but only {len(dataset)} windows are available.")

    rng = random.Random(args.seed)
    sample_indices = rng.sample(range(len(dataset)), args.count)

    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError("imageio-ffmpeg is required to save rollout MP4 animations.") from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)
    weights = rollout_weights(args.length, float(checkpoint.get("loss_alpha", LOSS_ALPHA)), device)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Dataset windows: {len(dataset)}")

    for output_index, sample_index in enumerate(sample_indices, start=1):
        batch = default_collate([dataset[sample_index]])
        batch = move_batch_to_device(batch, device)
        with torch.no_grad():
            rollout = boundary_conditioned_rollout(
                model=model,
                batch=batch,
                dataset=dataset,
                normalization_stats=normalization_stats,
                weights=weights,
            )

        frame_start = int(batch["frame_start"][0].detach().cpu())
        future_start = frame_start + T_HISTORY
        future_end = future_start + args.length - 1
        output_path = args.output_dir / (
            f"rollout_100_frame_{output_index:02d}_frames_{future_start}_{future_end}.mp4"
        )
        payload = {
            "pred_position": rollout["pred_position"][0].detach().cpu().numpy(),
            "true_position": rollout["true_position"][0].detach().cpu().numpy(),
            "mask": rollout["mask"][0].detach().cpu().numpy(),
            "boundary_mask": rollout["boundary_mask"][0].detach().cpu().numpy(),
            "droplet_ids": batch["droplet_ids"][0].detach().cpu().numpy(),
            "title": f"Rollout frames {future_start}-{future_end}",
        }
        save_rollout_animation(payload, output_path, imageio_ffmpeg.get_ffmpeg_exe(), fps=args.fps)
        print(f"Saved {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create random rollout prediction animations.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--npz-path", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--length", type=int, default=100)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def newest_rollout_checkpoint(model_dir: Path) -> Path:
    checkpoints = sorted(
        model_dir.glob("canonical_rollout_transformer*.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not checkpoints:
        raise FileNotFoundError(f"No rollout checkpoints found in {model_dir}")
    return checkpoints[0]


def build_dataset(npz_path: Path, horizon: int, stride: int, normalization_stats) -> CanonicalWindowDataset:
    data = np.load(npz_path, allow_pickle=False)
    total_frames = len(data["frames"])
    total_window = T_HISTORY + horizon
    start_frames = np.arange(0, total_frames - total_window + 1, stride, dtype=np.int64)
    return CanonicalWindowDataset(
        npz_path=npz_path,
        start_frames=start_frames,
        T_history=T_HISTORY,
        T_future=horizon,
        max_droplets=MAX_DROPLETS,
        normalization_stats=normalization_stats,
    )


def save_rollout_animation(payload, output_path: Path, ffmpeg_path: str, fps: int) -> None:
    pred_position = payload["pred_position"]
    true_position = payload["true_position"]
    mask = payload["mask"].astype(bool)
    boundary_mask = payload["boundary_mask"].astype(bool)

    valid_positions = np.concatenate([pred_position[mask], true_position[mask]], axis=0)
    valid_positions = valid_positions[np.isfinite(valid_positions).all(axis=1)]
    if valid_positions.size == 0:
        print(f"Skipping {output_path} because no valid rollout positions are available.")
        return

    x_min, y_min = np.nanmin(valid_positions, axis=0)
    x_max, y_max = np.nanmax(valid_positions, axis=0)
    x_pad = max((x_max - x_min) * 0.08, 1.0)
    y_pad = max((y_max - y_min) * 0.08, 1.0)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_max + y_pad, y_min - y_pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    gt_scatter = ax.scatter([], [], s=32, c="black", label="ground truth", alpha=0.75)
    pred_scatter = ax.scatter([], [], s=38, c="red", marker="x", label="prediction", alpha=0.85)
    boundary_scatter = ax.scatter([], [], s=52, facecolors="none", edgecolors="tab:blue", label="boundary injected")
    ax.legend(loc="upper right")

    def draw(step_index):
        valid = mask[step_index]
        boundary = boundary_mask[step_index]
        gt = true_position[step_index, valid]
        pred = pred_position[step_index, valid]
        injected = true_position[step_index, boundary]
        gt_scatter.set_offsets(gt if gt.size else np.empty((0, 2)))
        pred_scatter.set_offsets(pred if pred.size else np.empty((0, 2)))
        boundary_scatter.set_offsets(injected if injected.size else np.empty((0, 2)))
        ax.set_title(f"{payload['title']} - step {step_index + 1:03d}")
        return [gt_scatter, pred_scatter, boundary_scatter]

    plt.rcParams["animation.ffmpeg_path"] = ffmpeg_path
    animation = FuncAnimation(fig, draw, frames=np.arange(mask.shape[0]), interval=1000 / fps, blit=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=fps)
    animation.save(output_path, writer=writer)
    plt.close(fig)


if __name__ == "__main__":
    main()
