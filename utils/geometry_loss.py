from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


@dataclass
class GeometryLossConfig:
    sample_count: int = 2000
    random_seed: int = 123
    crop_bottom_px: int = 35
    torch_num_samples_x: int = 64
    torch_num_samples_y: int = 64
    consistency_count: int = 100
    worst_case_count: int = 25


def _validate_mask(channel_mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(channel_mask)
    if mask.ndim != 2:
        raise ValueError(f"channel_mask must be 2D, got shape {mask.shape}.")
    return mask.astype(bool)


def _validate_droplet_values(
    centroid_x: float,
    centroid_y: float,
    bbox_width: float,
    bbox_height: float,
) -> None:
    values = np.asarray([centroid_x, centroid_y, bbox_width, bbox_height], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Droplet values must be finite, got {values}.")
    if bbox_width <= 0 or bbox_height <= 0:
        raise ValueError(f"bbox_width and bbox_height must be positive, got {bbox_width}, {bbox_height}.")


def compute_ellipse_outside_fraction(
    centroid_x: float,
    centroid_y: float,
    bbox_width: float,
    bbox_height: float,
    channel_mask: np.ndarray,
) -> float:
    """Return the fraction of an axis-aligned ellipse footprint outside a channel mask.

    Pixel centers are tested at integer image coordinates. Ellipse pixels outside
    the image bounds count as outside the valid channel.
    """

    _validate_droplet_values(centroid_x, centroid_y, bbox_width, bbox_height)
    mask = _validate_mask(channel_mask)
    height, width = mask.shape
    axis_a = float(bbox_width) / 2.0
    axis_b = float(bbox_height) / 2.0

    x_min = math.floor(float(centroid_x) - axis_a)
    x_max = math.ceil(float(centroid_x) + axis_a)
    y_min = math.floor(float(centroid_y) - axis_b)
    y_max = math.ceil(float(centroid_y) + axis_b)

    xs = np.arange(x_min, x_max + 1, dtype=float)
    ys = np.arange(y_min, y_max + 1, dtype=float)
    grid_x, grid_y = np.meshgrid(xs, ys)
    ellipse = ((grid_x - centroid_x) / axis_a) ** 2 + ((grid_y - centroid_y) / axis_b) ** 2 <= 1.0
    total = int(ellipse.sum())
    if total == 0:
        return 1.0

    in_image = (
        (grid_x >= 0)
        & (grid_x < width)
        & (grid_y >= 0)
        & (grid_y < height)
        & ellipse
    )
    valid_inside = np.zeros_like(ellipse, dtype=bool)
    if np.any(in_image):
        image_x = grid_x[in_image].astype(int)
        image_y = grid_y[in_image].astype(int)
        valid_inside[in_image] = mask[image_y, image_x]
    outside = total - int(valid_inside.sum())
    return float(outside / total)


def compute_ellipse_outside_fraction_batch(
    centroids_x: np.ndarray,
    centroids_y: np.ndarray,
    bbox_widths: np.ndarray,
    bbox_heights: np.ndarray,
    channel_mask: np.ndarray,
) -> np.ndarray:
    """Vector-shaped NumPy wrapper returning one outside fraction per droplet."""

    arrays = [
        np.asarray(centroids_x, dtype=float),
        np.asarray(centroids_y, dtype=float),
        np.asarray(bbox_widths, dtype=float),
        np.asarray(bbox_heights, dtype=float),
    ]
    shape = arrays[0].shape
    if any(array.shape != shape for array in arrays):
        raise ValueError(f"All droplet input arrays must have matching shapes, got {[a.shape for a in arrays]}.")

    output = np.empty(shape, dtype=float)
    for index in np.ndindex(shape):
        output[index] = compute_ellipse_outside_fraction(
            arrays[0][index],
            arrays[1][index],
            arrays[2][index],
            arrays[3][index],
            channel_mask,
        )
    return output


def geometry_penalty(overlap: np.ndarray | float, tolerance: float = 0.0, sharpness: float = 20.0) -> np.ndarray:
    """Smooth thresholded penalty: softplus(sharpness * (overlap - tolerance)) / sharpness."""

    if sharpness <= 0:
        raise ValueError("sharpness must be positive.")
    values = np.asarray(overlap, dtype=float)
    return np.logaddexp(0.0, sharpness * (values - tolerance)) / sharpness


def geometry_penalty_torch(
    overlap: torch.Tensor,
    tolerance: float = 0.0,
    sharpness: float = 20.0,
) -> torch.Tensor:
    if sharpness <= 0:
        raise ValueError("sharpness must be positive.")
    return F.softplus(sharpness * (overlap - tolerance)) / sharpness


def compute_ellipse_outside_fraction_torch(
    centroids: torch.Tensor,
    bbox_sizes: torch.Tensor,
    channel_mask: torch.Tensor,
    num_samples_x: int = 64,
    num_samples_y: int = 64,
) -> torch.Tensor:
    """Differentiable approximate outside fraction for ellipses.

    centroids has shape (..., 2) in image pixel coordinates (x, y).
    bbox_sizes has shape (..., 2) containing width and height in pixels.
    channel_mask is a 2D tensor where 1/True means valid channel.

    grid_sample expects normalized coordinates in [-1, 1] ordered as (x, y).
    With align_corners=True, x_norm = 2 * x / (W - 1) - 1 and similarly for y.
    Sampling outside the image uses zeros, so out-of-frame ellipse area counts
    as outside the channel.
    """

    if centroids.shape[-1] != 2:
        raise ValueError(f"centroids must have shape (..., 2), got {tuple(centroids.shape)}.")
    if bbox_sizes.shape[-1] != 2 or bbox_sizes.shape[:-1] != centroids.shape[:-1]:
        raise ValueError("bbox_sizes must have shape matching centroids, (..., 2).")
    if channel_mask.ndim != 2:
        raise ValueError(f"channel_mask must be 2D, got {tuple(channel_mask.shape)}.")
    if num_samples_x <= 0 or num_samples_y <= 0:
        raise ValueError("num_samples_x and num_samples_y must be positive.")

    device = centroids.device
    dtype = centroids.dtype
    mask = channel_mask.to(device=device, dtype=dtype)
    height, width = mask.shape

    bbox_sizes = bbox_sizes.to(device=device, dtype=dtype)
    if torch.any(~torch.isfinite(centroids)) or torch.any(~torch.isfinite(bbox_sizes)):
        raise ValueError("centroids and bbox_sizes must be finite.")
    if torch.any(bbox_sizes <= 0):
        raise ValueError("bbox_sizes must be positive.")

    sample_x = torch.linspace(-1.0, 1.0, num_samples_x, device=device, dtype=dtype)
    sample_y = torch.linspace(-1.0, 1.0, num_samples_y, device=device, dtype=dtype)
    unit_y, unit_x = torch.meshgrid(sample_y, sample_x, indexing="ij")
    unit_disk = (unit_x.square() + unit_y.square()) <= 1.0
    offsets = torch.stack([unit_x[unit_disk], unit_y[unit_disk]], dim=-1)

    flat_centroids = centroids.reshape(-1, 2)
    flat_sizes = bbox_sizes.reshape(-1, 2)
    axes = flat_sizes / 2.0
    sample_pixels = flat_centroids[:, None, :] + offsets[None, :, :] * axes[:, None, :]

    denom_x = max(width - 1, 1)
    denom_y = max(height - 1, 1)
    normalized_x = 2.0 * sample_pixels[..., 0] / denom_x - 1.0
    normalized_y = 2.0 * sample_pixels[..., 1] / denom_y - 1.0
    grid = torch.stack([normalized_x, normalized_y], dim=-1).reshape(1, -1, 1, 2)
    sampled = F.grid_sample(
        mask.reshape(1, 1, height, width),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).reshape(flat_centroids.shape[0], -1)
    outside_fraction = 1.0 - sampled.mean(dim=1)
    return outside_fraction.reshape(centroids.shape[:-1])


def run_sanity_tests() -> dict[str, Any]:
    all_true = np.ones((100, 100), dtype=bool)
    all_false = np.zeros((100, 100), dtype=bool)
    half_true = np.zeros((100, 100), dtype=bool)
    half_true[:, :50] = True

    inside = compute_ellipse_outside_fraction(30, 30, 10, 10, all_true)
    outside = compute_ellipse_outside_fraction(30, 30, 10, 10, all_false)
    crossing = compute_ellipse_outside_fraction(50, 50, 20, 20, half_true)
    positions = [35, 42, 49, 56, 63]
    monotonic = [compute_ellipse_outside_fraction(x, 50, 20, 20, half_true) for x in positions]

    assert inside <= 1e-12
    assert outside >= 1.0 - 1e-12
    assert 0.0 < crossing < 1.0
    assert all(a <= b + 1e-12 for a, b in zip(monotonic, monotonic[1:]))

    mask_torch = torch.ones((64, 64), dtype=torch.float32)
    centroids = torch.tensor([[20.0, 20.0], [30.0, 30.0]], requires_grad=True)
    sizes = torch.tensor([[10.0, 10.0], [12.0, 8.0]])
    values = compute_ellipse_outside_fraction_torch(centroids, sizes, mask_torch, 24, 24)
    loss = values.sum()
    loss.backward()
    assert centroids.grad is not None
    assert torch.isfinite(centroids.grad).all()

    return {
        "all_true_overlap": inside,
        "all_false_overlap": outside,
        "boundary_crossing_overlap": crossing,
        "monotonic_positions": positions,
        "monotonic_overlaps": monotonic,
        "torch_gradient_finite": True,
    }


def _load_frame(video_path: Path, frame_number: int, crop_bottom_px: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
    success, frame = capture.read()
    capture.release()
    if not success:
        raise RuntimeError(f"Unable to read frame {frame_number} from {video_path}")
    if crop_bottom_px > 0 and crop_bottom_px < frame.shape[0]:
        frame = frame[: -crop_bottom_px, :]
    return frame


def _ellipse_pixel_masks(
    centroid_x: float,
    centroid_y: float,
    bbox_width: float,
    bbox_height: float,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape
    axis_a = bbox_width / 2.0
    axis_b = bbox_height / 2.0
    x_min = max(0, math.floor(centroid_x - axis_a) - 1)
    x_max = min(width - 1, math.ceil(centroid_x + axis_a) + 1)
    y_min = max(0, math.floor(centroid_y - axis_b) - 1)
    y_max = min(height - 1, math.ceil(centroid_y + axis_b) + 1)
    ys = np.arange(y_min, y_max + 1)
    xs = np.arange(x_min, x_max + 1)
    grid_x, grid_y = np.meshgrid(xs, ys)
    local = ((grid_x - centroid_x) / axis_a) ** 2 + ((grid_y - centroid_y) / axis_b) ** 2 <= 1.0
    full = np.zeros(shape, dtype=bool)
    full[y_min : y_max + 1, x_min : x_max + 1] = local
    return full, local


def _draw_case_overlay(frame: np.ndarray, row: pd.Series, channel_mask: np.ndarray) -> np.ndarray:
    output = frame.copy()
    boundary = cv2.Canny((channel_mask.astype(np.uint8) * 255), 50, 150) > 0
    output[boundary] = (255, 0, 0)

    ellipse_mask, _ = _ellipse_pixel_masks(
        float(row["centroid_x"]),
        float(row["centroid_y"]),
        float(row["bbox_w"]),
        float(row["bbox_h"]),
        channel_mask.shape,
    )
    outside = ellipse_mask & ~channel_mask
    color_layer = output.copy()
    color_layer[ellipse_mask] = (0, 220, 255)
    color_layer[outside] = (0, 0, 255)
    output = cv2.addWeighted(color_layer, 0.45, output, 0.55, 0)

    center = (int(round(row["centroid_x"])), int(round(row["centroid_y"])))
    axes = (max(1, int(round(row["bbox_w"] / 2))), max(1, int(round(row["bbox_h"] / 2))))
    cv2.ellipse(output, center, axes, 0, 0, 360, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(output, center, 2, (255, 255, 255), -1)
    track_text = f" track {int(row['track_id'])}" if "track_id" in row and pd.notna(row["track_id"]) else ""
    text = f"f {int(row['frame'])}{track_text} outside={row['outside_fraction']:.3f}"
    cv2.rectangle(output, (0, 0), (min(output.shape[1] - 1, 360), 24), (0, 0, 0), -1)
    cv2.putText(output, text, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def _make_montage(images: list[np.ndarray], columns: int = 5) -> np.ndarray:
    if not images:
        raise ValueError("No images provided for montage.")
    height, width = images[0].shape[:2]
    rows = math.ceil(len(images) / columns)
    montage = np.zeros((rows * height, columns * width, 3), dtype=np.uint8)
    for index, image in enumerate(images):
        row = index // columns
        col = index % columns
        montage[row * height : (row + 1) * height, col * width : (col + 1) * width] = image
    return montage


def _resize_for_montage(image: np.ndarray, max_width: int = 260) -> np.ndarray:
    scale = max_width / image.shape[1]
    return cv2.resize(image, (max_width, int(round(image.shape[0] * scale))), interpolation=cv2.INTER_AREA)


def _select_range_examples(results: pd.DataFrame) -> pd.DataFrame:
    ranges = [
        ("near_zero", 0.0, 0.001),
        ("small", 0.001, 0.01),
        ("moderate", 0.01, 0.05),
        ("large", 0.05, float("inf")),
    ]
    rows = []
    for label, lower, upper in ranges:
        candidates = results[(results["outside_fraction"] >= lower) & (results["outside_fraction"] < upper)]
        if candidates.empty:
            continue
        selected = candidates.iloc[(candidates["outside_fraction"] - lower).abs().argsort().iloc[0]].copy()
        selected["range_label"] = label
        rows.append(selected)
    return pd.DataFrame(rows)


def _summary_statistics(values: np.ndarray) -> dict[str, Any]:
    thresholds = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05]
    return {
        "sample_count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "maximum": float(np.max(values)),
        "above_thresholds": {
            str(threshold): {
                "count": int(np.sum(values > threshold)),
                "fraction": float(np.mean(values > threshold)),
            }
            for threshold in thresholds
        },
    }


def run_validation_diagnostic(
    detections_csv: Path,
    channel_mask_path: Path,
    video_path: Path,
    output_dir: Path,
    config: GeometryLossConfig,
) -> dict[str, Any]:
    channel_mask = np.load(channel_mask_path).astype(bool)
    detections = pd.read_csv(detections_csv)
    required = ["frame", "centroid_x", "centroid_y", "bbox_w", "bbox_h"]
    missing = [column for column in required if column not in detections.columns]
    if missing:
        raise ValueError(f"Missing required detection columns: {missing}")
    detections = detections.dropna(subset=required).copy()
    detections = detections[(detections["bbox_w"] > 0) & (detections["bbox_h"] > 0)]
    sample_count = min(config.sample_count, len(detections))
    sample = detections.sample(n=sample_count, random_state=config.random_seed).sort_values("frame").reset_index(drop=True)

    overlaps = compute_ellipse_outside_fraction_batch(
        sample["centroid_x"].to_numpy(),
        sample["centroid_y"].to_numpy(),
        sample["bbox_w"].to_numpy(),
        sample["bbox_h"].to_numpy(),
        channel_mask,
    )
    sample["outside_fraction"] = overlaps
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "true_droplet_geometry_overlap_sample.csv"
    output_columns = [column for column in ["frame", "track_id", "centroid_x", "centroid_y", "bbox_w", "bbox_h", "outside_fraction"] if column in sample.columns]
    sample[output_columns].to_csv(csv_path, index=False)

    histogram_path = output_dir / "outside_fraction_histogram.png"
    _save_histogram(overlaps, histogram_path)

    worst = sample.sort_values("outside_fraction", ascending=False).head(config.worst_case_count)
    worst_images = [
        _resize_for_montage(_draw_case_overlay(_load_frame(video_path, int(row["frame"]), config.crop_bottom_px), row, channel_mask))
        for _, row in worst.iterrows()
    ]
    worst_montage_path = output_dir / "worst_overlap_montage.png"
    cv2.imwrite(str(worst_montage_path), _make_montage(worst_images))

    examples = _select_range_examples(sample)
    example_montage_path = output_dir / "overlap_range_examples_montage.png"
    if not examples.empty:
        example_images = [
            _resize_for_montage(_draw_case_overlay(_load_frame(video_path, int(row["frame"]), config.crop_bottom_px), row, channel_mask))
            for _, row in examples.iterrows()
        ]
        cv2.imwrite(str(example_montage_path), _make_montage(example_images, columns=max(1, len(example_images))))

    consistency = _compare_numpy_torch(sample.head(config.consistency_count), channel_mask, config)
    sanity = run_sanity_tests()
    summary = _summary_statistics(overlaps)
    report = {
        "config": asdict(config),
        "data_sources": {
            "detections_csv": str(detections_csv),
            "channel_mask_path": str(channel_mask_path),
            "video_path": str(video_path),
            "coordinate_convention": "image coordinates: centroid_x/bbox_w use x columns, centroid_y/bbox_h use y rows on the cropped detection frame",
            "columns_used": output_columns,
        },
        "summary": summary,
        "numpy_torch_consistency": consistency,
        "sanity_tests": sanity,
        "outputs": {
            "csv": str(csv_path),
            "histogram": str(histogram_path),
            "worst_montage": str(worst_montage_path),
            "range_examples_montage": str(example_montage_path) if example_montage_path.exists() else None,
        },
    }
    report_path = output_dir / "geometry_overlap_summary.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    report["outputs"]["summary_json"] = str(report_path)
    return report


def _save_histogram(values: np.ndarray, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(7, 4))
    axis.hist(values, bins=60, color="#2a7fff", edgecolor="black", linewidth=0.4)
    axis.set_xlabel("Outside-overlap fraction")
    axis.set_ylabel("Droplet count")
    axis.set_title("True droplet geometry overlap against channel mask")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return float("nan")
    rank_a = np.argsort(np.argsort(a))
    rank_b = np.argsort(np.argsort(b))
    if np.std(rank_a) == 0 or np.std(rank_b) == 0:
        return float("nan")
    return float(np.corrcoef(rank_a, rank_b)[0, 1])


def _compare_numpy_torch(sample: pd.DataFrame, channel_mask: np.ndarray, config: GeometryLossConfig) -> dict[str, Any]:
    numpy_values = sample["outside_fraction"].to_numpy(dtype=float)
    centroids = torch.tensor(sample[["centroid_x", "centroid_y"]].to_numpy(dtype=np.float32))
    sizes = torch.tensor(sample[["bbox_w", "bbox_h"]].to_numpy(dtype=np.float32))
    mask = torch.tensor(channel_mask.astype(np.float32))
    with torch.no_grad():
        torch_values = compute_ellipse_outside_fraction_torch(
            centroids,
            sizes,
            mask,
            num_samples_x=config.torch_num_samples_x,
            num_samples_y=config.torch_num_samples_y,
        ).cpu().numpy()
    differences = np.abs(numpy_values - torch_values)
    table = []
    for index in range(min(12, len(sample))):
        row = sample.iloc[index]
        table.append(
            {
                "frame": int(row["frame"]),
                "track_id": int(row["track_id"]) if "track_id" in sample.columns and pd.notna(row["track_id"]) else None,
                "numpy": float(numpy_values[index]),
                "torch": float(torch_values[index]),
                "abs_diff": float(differences[index]),
            }
        )
    return {
        "count": int(len(sample)),
        "mean_absolute_difference": float(np.mean(differences)) if len(differences) else float("nan"),
        "maximum_absolute_difference": float(np.max(differences)) if len(differences) else float("nan"),
        "rank_correlation": _rank_correlation(numpy_values, torch_values),
        "comparison_table": table,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute droplet ellipse overlap outside a channel mask.")
    parser.add_argument("--detections-csv", type=Path, default=Path("outputs/processed/2/tracked_features.csv"))
    parser.add_argument("--channel-mask", type=Path, default=Path("outputs/diagnostics/channel_mask_centerline/channel_mask.npy"))
    parser.add_argument("--video-path", type=Path, default=Path(r"D:\Microfluidic loop projct\new loop experiments\confined droplets 2\2.avi"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/diagnostics/geometry_loss_validation"))
    parser.add_argument("--sample-count", type=int, default=2000)
    parser.add_argument("--random-seed", type=int, default=123)
    parser.add_argument("--crop-bottom-px", type=int, default=35)
    parser.add_argument("--torch-num-samples-x", type=int, default=64)
    parser.add_argument("--torch-num-samples-y", type=int, default=64)
    parser.add_argument("--consistency-count", type=int, default=100)
    parser.add_argument("--worst-case-count", type=int, default=25)
    parser.add_argument("--run-tests-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = GeometryLossConfig(
        sample_count=args.sample_count,
        random_seed=args.random_seed,
        crop_bottom_px=args.crop_bottom_px,
        torch_num_samples_x=args.torch_num_samples_x,
        torch_num_samples_y=args.torch_num_samples_y,
        consistency_count=args.consistency_count,
        worst_case_count=args.worst_case_count,
    )
    if args.run_tests_only:
        print(json.dumps(run_sanity_tests(), indent=2))
        return

    report = run_validation_diagnostic(
        detections_csv=args.detections_csv,
        channel_mask_path=args.channel_mask,
        video_path=args.video_path,
        output_dir=args.output_dir,
        config=config,
    )
    summary = report["summary"]
    consistency = report["numpy_torch_consistency"]
    print("Geometry overlap validation complete:")
    print(f"  sample count: {summary['sample_count']}")
    print(f"  mean: {summary['mean']:.6f}")
    print(f"  median: {summary['median']:.6f}")
    print(f"  p90/p95/p99: {summary['p90']:.6f} / {summary['p95']:.6f} / {summary['p99']:.6f}")
    print(f"  maximum: {summary['maximum']:.6f}")
    print(f"  NumPy/Torch mean abs diff: {consistency['mean_absolute_difference']:.6f}")
    print(f"  NumPy/Torch max abs diff: {consistency['maximum_absolute_difference']:.6f}")
    print(f"  NumPy/Torch rank correlation: {consistency['rank_correlation']:.6f}")
    print("Outputs:")
    for name, path in report["outputs"].items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
