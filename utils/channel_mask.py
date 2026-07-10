from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from skimage import measure


Point = tuple[float, float]


@dataclass
class ChannelMaskConfig:
    """Configuration for centerline-based microchannel mask generation.

    channel_half_width_px controls the half-width around the centerline. A value of 12
    produces an approximately 24-pixel-wide channel mask.
    """

    channel_half_width_px: float = 12.0
    crop_bottom_px: int = 35
    x_column: str = "x"
    y_column: str = "y"
    branch_column: str = "channel"


def read_centerline_csv(
    centerline_csv: str | Path,
    x_column: str = "x",
    y_column: str = "y",
    branch_column: str = "channel",
) -> tuple[dict[str, list[Point]], dict[str, Any]]:
    """Read centerline points grouped by branch while preserving CSV order."""

    path = Path(centerline_csv)
    if not path.exists():
        raise FileNotFoundError(f"Centerline CSV does not exist: {path}")

    branches: dict[str, list[Point]] = {}
    all_x: list[float] = []
    all_y: list[float] = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Centerline CSV has no header: {path}")
        missing = [name for name in (x_column, y_column, branch_column) if name not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing required centerline column(s): {missing}; found {reader.fieldnames}")

        for row_index, row in enumerate(reader, start=2):
            try:
                x = float(row[x_column])
                y = float(row[y_column])
            except ValueError as exc:
                raise ValueError(f"Invalid x/y value at CSV line {row_index}: {row}") from exc
            branch = str(row[branch_column])
            branches.setdefault(branch, []).append((x, y))
            all_x.append(x)
            all_y.append(y)

    if not all_x:
        raise ValueError(f"Centerline CSV contains no points: {path}")

    metadata = {
        "centerline_csv": str(path),
        "columns": {
            "x": x_column,
            "y": y_column,
            "branch": branch_column,
        },
        "coordinate_convention": "image coordinates: x is column index, y is row index, origin at top-left",
        "point_count": len(all_x),
        "branch_names": list(branches.keys()),
        "branch_counts": {name: len(points) for name, points in branches.items()},
        "x_min": float(min(all_x)),
        "x_max": float(max(all_x)),
        "y_min": float(min(all_y)),
        "y_max": float(max(all_y)),
    }
    return branches, metadata


def load_reference_image(
    video_path: str | Path | None = None,
    background_path: str | Path | None = None,
    crop_bottom_px: int = 35,
) -> tuple[np.ndarray | None, tuple[int, int] | None, dict[str, Any]]:
    """Load a representative frame/background and return image plus (height, width)."""

    if background_path is not None:
        path = Path(background_path)
        if not path.exists():
            raise FileNotFoundError(f"Background path does not exist: {path}")
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Unable to read background image: {path}")
        if crop_bottom_px > 0 and crop_bottom_px < image.shape[0]:
            image = image[: -crop_bottom_px, :]
        return image, image.shape[:2], {"background_path": str(path)}

    if video_path is not None:
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video path does not exist: {path}")
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video: {path}")
        success, frame = capture.read()
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.release()
        if not success:
            raise RuntimeError(f"Unable to read first frame from video: {path}")
        if crop_bottom_px > 0 and crop_bottom_px < frame.shape[0]:
            frame = frame[: -crop_bottom_px, :]
        return frame, frame.shape[:2], {
            "video_path": str(path),
            "video_frame_count": frame_count,
            "video_width": frame_width,
            "video_height": frame_height,
            "crop_bottom_px": crop_bottom_px,
        }

    return None, None, {}


def infer_shape_from_centerline(
    branches: dict[str, list[Point]],
    channel_half_width_px: float,
) -> tuple[int, int]:
    """Infer the smallest image shape that contains the expanded centerline."""

    all_points = [point for points in branches.values() for point in points]
    max_x = max(x for x, _ in all_points)
    max_y = max(y for _, y in all_points)
    pad = int(np.ceil(channel_half_width_px)) + 1
    return int(np.ceil(max_y)) + pad + 1, int(np.ceil(max_x)) + pad + 1


def rasterize_centerline(
    branches: dict[str, list[Point]],
    shape: tuple[int, int],
) -> np.ndarray:
    """Rasterize branch polylines into a one-pixel centerline image."""

    centerline = np.zeros(shape, dtype=np.uint8)
    height, width = shape
    for points in branches.values():
        if not points:
            continue
        rounded = np.rint(np.asarray(points, dtype=np.float32)).astype(np.int32)
        rounded[:, 0] = np.clip(rounded[:, 0], 0, width - 1)
        rounded[:, 1] = np.clip(rounded[:, 1], 0, height - 1)
        if len(rounded) == 1:
            x, y = rounded[0]
            centerline[y, x] = 255
            continue
        cv2.polylines(centerline, [rounded.reshape(-1, 1, 2)], isClosed=False, color=255, thickness=1)
        for x, y in rounded:
            centerline[y, x] = 255
    return centerline


def create_mask_from_centerline(
    branches: dict[str, list[Point]],
    shape: tuple[int, int],
    channel_half_width_px: float = 12.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand centerline by Euclidean distance threshold to create a channel mask."""

    if channel_half_width_px <= 0:
        raise ValueError(f"channel_half_width_px must be positive, got {channel_half_width_px}.")
    centerline = rasterize_centerline(branches, shape)
    distance_to_centerline = cv2.distanceTransform(255 - centerline, cv2.DIST_L2, 5)
    mask = distance_to_centerline <= float(channel_half_width_px)
    return mask.astype(bool), centerline, distance_to_centerline


def create_overlay(reference_image: np.ndarray | None, mask: np.ndarray, centerline: np.ndarray) -> np.ndarray:
    """Overlay the channel mask and centerline on a representative image."""

    if reference_image is None:
        base = np.zeros((*mask.shape, 3), dtype=np.uint8)
    elif reference_image.ndim == 2:
        base = cv2.cvtColor(reference_image, cv2.COLOR_GRAY2BGR)
    else:
        base = reference_image.copy()

    color_layer = base.copy()
    color_layer[mask] = (0, 180, 255)
    overlay = cv2.addWeighted(color_layer, 0.45, base, 0.55, 0)
    overlay[centerline > 0] = (0, 0, 255)
    return overlay


def validate_mask(mask: np.ndarray) -> dict[str, Any]:
    """Return connected-component and area diagnostics for the generated mask."""

    labels = measure.label(mask, connectivity=2)
    regions = measure.regionprops(labels)
    component_areas = sorted((int(region.area) for region in regions), reverse=True)
    return {
        "mask_area_pixels": int(mask.sum()),
        "mask_fraction": float(mask.sum() / mask.size),
        "connected_component_count": int(labels.max()),
        "component_areas": component_areas,
        "has_gaps_or_disconnected_regions": int(labels.max()) != 1,
    }


def create_channel_mask(
    centerline_csv: str | Path,
    video_path: str | Path | None = None,
    background_path: str | Path | None = None,
    config: ChannelMaskConfig | None = None,
    image_shape: tuple[int, int] | None = None,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Create a boolean channel mask from a centerline CSV."""

    cfg = config or ChannelMaskConfig()
    branches, centerline_metadata = read_centerline_csv(
        centerline_csv,
        x_column=cfg.x_column,
        y_column=cfg.y_column,
        branch_column=cfg.branch_column,
    )
    reference_image, reference_shape, reference_metadata = load_reference_image(
        video_path=video_path,
        background_path=background_path,
        crop_bottom_px=cfg.crop_bottom_px,
    )
    shape = image_shape or reference_shape or infer_shape_from_centerline(branches, cfg.channel_half_width_px)
    mask, centerline, distance_to_centerline = create_mask_from_centerline(
        branches,
        shape,
        cfg.channel_half_width_px,
    )
    overlay = create_overlay(reference_image, mask, centerline)
    validation = validate_mask(mask)

    diagnostics = {
        "centerline": centerline,
        "distance_to_centerline": distance_to_centerline,
        "channel_mask": mask.astype(np.uint8) * 255,
        "channel_mask_overlay": overlay,
        "metadata": {
            "config": asdict(cfg),
            "centerline": centerline_metadata,
            "reference": reference_metadata,
            "image_height": int(shape[0]),
            "image_width": int(shape[1]),
            "validation": validation,
        },
    }
    if return_diagnostics:
        return mask, diagnostics
    return mask


def save_outputs(
    output_dir: str | Path,
    mask: np.ndarray,
    diagnostics: dict[str, Any],
) -> dict[str, str]:
    """Save the required centerline-mask outputs and a concise report."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "channel_mask_npy": out / "channel_mask.npy",
        "channel_mask_png": out / "channel_mask.png",
        "channel_mask_overlay": out / "channel_mask_overlay.png",
        "centerline_png": out / "centerline.png",
        "report": out / "channel_mask_report.json",
    }
    np.save(paths["channel_mask_npy"], mask.astype(bool))
    cv2.imwrite(str(paths["channel_mask_png"]), diagnostics["channel_mask"])
    cv2.imwrite(str(paths["channel_mask_overlay"]), diagnostics["channel_mask_overlay"])
    cv2.imwrite(str(paths["centerline_png"]), diagnostics["centerline"])
    report = diagnostics["metadata"].copy()
    report["saved_outputs"] = {name: str(path) for name, path in paths.items()}
    with paths["report"].open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return {name: str(path) for name, path in paths.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a channel mask from centerlines.csv.")
    parser.add_argument("--centerline-csv", type=Path, required=True)
    parser.add_argument("--video-path", type=Path, default=None)
    parser.add_argument("--background-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/diagnostics/channel_mask_centerline"))
    parser.add_argument("--channel-half-width-px", type=float, default=12.0)
    parser.add_argument(
        "--half-width-px",
        type=float,
        default=None,
        help="Deprecated alias for --channel-half-width-px.",
    )
    parser.add_argument("--crop-bottom-px", type=int, default=35)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.height is None) != (args.width is None):
        raise ValueError("--height and --width must be provided together.")

    channel_half_width_px = (
        args.half_width_px if args.half_width_px is not None else args.channel_half_width_px
    )
    config = ChannelMaskConfig(
        channel_half_width_px=channel_half_width_px,
        crop_bottom_px=args.crop_bottom_px,
    )
    image_shape = (args.height, args.width) if args.height is not None else None
    mask, diagnostics = create_channel_mask(
        centerline_csv=args.centerline_csv,
        video_path=args.video_path,
        background_path=args.background_path,
        config=config,
        image_shape=image_shape,
        return_diagnostics=True,
    )
    saved = save_outputs(args.output_dir, mask, diagnostics)
    metadata = diagnostics["metadata"]
    validation = metadata["validation"]

    print("Centerline channel mask saved:")
    for name, path in saved.items():
        print(f"  {name}: {path}")
    print("Diagnostic summary:")
    print(f"  CSV columns: {metadata['centerline']['columns']}")
    print(f"  coordinate convention: {metadata['centerline']['coordinate_convention']}")
    print(f"  frame dimensions: {metadata['image_width']} x {metadata['image_height']}")
    print(f"  channel half-width px: {metadata['config']['channel_half_width_px']}")
    print(f"  mask area pixels: {validation['mask_area_pixels']}")
    print(f"  mask fraction: {validation['mask_fraction']:.6f}")
    print(f"  connected components: {validation['connected_component_count']}")
    print(f"  component areas: {validation['component_areas']}")
    print(f"  gaps/disconnected regions found: {validation['has_gaps_or_disconnected_regions']}")


if __name__ == "__main__":
    main()
