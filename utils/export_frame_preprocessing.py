from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from skimage import measure


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "droplet-detection-tracking"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from droplet_detection_tracking.configs.settings import DEFAULT_CONFIG
from droplet_detection_tracking.detection.pipeline import (
    BackgroundModel,
    ConnectedComponentDropletDetector,
)


def main() -> None:
    args = parse_args()
    video_path = args.video_path
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config = DEFAULT_CONFIG
    detector = ConnectedComponentDropletDetector(config.detection)

    background_gray = BackgroundModel(video_path, config.background).build()
    frame_bgr = read_frame(video_path, args.frame)
    result = detector.detect(frame_bgr, args.frame, background_gray)

    cropped_frame = result.cropped_frame
    gray_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_frame, background_gray)
    blurred_diff = cv2.GaussianBlur(diff, (5, 5), 0)
    _, binary = cv2.threshold(
        blurred_diff,
        config.detection.background_threshold,
        255,
        cv2.THRESH_BINARY,
    )

    binary_bool = binary > 0
    filled_bool = detector._fill_holes_with_temporary_border(binary_bool)
    cleaned_bool = detector._remove_small_objects(filled_bool, config.detection.min_object_area)
    hole_markers = detector._build_prefill_hole_markers(binary_bool, filled_bool, cleaned_bool)
    distance = cv2.distanceTransform(cleaned_bool.astype(np.uint8), cv2.DIST_L2, 5)

    save_image(output_dir / "01_raw_frame_bgr.png", frame_bgr)
    save_image(output_dir / "02_cropped_frame_bgr.png", cropped_frame)
    save_image(output_dir / "03_gray_frame.png", gray_frame)
    save_image(output_dir / "04_background_gray.png", background_gray)
    save_image(output_dir / "05_absdiff_gray.png", diff)
    save_image(output_dir / "06_absdiff_blurred.png", blurred_diff)
    save_image(output_dir / "07_threshold_binary.png", binary)
    save_image(output_dir / "08_filled_holes_mask.png", filled_bool.astype(np.uint8) * 255)
    save_image(output_dir / "09_cleaned_mask.png", cleaned_bool.astype(np.uint8) * 255)
    save_image(output_dir / "10_prefill_hole_markers.png", colorize_labels(hole_markers))
    save_image(output_dir / "11_distance_transform.png", normalize_to_uint8(distance))
    save_image(output_dir / "12_watershed_labels.png", colorize_labels(result.labels))
    save_image(output_dir / "13_debug_detections.png", result.debug_frame)

    np.save(output_dir / "prefill_hole_markers.npy", hole_markers.astype(np.int32))
    np.save(output_dir / "watershed_labels.npy", result.labels.astype(np.int32))
    result.detections.to_csv(output_dir / "detections.csv", index=False)

    metadata = {
        "frame": args.frame,
        "video_path": str(video_path),
        "output_dir": str(output_dir),
        "background": {
            "crop_bottom_px": config.background.crop_bottom_px,
            "sample_every_n_frames": config.background.sample_every_n_frames,
            "max_background_frames": config.background.max_background_frames,
            "method": config.background.method,
            "percentile": config.background.percentile,
        },
        "detection": {
            "crop_bottom_px": config.detection.crop_bottom_px,
            "background_threshold": config.detection.background_threshold,
            "fill_holes": config.detection.fill_holes,
            "min_object_area": config.detection.min_object_area,
            "use_watershed_split": config.detection.use_watershed_split,
            "use_prefill_hole_markers": config.detection.use_prefill_hole_markers,
            "prefill_hole_marker_min_area": config.detection.prefill_hole_marker_min_area,
            "prefill_hole_marker_max_area": config.detection.prefill_hole_marker_max_area,
            "watershed_min_distance": config.detection.watershed_min_distance,
            "fallback_watershed_min_distance": config.detection.fallback_watershed_min_distance,
            "merged_area_ratio": config.detection.merged_area_ratio,
            "max_split_objects": config.detection.max_split_objects,
        },
        "detections": int(len(result.detections)),
        "prefill_hole_markers": int(hole_markers.max()),
        "watershed_labels": int(result.labels.max()),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved preprocessing outputs to {output_dir}")
    print(f"detections: {len(result.detections)}")
    print(f"prefill_hole_markers: {int(hole_markers.max())}")
    print(f"watershed_labels: {int(result.labels.max())}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export preprocessing intermediates for one frame.")
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument(
        "--video-path",
        type=Path,
        default=DEFAULT_CONFIG.input.raw_video_dir / DEFAULT_CONFIG.input.video_file_name,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "processed" / "2" / "frame_0529_preprocessing",
    )
    return parser.parse_args()


def read_frame(video_path: Path, frame_id: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    success, frame = capture.read()
    capture.release()
    if not success:
        raise RuntimeError(f"Unable to read frame {frame_id} from {video_path}")
    return frame


def save_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to write image: {path}")


def normalize_to_uint8(values: np.ndarray) -> np.ndarray:
    if values.size == 0 or float(np.nanmax(values)) <= 0:
        return np.zeros(values.shape, dtype=np.uint8)
    normalized = values.astype(np.float32) / float(np.nanmax(values))
    return np.clip(normalized * 255, 0, 255).astype(np.uint8)


def colorize_labels(labels: np.ndarray) -> np.ndarray:
    color = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for region in measure.regionprops(labels.astype(np.int32)):
        label = int(region.label)
        rng = np.random.default_rng(label)
        region_color = rng.integers(40, 256, size=3, dtype=np.uint8)
        color[labels == label] = region_color
    return color


if __name__ == "__main__":
    main()
