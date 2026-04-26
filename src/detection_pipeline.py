from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
from typing import List, Optional

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import binary_fill_holes
from skimage import feature, measure, segmentation


@dataclass
class FrameExtractionConfig:
    every_n_frames: int = 1
    max_frames: int | None = None
    start_frame: int = 0


@dataclass
class ConnectedComponentDetectionConfig:
    min_area_ratio: float = 0.8
    max_area_ratio: float = 1.2
    min_object_area: int = 10
    crop_bottom_px: int = 0
    background_threshold: int = 0
    background_method: str = "median"
    background_percentile: int = 50
    fill_holes: bool = True
    use_watershed_split: bool = False
    watershed_min_distance: int = 5
    fallback_watershed_min_distance: int = 3
    merged_area_ratio: float = 1.4
    max_split_objects: int = 4


@dataclass
class DetectionResult:
    features: pd.DataFrame
    binary_image: np.ndarray
    filled_image: np.ndarray
    label_image: np.ndarray
    debug_frame: np.ndarray
    cropped_frame: np.ndarray


class BackgroundModel:
    def __init__(
        self,
        video_path: Path,
        crop_bottom_px: int = 0,
        sample_every_n_frames: int = 1,
        max_background_frames: int | None = None,
        method: str = "median",
        percentile_value: int = 50,
    ):
        self.video_path = video_path
        self.crop_bottom_px = crop_bottom_px
        self.sample_every_n_frames = sample_every_n_frames
        self.max_background_frames = max_background_frames
        self.method = method
        self.percentile_value = percentile_value

    def build(self) -> np.ndarray:
        capture = cv2.VideoCapture(str(self.video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video: {self.video_path}")

        frames: List[np.ndarray] = []
        frame_index = 0

        while True:
            success, frame = capture.read()
            if not success:
                break

            if frame_index % self.sample_every_n_frames != 0:
                frame_index += 1
                continue

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self.crop_bottom_px > 0 and self.crop_bottom_px < gray_frame.shape[0]:
                gray_frame = gray_frame[: -self.crop_bottom_px, :]

            frames.append(gray_frame)
            if self.max_background_frames is not None and len(frames) >= self.max_background_frames:
                break

            frame_index += 1

        capture.release()

        if not frames:
            raise RuntimeError("No background frames were captured for background model building.")

        stack = np.stack(frames, axis=0)
        if self.method == "percentile":
            background = np.percentile(stack, self.percentile_value, axis=0)
        elif self.method == "median":
            background = np.median(stack, axis=0)
        else:
            raise ValueError(f"Unsupported background method: {self.method}")

        return background.astype(np.uint8)


class VideoFrameExtractor:
    def __init__(self, config: FrameExtractionConfig):
        self.config = config

    def extract_frames(self, video_path: Path, output_dir: Path) -> List[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video: {video_path}")

        saved_frames: List[Path] = []
        frame_index = 0

        while True:
            success, frame = capture.read()
            if not success:
                break

            if frame_index < self.config.start_frame:
                frame_index += 1
                continue

            if self.config.max_frames is not None and len(saved_frames) >= self.config.max_frames:
                break

            if (frame_index - self.config.start_frame) % self.config.every_n_frames == 0:
                frame_name = f"frame_{len(saved_frames) + 1:05d}.png"
                frame_path = output_dir / frame_name
                if not cv2.imwrite(str(frame_path), frame):
                    raise RuntimeError(f"Unable to write frame image: {frame_path}")
                saved_frames.append(frame_path)

            frame_index += 1

        capture.release()
        return saved_frames


class ConnectedComponentDropletDetector:
    def __init__(self, config: ConnectedComponentDetectionConfig):
        self.config = config

    def detect(
        self,
        frame_bgr: np.ndarray,
        frame_id: int,
        background_gray: np.ndarray,
    ) -> DetectionResult:
        cropped_frame = self._crop_frame(frame_bgr)
        gray_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)

        if background_gray is None:
            raise ValueError("Background image is required for background-based detection.")

        diff = cv2.absdiff(gray_frame, background_gray)
        diff = cv2.GaussianBlur(diff, (5, 5), 0)
        _, binary = cv2.threshold(
            diff,
            self.config.background_threshold,
            255,
            cv2.THRESH_BINARY,
        )

        binary_bool = binary > 0
        filled_bool = binary_bool
        if self.config.fill_holes:
            filled_bool = binary_fill_holes(binary_bool)

        cleaned_bool = self._remove_small_objects(filled_bool, self.config.min_object_area)
        if self.config.use_watershed_split:
            label_image = self._segment_with_watershed(cleaned_bool)
        else:
            label_image = measure.label(cleaned_bool)

        features = self._extract_features(label_image, frame_id)
        debug_frame = self._render_debug_frame(cropped_frame, features)

        return DetectionResult(
            features=features,
            binary_image=binary,
            filled_image=(cleaned_bool.astype(np.uint8) * 255),
            label_image=label_image.astype(np.int32),
            debug_frame=debug_frame,
            cropped_frame=cropped_frame,
        )

    def _crop_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        if self.config.crop_bottom_px > 0 and self.config.crop_bottom_px < frame_bgr.shape[0]:
            return frame_bgr[: -self.config.crop_bottom_px, :]
        return frame_bgr

    def _remove_small_objects(self, mask_bool: np.ndarray, min_area: int) -> np.ndarray:
        labeled = measure.label(mask_bool)
        if labeled.max() == 0:
            return np.zeros_like(mask_bool, dtype=bool)

        cleaned = np.zeros_like(mask_bool, dtype=bool)
        for region in measure.regionprops(labeled):
            if region.area >= min_area:
                cleaned[labeled == region.label] = True
        return cleaned

    def _segment_with_watershed(self, mask_bool: np.ndarray) -> np.ndarray:
        if not np.any(mask_bool):
            return measure.label(mask_bool)

        dist = cv2.distanceTransform(mask_bool.astype(np.uint8), cv2.DIST_L2, 5)
        marker_mask = mask_bool.astype(np.uint8)
        coordinates = feature.peak_local_max(
            dist,
            min_distance=self.config.watershed_min_distance,
            labels=marker_mask,
        )

        markers = np.zeros(dist.shape, dtype=np.int32)
        for marker_id, (row, col) in enumerate(coordinates, start=1):
            markers[row, col] = marker_id

        if markers.max() == 0:
            label_image = self._segment_with_watershed_fallback(dist, mask_bool)
        else:
            label_image = segmentation.watershed(-dist, markers, mask=mask_bool)
            if label_image.max() <= 1:
                label_image = self._segment_with_watershed_fallback(dist, mask_bool)

        return self._apply_watershed_fallback_per_object(label_image, dist)

    def _segment_with_watershed_fallback(self, dist: np.ndarray, mask_bool: np.ndarray) -> np.ndarray:
        fallback_distance = max(1, self.config.fallback_watershed_min_distance)
        label_image = self._watershed_local_split(dist, mask_bool, fallback_distance)
        if label_image.max() <= 1:
            return measure.label(mask_bool)
        return label_image

    def _apply_watershed_fallback_per_object(self, label_image: np.ndarray, dist: np.ndarray) -> np.ndarray:
        regions = list(measure.regionprops(label_image))
        if not regions:
            return label_image

        areas = [region.area for region in regions]
        median_area = float(np.median(areas))
        if median_area <= 0:
            return label_image

        next_label = int(label_image.max())
        output_labels = label_image.copy()

        for region in regions:
            if region.area <= self.config.merged_area_ratio * median_area:
                continue

            object_mask = label_image == region.label
            if not np.any(object_mask):
                continue

            local_labels = self._watershed_local_split(dist, object_mask, self.config.fallback_watershed_min_distance)
            n_splits = int(local_labels.max())
            if n_splits <= 1 or n_splits > self.config.max_split_objects:
                continue

            output_labels[object_mask] = 0
            local_labels = np.where(local_labels > 0, local_labels + next_label, 0)
            output_labels = np.where(local_labels > 0, local_labels, output_labels)
            next_label = int(output_labels.max())

        return output_labels

    def _watershed_local_split(
        self,
        dist: np.ndarray,
        mask_bool: np.ndarray,
        min_distance: int,
    ) -> np.ndarray:
        if not np.any(mask_bool):
            return np.zeros_like(mask_bool, dtype=np.int32)

        local_dist = dist.copy()
        local_dist[~mask_bool] = 0
        coordinates = feature.peak_local_max(
            local_dist,
            min_distance=min_distance,
            labels=mask_bool.astype(np.uint8),
        )

        markers = np.zeros(dist.shape, dtype=np.int32)
        for marker_id, (row, col) in enumerate(coordinates, start=1):
            markers[row, col] = marker_id

        if markers.max() == 0:
            return np.zeros_like(dist, dtype=np.int32)

        return segmentation.watershed(-local_dist, markers, mask=mask_bool)

    def _extract_features(self, label_image: np.ndarray, frame_id: int) -> pd.DataFrame:
        rows: List[dict[str, float | int]] = []
        for region in measure.regionprops(label_image):
            if region.area == 0:
                continue

            minr, minc, maxr, maxc = region.bbox
            width = maxc - minc
            height = maxr - minr
            area = int(region.area)
            centroid_y, centroid_x = region.centroid
            perimeter = float(region.perimeter or 0.0)
            circularity = float(4 * math.pi * area / perimeter**2) if perimeter > 0 else float("nan")
            aspect_ratio = float(width / height) if height > 0 else float("nan")
            equivalent_radius = math.sqrt(area / math.pi) if area > 0 else 0.0
            solidity = float(region.solidity) if hasattr(region, "solidity") else float("nan")

            rows.append(
                {
                    "frame": frame_id,
                    "label": int(region.label),
                    "centroid_x": float(centroid_x),
                    "centroid_y": float(centroid_y),
                    "area": area,
                    "bbox_x": int(minc),
                    "bbox_y": int(minr),
                    "bbox_w": int(width),
                    "bbox_h": int(height),
                    "equivalent_radius": equivalent_radius,
                    "perimeter": perimeter,
                    "circularity": circularity,
                    "aspect_ratio": aspect_ratio,
                    "solidity": solidity,
                }
            )

        return pd.DataFrame(rows)

    def _render_debug_frame(self, frame_bgr: np.ndarray, features: pd.DataFrame) -> np.ndarray:
        debug_frame = frame_bgr.copy()
        for _, row in features.iterrows():
            x = int(row["bbox_x"])
            y = int(row["bbox_y"])
            w = int(row["bbox_w"])
            h = int(row["bbox_h"])
            centroid = (int(row["centroid_x"]), int(row["centroid_y"]))
            label_text = str(int(row["label"]))

            cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(debug_frame, centroid, 3, (0, 0, 255), -1)
            cv2.putText(
                debug_frame,
                label_text,
                (x, max(0, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return debug_frame
