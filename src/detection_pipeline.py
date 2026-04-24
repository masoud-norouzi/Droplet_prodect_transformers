from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "frame",
    "label",
    "centroid_x",
    "centroid_y",
    "area",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "equivalent_radius",
    "circularity",
    "aspect_ratio",
    "mask_label",
]


@dataclass
class FrameExtractionConfig:
    video_path: Path
    output_dir: Path
    every_n_frames: int = 1
    max_frames: int | None = None
    image_extension: str = ".png"


@dataclass
class ConnectedComponentDetectionConfig:
    threshold: int
    min_area_ratio: float = 0.8
    max_area_ratio: float = 1.2
    min_area: int = 10
    invert_threshold: bool = False
    crop_bottom_px: int = 0


class VideoFrameExtractor:
    def __init__(self, config: FrameExtractionConfig):
        self.config = config

    def extract(self) -> list[Path]:
        video_path = Path(self.config.video_path)
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        saved_paths: list[Path] = []
        frame_index = 0
        saved_count = 0

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if frame_index % self.config.every_n_frames == 0:
                    frame_path = output_dir / (
                        f"frame_{frame_index:06d}{self.config.image_extension}"
                    )
                    if not cv2.imwrite(str(frame_path), frame):
                        raise RuntimeError(f"Could not write frame: {frame_path}")

                    saved_paths.append(frame_path)
                    saved_count += 1

                    if (
                        self.config.max_frames is not None
                        and saved_count >= self.config.max_frames
                    ):
                        break

                frame_index += 1
        finally:
            cap.release()

        return saved_paths


class ConnectedComponentDropletDetector:
    def __init__(self, config: ConnectedComponentDetectionConfig):
        self.config = config

    def detect(self, frame_bgr: np.ndarray, frame_id: int) -> pd.DataFrame:
        frame_bgr = self.crop_frame(frame_bgr)
        gray_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        threshold_type = (
            cv2.THRESH_BINARY_INV
            if self.config.invert_threshold
            else cv2.THRESH_BINARY
        )
        _, thresh = cv2.threshold(
            gray_frame,
            self.config.threshold,
            255,
            threshold_type,
        )

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            thresh,
            connectivity=8,
        )

        component_areas = stats[1:, cv2.CC_STAT_AREA]
        if component_areas.size == 0:
            return self._empty_features()

        median_area = float(np.median(component_areas))
        rows = []

        for label_id in range(1, num_labels):
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if not self._area_is_droplet(area, median_area):
                continue

            bbox_x = int(stats[label_id, cv2.CC_STAT_LEFT])
            bbox_y = int(stats[label_id, cv2.CC_STAT_TOP])
            bbox_w = int(stats[label_id, cv2.CC_STAT_WIDTH])
            bbox_h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
            centroid_x = float(centroids[label_id][0])
            centroid_y = float(centroids[label_id][1])

            component_mask = (labels == label_id).astype(np.uint8) * 255
            circularity = self._calculate_circularity(component_mask, area)
            aspect_ratio = float(bbox_w / bbox_h) if bbox_h else np.nan
            equivalent_radius = float(np.sqrt(area / np.pi))

            rows.append(
                {
                    "frame": frame_id,
                    "label": label_id,
                    "centroid_x": centroid_x,
                    "centroid_y": centroid_y,
                    "area": area,
                    "bbox_x": bbox_x,
                    "bbox_y": bbox_y,
                    "bbox_w": bbox_w,
                    "bbox_h": bbox_h,
                    "equivalent_radius": equivalent_radius,
                    "circularity": circularity,
                    "aspect_ratio": aspect_ratio,
                    "mask_label": label_id,
                }
            )

        return pd.DataFrame(rows, columns=FEATURE_COLUMNS)

    def _area_is_droplet(self, area: int, median_area: float) -> bool:
        return (
            area >= self.config.min_area
            and area > self.config.min_area_ratio * median_area
            and area < self.config.max_area_ratio * median_area
        )

    @staticmethod
    def _calculate_circularity(component_mask: np.ndarray, area: int) -> float:
        contours, _ = cv2.findContours(
            component_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        perimeter = sum(cv2.arcLength(contour, True) for contour in contours)
        if perimeter == 0:
            return np.nan
        return float(4 * np.pi * area / (perimeter**2))

    @staticmethod
    def _empty_features() -> pd.DataFrame:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    def crop_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        if self.config.crop_bottom_px > 0:
            return frame_bgr[: -self.config.crop_bottom_px, :]
        return frame_bgr


class DropletDetectionPipeline:
    def __init__(self, detector: ConnectedComponentDropletDetector):
        self.detector = detector

    def process_single_frame(
        self,
        frame_bgr: np.ndarray,
        frame_id: int,
        output_dir: Path | None = None,
        frame_name: str | None = None,
        save_csv: bool = False,
        save_overlay: bool = False,
    ) -> pd.DataFrame:
        features = self.detector.detect(frame_bgr, frame_id)

        if output_dir is not None and (save_csv or save_overlay):
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        if output_dir is not None and save_csv:
            csv_name = f"{frame_name or f'frame_{frame_id:06d}'}_features.csv"
            features.to_csv(output_dir / csv_name, index=False)

        if output_dir is not None and save_overlay:
            overlay_name = f"{frame_name or f'frame_{frame_id:06d}'}_overlay.png"
            overlay_dir = output_dir / "overlays"
            overlay_dir.mkdir(parents=True, exist_ok=True)
            overlay_frame = self.detector.crop_frame(frame_bgr)
            overlay = self.create_debug_overlay(overlay_frame, features)
            cv2.imwrite(str(overlay_dir / overlay_name), overlay)

        return features

    def process_frame_folder(
        self,
        frame_dir: Path,
        output_dir: Path,
        save_per_frame_csvs: bool = True,
        save_overlays: bool = False,
    ) -> pd.DataFrame:
        frame_paths = self._find_frame_paths(frame_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        all_features = []
        for index, frame_path in enumerate(frame_paths):
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise RuntimeError(f"Could not read frame image: {frame_path}")

            frame_id = self._frame_id_from_path(frame_path, index)
            features = self.process_single_frame(
                frame,
                frame_id=frame_id,
                output_dir=output_dir,
                frame_name=frame_path.stem,
                save_csv=save_per_frame_csvs,
                save_overlay=save_overlays,
            )
            all_features.append(features)

        if all_features:
            combined = pd.concat(all_features, ignore_index=True)
        else:
            combined = pd.DataFrame(columns=FEATURE_COLUMNS)

        combined.to_csv(output_dir / "all_frame_features.csv", index=False)
        return combined

    @staticmethod
    def create_debug_overlay(frame_bgr: np.ndarray, features: pd.DataFrame) -> np.ndarray:
        overlay = frame_bgr.copy()

        for row in features.itertuples(index=False):
            x = int(row.bbox_x)
            y = int(row.bbox_y)
            w = int(row.bbox_w)
            h = int(row.bbox_h)
            cx = int(round(row.centroid_x))
            cy = int(round(row.centroid_y))
            label = str(row.label)

            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(overlay, (cx, cy), 3, (0, 0, 255), -1)
            cv2.putText(
                overlay,
                label,
                (x, max(0, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
                cv2.LINE_AA,
            )

        return overlay

    @staticmethod
    def _find_frame_paths(frame_dir: Path) -> list[Path]:
        frame_dir = Path(frame_dir)
        extensions = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        return sorted(
            path
            for path in frame_dir.iterdir()
            if path.is_file() and path.suffix.lower() in extensions
        )

    @staticmethod
    def _frame_id_from_path(frame_path: Path, fallback: int) -> int:
        digits = "".join(char for char in frame_path.stem if char.isdigit())
        if digits:
            return int(digits)
        return fallback
