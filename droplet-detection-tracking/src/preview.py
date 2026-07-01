from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from src import schema


def make_preview_frame(
    cropped_frame: np.ndarray,
    processed_image: np.ndarray,
    tracked_features: pd.DataFrame,
) -> np.ndarray:
    left = draw_tracked_frame(cropped_frame, tracked_features)
    left = add_panel_label(left, "Original + Tracks")
    right = draw_tracked_mask(processed_image, tracked_features)
    right = add_panel_label(right, "Processed Filled + Tracks")

    if left.shape[0] != right.shape[0]:
        height = max(left.shape[0], right.shape[0])
        left = pad_to_height(left, height)
        right = pad_to_height(right, height)

    return np.hstack([left, right])


def draw_tracked_frame(frame_bgr: np.ndarray, tracked_features: pd.DataFrame) -> np.ndarray:
    output = frame_bgr.copy()
    return draw_track_markers(output, tracked_features)


def draw_tracked_mask(processed_image: np.ndarray, tracked_features: pd.DataFrame) -> np.ndarray:
    output = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)
    return draw_track_markers(output, tracked_features)


def draw_track_markers(frame_bgr: np.ndarray, tracked_features: pd.DataFrame) -> np.ndarray:
    if tracked_features.empty:
        return frame_bgr

    for _, row in tracked_features.iterrows():
        track_id = row.get(schema.TRACK_ID)
        if pd.isna(track_id):
            continue

        centroid = (int(row[schema.CENTROID_X]), int(row[schema.CENTROID_Y]))
        cv2.circle(frame_bgr, centroid, 4, (0, 0, 255), -1)
        cv2.putText(
            frame_bgr,
            str(int(track_id)),
            (centroid[0] + 8, max(centroid[1] - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return frame_bgr


def add_panel_label(frame_bgr: np.ndarray, label: str) -> np.ndarray:
    panel = frame_bgr.copy()
    cv2.rectangle(panel, (0, 0), (260, 28), (0, 0, 0), -1)
    cv2.putText(
        panel,
        label,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return panel


def pad_to_height(frame_bgr: np.ndarray, target_height: int) -> np.ndarray:
    height = frame_bgr.shape[0]
    if height >= target_height:
        return frame_bgr
    padding = np.zeros((target_height - height, frame_bgr.shape[1], 3), dtype=frame_bgr.dtype)
    return np.vstack([frame_bgr, padding])
