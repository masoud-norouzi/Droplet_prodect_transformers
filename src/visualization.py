from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.detection_pipeline import ConnectedComponentDropletDetector
from src.tracking import DropletTracker


def live_detection_preview(
    video_path: str | Path,
    detector: ConnectedComponentDropletDetector,
    tracker: DropletTracker | None = None,
    every_n_frames: int = 1,
    max_frames: int | None = None,
    delay_ms: int = 1,
    window_name: str = "Detection Preview",
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    paused = False
    frame_index = 0
    processed_count = 0

    try:
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    break

                if frame_index % every_n_frames == 0:
                    detection_result = detector.detect_with_debug(frame, frame_index)
                    detections = detection_result.features
                    if tracker is not None:
                        detections = tracker.update(detections)

                    preview_frame = draw_live_preview_frame(
                        original_frame_bgr=frame,
                        threshold_image=detection_result.threshold_image,
                        detections=detections,
                    )
                    cv2.imshow(window_name, preview_frame)

                    processed_count += 1
                    if max_frames is not None and processed_count >= max_frames:
                        break

                frame_index += 1

            key = cv2.waitKey(delay_ms if not paused else 30) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                paused = not paused
    finally:
        cap.release()
        cv2.destroyWindow(window_name)


def show_background_model(
    background_gray,
    window_name: str = "Background Model",
) -> None:
    background_panel = cv2.cvtColor(background_gray, cv2.COLOR_GRAY2BGR)
    background_panel = add_panel_label(background_panel, "Median Background")
    cv2.imshow(window_name, background_panel)
    cv2.waitKey(1)


def visualize_single_track(
    video_path: str,
    tracked_csv_path: str,
    track_id: int,
    delay_ms: int = 1,
    window_name: str = "Single Track Preview",
) -> None:
    tracked = pd.read_csv(tracked_csv_path)
    track_df = tracked[tracked["track_id"] == track_id].copy()
    if track_df.empty:
        raise ValueError(f"No detections found for track_id={track_id}")

    x_col, y_col = _centroid_columns(track_df)
    track_df = track_df.sort_values("frame")
    start_frame = int(track_df["frame"].min())
    end_frame = int(track_df["frame"].max())
    frame_lookup = {
        int(frame): frame_df
        for frame, frame_df in track_df.groupby("frame", sort=True)
    }

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    history: list[tuple[float, float]] = []
    paused = False
    current_frame = start_frame

    try:
        while current_frame <= end_frame:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    break

                display_frame = frame.copy()
                if current_frame in frame_lookup:
                    row = frame_lookup[current_frame].iloc[0]
                    history.append((float(row[x_col]), float(row[y_col])))

                draw_single_track_frame(
                    display_frame,
                    track_id=track_id,
                    frame_id=current_frame,
                    history=history,
                )
                cv2.imshow(window_name, display_frame)
                current_frame += 1

            key = cv2.waitKey(delay_ms if not paused else 30) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                paused = not paused
    finally:
        cap.release()
        cv2.destroyWindow(window_name)


def draw_detection_frame(frame_bgr, detections: pd.DataFrame):
    output = frame_bgr.copy()
    if detections.empty:
        return output

    x_col, y_col = _centroid_columns(detections)
    for _, row in detections.iterrows():
        x = int(round(float(row[x_col])))
        y = int(round(float(row[y_col])))
        track_id = row["track_id"] if "track_id" in detections.columns else None
        color = _track_color(track_id) if not pd.isna(track_id) else (0, 255, 0)

        if {"bbox_x", "bbox_y", "bbox_w", "bbox_h"}.issubset(detections.columns):
            x0 = int(row["bbox_x"])
            y0 = int(row["bbox_y"])
            w = int(row["bbox_w"])
            h = int(row["bbox_h"])
            cv2.rectangle(output, (x0, y0), (x0 + w, y0 + h), color, 1)

        cv2.circle(output, (x, y), 4, color, -1)
        if track_id is not None and not pd.isna(track_id):
            cv2.putText(
                output,
                str(int(track_id)),
                (x + 6, max(0, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

    return output


def draw_live_preview_frame(
    original_frame_bgr,
    threshold_image,
    detections: pd.DataFrame,
):
    left_panel = draw_detection_frame(original_frame_bgr, detections)
    right_panel = cv2.cvtColor(threshold_image, cv2.COLOR_GRAY2BGR)
    right_panel = draw_detection_frame(right_panel, detections)

    left_panel = add_panel_label(left_panel, "Original")
    right_panel = add_panel_label(right_panel, "Threshold + Components")
    left_panel, right_panel = match_panel_heights(left_panel, right_panel)
    return np.hstack([left_panel, right_panel])


def add_panel_label(frame_bgr, label: str):
    output = frame_bgr.copy()
    cv2.rectangle(output, (0, 0), (260, 34), (0, 0, 0), -1)
    cv2.putText(
        output,
        label,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def match_panel_heights(left_panel, right_panel):
    target_height = max(left_panel.shape[0], right_panel.shape[0])
    return (
        pad_to_height(left_panel, target_height),
        pad_to_height(right_panel, target_height),
    )


def pad_to_height(frame_bgr, target_height: int):
    height = frame_bgr.shape[0]
    if height >= target_height:
        return frame_bgr

    pad_height = target_height - height
    padding = np.zeros((pad_height, frame_bgr.shape[1], 3), dtype=frame_bgr.dtype)
    return np.vstack([frame_bgr, padding])


def draw_single_track_frame(
    frame_bgr,
    track_id: int,
    frame_id: int,
    history: list[tuple[float, float]],
) -> None:
    color = _track_color(track_id)
    recent_history = history[-50:]

    for start, end in zip(recent_history, recent_history[1:]):
        start_point = (int(round(start[0])), int(round(start[1])))
        end_point = (int(round(end[0])), int(round(end[1])))
        cv2.line(frame_bgr, start_point, end_point, color, 2)

    if history:
        x, y = history[-1]
        point = (int(round(x)), int(round(y)))
        cv2.circle(frame_bgr, point, 6, color, -1)
        cv2.putText(
            frame_bgr,
            f"track {track_id}",
            (point[0] + 8, max(0, point[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        frame_bgr,
        f"frame {frame_id}",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def _track_color(track_id) -> tuple[int, int, int]:
    color_hash = abs(hash(int(track_id)))
    red = 64 + color_hash % 192
    green = 64 + (color_hash // 192) % 192
    blue = 64 + (color_hash // (192 * 192)) % 192
    return int(blue), int(green), int(red)


def _centroid_columns(features: pd.DataFrame) -> tuple[str, str]:
    if "centroid_x" in features.columns:
        x_col = "centroid_x"
    elif "centroid-1" in features.columns:
        x_col = "centroid-1"
    else:
        raise KeyError("Missing centroid x column: expected centroid_x or centroid-1")

    if "centroid_y" in features.columns:
        y_col = "centroid_y"
    elif "centroid-0" in features.columns:
        y_col = "centroid-0"
    else:
        raise KeyError("Missing centroid y column: expected centroid_y or centroid-0")

    return x_col, y_col
