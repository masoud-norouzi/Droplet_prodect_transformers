from __future__ import annotations
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import cv2
import numpy as np
import pandas as pd

from configs.constants import (
    BACKGROUND_METHOD,
    BACKGROUND_PERCENTILE,
    BACKGROUND_SAMPLE_EVERY_N_FRAMES,
    BACKGROUND_THRESHOLD,
    CROP_BOTTOM_PX,
    EXPERIMENT_NAME,
    FALLBACK_WATERSHED_MIN_DISTANCE,
    FILL_HOLES,
    MAX_BACKGROUND_FRAMES,
    MAX_ASSIGNMENT_DISTANCE,
    MAX_MISSED,
    MAX_SPLIT_OBJECTS,
    MERGED_AREA_RATIO,
    MIN_OBJECT_AREA,
    RUN_FULL_CSV_EXPORT,
    RUN_LIVE_PREVIEW,
    USE_WATERSHED_SPLIT,
    VIDEO_PLAYBACK_DELAY_MS,
    WATERSHED_MIN_DISTANCE,
)
from configs.paths import PROCESSED_DIR, RAW_VIDEO_DIR
from src.detection_pipeline import (
    BackgroundModel,
    ConnectedComponentDetectionConfig,
    ConnectedComponentDropletDetector,
    DetectionResult,
)
from src.tracking import DropletTracker


def find_first_video(raw_video_dir: Path) -> Path:
    supported_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv"]
    for pattern in supported_extensions:
        candidates = sorted(raw_video_dir.glob(pattern))
        if candidates:
            return candidates[0]
    raise FileNotFoundError(f"No supported video file found in {raw_video_dir}")


def build_background_model(video_path: Path) -> np.ndarray:
    background_model = BackgroundModel(
        video_path=video_path,
        crop_bottom_px=CROP_BOTTOM_PX,
        sample_every_n_frames=BACKGROUND_SAMPLE_EVERY_N_FRAMES,
        max_background_frames=MAX_BACKGROUND_FRAMES,
        method=BACKGROUND_METHOD,
        percentile_value=BACKGROUND_PERCENTILE,
    )
    return background_model.build()


def update_track_history(tracked_features: pd.DataFrame, track_history: dict[int, list[tuple[int, int]]]) -> None:
    x_col = "centroid_x"
    y_col = "centroid_y"

    for _, row in tracked_features.iterrows():
        track_id = row.get("track_id")
        if pd.isna(track_id):
            continue
        track_id = int(track_id)
        centroid = (int(row[x_col]), int(row[y_col]))
        history = track_history.setdefault(track_id, [])
        history.append(centroid)


def draw_left_panel(frame_bgr: np.ndarray, tracked_features: pd.DataFrame, track_history: dict[int, list[tuple[int, int]]]) -> np.ndarray:
    output = frame_bgr.copy()
    if tracked_features.empty:
        return output

    for _, row in tracked_features.iterrows():
        track_id = row.get("track_id")
        if pd.isna(track_id):
            continue
        track_id = int(track_id)
        centroid = (int(row["centroid_x"] ), int(row["centroid_y"]))

        cv2.circle(output, centroid, 4, (0, 0, 255), -1)
        cv2.putText(
            output,
            str(track_id),
            (centroid[0] + 8, max(centroid[1] - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return output


def draw_right_panel(processed_image: np.ndarray, tracked_features: pd.DataFrame, track_history: dict[int, list[tuple[int, int]]]) -> np.ndarray:
    output = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)
    if tracked_features.empty:
        return output

    for _, row in tracked_features.iterrows():
        track_id = row.get("track_id")
        if pd.isna(track_id):
            continue
        track_id = int(track_id)
        centroid = (int(row["centroid_x"] ), int(row["centroid_y"]))

        cv2.circle(output, centroid, 4, (0, 0, 255), -1)
        cv2.putText(
            output,
            str(track_id),
            (centroid[0] + 8, max(centroid[1] - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return output


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


def make_preview_frame(
    original_frame: np.ndarray,
    cropped_frame: np.ndarray,
    processed_image: np.ndarray,
    tracked_features: pd.DataFrame,
    track_history: dict[int, list[tuple[int, int]]],
) -> np.ndarray:
    update_track_history(tracked_features, track_history)

    left = draw_left_panel(cropped_frame, tracked_features, track_history)
    left = add_panel_label(left, "Original + Tracks")
    right = draw_right_panel(processed_image, tracked_features, track_history)
    right = add_panel_label(right, "Processed Filled + Tracks")

    if left.shape[0] != right.shape[0]:
        height = max(left.shape[0], right.shape[0])
        left = pad_to_height(left, height)
        right = pad_to_height(right, height)

    return np.hstack([left, right])


def pad_to_height(frame_bgr: np.ndarray, target_height: int) -> np.ndarray:
    height = frame_bgr.shape[0]
    if height >= target_height:
        return frame_bgr
    padding = np.zeros((target_height - height, frame_bgr.shape[1], 3), dtype=frame_bgr.dtype)
    return np.vstack([frame_bgr, padding])


def save_csvs(
    output_dir: Path,
    all_features: list[pd.DataFrame],
    all_tracked: list[pd.DataFrame],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if all_features:
        pd.concat(all_features, ignore_index=True).to_csv(output_dir / "all_frame_features.csv", index=False)
    else:
        pd.DataFrame().to_csv(output_dir / "all_frame_features.csv", index=False)

    if all_tracked:
        pd.concat(all_tracked, ignore_index=True).to_csv(output_dir / "tracked_features.csv", index=False)
    else:
        pd.DataFrame().to_csv(output_dir / "tracked_features.csv", index=False)


def main() -> None:
    if not RAW_VIDEO_DIR.exists():
        raise FileNotFoundError(f"RAW_VIDEO_DIR does not exist: {RAW_VIDEO_DIR}")

    experiment_name = EXPERIMENT_NAME or "debug_experiment"
    output_dir = PROCESSED_DIR / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    video_path = find_first_video(RAW_VIDEO_DIR)
    print("Background is being calculated...")
    background_gray = build_background_model(video_path)
    background_preview = cv2.cvtColor(background_gray, cv2.COLOR_GRAY2BGR)
    cv2.imshow("Background Model", background_preview)

    detector_config = ConnectedComponentDetectionConfig(
        min_object_area=MIN_OBJECT_AREA,
        crop_bottom_px=CROP_BOTTOM_PX,
        background_threshold=BACKGROUND_THRESHOLD,
        background_method=BACKGROUND_METHOD,
        background_percentile=BACKGROUND_PERCENTILE,
        fill_holes=FILL_HOLES,
        use_watershed_split=USE_WATERSHED_SPLIT,
        watershed_min_distance=WATERSHED_MIN_DISTANCE,
        fallback_watershed_min_distance=FALLBACK_WATERSHED_MIN_DISTANCE,
        merged_area_ratio=MERGED_AREA_RATIO,
        max_split_objects = MAX_SPLIT_OBJECTS,
    )

    detector = ConnectedComponentDropletDetector(detector_config)
    tracker = DropletTracker(
        max_assignment_distance=MAX_ASSIGNMENT_DISTANCE,
        max_missed=MAX_MISSED,
    )

    track_history: dict[int, list[tuple[int, int]]] = {}
    tracked_frames: list[pd.DataFrame] = []
    raw_frames: list[pd.DataFrame] = []

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    paused = False
    step_frame = False
    frame_id = 0

    print("Controls: space=toggle pause/play, n=next frame when paused, q=quit")

    while True:
        if not paused or step_frame:
            success, frame = capture.read()
            if not success:
                break

            result = detector.detect(frame, frame_id, background_gray)
            tracked = tracker.update(result.features)

            if RUN_FULL_CSV_EXPORT:
                raw_frames.append(result.features)
                tracked_frames.append(tracked)

            preview = make_preview_frame(
                original_frame=frame,
                cropped_frame=result.cropped_frame,
                processed_image=result.filled_image,
                tracked_features=tracked,
                track_history=track_history,
            )
            cv2.imshow("Droplet Live Preview", preview)
            frame_id += 1
            step_frame = False

        delay = VIDEO_PLAYBACK_DELAY_MS if not paused else 0
        key = cv2.waitKey(delay) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            paused = not paused
        if key == ord("n") and paused:
            step_frame = True

    capture.release()
    cv2.destroyAllWindows()

    if RUN_FULL_CSV_EXPORT:
        save_csvs(output_dir, raw_frames, tracked_frames)
        print(f"Saved all_frame_features.csv and tracked_features.csv to {output_dir}")


if __name__ == "__main__":
    main()
