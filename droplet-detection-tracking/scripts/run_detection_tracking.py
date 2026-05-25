from __future__ import annotations
import argparse
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
    MAX_ANALYZED_FRAMES,
    MAX_BACKGROUND_FRAMES,
    MAX_ASSIGNMENT_DISTANCE,
    MAX_MISSED,
    MAX_SPLIT_OBJECTS,
    MERGED_AREA_RATIO,
    MIN_OBJECT_AREA,
    RUN_FULL_CSV_EXPORT,
    RUN_LIVE_PREVIEW,
    USE_WATERSHED_SPLIT,
    VIDEO_FILE_NAME,
    VIDEO_PLAYBACK_DELAY_MS,
    WATERSHED_MIN_DISTANCE,
)
from configs.paths import PROCESSED_DIR, RAW_VIDEO_DIR
from src.detection import (
    BackgroundModel,
    ConnectedComponentDetectionConfig,
    ConnectedComponentDropletDetector,
    DetectionResult,
)
from src.tracking import DropletTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the behavior-preserving droplet detection + tracking pipeline."
    )
    parser.add_argument("--video-path", type=Path, default=None, help="Input video file.")
    parser.add_argument(
        "--raw-video-dir",
        type=Path,
        default=RAW_VIDEO_DIR,
        help="Directory searched when --video-path is not provided.",
    )
    parser.add_argument(
        "--video-file-name",
        default=VIDEO_FILE_NAME,
        help="Video filename to find inside --raw-video-dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DIR,
        help="Base output directory. The experiment folder is created inside this path.",
    )
    parser.add_argument(
        "--experiment-name",
        default=EXPERIMENT_NAME,
        help="Experiment/output folder name. Defaults to the copied config value.",
    )
    parser.add_argument(
        "--max-analyzed-frames",
        type=int,
        default=MAX_ANALYZED_FRAMES,
        help="Optional frame limit. Defaults to the copied config value.",
    )
    parser.add_argument(
        "--live-preview",
        action="store_true",
        default=None,
        help="Enable the existing OpenCV live preview.",
    )
    parser.add_argument(
        "--no-live-preview",
        action="store_false",
        dest="live_preview",
        help="Disable the existing OpenCV live preview.",
    )
    parser.add_argument(
        "--full-csv-export",
        action="store_true",
        default=None,
        help="Write all_frame_features.csv and tracked_features.csv.",
    )
    parser.add_argument(
        "--no-full-csv-export",
        action="store_false",
        dest="full_csv_export",
        help="Disable CSV export.",
    )
    return parser.parse_args()


def find_video(raw_video_dir: Path, video_file_name: str | None = None) -> Path:
    supported_extensions = ["mp4", "avi", "mov", "mkv"]

    if video_file_name:
        candidate = raw_video_dir / video_file_name
        if candidate.exists():
            return candidate

        root = raw_video_dir / Path(video_file_name).stem
        for extension in supported_extensions:
            candidate_with_ext = root.with_suffix(f".{extension}")
            if candidate_with_ext.exists():
                return candidate_with_ext

        raise FileNotFoundError(
            f"Unable to find video '{video_file_name}' in {raw_video_dir}."
        )

    for extension in supported_extensions:
        for candidate in sorted(raw_video_dir.glob(f"*.{extension}")):
            return candidate

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
    args = parse_args()

    if args.video_path is not None:
        video_path = args.video_path
        if not video_path.exists():
            raise FileNotFoundError(f"Video path does not exist: {video_path}")
    else:
        if not args.raw_video_dir.exists():
            raise FileNotFoundError(f"RAW_VIDEO_DIR does not exist: {args.raw_video_dir}")
        video_path = find_video(args.raw_video_dir, args.video_file_name)

    run_live_preview = RUN_LIVE_PREVIEW if args.live_preview is None else args.live_preview
    run_full_csv_export = RUN_FULL_CSV_EXPORT if args.full_csv_export is None else args.full_csv_export
    max_analyzed_frames = args.max_analyzed_frames

    print(f"Using video: {video_path.name}")
    print("Background is being calculated...")

    experiment_name = args.experiment_name or video_path.stem
    output_dir = args.output_dir / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    background_gray = build_background_model(video_path)
    if run_live_preview:
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

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    analyzed_frames = max_analyzed_frames if max_analyzed_frames is not None else None
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

            if run_full_csv_export:
                raw_frames.append(result.features)
                tracked_frames.append(tracked)

            if run_live_preview:
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

            if frame_id % 1000 == 0:
                if analyzed_frames is not None:
                    remaining = max(analyzed_frames - frame_id, 0)
                    print(f"Processed {frame_id} frames; {remaining} frames remain (limit).")
                elif total_frames > 0:
                    remaining = max(total_frames - frame_id, 0)
                    print(f"Processed {frame_id} frames; {remaining} frames remain.")
                else:
                    print(f"Processed {frame_id} frames.")

            if max_analyzed_frames is not None and frame_id >= max_analyzed_frames:
                print(f"Reached max analyzed frames: {max_analyzed_frames}")
                break

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

    if run_full_csv_export:
        save_csvs(output_dir, raw_frames, tracked_frames)
        print(f"Saved all_frame_features.csv and tracked_features.csv to {output_dir}")


if __name__ == "__main__":
    main()
