from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from droplet_detection_tracking.configs.settings import DEFAULT_CONFIG
from droplet_detection_tracking.detection import BackgroundModel, ConnectedComponentDropletDetector
from droplet_detection_tracking.io import save_detection_csvs
from droplet_detection_tracking.preview import make_preview_frame
from droplet_detection_tracking.tracking import DropletTracker


def parse_args() -> argparse.Namespace:
    config = DEFAULT_CONFIG
    parser = argparse.ArgumentParser(
        description="Run the behavior-preserving droplet detection + tracking pipeline."
    )
    parser.add_argument("--video-path", type=Path, default=None, help="Input video file.")
    parser.add_argument(
        "--raw-video-dir",
        type=Path,
        default=config.input.raw_video_dir,
        help="Directory searched when --video-path is not provided.",
    )
    parser.add_argument(
        "--video-file-name",
        default=config.input.video_file_name,
        help="Video filename to find inside --raw-video-dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.output.processed_dir,
        help="Base output directory. The experiment folder is created inside this path.",
    )
    parser.add_argument(
        "--experiment-name",
        default=config.output.experiment_name,
        help="Experiment/output folder name. Defaults to the configured value.",
    )
    parser.add_argument(
        "--max-analyzed-frames",
        type=int,
        default=config.input.max_analyzed_frames,
        help="Optional frame limit. Defaults to the configured value.",
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


def resolve_video_path(args: argparse.Namespace) -> Path:
    if args.video_path is not None:
        if not args.video_path.exists():
            raise FileNotFoundError(f"Video path does not exist: {args.video_path}")
        return args.video_path

    if not args.raw_video_dir.exists():
        raise FileNotFoundError(f"Raw video directory does not exist: {args.raw_video_dir}")
    return find_video(args.raw_video_dir, args.video_file_name)


def main() -> None:
    args = parse_args()
    config = DEFAULT_CONFIG

    video_path = resolve_video_path(args)
    run_live_preview = config.preview.run_live_preview if args.live_preview is None else args.live_preview
    run_full_csv_export = config.output.run_full_csv_export if args.full_csv_export is None else args.full_csv_export
    max_analyzed_frames = args.max_analyzed_frames

    print(f"Using video: {video_path.name}")
    print("Background is being calculated...")

    experiment_name = args.experiment_name or video_path.stem
    output_dir = args.output_dir / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    background_gray = BackgroundModel(video_path, config.background).build()
    if run_live_preview:
        background_preview = cv2.cvtColor(background_gray, cv2.COLOR_GRAY2BGR)
        cv2.imshow("Background Model", background_preview)

    detector = ConnectedComponentDropletDetector(config.detection)
    tracker = DropletTracker(config.tracking)

    tracked_frames = []
    raw_frames = []

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
            tracked = tracker.update(result.detections)

            if run_full_csv_export:
                raw_frames.append(result.detections)
                tracked_frames.append(tracked)

            if run_live_preview:
                preview = make_preview_frame(
                    cropped_frame=result.cropped_frame,
                    processed_image=result.filled_mask,
                    tracked_features=tracked,
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

        delay = config.preview.video_playback_delay_ms if not paused else 0
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
        save_detection_csvs(output_dir, raw_frames, tracked_frames)
        print(f"Saved all_frame_features.csv and tracked_features.csv to {output_dir}")
