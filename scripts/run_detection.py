from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.paths import FRAME_DIR, PROCESSED_DIR, RAW_VIDEO_DIR
from src.detection_pipeline import (
    ConnectedComponentDetectionConfig,
    ConnectedComponentDropletDetector,
    DropletDetectionPipeline,
    FrameExtractionConfig,
    VideoFrameExtractor,
)


VIDEO_EXTENSIONS = {".avi", ".cine", ".m4v", ".mov", ".mp4", ".mpeg", ".mpg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract frames and detect droplets with connected components.",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Optional explicit video path. Defaults to the first video in RAW_VIDEO_DIR.",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Optional output folder name. Defaults to the video stem.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=127,
        help="Fixed grayscale threshold for connected-components detection.",
    )
    parser.add_argument(
        "--invert-threshold",
        action="store_true",
        help="Use binary inverse thresholding.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = args.video or find_first_video(RAW_VIDEO_DIR)
    experiment_name = args.experiment_name or video_path.stem

    frame_output_dir = FRAME_DIR / experiment_name
    processed_output_dir = PROCESSED_DIR / experiment_name

    extractor = VideoFrameExtractor(
        FrameExtractionConfig(
            video_path=video_path,
            output_dir=frame_output_dir,
            every_n_frames=20,
            max_frames=50,
        )
    )
    frame_paths = extractor.extract()

    detector = ConnectedComponentDropletDetector(
        ConnectedComponentDetectionConfig(
            threshold=args.threshold,
            invert_threshold=args.invert_threshold,
        )
    )
    pipeline = DropletDetectionPipeline(detector)
    features = pipeline.process_frame_folder(
        frame_dir=frame_output_dir,
        output_dir=processed_output_dir,
        save_per_frame_csvs=True,
        save_overlays=True,
    )

    print(f"Video: {video_path}")
    print(f"Extracted frames: {len(frame_paths)}")
    print(f"Detected droplets: {len(features)}")
    print(f"Frames: {frame_output_dir}")
    print(f"Results: {processed_output_dir}")
    print(f"All features: {processed_output_dir / 'all_frame_features.csv'}")


def find_first_video(raw_video_dir: Path) -> Path:
    raw_video_dir = Path(raw_video_dir)
    if not raw_video_dir.exists():
        raise FileNotFoundError(f"RAW_VIDEO_DIR does not exist: {raw_video_dir}")

    video_paths = sorted(
        path
        for path in raw_video_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not video_paths:
        raise FileNotFoundError(f"No video files found in RAW_VIDEO_DIR: {raw_video_dir}")

    return video_paths[0]


if __name__ == "__main__":
    main()
