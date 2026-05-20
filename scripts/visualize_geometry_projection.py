from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import pandas as pd

from configs.constants import EXPERIMENT_NAME, VIDEO_FILE_NAME
from configs.paths import PROCESSED_DIR, PROJECT_ROOT, RAW_VIDEO_DIR
from src.geometry import ChannelGeometry


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize droplet projections onto manually defined centerlines."
    )
    parser.add_argument(
        "--experiment-name",
        default=EXPERIMENT_NAME,
        help=(
            "Experiment name under outputs/processed/. "
            "If omitted, uses configs.constants.EXPERIMENT_NAME."
        ),
    )
    parser.add_argument(
        "--frame-number",
        type=int,
        required=True,
        help="Zero-based video frame number to visualize.",
    )
    parser.add_argument(
        "--trajectories-csv",
        type=Path,
        default=None,
        help=(
            "Optional path to trajectories_geometry.csv. If omitted, uses "
            "outputs/processed/<experiment_name>/trajectories_geometry.csv."
        ),
    )
    parser.add_argument(
        "--centerlines-csv",
        type=Path,
        default=PROJECT_ROOT / "geometry" / "centerlines.csv",
        help="Path to centerlines.csv with channel_id, x, and y columns.",
    )
    parser.add_argument(
        "--video-path",
        type=Path,
        default=None,
        help=(
            "Optional path to the source video. If omitted, searches RAW_VIDEO_DIR "
            "for configs.constants.VIDEO_FILE_NAME."
        ),
    )
    return parser.parse_args(args)


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


def load_video_frame(video_path: Path, frame_number: int):
    if frame_number < 0:
        raise ValueError("--frame-number must be non-negative.")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0 and frame_number >= total_frames:
            raise ValueError(
                f"Frame {frame_number} is outside video range 0-{total_frames - 1}."
            )

        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Unable to read frame {frame_number} from {video_path}")
        return frame
    finally:
        capture.release()


def load_trajectories(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Geometry trajectories file not found: {path}")

    df = pd.read_csv(path)
    required_columns = {
        "frame",
        "track_id",
        "x",
        "y",
        "channel_id",
        "s_coord",
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise KeyError(f"Missing required trajectory columns: {sorted(missing)}")

    return df


def load_centerlines(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Centerlines file not found: {path}")

    df = pd.read_csv(path)
    required_columns = {"channel_id", "x", "y"}
    missing = required_columns - set(df.columns)
    if missing:
        raise KeyError(f"Missing required centerline columns: {sorted(missing)}")

    return df


def point_from_row(row: pd.Series, x_column: str, y_column: str) -> tuple[int, int]:
    return int(round(float(row[x_column]))), int(round(float(row[y_column])))


def draw_centerlines(frame, centerlines: pd.DataFrame) -> None:
    for _, channel_df in centerlines.groupby("channel_id", sort=False):
        ordered = channel_df.loc[:, ["x", "y"]].to_numpy(dtype=float)
        points = [
            (int(round(x)), int(round(y)))
            for x, y in ordered
        ]
        for start, end in zip(points, points[1:]):
            cv2.line(frame, start, end, (255, 0, 0), 1, cv2.LINE_AA)


def draw_projection_annotations(frame, frame_df: pd.DataFrame) -> None:
    for _, row in frame_df.iterrows():
        original = point_from_row(row, "x", "y")
        projection = point_from_row(row, "projection_x", "projection_y")
        track_id = int(row["track_id"])
        channel_id = str(row["channel_id"])
        s_coord = float(row["s_coord"])

        cv2.line(frame, original, projection, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(frame, original, 4, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(frame, projection, 4, (0, 0, 255), -1, cv2.LINE_AA)

        label = f"{track_id} {channel_id} sG={s_coord:.1f}"
        text_origin = (original[0] + 7, max(14, original[1] - 7))
        cv2.putText(
            frame,
            label,
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            label,
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def draw_legend(frame, frame_number: int, detection_count: int) -> None:
    legend_lines = [
        f"frame {frame_number} | detections {detection_count}",
        "green: droplet  red: projection  blue: centerline",
    ]
    box_width = 430
    box_height = 48
    cv2.rectangle(frame, (0, 0), (box_width, box_height), (0, 0, 0), -1)
    for index, text in enumerate(legend_lines):
        cv2.putText(
            frame,
            text,
            (10, 19 + index * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def resolve_inputs(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    if args.trajectories_csv is not None:
        trajectories_csv = args.trajectories_csv
        output_dir = args.trajectories_csv.parent
    else:
        if not args.experiment_name:
            raise ValueError(
                "Experiment name is required when --trajectories-csv is not provided."
            )
        output_dir = PROCESSED_DIR / args.experiment_name
        trajectories_csv = output_dir / "trajectories_geometry.csv"

    video_path = args.video_path
    if video_path is None:
        video_path = find_video(RAW_VIDEO_DIR, VIDEO_FILE_NAME)

    return trajectories_csv, video_path, output_dir


def main() -> None:
    args = parse_args()
    trajectories_csv, video_path, output_dir = resolve_inputs(args)

    trajectories = load_trajectories(trajectories_csv)
    geometry = ChannelGeometry(args.centerlines_csv)
    centerlines = load_centerlines(args.centerlines_csv)
    frame = load_video_frame(video_path, args.frame_number)

    frame_df = trajectories.loc[trajectories["frame"] == args.frame_number].copy()
    if frame_df.empty:
        print(f"No droplet rows found for frame {args.frame_number}.")
    else:
        frame_df = geometry.annotate_dataframe(frame_df, include_projection=True)

    output = frame.copy()
    draw_centerlines(output, centerlines)
    draw_projection_annotations(output, frame_df)
    draw_legend(output, args.frame_number, len(frame_df))

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"geometry_projection_frame_{args.frame_number}.png"
    if not cv2.imwrite(str(output_path), output):
        raise RuntimeError(f"Unable to write visualization image: {output_path}")

    print(f"Using video: {video_path}")
    print(f"Rows visualized: {len(frame_df)}")
    print(f"Saved projection visualization: {output_path}")


if __name__ == "__main__":
    main()
