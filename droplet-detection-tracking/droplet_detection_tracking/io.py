from __future__ import annotations

from pathlib import Path

import pandas as pd

from droplet_detection_tracking import schema
from droplet_detection_tracking.configs.settings import TrackingConfig
from droplet_detection_tracking.tracking import DropletTracker, centroid_columns


def save_detection_csvs(
    output_dir: Path,
    all_features: list[pd.DataFrame],
    all_tracked: list[pd.DataFrame],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    features = pd.concat(all_features, ignore_index=True) if all_features else pd.DataFrame()
    tracked = pd.concat(all_tracked, ignore_index=True) if all_tracked else pd.DataFrame()

    features.to_csv(output_dir / schema.ALL_FRAME_FEATURES_CSV, index=False)
    tracked.to_csv(output_dir / schema.TRACKED_FEATURES_CSV, index=False)


def track_detections_csv(
    features_csv_path: Path,
    output_csv_path: Path,
    config: TrackingConfig | None = None,
) -> pd.DataFrame:
    features = pd.read_csv(features_csv_path)
    if features.empty:
        features[schema.TRACK_ID] = pd.Series(dtype="Int64")
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(output_csv_path, index=False)
        return features

    centroid_columns(features)
    tracker = DropletTracker(config or TrackingConfig(max_assignment_distance=15, max_missed=5))
    tracked_frames = []

    features = features.sort_values(schema.FRAME).reset_index(drop=True)
    for _, frame_df in features.groupby(schema.FRAME, sort=True):
        tracked_frames.append(tracker.update(frame_df))

    tracked = pd.concat(tracked_frames, ignore_index=True)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    tracked.to_csv(output_csv_path, index=False)
    return tracked
