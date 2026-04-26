from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


@dataclass
class DropletTrackerConfig:
    max_assignment_distance: float = 3
    max_missed: int = 10


class KalmanTrack:
    def __init__(self, track_id: int, x: float, y: float):
        self.track_id = track_id
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.transitionMatrix = np.array(
            [
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        self.kalman.measurementMatrix = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ],
            dtype=np.float32,
        )
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 100
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32)
        self.kalman.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)

        self.age = 0
        self.missed = 0
        self.hits = 1
        self.latest_position = (float(x), float(y))

    def predict(self) -> tuple[float, float]:
        prediction = self.kalman.predict()
        x = float(prediction[0, 0])
        y = float(prediction[1, 0])
        self.age += 1
        self.latest_position = (x, y)
        return self.latest_position

    def update(self, x: float, y: float) -> tuple[float, float]:
        measurement = np.array([[x], [y]], dtype=np.float32)
        corrected = self.kalman.correct(measurement)
        self.missed = 0
        self.hits += 1
        self.latest_position = (float(corrected[0, 0]), float(corrected[1, 0]))
        return self.latest_position


class DropletTracker:
    def __init__(
        self,
        max_assignment_distance: float = 30.0,
        max_missed: int = 5,
    ):
        self.config = DropletTrackerConfig(
            max_assignment_distance=max_assignment_distance,
            max_missed=max_missed,
        )
        self.tracks: list[KalmanTrack] = []
        self.next_track_id = 1

    def update(self, detections_df_for_one_frame: pd.DataFrame) -> pd.DataFrame:
        detections = detections_df_for_one_frame.copy()
        detections["track_id"] = pd.Series(dtype="Int64")

        predicted_positions = [track.predict() for track in self.tracks]

        if detections.empty:
            for track in self.tracks:
                track.missed += 1
            self._delete_stale_tracks()
            return detections

        x_col, y_col = _centroid_columns(detections)
        detection_positions = detections[[x_col, y_col]].to_numpy(dtype=float)

        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()

        if self.tracks:
            distances = self._distance_matrix(predicted_positions, detection_positions)
            track_indices, detection_indices = linear_sum_assignment(distances)

            for track_index, detection_index in zip(track_indices, detection_indices):
                distance = distances[track_index, detection_index]
                if distance > self.config.max_assignment_distance:
                    continue

                x, y = detection_positions[detection_index]
                track = self.tracks[track_index]
                track.update(float(x), float(y))
                detections.iat[detection_index, detections.columns.get_loc("track_id")] = (
                    track.track_id
                )
                matched_tracks.add(track_index)
                matched_detections.add(detection_index)

        for track_index, track in enumerate(self.tracks):
            if track_index not in matched_tracks:
                track.missed += 1

        for detection_index, (x, y) in enumerate(detection_positions):
            if detection_index in matched_detections:
                continue

            track = self._create_track(float(x), float(y))
            detections.iat[detection_index, detections.columns.get_loc("track_id")] = (
                track.track_id
            )

        self._delete_stale_tracks()
        detections["track_id"] = detections["track_id"].astype("Int64")
        return detections

    @staticmethod
    def _distance_matrix(
        predicted_positions: list[tuple[float, float]],
        detection_positions: np.ndarray,
    ) -> np.ndarray:
        predictions = np.asarray(predicted_positions, dtype=float)
        deltas = predictions[:, None, :] - detection_positions[None, :, :]
        return np.linalg.norm(deltas, axis=2)

    def _create_track(self, x: float, y: float) -> KalmanTrack:
        track = KalmanTrack(self.next_track_id, x, y)
        self.next_track_id += 1
        self.tracks.append(track)
        return track

    def _delete_stale_tracks(self) -> None:
        self.tracks = [
            track
            for track in self.tracks
            if track.missed <= self.config.max_missed
        ]


def track_detections(features_csv_path: str, output_csv_path: str) -> pd.DataFrame:
    features_path = Path(features_csv_path)
    output_path = Path(output_csv_path)

    features = pd.read_csv(features_path)
    if features.empty:
        features["track_id"] = pd.Series(dtype="Int64")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(output_path, index=False)
        return features

    _centroid_columns(features)
    tracker = DropletTracker(max_assignment_distance=15)
    tracked_frames = []

    features = features.sort_values("frame").reset_index(drop=True)
    for _, frame_df in features.groupby("frame", sort=True):
        tracked_frames.append(tracker.update(frame_df))

    tracked = pd.concat(tracked_frames, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tracked.to_csv(output_path, index=False)
    return tracked


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
