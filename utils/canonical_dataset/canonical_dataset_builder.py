from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


class CanonicalDatasetBuilder:
    def __init__(self, input_csv, output_npz, feature_names=None):
        self.input_csv = Path(input_csv)
        self.output_npz = Path(output_npz)
        self.feature_names = feature_names or ["x", "y", "vx", "vy", "circularity"]

    def run(self):
        tracks = pd.read_csv(self.input_csv)
        tracks = self._standardize_columns(tracks)
        tracks = tracks[["frame", "track_id", "x", "y", "circularity"]]
        tracks = tracks.sort_values(["track_id", "frame"]).reset_index(drop=True)

        interpolated = self._interpolate_tracks(tracks)
        interpolated = self._add_velocities(interpolated)

        Z, mask, track_ids, frames = self._build_tensor(interpolated)

        self.output_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            self.output_npz,
            Z=Z,
            mask=mask,
            track_ids=track_ids,
            frames=frames,
            feature_names=np.asarray(self.feature_names, dtype=str),
        )

        coverage = mask.mean() if mask.size else 0.0
        print(f"N: {len(track_ids)}")
        print(f"T: {len(frames)}")
        print(f"F: {len(self.feature_names)}")
        print(f"Z shape: {Z.shape}")
        print(f"mask coverage: {coverage:.4f}")
        print(f"output path: {self.output_npz}")

    def _standardize_columns(self, tracks):
        rename_map = {}
        if "centroid_x" in tracks.columns and "x" not in tracks.columns:
            rename_map["centroid_x"] = "x"
        if "centroid_y" in tracks.columns and "y" not in tracks.columns:
            rename_map["centroid_y"] = "y"
        return tracks.rename(columns=rename_map)

    def _interpolate_tracks(self, tracks):
        filled_tracks = []

        for track_id, track in tracks.groupby("track_id", sort=True):
            track = track.sort_values("frame").set_index("frame")
            frame_index = np.arange(int(track.index.min()), int(track.index.max()) + 1)

            filled = track.reindex(frame_index)
            filled.index.name = "frame"
            filled["track_id"] = track_id
            filled[["x", "y", "circularity"]] = filled[["x", "y", "circularity"]].interpolate(
                method="linear"
            )
            filled_tracks.append(filled.reset_index())

        if not filled_tracks:
            return pd.DataFrame(columns=["frame", "track_id", "x", "y", "circularity"])

        return pd.concat(filled_tracks, ignore_index=True)

    def _add_velocities(self, tracks):
        tracks = tracks.sort_values(["track_id", "frame"]).reset_index(drop=True)
        tracks["vx"] = tracks.groupby("track_id")["x"].diff()
        tracks["vy"] = tracks.groupby("track_id")["y"].diff()
        return tracks

    def _build_tensor(self, tracks):
        missing_features = [name for name in self.feature_names if name not in tracks.columns]
        if missing_features:
            raise KeyError(f"Missing requested feature columns: {missing_features}")

        track_ids = np.asarray(sorted(tracks["track_id"].dropna().unique()))
        if tracks.empty:
            frames = np.asarray([], dtype=int)
        else:
            frames = np.arange(int(tracks["frame"].min()), int(tracks["frame"].max()) + 1)

        track_index = {track_id: i for i, track_id in enumerate(track_ids)}
        frame_index = {frame: i for i, frame in enumerate(frames)}

        Z = np.full((len(track_ids), len(frames), len(self.feature_names)), np.nan, dtype=np.float32)
        mask = np.zeros((len(track_ids), len(frames)), dtype=bool)

        for _, row in tracks.iterrows():
            i = track_index[row["track_id"]]
            t = frame_index[int(row["frame"])]
            Z[i, t, :] = row[self.feature_names].to_numpy(dtype=np.float32)
            mask[i, t] = True

        return Z, mask, track_ids, frames
