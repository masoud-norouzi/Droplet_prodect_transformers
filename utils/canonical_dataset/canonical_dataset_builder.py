from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


class CanonicalDatasetBuilder:
    def __init__(self, input_csv, output_npz, feature_names=None, inlet_y_max_px=100.0):
        self.input_csv = Path(input_csv)
        self.output_npz = Path(output_npz)
        self.feature_names = feature_names or ["x", "y", "vx", "vy", "circularity"]
        self.inlet_y_max_px = float(inlet_y_max_px)
        self.inlet_velocity_diagnostics = {}

    def run(self):
        tracks = pd.read_csv(self.input_csv)
        tracks = self._standardize_columns(tracks)
        tracks = tracks[["frame", "track_id", "x", "y", "circularity"]]
        tracks = tracks.sort_values(["track_id", "frame"]).reset_index(drop=True)

        interpolated = self._interpolate_tracks(tracks)
        interpolated = self._add_velocities(interpolated)
        interpolated = self._patch_new_inlet_velocities(interpolated)

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
        print("inlet velocity patch:")
        print(f"  criterion: first visible track frame with y <= {self.inlet_y_max_px:g} px")
        print(f"  mean_inlet_vx: {self.inlet_velocity_diagnostics['mean_inlet_vx']:.6f}")
        print(f"  mean_inlet_vy: {self.inlet_velocity_diagnostics['mean_inlet_vy']:.6f}")
        print(f"  finite inlet velocity observations: {self.inlet_velocity_diagnostics['finite_inlet_velocity_observations']}")
        print(f"  patched rows: {self.inlet_velocity_diagnostics['patched_rows']}")
        print(f"  patched velocity values: {self.inlet_velocity_diagnostics['patched_velocity_values']}")
        print(f"  affected unique tracks: {self.inlet_velocity_diagnostics['affected_unique_tracks']}")
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

    def _patch_new_inlet_velocities(self, tracks):
        tracks = tracks.sort_values(["track_id", "frame"]).reset_index(drop=True).copy()
        inlet_region = tracks["y"].le(self.inlet_y_max_px)
        finite_velocity = np.isfinite(tracks["vx"]) & np.isfinite(tracks["vy"])
        inlet_velocity_samples = tracks.loc[inlet_region & finite_velocity, ["vx", "vy"]]
        if inlet_velocity_samples.empty:
            raise ValueError(
                "No finite inlet velocity observations found for "
                f"y <= {self.inlet_y_max_px:g} px."
            )

        mean_inlet_vx = float(inlet_velocity_samples["vx"].mean())
        mean_inlet_vy = float(inlet_velocity_samples["vy"].mean())

        first_visible = tracks.groupby("track_id")["frame"].transform("min").eq(tracks["frame"])
        missing_vx = ~np.isfinite(tracks["vx"])
        missing_vy = ~np.isfinite(tracks["vy"])
        patch_rows = first_visible & inlet_region & (missing_vx | missing_vy)
        patch_vx = patch_rows & missing_vx
        patch_vy = patch_rows & missing_vy

        tracks.loc[patch_vx, "vx"] = mean_inlet_vx
        tracks.loc[patch_vy, "vy"] = mean_inlet_vy

        affected_tracks = tracks.loc[patch_rows, "track_id"].dropna().unique()
        self.inlet_velocity_diagnostics = {
            "inlet_y_max_px": self.inlet_y_max_px,
            "mean_inlet_vx": mean_inlet_vx,
            "mean_inlet_vy": mean_inlet_vy,
            "finite_inlet_velocity_observations": int(len(inlet_velocity_samples)),
            "patched_rows": int(patch_rows.sum()),
            "patched_velocity_values": int(patch_vx.sum() + patch_vy.sum()),
            "affected_unique_tracks": int(len(affected_tracks)),
        }
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
