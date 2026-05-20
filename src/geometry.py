from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class _Channel:
    channel_id: Any
    points: np.ndarray
    segment_starts: np.ndarray
    segment_vectors: np.ndarray
    segment_lengths: np.ndarray
    segment_lengths_squared: np.ndarray
    cumulative_s: np.ndarray
    length: float


class ChannelGeometry:
    """Geometry helper for projecting droplet positions onto channel centerlines."""

    def __init__(self, centerlines_csv: str | Path) -> None:
        self.centerlines_csv = Path(centerlines_csv)
        self.channels = self._load_channels(self.centerlines_csv)
        self._build_segment_index()

    @staticmethod
    def _load_channels(centerlines_csv: Path) -> dict[Any, _Channel]:
        if not centerlines_csv.exists():
            raise FileNotFoundError(f"Centerlines file not found: {centerlines_csv}")

        df = pd.read_csv(centerlines_csv)
        required_columns = {"channel_id", "x", "y"}
        missing = required_columns - set(df.columns)
        if missing:
            raise KeyError(f"Missing required centerline columns: {sorted(missing)}")

        channels: dict[Any, _Channel] = {}
        for channel_id, channel_df in df.groupby("channel_id", sort=False):
            points = channel_df.loc[:, ["x", "y"]].to_numpy(dtype=float)
            if len(points) < 2:
                raise ValueError(
                    f"Channel {channel_id!r} needs at least two centerline points."
                )

            segment_starts = points[:-1]
            segment_vectors = points[1:] - points[:-1]
            segment_lengths = np.linalg.norm(segment_vectors, axis=1)
            valid_segments = segment_lengths > 0
            if not valid_segments.any():
                raise ValueError(
                    f"Channel {channel_id!r} has no non-zero length centerline segments."
                )

            segment_starts = segment_starts[valid_segments]
            segment_vectors = segment_vectors[valid_segments]
            segment_lengths = segment_lengths[valid_segments]
            segment_lengths_squared = segment_lengths * segment_lengths
            cumulative_s = np.concatenate(([0.0], np.cumsum(segment_lengths[:-1])))

            channels[channel_id] = _Channel(
                channel_id=channel_id,
                points=points,
                segment_starts=segment_starts,
                segment_vectors=segment_vectors,
                segment_lengths=segment_lengths,
                segment_lengths_squared=segment_lengths_squared,
                cumulative_s=cumulative_s,
                length=float(segment_lengths.sum()),
            )

        if not channels:
            raise ValueError(f"No centerline channels found in {centerlines_csv}")

        return channels

    def _build_segment_index(self) -> None:
        channel_offsets = self._compute_channel_offsets()
        segment_starts = []
        segment_vectors = []
        segment_lengths = []
        segment_lengths_squared = []
        segment_cumulative_s = []
        segment_channel_ids = []

        for channel in self.channels.values():
            segment_count = len(channel.segment_lengths)
            segment_starts.append(channel.segment_starts)
            segment_vectors.append(channel.segment_vectors)
            segment_lengths.append(channel.segment_lengths)
            segment_lengths_squared.append(channel.segment_lengths_squared)
            segment_cumulative_s.append(
                channel.cumulative_s + channel_offsets[channel.channel_id]
            )
            segment_channel_ids.extend([channel.channel_id] * segment_count)

        self._segment_starts = np.vstack(segment_starts)
        self._segment_vectors = np.vstack(segment_vectors)
        self._segment_lengths = np.concatenate(segment_lengths)
        self._segment_lengths_squared = np.concatenate(segment_lengths_squared)
        self._segment_cumulative_s = np.concatenate(segment_cumulative_s)
        self._segment_channel_ids = np.asarray(segment_channel_ids, dtype=object)
        self.channel_offsets = channel_offsets

    def _compute_channel_offsets(self) -> dict[Any, float]:
        """Return global downstream offsets for known branch geometry."""
        channel_ids = list(self.channels)
        channel_names = {str(channel_id): channel_id for channel_id in channel_ids}

        if {"inlet", "outlet"}.issubset(channel_names):
            inlet_id = channel_names["inlet"]
            outlet_id = channel_names["outlet"]
            inlet_length = self.channels[inlet_id].length
            branch_ids = [
                channel_id
                for channel_id in channel_ids
                if channel_id not in {inlet_id, outlet_id}
            ]
            branch_length = max(
                (self.channels[channel_id].length for channel_id in branch_ids),
                default=0.0,
            )

            offsets = {channel_id: inlet_length for channel_id in branch_ids}
            offsets[inlet_id] = 0.0
            offsets[outlet_id] = inlet_length + branch_length
            return offsets

        offsets: dict[Any, float] = {}
        next_offset = 0.0
        for channel_id, channel in self.channels.items():
            offsets[channel_id] = next_offset
            next_offset += channel.length
        return offsets

    def project_point(self, x: float, y: float) -> dict[str, Any]:
        """Project one point to the nearest centerline segment across all channels."""
        projections = self._project_points(np.array([[x, y]], dtype=float))
        return projections.iloc[0].to_dict()

    def _project_points(
        self, points: np.ndarray, chunk_size: int = 2500
    ) -> pd.DataFrame:
        if len(points) == 0:
            return pd.DataFrame(
                columns=[
                    "channel_id",
                    "s_coord",
                    "d_centerline",
                    "projection_x",
                    "projection_y",
                ]
            )

        results: list[pd.DataFrame] = []
        for chunk_start in range(0, len(points), chunk_size):
            point_chunk = points[chunk_start : chunk_start + chunk_size]
            offsets = point_chunk[:, None, :] - self._segment_starts[None, :, :]
            t = (
                np.einsum("nsi,si->ns", offsets, self._segment_vectors)
                / self._segment_lengths_squared[None, :]
            )
            t = np.clip(t, 0.0, 1.0)

            projections = (
                self._segment_starts[None, :, :]
                + t[:, :, None] * self._segment_vectors[None, :, :]
            )
            deltas = point_chunk[:, None, :] - projections
            distances_squared = np.einsum("nsi,nsi->ns", deltas, deltas)
            segment_indices = np.argmin(distances_squared, axis=1)
            row_indices = np.arange(len(point_chunk))
            selected_t = t[row_indices, segment_indices]
            selected_projections = projections[row_indices, segment_indices]
            selected_distances = np.sqrt(distances_squared[row_indices, segment_indices])

            results.append(
                pd.DataFrame(
                    {
                        "channel_id": self._segment_channel_ids[segment_indices],
                        "s_coord": self._segment_cumulative_s[segment_indices]
                        + selected_t * self._segment_lengths[segment_indices],
                        "d_centerline": selected_distances,
                        "projection_x": selected_projections[:, 0],
                        "projection_y": selected_projections[:, 1],
                    }
                )
            )

        return pd.concat(results, ignore_index=True)

    def annotate_dataframe(
        self, df: pd.DataFrame, include_projection: bool = False
    ) -> pd.DataFrame:
        """Return a copy of df with nearest-channel geometry feature columns added."""
        required_columns = {"x", "y"}
        missing = required_columns - set(df.columns)
        if missing:
            raise KeyError(f"Missing required trajectory columns: {sorted(missing)}")

        points = df.loc[:, ["x", "y"]].to_numpy(dtype=float)
        annotation_df = self._project_points(points)
        annotation_df.index = df.index

        output_df = df.copy()
        columns = ["channel_id", "s_coord", "d_centerline"]
        if include_projection:
            columns.extend(["projection_x", "projection_y"])

        for column in columns:
            output_df[column] = annotation_df[column]

        return output_df
