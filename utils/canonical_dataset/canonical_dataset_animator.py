from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
import numpy as np


class CanonicalDatasetAnimator:
    def __init__(
        self,
        dataset_path="canonical_droplet_dataset.npz",
        output_video="canonical_animation.mp4",
        start_frame=None,
        end_frame=None,
        fps=30,
        marker_size=25,
        show_track_ids=False,
        trail_length=10,
        frame_sample_count=None,
        random_seed=0,
    ):
        self.dataset_path = Path(dataset_path)
        self.output_video = Path(output_video)
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.fps = fps
        self.marker_size = marker_size
        self.show_track_ids = show_track_ids
        self.trail_length = trail_length
        self.frame_sample_count = frame_sample_count
        self.random_seed = random_seed

        self.Z = None
        self.mask = None
        self.track_ids = None
        self.frames = None
        self.feature_names = None
        self.feature_indices = {}
        self.frame_indices = None
        self.colors = None

        self.fig = None
        self.ax = None
        self.scatter = None
        self.info_text = None
        self.trail_artists = []
        self.track_id_artists = []

    def run(self):
        self._load_dataset()
        self._prepare_frame_range()
        self._prepare_colors()
        self._setup_figure()

        animation = FuncAnimation(
            self.fig,
            self._draw_frame,
            frames=self.frame_indices,
            interval=1000 / self.fps,
            blit=False,
        )

        ffmpeg_path = self._resolve_ffmpeg_path()

        self.output_video.parent.mkdir(parents=True, exist_ok=True)
        plt.rcParams["animation.ffmpeg_path"] = ffmpeg_path
        writer = FFMpegWriter(fps=self.fps)
        animation.save(self.output_video, writer=writer)
        plt.close(self.fig)
        print(f"Saved animation: {self.output_video}")

    def _resolve_ffmpeg_path(self):
        try:
            import imageio_ffmpeg
        except ImportError as exc:
            raise RuntimeError(
                "imageio-ffmpeg is required to save MP4 animations. "
                "Install it with: pip install imageio-ffmpeg"
            ) from exc

        return imageio_ffmpeg.get_ffmpeg_exe()

    def _load_dataset(self):
        dataset = np.load(self.dataset_path, allow_pickle=False)
        self.Z = dataset["Z"]
        self.mask = dataset["mask"]
        self.track_ids = dataset["track_ids"]
        self.frames = dataset["frames"]
        self.feature_names = [str(name) for name in dataset["feature_names"]]

        for required_feature in ["x", "y", "circularity"]:
            if required_feature not in self.feature_names:
                raise KeyError(f"Missing required feature: {required_feature}")
            self.feature_indices[required_feature] = self.feature_names.index(required_feature)

    def _prepare_frame_range(self):
        if self.frames.size == 0:
            raise ValueError("Dataset contains no frames.")

        first_frame = self.frames[0] if self.start_frame is None else self.start_frame
        last_frame = self.frames[-1] if self.end_frame is None else self.end_frame

        frame_mask = (self.frames >= first_frame) & (self.frames <= last_frame)
        self.frame_indices = np.flatnonzero(frame_mask)
        if self.frame_indices.size == 0:
            raise ValueError("No dataset frames fall within the requested frame range.")

        if self.frame_sample_count is not None:
            sample_count = min(int(self.frame_sample_count), self.frame_indices.size)
            rng = np.random.default_rng(self.random_seed)
            start_offset = int(rng.integers(0, self.frame_indices.size - sample_count + 1))
            self.frame_indices = self.frame_indices[start_offset : start_offset + sample_count]

    def _prepare_colors(self):
        track_hash = (np.asarray(self.track_ids, dtype=np.uint64) * np.uint64(2654435761)) % np.uint64(2**32)
        hues = track_hash.astype(float) / float(2**32)
        self.colors = plt.cm.hsv(hues)

    def _setup_figure(self):
        x = self.Z[:, :, self.feature_indices["x"]]
        y = self.Z[:, :, self.feature_indices["y"]]
        valid = self.mask & np.isfinite(x) & np.isfinite(y)
        if not np.any(valid):
            raise ValueError("Dataset contains no valid x/y coordinates.")

        x_min, x_max = float(np.nanmin(x[valid])), float(np.nanmax(x[valid]))
        y_min, y_max = float(np.nanmin(y[valid])), float(np.nanmax(y[valid]))
        x_pad = max((x_max - x_min) * 0.05, 1.0)
        y_pad = max((y_max - y_min) * 0.05, 1.0)

        self.fig, self.ax = plt.subplots(figsize=(9, 7))
        self.ax.set_xlim(x_min - x_pad, x_max + x_pad)
        self.ax.set_ylim(y_max + y_pad, y_min - y_pad)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_title("Canonical Droplet Dataset")

        self.scatter = self.ax.scatter([], [], s=self.marker_size)
        self.info_text = self.ax.text(
            0.02,
            0.98,
            "",
            transform=self.ax.transAxes,
            va="top",
            ha="left",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )

    def _draw_frame(self, frame_index):
        self._clear_dynamic_artists()

        active = self.mask[:, frame_index]
        active_track_indices = np.flatnonzero(active)
        x = self.Z[active_track_indices, frame_index, self.feature_indices["x"]]
        y = self.Z[active_track_indices, frame_index, self.feature_indices["y"]]
        valid = np.isfinite(x) & np.isfinite(y)

        active_track_indices = active_track_indices[valid]
        x = x[valid]
        y = y[valid]

        offsets = np.column_stack([x, y]) if x.size else np.empty((0, 2))
        self.scatter.set_offsets(offsets)
        self.scatter.set_sizes(np.full(x.size, self.marker_size))
        self.scatter.set_color(self.colors[active_track_indices] if x.size else [])

        if self.trail_length > 0:
            self._draw_trails(active_track_indices, frame_index)

        if self.show_track_ids:
            self._draw_track_ids(active_track_indices, x, y)

        self.info_text.set_text(
            f"Frame: {int(self.frames[frame_index]):04d}\n"
            f"Active droplets: {len(active_track_indices)}"
        )

        return [self.scatter, self.info_text, *self.trail_artists, *self.track_id_artists]

    def _draw_trails(self, track_indices, frame_index):
        x_idx = self.feature_indices["x"]
        y_idx = self.feature_indices["y"]
        start = max(0, frame_index - self.trail_length)

        for track_index in track_indices:
            trail_frames = np.arange(start, frame_index + 1)
            valid = self.mask[track_index, trail_frames]
            trail_frames = trail_frames[valid]
            if trail_frames.size < 2:
                continue

            x = self.Z[track_index, trail_frames, x_idx]
            y = self.Z[track_index, trail_frames, y_idx]
            finite = np.isfinite(x) & np.isfinite(y)
            if finite.sum() < 2:
                continue

            artist, = self.ax.plot(
                x[finite],
                y[finite],
                color=self.colors[track_index],
                alpha=0.35,
                linewidth=1.2,
            )
            self.trail_artists.append(artist)

    def _draw_track_ids(self, track_indices, x, y):
        for track_index, x_value, y_value in zip(track_indices, x, y):
            artist = self.ax.text(
                x_value + 3,
                y_value + 3,
                str(self.track_ids[track_index]),
                color=self.colors[track_index],
                fontsize=7,
            )
            self.track_id_artists.append(artist)

    def _clear_dynamic_artists(self):
        for artist in self.trail_artists:
            artist.remove()
        for artist in self.track_id_artists:
            artist.remove()
        self.trail_artists = []
        self.track_id_artists = []
