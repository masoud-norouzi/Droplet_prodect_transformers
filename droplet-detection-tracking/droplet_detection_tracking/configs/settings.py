from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class InputConfig:
    raw_video_dir: Path = Path(r"D:\Microfluidic loop projct\new loop experiments\confined droplets 2")
    video_file_name: str = "2.avi"
    max_analyzed_frames: int | None = None


@dataclass(frozen=True)
class OutputConfig:
    experiment_name: str = "2"
    output_dir: Path = Path("outputs")
    processed_dir: Path = output_dir / "processed"
    run_full_csv_export: bool = True


@dataclass(frozen=True)
class BackgroundConfig:
    crop_bottom_px: int = 35
    sample_every_n_frames: int = 20
    max_background_frames: int | None = 200
    method: str = "percentile"
    percentile: int = 80


@dataclass(frozen=True)
class DropletDetectionConfig:
    min_object_area: int = 100
    crop_bottom_px: int = 35
    background_threshold: int = 40
    fill_holes: bool = True
    use_watershed_split: bool = True
    watershed_min_distance: int = 10
    fallback_watershed_min_distance: int = 5
    merged_area_ratio: float = 1.5
    max_split_objects: int = 3


@dataclass(frozen=True)
class TrackingConfig:
    max_assignment_distance: float = 25
    max_missed: int = 10
    assignment_prediction_weight: float = 1


@dataclass(frozen=True)
class PreviewConfig:
    run_live_preview: bool = False
    video_playback_delay_ms: int = 1


@dataclass(frozen=True)
class PipelineConfig:
    input: InputConfig = field(default_factory=InputConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    background: BackgroundConfig = field(default_factory=BackgroundConfig)
    detection: DropletDetectionConfig = field(default_factory=DropletDetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    preview: PreviewConfig = field(default_factory=PreviewConfig)


DEFAULT_CONFIG = PipelineConfig()
