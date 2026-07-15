from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Callable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.canonical_rollout_transformer import CanonicalRolloutTransformer
import train_canonical_rollout_transformer as history_rollout
import train_markovian_rollout_transformer as markovian_rollout
from utils.channel_mask import read_centerline_csv
from utils.geometry_loss import (
    compute_ellipse_outside_fraction,
    compute_ellipse_outside_fraction_torch,
    run_sanity_tests as run_geometry_loss_sanity_tests,
)
from train_geometry_aware_markovian_rollout import (
    CHANNEL_MASK_PATH as TRAINING_CHANNEL_MASK_PATH,
    DETECTIONS_CSV_PATH as TRAINING_DETECTIONS_CSV_PATH,
    GEOMETRY_TOLERANCE,
    FutureBBoxLookup,
)


DEFAULT_NPZ_PATH = Path("outputs/processed/2/canonical_dataset.npz")
DEFAULT_OUTPUT_DIR = Path("outputs/models/rollout_model_comparison")
DEFAULT_HISTORY_CHECKPOINT = Path(
    "outputs/models/train_canonical_rollout_transformer/canonical_rollout_transformer_best.pt"
)
DEFAULT_MARKOVIAN_CHECKPOINT = Path(
    "outputs/models/train_markovian_rollout_transformer/markovian_rollout_transformer_best.pt"
)
DEFAULT_GEOMETRY_AWARE_CHECKPOINT = Path(
    "outputs/models/train_geometry_aware_markovian_rollout/geometry_aware_markovian_rollout_best.pt"
)
DEFAULT_CENTERLINE_CSV = Path("centerlines.csv")
DEFAULT_CHANNEL_MASK_PATH = Path(TRAINING_CHANNEL_MASK_PATH)
DEFAULT_DETECTIONS_CSV_PATH = Path(TRAINING_DETECTIONS_CSV_PATH)

ROLLOUT_HORIZON = 50
MAX_DROPLETS = 64
STRIDE = 5
SOURCE_HISTORY_LENGTH = 20


@dataclass
class RolloutPrediction:
    model_name: str
    window_id: np.ndarray
    rollout_start_frame: np.ndarray
    frame_start: np.ndarray
    track_ids: np.ndarray
    pred_position: np.ndarray
    true_position: np.ndarray
    pred_velocity: np.ndarray
    true_velocity: np.ndarray
    valid_mask: np.ndarray
    boundary_mask: np.ndarray

    @property
    def metric_mask(self) -> np.ndarray:
        return self.valid_mask & ~self.boundary_mask


@dataclass
class StepwiseMetricCurves:
    position_rmse: np.ndarray
    velocity_rmse: np.ndarray
    vx_rmse: np.ndarray
    vy_rmse: np.ndarray
    n_valid_samples: np.ndarray


@dataclass
class IntegratedMetrics:
    integrated_position_rmse: float
    final_step_position_rmse: float


@dataclass
class RmseSufficientStats:
    position_sse: np.ndarray
    velocity_sse: np.ndarray
    vx_sse: np.ndarray
    vy_sse: np.ndarray
    count: np.ndarray


@dataclass
class DirectionalBasis:
    tangent: np.ndarray
    normal: np.ndarray
    axis_valid: np.ndarray
    orientation_valid: np.ndarray
    quality: np.ndarray
    distance_to_centerline: np.ndarray
    branch_name: np.ndarray
    second_nearest_branch_name: np.ndarray
    second_nearest_branch_distance: np.ndarray
    branch_distance_margin: np.ndarray
    branch_relative_margin: np.ndarray
    status: np.ndarray


@dataclass
class DirectionalSufficientStats:
    sse_parallel: np.ndarray
    sse_perp: np.ndarray
    sum_parallel: np.ndarray
    sum_perp: np.ndarray
    axis_count: np.ndarray
    orientation_count: np.ndarray
    global_count: np.ndarray


@dataclass
class DirectionalMetricCurves:
    tangential_rmse: np.ndarray
    normal_rmse: np.ndarray
    tangential_bias: np.ndarray
    normal_bias: np.ndarray
    anisotropy_ratio: np.ndarray
    n_axis_valid_samples: np.ndarray
    axis_valid_fraction: np.ndarray
    n_orientation_valid_samples: np.ndarray
    orientation_valid_fraction: np.ndarray


@dataclass
class BranchAssignment:
    branch_name: np.ndarray
    axis_valid: np.ndarray
    status: np.ndarray


@dataclass
class JunctionDecisionEvent:
    window_row: int
    window_id: int
    rollout_start_frame: int
    track_id: int
    slot: int
    true_incoming_branch: str
    true_branch: str
    true_commitment_step: int
    pred_branch_by_model: dict[str, str]
    pred_commitment_step_by_model: dict[str, int]
    decision_status_by_model: dict[str, str]
    commitment_step_delay_by_model: dict[str, float]


@dataclass
class JunctionDecisionMetrics:
    n_true_junction_events: int
    n_correct: int
    n_wrong_branch: int
    n_no_commitment: int
    n_unclassifiable: int
    wrong_decision_rate: float
    noncommitment_rate: float
    mean_commitment_step_delay: float
    median_commitment_step_delay: float
    confusion_counts: dict[tuple[str, str], int]
    true_event_transition_counts: dict[tuple[str, str], int]


@dataclass
class TangentEstimate:
    tangent: np.ndarray | None
    normal: np.ndarray | None
    axis_valid: bool
    orientation_valid: bool
    quality: float
    distance_to_centerline: float
    branch_name: str
    second_nearest_branch_name: str
    second_nearest_branch_distance: float
    branch_distance_margin: float
    branch_relative_margin: float
    status: str


@dataclass
class ChannelAdmissibilityContext:
    window_id: np.ndarray
    rollout_start_frame: np.ndarray
    track_ids: np.ndarray
    future_frame: np.ndarray
    true_position: np.ndarray
    bbox_w: np.ndarray
    bbox_h: np.ndarray
    bbox_valid: np.ndarray
    geometry_valid_mask: np.ndarray
    true_outside_fraction: np.ndarray
    global_count: np.ndarray
    bbox_lookup_coverage: float
    bbox_missing_count: int
    bbox_candidate_count: int
    channel_mask_path: Path
    detections_csv_path: Path


@dataclass
class ChannelAdmissibilitySufficientStats:
    sum_outside: np.ndarray
    count: np.ndarray
    count_viol_2pct: np.ndarray
    count_any_viol: np.ndarray
    count_viol_10pct: np.ndarray
    sum_excess: np.ndarray
    sum_penalty_equivalent: np.ndarray
    global_count: np.ndarray


@dataclass
class ChannelAdmissibilityMetricCurves:
    mean_outside_fraction: np.ndarray
    median_outside_fraction: np.ndarray
    violation_rate_2pct: np.ndarray
    any_violation_rate: np.ndarray
    violation_rate_10pct: np.ndarray
    mean_excess_outside_fraction: np.ndarray
    mean_geometry_penalty_equivalent: np.ndarray
    true_mean_outside_fraction: np.ndarray
    true_violation_rate_2pct: np.ndarray
    true_violation_rate_10pct: np.ndarray
    excess_mean_outside_fraction_vs_truth: np.ndarray
    excess_violation_rate_2pct_vs_truth: np.ndarray
    n_geometry_valid_samples: np.ndarray
    geometry_valid_fraction: np.ndarray


@dataclass
class TrajectoryAdmissibilityMetrics:
    n_evaluable_trajectories: int
    n_trajectories_with_violation: int
    trajectory_violation_fraction: float
    mean_first_violation_step: float
    median_first_violation_step: float
    n_violation_episodes: int
    mean_violation_episode_duration: float
    median_violation_episode_duration: float
    persistent_violation_fraction: float


class AlignedRolloutWindowDataset(Dataset):
    """Canonical rollout dataset with externally fixed rollout starts and slot order."""

    def __init__(
        self,
        npz_path: Path,
        rollout_starts: np.ndarray,
        selected_track_ids: np.ndarray,
        T_history: int,
        T_future: int,
        max_droplets: int,
        normalization_stats,
        target_features: tuple[str, str] = ("vx", "vy"),
    ) -> None:
        self.npz_path = Path(npz_path)
        self.rollout_starts = np.asarray(rollout_starts, dtype=np.int64)
        self.selected_track_ids = np.asarray(selected_track_ids, dtype=np.int64)
        self.T_history = int(T_history)
        self.T_future = int(T_future)
        self.T_total = self.T_history + self.T_future
        self.max_droplets = int(max_droplets)
        self.target_features = tuple(target_features)
        self.normalization_stats = normalization_stats

        dataset = np.load(self.npz_path, allow_pickle=False)
        self.Z = dataset["Z"]
        self.mask = dataset["mask"]
        self.track_ids = dataset["track_ids"]
        self.frames = dataset["frames"]
        self.feature_names = [str(name) for name in dataset["feature_names"]]
        self.feature_indices = self._feature_indices(self.feature_names)
        self.target_indices = [self.feature_indices[name] for name in self.target_features]
        self.track_id_to_index = {int(track_id): index for index, track_id in enumerate(self.track_ids)}

        if self.selected_track_ids.shape != (len(self.rollout_starts), self.max_droplets):
            raise ValueError(
                "selected_track_ids must have shape "
                f"({len(self.rollout_starts)}, {self.max_droplets}), got {self.selected_track_ids.shape}"
            )

    def __len__(self) -> int:
        return len(self.rollout_starts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rollout_start = int(self.rollout_starts[index])
        frame_start = rollout_start - self.T_history
        droplet_ids = self.selected_track_ids[index].copy()

        history_x = np.zeros((self.T_history, self.max_droplets, len(self.feature_names)), dtype=np.float32)
        future_y = np.zeros((self.T_future, self.max_droplets, len(self.target_features)), dtype=np.float32)
        history_mask = np.zeros((self.T_history, self.max_droplets), dtype=bool)
        future_mask = np.zeros((self.T_future, self.max_droplets), dtype=bool)

        history_slice = slice(frame_start, rollout_start)
        future_slice = slice(rollout_start, rollout_start + self.T_future)

        for slot_index, track_id in enumerate(droplet_ids):
            if track_id < 0:
                continue
            droplet_index = self.track_id_to_index.get(int(track_id))
            if droplet_index is None:
                continue

            raw_history = self.Z[droplet_index, history_slice, :]
            raw_future = self.Z[droplet_index, future_slice, :][:, self.target_indices]
            raw_history_mask = self.mask[droplet_index, history_slice]
            raw_future_mask = self.mask[droplet_index, future_slice]
            raw_future_mask = raw_future_mask & np.isfinite(raw_future).all(axis=1)

            history_x[:, slot_index, :] = np.nan_to_num(raw_history, nan=0.0)
            future_y[:, slot_index, :] = np.nan_to_num(raw_future, nan=0.0)
            history_mask[:, slot_index] = raw_history_mask
            future_mask[:, slot_index] = raw_future_mask

        self._normalize_in_place(history_x, future_y, history_mask, future_mask)

        return {
            "history_x": torch.as_tensor(history_x, dtype=torch.float32),
            "future_y": torch.as_tensor(future_y, dtype=torch.float32),
            "history_mask": torch.as_tensor(history_mask, dtype=torch.bool),
            "future_mask": torch.as_tensor(future_mask, dtype=torch.bool),
            "droplet_ids": torch.as_tensor(droplet_ids, dtype=torch.long),
            "frame_start": torch.as_tensor(frame_start, dtype=torch.long),
            "window_id": torch.as_tensor(index, dtype=torch.long),
            "rollout_start_frame": torch.as_tensor(rollout_start, dtype=torch.long),
        }

    def _feature_indices(self, feature_names: list[str]) -> dict[str, int]:
        feature_indices = {name: index for index, name in enumerate(feature_names)}
        for required_name in ["x", "y", "vx", "vy", "circularity"]:
            if required_name not in feature_indices:
                raise KeyError(f"Missing required feature: {required_name}")
        for target_name in self.target_features:
            if target_name not in feature_indices:
                raise KeyError(f"Missing target feature: {target_name}")
        return feature_indices

    def _normalize_in_place(self, history_x, future_y, history_mask, future_mask) -> None:
        input_mean = np.asarray(self.normalization_stats["input_mean"], dtype=np.float32)
        input_std = np.asarray(self.normalization_stats["input_std"], dtype=np.float32)
        target_mean = np.asarray(self.normalization_stats["target_mean"], dtype=np.float32)
        target_std = np.asarray(self.normalization_stats["target_std"], dtype=np.float32)

        valid_history = history_mask[:, :, None] & np.isfinite(history_x)
        history_x[valid_history] = ((history_x - input_mean) / input_std)[valid_history]
        history_x[~valid_history] = 0.0

        valid_future = future_mask[:, :, None] & np.isfinite(future_y)
        future_y[valid_future] = ((future_y - target_mean) / target_std)[valid_future]
        future_y[~valid_future] = 0.0


def build_validation_rollout_starts(npz_path: Path, stride: int, horizon: int, source_history: int) -> np.ndarray:
    dataset = np.load(npz_path, allow_pickle=False)
    total_frames = len(dataset["frames"])
    source_total = int(source_history) + int(horizon)
    all_source_starts = np.arange(0, total_frames - source_total + 1, stride, dtype=np.int64)
    train_end = int(0.70 * len(all_source_starts))
    val_end = int(0.85 * len(all_source_starts))
    source_val_starts = all_source_starts[train_end:val_end]
    return source_val_starts + int(source_history)


def build_common_track_slots(
    npz_path: Path,
    rollout_starts: np.ndarray,
    source_history: int,
    horizon: int,
    max_droplets: int,
) -> np.ndarray:
    dataset = np.load(npz_path, allow_pickle=False)
    Z = dataset["Z"]
    mask = dataset["mask"]
    track_ids = dataset["track_ids"]
    feature_names = [str(name) for name in dataset["feature_names"]]
    x_index = feature_names.index("x")

    selected_by_window = np.full((len(rollout_starts), max_droplets), -1, dtype=np.int64)
    for window_index, rollout_start in enumerate(rollout_starts):
        start = int(rollout_start) - int(source_history)
        stop = int(rollout_start) + int(horizon)
        window_mask = mask[:, start:stop]
        selected = np.flatnonzero(window_mask.any(axis=1))
        sort_keys = []
        for droplet_index in selected:
            valid_offsets = np.flatnonzero(window_mask[droplet_index])
            first_offset = int(valid_offsets[0])
            first_frame = start + first_offset
            first_x = float(Z[droplet_index, first_frame, x_index])
            sort_keys.append((first_frame, first_x, int(track_ids[droplet_index]), int(droplet_index)))
        sort_keys.sort()
        ordered_track_ids = [int(track_ids[item[3]]) for item in sort_keys[:max_droplets]]
        selected_by_window[window_index, : len(ordered_track_ids)] = ordered_track_ids
    return selected_by_window


class RolloutModelAdapter:
    def __init__(
        self,
        name: str,
        checkpoint_path: Path,
        rollout_fn: Callable,
        device: torch.device,
    ) -> None:
        self.name = name
        self.checkpoint_path = Path(checkpoint_path)
        self.rollout_fn = rollout_fn
        self.device = device
        self.checkpoint = torch.load(self.checkpoint_path, map_location=device, weights_only=False)
        self.model_config = dict(self.checkpoint["model_config"])
        self.history_length = int(self.model_config["T_history"])
        self.horizon = int(self.checkpoint.get("rollout_horizon", ROLLOUT_HORIZON))
        self.loss_alpha = float(self.checkpoint.get("loss_alpha", 2.0))
        self.normalization_stats = self.checkpoint["normalization_stats"]
        self.model = CanonicalRolloutTransformer(**self.model_config).to(device)
        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.model.eval()
        self.weights = self._rollout_weights(self.horizon, self.loss_alpha, device)
        self.dataset: AlignedRolloutWindowDataset | None = None

    def attach_dataset(self, dataset: AlignedRolloutWindowDataset) -> None:
        self.dataset = dataset

    def predict_rollout(self, batch: dict[str, torch.Tensor]) -> RolloutPrediction:
        if self.dataset is None:
            raise RuntimeError(f"Dataset has not been attached for adapter {self.name}.")
        with torch.inference_mode():
            rollout = self.rollout_fn(
                model=self.model,
                batch=batch,
                dataset=self.dataset,
                normalization_stats=self.normalization_stats,
                weights=self.weights,
            )
        return RolloutPrediction(
            model_name=self.name,
            window_id=batch["window_id"].detach().cpu().numpy(),
            rollout_start_frame=batch["rollout_start_frame"].detach().cpu().numpy(),
            frame_start=batch["frame_start"].detach().cpu().numpy(),
            track_ids=batch["droplet_ids"].detach().cpu().numpy(),
            pred_position=rollout["pred_position"].detach().cpu().numpy(),
            true_position=rollout["true_position"].detach().cpu().numpy(),
            pred_velocity=rollout["pred_velocity"].detach().cpu().numpy(),
            true_velocity=rollout["true_velocity"].detach().cpu().numpy(),
            valid_mask=rollout["mask"].detach().cpu().numpy().astype(bool),
            boundary_mask=rollout["boundary_mask"].detach().cpu().numpy().astype(bool),
        )

    @staticmethod
    def _rollout_weights(horizon: int, alpha: float, device: torch.device) -> torch.Tensor:
        if horizon == 1:
            return torch.ones(1, dtype=torch.float32, device=device)
        step_ids = torch.arange(horizon, dtype=torch.float32, device=device)
        return 1.0 + float(alpha) * step_ids / float(horizon - 1)


class HistoryRolloutModelAdapter(RolloutModelAdapter):
    def __init__(self, name: str, checkpoint_path: Path, device: torch.device) -> None:
        super().__init__(name, checkpoint_path, history_rollout.boundary_conditioned_rollout, device)


class MarkovianRolloutModelAdapter(RolloutModelAdapter):
    def __init__(self, name: str, checkpoint_path: Path, device: torch.device) -> None:
        super().__init__(name, checkpoint_path, markovian_rollout.boundary_conditioned_rollout, device)


class RolloutModelComparator:
    def __init__(
        self,
        adapters: dict[str, RolloutModelAdapter],
        npz_path: Path,
        output_dir: Path,
        batch_size: int = 4,
        n_bootstrap: int = 1000,
        seed: int = 123,
        max_windows: int | None = None,
        stride: int = STRIDE,
    ) -> None:
        self.adapters = adapters
        self.npz_path = Path(npz_path)
        self.output_dir = Path(output_dir)
        self.batch_size = int(batch_size)
        self.n_bootstrap = int(n_bootstrap)
        self.seed = int(seed)
        self.max_windows = max_windows
        self.stride = int(stride)
        self.predictions: dict[str, RolloutPrediction] = {}
        self.stepwise_metrics: dict[str, StepwiseMetricCurves] = {}
        self.integrated_metrics: dict[str, IntegratedMetrics] = {}
        self.bootstrap: dict[str, dict[str, np.ndarray]] = {}
        self.bootstrap_indices: np.ndarray | None = None
        self.directional_basis: DirectionalBasis | None = None
        self.directional_stats: dict[str, DirectionalSufficientStats] = {}
        self.directional_metrics: dict[str, DirectionalMetricCurves] = {}
        self.directional_bootstrap: dict[str, dict[str, np.ndarray]] = {}
        self.channel_context: ChannelAdmissibilityContext | None = None
        self.channel_outside_fraction: dict[str, np.ndarray] = {}
        self.channel_stats: dict[str, ChannelAdmissibilitySufficientStats] = {}
        self.channel_metrics: dict[str, ChannelAdmissibilityMetricCurves] = {}
        self.channel_bootstrap: dict[str, dict[str, np.ndarray]] = {}
        self.channel_trajectory_metrics: dict[str, TrajectoryAdmissibilityMetrics] = {}
        self.junction_events: list[JunctionDecisionEvent] = []
        self.junction_metrics: dict[str, JunctionDecisionMetrics] = {}
        self.junction_bootstrap: dict[str, dict[str, tuple[float, float]]] = {}

    def run_inference(self) -> dict[str, RolloutPrediction]:
        print("Building aligned validation rollout windows...", flush=True)
        rollout_starts = build_validation_rollout_starts(
            self.npz_path,
            stride=self.stride,
            horizon=ROLLOUT_HORIZON,
            source_history=SOURCE_HISTORY_LENGTH,
        )
        if self.max_windows is not None:
            rollout_starts = rollout_starts[: int(self.max_windows)]
        print(f"Validation rollout windows: {len(rollout_starts)}", flush=True)
        print("Selecting common droplet slots for all models...", flush=True)
        selected_track_ids = build_common_track_slots(
            self.npz_path,
            rollout_starts,
            source_history=SOURCE_HISTORY_LENGTH,
            horizon=ROLLOUT_HORIZON,
            max_droplets=MAX_DROPLETS,
        )

        total_models = len(self.adapters)
        for model_index, (model_name, adapter) in enumerate(self.adapters.items(), start=1):
            print(
                f"[{model_index}/{total_models}] Running model '{model_name}' "
                f"(history={adapter.history_length}, horizon={adapter.horizon})",
                flush=True,
            )
            dataset = AlignedRolloutWindowDataset(
                npz_path=self.npz_path,
                rollout_starts=rollout_starts,
                selected_track_ids=selected_track_ids,
                T_history=adapter.history_length,
                T_future=adapter.horizon,
                max_droplets=MAX_DROPLETS,
                normalization_stats=adapter.normalization_stats,
            )
            adapter.attach_dataset(dataset)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
            chunks = []
            total_batches = len(loader)
            for batch_index, batch in enumerate(loader, start=1):
                device_batch = {
                    key: value.to(adapter.device) if torch.is_tensor(value) else value
                    for key, value in batch.items()
                }
                chunks.append(adapter.predict_rollout(device_batch))
                remaining = total_batches - batch_index
                print(
                    f"  {model_name}: batch {batch_index}/{total_batches} "
                    f"processed, remaining {remaining}",
                    flush=True,
                )
            self.predictions[model_name] = concatenate_predictions(model_name, chunks)
            print(f"[{model_index}/{total_models}] Finished model '{model_name}'", flush=True)

        print("Validating common outputs across models...", flush=True)
        self.validate_common_outputs()
        return self.predictions

    def save_predictions(self, path: Path | None = None) -> Path:
        if path is None:
            path = self.output_dir / "rollout_predictions.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"model_names": np.asarray(list(self.predictions), dtype=str)}
        for model_name, prediction in self.predictions.items():
            prefix = f"{model_name}__"
            arrays[prefix + "window_id"] = prediction.window_id
            arrays[prefix + "rollout_start_frame"] = prediction.rollout_start_frame
            arrays[prefix + "frame_start"] = prediction.frame_start
            arrays[prefix + "track_ids"] = prediction.track_ids
            arrays[prefix + "pred_position"] = prediction.pred_position.astype(np.float32)
            arrays[prefix + "true_position"] = prediction.true_position.astype(np.float32)
            arrays[prefix + "pred_velocity"] = prediction.pred_velocity.astype(np.float32)
            arrays[prefix + "true_velocity"] = prediction.true_velocity.astype(np.float32)
            arrays[prefix + "valid_mask"] = prediction.valid_mask
            arrays[prefix + "boundary_mask"] = prediction.boundary_mask
        np.savez_compressed(path, **arrays)
        return path

    def load_predictions(self, path: Path) -> dict[str, RolloutPrediction]:
        loaded = load_predictions_npz(path)
        self.predictions = loaded
        self.validate_common_outputs()
        return self.predictions

    def compute_metrics(self) -> None:
        print("Computing stepwise and integrated metrics...", flush=True)
        for model_name, prediction in self.predictions.items():
            stepwise = compute_stepwise_rmse(prediction)
            integrated = compute_integrated_metrics(stepwise)
            self.stepwise_metrics[model_name] = stepwise
            self.integrated_metrics[model_name] = integrated
            print(f"  metrics complete for '{model_name}'", flush=True)

    def bootstrap_metrics(self) -> None:
        if not self.predictions:
            raise RuntimeError("No predictions available. Run inference first.")

        first_prediction = next(iter(self.predictions.values()))
        n_windows = int(first_prediction.window_id.shape[0])
        for model_name, prediction in self.predictions.items():
            if int(prediction.window_id.shape[0]) != n_windows:
                raise AssertionError(
                    f"{model_name}: bootstrap window count differs from the first model "
                    f"({prediction.window_id.shape[0]} != {n_windows})."
                )

        rng = np.random.default_rng(self.seed)
        print(
            f"Generating shared bootstrap index matrix: "
            f"{self.n_bootstrap} replicates x {n_windows} windows",
            flush=True,
        )
        self.bootstrap_indices = rng.integers(
            0,
            n_windows,
            size=(self.n_bootstrap, n_windows),
        )
        for model_name, prediction in self.predictions.items():
            print(f"  bootstrapping '{model_name}'...", flush=True)
            self.bootstrap[model_name] = bootstrap_prediction_metrics(prediction, self.bootstrap_indices)
            print(f"  bootstrap complete for '{model_name}'", flush=True)

    def compute_directional_metrics(
        self,
        centerline_csv: Path,
        pca_half_window: int = 8,
        min_quality: float = 3.0,
        max_centerline_distance: float = 30.0,
        min_orientation_speed: float = 1.0e-3,
        min_branch_distance_margin: float | None = None,
        min_branch_relative_margin: float | None = None,
    ) -> None:
        if not self.predictions:
            raise RuntimeError("No predictions available for directional analysis.")
        if self.bootstrap_indices is None:
            self.bootstrap_metrics()

        reference = next(iter(self.predictions.values()))
        print("Building true-position centerline tangent basis...", flush=True)
        estimator = CenterlineTangentEstimator(
            centerline_csv=centerline_csv,
            pca_half_window=pca_half_window,
            min_quality=min_quality,
            max_centerline_distance=max_centerline_distance,
            min_orientation_speed=min_orientation_speed,
            min_branch_distance_margin=min_branch_distance_margin,
            min_branch_relative_margin=min_branch_relative_margin,
        )
        self.directional_basis = estimator.build_basis(reference)
        summarize_tangent_basis(self.directional_basis, reference.metric_mask)

        print("Computing directional sufficient statistics and metrics...", flush=True)
        for model_name, prediction in self.predictions.items():
            stats = compute_directional_sufficient_stats(prediction, self.directional_basis)
            curves = directional_metrics_from_stats(stats)
            self.directional_stats[model_name] = stats
            self.directional_metrics[model_name] = curves
            self.directional_bootstrap[model_name] = bootstrap_directional_metrics(stats, self.bootstrap_indices)
            print(f"  directional metrics complete for '{model_name}'", flush=True)

    def save_directional_outputs(self) -> tuple[Path, Path]:
        csv_path = self.output_dir / "directional_error_metrics.csv"
        data_path = self.output_dir / "directional_error_data.npz"
        write_directional_metrics_csv(csv_path, self.directional_metrics, self.directional_bootstrap)
        save_directional_data(data_path, self.directional_basis, self.directional_stats)
        return csv_path, data_path

    def plot_directional_metrics(self) -> tuple[tuple[Path, Path], tuple[Path, Path], tuple[Path, Path]]:
        tangential = plot_directional_metric(
            self.output_dir,
            self.directional_metrics,
            self.directional_bootstrap,
            metric_name="tangential_rmse",
            ylabel="Tangential RMSE [pixels]",
            stem="tangential_rmse_vs_rollout_step",
        )
        normal = plot_directional_metric(
            self.output_dir,
            self.directional_metrics,
            self.directional_bootstrap,
            metric_name="normal_rmse",
            ylabel="Normal RMSE [pixels]",
            stem="normal_rmse_vs_rollout_step",
        )
        bias = plot_directional_metric(
            self.output_dir,
            self.directional_metrics,
            self.directional_bootstrap,
            metric_name="tangential_bias",
            ylabel="Mean signed tangential error [pixels]",
            stem="tangential_bias_vs_rollout_step",
            zero_line=True,
        )
        return tangential, normal, bias

    def compute_channel_admissibility_metrics(
        self,
        channel_mask_path: Path,
        detections_csv_path: Path,
        tolerance: float = GEOMETRY_TOLERANCE,
    ) -> None:
        if not self.predictions:
            raise RuntimeError("No predictions available for channel admissibility analysis.")
        if self.bootstrap_indices is None:
            self.bootstrap_metrics()

        reference = next(iter(self.predictions.values()))
        print("Building common channel footprint context...", flush=True)
        self.channel_context = build_channel_admissibility_context(
            reference=reference,
            channel_mask_path=channel_mask_path,
            detections_csv_path=detections_csv_path,
        )
        summarize_channel_context(self.channel_context, tolerance=tolerance)

        true_stats = compute_channel_admissibility_sufficient_stats(
            self.channel_context.true_outside_fraction,
            self.channel_context.geometry_valid_mask,
            self.channel_context.global_count,
            tolerance=tolerance,
        )
        true_curves = channel_admissibility_metrics_from_stats(
            true_stats,
            self.channel_context.true_outside_fraction,
            self.channel_context.geometry_valid_mask,
            true_stats,
            self.channel_context.true_outside_fraction,
            self.channel_context.geometry_valid_mask,
            tolerance=tolerance,
        )

        print("Computing predicted channel admissibility metrics...", flush=True)
        for model_name, prediction in self.predictions.items():
            outside = compute_model_outside_fraction(prediction, self.channel_context)
            stats = compute_channel_admissibility_sufficient_stats(
                outside,
                self.channel_context.geometry_valid_mask,
                self.channel_context.global_count,
                tolerance=tolerance,
            )
            curves = channel_admissibility_metrics_from_stats(
                stats,
                outside,
                self.channel_context.geometry_valid_mask,
                true_stats,
                self.channel_context.true_outside_fraction,
                self.channel_context.geometry_valid_mask,
                tolerance=tolerance,
            )
            self.channel_outside_fraction[model_name] = outside
            self.channel_stats[model_name] = stats
            self.channel_metrics[model_name] = curves
            self.channel_bootstrap[model_name] = bootstrap_channel_admissibility_metrics(stats, self.bootstrap_indices)
            self.channel_trajectory_metrics[model_name] = compute_trajectory_admissibility_metrics(
                outside,
                self.channel_context.geometry_valid_mask,
                tolerance=tolerance,
            )
            print(f"  channel admissibility complete for '{model_name}'", flush=True)

        validate_common_channel_masks(self.predictions, self.channel_context)
        validate_channel_sanity_checks(self.channel_context)

    def compute_junction_decision_metrics(
        self,
        centerline_csv: Path,
        pca_half_window: int = 8,
        min_quality: float = 3.0,
        max_centerline_distance: float = 30.0,
        min_orientation_speed: float = 1.0e-3,
        min_branch_distance_margin: float | None = None,
        min_branch_relative_margin: float | None = None,
        outgoing_branches: tuple[str, ...] = ("left", "right"),
        incoming_branches: tuple[str, ...] = ("inlet", "outlet"),
        commitment_steps: int = 3,
    ) -> None:
        if not self.predictions:
            raise RuntimeError("No predictions available for junction decision analysis.")
        if self.bootstrap_indices is None:
            self.bootstrap_metrics()

        summarize_junction_decision_logic(
            outgoing_branches=outgoing_branches,
            incoming_branches=incoming_branches,
            commitment_steps=commitment_steps,
        )
        estimator = CenterlineTangentEstimator(
            centerline_csv=centerline_csv,
            pca_half_window=pca_half_window,
            min_quality=min_quality,
            max_centerline_distance=max_centerline_distance,
            min_orientation_speed=min_orientation_speed,
            min_branch_distance_margin=min_branch_distance_margin,
            min_branch_relative_margin=min_branch_relative_margin,
        )
        reference = next(iter(self.predictions.values()))
        start_time = time.time()
        print("Assigning true trajectories to centerline branches...", flush=True)
        true_assignment = assign_centerline_branches(
            estimator,
            reference.true_position,
            reference.true_velocity,
            reference.metric_mask,
        )
        print(f"  true branch assignment complete in {time.time() - start_time:.2f}s", flush=True)
        print("Assigning predicted trajectories to centerline branches...", flush=True)
        predicted_assignments: dict[str, BranchAssignment] = {}
        for model_name, prediction in self.predictions.items():
            model_start = time.time()
            predicted_assignments[model_name] = assign_centerline_branches(
                estimator,
                prediction.pred_position,
                prediction.pred_velocity,
                prediction.metric_mask,
            )
            print(f"  {model_name}: branch assignment complete in {time.time() - model_start:.2f}s", flush=True)

        print("Identifying true junction events and classifying model decisions...", flush=True)
        event_start = time.time()
        legacy_event_count = count_legacy_loose_junction_events(
            true_assignment=true_assignment,
            metric_mask=reference.metric_mask,
            outgoing_branches=tuple(outgoing_branches),
            commitment_steps=commitment_steps,
        )
        self.junction_events = identify_junction_decision_events(
            reference=reference,
            true_assignment=true_assignment,
            predicted_assignments=predicted_assignments,
            outgoing_branches=outgoing_branches,
            incoming_branches=incoming_branches,
            commitment_steps=commitment_steps,
        )
        print(f"  junction event classification complete in {time.time() - event_start:.2f}s", flush=True)
        summarize_true_junction_event_counts(self.junction_events, legacy_event_count)
        self.junction_metrics = {
            model_name: compute_junction_decision_metrics_for_model(self.junction_events, model_name)
            for model_name in self.predictions
        }
        self.junction_bootstrap = {
            model_name: bootstrap_junction_decision_rates(
                self.junction_events,
                model_name,
                self.bootstrap_indices,
                n_windows=int(reference.window_id.shape[0]),
            )
            for model_name in self.predictions
        }
        summarize_junction_decision_metrics(self.junction_metrics)

    def save_junction_decision_outputs(self) -> tuple[Path, Path]:
        metrics_path = self.output_dir / "junction_decision_metrics.csv"
        events_path = self.output_dir / "junction_decision_events.csv"
        write_junction_decision_metrics_csv(metrics_path, self.junction_metrics, self.junction_bootstrap)
        write_junction_decision_events_csv(events_path, self.junction_events, list(self.predictions))
        return metrics_path, events_path

    def plot_junction_decision_outputs(self) -> tuple[tuple[Path, Path], dict[str, tuple[Path, Path]]]:
        outcome_paths = plot_junction_decision_outcomes(
            output_dir=self.output_dir,
            metrics=self.junction_metrics,
        )
        confusion_paths = {
            model_name: plot_junction_confusion_matrix(
                output_dir=self.output_dir,
                events=self.junction_events,
                model_name=model_name,
            )
            for model_name in self.predictions
        }
        return outcome_paths, confusion_paths

    def save_channel_admissibility_outputs(self) -> tuple[Path, Path, Path]:
        csv_path = self.output_dir / "channel_admissibility_metrics.csv"
        trajectory_csv_path = self.output_dir / "channel_admissibility_trajectory_metrics.csv"
        data_path = self.output_dir / "channel_admissibility_data.npz"
        write_channel_admissibility_metrics_csv(csv_path, self.channel_metrics, self.channel_bootstrap)
        write_channel_trajectory_metrics_csv(trajectory_csv_path, self.channel_trajectory_metrics)
        save_channel_admissibility_data(
            data_path,
            self.channel_context,
            self.channel_outside_fraction,
        )
        return csv_path, trajectory_csv_path, data_path

    def plot_channel_admissibility_metrics(self) -> tuple[tuple[Path, Path], tuple[Path, Path], tuple[Path, Path]]:
        mean_plot = plot_channel_metric(
            output_dir=self.output_dir,
            metrics=self.channel_metrics,
            bootstrap=self.channel_bootstrap,
            metric_name="mean_outside_fraction",
            truth_metric_name="true_mean_outside_fraction",
            ylabel="Mean footprint outside channel fraction",
            stem="mean_outside_fraction_vs_rollout_step",
        )
        rate_plot = plot_channel_metric(
            output_dir=self.output_dir,
            metrics=self.channel_metrics,
            bootstrap=self.channel_bootstrap,
            metric_name="violation_rate_2pct",
            truth_metric_name="true_violation_rate_2pct",
            ylabel="Fraction of footprints with >2% outside-channel area",
            stem="channel_violation_rate_2pct_vs_rollout_step",
        )
        penalty_plot = plot_channel_metric(
            output_dir=self.output_dir,
            metrics=self.channel_metrics,
            bootstrap=self.channel_bootstrap,
            metric_name="mean_geometry_penalty_equivalent",
            truth_metric_name=None,
            ylabel="Mean geometry penalty equivalent",
            stem="geometry_penalty_equivalent_vs_rollout_step",
        )
        return mean_plot, rate_plot, penalty_plot

    def save_metrics(self) -> tuple[Path, Path]:
        stepwise_path = self.output_dir / "stepwise_metrics.csv"
        integrated_path = self.output_dir / "integrated_metrics.csv"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        write_stepwise_metrics_csv(stepwise_path, self.stepwise_metrics, self.bootstrap)
        write_integrated_metrics_csv(integrated_path, self.integrated_metrics, self.bootstrap)
        return stepwise_path, integrated_path

    def plot_position_rmse(self) -> tuple[Path, Path]:
        return plot_stepwise_metric(
            output_dir=self.output_dir,
            stepwise_metrics=self.stepwise_metrics,
            bootstrap=self.bootstrap,
            metric_name="position_rmse",
            ylabel="Position RMSE [pixels]",
            stem="position_rmse_vs_rollout_step",
        )

    def plot_velocity_rmse(self) -> tuple[Path, Path]:
        return plot_stepwise_metric(
            output_dir=self.output_dir,
            stepwise_metrics=self.stepwise_metrics,
            bootstrap=self.bootstrap,
            metric_name="velocity_rmse",
            ylabel="Velocity RMSE [pixels/frame]",
            stem="velocity_rmse_vs_rollout_step",
        )

    def validate_common_outputs(self) -> None:
        if not self.predictions:
            raise RuntimeError("No predictions were produced.")
        reference_name = next(iter(self.predictions))
        reference = self.predictions[reference_name]
        for model_name, prediction in self.predictions.items():
            if prediction.window_id.shape != reference.window_id.shape:
                raise AssertionError(f"{model_name}: window count differs from {reference_name}.")
            np.testing.assert_array_equal(prediction.window_id, reference.window_id)
            np.testing.assert_array_equal(prediction.rollout_start_frame, reference.rollout_start_frame)
            np.testing.assert_array_equal(prediction.track_ids, reference.track_ids)
            np.testing.assert_array_equal(prediction.valid_mask, reference.valid_mask)
            np.testing.assert_array_equal(prediction.boundary_mask, reference.boundary_mask)
            compare_mask = reference.metric_mask
            np.testing.assert_allclose(
                prediction.true_position[compare_mask],
                reference.true_position[compare_mask],
                rtol=0,
                atol=1e-5,
                err_msg=f"{model_name}: true positions differ from {reference_name}.",
            )
            np.testing.assert_allclose(
                prediction.true_velocity[compare_mask],
                reference.true_velocity[compare_mask],
                rtol=0,
                atol=1e-5,
                err_msg=f"{model_name}: true velocities differ from {reference_name}.",
            )
            if not np.isfinite(prediction.pred_position[prediction.metric_mask]).all():
                raise AssertionError(f"{model_name}: non-finite predicted positions under metric mask.")
            if not np.isfinite(prediction.pred_velocity[prediction.metric_mask]).all():
                raise AssertionError(f"{model_name}: non-finite predicted velocities under metric mask.")
            if np.any(prediction.valid_mask & prediction.boundary_mask & prediction.metric_mask):
                raise AssertionError(f"{model_name}: boundary samples leaked into the metric mask.")


def concatenate_predictions(model_name: str, chunks: list[RolloutPrediction]) -> RolloutPrediction:
    if not chunks:
        raise ValueError(f"No prediction chunks for {model_name}.")
    return RolloutPrediction(
        model_name=model_name,
        window_id=np.concatenate([chunk.window_id for chunk in chunks], axis=0),
        rollout_start_frame=np.concatenate([chunk.rollout_start_frame for chunk in chunks], axis=0),
        frame_start=np.concatenate([chunk.frame_start for chunk in chunks], axis=0),
        track_ids=np.concatenate([chunk.track_ids for chunk in chunks], axis=0),
        pred_position=np.concatenate([chunk.pred_position for chunk in chunks], axis=0),
        true_position=np.concatenate([chunk.true_position for chunk in chunks], axis=0),
        pred_velocity=np.concatenate([chunk.pred_velocity for chunk in chunks], axis=0),
        true_velocity=np.concatenate([chunk.true_velocity for chunk in chunks], axis=0),
        valid_mask=np.concatenate([chunk.valid_mask for chunk in chunks], axis=0),
        boundary_mask=np.concatenate([chunk.boundary_mask for chunk in chunks], axis=0),
    )


def compute_stepwise_rmse(prediction: RolloutPrediction, window_indices: np.ndarray | None = None) -> StepwiseMetricCurves:
    pred_position = prediction.pred_position
    true_position = prediction.true_position
    pred_velocity = prediction.pred_velocity
    true_velocity = prediction.true_velocity
    mask = prediction.metric_mask
    if window_indices is not None:
        pred_position = pred_position[window_indices]
        true_position = true_position[window_indices]
        pred_velocity = pred_velocity[window_indices]
        true_velocity = true_velocity[window_indices]
        mask = mask[window_indices]

    position_error = pred_position - true_position
    velocity_error = pred_velocity - true_velocity
    if not np.isfinite(position_error[mask]).all():
        raise ValueError(f"{prediction.model_name}: non-finite position error under metric mask.")
    if not np.isfinite(velocity_error[mask]).all():
        raise ValueError(f"{prediction.model_name}: non-finite velocity error under metric mask.")
    metric_mask = mask

    counts = metric_mask.sum(axis=(0, 2)).astype(np.int64)
    if np.any(counts == 0):
        zero_steps = np.flatnonzero(counts == 0) + 1
        raise ValueError(f"Zero valid samples for rollout steps: {zero_steps.tolist()}")

    position_sse = np.where(metric_mask, np.sum(position_error**2, axis=-1), 0.0).sum(axis=(0, 2))
    velocity_sse = np.where(metric_mask, np.sum(velocity_error**2, axis=-1), 0.0).sum(axis=(0, 2))
    vx_sse = np.where(metric_mask, velocity_error[..., 0] ** 2, 0.0).sum(axis=(0, 2))
    vy_sse = np.where(metric_mask, velocity_error[..., 1] ** 2, 0.0).sum(axis=(0, 2))

    return StepwiseMetricCurves(
        position_rmse=np.sqrt(position_sse / counts),
        velocity_rmse=np.sqrt(velocity_sse / counts),
        vx_rmse=np.sqrt(vx_sse / counts),
        vy_rmse=np.sqrt(vy_sse / counts),
        n_valid_samples=counts,
    )


def compute_rmse_sufficient_stats(prediction: RolloutPrediction) -> RmseSufficientStats:
    position_error = prediction.pred_position - prediction.true_position
    velocity_error = prediction.pred_velocity - prediction.true_velocity
    mask = prediction.metric_mask
    if not np.isfinite(position_error[mask]).all():
        raise ValueError(f"{prediction.model_name}: non-finite position error under metric mask.")
    if not np.isfinite(velocity_error[mask]).all():
        raise ValueError(f"{prediction.model_name}: non-finite velocity error under metric mask.")
    return RmseSufficientStats(
        position_sse=np.where(mask, np.sum(position_error**2, axis=-1), 0.0).sum(axis=2),
        velocity_sse=np.where(mask, np.sum(velocity_error**2, axis=-1), 0.0).sum(axis=2),
        vx_sse=np.where(mask, velocity_error[..., 0] ** 2, 0.0).sum(axis=2),
        vy_sse=np.where(mask, velocity_error[..., 1] ** 2, 0.0).sum(axis=2),
        count=mask.sum(axis=2).astype(np.int64),
    )


def stepwise_rmse_from_stats(stats: RmseSufficientStats) -> StepwiseMetricCurves:
    counts = stats.count.sum(axis=0).astype(np.int64)
    if np.any(counts == 0):
        zero_steps = np.flatnonzero(counts == 0) + 1
        raise ValueError(f"Zero valid samples for rollout steps: {zero_steps.tolist()}")
    return StepwiseMetricCurves(
        position_rmse=np.sqrt(stats.position_sse.sum(axis=0) / counts),
        velocity_rmse=np.sqrt(stats.velocity_sse.sum(axis=0) / counts),
        vx_rmse=np.sqrt(stats.vx_sse.sum(axis=0) / counts),
        vy_rmse=np.sqrt(stats.vy_sse.sum(axis=0) / counts),
        n_valid_samples=counts,
    )


def compute_integrated_metrics(stepwise: StepwiseMetricCurves) -> IntegratedMetrics:
    integrated = float(np.mean(stepwise.position_rmse))
    final_step = float(stepwise.position_rmse[-1])
    return IntegratedMetrics(
        integrated_position_rmse=integrated,
        final_step_position_rmse=final_step,
    )


def bootstrap_prediction_metrics(
    prediction: RolloutPrediction,
    bootstrap_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    n_bootstrap, n_windows = bootstrap_indices.shape
    if n_windows != prediction.window_id.shape[0]:
        raise ValueError(
            f"{prediction.model_name}: bootstrap index width {n_windows} does not match "
            f"prediction windows {prediction.window_id.shape[0]}."
        )
    horizon = prediction.pred_position.shape[1]
    curves = {
        "position_rmse": np.empty((n_bootstrap, horizon), dtype=np.float64),
        "velocity_rmse": np.empty((n_bootstrap, horizon), dtype=np.float64),
        "vx_rmse": np.empty((n_bootstrap, horizon), dtype=np.float64),
        "vy_rmse": np.empty((n_bootstrap, horizon), dtype=np.float64),
        "integrated_position_rmse": np.empty((n_bootstrap,), dtype=np.float64),
        "final_step_position_rmse": np.empty((n_bootstrap,), dtype=np.float64),
    }
    stats = compute_rmse_sufficient_stats(prediction)
    for replicate in range(n_bootstrap):
        indices = bootstrap_indices[replicate]
        stepwise = stepwise_rmse_from_stats(
            RmseSufficientStats(
                position_sse=stats.position_sse[indices],
                velocity_sse=stats.velocity_sse[indices],
                vx_sse=stats.vx_sse[indices],
                vy_sse=stats.vy_sse[indices],
                count=stats.count[indices],
            )
        )
        integrated = compute_integrated_metrics(stepwise)
        curves["position_rmse"][replicate] = stepwise.position_rmse
        curves["velocity_rmse"][replicate] = stepwise.velocity_rmse
        curves["vx_rmse"][replicate] = stepwise.vx_rmse
        curves["vy_rmse"][replicate] = stepwise.vy_rmse
        curves["integrated_position_rmse"][replicate] = integrated.integrated_position_rmse
        curves["final_step_position_rmse"][replicate] = integrated.final_step_position_rmse
    return curves


def percentile_ci(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.percentile(samples, 2.5, axis=0), np.percentile(samples, 97.5, axis=0)


def write_stepwise_metrics_csv(
    path: Path,
    metrics: dict[str, StepwiseMetricCurves],
    bootstrap: dict[str, dict[str, np.ndarray]],
) -> None:
    fieldnames = [
        "model",
        "rollout_step",
        "position_rmse",
        "position_rmse_ci_low",
        "position_rmse_ci_high",
        "velocity_rmse",
        "velocity_rmse_ci_low",
        "velocity_rmse_ci_high",
        "vx_rmse",
        "vx_rmse_ci_low",
        "vx_rmse_ci_high",
        "vy_rmse",
        "vy_rmse_ci_low",
        "vy_rmse_ci_high",
        "n_valid_samples",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for model_name, stepwise in metrics.items():
            ci = {name: percentile_ci(bootstrap[model_name][name]) for name in ["position_rmse", "velocity_rmse", "vx_rmse", "vy_rmse"]}
            for step_index in range(len(stepwise.position_rmse)):
                writer.writerow(
                    {
                        "model": model_name,
                        "rollout_step": step_index + 1,
                        "position_rmse": stepwise.position_rmse[step_index],
                        "position_rmse_ci_low": ci["position_rmse"][0][step_index],
                        "position_rmse_ci_high": ci["position_rmse"][1][step_index],
                        "velocity_rmse": stepwise.velocity_rmse[step_index],
                        "velocity_rmse_ci_low": ci["velocity_rmse"][0][step_index],
                        "velocity_rmse_ci_high": ci["velocity_rmse"][1][step_index],
                        "vx_rmse": stepwise.vx_rmse[step_index],
                        "vx_rmse_ci_low": ci["vx_rmse"][0][step_index],
                        "vx_rmse_ci_high": ci["vx_rmse"][1][step_index],
                        "vy_rmse": stepwise.vy_rmse[step_index],
                        "vy_rmse_ci_low": ci["vy_rmse"][0][step_index],
                        "vy_rmse_ci_high": ci["vy_rmse"][1][step_index],
                        "n_valid_samples": int(stepwise.n_valid_samples[step_index]),
                    }
                )


def write_integrated_metrics_csv(
    path: Path,
    metrics: dict[str, IntegratedMetrics],
    bootstrap: dict[str, dict[str, np.ndarray]],
) -> None:
    fieldnames = [
        "model",
        "integrated_position_rmse",
        "integrated_position_rmse_ci_low",
        "integrated_position_rmse_ci_high",
        "final_step_position_rmse",
        "final_step_position_rmse_ci_low",
        "final_step_position_rmse_ci_high",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for model_name, integrated in metrics.items():
            integrated_ci = percentile_ci(bootstrap[model_name]["integrated_position_rmse"])
            final_ci = percentile_ci(bootstrap[model_name]["final_step_position_rmse"])
            writer.writerow(
                {
                    "model": model_name,
                    "integrated_position_rmse": integrated.integrated_position_rmse,
                    "integrated_position_rmse_ci_low": integrated_ci[0],
                    "integrated_position_rmse_ci_high": integrated_ci[1],
                    "final_step_position_rmse": integrated.final_step_position_rmse,
                    "final_step_position_rmse_ci_low": final_ci[0],
                    "final_step_position_rmse_ci_high": final_ci[1],
                }
            )


def write_channel_admissibility_metrics_csv(
    path: Path,
    metrics: dict[str, ChannelAdmissibilityMetricCurves],
    bootstrap: dict[str, dict[str, np.ndarray]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model", "rollout_step",
        "mean_outside_fraction", "mean_outside_fraction_ci_low", "mean_outside_fraction_ci_high",
        "median_outside_fraction",
        "violation_rate_2pct", "violation_rate_2pct_ci_low", "violation_rate_2pct_ci_high",
        "any_violation_rate", "any_violation_rate_ci_low", "any_violation_rate_ci_high",
        "violation_rate_10pct", "violation_rate_10pct_ci_low", "violation_rate_10pct_ci_high",
        "mean_excess_outside_fraction", "mean_excess_outside_fraction_ci_low", "mean_excess_outside_fraction_ci_high",
        "mean_geometry_penalty_equivalent", "mean_geometry_penalty_equivalent_ci_low", "mean_geometry_penalty_equivalent_ci_high",
        "true_mean_outside_fraction", "true_violation_rate_2pct", "true_violation_rate_10pct",
        "excess_mean_outside_fraction_vs_truth", "excess_violation_rate_2pct_vs_truth",
        "n_geometry_valid_samples", "geometry_valid_fraction",
    ]
    ci_metrics = [
        "mean_outside_fraction",
        "violation_rate_2pct",
        "any_violation_rate",
        "violation_rate_10pct",
        "mean_excess_outside_fraction",
        "mean_geometry_penalty_equivalent",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for model_name, curves in metrics.items():
            ci = {name: percentile_ci(bootstrap[model_name][name]) for name in ci_metrics}
            for step_index in range(len(curves.mean_outside_fraction)):
                row = {
                    "model": model_name,
                    "rollout_step": step_index + 1,
                    "median_outside_fraction": curves.median_outside_fraction[step_index],
                    "true_mean_outside_fraction": curves.true_mean_outside_fraction[step_index],
                    "true_violation_rate_2pct": curves.true_violation_rate_2pct[step_index],
                    "true_violation_rate_10pct": curves.true_violation_rate_10pct[step_index],
                    "excess_mean_outside_fraction_vs_truth": curves.excess_mean_outside_fraction_vs_truth[step_index],
                    "excess_violation_rate_2pct_vs_truth": curves.excess_violation_rate_2pct_vs_truth[step_index],
                    "n_geometry_valid_samples": int(curves.n_geometry_valid_samples[step_index]),
                    "geometry_valid_fraction": curves.geometry_valid_fraction[step_index],
                }
                for name in ci_metrics:
                    values = getattr(curves, name)
                    low, high = ci[name]
                    row[name] = values[step_index]
                    row[f"{name}_ci_low"] = low[step_index]
                    row[f"{name}_ci_high"] = high[step_index]
                writer.writerow(row)


def write_channel_trajectory_metrics_csv(
    path: Path,
    metrics: dict[str, TrajectoryAdmissibilityMetrics],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "n_evaluable_trajectories",
        "n_trajectories_with_violation",
        "trajectory_violation_fraction",
        "mean_first_violation_step",
        "median_first_violation_step",
        "n_violation_episodes",
        "mean_violation_episode_duration",
        "median_violation_episode_duration",
        "persistent_violation_fraction",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for model_name, item in metrics.items():
            writer.writerow({"model": model_name, **item.__dict__})


def save_channel_admissibility_data(
    path: Path,
    context: ChannelAdmissibilityContext | None,
    outside_fraction_by_model: dict[str, np.ndarray],
) -> None:
    if context is None:
        raise RuntimeError("Channel admissibility context has not been computed.")
    arrays: dict[str, np.ndarray] = {
        "window_id": context.window_id,
        "rollout_start_frame": context.rollout_start_frame,
        "track_ids": context.track_ids,
        "future_frame": context.future_frame,
        "bbox_w": context.bbox_w.astype(np.float32),
        "bbox_h": context.bbox_h.astype(np.float32),
        "bbox_valid": context.bbox_valid,
        "geometry_valid_mask": context.geometry_valid_mask,
        "true_outside_fraction": context.true_outside_fraction.astype(np.float32),
        "channel_mask_path": np.asarray(str(context.channel_mask_path), dtype="U512"),
        "detections_csv_path": np.asarray(str(context.detections_csv_path), dtype="U512"),
        "model_names": np.asarray(list(outside_fraction_by_model), dtype=str),
    }
    for model_name, outside in outside_fraction_by_model.items():
        arrays[f"{model_name}__outside_fraction"] = outside.astype(np.float32)
        arrays[f"{model_name}__violation_2pct"] = (outside > GEOMETRY_TOLERANCE) & context.geometry_valid_mask
        arrays[f"{model_name}__violation_10pct"] = (outside > 0.10) & context.geometry_valid_mask
    np.savez_compressed(path, **arrays)


def plot_channel_metric(
    output_dir: Path,
    metrics: dict[str, ChannelAdmissibilityMetricCurves],
    bootstrap: dict[str, dict[str, np.ndarray]],
    metric_name: str,
    truth_metric_name: str | None,
    ylabel: str,
    stem: str,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.8, 3.2), constrained_layout=True)
    steps = None
    reference_truth = None
    for model_name, curves in metrics.items():
        values = getattr(curves, metric_name)
        steps = np.arange(1, len(values) + 1)
        low, high = percentile_ci(bootstrap[model_name][metric_name])
        line = ax.plot(steps, values, linewidth=1.8, label=model_name)[0]
        ax.fill_between(steps, low, high, color=line.get_color(), alpha=0.18, linewidth=0)
        if truth_metric_name is not None and reference_truth is None:
            reference_truth = getattr(curves, truth_metric_name)
    if reference_truth is not None and steps is not None:
        ax.plot(steps, reference_truth, color="0.20", linewidth=1.4, linestyle="--", label="True footprint baseline")
    ax.set_xlabel("Rollout step")
    ax.set_ylabel(ylabel)
    ax.set_xlim(1, int(steps[-1]) if steps is not None else ROLLOUT_HORIZON)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="0.9", linewidth=0.6)
    ax.legend(frameon=False)
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    return pdf_path, png_path


def plot_stepwise_metric(
    output_dir: Path,
    stepwise_metrics: dict[str, StepwiseMetricCurves],
    bootstrap: dict[str, dict[str, np.ndarray]],
    metric_name: str,
    ylabel: str,
    stem: str,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )
    fig, ax = plt.subplots(figsize=(4.8, 3.2), constrained_layout=True)
    steps = None
    for model_name, stepwise in stepwise_metrics.items():
        values = getattr(stepwise, metric_name)
        steps = np.arange(1, len(values) + 1)
        low, high = percentile_ci(bootstrap[model_name][metric_name])
        line = ax.plot(steps, values, linewidth=1.8, label=model_name)[0]
        ax.fill_between(steps, low, high, color=line.get_color(), alpha=0.18, linewidth=0)
    ax.set_xlabel("Rollout step")
    ax.set_ylabel(ylabel)
    ax.set_xlim(1, int(steps[-1]) if steps is not None else ROLLOUT_HORIZON)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="0.9", linewidth=0.6)
    ax.legend(frameon=False)
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    return pdf_path, png_path


def load_predictions_npz(path: Path) -> dict[str, RolloutPrediction]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Prediction file does not exist: {path}")
    data = np.load(path, allow_pickle=True)
    model_names = [str(name) for name in data["model_names"]]
    predictions: dict[str, RolloutPrediction] = {}
    for model_name in model_names:
        prefix = f"{model_name}__"
        predictions[model_name] = RolloutPrediction(
            model_name=model_name,
            window_id=data[prefix + "window_id"],
            rollout_start_frame=data[prefix + "rollout_start_frame"],
            frame_start=data[prefix + "frame_start"],
            track_ids=data[prefix + "track_ids"],
            pred_position=data[prefix + "pred_position"],
            true_position=data[prefix + "true_position"],
            pred_velocity=data[prefix + "pred_velocity"],
            true_velocity=data[prefix + "true_velocity"],
            valid_mask=data[prefix + "valid_mask"].astype(bool),
            boundary_mask=data[prefix + "boundary_mask"].astype(bool),
        )
    return predictions


class CenterlineTangentEstimator:
    def __init__(
        self,
        centerline_csv: Path,
        pca_half_window: int,
        min_quality: float,
        max_centerline_distance: float,
        min_orientation_speed: float,
        min_branch_distance_margin: float | None,
        min_branch_relative_margin: float | None,
    ) -> None:
        branches, metadata = read_centerline_csv(centerline_csv)
        self.metadata = metadata
        self.pca_half_window = int(pca_half_window)
        self.min_quality = float(min_quality)
        self.max_centerline_distance = float(max_centerline_distance)
        self.min_orientation_speed = float(min_orientation_speed)
        self.min_branch_distance_margin = min_branch_distance_margin
        self.min_branch_relative_margin = min_branch_relative_margin
        min_points = max(3, 2 * self.pca_half_window + 1)
        self.branches = {
            name: np.asarray(points, dtype=np.float64)
            for name, points in branches.items()
            if len(points) >= min_points
        }
        if not self.branches:
            raise ValueError(f"No usable centerline branches found in {centerline_csv}")
        print(
            "Centerline loaded: "
            f"{metadata['point_count']} points, branches={metadata['branch_counts']}",
            flush=True,
        )

    def build_basis(self, prediction: RolloutPrediction) -> DirectionalBasis:
        shape = prediction.metric_mask.shape
        tangent = np.full((*shape, 2), np.nan, dtype=np.float32)
        normal = np.full((*shape, 2), np.nan, dtype=np.float32)
        quality = np.full(shape, np.nan, dtype=np.float32)
        distance = np.full(shape, np.nan, dtype=np.float32)
        branch_name = np.full(shape, "", dtype="<U64")
        second_branch_name = np.full(shape, "", dtype="<U64")
        second_distance = np.full(shape, np.nan, dtype=np.float32)
        branch_margin = np.full(shape, np.nan, dtype=np.float32)
        branch_relative_margin = np.full(shape, np.nan, dtype=np.float32)
        status = np.full(shape, "not_evaluated", dtype="<U64")
        axis_valid = np.zeros(shape, dtype=bool)
        orientation_valid = np.zeros(shape, dtype=bool)

        for index in zip(*np.nonzero(prediction.metric_mask)):
            estimate = self.estimate_one(prediction.true_position[index], prediction.true_velocity[index])
            quality[index] = estimate.quality
            distance[index] = estimate.distance_to_centerline
            branch_name[index] = estimate.branch_name
            second_branch_name[index] = estimate.second_nearest_branch_name
            second_distance[index] = estimate.second_nearest_branch_distance
            branch_margin[index] = estimate.branch_distance_margin
            branch_relative_margin[index] = estimate.branch_relative_margin
            status[index] = estimate.status
            axis_valid[index] = estimate.axis_valid
            orientation_valid[index] = estimate.orientation_valid
            if estimate.tangent is not None and estimate.normal is not None:
                tangent[index] = estimate.tangent
                normal[index] = estimate.normal

        check_directional_basis(tangent, normal, axis_valid)
        check_orientation(prediction.true_velocity, tangent, axis_valid & orientation_valid)
        return DirectionalBasis(
            tangent=tangent,
            normal=normal,
            axis_valid=axis_valid,
            orientation_valid=orientation_valid,
            quality=quality,
            distance_to_centerline=distance,
            branch_name=branch_name,
            second_nearest_branch_name=second_branch_name,
            second_nearest_branch_distance=second_distance,
            branch_distance_margin=branch_margin,
            branch_relative_margin=branch_relative_margin,
            status=status,
        )

    def estimate_one(self, point: np.ndarray, velocity: np.ndarray) -> TangentEstimate:
        empty = TangentEstimate(
            tangent=None,
            normal=None,
            axis_valid=False,
            orientation_valid=False,
            quality=np.nan,
            distance_to_centerline=np.nan,
            branch_name="",
            second_nearest_branch_name="",
            second_nearest_branch_distance=np.nan,
            branch_distance_margin=np.nan,
            branch_relative_margin=np.nan,
            status="nonfinite_input",
        )
        if not np.isfinite(point).all() or not np.isfinite(velocity).all():
            return empty

        branch_candidates = []
        for branch, points in self.branches.items():
            distances = np.linalg.norm(points - point[None, :], axis=1)
            nearest_index = int(np.argmin(distances))
            nearest_distance = float(distances[nearest_index])
            branch_candidates.append((nearest_distance, branch, nearest_index))
        if not branch_candidates:
            empty.status = "no_usable_branch"
            return empty
        branch_candidates.sort(key=lambda item: item[0])
        nearest_distance, branch, nearest_index = branch_candidates[0]
        if len(branch_candidates) > 1:
            second_distance, second_branch, _ = branch_candidates[1]
        else:
            second_distance, second_branch = np.inf, ""
        branch_margin = float(second_distance - nearest_distance)
        branch_relative_margin = float(branch_margin / (nearest_distance + 1.0e-12))

        def make_estimate(status: str, axis_valid: bool = False, orientation_valid: bool = False,
                          tangent=None, normal=None, quality=np.nan) -> TangentEstimate:
            return TangentEstimate(
                tangent=tangent,
                normal=normal,
                axis_valid=axis_valid,
                orientation_valid=orientation_valid,
                quality=float(quality),
                distance_to_centerline=float(nearest_distance),
                branch_name=str(branch),
                second_nearest_branch_name=str(second_branch),
                second_nearest_branch_distance=float(second_distance),
                branch_distance_margin=branch_margin,
                branch_relative_margin=branch_relative_margin,
                status=status,
            )

        if nearest_distance > self.max_centerline_distance:
            return make_estimate("too_far_from_centerline")
        if self.min_branch_distance_margin is not None and branch_margin < float(self.min_branch_distance_margin):
            return make_estimate("branch_ambiguous")
        if self.min_branch_relative_margin is not None and branch_relative_margin < float(self.min_branch_relative_margin):
            return make_estimate("branch_ambiguous")

        points = self.branches[branch]
        start = max(0, nearest_index - self.pca_half_window)
        stop = min(len(points), nearest_index + self.pca_half_window + 1)
        local = points[start:stop]
        if len(local) < 3:
            return make_estimate("insufficient_local_points")
        centered = local - local.mean(axis=0, keepdims=True)
        covariance = centered.T @ centered / float(len(local))
        eigvals, eigvecs = np.linalg.eigh(covariance)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        if eigvals[0] <= 0:
            return make_estimate("degenerate_pca")
        q_value = float(eigvals[0] / (eigvals[1] + 1.0e-12))
        if q_value < self.min_quality:
            return make_estimate("low_pca_quality", quality=q_value)
        t_hat = eigvecs[:, 0]
        t_hat = t_hat / np.linalg.norm(t_hat)
        speed = float(np.linalg.norm(velocity))
        orientation_valid = speed >= self.min_orientation_speed
        status_value = "valid_oriented" if orientation_valid else "valid_unoriented_low_speed"
        if orientation_valid and float(np.dot(velocity, t_hat)) < 0:
            t_hat = -t_hat
        n_hat = np.asarray([-t_hat[1], t_hat[0]], dtype=np.float64)
        return make_estimate(
            status_value,
            axis_valid=True,
            orientation_valid=orientation_valid,
            tangent=t_hat.astype(np.float32),
            normal=n_hat.astype(np.float32),
            quality=q_value,
        )


def check_directional_basis(tangent: np.ndarray, normal: np.ndarray, valid: np.ndarray) -> None:
    if not valid.any():
        raise ValueError("No axis-valid samples were found.")
    t = tangent[valid].astype(np.float64)
    n = normal[valid].astype(np.float64)
    if not np.allclose(np.linalg.norm(t, axis=1), 1.0, atol=1.0e-4):
        raise ValueError("Invalid tangent basis: tangent vectors are not unit length.")
    if not np.allclose(np.linalg.norm(n, axis=1), 1.0, atol=1.0e-4):
        raise ValueError("Invalid tangent basis: normal vectors are not unit length.")
    if not np.allclose(np.sum(t * n, axis=1), 0.0, atol=1.0e-4):
        raise ValueError("Invalid tangent basis: tangent and normal are not orthogonal.")


def check_orientation(true_velocity: np.ndarray, tangent: np.ndarray, valid: np.ndarray) -> None:
    if not valid.any():
        return
    dot = np.sum(true_velocity[valid] * tangent[valid], axis=-1)
    if np.any(dot < -1.0e-5):
        raise ValueError("Invalid oriented tangent basis: true_velocity dot tangent is negative.")


def compute_directional_sufficient_stats(
    prediction: RolloutPrediction,
    basis: DirectionalBasis,
) -> DirectionalSufficientStats:
    metric_mask = prediction.metric_mask
    axis_mask = metric_mask & basis.axis_valid
    orientation_mask = axis_mask & basis.orientation_valid
    error = prediction.pred_position - prediction.true_position
    e_parallel = np.sum(error * basis.tangent, axis=-1)
    e_perp = np.sum(error * basis.normal, axis=-1)
    if not np.isfinite(e_parallel[axis_mask]).all() or not np.isfinite(e_perp[axis_mask]).all():
        raise ValueError(f"{prediction.model_name}: non-finite directional error under axis-valid mask.")
    euclidean_sq = np.sum(error[axis_mask] ** 2, axis=-1)
    decomposed_sq = e_parallel[axis_mask] ** 2 + e_perp[axis_mask] ** 2
    if not np.allclose(euclidean_sq, decomposed_sq, rtol=1.0e-4, atol=1.0e-3):
        raise ValueError(f"{prediction.model_name}: directional decomposition sanity check failed.")
    return DirectionalSufficientStats(
        sse_parallel=np.where(axis_mask, e_parallel**2, 0.0).sum(axis=2),
        sse_perp=np.where(axis_mask, e_perp**2, 0.0).sum(axis=2),
        sum_parallel=np.where(orientation_mask, e_parallel, 0.0).sum(axis=2),
        sum_perp=np.where(orientation_mask, e_perp, 0.0).sum(axis=2),
        axis_count=axis_mask.sum(axis=2).astype(np.int64),
        orientation_count=orientation_mask.sum(axis=2).astype(np.int64),
        global_count=metric_mask.sum(axis=2).astype(np.int64),
    )


def directional_metrics_from_stats(stats: DirectionalSufficientStats) -> DirectionalMetricCurves:
    axis_count = stats.axis_count.sum(axis=0)
    orientation_count = stats.orientation_count.sum(axis=0)
    global_count = stats.global_count.sum(axis=0)
    if np.any(axis_count == 0):
        raise ValueError(f"Zero axis-valid samples for steps {(np.flatnonzero(axis_count == 0) + 1).tolist()}")
    if np.any(orientation_count == 0):
        raise ValueError(f"Zero orientation-valid samples for steps {(np.flatnonzero(orientation_count == 0) + 1).tolist()}")
    tangential_rmse = np.sqrt(stats.sse_parallel.sum(axis=0) / axis_count)
    normal_rmse = np.sqrt(stats.sse_perp.sum(axis=0) / axis_count)
    return DirectionalMetricCurves(
        tangential_rmse=tangential_rmse,
        normal_rmse=normal_rmse,
        tangential_bias=stats.sum_parallel.sum(axis=0) / orientation_count,
        normal_bias=stats.sum_perp.sum(axis=0) / orientation_count,
        anisotropy_ratio=tangential_rmse / (normal_rmse + 1.0e-12),
        n_axis_valid_samples=axis_count.astype(np.int64),
        axis_valid_fraction=axis_count / np.maximum(global_count, 1),
        n_orientation_valid_samples=orientation_count.astype(np.int64),
        # Orientation validity is a second-stage concept, conditional on a valid axis.
        orientation_valid_fraction=orientation_count / np.maximum(axis_count, 1),
    )


def bootstrap_directional_metrics(
    stats: DirectionalSufficientStats,
    bootstrap_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    n_bootstrap, _ = bootstrap_indices.shape
    horizon = stats.axis_count.shape[1]
    output = {name: np.empty((n_bootstrap, horizon), dtype=np.float64) for name in [
        "tangential_rmse",
        "normal_rmse",
        "tangential_bias",
        "normal_bias",
        "anisotropy_ratio",
    ]}
    for replicate, indices in enumerate(bootstrap_indices):
        curves = directional_metrics_from_stats(
            DirectionalSufficientStats(
                sse_parallel=stats.sse_parallel[indices],
                sse_perp=stats.sse_perp[indices],
                sum_parallel=stats.sum_parallel[indices],
                sum_perp=stats.sum_perp[indices],
                axis_count=stats.axis_count[indices],
                orientation_count=stats.orientation_count[indices],
                global_count=stats.global_count[indices],
            )
        )
        output["tangential_rmse"][replicate] = curves.tangential_rmse
        output["normal_rmse"][replicate] = curves.normal_rmse
        output["tangential_bias"][replicate] = curves.tangential_bias
        output["normal_bias"][replicate] = curves.normal_bias
        output["anisotropy_ratio"][replicate] = curves.anisotropy_ratio
    return output


def assign_centerline_branches(
    estimator: CenterlineTangentEstimator,
    positions: np.ndarray,
    velocities: np.ndarray,
    valid_mask: np.ndarray,
) -> BranchAssignment:
    shape = valid_mask.shape
    branch_name = np.full(shape, "", dtype="<U64")
    axis_valid = np.zeros(shape, dtype=bool)
    status = np.full(shape, "not_evaluated", dtype="<U64")
    flat_positions = positions.reshape(-1, 2)
    flat_valid = valid_mask.reshape(-1)
    valid_indices = np.flatnonzero(flat_valid)
    if valid_indices.size == 0:
        return BranchAssignment(branch_name=branch_name, axis_valid=axis_valid, status=status)

    flat_branch_name = branch_name.reshape(-1)
    flat_axis_valid = axis_valid.reshape(-1)
    flat_status = status.reshape(-1)
    candidate_positions = flat_positions[valid_indices].astype(np.float64, copy=False)
    finite = np.isfinite(candidate_positions).all(axis=1)
    finite_indices = valid_indices[finite]
    if finite_indices.size == 0:
        flat_status[valid_indices] = "nonfinite_input"
        return BranchAssignment(branch_name=branch_name, axis_valid=axis_valid, status=status)

    finite_positions = candidate_positions[finite]
    branch_items = list(estimator.branches.items())
    branch_names = [name for name, _ in branch_items]
    all_branch_points = np.concatenate([np.asarray(points, dtype=np.float64) for _, points in branch_items], axis=0)
    max_x = float(np.nanmax(all_branch_points[:, 0]))
    max_y = float(np.nanmax(all_branch_points[:, 1]))
    width = int(np.ceil(max_x + estimator.max_centerline_distance + 4))
    height = int(np.ceil(max_y + estimator.max_centerline_distance + 4))
    width = max(width, 1)
    height = max(height, 1)
    distance_by_branch = np.empty((finite_positions.shape[0], len(branch_items)), dtype=np.float64)
    sample_x = np.rint(finite_positions[:, 0]).astype(np.int64)
    sample_y = np.rint(finite_positions[:, 1]).astype(np.int64)
    in_image = (sample_x >= 0) & (sample_x < width) & (sample_y >= 0) & (sample_y < height)
    for branch_index, (_, branch_points) in enumerate(branch_items):
        branch_image = np.zeros((height, width), dtype=np.uint8)
        rounded = np.rint(np.asarray(branch_points, dtype=np.float64)).astype(np.int32)
        rounded[:, 0] = np.clip(rounded[:, 0], 0, width - 1)
        rounded[:, 1] = np.clip(rounded[:, 1], 0, height - 1)
        if len(rounded) == 1:
            x, y = rounded[0]
            branch_image[y, x] = 255
        else:
            cv2.polylines(branch_image, [rounded.reshape(-1, 1, 2)], isClosed=False, color=255, thickness=1)
        distance_map = cv2.distanceTransform(255 - branch_image, cv2.DIST_L2, 5)
        branch_distance = np.full((finite_positions.shape[0],), np.inf, dtype=np.float64)
        branch_distance[in_image] = distance_map[sample_y[in_image], sample_x[in_image]]
        distance_by_branch[:, branch_index] = branch_distance

    order = np.argsort(distance_by_branch, axis=1)
    nearest_index = order[:, 0]
    nearest_distance = distance_by_branch[np.arange(distance_by_branch.shape[0]), nearest_index]
    if len(branch_items) > 1:
        second_index = order[:, 1]
        second_distance = distance_by_branch[np.arange(distance_by_branch.shape[0]), second_index]
    else:
        second_distance = np.full_like(nearest_distance, np.inf)
    branch_margin = second_distance - nearest_distance
    branch_relative_margin = branch_margin / (nearest_distance + 1.0e-12)

    for branch_index, name in enumerate(branch_names):
        flat_branch_name[finite_indices[nearest_index == branch_index]] = str(name)
    flat_status[finite_indices] = "valid_branch"
    too_far = nearest_distance > estimator.max_centerline_distance
    flat_status[finite_indices[too_far]] = "too_far_from_centerline"
    ambiguous = np.zeros_like(too_far, dtype=bool)
    if estimator.min_branch_distance_margin is not None:
        ambiguous |= branch_margin < float(estimator.min_branch_distance_margin)
    if estimator.min_branch_relative_margin is not None:
        ambiguous |= branch_relative_margin < float(estimator.min_branch_relative_margin)
    flat_status[finite_indices[ambiguous & ~too_far]] = "branch_ambiguous"
    flat_axis_valid[finite_indices[~too_far & ~ambiguous]] = True
    flat_status[valid_indices[~finite]] = "nonfinite_input"
    return BranchAssignment(branch_name=branch_name, axis_valid=axis_valid, status=status)


def identify_junction_decision_events(
    reference: RolloutPrediction,
    true_assignment: BranchAssignment,
    predicted_assignments: dict[str, BranchAssignment],
    outgoing_branches: tuple[str, ...],
    incoming_branches: tuple[str, ...],
    commitment_steps: int,
) -> list[JunctionDecisionEvent]:
    outgoing_branch_to_code = {branch: index + 1 for index, branch in enumerate(outgoing_branches)}
    outgoing_code_to_branch = {code: branch for branch, code in outgoing_branch_to_code.items()}
    incoming_branch_to_code = {branch: index + 1 for index, branch in enumerate(incoming_branches)}
    incoming_code_to_branch = {code: branch for branch, code in incoming_branch_to_code.items()}
    true_codes = encode_branch_commitment_codes(true_assignment, reference.metric_mask, outgoing_branch_to_code)
    true_incoming_codes = encode_specific_branch_codes(true_assignment, reference.metric_mask, incoming_branch_to_code)
    pred_codes = {
        model_name: encode_branch_commitment_codes(assignment, reference.metric_mask, outgoing_branch_to_code)
        for model_name, assignment in predicted_assignments.items()
    }
    events: list[JunctionDecisionEvent] = []
    n_windows, _, n_slots = reference.metric_mask.shape
    candidate_windows, candidate_slots = np.nonzero(np.any(true_codes != -99, axis=1))
    for window_row, slot in zip(candidate_windows.tolist(), candidate_slots.tolist()):
        codes = true_codes[window_row, :, slot]
        true_code, true_step = find_persistent_branch_commitment_code(codes, commitment_steps)
        if true_code is None or true_step is None:
            continue
        true_incoming_code = find_final_incoming_before_commitment(
            outgoing_codes=codes,
            incoming_codes=true_incoming_codes[window_row, :, slot],
            commitment_end_step=int(true_step),
            commitment_steps=commitment_steps,
        )
        if true_incoming_code is None:
            continue
        true_branch = outgoing_code_to_branch[int(true_code)]
        true_incoming_branch = incoming_code_to_branch[int(true_incoming_code)]

        pred_branch_by_model: dict[str, str] = {}
        pred_step_by_model: dict[str, int] = {}
        status_by_model: dict[str, str] = {}
        delay_by_model: dict[str, float] = {}
        for model_name, codes_by_model in pred_codes.items():
            pred_code, pred_step = find_persistent_branch_commitment_code(
                codes_by_model[window_row, :, slot],
                commitment_steps,
            )
            if pred_code is None or pred_step is None:
                assignment = predicted_assignments[model_name]
                pred_valid = reference.metric_mask[window_row, :, slot] & assignment.axis_valid[window_row, :, slot]
                status = classify_missing_prediction_commitment(pred_valid, assignment.status[window_row, :, slot])
                pred_branch_value = ""
                pred_step_value = -1
                delay_value = np.nan
            else:
                pred_branch = outgoing_code_to_branch[int(pred_code)]
                status = "correct" if pred_branch == true_branch else "wrong_branch"
                pred_branch_value = pred_branch
                pred_step_value = int(pred_step)
                delay_value = float(pred_step - true_step) if status == "correct" else np.nan
            pred_branch_by_model[model_name] = pred_branch_value
            pred_step_by_model[model_name] = pred_step_value
            status_by_model[model_name] = status
            delay_by_model[model_name] = delay_value

        events.append(
            JunctionDecisionEvent(
                window_row=int(window_row),
                window_id=int(reference.window_id[window_row]),
                rollout_start_frame=int(reference.rollout_start_frame[window_row]),
                track_id=int(reference.track_ids[window_row, slot]),
                slot=int(slot),
                true_incoming_branch=str(true_incoming_branch),
                true_branch=str(true_branch),
                true_commitment_step=int(true_step),
                pred_branch_by_model=pred_branch_by_model,
                pred_commitment_step_by_model=pred_step_by_model,
                decision_status_by_model=status_by_model,
                commitment_step_delay_by_model=delay_by_model,
            )
        )
    return events


def encode_branch_commitment_codes(
    assignment: BranchAssignment,
    metric_mask: np.ndarray,
    branch_to_code: dict[str, int],
) -> np.ndarray:
    codes = np.full(metric_mask.shape, -99, dtype=np.int16)
    valid = metric_mask & assignment.axis_valid
    codes[valid] = 0
    for branch, code in branch_to_code.items():
        codes[valid & (assignment.branch_name == branch)] = int(code)
    return codes


def encode_specific_branch_codes(
    assignment: BranchAssignment,
    metric_mask: np.ndarray,
    branch_to_code: dict[str, int],
) -> np.ndarray:
    codes = np.full(metric_mask.shape, -99, dtype=np.int16)
    valid = metric_mask & assignment.axis_valid
    codes[valid] = 0
    for branch, code in branch_to_code.items():
        codes[valid & (assignment.branch_name == branch)] = int(code)
    return codes


def find_final_incoming_before_commitment(
    outgoing_codes: np.ndarray,
    incoming_codes: np.ndarray,
    commitment_end_step: int,
    commitment_steps: int,
) -> int | None:
    valid_steps = np.flatnonzero(outgoing_codes != -99)
    if valid_steps.size == 0:
        return None
    if int(outgoing_codes[int(valid_steps[0])]) > 0:
        return None

    commitment_start_step = int(commitment_end_step) - int(commitment_steps) + 1
    if commitment_start_step <= 0:
        return None
    prior_incoming_steps = np.flatnonzero(incoming_codes[:commitment_start_step] > 0)
    if prior_incoming_steps.size == 0:
        return None
    return int(incoming_codes[int(prior_incoming_steps[-1])])


def count_legacy_loose_junction_events(
    true_assignment: BranchAssignment,
    metric_mask: np.ndarray,
    outgoing_branches: tuple[str, ...],
    commitment_steps: int,
) -> int:
    branch_to_code = {branch: index + 1 for index, branch in enumerate(outgoing_branches)}
    codes = encode_branch_commitment_codes(true_assignment, metric_mask, branch_to_code)
    count = 0
    candidate_windows, candidate_slots = np.nonzero(np.any(codes != -99, axis=1))
    for window_row, slot in zip(candidate_windows.tolist(), candidate_slots.tolist()):
        trajectory_codes = codes[window_row, :, slot]
        if not starts_as_legacy_uncommitted_code(trajectory_codes, commitment_steps):
            continue
        true_code, _ = find_persistent_branch_commitment_code(trajectory_codes, commitment_steps)
        if true_code is not None:
            count += 1
    return count


def starts_as_legacy_uncommitted_code(codes: np.ndarray, commitment_steps: int) -> bool:
    valid_steps = np.flatnonzero(codes != -99)
    if valid_steps.size == 0:
        return False
    first_code = int(codes[int(valid_steps[0])])
    if first_code <= 0:
        return True
    first_run = 0
    for step in valid_steps:
        if int(codes[int(step)]) != first_code:
            break
        first_run += 1
    return first_run < commitment_steps


def find_persistent_branch_commitment_code(
    codes: np.ndarray,
    commitment_steps: int,
) -> tuple[int | None, int | None]:
    horizon = int(codes.shape[0])
    for start in range(0, horizon - commitment_steps + 1):
        code = int(codes[start])
        if code <= 0:
            continue
        committed = True
        for offset in range(1, commitment_steps):
            if int(codes[start + offset]) != code:
                committed = False
                break
        if committed:
            return code, start + commitment_steps - 1
    return None, None


def classify_missing_prediction_commitment(pred_valid: np.ndarray, pred_status: np.ndarray) -> str:
    if bool(pred_valid.any()):
        return "no_commitment"
    usable_status = [str(value) for value in pred_status if str(value) not in ("", "not_evaluated")]
    return "unclassifiable" if usable_status else "no_commitment"


def compute_junction_decision_metrics_for_model(
    events: list[JunctionDecisionEvent],
    model_name: str,
) -> JunctionDecisionMetrics:
    statuses = [event.decision_status_by_model[model_name] for event in events]
    n_correct = int(sum(status == "correct" for status in statuses))
    n_wrong = int(sum(status == "wrong_branch" for status in statuses))
    n_no_commitment = int(sum(status == "no_commitment" for status in statuses))
    n_unclassifiable = int(sum(status == "unclassifiable" for status in statuses))
    wrong_denominator = n_correct + n_wrong
    wrong_rate = float(n_wrong / wrong_denominator) if wrong_denominator else np.nan
    noncommitment_rate = float(n_no_commitment / len(events)) if events else np.nan
    delays = np.asarray(
        [
            event.commitment_step_delay_by_model[model_name]
            for event in events
            if event.decision_status_by_model[model_name] == "correct"
            and np.isfinite(event.commitment_step_delay_by_model[model_name])
        ],
        dtype=np.float64,
    )
    confusion: dict[tuple[str, str], int] = {}
    for event in events:
        pred_branch = event.pred_branch_by_model[model_name]
        if event.decision_status_by_model[model_name] in ("correct", "wrong_branch") and pred_branch:
            key = (event.true_branch, pred_branch)
            confusion[key] = confusion.get(key, 0) + 1
    return JunctionDecisionMetrics(
        n_true_junction_events=int(len(events)),
        n_correct=n_correct,
        n_wrong_branch=n_wrong,
        n_no_commitment=n_no_commitment,
        n_unclassifiable=n_unclassifiable,
        wrong_decision_rate=wrong_rate,
        noncommitment_rate=noncommitment_rate,
        mean_commitment_step_delay=float(np.mean(delays)) if delays.size else np.nan,
        median_commitment_step_delay=float(np.median(delays)) if delays.size else np.nan,
        confusion_counts=confusion,
        true_event_transition_counts=count_true_event_transitions(events),
    )


def count_true_event_transitions(events: list[JunctionDecisionEvent]) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for event in events:
        key = (event.true_incoming_branch, event.true_branch)
        counts[key] = counts.get(key, 0) + 1
    return counts


def bootstrap_junction_decision_rates(
    events: list[JunctionDecisionEvent],
    model_name: str,
    bootstrap_indices: np.ndarray,
    n_windows: int,
) -> dict[str, tuple[float, float]]:
    if not events:
        return {"wrong_decision_rate": (np.nan, np.nan), "noncommitment_rate": (np.nan, np.nan)}
    event_windows = np.asarray([event.window_row for event in events], dtype=np.int64)
    statuses = np.asarray([event.decision_status_by_model[model_name] for event in events], dtype="<U32")
    wrong_rates = np.empty((bootstrap_indices.shape[0],), dtype=np.float64)
    noncommit_rates = np.empty((bootstrap_indices.shape[0],), dtype=np.float64)
    for replicate, indices in enumerate(bootstrap_indices):
        window_weights = np.bincount(indices.astype(np.int64), minlength=n_windows)
        weights = window_weights[event_windows].astype(np.float64)
        correct = float(weights[statuses == "correct"].sum())
        wrong = float(weights[statuses == "wrong_branch"].sum())
        no_commit = float(weights[statuses == "no_commitment"].sum())
        total = float(weights.sum())
        wrong_rates[replicate] = wrong / (correct + wrong) if (correct + wrong) > 0 else np.nan
        noncommit_rates[replicate] = no_commit / total if total > 0 else np.nan
    return {
        "wrong_decision_rate": finite_percentile_ci(wrong_rates),
        "noncommitment_rate": finite_percentile_ci(noncommit_rates),
    }


def finite_percentile_ci(samples: np.ndarray) -> tuple[float, float]:
    finite = samples[np.isfinite(samples)]
    if finite.size == 0:
        return np.nan, np.nan
    low, high = percentile_ci(finite)
    return float(low), float(high)


def format_confusion_counts(confusion: dict[tuple[str, str], int]) -> str:
    return ";".join(
        f"{true_branch}->{pred_branch}:{count}"
        for (true_branch, pred_branch), count in sorted(confusion.items())
    )


def summarize_junction_decision_logic(
    outgoing_branches: tuple[str, ...],
    incoming_branches: tuple[str, ...],
    commitment_steps: int,
) -> None:
    print("Junction branch-decision diagnostic:", flush=True)
    print(
        "  Junction events are identified from TRUE trajectories only: "
        f"at least one explicit incoming-branch assignment before {commitment_steps} consecutive "
        "evaluable steps on the same outgoing branch.",
        flush=True,
    )
    print(f"  Outgoing branches: {', '.join(outgoing_branches)}", flush=True)
    print(f"  Incoming/non-decision branches: {', '.join(incoming_branches)}", flush=True)
    print("  Branch assignments reuse the centerline nearest-branch logic from the tangential/normal diagnostic.", flush=True)
    print(
        "  Arbitrary non-outgoing branches are not treated as incoming, and trajectories that begin "
        "on an outgoing branch are not true junction events.",
        flush=True,
    )
    print(
        "  no_commitment means the prediction never persistently reaches an outgoing branch before its final "
        "evaluable step; progression lag cannot be counted as wrong_branch.",
        flush=True,
    )
    print("  unclassifiable means there was no usable branch assignment for that predicted event.", flush=True)


def summarize_true_junction_event_counts(
    events: list[JunctionDecisionEvent],
    legacy_event_count: int,
) -> None:
    transition_counts = count_true_event_transitions(events)
    removed_count = int(legacy_event_count) - len(events)
    print("True junction-event identification:", flush=True)
    print(f"  legacy loose event count: {legacy_event_count}", flush=True)
    print(f"  corrected explicit-incoming event count: {len(events)}", flush=True)
    print(f"  events removed relative to legacy loose logic: {removed_count}", flush=True)
    print("  true event counts by incoming_branch -> true_outgoing_branch:", flush=True)
    if transition_counts:
        for (incoming_branch, outgoing_branch), count in sorted(transition_counts.items()):
            print(f"    {incoming_branch} -> {outgoing_branch}: {count}", flush=True)
    else:
        print("    none", flush=True)


def summarize_junction_decision_metrics(metrics: dict[str, JunctionDecisionMetrics]) -> None:
    print("Junction branch-decision summary:", flush=True)
    for model_name, item in metrics.items():
        print(
            f"  {model_name}: events={item.n_true_junction_events} "
            f"correct={item.n_correct} wrong_branch={item.n_wrong_branch} "
            f"no_commitment={item.n_no_commitment} unclassifiable={item.n_unclassifiable} "
            f"wrong_decision_rate={item.wrong_decision_rate:.6f} "
            f"noncommitment_rate={item.noncommitment_rate:.6f}",
            flush=True,
        )


def write_junction_decision_metrics_csv(
    path: Path,
    metrics: dict[str, JunctionDecisionMetrics],
    bootstrap: dict[str, dict[str, tuple[float, float]]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "model",
        "n_true_junction_events",
        "n_correct",
        "n_wrong_branch",
        "n_no_commitment",
        "n_unclassifiable",
        "wrong_decision_rate",
        "wrong_decision_rate_ci_low",
        "wrong_decision_rate_ci_high",
        "noncommitment_rate",
        "noncommitment_rate_ci_low",
        "noncommitment_rate_ci_high",
        "mean_commitment_step_delay",
        "median_commitment_step_delay",
        "confusion_counts",
        "true_event_transition_counts",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for model_name, item in metrics.items():
            wrong_ci = bootstrap.get(model_name, {}).get("wrong_decision_rate", (np.nan, np.nan))
            noncommit_ci = bootstrap.get(model_name, {}).get("noncommitment_rate", (np.nan, np.nan))
            writer.writerow(
                {
                    "model": model_name,
                    "n_true_junction_events": item.n_true_junction_events,
                    "n_correct": item.n_correct,
                    "n_wrong_branch": item.n_wrong_branch,
                    "n_no_commitment": item.n_no_commitment,
                    "n_unclassifiable": item.n_unclassifiable,
                    "wrong_decision_rate": item.wrong_decision_rate,
                    "wrong_decision_rate_ci_low": wrong_ci[0],
                    "wrong_decision_rate_ci_high": wrong_ci[1],
                    "noncommitment_rate": item.noncommitment_rate,
                    "noncommitment_rate_ci_low": noncommit_ci[0],
                    "noncommitment_rate_ci_high": noncommit_ci[1],
                    "mean_commitment_step_delay": item.mean_commitment_step_delay,
                    "median_commitment_step_delay": item.median_commitment_step_delay,
                    "confusion_counts": format_confusion_counts(item.confusion_counts),
                    "true_event_transition_counts": format_confusion_counts(item.true_event_transition_counts),
                }
            )


def write_junction_decision_events_csv(
    path: Path,
    events: list[JunctionDecisionEvent],
    model_names: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "window_id",
        "rollout_start_frame",
        "track_id",
        "slot",
        "true_incoming_branch",
        "true_branch",
        "true_commitment_step",
    ]
    for model_name in model_names:
        columns.extend(
            [
                f"{model_name}_pred_branch",
                f"{model_name}_pred_commitment_step",
                f"{model_name}_decision_status",
                f"{model_name}_commitment_step_delay",
            ]
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for event in events:
            row = {
                "window_id": event.window_id,
                "rollout_start_frame": event.rollout_start_frame,
                "track_id": event.track_id,
                "slot": event.slot,
                "true_incoming_branch": event.true_incoming_branch,
                "true_branch": event.true_branch,
                "true_commitment_step": event.true_commitment_step,
            }
            for model_name in model_names:
                row[f"{model_name}_pred_branch"] = event.pred_branch_by_model[model_name]
                row[f"{model_name}_pred_commitment_step"] = event.pred_commitment_step_by_model[model_name]
                row[f"{model_name}_decision_status"] = event.decision_status_by_model[model_name]
                row[f"{model_name}_commitment_step_delay"] = event.commitment_step_delay_by_model[model_name]
            writer.writerow(row)


def load_channel_mask_bool(mask_path: Path) -> np.ndarray:
    mask = np.load(mask_path)
    if mask.ndim != 2:
        raise ValueError(f"Channel mask must be 2D, got {mask.shape}.")
    mask = mask.astype(bool)
    if not mask.any():
        raise ValueError(f"Channel mask has no admissible pixels: {mask_path}")
    return mask


def compute_ellipse_outside_fraction_vectorized(
    positions: np.ndarray,
    bbox_w: np.ndarray,
    bbox_h: np.ndarray,
    channel_mask: np.ndarray,
    valid_mask: np.ndarray,
    chunk_size: int = 8192,
) -> np.ndarray:
    """Vectorized equivalent of compute_ellipse_outside_fraction for many ellipses."""

    output = np.full(valid_mask.shape, np.nan, dtype=np.float32)
    flat_valid = np.flatnonzero(valid_mask.reshape(-1))
    if flat_valid.size == 0:
        return output

    flat_positions = positions.reshape(-1, 2)
    flat_w = bbox_w.reshape(-1).astype(np.float64)
    flat_h = bbox_h.reshape(-1).astype(np.float64)
    height, width = channel_mask.shape
    max_axis = float(max(np.nanmax(flat_w[flat_valid]) / 2.0, np.nanmax(flat_h[flat_valid]) / 2.0))
    radius = int(np.ceil(max_axis)) + 3
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    offset_x, offset_y = np.meshgrid(offsets, offsets)
    offset_x = offset_x.reshape(1, -1)
    offset_y = offset_y.reshape(1, -1)

    flat_output = output.reshape(-1)
    for start in range(0, flat_valid.size, chunk_size):
        indices = flat_valid[start : start + chunk_size]
        cx = flat_positions[indices, 0].astype(np.float64)
        cy = flat_positions[indices, 1].astype(np.float64)
        bw = flat_w[indices]
        bh = flat_h[indices]
        finite = np.isfinite(cx) & np.isfinite(cy) & np.isfinite(bw) & np.isfinite(bh) & (bw > 0) & (bh > 0)
        if not finite.all():
            raise ValueError("Non-finite or non-positive ellipse inputs under geometry-valid mask.")
        axis_a = bw / 2.0
        axis_b = bh / 2.0
        base_x = np.floor(cx)[:, None]
        base_y = np.floor(cy)[:, None]
        grid_x = base_x + offset_x
        grid_y = base_y + offset_y
        ellipse = ((grid_x - cx[:, None]) / axis_a[:, None]) ** 2 + ((grid_y - cy[:, None]) / axis_b[:, None]) ** 2 <= 1.0
        total = ellipse.sum(axis=1)
        in_image = (
            ellipse
            & (grid_x >= 0)
            & (grid_x < width)
            & (grid_y >= 0)
            & (grid_y < height)
        )
        clipped_x = np.clip(grid_x, 0, width - 1).astype(np.int64)
        clipped_y = np.clip(grid_y, 0, height - 1).astype(np.int64)
        inside = in_image & channel_mask[clipped_y, clipped_x]
        inside_count = inside.sum(axis=1)
        flat_output[indices] = np.where(total > 0, 1.0 - inside_count / total, 1.0).astype(np.float32)
    return output


def build_channel_admissibility_context(
    reference: RolloutPrediction,
    channel_mask_path: Path,
    detections_csv_path: Path,
) -> ChannelAdmissibilityContext:
    channel_mask_path = Path(channel_mask_path)
    detections_csv_path = Path(detections_csv_path)
    channel_mask = load_channel_mask_bool(channel_mask_path)
    bbox_lookup = FutureBBoxLookup(detections_csv_path)

    n_windows, horizon, max_droplets = reference.valid_mask.shape
    future_frame = reference.rollout_start_frame[:, None] + np.arange(horizon, dtype=np.int64)[None, :]
    bbox_w = np.zeros((n_windows, horizon, max_droplets), dtype=np.float32)
    bbox_h = np.zeros_like(bbox_w)
    bbox_valid = np.zeros((n_windows, horizon, max_droplets), dtype=bool)

    metric_mask = reference.metric_mask
    candidate_mask = metric_mask & (reference.track_ids[:, None, :] >= 0)
    candidate_count = int(candidate_mask.sum())
    missing_count = 0
    for window_index, step_index, slot_index in zip(*np.nonzero(candidate_mask)):
        frame = int(future_frame[window_index, step_index])
        track_id = int(reference.track_ids[window_index, slot_index])
        lookup = bbox_lookup.lookup(frame, track_id)
        if lookup is None:
            missing_count += 1
            continue
        width, height = lookup
        if not (np.isfinite(width) and np.isfinite(height) and width > 0 and height > 0):
            missing_count += 1
            continue
        bbox_w[window_index, step_index, slot_index] = float(width)
        bbox_h[window_index, step_index, slot_index] = float(height)
        bbox_valid[window_index, step_index, slot_index] = True

    candidate_geometry_mask = metric_mask & bbox_valid & np.isfinite(reference.true_position).all(axis=-1)
    true_outside = compute_ellipse_outside_fraction_vectorized(
        reference.true_position,
        bbox_w,
        bbox_h,
        channel_mask,
        candidate_geometry_mask,
    )
    geometry_valid = candidate_geometry_mask & np.isfinite(true_outside)
    coverage = float(bbox_valid[metric_mask].sum() / max(candidate_count, 1))
    global_count = metric_mask.sum(axis=2).astype(np.int64)
    if not np.all((true_outside[geometry_valid] >= -1.0e-7) & (true_outside[geometry_valid] <= 1.0 + 1.0e-7)):
        raise ValueError("True outside fractions outside [0, 1].")

    return ChannelAdmissibilityContext(
        window_id=reference.window_id,
        rollout_start_frame=reference.rollout_start_frame,
        track_ids=reference.track_ids,
        future_frame=future_frame,
        true_position=reference.true_position.astype(np.float32),
        bbox_w=bbox_w,
        bbox_h=bbox_h,
        bbox_valid=bbox_valid,
        geometry_valid_mask=geometry_valid,
        true_outside_fraction=true_outside,
        global_count=global_count,
        bbox_lookup_coverage=coverage,
        bbox_missing_count=missing_count,
        bbox_candidate_count=candidate_count,
        channel_mask_path=channel_mask_path,
        detections_csv_path=detections_csv_path,
    )


def compute_model_outside_fraction(
    prediction: RolloutPrediction,
    context: ChannelAdmissibilityContext,
) -> np.ndarray:
    channel_mask = load_channel_mask_bool(context.channel_mask_path)
    outside = compute_ellipse_outside_fraction_vectorized(
        prediction.pred_position,
        context.bbox_w,
        context.bbox_h,
        channel_mask,
        context.geometry_valid_mask,
    )
    if np.any(outside[context.geometry_valid_mask] < -1.0e-7) or np.any(outside[context.geometry_valid_mask] > 1.0 + 1.0e-7):
        raise ValueError(f"{prediction.model_name}: outside fraction outside [0, 1].")
    return outside


def compute_channel_admissibility_sufficient_stats(
    outside_fraction: np.ndarray,
    geometry_mask: np.ndarray,
    global_count: np.ndarray,
    tolerance: float,
) -> ChannelAdmissibilitySufficientStats:
    values = np.where(geometry_mask, outside_fraction, 0.0).astype(np.float64)
    if not np.isfinite(outside_fraction[geometry_mask]).all():
        raise ValueError("Non-finite outside fraction under geometry-valid mask.")
    if np.any(outside_fraction[geometry_mask] < -1.0e-7) or np.any(outside_fraction[geometry_mask] > 1.0 + 1.0e-7):
        raise ValueError("Outside fraction outside [0, 1] under geometry-valid mask.")
    excess = np.maximum(values - float(tolerance), 0.0)
    return ChannelAdmissibilitySufficientStats(
        sum_outside=values.sum(axis=2),
        count=geometry_mask.sum(axis=2).astype(np.int64),
        count_viol_2pct=((outside_fraction > float(tolerance)) & geometry_mask).sum(axis=2).astype(np.int64),
        count_any_viol=((outside_fraction > 0.0) & geometry_mask).sum(axis=2).astype(np.int64),
        count_viol_10pct=((outside_fraction > 0.10) & geometry_mask).sum(axis=2).astype(np.int64),
        sum_excess=np.where(geometry_mask, excess, 0.0).sum(axis=2),
        sum_penalty_equivalent=np.where(geometry_mask, excess**2, 0.0).sum(axis=2),
        global_count=global_count.astype(np.int64),
    )


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(np.asarray(numerator, dtype=np.float64), np.nan, dtype=np.float64),
        where=np.asarray(denominator) > 0,
    )


def _stepwise_median(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    horizon = values.shape[1]
    output = np.full((horizon,), np.nan, dtype=np.float64)
    for step_index in range(horizon):
        step_values = values[:, step_index, :][mask[:, step_index, :]]
        if step_values.size:
            output[step_index] = float(np.median(step_values))
    return output


def channel_admissibility_metrics_from_stats(
    stats: ChannelAdmissibilitySufficientStats,
    outside_fraction: np.ndarray,
    geometry_mask: np.ndarray,
    true_stats: ChannelAdmissibilitySufficientStats,
    true_outside_fraction: np.ndarray,
    true_geometry_mask: np.ndarray,
    tolerance: float,
) -> ChannelAdmissibilityMetricCurves:
    count = stats.count.sum(axis=0)
    true_count = true_stats.count.sum(axis=0)
    global_count = stats.global_count.sum(axis=0)
    mean_outside = _safe_divide(stats.sum_outside.sum(axis=0), count)
    true_mean = _safe_divide(true_stats.sum_outside.sum(axis=0), true_count)
    viol_2 = _safe_divide(stats.count_viol_2pct.sum(axis=0), count)
    true_viol_2 = _safe_divide(true_stats.count_viol_2pct.sum(axis=0), true_count)
    viol_10 = _safe_divide(stats.count_viol_10pct.sum(axis=0), count)
    true_viol_10 = _safe_divide(true_stats.count_viol_10pct.sum(axis=0), true_count)
    return ChannelAdmissibilityMetricCurves(
        mean_outside_fraction=mean_outside,
        median_outside_fraction=_stepwise_median(outside_fraction, geometry_mask),
        violation_rate_2pct=viol_2,
        any_violation_rate=_safe_divide(stats.count_any_viol.sum(axis=0), count),
        violation_rate_10pct=viol_10,
        mean_excess_outside_fraction=_safe_divide(stats.sum_excess.sum(axis=0), count),
        mean_geometry_penalty_equivalent=_safe_divide(stats.sum_penalty_equivalent.sum(axis=0), count),
        true_mean_outside_fraction=true_mean,
        true_violation_rate_2pct=true_viol_2,
        true_violation_rate_10pct=true_viol_10,
        excess_mean_outside_fraction_vs_truth=mean_outside - true_mean,
        excess_violation_rate_2pct_vs_truth=viol_2 - true_viol_2,
        n_geometry_valid_samples=count.astype(np.int64),
        geometry_valid_fraction=_safe_divide(count, global_count),
    )


def bootstrap_channel_admissibility_metrics(
    stats: ChannelAdmissibilitySufficientStats,
    bootstrap_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    n_bootstrap, _ = bootstrap_indices.shape
    horizon = stats.count.shape[1]
    names = [
        "mean_outside_fraction",
        "violation_rate_2pct",
        "any_violation_rate",
        "violation_rate_10pct",
        "mean_excess_outside_fraction",
        "mean_geometry_penalty_equivalent",
    ]
    output = {name: np.empty((n_bootstrap, horizon), dtype=np.float64) for name in names}
    for replicate, indices in enumerate(bootstrap_indices):
        count = stats.count[indices].sum(axis=0)
        output["mean_outside_fraction"][replicate] = _safe_divide(stats.sum_outside[indices].sum(axis=0), count)
        output["violation_rate_2pct"][replicate] = _safe_divide(stats.count_viol_2pct[indices].sum(axis=0), count)
        output["any_violation_rate"][replicate] = _safe_divide(stats.count_any_viol[indices].sum(axis=0), count)
        output["violation_rate_10pct"][replicate] = _safe_divide(stats.count_viol_10pct[indices].sum(axis=0), count)
        output["mean_excess_outside_fraction"][replicate] = _safe_divide(stats.sum_excess[indices].sum(axis=0), count)
        output["mean_geometry_penalty_equivalent"][replicate] = _safe_divide(
            stats.sum_penalty_equivalent[indices].sum(axis=0),
            count,
        )
    return output


def compute_trajectory_admissibility_metrics(
    outside_fraction: np.ndarray,
    geometry_mask: np.ndarray,
    tolerance: float,
) -> TrajectoryAdmissibilityMetrics:
    first_steps = []
    episode_durations = []
    persistent_count = 0
    n_evaluable = 0
    n_violating = 0
    n_windows, _, max_droplets = geometry_mask.shape
    for window_index in range(n_windows):
        for slot_index in range(max_droplets):
            valid_steps = np.flatnonzero(geometry_mask[window_index, :, slot_index])
            if valid_steps.size == 0:
                continue
            n_evaluable += 1
            violations = outside_fraction[window_index, valid_steps, slot_index] > float(tolerance)
            if not violations.any():
                continue
            n_violating += 1
            first_local = int(np.flatnonzero(violations)[0])
            first_step = int(valid_steps[first_local] + 1)
            first_steps.append(first_step)
            if violations[first_local:].all():
                persistent_count += 1

            run_length = 0
            for is_violation in violations:
                if is_violation:
                    run_length += 1
                elif run_length:
                    episode_durations.append(run_length)
                    run_length = 0
            if run_length:
                episode_durations.append(run_length)

    return TrajectoryAdmissibilityMetrics(
        n_evaluable_trajectories=n_evaluable,
        n_trajectories_with_violation=n_violating,
        trajectory_violation_fraction=n_violating / n_evaluable if n_evaluable else float("nan"),
        mean_first_violation_step=float(np.mean(first_steps)) if first_steps else float("nan"),
        median_first_violation_step=float(np.median(first_steps)) if first_steps else float("nan"),
        n_violation_episodes=len(episode_durations),
        mean_violation_episode_duration=float(np.mean(episode_durations)) if episode_durations else float("nan"),
        median_violation_episode_duration=float(np.median(episode_durations)) if episode_durations else float("nan"),
        persistent_violation_fraction=persistent_count / n_violating if n_violating else float("nan"),
    )


def summarize_channel_context(context: ChannelAdmissibilityContext, tolerance: float) -> None:
    total_metric = int(context.global_count.sum())
    geometry_count = int(context.geometry_valid_mask.sum())
    print(f"Channel mask path: {context.channel_mask_path}", flush=True)
    print(f"BBox source path: {context.detections_csv_path}", flush=True)
    print("BBox lookup key: (frame, track_id) -> (bbox_w, bbox_h)", flush=True)
    print("Future frame convention: future_frame = rollout_start_frame + zero_based_rollout_step", flush=True)
    print(
        f"BBox coverage under global metric mask: "
        f"{context.bbox_lookup_coverage:.6f} "
        f"({context.bbox_candidate_count - context.bbox_missing_count}/{context.bbox_candidate_count})",
        flush=True,
    )
    print(
        f"Geometry-valid samples: {geometry_count}/{total_metric} "
        f"({geometry_count / max(total_metric, 1):.6f})",
        flush=True,
    )
    true_values = context.true_outside_fraction[context.geometry_valid_mask]
    if true_values.size:
        print(
            "True footprint baseline outside_fraction "
            f"mean={float(np.mean(true_values)):.6f} "
            f"p50={float(np.median(true_values)):.6f} "
            f"rate>{tolerance:.2f}={float(np.mean(true_values > tolerance)):.6f} "
            f"rate>0.10={float(np.mean(true_values > 0.10)):.6f}",
            flush=True,
        )


def validate_common_channel_masks(
    predictions: dict[str, RolloutPrediction],
    context: ChannelAdmissibilityContext,
) -> None:
    reference = next(iter(predictions.values()))
    for model_name, prediction in predictions.items():
        np.testing.assert_array_equal(prediction.window_id, context.window_id)
        np.testing.assert_array_equal(prediction.rollout_start_frame, context.rollout_start_frame)
        np.testing.assert_array_equal(prediction.track_ids, context.track_ids)
        np.testing.assert_array_equal(prediction.metric_mask, reference.metric_mask)
    print("Common geometry-valid mask reused across models: ok", flush=True)


def validate_channel_sanity_checks(context: ChannelAdmissibilityContext) -> None:
    sanity = run_geometry_loss_sanity_tests()
    inside = sanity["all_true_overlap"]
    outside = sanity["all_false_overlap"]
    partial = sanity["boundary_crossing_overlap"]
    if not (inside <= 1.0e-12 and outside >= 1.0 - 1.0e-12 and 0.0 < partial < 1.0):
        raise AssertionError(f"Unexpected geometry sanity-test result: {sanity}")
    compare_vectorized_scalar_geometry(context, max_samples=50)
    compare_numpy_torch_geometry(context, max_samples=50)
    print(
        "Synthetic ellipse sanity: inside~0, outside~1, partial in (0,1): ok",
        flush=True,
    )


def compare_vectorized_scalar_geometry(context: ChannelAdmissibilityContext, max_samples: int) -> None:
    indices = list(zip(*np.nonzero(context.geometry_valid_mask)))[:max_samples]
    if not indices:
        return
    channel_mask = load_channel_mask_bool(context.channel_mask_path)
    diffs = []
    for window_index, step_index, slot_index in indices:
        x, y = context.true_position[window_index, step_index, slot_index]
        width = float(context.bbox_w[window_index, step_index, slot_index])
        height = float(context.bbox_h[window_index, step_index, slot_index])
        scalar = compute_ellipse_outside_fraction(float(x), float(y), width, height, channel_mask)
        vectorized = float(context.true_outside_fraction[window_index, step_index, slot_index])
        diffs.append(abs(scalar - vectorized))
    max_abs = float(np.max(diffs))
    if max_abs > 1.0e-7:
        raise AssertionError(f"Vectorized ellipse rasterizer differs from scalar helper: max_abs_diff={max_abs}")
    print(
        f"Vectorized rasterizer vs scalar helper on {len(indices)} samples: max_abs_diff={max_abs:.6g}",
        flush=True,
    )


def compare_numpy_torch_geometry(context: ChannelAdmissibilityContext, max_samples: int) -> None:
    indices = list(zip(*np.nonzero(context.geometry_valid_mask)))[:max_samples]
    if not indices:
        return
    centroids = []
    sizes = []
    numpy_values = []
    for window_index, step_index, slot_index in indices:
        centroids.append(context.true_position[window_index, step_index, slot_index].tolist())
        sizes.append([
            context.bbox_w[window_index, step_index, slot_index],
            context.bbox_h[window_index, step_index, slot_index],
        ])
        numpy_values.append(context.true_outside_fraction[window_index, step_index, slot_index])
    channel_mask = torch.as_tensor(load_channel_mask_bool(context.channel_mask_path).astype(np.float32))
    with torch.inference_mode():
        torch_values = compute_ellipse_outside_fraction_torch(
            torch.as_tensor(centroids, dtype=torch.float32),
            torch.as_tensor(sizes, dtype=torch.float32),
            channel_mask,
            64,
            64,
        ).detach().cpu().numpy()
    numpy_values = np.asarray(numpy_values, dtype=np.float32)
    max_abs = float(np.max(np.abs(torch_values - numpy_values)))
    mean_abs = float(np.mean(np.abs(torch_values - numpy_values)))
    print(
        f"NumPy exact rasterizer vs training torch helper on {len(indices)} samples: "
        f"mean_abs_diff={mean_abs:.6f} max_abs_diff={max_abs:.6f}",
        flush=True,
    )


def summarize_tangent_basis(basis: DirectionalBasis, metric_mask: np.ndarray) -> None:
    axis_valid = metric_mask & basis.axis_valid
    orientation_valid = metric_mask & basis.orientation_valid
    total = int(metric_mask.sum())
    axis_count = int(axis_valid.sum())
    orientation_count = int(orientation_valid.sum())
    print(f"Metric-valid samples: {total}", flush=True)
    print(f"Axis-valid samples: {axis_count}/{total} ({axis_count / max(total, 1):.6f})", flush=True)
    print(
        f"Orientation-valid samples: {orientation_count}/{axis_count} "
        f"({orientation_count / max(axis_count, 1):.6f} of axis-valid)",
        flush=True,
    )

    raw_quality = basis.quality[metric_mask & np.isfinite(basis.quality)]
    accepted_quality = basis.quality[axis_valid & np.isfinite(basis.quality)]
    raw_distance = basis.distance_to_centerline[metric_mask & np.isfinite(basis.distance_to_centerline)]
    branch_margin = basis.branch_distance_margin[metric_mask & np.isfinite(basis.branch_distance_margin)]
    relative_margin = basis.branch_relative_margin[metric_mask & np.isfinite(basis.branch_relative_margin)]

    print_distribution("Raw PCA quality ratio", raw_quality)
    print_distribution("Accepted-axis PCA quality ratio", accepted_quality)
    print_distribution("Raw distance to centerline [px]", raw_distance)
    print_distribution("Branch distance margin [px]", branch_margin)
    print_distribution("Branch relative margin", relative_margin)
    for threshold in (0.5, 1.0, 2.0, 5.0):
        count = int((branch_margin < threshold).sum())
        print(f"Branch margin < {threshold:g} px: {count}/{branch_margin.size}", flush=True)

    statuses, counts = np.unique(basis.status[metric_mask], return_counts=True)
    print("Tangent status counts:", flush=True)
    for status, count in zip(statuses, counts):
        print(f"  {status}: {int(count)} ({int(count) / max(total, 1):.6f})", flush=True)


def print_distribution(label: str, values: np.ndarray) -> None:
    if values.size == 0:
        print(f"{label}: no finite values", flush=True)
        return
    print(
        f"{label}: n={values.size} min={np.nanmin(values):.3f} "
        f"p05={np.nanpercentile(values,5):.3f} p25={np.nanpercentile(values,25):.3f} "
        f"median={np.nanmedian(values):.3f} p75={np.nanpercentile(values,75):.3f} "
        f"p95={np.nanpercentile(values,95):.3f} max={np.nanmax(values):.3f}",
        flush=True,
    )


def write_directional_metrics_csv(
    path: Path,
    metrics: dict[str, DirectionalMetricCurves],
    bootstrap: dict[str, dict[str, np.ndarray]],
) -> None:
    fields = [
        "model", "rollout_step",
        "tangential_rmse", "tangential_rmse_ci_low", "tangential_rmse_ci_high",
        "normal_rmse", "normal_rmse_ci_low", "normal_rmse_ci_high",
        "tangential_bias", "tangential_bias_ci_low", "tangential_bias_ci_high",
        "normal_bias", "normal_bias_ci_low", "normal_bias_ci_high",
        "anisotropy_ratio", "anisotropy_ratio_ci_low", "anisotropy_ratio_ci_high",
        "n_axis_valid_samples", "axis_valid_fraction",
        "n_orientation_valid_samples", "orientation_valid_fraction",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model_name, curves in metrics.items():
            ci = {name: percentile_ci(bootstrap[model_name][name]) for name in [
                "tangential_rmse", "normal_rmse", "tangential_bias", "normal_bias", "anisotropy_ratio"
            ]}
            for step in range(len(curves.tangential_rmse)):
                writer.writerow({
                    "model": model_name,
                    "rollout_step": step + 1,
                    "tangential_rmse": curves.tangential_rmse[step],
                    "tangential_rmse_ci_low": ci["tangential_rmse"][0][step],
                    "tangential_rmse_ci_high": ci["tangential_rmse"][1][step],
                    "normal_rmse": curves.normal_rmse[step],
                    "normal_rmse_ci_low": ci["normal_rmse"][0][step],
                    "normal_rmse_ci_high": ci["normal_rmse"][1][step],
                    "tangential_bias": curves.tangential_bias[step],
                    "tangential_bias_ci_low": ci["tangential_bias"][0][step],
                    "tangential_bias_ci_high": ci["tangential_bias"][1][step],
                    "normal_bias": curves.normal_bias[step],
                    "normal_bias_ci_low": ci["normal_bias"][0][step],
                    "normal_bias_ci_high": ci["normal_bias"][1][step],
                    "anisotropy_ratio": curves.anisotropy_ratio[step],
                    "anisotropy_ratio_ci_low": ci["anisotropy_ratio"][0][step],
                    "anisotropy_ratio_ci_high": ci["anisotropy_ratio"][1][step],
                    "n_axis_valid_samples": int(curves.n_axis_valid_samples[step]),
                    "axis_valid_fraction": curves.axis_valid_fraction[step],
                    "n_orientation_valid_samples": int(curves.n_orientation_valid_samples[step]),
                    "orientation_valid_fraction": curves.orientation_valid_fraction[step],
                })


def save_directional_data(path: Path, basis: DirectionalBasis | None, stats: dict[str, DirectionalSufficientStats]) -> None:
    if basis is None:
        raise RuntimeError("No directional basis available.")
    arrays = {
        "tangent": basis.tangent,
        "normal": basis.normal,
        "axis_valid": basis.axis_valid,
        "orientation_valid": basis.orientation_valid,
        "tangent_quality": basis.quality,
        "distance_to_centerline": basis.distance_to_centerline,
        "branch_name": basis.branch_name,
        "second_nearest_branch_name": basis.second_nearest_branch_name,
        "second_nearest_branch_distance": basis.second_nearest_branch_distance,
        "branch_distance_margin": basis.branch_distance_margin,
        "branch_relative_margin": basis.branch_relative_margin,
        "tangent_status": basis.status,
        "model_names": np.asarray(list(stats), dtype=str),
    }
    for model_name, item in stats.items():
        prefix = f"{model_name}__"
        arrays[prefix + "sse_parallel"] = item.sse_parallel
        arrays[prefix + "sse_perp"] = item.sse_perp
        arrays[prefix + "sum_parallel"] = item.sum_parallel
        arrays[prefix + "sum_perp"] = item.sum_perp
        arrays[prefix + "axis_count"] = item.axis_count
        arrays[prefix + "orientation_count"] = item.orientation_count
        arrays[prefix + "global_count"] = item.global_count
    np.savez_compressed(path, **arrays)


def plot_directional_metric(
    output_dir: Path,
    metrics: dict[str, DirectionalMetricCurves],
    bootstrap: dict[str, dict[str, np.ndarray]],
    metric_name: str,
    ylabel: str,
    stem: str,
    zero_line: bool = False,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.8, 3.2), constrained_layout=True)
    steps = None
    for model_name, curves in metrics.items():
        values = getattr(curves, metric_name)
        steps = np.arange(1, len(values) + 1)
        low, high = percentile_ci(bootstrap[model_name][metric_name])
        line = ax.plot(steps, values, linewidth=1.8, label=model_name)[0]
        ax.fill_between(steps, low, high, color=line.get_color(), alpha=0.18, linewidth=0)
    if zero_line:
        ax.axhline(0.0, color="0.35", linewidth=0.8)
    ax.set_xlabel("Rollout step")
    ax.set_ylabel(ylabel)
    ax.set_xlim(1, int(steps[-1]) if steps is not None else ROLLOUT_HORIZON)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="0.9", linewidth=0.6)
    ax.legend(frameon=False)
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    return pdf_path, png_path


def plot_junction_decision_outcomes(
    output_dir: Path,
    metrics: dict[str, JunctionDecisionMetrics],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_names = list(metrics)
    outcome_names = ["correct", "wrong_branch", "no_commitment", "unclassifiable"]
    colors = {
        "correct": "#4C78A8",
        "wrong_branch": "#E45756",
        "no_commitment": "#F2CF5B",
        "unclassifiable": "#BAB0AC",
    }
    values = np.zeros((len(outcome_names), len(model_names)), dtype=np.float64)
    for model_index, model_name in enumerate(model_names):
        item = metrics[model_name]
        denominator = max(int(item.n_true_junction_events), 1)
        values[:, model_index] = [
            item.n_correct / denominator,
            item.n_wrong_branch / denominator,
            item.n_no_commitment / denominator,
            item.n_unclassifiable / denominator,
        ]

    fig, ax = plt.subplots(figsize=(4.8, 3.2), constrained_layout=True)
    x = np.arange(len(model_names))
    bottom = np.zeros((len(model_names),), dtype=np.float64)
    for outcome_index, outcome_name in enumerate(outcome_names):
        ax.bar(
            x,
            values[outcome_index],
            bottom=bottom,
            width=0.62,
            label=outcome_name,
            color=colors[outcome_name],
            edgecolor="white",
            linewidth=0.6,
        )
        bottom += values[outcome_index]
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Fraction of true junction events")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="0.9", linewidth=0.6)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2)
    pdf_path = output_dir / "junction_decision_outcomes.pdf"
    png_path = output_dir / "junction_decision_outcomes.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    return pdf_path, png_path


def plot_junction_confusion_matrix(
    output_dir: Path,
    events: list[JunctionDecisionEvent],
    model_name: str,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    branches = ["left", "right"]
    branch_to_index = {branch: index for index, branch in enumerate(branches)}
    matrix = np.zeros((2, 2), dtype=np.int64)
    for event in events:
        status = event.decision_status_by_model[model_name]
        if status not in ("correct", "wrong_branch"):
            continue
        true_branch = event.true_branch
        pred_branch = event.pred_branch_by_model[model_name]
        if true_branch not in branch_to_index or pred_branch not in branch_to_index:
            continue
        matrix[branch_to_index[true_branch], branch_to_index[pred_branch]] += 1

    row_totals = matrix.sum(axis=1, keepdims=True)
    percentages = np.divide(
        matrix,
        np.maximum(row_totals, 1),
        out=np.zeros_like(matrix, dtype=np.float64),
        where=row_totals > 0,
    )

    fig, ax = plt.subplots(figsize=(3.5, 3.2), constrained_layout=True)
    image = ax.imshow(percentages, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(2))
    ax.set_yticks(np.arange(2))
    ax.set_xticklabels(["pred left", "pred right"])
    ax.set_yticklabels(["true left", "true right"])
    ax.set_title(model_name)
    for row in range(2):
        for col in range(2):
            value = percentages[row, col]
            text_color = "white" if value >= 0.55 else "black"
            ax.text(
                col,
                row,
                f"{matrix[row, col]}\n{100.0 * value:.1f}%",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
            )
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Row-normalized fraction")
    stem = f"junction_confusion_{model_name}"
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    return pdf_path, png_path


def build_default_adapters(args, device: torch.device) -> dict[str, RolloutModelAdapter]:
    if args.predictions_path is not None:
        return {}
    adapters: dict[str, RolloutModelAdapter] = {}
    if args.history_checkpoint:
        adapters["history_20"] = HistoryRolloutModelAdapter("history_20", Path(args.history_checkpoint), device)
    if args.markovian_checkpoint:
        adapters["markovian"] = MarkovianRolloutModelAdapter("markovian", Path(args.markovian_checkpoint), device)
    if args.geometry_aware_checkpoint:
        adapters["geometry_aware"] = MarkovianRolloutModelAdapter(
            "geometry_aware",
            Path(args.geometry_aware_checkpoint),
            device,
        )
    if not adapters:
        raise ValueError("At least one checkpoint must be provided.")
    horizons = {adapter.horizon for adapter in adapters.values()}
    if horizons != {ROLLOUT_HORIZON}:
        raise ValueError(f"All adapters must use horizon {ROLLOUT_HORIZON}; got {horizons}")
    return adapters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare trained droplet rollout models on aligned validation windows.")
    parser.add_argument("--npz-path", type=Path, default=DEFAULT_NPZ_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--predictions-path", type=Path, default=None, help="Load saved rollout_predictions.npz and skip inference.")
    parser.add_argument("--centerline-csv", type=Path, default=DEFAULT_CENTERLINE_CSV)
    parser.add_argument("--history-checkpoint", type=Path, default=DEFAULT_HISTORY_CHECKPOINT)
    parser.add_argument("--markovian-checkpoint", type=Path, default=DEFAULT_MARKOVIAN_CHECKPOINT)
    parser.add_argument("--geometry-aware-checkpoint", type=Path, default=DEFAULT_GEOMETRY_AWARE_CHECKPOINT)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--stride", type=int, default=STRIDE)
    parser.add_argument("--pca-half-window", type=int, default=8)
    parser.add_argument("--min-tangent-quality", type=float, default=3.0)
    parser.add_argument("--max-centerline-distance", type=float, default=30.0)
    parser.add_argument("--min-orientation-speed", type=float, default=1.0e-3)
    parser.add_argument("--channel-mask", type=Path, default=DEFAULT_CHANNEL_MASK_PATH)
    parser.add_argument("--detections-csv", type=Path, default=DEFAULT_DETECTIONS_CSV_PATH)
    parser.add_argument("--geometry-tolerance", type=float, default=GEOMETRY_TOLERANCE)
    parser.add_argument(
        "--min-branch-distance-margin",
        type=float,
        default=None,
        help="Optional absolute nearest-vs-second-nearest branch margin threshold in pixels.",
    )
    parser.add_argument(
        "--min-branch-relative-margin",
        type=float,
        default=None,
        help="Optional relative nearest-vs-second-nearest branch margin threshold.",
    )
    parser.add_argument(
        "--junction-outgoing-branches",
        nargs="+",
        default=["left", "right"],
        help="Centerline branch names that represent outgoing decision branches.",
    )
    parser.add_argument(
        "--junction-incoming-branches",
        nargs="+",
        default=["inlet", "outlet"],
        help="Centerline branch names treated as incoming/non-decision branches for event starts.",
    )
    parser.add_argument(
        "--junction-commitment-steps",
        type=int,
        default=3,
        help="Number of consecutive evaluable rollout steps required for branch commitment.",
    )
    parser.add_argument("--skip-directional", action="store_true")
    parser.add_argument("--skip-channel-admissibility", action="store_true")
    parser.add_argument("--skip-junction-decisions", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    adapters = build_default_adapters(args, device)
    comparator = RolloutModelComparator(
        adapters=adapters,
        npz_path=args.npz_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        max_windows=args.max_windows,
        stride=args.stride,
    )
    if args.predictions_path is not None:
        print(f"Loading saved predictions and skipping inference: {args.predictions_path}", flush=True)
        comparator.load_predictions(args.predictions_path)
        prediction_path = Path(args.predictions_path)
    else:
        comparator.run_inference()
        prediction_path = comparator.save_predictions()
    comparator.compute_metrics()
    comparator.bootstrap_metrics()
    stepwise_path, integrated_path = comparator.save_metrics()
    position_pdf, position_png = comparator.plot_position_rmse()
    velocity_pdf, velocity_png = comparator.plot_velocity_rmse()
    directional_csv = directional_data = None
    directional_plots = None
    channel_csv = channel_trajectory_csv = channel_data = None
    channel_plots = None
    junction_metrics_csv = junction_events_csv = None
    junction_outcome_plots = None
    junction_confusion_plots = None
    if not args.skip_directional:
        comparator.compute_directional_metrics(
            centerline_csv=args.centerline_csv,
            pca_half_window=args.pca_half_window,
            min_quality=args.min_tangent_quality,
            max_centerline_distance=args.max_centerline_distance,
            min_orientation_speed=args.min_orientation_speed,
            min_branch_distance_margin=args.min_branch_distance_margin,
            min_branch_relative_margin=args.min_branch_relative_margin,
        )
        directional_csv, directional_data = comparator.save_directional_outputs()
        directional_plots = comparator.plot_directional_metrics()
    if not args.skip_channel_admissibility:
        comparator.compute_channel_admissibility_metrics(
            channel_mask_path=args.channel_mask,
            detections_csv_path=args.detections_csv,
            tolerance=args.geometry_tolerance,
        )
        channel_csv, channel_trajectory_csv, channel_data = comparator.save_channel_admissibility_outputs()
        channel_plots = comparator.plot_channel_admissibility_metrics()
    if not args.skip_junction_decisions:
        comparator.compute_junction_decision_metrics(
            centerline_csv=args.centerline_csv,
            pca_half_window=args.pca_half_window,
            min_quality=args.min_tangent_quality,
            max_centerline_distance=args.max_centerline_distance,
            min_orientation_speed=args.min_orientation_speed,
            min_branch_distance_margin=args.min_branch_distance_margin,
            min_branch_relative_margin=args.min_branch_relative_margin,
            outgoing_branches=tuple(args.junction_outgoing_branches),
            incoming_branches=tuple(args.junction_incoming_branches),
            commitment_steps=args.junction_commitment_steps,
        )
        junction_metrics_csv, junction_events_csv = comparator.save_junction_decision_outputs()
        junction_outcome_plots, junction_confusion_plots = comparator.plot_junction_decision_outputs()

    print("Models evaluated:")
    for model_name in comparator.predictions:
        print(f"  {model_name}")
    first_prediction = next(iter(comparator.predictions.values()))
    print(f"Validation windows: {len(first_prediction.window_id)}")
    print(f"Rollout horizon: {first_prediction.pred_position.shape[1]}")
    print(f"Bootstrap replicates: {args.n_bootstrap}")
    print("Output:")
    print(f"  predictions: {prediction_path}")
    print(f"  stepwise metrics: {stepwise_path}")
    print(f"  integrated metrics: {integrated_path}")
    print(f"  position RMSE plot: {position_pdf}, {position_png}")
    print(f"  velocity RMSE plot: {velocity_pdf}, {velocity_png}")
    if directional_csv is not None:
        print(f"  directional metrics: {directional_csv}")
        print(f"  directional data: {directional_data}")
        print(f"  tangential RMSE plot: {directional_plots[0][0]}, {directional_plots[0][1]}")
        print(f"  normal RMSE plot: {directional_plots[1][0]}, {directional_plots[1][1]}")
        print(f"  tangential bias plot: {directional_plots[2][0]}, {directional_plots[2][1]}")
    if channel_csv is not None:
        print(f"  channel admissibility metrics: {channel_csv}")
        print(f"  channel trajectory metrics: {channel_trajectory_csv}")
        print(f"  channel admissibility data: {channel_data}")
        print(f"  mean outside plot: {channel_plots[0][0]}, {channel_plots[0][1]}")
        print(f"  2pct violation plot: {channel_plots[1][0]}, {channel_plots[1][1]}")
        print(f"  geometry penalty equivalent plot: {channel_plots[2][0]}, {channel_plots[2][1]}")
    if junction_metrics_csv is not None:
        print(f"  junction decision metrics: {junction_metrics_csv}")
        print(f"  junction decision events: {junction_events_csv}")
        print(f"  junction decision outcomes plot: {junction_outcome_plots[0]}, {junction_outcome_plots[1]}")
        for model_name, paths in junction_confusion_plots.items():
            print(f"  junction confusion {model_name}: {paths[0]}, {paths[1]}")


if __name__ == "__main__":
    main()
