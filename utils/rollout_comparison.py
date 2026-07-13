from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Callable

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
        raise ValueError("No tangent-valid samples were found.")
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
    parser.add_argument("--skip-directional", action="store_true")
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


if __name__ == "__main__":
    main()
