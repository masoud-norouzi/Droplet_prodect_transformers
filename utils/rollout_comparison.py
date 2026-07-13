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
        arrays = {"model_names": np.asarray(list(self.predictions), dtype=object)}
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
    for replicate in range(n_bootstrap):
        indices = bootstrap_indices[replicate]
        stepwise = compute_stepwise_rmse(prediction, window_indices=indices)
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


def build_default_adapters(args, device: torch.device) -> dict[str, RolloutModelAdapter]:
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
    parser.add_argument("--history-checkpoint", type=Path, default=DEFAULT_HISTORY_CHECKPOINT)
    parser.add_argument("--markovian-checkpoint", type=Path, default=DEFAULT_MARKOVIAN_CHECKPOINT)
    parser.add_argument("--geometry-aware-checkpoint", type=Path, default=DEFAULT_GEOMETRY_AWARE_CHECKPOINT)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--stride", type=int, default=STRIDE)
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
    comparator.run_inference()
    prediction_path = comparator.save_predictions()
    comparator.compute_metrics()
    comparator.bootstrap_metrics()
    stepwise_path, integrated_path = comparator.save_metrics()
    position_pdf, position_png = comparator.plot_position_rmse()
    velocity_pdf, velocity_png = comparator.plot_velocity_rmse()

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


if __name__ == "__main__":
    main()
