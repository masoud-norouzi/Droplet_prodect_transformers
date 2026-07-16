from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import utils.rollout_comparison as comparison
from utils.channel_mask import read_centerline_csv


DEFAULT_PREDICTIONS_PATH = Path("outputs/models/rollout_model_comparison/rollout_predictions.npz")
DEFAULT_DIRECTIONAL_DATA_PATH = Path("outputs/models/rollout_model_comparison/directional_error_data.npz")
DEFAULT_CHANNEL_DATA_PATH = Path("outputs/models/rollout_model_comparison_channel_full_smoke/channel_admissibility_data.npz")
DEFAULT_OUTPUT_DIR = Path("outputs/models/rollout_model_comparison")
ARCHIVE_ROOT = Path("outputs/models/Archive-round 1- wrong droplet velocity at the inlet")
FALLBACK_PREDICTIONS_PATH = ARCHIVE_ROOT / "rollout_model_comparison/rollout_predictions.npz"
FALLBACK_DIRECTIONAL_DATA_PATH = ARCHIVE_ROOT / "rollout_model_comparison/directional_error_data.npz"
FALLBACK_CHANNEL_DATA_PATH = ARCHIVE_ROOT / "rollout_model_comparison_channel_full_smoke/channel_admissibility_data.npz"
MODEL_ORDER = ("history_20", "markovian", "geometry_aware")

STEPWISE_CSV = "phase_direction_stepwise_summary.csv"
ALIGNED_CSV = "phase_direction_escape_aligned_summary.csv"
LAG_LEAD_CSV = "phase_direction_lag_lead_summary.csv"
RISK_CSV = "phase_direction_conditional_escape_risk.csv"
COVERAGE_CSV = "phase_direction_coverage_summary.csv"
EVENT_CSV = "phase_direction_event_level_summary.csv"


@dataclass
class CenterlineTangentLookup:
    tree: cKDTree
    points: np.ndarray
    tangents: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze phase error and velocity/local-geometry mismatch before catastrophic escapes."
    )
    parser.add_argument("--predictions-path", type=Path, default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--directional-data-path", type=Path, default=DEFAULT_DIRECTIONAL_DATA_PATH)
    parser.add_argument("--channel-data-path", type=Path, default=DEFAULT_CHANNEL_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--centerline-csv", type=Path, default=comparison.DEFAULT_CENTERLINE_CSV)
    parser.add_argument("--escape-threshold", type=float, default=0.95)
    parser.add_argument("--phase-deadband", type=float, default=1.0)
    parser.add_argument("--velocity-eps", type=float, default=1.0e-6)
    parser.add_argument("--max-predicted-centerline-distance", type=float, default=40.0)
    parser.add_argument("--escape-aligned-min-step", type=int, default=-15)
    parser.add_argument("--near-term-steps", type=int, default=5)
    parser.add_argument("--min-bin-count", type=int, default=25)
    parser.add_argument("--smoke-max-windows", type=int, default=None)
    return parser.parse_args()


def resolve_existing_path(path: Path, fallbacks: list[Path], label: str) -> Path:
    if path.exists():
        return path
    for fallback in fallbacks:
        if fallback.exists():
            print(f"{label} not found at {path}; using fallback {fallback}", flush=True)
            return fallback
    raise FileNotFoundError(f"{label} not found: {path}; checked fallbacks: {fallbacks}")


def wrapped_angle(local_tangent: np.ndarray, velocity_unit: np.ndarray) -> np.ndarray:
    cross = local_tangent[..., 0] * velocity_unit[..., 1] - local_tangent[..., 1] * velocity_unit[..., 0]
    dot = np.sum(local_tangent * velocity_unit, axis=-1)
    return np.arctan2(cross, dot)


def run_angle_sanity_checks() -> None:
    tangent = np.asarray([[1.0, 0.0]], dtype=np.float64)
    tests = [
        ("identical", np.asarray([[1.0, 0.0]]), 0.0),
        ("perpendicular_ccw", np.asarray([[0.0, 1.0]]), 90.0),
        ("perpendicular_cw", np.asarray([[0.0, -1.0]]), -90.0),
        ("opposite", np.asarray([[-1.0, 0.0]]), 180.0),
    ]
    print("Angle sanity checks:", flush=True)
    for name, vector, expected in tests:
        angle = float(np.degrees(wrapped_angle(tangent, vector))[0])
        print(f"  {name}: {angle:.3f} deg (expected about {expected:g})", flush=True)


def load_predictions(path: Path, max_windows: int | None) -> dict[str, comparison.RolloutPrediction]:
    print(f"Loading saved predictions: {path}", flush=True)
    predictions = comparison.load_predictions_npz(path)
    if max_windows is None:
        return predictions
    truncated: dict[str, comparison.RolloutPrediction] = {}
    for model_name, item in predictions.items():
        n = min(int(max_windows), item.window_id.shape[0])
        truncated[model_name] = comparison.RolloutPrediction(
            model_name=item.model_name,
            window_id=item.window_id[:n],
            rollout_start_frame=item.rollout_start_frame[:n],
            frame_start=item.frame_start[:n],
            track_ids=item.track_ids[:n],
            pred_position=item.pred_position[:n],
            true_position=item.true_position[:n],
            pred_velocity=item.pred_velocity[:n],
            true_velocity=item.true_velocity[:n],
            valid_mask=item.valid_mask[:n],
            boundary_mask=item.boundary_mask[:n],
        )
    return truncated


def load_directional_basis(path: Path, reference: comparison.RolloutPrediction) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing cached directional basis: {path}")
    print(f"Loading true-position tangent/normal basis: {path}", flush=True)
    with np.load(path, allow_pickle=False) as data:
        tangent = data["tangent"][: reference.window_id.shape[0]]
        normal = data["normal"][: reference.window_id.shape[0]]
        axis_valid = data["axis_valid"][: reference.window_id.shape[0]].astype(bool)
    return tangent, normal, axis_valid


def load_channel_data(path: Path, model_name: str, n_windows: int) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing cached channel outside-fraction data: {path}")
    with np.load(path, allow_pickle=False) as data:
        outside_key = f"{model_name}__outside_fraction"
        if outside_key not in data:
            raise KeyError(f"Missing {outside_key} in {path}")
        outside = data[outside_key][:n_windows].astype(np.float32)
        geometry_valid = data["geometry_valid_mask"][:n_windows].astype(bool)
    return outside, geometry_valid


def build_centerline_tangent_lookup(centerline_csv: Path, half_window: int = 8) -> CenterlineTangentLookup:
    branches, _ = read_centerline_csv(centerline_csv)
    points_list: list[np.ndarray] = []
    tangents_list: list[np.ndarray] = []
    for branch_points in branches.values():
        points = np.asarray(branch_points, dtype=np.float64)
        if len(points) < 2:
            continue
        tangents = np.zeros_like(points)
        for index in range(len(points)):
            lo = max(0, index - half_window)
            hi = min(len(points), index + half_window + 1)
            if hi - lo < 2:
                lo = max(0, index - 1)
                hi = min(len(points), index + 2)
            delta = points[hi - 1] - points[lo]
            norm = float(np.linalg.norm(delta))
            if norm <= 0:
                delta = points[min(index + 1, len(points) - 1)] - points[max(index - 1, 0)]
                norm = float(np.linalg.norm(delta))
            tangents[index] = delta / max(norm, 1.0e-12)
        points_list.append(points)
        tangents_list.append(tangents)
    all_points = np.concatenate(points_list, axis=0)
    all_tangents = np.concatenate(tangents_list, axis=0)
    return CenterlineTangentLookup(tree=cKDTree(all_points), points=all_points, tangents=all_tangents)


def classify_phase(signed_tangential_error: np.ndarray, deadband: float) -> np.ndarray:
    labels = np.full(signed_tangential_error.shape, "near_aligned", dtype="<U16")
    labels[signed_tangential_error < -float(deadband)] = "lagging"
    labels[signed_tangential_error > float(deadband)] = "leading"
    labels[~np.isfinite(signed_tangential_error)] = "invalid"
    return labels


def event_arrays(
    outside_fraction: np.ndarray,
    valid_mask: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    catastrophic_step_mask = valid_mask & np.isfinite(outside_fraction) & (outside_fraction >= threshold)
    escaping = np.any(catastrophic_step_mask, axis=1)
    first_escape = np.full(escaping.shape, -1, dtype=np.int16)
    windows, slots = np.nonzero(escaping)
    for window, slot in zip(windows.tolist(), slots.tolist()):
        first_escape[window, slot] = int(np.flatnonzero(catastrophic_step_mask[window, :, slot])[0])
    valid_outside = valid_mask & np.isfinite(outside_fraction)
    max_input = np.where(valid_outside, outside_fraction, -np.inf)
    max_outside = np.max(max_input, axis=1)
    max_outside[~np.any(valid_outside, axis=1)] = np.nan
    return escaping, first_escape, max_outside


def summarize(values: np.ndarray) -> dict[str, float | int]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "mean": np.nan, "median": np.nan, "q25": np.nan, "q75": np.nan}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "q25": float(np.percentile(values, 25)),
        "q75": float(np.percentile(values, 75)),
    }


def build_sample_arrays(
    prediction: comparison.RolloutPrediction,
    true_tangent: np.ndarray,
    true_normal: np.ndarray,
    true_axis_valid: np.ndarray,
    lookup: CenterlineTangentLookup,
    outside_fraction: np.ndarray,
    geometry_valid: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    metric_mask = prediction.metric_mask
    position_error = prediction.pred_position - prediction.true_position
    signed_parallel = np.sum(position_error * true_tangent, axis=-1)
    true_normal_error = np.sum(position_error * true_normal, axis=-1)

    pred_velocity = prediction.pred_velocity.astype(np.float64)
    pred_speed = np.linalg.norm(pred_velocity, axis=-1)
    velocity_valid = np.isfinite(pred_velocity).all(axis=-1) & (pred_speed > float(args.velocity_eps))
    velocity_unit = np.divide(
        pred_velocity,
        pred_speed[..., None],
        out=np.zeros_like(pred_velocity, dtype=np.float64),
        where=pred_speed[..., None] > float(args.velocity_eps),
    )

    flat_positions = prediction.pred_position.reshape(-1, 2).astype(np.float64)
    finite_positions = np.isfinite(flat_positions).all(axis=1)
    distance = np.full((flat_positions.shape[0],), np.nan, dtype=np.float64)
    local_tangent_flat = np.full_like(flat_positions, np.nan, dtype=np.float64)
    if finite_positions.any():
        distance_values, nearest = lookup.tree.query(flat_positions[finite_positions], k=1)
        distance[finite_positions] = distance_values
        local_tangent_flat[finite_positions] = lookup.tangents[nearest]
    local_tangent = local_tangent_flat.reshape(prediction.pred_position.shape)
    distance_to_centerline = distance.reshape(metric_mask.shape)

    orient_valid = true_axis_valid & np.isfinite(true_tangent).all(axis=-1) & np.isfinite(local_tangent).all(axis=-1)
    flip = np.sum(local_tangent * true_tangent, axis=-1) < 0
    local_tangent[orient_valid & flip] *= -1.0
    local_normal = np.stack([-local_tangent[..., 1], local_tangent[..., 0]], axis=-1)

    local_tangent_valid = (
        np.isfinite(local_tangent).all(axis=-1)
        & np.isfinite(distance_to_centerline)
        & (distance_to_centerline <= float(args.max_predicted_centerline_distance))
    )

    angle_rad = wrapped_angle(local_tangent, velocity_unit)
    angle_deg = np.degrees(angle_rad)
    pred_tangential_velocity = np.sum(pred_velocity * local_tangent, axis=-1)
    pred_normal_velocity = np.sum(pred_velocity * local_normal, axis=-1)

    final_valid = metric_mask & geometry_valid & true_axis_valid & velocity_valid & local_tangent_valid
    return {
        "metric_mask": metric_mask,
        "geometry_valid": geometry_valid,
        "velocity_valid": velocity_valid,
        "local_tangent_valid": local_tangent_valid,
        "final_valid": final_valid,
        "signed_tangential_error": signed_parallel,
        "abs_tangential_error": np.abs(signed_parallel),
        "true_normal_error": true_normal_error,
        "direction_mismatch_rad": angle_rad,
        "direction_mismatch_deg": angle_deg,
        "absolute_direction_mismatch_deg": np.abs(angle_deg),
        "predicted_normal_velocity": pred_normal_velocity,
        "predicted_tangential_velocity": pred_tangential_velocity,
        "distance_to_centerline": distance_to_centerline,
        "outside_fraction": outside_fraction,
    }


def add_stepwise_rows(
    rows: list[dict[str, object]],
    model_name: str,
    arrays: dict[str, np.ndarray],
    escaping: np.ndarray,
    phase_labels: np.ndarray,
) -> None:
    horizon = arrays["final_valid"].shape[1]
    for step in range(horizon):
        for trajectory_class, trajectory_mask in (
            ("escaping", escaping),
            ("non_escaping", ~escaping),
        ):
            step_mask_2d = arrays["final_valid"][:, step, :] & trajectory_mask
            mask = np.zeros_like(arrays["final_valid"], dtype=bool)
            mask[:, step, :] = step_mask_2d
            append_stats_row(rows, model_name, step + 1, trajectory_class, "all", mask, arrays)

            if trajectory_class == "escaping":
                for phase_class in ("lagging", "leading", "near_aligned"):
                    phase_mask_2d = step_mask_2d & (phase_labels[:, step, :] == phase_class)
                    phase_mask = np.zeros_like(arrays["final_valid"], dtype=bool)
                    phase_mask[:, step, :] = phase_mask_2d
                    append_stats_row(rows, model_name, step + 1, trajectory_class, phase_class, phase_mask, arrays)


def append_stats_row(
    rows: list[dict[str, object]],
    model_name: str,
    step_value: int,
    trajectory_class: str,
    phase_class: str,
    mask: np.ndarray,
    arrays: dict[str, np.ndarray],
    relative: bool = False,
) -> None:
    signed_phase = summarize(arrays["signed_tangential_error"][mask])
    signed_mismatch = summarize(arrays["direction_mismatch_deg"][mask])
    abs_mismatch = summarize(arrays["absolute_direction_mismatch_deg"][mask])
    normal_velocity = summarize(arrays["predicted_normal_velocity"][mask])
    row = {
        "model": model_name,
        "trajectory_class": trajectory_class,
        "phase_class": phase_class,
        "sample_count": signed_phase["count"],
        "mean_signed_tangential_error": signed_phase["mean"],
        "median_signed_tangential_error": signed_phase["median"],
        "q25_signed_tangential_error": signed_phase["q25"],
        "q75_signed_tangential_error": signed_phase["q75"],
        "mean_signed_direction_mismatch_deg": signed_mismatch["mean"],
        "median_signed_direction_mismatch_deg": signed_mismatch["median"],
        "mean_absolute_direction_mismatch_deg": abs_mismatch["mean"],
        "median_absolute_direction_mismatch_deg": abs_mismatch["median"],
        "q25_absolute_direction_mismatch_deg": abs_mismatch["q25"],
        "q75_absolute_direction_mismatch_deg": abs_mismatch["q75"],
        "mean_predicted_normal_velocity": normal_velocity["mean"],
        "median_predicted_normal_velocity": normal_velocity["median"],
    }
    row["relative_step" if relative else "rollout_step"] = step_value
    rows.append(row)


def add_escape_aligned_rows(
    rows: list[dict[str, object]],
    model_name: str,
    arrays: dict[str, np.ndarray],
    first_escape: np.ndarray,
    phase_labels: np.ndarray,
    min_relative_step: int,
) -> None:
    for relative_step in range(int(min_relative_step), 1):
        base_mask = np.zeros_like(arrays["final_valid"], dtype=bool)
        windows, slots = np.nonzero(first_escape >= 0)
        for window, slot in zip(windows.tolist(), slots.tolist()):
            step = int(first_escape[window, slot]) + relative_step
            if 0 <= step < base_mask.shape[1]:
                base_mask[window, step, slot] = True
        mask = base_mask & arrays["final_valid"]
        append_escape_aligned_stats_row(rows, model_name, relative_step, "all", mask, arrays)
        for phase_class in ("lagging", "leading", "near_aligned"):
            phase_mask = mask & (phase_labels == phase_class)
            append_escape_aligned_stats_row(rows, model_name, relative_step, phase_class, phase_mask, arrays)


def append_escape_aligned_stats_row(
    rows: list[dict[str, object]],
    model_name: str,
    relative_step: int,
    phase_class: str,
    mask: np.ndarray,
    arrays: dict[str, np.ndarray],
) -> None:
    signed_phase = summarize(arrays["signed_tangential_error"][mask])
    abs_phase = summarize(arrays["abs_tangential_error"][mask])
    signed_mismatch = summarize(arrays["direction_mismatch_deg"][mask])
    abs_mismatch = summarize(arrays["absolute_direction_mismatch_deg"][mask])
    normal_velocity = summarize(arrays["predicted_normal_velocity"][mask])
    rows.append(
        {
            "model": model_name,
            "relative_step": relative_step,
            "phase_class": phase_class,
            "sample_count": signed_phase["count"],
            "mean_signed_tangential_error": signed_phase["mean"],
            "median_signed_tangential_error": signed_phase["median"],
            "mean_absolute_tangential_error": abs_phase["mean"],
            "median_absolute_tangential_error": abs_phase["median"],
            "mean_signed_direction_mismatch_deg": signed_mismatch["mean"],
            "median_signed_direction_mismatch_deg": signed_mismatch["median"],
            "mean_absolute_direction_mismatch_deg": abs_mismatch["mean"],
            "median_absolute_direction_mismatch_deg": abs_mismatch["median"],
            "mean_predicted_normal_velocity": normal_velocity["mean"],
            "median_predicted_normal_velocity": normal_velocity["median"],
        }
    )


def add_risk_rows(
    rows: list[dict[str, object]],
    model_name: str,
    arrays: dict[str, np.ndarray],
    escaping: np.ndarray,
    first_escape: np.ndarray,
    near_term_steps: int,
    min_bin_count: int,
) -> None:
    phase_bins = np.asarray([-np.inf, -20, -10, -5, -2, 2, 5, 10, 20, np.inf], dtype=np.float64)
    mismatch_bins = np.asarray([0, 5, 10, 20, 30, 45, 90, 180], dtype=np.float64)
    abs_phase_bins = np.asarray([0, 2, 5, 10, 20, np.inf], dtype=np.float64)

    final_valid = arrays["final_valid"]
    horizon = final_valid.shape[1]
    pre_escape = final_valid.copy()
    near_term = np.zeros_like(final_valid, dtype=bool)
    for window in range(final_valid.shape[0]):
        for slot in range(final_valid.shape[2]):
            escape_step = int(first_escape[window, slot])
            if escape_step >= 0:
                pre_escape[window, escape_step:, slot] = False
                for step in range(max(0, escape_step - near_term_steps), escape_step):
                    near_term[window, step, slot] = True
    # Non-escaping samples remain pre_escape candidates. Escaping samples at/after escape are excluded.
    pre_escape &= final_valid

    add_binned_rows(
        rows,
        model_name,
        "joint_signed_phase_x_abs_direction_mismatch",
        arrays["signed_tangential_error"],
        phase_bins,
        arrays["absolute_direction_mismatch_deg"],
        mismatch_bins,
        pre_escape,
        escaping,
        near_term,
        min_bin_count,
    )
    add_binned_rows(
        rows,
        model_name,
        "control_abs_tangential_error_only",
        arrays["abs_tangential_error"],
        abs_phase_bins,
        None,
        None,
        pre_escape,
        escaping,
        near_term,
        min_bin_count,
    )
    add_binned_rows(
        rows,
        model_name,
        "control_abs_direction_mismatch_only",
        arrays["absolute_direction_mismatch_deg"],
        mismatch_bins,
        None,
        None,
        pre_escape,
        escaping,
        near_term,
        min_bin_count,
    )
    add_high_risk_comparison_rows(
        rows,
        model_name,
        arrays,
        pre_escape,
        escaping,
        near_term,
        min_bin_count,
    )


def add_binned_rows(
    rows: list[dict[str, object]],
    model_name: str,
    analysis_type: str,
    x_values: np.ndarray,
    x_bins: np.ndarray,
    y_values: np.ndarray | None,
    y_bins: np.ndarray | None,
    sample_mask: np.ndarray,
    escaping: np.ndarray,
    near_term: np.ndarray,
    min_bin_count: int,
) -> None:
    if y_values is None or y_bins is None:
        y_iter = [(None, None, None)]
    else:
        y_iter = [(idx, y_bins[idx], y_bins[idx + 1]) for idx in range(len(y_bins) - 1)]
    for x_idx in range(len(x_bins) - 1):
        x_lo, x_hi = x_bins[x_idx], x_bins[x_idx + 1]
        x_mask = sample_mask & (x_values >= x_lo) & (x_values < x_hi)
        for y_idx, y_lo, y_hi in y_iter:
            mask = x_mask
            if y_values is not None:
                mask = mask & (y_values >= y_lo) & (y_values < y_hi)
            n = int(mask.sum())
            n_escaping = int((mask & escaping[:, None, :]).sum())
            n_near = int((mask & near_term).sum())
            rows.append(
                {
                    "model": model_name,
                    "analysis_type": analysis_type,
                    "signed_tangential_error_bin": format_bin(x_lo, x_hi),
                    "absolute_direction_mismatch_bin_deg": "" if y_idx is None else format_bin(y_lo, y_hi),
                    "sample_count": n,
                    "n_eventually_escaping": n_escaping,
                    "eventual_escape_fraction": n_escaping / n if n else np.nan,
                    "n_escape_within_next_5_steps": n_near,
                    "conditional_5_step_escape_fraction": n_near / n if n else np.nan,
                    "reliable": bool(n >= min_bin_count),
                }
            )


def add_high_risk_comparison_rows(
    rows: list[dict[str, object]],
    model_name: str,
    arrays: dict[str, np.ndarray],
    sample_mask: np.ndarray,
    escaping: np.ndarray,
    near_term: np.ndarray,
    min_bin_count: int,
) -> None:
    conditions = {
        "control_phase_high_abs_tangential_ge_20": arrays["abs_tangential_error"] >= 20.0,
        "control_direction_high_abs_mismatch_ge_45": arrays["absolute_direction_mismatch_deg"] >= 45.0,
        "joint_high_phase_ge_20_and_mismatch_ge_45": (
            (arrays["abs_tangential_error"] >= 20.0)
            & (arrays["absolute_direction_mismatch_deg"] >= 45.0)
        ),
    }
    for name, condition in conditions.items():
        mask = sample_mask & condition
        n = int(mask.sum())
        n_escaping = int((mask & escaping[:, None, :]).sum())
        n_near = int((mask & near_term).sum())
        rows.append(
            {
                "model": model_name,
                "analysis_type": name,
                "signed_tangential_error_bin": "high_risk_condition",
                "absolute_direction_mismatch_bin_deg": "high_risk_condition",
                "sample_count": n,
                "n_eventually_escaping": n_escaping,
                "eventual_escape_fraction": n_escaping / n if n else np.nan,
                "n_escape_within_next_5_steps": n_near,
                "conditional_5_step_escape_fraction": n_near / n if n else np.nan,
                "reliable": bool(n >= min_bin_count),
            }
        )


def format_bin(lo: float, hi: float) -> str:
    def fmt(value: float) -> str:
        if np.isneginf(value):
            return "-inf"
        if np.isposinf(value):
            return "inf"
        return f"{value:g}"
    return f"[{fmt(float(lo))}, {fmt(float(hi))})"


def add_event_rows(
    rows: list[dict[str, object]],
    model_name: str,
    prediction: comparison.RolloutPrediction,
    arrays: dict[str, np.ndarray],
    escaping: np.ndarray,
    first_escape: np.ndarray,
    max_outside: np.ndarray,
    phase_labels: np.ndarray,
) -> None:
    for window in range(prediction.window_id.shape[0]):
        for slot in range(prediction.track_ids.shape[1]):
            trajectory_valid = arrays["final_valid"][window, :, slot]
            if not trajectory_valid.any():
                continue
            escape_step = int(first_escape[window, slot])
            pre_mask = trajectory_valid.copy()
            if escape_step >= 0:
                pre_mask[escape_step:,] = False
                if not pre_mask.any():
                    pre_mask = trajectory_valid.copy()
            phase = arrays["signed_tangential_error"][window, :, slot][pre_mask]
            abs_phase = arrays["abs_tangential_error"][window, :, slot][pre_mask]
            abs_mismatch = arrays["absolute_direction_mismatch_deg"][window, :, slot][pre_mask]
            normal_velocity = arrays["predicted_normal_velocity"][window, :, slot][pre_mask]
            immediate_label = ""
            if escape_step > 0:
                immediate_label = str(phase_labels[window, escape_step - 1, slot])
            rows.append(
                {
                    "model": model_name,
                    "window_id": int(prediction.window_id[window]),
                    "rollout_start_frame": int(prediction.rollout_start_frame[window]),
                    "track_id": int(prediction.track_ids[window, slot]),
                    "slot": int(slot),
                    "catastrophic_escape": bool(escaping[window, slot]),
                    "first_catastrophic_step": "" if escape_step < 0 else int(escape_step + 1),
                    "maximum_outside_fraction": float(max_outside[window, slot]),
                    "mean_signed_tangential_error_before_escape": float(np.nanmean(phase)),
                    "median_signed_tangential_error_before_escape": float(np.nanmedian(phase)),
                    "maximum_absolute_tangential_error_before_escape": float(np.nanmax(abs_phase)),
                    "mean_absolute_direction_mismatch_before_escape": float(np.nanmean(abs_mismatch)),
                    "maximum_absolute_direction_mismatch_before_escape": float(np.nanmax(abs_mismatch)),
                    "mean_predicted_normal_velocity_before_escape": float(np.nanmean(normal_velocity)),
                    "phase_class_immediately_before_escape": immediate_label,
                    "n_metric_valid_steps": int(arrays["metric_mask"][window, :, slot].sum()),
                    "n_geometry_valid_steps": int(arrays["geometry_valid"][window, :, slot].sum()),
                    "n_velocity_valid_steps": int(arrays["velocity_valid"][window, :, slot].sum()),
                    "n_predicted_tangent_valid_steps": int(arrays["local_tangent_valid"][window, :, slot].sum()),
                    "n_final_analysis_valid_steps": int(trajectory_valid.sum()),
                }
            )


def coverage_row(model_name: str, arrays: dict[str, np.ndarray], escaping: np.ndarray) -> dict[str, object]:
    metric = arrays["metric_mask"]
    geometry = arrays["geometry_valid"]
    velocity = arrays["velocity_valid"]
    tangent = arrays["local_tangent_valid"]
    final = arrays["final_valid"]
    return {
        "model": model_name,
        "n_evaluable_trajectories": int(np.any(metric, axis=1).sum()),
        "n_catastrophic_trajectories": int(escaping.sum()),
        "total_valid_trajectory_step_samples": int(metric.sum()),
        "samples_with_geometry_valid": int((metric & geometry).sum()),
        "samples_with_valid_predicted_velocity": int((metric & velocity).sum()),
        "samples_with_valid_predicted_position_tangent": int((metric & tangent).sum()),
        "final_samples_included": int(final.sum()),
        "final_inclusion_fraction": float(final.sum() / max(metric.sum(), 1)),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_stepwise(output_dir: Path, rows: list[dict[str, object]], metric: str, stem: str, ylabel: str) -> tuple[Path, Path]:
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.2), sharex=True, constrained_layout=True)
    for axis, model_name in zip(axes, MODEL_ORDER):
        for trajectory_class, color in (("escaping", "#E45756"), ("non_escaping", "#4C78A8")):
            model_rows = [
                row for row in rows
                if row["model"] == model_name and row["trajectory_class"] == trajectory_class and row["phase_class"] == "all"
            ]
            x = np.asarray([row["rollout_step"] for row in model_rows], dtype=np.float64)
            median = np.asarray([row[f"median_{metric}"] for row in model_rows], dtype=np.float64)
            q25_name = f"q25_{metric}" if f"q25_{metric}" in model_rows[0] else None
            q75_name = f"q75_{metric}" if f"q75_{metric}" in model_rows[0] else None
            axis.plot(x, median, label=trajectory_class, color=color, linewidth=1.8)
            if q25_name and q75_name:
                q25 = np.asarray([row[q25_name] for row in model_rows], dtype=np.float64)
                q75 = np.asarray([row[q75_name] for row in model_rows], dtype=np.float64)
                axis.fill_between(x, q25, q75, color=color, alpha=0.18, linewidth=0)
        axis.axhline(0.0, color="0.35", linewidth=0.8)
        axis.set_title(model_name)
        axis.set_xlabel("Rollout step")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", color="0.9", linewidth=0.6)
    axes[0].set_ylabel(ylabel)
    axes[-1].legend(frameon=False)
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    return png_path, pdf_path


def plot_aligned(output_dir: Path, rows: list[dict[str, object]], metric: str, stem: str, ylabel: str) -> tuple[Path, Path]:
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.2), sharex=True, constrained_layout=True)
    for axis, model_name in zip(axes, MODEL_ORDER):
        for phase_class, color in (("lagging", "#F58518"), ("leading", "#54A24B"), ("near_aligned", "#4C78A8")):
            model_rows = [
                row for row in rows
                if row["model"] == model_name and row["phase_class"] == phase_class
            ]
            x = np.asarray([row["relative_step"] for row in model_rows], dtype=np.float64)
            y = np.asarray([row[f"median_{metric}"] for row in model_rows], dtype=np.float64)
            axis.plot(x, y, label=phase_class, color=color, linewidth=1.8)
        axis.axvline(0.0, color="0.35", linestyle="--", linewidth=0.9)
        axis.axhline(0.0, color="0.35", linewidth=0.8)
        axis.set_title(model_name)
        axis.set_xlabel("Steps relative to escape")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", color="0.9", linewidth=0.6)
    axes[0].set_ylabel(ylabel)
    axes[-1].legend(frameon=False)
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    return png_path, pdf_path


def plot_heatmaps(output_dir: Path, risk_rows: list[dict[str, object]]) -> list[tuple[Path, Path]]:
    paths: list[tuple[Path, Path]] = []
    joint_rows = [row for row in risk_rows if row["analysis_type"] == "joint_signed_phase_x_abs_direction_mismatch"]
    x_bins = sorted({row["signed_tangential_error_bin"] for row in joint_rows}, key=bin_sort_key)
    y_bins = sorted({row["absolute_direction_mismatch_bin_deg"] for row in joint_rows}, key=bin_sort_key)
    for model_name in MODEL_ORDER:
        matrix = np.full((len(y_bins), len(x_bins)), np.nan, dtype=np.float64)
        counts = np.zeros((len(y_bins), len(x_bins)), dtype=np.int64)
        for row in joint_rows:
            if row["model"] != model_name:
                continue
            yi = y_bins.index(row["absolute_direction_mismatch_bin_deg"])
            xi = x_bins.index(row["signed_tangential_error_bin"])
            matrix[yi, xi] = float(row["conditional_5_step_escape_fraction"])
            counts[yi, xi] = int(row["sample_count"])
        fig, ax = plt.subplots(figsize=(8.0, 4.0), constrained_layout=True)
        image = ax.imshow(matrix, cmap="magma", vmin=0.0, vmax=np.nanpercentile(matrix, 95) if np.isfinite(matrix).any() else 1.0, aspect="auto")
        ax.set_xticks(np.arange(len(x_bins)))
        ax.set_xticklabels(x_bins, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(np.arange(len(y_bins)))
        ax.set_yticklabels(y_bins, fontsize=8)
        ax.set_xlabel("Signed tangential error bin [px]")
        ax.set_ylabel("Absolute direction mismatch bin [deg]")
        ax.set_title(f"{model_name}: P(escape within next 5 steps)")
        for yi in range(len(y_bins)):
            for xi in range(len(x_bins)):
                if counts[yi, xi] <= 0 or not np.isfinite(matrix[yi, xi]):
                    continue
                ax.text(xi, yi, f"{matrix[yi, xi]:.3f}\n{counts[yi, xi]}", ha="center", va="center", fontsize=6, color="white" if matrix[yi, xi] > 0.02 else "black")
        fig.colorbar(image, ax=ax, label="Conditional 5-step escape fraction")
        pdf_path = output_dir / f"phase_direction_conditional_escape_heatmap_{model_name}.pdf"
        png_path = output_dir / f"phase_direction_conditional_escape_heatmap_{model_name}.png"
        fig.savefig(pdf_path)
        fig.savefig(png_path, dpi=300)
        plt.close(fig)
        paths.append((png_path, pdf_path))
    return paths


def bin_sort_key(label: str) -> float:
    first = label.strip("[]()").split(",")[0]
    if first == "-inf":
        return -1.0e99
    if first == "inf":
        return 1.0e99
    return float(first)


def plot_high_risk_controls(output_dir: Path, risk_rows: list[dict[str, object]]) -> tuple[Path, Path]:
    condition_names = [
        "control_phase_high_abs_tangential_ge_20",
        "control_direction_high_abs_mismatch_ge_45",
        "joint_high_phase_ge_20_and_mismatch_ge_45",
    ]
    labels = ["phase only", "direction only", "joint"]
    x = np.arange(len(condition_names))
    width = 0.24
    fig, ax = plt.subplots(figsize=(6.2, 3.4), constrained_layout=True)
    for model_index, model_name in enumerate(MODEL_ORDER):
        values = []
        for condition in condition_names:
            match = [row for row in risk_rows if row["model"] == model_name and row["analysis_type"] == condition]
            values.append(float(match[0]["conditional_5_step_escape_fraction"]) if match else np.nan)
        ax.bar(x + (model_index - 1) * width, values, width=width, label=model_name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("P(escape within next 5 steps)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="0.9", linewidth=0.6)
    ax.legend(frameon=False)
    pdf_path = output_dir / "phase_direction_high_risk_condition_comparison.pdf"
    png_path = output_dir / "phase_direction_high_risk_condition_comparison.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    return png_path, pdf_path


def write_outputs(
    output_dir: Path,
    stepwise_rows: list[dict[str, object]],
    lag_lead_rows: list[dict[str, object]],
    aligned_rows: list[dict[str, object]],
    risk_rows: list[dict[str, object]],
    coverage_rows: list[dict[str, object]],
    event_rows: list[dict[str, object]],
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        output_dir / STEPWISE_CSV,
        output_dir / LAG_LEAD_CSV,
        output_dir / ALIGNED_CSV,
        output_dir / RISK_CSV,
        output_dir / COVERAGE_CSV,
        output_dir / EVENT_CSV,
    ]
    for path, rows in zip(paths, [stepwise_rows, lag_lead_rows, aligned_rows, risk_rows, coverage_rows, event_rows]):
        write_csv(path, rows)
    return paths


def print_examples(model_name: str, prediction: comparison.RolloutPrediction, arrays: dict[str, np.ndarray], escaping: np.ndarray, first_escape: np.ndarray) -> None:
    windows, steps, slots = np.nonzero(arrays["final_valid"])
    print(f"\nInspectable examples for {model_name}:", flush=True)
    printed = 0
    for window, step, slot in zip(windows.tolist(), steps.tolist(), slots.tolist()):
        if printed >= 5:
            break
        print(
            f"  window={int(prediction.window_id[window])} step={step + 1} slot={slot} track={int(prediction.track_ids[window, slot])} "
            f"true=({prediction.true_position[window, step, slot, 0]:.2f},{prediction.true_position[window, step, slot, 1]:.2f}) "
            f"pred=({prediction.pred_position[window, step, slot, 0]:.2f},{prediction.pred_position[window, step, slot, 1]:.2f}) "
            f"signed_phase={arrays['signed_tangential_error'][window, step, slot]:.2f} "
            f"pred_vel=({prediction.pred_velocity[window, step, slot, 0]:.2f},{prediction.pred_velocity[window, step, slot, 1]:.2f}) "
            f"mismatch={arrays['direction_mismatch_deg'][window, step, slot]:.2f}deg "
            f"outside={arrays['outside_fraction'][window, step, slot]:.3f} "
            f"escape_step={int(first_escape[window, slot]) + 1 if escaping[window, slot] else ''}",
            flush=True,
        )
        printed += 1


def main() -> None:
    args = parse_args()
    args.predictions_path = resolve_existing_path(
        args.predictions_path,
        [FALLBACK_PREDICTIONS_PATH],
        "Saved rollout predictions",
    )
    args.directional_data_path = resolve_existing_path(
        args.directional_data_path,
        [FALLBACK_DIRECTIONAL_DATA_PATH],
        "Directional data",
    )
    args.channel_data_path = resolve_existing_path(
        args.channel_data_path,
        [FALLBACK_CHANNEL_DATA_PATH],
        "Channel outside-fraction data",
    )
    run_angle_sanity_checks()
    predictions = load_predictions(args.predictions_path, args.smoke_max_windows)
    reference = next(iter(predictions.values()))
    true_tangent, true_normal, true_axis_valid = load_directional_basis(args.directional_data_path, reference)
    lookup = build_centerline_tangent_lookup(args.centerline_csv)

    stepwise_rows: list[dict[str, object]] = []
    lag_lead_rows: list[dict[str, object]] = []
    aligned_rows: list[dict[str, object]] = []
    risk_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []

    print(
        "\nSigned tangential error convention: predicted_position - true_position dotted with the true-position tangent.",
        flush=True,
    )
    print("Positive means predicted droplet is ahead of truth along the true trajectory tangent.", flush=True)
    print("Negative means predicted droplet is behind truth along the true trajectory tangent.", flush=True)
    print("Near-term escape risk uses only samples before the first catastrophic escape step.", flush=True)
    print("Lagging and leading samples are summarized separately before any aggregation.", flush=True)

    for model_name in MODEL_ORDER:
        if model_name not in predictions:
            continue
        prediction = predictions[model_name]
        outside, geometry_valid = load_channel_data(args.channel_data_path, model_name, prediction.window_id.shape[0])
        arrays = build_sample_arrays(
            prediction=prediction,
            true_tangent=true_tangent,
            true_normal=true_normal,
            true_axis_valid=true_axis_valid,
            lookup=lookup,
            outside_fraction=outside,
            geometry_valid=geometry_valid,
            args=args,
        )
        escaping, first_escape, max_outside = event_arrays(outside, arrays["final_valid"], float(args.escape_threshold))
        phase_labels = classify_phase(arrays["signed_tangential_error"], float(args.phase_deadband))

        add_stepwise_rows(stepwise_rows, model_name, arrays, escaping, phase_labels)
        lag_lead_rows.extend([row for row in stepwise_rows if row["model"] == model_name and row["trajectory_class"] == "escaping" and row["phase_class"] != "all"])
        add_escape_aligned_rows(aligned_rows, model_name, arrays, first_escape, phase_labels, int(args.escape_aligned_min_step))
        add_risk_rows(risk_rows, model_name, arrays, escaping, first_escape, int(args.near_term_steps), int(args.min_bin_count))
        add_event_rows(event_rows, model_name, prediction, arrays, escaping, first_escape, max_outside, phase_labels)
        coverage_rows.append(coverage_row(model_name, arrays, escaping))
        print_examples(model_name, prediction, arrays, escaping, first_escape)

    output_paths = write_outputs(
        args.output_dir,
        stepwise_rows,
        lag_lead_rows,
        aligned_rows,
        risk_rows,
        coverage_rows,
        event_rows,
    )
    plot_paths = [
        *plot_stepwise(args.output_dir, stepwise_rows, "signed_tangential_error", "phase_direction_stepwise_signed_tangential_error", "Median signed tangential error [px]"),
        *plot_stepwise(args.output_dir, stepwise_rows, "absolute_direction_mismatch_deg", "phase_direction_stepwise_absolute_mismatch", "Median absolute direction mismatch [deg]"),
        *plot_aligned(args.output_dir, aligned_rows, "signed_tangential_error", "phase_direction_escape_aligned_signed_tangential_error", "Median signed tangential error [px]"),
        *plot_aligned(args.output_dir, aligned_rows, "absolute_direction_mismatch_deg", "phase_direction_escape_aligned_absolute_mismatch", "Median absolute direction mismatch [deg]"),
        *plot_high_risk_controls(args.output_dir, risk_rows),
    ]
    for png_path, pdf_path in plot_heatmaps(args.output_dir, risk_rows):
        plot_paths.extend([png_path, pdf_path])

    print("\nCoverage summary:")
    for row in coverage_rows:
        print(
            f"  {row['model']}: catastrophic={row['n_catastrophic_trajectories']}, "
            f"included={row['final_samples_included']}/{row['total_valid_trajectory_step_samples']} "
            f"({100.0 * row['final_inclusion_fraction']:.1f}%)",
            flush=True,
        )

    print("\nOutput files:")
    for path in output_paths + plot_paths:
        print(f"  {path}", flush=True)


if __name__ == "__main__":
    main()
