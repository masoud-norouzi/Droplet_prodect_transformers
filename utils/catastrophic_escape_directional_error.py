from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import utils.rollout_comparison as comparison


DEFAULT_PREDICTIONS_PATH = Path("outputs/models/rollout_model_comparison/rollout_predictions.npz")
DEFAULT_DIRECTIONAL_DATA_PATH = Path("outputs/models/rollout_model_comparison/directional_error_data.npz")
DEFAULT_CHANNEL_DATA_PATH = Path("outputs/models/rollout_model_comparison_channel_full_smoke/channel_admissibility_data.npz")
DEFAULT_OUTPUT_DIR = Path("outputs/models/rollout_model_comparison")
SUMMARY_CSV_NAME = "catastrophic_escape_preceding_directional_error.csv"
EVENT_CSV_NAME = "catastrophic_escape_directional_event_summary.csv"
PLOT_STEM = "catastrophic_escape_preceding_directional_error"
MODEL_ORDER = ("history_20", "markovian", "geometry_aware")
RELATIVE_STEPS = tuple(range(-10, 1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze tangential/normal errors before catastrophic channel escape."
    )
    parser.add_argument("--predictions-path", type=Path, default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--directional-data-path", type=Path, default=DEFAULT_DIRECTIONAL_DATA_PATH)
    parser.add_argument("--channel-data-path", type=Path, default=DEFAULT_CHANNEL_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--centerline-csv", type=Path, default=comparison.DEFAULT_CENTERLINE_CSV)
    parser.add_argument("--escape-threshold", type=float, default=0.95)
    parser.add_argument("--pca-half-window", type=int, default=8)
    parser.add_argument("--min-tangent-quality", type=float, default=3.0)
    parser.add_argument("--max-centerline-distance", type=float, default=30.0)
    parser.add_argument("--min-orientation-speed", type=float, default=1.0e-3)
    return parser.parse_args()


def load_predictions(path: Path) -> dict[str, comparison.RolloutPrediction]:
    print(f"Loading saved predictions: {path}", flush=True)
    return comparison.load_predictions_npz(path)


def load_or_build_basis(
    args: argparse.Namespace,
    reference: comparison.RolloutPrediction,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if args.directional_data_path.exists():
        print(f"Loading true-position tangent/normal basis: {args.directional_data_path}", flush=True)
        with np.load(args.directional_data_path, allow_pickle=False) as data:
            required = ("tangent", "normal", "axis_valid")
            missing = [name for name in required if name not in data]
            if missing:
                raise ValueError(f"Directional data is missing required arrays: {missing}")
            return data["tangent"], data["normal"], data["axis_valid"].astype(bool)

    print("Directional data cache not found; rebuilding true-position basis without model inference.", flush=True)
    estimator = comparison.CenterlineTangentEstimator(
        centerline_csv=args.centerline_csv,
        pca_half_window=args.pca_half_window,
        min_quality=args.min_tangent_quality,
        max_centerline_distance=args.max_centerline_distance,
        min_orientation_speed=args.min_orientation_speed,
        min_branch_distance_margin=None,
        min_branch_relative_margin=None,
    )
    basis = estimator.build_basis(reference)
    return basis.tangent, basis.normal, basis.axis_valid


def load_channel_arrays(path: Path, model_name: str) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(
            f"Channel outside-fraction data is required for this analysis and was not found: {path}"
        )
    with np.load(path, allow_pickle=False) as data:
        outside_key = f"{model_name}__outside_fraction"
        if outside_key not in data:
            raise KeyError(f"Missing outside-fraction array in {path}: {outside_key}")
        if "geometry_valid_mask" not in data:
            raise KeyError(f"Missing geometry_valid_mask in {path}")
        return data[outside_key].astype(np.float32), data["geometry_valid_mask"].astype(bool)


def compute_abs_directional_errors(
    prediction: comparison.RolloutPrediction,
    tangent: np.ndarray,
    normal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    error = prediction.pred_position - prediction.true_position
    e_parallel = np.sum(error * tangent, axis=-1)
    e_perp = np.sum(error * normal, axis=-1)
    return np.abs(e_parallel), np.abs(e_perp)


def find_catastrophic_events(
    outside_fraction: np.ndarray,
    geometry_valid_mask: np.ndarray,
    metric_mask: np.ndarray,
    threshold: float,
) -> list[tuple[int, int, int]]:
    event_mask = geometry_valid_mask & metric_mask & np.isfinite(outside_fraction) & (outside_fraction >= threshold)
    windows, slots = np.nonzero(np.any(event_mask, axis=1))
    events: list[tuple[int, int, int]] = []
    for window_index, slot in zip(windows.tolist(), slots.tolist()):
        steps = np.flatnonzero(event_mask[window_index, :, slot])
        if steps.size:
            events.append((int(window_index), int(slot), int(steps[0])))
    return events


def sample_event_errors(
    abs_parallel: np.ndarray,
    abs_perp: np.ndarray,
    valid_mask: np.ndarray,
    window_index: int,
    slot: int,
    escape_step: int,
) -> dict[int, tuple[float, float] | None]:
    samples: dict[int, tuple[float, float] | None] = {}
    for relative_step in RELATIVE_STEPS:
        step = escape_step + relative_step
        if step < 0 or step >= abs_parallel.shape[1] or not bool(valid_mask[window_index, step, slot]):
            samples[relative_step] = None
            continue
        parallel = float(abs_parallel[window_index, step, slot])
        perp = float(abs_perp[window_index, step, slot])
        if not (np.isfinite(parallel) and np.isfinite(perp)):
            samples[relative_step] = None
            continue
        samples[relative_step] = parallel, perp
    return samples


def first_parallel_exceeds_perp(samples: dict[int, tuple[float, float] | None]) -> tuple[int | None, int | None]:
    for relative_step in RELATIVE_STEPS:
        value = samples[relative_step]
        if value is None:
            continue
        if value[0] > value[1]:
            return relative_step, relative_step
    return None, None


def build_analysis(args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, dict[str, object]]]:
    predictions = load_predictions(args.predictions_path)
    reference = next(iter(predictions.values()))
    tangent, normal, axis_valid = load_or_build_basis(args, reference)

    aggregate: dict[tuple[str, int], dict[str, list[float]]] = {}
    event_rows: list[dict[str, object]] = []
    model_report: dict[str, dict[str, object]] = {}

    for model_name in MODEL_ORDER:
        if model_name not in predictions:
            continue
        prediction = predictions[model_name]
        print(f"Analyzing catastrophic escapes for {model_name}...", flush=True)
        outside_fraction, geometry_valid_mask = load_channel_arrays(args.channel_data_path, model_name)
        events = find_catastrophic_events(
            outside_fraction=outside_fraction,
            geometry_valid_mask=geometry_valid_mask,
            metric_mask=prediction.metric_mask,
            threshold=float(args.escape_threshold),
        )
        abs_parallel, abs_perp = compute_abs_directional_errors(prediction, tangent, normal)
        directional_valid = prediction.metric_mask & axis_valid & np.isfinite(abs_parallel) & np.isfinite(abs_perp)

        total_possible = 0
        total_valid = 0
        for window_index, slot, escape_step in events:
            samples = sample_event_errors(
                abs_parallel=abs_parallel,
                abs_perp=abs_perp,
                valid_mask=directional_valid,
                window_index=window_index,
                slot=slot,
                escape_step=escape_step,
            )
            for relative_step, value in samples.items():
                total_possible += 1
                if value is None:
                    continue
                total_valid += 1
                bucket = aggregate.setdefault(
                    (model_name, relative_step),
                    {"parallel": [], "perp": []},
                )
                bucket["parallel"].append(value[0])
                bucket["perp"].append(value[1])

            first_rel, first_rollout = first_parallel_exceeds_perp(samples)
            event_rows.append(
                {
                    "model": model_name,
                    "window_id": int(prediction.window_id[window_index]),
                    "rollout_start_frame": int(prediction.rollout_start_frame[window_index]),
                    "track_id": int(prediction.track_ids[window_index, slot]),
                    "slot": int(slot),
                    "first_catastrophic_escape_step": int(escape_step + 1),
                    "first_catastrophic_escape_relative_step": 0,
                    "abs_tangential_error_m10": value_or_nan(samples[-10], 0),
                    "abs_normal_error_m10": value_or_nan(samples[-10], 1),
                    "abs_tangential_error_m5": value_or_nan(samples[-5], 0),
                    "abs_normal_error_m5": value_or_nan(samples[-5], 1),
                    "abs_tangential_error_m1": value_or_nan(samples[-1], 0),
                    "abs_normal_error_m1": value_or_nan(samples[-1], 1),
                    "first_relative_step_parallel_exceeds_normal": "" if first_rel is None else int(first_rel),
                    "first_rollout_step_parallel_exceeds_normal": "" if first_rollout is None else int(escape_step + first_rollout + 1),
                }
            )

        model_report[model_name] = {
            "n_catastrophic_trajectories": len(events),
            "valid_directional_samples": total_valid,
            "possible_directional_samples": total_possible,
            "coverage": total_valid / total_possible if total_possible else np.nan,
        }

    summary_rows: list[dict[str, object]] = []
    for model_name in MODEL_ORDER:
        if model_name not in predictions:
            continue
        for relative_step in RELATIVE_STEPS:
            bucket = aggregate.get((model_name, relative_step), {"parallel": [], "perp": []})
            parallel = np.asarray(bucket["parallel"], dtype=np.float64)
            perp = np.asarray(bucket["perp"], dtype=np.float64)
            summary_rows.append(
                {
                    "model": model_name,
                    "relative_step": int(relative_step),
                    "mean_abs_tangential_error": float(np.mean(parallel)) if parallel.size else np.nan,
                    "median_abs_tangential_error": float(np.median(parallel)) if parallel.size else np.nan,
                    "mean_abs_normal_error": float(np.mean(perp)) if perp.size else np.nan,
                    "median_abs_normal_error": float(np.median(perp)) if perp.size else np.nan,
                    "n_valid_trajectories": int(parallel.size),
                }
            )
    return summary_rows, event_rows, model_report


def value_or_nan(value: tuple[float, float] | None, index: int) -> float:
    return np.nan if value is None else float(value[index])


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(output_dir: Path, summary_rows: list[dict[str, object]]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.2), sharex=True, sharey=False, constrained_layout=True)
    for axis, model_name in zip(axes, MODEL_ORDER):
        rows = [row for row in summary_rows if row["model"] == model_name]
        x = np.asarray([row["relative_step"] for row in rows], dtype=np.int64)
        parallel = np.asarray([row["mean_abs_tangential_error"] for row in rows], dtype=np.float64)
        perp = np.asarray([row["mean_abs_normal_error"] for row in rows], dtype=np.float64)
        axis.plot(x, parallel, marker="o", linewidth=1.8, label=r"$|e_{\parallel}|$")
        axis.plot(x, perp, marker="s", linewidth=1.8, label=r"$|e_{\perp}|$")
        axis.axvline(0, color="0.35", linestyle="--", linewidth=0.9)
        axis.set_title(model_name)
        axis.set_xlabel("Steps relative to first escape")
        axis.set_xlim(-10, 0)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", color="0.9", linewidth=0.6)
    axes[0].set_ylabel("Mean absolute position error [px]")
    axes[-1].legend(frameon=False, loc="upper left")
    pdf_path = output_dir / f"{PLOT_STEM}.pdf"
    png_path = output_dir / f"{PLOT_STEM}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    return png_path, pdf_path


def print_report(model_report: dict[str, dict[str, object]], summary_rows: list[dict[str, object]]) -> None:
    print("\nCatastrophic escape directional-error report")
    for model_name in MODEL_ORDER:
        if model_name not in model_report:
            continue
        item = model_report[model_name]
        print(
            f"{model_name}: catastrophic trajectories={item['n_catastrophic_trajectories']}, "
            f"directional coverage={item['valid_directional_samples']}/{item['possible_directional_samples']} "
            f"({100.0 * item['coverage']:.1f}%)"
        )
        model_rows = {int(row["relative_step"]): row for row in summary_rows if row["model"] == model_name}
        for relative_step in (-10, -5, -1, 0):
            row = model_rows[relative_step]
            print(
                f"  step {relative_step:>3}: "
                f"mean |e_parallel|={row['mean_abs_tangential_error']:.2f}, "
                f"mean |e_perp|={row['mean_abs_normal_error']:.2f}, "
                f"n={row['n_valid_trajectories']}"
            )


def main() -> None:
    args = parse_args()
    summary_rows, event_rows, model_report = build_analysis(args)
    summary_path = args.output_dir / SUMMARY_CSV_NAME
    event_path = args.output_dir / EVENT_CSV_NAME
    write_csv(summary_path, summary_rows)
    write_csv(event_path, event_rows)
    png_path, pdf_path = plot_summary(args.output_dir, summary_rows)
    print_report(model_report, summary_rows)
    print("\nOutput files:")
    print(f"  {summary_path}")
    print(f"  {event_path}")
    print(f"  {png_path}")
    print(f"  {pdf_path}")


if __name__ == "__main__":
    main()
