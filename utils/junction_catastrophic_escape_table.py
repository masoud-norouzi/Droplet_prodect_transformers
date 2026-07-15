from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import utils.rollout_comparison as comparison


DEFAULT_PREDICTIONS_PATH = Path("outputs/models/rollout_model_comparison/rollout_predictions.npz")
DEFAULT_OUTPUT_DIR = Path("outputs/models/rollout_model_comparison")
DEFAULT_OUTPUT_CSV = DEFAULT_OUTPUT_DIR / "junction_decision_catastrophic_escape_table.csv"
DEFAULT_CHANNEL_DATA_PATH = DEFAULT_OUTPUT_DIR / "channel_admissibility_data.npz"

MODEL_ORDER = ("history_20", "markovian", "geometry_aware")
TABLE_ROWS = (
    ("left", "left", "correct", "correct"),
    ("left", "right", "wrong", "wrong_branch"),
    ("right", "right", "correct", "correct"),
    ("right", "left", "wrong", "wrong_branch"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize catastrophic channel escape rates for committed junction "
            "branch decisions."
        )
    )
    parser.add_argument("--predictions-path", type=Path, default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument(
        "--channel-data-path",
        type=Path,
        default=DEFAULT_CHANNEL_DATA_PATH,
        help="Optional saved channel_admissibility_data.npz to reuse if present.",
    )
    parser.add_argument("--centerline-csv", type=Path, default=comparison.DEFAULT_CENTERLINE_CSV)
    parser.add_argument("--channel-mask", type=Path, default=comparison.DEFAULT_CHANNEL_MASK_PATH)
    parser.add_argument("--detections-csv", type=Path, default=comparison.DEFAULT_DETECTIONS_CSV_PATH)
    parser.add_argument("--escape-threshold", type=float, default=0.95)
    parser.add_argument("--n-bootstrap", type=int, default=2, help="Only needed to initialize shared comparison helper.")
    return parser.parse_args()


def load_or_build_context(args: argparse.Namespace, reference: comparison.RolloutPrediction) -> comparison.ChannelAdmissibilityContext:
    print("Building channel/bbox context...", flush=True)
    return comparison.build_channel_admissibility_context(
        reference=reference,
        channel_mask_path=args.channel_mask,
        detections_csv_path=args.detections_csv,
    )


def load_saved_outside_fraction(channel_data_path: Path, model_name: str) -> tuple[np.ndarray, np.ndarray] | None:
    if not channel_data_path.exists():
        return None
    with np.load(channel_data_path, allow_pickle=False) as data:
        key = f"{model_name}__outside_fraction"
        if key not in data or "geometry_valid_mask" not in data:
            return None
        return data[key].astype(np.float32), data["geometry_valid_mask"].astype(bool)


def event_has_catastrophic_escape(
    event: comparison.JunctionDecisionEvent,
    model_name: str,
    prediction: comparison.RolloutPrediction,
    context: comparison.ChannelAdmissibilityContext,
    channel_mask: np.ndarray,
    escape_threshold: float,
    saved_outside: tuple[np.ndarray, np.ndarray] | None,
) -> bool:
    pred_commit_step = int(event.pred_commitment_step_by_model[model_name])
    if pred_commit_step < 0:
        return False

    window_index = int(event.window_row)
    slot = int(event.slot)
    if saved_outside is not None:
        outside_fraction, geometry_valid_mask = saved_outside
        values = outside_fraction[window_index, pred_commit_step:, slot]
        valid = geometry_valid_mask[window_index, pred_commit_step:, slot] & np.isfinite(values)
        return bool(valid.any() and np.any(values[valid] >= escape_threshold))

    for step in range(pred_commit_step, prediction.pred_position.shape[1]):
        if not bool(context.geometry_valid_mask[window_index, step, slot]):
            continue
        x, y = prediction.pred_position[window_index, step, slot]
        bbox_w = float(context.bbox_w[window_index, step, slot])
        bbox_h = float(context.bbox_h[window_index, step, slot])
        if not (
            np.isfinite(x)
            and np.isfinite(y)
            and np.isfinite(bbox_w)
            and np.isfinite(bbox_h)
            and bbox_w > 0
            and bbox_h > 0
        ):
            continue
        outside = comparison.compute_ellipse_outside_fraction(
            float(x),
            float(y),
            bbox_w,
            bbox_h,
            channel_mask,
        )
        if outside >= escape_threshold:
            return True
    return False


def build_junction_events(args: argparse.Namespace) -> comparison.RolloutModelComparator:
    comparator = comparison.RolloutModelComparator(
        adapters={},
        npz_path=comparison.DEFAULT_NPZ_PATH,
        output_dir=args.output_csv.parent,
        n_bootstrap=int(args.n_bootstrap),
    )
    print(f"Loading saved predictions: {args.predictions_path}", flush=True)
    comparator.load_predictions(args.predictions_path)
    comparator.bootstrap_metrics()
    comparator.compute_junction_decision_metrics(
        centerline_csv=args.centerline_csv,
        pca_half_window=8,
        min_quality=3.0,
        max_centerline_distance=30.0,
        min_orientation_speed=1.0e-3,
        min_branch_distance_margin=None,
        min_branch_relative_margin=None,
        outgoing_branches=("left", "right"),
        incoming_branches=("inlet", "outlet"),
        commitment_steps=3,
    )
    return comparator


def build_table(args: argparse.Namespace) -> list[dict[str, object]]:
    comparator = build_junction_events(args)
    reference = next(iter(comparator.predictions.values()))
    context = load_or_build_context(args, reference)
    channel_mask = comparison.load_channel_mask_bool(context.channel_mask_path)

    rows: list[dict[str, object]] = []
    for model_name in MODEL_ORDER:
        if model_name not in comparator.predictions:
            continue
        prediction = comparator.predictions[model_name]
        saved_outside = load_saved_outside_fraction(args.channel_data_path, model_name)
        source = args.channel_data_path if saved_outside is not None else "on-demand ellipse evaluation"
        print(f"Analyzing {model_name}; outside fractions source: {source}", flush=True)

        for true_branch, pred_branch, display_status, decision_status in TABLE_ROWS:
            matching_events = [
                event
                for event in comparator.junction_events
                if event.true_branch == true_branch
                and event.pred_branch_by_model[model_name] == pred_branch
                and event.decision_status_by_model[model_name] == decision_status
            ]
            escape_count = sum(
                event_has_catastrophic_escape(
                    event=event,
                    model_name=model_name,
                    prediction=prediction,
                    context=context,
                    channel_mask=channel_mask,
                    escape_threshold=float(args.escape_threshold),
                    saved_outside=saved_outside,
                )
                for event in matching_events
            )
            n_events = len(matching_events)
            rows.append(
                {
                    "model": model_name,
                    "true_branch": true_branch,
                    "pred_branch": pred_branch,
                    "status": display_status,
                    "N": n_events,
                    "complete_channel_escape_count": int(escape_count),
                    "complete_channel_escape_percent": 100.0 * escape_count / n_events if n_events else np.nan,
                }
            )
    return rows


def write_table(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "model",
        "true_branch",
        "pred_branch",
        "status",
        "N",
        "complete_channel_escape_count",
        "complete_channel_escape_percent",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows: list[dict[str, object]]) -> None:
    print("\nCatastrophic channel escape after predicted branch commitment")
    print("Definition: predicted ellipse outside-channel fraction >= 0.95")
    for model_name in MODEL_ORDER:
        model_rows = [row for row in rows if row["model"] == model_name]
        if not model_rows:
            continue
        print(f"\n{model_name}")
        print(f"{'True branch':<12} {'Pred branch':<12} {'Status':<8} {'N':>6} {'Complete channel escape %':>28}")
        print("-" * 74)
        for row in model_rows:
            percent = row["complete_channel_escape_percent"]
            percent_text = "nan" if not np.isfinite(percent) else f"{percent:.2f}"
            print(
                f"{row['true_branch']:<12} {row['pred_branch']:<12} {row['status']:<8} "
                f"{int(row['N']):>6} {percent_text:>28}"
            )


def main() -> None:
    args = parse_args()
    rows = build_table(args)
    write_table(args.output_csv, rows)
    print_table(rows)
    print(f"\nSaved: {args.output_csv}")


if __name__ == "__main__":
    main()
