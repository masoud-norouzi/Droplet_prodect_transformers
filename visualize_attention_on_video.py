from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import default_collate


REPO_ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = REPO_ROOT / "droplet-detection-tracking"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from droplet_detection_tracking.configs.settings import DEFAULT_CONFIG
from models.canonical_rollout_transformer import CanonicalRolloutTransformer
from train_canonical_rollout_transformer import (
    LOSS_ALPHA,
    MAX_DROPLETS,
    T_HISTORY,
    denormalize_features,
    denormalize_targets,
    move_batch_to_device,
    normalize_features,
    rollout_weights,
)
from utils.canonical_dataset.canonical_window_dataset import CanonicalWindowDataset


CSV_COLUMNS = [
    "window_index",
    "absolute_frame",
    "rollout_step",
    "target_slot",
    "target_track_id",
    "attended_slot",
    "attended_track_id",
    "aggregated_attention",
    "normalized_attention",
    "current_x",
    "current_y",
    "distance_to_target",
    "rank",
]


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    checkpoint = load_checkpoint(args.checkpoint, device)
    model = CanonicalRolloutTransformer(**checkpoint["model_config"]).to(device)
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as exc:
        raise SystemExit(
            "Checkpoint loading failed after enabling optional attention output. "
            f"No video was written. Original error:\n{exc}"
        ) from exc
    model.eval()

    horizon = args.length
    history_length = int(checkpoint["model_config"].get("T_history", T_HISTORY))
    normalization_stats = checkpoint["normalization_stats"]
    dataset = build_dataset(args.npz_path, horizon, args.stride, normalization_stats, history_length)
    if args.window_index < 0 or args.window_index >= len(dataset):
        raise IndexError(f"--window-index must be in 0..{len(dataset) - 1}; got {args.window_index}")
    if args.target_slot < 0 or args.target_slot >= MAX_DROPLETS:
        raise IndexError(f"--target-slot must be in 0..{MAX_DROPLETS - 1}; got {args.target_slot}")

    batch = default_collate([dataset[args.window_index]])
    batch = move_batch_to_device(batch, device)
    sample = dataset[args.window_index]
    frame_start_index = int(sample["frame_start"])
    droplet_ids = sample["droplet_ids"].numpy()
    target_track_id = int(droplet_ids[args.target_slot]) if droplet_ids.size else -1

    print_mapping(args, dataset, frame_start_index, target_track_id, device, history_length)

    weights = rollout_weights(horizon, float(checkpoint.get("loss_alpha", LOSS_ALPHA)), device)
    dry_run_steps = 1 if args.dry_run else horizon
    print("Attention links show top-k droplets after summing attention over each droplet's full history buffer.")
    print("If --exclude-self is used, same-slot/same-track history tokens are removed to emphasize neighbor interactions.")

    with torch.inference_mode():
        rows, render_payload = collect_attention_rollout(
            args=args,
            model=model,
            batch=batch,
            dataset=dataset,
            normalization_stats=normalization_stats,
            weights=weights,
            steps=dry_run_steps,
            device=device,
            history_length=history_length,
        )

    if args.dry_run:
        with torch.inference_mode():
            old_prediction = model(batch["history_x"], batch["history_mask"])
            attention_probe = model(batch["history_x"], batch["history_mask"], return_attention=True, attention_layer=args.layer)
        print(f"normal model(...) prediction shape: {tuple(old_prediction.shape)}")
        print(f"model(..., return_attention=True) prediction shape: {tuple(attention_probe['prediction'].shape)}")
        print(f"selected attention shape: {tuple(attention_probe['attention'].shape)}")
        print(f"attention layers returned: {len(attention_probe['attention_layers'])}")
        print("attention format: (batch, heads, query_tokens, key_tokens)")
        print("token metadata examples:")
        for token in render_payload["metadata_examples"]:
            print(f"  {token}")
        print("top attended droplet rows from first rollout step:")
        for row in rows[: min(args.top_k, len(rows))]:
            print(f"  {row}")
        print("dry run complete; no video or CSV was written.")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.csv_output, rows)
    write_attention_video(args, render_payload)
    print(f"Saved video: {args.output}")
    print(f"Saved CSV:   {args.csv_output}")


def parse_args() -> argparse.Namespace:
    default_video = DEFAULT_CONFIG.input.raw_video_dir / DEFAULT_CONFIG.input.video_file_name
    parser = argparse.ArgumentParser(description="Overlay rollout Transformer attention on real video frames.")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/models/canonical_rollout_transformer_best.pt"))
    parser.add_argument("--video-path", type=Path, default=default_video)
    parser.add_argument("--npz-path", type=Path, default=Path("outputs/processed/2/canonical_dataset.npz"))
    parser.add_argument("--window-index", type=int, required=True)
    parser.add_argument("--target-slot", type=int, required=True)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--layer", default="final")
    parser.add_argument("--head", default="mean", help="'mean' or a zero-based attention head index.")
    parser.add_argument("--line-start", choices=("pred", "true"), default="pred")
    parser.add_argument("--length", type=int, default=100)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--min-line-thickness", type=int, default=1)
    parser.add_argument("--max-line-thickness", type=int, default=8)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--csv-output", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-labels", action="store_true")
    parser.add_argument("--terminate-on-target-exit", action="store_true")
    parser.add_argument(
        "--exclude-self",
        action="store_true",
        help="Exclude attention tokens from the same droplet slot/track as the target.",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = Path(f"outputs/attention_video_window_{args.window_index}_slot_{args.target_slot}.mp4")
    if args.csv_output is None:
        args.csv_output = Path(f"outputs/attention_video_window_{args.window_index}_slot_{args.target_slot}.csv")
    return args


def load_checkpoint(path: Path, device: torch.device) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=device, weights_only=False)


def build_dataset(
    npz_path: Path,
    horizon: int,
    stride: int,
    normalization_stats,
    history_length: int,
) -> CanonicalWindowDataset:
    data = np.load(npz_path, allow_pickle=False)
    total_frames = len(data["frames"])
    total_window = history_length + horizon
    start_frames = np.arange(0, total_frames - total_window + 1, stride, dtype=np.int64)
    return CanonicalWindowDataset(
        npz_path=npz_path,
        start_frames=start_frames,
        T_history=history_length,
        T_future=horizon,
        max_droplets=MAX_DROPLETS,
        normalization_stats=normalization_stats,
    )


def print_mapping(
    args,
    dataset,
    frame_start_index: int,
    target_track_id: int,
    device: torch.device,
    history_length: int,
) -> None:
    first_history_frame = int(dataset.frames[frame_start_index])
    last_history_frame = int(dataset.frames[frame_start_index + history_length - 1])
    first_rollout_frame = int(dataset.frames[frame_start_index + history_length])
    last_rollout_frame = int(dataset.frames[frame_start_index + history_length + args.length - 1])
    print(f"checkpoint: {args.checkpoint}")
    print(f"video:      {args.video_path}")
    print(f"device:     {device}")
    print(f"window_index {args.window_index} maps to dataset start index {frame_start_index}")
    print(f"history length: {history_length}")
    print(f"history frames: {first_history_frame}-{last_history_frame}")
    print(f"rollout step 1 maps to absolute frame {first_rollout_frame}")
    print(f"rollout step {args.length} maps to absolute frame {last_rollout_frame}")
    print(f"target slot {args.target_slot}, target track_id {target_track_id}")


def collect_attention_rollout(
    args,
    model,
    batch,
    dataset,
    normalization_stats,
    weights,
    steps: int,
    device,
    history_length: int,
):
    history = batch["history_x"].clone()
    history_mask = batch["history_mask"].clone()
    frame_start = int(batch["frame_start"][0].detach().cpu())
    droplet_ids = batch["droplet_ids"][0].detach().cpu().numpy()
    feature_index = dataset.feature_indices
    true_future_features = get_true_future_features(batch, dataset, device, args.length, history_length)
    target_query_token = (history_length - 1) * MAX_DROPLETS + args.target_slot

    rows = []
    frame_payloads = []
    target_pred_trail = []
    target_true_trail = []
    metadata_examples = []

    for step_index in range(steps):
        history_phys = denormalize_features(history, normalization_stats, device)
        new_mask = batch["future_mask"][:, step_index, :]
        target_valid = bool(new_mask[0, args.target_slot].detach().cpu())

        if target_valid:
            attention_output = model(
                history,
                history_mask,
                return_attention=True,
                attention_layer=args.layer,
            )
            pred_step_norm_raw = attention_output["prediction"]
            attention = attention_output["attention"][0].detach().cpu().numpy()
        else:
            pred_step_norm_raw = model(history, history_mask)
            attention = None

        pred_step_phys_raw = denormalize_targets(
            pred_step_norm_raw[:, None, :, :],
            normalization_stats,
            device,
        )[:, 0, :, :]

        last_frame = history_phys[:, -1, :, :]
        x_next = last_frame[:, :, feature_index["x"]] + pred_step_phys_raw[:, :, 0]
        y_next = last_frame[:, :, feature_index["y"]] + pred_step_phys_raw[:, :, 1]

        new_frame_phys = last_frame.clone()
        new_frame_phys[:, :, feature_index["x"]] = x_next
        new_frame_phys[:, :, feature_index["y"]] = y_next
        new_frame_phys[:, :, feature_index["vx"]] = pred_step_phys_raw[:, :, 0]
        new_frame_phys[:, :, feature_index["vy"]] = pred_step_phys_raw[:, :, 1]

        previous_last_mask = history_mask[:, -1, :]
        true_step_features = true_future_features[:, step_index, :, :]
        true_step_features_finite = torch.isfinite(true_step_features).all(dim=-1)
        boundary_mask = new_mask & ~previous_last_mask & true_step_features_finite
        new_frame_phys[boundary_mask] = true_step_features[boundary_mask]

        circularity_index = feature_index.get("circularity")
        if circularity_index is not None:
            true_circularity = true_step_features[:, :, circularity_index]
            circularity_mask = new_mask & torch.isfinite(true_circularity)
            new_frame_phys[:, :, circularity_index] = torch.where(
                circularity_mask,
                true_circularity,
                new_frame_phys[:, :, circularity_index],
            )

        pred_position = new_frame_phys[0, :, [feature_index["x"], feature_index["y"]]].detach().cpu().numpy()
        true_position = true_step_features[0, :, [feature_index["x"], feature_index["y"]]].detach().cpu().numpy()
        future_mask = new_mask[0].detach().cpu().numpy().astype(bool)
        boundary_mask_np = boundary_mask[0].detach().cpu().numpy().astype(bool)
        absolute_frame = int(dataset.frames[frame_start + history_length + step_index])

        target_pred = pred_position[args.target_slot]
        target_true = true_position[args.target_slot]
        if target_valid and np.isfinite(target_pred).all():
            target_pred_trail.append(tuple_float(target_pred))
        if target_valid and np.isfinite(target_true).all():
            target_true_trail.append(tuple_float(target_true))

        key_metadata = build_token_metadata(
            dataset=dataset,
            history_phys=history_phys[0].detach().cpu().numpy(),
            history_mask=history_mask[0].detach().cpu().numpy().astype(bool),
            droplet_ids=droplet_ids,
            frame_start=frame_start,
            step_index=step_index,
            history_length=history_length,
        )
        if step_index == 0:
            metadata_examples = key_metadata[:5]

        top_droplets = []
        if target_valid:
            top_droplets = aggregate_top_attention_droplets(
                attention=attention,
                query_token=target_query_token,
                key_metadata=key_metadata,
                current_positions=true_position,
                current_mask=future_mask,
                top_k=args.top_k,
                head=args.head,
                target_slot=args.target_slot,
                target_track_id=int(droplet_ids[args.target_slot]),
                exclude_self=args.exclude_self,
            )

        step_rows = []
        target_track_id = int(droplet_ids[args.target_slot])
        target_line_start = target_pred if args.line_start == "pred" else target_true
        for rank, droplet in enumerate(top_droplets, start=1):
            attended_point = np.asarray([droplet["current_x"], droplet["current_y"]], dtype=float)
            distance_to_target = float("nan")
            if np.isfinite(target_line_start).all() and np.isfinite(attended_point).all():
                distance_to_target = float(np.linalg.norm(target_line_start - attended_point))
            row = {
                "window_index": args.window_index,
                "absolute_frame": absolute_frame,
                "rollout_step": step_index + 1,
                "target_slot": args.target_slot,
                "target_track_id": target_track_id,
                "attended_slot": droplet["slot"],
                "attended_track_id": droplet["track_id"],
                "aggregated_attention": droplet["aggregated_attention"],
                "normalized_attention": droplet["normalized_attention"],
                "current_x": droplet["current_x"],
                "current_y": droplet["current_y"],
                "distance_to_target": distance_to_target,
                "rank": rank,
            }
            rows.append(row)
            step_rows.append(row)

        frame_payloads.append(
            {
                "absolute_frame": absolute_frame,
                "rollout_step": step_index + 1,
                "pred_position": pred_position,
                "true_position": true_position,
                "future_mask": future_mask,
                "boundary_mask": boundary_mask_np,
                "target_valid": target_valid,
                "target_pred_trail": list(target_pred_trail),
                "target_true_trail": list(target_true_trail),
                "top_rows": step_rows,
                "target_track_id": int(droplet_ids[args.target_slot]),
            }
        )

        new_frame_norm = normalize_features(new_frame_phys, normalization_stats, device)
        new_frame_norm = torch.where(new_mask[:, :, None], new_frame_norm, torch.zeros_like(new_frame_norm))
        history = torch.cat([history[:, 1:, :, :], new_frame_norm[:, None, :, :]], dim=1)
        history_mask = torch.cat([history_mask[:, 1:, :], new_mask[:, None, :]], dim=1)

        if args.terminate_on_target_exit and not target_valid:
            break

    return rows, {
        "frame_payloads": frame_payloads,
        "metadata_examples": metadata_examples,
        "droplet_ids": droplet_ids,
    }


def get_true_future_features(batch, dataset, device, horizon: int, history_length: int) -> torch.Tensor:
    droplet_ids = batch["droplet_ids"].detach().cpu().numpy()
    frame_starts = batch["frame_start"].detach().cpu().numpy()
    track_id_to_index = {int(track_id): index for index, track_id in enumerate(dataset.track_ids)}
    B, M = droplet_ids.shape
    true_features = np.full((B, horizon, M, len(dataset.feature_names)), np.nan, dtype=np.float32)

    for batch_index in range(B):
        start = int(frame_starts[batch_index]) + history_length
        end = start + horizon
        for slot_index in range(M):
            track_id = int(droplet_ids[batch_index, slot_index])
            if track_id < 0:
                continue
            droplet_index = track_id_to_index.get(track_id)
            if droplet_index is None:
                continue
            true_features[batch_index, :, slot_index, :] = dataset.Z[droplet_index, start:end, :]

    return torch.as_tensor(true_features, dtype=torch.float32, device=device)


def build_token_metadata(
    dataset,
    history_phys,
    history_mask,
    droplet_ids,
    frame_start: int,
    step_index: int,
    history_length: int,
):
    x_index = dataset.feature_indices["x"]
    y_index = dataset.feature_indices["y"]
    metadata = []
    for time_index in range(history_length):
        absolute_frame = int(dataset.frames[frame_start + step_index + time_index])
        for slot in range(MAX_DROPLETS):
            x = float_or_nan(history_phys[time_index, slot, x_index])
            y = float_or_nan(history_phys[time_index, slot, y_index])
            valid = bool(history_mask[time_index, slot] and np.isfinite(x) and np.isfinite(y))
            metadata.append(
                {
                    "token": time_index * MAX_DROPLETS + slot,
                    "time_index": time_index,
                    "slot": slot,
                    "track_id": int(droplet_ids[slot]) if slot < len(droplet_ids) else -1,
                    "absolute_frame": absolute_frame,
                    "x": x,
                    "y": y,
                    "valid": valid,
                }
            )
    return metadata


def aggregate_top_attention_droplets(
    attention,
    query_token: int,
    key_metadata,
    current_positions: np.ndarray,
    current_mask: np.ndarray,
    top_k: int,
    head: str,
    target_slot: int,
    target_track_id: int,
    exclude_self: bool,
):
    if head == "mean":
        scores = attention[:, query_token, :].mean(axis=0)
    else:
        head_index = int(head)
        if head_index < 0 or head_index >= attention.shape[0]:
            raise IndexError(f"--head must be 'mean' or 0..{attention.shape[0] - 1}; got {head}")
        scores = attention[head_index, query_token, :]

    grouped = {}
    for token in key_metadata:
        if not token["valid"]:
            continue
        if not np.isfinite([token["x"], token["y"]]).all():
            continue
        if exclude_self:
            same_slot = token["slot"] == target_slot
            same_track = target_track_id >= 0 and token["track_id"] == target_track_id
            if same_slot or same_track:
                continue

        key = ("track", token["track_id"]) if token["track_id"] >= 0 else ("slot", token["slot"])
        if key not in grouped:
            grouped[key] = {
                "slot": token["slot"],
                "track_id": token["track_id"],
                "aggregated_attention": 0.0,
            }
        grouped[key]["aggregated_attention"] += float(scores[token["token"]])

    visible = []
    for item in grouped.values():
        slot = int(item["slot"])
        if slot < 0 or slot >= len(current_mask):
            continue
        if not bool(current_mask[slot]):
            continue
        current_xy = current_positions[slot]
        if not np.isfinite(current_xy).all():
            continue
        item = dict(item)
        item["current_x"] = float(current_xy[0])
        item["current_y"] = float(current_xy[1])
        visible.append(item)

    total_attention = sum(item["aggregated_attention"] for item in visible)
    if total_attention > 0:
        for item in visible:
            item["normalized_attention"] = item["aggregated_attention"] / total_attention
    else:
        for item in visible:
            item["normalized_attention"] = 0.0

    visible.sort(key=lambda item: item["normalized_attention"], reverse=True)
    return visible[:top_k]


def write_attention_video(args, render_payload) -> None:
    if not args.video_path.exists():
        raise FileNotFoundError(f"Video path does not exist: {args.video_path}")
    capture = cv2.VideoCapture(str(args.video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {args.video_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.output), fourcc, args.fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Unable to create output video: {args.output}")

    try:
        for payload in render_payload["frame_payloads"]:
            frame = read_video_frame(capture, payload["absolute_frame"])
            annotate_frame(frame, args, payload, render_payload["droplet_ids"])
            writer.write(frame)
    finally:
        writer.release()
        capture.release()


def read_video_frame(capture, absolute_frame: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, absolute_frame)
    success, frame = capture.read()
    if not success:
        raise RuntimeError(f"Unable to read video frame {absolute_frame}")
    return frame


def annotate_frame(frame, args, payload, droplet_ids) -> None:
    pred = payload["pred_position"]
    true = payload["true_position"]
    mask = payload["future_mask"]
    target_slot = args.target_slot
    target_pred = pred[target_slot]
    target_true = true[target_slot]
    line_start = target_pred if args.line_start == "pred" else target_true
    target_valid = payload["target_valid"]

    for slot in np.flatnonzero(mask):
        if np.isfinite(true[slot]).all():
            cv2.circle(frame, point(true[slot]), 4, (30, 30, 30), -1)
        if np.isfinite(pred[slot]).all():
            cv2.drawMarker(frame, point(pred[slot]), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 9, 1)

    if target_valid:
        draw_polyline(frame, payload["target_true_trail"], (40, 40, 40), 1)
        draw_polyline(frame, payload["target_pred_trail"], (0, 0, 255), 2)

    if target_valid and np.isfinite(target_true).all():
        cv2.circle(frame, point(target_true), 8, (0, 255, 255), 2)
    if target_valid and np.isfinite(target_pred).all():
        cv2.circle(frame, point(target_pred), 7, (0, 0, 255), 2)

    max_weight = max([row["normalized_attention"] for row in payload["top_rows"]] or [1.0])
    if target_valid:
        for row in reversed(payload["top_rows"]):
            if not np.isfinite([row["current_x"], row["current_y"]]).all() or not np.isfinite(line_start).all():
                continue
            relative_weight = row["normalized_attention"] / max(max_weight, 1e-12)
            thickness = args.min_line_thickness
            brightness = int(round(80 + 175 * relative_weight))
            radius = 4 + int(round(10 * relative_weight))
            line_color = (0, brightness, brightness)
            marker_color = (brightness, brightness, 0)
            end = (int(round(row["current_x"])), int(round(row["current_y"])))
            cv2.line(frame, point(line_start), end, line_color, thickness, cv2.LINE_AA)
            cv2.circle(frame, end, radius, marker_color, -1, cv2.LINE_AA)
            cv2.circle(frame, end, radius + 2, (255, 255, 255), 1, cv2.LINE_AA)
            if not args.no_labels:
                label = f"t{row['attended_track_id']} a={row['normalized_attention']:.2f}"
                cv2.putText(frame, label, (end[0] + radius + 4, end[1] - radius - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    top_text = ", ".join(
        f"{row['attended_track_id']}:{row['normalized_attention']:.2f}"
        for row in payload["top_rows"][: min(5, len(payload["top_rows"]))]
    )
    lines = [
        f"window {args.window_index} step {payload['rollout_step']} frame {payload['absolute_frame']}",
        f"target slot {target_slot} track {payload['target_track_id']}",
        f"attention layer {args.layer} head {args.head} line_start {args.line_start}",
        f"top droplets: {top_text}",
    ]
    if not target_valid:
        lines.append("target exited FOV")
    draw_text_box(frame, lines, (10, 22))


def draw_text_box(frame, lines, origin) -> None:
    x, y = origin
    line_height = 18
    width = max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0][0] for line in lines) + 12
    height = line_height * len(lines) + 8
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 6, y - 16), (x - 6 + width, y - 16 + height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for index, line in enumerate(lines):
        cv2.putText(frame, line, (x, y + index * line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)


def draw_polyline(frame, points, color, thickness) -> None:
    clean_points = [point(np.asarray(item, dtype=float)) for item in points if np.isfinite(item).all()]
    if len(clean_points) >= 2:
        cv2.polylines(frame, [np.asarray(clean_points, dtype=np.int32)], False, color, thickness, cv2.LINE_AA)


def write_csv(path: Path, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def point(values) -> tuple[int, int]:
    return int(round(float(values[0]))), int(round(float(values[1])))


def tuple_float(values) -> tuple[float, float]:
    return float(values[0]), float(values[1])


def float_or_nan(value) -> float:
    value = float(value)
    return value if np.isfinite(value) else float("nan")


if __name__ == "__main__":
    main()
