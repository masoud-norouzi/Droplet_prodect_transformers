from __future__ import annotations

import argparse
import csv
from pathlib import Path
import random

import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, default_collate

from models.canonical_rollout_transformer import CanonicalRolloutTransformer
from utils.geometry_loss import compute_ellipse_outside_fraction, compute_ellipse_outside_fraction_torch
from utils.canonical_dataset.canonical_window_dataset import create_train_val_test_datasets


NPZ_PATH = "outputs/processed/2/canonical_dataset.npz"
DETECTIONS_CSV_PATH = "outputs/processed/2/tracked_features.csv"
CHANNEL_MASK_PATH = "outputs/processed/2/channel_mask.npy"
VIDEO_PATH = r"D:\Microfluidic loop projct\new loop experiments\confined droplets 2\2.avi"
OUTPUT_DIR = "outputs/models/train_geometry_aware_markovian_rollout"
CHECKPOINT_STEM = "geometry_aware_markovian_rollout"

T_HISTORY = 1
ROLLOUT_HORIZON = 50
MAX_DROPLETS = 64
INPUT_DIM = 5
TARGET_DIM = 2
STRIDE = 5
LOSS_ALPHA = 2.0
GEOMETRY_TOLERANCE = 0.02
GEOMETRY_LOSS_WEIGHT = 1.0
GEOMETRY_NUM_SAMPLES_X = 64
GEOMETRY_NUM_SAMPLES_Y = 64

BATCH_SIZE = 4
EPOCHS = 50
LR = 1e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
NUM_WORKERS = 0
LOG_EVERY_N_BATCHES = 25
ANIMATE_EVERY_N_EPOCHS = 10
DIAGNOSTIC_STEPS = (1, 5, 10, 20, 30, 40, 50)

MODEL_CONFIG = {
    "input_dim": INPUT_DIM,
    "target_dim": TARGET_DIM,
    "T_history": T_HISTORY,
    "max_droplets": MAX_DROPLETS,
    "d_model": 128,
    "n_heads": 4,
    "num_layers": 4,
    "dim_feedforward": 512,
    "dropout": 0.1,
}


class FutureBBoxLookup:
    def __init__(self, detections_csv: str | Path):
        self.source_path = Path(detections_csv)
        self.resolved_source_path = self.source_path.resolve()
        self.source_size_bytes = self.source_path.stat().st_size
        self.source_mtime = self.source_path.stat().st_mtime
        detections = pd.read_csv(self.source_path, usecols=["frame", "track_id", "bbox_w", "bbox_h"])
        self.raw_rows = int(len(detections))
        self.raw_frame_min = int(detections["frame"].min()) if len(detections) else None
        self.raw_frame_max = int(detections["frame"].max()) if len(detections) else None
        detections = detections.dropna(subset=["frame", "track_id", "bbox_w", "bbox_h"])
        self.valid_rows = int(len(detections))
        self.values: dict[tuple[int, int], tuple[float, float]] = {}
        self.record_counts: dict[tuple[int, int], int] = {}
        self.frame_min = int(detections["frame"].min()) if len(detections) else None
        self.frame_max = int(detections["frame"].max()) if len(detections) else None
        for row in detections.itertuples(index=False):
            key = (int(row.frame), int(row.track_id))
            self.record_counts[key] = self.record_counts.get(key, 0) + 1
            self.values[key] = (float(row.bbox_w), float(row.bbox_h))
        self.duplicate_keys = [key for key, count in self.record_counts.items() if count > 1]

    def lookup(self, frame: int, track_id: int) -> tuple[float, float] | None:
        return self.values.get((int(frame), int(track_id)))


class GeometryAwareDataset(Dataset):
    def __init__(self, base_dataset, bbox_lookup: FutureBBoxLookup):
        self.base_dataset = base_dataset
        self.bbox_lookup = bbox_lookup

    def __len__(self):
        return len(self.base_dataset)

    def __getattr__(self, name):
        return getattr(self.base_dataset, name)

    def __getitem__(self, index):
        item = self.base_dataset[index]
        frame_start = int(item["frame_start"])
        future_bbox = np.zeros((self.base_dataset.T_future, self.base_dataset.max_droplets, 2), dtype=np.float32)
        future_bbox_mask = np.zeros((self.base_dataset.T_future, self.base_dataset.max_droplets), dtype=bool)
        droplet_ids = item["droplet_ids"].numpy()
        future_mask = item["future_mask"].numpy()

        first_future_frame = frame_start + self.base_dataset.T_history
        for step_index in range(self.base_dataset.T_future):
            frame = first_future_frame + step_index
            for slot_index, track_id in enumerate(droplet_ids):
                if track_id < 0 or not future_mask[step_index, slot_index]:
                    continue
                bbox = self.bbox_lookup.lookup(frame, int(track_id))
                if bbox is None:
                    continue
                bbox_w, bbox_h = bbox
                if np.isfinite(bbox_w) and np.isfinite(bbox_h) and bbox_w > 0 and bbox_h > 0:
                    future_bbox[step_index, slot_index, 0] = bbox_w
                    future_bbox[step_index, slot_index, 1] = bbox_h
                    future_bbox_mask[step_index, slot_index] = True

        item["future_bbox"] = torch.as_tensor(future_bbox, dtype=torch.float32)
        item["future_bbox_mask"] = torch.as_tensor(future_bbox_mask, dtype=torch.bool)
        return item


def main() -> None:
    args = parse_args()
    configure_output_paths(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_startup_banner()
    print(f"Device: {device}")
    print(f"Geometry loss weight: {args.geometry_loss_weight:.6g}")
    print(f"Geometry loss weight tag: {args.geometry_loss_weight_tag}")
    print(f"Output directory: {args.output_dir}")

    train_ds, val_ds, test_ds, normalization_stats = create_train_val_test_datasets(
        npz_path=args.npz_path,
        stride=args.stride,
        T_history=T_HISTORY,
        T_future=ROLLOUT_HORIZON,
        max_droplets=MAX_DROPLETS,
    )
    bbox_lookup = FutureBBoxLookup(args.detections_csv)
    train_ds = GeometryAwareDataset(train_ds, bbox_lookup)
    val_ds = GeometryAwareDataset(val_ds, bbox_lookup)
    test_ds = GeometryAwareDataset(test_ds, bbox_lookup)
    print(f"Train windows: {len(train_ds)}")
    print(f"Val windows: {len(val_ds)}")
    print(f"Test windows: {len(test_ds)}")
    print(f"BBox CSV path: {bbox_lookup.resolved_source_path}")
    print(f"BBox CSV size bytes: {bbox_lookup.source_size_bytes}")
    print(f"BBox CSV modified unix time: {bbox_lookup.source_mtime:.0f}")
    print(f"BBox CSV raw rows: {bbox_lookup.raw_rows}")
    print(f"BBox CSV raw frame range: {bbox_lookup.raw_frame_min}..{bbox_lookup.raw_frame_max}")
    print(f"BBox CSV rows with non-null frame/track_id/bbox_w/bbox_h: {bbox_lookup.valid_rows}")
    print(f"BBox lookup entries: {len(bbox_lookup.values)}")
    print(f"BBox frame range: {bbox_lookup.frame_min}..{bbox_lookup.frame_max}")
    print(f"Duplicate (frame, track_id) bbox keys: {len(bbox_lookup.duplicate_keys)}")
    report_bbox_coverage_by_split(
        {"train": train_ds, "val": val_ds, "test": test_ds},
        bbox_lookup=bbox_lookup,
        max_windows=args.bbox_coverage_check_windows,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    model = CanonicalRolloutTransformer(**MODEL_CONFIG).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    weights = rollout_weights(ROLLOUT_HORIZON, LOSS_ALPHA, device)
    channel_mask = load_channel_mask(args.channel_mask, device)

    if args.validate_geometry_integration:
        validate_geometry_integration(
            model=model,
            loader=val_loader,
            dataset=val_ds,
            bbox_lookup=bbox_lookup,
            normalization_stats=normalization_stats,
            weights=weights,
            channel_mask=channel_mask,
            optimizer=optimizer,
            device=device,
            args=args,
        )
        return

    run_shape_test(model, train_loader, train_ds, normalization_stats, weights, channel_mask, optimizer, args, device)
    if args.validation_smoke_test:
        run_validation_smoke_test(
            model=model,
            optimizer=optimizer,
            loader=train_loader,
            dataset=train_ds,
            normalization_stats=normalization_stats,
            weights=weights,
            channel_mask=channel_mask,
            device=device,
            output_dir=Path(args.validation_dir),
            args=args,
        )
        return
    if args.shape_test_only:
        return
    if args.sanity_train_batches > 0:
        run_short_sanity_training(
            model=model,
            loader=train_loader,
            dataset=train_ds,
            optimizer=optimizer,
            normalization_stats=normalization_stats,
            weights=weights,
            channel_mask=channel_mask,
            device=device,
            args=args,
        )
        return

    best_val_loss = float("inf")
    best_val_geometry_loss = float("inf")
    best_checkpoint_path = Path(args.best_checkpoint)
    latest_checkpoint_path = Path(args.latest_checkpoint)
    curves_csv_path = Path(args.curves_csv)
    best_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    curves_csv_path.parent.mkdir(parents=True, exist_ok=True)
    initialize_curves_csv(curves_csv_path)

    for epoch in range(1, args.epochs + 1):
        train_summary = train_one_epoch(
            model=model,
            loader=train_loader,
            dataset=train_ds,
            optimizer=optimizer,
            normalization_stats=normalization_stats,
            weights=weights,
            channel_mask=channel_mask,
            device=device,
            args=args,
            log_every=args.log_every,
        )
        val_summary = evaluate(
            model=model,
            loader=val_loader,
            dataset=val_ds,
            normalization_stats=normalization_stats,
            weights=weights,
            channel_mask=channel_mask,
            device=device,
            args=args,
            log_every=args.log_every,
        )

        print_epoch_summary(epoch, train_summary, val_summary)
        append_curves_csv(curves_csv_path, epoch, train_summary, val_summary)

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": val_summary["weighted_loss_internal_only"],
            "val_geometry_loss": val_summary["geometry_loss"],
            "val_total_loss": val_summary["total_loss"],
            "normalization_stats": normalization_stats,
            "model_config": MODEL_CONFIG,
            "rollout_horizon": ROLLOUT_HORIZON,
            "loss_alpha": LOSS_ALPHA,
            "stride": args.stride,
            "geometry_tolerance": args.geometry_tolerance,
            "geometry_loss_weight": args.geometry_loss_weight,
            "geometry_loss_weight_tag": args.geometry_loss_weight_tag,
            "output_dir": args.output_dir,
            "channel_mask_path": args.channel_mask,
            "detections_csv": args.detections_csv,
            "geometry_num_samples_x": args.geometry_num_samples_x,
            "geometry_num_samples_y": args.geometry_num_samples_y,
            "best_checkpoint_selection": "validation weighted rollout loss",
        }
        torch.save(checkpoint, latest_checkpoint_path)

        if val_summary["weighted_loss_internal_only"] < best_val_loss:
            best_val_loss = val_summary["weighted_loss_internal_only"]
            torch.save(checkpoint, best_checkpoint_path)
            print(f"Saved best checkpoint: {best_checkpoint_path}")
        if val_summary["geometry_loss"] < best_val_geometry_loss:
            best_val_geometry_loss = val_summary["geometry_loss"]
            geometry_checkpoint_path = Path(args.best_geometry_checkpoint)
            geometry_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    **checkpoint,
                    "best_checkpoint_selection": "validation geometry loss",
                },
                geometry_checkpoint_path,
            )
            print(f"Saved best geometry checkpoint: {geometry_checkpoint_path}")

        if args.animate_every > 0 and epoch % args.animate_every == 0:
            make_random_validation_animation(
                model=model,
                dataset=val_ds,
                normalization_stats=normalization_stats,
                channel_mask=channel_mask,
                device=device,
                args=args,
                output_path=Path(args.animation_dir) / f"rollout_epoch_{epoch:03d}.mp4",
            )

    make_random_validation_animation(
        model=model,
        dataset=val_ds,
        normalization_stats=normalization_stats,
        channel_mask=channel_mask,
        device=device,
        args=args,
        output_path=Path(args.animation_dir) / "rollout_after_training.mp4",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the geometry-aware Markovian rollout Transformer control.")
    parser.add_argument("--npz-path", default=NPZ_PATH)
    parser.add_argument("--detections-csv", default=DETECTIONS_CSV_PATH)
    parser.add_argument("--channel-mask", default=CHANNEL_MASK_PATH)
    parser.add_argument("--video-path", default=VIDEO_PATH)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--stride", type=int, default=STRIDE)
    parser.add_argument("--log-every", type=int, default=LOG_EVERY_N_BATCHES)
    parser.add_argument("--animate-every", type=int, default=ANIMATE_EVERY_N_EPOCHS)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--best-checkpoint", default=None)
    parser.add_argument("--best-geometry-checkpoint", default=None)
    parser.add_argument("--latest-checkpoint", default=None)
    parser.add_argument("--curves-csv", default=None)
    parser.add_argument("--animation-dir", default=None)
    parser.add_argument("--validation-dir", default=None)
    parser.add_argument("--geometry-tolerance", type=float, default=GEOMETRY_TOLERANCE)
    parser.add_argument("--geometry-loss-weight", type=float, default=GEOMETRY_LOSS_WEIGHT)
    parser.add_argument("--geometry-num-samples-x", type=int, default=GEOMETRY_NUM_SAMPLES_X)
    parser.add_argument("--geometry-num-samples-y", type=int, default=GEOMETRY_NUM_SAMPLES_Y)
    parser.add_argument("--sanity-train-batches", type=int, default=0)
    parser.add_argument("--validate-geometry-integration", action="store_true")
    parser.add_argument("--geometry-validation-batches", type=int, default=3)
    parser.add_argument("--geometry-validation-examples", type=int, default=20)
    parser.add_argument(
        "--bbox-coverage-check-windows",
        type=int,
        default=128,
        help="Windows per split to check for future bbox coverage before training; 0 checks all, negative skips.",
    )
    parser.add_argument(
        "--geometry-validation-montage",
        default=None,
    )
    parser.add_argument("--shape-test-only", action="store_true")
    parser.add_argument("--validation-smoke-test", action="store_true")
    return parser.parse_args()


def geometry_loss_weight_tag(weight: float) -> str:
    raw = f"{float(weight):.12g}"
    safe = raw.replace("-", "neg").replace("+", "").replace(".", "p")
    return f"geom_weight_{safe}"


def configure_output_paths(args: argparse.Namespace) -> None:
    tag = geometry_loss_weight_tag(args.geometry_loss_weight)
    args.geometry_loss_weight_tag = tag
    if args.output_dir is None:
        args.output_dir = str(Path(f"{OUTPUT_DIR}_{tag}"))
    output_dir = Path(args.output_dir)
    artifact_stem = f"{CHECKPOINT_STEM}_{tag}"
    if args.best_checkpoint is None:
        args.best_checkpoint = str(output_dir / f"{artifact_stem}_best.pt")
    if args.best_geometry_checkpoint is None:
        args.best_geometry_checkpoint = str(output_dir / f"{artifact_stem}_best_geometry.pt")
    if args.latest_checkpoint is None:
        args.latest_checkpoint = str(output_dir / f"{artifact_stem}_latest.pt")
    if args.curves_csv is None:
        args.curves_csv = str(output_dir / f"{artifact_stem}_training_curves.csv")
    if args.animation_dir is None:
        args.animation_dir = str(output_dir / "rollout_training_animations")
    if args.validation_dir is None:
        args.validation_dir = str(output_dir / "validation")
    if args.geometry_validation_montage is None:
        args.geometry_validation_montage = str(output_dir / "validation" / f"{artifact_stem}_bbox_alignment_montage.png")


def print_startup_banner() -> None:
    print("======================================")
    print("GEOMETRY-AWARE MARKOVIAN ROLLOUT EXPERIMENT")
    print(f"History length : {T_HISTORY}")
    print(f"Future rollout : {ROLLOUT_HORIZON}")
    print("======================================")
    print("Scientific goal: test whether channel geometry regularizes Markovian rollout dynamics.")


def load_channel_mask(mask_path, device) -> torch.Tensor:
    mask = np.load(mask_path)
    if mask.ndim != 2:
        raise ValueError(f"Channel mask must be 2D, got {mask.shape}")
    if mask.dtype != bool:
        mask = mask.astype(bool)
    if not mask.any():
        raise ValueError("Channel mask has no True pixels.")
    print(f"Channel mask: {mask_path} shape={mask.shape} true_fraction={mask.mean():.6f}")
    return torch.as_tensor(mask.astype(np.float32), dtype=torch.float32, device=device)


def report_bbox_coverage_by_split(datasets, bbox_lookup: FutureBBoxLookup, max_windows: int) -> None:
    if max_windows < 0:
        return

    print("BBox coverage diagnostics:")
    print("  lookup key: (frame, track_id) -> (bbox_w, bbox_h)")
    for split_name, dataset in datasets.items():
        if len(dataset) == 0:
            print(f"  {split_name}: empty split")
            continue

        if max_windows == 0:
            indices = range(len(dataset))
            checked_text = "all"
        else:
            checked = min(int(max_windows), len(dataset))
            indices = range(checked)
            checked_text = str(checked)

        starts = np.asarray(dataset.start_frames, dtype=np.int64)
        future_start = int(starts.min()) + dataset.T_history
        future_end = int(starts.max()) + dataset.T_history + dataset.T_future - 1
        frame_labels = getattr(dataset, "frames", None)
        if frame_labels is not None and future_end < len(frame_labels):
            future_label_start = int(frame_labels[future_start])
            future_label_end = int(frame_labels[future_end])
            future_range_text = f"indices {future_start}..{future_end}, labels {future_label_start}..{future_label_end}"
        else:
            future_range_text = f"indices {future_start}..{future_end}"

        valid_count = 0
        bbox_count = 0
        for index in indices:
            item = dataset[index]
            valid = item["future_mask"] & (item["droplet_ids"][None, :] >= 0)
            valid_count += int(valid.sum().item())
            bbox_count += int((item["future_bbox_mask"] & valid).sum().item())

        missing_count = valid_count - bbox_count
        coverage = bbox_count / valid_count if valid_count else 0.0
        print(
            f"  {split_name}: checked_windows={checked_text} "
            f"future_frames=({future_range_text}) "
            f"valid_future={valid_count} bbox_matches={bbox_count} "
            f"missing={missing_count} coverage={coverage:.6f}"
        )
        if valid_count > 0 and bbox_count == 0:
            print(
                f"  WARNING: {split_name} has valid future droplets but zero bbox matches. "
                "Geometry loss for this split will be exactly zero; check that the detections CSV covers these frames."
            )


def validate_geometry_integration(model, loader, dataset, bbox_lookup, normalization_stats, weights, channel_mask, optimizer, device, args) -> None:
    print("======================================")
    print("GEOMETRY INTEGRATION VALIDATION ONLY")
    print("======================================")
    print("bbox lookup key: (frame, track_id) -> (bbox_w, bbox_h)")
    print(f"duplicate bbox records: {len(bbox_lookup.duplicate_keys)}")
    if bbox_lookup.duplicate_keys[:5]:
        print(f"duplicate key examples: {bbox_lookup.duplicate_keys[:5]}")

    channel_mask_np = channel_mask.detach().cpu().numpy().astype(bool)
    model.eval()
    first_batch = None
    first_rollout = None
    loss_rows = []

    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            if batch_index > args.geometry_validation_batches:
                break
            batch = move_batch_to_device(batch, device)
            rollout = boundary_conditioned_rollout(
                model=model,
                batch=batch,
                dataset=dataset,
                normalization_stats=normalization_stats,
                weights=weights,
                channel_mask=channel_mask,
                geometry_tolerance=args.geometry_tolerance,
                geometry_loss_weight=args.geometry_loss_weight,
                geometry_num_samples_x=args.geometry_num_samples_x,
                geometry_num_samples_y=args.geometry_num_samples_y,
            )
            rollout_loss = float(rollout["weighted_loss_internal_only"].detach().cpu())
            geometry_loss = float(rollout["geometry_loss"].detach().cpu())
            weighted_geometry = float(args.geometry_loss_weight) * geometry_loss
            total_loss = float(rollout["total_loss"].detach().cpu())
            ratio = weighted_geometry / rollout_loss if rollout_loss > 0 else float("inf")
            loss_rows.append((batch_index, rollout_loss, geometry_loss, weighted_geometry, total_loss, ratio, int(rollout["geometry_count"])))
            if first_batch is None:
                first_batch = batch
                first_rollout = rollout

    assert first_batch is not None and first_rollout is not None
    examples, missing_count = collect_bbox_alignment_examples(
        batch=first_batch,
        rollout=first_rollout,
        dataset=dataset,
        bbox_lookup=bbox_lookup,
        channel_mask_np=channel_mask_np,
        max_examples=args.geometry_validation_examples,
    )
    assert examples, "No valid examples found for bbox alignment validation."
    assert int((first_rollout["geometry_mask"] & first_rollout["boundary_mask"]).sum().detach().cpu()) == 0

    montage_path = Path(args.geometry_validation_montage)
    montage_saved = False
    try:
        save_bbox_alignment_montage(
            examples[: min(len(examples), 20)],
            channel_mask_np,
            Path(args.video_path),
            montage_path,
            crop_bottom_px=35,
        )
        montage_saved = True
    except RuntimeError as exc:
        print(f"WARNING: skipped bbox alignment montage: {exc}")

    gradient_report = compute_gradient_diagnostics(
        model=model,
        batch=first_batch,
        dataset=dataset,
        normalization_stats=normalization_stats,
        weights=weights,
        channel_mask=channel_mask,
        args=args,
    )

    print("BBox alignment examples:")
    header = (
        "frame step slot track_id true_x true_y bbox_w bbox_h "
        "boundary_injected internal_loss_mask future_bbox_mask outside_fraction"
    )
    print(header)
    for row in examples:
        print(
            f"{row['frame']:5d} {row['rollout_step']:4d} {row['slot']:4d} {row['track_id']:8d} "
            f"{row['true_centroid_x']:7.2f} {row['true_centroid_y']:7.2f} "
            f"{row['bbox_w']:6.2f} {row['bbox_h']:6.2f} "
            f"{str(row['boundary_injected']):>17} {str(row['internal_loss_mask']):>18} "
            f"{str(row['future_bbox_mask']):>16} {row['outside_fraction']:.6f}"
        )

    print("Loss-scale diagnostics:")
    for batch_index, rollout_loss, geometry_loss, weighted_geometry, total_loss, ratio, geometry_count in loss_rows:
        print(
            f"batch {batch_index:03d}: rollout_loss={rollout_loss:.6f} "
            f"geometry_loss={geometry_loss:.6f} weight={args.geometry_loss_weight:.6f} "
            f"weighted_geometry={weighted_geometry:.6f} total_loss={total_loss:.6f} "
            f"weighted_geometry/rollout={ratio:.6f} geometry_count={geometry_count}"
        )

    print("Gradient diagnostics:")
    for key, value in gradient_report.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for sub_key, sub_value in value.items():
                print(f"    {sub_key}: {sub_value:.6f}")
        else:
            print(f"  {key}: {value:.6f}")

    aligned = missing_count == 0 and len(bbox_lookup.duplicate_keys) == 0
    print(f"all_checked_identities_aligned: {aligned}")
    print(f"missing bbox records among checked valid droplets: {missing_count}")
    print(f"montage_path: {montage_path if montage_saved else 'not saved'}")
    if len(bbox_lookup.duplicate_keys) > 0:
        print("BLOCKER: duplicate (frame, track_id) bbox records exist; inspect detections CSV before full training.")
    elif missing_count > 0:
        print("BLOCKER: missing bbox records found for valid future droplets.")
    else:
        print("No geometry-integration blocker found in checked batch.")


def collect_bbox_alignment_examples(batch, rollout, dataset, bbox_lookup, channel_mask_np, max_examples):
    droplet_ids = batch["droplet_ids"].detach().cpu().numpy()
    frame_starts = batch["frame_start"].detach().cpu().numpy()
    true_positions = rollout["true_position"].detach().cpu().numpy()
    future_bbox = batch["future_bbox"].detach().cpu().numpy()
    future_bbox_mask = batch["future_bbox_mask"].detach().cpu().numpy()
    future_mask = batch["future_mask"].detach().cpu().numpy()
    boundary_mask = rollout["boundary_mask"].detach().cpu().numpy()
    internal_loss_mask = rollout["internal_loss_mask"].detach().cpu().numpy()

    candidates = []
    missing_count = 0
    B, T, M = future_mask.shape
    for batch_index in range(B):
        first_future_frame = int(frame_starts[batch_index]) + dataset.T_history
        for step_index in range(T):
            frame = first_future_frame + step_index
            for slot_index in range(M):
                if not future_mask[batch_index, step_index, slot_index]:
                    continue
                track_id = int(droplet_ids[batch_index, slot_index])
                if track_id < 0:
                    continue
                lookup = bbox_lookup.lookup(frame, track_id)
                has_bbox = bool(future_bbox_mask[batch_index, step_index, slot_index])
                if lookup is None:
                    missing_count += 1
                    assert not has_bbox
                    continue
                bbox_w, bbox_h = lookup
                assert bbox_w > 0 and bbox_h > 0
                assert has_bbox
                np.testing.assert_allclose(
                    future_bbox[batch_index, step_index, slot_index],
                    np.asarray([bbox_w, bbox_h], dtype=np.float32),
                    rtol=0,
                    atol=1e-5,
                )
                true_x, true_y = true_positions[batch_index, step_index, slot_index]
                outside_fraction = compute_ellipse_outside_fraction(true_x, true_y, bbox_w, bbox_h, channel_mask_np)
                candidates.append(
                    {
                        "batch_index": batch_index,
                        "frame": int(frame),
                        "rollout_step": int(step_index + 1),
                        "slot": int(slot_index),
                        "track_id": int(track_id),
                        "true_centroid_x": float(true_x),
                        "true_centroid_y": float(true_y),
                        "bbox_w": float(bbox_w),
                        "bbox_h": float(bbox_h),
                        "boundary_injected": bool(boundary_mask[batch_index, step_index, slot_index]),
                        "internal_loss_mask": bool(internal_loss_mask[batch_index, step_index, slot_index]),
                        "future_bbox_mask": has_bbox,
                        "outside_fraction": float(outside_fraction),
                    }
                )

    if len(candidates) <= max_examples:
        return candidates, missing_count

    step_buckets = {}
    for row in candidates:
        bucket = min(4, (row["rollout_step"] - 1) // 10)
        step_buckets.setdefault(bucket, []).append(row)
    selected = []
    for bucket in sorted(step_buckets):
        selected.extend(step_buckets[bucket][: max(1, max_examples // 5)])
    if len(selected) < max_examples:
        selected_ids = {(row["frame"], row["slot"], row["track_id"]) for row in selected}
        for row in candidates:
            key = (row["frame"], row["slot"], row["track_id"])
            if key in selected_ids:
                continue
            selected.append(row)
            if len(selected) >= max_examples:
                break
    return selected[:max_examples], missing_count


def save_bbox_alignment_montage(examples, channel_mask_np, video_path, output_path, crop_bottom_px):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panels = []
    for row in examples:
        frame = load_video_frame(video_path, row["frame"], crop_bottom_px)
        panels.append(resize_panel(draw_bbox_alignment_panel(frame, row, channel_mask_np), width=280))
    montage = make_montage(panels, columns=5)
    if not cv2.imwrite(str(output_path), montage):
        raise RuntimeError(f"Failed to write montage: {output_path}")


def load_video_frame(video_path, frame_number, crop_bottom_px):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video for validation montage: {video_path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
    success, frame = capture.read()
    capture.release()
    if not success:
        raise RuntimeError(f"Unable to read frame {frame_number} from {video_path}")
    if crop_bottom_px > 0 and crop_bottom_px < frame.shape[0]:
        frame = frame[: -crop_bottom_px, :]
    return frame


def draw_bbox_alignment_panel(frame, row, channel_mask_np):
    output = frame.copy()
    boundary = cv2.Canny(channel_mask_np.astype(np.uint8) * 255, 50, 150) > 0
    output[boundary] = (255, 0, 0)
    center = (int(round(row["true_centroid_x"])), int(round(row["true_centroid_y"])))
    axes = (max(1, int(round(row["bbox_w"] / 2))), max(1, int(round(row["bbox_h"] / 2))))
    cv2.ellipse(output, center, axes, 0, 0, 360, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(output, center, 3, (0, 0, 255), -1)
    text = (
        f"f {row['frame']} tr {row['track_id']} s {row['slot']} "
        f"bb {row['bbox_w']:.0f}x{row['bbox_h']:.0f} O={row['outside_fraction']:.3f}"
    )
    cv2.rectangle(output, (0, 0), (min(output.shape[1] - 1, 520), 25), (0, 0, 0), -1)
    cv2.putText(output, text, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def resize_panel(image, width):
    scale = width / image.shape[1]
    return cv2.resize(image, (width, int(round(image.shape[0] * scale))), interpolation=cv2.INTER_AREA)


def make_montage(images, columns):
    height, width = images[0].shape[:2]
    rows = int(np.ceil(len(images) / columns))
    montage = np.zeros((rows * height, columns * width, 3), dtype=np.uint8)
    for index, image in enumerate(images):
        row = index // columns
        col = index % columns
        montage[row * height : (row + 1) * height, col * width : (col + 1) * width] = image
    return montage


def compute_gradient_diagnostics(model, batch, dataset, normalization_stats, weights, channel_mask, args):
    def rollout_for_grad():
        return boundary_conditioned_rollout(
            model=model,
            batch=batch,
            dataset=dataset,
            normalization_stats=normalization_stats,
            weights=weights,
            channel_mask=channel_mask,
            geometry_tolerance=args.geometry_tolerance,
            geometry_loss_weight=args.geometry_loss_weight,
            geometry_num_samples_x=args.geometry_num_samples_x,
            geometry_num_samples_y=args.geometry_num_samples_y,
        )

    rollout = rollout_for_grad()
    rollout_norms = gradient_norms_for_loss(model, rollout["weighted_loss_internal_only"])
    geometry_weighted_loss = float(args.geometry_loss_weight) * rollout_for_grad()["geometry_loss"]
    geometry_norms = gradient_norms_for_loss(model, geometry_weighted_loss)
    total_norms = gradient_norms_for_loss(model, rollout_for_grad()["total_loss"])
    ratio = geometry_norms["global"] / rollout_norms["global"] if rollout_norms["global"] > 0 else float("inf")
    return {
        "rollout_global_grad_norm": rollout_norms["global"],
        "weighted_geometry_global_grad_norm": geometry_norms["global"],
        "geometry_to_rollout_grad_norm_ratio": ratio,
        "total_global_grad_norm": total_norms["global"],
        "rollout_groups": rollout_norms["groups"],
        "weighted_geometry_groups": geometry_norms["groups"],
        "total_groups": total_norms["groups"],
    }


def gradient_norms_for_loss(model, loss):
    model.zero_grad(set_to_none=True)
    loss.backward()
    total_sq = 0.0
    group_sq = {"output_head": 0.0, "final_transformer_block": 0.0, "input_embedding": 0.0}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        grad_sq = float(parameter.grad.detach().pow(2).sum().cpu())
        total_sq += grad_sq
        if "output_head" in name or "prediction_head" in name or "velocity_head" in name or name.endswith("head"):
            group_sq["output_head"] += grad_sq
        if "encoder.layers.3" in name or "transformer_encoder.layers.3" in name or "transformer.layers.3" in name:
            group_sq["final_transformer_block"] += grad_sq
        if "droplet_mlp" in name or "time_embedding" in name or "slot_embedding" in name or "mask_embedding" in name:
            group_sq["input_embedding"] += grad_sq
    model.zero_grad(set_to_none=True)
    return {
        "global": float(np.sqrt(total_sq)),
        "groups": {key: float(np.sqrt(value)) for key, value in group_sq.items()},
    }


def rollout_weights(horizon, alpha, device):
    if horizon == 1:
        return torch.ones(1, dtype=torch.float32, device=device)
    step_ids = torch.arange(horizon, dtype=torch.float32, device=device)
    return 1.0 + alpha * step_ids / float(horizon - 1)


def run_shape_test(model, train_loader, dataset, normalization_stats, weights, channel_mask, optimizer, args, device) -> None:
    model.train()
    batch = move_batch_to_device(next(iter(train_loader)), device)
    optimizer.zero_grad(set_to_none=True)
    rollout = boundary_conditioned_rollout(
        model=model,
        batch=batch,
        dataset=dataset,
        normalization_stats=normalization_stats,
        weights=weights,
        channel_mask=channel_mask,
        geometry_tolerance=args.geometry_tolerance,
        geometry_loss_weight=args.geometry_loss_weight,
        geometry_num_samples_x=args.geometry_num_samples_x,
        geometry_num_samples_y=args.geometry_num_samples_y,
    )
    rollout["total_loss"].backward()
    finite_gradients = all(
        parameter.grad is None or torch.isfinite(parameter.grad).all().item()
        for parameter in model.parameters()
    )
    optimizer.zero_grad(set_to_none=True)

    print(f"history_x:       {tuple(batch['history_x'].shape)}")
    print(f"history_mask:    {tuple(batch['history_mask'].shape)}")
    print(f"future_y:        {tuple(batch['future_y'].shape)}")
    print(f"future_mask:     {tuple(batch['future_mask'].shape)}")
    print(f"future_bbox:     {tuple(batch['future_bbox'].shape)}")
    print(f"future_bbox_mask:{tuple(batch['future_bbox_mask'].shape)}")
    print(f"pred_velocity:   {tuple(rollout['pred_velocity'].shape)}")
    print(f"pred_position:   {tuple(rollout['pred_position'].shape)}")
    print(f"weighted_loss_internal_only: {float(rollout['weighted_loss_internal_only'].detach().cpu()):.6f}")
    print(f"geometry_loss:              {float(rollout['geometry_loss'].detach().cpu()):.6f}")
    print(f"total_loss:                 {float(rollout['total_loss'].detach().cpu()):.6f}")
    print(f"geometry_count:             {int(rollout['geometry_count'])}")
    print(f"geometry_overlap_mean:      {rollout['geometry_metrics']['overlap_mean']:.6f}")
    print(f"boundary_injected:           {int(rollout['boundary_mask'].sum().detach().cpu())}")
    print(f"valid_future_samples:        {int(rollout['mask'].sum().detach().cpu())}")
    print(f"finite_gradients_after_backward: {finite_gradients}")

    assert rollout["pred_velocity"].shape == batch["future_y"].shape
    assert rollout["mask"].shape == batch["future_mask"].shape
    assert batch["future_bbox"].shape == (*batch["future_mask"].shape, 2)
    assert torch.isfinite(rollout["geometry_loss"])
    assert torch.isfinite(rollout["total_loss"])
    assert finite_gradients
    assert int(rollout["geometry_count"]) > 0
    assert int((rollout["geometry_mask"] & rollout["boundary_mask"]).sum().detach().cpu()) == 0


def run_validation_smoke_test(model, optimizer, loader, dataset, normalization_stats, weights, channel_mask, device, output_dir, args) -> None:
    print("Running geometry-aware Markovian validation smoke test...")
    model.eval()
    batch = move_batch_to_device(next(iter(loader)), device)

    with torch.no_grad():
        rollout = boundary_conditioned_rollout(
            model=model,
            batch=batch,
            dataset=dataset,
            normalization_stats=normalization_stats,
            weights=weights,
            channel_mask=channel_mask,
            geometry_tolerance=args.geometry_tolerance,
            geometry_loss_weight=args.geometry_loss_weight,
            geometry_num_samples_x=args.geometry_num_samples_x,
            geometry_num_samples_y=args.geometry_num_samples_y,
        )
        attention_probe = model(batch["history_x"], batch["history_mask"], return_attention=True)

    assert batch["history_x"].shape[1] == T_HISTORY
    assert batch["future_y"].shape[1] == ROLLOUT_HORIZON
    assert rollout["pred_velocity"].shape == batch["future_y"].shape
    assert attention_probe["prediction"].shape == (batch["history_x"].shape[0], MAX_DROPLETS, TARGET_DIM)
    assert attention_probe["attention"].shape[2:] == (T_HISTORY * MAX_DROPLETS, T_HISTORY * MAX_DROPLETS)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": 0,
        "val_loss": float(rollout["weighted_loss_internal_only"].detach().cpu()),
        "normalization_stats": normalization_stats,
        "model_config": MODEL_CONFIG,
        "rollout_horizon": ROLLOUT_HORIZON,
        "loss_alpha": LOSS_ALPHA,
        "stride": STRIDE,
        "geometry_tolerance": args.geometry_tolerance,
        "geometry_loss_weight": args.geometry_loss_weight,
        "channel_mask_path": args.channel_mask,
        "geometry_num_samples_x": args.geometry_num_samples_x,
        "geometry_num_samples_y": args.geometry_num_samples_y,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    smoke_checkpoint = output_dir / "markovian_smoke_checkpoint.pt"
    torch.save(checkpoint, smoke_checkpoint)

    loaded = torch.load(smoke_checkpoint, map_location=device, weights_only=False)
    reloaded_model = CanonicalRolloutTransformer(**loaded["model_config"]).to(device)
    reloaded_model.load_state_dict(loaded["model_state_dict"])
    reloaded_model.eval()
    with torch.no_grad():
        reloaded_prediction = reloaded_model(batch["history_x"], batch["history_mask"])

    assert reloaded_prediction.shape == (batch["history_x"].shape[0], MAX_DROPLETS, TARGET_DIM)

    print(f"checkpoint_smoke_path: {smoke_checkpoint}")
    print(f"checkpoint_save_reload: ok")
    print(f"rollout_inference:      ok, pred_position={tuple(rollout['pred_position'].shape)}")
    print(f"geometry_loss:          ok, value={float(rollout['geometry_loss'].detach().cpu()):.6f}")
    print(f"attention_probe:        ok, attention={tuple(attention_probe['attention'].shape)}")
    print("attention_visualization_compatibility: ok; checkpoint model_config carries T_history=1")


def run_short_sanity_training(model, loader, dataset, optimizer, normalization_stats, weights, channel_mask, device, args) -> None:
    print(f"Running short sanity training for {args.sanity_train_batches} batches...")
    model.train()
    geometry_losses = []
    for batch_index, batch in enumerate(loader, start=1):
        if batch_index > args.sanity_train_batches:
            break
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        rollout = boundary_conditioned_rollout(
            model=model,
            batch=batch,
            dataset=dataset,
            normalization_stats=normalization_stats,
            weights=weights,
            channel_mask=channel_mask,
            geometry_tolerance=args.geometry_tolerance,
            geometry_loss_weight=args.geometry_loss_weight,
            geometry_num_samples_x=args.geometry_num_samples_x,
            geometry_num_samples_y=args.geometry_num_samples_y,
        )
        rollout["total_loss"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        finite_gradients = all(
            parameter.grad is None or torch.isfinite(parameter.grad).all().item()
            for parameter in model.parameters()
        )
        optimizer.step()
        geometry_losses.append(float(rollout["geometry_loss"].detach().cpu()))
        metrics = rollout["geometry_metrics"]
        print(
            f"sanity batch {batch_index:03d} "
            f"rollout_loss={float(rollout['weighted_loss_internal_only'].detach().cpu()):.6f} "
            f"geometry_loss={float(rollout['geometry_loss'].detach().cpu()):.6f} "
            f"total_loss={float(rollout['total_loss'].detach().cpu()):.6f} "
            f"overlap_mean={metrics['overlap_mean']:.6f} "
            f"overlap_p95={metrics['overlap_p95']:.6f} "
            f"overlap_max={metrics['overlap_max']:.6f} "
            f"geometry_count={int(rollout['geometry_count'])} "
            f"grad_norm={float(grad_norm):.6f} "
            f"finite_gradients={finite_gradients}"
        )
        if not finite_gradients or not torch.isfinite(rollout["total_loss"]):
            raise RuntimeError("NaN/inf detected during short sanity training.")

    if geometry_losses:
        print(
            "short_sanity_training_summary "
            f"first_geometry_loss={geometry_losses[0]:.6f} "
            f"last_geometry_loss={geometry_losses[-1]:.6f} "
            f"changed={abs(geometry_losses[-1] - geometry_losses[0]) > 1e-12}"
        )


def train_one_epoch(model, loader, dataset, optimizer, normalization_stats, weights, channel_mask, device, args, log_every):
    model.train()
    total_loss = 0.0
    total_boundary = 0
    total_valid = 0
    total_rollouts = 0
    num_batches = 0
    total_batches = len(loader)
    geometry_accumulator = new_geometry_epoch_accumulator()

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)
        rollout = boundary_conditioned_rollout(
            model=model,
            batch=batch,
            dataset=dataset,
            normalization_stats=normalization_stats,
            weights=weights,
            channel_mask=channel_mask,
            geometry_tolerance=args.geometry_tolerance,
            geometry_loss_weight=args.geometry_loss_weight,
            geometry_num_samples_x=args.geometry_num_samples_x,
            geometry_num_samples_y=args.geometry_num_samples_y,
        )
        loss = rollout["total_loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        total_loss += float(loss.detach().cpu())
        update_geometry_epoch_accumulator(geometry_accumulator, rollout)
        total_boundary += int(rollout["boundary_mask"].sum().detach().cpu())
        total_valid += int(rollout["mask"].sum().detach().cpu())
        total_rollouts += int(batch["history_x"].shape[0])
        num_batches += 1
        if log_every > 0 and (num_batches % log_every == 0 or num_batches == total_batches):
            print_progress("train", num_batches, total_batches, total_loss / num_batches)

    return {
        "total_loss": total_loss / max(num_batches, 1),
        **geometry_summary_from_accumulator(geometry_accumulator),
        "boundary_injected_per_rollout": total_boundary / max(total_rollouts, 1),
        "boundary_fraction": total_boundary / max(total_valid, 1),
    }


def evaluate(model, loader, dataset, normalization_stats, weights, channel_mask, device, args, log_every=0):
    model.eval()
    total_loss = 0.0
    total_boundary = 0
    total_valid = 0
    total_rollouts = 0
    num_batches = 0
    total_batches = len(loader)
    accumulators = create_accumulators(ROLLOUT_HORIZON)
    geometry_accumulator = new_geometry_epoch_accumulator()

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            rollout = boundary_conditioned_rollout(
                model=model,
                batch=batch,
                dataset=dataset,
                normalization_stats=normalization_stats,
                weights=weights,
                channel_mask=channel_mask,
                geometry_tolerance=args.geometry_tolerance,
                geometry_loss_weight=args.geometry_loss_weight,
                geometry_num_samples_x=args.geometry_num_samples_x,
                geometry_num_samples_y=args.geometry_num_samples_y,
            )
            total_loss += float(rollout["total_loss"].detach().cpu())
            total_boundary += int(rollout["boundary_mask"].sum().detach().cpu())
            total_valid += int(rollout["mask"].sum().detach().cpu())
            total_rollouts += int(batch["history_x"].shape[0])
            num_batches += 1
            update_metric_accumulators(accumulators, rollout)
            update_geometry_epoch_accumulator(geometry_accumulator, rollout)
            if log_every > 0 and (num_batches % log_every == 0 or num_batches == total_batches):
                print_progress("val", num_batches, total_batches, total_loss / num_batches)

    summary = metrics_from_accumulator(accumulators["overall"])
    summary["total_loss"] = total_loss / max(num_batches, 1)
    summary.update(geometry_summary_from_accumulator(geometry_accumulator))
    summary["boundary_injected_per_rollout"] = total_boundary / max(total_rollouts, 1)
    summary["boundary_fraction"] = total_boundary / max(total_valid, 1)
    summary["boundary_injected_total"] = total_boundary
    summary["valid_future_samples_total"] = total_valid
    summary["step_rmse_position"] = [
        metrics_from_accumulator(accumulator)["rmse_position"]
        for accumulator in accumulators["steps"]
    ]
    return summary


def boundary_conditioned_rollout(
    model,
    batch,
    dataset,
    normalization_stats,
    weights,
    channel_mask,
    geometry_tolerance,
    geometry_loss_weight,
    geometry_num_samples_x,
    geometry_num_samples_y,
):
    device = batch["history_x"].device
    rollout_history = batch["history_x"].clone()
    history_mask = batch["history_mask"].clone()

    pred_velocities_norm = []
    true_velocities_norm = []
    pred_velocities_phys = []
    true_velocities_phys = []
    pred_positions = []
    true_positions = []
    step_masks = []
    boundary_masks = []
    internal_loss_masks = []
    step_losses = []

    feature_index = dataset.feature_indices
    true_future_features = get_true_future_features(batch, dataset, device, weights.numel())
    true_future_xy = true_future_features[:, :, :, [
        feature_index["x"],
        feature_index["y"],
    ]]

    for step_index in range(weights.numel()):
        previous_last_mask = history_mask[:, -1, :]
        pred_step_norm_raw = model(rollout_history, history_mask)
        pred_step_phys_raw = denormalize_targets(
            pred_step_norm_raw[:, None, :, :],
            normalization_stats,
            device,
        )[:, 0, :, :]

        true_step_norm = batch["future_y"][:, step_index, :, :]
        true_step_phys = denormalize_targets(
            true_step_norm[:, None, :, :],
            normalization_stats,
            device,
        )[:, 0, :, :]

        history_phys = denormalize_features(rollout_history, normalization_stats, device)
        last_frame = history_phys[:, -1, :, :]
        x_next = last_frame[:, :, feature_index["x"]] + pred_step_phys_raw[:, :, 0]
        y_next = last_frame[:, :, feature_index["y"]] + pred_step_phys_raw[:, :, 1]

        new_frame_phys = last_frame.clone()
        new_frame_phys[:, :, feature_index["x"]] = x_next
        new_frame_phys[:, :, feature_index["y"]] = y_next
        new_frame_phys[:, :, feature_index["vx"]] = pred_step_phys_raw[:, :, 0]
        new_frame_phys[:, :, feature_index["vy"]] = pred_step_phys_raw[:, :, 1]

        new_mask = batch["future_mask"][:, step_index, :]
        entering_mask = new_mask & ~previous_last_mask
        true_step_features = true_future_features[:, step_index, :, :]
        true_step_features_finite = torch.isfinite(true_step_features).all(dim=-1)
        boundary_mask = entering_mask & true_step_features_finite
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

        pred_step_norm = pred_step_norm_raw.clone()
        pred_step_phys = pred_step_phys_raw.clone()
        pred_step_norm[boundary_mask] = true_step_norm[boundary_mask]
        pred_step_phys[boundary_mask] = true_step_phys[boundary_mask]

        loss_mask = new_mask & ~boundary_mask
        step_loss = masked_velocity_mse(pred_step_norm, true_step_norm, loss_mask)
        step_losses.append(step_loss)

        new_frame_norm = normalize_features(new_frame_phys, normalization_stats, device)
        new_frame_norm = torch.where(
            new_mask[:, :, None],
            new_frame_norm,
            torch.zeros_like(new_frame_norm),
        )
        rollout_history = torch.cat(
            [rollout_history[:, 1:, :, :], new_frame_norm[:, None, :, :]],
            dim=1,
        )
        history_mask = torch.cat(
            [history_mask[:, 1:, :], new_mask[:, None, :]],
            dim=1,
        )

        pred_velocities_norm.append(pred_step_norm)
        true_velocities_norm.append(true_step_norm)
        pred_velocities_phys.append(pred_step_phys)
        true_velocities_phys.append(true_step_phys)
        pred_positions.append(new_frame_phys[:, :, [feature_index["x"], feature_index["y"]]])
        true_positions.append(true_future_xy[:, step_index, :, :])
        step_masks.append(new_mask)
        boundary_masks.append(boundary_mask)
        internal_loss_masks.append(loss_mask)

    step_loss_tensor = torch.stack(step_losses)
    weighted_loss_internal_only = (step_loss_tensor * weights).sum() / weights.sum()
    pred_position_tensor = torch.stack(pred_positions, dim=1)
    mask_tensor = torch.stack(step_masks, dim=1)
    boundary_mask_tensor = torch.stack(boundary_masks, dim=1)
    internal_loss_mask_tensor = torch.stack(internal_loss_masks, dim=1)
    geometry_loss, geometry_payload = compute_geometry_loss(
        pred_position=pred_position_tensor,
        future_bbox=batch["future_bbox"],
        future_bbox_mask=batch["future_bbox_mask"],
        geometry_mask=internal_loss_mask_tensor,
        channel_mask=channel_mask,
        tolerance=geometry_tolerance,
        num_samples_x=geometry_num_samples_x,
        num_samples_y=geometry_num_samples_y,
    )
    total_loss = weighted_loss_internal_only + float(geometry_loss_weight) * geometry_loss

    return {
        "weighted_loss": weighted_loss_internal_only,
        "total_loss": total_loss,
        "weighted_loss_internal_only": weighted_loss_internal_only,
        "rollout_loss": weighted_loss_internal_only,
        "geometry_loss": geometry_loss,
        "geometry_count": geometry_payload["count"],
        "geometry_overlaps": geometry_payload["overlaps"],
        "geometry_mask": geometry_payload["mask"],
        "geometry_metrics": geometry_payload["metrics"],
        "step_losses": step_loss_tensor,
        "pred_velocity_norm": torch.stack(pred_velocities_norm, dim=1),
        "true_velocity_norm": torch.stack(true_velocities_norm, dim=1),
        "pred_velocity": torch.stack(pred_velocities_phys, dim=1),
        "true_velocity": torch.stack(true_velocities_phys, dim=1),
        "pred_position": pred_position_tensor,
        "true_position": torch.stack(true_positions, dim=1),
        "mask": mask_tensor,
        "boundary_mask": boundary_mask_tensor,
        "internal_loss_mask": internal_loss_mask_tensor,
    }


def compute_geometry_loss(
    pred_position,
    future_bbox,
    future_bbox_mask,
    geometry_mask,
    channel_mask,
    tolerance,
    num_samples_x,
    num_samples_y,
):
    finite = torch.isfinite(pred_position).all(dim=-1) & torch.isfinite(future_bbox).all(dim=-1)
    positive_bbox = (future_bbox > 0).all(dim=-1)
    mask = geometry_mask & future_bbox_mask & finite & positive_bbox
    if mask.sum().item() == 0:
        zero = pred_position.sum() * 0.0
        return zero, {
            "count": 0,
            "overlaps": torch.empty(0, dtype=pred_position.dtype, device=pred_position.device),
            "mask": mask,
            "metrics": empty_geometry_metrics(),
        }

    overlaps = compute_ellipse_outside_fraction_torch(
        pred_position[mask],
        future_bbox[mask],
        channel_mask,
        num_samples_x=num_samples_x,
        num_samples_y=num_samples_y,
    )
    penalties = torch.relu(overlaps - float(tolerance)).square()
    geometry_loss = penalties.mean()
    return geometry_loss, {
        "count": int(mask.sum().detach().cpu()),
        "overlaps": overlaps,
        "mask": mask,
        "metrics": geometry_metrics_from_overlaps(overlaps),
    }


def empty_geometry_metrics():
    return {
        "overlap_mean": 0.0,
        "overlap_median": 0.0,
        "overlap_p95": 0.0,
        "overlap_max": 0.0,
        "overlap_fraction_gt_0": 0.0,
        "overlap_fraction_gt_0p01": 0.0,
        "overlap_fraction_gt_0p02": 0.0,
        "overlap_fraction_gt_0p05": 0.0,
    }


def geometry_metrics_from_overlaps(overlaps):
    if overlaps.numel() == 0:
        return empty_geometry_metrics()
    detached = overlaps.detach()
    return {
        "overlap_mean": float(detached.mean().cpu()),
        "overlap_median": float(detached.median().cpu()),
        "overlap_p95": float(torch.quantile(detached, 0.95).cpu()),
        "overlap_max": float(detached.max().cpu()),
        "overlap_fraction_gt_0": float((detached > 0).float().mean().cpu()),
        "overlap_fraction_gt_0p01": float((detached > 0.01).float().mean().cpu()),
        "overlap_fraction_gt_0p02": float((detached > 0.02).float().mean().cpu()),
        "overlap_fraction_gt_0p05": float((detached > 0.05).float().mean().cpu()),
    }


def new_geometry_epoch_accumulator():
    return {
        "rollout_loss_sum": 0.0,
        "geometry_loss_sum": 0.0,
        "total_loss_sum": 0.0,
        "batches": 0,
        "geometry_count": 0,
        "boundary_excluded_count": 0,
        "overlaps": [],
    }


def update_geometry_epoch_accumulator(accumulator, rollout):
    accumulator["rollout_loss_sum"] += float(rollout["weighted_loss_internal_only"].detach().cpu())
    accumulator["geometry_loss_sum"] += float(rollout["geometry_loss"].detach().cpu())
    accumulator["total_loss_sum"] += float(rollout["total_loss"].detach().cpu())
    accumulator["batches"] += 1
    accumulator["geometry_count"] += int(rollout["geometry_count"])
    accumulator["boundary_excluded_count"] += int(rollout["boundary_mask"].sum().detach().cpu())
    overlaps = rollout["geometry_overlaps"].detach().cpu()
    if overlaps.numel() > 0:
        accumulator["overlaps"].append(overlaps)


def geometry_summary_from_accumulator(accumulator):
    batches = max(accumulator["batches"], 1)
    if accumulator["overlaps"]:
        overlaps = torch.cat(accumulator["overlaps"])
        overlap_metrics = geometry_metrics_from_overlaps(overlaps)
    else:
        overlap_metrics = empty_geometry_metrics()
    return {
        "weighted_loss_internal_only": accumulator["rollout_loss_sum"] / batches,
        "geometry_loss": accumulator["geometry_loss_sum"] / batches,
        "total_loss": accumulator["total_loss_sum"] / batches,
        "geometry_count": accumulator["geometry_count"],
        "boundary_excluded_from_geometry": accumulator["boundary_excluded_count"],
        **overlap_metrics,
    }


def masked_velocity_mse(prediction, target, mask):
    expanded_mask = mask.unsqueeze(-1).expand_as(target)
    squared_error = (prediction - target) ** 2
    valid_error = squared_error[expanded_mask]
    if valid_error.numel() == 0:
        return squared_error.sum() * 0.0
    return valid_error.mean()


def get_true_future_features(batch, dataset, device, horizon):
    droplet_ids = batch["droplet_ids"].detach().cpu().numpy()
    frame_starts = batch["frame_start"].detach().cpu().numpy()
    track_id_to_index = {int(track_id): index for index, track_id in enumerate(dataset.track_ids)}

    B, M = droplet_ids.shape
    true_features = np.full((B, horizon, M, len(dataset.feature_names)), np.nan, dtype=np.float32)

    for batch_index in range(B):
        start = int(frame_starts[batch_index]) + dataset.T_history
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


def move_batch_to_device(batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def denormalize_features(features, normalization_stats, device):
    mean = torch.as_tensor(normalization_stats["input_mean"], dtype=torch.float32, device=device)
    std = torch.as_tensor(normalization_stats["input_std"], dtype=torch.float32, device=device)
    return features * std.view(1, 1, 1, -1) + mean.view(1, 1, 1, -1)


def normalize_features(features, normalization_stats, device):
    mean = torch.as_tensor(normalization_stats["input_mean"], dtype=torch.float32, device=device)
    std = torch.as_tensor(normalization_stats["input_std"], dtype=torch.float32, device=device)
    return (features - mean.view(1, 1, -1)) / std.view(1, 1, -1)


def denormalize_targets(targets, normalization_stats, device):
    mean = torch.as_tensor(normalization_stats["target_mean"], dtype=torch.float32, device=device)
    std = torch.as_tensor(normalization_stats["target_std"], dtype=torch.float32, device=device)
    return targets * std.view(1, 1, 1, -1) + mean.view(1, 1, 1, -1)


def create_accumulators(num_steps):
    return {
        "overall": new_accumulator(),
        "steps": [new_accumulator() for _ in range(num_steps)],
    }


def new_accumulator():
    return {
        "count": 0,
        "sum_sq_vx": 0.0,
        "sum_sq_vy": 0.0,
        "sum_sq_speed": 0.0,
        "position_count": 0,
        "sum_sq_position": 0.0,
    }


def update_metric_accumulators(accumulators, rollout):
    velocity_error = rollout["pred_velocity"] - rollout["true_velocity"]
    speed_error = torch.sqrt(velocity_error[..., 0] ** 2 + velocity_error[..., 1] ** 2)
    position_error = rollout["pred_position"] - rollout["true_position"]
    position_error_norm = torch.sqrt(position_error[..., 0] ** 2 + position_error[..., 1] ** 2)
    position_finite = torch.isfinite(position_error).all(dim=-1)

    update_one_accumulator(
        accumulators["overall"],
        velocity_error,
        speed_error,
        rollout["mask"],
        position_error_norm,
        rollout["mask"] & position_finite,
    )
    for step_index in range(rollout["mask"].shape[1]):
        update_one_accumulator(
            accumulators["steps"][step_index],
            velocity_error[:, step_index, :, :],
            speed_error[:, step_index, :],
            rollout["mask"][:, step_index, :],
            position_error_norm[:, step_index, :],
            rollout["mask"][:, step_index, :] & position_finite[:, step_index, :],
        )


def update_one_accumulator(accumulator, velocity_error, speed_error, velocity_mask, position_error_norm, position_mask):
    valid = velocity_mask.bool()
    if valid.sum().item() > 0:
        vx_error = velocity_error[..., 0][valid]
        vy_error = velocity_error[..., 1][valid]
        speed = speed_error[valid]
        accumulator["count"] += int(valid.sum().item())
        accumulator["sum_sq_vx"] += float((vx_error**2).sum().detach().cpu())
        accumulator["sum_sq_vy"] += float((vy_error**2).sum().detach().cpu())
        accumulator["sum_sq_speed"] += float((speed**2).sum().detach().cpu())

    valid_position = position_mask.bool()
    if valid_position.sum().item() > 0:
        position = position_error_norm[valid_position]
        accumulator["position_count"] += int(valid_position.sum().item())
        accumulator["sum_sq_position"] += float((position**2).sum().detach().cpu())


def metrics_from_accumulator(accumulator):
    count = accumulator["count"]
    position_count = accumulator["position_count"]
    return {
        "valid_samples": count,
        "valid_position_samples": position_count,
        "rmse_vx": safe_rmse(accumulator["sum_sq_vx"], count),
        "rmse_vy": safe_rmse(accumulator["sum_sq_vy"], count),
        "rmse_speed": safe_rmse(accumulator["sum_sq_speed"], count),
        "rmse_position": safe_rmse(accumulator["sum_sq_position"], position_count),
    }


def safe_rmse(sum_sq, count):
    return np.sqrt(sum_sq / count) if count else np.nan


def print_progress(label, num_batches, total_batches, running_loss):
    percent = 100.0 * num_batches / max(total_batches, 1)
    print(
        f"  {label:<5} batch {num_batches:04d}/{total_batches:04d} "
        f"({percent:5.1f}%) weighted_loss={running_loss:.6f}"
    )


def print_epoch_summary(epoch, train_summary, val_summary):
    step_text = " ".join(
        f"s{step}={val_summary['step_rmse_position'][step - 1]:.6f}"
        for step in DIAGNOSTIC_STEPS
    )
    print(
        f"epoch {epoch:03d} "
        f"train_weighted_loss_internal_only={train_summary['weighted_loss_internal_only']:.6f} "
        f"train_geometry_loss={train_summary['geometry_loss']:.6f} "
        f"train_total_loss={train_summary['total_loss']:.6f} "
        f"val_weighted_loss_internal_only={val_summary['weighted_loss_internal_only']:.6f} "
        f"val_geometry_loss={val_summary['geometry_loss']:.6f} "
        f"val_total_loss={val_summary['total_loss']:.6f} "
        f"val_rmse_vx={val_summary['rmse_vx']:.6f} "
        f"val_rmse_vy={val_summary['rmse_vy']:.6f} "
        f"val_rmse_speed={val_summary['rmse_speed']:.6f} "
        f"val_rmse_position={val_summary['rmse_position']:.6f}"
    )
    print(
        "  boundary_diagnostics "
        f"train_boundary_injected_per_rollout={train_summary['boundary_injected_per_rollout']:.3f} "
        f"train_boundary_fraction={train_summary['boundary_fraction']:.6f} "
        f"val_boundary_injected_per_rollout={val_summary['boundary_injected_per_rollout']:.3f} "
        f"val_boundary_fraction={val_summary['boundary_fraction']:.6f}"
    )
    print(
        "  geometry_diagnostics "
        f"train_count={train_summary['geometry_count']} "
        f"train_overlap_mean={train_summary['overlap_mean']:.6f} "
        f"train_overlap_p95={train_summary['overlap_p95']:.6f} "
        f"train_overlap_max={train_summary['overlap_max']:.6f} "
        f"val_count={val_summary['geometry_count']} "
        f"val_overlap_mean={val_summary['overlap_mean']:.6f} "
        f"val_overlap_p95={val_summary['overlap_p95']:.6f} "
        f"val_overlap_max={val_summary['overlap_max']:.6f}"
    )
    print(f"  stepwise_val_rmse_position {step_text}")


def initialize_curves_csv(path):
    if path.exists():
        return
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "epoch",
                "train_weighted_loss_internal_only",
                "train_geometry_loss",
                "train_total_loss",
                "val_weighted_loss_internal_only",
                "val_geometry_loss",
                "val_total_loss",
                "train_boundary_injected_per_rollout",
                "train_boundary_fraction",
                "train_boundary_excluded_from_geometry",
                "train_geometry_count",
                "train_overlap_mean",
                "train_overlap_median",
                "train_overlap_p95",
                "train_overlap_max",
                "train_overlap_fraction_gt_0",
                "train_overlap_fraction_gt_0p01",
                "train_overlap_fraction_gt_0p02",
                "train_overlap_fraction_gt_0p05",
                "val_boundary_injected_per_rollout",
                "val_boundary_fraction",
                "val_boundary_excluded_from_geometry",
                "val_geometry_count",
                "val_overlap_mean",
                "val_overlap_median",
                "val_overlap_p95",
                "val_overlap_max",
                "val_overlap_fraction_gt_0",
                "val_overlap_fraction_gt_0p01",
                "val_overlap_fraction_gt_0p02",
                "val_overlap_fraction_gt_0p05",
                "val_rmse_vx",
                "val_rmse_vy",
                "val_rmse_speed",
                "val_rmse_position",
                *[f"val_rmse_position_step_{step}" for step in DIAGNOSTIC_STEPS],
            ]
        )


def append_curves_csv(path, epoch, train_summary, val_summary):
    with path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                epoch,
                train_summary["weighted_loss_internal_only"],
                train_summary["geometry_loss"],
                train_summary["total_loss"],
                val_summary["weighted_loss_internal_only"],
                val_summary["geometry_loss"],
                val_summary["total_loss"],
                train_summary["boundary_injected_per_rollout"],
                train_summary["boundary_fraction"],
                train_summary["boundary_excluded_from_geometry"],
                train_summary["geometry_count"],
                train_summary["overlap_mean"],
                train_summary["overlap_median"],
                train_summary["overlap_p95"],
                train_summary["overlap_max"],
                train_summary["overlap_fraction_gt_0"],
                train_summary["overlap_fraction_gt_0p01"],
                train_summary["overlap_fraction_gt_0p02"],
                train_summary["overlap_fraction_gt_0p05"],
                val_summary["boundary_injected_per_rollout"],
                val_summary["boundary_fraction"],
                val_summary["boundary_excluded_from_geometry"],
                val_summary["geometry_count"],
                val_summary["overlap_mean"],
                val_summary["overlap_median"],
                val_summary["overlap_p95"],
                val_summary["overlap_max"],
                val_summary["overlap_fraction_gt_0"],
                val_summary["overlap_fraction_gt_0p01"],
                val_summary["overlap_fraction_gt_0p02"],
                val_summary["overlap_fraction_gt_0p05"],
                val_summary["rmse_vx"],
                val_summary["rmse_vy"],
                val_summary["rmse_speed"],
                val_summary["rmse_position"],
                *[val_summary["step_rmse_position"][step - 1] for step in DIAGNOSTIC_STEPS],
            ]
        )


def make_random_validation_animation(model, dataset, normalization_stats, channel_mask, device, args, output_path):
    if len(dataset) == 0:
        print("Skipping animation because the validation dataset is empty.")
        return
    try:
        import imageio_ffmpeg
    except ImportError:
        print("Skipping animation because imageio-ffmpeg is not installed.")
        return

    model.eval()
    sample_index = random.randrange(len(dataset))
    batch = default_collate([dataset[sample_index]])
    batch = move_batch_to_device(batch, device)
    weights = rollout_weights(ROLLOUT_HORIZON, LOSS_ALPHA, device)
    with torch.no_grad():
        rollout = boundary_conditioned_rollout(
            model=model,
            batch=batch,
            dataset=dataset,
            normalization_stats=normalization_stats,
            weights=weights,
            channel_mask=channel_mask,
            geometry_tolerance=args.geometry_tolerance,
            geometry_loss_weight=args.geometry_loss_weight,
            geometry_num_samples_x=args.geometry_num_samples_x,
            geometry_num_samples_y=args.geometry_num_samples_y,
        )

    payload = {
        "pred_position": rollout["pred_position"][0].detach().cpu().numpy(),
        "true_position": rollout["true_position"][0].detach().cpu().numpy(),
        "mask": rollout["mask"][0].detach().cpu().numpy(),
        "droplet_ids": batch["droplet_ids"][0].detach().cpu().numpy(),
    }
    save_rollout_animation(payload, output_path, imageio_ffmpeg.get_ffmpeg_exe())


def save_rollout_animation(payload, output_path, ffmpeg_path):
    pred_position = payload["pred_position"]
    true_position = payload["true_position"]
    mask = payload["mask"].astype(bool)
    droplet_ids = payload["droplet_ids"]

    valid_positions = np.concatenate([pred_position[mask], true_position[mask]], axis=0)
    if valid_positions.size == 0:
        print("Skipping animation because no valid rollout positions are available.")
        return

    x_min, y_min = np.nanmin(valid_positions, axis=0)
    x_max, y_max = np.nanmax(valid_positions, axis=0)
    x_pad = max((x_max - x_min) * 0.05, 1.0)
    y_pad = max((y_max - y_min) * 0.05, 1.0)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_max + y_pad, y_min - y_pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    gt_scatter = ax.scatter([], [], s=35, c="black", label="ground truth", alpha=0.8)
    pred_scatter = ax.scatter([], [], s=35, c="red", marker="x", label="prediction", alpha=0.8)
    ax.legend(loc="upper right")

    def draw(step_index):
        valid = mask[step_index]
        gt = true_position[step_index, valid]
        pred = pred_position[step_index, valid]
        gt_scatter.set_offsets(gt if gt.size else np.empty((0, 2)))
        pred_scatter.set_offsets(pred if pred.size else np.empty((0, 2)))
        ax.set_title(f"Boundary-conditioned rollout - step {step_index + 1}")
        return [gt_scatter, pred_scatter]

    plt.rcParams["animation.ffmpeg_path"] = ffmpeg_path
    animation = FuncAnimation(fig, draw, frames=np.arange(ROLLOUT_HORIZON), interval=200, blit=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=5)
    animation.save(output_path, writer=writer)
    plt.close(fig)
    print(f"Saved rollout animation: {output_path}")


if __name__ == "__main__":
    main()
