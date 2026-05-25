# Droplet Detection + Tracking

Standalone extraction of the existing droplet detection and tracking pipeline.

This project intentionally preserves the current behavior. It does not include
geometry projection, Transformer training, rollout evaluation, or ML prediction
code.

## What It Does

Input:

- raw video of droplets

Output:

- `all_frame_features.csv`
- `tracked_features.csv`
- optional OpenCV debug/live preview

The key compatibility output is `tracked_features.csv`, with the same schema as
the source project.

## Project Layout

```text
droplet-detection-tracking/
  configs/
  scripts/
  src/
    detection/
    tracking/
    visualization/
    io/
  outputs/
```

## Install

From this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

You can also run it with the parent repository virtual environment if those
dependencies are already installed.

## Run

```powershell
python scripts\run_detection_tracking.py --video-path "D:\path\to\video.avi" --experiment-name 2
```

By default, output is written to:

```text
outputs/processed/<experiment-name>/
```

For example:

```text
outputs/processed/2/tracked_features.csv
```

Useful options:

```powershell
python scripts\run_detection_tracking.py `
  --video-path "D:\path\to\video.avi" `
  --output-dir outputs\processed `
  --experiment-name 2 `
  --no-live-preview
```

If `--video-path` is omitted, the script uses the copied config defaults in
`configs/constants.py` and `configs/paths.py`.

## Compare Old And New Outputs

Run the standalone pipeline on the same video, then compare:

```powershell
python scripts\compare_old_new_outputs.py `
  --old-csv "..\outputs\processed\2\tracked_features.csv" `
  --new-csv "outputs\processed\2\tracked_features.csv" `
  --output-dir outputs\comparison
```

This writes:

- `comparison_report.txt`
- `comparison_summary.csv`

The comparison checks row count, column names, frame range, unique track count,
per-frame detection counts, and centroid differences when row alignment is
possible.

## Behavior-Preserving Note

This extraction does not change watershed parameters, Kalman/Hungarian tracking,
background subtraction, hole filling, area filtering, track splitting, CSV
columns, or debug preview behavior. Robustness improvements should happen in a
separate phase after regression comparison.
