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
  droplet_detection_tracking/
    configs/
      settings.py
    detection/
    tracking/
    io.py
    preview.py
    schema.py
    cli.py
  pyproject.toml
```

## Install

From this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

From another project, install this package from a local path:

```powershell
python -m pip install -e "D:\path\to\droplet-detection-tracking"
```

Or install from a Git repo once this folder is pushed to its own repository:

```powershell
python -m pip install "git+https://github.com/<owner>/droplet-detection-tracking.git"
```

## Run

```powershell
droplet-detect-track --video-path "D:\path\to\video.avi" --experiment-name 2
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
droplet-detect-track `
  --video-path "D:\path\to\video.avi" `
  --output-dir outputs\processed `
  --experiment-name 2 `
  --no-live-preview
```

If `--video-path` is omitted, the script uses the config defaults in
`droplet_detection_tracking/configs/settings.py`.

## Behavior-Preserving Note

This extraction does not change watershed parameters, Kalman/Hungarian tracking,
background subtraction, hole filling, area filtering, track splitting, CSV
columns, or debug preview behavior. Robustness improvements should happen in a
separate phase after regression comparison.
