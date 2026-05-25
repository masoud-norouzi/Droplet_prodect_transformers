from __future__ import annotations

import argparse
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare old and standalone tracked_features.csv outputs."
    )
    parser.add_argument("--old-csv", type=Path, required=True, help="Original tracked_features.csv.")
    parser.add_argument("--new-csv", type=Path, required=True, help="Standalone tracked_features.csv.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "comparison",
        help="Directory for comparison_report.txt and comparison_summary.csv.",
    )
    return parser.parse_args()


def frame_range(df: pd.DataFrame) -> str:
    if "frame" not in df.columns or df.empty:
        return ""
    return f"{int(df['frame'].min())}-{int(df['frame'].max())}"


def unique_track_count(df: pd.DataFrame) -> int | None:
    if "track_id" not in df.columns:
        return None
    return int(df["track_id"].dropna().nunique())


def per_frame_count_diffs(old_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    if "frame" not in old_df.columns or "frame" not in new_df.columns:
        return pd.DataFrame()

    old_counts = old_df.groupby("frame").size().rename("old_count")
    new_counts = new_df.groupby("frame").size().rename("new_count")
    counts = pd.concat([old_counts, new_counts], axis=1).fillna(0).astype(int).reset_index()
    counts["count_diff"] = counts["new_count"] - counts["old_count"]
    return counts


def sorted_for_alignment(df: pd.DataFrame) -> pd.DataFrame:
    sort_columns = [column for column in ["frame", "track_id", "label"] if column in df.columns]
    if sort_columns:
        return df.sort_values(sort_columns).reset_index(drop=True)
    return df.reset_index(drop=True)


def centroid_diff_summary(old_df: pd.DataFrame, new_df: pd.DataFrame) -> dict[str, float | str]:
    required = {"centroid_x", "centroid_y"}
    if not required.issubset(old_df.columns) or not required.issubset(new_df.columns):
        return {"centroid_alignment": "missing centroid columns"}
    if len(old_df) != len(new_df):
        return {"centroid_alignment": "row counts differ"}

    old_aligned = sorted_for_alignment(old_df)
    new_aligned = sorted_for_alignment(new_df)
    dx = new_aligned["centroid_x"].to_numpy(float) - old_aligned["centroid_x"].to_numpy(float)
    dy = new_aligned["centroid_y"].to_numpy(float) - old_aligned["centroid_y"].to_numpy(float)
    distance = np.sqrt(dx**2 + dy**2)
    return {
        "centroid_alignment": "sorted row alignment",
        "max_abs_centroid_dx": float(np.max(np.abs(dx))) if len(dx) else 0.0,
        "max_abs_centroid_dy": float(np.max(np.abs(dy))) if len(dy) else 0.0,
        "max_centroid_distance": float(np.max(distance)) if len(distance) else 0.0,
        "mean_centroid_distance": float(np.mean(distance)) if len(distance) else 0.0,
    }


def main() -> None:
    args = parse_args()
    old_df = pd.read_csv(args.old_csv)
    new_df = pd.read_csv(args.new_csv)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    old_columns = list(old_df.columns)
    new_columns = list(new_df.columns)
    count_diffs = per_frame_count_diffs(old_df, new_df)
    centroid_summary = centroid_diff_summary(old_df, new_df)

    summary_rows = [
        {
            "metric": "row_count",
            "old_value": len(old_df),
            "new_value": len(new_df),
            "match": len(old_df) == len(new_df),
        },
        {
            "metric": "column_names",
            "old_value": "|".join(old_columns),
            "new_value": "|".join(new_columns),
            "match": old_columns == new_columns,
        },
        {
            "metric": "frame_range",
            "old_value": frame_range(old_df),
            "new_value": frame_range(new_df),
            "match": frame_range(old_df) == frame_range(new_df),
        },
        {
            "metric": "unique_track_ids",
            "old_value": unique_track_count(old_df),
            "new_value": unique_track_count(new_df),
            "match": unique_track_count(old_df) == unique_track_count(new_df),
        },
    ]

    if not count_diffs.empty:
        summary_rows.append(
            {
                "metric": "per_frame_detection_counts",
                "old_value": "",
                "new_value": "",
                "match": bool((count_diffs["count_diff"] == 0).all()),
            }
        )

    for key, value in centroid_summary.items():
        summary_rows.append(
            {
                "metric": key,
                "old_value": "",
                "new_value": value,
                "match": value == 0.0 if isinstance(value, float) else "",
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary_path = args.output_dir / "comparison_summary.csv"
    report_path = args.output_dir / "comparison_report.txt"
    summary.to_csv(summary_path, index=False)

    lines = [
        "Droplet detection/tracking output comparison",
        "",
        f"Old CSV: {args.old_csv}",
        f"New CSV: {args.new_csv}",
        "",
        summary.to_string(index=False),
    ]
    if not count_diffs.empty:
        differing = count_diffs[count_diffs["count_diff"] != 0]
        lines.extend(["", "Per-frame detection count differences:"])
        lines.append(differing.to_string(index=False) if not differing.empty else "None")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {report_path}")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
