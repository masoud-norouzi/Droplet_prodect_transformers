from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# External location — change this to where your videos actually live
RAW_VIDEO_DIR = Path(r"D:\Microfluidic loop projct\new loop experiments\confined droplets 2")

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FRAME_DIR = OUTPUT_DIR / "frames"
PROCESSED_DIR = OUTPUT_DIR / "processed"