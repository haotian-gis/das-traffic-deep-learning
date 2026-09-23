"""
Central configuration for the DAS traffic-monitoring workflow.

All locations derive from ``DAS_DATA_ROOT`` (default ``./data``); outputs go to
``DAS_OUTPUT_DIR`` (default ``./outputs``). Edit the constants here rather than
hard-coding paths in the modules.
"""
import os
from pathlib import Path

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("DAS_DATA_ROOT", REPO_ROOT / "data"))
OUTPUT_DIR = Path(os.environ.get("DAS_OUTPUT_DIR", REPO_ROOT / "outputs"))

# Detection datasets (YOLO format: images/*.png + labels/*.txt)
LABELED_IMG_DIR = DATA_ROOT / "labeled" / "images"      # manually annotated real images
LABELED_LBL_DIR = DATA_ROOT / "labeled" / "labels"
DATASET_SPLIT_DIR = DATA_ROOT / "labeled" / "dataset_split"
REAL_YAML = DATASET_SPLIT_DIR / "dataset_real.yaml"
SYNTH_YAML = DATA_ROOT / "synthetic" / "dataset_synth.yaml"   # synthetic-only dataset
MIX_YAML = DATA_ROOT / "synthetic" / "dataset_mix.yaml"       # synthetic + real training images
HOLDOUT_DIR = DATA_ROOT / "holdout"                           # independent hold-out: images/ + labels/
RUNS_DIR = DATA_ROOT / "runs_compare"                         # Ultralytics training runs
PRETRAINED_DIR = Path(os.environ.get("DAS_WEIGHTS_DIR", REPO_ROOT / "weights"))

# Selected detector for traffic-state inference (YOLOv8, Stage 2 hybrid)
BEST_WEIGHTS = RUNS_DIR / "y8_stage2_mix" / "weights" / "best.pt"

# Spatiotemporal images for inference, one folder per day: IMAGES_DIR/<YYYYMMDD>/*.png
IMAGES_DIR = DATA_ROOT / "images"
NPY_DIR = DATA_ROOT / "npy"                  # same windows as .npy arrays (for slant stacking)
RESULTS_DIR = DATA_ROOT / "results"          # per-day detection results: RESULTS_DIR/<YYYYMMDD>/speeds.csv

# Fiber geometry (one KML point per DAS channel)
FIBER_KMZ = DATA_ROOT / "fiber_path_KMZ" / "fiber_path_vertices.kmz"

# -----------------------------------------------------------------------------
# DAS acquisition / image geometry
# -----------------------------------------------------------------------------
FS_HZ = 100.0            # sampling rate of the preprocessed arrays
WINDOW_SEC = 180.0       # duration of one spatiotemporal image
CHANNELS_PER_IMAGE = 50  # channels per image (~200 m)
CHANNEL_SPACING_M = 4.0

# Channel numbering and roadway segments
KMZ_FIRST_CHANNEL = 340              # channel number of the first KMZ vertex
SEG_START_CH, SEG_END_CH = 2400, 3400
SEG_SIZE = 50                        # c2400-2449 -> seg 0, c2450-2499 -> seg 1, ...

# -----------------------------------------------------------------------------
# Detection / training
# -----------------------------------------------------------------------------
IMG_SIZE = 1024
DEVICE = os.environ.get("DAS_DEVICE", "0")   # GPU index, or "cpu"

# -----------------------------------------------------------------------------
# Events (local time) and matched baselines
# -----------------------------------------------------------------------------
HOME_GAMEDAYS = ["2023-10-07", "2023-10-28", "2023-11-11", "2023-11-18"]
GAMEDAY = "2023-10-28"                                   # representative gameday (vs South Carolina)
GAMEDAY_BASELINE = ["2023-10-14", "2023-10-21", "2023-11-04"]   # non-gameday Saturdays
THANKSGIVING = ["2023-11-23", "2023-11-24", "2023-11-25", "2023-11-26"]
THANKSGIVING_BASELINE = [
    "2023-10-12", "2023-10-13", "2023-10-14", "2023-10-15",
    "2023-10-19", "2023-10-20", "2023-10-21", "2023-10-22",
    "2023-11-02", "2023-11-03", "2023-11-04", "2023-11-05",
]


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
