"""
config.py
Shared settings for all scripts. Change paths here only.
"""
from pathlib import Path

# Project folder (one level above src/)
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Folder that contains Bottle_A, Bottle_B, Bottle_C, Bottle_D
DATA_DIR = Path(r"D:\AI_Project\Bottle_dataset")
LABELS_CSV = DATA_DIR / "labels.csv"

# Smaller copies of the photos for fast training (created automatically).
# Kept OUTSIDE DATA_DIR so make_labels.py does not see it as a bottle folder.
CACHE_DIR = Path(r"D:\AI_Project\Bottle_dataset_cache")

# Professor's validation set: folder with the photos and labels.txt.
# Set to None to skip validation.
VAL_DIR = Path(r"D:\AI_Project\validation")

# Outputs inside the project
MODELS_DIR = PROJECT_DIR / "models"      # ignored by Git
RESULTS_DIR = PROJECT_DIR / "results"    # plots and CSVs, small, in Git