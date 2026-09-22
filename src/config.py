"""
Central configuration for the F1 Virtual Race Engineer project.

Every script imports its paths, split definitions and hyper-parameters from
here so that all three models are trained and evaluated under identical
conditions -- which is what makes the Review 3 comparison fair.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

CACHE_DIR = PROJECT_ROOT / "cache"            # FastF1 on-disk cache (gitignored)
DATA_RAW = PROJECT_ROOT / "data" / "raw"      # one parquet per session
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"   # model-ready tensors
MODELS_DIR = PROJECT_ROOT / "models"          # trained weights
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"

ALL_DIRS = [CACHE_DIR, DATA_RAW, DATA_PROCESSED, MODELS_DIR, FIGURES_DIR, METRICS_DIR]


def ensure_dirs() -> None:
    """Create every project directory if it does not already exist."""
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Dataset scope  (as specified in the project proposal)
# --------------------------------------------------------------------------
SEASONS = [2021, 2022, 2023, 2024]

# Split is by SEASON/RACE, never by random row: two laps from the same race are
# highly correlated, so a random split would leak the test set into training and
# inflate the reported accuracy.
TRAIN_SEASONS = [2021, 2022]
VAL_SEASONS = [2023]
TEST_SEASONS = [2024]

SESSION_TYPE = "R"   # Race sessions only

# --------------------------------------------------------------------------
# Label definition
# --------------------------------------------------------------------------
# A driver pits roughly 2-3 times in a ~55 lap race, so "pits on this exact lap"
# is only ~5% positive -- a model that always predicts "no pit" would score 95%
# accuracy while being useless. Widening the target to "pits within the next
# PIT_HORIZON laps" raises the positive rate to a trainable ~12-15%.
PIT_HORIZON = 3

# --------------------------------------------------------------------------
# Sequence construction
# --------------------------------------------------------------------------
SEQ_LEN = 10          # number of past laps fed to the model at each prediction
MIN_LAPS_PER_DRIVER = SEQ_LEN + PIT_HORIZON

# --------------------------------------------------------------------------
# Training hyper-parameters (shared by all three models)
# --------------------------------------------------------------------------
RANDOM_SEED = 42
BATCH_SIZE = 256
EPOCHS = 60
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 10
DECISION_THRESHOLD = 0.50   # tuned on the validation set in Phase 3


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Seed every RNG so all three model runs are reproducible."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass
