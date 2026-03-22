"""
Project-wide configuration for Hurricane Damage Detector.
All hyperparameters, paths, and constants in one place.
"""

from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DIR = DATA_DIR / "train_another"
VALIDATION_DIR = DATA_DIR / "validation_another"
TEST_DIR = DATA_DIR / "test_another"
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_DIR = PROJECT_ROOT / "saved_model"

# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
IMG_DIMS = (128, 128)
IMG_SHAPE = IMG_DIMS + (3,)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2  # Only used if VALIDATION_DIR is missing
CLASS_NAMES = ["no_damage", "damage"]
SEED = 42

# ──────────────────────────────────────────────
# Data source (Kaggle)
# ──────────────────────────────────────────────
KAGGLE_DATASET = "kmader/satellite-images-of-hurricane-damage"

# ──────────────────────────────────────────────
# Training — Phase 1 (frozen base)
# ──────────────────────────────────────────────
PHASE1_EPOCHS = 10
PHASE1_LEARNING_RATE = 1e-3

# ──────────────────────────────────────────────
# Training — Phase 2 (fine-tuning conv5 block)
# ──────────────────────────────────────────────
PHASE2_EPOCHS = 10
PHASE2_LEARNING_RATE = 1e-5
EARLY_STOPPING_PATIENCE = 3
REDUCE_LR_PATIENCE = 2
REDUCE_LR_FACTOR = 0.5

# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────
CLASSIFICATION_THRESHOLD = 0.5
TEST_BATCH_SIZE = 128

# ──────────────────────────────────────────────
# Grad-CAM
# ──────────────────────────────────────────────
GRADCAM_LAYER_NAME = "conv5_block3_out"  # Last conv layer in ResNet50
GRADCAM_NUM_SAMPLES = 8  # Number of sample images to visualize
