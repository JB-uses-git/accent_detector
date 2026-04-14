"""
Centralized configuration for the Indian Accent Detector project.

All hyperparameters, paths, label mappings, and constants live here.
Every other module imports from this file — no magic numbers elsewhere.
"""

import os

# ─── Random Seed ──────────────────────────────────────────────────────────────
SEED = 42

# ─── Accent Labels (4-class) ─────────────────────────────────────────────────
ACCENT_LABELS = [
    "american", "british", "canadian", "indian",
]
NUM_LABELS = len(ACCENT_LABELS)
LABEL2ID = {label: idx for idx, label in enumerate(ACCENT_LABELS)}
ID2LABEL = {idx: label for idx, label in enumerate(ACCENT_LABELS)}

DISPLAY_LABELS = [
    "🇺🇸 American", "🇬🇧 British", "🇨🇦 Canadian", "🇮🇳 Indian",
]

# ─── Clip Lengths ─────────────────────────────────────────────────────────────
CLIP_LENGTHS = [1, 2, 3]  # seconds

# ─── Audio ────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000  # Wav2Vec2 requires 16 kHz

# ─── Model ────────────────────────────────────────────────────────────────────
MODEL_NAME = "facebook/wav2vec2-base"

# ─── Training Hyperparameters ─────────────────────────────────────────────────
BATCH_SIZE = 16
LEARNING_RATE = 3e-5
NUM_EPOCHS = 5
WARMUP_RATIO = 0.1

# ─── Paths ────────────────────────────────────────────────────────────────────
PROCESSED_DATA_DIR = "processed_data"
MODEL_OUTPUT_DIR = "accent-classifier-final"
RESULTS_DIR = "results"
SAMPLES_DIR = "samples"

# ─── Dataset Source ───────────────────────────────────────────────────────────
# Westbrook English Accent Dataset (79 hrs, 53K samples, Parquet, free)
ACCENT_DATASET = "westbrook/English_Accent_DataSet"

# ─── Accent Mapping ──────────────────────────────────────────────────────────
# Maps Westbrook ClassLabel names to our label scheme
ACCENT_MAP = {
    "American": "american",
    "English": "british",
    "Canadian": "canadian",
    "Indian": "indian",
}
TARGET_ACCENTS = list(ACCENT_MAP.keys())

# ─── Dry Run ──────────────────────────────────────────────────────────────────
DRY_RUN_SAMPLES_PER_CLASS = 50

# ─── Split Ratios ────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
