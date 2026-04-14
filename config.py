"""
Centralized configuration for the Indian Accent Detector project.

All hyperparameters, paths, label mappings, and constants live here.
Every other module imports from this file — no magic numbers elsewhere.
"""

import os

# ─── Random Seed ──────────────────────────────────────────────────────────────
SEED = 42

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — Global Accent Classifier (4-class)
# ═══════════════════════════════════════════════════════════════════════════════

ACCENT_LABELS = [
    "american", "british", "canadian", "indian",
]
NUM_LABELS = len(ACCENT_LABELS)
LABEL2ID = {label: idx for idx, label in enumerate(ACCENT_LABELS)}
ID2LABEL = {idx: label for idx, label in enumerate(ACCENT_LABELS)}

DISPLAY_LABELS = [
    "🇺🇸 American", "🇬🇧 British", "🇨🇦 Canadian", "🇮🇳 Indian",
]

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — Indian Sub-Accent Classifier (3-class)
# ═══════════════════════════════════════════════════════════════════════════════

INDIAN_SUB_LABELS = [
    "indian_north", "indian_south", "indian_west",
]
INDIAN_NUM_LABELS = len(INDIAN_SUB_LABELS)
INDIAN_LABEL2ID = {label: idx for idx, label in enumerate(INDIAN_SUB_LABELS)}
INDIAN_ID2LABEL = {idx: label for idx, label in enumerate(INDIAN_SUB_LABELS)}

INDIAN_DISPLAY_LABELS = [
    "🇮🇳 North (Hindi Belt)", "🇮🇳 South (Dravidian)", "🇮🇳 West (Gujarati)",
]

# ─── Clip Lengths ─────────────────────────────────────────────────────────────
CLIP_LENGTHS = [3]  # seconds

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
INDIAN_MODEL_OUTPUT_DIR = "indian-subaccent-classifier"
RESULTS_DIR = "results"
SAMPLES_DIR = "samples"

# ─── Dataset Sources ──────────────────────────────────────────────────────────
# Westbrook English Accent Dataset (79 hrs, 53K samples, Parquet, free)
ACCENT_DATASET = "westbrook/English_Accent_DataSet"
# IndicAccentDb (8K samples, 3.2 GB, Parquet, free — Indian sub-accents)
INDIAN_ACCENT_DATASET = "DarshanaS/IndicAccentDb"

# ─── Westbrook Accent Mapping (Stage 1) ──────────────────────────────────────
ACCENT_MAP = {
    "American": "american",
    "English": "british",
    "Canadian": "canadian",
    "Indian": "indian",
}
TARGET_ACCENTS = list(ACCENT_MAP.keys())

# ─── IndicAccentDb Region Mapping (Stage 2) ──────────────────────────────────
# Maps native language/state labels → Indian sub-regions
INDIAN_ACCENT_MAP = {
    # By language name
    "hindi": "indian_north",
    "tamil": "indian_south",
    "telugu": "indian_south",
    "kannada": "indian_south",
    "malayalam": "indian_south",
    "gujarati": "indian_west",
    # By state name (in case dataset uses these)
    "andhra_pradesh": "indian_south",
    "tamil_nadu": "indian_south",
    "karnataka": "indian_south",
    "kerala": "indian_south",
    "gujarat": "indian_west",
    "uttar_pradesh": "indian_north",
    "delhi": "indian_north",
    "rajasthan": "indian_north",
    "maharashtra": "indian_west",
}

# ─── Dry Run ──────────────────────────────────────────────────────────────────
DRY_RUN_SAMPLES_PER_CLASS = 50

# ─── Split Ratios ────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
