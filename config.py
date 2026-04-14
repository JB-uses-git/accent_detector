"""
Centralized configuration for the Indian Accent Detector project.

All hyperparameters, paths, label mappings, and constants live here.
Every other module imports from this file — no magic numbers elsewhere.
"""

import os

# ─── Random Seed ──────────────────────────────────────────────────────────────
SEED = 42

# ─── Accent Labels (8-class) ─────────────────────────────────────────────────
ACCENT_LABELS = [
    "american", "british", "australian", "canadian",
    "indian_north", "indian_south", "indian_east", "indian_west",
]
NUM_LABELS = len(ACCENT_LABELS)
LABEL2ID = {label: idx for idx, label in enumerate(ACCENT_LABELS)}
ID2LABEL = {idx: label for idx, label in enumerate(ACCENT_LABELS)}

DISPLAY_LABELS = [
    "🇺🇸 American", "🇬🇧 British", "🇦🇺 Australian", "🇨🇦 Canadian",
    "🇮🇳 Indian-North", "🇮🇳 Indian-South", "🇮🇳 Indian-East", "🇮🇳 Indian-West",
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

# ─── Dataset Sources ─────────────────────────────────────────────────────────
COMMON_VOICE_DATASET = "mozilla-foundation/common_voice_13_0"
COMMON_VOICE_LANG = "en"
SVARAH_DATASET = "iitb-monolingual/svarah"

# ─── Common Voice Accent Mapping ─────────────────────────────────────────────
# Maps Common Voice accent tags to our label scheme
CV_ACCENT_MAP = {
    "us": "american",
    "england": "british",
    "australia": "australian",
    "canada": "canadian",
}
CV_TARGET_ACCENTS = list(CV_ACCENT_MAP.keys())

# ─── Svarah Region Mapping ───────────────────────────────────────────────────
# Maps Indian states/regions from Svarah dataset to our 4 sub-accent classes
SVARAH_REGION_MAP = {
    # North India
    "uttarakhand": "indian_north",
    "himachal": "indian_north",
    "himachal_pradesh": "indian_north",
    "punjab": "indian_north",
    "haryana": "indian_north",
    "delhi": "indian_north",
    "up": "indian_north",
    "uttar_pradesh": "indian_north",
    "rajasthan": "indian_north",
    "jammu_kashmir": "indian_north",
    "jammu": "indian_north",
    "kashmir": "indian_north",
    "chandigarh": "indian_north",
    "ladakh": "indian_north",
    # West India
    "gujarat": "indian_west",
    "maharashtra": "indian_west",
    "goa": "indian_west",
    "madhya_pradesh": "indian_west",
    "mp": "indian_west",
    "chhattisgarh": "indian_west",
    "daman": "indian_west",
    "dadra": "indian_west",
    # East India
    "west_bengal": "indian_east",
    "odisha": "indian_east",
    "assam": "indian_east",
    "bihar": "indian_east",
    "jharkhand": "indian_east",
    "northeast": "indian_east",
    "meghalaya": "indian_east",
    "manipur": "indian_east",
    "mizoram": "indian_east",
    "nagaland": "indian_east",
    "tripura": "indian_east",
    "arunachal_pradesh": "indian_east",
    "sikkim": "indian_east",
    # South India
    "tamil_nadu": "indian_south",
    "kerala": "indian_south",
    "karnataka": "indian_south",
    "andhra": "indian_south",
    "andhra_pradesh": "indian_south",
    "telangana": "indian_south",
    "pondicherry": "indian_south",
    "puducherry": "indian_south",
    "lakshadweep": "indian_south",
}

# ─── Dry Run ──────────────────────────────────────────────────────────────────
DRY_RUN_SAMPLES_PER_CLASS = 50

# ─── Split Ratios ────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
