"""
Centralized configuration for the Accent Detector project.
All hyperparameters, paths, and label mappings live here.
"""

# ─── Model ────────────────────────────────────────────────────────────────────
BASE_MODEL = "facebook/wav2vec2-base"
# For better multilingual/Indian accent performance, swap to:
# BASE_MODEL = "facebook/wav2vec2-large-xlsr-53"

# ─── Labels ───────────────────────────────────────────────────────────────────
ACCENT_LABELS = ["us", "england", "indian", "australia", "canada"]
LABEL2ID = {label: idx for idx, label in enumerate(ACCENT_LABELS)}
ID2LABEL = {idx: label for idx, label in enumerate(ACCENT_LABELS)}
NUM_LABELS = len(ACCENT_LABELS)

DISPLAY_LABELS = [
    "🇺🇸 American",
    "🇬🇧 British",
    "🇮🇳 Indian",
    "🇦🇺 Australian",
    "🇨🇦 Canadian",
]

# ─── Audio ────────────────────────────────────────────────────────────────────
SAMPLING_RATE = 16_000          # Wav2Vec2 requires 16 kHz
MAX_LENGTH_SAMPLES = 48_000     # 3 seconds of audio at 16 kHz
ORIGINAL_SR = 48_000            # Common Voice native sample rate

# ─── Dataset ──────────────────────────────────────────────────────────────────
DATASET_NAME = "mozilla-foundation/common_voice_13_0"
DATASET_LANG = "en"
TRAIN_SPLIT = "train"
VALIDATION_SPLIT = "validation"
TEST_SPLIT = "test"

# ─── Training ─────────────────────────────────────────────────────────────────
OUTPUT_DIR = "accent-classifier"
FINAL_MODEL_DIR = "accent-classifier-final"
LEARNING_RATE = 3e-5
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
NUM_EPOCHS = 5
WARMUP_RATIO = 0.1
FP16 = True                    # Use mixed precision on GPU (set False for CPU)

# ─── Svarah (Stage 2 — Indian regional accents) ──────────────────────────────
SVARAH_DATASET = "ai4bharat/Svarah"
INDIAN_REGIONAL_LABELS = ["north", "south", "east", "west"]
