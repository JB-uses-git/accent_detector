"""
Load pre-trained accent classifier models from disk (no retraining).

Use this when you already have trained model weights saved to disk
(e.g., copied from Google Drive) and want to run inference or evaluation
without re-running the full prepare + train pipeline.

Usage (Colab):
    # 1. Mount Drive and copy model
    from google.colab import drive
    drive.mount('/content/drive')
    !cp -r /content/drive/MyDrive/accent_detector_model /content/accent_detector/accent-classifier-final

    # 2. Run inference
    !python load_model.py --audio /path/to/audio.wav

    # Or import in a notebook:
    from load_model import load_stage1_model, load_stage2_model, predict_accent
"""

import argparse
import logging
import os
import sys

import librosa
import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

from config import (
    ACCENT_LABELS,
    DISPLAY_LABELS,
    INDIAN_DISPLAY_LABELS,
    INDIAN_MODEL_OUTPUT_DIR,
    INDIAN_SUB_LABELS,
    MODEL_NAME,
    MODEL_OUTPUT_DIR,
    SAMPLE_RATE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

CLIP_LENGTH = 3
MAX_SAMPLES = SAMPLE_RATE * CLIP_LENGTH
TEMPERATURE = 3.0


# ═══════════════════════════════════════════════════════════════════════════════
# Model Loading — uses saved weights, NO retraining
# ═══════════════════════════════════════════════════════════════════════════════

def load_extractor():
    """Load the Wav2Vec2 feature extractor (from HuggingFace cache, lightweight)."""
    logger.info("Loading Wav2Vec2 feature extractor...")
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    logger.info("  ✅ Feature extractor ready")
    return extractor


def load_stage1_model(model_dir: str = None):
    """
    Load the Stage 1 (global accent) model from a local directory.

    Args:
        model_dir: Path to the saved model directory.
                   Defaults to: accent-classifier-final/clips_3s/
    """
    if model_dir is None:
        model_dir = os.path.join(MODEL_OUTPUT_DIR, f"clips_{CLIP_LENGTH}s")

    if not os.path.exists(model_dir):
        logger.error("❌ Stage 1 model NOT found at: %s", model_dir)
        logger.error("   Make sure you copied the model from Google Drive:")
        logger.error("   !cp -r /content/drive/MyDrive/accent_detector_model %s", MODEL_OUTPUT_DIR)
        return None

    logger.info("Loading Stage 1 model from: %s", model_dir)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("  ✅ Stage 1 model loaded (%s parameters)", f"{total_params:,}")
    logger.info("  Labels: %s", ACCENT_LABELS)
    return model


def load_stage2_model(model_dir: str = None):
    """
    Load the Stage 2 (Indian sub-accent) model from a local directory.

    Args:
        model_dir: Path to the saved model directory.
                   Defaults to: indian-subaccent-classifier/clips_3s/
    """
    if model_dir is None:
        model_dir = os.path.join(INDIAN_MODEL_OUTPUT_DIR, f"clips_{CLIP_LENGTH}s")

    if not os.path.exists(model_dir):
        logger.warning("⚠️ Stage 2 model NOT found at: %s", model_dir)
        logger.warning("   Indian sub-accent detection will be skipped.")
        logger.warning("   Train with: python prepare_indian.py")
        return None

    logger.info("Loading Stage 2 model from: %s", model_dir)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("  ✅ Stage 2 model loaded (%s parameters)", f"{total_params:,}")
    logger.info("  Labels: %s", INDIAN_SUB_LABELS)
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Inference — predict accent from audio file
# ═══════════════════════════════════════════════════════════════════════════════

def predict_accent(
    audio_path: str,
    extractor: Wav2Vec2FeatureExtractor,
    stage1_model: Wav2Vec2ForSequenceClassification,
    stage2_model: Wav2Vec2ForSequenceClassification = None,
):
    """
    Run two-stage accent prediction on an audio file.

    Returns:
        dict: {label: probability} for all accent classes
        str:  status message
    """
    # Load audio
    audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    duration = len(audio) / SAMPLE_RATE
    logger.info("Audio loaded: %.1fs at %d Hz", duration, sr)

    if duration < 0.3:
        return {}, "⚠️ Audio too short — need at least 0.3 seconds."

    # Truncate / pad to clip length
    if len(audio) > MAX_SAMPLES:
        audio = audio[:MAX_SAMPLES]
    elif len(audio) < MAX_SAMPLES:
        audio = np.pad(audio, (0, MAX_SAMPLES - len(audio)), mode="constant")

    # Extract features
    inputs = extractor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding="max_length",
        max_length=MAX_SAMPLES,
        truncation=True,
    )

    # ── Stage 1: Global accent ──
    with torch.no_grad():
        logits = stage1_model(**inputs).logits
    probs = torch.softmax(logits / TEMPERATURE, dim=-1)[0]
    predicted_idx = probs.argmax().item()
    predicted_label = ACCENT_LABELS[predicted_idx]
    confidence = float(probs[predicted_idx])

    logger.info("Stage 1 prediction: %s (%.1f%%)", predicted_label, confidence * 100)

    # ── Stage 2: Indian sub-accent (if applicable) ──
    if predicted_label == "indian" and stage2_model is not None:
        with torch.no_grad():
            indian_logits = stage2_model(**inputs).logits
        indian_probs = torch.softmax(indian_logits / TEMPERATURE, dim=-1)[0]

        result = {}
        for i, label in enumerate(DISPLAY_LABELS):
            if ACCENT_LABELS[i] == "indian":
                indian_confidence = float(probs[i])
                for j, sub_label in enumerate(INDIAN_DISPLAY_LABELS):
                    result[sub_label] = indian_confidence * float(indian_probs[j])
            else:
                result[label] = float(probs[i])

        return result, f"🔍 Indian accent detected → sub-regional: {INDIAN_SUB_LABELS[indian_probs.argmax().item()]}"

    # No Stage 2 needed
    result = {DISPLAY_LABELS[i]: float(probs[i]) for i in range(len(DISPLAY_LABELS))}
    return result, f"Predicted: {DISPLAY_LABELS[predicted_idx]}"


# ═══════════════════════════════════════════════════════════════════════════════
# Main — CLI usage
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Load pre-trained accent models and run inference."
    )
    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Path to an audio file to classify.",
    )
    parser.add_argument(
        "--stage1_dir",
        type=str,
        default=None,
        help="Custom path to Stage 1 model directory.",
    )
    parser.add_argument(
        "--stage2_dir",
        type=str,
        default=None,
        help="Custom path to Stage 2 model directory.",
    )
    parser.add_argument(
        "--verify_only",
        action="store_true",
        help="Just verify models can be loaded, don't run inference.",
    )
    args = parser.parse_args()

    # Load models
    extractor = load_extractor()
    stage1 = load_stage1_model(args.stage1_dir)
    stage2 = load_stage2_model(args.stage2_dir)

    if stage1 is None:
        logger.error("Cannot proceed without Stage 1 model.")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("✅ Model loading complete!")
    logger.info("  Stage 1 (global accent): READY")
    logger.info("  Stage 2 (Indian sub):    %s", "READY" if stage2 else "NOT AVAILABLE")
    logger.info("=" * 60)

    if args.verify_only:
        return

    if args.audio:
        results, status = predict_accent(args.audio, extractor, stage1, stage2)
        logger.info("Status: %s", status)
        logger.info("Results:")
        for label, prob in sorted(results.items(), key=lambda x: -x[1]):
            logger.info("  %-25s %.1f%%", label, prob * 100)
    else:
        logger.info("No audio file provided. Use --audio to classify a file.")
        logger.info("Or import in Python:")
        logger.info("  from load_model import load_stage1_model, predict_accent")


if __name__ == "__main__":
    main()
