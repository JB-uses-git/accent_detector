"""
Phase 5 — Gradio Demo for Indian Accent Detection (Two-Stage).

Stage 1: Classify as American / British / Canadian / Indian
Stage 2: If Indian → classify as North / South / West

Usage:
    python app.py [--share] [--dry_run]
"""

import argparse
import json
import logging
import os
import sys

import gradio as gr
import librosa
import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

from config import (
    ACCENT_LABELS,
    CLIP_LENGTHS,
    DISPLAY_LABELS,
    INDIAN_DISPLAY_LABELS,
    INDIAN_MODEL_OUTPUT_DIR,
    INDIAN_SUB_LABELS,
    MODEL_NAME,
    MODEL_OUTPUT_DIR,
    RESULTS_DIR,
    SAMPLE_RATE,
    SAMPLES_DIR,
)

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ─── Model Cache ──────────────────────────────────────────────────────────────
_model_cache = {}
_indian_model_cache = {}
_extractor = None


def get_extractor():
    """Get or load the feature extractor (shared across all models)."""
    global _extractor
    if _extractor is None:
        logger.info("Loading feature extractor from %s...", MODEL_NAME)
        _extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    return _extractor


def get_model(clip_length: int):
    """Get or load Stage 1 model (global accent classifier)."""
    if clip_length not in _model_cache:
        model_path = os.path.join(MODEL_OUTPUT_DIR, f"clips_{clip_length}s")
        if not os.path.exists(model_path):
            logger.error("Stage 1 model not found at %s", model_path)
            return None
        logger.info("Loading Stage 1 model from %s...", model_path)
        model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
        model.eval()
        _model_cache[clip_length] = model
    return _model_cache[clip_length]


def get_indian_model(clip_length: int):
    """Get or load Stage 2 model (Indian sub-accent classifier)."""
    if clip_length not in _indian_model_cache:
        model_path = os.path.join(INDIAN_MODEL_OUTPUT_DIR, f"clips_{clip_length}s")
        if not os.path.exists(model_path):
            logger.warning("Stage 2 model not found at %s — sub-accents unavailable", model_path)
            return None
        logger.info("Loading Stage 2 model from %s...", model_path)
        model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
        model.eval()
        _indian_model_cache[clip_length] = model
    return _indian_model_cache[clip_length]


def get_accuracy_note(clip_length: int) -> str:
    """Load accuracy from evaluation results if available."""
    overall_csv = os.path.join(RESULTS_DIR, "overall_metrics.csv")
    if os.path.exists(overall_csv):
        import pandas as pd
        df = pd.read_csv(overall_csv)
        row = df[df["clip_length"] == clip_length]
        if len(row) > 0:
            f1 = row.iloc[0]["macro_f1"]
            acc = row.iloc[0]["accuracy"]
            return f"📊 Accuracy at {clip_length}s: {acc*100:.1f}% accuracy, {f1*100:.1f}% macro F1"
    return f"📊 Accuracy at {clip_length}s: not yet evaluated (run evaluate.py)"


def classify_accent(audio_path: str, clip_length_str: str):
    """Two-stage accent classification."""
    if audio_path is None:
        return (
            {label: 0.0 for label in DISPLAY_LABELS},
            "⚠️ No audio provided. Please record or upload audio."
        )

    clip_length = int(clip_length_str.replace("s", ""))
    max_samples = SAMPLE_RATE * clip_length

    # Load Stage 1 model
    model = get_model(clip_length)
    if model is None:
        return (
            {label: 0.0 for label in DISPLAY_LABELS},
            f"❌ Stage 1 model for {clip_length}s clips not found. Train the model first."
        )

    extractor = get_extractor()

    # Load and resample audio
    try:
        audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    except Exception as e:
        return (
            {label: 0.0 for label in DISPLAY_LABELS},
            f"❌ Error loading audio: {str(e)}"
        )

    if len(audio) < SAMPLE_RATE * 0.3:
        return (
            {label: 0.0 for label in DISPLAY_LABELS},
            "⚠️ Audio too short. Please provide at least 0.3 seconds."
        )

    # Truncate or pad
    if len(audio) > max_samples:
        audio = audio[:max_samples]
    elif len(audio) < max_samples:
        audio = np.pad(audio, (0, max_samples - len(audio)), mode="constant")

    # Extract features
    inputs = extractor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding="max_length",
        max_length=max_samples,
        truncation=True,
    )

    # ── Stage 1: Global classification ──
    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1)[0]
    predicted_idx = probs.argmax().item()
    predicted_label = ACCENT_LABELS[predicted_idx]

    # ── Stage 2: If Indian, run sub-accent classifier ──
    sub_result = None
    if predicted_label == "indian":
        indian_model = get_indian_model(clip_length)
        if indian_model is not None:
            with torch.no_grad():
                indian_logits = indian_model(**inputs).logits
            indian_probs = torch.softmax(indian_logits, dim=-1)[0]

            # Build combined result with Indian sub-accents
            result = {}
            for i, label in enumerate(DISPLAY_LABELS):
                if ACCENT_LABELS[i] == "indian":
                    # Replace single "Indian" with sub-accent breakdown
                    indian_confidence = float(probs[i])
                    for j, sub_label in enumerate(INDIAN_DISPLAY_LABELS):
                        result[sub_label] = indian_confidence * float(indian_probs[j])
                else:
                    result[label] = float(probs[i])

            note = get_accuracy_note(clip_length)
            note += "\n🔍 Indian accent detected → Sub-regional classification applied"
            return result, note

    # No Indian detected or no Stage 2 model — return Stage 1 results
    result = {DISPLAY_LABELS[i]: float(probs[i]) for i in range(len(DISPLAY_LABELS))}
    note = get_accuracy_note(clip_length)
    return result, note


def build_demo(share: bool = False):
    """Build and launch the Gradio interface."""

    clip_choices = [f"{cl}s" for cl in CLIP_LENGTHS]

    examples = []
    if os.path.exists(SAMPLES_DIR):
        for f in sorted(os.listdir(SAMPLES_DIR)):
            if f.endswith((".wav", ".mp3", ".flac", ".ogg")):
                examples.append([os.path.join(SAMPLES_DIR, f), "3s"])

    # Check which models are available
    has_stage2 = os.path.exists(os.path.join(INDIAN_MODEL_OUTPUT_DIR, "clips_3s"))

    with gr.Blocks(
        title="🎙️ Indian Accent Detector",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="orange",
        ),
    ) as demo:
        gr.Markdown(
            f"""
            # 🎙️ Indian Accent Detector

            A **two-stage** English accent classifier with **hierarchical Indian sub-accent detection**.

            ### How It Works
            1. **Stage 1**: Classifies accent as 🇺🇸 American · 🇬🇧 British · 🇨🇦 Canadian · 🇮🇳 Indian
            2. **Stage 2**: If Indian detected → further classifies as North (Hindi Belt) · South (Dravidian) · West (Gujarati)

            {"✅ Both stages loaded!" if has_stage2 else "⚠️ Stage 2 (Indian sub-accents) not trained yet. Run `python prepare_indian.py`"}

            ---
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="🎤 Record or Upload Audio",
                )
                clip_length_dropdown = gr.Dropdown(
                    choices=clip_choices,
                    value="3s",
                    label="⏱️ Clip Length for Inference",
                    info="Select how much audio to use for classification",
                )
                submit_btn = gr.Button(
                    "🔍 Classify Accent",
                    variant="primary",
                    size="lg",
                )
                gr.Markdown(
                    """
                    > **Tips:**
                    > - Speak naturally for a few seconds
                    > - Read a sentence aloud for best results
                    > - Minimize background noise
                    > - Longer clips (3s) tend to be more accurate
                    """
                )

            with gr.Column(scale=1):
                output_label = gr.Label(
                    num_top_classes=6,
                    label="🌍 Accent Prediction",
                )
                accuracy_note = gr.Textbox(
                    label="📈 Model Info",
                    interactive=False,
                )

        submit_btn.click(
            fn=classify_accent,
            inputs=[audio_input, clip_length_dropdown],
            outputs=[output_label, accuracy_note],
        )

        if examples:
            gr.Examples(
                examples=examples,
                inputs=[audio_input, clip_length_dropdown],
                outputs=[output_label, accuracy_note],
                fn=classify_accent,
                cache_examples=False,
            )

        gr.Markdown(
            """
            ---
            **Stage 1**: 🇺🇸 American · 🇬🇧 British · 🇨🇦 Canadian · 🇮🇳 Indian  
            **Stage 2** (if Indian): 🇮🇳 North (Hindi Belt) · 🇮🇳 South (Dravidian) · 🇮🇳 West (Gujarati)

            *Powered by [Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base) · 
            Trained on [Westbrook](https://huggingface.co/datasets/westbrook/English_Accent_DataSet) + 
            [IndicAccentDb](https://huggingface.co/datasets/DarshanaS/IndicAccentDb)*
            """
        )

    demo.launch(share=share)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Create public link")
    args = parser.parse_args()
    build_demo(share=args.share)
