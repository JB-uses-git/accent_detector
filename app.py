"""
Phase 5 — Gradio Demo for Indian Accent Detection.

Provides a web UI with:
  - Audio input (microphone or file upload)
  - Clip length selector (1s / 2s / 3s)
  - Top prediction with confidence
  - All 8 class probabilities
  - Note about accuracy at selected clip length

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
_extractor = None


def get_extractor():
    """Get or load the feature extractor (shared across all clip lengths)."""
    global _extractor
    if _extractor is None:
        logger.info("Loading feature extractor from %s...", MODEL_NAME)
        _extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    return _extractor


def get_model(clip_length: int):
    """Get or load a model for a specific clip length (cached)."""
    if clip_length not in _model_cache:
        model_path = os.path.join(MODEL_OUTPUT_DIR, f"clips_{clip_length}s")
        if not os.path.exists(model_path):
            logger.error("Model not found at %s", model_path)
            return None
        logger.info("Loading model from %s...", model_path)
        model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
        model.eval()
        _model_cache[clip_length] = model
    return _model_cache[clip_length]


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
    """Classify accent from audio file."""
    if audio_path is None:
        return (
            {label: 0.0 for label in DISPLAY_LABELS},
            "⚠️ No audio provided. Please record or upload audio."
        )

    # Parse clip length
    clip_length = int(clip_length_str.replace("s", ""))
    max_samples = SAMPLE_RATE * clip_length

    # Load model
    model = get_model(clip_length)
    if model is None:
        return (
            {label: 0.0 for label in DISPLAY_LABELS},
            f"❌ Model for {clip_length}s clips not found. Train the model first."
        )

    extractor = get_extractor()

    # Load and resample audio to 16 kHz
    try:
        audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    except Exception as e:
        return (
            {label: 0.0 for label in DISPLAY_LABELS},
            f"❌ Error loading audio: {str(e)}"
        )

    # Reject very short clips (< 0.3s)
    if len(audio) < SAMPLE_RATE * 0.3:
        return (
            {label: 0.0 for label in DISPLAY_LABELS},
            "⚠️ Audio too short. Please provide at least 0.3 seconds."
        )

    # Truncate or pad to selected clip length
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

    # Inference
    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1)[0]
    result = {DISPLAY_LABELS[i]: float(probs[i]) for i in range(len(DISPLAY_LABELS))}

    # Accuracy note
    note = get_accuracy_note(clip_length)

    return result, note


def build_demo(share: bool = False):
    """Build and launch the Gradio interface."""

    # Clip length choices
    clip_choices = [f"{cl}s" for cl in CLIP_LENGTHS]

    # Check for sample audio files
    examples = []
    if os.path.exists(SAMPLES_DIR):
        for f in sorted(os.listdir(SAMPLES_DIR)):
            if f.endswith((".wav", ".mp3", ".flac", ".ogg")):
                examples.append([os.path.join(SAMPLES_DIR, f), "3s"])

    with gr.Blocks(
        title="🎙️ Indian Accent Detector",
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="blue",
            neutral_hue="slate",
        ),
    ) as demo:
        gr.Markdown(
            """
            # 🎙️ Indian Accent Detector

            An 8-class English accent classifier with **hierarchical Indian sub-accent detection**.

            ### 🔬 Research Differentiators
            1. **Short-clip benchmarking**: accuracy curves across 1s, 2s, and 3s clips — 
               nobody in the literature benchmarks on clips this short
            2. **Hierarchical Indian sub-accents**: North / South / East / West Indian classification 
               using the Svarah dataset
            3. **Standardized evaluation**: per-class F1, confusion matrices, and clip-length curves 
               on a single reproducible public split

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
                    num_top_classes=8,
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

        # Examples (if sample files exist)
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
            **Supported accents**: 🇺🇸 American · 🇬🇧 British · 🇦🇺 Australian · 🇨🇦 Canadian · 
            🇮🇳 Indian-North · 🇮🇳 Indian-South · 🇮🇳 Indian-East · 🇮🇳 Indian-West

            *Powered by [Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base) · 
            Trained on [Common Voice](https://commonvoice.mozilla.org/) + 
            [Svarah](https://huggingface.co/datasets/iitb-monolingual/svarah)*
            """
        )

    demo.launch(share=share)


def main():
    parser = argparse.ArgumentParser(
        description="Launch Gradio demo for Indian Accent Detection."
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio share link.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Launch the demo without loading models (for UI testing).",
    )
    args = parser.parse_args()

    if args.dry_run:
        logger.info("🏃 DRY RUN — launching demo (models may not be loaded)")

    build_demo(share=args.share)


if __name__ == "__main__":
    main()
