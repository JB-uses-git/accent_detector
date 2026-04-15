"""
Phase 5 — Gradio Demo for Indian Accent Detection (Two-Stage).

Stage 1: Classify as American / British / Canadian / Indian
Stage 2: If Indian → classify as North / South / West

Usage:
    python app.py [--share]
"""

import argparse
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
    DISPLAY_LABELS,
    INDIAN_DISPLAY_LABELS,
    INDIAN_MODEL_OUTPUT_DIR,
    INDIAN_SUB_LABELS,
    MODEL_NAME,
    MODEL_OUTPUT_DIR,
    RESULTS_DIR,
    SAMPLE_RATE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Directory where app.py lives
CLIP_LENGTH = 3  # Always use 3s clips
MAX_SAMPLES = SAMPLE_RATE * CLIP_LENGTH
TEMPERATURE = 8.0  # Softens predictions to look more natural

# ─── Model Cache ──────────────────────────────────────────────────────────────
_stage1_model = None
_stage2_model = None
_extractor = None


def get_extractor():
    global _extractor
    if _extractor is None:
        _extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    return _extractor


def get_stage1_model():
    global _stage1_model
    if _stage1_model is None:
        model_path = os.path.join(BASE_DIR, MODEL_OUTPUT_DIR, f"clips_{CLIP_LENGTH}s")
        logger.info("Stage 1 model path: %s (exists: %s)", model_path, os.path.exists(model_path))
        if not os.path.exists(model_path):
            return None
        _stage1_model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
        _stage1_model.eval()
        logger.info("✅ Stage 1 model loaded!")
    return _stage1_model


def get_stage2_model():
    global _stage2_model
    if _stage2_model is None:
        model_path = os.path.join(BASE_DIR, INDIAN_MODEL_OUTPUT_DIR, f"clips_{CLIP_LENGTH}s")
        logger.info("Stage 2 model path: %s (exists: %s)", model_path, os.path.exists(model_path))
        if not os.path.exists(model_path):
            logger.warning("⚠️ Stage 2 model NOT found — Indian sub-accent classification disabled")
            return None
        _stage2_model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
        _stage2_model.eval()
        logger.info("✅ Stage 2 model loaded!")
    return _stage2_model


def classify_accent(audio_path: str):
    """Two-stage accent classification."""
    if audio_path is None:
        return {label: 0.0 for label in DISPLAY_LABELS}, ""

    model = get_stage1_model()
    if model is None:
        return {label: 0.0 for label in DISPLAY_LABELS}, "❌ Model not found. Train the model first."

    extractor = get_extractor()

    try:
        audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    except Exception as e:
        return {label: 0.0 for label in DISPLAY_LABELS}, f"❌ Error: {str(e)}"

    if len(audio) < SAMPLE_RATE * 0.3:
        return {label: 0.0 for label in DISPLAY_LABELS}, "⚠️ Audio too short — need at least 0.3 seconds."

    # Truncate or pad to 3s
    if len(audio) > MAX_SAMPLES:
        audio = audio[:MAX_SAMPLES]
    elif len(audio) < MAX_SAMPLES:
        audio = np.pad(audio, (0, MAX_SAMPLES - len(audio)), mode="constant")

    inputs = extractor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding="max_length",
        max_length=MAX_SAMPLES,
        truncation=True,
    )

    # ── Stage 1 ──
    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits / TEMPERATURE, dim=-1)[0]
    predicted_idx = probs.argmax().item()
    predicted_label = ACCENT_LABELS[predicted_idx]

    # ── Stage 2 ──
    if predicted_label == "indian":
        indian_model = get_stage2_model()
        if indian_model is not None:
            with torch.no_grad():
                indian_logits = indian_model(**inputs).logits
            indian_probs = torch.softmax(indian_logits / TEMPERATURE, dim=-1)[0]

            result = {}
            for i, label in enumerate(DISPLAY_LABELS):
                if ACCENT_LABELS[i] == "indian":
                    indian_confidence = float(probs[i])
                    for j, sub_label in enumerate(INDIAN_DISPLAY_LABELS):
                        result[sub_label] = indian_confidence * float(indian_probs[j])
                else:
                    result[label] = float(probs[i])

            return result, "🔍 Indian accent detected → sub-regional classification applied"

    result = {DISPLAY_LABELS[i]: float(probs[i]) for i in range(len(DISPLAY_LABELS))}
    return result, ""


# ─── Custom CSS ───────────────────────────────────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

* {
    font-family: 'Inter', sans-serif !important;
}

.gradio-container {
    max-width: 800px !important;
    margin: auto !important;
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%) !important;
    min-height: 100vh;
}

.main-header {
    text-align: center;
    padding: 2rem 1rem;
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(236, 72, 153, 0.15));
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
    margin-bottom: 1.5rem;
}

.main-header h1 {
    background: linear-gradient(135deg, #818cf8, #c084fc, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.2rem !important;
    font-weight: 700 !important;
    margin-bottom: 0.5rem !important;
}

.main-header p {
    color: rgba(255, 255, 255, 0.7) !important;
    font-size: 1rem !important;
}

.card {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
    backdrop-filter: blur(10px) !important;
}

footer { display: none !important; }

.accent-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.85rem;
    margin: 3px;
    background: rgba(99, 102, 241, 0.2);
    border: 1px solid rgba(99, 102, 241, 0.3);
    color: #c4b5fd;
}

button.primary {
    background: linear-gradient(135deg, #6366f1, #8b5cf6, #a855f7) !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 1.05rem !important;
    padding: 12px 24px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
}

button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 25px rgba(99, 102, 241, 0.5) !important;
}
"""


def build_demo(share: bool = False):
    """Build and launch the Gradio interface."""

    has_stage1 = os.path.exists(os.path.join(BASE_DIR, MODEL_OUTPUT_DIR, f"clips_{CLIP_LENGTH}s"))
    has_stage2 = os.path.exists(os.path.join(BASE_DIR, INDIAN_MODEL_OUTPUT_DIR, f"clips_{CLIP_LENGTH}s"))

    with gr.Blocks(
        title="Indian Accent Detector",
        css=CUSTOM_CSS,
        theme=gr.themes.Base(
            primary_hue="indigo",
            secondary_hue="purple",
            neutral_hue="slate",
            font=("Inter", "sans-serif"),
        ).set(
            body_background_fill="linear-gradient(135deg, #0f0c29, #302b63, #24243e)",
            body_text_color="rgba(255,255,255,0.9)",
            block_background_fill="rgba(255,255,255,0.05)",
            block_border_color="rgba(255,255,255,0.1)",
            block_label_text_color="rgba(255,255,255,0.7)",
            input_background_fill="rgba(255,255,255,0.08)",
            input_border_color="rgba(255,255,255,0.15)",
        ),
    ) as demo:

        # ── Header ──
        gr.HTML("""
        <div class="main-header">
            <h1>🎙️ Accent Detector</h1>
            <p>Two-stage AI that detects global accents and drills into Indian sub-regions</p>
        </div>
        """)

        with gr.Row(equal_height=True):
            # ── Left: Input ──
            with gr.Column(scale=1):
                gr.Markdown("### 🎤 Record or Upload")
                audio_input = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="Audio Input",
                    show_label=False,
                )
                submit_btn = gr.Button(
                    "✨ Detect Accent",
                    variant="primary",
                    size="lg",
                )
                gr.Markdown(
                    """
                    <div style="color: rgba(255,255,255,0.5); font-size: 0.85rem; margin-top: 8px;">
                    💡 Speak naturally for 3+ seconds for best results
                    </div>
                    """,
                )

            # ── Right: Output ──
            with gr.Column(scale=1):
                gr.Markdown("### 🌍 Prediction")
                output_label = gr.Label(
                    num_top_classes=6,
                    label="Accent Probabilities",
                    show_label=False,
                )
                status_text = gr.Textbox(
                    label="Status",
                    interactive=False,
                    show_label=False,
                    max_lines=1,
                )

        submit_btn.click(
            fn=classify_accent,
            inputs=[audio_input],
            outputs=[output_label, status_text],
        )

        # ── Footer ──
        gr.HTML(f"""
        <div style="text-align:center; padding: 1.5rem 0; margin-top: 1rem; border-top: 1px solid rgba(255,255,255,0.08);">
            <div style="margin-bottom: 8px;">
                <span class="accent-badge">🇺🇸 American</span>
                <span class="accent-badge">🇬🇧 British</span>
                <span class="accent-badge">🇨🇦 Canadian</span>
                <span class="accent-badge">🇮🇳 Indian</span>
            </div>
            <div style="margin-bottom: 12px;">
                <span class="accent-badge" style="background:rgba(236,72,153,0.15); border-color:rgba(236,72,153,0.3); color:#f9a8d4;">
                    North · Hindi Belt</span>
                <span class="accent-badge" style="background:rgba(236,72,153,0.15); border-color:rgba(236,72,153,0.3); color:#f9a8d4;">
                    South · Dravidian</span>
                <span class="accent-badge" style="background:rgba(236,72,153,0.15); border-color:rgba(236,72,153,0.3); color:#f9a8d4;">
                    West · Gujarati</span>
            </div>
            <p style="color: rgba(255,255,255,0.35); font-size: 0.8rem;">
                Powered by Wav2Vec2 · Stage 1 {"✅" if has_stage1 else "❌"} · Stage 2 {"✅" if has_stage2 else "❌"}
            </p>
        </div>
        """)

    demo.launch(share=share)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Create public link")
    args = parser.parse_args()
    build_demo(share=args.share)
