"""
Phase 5 — Gradio frontend for live accent classification.

Usage:
    python app.py [--model_dir accent-classifier-final] [--share]

Speak into your mic for 3–5 seconds and get accent predictions.
"""

import argparse
import torch
import librosa
import gradio as gr
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
from config import BASE_MODEL, DISPLAY_LABELS, SAMPLING_RATE


def load_model(model_dir: str):
    """Load the fine-tuned model and feature extractor."""
    print(f"Loading model from '{model_dir}'...")
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_dir)
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(BASE_MODEL)
    model.eval()
    print("  Model loaded ✅")
    return model, extractor


def classify_accent(audio_path: str, model, extractor) -> dict:
    """Classify the accent from an audio file path."""
    if audio_path is None:
        return {label: 0.0 for label in DISPLAY_LABELS}

    # Load and resample to 16 kHz
    audio, sr = librosa.load(audio_path, sr=SAMPLING_RATE)

    # Reject very short clips (< 0.5s)
    if len(audio) < SAMPLING_RATE * 0.5:
        return {label: 0.0 for label in DISPLAY_LABELS}

    # Extract features
    inputs = extractor(
        audio,
        sampling_rate=SAMPLING_RATE,
        return_tensors="pt",
        padding=True,
    )

    # Inference
    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1)[0]
    return {DISPLAY_LABELS[i]: float(probs[i]) for i in range(len(DISPLAY_LABELS))}


def build_demo(model, extractor, share: bool = False):
    """Build and launch the Gradio interface."""

    def predict(audio_path):
        return classify_accent(audio_path, model, extractor)

    with gr.Blocks(
        title="🎙️ Accent Classifier",
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="blue",
            neutral_hue="slate",
        ),
    ) as demo:
        gr.Markdown(
            """
            # 🎙️ English Accent Classifier
            
            Speak into your microphone for **3–5 seconds** and the model will predict 
            which English accent you have.
            
            Supported accents: **American 🇺🇸 · British 🇬🇧 · Indian 🇮🇳 · Australian 🇦🇺 · Canadian 🇨🇦**
            
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
                submit_btn = gr.Button("🔍 Classify Accent", variant="primary", size="lg")
                gr.Markdown(
                    """
                    > **Tips:**
                    > - Speak naturally for 3–5 seconds
                    > - Read a sentence aloud for best results
                    > - Minimize background noise
                    """
                )

            with gr.Column(scale=1):
                output_label = gr.Label(
                    num_top_classes=5,
                    label="🌍 Accent Prediction",
                )

        submit_btn.click(
            fn=predict,
            inputs=[audio_input],
            outputs=[output_label],
        )

        gr.Markdown(
            """
            ---
            *Powered by [Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base) · 
            Fine-tuned on [Mozilla Common Voice](https://commonvoice.mozilla.org/)*
            """
        )

    demo.launch(share=share)


def main():
    parser = argparse.ArgumentParser(description="Launch Gradio accent classifier.")
    parser.add_argument("--model_dir", type=str, default="accent-classifier-final", help="Path to trained model.")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link.")
    args = parser.parse_args()

    model, extractor = load_model(args.model_dir)
    build_demo(model, extractor, share=args.share)


if __name__ == "__main__":
    main()
