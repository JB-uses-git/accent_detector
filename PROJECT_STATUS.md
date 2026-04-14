# Project Status & Working

---

## What It Does

An **English accent classifier** that takes a short audio clip of someone speaking English and predicts their accent. Supports 5 accents:

🇺🇸 American · 🇬🇧 British · 🇮🇳 Indian · 🇦🇺 Australian · 🇨🇦 Canadian

---

## How It Works

### The Model

- **Base:** `facebook/wav2vec2-base` (95M parameters)
- **Approach:** Transfer learning — the CNN feature encoder (~7M params) is frozen, only the Transformer layers + classification head (~88M params) are fine-tuned
- Freezing the CNN prevents catastrophic forgetting since it already knows how to extract audio features from pretraining

### The Dataset

- **Mozilla Common Voice 13.0** (English subset)
- Filtered to only keep clips tagged with one of the 5 target accents
- Audio resampled from 48kHz → 16kHz (what Wav2Vec2 expects)
- Clips truncated/padded to 3 seconds

---

## Workflow

### Step 1 — Data Preparation (`prepare_data.py`)

Downloads Mozilla Common Voice English, loads train/validation/test splits, filters to the 5 target accents, resamples audio to 16kHz, extracts features using `Wav2Vec2FeatureExtractor`, and saves the processed dataset to `processed_data/`.

### Step 2 — Training (`train.py`)

Loads the processed dataset, initializes `Wav2Vec2ForSequenceClassification` with a 5-class head, freezes the CNN encoder, and trains using HuggingFace Trainer with:
- Learning rate `3e-5` with 10% warmup
- Batch size 16, FP16 mixed precision on GPU
- 5 epochs, saves best model by accuracy

Outputs the final model to `accent-classifier-final/`.

### Step 3 — Evaluation (`evaluate_model.py`)

Loads the trained model and test split, runs inference, and prints overall accuracy, per-class precision/recall/F1, and a confusion matrix.

### Step 4 — Gradio Frontend (`app.py`)

Loads the trained model and launches a web UI where users can record from their microphone or upload an audio file. The app resamples the audio, runs it through the model, and displays accent probabilities for all 5 classes.

---

## File Structure

```
accent_detector/
├── config.py              # All hyperparameters, labels, paths (everything imports from here)
├── prepare_data.py        # Downloads + processes dataset → processed_data/
├── train.py               # Fine-tunes model → accent-classifier-final/
├── evaluate_model.py      # Evaluates on test set, prints metrics
├── app.py                 # Gradio web UI for live inference
├── notebook_train.ipynb   # Self-contained Kaggle/Colab notebook (runs full pipeline)
├── requirements.txt       # Python dependencies
└── README.md              # Setup instructions
```

---

## How to Run

```bash
pip install -r requirements.txt
python prepare_data.py
python train.py
python evaluate_model.py
python app.py
```

Training needs a GPU. Use `notebook_train.ipynb` on Kaggle (P100) or Colab (T4) for the easiest experience.
