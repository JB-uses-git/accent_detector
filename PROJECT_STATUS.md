# Project Status & Working

---

## What It Does

An **8-class English accent classifier** with hierarchical Indian sub-accent detection. Takes a short audio clip (1s, 2s, or 3s) and predicts the speaker's accent.

### Supported Accents (8 classes)
🇺🇸 American · 🇬🇧 British · 🇦🇺 Australian · 🇨🇦 Canadian  
🇮🇳 Indian-North · 🇮🇳 Indian-South · 🇮🇳 Indian-East · 🇮🇳 Indian-West

---

## Research Differentiators

1. **Short-clip benchmarking** — accuracy curves across 1s, 2s, and 3s clips (nobody in literature does this)
2. **Hierarchical Indian sub-accents** — North/South/East/West using Svarah dataset
3. **Standardized evaluation** — per-class F1, confusion matrices, clip-length curves on one reproducible split

---

## How It Works

### The Model
- **Base:** `facebook/wav2vec2-base` (95M parameters)
- **Approach:** Transfer learning — CNN feature encoder frozen, Transformer + 8-class head fine-tuned
- **Training:** AdamW, cosine scheduler with 10% warmup, FP16, macro F1 for best model selection

### The Datasets
- **Mozilla Common Voice 13.0** (English) — American, British, Australian, Canadian accents
- **Svarah** (IIT Bombay) — Indian regional accents mapped to North/South/East/West
- 80/10/10 stratified split with reproducible manifest CSV

---

## Workflow

### Step 1 — Data Preparation (`prepare_data.py`)
Downloads Common Voice + Svarah, maps accents, performs stratified split, creates 3 clip-length variants (1s/2s/3s), saves manifest.

### Step 2 — Training (`train.py`)
Fine-tunes Wav2Vec2 for each clip length with frozen CNN encoder. Selects best model by macro F1.

### Step 3 — Evaluation (`evaluate.py`)
Generates per-class CSVs, normalized confusion matrix PNGs, overall metrics CSV, clip-length accuracy curve PNG, and baseline comparison.

### Step 4 — Gradio Demo (`app.py`)
Web UI with microphone/upload input, clip-length selector, and full 8-class probability output.

---

## File Structure

```
accent_detector/
├── config.py              # All hyperparameters, labels, paths
├── prepare_data.py        # Data download + processing pipeline
├── train.py               # Model training (per clip length)
├── evaluate.py            # Full evaluation pipeline
├── app.py                 # Gradio web demo
├── requirements.txt       # Python dependencies
├── README.md              # Setup instructions + results
├── notebooks/
│   └── train_colab.ipynb  # Self-contained Colab/Kaggle notebook
├── samples/               # Example audio for Gradio demo
└── results/               # Evaluation outputs (CSVs + PNGs)
```

---

## How to Run

```bash
pip install -r requirements.txt
python prepare_data.py          # ~1-2 hours (downloads datasets)
python train.py --all           # ~2-3 hours on GPU
python evaluate.py              # ~10 minutes
python app.py --share           # Launch demo
```

GPU required for training. Use `notebooks/train_colab.ipynb` on Kaggle P100 or Colab T4.

Quick test: add `--dry_run` to any script to run on 50 samples per class.
