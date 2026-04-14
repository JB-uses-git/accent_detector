# Project Status & Working

---

## What It Does

A **7-class English accent classifier** with hierarchical Indian sub-accent detection. Takes a short audio clip (1s, 2s, or 3s) and predicts the speaker's accent.

### Supported Accents (7 classes)
🇺🇸 American · 🇬🇧 British · 🇨🇦 Canadian  
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
- **Approach:** Transfer learning — CNN feature encoder frozen, Transformer + 7-class head fine-tuned
- **Training:** AdamW, cosine scheduler with 10% warmup, FP16, macro F1 for best model selection

### The Datasets
- **Westbrook English Accent Dataset** (79 hrs, 53K samples) — American, British, Canadian accents
- **Svarah** (IIT Bombay) — Indian regional accents mapped to North/South/East/West
- 80/10/10 stratified split with reproducible manifest CSV

---

## How to Run

```bash
pip install -r requirements.txt
python prepare_data.py          # ~30-60 min (downloads datasets)
python train.py --all           # ~2-3 hours on GPU
python evaluate.py              # ~10 minutes
python app.py --share           # Launch demo
```

GPU required for training. Use `notebooks/train_colab.ipynb` on Kaggle P100 or Colab T4.

Quick test: add `--dry_run` to any script to run on 50 samples per class.
