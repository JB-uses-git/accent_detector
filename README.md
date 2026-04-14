# 🎙️ Indian Accent Detector

A 7-class English accent classifier with **hierarchical Indian sub-accent detection** and **short-clip benchmarking** — built on Wav2Vec2, trained on Westbrook English Accent Dataset + Svarah.

---

## Research Gaps This Closes

### 1. No Short-Clip Benchmarking
Prior work evaluates only on clips ≥ 5 seconds. Real-world applications often have only 1–3 seconds of speech. We benchmark accuracy across **1s, 2s, and 3s clips** and publish the degradation curve.

### 2. No Indian Sub-Accent Classification
All existing accent classifiers group Indian English into a single "Indian" category. We use the **Svarah dataset** to classify into 4 Indian sub-regions:
- 🇮🇳 **Indian-North**: Uttarakhand, Himachal, Punjab, Haryana, Delhi, UP, Rajasthan
- 🇮🇳 **Indian-South**: Tamil Nadu, Kerala, Karnataka, Andhra, Telangana
- 🇮🇳 **Indian-East**: West Bengal, Odisha, Assam, Bihar, Jharkhand, Northeast
- 🇮🇳 **Indian-West**: Gujarat, Maharashtra, Goa, Madhya Pradesh, Chhattisgarh

### 3. No Standardized Multi-Metric Evaluation
Prior papers report only overall accuracy. We provide per-class F1, normalized confusion matrices, and clip-length curves on a single reproducible public split.

---

## Model Architecture

- **Base model**: `facebook/wav2vec2-base` (95M parameters)
- **Approach**: Transfer learning — CNN feature encoder frozen, Transformer + 7-class head fine-tuned
- **Training**: AdamW optimizer, cosine scheduler with 10% warmup, FP16 on GPU
- **Best model selection**: Macro F1 (not accuracy, to handle class imbalance)

```
Wav2Vec2 Feature Encoder (FROZEN)
        ↓
Wav2Vec2 Transformer Encoder (FINE-TUNED)
        ↓
Mean Pooling
        ↓
7-Class Classification Head (FINE-TUNED)
        ↓
Softmax → Accent Prediction
```

---

## Dataset

| Source | Classes | Samples | Purpose |
|--------|---------|---------|---------|
| [Westbrook English Accent Dataset](https://huggingface.co/datasets/westbrook/English_Accent_DataSet) | American, British, Canadian | ~20K+ | Global accents |
| [Svarah](https://huggingface.co/datasets/iitb-monolingual/svarah) | Indian-North, Indian-South, Indian-East, Indian-West | ~10K+ | Indian sub-accents |

- Split: 80% train / 10% validation / 10% test (stratified by accent)
- Reproducible split manifest: `processed_data/split_manifest.csv`

> **Note:** Mozilla Common Voice was removed from HuggingFace in October 2025. We use the Westbrook English Accent Dataset (79 hrs, 53K total samples from VCTK + EDACC + Voxpopuli) as the primary source for global accents.

---

## Results

### Overall Metrics by Clip Length

| Clip Length | Accuracy | Macro F1 | Weighted F1 |
|:-----------:|:--------:|:--------:|:-----------:|
| 1s          |    __    |    __    |     __      |
| 2s          |    __    |    __    |     __      |
| 3s          |    __    |    __    |     __      |

*Run `python evaluate.py` after training to populate.*

### Per-Class F1 by Clip Length

| Accent       | F1 (1s) | F1 (2s) | F1 (3s) |
|:-------------|:-------:|:-------:|:-------:|
| American     |   __    |   __    |   __    |
| British      |   __    |   __    |   __    |
| Canadian     |   __    |   __    |   __    |
| Indian-North |   __    |   __    |   __    |
| Indian-South |   __    |   __    |   __    |
| Indian-East  |   __    |   __    |   __    |
| Indian-West  |   __    |   __    |   __    |

### Confusion Matrix (3s)

![Confusion Matrix](results/confusion_matrix_3s.png)

### Clip-Length Accuracy Curve

![Clip Length Curve](results/clip_length_curve.png)

---

## Quickstart

```bash
pip install -r requirements.txt
python prepare_data.py
python train.py --all
python evaluate.py
python app.py --share
```

For a quick test run:
```bash
python prepare_data.py --dry_run
python train.py --all --dry_run
python evaluate.py --dry_run
```

---

## Full Pipeline

### Step 1 — Data Preparation (`prepare_data.py`)
Downloads Westbrook + Svarah, maps accents, performs stratified 80/10/10 split, creates 3 clip-length variants (1s/2s/3s), saves a reproducible manifest.

### Step 2 — Training (`train.py`)
Fine-tunes Wav2Vec2 with frozen CNN encoder for each clip length. Uses macro F1 for model selection.

### Step 3 — Evaluation (`evaluate.py`)
Generates all research artifacts: per-class CSVs, confusion matrix PNGs, clip-length curves, and baseline comparison.

### Step 4 — Demo (`app.py`)
Gradio web UI with microphone/upload support and clip-length selector.

---

## File Structure

```
accent_detector/
├── config.py                  # All hyperparameters, labels, paths
├── prepare_data.py            # Data download + processing pipeline
├── train.py                   # Model training (per clip length)
├── evaluate.py                # Full evaluation pipeline
├── app.py                     # Gradio web demo
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── notebooks/
│   └── train_colab.ipynb      # Self-contained Colab/Kaggle notebook
├── processed_data/            # Generated by prepare_data.py
├── accent-classifier-final/   # Generated by train.py
├── results/                   # Generated by evaluate.py
└── samples/                   # Example audio for Gradio demo
```

---

## Citation / Acknowledgements

### Datasets
- **Westbrook English Accent Dataset**: 79-hour dataset from VCTK, EDACC, and Voxpopuli.
- **Svarah**: IIT Bombay. Indian accented English speech data.

### Models
- **Wav2Vec2**: Baevski, A., et al. (2020). "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations." NeurIPS 2020.

### Tools
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [Gradio](https://gradio.app/)
- [PyTorch](https://pytorch.org/)
