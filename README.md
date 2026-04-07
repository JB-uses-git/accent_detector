# 🎙️ English Accent Classifier

Classify English accents from speech using **Wav2Vec2** fine-tuned on **Mozilla Common Voice**.

**Supported Accents:** 🇺🇸 American · 🇬🇧 British · 🇮🇳 Indian · 🇦🇺 Australian · 🇨🇦 Canadian

---

## Project Structure

```
accent_detector/
├── config.py              # All hyperparameters & label mappings
├── prepare_data.py        # Phase 1-2: Load, filter, preprocess data
├── train.py               # Phase 3-4: Fine-tune Wav2Vec2
├── evaluate_model.py      # Detailed evaluation with confusion matrix
├── app.py                 # Phase 5: Gradio frontend (mic + upload)
├── notebook_train.ipynb   # Self-contained Kaggle/Colab notebook
├── requirements.txt       # Python dependencies
└── .gitignore
```

## Quick Start

### Option A — Kaggle/Colab Notebook (Recommended)

1. Upload `notebook_train.ipynb` to [Kaggle](https://www.kaggle.com/) or [Google Colab](https://colab.research.google.com/)
2. Enable **GPU** (Kaggle: P100 / Colab: T4)
3. Run all cells top-to-bottom
4. The Gradio demo launches in the last cell with a public URL

### Option B — Local Scripts

#### 1. Install dependencies
```bash
pip install -r requirements.txt
```

#### 2. Prepare data
```bash
# Full dataset (takes a while to download)
python prepare_data.py

# Quick test with limited samples
python prepare_data.py --max_train_samples 500 --max_eval_samples 100
```

#### 3. Train
```bash
# Full training (needs GPU, ~2-3 hours)
python train.py

# CPU mode (no mixed precision)
python train.py --no_fp16 --batch_size 4 --epochs 1
```

#### 4. Evaluate
```bash
python evaluate_model.py
```

#### 5. Run Gradio frontend
```bash
# Local only
python app.py

# With public share link
python app.py --share
```

---

## Architecture

```
Raw Audio (48kHz) → Resample (16kHz) → Wav2Vec2 Feature Extractor
    → Wav2Vec2 Transformer (frozen CNN encoder) → Classification Head → 5 accent classes
```

- **Base model:** `facebook/wav2vec2-base` (95M params)
- **Frozen:** CNN feature encoder (~7M params) — prevents catastrophic forgetting
- **Trainable:** Transformer layers + classification head (~88M params)

## Expected Results

| Metric | Value |
|--------|-------|
| Validation Accuracy | 80–90% |
| Training Time (P100) | ~2–3 hours |
| Epochs | 5 |

## Optional Upgrades

- **Stronger model:** Replace `wav2vec2-base` with `facebook/wav2vec2-large-xlsr-53` in `config.py` for better multilingual accent detection
- **Two-stage pipeline:** If classified as Indian → pass to a second model fine-tuned on [Svarah](https://huggingface.co/datasets/ai4bharat/Svarah) for North/South/East/West Indian sub-classification
- **Deploy:** Use `python app.py --share` for instant public URL, or push to HuggingFace Spaces

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Model | Wav2Vec2 (HuggingFace Transformers) |
| Dataset | Mozilla Common Voice 13.0 |
| Training | HuggingFace Trainer API |
| Frontend | Gradio |
| Runtime | Kaggle P100 / Colab T4 |
