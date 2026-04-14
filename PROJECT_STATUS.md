# Project Status — Indian Accent Detector

## ✅ COMPLETE

### Architecture
Two-stage cascaded Wav2Vec2 classifier:
- **Stage 1**: 4-class global accent (American, British, Canadian, Indian) — **99.66% accuracy**
- **Stage 2**: 3-class Indian sub-accent (North, South, West) — **100% accuracy**

### Datasets
| Dataset | Samples | Access | Used For |
|---------|---------|--------|----------|
| Westbrook English Accent Dataset | 26,206 (filtered) | Open | Stage 1 |
| IndicAccentDb | 8,116 | Open | Stage 2 |

### Pipeline
- [x] `prepare_data.py` — Stage 1 data pipeline
- [x] `train.py` — Stage 1 training
- [x] `evaluate.py` — Stage 1 evaluation
- [x] `prepare_indian.py` — Stage 2 data + training pipeline
- [x] `app.py` — Two-stage Gradio demo

### Models
- Stage 1: `accent-classifier-final/clips_3s/`
- Stage 2: `indian-subaccent-classifier/clips_3s/`
- Both saved to Google Drive for persistence

### Research Differentiators
1. **Two-stage hierarchical classification** — global accent first, then Indian sub-regional
2. **Indian sub-accent detection** — North (Hindi Belt) / South (Dravidian) / West (Gujarati)
3. **Open datasets only** — no gated access required, fully reproducible
4. **Per-class F1 evaluation** — not just overall accuracy
