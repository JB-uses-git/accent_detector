"""
Stage 2 — Indian Sub-Accent Pipeline.

Downloads IndicAccentDb, maps accents to North/South/West,
processes audio, trains a 3-class Wav2Vec2 classifier.

Usage:
    python prepare_indian.py           # Prepare data + train
    python prepare_indian.py --prepare_only  # Just prepare data
"""

import argparse
import json
import logging
import os
import sys
from collections import Counter, defaultdict

import numpy as np
from datasets import Audio, Dataset, DatasetDict, Value, load_dataset, load_from_disk
from sklearn.model_selection import train_test_split
from transformers import (
    Trainer,
    TrainingArguments,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
)

from config import (
    BATCH_SIZE,
    CLIP_LENGTHS,
    INDIAN_ACCENT_DATASET,
    INDIAN_ACCENT_MAP,
    INDIAN_ID2LABEL,
    INDIAN_LABEL2ID,
    INDIAN_MODEL_OUTPUT_DIR,
    INDIAN_NUM_LABELS,
    INDIAN_SUB_LABELS,
    LEARNING_RATE,
    MODEL_NAME,
    NUM_EPOCHS,
    PROCESSED_DATA_DIR,
    SAMPLE_RATE,
    SEED,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
    WARMUP_RATIO,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load and process IndicAccentDb
# ═══════════════════════════════════════════════════════════════════════════════

def load_indian_data() -> Dataset:
    """Load IndicAccentDb and map to North/South/West regions."""
    logger.info("STEP 1 — Loading IndicAccentDb")

    ds = load_dataset(INDIAN_ACCENT_DATASET, split="train")
    logger.info("  Loaded %d samples", len(ds))
    logger.info("  Columns: %s", ds.column_names)

    # Find the label column
    label_col = None
    for candidate in ["label", "accent", "class"]:
        if candidate in ds.column_names:
            label_col = candidate
            break

    if label_col is None:
        raise ValueError(f"No label column found. Available: {ds.column_names}")

    logger.info("  Label column: '%s'", label_col)

    # Convert ClassLabel to string
    label_feature = ds.features[label_col]
    if hasattr(label_feature, "int2str"):
        label_names = label_feature.names
        logger.info("  ClassLabel names: %s", label_names)

        label_ints = ds[label_col]
        label_strings = [label_names[i] for i in label_ints]
        ds = ds.remove_columns([label_col])
        ds = ds.add_column("accent_raw", label_strings)
    else:
        ds = ds.rename_column(label_col, "accent_raw")

    # Normalize labels
    raw_labels = ds["accent_raw"]
    normalized = [s.strip().lower().replace(" ", "_") for s in raw_labels]
    ds = ds.remove_columns(["accent_raw"])
    ds = ds.add_column("accent_raw", normalized)

    unique_raw = set(normalized)
    logger.info("  Unique labels (normalized): %s", unique_raw)

    # Map to regions
    mapped = []
    unmapped = set()
    for raw in ds["accent_raw"]:
        if raw in INDIAN_ACCENT_MAP:
            mapped.append(INDIAN_ACCENT_MAP[raw])
        else:
            unmapped.add(raw)
            mapped.append(None)

    if unmapped:
        logger.warning("  Unmapped labels (dropped): %s", unmapped)

    ds = ds.remove_columns(["accent_raw"])
    ds = ds.add_column("accent", mapped)

    # Drop unmapped
    ds = ds.filter(lambda x: x["accent"] is not None)
    logger.info("  After mapping: %d samples", len(ds))

    # Keep only audio + accent
    columns_to_remove = [c for c in ds.column_names if c not in ["audio", "accent"]]
    if columns_to_remove:
        ds = ds.remove_columns(columns_to_remove)

    # Log distribution
    counts = Counter(ds["accent"])
    total = sum(counts.values())
    logger.info("  Distribution (%d total):", total)
    for label in INDIAN_SUB_LABELS:
        if label in counts:
            pct = 100.0 * counts[label] / total
            logger.info("    %-15s %6d  (%5.1f%%)", label, counts[label], pct)

    return ds


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Split and process
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_splits(ds: Dataset) -> dict:
    """Stratified split + clip-length processing for Indian sub-accents."""
    logger.info("STEP 2 — Stratified split (80/10/10)")

    # Resample
    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))

    labels = ds["accent"]
    indices = list(range(len(ds)))

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=(VAL_RATIO + TEST_RATIO),
        stratify=labels,
        random_state=SEED,
    )
    temp_labels = [labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        stratify=temp_labels,
        random_state=SEED,
    )

    split_dict = DatasetDict({
        "train": ds.select(train_idx),
        "val": ds.select(val_idx),
        "test": ds.select(test_idx),
    })
    logger.info("  Train: %d | Val: %d | Test: %d",
                len(split_dict["train"]), len(split_dict["val"]), len(split_dict["test"]))

    # Process clips
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    clip_datasets = {}

    for clip_len in CLIP_LENGTHS:
        max_samples = SAMPLE_RATE * clip_len
        logger.info("  Processing %ds clips...", clip_len)

        def preprocess_fn(batch, _max_samples=max_samples):
            audio_arrays = []
            for audio in batch["audio"]:
                arr = audio["array"]
                if len(arr) > _max_samples:
                    arr = arr[:_max_samples]
                elif len(arr) < _max_samples:
                    arr = np.pad(arr, (0, _max_samples - len(arr)), mode="constant")
                audio_arrays.append(arr)

            inputs = extractor(
                audio_arrays,
                sampling_rate=SAMPLE_RATE,
                max_length=_max_samples,
                truncation=True,
                padding="max_length",
                return_tensors="np",
            )
            inputs["labels"] = [INDIAN_LABEL2ID[a] for a in batch["accent"]]
            return inputs

        processed_dict = DatasetDict()
        for split_name, split_ds in split_dict.items():
            processed = split_ds.map(
                preprocess_fn,
                batched=True,
                batch_size=32,
                remove_columns=split_ds.column_names,
                num_proc=1,
            )
            processed.set_format("torch")
            processed_dict[split_name] = processed
            logger.info("    %s: %d samples", split_name, len(processed))

        save_path = os.path.join(PROCESSED_DATA_DIR, f"indian_clips_{clip_len}s")
        processed_dict.save_to_disk(save_path)
        logger.info("    Saved to %s", save_path)
        clip_datasets[clip_len] = processed_dict

    return clip_datasets


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Train
# ═══════════════════════════════════════════════════════════════════════════════

def train_model(clip_datasets: dict):
    """Train the Indian sub-accent classifier."""
    import torch
    from sklearn.metrics import f1_score

    for clip_len, ds_dict in clip_datasets.items():
        logger.info("STEP 3 — Training Indian sub-accent model (%ds clips)", clip_len)

        train_ds = ds_dict["train"]
        val_ds = ds_dict["val"]

        model = Wav2Vec2ForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=INDIAN_NUM_LABELS,
            label2id=INDIAN_LABEL2ID,
            id2label=INDIAN_ID2LABEL,
        )

        # Freeze CNN encoder
        model.freeze_feature_encoder()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info("  Trainable: %d / Total: %d", trainable, total)

        output_dir = os.path.join(INDIAN_MODEL_OUTPUT_DIR, f"clips_{clip_len}s")

        def compute_metrics(pred):
            preds = np.argmax(pred.predictions, axis=-1)
            labels = pred.label_ids
            macro_f1 = f1_score(labels, preds, average="macro")
            acc = (preds == labels).mean()
            return {"accuracy": acc, "macro_f1": macro_f1}

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE * 2,
            learning_rate=LEARNING_RATE,
            warmup_ratio=WARMUP_RATIO,
            lr_scheduler_type="cosine",
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            logging_steps=50,
            fp16=torch.cuda.is_available(),
            save_total_limit=2,
            seed=SEED,
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
        )

        logger.info("  Starting training 🚀")
        trainer.train()

        # Save best model
        trainer.save_model(output_dir)
        logger.info("  ✅ Model saved to %s", output_dir)

        # Quick test eval
        results = trainer.evaluate(ds_dict["test"])
        logger.info("  Test results: %s", results)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Prepare and train the Indian sub-accent classifier (Stage 2)."
    )
    parser.add_argument(
        "--prepare_only",
        action="store_true",
        help="Only prepare data, don't train.",
    )
    args = parser.parse_args()

    np.random.seed(SEED)

    # Load and process
    ds = load_indian_data()
    clip_datasets = prepare_splits(ds)

    if not args.prepare_only:
        train_model(clip_datasets)

    logger.info("=" * 60)
    logger.info("✅ Indian sub-accent pipeline complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
