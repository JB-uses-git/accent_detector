"""
Phase 3 — Model Training.

Fine-tunes Wav2Vec2ForSequenceClassification for each clip length (1s, 2s, 3s).
Uses macro F1 as the metric for best model selection (handles class imbalance).

Usage:
    python train.py [--clip_length 3] [--dry_run]
    python train.py --all                  # Train for all clip lengths
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
import torch
from datasets import load_from_disk
from sklearn.metrics import f1_score
from transformers import (
    Trainer,
    TrainingArguments,
    Wav2Vec2ForSequenceClassification,
)

from config import (
    ACCENT_LABELS,
    BATCH_SIZE,
    CLIP_LENGTHS,
    ID2LABEL,
    LABEL2ID,
    LEARNING_RATE,
    MODEL_NAME,
    MODEL_OUTPUT_DIR,
    NUM_EPOCHS,
    NUM_LABELS,
    PROCESSED_DATA_DIR,
    SEED,
    WARMUP_RATIO,
)

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model() -> Wav2Vec2ForSequenceClassification:
    """Load Wav2Vec2 with a classification head and freeze the CNN feature encoder."""
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        label2id=LABEL2ID,
        id2label=ID2LABEL,
    )
    # Freeze the CNN feature encoder — only fine-tune transformer + classifier
    model.wav2vec2.feature_extractor._freeze_parameters()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("Model loaded: %s", MODEL_NAME)
    logger.info("  Trainable params: %s", f"{trainable:,}")
    logger.info("  Total params:     %s", f"{total:,}")
    return model


def get_compute_metrics():
    """Return a compute_metrics function that reports macro F1 and accuracy."""

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        macro_f1 = f1_score(labels, predictions, average="macro", zero_division=0)
        accuracy = np.mean(predictions == labels)
        return {
            "macro_f1": macro_f1,
            "accuracy": accuracy,
        }

    return compute_metrics


def train_for_clip_length(clip_length: int, dry_run: bool = False):
    """Train a model for a specific clip length."""
    logger.info("=" * 60)
    logger.info("Training for %ds clips", clip_length)
    logger.info("=" * 60)

    # Load processed dataset
    data_path = os.path.join(PROCESSED_DATA_DIR, f"clips_{clip_length}s")
    if not os.path.exists(data_path):
        logger.error("Processed data not found at %s. Run prepare_data.py first.", data_path)
        return

    ds = load_from_disk(data_path)
    logger.info("  Train: %d samples", len(ds["train"]))
    logger.info("  Val:   %d samples", len(ds["val"]))

    if dry_run:
        # Use a small subset for dry run
        max_samples = 50
        ds["train"] = ds["train"].select(range(min(max_samples, len(ds["train"]))))
        ds["val"] = ds["val"].select(range(min(max_samples, len(ds["val"]))))
        logger.info("  🏃 DRY RUN — capped to %d samples", max_samples)

    # Build model
    model = build_model()

    # Output directory
    output_dir = os.path.join(MODEL_OUTPUT_DIR, f"clips_{clip_length}s")
    checkpoint_dir = os.path.join("accent-classifier-checkpoints", f"clips_{clip_length}s")

    # Determine FP16
    use_fp16 = torch.cuda.is_available()

    # Training arguments
    num_epochs = 1 if dry_run else NUM_EPOCHS

    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=num_epochs,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        fp16=use_fp16,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        push_to_hub=False,
        save_total_limit=2,
        dataloader_num_workers=2,
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["val"],
        compute_metrics=get_compute_metrics(),
    )

    logger.info("  FP16: %s", use_fp16)
    logger.info("  Epochs: %d", num_epochs)
    logger.info("  Batch size: %d", BATCH_SIZE)
    logger.info("  Learning rate: %s", LEARNING_RATE)
    logger.info("  Scheduler: cosine with warmup (ratio=%s)", WARMUP_RATIO)
    logger.info("  Best model metric: macro_f1")

    # Train
    logger.info("Starting training 🚀")
    train_result = trainer.train()

    # Evaluate on validation
    logger.info("Evaluating on validation set...")
    val_metrics = trainer.evaluate()
    logger.info("  Val macro F1:  %.4f", val_metrics.get("eval_macro_f1", 0))
    logger.info("  Val accuracy:  %.4f", val_metrics.get("eval_accuracy", 0))
    logger.info("  Val loss:      %.4f", val_metrics.get("eval_loss", 0))

    # Save best model
    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)
    logger.info("  Saved model to %s", output_dir)

    # Save training log
    training_log = {
        "clip_length": clip_length,
        "num_epochs": num_epochs,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "train_loss": train_result.training_loss,
        "val_metrics": {
            "macro_f1": val_metrics.get("eval_macro_f1", 0),
            "accuracy": val_metrics.get("eval_accuracy", 0),
            "loss": val_metrics.get("eval_loss", 0),
        },
        "train_samples": len(ds["train"]),
        "val_samples": len(ds["val"]),
    }

    log_path = os.path.join(MODEL_OUTPUT_DIR, f"training_log_{clip_length}s.json")
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)
    logger.info("  Saved training log to %s", log_path)

    logger.info("✅ Training complete for %ds clips", clip_length)
    return training_log


def main():
    parser = argparse.ArgumentParser(
        description="Train accent classifier for specific clip lengths."
    )
    parser.add_argument(
        "--clip_length",
        type=int,
        choices=CLIP_LENGTHS,
        default=None,
        help="Clip length to train for (1, 2, or 3 seconds).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Train for all clip lengths (1s, 2s, 3s).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run on a small subset for fast testing.",
    )
    args = parser.parse_args()

    set_seed(SEED)

    if args.all:
        clip_lengths = CLIP_LENGTHS
    elif args.clip_length is not None:
        clip_lengths = [args.clip_length]
    else:
        # Default: train all
        clip_lengths = CLIP_LENGTHS

    logger.info("Will train for clip lengths: %s", clip_lengths)

    all_logs = {}
    for clip_len in clip_lengths:
        log = train_for_clip_length(clip_len, dry_run=args.dry_run)
        if log:
            all_logs[f"{clip_len}s"] = log

    # Save combined training summary
    summary_path = os.path.join(MODEL_OUTPUT_DIR, "training_summary.json")
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_logs, f, indent=2)
    logger.info("Saved training summary to %s", summary_path)

    logger.info("=" * 60)
    logger.info("✅ All training complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
