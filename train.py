"""
Phase 3 & 4 — Fine-tune Wav2Vec2 for accent classification.

Usage:
    python train.py [--data_dir processed_data] [--output_dir accent-classifier]

Expects the processed dataset from prepare_data.py.
"""

import argparse
import numpy as np
import evaluate
from datasets import load_from_disk
from transformers import (
    Wav2Vec2ForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from config import (
    BASE_MODEL,
    NUM_LABELS,
    LABEL2ID,
    ID2LABEL,
    OUTPUT_DIR,
    FINAL_MODEL_DIR,
    LEARNING_RATE,
    TRAIN_BATCH_SIZE,
    EVAL_BATCH_SIZE,
    NUM_EPOCHS,
    WARMUP_RATIO,
    FP16,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
)


def build_model():
    """Load Wav2Vec2 with a classification head and freeze the feature encoder."""
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=NUM_LABELS,
        label2id=LABEL2ID,
        id2label=ID2LABEL,
    )
    # Freeze the CNN feature encoder — only train transformer + classifier
    model.freeze_feature_encoder()
    print(f"  Model loaded: {BASE_MODEL}")
    print(f"  Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"  Total params:     {sum(p.numel() for p in model.parameters()):,}")
    return model


def get_compute_metrics():
    """Return a compute_metrics function for the Trainer."""
    accuracy_metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return accuracy_metric.compute(predictions=predictions, references=labels)

    return compute_metrics


def main():
    parser = argparse.ArgumentParser(description="Train accent classifier.")
    parser.add_argument("--data_dir", type=str, default="processed_data", help="Path to processed dataset.")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="Checkpoint output directory.")
    parser.add_argument("--final_model_dir", type=str, default=FINAL_MODEL_DIR, help="Where to save the final model.")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=TRAIN_BATCH_SIZE, help="Batch size.")
    parser.add_argument("--no_fp16", action="store_true", help="Disable mixed-precision training.")
    args = parser.parse_args()

    print("=" * 60)
    print("Loading processed dataset")
    print("=" * 60)
    ds = load_from_disk(args.data_dir)
    print(f"  Train:      {len(ds[TRAIN_SPLIT])} samples")
    print(f"  Validation: {len(ds[VALIDATION_SPLIT])} samples")

    print("\n" + "=" * 60)
    print("Building model")
    print("=" * 60)
    model = build_model()

    print("\n" + "=" * 60)
    print("Setting up training")
    print("=" * 60)

    use_fp16 = FP16 and not args.no_fp16

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=args.epochs,
        warmup_ratio=WARMUP_RATIO,
        fp16=use_fp16,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",         # Disable W&B / MLflow unless explicitly configured
        push_to_hub=False,
        save_total_limit=2,       # Keep only 2 best checkpoints to save disk
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds[TRAIN_SPLIT],
        eval_dataset=ds[VALIDATION_SPLIT],
        compute_metrics=get_compute_metrics(),
    )

    print(f"  FP16: {use_fp16}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")

    print("\n" + "=" * 60)
    print("Starting training 🚀")
    print("=" * 60)
    trainer.train()

    print("\n" + "=" * 60)
    print("Evaluating on validation set")
    print("=" * 60)
    metrics = trainer.evaluate()
    print(f"  Validation accuracy: {metrics.get('eval_accuracy', 'N/A'):.4f}")

    print("\n" + "=" * 60)
    print(f"Saving final model to '{args.final_model_dir}'")
    print("=" * 60)
    trainer.save_model(args.final_model_dir)
    print("  Done! ✅\n")


if __name__ == "__main__":
    main()
