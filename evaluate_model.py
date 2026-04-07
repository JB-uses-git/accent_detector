"""
Evaluate a trained accent classifier on the test set.

Usage:
    python evaluate_model.py [--model_dir accent-classifier-final] [--data_dir processed_data]
"""

import argparse
import numpy as np
import torch
from datasets import load_from_disk
from transformers import Wav2Vec2ForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
from config import (
    ACCENT_LABELS,
    DISPLAY_LABELS,
    TEST_SPLIT,
    FINAL_MODEL_DIR,
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate accent classifier.")
    parser.add_argument("--model_dir", type=str, default=FINAL_MODEL_DIR)
    parser.add_argument("--data_dir", type=str, default="processed_data")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    print("Loading model...")
    model = Wav2Vec2ForSequenceClassification.from_pretrained(args.model_dir)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("Loading test data...")
    ds = load_from_disk(args.data_dir)
    test_ds = ds[TEST_SPLIT]

    all_preds = []
    all_labels = []

    print("Running inference...")
    for i in range(0, len(test_ds), args.batch_size):
        batch = test_ds[i:i + args.batch_size]
        input_values = batch["input_values"].to(device)
        labels = batch["labels"]

        with torch.no_grad():
            logits = model(input_values=input_values).logits

        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy() if hasattr(labels, 'numpy') else labels)

        if (i // args.batch_size) % 10 == 0:
            print(f"  Processed {min(i + args.batch_size, len(test_ds))}/{len(test_ds)}")

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Accuracy
    accuracy = np.mean(all_preds == all_labels)
    print(f"\n{'=' * 60}")
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"{'=' * 60}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        all_labels, all_preds,
        target_names=DISPLAY_LABELS,
        digits=4,
    ))

    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(f"{'':>15}", end="")
    for label in ACCENT_LABELS:
        print(f"{label:>10}", end="")
    print()
    for i, row in enumerate(cm):
        print(f"{ACCENT_LABELS[i]:>15}", end="")
        for val in row:
            print(f"{val:>10}", end="")
        print()


if __name__ == "__main__":
    main()
