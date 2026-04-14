"""
Phase 4 — Evaluation Pipeline.

This is the core research contribution. Produces:
  - Per-class metrics CSV for each clip length
  - Normalized confusion matrix PNG + CSV for each clip length
  - Overall metrics CSV across all clip lengths
  - Clip-length accuracy curve PNG + CSV
  - Baseline comparison table

Usage:
    python evaluate.py [--dry_run]
    python evaluate.py --clip_length 3
"""

import argparse
import json
import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving figures

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datasets import load_from_disk
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    accuracy_score,
)
from transformers import Wav2Vec2ForSequenceClassification

from config import (
    ACCENT_LABELS,
    CLIP_LENGTHS,
    DISPLAY_LABELS,
    ID2LABEL,
    MODEL_OUTPUT_DIR,
    NUM_LABELS,
    PROCESSED_DATA_DIR,
    RESULTS_DIR,
    SEED,
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


def run_inference(model, test_ds, batch_size: int = 32):
    """Run inference on the test split and return predictions + labels."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    all_logits = []

    for i in range(0, len(test_ds), batch_size):
        batch = test_ds[i:i + batch_size]
        input_values = batch["input_values"]
        if isinstance(input_values, list):
            input_values = torch.stack(input_values)
        input_values = input_values.to(device)

        labels = batch["labels"]

        with torch.no_grad():
            outputs = model(input_values=input_values)
            logits = outputs.logits

        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_logits.extend(logits.cpu().numpy())

        if hasattr(labels, "numpy"):
            all_labels.extend(labels.numpy())
        elif isinstance(labels, torch.Tensor):
            all_labels.extend(labels.numpy())
        else:
            all_labels.extend(labels)

        if (i // batch_size) % 20 == 0:
            logger.info("  Processed %d/%d", min(i + batch_size, len(test_ds)), len(test_ds))

    return np.array(all_preds), np.array(all_labels), np.array(all_logits)


# ═══════════════════════════════════════════════════════════════════════════════
# a) Per-class metrics
# ═══════════════════════════════════════════════════════════════════════════════

def save_per_class_metrics(labels, preds, clip_length: int):
    """Compute and save per-class precision, recall, F1, and support."""
    logger.info("  Saving per-class metrics...")

    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, labels=list(range(NUM_LABELS)), zero_division=0
    )

    # Determine which labels are actually present in the data
    present_labels = sorted(set(labels) | set(preds))

    rows = []
    for i in range(NUM_LABELS):
        rows.append({
            "accent": ACCENT_LABELS[i],
            "precision": round(precision[i], 4),
            "recall": round(recall[i], 4),
            "f1": round(f1[i], 4),
            "support": int(support[i]),
        })

    df = pd.DataFrame(rows)

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, f"per_class_{clip_length}s.csv")
    df.to_csv(csv_path, index=False)
    logger.info("    Saved: %s", csv_path)

    # Print table
    print(f"\n{'='*60}")
    print(f"Per-Class Metrics — {clip_length}s clips")
    print(f"{'='*60}")
    print(df.to_string(index=False))
    print()

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# b) Confusion matrix
# ═══════════════════════════════════════════════════════════════════════════════

def save_confusion_matrix(labels, preds, clip_length: int):
    """Generate and save normalized confusion matrix as PNG and CSV."""
    logger.info("  Saving confusion matrix...")

    cm = confusion_matrix(labels, preds, labels=list(range(NUM_LABELS)))

    # Normalize by true label (rows sum to 1)
    cm_normalized = cm.astype(float)
    row_sums = cm_normalized.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1
    cm_normalized = cm_normalized / row_sums

    # Save CSV (normalized)
    cm_df = pd.DataFrame(
        cm_normalized,
        index=ACCENT_LABELS,
        columns=ACCENT_LABELS,
    )
    csv_path = os.path.join(RESULTS_DIR, f"confusion_matrix_{clip_length}s.csv")
    cm_df.to_csv(csv_path)
    logger.info("    Saved: %s", csv_path)

    # Save PNG (seaborn heatmap)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=ACCENT_LABELS,
        yticklabels=ACCENT_LABELS,
        ax=ax,
        vmin=0,
        vmax=1,
        linewidths=0.5,
        linecolor="white",
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(f"Normalized Confusion Matrix — {clip_length}s clips", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    png_path = os.path.join(RESULTS_DIR, f"confusion_matrix_{clip_length}s.png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    logger.info("    Saved: %s", png_path)

    return cm_normalized


# ═══════════════════════════════════════════════════════════════════════════════
# c) Overall metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_overall_metrics(labels, preds, clip_length: int) -> dict:
    """Compute overall accuracy, macro F1, and weighted F1."""
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)

    metrics = {
        "clip_length": clip_length,
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
    }

    print(f"\nOverall Metrics — {clip_length}s clips:")
    print(f"  Accuracy:    {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Macro F1:    {macro_f1:.4f}")
    print(f"  Weighted F1: {weighted_f1:.4f}")

    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# d) Clip-length accuracy curve
# ═══════════════════════════════════════════════════════════════════════════════

def save_clip_length_curve(all_results: dict):
    """Generate clip-length vs F1 curve with per-class breakdown."""
    logger.info("Generating clip-length accuracy curve...")

    clip_lengths_evaluated = sorted(all_results.keys())

    # Collect overall macro F1 per clip length
    overall_f1s = []
    for cl in clip_lengths_evaluated:
        overall_f1s.append(all_results[cl]["overall"]["macro_f1"])

    # Collect per-class F1 per clip length
    per_class_f1s = {label: [] for label in ACCENT_LABELS}
    for cl in clip_lengths_evaluated:
        per_class_df = all_results[cl]["per_class_df"]
        for _, row in per_class_df.iterrows():
            per_class_f1s[row["accent"]].append(row["f1"])

    # Save underlying data as CSV
    curve_data = {"clip_length": [f"{cl}s" for cl in clip_lengths_evaluated]}
    curve_data["overall_macro_f1"] = overall_f1s
    for label in ACCENT_LABELS:
        curve_data[f"{label}_f1"] = per_class_f1s[label]

    curve_df = pd.DataFrame(curve_data)
    csv_path = os.path.join(RESULTS_DIR, "clip_length_curve.csv")
    curve_df.to_csv(csv_path, index=False)
    logger.info("  Saved: %s", csv_path)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 7))
    x_labels = [f"{cl}s" for cl in clip_lengths_evaluated]
    x_pos = list(range(len(clip_lengths_evaluated)))

    # Overall line (thick, black)
    ax.plot(x_pos, overall_f1s, "k-o", linewidth=3, markersize=10,
            label="Overall (macro F1)", zorder=10)

    # Per-class lines
    colors = plt.cm.Set2(np.linspace(0, 1, len(ACCENT_LABELS)))
    for i, label in enumerate(ACCENT_LABELS):
        ax.plot(x_pos, per_class_f1s[label], "-s", color=colors[i],
                linewidth=1.5, markersize=6, label=label, alpha=0.8)

    ax.set_xlabel("Clip Length", fontsize=13)
    ax.set_ylabel("Macro F1 Score", fontsize=13)
    ax.set_title("F1 Score vs Clip Length — Per-Class Breakdown", fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    png_path = os.path.join(RESULTS_DIR, "clip_length_curve.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: %s", png_path)


# ═══════════════════════════════════════════════════════════════════════════════
# e) Baseline comparison
# ═══════════════════════════════════════════════════════════════════════════════

def print_baseline_comparison(all_results: dict):
    """Print comparison table vs literature baselines."""
    print("\n" + "=" * 60)
    print("BASELINE COMPARISON")
    print("=" * 60)
    print()
    print("┌─────────────────────────────┬──────────┬──────────┬───────────────────┐")
    print("│ Method                      │ Accuracy │ Macro F1 │ Indian Sub-Accents│")
    print("├─────────────────────────────┼──────────┼──────────┼───────────────────┤")
    print("│ MPSA-DenseNet (lit.)        │ ~65%     │   N/R    │        No         │")
    print("│ AccentDB CNN (lit.)         │ ~60%     │   N/R    │        No         │")

    # Our results for each clip length
    for cl in sorted(all_results.keys()):
        metrics = all_results[cl]["overall"]
        acc = metrics["accuracy"]
        f1 = metrics["macro_f1"]
        print(f"│ Ours ({cl}s clips)            │ {acc*100:5.1f}%   │ {f1:.4f}  │       Yes         │")

    print("└─────────────────────────────┴──────────┴──────────┴───────────────────┘")
    print()
    print("Note: MPSA-DenseNet and AccentDB do not report per-class F1 or Indian")
    print("sub-regional classification. N/R = Not Reported. Comparison is indicative")
    print("only — datasets and class sets differ.")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# Main evaluation loop
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_clip_length(clip_length: int, dry_run: bool = False) -> dict:
    """Evaluate a single clip length and save all artifacts."""
    logger.info("=" * 60)
    logger.info("Evaluating %ds clips", clip_length)
    logger.info("=" * 60)

    # Load model
    model_path = os.path.join(MODEL_OUTPUT_DIR, f"clips_{clip_length}s")
    if not os.path.exists(model_path):
        logger.error("Model not found at %s. Run train.py first.", model_path)
        return None

    logger.info("  Loading model from %s", model_path)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)

    # Load test data
    data_path = os.path.join(PROCESSED_DATA_DIR, f"clips_{clip_length}s")
    if not os.path.exists(data_path):
        logger.error("Test data not found at %s. Run prepare_data.py first.", data_path)
        return None

    ds = load_from_disk(data_path)
    test_ds = ds["test"]
    logger.info("  Test samples: %d", len(test_ds))

    if dry_run:
        max_samples = 50
        test_ds = test_ds.select(range(min(max_samples, len(test_ds))))
        logger.info("  🏃 DRY RUN — capped to %d samples", max_samples)

    # Run inference
    logger.info("  Running inference...")
    preds, labels, logits = run_inference(model, test_ds)

    # a) Per-class metrics
    per_class_df = save_per_class_metrics(labels, preds, clip_length)

    # b) Confusion matrix
    cm_normalized = save_confusion_matrix(labels, preds, clip_length)

    # c) Overall metrics
    overall = compute_overall_metrics(labels, preds, clip_length)

    return {
        "overall": overall,
        "per_class_df": per_class_df,
        "cm": cm_normalized,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate accent classifier across clip lengths."
    )
    parser.add_argument(
        "--clip_length",
        type=int,
        choices=CLIP_LENGTHS,
        default=None,
        help="Evaluate only this clip length.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run on a small subset for fast testing.",
    )
    args = parser.parse_args()

    set_seed(SEED)

    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Determine which clip lengths to evaluate
    if args.clip_length is not None:
        clip_lengths = [args.clip_length]
    else:
        clip_lengths = CLIP_LENGTHS

    # Evaluate each clip length
    all_results = {}
    overall_rows = []

    for cl in clip_lengths:
        result = evaluate_clip_length(cl, dry_run=args.dry_run)
        if result is not None:
            all_results[cl] = result
            overall_rows.append(result["overall"])

    # Save overall metrics CSV (append-style, but overwrite for consistency)
    if overall_rows:
        overall_df = pd.DataFrame(overall_rows)
        overall_csv_path = os.path.join(RESULTS_DIR, "overall_metrics.csv")
        overall_df.to_csv(overall_csv_path, index=False)
        logger.info("Saved overall metrics to %s", overall_csv_path)

        print(f"\n{'='*60}")
        print("Overall Metrics Summary")
        print(f"{'='*60}")
        print(overall_df.to_string(index=False))
        print()

    # Generate clip-length curve (only if we have multiple clip lengths)
    if len(all_results) > 1:
        save_clip_length_curve(all_results)
    elif len(all_results) == 1:
        logger.info("Only 1 clip length evaluated — skipping curve generation.")

    # Baseline comparison
    if all_results:
        print_baseline_comparison(all_results)

    logger.info("=" * 60)
    logger.info("✅ Evaluation complete! Results saved to %s/", RESULTS_DIR)
    logger.info("=" * 60)

    # Print file listing
    if os.path.exists(RESULTS_DIR):
        logger.info("Generated files:")
        for f in sorted(os.listdir(RESULTS_DIR)):
            fpath = os.path.join(RESULTS_DIR, f)
            size = os.path.getsize(fpath)
            logger.info("  %s (%s bytes)", f, f"{size:,}")


if __name__ == "__main__":
    main()
