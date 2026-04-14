"""
Phase 2 — Data Preparation Pipeline.

Loads Westbrook English Accent Dataset, filters to target accents,
creates 3 clip-length variants, performs stratified split, and saves
reproducible processed datasets.

Usage:
    python prepare_data.py [--dry_run]
"""

import argparse
import hashlib
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
from datasets import (
    Audio,
    Dataset,
    DatasetDict,
    Value,
    load_dataset,
)
from sklearn.model_selection import train_test_split
from transformers import Wav2Vec2FeatureExtractor

from config import (
    ACCENT_DATASET,
    ACCENT_LABELS,
    ACCENT_MAP,
    CLIP_LENGTHS,
    DRY_RUN_SAMPLES_PER_CLASS,
    ID2LABEL,
    LABEL2ID,
    MODEL_NAME,
    PROCESSED_DATA_DIR,
    SAMPLE_RATE,
    SEED,
    TARGET_ACCENTS,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP A — Load Westbrook English Accent Dataset
# ═══════════════════════════════════════════════════════════════════════════════

def load_accent_data(dry_run: bool = False) -> Dataset:
    """Load Westbrook dataset, convert ClassLabel to string, filter accents."""
    logger.info("STEP A — Loading Westbrook English Accent Dataset")

    ds = load_dataset(ACCENT_DATASET, split="train")
    logger.info("  Loaded %d total samples", len(ds))

    # The 'accent' column is a ClassLabel (integer-encoded).
    # Convert to string names so our filter/mapping works.
    accent_feature = ds.features["accent"]
    logger.info("  Accent feature type: %s", type(accent_feature).__name__)

    if hasattr(accent_feature, "int2str"):
        logger.info("  Converting ClassLabel integers to string names...")
        label_names = accent_feature.names
        logger.info("  Available accents: %s", label_names)
        ds = ds.cast_column("accent", Value("string"))
        # After cast, values become the string names automatically
        # Verify
        sample_accents = set(ds[:10]["accent"])
        logger.info("  Sample accent values after cast: %s", sample_accents)

    # Filter to target accents
    ds = ds.filter(
        lambda batch: [a in TARGET_ACCENTS for a in batch["accent"]],
        batched=True,
        batch_size=1000,
    )
    logger.info("  After accent filter: %d samples", len(ds))

    if len(ds) == 0:
        # Debug: print what values we actually have
        logger.error("  No samples matched! Check accent values vs TARGET_ACCENTS=%s", TARGET_ACCENTS)
        raise ValueError("No samples matched the target accents. Check the accent mapping.")

    # Map accent names: American→american, English→british, etc.
    def map_accent(example):
        example["accent"] = ACCENT_MAP[example["accent"]]
        return example

    ds = ds.map(map_accent)

    # Keep only needed columns
    columns_to_keep = ["audio", "accent"]
    if "raw_text" in ds.column_names:
        ds = ds.rename_column("raw_text", "sentence")
        columns_to_keep.append("sentence")

    columns_to_remove = [c for c in ds.column_names if c not in columns_to_keep]
    ds = ds.remove_columns(columns_to_remove)

    if dry_run:
        ds = _subsample_per_class(ds, DRY_RUN_SAMPLES_PER_CLASS)

    _log_class_distribution(ds, "Filtered dataset")
    return ds


# ═══════════════════════════════════════════════════════════════════════════════
# STEP B — Create 3 clip-length variants
# ═══════════════════════════════════════════════════════════════════════════════

def create_clip_length_variants(
    dataset_dict: DatasetDict,
    extractor: Wav2Vec2FeatureExtractor,
) -> dict:
    """Create processed datasets for each clip length (1s, 2s, 3s)."""
    logger.info("STEP B — Creating clip-length variants")

    clip_datasets = {}

    for clip_len in CLIP_LENGTHS:
        max_samples = SAMPLE_RATE * clip_len
        logger.info("  Processing %ds clips (%d samples)...", clip_len, max_samples)

        def preprocess_fn(batch, _max_samples=max_samples):
            """Resample, truncate/pad, and extract features."""
            audio_arrays = []
            for audio in batch["audio"]:
                arr = audio["array"]
                # Truncate or zero-pad to exact clip length
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
            inputs["labels"] = [LABEL2ID[a] for a in batch["accent"]]
            return inputs

        processed_dict = DatasetDict()
        for split_name, split_ds in dataset_dict.items():
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

        # Save to disk
        save_path = os.path.join(PROCESSED_DATA_DIR, f"clips_{clip_len}s")
        processed_dict.save_to_disk(save_path)
        logger.info("    Saved to %s", save_path)
        clip_datasets[clip_len] = processed_dict

    return clip_datasets


# ═══════════════════════════════════════════════════════════════════════════════
# STEP C — Stratified split
# ═══════════════════════════════════════════════════════════════════════════════

def stratified_split(dataset: Dataset) -> DatasetDict:
    """Perform 80/10/10 stratified split and generate manifest CSV."""
    logger.info("STEP C — Stratified split (80/10/10)")

    labels = dataset["accent"]
    indices = list(range(len(dataset)))

    # First split: 80% train, 20% temp
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=(VAL_RATIO + TEST_RATIO),
        stratify=labels,
        random_state=SEED,
    )

    # Second split: 50/50 of temp → val and test
    temp_labels = [labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        stratify=temp_labels,
        random_state=SEED,
    )

    train_ds = dataset.select(train_idx)
    val_ds = dataset.select(val_idx)
    test_ds = dataset.select(test_idx)

    logger.info("  Train: %d | Val: %d | Test: %d", len(train_ds), len(val_ds), len(test_ds))

    # Generate manifest CSV with sha256 hashes
    manifest_rows = []
    for split_name, split_ds, split_indices in [
        ("train", train_ds, train_idx),
        ("val", val_ds, val_idx),
        ("test", test_ds, test_idx),
    ]:
        for i, idx in enumerate(split_indices):
            audio_array = split_ds[i]["audio"]["array"]
            audio_bytes = audio_array.tobytes() if hasattr(audio_array, "tobytes") else bytes(audio_array)
            sha = hashlib.sha256(audio_bytes).hexdigest()
            manifest_rows.append({
                "file_id": f"{split_name}_{i:06d}",
                "accent": split_ds[i]["accent"],
                "split": split_name,
                "sha256": sha,
            })

    manifest_df = pd.DataFrame(manifest_rows)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    manifest_path = os.path.join(PROCESSED_DATA_DIR, "split_manifest.csv")
    manifest_df.to_csv(manifest_path, index=False)
    logger.info("  Saved manifest to %s (%d rows)", manifest_path, len(manifest_df))

    return DatasetDict({"train": train_ds, "val": val_ds, "test": test_ds})


# ═══════════════════════════════════════════════════════════════════════════════
# STEP D — Validation
# ═══════════════════════════════════════════════════════════════════════════════

def validate_splits(clip_datasets: dict):
    """Validate processed datasets and print summary statistics."""
    logger.info("STEP D — Validation")

    manifest_path = os.path.join(PROCESSED_DATA_DIR, "split_manifest.csv")
    if os.path.exists(manifest_path):
        manifest = pd.read_csv(manifest_path)
        train_hashes = set(manifest[manifest["split"] == "train"]["sha256"])
        val_hashes = set(manifest[manifest["split"] == "val"]["sha256"])
        test_hashes = set(manifest[manifest["split"] == "test"]["sha256"])

        assert len(train_hashes & val_hashes) == 0, "Train/Val overlap!"
        assert len(train_hashes & test_hashes) == 0, "Train/Test overlap!"
        assert len(val_hashes & test_hashes) == 0, "Val/Test overlap!"
        logger.info("  ✅ No overlap between train/val/test splits")

    data_stats = {"clip_lengths": {}}
    for clip_len, ds_dict in clip_datasets.items():
        logger.info("  ── %ds clips ──", clip_len)
        clip_stats = {}
        for split_name, split_ds in ds_dict.items():
            labels = split_ds["labels"]
            if hasattr(labels, "numpy"):
                labels = labels.numpy()
            else:
                labels = np.array(labels)

            unique, counts = np.unique(labels, return_counts=True)
            count_dict = {ID2LABEL[int(u)]: int(c) for u, c in zip(unique, counts)}
            clip_stats[split_name] = count_dict

            parts = [f"{ID2LABEL[int(u)]}: {c}" for u, c in zip(unique, counts)]
            logger.info("    %s (%d total): %s", split_name, len(split_ds), " | ".join(parts))

        data_stats["clip_lengths"][f"{clip_len}s"] = clip_stats

    stats_path = os.path.join(PROCESSED_DATA_DIR, "data_stats.json")
    with open(stats_path, "w") as f:
        json.dump(data_stats, f, indent=2)
    logger.info("  Saved data stats to %s", stats_path)
    logger.info("  ✅ Validation complete")


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _get_class_counts(ds: Dataset) -> dict:
    """Get per-class counts from a dataset."""
    from collections import Counter
    return dict(Counter(ds["accent"]))


def _log_class_distribution(ds: Dataset, name: str):
    """Log the class distribution of a dataset."""
    counts = _get_class_counts(ds)
    total = sum(counts.values())
    logger.info("  %s distribution (%d total):", name, total)
    for label in ACCENT_LABELS:
        if label in counts:
            pct = 100.0 * counts[label] / total
            logger.info("    %-15s %6d  (%5.1f%%)", label, counts[label], pct)


def _subsample_per_class(ds: Dataset, n: int) -> Dataset:
    """Subsample to at most n examples per class."""
    from collections import defaultdict

    class_indices = defaultdict(list)
    for i, accent in enumerate(ds["accent"]):
        class_indices[accent].append(i)

    selected = []
    for accent, indices in class_indices.items():
        np.random.seed(SEED)
        if len(indices) > n:
            indices = np.random.choice(indices, size=n, replace=False).tolist()
        selected.extend(indices)

    selected.sort()
    logger.info("  Subsampled to %d samples (%d per class max)", len(selected), n)
    return ds.select(selected)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Prepare data for the Indian Accent Detector."
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help=f"Run on {DRY_RUN_SAMPLES_PER_CLASS} samples per class for fast testing.",
    )
    args = parser.parse_args()

    if args.dry_run:
        logger.info("🏃 DRY RUN MODE — %d samples per class", DRY_RUN_SAMPLES_PER_CLASS)

    np.random.seed(SEED)

    # ── Step A: Load and filter ──
    ds = load_accent_data(dry_run=args.dry_run)

    # ── Resample audio to 16kHz ──
    logger.info("Resampling all audio to %d Hz...", SAMPLE_RATE)
    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))

    # ── Step C: Stratified split ──
    split_dict = stratified_split(ds)

    # ── Step B: Create clip-length variants ──
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    clip_datasets = create_clip_length_variants(split_dict, extractor)

    # ── Step D: Validate ──
    validate_splits(clip_datasets)

    logger.info("=" * 60)
    logger.info("✅ Data preparation complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
