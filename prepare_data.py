"""
Phase 2 — Data Preparation Pipeline.

Loads Westbrook English Accent Dataset (global accents) and
IndicAccentDb (Indian sub-accents), merges them, creates clip-length
variants, performs stratified split, and saves reproducible processed
datasets.

Usage:
    python prepare_data.py [--dry_run]
"""

import argparse
import hashlib
import json
import logging
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from datasets import (
    Audio,
    Dataset,
    DatasetDict,
    Value,
    concatenate_datasets,
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
    INDIAN_ACCENT_DATASET,
    INDIAN_ACCENT_MAP,
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
# STEP A — Load Global Accents (Westbrook)
# ═══════════════════════════════════════════════════════════════════════════════

def load_global_accents() -> Dataset:
    """Load Westbrook dataset for American, British, Canadian accents."""
    logger.info("STEP A.1 — Loading Westbrook English Accent Dataset")

    ds = load_dataset(ACCENT_DATASET, split="train")
    logger.info("  Loaded %d total samples", len(ds))

    # Convert ClassLabel integers to string names
    accent_feature = ds.features["accent"]
    if hasattr(accent_feature, "int2str"):
        label_names = accent_feature.names
        logger.info("  Available accents: %s", label_names)

        # ClassLabel auto-encodes strings back to ints, so use remove+add approach
        accent_ints = ds["accent"]
        accent_strings = [label_names[i] for i in accent_ints]
        ds = ds.remove_columns(["accent"])
        ds = ds.add_column("accent", accent_strings)

        sample_accents = set(ds[:10]["accent"])
        logger.info("  Sample values after conversion: %s", sample_accents)

    # Filter to target global accents (American, English, Canadian)
    ds = ds.filter(
        lambda batch: [a in TARGET_ACCENTS for a in batch["accent"]],
        batched=True,
        batch_size=1000,
    )
    logger.info("  After accent filter: %d samples", len(ds))

    if len(ds) == 0:
        raise ValueError("No samples matched the target accents.")

    # Map accent names: American→american, English→british, etc.
    def map_accent(example):
        example["accent"] = ACCENT_MAP[example["accent"]]
        return example

    ds = ds.map(map_accent)

    # Keep only needed columns
    columns_to_keep = ["audio", "accent"]
    columns_to_remove = [c for c in ds.column_names if c not in columns_to_keep]
    ds = ds.remove_columns(columns_to_remove)

    _log_class_distribution(ds, "Global accents")
    return ds


# ═══════════════════════════════════════════════════════════════════════════════
# STEP A.2 — Load Indian Sub-Accents (IndicAccentDb)
# ═══════════════════════════════════════════════════════════════════════════════

def load_indian_accents() -> Dataset:
    """Load IndicAccentDb for Indian sub-regional accents."""
    logger.info("STEP A.2 — Loading IndicAccentDb (Indian sub-accents)")

    ds = load_dataset(INDIAN_ACCENT_DATASET, split="train")
    logger.info("  Loaded %d total samples", len(ds))
    logger.info("  Column names: %s", ds.column_names)

    # Identify the label column (could be 'label' or 'accent')
    label_col = None
    for candidate in ["label", "accent", "class"]:
        if candidate in ds.column_names:
            label_col = candidate
            break

    if label_col is None:
        raise ValueError(f"Cannot find label column. Available: {ds.column_names}")

    logger.info("  Using label column: '%s'", label_col)

    # Convert ClassLabel to string if needed
    label_feature = ds.features[label_col]
    if hasattr(label_feature, "int2str"):
        label_names = label_feature.names
        logger.info("  Available Indian accent labels: %s", label_names)

        label_ints = ds[label_col]
        label_strings = [label_names[i] for i in label_ints]
        ds = ds.remove_columns([label_col])
        ds = ds.add_column("accent_raw", label_strings)
    else:
        # Already strings
        ds = ds.rename_column(label_col, "accent_raw")

    # Normalize label strings (lowercase, strip)
    raw_labels = ds["accent_raw"]
    normalized = [s.strip().lower().replace(" ", "_") for s in raw_labels]
    ds = ds.remove_columns(["accent_raw"])
    ds = ds.add_column("accent_raw", normalized)

    unique_raw = set(normalized)
    logger.info("  Normalized unique labels: %s", unique_raw)

    # Map to our Indian sub-regions
    mapped_accents = []
    unmapped = set()
    for raw in ds["accent_raw"]:
        if raw in INDIAN_ACCENT_MAP:
            mapped_accents.append(INDIAN_ACCENT_MAP[raw])
        else:
            unmapped.add(raw)
            mapped_accents.append(None)

    if unmapped:
        logger.warning("  Unmapped labels (will be dropped): %s", unmapped)

    ds = ds.remove_columns(["accent_raw"])
    ds = ds.add_column("accent", mapped_accents)

    # Drop unmapped samples
    ds = ds.filter(lambda x: x["accent"] is not None)
    logger.info("  After mapping: %d samples", len(ds))

    # Keep only needed columns
    columns_to_keep = ["audio", "accent"]
    columns_to_remove = [c for c in ds.column_names if c not in columns_to_keep]
    if columns_to_remove:
        ds = ds.remove_columns(columns_to_remove)

    _log_class_distribution(ds, "Indian sub-accents")
    return ds


# ═══════════════════════════════════════════════════════════════════════════════
# STEP A.3 — Merge Datasets
# ═══════════════════════════════════════════════════════════════════════════════

def merge_datasets(global_ds: Dataset, indian_ds: Dataset) -> Dataset:
    """Merge global and Indian sub-accent datasets."""
    logger.info("STEP A.3 — Merging datasets")

    # Ensure both have same Audio feature format
    if "audio" in global_ds.features and "audio" in indian_ds.features:
        # Cast both to same Audio format
        global_ds = global_ds.cast_column("audio", Audio())
        indian_ds = indian_ds.cast_column("audio", Audio())

    merged = concatenate_datasets([global_ds, indian_ds])
    logger.info("  Merged total: %d samples", len(merged))
    _log_class_distribution(merged, "Merged dataset")
    return merged


# ═══════════════════════════════════════════════════════════════════════════════
# STEP B — Create clip-length variants
# ═══════════════════════════════════════════════════════════════════════════════

def create_clip_length_variants(
    dataset_dict: DatasetDict,
    extractor: Wav2Vec2FeatureExtractor,
) -> dict:
    """Create processed datasets for each clip length."""
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

    train_ds = dataset.select(train_idx)
    val_ds = dataset.select(val_idx)
    test_ds = dataset.select(test_idx)

    logger.info("  Train: %d | Val: %d | Test: %d", len(train_ds), len(val_ds), len(test_ds))

    # Generate manifest CSV
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

def _log_class_distribution(ds: Dataset, name: str):
    """Log the class distribution of a dataset."""
    counts = dict(Counter(ds["accent"]))
    total = sum(counts.values())
    logger.info("  %s distribution (%d total):", name, total)
    for label in ACCENT_LABELS:
        if label in counts:
            pct = 100.0 * counts[label] / total
            logger.info("    %-15s %6d  (%5.1f%%)", label, counts[label], pct)


def _subsample_per_class(ds: Dataset, n: int) -> Dataset:
    """Subsample to at most n examples per class."""
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

    # ── Step A.1: Load global accents ──
    global_ds = load_global_accents()

    # ── Step A.2: Load Indian sub-accents ──
    indian_ds = load_indian_accents()

    # ── Step A.3: Merge ──
    merged_ds = merge_datasets(global_ds, indian_ds)

    if args.dry_run:
        merged_ds = _subsample_per_class(merged_ds, DRY_RUN_SAMPLES_PER_CLASS)

    # ── Resample audio to 16kHz ──
    logger.info("Resampling all audio to %d Hz...", SAMPLE_RATE)
    merged_ds = merged_ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))

    # ── Step C: Stratified split ──
    split_dict = stratified_split(merged_ds)

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
