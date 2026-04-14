"""
Phase 2 — Data Preparation Pipeline.

Loads Common Voice (global accents) and Svarah (Indian sub-accents),
merges, balances, creates 3 clip-length variants, performs stratified
split, and saves reproducible processed datasets.

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
    concatenate_datasets,
    load_dataset,
)
from sklearn.model_selection import train_test_split
from transformers import Wav2Vec2FeatureExtractor

from config import (
    ACCENT_LABELS,
    CLIP_LENGTHS,
    COMMON_VOICE_DATASET,
    COMMON_VOICE_LANG,
    CV_ACCENT_MAP,
    CV_TARGET_ACCENTS,
    DRY_RUN_SAMPLES_PER_CLASS,
    ID2LABEL,
    LABEL2ID,
    MODEL_NAME,
    PROCESSED_DATA_DIR,
    SAMPLE_RATE,
    SEED,
    SVARAH_DATASET,
    SVARAH_REGION_MAP,
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
# STEP A — Load Common Voice (global accents)
# ═══════════════════════════════════════════════════════════════════════════════

def load_common_voice(dry_run: bool = False) -> Dataset:
    """Load Mozilla Common Voice English, filter to 4 global accents."""
    logger.info("STEP A — Loading Common Voice (English)")

    # Load the train split (largest) — we'll do our own stratified split later
    ds = load_dataset(
        COMMON_VOICE_DATASET,
        COMMON_VOICE_LANG,
        split="train",
        trust_remote_code=True,
    )
    logger.info("  Loaded %d total train samples from Common Voice", len(ds))

    # Filter to target accents
    ds = ds.filter(
        lambda batch: [a in CV_TARGET_ACCENTS for a in batch["accent"]],
        batched=True,
        batch_size=1000,
    )
    logger.info("  After accent filter: %d samples", len(ds))

    # Map accent names: us→american, england→british, etc.
    def map_accent(example):
        example["accent"] = CV_ACCENT_MAP[example["accent"]]
        return example

    ds = ds.map(map_accent)

    # Keep only needed columns
    columns_to_keep = ["audio", "accent", "sentence"]
    columns_to_remove = [c for c in ds.column_names if c not in columns_to_keep]
    ds = ds.remove_columns(columns_to_remove)

    if dry_run:
        ds = _subsample_per_class(ds, DRY_RUN_SAMPLES_PER_CLASS)

    # Log class distribution
    _log_class_distribution(ds, "Common Voice")
    return ds


# ═══════════════════════════════════════════════════════════════════════════════
# STEP B — Load Svarah (Indian sub-accents)
# ═══════════════════════════════════════════════════════════════════════════════

def load_svarah(dry_run: bool = False) -> Dataset:
    """Load Svarah dataset and map speakers' states to Indian sub-regions."""
    logger.info("STEP B — Loading Svarah (Indian sub-accents)")

    try:
        ds = load_dataset(SVARAH_DATASET, split="train", trust_remote_code=True)
    except Exception as e:
        logger.warning(
            "⚠ Failed to load Svarah dataset: %s\n"
            "  This dataset may be gated on HuggingFace. Run 'huggingface-cli login' first.\n"
            "  Falling back to 4-class global-only mode.",
            str(e),
        )
        return None

    logger.info("  Loaded %d samples from Svarah", len(ds))

    # Identify the state/region column — try common column names
    state_col = None
    for candidate in ["state", "region", "native_region", "speaker_region", "dialect_region"]:
        if candidate in ds.column_names:
            state_col = candidate
            break

    if state_col is None:
        logger.warning(
            "  Could not find a state/region column in Svarah. Columns: %s\n"
            "  Falling back to 4-class global-only mode.",
            ds.column_names,
        )
        return None

    logger.info("  Using column '%s' for state/region mapping", state_col)

    # Normalize state names and map
    def normalize_state_name(name):
        """Normalize state name to match SVARAH_REGION_MAP keys."""
        if name is None:
            return None
        return name.strip().lower().replace(" ", "_").replace("-", "_")

    # Map each example to a sub-region
    unmapped_count = 0
    unmapped_states = set()

    def map_region(example):
        nonlocal unmapped_count, unmapped_states
        raw_state = example.get(state_col, None)
        normalized = normalize_state_name(raw_state)

        if normalized and normalized in SVARAH_REGION_MAP:
            example["accent"] = SVARAH_REGION_MAP[normalized]
            example["_keep"] = True
        else:
            example["accent"] = None
            example["_keep"] = False
            unmapped_count += 1
            if normalized:
                unmapped_states.add(normalized)
        return example

    ds = ds.map(map_region)
    logger.info("  Unmapped samples: %d (states: %s)", unmapped_count, unmapped_states)

    # Filter out unmapped
    ds = ds.filter(lambda x: x["_keep"])
    ds = ds.remove_columns(["_keep"])
    logger.info("  After filtering unmapped: %d samples", len(ds))

    # Keep only audio + accent columns (and sentence if it exists)
    columns_to_keep = ["audio", "accent"]
    if "sentence" in ds.column_names:
        columns_to_keep.append("sentence")
    elif "text" in ds.column_names:
        # Rename 'text' to 'sentence' for consistency
        ds = ds.rename_column("text", "sentence")
        columns_to_keep.append("sentence")

    columns_to_remove = [c for c in ds.column_names if c not in columns_to_keep]
    if columns_to_remove:
        ds = ds.remove_columns(columns_to_remove)

    if dry_run:
        ds = _subsample_per_class(ds, DRY_RUN_SAMPLES_PER_CLASS)

    _log_class_distribution(ds, "Svarah")
    return ds


# ═══════════════════════════════════════════════════════════════════════════════
# STEP C — Merge & balance
# ═══════════════════════════════════════════════════════════════════════════════

def merge_datasets(cv_ds: Dataset, svarah_ds: Dataset) -> Dataset:
    """Concatenate Common Voice and Svarah datasets."""
    logger.info("STEP C — Merging datasets")

    if svarah_ds is not None and len(svarah_ds) > 0:
        # Ensure both have the same columns
        cv_cols = set(cv_ds.column_names)
        sv_cols = set(svarah_ds.column_names)
        common_cols = cv_cols & sv_cols

        if cv_cols != sv_cols:
            # Keep only common columns
            for col in cv_cols - common_cols:
                cv_ds = cv_ds.remove_columns([col])
            for col in sv_cols - common_cols:
                svarah_ds = svarah_ds.remove_columns([col])

        merged = concatenate_datasets([cv_ds, svarah_ds])
        logger.info("  Merged: %d samples (CV: %d + Svarah: %d)",
                     len(merged), len(cv_ds), len(svarah_ds))
    else:
        merged = cv_ds
        logger.warning(
            "  ⚠ No Svarah data — running in 4-class global-only mode"
        )

    _log_class_distribution(merged, "Merged")

    # Warn about small classes
    counts = _get_class_counts(merged)
    for accent, count in counts.items():
        if count < 500:
            logger.warning("  ⚠ Class '%s' has only %d training clips (< 500)", accent, count)

    return merged


# ═══════════════════════════════════════════════════════════════════════════════
# STEP D — Create 3 clip-length variants
# ═══════════════════════════════════════════════════════════════════════════════

def create_clip_length_variants(
    dataset_dict: DatasetDict,
    extractor: Wav2Vec2FeatureExtractor,
) -> dict:
    """Create processed datasets for each clip length (1s, 2s, 3s)."""
    logger.info("STEP D — Creating clip-length variants")

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
# STEP E — Stratified split
# ═══════════════════════════════════════════════════════════════════════════════

def stratified_split(dataset: Dataset) -> tuple:
    """Perform 80/10/10 stratified split and generate manifest CSV."""
    logger.info("STEP E — Stratified split (80/10/10)")

    labels = dataset["accent"]
    indices = list(range(len(dataset)))

    # First split: 80% train, 20% temp
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=(VAL_RATIO + TEST_RATIO),
        stratify=labels,
        random_state=SEED,
    )

    # Second split: 50/50 of temp → val and test (10% each of original)
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
# STEP F — Validation
# ═══════════════════════════════════════════════════════════════════════════════

def validate_splits(clip_datasets: dict):
    """Validate processed datasets and print summary statistics."""
    logger.info("STEP F — Validation")

    # Load manifest and check for overlaps
    manifest_path = os.path.join(PROCESSED_DATA_DIR, "split_manifest.csv")
    if os.path.exists(manifest_path):
        manifest = pd.read_csv(manifest_path)
        train_hashes = set(manifest[manifest["split"] == "train"]["sha256"])
        val_hashes = set(manifest[manifest["split"] == "val"]["sha256"])
        test_hashes = set(manifest[manifest["split"] == "test"]["sha256"])

        train_val_overlap = train_hashes & val_hashes
        train_test_overlap = train_hashes & test_hashes
        val_test_overlap = val_hashes & test_hashes

        assert len(train_val_overlap) == 0, f"Train/Val overlap: {len(train_val_overlap)} samples!"
        assert len(train_test_overlap) == 0, f"Train/Test overlap: {len(train_test_overlap)} samples!"
        assert len(val_test_overlap) == 0, f"Val/Test overlap: {len(val_test_overlap)} samples!"
        logger.info("  ✅ No overlap between train/val/test splits")
    else:
        logger.warning("  ⚠ Manifest file not found — skipping overlap check")

    # Print per-class counts per split per clip length
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

            # Print table
            parts = [f"{ID2LABEL[int(u)]}: {c}" for u, c in zip(unique, counts)]
            logger.info("    %s (%d total): %s", split_name, len(split_ds), " | ".join(parts))

        data_stats["clip_lengths"][f"{clip_len}s"] = clip_stats

    # Save data_stats.json
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

    # Set seeds
    np.random.seed(SEED)

    # ── Step A: Common Voice ──
    cv_ds = load_common_voice(dry_run=args.dry_run)

    # ── Step B: Svarah ──
    svarah_ds = load_svarah(dry_run=args.dry_run)

    # ── Step C: Merge ──
    merged = merge_datasets(cv_ds, svarah_ds)

    # ── Step D+E: Resample audio to 16kHz, then stratified split ──
    logger.info("Resampling all audio to %d Hz...", SAMPLE_RATE)
    merged = merged.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))

    # ── Step E: Stratified split ──
    split_dict = stratified_split(merged)

    # ── Step D: Create clip-length variants ──
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    clip_datasets = create_clip_length_variants(split_dict, extractor)

    # ── Step F: Validate ──
    validate_splits(clip_datasets)

    logger.info("=" * 60)
    logger.info("✅ Data preparation complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
