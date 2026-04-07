"""
Phase 1 & 2 — Load Mozilla Common Voice, filter accents, resample, and extract features.

Usage:
    python prepare_data.py [--output_dir processed_data]

The processed dataset is saved to disk so training doesn't re-process every time.
"""

import argparse
from datasets import load_dataset, Audio, DatasetDict
from transformers import Wav2Vec2FeatureExtractor
from config import (
    BASE_MODEL,
    DATASET_NAME,
    DATASET_LANG,
    ACCENT_LABELS,
    LABEL2ID,
    SAMPLING_RATE,
    MAX_LENGTH_SAMPLES,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
    TEST_SPLIT,
)


def filter_accents(dataset, accents: list[str]):
    """Keep only rows whose accent field is in the target list."""
    before = len(dataset)
    dataset = dataset.filter(
        lambda batch: [a in accents for a in batch["accent"]],
        batched=True,
        batch_size=1000,
    )
    after = len(dataset)
    print(f"  Filtered {before} → {after} samples (kept accents: {accents})")
    return dataset


def preprocess_fn(batch, extractor):
    """Resample to 16 kHz had already been done via cast_column; extract features."""
    audio_arrays = [x["array"] for x in batch["audio"]]
    inputs = extractor(
        audio_arrays,
        sampling_rate=SAMPLING_RATE,
        max_length=MAX_LENGTH_SAMPLES,
        truncation=True,
        padding=True,
    )
    inputs["labels"] = [LABEL2ID[a] for a in batch["accent"]]
    return inputs


def main():
    parser = argparse.ArgumentParser(description="Prepare Common Voice data for accent classification.")
    parser.add_argument("--output_dir", type=str, default="processed_data", help="Where to save the processed dataset.")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Cap training samples (for quick testing).")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Cap eval samples (for quick testing).")
    args = parser.parse_args()

    print("=" * 60)
    print("STEP 1 — Loading Mozilla Common Voice (English)")
    print("=" * 60)

    # Load train and validation splits
    ds = DatasetDict()
    for split_name in [TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT]:
        print(f"\nLoading split: {split_name}")
        split_ds = load_dataset(
            DATASET_NAME,
            DATASET_LANG,
            split=split_name,
            trust_remote_code=True,
        )
        ds[split_name] = split_ds

    print("\n" + "=" * 60)
    print("STEP 2 — Filtering to target accents")
    print("=" * 60)

    for split_name in ds:
        print(f"\n[{split_name}]")
        ds[split_name] = filter_accents(ds[split_name], ACCENT_LABELS)

    # Optional: cap samples for fast dev runs
    if args.max_train_samples:
        ds[TRAIN_SPLIT] = ds[TRAIN_SPLIT].select(range(min(args.max_train_samples, len(ds[TRAIN_SPLIT]))))
        print(f"\n  Capped train to {len(ds[TRAIN_SPLIT])} samples")
    if args.max_eval_samples:
        for split in [VALIDATION_SPLIT, TEST_SPLIT]:
            ds[split] = ds[split].select(range(min(args.max_eval_samples, len(ds[split]))))
            print(f"  Capped {split} to {len(ds[split])} samples")

    print("\n" + "=" * 60)
    print("STEP 3 — Resampling audio to 16 kHz")
    print("=" * 60)
    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
    print("  Done.")

    print("\n" + "=" * 60)
    print("STEP 4 — Extracting features with Wav2Vec2FeatureExtractor")
    print("=" * 60)

    extractor = Wav2Vec2FeatureExtractor.from_pretrained(BASE_MODEL)

    ds = ds.map(
        lambda batch: preprocess_fn(batch, extractor),
        batched=True,
        batch_size=32,
        remove_columns=ds[TRAIN_SPLIT].column_names,
        num_proc=1,  # Increase if you have multiple CPUs available
    )

    # Set format for PyTorch
    ds.set_format("torch")

    print("\n" + "=" * 60)
    print(f"STEP 5 — Saving processed dataset to '{args.output_dir}'")
    print("=" * 60)
    ds.save_to_disk(args.output_dir)
    print("  Done! Dataset saved.\n")

    # Print summary
    for split_name in ds:
        print(f"  {split_name}: {len(ds[split_name])} samples")


if __name__ == "__main__":
    main()
