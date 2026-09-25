"""Inspect the fields needed for LLMTrace window-level authorship detection."""

from collections import Counter

from datasets import load_dataset


DATASET_NAME = "iitolstykh/LLMTrace_detection"


def main():
    dataset = load_dataset(DATASET_NAME)

    print(f"Dataset: {DATASET_NAME}")
    print(f"Splits: {dict(dataset.num_rows)}")

    for split_name, split in dataset.items():
        label_counts = Counter(split["label"])
        print(f"\n{split_name} columns: {split.column_names}")
        print(f"{split_name} label counts: {dict(sorted(label_counts.items()))}")

    train = dataset["train"]
    mixed_example = next(row for row in train if row["label"] == "mixed")
    print("\nExample mixed document:")
    print(f"  document length: {len(mixed_example['text'])} characters")
    print(f"  AI character intervals: {mixed_example['ai_char_intervals']}")
    print(f"  model: {mixed_example['model']}")


if __name__ == "__main__":
    main()
