import argparse
from pathlib import Path

import pandas as pd

from code_snippets import print_binary_classification_results, train_binary_text_classifiers


def make_binary_label(source):
    if pd.isna(source):
        return None

    return "Human" if str(source).strip().lower() == "human" else "AI"


def load_dataset_sample(
    csv_path,
    text_column,
    label_column,
    rows_per_label,
    random_state=42,
    chunk_size=100000,
):
    if rows_per_label is None:
        sample = pd.read_csv(csv_path)
        return prepare_labels(sample, label_column)

    sample_parts = {}

    for chunk_number, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunk_size)):
        chunk = prepare_labels(chunk, label_column)

        if text_column not in chunk.columns:
            raise ValueError(f"Could not find text column '{text_column}' in {csv_path}.")

        chunk = chunk[[text_column, label_column]].dropna()

        for label, label_rows in chunk.groupby(label_column):
            old_rows = sample_parts.get(label)
            candidate_rows = label_rows if old_rows is None else pd.concat([old_rows, label_rows])

            if len(candidate_rows) > rows_per_label:
                seed = random_state + chunk_number
                candidate_rows = candidate_rows.sample(n=rows_per_label, random_state=seed)

            sample_parts[label] = candidate_rows

    if not sample_parts:
        raise ValueError(f"No usable rows were found in {csv_path}.")

    return pd.concat(sample_parts.values(), ignore_index=True)


def load_prompt_matched_sample(
    csv_path,
    text_column,
    label_column,
    prompt_column,
    rows_per_label,
    random_state=42,
    chunk_size=100000,
):
    count_parts = []
    label_source_column = "source" if label_column == "label" else label_column
    count_columns = sorted({prompt_column, label_source_column})

    for chunk in pd.read_csv(csv_path, usecols=count_columns, chunksize=chunk_size):
        chunk = prepare_labels(chunk, label_column)
        chunk = chunk[[prompt_column, label_column]].dropna()
        counts = chunk.groupby([prompt_column, label_column]).size().reset_index(name="rows")
        count_parts.append(counts)

    if not count_parts:
        raise ValueError(f"No usable rows were found in {csv_path}.")

    prompt_counts = pd.concat(count_parts, ignore_index=True)
    prompt_counts = (
        prompt_counts.groupby([prompt_column, label_column])["rows"]
        .sum()
        .unstack(fill_value=0)
    )

    if not {"Human", "AI"}.issubset(prompt_counts.columns):
        raise ValueError("Prompt matching requires both Human and AI rows.")

    prompt_counts["matched_rows"] = prompt_counts[["Human", "AI"]].min(axis=1)
    prompt_counts = prompt_counts[prompt_counts["matched_rows"] > 0]

    if prompt_counts.empty:
        raise ValueError("No prompts had both Human and AI rows.")

    prompt_counts = prompt_counts.sample(frac=1, random_state=random_state)
    quotas = {}
    total_per_label = 0

    for prompt_id, row in prompt_counts.iterrows():
        if rows_per_label is not None and total_per_label >= rows_per_label:
            break

        prompt_quota = int(row["matched_rows"])

        if rows_per_label is not None:
            prompt_quota = min(prompt_quota, rows_per_label - total_per_label)

        if prompt_quota > 0:
            quotas[prompt_id] = prompt_quota
            total_per_label += prompt_quota

    data_columns = sorted({text_column, prompt_column, label_source_column})
    selected_parts = []
    selected_counts = {(prompt_id, label): 0 for prompt_id in quotas for label in ["Human", "AI"]}

    for chunk_number, chunk in enumerate(pd.read_csv(csv_path, usecols=data_columns, chunksize=chunk_size)):
        chunk = prepare_labels(chunk, label_column)
        chunk = chunk[[text_column, prompt_column, label_column]].dropna()
        chunk = chunk[chunk[prompt_column].isin(quotas)]

        if chunk.empty:
            continue

        for (prompt_id, label), rows in chunk.groupby([prompt_column, label_column]):
            target_count = quotas.get(prompt_id, 0)
            already_selected = selected_counts.get((prompt_id, label), 0)
            needed = target_count - already_selected

            if needed <= 0:
                continue

            take_count = min(needed, len(rows))
            selected = rows.sample(n=take_count, random_state=random_state + chunk_number)
            selected_parts.append(selected)
            selected_counts[(prompt_id, label)] = already_selected + take_count

    if not selected_parts:
        raise ValueError("Prompt matching did not select any rows.")

    sample = pd.concat(selected_parts, ignore_index=True)
    sample = sample.sample(frac=1, random_state=random_state).reset_index(drop=True)

    summary = (
        sample.groupby([prompt_column, label_column])
        .size()
        .unstack(fill_value=0)
    )
    matched_prompt_count = int((summary[["Human", "AI"]].min(axis=1) > 0).sum())
    print(
        f"Created prompt-matched sample with {len(sample)} rows "
        f"from {matched_prompt_count} prompts."
    )

    return sample


def prepare_labels(sample, label_column):
    if label_column in sample.columns:
        return sample

    if "source" in sample.columns:
        sample = sample.copy()
        sample[label_column] = sample["source"].map(make_binary_label)
        return sample

    raise ValueError(
        f"Could not find label column '{label_column}'. "
        "This script can also create labels automatically if the CSV has a 'source' column."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compare multiple models for Human vs AI text classification."
    )
    parser.add_argument("csv_path", help="Path to a CSV file with text and label columns.")
    parser.add_argument("--text-column", default="text", help="Name of the text column.")
    parser.add_argument("--label-column", default="label", help="Name of the Human/AI label column.")
    parser.add_argument("--prompt-column", default="prompt_id", help="Name of the prompt ID column.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of rows used for testing.")
    parser.add_argument("--max-features", type=int, default=30000, help="Maximum TF-IDF features.")
    parser.add_argument(
        "--match-prompt",
        action="store_true",
        help="Build a sample with Human and AI texts matched by the same prompt.",
    )
    parser.add_argument(
        "--rows-per-label",
        type=int,
        default=12000,
        help="Balanced sample size per class. Use 0 to read the whole CSV.",
    )
    parser.add_argument(
        "--save-results",
        default="classification_results.csv",
        help="Where to save the model comparison table.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)

    if not csv_path.exists():
        available_csvs = sorted(Path(".").glob("*.csv"))
        available_text = ", ".join(str(path) for path in available_csvs) if available_csvs else "none"

        raise FileNotFoundError(
            f"Could not find '{csv_path}'. Replace it with the actual dataset filename/Username. "
            f"CSV files in this folder: {available_text}"
        )

    rows_per_label = args.rows_per_label if args.rows_per_label > 0 else None
    if args.match_prompt:
        sample = load_prompt_matched_sample(
            csv_path,
            text_column=args.text_column,
            label_column=args.label_column,
            prompt_column=args.prompt_column,
            rows_per_label=rows_per_label,
        )
    else:
        sample = load_dataset_sample(
            csv_path,
            text_column=args.text_column,
            label_column=args.label_column,
            rows_per_label=rows_per_label,
        )

    results_table, reports, confusion_matrices, trained_models = train_binary_text_classifiers(
        sample,
        text_column=args.text_column,
        label_column=args.label_column,
        test_size=args.test_size,
        max_features=args.max_features,
    )

    print_binary_classification_results(results_table, reports, confusion_matrices)
    results_table.to_csv(args.save_results, index=False)

    print(f"\nSaved comparison table to {args.save_results}")
    print(f"Trained {len(trained_models)} models.")


if __name__ == "__main__":
    main()
