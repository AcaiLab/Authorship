import argparse
from pathlib import Path

import pandas as pd

from code_snippets import (
    print_binary_classification_results,
    train_binary_text_classifiers,
    train_binary_text_classifiers_on_split,
)


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


def load_word_count_matched_sample(
    csv_path,
    text_column,
    label_column,
    word_count_column,
    word_count_bin_size,
    max_rows_per_word_count_bin,
    rows_per_label,
    random_state=42,
    chunk_size=100000,
):
    count_parts = []
    label_source_column = "source" if label_column == "label" else label_column
    count_columns = sorted({word_count_column, label_source_column})

    for chunk in pd.read_csv(csv_path, usecols=count_columns, chunksize=chunk_size):
        chunk = prepare_labels(chunk, label_column)
        chunk = chunk[[word_count_column, label_column]].dropna()
        chunk[word_count_column] = pd.to_numeric(chunk[word_count_column], errors="coerce")
        chunk = chunk.dropna()
        chunk["word_count_bin"] = (chunk[word_count_column] // word_count_bin_size).astype(int)
        counts = chunk.groupby(["word_count_bin", label_column]).size().reset_index(name="rows")
        count_parts.append(counts)

    if not count_parts:
        raise ValueError(f"No usable rows were found in {csv_path}.")

    bin_counts = pd.concat(count_parts, ignore_index=True)
    bin_counts = (
        bin_counts.groupby(["word_count_bin", label_column])["rows"]
        .sum()
        .unstack(fill_value=0)
    )

    if not {"Human", "AI"}.issubset(bin_counts.columns):
        raise ValueError("Word-count matching requires both Human and AI rows.")

    bin_counts["matched_rows"] = bin_counts[["Human", "AI"]].min(axis=1)
    bin_counts = bin_counts[bin_counts["matched_rows"] > 0]

    if bin_counts.empty:
        raise ValueError("No word-count bins had both Human and AI rows.")

    bin_counts = bin_counts.sample(frac=1, random_state=random_state)
    quotas = {}
    total_per_label = 0

    for word_count_bin, row in bin_counts.iterrows():
        if rows_per_label is not None and total_per_label >= rows_per_label:
            break

        bin_quota = min(int(row["matched_rows"]), max_rows_per_word_count_bin)

        if rows_per_label is not None:
            bin_quota = min(bin_quota, rows_per_label - total_per_label)

        if bin_quota > 0:
            quotas[word_count_bin] = bin_quota
            total_per_label += bin_quota

    data_columns = sorted({text_column, word_count_column, label_source_column})
    selected_parts = []
    selected_counts = {(word_count_bin, label): 0 for word_count_bin in quotas for label in ["Human", "AI"]}

    for chunk_number, chunk in enumerate(pd.read_csv(csv_path, usecols=data_columns, chunksize=chunk_size)):
        chunk = prepare_labels(chunk, label_column)
        chunk = chunk[[text_column, word_count_column, label_column]].dropna()
        chunk[word_count_column] = pd.to_numeric(chunk[word_count_column], errors="coerce")
        chunk = chunk.dropna()
        chunk["word_count_bin"] = (chunk[word_count_column] // word_count_bin_size).astype(int)
        chunk = chunk[chunk["word_count_bin"].isin(quotas)]

        if chunk.empty:
            continue

        for (word_count_bin, label), rows in chunk.groupby(["word_count_bin", label_column]):
            target_count = quotas.get(word_count_bin, 0)
            already_selected = selected_counts.get((word_count_bin, label), 0)
            needed = target_count - already_selected

            if needed <= 0:
                continue

            take_count = min(needed, len(rows))
            selected = rows.sample(n=take_count, random_state=random_state + chunk_number)
            selected_parts.append(selected)
            selected_counts[(word_count_bin, label)] = already_selected + take_count

    if not selected_parts:
        raise ValueError("Word-count matching did not select any rows.")

    sample = pd.concat(selected_parts, ignore_index=True)
    sample = sample.sample(frac=1, random_state=random_state).reset_index(drop=True)

    word_summary = sample.groupby(label_column)[word_count_column].agg(["count", "mean", "median"]).round(2)
    matched_bin_count = int(sample["word_count_bin"].nunique())
    print(
        f"Created word-count-matched sample with {len(sample)} rows "
        f"from {matched_bin_count} word-count bins."
    )
    print("\nWord-count summary:")
    print(word_summary.to_string())

    return sample


def load_prompt_and_word_count_matched_sample(
    csv_path,
    text_column,
    label_column,
    prompt_column,
    word_count_column,
    word_count_bin_size,
    max_rows_per_prompt_word_count_bin,
    rows_per_label,
    random_state=42,
    chunk_size=100000,
):
    count_parts = []
    label_source_column = "source" if label_column == "label" else label_column
    count_columns = sorted({prompt_column, word_count_column, label_source_column})

    for chunk in pd.read_csv(csv_path, usecols=count_columns, chunksize=chunk_size):
        chunk = prepare_labels(chunk, label_column)
        chunk = chunk[[prompt_column, word_count_column, label_column]].dropna()
        chunk[word_count_column] = pd.to_numeric(chunk[word_count_column], errors="coerce")
        chunk = chunk.dropna()
        chunk["word_count_bin"] = (chunk[word_count_column] // word_count_bin_size).astype(int)
        counts = chunk.groupby([prompt_column, "word_count_bin", label_column]).size().reset_index(name="rows")
        count_parts.append(counts)

    if not count_parts:
        raise ValueError(f"No usable rows were found in {csv_path}.")

    group_counts = pd.concat(count_parts, ignore_index=True)
    group_counts = (
        group_counts.groupby([prompt_column, "word_count_bin", label_column])["rows"]
        .sum()
        .unstack(fill_value=0)
    )

    if not {"Human", "AI"}.issubset(group_counts.columns):
        raise ValueError("Prompt and word-count matching requires both Human and AI rows.")

    group_counts["matched_rows"] = group_counts[["Human", "AI"]].min(axis=1)
    group_counts = group_counts[group_counts["matched_rows"] > 0]

    if group_counts.empty:
        raise ValueError("No prompt and word-count groups had both Human and AI rows.")

    group_counts = group_counts.sample(frac=1, random_state=random_state)
    quotas = {}
    total_per_label = 0

    for group_key, row in group_counts.iterrows():
        if rows_per_label is not None and total_per_label >= rows_per_label:
            break

        group_quota = min(int(row["matched_rows"]), max_rows_per_prompt_word_count_bin)

        if rows_per_label is not None:
            group_quota = min(group_quota, rows_per_label - total_per_label)

        if group_quota > 0:
            quotas[group_key] = group_quota
            total_per_label += group_quota

    data_columns = sorted({text_column, prompt_column, word_count_column, label_source_column})
    selected_parts = []
    selected_counts = {(group_key, label): 0 for group_key in quotas for label in ["Human", "AI"]}

    for chunk_number, chunk in enumerate(pd.read_csv(csv_path, usecols=data_columns, chunksize=chunk_size)):
        chunk = prepare_labels(chunk, label_column)
        chunk = chunk[[text_column, prompt_column, word_count_column, label_column]].dropna()
        chunk[word_count_column] = pd.to_numeric(chunk[word_count_column], errors="coerce")
        chunk = chunk.dropna()
        chunk["word_count_bin"] = (chunk[word_count_column] // word_count_bin_size).astype(int)
        chunk["group_key"] = list(zip(chunk[prompt_column], chunk["word_count_bin"]))
        chunk = chunk[chunk["group_key"].isin(quotas)]

        if chunk.empty:
            continue

        for (group_key, label), rows in chunk.groupby(["group_key", label_column]):
            target_count = quotas.get(group_key, 0)
            already_selected = selected_counts.get((group_key, label), 0)
            needed = target_count - already_selected

            if needed <= 0:
                continue

            take_count = min(needed, len(rows))
            selected = rows.sample(n=take_count, random_state=random_state + chunk_number)
            selected_parts.append(selected.drop(columns=["group_key"]))
            selected_counts[(group_key, label)] = already_selected + take_count

    if not selected_parts:
        raise ValueError("Prompt and word-count matching did not select any rows.")

    sample = pd.concat(selected_parts, ignore_index=True)
    sample = sample.sample(frac=1, random_state=random_state).reset_index(drop=True)

    prompt_count = int(sample[prompt_column].nunique())
    bin_count = int(sample["word_count_bin"].nunique())
    word_summary = sample.groupby(label_column)[word_count_column].agg(["count", "mean", "median"]).round(2)
    print(
        f"Created prompt-and-word-count-matched sample with {len(sample)} rows "
        f"from {prompt_count} prompts and {bin_count} word-count bins."
    )
    print("\nWord-count summary:")
    print(word_summary.to_string())

    return sample


def load_unseen_prompt_split(
    csv_path,
    text_column,
    label_column,
    prompt_column,
    rows_per_label,
    test_size,
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
        raise ValueError("Unseen-prompt split requires both Human and AI rows.")

    prompt_counts["matched_rows"] = prompt_counts[["Human", "AI"]].min(axis=1)
    prompt_counts = prompt_counts[prompt_counts["matched_rows"] > 0]

    if prompt_counts.empty:
        raise ValueError("No prompts had both Human and AI rows.")

    target_rows_per_label = rows_per_label or int(prompt_counts["matched_rows"].sum())
    test_rows_per_label = max(1, int(round(target_rows_per_label * test_size)))
    train_rows_per_label = max(1, target_rows_per_label - test_rows_per_label)
    shuffled_counts = prompt_counts.sample(frac=1, random_state=random_state)

    train_quotas = {}
    test_quotas = {}
    train_total = 0
    test_total = 0

    for prompt_id, row in shuffled_counts.iterrows():
        prompt_quota = int(row["matched_rows"])

        if test_total < test_rows_per_label:
            quota = min(prompt_quota, test_rows_per_label - test_total)
            test_quotas[prompt_id] = quota
            test_total += quota
        elif train_total < train_rows_per_label:
            quota = min(prompt_quota, train_rows_per_label - train_total)
            train_quotas[prompt_id] = quota
            train_total += quota

        if train_total >= train_rows_per_label and test_total >= test_rows_per_label:
            break

    data_columns = sorted({text_column, prompt_column, label_source_column})
    train_parts = []
    test_parts = []
    train_counts = {(prompt_id, label): 0 for prompt_id in train_quotas for label in ["Human", "AI"]}
    test_counts = {(prompt_id, label): 0 for prompt_id in test_quotas for label in ["Human", "AI"]}
    selected_prompts = set(train_quotas).union(test_quotas)

    for chunk_number, chunk in enumerate(pd.read_csv(csv_path, usecols=data_columns, chunksize=chunk_size)):
        chunk = prepare_labels(chunk, label_column)
        chunk = chunk[[text_column, prompt_column, label_column]].dropna()
        chunk = chunk[chunk[prompt_column].isin(selected_prompts)]

        if chunk.empty:
            continue

        for (prompt_id, label), rows in chunk.groupby([prompt_column, label_column]):
            if prompt_id in test_quotas:
                target_count = test_quotas[prompt_id]
                already_selected = test_counts.get((prompt_id, label), 0)
                needed = target_count - already_selected

                if needed > 0:
                    take_count = min(needed, len(rows))
                    selected = rows.sample(n=take_count, random_state=random_state + chunk_number)
                    test_parts.append(selected)
                    test_counts[(prompt_id, label)] = already_selected + take_count

            if prompt_id in train_quotas:
                target_count = train_quotas[prompt_id]
                already_selected = train_counts.get((prompt_id, label), 0)
                needed = target_count - already_selected

                if needed > 0:
                    take_count = min(needed, len(rows))
                    selected = rows.sample(n=take_count, random_state=random_state + chunk_number)
                    train_parts.append(selected)
                    train_counts[(prompt_id, label)] = already_selected + take_count

    if not train_parts or not test_parts:
        raise ValueError("Unseen-prompt split did not select both train and test rows.")

    train_sample = pd.concat(train_parts, ignore_index=True).sample(frac=1, random_state=random_state).reset_index(drop=True)
    test_sample = pd.concat(test_parts, ignore_index=True).sample(frac=1, random_state=random_state).reset_index(drop=True)
    overlap = set(train_sample[prompt_column]).intersection(set(test_sample[prompt_column]))

    if overlap:
        raise ValueError("Train and test prompts overlap.")

    print(
        f"Created unseen-prompt split with {len(train_sample)} train rows "
        f"from {train_sample[prompt_column].nunique()} prompts and {len(test_sample)} test rows "
        f"from {test_sample[prompt_column].nunique()} unseen prompts."
    )

    return train_sample, test_sample


def load_unseen_ai_source_split(
    csv_path,
    text_column,
    label_column,
    source_column,
    unseen_ai_source,
    rows_per_label,
    test_size,
    random_state=42,
    chunk_size=100000,
):
    """Create a balanced split whose test AI texts come from one held-out source."""
    if not 0 < test_size < 1:
        raise ValueError("--test-size must be between 0 and 1 for an unseen-AI-source split.")

    source_key = str(unseen_ai_source).strip().casefold()
    if source_key == "human":
        raise ValueError("--unseen-ai-source must name an AI source, not 'Human'.")

    label_source_column = "source" if label_column == "label" else label_column
    count_columns = sorted({source_column, label_source_column})
    source_counts = {}

    for chunk in pd.read_csv(csv_path, usecols=count_columns, chunksize=chunk_size):
        chunk = prepare_labels(chunk, label_column)
        chunk = chunk[[source_column, label_column]].dropna()
        chunk["source_key"] = chunk[source_column].astype(str).str.strip().str.casefold()
        counts = chunk.groupby(["source_key", label_column]).size()
        for (key, label), count in counts.items():
            source_counts[(key, label)] = source_counts.get((key, label), 0) + int(count)

    held_out_ai_rows = source_counts.get((source_key, "AI"), 0)
    other_ai_rows = sum(
        count for (key, label), count in source_counts.items() if label == "AI" and key != source_key
    )
    human_rows = sum(count for (_, label), count in source_counts.items() if label == "Human")

    if held_out_ai_rows == 0:
        available_sources = sorted(
            key for (key, label) in source_counts if label == "AI"
        )
        raise ValueError(
            f"No AI rows were found for '{unseen_ai_source}' in '{source_column}'. "
            f"Available AI sources: {', '.join(available_sources) or 'none'}"
        )
    if other_ai_rows == 0 or human_rows == 0:
        raise ValueError("This experiment needs Human rows, the held-out AI source, and at least one other AI source.")

    if rows_per_label is None:
        test_rows_per_label = min(held_out_ai_rows, human_rows)
        train_rows_per_label = min(other_ai_rows, human_rows - test_rows_per_label)
    else:
        requested_test_rows = max(1, int(round(rows_per_label * test_size)))
        test_rows_per_label = min(requested_test_rows, held_out_ai_rows, human_rows)
        train_rows_per_label = min(rows_per_label - test_rows_per_label, other_ai_rows, human_rows - test_rows_per_label)

    if train_rows_per_label < 1 or test_rows_per_label < 1:
        raise ValueError("There are not enough balanced rows to create train and test sets for this experiment.")

    def keep_sample(existing, candidate, limit, seed):
        combined = candidate if existing is None else pd.concat([existing, candidate], ignore_index=True)
        if len(combined) > limit:
            return combined.sample(n=limit, random_state=seed)
        return combined

    data_columns = sorted({text_column, source_column, label_source_column})
    train_ai = None
    test_ai = None
    human = None
    needed_human_rows = train_rows_per_label + test_rows_per_label

    for chunk_number, chunk in enumerate(pd.read_csv(csv_path, usecols=data_columns, chunksize=chunk_size)):
        chunk = prepare_labels(chunk, label_column)
        chunk = chunk[[text_column, source_column, label_column]].dropna()
        chunk["source_key"] = chunk[source_column].astype(str).str.strip().str.casefold()

        human = keep_sample(
            human,
            chunk[chunk[label_column] == "Human"],
            needed_human_rows,
            random_state + chunk_number,
        )
        test_ai = keep_sample(
            test_ai,
            chunk[(chunk[label_column] == "AI") & (chunk["source_key"] == source_key)],
            test_rows_per_label,
            random_state + 10_000 + chunk_number,
        )
        train_ai = keep_sample(
            train_ai,
            chunk[(chunk[label_column] == "AI") & (chunk["source_key"] != source_key)],
            train_rows_per_label,
            random_state + 20_000 + chunk_number,
        )

    if any(part is None for part in (train_ai, test_ai, human)):
        raise ValueError("Could not collect rows for the unseen-AI-source split.")

    human = human.sample(frac=1, random_state=random_state).reset_index(drop=True)
    test_human = human.iloc[:test_rows_per_label]
    train_human = human.iloc[test_rows_per_label : test_rows_per_label + train_rows_per_label]
    train_sample = pd.concat([train_ai, train_human], ignore_index=True).sample(
        frac=1, random_state=random_state
    ).reset_index(drop=True)
    test_sample = pd.concat([test_ai, test_human], ignore_index=True).sample(
        frac=1, random_state=random_state
    ).reset_index(drop=True)

    print(
        f"Created unseen-AI-source split holding out '{unseen_ai_source}': "
        f"{len(train_sample)} train rows (AI from other sources) and {len(test_sample)} test rows "
        f"(AI only from '{unseen_ai_source}')."
    )
    return train_sample, test_sample


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
    parser.add_argument(
        "--source-column",
        default="source",
        help="Name of the column identifying the original human or AI source/model.",
    )
    parser.add_argument("--word-count-column", default="word_count", help="Name of the word count column.")
    parser.add_argument(
        "--word-count-bin-size",
        type=int,
        default=25,
        help="Word-count range size used when matching texts by similar length.",
    )
    parser.add_argument(
        "--max-rows-per-word-count-bin",
        type=int,
        default=100,
        help="Maximum Human and AI rows sampled from each word-count range.",
    )
    parser.add_argument(
        "--max-rows-per-prompt-word-count-bin",
        type=int,
        default=25,
        help="Maximum Human and AI rows sampled from each prompt and word-count group.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of rows used for testing.")
    parser.add_argument("--max-features", type=int, default=30000, help="Maximum TF-IDF features.")
    parser.add_argument(
        "--match-prompt",
        action="store_true",
        help="Build a sample with Human and AI texts matched by the same prompt.",
    )
    parser.add_argument(
        "--match-word-count",
        action="store_true",
        help="Build a sample with Human and AI texts matched by similar word counts.",
    )
    parser.add_argument(
        "--unseen-prompts",
        action="store_true",
        help="Train on one set of prompts and test on completely unseen prompts.",
    )
    parser.add_argument(
        "--unseen-ai-source",
        help="Hold out this AI source/model during training and use it only for the test AI texts.",
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
    split_experiments = sum(bool(value) for value in (args.unseen_prompts, args.unseen_ai_source))
    if split_experiments and (args.match_prompt or args.match_word_count):
        raise ValueError("Use --unseen-prompts or --unseen-ai-source by itself.")
    if split_experiments > 1:
        raise ValueError("Choose either --unseen-prompts or --unseen-ai-source, not both.")

    if args.unseen_ai_source:
        train_sample, test_sample = load_unseen_ai_source_split(
            csv_path,
            text_column=args.text_column,
            label_column=args.label_column,
            source_column=args.source_column,
            unseen_ai_source=args.unseen_ai_source,
            rows_per_label=rows_per_label,
            test_size=args.test_size,
        )
        results_table, reports, confusion_matrices, trained_models = train_binary_text_classifiers_on_split(
            train_sample,
            test_sample,
            text_column=args.text_column,
            label_column=args.label_column,
            max_features=args.max_features,
        )
    elif args.unseen_prompts:
        train_sample, test_sample = load_unseen_prompt_split(
            csv_path,
            text_column=args.text_column,
            label_column=args.label_column,
            prompt_column=args.prompt_column,
            rows_per_label=rows_per_label,
            test_size=args.test_size,
        )
        results_table, reports, confusion_matrices, trained_models = train_binary_text_classifiers_on_split(
            train_sample,
            test_sample,
            text_column=args.text_column,
            label_column=args.label_column,
            max_features=args.max_features,
        )
    elif args.match_prompt and args.match_word_count:
        sample = load_prompt_and_word_count_matched_sample(
            csv_path,
            text_column=args.text_column,
            label_column=args.label_column,
            prompt_column=args.prompt_column,
            word_count_column=args.word_count_column,
            word_count_bin_size=args.word_count_bin_size,
            max_rows_per_prompt_word_count_bin=args.max_rows_per_prompt_word_count_bin,
            rows_per_label=rows_per_label,
        )
        results_table, reports, confusion_matrices, trained_models = train_binary_text_classifiers(
            sample,
            text_column=args.text_column,
            label_column=args.label_column,
            test_size=args.test_size,
            max_features=args.max_features,
        )
    elif args.match_prompt:
        sample = load_prompt_matched_sample(
            csv_path,
            text_column=args.text_column,
            label_column=args.label_column,
            prompt_column=args.prompt_column,
            rows_per_label=rows_per_label,
        )
        results_table, reports, confusion_matrices, trained_models = train_binary_text_classifiers(
            sample,
            text_column=args.text_column,
            label_column=args.label_column,
            test_size=args.test_size,
            max_features=args.max_features,
        )
    elif args.match_word_count:
        sample = load_word_count_matched_sample(
            csv_path,
            text_column=args.text_column,
            label_column=args.label_column,
            word_count_column=args.word_count_column,
            word_count_bin_size=args.word_count_bin_size,
            max_rows_per_word_count_bin=args.max_rows_per_word_count_bin,
            rows_per_label=rows_per_label,
        )
        results_table, reports, confusion_matrices, trained_models = train_binary_text_classifiers(
            sample,
            text_column=args.text_column,
            label_column=args.label_column,
            test_size=args.test_size,
            max_features=args.max_features,
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
