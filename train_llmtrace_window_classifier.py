"""Train Human-vs-AI window classifiers and test them on mixed LLMTrace documents."""

import argparse
from pathlib import Path

import joblib
import pandas as pd
from datasets import load_dataset

from code_snippets import train_binary_text_classifiers_on_split
from prepare_llmtrace_windows import DATASET_NAME, make_windows


def collect_windows(split, document_labels, window_words, max_per_label, max_documents, random_state):
    """Collect a balanced, deterministic sample of labeled windows from documents."""
    rows = []

    selected_documents = 0
    for document_number, document in enumerate(split):
        if document["label"] not in document_labels:
            continue

        if max_documents > 0 and selected_documents >= max_documents:
            break
        selected_documents += 1

        for window in make_windows(document["text"], document["ai_char_intervals"], window_words):
            rows.append(
                {
                    "document_number": document_number,
                    "document_label": document["label"],
                    "text": window["text"],
                    "label": window["label"],
                }
            )

    windows = pd.DataFrame(rows)
    if windows.empty:
        raise ValueError("No windows were created from the selected documents.")

    labels = {"Human", "AI"}
    if not labels.issubset(windows["label"].unique()):
        raise ValueError("The selected data must contain both Human and AI windows.")

    sampled = []
    for label in sorted(labels):
        label_windows = windows[windows["label"] == label]
        take = min(max_per_label, len(label_windows))
        sampled.append(label_windows.sample(n=take, random_state=random_state))

    sampled_windows = pd.concat(sampled, ignore_index=True).sample(
        frac=1, random_state=random_state
    ).reset_index(drop=True)
    print(
        f"Collected {len(sampled_windows):,} windows from {selected_documents:,} documents "
        f"with labels {sorted(document_labels)}.",
        flush=True,
    )
    return sampled_windows


def main():
    parser = argparse.ArgumentParser(
        description="Train on pure LLMTrace documents and evaluate on windows from mixed documents."
    )
    parser.add_argument("--window-words", type=int, default=100)
    parser.add_argument("--train-windows-per-label", type=int, default=6000)
    parser.add_argument("--test-windows-per-label", type=int, default=1500)
    parser.add_argument(
        "--max-train-documents",
        type=int,
        default=12000,
        help="Number of pure training documents to inspect; use 0 for all.",
    )
    parser.add_argument(
        "--max-test-documents",
        type=int,
        default=6000,
        help="Number of mixed validation documents to inspect; use 0 for all.",
    )
    parser.add_argument("--max-features", type=int, default=30000)
    parser.add_argument("--results", default="llmtrace_window_model_results.csv")
    parser.add_argument("--model-output", default="models/llmtrace_window_best_model.joblib")
    args = parser.parse_args()

    dataset = load_dataset(DATASET_NAME)
    train_windows = collect_windows(
        dataset["train"],
        document_labels={"human", "ai"},
        window_words=args.window_words,
        max_per_label=args.train_windows_per_label,
        max_documents=args.max_train_documents,
        random_state=42,
    )
    test_windows = collect_windows(
        dataset["validation"],
        document_labels={"mixed"},
        window_words=args.window_words,
        max_per_label=args.test_windows_per_label,
        max_documents=args.max_test_documents,
        random_state=43,
    )

    results, reports, matrices, models = train_binary_text_classifiers_on_split(
        train_windows,
        test_windows,
        text_column="text",
        label_column="label",
        max_features=args.max_features,
    )
    results.to_csv(args.results, index=False)

    best_model_name = results.loc[0, "model"]
    model_path = Path(args.model_output)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(models[best_model_name], model_path)

    print(f"Training windows: {len(train_windows):,}")
    print(f"Mixed-document test windows: {len(test_windows):,}")
    print("\nModel comparison:")
    print(results.round(4).to_string(index=False))
    print(f"\nBest model: {best_model_name}")
    print(f"Saved result table to {args.results}")
    print(f"Saved best model to {model_path}")
    print("\nBest-model confusion matrix:")
    print(matrices[best_model_name].to_string())


if __name__ == "__main__":
    main()
