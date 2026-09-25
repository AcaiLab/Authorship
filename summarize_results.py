import argparse
from pathlib import Path

import pandas as pd


EXPERIMENTS = {
    "classification_results.csv": {
        "experiment": "Baseline random split",
        "setup": "Random train/test split",
        "control": "No prompt or length control",
        "sample": "Balanced Human vs AI sample",
        "purpose": "Initial baseline",
    },
    "prompt_matched_classification_results.csv": {
        "experiment": "Prompt matched",
        "setup": "Human and AI texts use the same prompts",
        "control": "Controls prompt/topic differences",
        "sample": "4,000 rows; 2,000 Human and 2,000 AI; 171 shared prompts",
        "purpose": "Check whether the model is only learning topic differences",
    },
    "word_count_matched_classification_results.csv": {
        "experiment": "Word-count matched",
        "setup": "Human and AI texts have similar word counts",
        "control": "Controls text length differences",
        "sample": "4,000 rows; 65 word-count bins; Human mean 786.51, AI mean 786.73",
        "purpose": "Check whether the model is relying on text length",
    },
    "prompt_and_word_count_matched_classification_results.csv": {
        "experiment": "Prompt + word-count matched",
        "setup": "Same prompts and similar word-count ranges",
        "control": "Controls both topic and length",
        "sample": "4,000 rows; 45 prompts; 41 word-count bins; Human mean 447.90, AI mean 447.89",
        "purpose": "Stronger controlled comparison",
    },
    "unseen_prompt_classification_results.csv": {
        "experiment": "Unseen-prompt test",
        "setup": "Train prompts and test prompts do not overlap",
        "control": "Tests generalization to new writing tasks",
        "sample": "3,200 train rows from 91 prompts; 800 test rows from 80 unseen prompts",
        "purpose": "Check whether performance holds on prompts not seen during training",
    },
}


def read_experiment(path):
    metadata = EXPERIMENTS[path.name]
    results = pd.read_csv(path)
    rows = []

    for rank, (_, row) in enumerate(results.iterrows(), start=1):
        rows.append(
            {
                "experiment": metadata["experiment"],
                "setup": metadata["setup"],
                "control": metadata["control"],
                "sample": metadata["sample"],
                "purpose": metadata["purpose"],
                "model_rank": rank,
                "model": row["model"],
                "accuracy_percent": round(float(row["accuracy"]) * 100, 2),
                "macro_precision_percent": round(float(row["macro_precision"]) * 100, 2),
                "macro_recall_percent": round(float(row["macro_recall"]) * 100, 2),
                "macro_f1_percent": round(float(row["macro_f1"]) * 100, 2),
                "weighted_f1_percent": round(float(row["weighted_f1"]) * 100, 2),
                "results_file": path.name,
            }
        )

    return rows


def save_markdown_table(dataframe, path):
    rows = []
    columns = list(dataframe.columns)
    rows.append("| " + " | ".join(columns) + " |")
    rows.append("| " + " | ".join(["---"] * len(columns)) + " |")

    for _, row in dataframe.iterrows():
        values = [str(row[column]).replace("|", "/") for column in columns]
        rows.append("| " + " | ".join(values) + " |")

    path.write_text("\n".join(rows) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Create comparison tables for Human vs AI experiments.")
    parser.add_argument("--summary-output", default="experiment_summary.csv")
    parser.add_argument("--full-output", default="experiment_model_comparison.csv")
    parser.add_argument("--summary-markdown", default="experiment_summary.md")
    parser.add_argument("--full-markdown", default="experiment_model_comparison.md")
    args = parser.parse_args()

    all_rows = []
    missing_files = []

    for filename in EXPERIMENTS:
        path = Path(filename)
        if path.exists():
            all_rows.extend(read_experiment(path))
        else:
            missing_files.append(filename)

    if not all_rows:
        raise FileNotFoundError("No experiment result files were found.")

    full_table = pd.DataFrame(all_rows)
    summary_table = full_table[full_table["model_rank"] == 1].copy()
    summary_table = summary_table[
        [
            "experiment",
            "setup",
            "control",
            "sample",
            "best_model",
            "accuracy_percent",
            "macro_f1_percent",
            "purpose",
            "results_file",
        ]
        if "best_model" in summary_table.columns
        else [
            "experiment",
            "setup",
            "control",
            "sample",
            "model",
            "accuracy_percent",
            "macro_f1_percent",
            "purpose",
            "results_file",
        ]
    ]

    if "model" in summary_table.columns:
        summary_table = summary_table.rename(columns={"model": "best_model"})

    full_table.to_csv(args.full_output, index=False)
    summary_table.to_csv(args.summary_output, index=False)
    save_markdown_table(full_table, Path(args.full_markdown))
    save_markdown_table(summary_table, Path(args.summary_markdown))

    print("\nBest model summary:")
    print(summary_table.to_string(index=False))
    print(f"\nSaved {args.summary_output}")
    print(f"Saved {args.full_output}")
    print(f"Saved {args.summary_markdown}")
    print(f"Saved {args.full_markdown}")

    if missing_files:
        print("\nMissing result files:")
        for filename in missing_files:
            print(f"- {filename}")


if __name__ == "__main__":
    main()
