"""Split LLMTrace documents into labeled word windows for authorship detection."""

import argparse
import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset


DATASET_NAME = "iitolstykh/LLMTrace_detection"
WORD_PATTERN = re.compile(r"\S+")


def interval_overlap(start, end, intervals):
    """Return the number of characters in [start, end) covered by AI intervals."""
    return sum(max(0, min(end, ai_end) - max(start, ai_start)) for ai_start, ai_end in intervals)


def make_windows(text, ai_char_intervals, window_words=100, ai_threshold=0.5):
    """Return consecutive word windows labeled by their majority authorship.

    A window is AI when at least ``ai_threshold`` of its characters overlap an
    annotated AI interval. Otherwise, it is Human. Window positions remain in
    original-document character coordinates for later boundary evaluation.
    """
    if window_words < 1:
        raise ValueError("window_words must be at least 1.")
    if not 0 < ai_threshold <= 1:
        raise ValueError("ai_threshold must be greater than 0 and at most 1.")

    text = str(text)
    intervals = [(int(start), int(end)) for start, end in ai_char_intervals]
    words = list(WORD_PATTERN.finditer(text))
    windows = []

    for window_number, first_word in enumerate(range(0, len(words), window_words)):
        window_words_matches = words[first_word : first_word + window_words]
        start = window_words_matches[0].start()
        end = window_words_matches[-1].end()
        window_text = text[start:end]
        ai_characters = interval_overlap(start, end, intervals)
        ai_fraction = ai_characters / max(1, end - start)

        windows.append(
            {
                "window_number": window_number,
                "window_start": start,
                "window_end": end,
                "text": window_text,
                "ai_character_fraction": ai_fraction,
                "label": "AI" if ai_fraction >= ai_threshold else "Human",
            }
        )

    return windows


def main():
    parser = argparse.ArgumentParser(description="Create labeled word windows from LLMTrace documents.")
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"])
    parser.add_argument("--window-words", type=int, default=100, help="Words in each consecutive window.")
    parser.add_argument(
        "--max-documents",
        type=int,
        default=1000,
        help="Maximum documents to process; use 0 to process the entire split.",
    )
    parser.add_argument(
        "--output",
        default="data/llmtrace_train_windows.csv",
        help="CSV file for the labeled windows.",
    )
    args = parser.parse_args()

    split = load_dataset(DATASET_NAME, split=args.split)
    if args.max_documents > 0:
        split = split.select(range(min(args.max_documents, len(split))))

    rows = []
    for document_number, document in enumerate(split):
        for window in make_windows(document["text"], document["ai_char_intervals"], args.window_words):
            rows.append(
                {
                    "document_number": document_number,
                    "document_label": document["label"],
                    "source_model": document["model"],
                    **window,
                }
            )

    windows = pd.DataFrame(rows)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    windows.to_csv(output_path, index=False)

    print(f"Created {len(windows):,} windows from {len(split):,} {args.split} documents.")
    print("Window labels:")
    print(windows["label"].value_counts().to_string())
    print(f"Saved windows to {output_path}")


if __name__ == "__main__":
    main()
