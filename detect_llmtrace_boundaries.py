"""Detect Human/AI authorship boundaries in mixed LLMTrace documents."""

import argparse
from pathlib import Path

import joblib
import pandas as pd
from datasets import load_dataset

from prepare_llmtrace_windows import DATASET_NAME, make_windows


def true_boundaries(ai_char_intervals, text_length):
    """Return internal points where ground-truth authorship switches."""
    boundaries = set()
    for start, end in ai_char_intervals:
        if 0 < start < text_length:
            boundaries.add(int(start))
        if 0 < end < text_length:
            boundaries.add(int(end))
    return sorted(boundaries)


def match_boundaries(predicted, actual, tolerance_chars):
    """Greedily match each predicted point to at most one nearby true boundary."""
    unused_actual = set(actual)
    matched_distances = []

    for prediction in predicted:
        nearby = [point for point in unused_actual if abs(prediction - point) <= tolerance_chars]
        if nearby:
            closest = min(nearby, key=lambda point: abs(prediction - point))
            unused_actual.remove(closest)
            matched_distances.append(abs(prediction - closest))

    return matched_distances


def main():
    parser = argparse.ArgumentParser(description="Evaluate authorship-boundary detection on mixed LLMTrace documents.")
    parser.add_argument("--model", default="models/llmtrace_window_best_model.joblib")
    parser.add_argument("--window-words", type=int, default=100)
    parser.add_argument("--tolerance-chars", type=int, default=500)
    parser.add_argument(
        "--max-documents",
        type=int,
        default=1000,
        help="Mixed test documents to evaluate; use 0 to evaluate all.",
    )
    parser.add_argument("--results", default="llmtrace_boundary_results.csv")
    parser.add_argument("--document-results", default="llmtrace_boundary_document_results.csv")
    args = parser.parse_args()

    model = joblib.load(args.model)
    test_split = load_dataset(DATASET_NAME, split="test")
    mixed_documents = [document for document in test_split if document["label"] == "mixed"]
    if args.max_documents > 0:
        mixed_documents = mixed_documents[: args.max_documents]

    document_rows = []
    predicted_windows = 0
    correct_windows = 0
    total_predicted_boundaries = 0
    total_actual_boundaries = 0
    total_matches = 0
    all_distances = []

    for document_number, document in enumerate(mixed_documents):
        windows = make_windows(document["text"], document["ai_char_intervals"], args.window_words)
        if len(windows) < 2:
            continue

        predicted_labels = model.predict([window["text"] for window in windows])
        actual_labels = [window["label"] for window in windows]
        predicted_points = [
            windows[index]["window_start"]
            for index in range(1, len(windows))
            if predicted_labels[index] != predicted_labels[index - 1]
        ]
        actual_points = true_boundaries(document["ai_char_intervals"], len(document["text"]))
        distances = match_boundaries(predicted_points, actual_points, args.tolerance_chars)

        predicted_windows += len(windows)
        correct_windows += sum(predicted == actual for predicted, actual in zip(predicted_labels, actual_labels))
        total_predicted_boundaries += len(predicted_points)
        total_actual_boundaries += len(actual_points)
        total_matches += len(distances)
        all_distances.extend(distances)

        document_rows.append(
            {
                "document_number": document_number,
                "source_model": document["model"],
                "windows": len(windows),
                "predicted_boundaries": len(predicted_points),
                "true_boundaries": len(actual_points),
                "matched_boundaries": len(distances),
                "mean_matched_distance_chars": round(sum(distances) / len(distances), 2) if distances else None,
            }
        )

    precision = total_matches / total_predicted_boundaries if total_predicted_boundaries else 0.0
    recall = total_matches / total_actual_boundaries if total_actual_boundaries else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    summary = pd.DataFrame(
        [
            {
                "experiment": "Mixed-document boundary detection",
                "model": type(model.named_steps["model"]).__name__,
                "window_words": args.window_words,
                "tolerance_chars": args.tolerance_chars,
                "documents_evaluated": len(document_rows),
                "window_accuracy_percent": round(100 * correct_windows / predicted_windows, 2),
                "boundary_precision_percent": round(100 * precision, 2),
                "boundary_recall_percent": round(100 * recall, 2),
                "boundary_f1_percent": round(100 * f1, 2),
                "mean_matched_distance_chars": round(sum(all_distances) / len(all_distances), 2)
                if all_distances
                else None,
                "predicted_boundaries": total_predicted_boundaries,
                "true_boundaries": total_actual_boundaries,
            }
        ]
    )

    summary.to_csv(args.results, index=False)
    pd.DataFrame(document_rows).to_csv(args.document_results, index=False)
    print("\nBoundary-detection summary:")
    print(summary.to_string(index=False))
    print(f"\nSaved summary table to {args.results}")
    print(f"Saved per-document results to {args.document_results}")


if __name__ == "__main__":
    main()
