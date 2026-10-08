"""Put the LLMTrace classifier and boundary-detection results in one table."""

import pandas as pd


def main():
    classifier_results = pd.read_csv("llmtrace_window_model_results.csv")
    boundary_results = pd.read_csv("llmtrace_boundary_results.csv")
    best_classifier = classifier_results.loc[classifier_results["macro_f1"].idxmax()]
    boundary = boundary_results.iloc[0]

    summary = pd.DataFrame(
        [
            {
                "stage": "Window classification",
                "best_model": best_classifier["model"],
                "main_measure": "Accuracy",
                "score_percent": round(100 * best_classifier["accuracy"], 2),
                "details": "Trained on pure Human/AI windows; tested on mixed-document windows.",
            },
            {
                "stage": "Boundary detection",
                "best_model": boundary["model"],
                "main_measure": "Boundary F1",
                "score_percent": boundary["boundary_f1_percent"],
                "details": (
                    f"100-word windows; a match is within {boundary['tolerance_chars']} characters of "
                    "a true Human/AI transition."
                ),
            },
        ]
    )
    summary.to_csv("llmtrace_experiment_summary.csv", index=False)
    print(summary.to_string(index=False))
    print("\nSaved summary table to llmtrace_experiment_summary.csv")


if __name__ == "__main__":
    main()
