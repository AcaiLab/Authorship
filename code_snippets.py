from pathlib import Path

import numpy as np
import pandas as pd


def read_data_in_chunks(paths, columns=None, chunk_size=200000):
    return pd.read_csv(paths["data_file"], usecols=columns, chunksize=chunk_size)


# convert source labels into either human or ai
def make_source_label_map(distribution):
    label_map = {}

    for source in distribution["Source"]:
        if source == "Human":
            label_map[source] = "Human"
        else:
            label_map[source] = "AI"

    return label_map


# add simple text features: character count, word count, sentence count
# avg word length, avg sentence length
def add_basic_text_features(data, text_column="text"):
    result = data.copy()

    text = result[text_column].fillna("").astype(str)

    result["char_len"] = text.str.len()
    result["word_len"] = text.str.split().str.len()
    result["sentence_count"] = text.str.count(r"[.!?]+").clip(lower=1)

    characters_without_spaces = text.str.replace(r"\s+", "", regex=True).str.len()
    result["avg_word_len"] = characters_without_spaces / result["word_len"].clip(lower=1)

    result["avg_sentence_words"] = result["word_len"] / result["sentence_count"]

    return result


# balanced human vs ai sample from larger dataset
def load_balanced_sample(paths, rows_per_label=12000, random_state=42, save_cache=True):
    cache_file = paths["processed_folder"] / f"balanced_human_ai_sample_{rows_per_label}_per_label.csv"

    if save_cache and cache_file.exists():
        return pd.read_csv(cache_file)

    distribution = load_distribution(paths)
    label_map = make_source_label_map(distribution)

    columns_to_read = ["text", "source", "prompt_id", "text_length", "word_count"]
    sample_parts = {}
    rng = np.random.default_rng(random_state)

    for chunk in read_data_in_chunks(paths, columns=columns_to_read):
        chunk = chunk.dropna(subset=["text", "source"]).copy()

        # original source name, then create the simple Human/AI label.
        chunk["original_source"] = chunk["source"]
        chunk["label"] = chunk["source"].map(label_map)
        chunk = chunk.dropna(subset=["label"])

        for label, label_rows in chunk.groupby("label"):
            old_rows = sample_parts.get(label)

            if old_rows is None:
                candidate_rows = label_rows
            else:
                candidate_rows = pd.concat([old_rows, label_rows], ignore_index=True)

            # if the temporary sample is too large, randomly reduce it.
            if len(candidate_rows) > rows_per_label:
                seed = int(rng.integers(0, 1_000_000_000))
                candidate_rows = candidate_rows.sample(n=rows_per_label, random_state=seed)

            sample_parts[label] = candidate_rows

    sample = pd.concat(sample_parts.values(), ignore_index=True)
    sample = sample.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Rename columns to make the final sample easier to read.
    sample = sample.rename(columns={"source": "old_source_column", "original_source": "source"})
    sample = sample.drop(columns=["old_source_column"])

    sample = add_basic_text_features(sample)

    if save_cache:
        sample.to_csv(cache_file, index=False)

    return sample


def summarize_sources(paths):
    distribution = load_distribution(paths)

    total_rows = int(distribution["Number of Samples"].sum())
    human_rows = int(distribution.loc[distribution["Source"] == "Human", "Number of Samples"].sum())
    ai_rows = total_rows - human_rows

    summary = {
        "total_rows": total_rows,
        "human_rows": human_rows,
        "ai_rows": ai_rows,
        "number_of_sources": int(distribution["Source"].nunique()),
    }

    return summary, distribution


def summarize_text_lengths(sample):
    summary = (
        sample.groupby("label")
        .agg(
            rows=("text", "size"),
            word_mean=("word_len", "mean"),
            word_median=("word_len", "median"),
            word_p90=("word_len", lambda values: values.quantile(0.90)),
            word_p99=("word_len", lambda values: values.quantile(0.99)),
            char_mean=("char_len", "mean"),
        )
        .round(2)
    )

    return summary


def summarize_prompt_coverage(sample, paths):
    prompts = load_prompts(paths)

    prompt_counts = (
        sample.groupby(["label", "prompt_id"])
        .size()
        .reset_index(name="rows")
    )

    prompt_counts = prompt_counts.merge(
        prompts,
        left_on="prompt_id",
        right_on="Prompt ID",
        how="left",
    )

    coverage = (
        prompt_counts.groupby("label")["prompt_id"]
        .nunique()
        .reset_index(name="unique_prompt_ids")
    )

    return coverage, prompt_counts


def find_exact_text_overlap(sample):
    normalized_text = (
        sample["text"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    human_texts = set(normalized_text[sample["label"] == "Human"])
    ai_texts = set(normalized_text[sample["label"] == "AI"])

    return len(human_texts.intersection(ai_texts))


def get_top_terms(sample, label_column="label", text_column="text", top_count=20):
    from sklearn.feature_extraction.text import CountVectorizer

    all_results = []

    for label, rows_for_label in sample.groupby(label_column):
        vectorizer = CountVectorizer(
            stop_words="english",
            max_features=5000,
            ngram_range=(1, 2),
            min_df=3,
        )

        text_for_label = rows_for_label[text_column].fillna("").astype(str).str.slice(0, 20_000)
        term_matrix = vectorizer.fit_transform(text_for_label)

        counts = np.asarray(term_matrix.sum(axis=0)).ravel()
        terms = np.array(vectorizer.get_feature_names_out())

        term_table = pd.DataFrame({"term": terms, "count": counts})
        term_table = term_table.sort_values("count", ascending=False).head(top_count)
        term_table[label_column] = label

        all_results.append(term_table)

    return pd.concat(all_results, ignore_index=True)



def make_tfidf_svd_embeddings(sample, rows_per_label=2500, random_state=42):
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    human_rows = sample[sample["label"] == "Human"].sample(n=rows_per_label, random_state=random_state)
    ai_rows = sample[sample["label"] == "AI"].sample(n=rows_per_label, random_state=random_state)

    embedding_sample = pd.concat([human_rows, ai_rows], ignore_index=True)
    embedding_sample = embedding_sample.sample(frac=1, random_state=random_state).reset_index(drop=True)
    embedding_sample["model_text"] = embedding_sample["text"].fillna("").astype(str).str.slice(0, 20000)

    vectorizer = TfidfVectorizer(
        max_features=30000,
        min_df=3,
        max_df=0.9,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    tfidf_matrix = vectorizer.fit_transform(embedding_sample["model_text"])

    svd = TruncatedSVD(n_components=100, random_state=random_state)
    embedding_matrix = svd.fit_transform(tfidf_matrix)
    # nomarlize
    embedding_matrix = normalize(embedding_matrix)

    return embedding_sample, embedding_matrix, vectorizer, svd


def compare_embedding_distances(embedding_sample, embedding_matrix, pair_count=3000, random_state=42):
    rng = np.random.default_rng(random_state)
    labels = embedding_sample["label"].to_numpy()

    pair_definitions = [
        ("Human-Human", "Human", "Human"),
        ("AI-AI", "AI", "AI"),
        ("Human-AI", "Human", "AI"),
    ]

    rows = []

    for pair_name, left_label, right_label in pair_definitions:
        left_indices = np.flatnonzero(labels == left_label)
        right_indices = np.flatnonzero(labels == right_label)

        sampled_left = rng.choice(left_indices, size=pair_count, replace=True)
        sampled_right = rng.choice(right_indices, size=pair_count, replace=True)

        similarities = np.sum(embedding_matrix[sampled_left] * embedding_matrix[sampled_right], axis=1)

        for similarity in similarities:
            rows.append(
                {
                    "pair": pair_name,
                    "cosine_similarity": similarity,
                    "cosine_distance": 1 - similarity,
                }
            )

    return pd.DataFrame(rows)


def evaluate_classifiers(X_train, X_test, y_train, y_test, max_features=30000, random_state=42):
    """Fit the standard model comparison suite on an already-built train/test split. """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.naive_bayes import ComplementNB, MultinomialNB
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC

    tfidf_settings = {
        "max_features": max_features,
        "stop_words": "english",
        "ngram_range": (1, 2),
        "min_df": 2,
        "sublinear_tf": True,
    }

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "Linear SVM": LinearSVC(random_state=random_state),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Complement Naive Bayes": ComplementNB(),
        "SGD Linear Classifier": SGDClassifier(
            loss="modified_huber",
            max_iter=1000,
            random_state=random_state,
        ),
    }

    label_order = ["Human", "AI"] if set(y_train.unique()) | set(y_test.unique()) == {"Human", "AI"} else None
    results = []
    reports = {}
    confusion_matrices = {}
    trained_models = {}

    for model_name, model in models.items():
        pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(**tfidf_settings)),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
        matrix = confusion_matrix(y_test, predictions, labels=label_order)

        results.append(
            {
                "model": model_name,
                "accuracy": accuracy_score(y_test, predictions),
                "macro_precision": report["macro avg"]["precision"],
                "macro_recall": report["macro avg"]["recall"],
                "macro_f1": report["macro avg"]["f1-score"],
                "weighted_f1": report["weighted avg"]["f1-score"],
                "n_train": len(X_train),
                "n_test": len(X_test),
            }
        )

        reports[model_name] = classification_report(y_test, predictions, zero_division=0)
        confusion_matrices[model_name] = pd.DataFrame(
            matrix,
            index=[f"actual_{label}" for label in label_order] if label_order else None,
            columns=[f"predicted_{label}" for label in label_order] if label_order else None,
        )
        trained_models[model_name] = pipeline

    results_table = pd.DataFrame(results).sort_values("macro_f1", ascending=False).reset_index(drop=True)

    return results_table, reports, confusion_matrices, trained_models


def train_binary_text_classifiers(
    sample,
    text_column="text",
    label_column="label",
    test_size=0.2,
    random_state=42,
    max_features=30000,
):
    from sklearn.model_selection import train_test_split

    required_columns = {text_column, label_column}
    missing_columns = required_columns.difference(sample.columns)

    if missing_columns:
        raise ValueError(f"Sample is missing required columns: {sorted(missing_columns)}")

    data = sample[[text_column, label_column]].dropna().copy()
    data[text_column] = data[text_column].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        data[text_column],
        data[label_column],
        test_size=test_size,
        random_state=random_state,
        stratify=data[label_column],
    )

    return evaluate_classifiers(
        X_train, X_test, y_train, y_test, max_features=max_features, random_state=random_state
    )


def print_binary_classification_results(results_table, reports, confusion_matrices):
    print("\nModel comparison:")
    print(results_table.round(4).to_string(index=False))

    best_model = results_table.loc[0, "model"]
    print(f"\nBest model by macro F1: {best_model}")

    print("\nClassification report for best model:")
    print(reports[best_model])

    print("Confusion matrix for best model:")
    print(confusion_matrices[best_model].to_string())
             