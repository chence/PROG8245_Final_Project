from __future__ import annotations

import argparse
from dataclasses import dataclass

import joblib
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.config import get_config
from src.utils import DenseTransformer, ensure_directory, save_json

RANDOM_STATE = 42
MODEL_NAMES = ("baseline_nb", "svd_logreg", "pca_logreg")


@dataclass
class TrainingParams:
    tfidf_max_features: int = 4000
    tfidf_ngram_max: int = 2
    svd_components: int = 100
    pca_components: int = 100
    logistic_max_iter: int = 2000
def _vectorizer(params: TrainingParams) -> TfidfVectorizer:
    return TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, params.tfidf_ngram_max),
        max_features=params.tfidf_max_features,
    )


def build_model(model_name: str, params: TrainingParams) -> Pipeline:
    vectorizer = _vectorizer(params)
    if model_name == "baseline_nb":
        return Pipeline(
            [
                ("tfidf", vectorizer),
                ("clf", MultinomialNB()),
            ]
        )
    if model_name == "svd_logreg":
        return Pipeline(
            [
                ("tfidf", vectorizer),
                ("svd", TruncatedSVD(n_components=params.svd_components, random_state=RANDOM_STATE)),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=params.logistic_max_iter,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
    if model_name == "pca_logreg":
        return Pipeline(
            [
                ("tfidf", vectorizer),
                ("dense", DenseTransformer()),
                ("pca", PCA(n_components=params.pca_components, random_state=RANDOM_STATE)),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=params.logistic_max_iter,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unsupported model name: {model_name}")


def train_models(train_path: str, model_dir: str | None = None, params: TrainingParams | None = None) -> list[dict]:
    config = get_config()
    output_dir = ensure_directory(model_dir or config.models_dir)
    training_params = params or TrainingParams()
    train_df = pd.read_csv(train_path)
    X_train = train_df["text"].astype(str)
    y_train = train_df["label"].astype(str)

    summaries: list[dict] = []
    for model_name in MODEL_NAMES:
        model = build_model(model_name, training_params)
        model.fit(X_train, y_train)
        artifact_path = output_dir / f"{model_name}.joblib"
        joblib.dump(model, artifact_path)

        summary = {
            "model_name": model_name,
            "artifact_path": str(artifact_path),
            "train_rows": int(len(train_df)),
            "labels": sorted(y_train.unique().tolist()),
            "params": training_params.__dict__,
        }
        save_json(summary, output_dir / f"{model_name}.metadata.json")
        summaries.append(summary)

    save_json({"models": summaries}, output_dir / "training_summary.json")
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train all MediChat intent models.")
    parser.add_argument("--train-path", default=str(get_config().train_split_path))
    parser.add_argument("--model-dir", default=str(get_config().models_dir))
    parser.add_argument("--tfidf-max-features", type=int, default=4000)
    parser.add_argument("--tfidf-ngram-max", type=int, default=2)
    parser.add_argument("--svd-components", type=int, default=100)
    parser.add_argument("--pca-components", type=int, default=100)
    parser.add_argument("--logistic-max-iter", type=int, default=2000)
    args = parser.parse_args()

    training_params = TrainingParams(
        tfidf_max_features=args.tfidf_max_features,
        tfidf_ngram_max=args.tfidf_ngram_max,
        svd_components=args.svd_components,
        pca_components=args.pca_components,
        logistic_max_iter=args.logistic_max_iter,
    )
    result = train_models(args.train_path, args.model_dir, training_params)
    print(result)
