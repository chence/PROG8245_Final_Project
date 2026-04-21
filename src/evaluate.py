from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.config import get_config
from src.train import MODEL_NAMES
from src.utils import ensure_directory, save_json, save_text


def evaluate_models(test_path: str, model_dir: str | None = None) -> list[dict]:
    config = get_config()
    output_dir = ensure_directory(config.documentation_dir)
    model_root = Path(model_dir or config.models_dir)
    test_df = pd.read_csv(test_path)
    X_test = test_df["text"].astype(str)
    y_true = test_df["label"].astype(str)

    metrics: list[dict] = []
    metrics_payload: dict[str, dict] = {}
    labels = sorted(y_true.unique().tolist())

    for model_name in MODEL_NAMES:
        artifact_path = model_root / f"{model_name}.joblib"
        model = joblib.load(artifact_path)
        y_pred = model.predict(X_test)
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        matrix = confusion_matrix(y_true, y_pred, labels=labels)
        row = {
            "model_name": model_name,
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "precision_macro": round(report["macro avg"]["precision"], 4),
            "recall_macro": round(report["macro avg"]["recall"], 4),
            "f1_macro": round(report["macro avg"]["f1-score"], 4),
            "precision_weighted": round(report["weighted avg"]["precision"], 4),
            "recall_weighted": round(report["weighted avg"]["recall"], 4),
            "f1_weighted": round(report["weighted avg"]["f1-score"], 4),
        }
        metrics.append(row)
        metrics_payload[model_name] = {
            "summary": row,
            "classification_report": report,
            "labels": labels,
            "confusion_matrix": matrix.tolist(),
        }

        plt.figure(figsize=(8, 6))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.title(f"Confusion Matrix - {model_name}")
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.tight_layout()
        plt.savefig(output_dir / f"confusion_matrix_{model_name}.png", dpi=150, bbox_inches="tight")
        plt.close()

    results_df = pd.DataFrame(metrics).sort_values(by="f1_weighted", ascending=False).reset_index(drop=True)
    results_df.to_csv(config.comparison_csv_path, index=False)
    save_json(metrics_payload, config.evaluation_json_path)

    markdown_lines = [
        "# Model Comparison",
        "",
        "| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Precision (weighted) | Recall (weighted) | F1 (weighted) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results_df.to_dict(orient="records"):
        markdown_lines.append(
            "| {model_name} | {accuracy:.4f} | {precision_macro:.4f} | {recall_macro:.4f} | {f1_macro:.4f} | "
            "{precision_weighted:.4f} | {recall_weighted:.4f} | {f1_weighted:.4f} |".format(**row)
        )
    save_text("\n".join(markdown_lines) + "\n", config.comparison_md_path)
    return results_df.to_dict(orient="records")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate all MediChat models.")
    parser.add_argument("--test-path", default=str(get_config().test_split_path))
    parser.add_argument("--model-dir", default=str(get_config().models_dir))
    args = parser.parse_args()
    print(evaluate_models(args.test_path, args.model_dir))
