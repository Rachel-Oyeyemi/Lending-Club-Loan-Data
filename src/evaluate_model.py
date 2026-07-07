"""
Evaluate trained models on the Lending Club dataset.

This script loads trained models from the `models/` directory and evaluates
them on a given engineered dataset.  It computes accuracy, precision, recall,
F1‑score and ROC–AUC, saves confusion matrix and ROC curve plots into
`visuals/`, and writes a markdown report summarising the results.
"""

import logging
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

logger.remove()
logger.add(lambda msg: logging.getLogger("evaluate_model").info(msg), format="{message}")


def load_models(model_dir: str) -> dict:
    """Load all available models from a directory.

    Args:
        model_dir: Directory containing pickled model files.

    Returns:
        Dictionary mapping model name to loaded estimator.
    """
    models = {}
    for fname in Path(model_dir).glob("*.pkl"):
        name = fname.stem
        try:
            models[name] = joblib.load(fname)
        except Exception as exc:
            logger.error(f"Failed to load {fname}: {exc}")
    return models


def evaluate_models(models: dict, X: pd.DataFrame, y: pd.Series, visuals_dir: str, report_path: str) -> None:
    """Evaluate models on test data and produce plots and a report.

    Args:
        models: Dictionary of model name -> estimator.
        X: Features DataFrame.
        y: Target Series.
        visuals_dir: Directory to save plots.
        report_path: Path to save the Markdown summary.
    """
    Path(visuals_dir).mkdir(parents=True, exist_ok=True)
    metrics = []
    for name, model in models.items():
        logger.info(f"Evaluating {name}")
        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(X)[:, 1]
        else:
            # Some models (e.g. SVM) may use decision_function; fallback to predictions
            probas = model.predict(X)
        preds = (probas >= 0.5).astype(int)
        metrics.append({
            "model": name,
            "accuracy": accuracy_score(y, preds),
            "precision": precision_score(y, preds),
            "recall": recall_score(y, preds),
            "f1": f1_score(y, preds),
            "roc_auc": roc_auc_score(y, probas),
        })
        # Confusion matrix
        cm = confusion_matrix(y, preds)
        plt.figure(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(f"Confusion Matrix – {name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(Path(visuals_dir) / f"confusion_matrix_{name}.png")
        plt.close()
        # ROC curve
        fpr, tpr, _ = roc_curve(y, probas)
        plt.figure(figsize=(4, 3))
        plt.plot(fpr, tpr, label=f"ROC – {name}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve – {name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(Path(visuals_dir) / f"roc_curve_{name}.png")
        plt.close()

    # Save metrics to CSV
    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_csv(Path(visuals_dir) / "model_metrics.csv", index=False)

    # Write markdown summary
    lines = [
        "# Model Evaluation Report",
        "",
        "| Model | Accuracy | Precision | Recall | F1‑Score | ROC–AUC |",
        "|---|---|---|---|---|---|",
    ]
    for row in metrics:
        lines.append(
            f"| {row['model']} | {row['accuracy']:.3f} | {row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} | {row['roc_auc']:.3f} |"
        )
    Path(report_path).write_text("\n".join(lines))
    logger.info(f"Evaluation report saved to {report_path}")


def main() -> None:
    """Command‑line entry point for model evaluation."""
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate trained models on the Lending Club dataset")
    parser.add_argument("--model_dir", "-m", required=True, help="Directory containing trained models")
    parser.add_argument("--data", "-d", required=True, help="CSV file with engineered features and target")
    parser.add_argument("--report", "-r", required=True, help="Path to save the markdown report")
    parser.add_argument("--visuals", "-v", default="visuals/", help="Directory to save figures")
    args = parser.parse_args()

    logger.info("Loading data")
    df = pd.read_csv(args.data)
    X = df.drop(columns=["DEFAULT"])
    y = df["DEFAULT"]

    models = load_models(args.model_dir)
    if not models:
        logger.error("No models found in the specified directory")
        return
    evaluate_models(models, X, y, args.visuals, args.report)


if __name__ == "__main__":
    main()
