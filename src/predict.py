"""
Make predictions using a trained model on Lending Club data.

This script loads a saved model and applies it to a new dataset of features.
The input CSV should contain the same feature columns (excluding the target)
used during training.  The script outputs a CSV with predicted default
probabilities and binary predictions.
"""

import logging
from pathlib import Path
import joblib
import pandas as pd
from loguru import logger

logger.remove()
logger.add(lambda msg: logging.getLogger("predict").info(msg), format="{message}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Predict default probabilities using a trained model")
    parser.add_argument("--model", required=True, help="Path to the trained model file (.pkl)")
    parser.add_argument("--input", required=True, help="CSV file containing feature columns")
    parser.add_argument("--output", required=True, help="Path to save predictions")
    args = parser.parse_args()

    logger.info(f"Loading model from {args.model}")
    model = joblib.load(args.model)

    logger.info(f"Loading input data from {args.input}")
    df = pd.read_csv(args.input)

    # Predict probabilities and labels
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(df)[:, 1]
    else:
        probas = model.predict(df)
    preds = (probas >= 0.5).astype(int)

    result = df.copy()
    result["prob_default"] = probas
    result["prediction"] = preds

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    logger.info(f"Predictions saved to {args.output}")


if __name__ == "__main__":
    main()
