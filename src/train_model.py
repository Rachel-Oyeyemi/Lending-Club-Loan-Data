"""
Train machine‑learning models on Lending Club Loan Data.

This script trains a baseline **Logistic Regression** model and more advanced
algorithms (**Random Forest**, **XGBoost**, and **LightGBM**) on the engineered
dataset.  Models are saved into the `models/` directory for later evaluation
and deployment.  Stratified train/test splits are used to maintain class
balance.
"""

import logging
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from loguru import logger

try:
    import xgboost as xgb
except ImportError:
    xgb = None  # type: ignore
try:
    import lightgbm as lgb
except ImportError:
    lgb = None  # type: ignore

logger.remove()
logger.add(lambda msg: logging.getLogger("train_model").info(msg), format="{message}")


def train_models(df: pd.DataFrame, model_dir: str) -> None:
    """Train baseline and advanced models and persist them.

    Args:
        df: DataFrame containing engineered features and the target column `DEFAULT`.
        model_dir: Directory where trained models will be saved.
    """
    # Separate features and target
    X = df.drop(columns=["DEFAULT"])
    y = df["DEFAULT"]

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    model_dir_path = Path(model_dir)
    model_dir_path.mkdir(parents=True, exist_ok=True)

    # Baseline: Logistic Regression
    logger.info("Training Logistic Regression baseline model")
    lr = LogisticRegression(max_iter=1000, n_jobs=-1, solver="lbfgs")
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    logger.info(f"Logistic Regression accuracy: {acc_lr:.3f}")
    joblib.dump(lr, model_dir_path / "logistic_regression.pkl")

    # Random Forest
    logger.info("Training Random Forest model")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    joblib.dump(rf, model_dir_path / "random_forest.pkl")

    # XGBoost
    if xgb is not None:
        logger.info("Training XGBoost model")
        xgb_clf = xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss",
        )
        xgb_clf.fit(X_train, y_train)
        joblib.dump(xgb_clf, model_dir_path / "xgboost.pkl")
    else:
        logger.warning("XGBoost not installed; skipping XGBoost model.")

    # LightGBM
    if lgb is not None:
        logger.info("Training LightGBM model")
        lgb_clf = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.1,
            num_leaves=31,
            max_depth=-1,
            random_state=42,
        )
        lgb_clf.fit(X_train, y_train)
        joblib.dump(lgb_clf, model_dir_path / "lightgbm.pkl")
    else:
        logger.warning("LightGBM not installed; skipping LightGBM model.")

    logger.info(f"Models saved to {model_dir}")


def main() -> None:
    """Command‑line entry point for model training."""
    import argparse
    parser = argparse.ArgumentParser(description="Train models on engineered Lending Club data")
    parser.add_argument("--input", "-i", required=True, help="Path to engineered CSV file")
    parser.add_argument("--model-output", "-o", required=True, help="Directory to save trained models")
    args = parser.parse_args()

    logger.info(f"Loading engineered data from {args.input}")
    df = pd.read_csv(args.input)
    logger.info(f"Data shape: {df.shape}")

    train_models(df, args.model_output)


if __name__ == "__main__":
    main()
