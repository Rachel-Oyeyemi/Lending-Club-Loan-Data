"""
Feature engineering for Lending Club Loan Data.

This script reads the processed data (with numeric features and a binary target)
and creates additional features that may improve model performance:

* **loan_to_income_ratio** – Ratio of loan amount to annual income.
* **installment_to_income_ratio** – Ratio of monthly installment to annual income.
* **interest_installment_product** – Product of interest rate and installment amount.
* **credit_grade_value** – If the original grade (A–G) was one‑hot encoded, this feature
  reconstructs an ordinal grade score from the encoded columns (A=1 … G=7).

These engineered features are appended to the DataFrame and saved to disk.
"""

import logging
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger

logger.remove()
logger.add(lambda msg: logging.getLogger("feature_engineering").info(msg), format="{message}")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create additional features from existing numeric columns.

    Args:
        df: Input DataFrame containing processed numeric features and `DEFAULT`.

    Returns:
        DataFrame with new feature columns added.
    """
    df = df.copy()
    # Compute ratio features if the necessary columns exist
    if set(["loan_amnt", "annual_inc"]).issubset(df.columns):
        df["loan_to_income_ratio"] = df["loan_amnt"] / (df["annual_inc"] + 1e-6)
    if set(["installment", "annual_inc"]).issubset(df.columns):
        df["installment_to_income_ratio"] = df["installment"] / (df["annual_inc"] + 1e-6)
    if set(["int_rate", "installment"]).issubset(df.columns):
        df["interest_installment_product"] = df["int_rate"] * df["installment"]

    # Reconstruct ordinal credit grade if one‑hot encoded columns are present
    grade_columns = [col for col in df.columns if col.startswith("grade_")]
    if grade_columns:
        grade_mapping = {f"grade_{letter}": idx for idx, letter in enumerate(['A','B','C','D','E','F','G'], start=1)}
        # Multiply each one‑hot column by its ordinal and sum across rows
        df["credit_grade_value"] = 0
        for col, val in grade_mapping.items():
            if col in df.columns:
                df["credit_grade_value"] += df[col] * val

    return df


def main() -> None:
    """Command‑line interface for feature engineering."""
    import argparse
    parser = argparse.ArgumentParser(description="Perform feature engineering on Lending Club data")
    parser.add_argument("--input", "-i", required=True, help="Path to the processed CSV file")
    parser.add_argument("--output", "-o", required=True, help="Path to save the engineered CSV file")
    args = parser.parse_args()

    logger.info(f"Loading processed data from {args.input}")
    df = pd.read_csv(args.input)
    logger.info(f"Data shape before engineering: {df.shape}")

    df_fe = engineer_features(df)
    logger.info(f"Data shape after engineering: {df_fe.shape}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df_fe.to_csv(args.output, index=False)
    logger.info(f"Engineered data saved to {args.output}")


if __name__ == "__main__":
    main()
