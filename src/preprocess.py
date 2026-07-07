"""
Preprocess the Lending Club Loan Data dataset.

This script reads the raw Lending Club CSV, performs cleaning and basic
preprocessing, and writes a processed CSV for downstream modelling.
Key steps include:

* Dropping columns with a high proportion of missing values (default 40 %).
* Mapping the target variable `loan_status` to a binary default flag and
  filtering out records with other statuses.
* Converting string representations (e.g. interest rates, terms) to numeric.
* Imputing missing values for numeric and categorical features.
* Encoding categorical variables via one‑hot or ordinal encoding.

The resulting dataset contains only numeric features and the binary target
column `DEFAULT`.  Identifiers and textual description fields are removed.
"""

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from loguru import logger

# Configure logging
logger.remove()
logger.add(lambda msg: logging.getLogger("preprocess").info(msg), format="{message}")


def load_dataset(path: str) -> pd.DataFrame:
    """Load the Lending Club dataset from CSV.

    Args:
        path: Path to the CSV file.

    Returns:
        DataFrame containing the raw data.
    """
    return pd.read_csv(path, low_memory=False)


def drop_high_missing(df: pd.DataFrame, threshold: float = 0.4) -> pd.DataFrame:
    """Drop columns with more than `threshold` proportion of missing values.

    Args:
        df: Input DataFrame.
        threshold: Fraction of allowed missing values (e.g. 0.4 drops columns
            with >40 % missing).

    Returns:
        DataFrame with high‑missingness columns removed.
    """
    missing_frac = df.isna().mean()
    cols_to_keep = missing_frac[missing_frac <= threshold].index
    dropped = set(df.columns) - set(cols_to_keep)
    if dropped:
        logger.info(f"Dropping {len(dropped)} columns with >{threshold:.0%} missing values: {sorted(list(dropped))}")
    return df[cols_to_keep].copy()


def map_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create a binary target column `DEFAULT` from `loan_status` and filter rows.

    The dataset contains multiple statuses.  We map `Charged Off`, `Default` and
    `Late (31-120 days)` to 1 (default) and `Fully Paid` to 0 (non‑default).
    Records with other statuses are dropped.
    """
    mapping = {
        "Charged Off": 1,
        "Default": 1,
        "Late (31-120 days)": 1,
        "Fully Paid": 0,
    }
    df = df[df["loan_status"].isin(mapping.keys())].copy()
    df["DEFAULT"] = df["loan_status"].map(mapping)
    return df


def convert_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Convert string fields to numeric representations.

    * `term`: Extract the number of months from strings like '36 months'.
    * `int_rate`: Strip the percent sign and cast to float.
    * `emp_length`: Map textual employment length to numeric years.

    Args:
        df: DataFrame with raw columns.

    Returns:
        DataFrame with converted columns.
    """
    df = df.copy()
    if "term" in df.columns:
        df["term"] = df["term"].str.extract(r"(\d+)").astype(float)
    if "int_rate" in df.columns:
        df["int_rate"] = df["int_rate"].str.rstrip("%").astype(float)
    if "emp_length" in df.columns:
        # Convert employment length to numeric years (e.g. '< 1 year' -> 0, '10+ years' -> 10)
        def parse_emp_length(x: str) -> float:
            if pd.isna(x):
                return np.nan  # preserve NaN for imputation
            x = x.strip()
            if x == '< 1 year':
                return 0
            if x == '10+ years':
                return 10
            # Extract number
            try:
                return float(x.split()[0])
            except Exception:
                return np.nan
        df["emp_length"] = df["emp_length"].apply(parse_emp_length)
    return df


def encode_and_impute(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values and one‑hot encode categorical variables.

    Numeric columns are imputed with the median.  Categorical columns are
    imputed with the most frequent value and then one‑hot encoded.  The
    transformer returns a NumPy array which is converted back into a DataFrame
    with appropriate column names.
    """
    df = df.copy()
    # Identify features to drop
    drop_cols = [
        'loan_status',
        'issue_d',
        'zip_code',
        'addr_state',
        'title',
        'desc',
        'url',
        'member_id',
        'id'
    ]
    for col in drop_cols:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    # Separate target
    y = df["DEFAULT"]
    X = df.drop(columns=["DEFAULT"])

    # Determine numeric and categorical columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    # Preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    # Fit and transform
    X_transformed = preprocessor.fit_transform(X)

    # Build DataFrame with new feature names
    num_features = numeric_cols
    cat_features = preprocessor.named_transformers_["cat"]["onehot"].get_feature_names_out(categorical_cols)
    feature_names = np.concatenate([num_features, cat_features])
    X_df = pd.DataFrame(X_transformed, columns=feature_names, index=df.index)
    X_df["DEFAULT"] = y.values
    return X_df


def preprocess(path: str, output_path: str, missing_threshold: float = 0.4) -> None:
    """End‑to‑end preprocessing pipeline.

    Args:
        path: Path to the raw CSV file.
        output_path: Path to save the processed CSV file.
        missing_threshold: Fraction of missing values allowed per column.
    """
    logger.info(f"Loading dataset from {path}")
    df = load_dataset(path)
    logger.info(f"Raw data shape: {df.shape}")

    df = drop_high_missing(df, threshold=missing_threshold)
    logger.info(f"Shape after dropping high‑missing columns: {df.shape}")

    df = map_target(df)
    logger.info(f"Shape after filtering target statuses: {df.shape}")

    df = convert_strings(df)
    logger.info("Converted string columns to numeric formats")

    df = encode_and_impute(df)
    logger.info(f"Final processed shape: {df.shape}")

    # Save processed data
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Processed data saved to {output_path}")


def main() -> None:
    """Command‑line entry point for preprocessing."""
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess the Lending Club dataset")
    parser.add_argument("--input", "-i", required=True, help="Path to the raw CSV file")
    parser.add_argument("--output", "-o", required=True, help="Path to save the processed CSV file")
    parser.add_argument(
        "--missing-threshold",
        "-t",
        type=float,
        default=0.4,
        help="Proportion threshold for dropping columns (default 0.4)",
    )
    args = parser.parse_args()

    preprocess(args.input, args.output, missing_threshold=args.missing_threshold)


if __name__ == "__main__":
    main()
