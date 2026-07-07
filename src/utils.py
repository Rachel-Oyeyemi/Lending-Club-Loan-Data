"""
Utility functions for the Lending Club project.

This module collects small helper functions used across the codebase. It contains reusable logic for data sampling.
"""

from pathlib import Path
import pandas as pd


def sample_data(input_path: str, output_path: str, n: int = 500) -> None:
    """Create a random sample of `n` rows from a CSV file.

    Args:
        input_path: Path to the input CSV.
        output_path: Path to save the sampled CSV.
        n: Number of rows to sample.
    """
    df = pd.read_csv(input_path)
    sample_df = df.sample(n=min(n, len(df)), random_state=42)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(output_path, index=False)
