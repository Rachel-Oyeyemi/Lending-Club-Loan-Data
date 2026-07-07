"""
Download the Lending Club Loan Data dataset from Kaggle.

This script provides a function to download the Lending Club dataset via the
Kaggle API.  Because Kaggle requires user authentication, you must supply a
valid Kaggle API key via the `KAGGLE_USERNAME` and `KAGGLE_KEY` environment
variables or have a `~/.kaggle/kaggle.json` file.  If the Kaggle API is
unavailable, you can manually download the CSV from the Kaggle web page and
place it into the `data/raw` directory.

The dataset on Kaggle is published under the slug
`adarshsng/lending-club-loan-data-csv`.  When downloaded via the API, it comes
as a zipped file containing one or more CSVs.  The `download_dataset`
function will unzip the archive into the specified output directory.
"""

import logging
import os
from pathlib import Path
from typing import Optional

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except ImportError:
    KaggleApi = None  # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
LOGGER = logging.getLogger(__name__)

# Kaggle dataset identifier
DATASET_SLUG = "adarshsng/lending-club-loan-data-csv"


def download_dataset(output_dir: str, unzip: bool = True) -> None:
    """Download the Lending Club dataset via the Kaggle API.

    Args:
        output_dir: Directory where the downloaded files will be saved.
        unzip: Whether to unzip the downloaded archive.  If true, the
            archive is unzipped into the output directory and the original
            `.zip` file is removed.

    Raises:
        RuntimeError: If the Kaggle API is not installed or authentication fails.
    """
    dest_dir = Path(output_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    if KaggleApi is None:
        LOGGER.error(
            "kaggle API is not installed. Install it with 'pip install kaggle' or download the dataset manually."
        )
        raise RuntimeError("kaggle API not available")

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:
        LOGGER.error(
            "Failed to authenticate with Kaggle API. Ensure KAGGLE_USERNAME and KAGGLE_KEY are set."
        )
        raise RuntimeError("Kaggle authentication failed") from exc

    LOGGER.info("Downloading dataset %s to %s", DATASET_SLUG, dest_dir)
    api.dataset_download_files(DATASET_SLUG, path=str(dest_dir), unzip=unzip)
    LOGGER.info("Download complete")


def main() -> None:
    """Command‑line entry point for downloading the dataset."""
    import argparse

    parser = argparse.ArgumentParser(description="Download the Lending Club Loan Data from Kaggle")
    parser.add_argument(
        "--output-dir",
        "-o",
        default=str(Path(__file__).resolve().parent.parent / "data/raw"),
        help="Directory to save the downloaded data",
    )
    parser.add_argument(
        "--no-unzip", action="store_true", help="Do not unzip the downloaded archive"
    )
    args = parser.parse_args()
    download_dataset(args.output_dir, unzip=not args.no_unzip)


if __name__ == "__main__":
    main()
