"""
Download the Customer Support on Twitter dataset from Kaggle.
Uses kagglehub for programmatic download, with a manual fallback.

Usage:
    python -m data.download
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAW_DATA_DIR, KAGGLE_DATASET, RAW_CSV_FILENAME


def download_dataset():
    """Download the Twitter Customer Support dataset from Kaggle."""
    output_path = RAW_DATA_DIR / RAW_CSV_FILENAME

    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"[OK] Dataset already exists at {output_path} ({size_mb:.1f} MB)")
        return output_path

    print(f"Downloading dataset: {KAGGLE_DATASET}")
    print("This may take a few minutes depending on your connection...")

    try:
        import kagglehub
        # Download dataset — returns path to the downloaded directory
        dataset_path = kagglehub.dataset_download(KAGGLE_DATASET)
        dataset_path = Path(dataset_path)

        # Find the CSV file in the downloaded directory
        csv_files = list(dataset_path.rglob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {dataset_path}")

        # Copy the CSV to our raw data directory
        import shutil
        source_csv = csv_files[0]
        shutil.copy2(source_csv, output_path)
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"[OK] Dataset downloaded to {output_path} ({size_mb:.1f} MB)")
        return output_path

    except ImportError:
        print("\n[!] kagglehub not installed. Install with: pip install kagglehub")
        print(_manual_instructions())
        sys.exit(1)
    except Exception as e:
        print(f"\n[!] Automatic download failed: {e}")
        print(_manual_instructions())
        sys.exit(1)


def _manual_instructions():
    return f"""
╔══════════════════════════════════════════════════════════════╗
║                   Manual Download Instructions               ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  1. Visit: https://www.kaggle.com/datasets/                  ║
║     thoughtvector/customer-support-on-twitter                ║
║                                                              ║
║  2. Download the dataset (you'll need a Kaggle account)      ║
║                                                              ║
║  3. Extract the CSV file and place it at:                    ║
║     {RAW_DATA_DIR / RAW_CSV_FILENAME}
║                                                              ║
║  4. Re-run this script to verify.                            ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""


if __name__ == "__main__":
    download_dataset()
