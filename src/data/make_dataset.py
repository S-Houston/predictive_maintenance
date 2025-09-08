"""
Make Dataset Script
-------------------
This script loads the raw CMAPSS FD001 datasets (train and test) from .txt,
applies standard column names, and saves CSVs into:
- data/processed/ (raw converted CSVs)
- data/cleaned/   (ready-to-use cleaned CSVs)
"""

import pandas as pd
from pathlib import Path


def load_and_process_txt(input_file: Path):
    """
    Load a CMAPSS .txt dataset, apply column names, and rename the time column.
    Returns the processed DataFrame.
    """
    # Column names based on CMAPSS FD001 documentation
    column_names = (
        ["unit", "time"]  # identifiers
        + [f"op_setting_{i}" for i in range(1, 4)]  # 3 operating settings
        + [f"sensor_{i}" for i in range(1, 22)]  # 21 sensors
    )

    # Load space-separated txt file
    df = pd.read_csv(input_file, sep=r"\s+", header=None, names=column_names)

    # Standardize column name
    df = df.rename(columns={"time": "time_in_cycles"})

    return df


def save_csv(df: pd.DataFrame, output_path: Path):
    """Save the DataFrame to CSV, creating parent directories if needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved file: {output_path}")


def main():
    # Define directories
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    cleaned_dir = Path("data/cleaned")

    datasets = {
        "train": "train_FD001.txt",
        "test": "test_FD001.txt",
    }

    for name, raw_file in datasets.items():
        input_path = raw_dir / raw_file
        if not input_path.exists():
            print(f"Skipped {name}: {input_path} not found")
            continue

        df = load_and_process_txt(input_path)

        # Save to processed folder
        processed_path = processed_dir / f"{name}_FD001.csv"
        save_csv(df, processed_path)

        # Save to cleaned folder
        cleaned_path = cleaned_dir / f"{name}_FD001_cleaned.csv"
        save_csv(df, cleaned_path)


if __name__ == "__main__":
    main()
