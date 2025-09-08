# Script to generate failure labels for predictive maintenance datasets

# Import necessary libraries
import pandas as pd
from pathlib import Path

def generate_failure_labels(input_path, output_path, failure_threshold=30):
    """
    Generate Remaining Useful Life (RUL) and binary failure labels.

    Parameters:
    - input_path (str or Path): Path to the input CSV file containing engine data.
                                Must include 'unit' and 'time_in_cycles' columns.
    - output_path (str or Path): Path to save the output CSV file with failure labels.
    - failure_threshold (int): Number of cycles before failure at which to flag
                               as a binary failure (default = 30 cycles).
    
    Returns:
    - df (pd.DataFrame): DataFrame with added RUL and binary failure labels.
    """
    # Load the dataset
    df = pd.read_csv(input_path)

    # Validate required columns exist
    required_cols = {"unit", "time_in_cycles"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Input data must contain columns {required_cols}, but got {df.columns.tolist()}")

    # Calculate the maximum cycle per engine (end of life for each unit)
    max_cycle = df.groupby("unit")["time_in_cycles"].transform("max")

    # Remaining Useful Life (RUL) = max cycle - current cycle
    df["RUL"] = max_cycle - df["time_in_cycles"]

    # Generate binary failure labels: 1 if within threshold cycles of failure, else 0
    df["failure_binary"] = (df["RUL"] <= failure_threshold).astype(int)

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Save the labeled dataset
    df.to_csv(output_path, index=False)
    print(f"Failure labels generated and saved to {output_path}")

    return df  
