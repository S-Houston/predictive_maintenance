# Script to generate failure labels for predictive maintenance datasets

# Import necessary libraries
import pandas as pd
from pathlib import Path

def generate_failure_labels(input_path, output_path, failure_threshold=30,
                            true_rul_path=None):
    """
    Generate Remaining Useful Life (RUL) and binary failure labels.

    Parameters:
    - input_path (str or Path): Path to the input CSV file containing engine data.
                                Must include 'unit' and 'time_in_cycles' columns.
    - output_path (str or Path): Path to save the output CSV file with failure labels.
    - failure_threshold (int): Number of cycles before failure at which to flag
                               as a binary failure (default = 30 cycles).
    - true_rul_path (str or Path, optional): CSV with a single 'RUL' column giving
                               the true RUL at the last observed cycle of each unit
                               (row i = unit i + 1), e.g. data/processed/rul_FD001.csv.
                               Required for truncated (test) series, where the last
                               observed cycle is not the failure point. Omit for
                               run-to-failure (train) series.

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

    # Truncated series: add the true RUL remaining at each unit's last observed cycle
    if true_rul_path is not None:
        true_rul = pd.read_csv(true_rul_path)["RUL"]
        units = sorted(df["unit"].unique())
        if len(true_rul) != len(units):
            raise ValueError(
                f"True RUL file has {len(true_rul)} rows but data has {len(units)} units"
            )
        offsets = pd.Series(true_rul.values, index=range(1, len(true_rul) + 1))
        missing = set(units) - set(offsets.index)
        if missing:
            raise ValueError(f"Units {sorted(missing)} have no entry in the true RUL file")
        df["RUL"] = df["RUL"] + df["unit"].map(offsets)

    # Generate binary failure labels: 1 if within threshold cycles of failure, else 0
    df["failure_binary"] = (df["RUL"] <= failure_threshold).astype(int)

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Save the labeled dataset
    df.to_csv(output_path, index=False)
    print(f"Failure labels generated and saved to {output_path}")

    return df  
