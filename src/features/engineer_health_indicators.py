# Create health indicators

# Import necessary libraries
import pandas as pd
import logging

# Add logging to trace pipeline execution steps
logging.basicConfig(level=logging.INFO)


def compute_sensor_baselines(df, sensor_cols, baseline_window=5):
    """
    Compute mean baseline values for each sensor using the first N cycles per unit.
    
    Note:
    - Takes the first N cycles per unit with nsmallest().
    - This prevents issues if cycles start at different offsets.
    """
    baselines = (
        df.groupby("unit")
        .apply(lambda x: x.nsmallest(baseline_window, "time_in_cycles")[sensor_cols].mean())
        .reset_index()
    )
    return baselines

def merge_sensor_baselines(df, baselines):
    """Merge computed baseline values back into the main DataFrame."""
    return df.merge(baselines, on="unit", suffixes=("", "_baseline"))


def apply_75pct_degradation_flag(df, sensor_cols, threshold=0.75):
    """
    Create binary flags for whether each sensor is below a threshold % of its baseline.
    
    Note:
    - Using param allows experimenting with 0.7, 0.8, etc. for sensitivity analysis.
    """
    for sensor in sensor_cols:
        baseline_col = f"{sensor}_baseline"
        flag_col = f"{sensor}_degraded"
        # Flag degraded if below threshold * baseline
        df[flag_col] = (df[sensor] < threshold * df[baseline_col]).astype(int)
    return df


def add_rolling_features(df, sensor_cols, window=5):
    """
    Add rolling mean, std, and cycle-to-cycle change for each sensor.

    Parameters:
    - df: Input DataFrame
    - sensor_cols: List of sensor columns to process
    - window: Rolling window size

    Returns:
    - DataFrame with additional rolling features:
        <sensor>_rolling_mean
        <sensor>_rolling_std
        <sensor>_cycle_change
    """
    df_rolled = df.copy()

    for sensor in sensor_cols:
        # Rolling mean
        df_rolled[f"{sensor}_rolling_mean"] = df.groupby("unit")[sensor].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )

        # Rolling std
        df_rolled[f"{sensor}_rolling_std"] = df.groupby("unit")[sensor].transform(
            lambda x: x.rolling(window, min_periods=1).std()
        )

        # Cycle-to-cycle change (previously called slope)
        df_rolled[f"{sensor}_cycle_change"] = df.groupby("unit")[sensor].transform(
            lambda x: x.diff()
        )

    return df_rolled


def compute_health_score(df, weights=None):
    """
    Aggregate all sensor degradation flags to compute a health score per row.
       
    """
    degradation_flags = df.filter(like="_degraded")

    if weights:
        # Weighted score: multiply each column by its weight
        df["health_score"] = sum(
            df[col] * weights.get(col.replace("_degraded", ""), 1.0) for col in degradation_flags.columns
        )
    else:
        # Simple sum of flags (equal weight)
        df["health_score"] = degradation_flags.sum(axis=1)
    return df


def engineer_health_indicators(df, sensor_cols, baseline_window=5, rolling_window=5, threshold=0.75):
    """
    Complete pipeline to engineer health-related features.

    Steps:
    1. Compute sensor baselines from the first N cycles per unit.
    2. Merge baselines into the main DataFrame.
    3. Apply degradation flags (e.g., below 75% of baseline).
    4. Add rolling features (mean, std, cycle-to-cycle change).
    5. Compute composite health score.

    Parameters:
    - df: Input DataFrame
    - sensor_cols: List of sensor columns to process
    - baseline_window: Number of initial cycles to use for baselines
    - rolling_window: Window size for rolling features
    - threshold: Degradation threshold (default 0.75)

    Returns:
    - DataFrame with engineered health indicators
    """
    logging.info(f"Step 1: Computing baselines with window={baseline_window}")
    baselines = compute_sensor_baselines(df, sensor_cols, baseline_window)

    logging.info("Step 2: Merging baselines back into DataFrame")
    df = merge_sensor_baselines(df, baselines)

    logging.info(f"Step 3: Applying degradation flags with threshold={threshold}")
    df = apply_75pct_degradation_flag(df, sensor_cols, threshold)

    logging.info(f"Step 4: Adding rolling features (mean, std, cycle_change) with window={rolling_window}")
    df = add_rolling_features(df, sensor_cols, rolling_window)

    logging.info("Step 5: Computing composite health score")
    df = compute_health_score(df)

    logging.info("Health indicators successfully engineered.")
    return df
