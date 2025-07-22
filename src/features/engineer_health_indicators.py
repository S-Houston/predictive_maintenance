# Create health indicators

# Import necessary libraries
import pandas as pd

def compute_sensor_baselines(df, sensor_cols, baseline_window=5):
    """Compute mean baseline values for each sensor using the first N cycles per unit."""
    return (
        df[df["time_in_cycles"] < baseline_window]
        .groupby("unit")[sensor_cols]
        .mean()
        .reset_index()
    )

def merge_sensor_baselines(df, baselines):
    """Merge computed baseline values back into the main DataFrame."""
    return df.merge(baselines, on="unit", suffixes=("", "_baseline"))

def apply_75pct_degradation_flag(df, sensor_cols):
    """Create binary flags for whether each sensor is below 75% of its baseline."""
    for sensor in sensor_cols:
        baseline_col = f"{sensor}_baseline"
        flag_col = f"{sensor}_degraded"
        df[flag_col] = (df[sensor] < 0.75 * df[baseline_col]).astype(int)
    return df

def add_rolling_features(df, sensor_cols, window=5):
    """Add rolling mean, std, and slope for each sensor."""
    df_rolled = df.copy()
    for sensor in sensor_cols:
        df_rolled[f"{sensor}_rolling_mean"] = df.groupby("unit")[sensor].transform(lambda x: x.rolling(window, min_periods=1).mean())
        df_rolled[f"{sensor}_rolling_std"] = df.groupby("unit")[sensor].transform(lambda x: x.rolling(window, min_periods=1).std())
        df_rolled[f"{sensor}_slope"] = df.groupby("unit")[sensor].transform(lambda x: x.diff())
    return df_rolled

def compute_health_score(df):
    """Aggregate all sensor degradation flags to compute a health score per row."""
    degradation_flags = df.filter(like="_degraded")
    df["health_score"] = degradation_flags.sum(axis=1)
    return df

def engineer_health_indicators(df, sensor_cols, baseline_window=5, rolling_window=5):
    """Complete pipeline to engineer health-related features."""
    # Step 1: Compute baselines
    baselines = compute_sensor_baselines(df, sensor_cols, baseline_window)

    # Step 2: Merge into main data
    df = merge_sensor_baselines(df, baselines)

    # Step 3: Apply 75% degradation flags
    df = apply_75pct_degradation_flag(df, sensor_cols)

    # Step 4: Add rolling features
    df = add_rolling_features(df, sensor_cols, rolling_window)

    # Step 5: Compute composite health score
    df = compute_health_score(df)

    return df