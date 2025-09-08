# tests/test_generate_health_indicators.py
"""
Unit tests for the engineer_health_indicators function
"""

import pytest
import pandas as pd
from features.engineer_health_indicators import engineer_health_indicators


# Fixtures

@pytest.fixture
def sample_df():
    """Creates a small example DataFrame to use across tests."""
    return pd.DataFrame({
        "unit": [1]*6 + [2]*6,
        "time_in_cycles": list(range(6)) * 2,
        "sensor_1": [100, 95, 90, 85, 70, 65, 120, 115, 110, 105, 80, 75],
        "sensor_2": [200, 190, 180, 170, 160, 150, 250, 240, 230, 220, 210, 200],
    })

# Parameterized tests

@pytest.mark.parametrize("sensor", ["sensor_1", "sensor_2"])
def test_engineered_columns_exist(sample_df, sensor):
    """Checks that all expected engineered columns are present and of correct type."""
    result = engineer_health_indicators(sample_df, ["sensor_1", "sensor_2"])

    expected_cols = [
        f"{sensor}_baseline",
        f"{sensor}_degraded",
        f"{sensor}_rolling_mean",
        f"{sensor}_rolling_std",
        f"{sensor}_slope",
    ]
    for col in expected_cols:
        assert col in result.columns, f"Missing expected column: {col}"
        assert not result[col].isnull().all(), f"{col} contains only NaNs"

# Core Tests

def test_engineered_shape(sample_df):
    """Ensure the engineered DataFrame has the same number of rows and additional columns."""
    result = engineer_health_indicators(sample_df, ["sensor_1", "sensor_2"])
    assert len(result) == len(sample_df)
    assert "health_score" in result.columns
    assert result["health_score"].dtype in [int, "int64"]

def test_degradation_flags_correct(sample_df):
    """Checks degradation flags based on the 75% threshold rule."""
    df = engineer_health_indicators(sample_df, ["sensor_1", "sensor_2"])
    
    # sensor_1 baseline for unit 1: mean of first 5 values
    expected_baseline = (100 + 95 + 90 + 85 + 70) / 5
    baseline_val = df[df["unit"] == 1]["sensor_1_baseline"].iloc[0]
    assert baseline_val == expected_baseline
    
    # Row that should be flagged as degraded
    degraded_row = df[(df["unit"] == 1) & (df["sensor_1"] == 65)]
    assert degraded_row["sensor_1_degraded"].iloc[0] == 1
    
    # Row that should not be flagged
    non_degraded_row = df[(df["unit"] == 1) & (df["sensor_1"] == 100)]
    assert non_degraded_row["sensor_1_degraded"].iloc[0] == 0

def test_health_score_sum(sample_df):
    """Verifies health_score equals sum of all individual sensor degradation flags."""
    df = engineer_health_indicators(sample_df, ["sensor_1", "sensor_2"])
    for _, row in df.iterrows():
        expected_score = row["sensor_1_degraded"] + row["sensor_2_degraded"]
        assert row["health_score"] == expected_score

def test_short_unit_time_series():
    """Tests behavior for units with fewer than 5 cycles; baseline should use available data."""
    short_df = pd.DataFrame({
        "unit": [1, 1],
        "time_in_cycles": [0, 1],
        "sensor_1": [100, 95],
        "sensor_2": [200, 190],
    })
    result = engineer_health_indicators(short_df, ["sensor_1", "sensor_2"])
    assert result["sensor_1_baseline"].iloc[0] == 97.5
    assert "sensor_1_degraded" in result.columns

def test_missing_sensor_values():
    """Tests handling of missing sensor values (NaNs)."""
    missing_df = pd.DataFrame({
        "unit": [1, 1, 1, 1, 1],
        "time_in_cycles": [0, 1, 2, 3, 4],
        "sensor_1": [100, None, 90, None, 85],
        "sensor_2": [200, 190, None, 170, None],
    })
    result = engineer_health_indicators(missing_df, ["sensor_1", "sensor_2"])
    # Ensure health_score exists and is computed where possible
    assert "health_score" in result.columns
    assert result["health_score"].notnull().any()
    # Degraded flags should never be NaN
    assert result["sensor_1_degraded"].notnull().all()
    assert result["sensor_2_degraded"].notnull().all()

def test_identical_sensor_values():
    """Tests that identical sensor values are not falsely flagged as degraded."""
    flat_df = pd.DataFrame({
        "unit": [1]*6,
        "time_in_cycles": list(range(6)),
        "sensor_1": [100]*6,
        "sensor_2": [200]*6,
    })
    result = engineer_health_indicators(flat_df, ["sensor_1", "sensor_2"])
    assert result["sensor_1_degraded"].sum() == 0
    assert result["sensor_2_degraded"].sum() == 0
    assert result["health_score"].sum() == 0
