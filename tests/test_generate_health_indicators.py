# Script to test the generation of health indicators

# Import standard testing and data libraries
import pytest
import pandas as pd

# Import the function we’re testing
from features.engineer_health_indicators import engineer_health_indicators

# Define a basic synthetic dataset that mimics CMAPSS structure
@pytest.fixture
def sample_df():
    """Creates a tiny example DataFrame to use across tests."""
    return pd.DataFrame({
        "unit": [1]*6 + [2]*6,
        "time_in_cycles": list(range(6)) * 2,
        "sensor_1": [100, 95, 90, 85, 70, 65, 120, 115, 110, 105, 80, 75],
        "sensor_2": [200, 190, 180, 170, 160, 150, 250, 240, 230, 220, 210, 200],
    })

# Test 1: Basic sanity check on dimensions
def test_engineered_shape(sample_df):
    """
    Ensures the engineered DataFrame has the same number of rows as input
    and additional columns added.
    """
    sensor_cols = ["sensor_1", "sensor_2"]
    result = engineer_health_indicators(sample_df, sensor_cols)

    # Assert the same number of rows as input
    assert len(result) == len(sample_df)

    # Check expected new columns exist
    expected_cols = [
        "sensor_1_baseline", "sensor_2_baseline",
        "sensor_1_degraded", "sensor_2_degraded",
        "sensor_1_rolling_mean", "sensor_2_rolling_mean",
        "sensor_1_rolling_std", "sensor_2_rolling_std",
        "sensor_1_slope", "sensor_2_slope",
        "health_score"
    ]
    
    for col in expected_cols:
        assert col in result.columns, f"Missing expected column: {col}"

# Test 2: Test degradation flag logic
def test_degradation_flags_correct(sample_df):
    """
    Checks that degradation flags are correctly assigned based on 75% rule.
    """
    sensor_cols = ["sensor_1", "sensor_2"]
    df = engineer_health_indicators(sample_df, sensor_cols)

    # Extract degradation flags and baselines
    sensor1_baseline_unit1 = df[df["unit"] == 1]["sensor_1_baseline"].iloc[0]
    
    # For unit 1, baseline for sensor_1 is average of first 5 cycles: (100+95+90+85+70)/5 = 88.0
    expected_baseline = (100 + 95 + 90 + 85 + 70) / 5
    assert sensor1_baseline_unit1 == expected_baseline

    # Row with sensor_1 = 65 → 65 < 0.75 * 88 → Should be flagged as degraded
    degraded_row = df[(df["unit"] == 1) & (df["sensor_1"] == 65)]
    assert degraded_row["sensor_1_degraded"].iloc[0] == 1

    # Row with sensor_1 = 100 → Should NOT be flagged
    non_degraded_row = df[(df["unit"] == 1) & (df["sensor_1"] == 100)]
    assert non_degraded_row["sensor_1_degraded"].iloc[0] == 0

# Test 3: Test health score calculation
def test_health_score_sum(sample_df):
    """
    Verifies that health score equals the sum of all individual sensor degradation flags.
    """
    sensor_cols = ["sensor_1", "sensor_2"]
    df = engineer_health_indicators(sample_df, sensor_cols)

    # Grab a few sample rows and check health score calculation
    for _, row in df.iterrows():
        expected_score = row["sensor_1_degraded"] + row["sensor_2_degraded"]
        assert row["health_score"] == expected_score
        
# Test 4: Short Time Series Edge Case
def test_short_unit_time_series():
    """
    Tests behavior when a unit has fewer than 5 cycles.
    The baseline should be computed from available data only.
    """
    short_df = pd.DataFrame({
        "unit": [1, 1],
        "time_in_cycles": [0, 1],
        "sensor_1": [100, 95],
        "sensor_2": [200, 190],
    })
    
    result = engineer_health_indicators(short_df, ["sensor_1", "sensor_2"])
    
    # Baseline should be mean of [100, 95] = 97.5
    assert result["sensor_1_baseline"].iloc[0] == 97.5
    assert "sensor_1_degraded" in result.columns
    
# Test 5: Missing Sensor Values
def test_missing_sensor_values():
    """
    Tests handling of missing sensor values (NaNs).
    """
    missing_df = pd.DataFrame({
        "unit": [1, 1, 1, 1, 1],
        "time_in_cycles": [0, 1, 2, 3, 4],
        "sensor_1": [100, None, 90, None, 85],
        "sensor_2": [200, 190, None, 170, None],
    })

    result = engineer_health_indicators(missing_df, ["sensor_1", "sensor_2"])
    
    # Make sure health score column exists even with NaNs
    assert "health_score" in result.columns
    assert not result["health_score"].isnull().all()  # Some should still be computed

# Test 6: All Sensor Values Identical
def test_identical_sensor_values():
    """
    Tests that identical sensor values don’t get falsely flagged.
    """
    flat_df = pd.DataFrame({
        "unit": [1]*6,
        "time_in_cycles": list(range(6)),
        "sensor_1": [100]*6,
        "sensor_2": [200]*6,
    })

    result = engineer_health_indicators(flat_df, ["sensor_1", "sensor_2"])
    
    # No degradation should be flagged
    assert result["sensor_1_degraded"].sum() == 0
    assert result["sensor_2_degraded"].sum() == 0
    assert result["health_score"].sum() == 0
