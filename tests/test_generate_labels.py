# Script to test the generate_failure_labels function
import pandas as pd
import pytest
import os
from pathlib import Path
from features.generate_failure_labels import generate_failure_labels

def test_generate_failure_labels_basic(tmp_path):
    """
    Test generate_failure_labels by creating a sample CSV, running the function,
    and validating the output CSV.
    """
    # Create sample data matching your columns (minimal for test)
    sample_data = pd.DataFrame({
        'unit': [1, 1, 1, 2, 2],
        'time_in_cycles': [10, 20, 30, 5, 15],
        'op_setting_1': [0, 0, 0, 0, 0],  # dummy values
        'op_setting_2': [0, 0, 0, 0, 0],
        'sensor_2': [0, 0, 0, 0, 0],
        'sensor_3': [0, 0, 0, 0, 0],
        'sensor_4': [0, 0, 0, 0, 0],
        'sensor_6': [0, 0, 0, 0, 0],
        'sensor_7': [0, 0, 0, 0, 0],
        'sensor_8': [0, 0, 0, 0, 0],
        'sensor_9': [0, 0, 0, 0, 0],
        'sensor_11': [0, 0, 0, 0, 0],
        'sensor_12': [0, 0, 0, 0, 0],
        'sensor_13': [0, 0, 0, 0, 0],
        'sensor_14': [0, 0, 0, 0, 0],
        'sensor_15': [0, 0, 0, 0, 0],
        'sensor_17': [0, 0, 0, 0, 0],
        'sensor_20': [0, 0, 0, 0, 0],
        'sensor_21': [0, 0, 0, 0, 0]
    })

    # Paths for input and output files in a temp directory
    input_csv = tmp_path / "input.csv"
    output_csv = tmp_path / "output.csv"
    
    # Save sample data to CSV
    sample_data.to_csv(input_csv, index=False)
    
    # Call your function
    failure_threshold = 10  # set threshold to test labels
    generate_failure_labels(str(input_csv), str(output_csv), failure_threshold=failure_threshold)
    
    # Read the output CSV
    df_out = pd.read_csv(output_csv)
    
    # Check that 'failure_binary' column was added
    assert 'failure_binary' in df_out.columns
    
    # Check the dtype of the label column is int (or convertible)
    assert pd.api.types.is_integer_dtype(df_out['failure_binary']), "failure_binary should be integer dtype"
    
    # Manually calculate expected failure labels for the sample data
    # For unit 1: max time = 30
    # failure if RUL <= 10 → RUL = 30 - time
    # times: 10 -> RUL=20 -> label=0, 20 -> RUL=10 -> label=1, 30 -> RUL=0 -> label=1
    # For unit 2: max time = 15
    # times: 5 -> RUL=10 -> label=1, 15 -> RUL=0 -> label=1
    
    assert df_out['RUL'].tolist() == [20, 10, 0, 10, 0]

    expected_labels = [0, 1, 1, 1, 1]
    assert df_out['failure_binary'].tolist() == expected_labels


def test_generate_failure_labels_with_true_rul(tmp_path):
    """
    For truncated (test) series, RUL at the last observed cycle must equal the
    ground-truth value, and earlier cycles count up from there.
    """
    sample_data = pd.DataFrame({
        'unit': [1, 1, 1, 2, 2],
        'time_in_cycles': [1, 2, 3, 1, 2],
    })
    input_csv = tmp_path / "input.csv"
    output_csv = tmp_path / "output.csv"
    true_rul_csv = tmp_path / "rul.csv"
    sample_data.to_csv(input_csv, index=False)
    pd.DataFrame({'RUL': [50, 5]}).to_csv(true_rul_csv, index=False)

    df_out = generate_failure_labels(str(input_csv), str(output_csv),
                                     failure_threshold=10,
                                     true_rul_path=str(true_rul_csv))

    assert df_out['RUL'].tolist() == [52, 51, 50, 6, 5]
    last_rul = df_out.groupby('unit')['RUL'].last().tolist()
    assert last_rul == [50, 5]
    assert df_out['failure_binary'].tolist() == [0, 0, 0, 1, 1]


def test_generate_failure_labels_true_rul_unit_mismatch(tmp_path):
    """A truth file that doesn't match the number of units is rejected."""
    input_csv = tmp_path / "input.csv"
    true_rul_csv = tmp_path / "rul.csv"
    pd.DataFrame({'unit': [1, 2], 'time_in_cycles': [1, 1]}).to_csv(input_csv, index=False)
    pd.DataFrame({'RUL': [50]}).to_csv(true_rul_csv, index=False)

    with pytest.raises(ValueError):
        generate_failure_labels(str(input_csv), str(tmp_path / "out.csv"),
                                true_rul_path=str(true_rul_csv))
