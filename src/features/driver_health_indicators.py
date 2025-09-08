# Script to generate failure labels and engineer health indicators for predictive maintenance datasets

"""
Driver script to:
1. Generate failure labels for train/test datasets.
2. Apply health indicator feature engineering.
3. Save engineered features to disk.
"""

import os
import logging
import pandas as pd
from features.engineer_health_indicators import engineer_health_indicators
from features.generate_failure_labels import generate_failure_labels

logging.basicConfig(level=logging.INFO)

def main(train_input='data/cleaned/train_FD001_cleaned.csv',
         test_input='data/cleaned/test_FD001_cleaned.csv',
         failure_threshold=30):
    
    # Paths for intermediate and output files
    train_labeled = 'data/cleaned/train_FD001_labeled.csv'
    train_features = 'data/features/train_FD001_features.csv'
    test_labeled = 'data/cleaned/test_FD001_labeled.csv'
    test_features = 'data/features/test_FD001_features.csv'

    # Generate failure labels
    generate_failure_labels(train_input, train_labeled, failure_threshold)
    generate_failure_labels(test_input, test_labeled, failure_threshold)

    # Load labeled data
    df_train = pd.read_csv(train_labeled)
    df_test = pd.read_csv(test_labeled)

    sensor_cols = [
        'sensor_2', 'sensor_3', 'sensor_4', 'sensor_6', 'sensor_7', 'sensor_8',
        'sensor_9', 'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14',
        'sensor_15', 'sensor_17', 'sensor_20', 'sensor_21'
    ]

    # Apply health indicators
    df_train_features = engineer_health_indicators(df_train, sensor_cols)
    df_test_features = engineer_health_indicators(df_test, sensor_cols)

    # Ensure output folders exist
    os.makedirs(os.path.dirname(train_features), exist_ok=True)
    os.makedirs(os.path.dirname(test_features), exist_ok=True)

    # Save engineered features
    df_train_features.to_csv(train_features, index=False)
    df_test_features.to_csv(test_features, index=False)
    
    logging.info(f"Train features saved to {train_features}")
    logging.info(f"Test features saved to {test_features}")

    return df_train_features, df_test_features

if __name__ == '__main__':
    main()
