# Script to run the engineer_health_indicators script and save the output
# This script will read the cleaned training data, apply health indicator feature engineering,

# Import necessary libraries
import os
import pandas as pd
from features.engineer_health_indicators import engineer_health_indicators

def main():
    # Path to your cleaned training data CSV
    input_csv = 'data/cleaned/train_FD001_cleaned.csv'
    
    # Path where you want to save the output CSV with engineered features
    output_csv = 'data/features/train_FD001_features.csv'
    
    # Load the cleaned data
    df = pd.read_csv(input_csv)
    
    # List your sensor columns to process (match columns in your CSV)
    sensor_cols = [
        'sensor_2', 'sensor_3', 'sensor_4', 'sensor_6', 'sensor_7', 'sensor_8',
        'sensor_9', 'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14',
        'sensor_15', 'sensor_17', 'sensor_20', 'sensor_21'
    ]
    
    # Apply health indicator feature engineering
    df_features = engineer_health_indicators(df, sensor_cols)
    
    # Ensure output folder exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # Save the dataframe with new features
    df_features.to_csv(output_csv, index=False)
    print(f"Engineered health features saved to {output_csv}")

if __name__ == '__main__':
    main()
