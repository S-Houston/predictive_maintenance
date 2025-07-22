# Script to generate failure labels for predictive maintenance datasets
import pandas as pd
from pathlib import Path

# Create a function to generate failure labels
def generate_failure_labels(input_path, output_path, failure_threshold=30):
    """
    Generate failure labels based on a threshold for the number of failures.
    
    Parameters:
    - input_path: Path to the input CSV file containing failure data.
    - output_path: Path to save the output CSV file with failure labels.
    - failure_threshold: Number of failures to consider for labeling.
    """
    # Load the dataset
    df = pd.read_csv(input_path)
    
    # Calculate max cycle per engine
    max_cycle = df.groupby('unit')['time'].transform('max')
    
    # Calculate RUL (Remaining Useful Life)
    df['RUL'] = max_cycle - df['time']
    
    # Generate binary failure labels
    df['failure_binary'] = (df['RUL'] <= failure_threshold).astype(int)
    
    # Save the labeled dataset
    Path(output_path).parent.mkdir(parents=True, exist_ok=True) # Ensure the output directory exists
    df.to_csv(output_path, index=False)
    print(f"Failure labels generated and saved to {output_path}")