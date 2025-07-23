# Script to load a LightGBM model from MLflow and predict Remaining Useful Life (RUL) on a test dataset since LightGBM was the best model in the tuning phase.

# Import necessary libraries
import mlflow
import mlflow.lightgbm
import pandas as pd
import numpy as np
from pathlib import Path

# Config
mlflow.set_tracking_uri("file:///C:/Users/Stuart/mlflow_tracking")
RUN_ID = "1bab934b5f99407ea6899069f3ad2e60"  # Replace with actual run_id from MLflow UI
MODEL_URI = f"runs:/{RUN_ID}/lightgbm_model"

# Load model from MLflow
print(f"Loading model from MLflow run: {RUN_ID}")
model = mlflow.lightgbm.load_model(MODEL_URI)

# Load training data to get feature columns
train_path = Path("data/features/train_FD001_features.csv")
train_df = pd.read_csv(train_path)

# Select the same features used in training
feature_cols = [col for col in train_df.columns if ("sensor" in col or "health_score" in col) and col != "failure_binary"]

# Load test set
test_path = Path("data/features/test_FD001_features.csv")
test_df = pd.read_csv(test_path)

# Select only the features used during training
X_test = test_df[feature_cols]

# Predict RUL
rul_preds = model.predict(X_test)
rul_df = pd.DataFrame({
    "unit": test_df["unit"],
    "RUL": np.round(rul_preds, 2)
})

# Save predictions
output_path = Path("data/processed/rul_predictions.csv")
rul_df.to_csv(output_path, index=False)
print(f"Saved predictions to {output_path.resolve()}")