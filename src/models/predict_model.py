"""
Predictive Maintenance Inference Script
=======================================

Loads the best trained model from MLflow, predicts Remaining Useful Life (RUL)
for test data, classifies risk, and logs the inference run in MLflow.

Intended Use:
-------------
- Load the best model from hyperparameter tuning experiments
- Predict RUL for test data
- Classify risk levels based on RUL
- Save predictions to CSV
- Log metrics and artifacts to MLflow
- Can be run as a standalone script or integrated into a larger pipeline
- Return predictions in a structured format
- Can be used for batch inference or single-unit queries

Author: Stuart Houston
Date: 20-08-2025
"""

# Import necessary libraries
import mlflow
import mlflow.lightgbm
import mlflow.xgboost
import mlflow.sklearn
import pandas as pd
import numpy as np
from pathlib import Path
from mlflow.tracking import MlflowClient
from mlflow import log_artifact, log_metric, start_run

# Config
mlflow.set_tracking_uri("http://localhost:5000")
experiment_name = "FD001 RUL Inference"
metric_to_optimise = "RMSE"

training_experiments = [
    "FD001 RUL Hyperparam Tuning",             
    "FD001 RUL XGBoost Hyperparam Tuning",
    "FD001 RUL LightGBM Hyperparam Tuning"
]

# Create or get the experiment
def get_best_training_run(metric=metric_to_optimise):
    client = MlflowClient()
    best_runs = []
    for exp_name in training_experiments:
        experiment = client.get_experiment_by_name(exp_name)
        if experiment is None:
            print(f"Experiment '{exp_name}' not found.")
            continue

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="attributes.status = 'FINISHED'",
            order_by=[f"metrics.{metric} ASC"],
            max_results=1
        )
        if runs:
            run = runs[0]
            best_runs.append({
                "experiment": exp_name,
                "run_id": run.info.run_id,
                metric: run.data.metrics.get(metric)
            })
    if not best_runs:
        raise RuntimeError("No completed runs found in any training experiment.")
    return min(best_runs, key=lambda x: x[metric])

def classify_risk(rul: float) -> str:
    """
    Classifies risk level based on Remaining Useful Life (RUL).
    """
    if rul < 30:
        return "High"
    elif rul < 100:
        return "Medium"
    return "Low"

def main():
    best_run = get_best_training_run()
    print(f"Best overall run: {best_run}")

    # Determine model type from experiment name (optional, for loading)
    if "LightGBM" in best_run["experiment"]:
        model_loader = mlflow.lightgbm.load_model
        model_name = "lightgbm_model"
    elif "XGBoost" in best_run["experiment"]:
        model_loader = mlflow.xgboost.load_model
        model_name = "xgboost_model"
    else:
        # Default to sklearn (assumed RandomForest or baseline)
        model_loader = mlflow.sklearn.load_model
        model_name = "model"

    model_uri = f"runs:/{best_run['run_id']}/{model_name}"
    print(f"Loading model from: {model_uri}")

    model = model_loader(model_uri)

    # Load train data to get features
    train_path = Path("data/features/train_FD001_features.csv")
    train_df = pd.read_csv(train_path)
    feature_cols = [col for col in train_df.columns if ("sensor" in col or "health_score" in col) and col != "failure_binary"]

    # Load test data
    test_path = Path("data/features/test_FD001_features.csv")
    test_df = pd.read_csv(test_path)
    X_test = test_df[feature_cols]

    # Predict
    rul_preds = model.predict(X_test)
    rul_df = pd.DataFrame({
        "unit": test_df["unit"],
        "RUL": np.round(rul_preds, 2)
    })
    rul_df["risk_level"] = rul_df["RUL"].apply(classify_risk)

    # Save predictions
    output_path = Path("data/processed/rul_predictions.csv")
    rul_df.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path.resolve()}")
    
    # Log inference run in MLflow
    mlflow.set_experiment(experiment_name)
    print(f"Logging inference run in experiment: {experiment_name}")
    with mlflow.start_run(run_name="Inference Run") as run:
        mlflow.log_param("model_run_id", best_run["run_id"])
        mlflow.log_param("num_predictions", len(rul_df))
        # Optionally log aggregated metrics
        mlflow.log_metric("mean_RUL", rul_df["RUL"].mean())
        mlflow.log_metric("high_risk_count", (rul_df["risk_level"] == "High").sum())
        # Log predictions CSV as artifact
        mlflow.log_artifact(str(output_path))
        print(f"Inference logged in MLflow under run ID: {run.info.run_id}")

if __name__ == "__main__":
    main()
