# Script to load the best model from MLflow and predict Remaining Useful Life (RUL) on a test dataset.

# Import necessary libraries
import mlflow
import mlflow.lightgbm
import mlflow.xgboost
import mlflow.sklearn
import pandas as pd
import numpy as np
from pathlib import Path
from mlflow.tracking import MlflowClient

# Config
mlflow.set_tracking_uri("http://localhost:5000")

experiment_names = [
    "FD001 RUL Hyperparam Tuning",             
    "FD001 RUL XGBoost Hyperparam Tuning",
    "FD001 RUL LightGBM Hyperparam Tuning"
]

metric_to_optimize = "RMSE"  # Or "MAE", etc.

def get_best_run_for_experiment(experiment_name, metric):
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Experiment '{experiment_name}' not found.")
        return None

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=[f"metrics.{metric} ASC"],
        max_results=1
    )
    return runs[0] if runs else None

def main():
    best_runs = []
    for exp_name in experiment_names:
        run = get_best_run_for_experiment(exp_name, metric_to_optimize)
        if run:
            best_runs.append({
                "experiment": exp_name,
                "run_id": run.info.run_id,
                metric_to_optimize: run.data.metrics.get(metric_to_optimize)
            })

    if not best_runs:
        raise RuntimeError("No completed runs found in any experiment.")

    # Find overall best run based on the metric
    best_run = min(best_runs, key=lambda x: x[metric_to_optimize])
    print(f"Best overall run: {best_run}")

    run_id = best_run["run_id"]

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

    model_uri = f"runs:/{run_id}/{model_name}"
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

    # Save predictions
    output_path = Path("data/processed/rul_predictions.csv")
    rul_df.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path.resolve()}")

if __name__ == "__main__":
    main()
