# This script trains a Random Forest model for predicting Remaining Useful Life (RUL) using feature-engineered data.

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, max_error
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
from pathlib import Path
import random
import os
import json

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

import json

# Utility function to save feature column names
def save_feature_columns(feature_cols, filepath):
    with open(filepath, "w") as f:
        json.dump(feature_cols, f)

def train_and_evaluate(params, X_train, X_test, y_train, y_test, feature_cols):
    with mlflow.start_run():
        mlflow.log_params(params)

        model = RandomForestRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            random_state=params["random_state"]
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        max_err = max_error(y_test, y_pred)

        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)
        mlflow.log_metric("Max Error", max_err)

        # Save and log feature importances
        importances = pd.DataFrame({
            "feature": feature_cols,
            "importance": model.feature_importances_
        }).sort_values(by="importance", ascending=False)

        importances_path = Path("feature_importances_rf_tuning.csv")
        importances.to_csv(importances_path, index=False)
        mlflow.log_artifact(str(importances_path))

        # Save and log feature column names
        feature_path = Path("feature_columns.json")
        save_feature_columns(feature_cols, feature_path)
        mlflow.log_artifact(str(feature_path))

        # Log the model
        mlflow.sklearn.log_model(
            sk_model=model,
            name="random_forest_model",
            input_example=X_test.iloc[:5],
            signature=mlflow.models.infer_signature(X_test, y_pred)
        )

        print(f"Run params: {params}")
        print(f"Metrics: MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}, Max Error={max_err:.3f}")
        return mae

def main():
    # --- [MLflow Setup] ---
    mlflow.set_tracking_uri("file:///C:/Users/Stuart/mlflow_tracking")
    experiment_name = "FD001 RUL Hyperparam Tuning"
    mlflow.set_experiment(experiment_name)
    print(f"Using MLflow experiment: '{experiment_name}'")
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")

    # --- [Data Load] ---
    data_path = Path("data/features/train_FD001_features.csv")
    df = pd.read_csv(data_path)

    feature_cols = [col for col in df.columns if ("sensor" in col or "health_score" in col) and col != "failure_binary"]
    target_col = "RUL"

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- [Hyperparameter Search] ---
    param_grid = {
        "n_estimators": [50, 100, 150, 200],
        "max_depth": [None, 5, 10, 15],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
        "random_state": [42]
    }

    n_iterations = 10
    best_mae = float("inf")
    best_params = None

    for _ in range(n_iterations):
        params = {k: random.choice(v) for k, v in param_grid.items()}
        mae = train_and_evaluate(params, X_train, X_test, y_train, y_test, feature_cols)

        if mae < best_mae:
            best_mae = mae
            best_params = params

    print(f"\nBest MAE: {best_mae:.3f}")
    print(f"Best Parameters: {best_params}")

if __name__ == "__main__":
    main()