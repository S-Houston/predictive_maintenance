# This script trains a LightGBM model for predicting Remaining Useful Life (RUL) using feature-engineered data.

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, max_error
import mlflow
import mlflow.lightgbm
from pathlib import Path
from mlflow.models import infer_signature
import json
import random

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def train_and_evaluate(params, X_train, X_test, y_train, y_test, feature_cols):
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.set_tag("model_type", "lightgbm")

        model = lgb.LGBMRegressor(**params)
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

        # Save feature importances
        importances = pd.DataFrame({
            "feature": feature_cols,
            "importance": model.feature_importances_
        }).sort_values(by="importance", ascending=False)

        importances_path = Path("feature_importances_lightgbm_tuning.csv")
        importances.to_csv(importances_path, index=False)
        mlflow.log_artifact(str(importances_path))

        # ✅ Save feature columns used in training
        feature_list_path = Path("lightgbm_feature_columns.json")
        with open(feature_list_path, "w") as f:
            json.dump(feature_cols, f)
        mlflow.log_artifact(str(feature_list_path))

        # Log the model to MLflow
        X_test_float = X_test.astype(np.float64)
        mlflow.lightgbm.log_model(
            lgb_model=model,
            name="lightgbm_model",
            input_example=X_test_float.iloc[:5],
            signature=infer_signature(X_test_float, y_pred)
        )

        print(f"Run params: {params}")
        print(f"Metrics: MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}, Max Error={max_err:.3f}")

        return mae

def main():
    mlflow.set_tracking_uri("http://localhost:5000")
    experiment_name = "FD001 RUL LightGBM Hyperparam Tuning"
    mlflow.set_experiment(experiment_name)
    print(f"Using MLflow experiment: '{experiment_name}'")
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")

    data_path = Path("data/features/train_FD001_features.csv")
    df = pd.read_csv(data_path)

    # Dynamically select relevant features
    feature_cols = [col for col in df.columns if ("sensor" in col or "health_score" in col) and col != "failure_binary"]
    target_col = "RUL"
    X = df[feature_cols].astype(np.float64)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Random hyperparameter search
    param_grid = {
        "objective": ["regression"],
        "metric": ["rmse"],
        "boosting_type": ["gbdt"],
        "num_leaves": [15, 31, 50],
        "learning_rate": [0.01, 0.05, 0.1],
        "n_estimators": [50, 100, 150],
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
