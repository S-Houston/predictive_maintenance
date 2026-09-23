# This script trains a baseline XGBoost model for predicting Remaining Useful Life (RUL) using feature-engineered data.

# Import necessary libraries
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, max_error
from sklearn.model_selection import GroupShuffleSplit
import mlflow
import os
import mlflow.xgboost
from mlflow.models import infer_signature
from pathlib import Path
import random

# Single seed for the grouped validation split, the hyperparameter search and
# model initialisation. Changing it changes the split too, so all runs in an
# experiment share one validation set.
SEED = 42

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def train_and_evaluate(params, X_train, X_test, y_train, y_test, feature_cols):
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_param("search_seed", SEED)
        mlflow.set_tag("model_type", "xgboost")

        # Ensure correct feature columns
        X_train = X_train[feature_cols]
        X_test = X_test[feature_cols]

        model = XGBRegressor(**params, verbosity=0)
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

        importances = pd.DataFrame({
            "feature": feature_cols,
            "importance": model.feature_importances_
        }).sort_values(by="importance", ascending=False)

        importances_path = Path("feature_importances_xgb_tuning.csv")
        importances.to_csv(importances_path, index=False)
        mlflow.log_artifact(str(importances_path))

        X_test_float = X_test.astype(np.float64)

        mlflow.xgboost.log_model(
            xgb_model=model,
            name="xgboost_model",
            input_example=X_test_float.iloc[:5],
            signature=infer_signature(X_test_float, y_pred)
        )

        print(f"Run params: {params}")
        print(f"Metrics: MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}, Max Error={max_err:.3f}")

        return mae

def main():
    # --- MLflow setup ---
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    experiment_name = "FD001 RUL XGBoost Hyperparam Tuning (unit split, seeded)"
    mlflow.set_experiment(experiment_name)
    print(f"Using MLflow experiment: '{experiment_name}'")
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")

    # Load data
    data_path = Path("data/features/train_FD001_features.csv")
    df = pd.read_csv(data_path)

    feature_cols = [col for col in df.columns if ("sensor" in col or "health_score" in col) and col != "failure_binary"]
    target_col = "RUL"

    X = df[feature_cols]
    y = df[target_col]

    # Split by engine so no unit's cycles appear in both train and validation
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, test_idx = next(splitter.split(X, y, groups=df["unit"]))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Define hyperparameter grid for random sampling
    param_grid = {
        "n_estimators": [50, 100, 150, 200],
        "max_depth": [3, 5, 7, 10],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "random_state": [SEED]
    }

    n_iterations = 10
    best_mae = float("inf")
    best_params = None

    rng = random.Random(SEED)  # local RNG: unaffected by global random state
    for _ in range(n_iterations):
        params = {k: rng.choice(v) for k, v in param_grid.items()}
        mae = train_and_evaluate(params, X_train, X_test, y_train, y_test, feature_cols)

        if mae < best_mae:
            best_mae = mae
            best_params = params

    print(f"\nBest MAE: {best_mae:.3f}")
    print(f"Best Parameters: {best_params}")

if __name__ == "__main__":
    main()
