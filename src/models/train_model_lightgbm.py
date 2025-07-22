# This script trains a LightGBM model for predicting Remaining Useful Life (RUL) using feature-engineered data.

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, max_error
import mlflow
import mlflow.lightgbm
from pathlib import Path
import mlflow.models

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def main():
    # Set MLflow experiment
    experiment_name = "FD001 RUL Prediction"
    mlflow.set_experiment(experiment_name)

    # Load dataset
    data_path = Path("data/features/train_FD001_features.csv")
    df = pd.read_csv(data_path)

    # Define features and target
    feature_cols = [col for col in df.columns if ("sensor" in col or "health_score" in col) and col != "failure_binary"]
    target_col = "RUL"
    X = df[feature_cols]
    y = df[target_col]

    # Cast all features to float64 to avoid MLflow integer missing value warnings
    X = X.astype(np.float64)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # LightGBM hyperparameters
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "n_estimators": 100,
        "random_state": 42
    }

    # Initialize model
    model = lgb.LGBMRegressor(**params)

    # Start MLflow tracking
    with mlflow.start_run():
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # Evaluation metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        max_err = max_error(y_test, y_pred)

        # Log hyperparameters
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)

        # Log metrics
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)
        mlflow.log_metric("Max Error", max_err)

        # Log feature importances
        importances = pd.DataFrame({
            "feature": feature_cols,
            "importance": model.feature_importances_
        }).sort_values(by="importance", ascending=False)

        importances_path = Path("feature_importances_lightgbm.csv")
        importances.to_csv(importances_path, index=False)
        mlflow.log_artifact(str(importances_path))

        # Log the model (using name= instead of deprecated artifact_path=)
        mlflow.lightgbm.log_model(
            lgb_model=model,
            name="lightgbm_model",
            input_example=X_test.iloc[:5],
            signature=mlflow.models.infer_signature(X_test, y_pred)
        )

        print(f"Model trained and logged under experiment '{experiment_name}'.")
        print(f"Metrics: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.2f}, Max Error={max_err:.2f}")

if __name__ == "__main__":
    main()
