# This script trains a baseline XGBoost model for predicting Remaining Useful Life (RUL) using feature-engineered data.

# Import necessary libraries
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, max_error
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.xgboost
from mlflow.models import infer_signature
from pathlib import Path

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def main():
    # Set MLflow tracking URI - adjust if needed, else defaults to ./mlruns
    experiment_name = "FD001 RUL Prediction"
    mlflow.set_experiment(experiment_name)

    # Load your dataset
    data_path = Path("data/features/train_FD001_features.csv")
    df = pd.read_csv(data_path)

    # Define features and target
    feature_cols = [col for col in df.columns if ("sensor" in col or "health_score" in col) and col != "failure_binary"]
    target_col = "RUL"
    
    X = df[feature_cols]
    y = df[target_col]

    # Split into train and test for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define model hyperparameters
    n_estimators = 100
    max_depth = 5
    learning_rate = 0.1
    subsample = 0.8
    colsample_bytree = 0.8
    random_state = 42

    # Initialize model
    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        verbosity=0
    )

    # Start MLflow run context
    with mlflow.start_run():
        # Train the model
        model.fit(X_train, y_train)

        # Predict on test set
        y_pred = model.predict(X_test)

        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        max_err = max_error(y_test, y_pred)

        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("subsample", subsample)
        mlflow.log_param("colsample_bytree", colsample_bytree)
        mlflow.log_param("random_state", random_state)

        # Log metrics
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)
        mlflow.log_metric("Max Error", max_err)

        # Log feature importances as a CSV artifact
        importances = pd.DataFrame({
            "feature": feature_cols,
            "importance": model.feature_importances_
        }).sort_values(by="importance", ascending=False)

        importances_path = Path("feature_importances_xgb.csv")
        importances.to_csv(importances_path, index=False)
        mlflow.log_artifact(str(importances_path))

        # Safely convert X_test input to float64 to avoid MLflow schema warnings
        X_test_float = X_test.astype(np.float64)

        # Log the trained model with input example and schema signature
        mlflow.xgboost.log_model(
            xgb_model=model,
            name="xgboost_model",
            input_example=X_test_float.iloc[:5],
            signature=infer_signature(X_test_float, y_pred)
        )

        print(f"Model trained and logged under experiment '{experiment_name}'.")
        print(f"Metrics: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.2f}, Max Error={max_err:.2f}")

if __name__ == "__main__":
    main()
