# This script trains a baseline Random Forest model for predicting Remaining Useful Life (RUL) using feature-engineered data.

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, max_error
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
from pathlib import Path

def main():
    # Set MLflow tracking URI - adjust if needed, else defaults to ./mlruns
    experiment_name = "FD001 RUL Prediction"
    mlflow.set_experiment(experiment_name)

    # Load your dataset
    data_path = Path("data/features/train_FD001_features.csv")
    df = pd.read_csv(data_path)

    # Define features and target
    # Exclude columns not relevant to features, and use engineered sensor columns
    feature_cols = [col for col in df.columns if ("sensor" in col or "health_score" in col) and col != "failure_binary"]
    target_col = "RUL"  # Assuming you have RUL column labeled
    
    X = df[feature_cols]
    y = df[target_col]

    # Split into train and test for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define model hyperparameters
    n_estimators = 100
    max_depth = 10
    random_state = 42

    # Initialize model
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
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

        # Save to a temp CSV
        importances_path = Path("feature_importances.csv")
        importances.to_csv(importances_path, index=False)
        mlflow.log_artifact(str(importances_path))

        # Log the trained model itself
        mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="random_forest_model",
        input_example=X_test.iloc[:5],
        signature=mlflow.models.infer_signature(X_test, y_pred))

        print(f"Model trained and logged under experiment '{experiment_name}'.")
        print(f"Metrics: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.2f}, Max Error={max_err:.2f}")

if __name__ == "__main__":
    main()