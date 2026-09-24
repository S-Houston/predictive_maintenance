# Script to compare the best models from hyperparameter tuning experiments in MLflow

# Import necessary libraries
import mlflow
import os
from mlflow.tracking import MlflowClient
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set the local MLflow tracking URI
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))

# List of experiment names
experiment_names = [
    "FD001 RUL Hyperparam Tuning",
    "FD001 RUL XGBoost Hyperparam Tuning",
    "FD001 RUL LightGBM Hyperparam Tuning"
]

# Metrics to compare
metrics_to_compare = ["MAE", "RMSE", "R2", "Max Error"]

# Function to fetch best run based on lowest RMSE
def get_best_run_by_rmse(experiment_name):
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Experiment '{experiment_name}' not found.")
        return None
    
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["metrics.RMSE ASC"],
        max_results=1
    )
    return runs[0] if runs else None

# Collect best run from each experiment
best_runs = []
for exp_name in experiment_names:
    best_run = get_best_run_by_rmse(exp_name)
    if best_run:
        best_runs.append({
            "Model": exp_name.split("-")[-1].strip(),
            "Run ID": best_run.info.run_id,
            **{metric: best_run.data.metrics.get(metric) for metric in metrics_to_compare}
        })

# Convert to DataFrame
results_df = pd.DataFrame(best_runs)
print("Best Runs Across Experiments:\n", results_df)

# ----- Plotting -----

# Reshape DataFrame for easier plotting
df_melted = results_df.melt(id_vars=["Model"], 
                            value_vars=metrics_to_compare, 
                            var_name="Metric", 
                            value_name="Score")

# Optional: Invert metrics where lower is better (like MAE/RMSE/Max Error)
df_melted["Direction"] = df_melted["Metric"].apply(
    lambda m: "Higher is better" if m == "R2" else "Lower is better"
)

# Set Seaborn theme
sns.set(style="whitegrid", context="talk")

# Create the barplot
plt.figure(figsize=(12, 6))
sns.barplot(data=df_melted, x="Metric", y="Score", hue="Model")
plt.title("Top Model Comparison Across Metrics")
plt.ylabel("Metric Value")
plt.xlabel("Metric")
plt.legend(title="Model")
plt.tight_layout()
plt.show()
