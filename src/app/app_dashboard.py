"""
Predictive Maintenance - Streanmlit Dashboard
==========================

This Streamlit Dashboard is intended to provide users with an interactive overview of engine performance. 
It visualises sensor data trends, Remaining Useful Life (RUL) predictions, and risk classifications for predictive maintenance.
Data is sourced from CSV files and displayed in an intuitive format.  Cached data loading is implemented to improve performance.

Intended Use:
-------------
- Provide visual insights into sensor data trends
- Display Remaining Useful Life (RUL) predictions
- Allow unit-level analysis and risk classification
- Enable comparison of predicted vs. true RUL values

Author: [Stuart Houston]
Date: 19-08-2025
"""

# Import necessary libraries
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Data Loading Functions

@st.cache_data # Cache data loading to improve performance
def load_features(path):
    """Loads sensor feature data from a CSV file."""
    try:
        df = pd.read_csv(path)
        if "unit" not in df.columns or "time_in_cycles" not in df.columns:
            st.error(f"⚠️ Sensor data missing 'unit' or 'time_in_cycles' columns.\nColumns found: {df.columns.tolist()}")
            return None
        df["unit"] = pd.to_numeric(df["unit"], errors="coerce").astype("Int64")
        df["time_in_cycles"] = pd.to_numeric(df["time_in_cycles"], errors="coerce").astype("Int64")
        df.dropna(subset=["unit", "time_in_cycles"], inplace=True)
        return df
    except FileNotFoundError:
        st.error(f"🚨 Critical Error: Main feature file not found at '{path}'.")
        return None

@st.cache_data # Cache RUL predictions loading
def load_rul_predictions(path):
    """Loads RUL predictions from a CSV file."""
    try:
        rul_df = pd.read_csv(path)
        if not {"unit", "RUL"}.issubset(rul_df.columns):
            st.warning("RUL predictions file missing required columns 'unit' and 'RUL'.")
            return None
        return rul_df
    except FileNotFoundError:
        st.info("RUL prediction file not found. Some features will be disabled.")
        return None

@st.cache_data # Cache true RUL labels loading
def load_true_rul_labels(possible_paths):
    """Loads true RUL labels by checking a list of possible paths."""
    for p in possible_paths:
        path = Path(p)
        if path.exists():
            df = pd.read_csv(path)
            if {"unit", "RUL"}.issubset(df.columns):
                return df[["unit", "RUL"]].drop_duplicates()
    st.info("True RUL labels not found. Model performance comparison will not be available.")
    return None

# Helper Functions

def classify_risk(rul, high_threshold, medium_threshold):
    """Classifies RUL into 'High', 'Medium', or 'Low' risk categories."""
    if rul < high_threshold:
        return "High"
    elif rul < medium_threshold:
        return "Medium"
    else:
        return "Low"

def get_sensor_type(col_name, sensor_tooltips):
    """Determines the type of a sensor column based on its suffix."""
    for key in sensor_tooltips.keys():
        if col_name.endswith(key):
            return key
    return "raw"

# UI Rendering Functions

def render_summary_tab(rul_df, risk_color_map, high_risk_threshold, medium_risk_threshold):
    """Renders the content for the Summary tab."""
    st.header("Summary Overview")
    if rul_df is None or rul_df.empty:
        st.warning("No RUL prediction data available to summarize.")
        return

    # Top 5 units approaching failure
    st.subheader("Top 5 Units Approaching Failure (Lowest RUL)")
    top5_high_risk = rul_df.sort_values("RUL").head(5)
    cols = st.columns(len(top5_high_risk))
    for i, row in enumerate(top5_high_risk.itertuples()):
        with cols[i]:
            risk_color = risk_color_map.get(row.risk_level, "black")
            st.markdown(f"**Unit {int(row.unit)}**")
            st.markdown(f"<span style='color:{risk_color};font-weight:bold;'>Risk: {row.risk_level}</span>", unsafe_allow_html=True)
            st.metric(label="Remaining Useful Life", value=f"{row.RUL:.1f} cycles")

    # Risk distribution pie chart
    st.subheader("Unit Risk Distribution")
    risk_counts = rul_df["risk_level"].value_counts().reindex(["High", "Medium", "Low"], fill_value=0)
    fig_pie = px.pie(
        names=risk_counts.index,
        values=risk_counts.values,
        color=risk_counts.index,
        color_discrete_map=risk_color_map,
        title="Units by Risk Level"
    )
    st.plotly_chart(fig_pie, use_container_width=True)
    
    # Add the explanation in an expander
    with st.expander("How is risk calculated?"):
        st.markdown(f"""
        The risk level for each engine unit is determined by its **Predicted Remaining Useful Life (RUL)**.
        
        - A **cycle** represents one complete engine operation (e.g., a flight).
        - **Risk thresholds** are defined as follows:
            - **High Risk**: RUL < **{high_risk_threshold} cycles**
            - **Medium Risk**: RUL is between **{high_risk_threshold}** and **{medium_risk_threshold} cycles**
            - **Low Risk**: RUL ≥ **{medium_risk_threshold} cycles**
        
        Units flagged as **High Risk** should be prioritised for inspection or maintenance.
        """)

def render_overview_tab(df, sensor_tooltips):
    """Renders the content for the Sensor Overview tab."""
    st.header("Sensor Trends Across All Units")
    sensor_cols = [col for col in df.columns if col.startswith("sensor_")]
    
    # Create display names with type hint for the selectbox
    sensor_display_names = [f"{col} ({get_sensor_type(col, sensor_tooltips)})" for col in sensor_cols]
    display_to_col = dict(zip(sensor_display_names, sensor_cols))
    
    selected_display = st.selectbox("Select a sensor to visualize", sensor_display_names)
    selected_sensor = display_to_col[selected_display]

    fig = px.line(
        df, x="time_in_cycles", y=selected_sensor, color="unit",
        title=f"Trend of {selected_sensor} Across All Units"
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Sensor Explanation Glossary"):
        for stype, desc in sensor_tooltips.items():
            st.markdown(f"**{stype.capitalize()}**: {desc}")

def render_unit_analysis_tab(df, failure_threshold):
    """Renders the content for the Individual Unit Analysis tab."""
    st.header("Individual Unit Analysis")
    unit_ids = sorted(df["unit"].unique())
    selected_unit = st.selectbox("Select a unit", unit_ids)

    st.markdown(f"""
    **Note:** The red shaded area marks the last **{failure_threshold} cycles** before the unit's recorded failure (the "failure zone"). 
    Sensor readings in this zone often show accelerated degradation.
    """)

    # Filter multiselect to only show base sensor names for clarity
    base_sensor_cols = sorted([
        col for col in df.columns if col.startswith("sensor_") and not any(
            suffix in col for suffix in ["_rolling_mean", "_rolling_std", "_slope"])
    ])
    
    sensor_choices = st.multiselect(
        "Select sensors to display",
        base_sensor_cols,
        default=["sensor_2", "sensor_3"]
    )

    df_unit = df[df["unit"] == selected_unit].copy()
    for sensor in sensor_choices:
        rolling_col = f"{sensor}_rolling_mean"
        if rolling_col in df_unit.columns:
            fig = go.Figure()
            # Raw sensor data
            fig.add_trace(go.Scatter(x=df_unit["time_in_cycles"], y=df_unit[sensor],
                                     mode="lines", name=f"{sensor} (Raw)"))
            # Rolling average
            fig.add_trace(go.Scatter(x=df_unit["time_in_cycles"], y=df_unit[rolling_col],
                                     mode="lines", name=f"{sensor} (Rolling Avg)", line=dict(dash='dot')))
            # Failure zone rectangle
            failure_start = df_unit["max_cycle"].max() - failure_threshold
            fig.add_vrect(x0=failure_start, x1=df_unit["max_cycle"].max(), fillcolor="red", opacity=0.2,
                          layer="below", line_width=0)
            
            fig.update_layout(title=f"Analysis of {sensor} for Unit {selected_unit}",
                              xaxis_title="Time (cycles)", yaxis_title="Sensor Reading", legend=dict(x=0.01, y=0.99))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"Data for rolling average ('{rolling_col}') not found for {sensor}.")

def render_rul_predictions_tab(rul_df, true_rul_df, high_risk_threshold, alert_emojis):
    """Renders the content for the RUL Predictions tab."""
    st.header("Remaining Useful Life (RUL) Predictions")
    
    if rul_df is None or rul_df.empty:
        st.warning("No RUL predictions available to display.")
        return

    st.subheader("Predicted RUL and Risk Levels by Unit")
    rul_display_df = rul_df.copy()
    rul_display_df["Alert"] = rul_display_df["risk_level"].map(alert_emojis)
    st.dataframe(
        rul_display_df[["unit", "RUL", "Alert"]].sort_values("RUL").reset_index(drop=True),
        use_container_width=True
    )

    # Comparison with True RUL
    if true_rul_df is None:
        st.info("True RUL data is not available, so performance comparison cannot be shown.")
        return

    st.subheader("Model Performance: Prediction vs. Reality")
    merged = pd.merge(
        rul_df.rename(columns={"RUL": "RUL_pred"}),
        true_rul_df.rename(columns={"RUL": "RUL_true"}),
        on="unit"
    )

    if merged.empty:
        st.warning("No overlapping units found between predictions and true RUL data.")
        return

    merged["error"] = (merged["RUL_true"] - merged["RUL_pred"]).abs()
    unit_list = sorted(merged["unit"].unique())
    
    with st.expander("How to interpret this section"):
        st.markdown("""
        This view helps you assess the model's reliability on a per-engine basis.
        - **True RUL**: The actual number of cycles the engine ran before failure.
        - **Predicted RUL**: The model's estimate of remaining cycles.
        - **Absolute Error**: The difference between the true and predicted RUL.
        
        An **underestimate** (Predicted < True) is a "safe" error, leading to early maintenance.
        An **overestimate** (Predicted > True) is a "risky" error that could lead to unplanned failure.
        """)

    selected_unit_comp = st.selectbox("Select a Unit for Comparison", unit_list)
    row = merged[merged["unit"] == selected_unit_comp].iloc[0]

    col1, col2, col3 = st.columns(3)
    col1.metric("True RUL", f"{row['RUL_true']} cycles")
    col2.metric("Predicted RUL", f"{row['RUL_pred']:.2f} cycles")
    col3.metric("Absolute Error", f"{row['error']:.2f} cycles", delta=f"{row['RUL_pred'] - row['RUL_true']:.2f}", delta_color="inverse")

    if row['RUL_pred'] < high_risk_threshold:
        st.warning("This unit's predicted RUL is very low, flagging it for immediate attention.")

    # Bar chart for visual comparison
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=["True RUL", "Predicted RUL"],
        y=[row["RUL_true"], row["RUL_pred"]],
        marker_color=["#2ca02c", "#1f77b4"], # Green for true, Blue for predicted
        text=[f"{row['RUL_true']}", f"{row['RUL_pred']:.2f}"],
        textposition='auto'
    ))
    fig_bar.update_layout(title=f"RUL Comparison for Unit {selected_unit_comp}", yaxis_title="RUL (cycles)")
    st.plotly_chart(fig_bar, use_container_width=True)

# Main Application
def main():
    """Main function to run the Streamlit dashboard."""
    
    # App Configuration & Constants
    st.set_page_config(page_title="Engine Health Dashboard", layout="wide", initial_sidebar_state="expanded")
    
    # Paths and Thresholds
    FEATURES_PATH = "data/features/train_FD001_features.csv"
    PREDICTIONS_PATH = "data/processed/rul_predictions.csv"
    TRUE_RUL_PATHS = ["data/cleaned/train_FD001_labeled.csv", "data/cleaned/test_FD001_labeled.csv"]
    FAILURE_THRESHOLD = 30
    HIGH_RISK_THRESHOLD = 30
    MEDIUM_RISK_THRESHOLD = 100

    # UI Constants
    RISK_COLOR_MAP = {"High": "#d9534f", "Medium": "#f0ad4e", "Low": "#5cb85c"} # Red, Orange, Green
    ALERT_EMOJIS = {"High": "🔴 High", "Medium": "🟠 Medium", "Low": "🟢 Low"}
    SENSOR_TOOLTIPS = {
        "raw": "Raw sensor reading as collected from the engine.",
        "baseline": "Baseline value representing normal operating condition, typically the mean of the first few cycles per unit.",
        "degraded": "Deviation from baseline beyond the 75% threshold; 1 indicates degradation, 0 is normal.",
        "rolling_mean": "Rolling average of sensor readings to smooth short-term fluctuations.",
        "rolling_std": "Rolling standard deviation over a rolling window, showing variability.",
        "cycle_to_cycle_change": "Difference between consecutive cycles, highlighting abrupt changes or trends in sensor readings.",
        "health_score": "Sum of all degradation flags across sensors, providing a composite measure of overall unit health."
    }

    # App Title
    st.title("Predictive Maintenance Dashboard")

    # Load Data
    df = load_features(FEATURES_PATH)
    rul_df = load_rul_predictions(PREDICTIONS_PATH)
    true_rul_df = load_true_rul_labels(TRUE_RUL_PATHS)

    # Main Validation Block
    if df is None:
        st.stop()

    # Data Processing
    df["max_cycle"] = df.groupby("unit")["time_in_cycles"].transform("max")
    df["failure_zone"] = df["time_in_cycles"] >= (df["max_cycle"] - FAILURE_THRESHOLD)
    if rul_df is not None and "risk_level" not in rul_df.columns:
        st.warning("'risk_level' column not found in predictions CSV. Default thresholds will be applied.")
        rul_df["risk_level"] = rul_df["RUL"].apply(
            classify_risk, args=(HIGH_RISK_THRESHOLD, MEDIUM_RISK_THRESHOLD)
        )

    # UI Tabs
    tabs = st.tabs(["Summary", "Overview", "Unit Analysis", "RUL Predictions"])

    with tabs[0]:
        render_summary_tab(rul_df, RISK_COLOR_MAP, HIGH_RISK_THRESHOLD, MEDIUM_RISK_THRESHOLD)

    with tabs[1]:
        render_overview_tab(df, SENSOR_TOOLTIPS)

    with tabs[2]:
        render_unit_analysis_tab(df, FAILURE_THRESHOLD)

    with tabs[3]:
        render_rul_predictions_tab(rul_df, true_rul_df, HIGH_RISK_THRESHOLD, ALERT_EMOJIS)

if __name__ == "__main__":
    main()