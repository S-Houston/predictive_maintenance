"""
Predictive Maintenance - Streamlit Dashboard
============================================

Author: Stuart Houston
Date: 19-08-2025

This Streamlit dashboard visualizes engine sensor data, RUL predictions, and risk classifications
to support predictive maintenance. Features include:
- Overview of sensor trends across units
- Unit-level analysis with failure zone highlighting
- RUL predictions with model vs true comparisons
- Multi-unit comparison for selected sensors
"""

# Import necessary libraries
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# -------------------- Data Loading --------------------

@st.cache_data
def load_features(path):
    try:
        df = pd.read_csv(path)
        if "unit" not in df.columns or "time_in_cycles" not in df.columns:
            st.error(f"Sensor data missing 'unit' or 'time_in_cycles'. Columns found: {df.columns.tolist()}")
            return None
        df["unit"] = pd.to_numeric(df["unit"], errors="coerce").astype("Int64")
        df["time_in_cycles"] = pd.to_numeric(df["time_in_cycles"], errors="coerce").astype("Int64")
        if "health_score" in df.columns:
            df["health_score"] = pd.to_numeric(df["health_score"], errors="coerce").fillna(0)
        else:
            df["health_score"] = 0
        df.dropna(subset=["unit", "time_in_cycles"], inplace=True)
        return df
    except FileNotFoundError:
        st.error(f"Critical Error: Feature file not found at '{path}'")
        return None

@st.cache_data
def load_rul_predictions(path):
    try:
        rul_df = pd.read_csv(path)
        if not {"unit", "RUL"}.issubset(rul_df.columns):
            st.warning("RUL predictions file missing required columns 'unit' and 'RUL'.")
            return None
        return rul_df
    except FileNotFoundError:
        st.info("RUL predictions file not found. Some features will be disabled.")
        return None

@st.cache_data
def load_true_rul_labels(possible_paths):
    for p in possible_paths:
        path = Path(p)
        if path.exists():
            df = pd.read_csv(path)
            if {"unit", "RUL"}.issubset(df.columns):
                return df[["unit", "RUL"]].drop_duplicates()
    st.info("True RUL labels not found. Model performance comparison will not be available.")
    return None

# -------------------- Helper Functions --------------------

def classify_risk(rul, high_threshold, medium_threshold):
    if rul < high_threshold:
        return "High"
    elif rul < medium_threshold:
        return "Medium"
    else:
        return "Low"

def get_sensor_type(col_name, sensor_tooltips):
    for key in sensor_tooltips.keys():
        if col_name.endswith(key):
            return key
    return "raw"

# -------------------- UI Rendering Functions --------------------

def render_summary_tab(rul_df, risk_color_map, high_risk_threshold, medium_risk_threshold):
    st.header("Summary Overview")
    if rul_df is None or rul_df.empty:
        st.warning("No RUL prediction data available.")
        return

    top5_high_risk = rul_df.sort_values("RUL").head(5)
    cols = st.columns(len(top5_high_risk))
    for i, row in enumerate(top5_high_risk.itertuples()):
        with cols[i]:
            risk_color = risk_color_map.get(row.risk_level, "black")
            st.markdown(f"**Unit {int(row.unit)}**")
            st.markdown(f"<span style='color:{risk_color};font-weight:bold;'>Risk: {row.risk_level}</span>", unsafe_allow_html=True)
            st.metric(label="Remaining Useful Life", value=f"{row.RUL:.1f} cycles")

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

    with st.expander("How is risk calculated?"):
        st.markdown(f"""
        Risk is based on **Predicted Remaining Useful Life (RUL)**:
        - **High Risk**: RUL < **{high_risk_threshold} cycles**
        - **Medium Risk**: RUL between **{high_risk_threshold}** and **{medium_risk_threshold} cycles**
        - **Low Risk**: RUL ≥ **{medium_risk_threshold} cycles**
        """)

def render_overview_tab(df, sensor_tooltips):
    """Renders the Sensor Overview tab with filtered sensors and multi-unit comparison."""
    st.header("Sensor Trends Overview")

    # Only include processed/degraded sensors + derived metrics by default
    processed_cols = [col for col in df.columns if any(suffix in col for suffix in ["degraded", "rolling_mean", "rolling_std", "health_score"])]
    
    # Map columns to display names with type hint
    sensor_display_names = [f"{col} ({get_sensor_type(col, sensor_tooltips)})" for col in processed_cols]
    display_to_col = dict(zip(sensor_display_names, processed_cols))

    selected_display = st.selectbox("Select a sensor to visualize", sensor_display_names)
    selected_sensor = display_to_col[selected_display]

    # Unit selection controls
    unit_ids = sorted(df["unit"].unique())
    view_mode = st.radio(
        "Select view mode:",
        options=["Single Unit", "Compare Units", "All Units"]
    )

    if view_mode == "Single Unit":
        selected_unit = st.selectbox("Select a unit", unit_ids, index=0)
        df_plot = df[df["unit"] == selected_unit]
        title = f"Trend of {selected_sensor} for Unit {selected_unit}"

    elif view_mode == "Compare Units":
        selected_units = st.multiselect("Select units to compare", unit_ids, default=unit_ids[:2])
        if not selected_units:
            st.warning("Please select at least one unit to compare.")
            return
        df_plot = df[df["unit"].isin(selected_units)]
        title = f"Trend of {selected_sensor} for Units {', '.join(map(str, selected_units))}"

    else:  # All Units
        df_plot = df
        title = f"Trend of {selected_sensor} Across All Units"

    # Plot
    fig = px.line(
        df_plot,
        x="time_in_cycles",
        y=selected_sensor,
        color="unit" if view_mode != "Single Unit" else None,
        title=title
    )
    st.plotly_chart(fig, use_container_width=True)

    # Sensor glossary
    with st.expander("Sensor Explanation Glossary"):
        for stype, desc in sensor_tooltips.items():
            st.markdown(f"**{stype.capitalize()}**: {desc}")


def render_unit_analysis_tab(df, failure_threshold):
    """Renders the Individual Unit Analysis tab with multi-unit comparison."""
    st.header("Individual Unit Analysis")

    unit_ids = sorted(df["unit"].unique())
    selected_units = st.multiselect("Select units to display", unit_ids, default=unit_ids[:2])
    if not selected_units:
        st.warning("Please select at least one unit.")
        return

    st.markdown(f"""
    **Note:** The red shaded area marks the last **{failure_threshold} cycles** before each unit's recorded failure (the "failure zone"). 
    Sensor readings in this zone often show accelerated degradation.
    """)

    # Filter for base sensors (no rolling stats)
    base_sensor_cols = sorted([
        col for col in df.columns if col.startswith("sensor_") and not any(
            suffix in col for suffix in ["_rolling_mean", "_rolling_std", "_slope"])
    ])

    sensor_choices = st.multiselect(
        "Select sensors to display",
        base_sensor_cols,
        default=["sensor_2", "sensor_3"]
    )

    for sensor in sensor_choices:
        fig = go.Figure()
        for unit in selected_units:
            df_unit = df[df["unit"] == unit].copy()
            rolling_col = f"{sensor}_rolling_mean"

            # Raw sensor data
            fig.add_trace(go.Scatter(
                x=df_unit["time_in_cycles"], y=df_unit[sensor],
                mode="lines", name=f"{sensor} (Raw) - Unit {unit}"
            ))

            # Rolling average if exists
            if rolling_col in df_unit.columns:
                fig.add_trace(go.Scatter(
                    x=df_unit["time_in_cycles"], y=df_unit[rolling_col],
                    mode="lines", name=f"{sensor} (Rolling Avg) - Unit {unit}", line=dict(dash='dot')
                ))

            # Failure zone rectangle
            failure_start = df_unit["max_cycle"].max() - failure_threshold
            fig.add_vrect(
                x0=failure_start, x1=df_unit["max_cycle"].max(),
                fillcolor="red", opacity=0.15, layer="below", line_width=0
            )

        fig.update_layout(
            title=f"Comparison of {sensor} Across Units {', '.join(map(str, selected_units))}",
            xaxis_title="Time (cycles)", yaxis_title="Sensor Reading",
            legend=dict(x=0.01, y=0.99)
        )
        st.plotly_chart(fig, use_container_width=True)
        
def render_rul_predictions_tab(rul_df, true_rul_df, high_risk_threshold, alert_emojis):
    st.header("Remaining Useful Life (RUL) Predictions")
    if rul_df is None or rul_df.empty:
        st.warning("No RUL predictions available.")
        return

    rul_display_df = rul_df.copy()
    rul_display_df["Alert"] = rul_display_df["risk_level"].map(alert_emojis)
    st.dataframe(rul_display_df[["unit", "RUL", "Alert"]].sort_values("RUL").reset_index(drop=True), use_container_width=True)

    if true_rul_df is None:
        st.info("True RUL data not available for comparison.")
        return

    merged = pd.merge(rul_df.rename(columns={"RUL": "RUL_pred"}), true_rul_df.rename(columns={"RUL": "RUL_true"}), on="unit")
    if merged.empty:
        st.warning("No overlapping units found between predictions and true RUL data.")
        return
    merged["error"] = (merged["RUL_true"] - merged["RUL_pred"]).abs()
    unit_list = sorted(merged["unit"].unique())

    selected_unit_comp = st.selectbox("Select a Unit for Comparison", unit_list, key="rul_comp_unit")
    row = merged[merged["unit"] == selected_unit_comp].iloc[0]

    col1, col2, col3 = st.columns(3)
    col1.metric("True RUL", f"{row['RUL_true']} cycles")
    col2.metric("Predicted RUL", f"{row['RUL_pred']:.2f} cycles")
    col3.metric("Absolute Error", f"{row['error']:.2f} cycles", delta=f"{row['RUL_pred'] - row['RUL_true']:.2f}", delta_color="inverse")
    if row['RUL_pred'] < high_risk_threshold:
        st.warning("Predicted RUL is very low – prioritize maintenance.")

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(x=["True RUL", "Predicted RUL"], y=[row["RUL_true"], row["RUL_pred"]],
                             marker_color=["#2ca02c", "#1f77b4"], text=[f"{row['RUL_true']}", f"{row['RUL_pred']:.2f}"],
                             textposition='auto'))
    fig_bar.update_layout(title=f"RUL Comparison for Unit {selected_unit_comp}", yaxis_title="RUL (cycles)")
    st.plotly_chart(fig_bar, use_container_width=True)

# -------------------- Main Application --------------------

def main():
    st.set_page_config(page_title="Engine Health Dashboard", layout="wide", initial_sidebar_state="expanded")

    FEATURES_PATH = "data/features/train_FD001_features.csv"
    PREDICTIONS_PATH = "data/processed/rul_predictions.csv"
    TRUE_RUL_PATHS = ["data/cleaned/train_FD001_labeled.csv", "data/cleaned/test_FD001_labeled.csv"]
    FAILURE_THRESHOLD = 30
    HIGH_RISK_THRESHOLD = 30
    MEDIUM_RISK_THRESHOLD = 100

    RISK_COLOR_MAP = {"High": "#d9534f", "Medium": "#f0ad4e", "Low": "#5cb85c"}
    ALERT_EMOJIS = {"High": "🔴 High", "Medium": "🟠 Medium", "Low": "🟢 Low"}
    SENSOR_TOOLTIPS = {
        "raw": "Raw sensor reading as collected from the engine.",
        "baseline": "Baseline value representing normal operating condition.",
        "degraded": "Deviation from baseline beyond threshold.",
        "rolling_mean": "Rolling average of sensor readings.",
        "rolling_std": "Rolling standard deviation.",
        "cycle_to_cycle_change": "Difference between consecutive cycles.",
        "health_score": "Composite measure of overall unit health."
    }

    st.title("Predictive Maintenance Dashboard")

    df = load_features(FEATURES_PATH)
    rul_df = load_rul_predictions(PREDICTIONS_PATH)
    true_rul_df = load_true_rul_labels(TRUE_RUL_PATHS)

    if df is None:
        st.stop()

    df["max_cycle"] = df.groupby("unit")["time_in_cycles"].transform("max")
    df["failure_zone"] = df["time_in_cycles"] >= (df["max_cycle"] - FAILURE_THRESHOLD)

    if rul_df is not None and "risk_level" not in rul_df.columns:
        rul_df["risk_level"] = rul_df["RUL"].apply(classify_risk, args=(HIGH_RISK_THRESHOLD, MEDIUM_RISK_THRESHOLD))

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
