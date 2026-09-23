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
- Commercial analysis with estimated costs and savings
"""

# Import necessary libraries
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# --- Assumptions for demo ---
COST_PER_HOUR = 5000
DOWNTIME_HOURS_FAILURE = 48
DOWNTIME_HOURS_PROACTIVE = 8
PROACTIVE_FIXED_COST = 20000
CRITICAL_THRESHOLD = 30
WARNING_THRESHOLD = 60

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
def load_true_rul_labels(path):
    """True RUL at each test unit's last observed cycle (labels include the RUL_FD001 offset)."""
    path = Path(path)
    if path.exists():
        df = pd.read_csv(path)
        if {"unit", "time_in_cycles", "RUL"}.issubset(df.columns):
            return latest_per_unit(df)[["unit", "RUL"]]
    st.info("True RUL labels not found. Model performance comparison will not be available.")
    return None

# -------------------- Helper Functions --------------------

def latest_per_unit(df):
    """One row per unit: the most recent observed cycle (the unit's current state)."""
    if "time_in_cycles" in df.columns:
        df = df.sort_values(["unit", "time_in_cycles"])
    return df.groupby("unit").tail(1).reset_index(drop=True)

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
    st.header("Sensor Trends Overview")

    processed_cols = [col for col in df.columns if any(suffix in col for suffix in ["degraded", "rolling_mean", "rolling_std"])]
    sensor_display_names = [f"{col} ({get_sensor_type(col, sensor_tooltips)})" for col in processed_cols]
    display_to_col = dict(zip(sensor_display_names, processed_cols))

    selected_display = st.selectbox("Select a sensor to visualize", sensor_display_names)
    selected_sensor = display_to_col[selected_display]

    unit_ids = sorted(df["unit"].unique())
    view_mode = st.radio("Select view mode:", options=["Single Unit", "Compare Units", "All Units"])

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
    else:
        df_plot = df
        title = f"Trend of {selected_sensor} Across All Units"

    fig = px.line(df_plot, x="time_in_cycles", y=selected_sensor,
                  color="unit" if view_mode != "Single Unit" else None, title=title)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Sensor Explanation Glossary"):
        for stype, desc in sensor_tooltips.items():
            st.markdown(f"**{stype.capitalize()}**: {desc}")

def render_unit_analysis_tab(df, failure_threshold, rul_df=None):
    st.header("Individual Unit Analysis")

    unit_ids = sorted(df["unit"].unique())
    selected_units = st.multiselect("Select units to display", unit_ids, default=unit_ids[:2])
    if not selected_units:
        st.warning("Please select at least one unit.")
        return

    st.markdown(f"""
    **Note:**  
    - The **orange shaded area** marks the recommended **optimum maintenance window** (~20 cycles before the critical threshold).  
    - The **red shaded area** marks the last **{failure_threshold} cycles** before each unit's recorded failure.
    """)

    base_sensor_cols = sorted([col for col in df.columns if col.startswith("sensor_") and not any(suffix in col for suffix in ["_rolling_mean", "_rolling_std", "_slope"])])
    sensor_choices = st.multiselect("Select sensors to display", base_sensor_cols, default=["sensor_2", "sensor_3"])

    for sensor in sensor_choices:
        fig = go.Figure()
        for unit in selected_units:
            df_unit = df[df["unit"] == unit].copy()
            rolling_col = f"{sensor}_rolling_mean"

            fig.add_trace(go.Scatter(
                x=df_unit["time_in_cycles"], y=df_unit[sensor],
                mode="lines", name=f"{sensor} (Raw) - Unit {unit}"
            ))

            if rolling_col in df_unit.columns:
                fig.add_trace(go.Scatter(
                    x=df_unit["time_in_cycles"], y=df_unit[rolling_col],
                    mode="lines", name=f"{sensor} (Rolling Avg) - Unit {unit}", line=dict(dash='dot')
                ))

            max_cycle = df_unit["max_cycle"].max()
            failure_start = max_cycle - failure_threshold
            optimum_start = failure_start - 20  # 20 cycles before critical zone

            # Orange zone (optimum proactive window: 50–30 cycles before failure)
            fig.add_vrect(
                x0=optimum_start, x1=failure_start,
                fillcolor="orange", opacity=0.15, layer="below", line_width=0,
                annotation_text="Optimum Maintenance Window", annotation_position="top right"
            )

            # Red zone (critical: <30 cycles to failure)
            fig.add_vrect(
                x0=failure_start, x1=max_cycle,
                fillcolor="red", opacity=0.15, layer="below", line_width=0,
                annotation_text="Critical Zone", annotation_position="top left"
            )
        fig.update_layout(title=f"Comparison of {sensor} Across Units {', '.join(map(str, selected_units))}",
                          xaxis_title="Time (cycles)", yaxis_title="Sensor Reading",
                          legend=dict(x=0.01, y=0.99))
        st.plotly_chart(fig, use_container_width=True)

    # Recommended Intervention with Cost Savings
    if rul_df is not None:
        selected_rul = rul_df[rul_df["unit"].isin(selected_units)]
        min_rul = selected_rul["RUL"].min()
        if min_rul < CRITICAL_THRESHOLD:
            recommendation = "Immediate Maintenance"
            color = "red"
            estimated_cost = DOWNTIME_HOURS_FAILURE * COST_PER_HOUR
            proactive_cost = DOWNTIME_HOURS_PROACTIVE * COST_PER_HOUR + PROACTIVE_FIXED_COST
            savings = estimated_cost - proactive_cost
        elif min_rul < WARNING_THRESHOLD:
            recommendation = "Schedule Maintenance"
            color = "orange"
            estimated_cost = DOWNTIME_HOURS_FAILURE * COST_PER_HOUR
            proactive_cost = DOWNTIME_HOURS_PROACTIVE * COST_PER_HOUR + PROACTIVE_FIXED_COST
            savings = estimated_cost - proactive_cost
        else:
            recommendation = "No Action Needed"
            color = "green"
            savings = 0

        st.markdown(
            f"**Recommended Intervention for Selected Units:** "
            f"<span style='color:{color};font-weight:bold'>{recommendation}</span>", unsafe_allow_html=True
        )
        st.caption(f"Based on minimum predicted RUL ({min_rul:.1f} cycles). "
                   f"Estimated cost savings if proactive maintenance: £{savings:,.0f}.")

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

    st.subheader("Test-set Performance")
    rmse = (merged["error"] ** 2).mean() ** 0.5
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"{merged['error'].mean():.2f} cycles")
    m2.metric("RMSE", f"{rmse:.2f} cycles")
    m3.metric("Units evaluated", f"{len(merged)}")
    st.caption("Predicted RUL at each unit's last observed cycle vs the ground truth in RUL_FD001.")

    unit_list = sorted(merged["unit"].unique())

    selected_unit_comp = st.selectbox("Select a Unit for Comparison", unit_list, key="rul_comp_unit")
    row = merged[merged["unit"] == selected_unit_comp].iloc[0]

    col1, col2, col3 = st.columns(3)
    col1.metric("True RUL", f"{row['RUL_true']} cycles")
    col2.metric("Predicted RUL", f"{row['RUL_pred']:.2f} cycles")
    col3.metric("Absolute Error", f"{row['error']:.2f} cycles", 
                delta=f"{row['RUL_pred'] - row['RUL_true']:.2f}", delta_color="inverse")

    with st.expander("What does Absolute Error mean?"):
        st.markdown("""
        **Absolute Error** = | True RUL − Predicted RUL |  
        - Smaller error = closer to reality
        - Example: True RUL = 20, Predicted RUL = 15 → Absolute Error = 5 cycles
        """)

    if row['RUL_pred'] < high_risk_threshold:
        st.warning("Predicted RUL is very low – prioritize maintenance.")

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=["True RUL", "Predicted RUL"],
        y=[row["RUL_true"], row["RUL_pred"]],
        marker_color=["#2ca02c", "#1f77b4"],
        text=[f"{row['RUL_true']}", f"{row['RUL_pred']:.2f}"],
        textposition='auto'
    ))
    fig_bar.update_layout(title=f"RUL Comparison for Unit {selected_unit_comp}", yaxis_title="RUL (cycles)")
    st.plotly_chart(fig_bar, use_container_width=True)

def render_commercial_analysis_tab(rul_df):
    st.header("Commercial Analysis – Estimated Costs")

    if rul_df is None or rul_df.empty:
        st.warning("RUL prediction data required for commercial analysis.")
        return

    analysis_df = rul_df.copy()

    def calculate_cost(rul, risk):
        if risk == "High":
            return DOWNTIME_HOURS_FAILURE * COST_PER_HOUR
        elif risk == "Medium":
            return DOWNTIME_HOURS_PROACTIVE * COST_PER_HOUR + PROACTIVE_FIXED_COST
        else:
            return 0

    analysis_df["Estimated Cost (£)"] = analysis_df.apply(
        lambda row: calculate_cost(row["RUL"], row["risk_level"]), axis=1
    )
    
    def highlight_cost(val):
        if val > 200000:
            color = 'red'
        elif val > 50000:
            color = 'orange'
        else:
            color = 'green'
        return f'color: {color}; font-weight:bold'

    st.subheader("Unit-wise Estimated Costs")
    st.dataframe(
        analysis_df[["unit", "RUL", "risk_level", "Estimated Cost (£)"]]
        .sort_values("Estimated Cost (£)", ascending=False)
        .style.map(highlight_cost, subset=["Estimated Cost (£)"]),
        use_container_width=True
    )

    total_cost = analysis_df["Estimated Cost (£)"].sum()
    st.markdown(f"**Total Estimated Cost Across Units:** £{total_cost:,.0f}")
    st.caption("Costs are calculated based on predicted RUL and the assumptions for downtime and proactive maintenance.")

    st.subheader("Estimated Costs by Risk Level")
    risk_summary = analysis_df.groupby("risk_level")["Estimated Cost (£)"].sum().reindex(["High", "Medium", "Low"], fill_value=0)
    st.bar_chart(risk_summary)

    st.subheader("Top Cost Drivers")
    fig = px.bar(
        analysis_df.sort_values("Estimated Cost (£)", ascending=False),
        x="unit",
        y="Estimated Cost (£)",
        color="risk_level",
        color_discrete_map={"High": "red", "Medium": "orange", "Low": "green"},
        title="Unit-wise Estimated Cost Ranking",
        text="Estimated Cost (£)"
    )
    fig.update_layout(xaxis_title="Unit", yaxis_title="Estimated Cost (£)")
    st.plotly_chart(fig, use_container_width=True)

# -------------------- Main Application --------------------

def main():
    st.set_page_config(page_title="Engine Health Dashboard", layout="wide", initial_sidebar_state="expanded")

    FEATURES_PATH = "data/features/train_FD001_features.csv"
    PREDICTIONS_PATH = "data/processed/rul_predictions.csv"
    TRUE_RUL_PATH = "data/cleaned/test_FD001_labeled.csv"
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
    }

    st.title("Predictive Maintenance Dashboard")

    df = load_features(FEATURES_PATH)
    rul_df = load_rul_predictions(PREDICTIONS_PATH)
    true_rul_df = load_true_rul_labels(TRUE_RUL_PATH)

    if df is None:
        st.stop()

    df["max_cycle"] = df.groupby("unit")["time_in_cycles"].transform("max")
    df["failure_zone"] = df["time_in_cycles"] >= (df["max_cycle"] - FAILURE_THRESHOLD)

    if rul_df is not None:
        if "risk_level" not in rul_df.columns:
            rul_df["risk_level"] = rul_df["RUL"].apply(classify_risk, args=(HIGH_RISK_THRESHOLD, MEDIUM_RISK_THRESHOLD))
        # Predictions cover every test cycle; the dashboard reports each unit's current state
        rul_df = latest_per_unit(rul_df)

    tabs = st.tabs(["Summary", "Overview", "Unit Analysis", "RUL Predictions", "Commercial Analysis"])
    with tabs[0]:
        render_summary_tab(rul_df, RISK_COLOR_MAP, HIGH_RISK_THRESHOLD, MEDIUM_RISK_THRESHOLD)
    with tabs[1]:
        render_overview_tab(df, SENSOR_TOOLTIPS)
    with tabs[2]:
        render_unit_analysis_tab(df, FAILURE_THRESHOLD, rul_df)
    with tabs[3]:
        render_rul_predictions_tab(rul_df, true_rul_df, HIGH_RISK_THRESHOLD, ALERT_EMOJIS)
    with tabs[4]:
        render_commercial_analysis_tab(rul_df)

if __name__ == "__main__":
    main()
