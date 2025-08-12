# Engine Health Dashboard
# This Streamlit app provides an interactive dashboard for predictive maintenance of engines.

# Import necessary libraries
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# --- Load Data Functions ---

@st.cache_data # Cache data loading to improve performance
def load_features():
    path = "data/features/train_FD001_features.csv"
    df = pd.read_csv(path)
    if "unit" not in df.columns or "time_in_cycles" not in df.columns:
        st.error(f"⚠️ Sensor data missing 'unit' or 'time_in_cycles' columns.\nColumns found: {df.columns.tolist()}")
        return None
    df["unit"] = pd.to_numeric(df["unit"], errors="coerce").astype("Int64")
    df["time_in_cycles"] = pd.to_numeric(df["time_in_cycles"], errors="coerce").astype("Int64")
    df.dropna(subset=["unit", "time_in_cycles"], inplace=True)
    return df

@st.cache_data # Cache RUL predictions loading
def load_rul_predictions():
    path = "data/processed/rul_predictions.csv"
    try:
        rul_df = pd.read_csv(path)
        if not {"unit", "RUL"}.issubset(rul_df.columns):
            st.warning("RUL predictions file missing required columns 'unit' and 'RUL'.")
            return None
        return rul_df
    except FileNotFoundError:
        return None

@st.cache_data # Cache true RUL labels loading
def load_true_rul_labels():
    # Check labelled RUL files only in cleaned folder based on your folder structure
    possible_paths = [
        "data/cleaned/train_FD001_labeled.csv",
        "data/cleaned/test_FD001_labeled.csv",
    ]
    for p in possible_paths:
        path = Path(p)
        if path.exists():
            df = pd.read_csv(path)
            if {"unit", "RUL"}.issubset(df.columns):
                return df[["unit", "RUL"]].drop_duplicates()
    return None

df = load_features()
rul_df = load_rul_predictions()
true_rul_df = load_true_rul_labels()

failure_threshold = 30  # cycles before max_cycle for failure zone

if df is not None:
    df["max_cycle"] = df.groupby("unit")["time_in_cycles"].transform("max")
    df["failure_zone"] = df["time_in_cycles"] >= (df["max_cycle"] - failure_threshold)

HIGH_RISK_THRESHOLD = 30
MEDIUM_RISK_THRESHOLD = 100

def classify_risk(rul):
    if rul < HIGH_RISK_THRESHOLD:
        return "High"
    elif rul >= HIGH_RISK_THRESHOLD and rul < MEDIUM_RISK_THRESHOLD:
        return "Medium"
    else:
        return "Low"

if rul_df is not None:
    rul_df["risk_level"] = rul_df["RUL"].apply(classify_risk)

SENSOR_TOOLTIPS = {
    "raw": "Raw sensor reading as collected from the engine.",
    "baseline": "Baseline value representing normal operating condition.",
    "degraded": "Deviation from baseline indicating sensor degradation.",
    "rolling_mean": "Rolling average smoothing short-term fluctuations.",
    "rolling_std": "Rolling standard deviation showing variability over time.",
    "slope": "Rate of change over a rolling window, indicating trends."
}

def get_sensor_type(col_name):
    for key in SENSOR_TOOLTIPS.keys():
        if col_name.endswith(key):
            return key
    return "raw"

# --- Streamlit UI ---

st.set_page_config(page_title="Engine Health Dashboard", layout="wide")
st.title("🚀 Predictive Maintenance Dashboard")

tabs = st.tabs(["📝 Summary", "📊 Overview", "🔍 Unit Analysis", "⏳ RUL Predictions"])

# --- Summary Tab ---
with tabs[0]:
    st.header("📝 Summary Overview")

    if rul_df is not None and len(rul_df) > 0:
        top5_high_risk = rul_df.sort_values("RUL").head(5)
        st.subheader("Top 5 Units Approaching Failure (Lowest RUL)")
        cols = st.columns(len(top5_high_risk))
        risk_color_map = {"High": "red", "Medium": "orange", "Low": "green"}
        for i, row in enumerate(top5_high_risk.itertuples()):
            with cols[i]:
                risk_color = risk_color_map.get(row.risk_level, "black")
                st.markdown(f"**Unit {int(row.unit)}**")
                st.markdown(f"<span style='color:{risk_color};font-weight:bold;'>Risk: {row.risk_level}</span>", unsafe_allow_html=True)
                st.metric(label="Remaining Useful Life", value=f"{row.RUL:.1f} cycles")

        risk_counts = rul_df["risk_level"].value_counts().reindex(["High","Medium","Low"], fill_value=0)
        st.subheader("Unit Risk Distribution")
        fig_pie = px.pie(
            names=risk_counts.index,
            values=risk_counts.values,
            color=risk_counts.index,
            color_discrete_map={"High":"red", "Medium":"orange", "Low":"green"},
            title="Units by Risk Level"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown(
            """
            **Cycle Explanation:**  
            - A **cycle** is one complete engine operation cycle.  
            - **Risk thresholds:**  
              - High Risk: RUL < 30 cycles  
              - Medium Risk: RUL ≥ 30 and < 100 cycles  
              - Low Risk: RUL ≥ 100 cycles  
            - Units flagged as **High Risk** should be prioritised for inspection or maintenance.
            """
        )
    else:
        st.warning("No RUL prediction data available to summarize.")

# --- Overview Tab ---
with tabs[1]:
    st.header("📊 Sensor Trends Across Units")
    if df is not None:
        sensor_cols = [col for col in df.columns if col.startswith("sensor_")]
        sensor_display_names = []
        for col in sensor_cols:
            stype = get_sensor_type(col)
            sensor_display_names.append(f"{col} ({stype})")

        display_to_col = dict(zip(sensor_display_names, sensor_cols))
        selected_display = st.selectbox("Select a sensor", sensor_display_names)

        selected_sensor = display_to_col[selected_display]

        fig = px.line(
            df,
            x="time_in_cycles",
            y=selected_sensor,
            color="unit",
            title=f"{selected_sensor} across all units"
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("ℹ️ Sensor Explanation Glossary"):
            for stype, desc in SENSOR_TOOLTIPS.items():
                st.markdown(f"**{stype.capitalize()}**: {desc}")

    else:
        st.warning("Sensor data not loaded.")

# --- Unit Analysis Tab ---
with tabs[2]:
    st.header("🔍 Individual Unit Analysis")
    if df is not None:
        unit_ids = df["unit"].unique()
        selected_unit = st.selectbox("Select a unit", sorted(unit_ids))

        st.markdown(
            """
            **Note:** The red shaded area marks the last 30 cycles before the unit's failure (failure zone). Sensor readings here typically degrade rapidly.
            """
        )

        sensor_choices = st.multiselect(
            "Select sensors to display",
            [col for col in df.columns if col.startswith("sensor_") and not any(suffix in col for suffix in ["_rolling_mean", "_rolling_std", "_slope"])],
            default=["sensor_2", "sensor_3"]
        )

        df_unit = df[df["unit"] == selected_unit]
        for sensor in sensor_choices:
            rolling_col = f"{sensor}_rolling_mean"
            if rolling_col in df_unit.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_unit["time_in_cycles"], y=df_unit[sensor],
                                         mode="lines", name=f"{sensor} (raw)"))
                fig.add_trace(go.Scatter(x=df_unit["time_in_cycles"], y=df_unit[rolling_col],
                                         mode="lines", name=f"{sensor} (rolling avg)"))
                failure_start = df_unit["max_cycle"].max() - failure_threshold
                fig.add_vrect(x0=failure_start, x1=df_unit["max_cycle"].max(), fillcolor="red", opacity=0.2,
                              layer="below", line_width=0)
                fig.update_layout(title=f"{sensor} - Unit {selected_unit}",
                                  xaxis_title="Time (cycles)", yaxis_title="Sensor Reading")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"{rolling_col} not found for {sensor}")
    else:
        st.warning("Sensor data not loaded.")

# --- RUL Predictions Tab ---
with tabs[3]:
    st.header("⏳ Remaining Useful Life (RUL) Predictions")

    if rul_df is not None:
        st.subheader("Predicted RUL and Risk Levels by Unit")

        alert_emojis = {"High": "🔴 High", "Medium": "🟠 Medium", "Low": "🟢 Low"}
        rul_display_df = rul_df.copy()
        rul_display_df["Alert"] = rul_display_df["risk_level"].map(alert_emojis)
        display_cols = ["unit", "RUL", "Alert"]
        st.dataframe(
            rul_display_df[display_cols].sort_values("RUL").reset_index(drop=True),
            use_container_width=True
        )

        if true_rul_df is not None:
            st.subheader("📊 Per Unit Comparison")

            # Merge predictions and true RULs
            merged = pd.merge(
                rul_df.rename(columns={"RUL": "RUL_pred"}),
                true_rul_df.rename(columns={"RUL": "RUL_true"}),
                on="unit"
            )
            merged["error"] = (merged["RUL_true"] - merged["RUL_pred"]).abs()

            # Defensive: check merged is not empty
            if merged.empty:
                st.warning("No overlapping units found between predictions and true RUL data.")
            else:
                # Units list for dropdown
                unit_list = sorted(merged["unit"].unique())

                # Defensive: If no units, warn and skip UI
                if len(unit_list) == 0:
                    st.warning("No units available for per-unit comparison.")
                else:
                    with st.expander("ℹ️ How to interpret the chart and metrics"):
                        st.markdown("""
                        Each unit represents a single engine at the start of its test. You’ll see:

                        - **🔧 True RUL**: Actual cycles before failure.
                        - **📡 Predicted RUL**: Model’s estimate at prediction time.
                        - **⚠️ Absolute Error**: The difference between prediction and reality.

                        ### Example:
                        - If **True RUL** = 268 and **Predicted RUL** = 193.5 →  
                          The model flagged the unit **~75 cycles early**.

                        🔄 Underestimate = early (safe, costly)  
                        ⏳ Overestimate = risk of unplanned failure

                        Use this view to assess model reliability per unit.
                        """)

                    # Dropdown for unit selection
                    selected_unit = st.selectbox("Select a Unit", unit_list)

                    # Defensive: check if selected_unit is in merged data
                    filtered_rows = merged[merged["unit"] == selected_unit]
                    if filtered_rows.empty:
                        st.error(f"No data found for unit {selected_unit}. Please select another unit.")
                    else:
                        row = filtered_rows.iloc[0]

                        col1, col2, col3 = st.columns(3)
                        col1.metric("🔧 True RUL", f"{row['RUL_true']} cycles")
                        col2.metric("📡 Predicted RUL", f"{row['RUL_pred']:.2f} cycles")
                        col3.metric("⚠️ Absolute Error", f"{row['error']:.2f} cycles")

                        if row['RUL_pred'] < HIGH_RISK_THRESHOLD:
                            st.warning("⚠️ Predicted RUL is very low. This unit may be approaching failure.")

                        # Bar chart
                        fig_bar = go.Figure()
                        fig_bar.add_trace(go.Bar(
                            x=["True RUL", "Predicted RUL"],
                            y=[row["RUL_true"], row["RUL_pred"]],
                            marker_color=["green", "blue"]
                        ))
                        fig_bar.update_layout(
                            title=f"RUL Comparison for Unit {selected_unit}",
                            yaxis_title="RUL (cycles)"
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)

                        st.info("This view lets engineers compare model predictions against actual outcomes on a per-unit basis.")
    else:
        st.warning("No RUL predictions found.")
