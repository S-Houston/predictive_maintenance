# Script to create a Streamlit dashboard for predictive maintenance
# This dashboard visualises sensor data, unit analysis, and remaining useful life (RUL)

# Import necessary librariesy
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load engineered features
@st.cache_data
def load_features():
    return pd.read_csv("data/features/train_FD001_features.csv")

@st.cache_data
def load_rul_predictions():
    try:
        return pd.read_csv("data/processed/rul_predictions.csv")
    except FileNotFoundError:
        return None

df = load_features()
rul_df = load_rul_predictions()

# Calculate failure thresholds
failure_threshold = 30
df["max_cycle"] = df.groupby("unit")["time_in_cycles"].transform("max")
df["failure_zone"] = df["time_in_cycles"] >= (df["max_cycle"] - failure_threshold)

# --- STREAMLIT UI START ---
st.set_page_config(page_title="Engine Health Dashboard", layout="wide")
st.title("🚀 Predictive Maintenance Dashboard")

# --- TAB SETUP ---
tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔍 Unit Analysis", "⏳ RUL Predictions"])

# --- TAB 1: Overview ---
with tab1:
    st.subheader("Remaining Useful Life Summary")
    if rul_df is not None:
        top_k = rul_df.sort_values("RUL").head(5)
        cols = st.columns(len(top_k))
        for i, row in top_k.iterrows():
            with cols[i]:
                st.metric(label=f"Unit {int(row.unit)}", value=f"{int(row.RUL)} cycles")
    else:
        st.warning("No RUL predictions available yet. Please upload to `data/processed/rul_predictions.csv`.")

    st.divider()
    st.subheader("Sensor Trends Across Units")
    selected_sensor = st.selectbox("Select a sensor", [col for col in df.columns if "sensor_" in col])
    fig = px.line(df, x="time_in_cycles", y=selected_sensor, color="unit", title=f"{selected_sensor} across all units")
    st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: Unit Analysis ---
with tab2:
    st.subheader("Individual Unit Analysis")
    unit_ids = df["unit"].unique()
    selected_unit = st.selectbox("Select a unit", sorted(unit_ids))
    sensor_choices = st.multiselect(
        "Select sensors to display",
        [col for col in df.columns if "sensor_" in col and "_rolling_mean" not in col],
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
            # Shaded failure region
            failure_start = df_unit["max_cycle"].max() - failure_threshold
            fig.add_vrect(x0=failure_start, x1=df_unit["max_cycle"].max(), fillcolor="red", opacity=0.2,
                          layer="below", line_width=0)
            fig.update_layout(title=f"{sensor} - Unit {selected_unit}",
                              xaxis_title="Time (cycles)", yaxis_title="Sensor Reading")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"{rolling_col} not found for {sensor}")

# --- TAB 3: RUL Predictions ---
with tab3:
    st.subheader("Remaining Useful Life (RUL) Predictions")
    if rul_df is not None:
        fig = px.bar(rul_df, x="unit", y="RUL", color="RUL", color_continuous_scale="reds",
                     title="Predicted Remaining Useful Life by Unit")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(rul_df.sort_values("RUL"), use_container_width=True)
    else:
        st.warning("No RUL predictions found.")
