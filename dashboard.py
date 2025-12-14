import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Data Center Energy Optimization",
    page_icon="⚡",
    layout="wide"
)

# Load data and models
@st.cache_data
def load_data():
    return pd.read_csv('cleaned_data.csv')

@st.cache_resource
def load_model_and_scaler():
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

# Load everything
df = load_data()
model, scaler = load_model_and_scaler()

# Recommendation function
def recommend_actions(predicted_energy, server_load, cooling_efficiency):
    recommendations = []

    if predicted_energy > 450:
        recommendations.append("⚠️ HIGH - Reduce server load 15-20%")
    elif predicted_energy > 420:
        recommendations.append("⚡ ELEVATED - Monitor cooling")
    else:
        recommendations.append("✅ OPTIMAL - Maintain settings")

    if cooling_efficiency < 75:
        recommendations.append("🔧 Improve cooling efficiency")

    if server_load > 80:
        recommendations.append("📊 Reduce server load")

    return recommendations

def get_energy_color(energy):
    if energy > 450:
        return "red"
    elif energy > 420:
        return "orange"
    else:
        return "green"

# HEADER
st.title("⚡ Data Center Energy Optimization Dashboard")
st.markdown("### AI-Powered Real-Time Energy Management")
st.markdown("---")

# SECTION 1: VISUALIZATIONS
st.header("📊 Historical Data Analysis")

col1, col2 = st.columns(2)

with col1:
    # Energy Consumption Over Time
    fig_energy = px.line(
        df,
        x='Time',
        y='Energy_Consumption_kWh',
        title='Energy Consumption Over Time',
        labels={'Energy_Consumption_kWh': 'Energy (kWh)', 'Time': 'Time'}
    )
    fig_energy.update_traces(line_color='#1f77b4', line_width=2)
    st.plotly_chart(fig_energy, use_container_width=True)

    # Ambient Temperature Distribution
    fig_temp = px.histogram(
        df,
        x='Ambient_Temperature_C',
        nbins=30,
        title='Ambient Temperature Distribution',
        labels={'Ambient_Temperature_C': 'Temperature (°C)', 'count': 'Frequency'}
    )
    fig_temp.update_traces(marker_color='#ff7f0e')
    st.plotly_chart(fig_temp, use_container_width=True)

with col2:
    # Server Load Distribution
    fig_load = px.histogram(
        df,
        x='Server_Load_percent',
        nbins=30,
        title='Server Load Distribution',
        labels={'Server_Load_percent': 'Server Load (%)', 'count': 'Frequency'}
    )
    fig_load.update_traces(marker_color='#2ca02c')
    st.plotly_chart(fig_load, use_container_width=True)

    # Cooling Efficiency Distribution
    fig_cooling = px.histogram(
        df,
        x='Cooling_Efficiency_percent',
        nbins=30,
        title='Cooling Efficiency Distribution',
        labels={'Cooling_Efficiency_percent': 'Cooling Efficiency (%)', 'count': 'Frequency'}
    )
    fig_cooling.update_traces(marker_color='#d62728')
    st.plotly_chart(fig_cooling, use_container_width=True)

st.markdown("---")

# SECTION 2: REAL-TIME PREDICTION
st.header("🔮 Real-Time Energy Prediction")

col_slider1, col_slider2 = st.columns(2)

with col_slider1:
    server_load = st.slider(
        "Server Load (%)",
        min_value=50,
        max_value=100,
        value=70,
        step=1
    )

    ambient_temp = st.slider(
        "Ambient Temperature (°C)",
        min_value=18,
        max_value=30,
        value=24,
        step=1
    )

with col_slider2:
    cooling_efficiency = st.slider(
        "Cooling Efficiency (%)",
        min_value=70,
        max_value=95,
        value=85,
        step=1
    )

    hour_of_day = st.slider(
        "Hour of Day",
        min_value=0,
        max_value=23,
        value=12,
        step=1
    )

# Additional inputs for prediction
day_of_week = st.selectbox("Day of Week", [0, 1, 2, 3, 4, 5, 6], index=2,
                           format_func=lambda x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][x])
month = st.selectbox("Month", list(range(1, 13)), index=5,
                     format_func=lambda x: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][x-1])

# Prepare input for prediction
input_data = np.array([[
    server_load,
    ambient_temp,
    cooling_efficiency,
    hour_of_day,
    day_of_week,
    month
]])

# Scale and predict
input_scaled = scaler.transform(input_data)
predicted_energy = model.predict(input_scaled)[0]

# Get recommendations
recommendations = recommend_actions(predicted_energy, server_load, cooling_efficiency)
energy_color = get_energy_color(predicted_energy)

# Display prediction with color coding
st.markdown("### Prediction Results")

col_pred1, col_pred2 = st.columns([1, 2])

with col_pred1:
    st.metric(
        label="Predicted Energy Consumption",
        value=f"{predicted_energy:.2f} kWh"
    )

    if energy_color == "green":
        st.success("Energy level: OPTIMAL")
    elif energy_color == "orange":
        st.warning("Energy level: ELEVATED")
    else:
        st.error("Energy level: HIGH")

with col_pred2:
    st.markdown("#### Recommendations")
    for rec in recommendations:
        if "HIGH" in rec or "⚠️" in rec:
            st.error(rec)
        elif "ELEVATED" in rec or "⚡" in rec:
            st.warning(rec)
        elif "OPTIMAL" in rec or "✅" in rec:
            st.success(rec)
        else:
            st.info(rec)

st.markdown("---")

# SECTION 3: KEY METRICS
st.header("📈 Key Performance Metrics")

col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)

with col_metric1:
    avg_energy = df['Energy_Consumption_kWh'].mean()
    st.metric("Average Energy", f"{avg_energy:.2f} kWh")

with col_metric2:
    peak_energy = df['Energy_Consumption_kWh'].max()
    st.metric("Peak Energy", f"{peak_energy:.2f} kWh")

with col_metric3:
    min_energy = df['Energy_Consumption_kWh'].min()
    st.metric("Minimum Energy", f"{min_energy:.2f} kWh")

with col_metric4:
    avg_cooling = df['Cooling_Efficiency_percent'].mean()
    st.metric("Avg Cooling Efficiency", f"{avg_cooling:.2f}%")

st.markdown("---")

# SECTION 4: MODEL INFO
st.header("🤖 Model Information")

col_info1, col_info2 = st.columns(2)

with col_info1:
    st.markdown("**Model Type:** Random Forest Regressor")
    st.markdown("**Features Used:**")
    st.markdown("""
    - Server Load (%)
    - Ambient Temperature (°C)
    - Cooling Efficiency (%)
    - Hour of Day
    - Day of Week
    - Month
    """)

with col_info2:
    st.markdown("**Model Performance (Test Set):**")
    st.markdown("- **MAE:** 51.20 kWh")
    st.markdown("- **RMSE:** 61.00 kWh")
    st.markdown("- **Training Data:** 500 samples")
    st.markdown("- **Train/Test Split:** 80/20")

# Footer
st.markdown("---")
st.markdown("*Dashboard powered by Streamlit and Machine Learning*")
