import os
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

# Define paths to model and scaler
MODEL_PATH = r"C:\Users\aaaji\Downloads\WORKING DISSERTATION\tuned_random_forest_model.pkl"
SCALER_PATH = r"C:\Users\aaaji\Downloads\WORKING DISSERTATION\scaler.pkl"

# Verify model and scaler files
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    st.error("Model or scaler file not found. Ensure both are in the correct directory.")
    st.stop()

# Load model and scaler
rf_model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# App title
st.title("Random Forest Regressor with Time Series Analysis")

# Section 1: Random Forest Predictions
st.header("1. Random Forest Predictions")
st.write("Provide input data to predict Base Scores using the trained Random Forest model.")

# User input for prediction
st.sidebar.header("Input Parameters for Prediction")
v3_attackVector = st.sidebar.slider("Attack Vector (v3)", 0.0, 1.0, 0.5)
v3_attackComplexity = st.sidebar.slider("Attack Complexity (v3)", 0.0, 1.0, 0.5)
v3_privilegesRequired = st.sidebar.slider("Privileges Required (v3)", 0.0, 1.0, 0.5)
v3_userInteraction = st.sidebar.slider("User Interaction (v3)", 0.0, 1.0, 0.5)
v3_confidentialityImpact = st.sidebar.slider("Confidentiality Impact (v3)", 0.0, 1.0, 0.5)
v3_integrityImpact = st.sidebar.slider("Integrity Impact (v3)", 0.0, 1.0, 0.5)
v3_availabilityImpact = st.sidebar.slider("Availability Impact (v3)", 0.0, 1.0, 0.5)
ExploitabilityScore = st.sidebar.number_input("Exploitability Score", 0.0, 10.0, 5.0)
ImpactScore = st.sidebar.number_input("Impact Score", 0.0, 10.0, 5.0)

# Create input dataframe
input_data = pd.DataFrame({
    'v3_attackVector': [v3_attackVector],
    'v3_attackComplexity': [v3_attackComplexity],
    'v3_privilegesRequired': [v3_privilegesRequired],
    'v3_userInteraction': [v3_userInteraction],
    'v3_confidentialityImpact': [v3_confidentialityImpact],
    'v3_integrityImpact': [v3_integrityImpact],
    'v3_availabilityImpact': [v3_availabilityImpact],
    'ExploitabilityScore': [ExploitabilityScore],
    'ImpactScore': [ImpactScore]
})

# Preprocessing for prediction
try:
    X_scaled = scaler.transform(input_data)
    prediction = rf_model.predict(X_scaled)[0]
    st.success(f"Predicted Base Score: {prediction:.2f}")
except Exception as e:
    st.error(f"Error during prediction: {e}")

# Section 2: Time Series Analysis
st.header("2. Time Series Analysis")
uploaded_file = st.file_uploader("Upload CSV File for Time Series Analysis", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)

    st.write("Preview of the uploaded data:")
    st.dataframe(data.head())

    # Step 1: Generate Synthetic Time Column
    st.subheader("Step 1: Generate a Synthetic Time Column")
    start_date = st.date_input("Select Start Date", value=pd.to_datetime("2023-01-01"))
    frequency = st.selectbox("Select Frequency", options=["Daily", "Weekly", "Monthly"])

    # Generate time column
    if frequency == "Daily":
        data["Time"] = pd.date_range(start=start_date, periods=len(data), freq="D")
    elif frequency == "Weekly":
        data["Time"] = pd.date_range(start=start_date, periods=len(data), freq="W")
    elif frequency == "Monthly":
        data["Time"] = pd.date_range(start=start_date, periods=len(data), freq="M")

    st.write("Data with Synthesized Time Column:")
    st.dataframe(data.head())

    # Step 2: Time Series Visualization
    st.subheader("Step 2: Visualize Time Series Trends")
    target_column = st.selectbox("Select Target Column for Time Series Analysis", options=data.columns)

    if target_column:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(data["Time"], data[target_column], label=target_column)
        ax.set_title(f"Time Series Plot: {target_column}")
        ax.set_xlabel("Time")
        ax.set_ylabel(target_column)
        ax.legend()
        st.pyplot(fig)

    # Step 3: Decomposition
    st.subheader("Step 3: Decompose Time Series")
    if st.button("Perform Decomposition"):
        try:
            decomposition = seasonal_decompose(data[target_column], model="additive", period=12)
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 12))
            decomposition.observed.plot(ax=ax1, title="Observed")
            decomposition.trend.plot(ax=ax2, title="Trend")
            decomposition.seasonal.plot(ax=ax3, title="Seasonal")
            decomposition.resid.plot(ax=ax4, title="Residual")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error during decomposition: {e}")

    # Step 4: ARIMA Forecasting
    st.subheader("Step 4: Forecast Future Values")
    forecast_steps = st.number_input("Number of Steps to Forecast", min_value=1, max_value=50, value=10)

    if st.button("Perform Forecasting"):
        try:
            model = ARIMA(data[target_column], order=(1, 1, 1))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=forecast_steps)

            # Display Forecast
            st.write("Forecasted Values:")
            st.write(forecast)

            # Plot Forecast
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(data["Time"], data[target_column], label="Historical Data")
            future_dates = pd.date_range(start=data["Time"].iloc[-1], periods=forecast_steps + 1, freq="D")[1:]
            ax.plot(future_dates, forecast, label="Forecast", linestyle="--")
            ax.set_title("Forecast Plot")
            ax.legend()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error during forecasting: {e}")
