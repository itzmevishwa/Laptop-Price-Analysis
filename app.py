import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load("best_model.pkl")

# Load dataset
df = pd.read_csv("laptop_prices.csv")

# Build dictionary of valid options per company
features_per_company = {}
for company in df["Company"].unique():
    subset = df[df["Company"] == company]
    features_per_company[company] = {
        "CPU_company": subset["CPU_company"].unique().tolist(),
        "GPU_company": subset["GPU_company"].unique().tolist(),
        "Ram": sorted(subset["Ram"].unique().tolist()),
        "PrimaryStorage": sorted(subset["PrimaryStorage"].unique().tolist()),
        "SecondaryStorage": sorted(subset["SecondaryStorage"].unique().tolist()),
        "Inches": (subset["Inches"].min(), subset["Inches"].max()),
        "Weight": (subset["Weight"].min(), subset["Weight"].max()),
        "CPU_freq": (subset["CPU_freq"].min(), subset["CPU_freq"].max())
    }

# ==============================
# Streamlit App
# ==============================
st.title("💻 Laptop Price Prediction")
st.write("Select laptop features to predict price (restricted to real dataset values).")

# Step 1: Select Company
company = st.selectbox("Company", sorted(features_per_company.keys()))
valids = features_per_company[company]

# Step 2: Dynamically update options based on company
cpu_company = st.selectbox("CPU Company", valids["CPU_company"])
gpu_company = st.selectbox("GPU Company", valids["GPU_company"])
ram = st.selectbox("RAM (GB)", valids["Ram"])
primary_storage = st.selectbox("Primary Storage (GB)", valids["PrimaryStorage"])
secondary_storage = st.selectbox("Secondary Storage (GB)", valids["SecondaryStorage"])
inches = st.selectbox("Screen Size (inches)", sorted(df[df["Company"] == company]["Inches"].unique()))
weight = st.selectbox("Weight (kg)", sorted(df[df["Company"] == company]["Weight"].unique()))
cpu_freq = st.selectbox("CPU Frequency (GHz)", sorted(df[df["Company"] == company]["CPU_freq"].unique()))

# ==============================
# Prediction
# ==============================
if st.button("Predict Price"):
    # Load encoders (must match training)
    company_encoder = joblib.load("company_encoder.pkl")
    cpu_encoder = joblib.load("cpu_company_encoder.pkl")
    gpu_encoder = joblib.load("gpu_company_encoder.pkl")

    # Encode categorical features
    company_encoded = company_encoder.transform([company])[0]
    cpu_company_encoded = cpu_encoder.transform([cpu_company])[0]
    gpu_company_encoded = gpu_encoder.transform([gpu_company])[0]

    features = np.array([
        company_encoded, cpu_company_encoded, gpu_company_encoded, ram,
        primary_storage, secondary_storage, inches, weight, cpu_freq
    ]).reshape(1, -1)

    price = model.predict(features)[0]
    st.session_state["predicted_euro"] = price
    st.success(f"💰 Predicted Laptop Price: €{price:.2f}")

if "predicted_euro" in st.session_state:
    if st.button("Convert to INR"):
        inr = st.session_state["predicted_euro"] * 90  # Example rate
        st.info(f"Predicted Price in Indian Rupees: ₹{inr:,.2f}")

st.info("Inputs are restricted per company — no unrealistic combos possible ✅")
