import streamlit as st
st.set_page_config(page_title="💻 Laptop Price Predictor", page_icon="💻", layout="wide")
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
st.markdown("""
<h1 style='text-align: center; color: #2E8B57;'>💻 Laptop Price Prediction</h1>
<p style='text-align: center; font-size: 18px;'>Select features below to get an instant price prediction! 🎯</p>
""", unsafe_allow_html=True)

# Step 1: Select Company
st.sidebar.title("💻 Laptop Price Analysis")
st.sidebar.markdown("Predict laptop prices based on real features. Select a company and customize specs!")
st.sidebar.info("Made with Streamlit | Powered by ML")

company = st.sidebar.selectbox("Company", sorted(features_per_company.keys()))
valids = features_per_company[company]

# Use columns for inputs
c1, c2, c3 = st.columns([1,1,1])
with c1:
    cpu_company = st.selectbox("🖥️ CPU Company", valids["CPU_company"])
    gpu_company = st.selectbox("🎮 GPU Company", valids["GPU_company"])
    ram = st.selectbox("🧠 RAM (GB)", valids["Ram"])
    primary_storage = st.selectbox("💾 Primary Storage (GB)", valids["PrimaryStorage"])
    secondary_storage = st.selectbox("🗄️ Secondary Storage (GB)", valids["SecondaryStorage"])
with c2:
    inches = st.selectbox("📏 Screen Size (inches)", sorted(df[df["Company"] == company]["Inches"].unique()))
    weight = st.selectbox("⚖️ Weight (kg)", sorted(df[df["Company"] == company]["Weight"].unique()))
    cpu_freq = st.selectbox("⚡ CPU Frequency (GHz)", sorted(df[df["Company"] == company]["CPU_freq"].unique()))
with c3:
    typename = st.selectbox("💼 TypeName", sorted(df[df["Company"] == company]["TypeName"].unique()))
    os = st.selectbox("🖥️ Operating System", sorted(df[df["Company"] == company]["OS"].unique()))
screen_res_col = None

# Detect screenw and screenh columns
screen_width_col = None
screen_height_col = None
for col in df.columns:
    if col.lower() == 'screenw':
        screen_width_col = col
    if col.lower() == 'screenh':
        screen_height_col = col

# Screen Width
if screen_width_col:
    screen_width = st.selectbox("Screen Width (cm)", sorted(df[df["Company"] == company][screen_width_col].unique()))
else:
    st.warning("Screen Width column not found in dataset.")

# Screen Height
if screen_height_col:
    screen_height = st.selectbox("Screen Height (cm)", sorted(df[df["Company"] == company][screen_height_col].unique()))
else:
    st.warning("Screen Height column not found in dataset.")

# Touchscreen
if "Touchscreen" in df.columns:
    touchscreen = st.selectbox("Touchscreen", sorted(df[df["Company"] == company]["Touchscreen"].unique()))
else:
    st.warning("Touchscreen column not found in dataset.")

# IPS Panel

# Detect IPSpanel column (case-insensitive)
ips_panel_col = None
for col in df.columns:
    if col.lower() == "ipspanel":
        ips_panel_col = col
        break
if ips_panel_col:
    ips_panel = st.selectbox("IPS Panel", sorted(df[df["Company"] == company][ips_panel_col].unique()))
else:
    st.warning("IPS Panel column not found in dataset.")

# Retina Display
if "RetinaDisplay" in df.columns:
    retina_display = st.selectbox("Retina Display", sorted(df[df["Company"] == company]["RetinaDisplay"].unique()))
else:
    st.warning("Retina Display column not found in dataset.")

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
    st.markdown(f"""
    <div style='background: linear-gradient(90deg, #232526 0%, #414345 100%); padding: 24px; border-radius: 14px; text-align: center;'>
        <h2 style='color:#fff; font-size:2.2em; font-weight:700; letter-spacing:1px;'>💰 Predicted Laptop Price:</h2>
        <span style='color:#00e676; font-size:2.5em; font-weight:900; background: #222; padding: 8px 24px; border-radius: 8px;'>€{price:.2f}</span>
    </div>
    """, unsafe_allow_html=True)

if "predicted_euro" in st.session_state:
    if st.button("🇮🇳 Convert to INR"):
        inr = st.session_state["predicted_euro"] * 90  # Example rate
        st.markdown(f"""
        <div style='background: linear-gradient(90deg, #232526 0%, #414345 100%); padding: 24px; border-radius: 14px; text-align: center;'>
            <h2 style='color:#fff; font-size:2.2em; font-weight:700; letter-spacing:1px;'>🇮🇳 Predicted Price in Indian Rupees:</h2>
            <span style='color:#00e676; font-size:2.5em; font-weight:900; background: #222; padding: 8px 24px; border-radius: 8px;'>₹{inr:,.2f}</span>
        </div>
        """, unsafe_allow_html=True)

st.info("Inputs are restricted per company — no unrealistic combos possible ✅")
