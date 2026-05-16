"""
Streamlit web app for the Laptop Price Prediction project.
Loads the trained Random Forest model and lets users predict laptop prices
from hardware specifications.
"""

import sys
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Add src/ to path so we can import preprocessing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from preprocessing import prepare_single_prediction

# ---------- Page Config ----------
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide"
)

# ---------- Load model and metadata ----------
@st.cache_resource
def load_model():
    model = joblib.load('models/random_forest_model.pkl')
    feature_columns = joblib.load('models/feature_columns.pkl')
    return model, feature_columns

@st.cache_data
def load_data():
    """Load the original dataset to populate dropdown options."""
    return pd.read_csv('data/laptop_prices.csv')

model, feature_columns = load_model()
df = load_data()

# ---------- Header ----------
st.title("💻 Laptop Price Predictor")
st.markdown("Predict the price of a laptop based on its specifications using a Random Forest model.")
st.markdown("---")

# ---------- Sidebar: About ----------
with st.sidebar:
    st.header("About")
    st.write("""
    This app predicts laptop prices using a Random Forest model trained on 
    1,275 laptops with 23 features.
    
    **Model Performance:**
    - R² Score: 0.85
    - Mean Absolute Error: €184
    """)
    st.markdown("---")
    st.write("**Built by Vishwa A**")
    st.write("[GitHub](https://github.com/itzmevishwa)")

# ---------- Input Form ----------
st.subheader("🔧 Configure Your Laptop")

col1, col2, col3 = st.columns(3)

with col1:
    company = st.selectbox("Brand", sorted(df['Company'].unique()))
    type_name = st.selectbox("Laptop Type", sorted(df['TypeName'].unique()))
    os_choice = st.selectbox("Operating System", sorted(df['OS'].unique()))
    ram = st.selectbox("RAM (GB)", sorted(df['Ram'].unique()))
    weight = st.number_input("Weight (kg)", min_value=0.5, max_value=5.0, value=2.0, step=0.1)

with col2:
    inches = st.number_input("Screen Size (inches)", min_value=10.0, max_value=20.0, value=15.6, step=0.1)
    screen = st.selectbox("Screen Type", sorted(df['Screen'].unique()))
    screen_w = st.number_input("Screen Width (px)", min_value=800, max_value=4000, value=1920, step=1)
    screen_h = st.number_input("Screen Height (px)", min_value=600, max_value=3000, value=1080, step=1)
    touchscreen = st.selectbox("Touchscreen", ["No", "Yes"])

with col3:
    ips = st.selectbox("IPS Panel", ["No", "Yes"])
    retina = st.selectbox("Retina Display", ["No", "Yes"])
    cpu_company = st.selectbox("CPU Brand", sorted(df['CPU_company'].unique()))
    cpu_freq = st.number_input("CPU Frequency (GHz)", min_value=0.5, max_value=4.0, value=2.5, step=0.1)
    gpu_company = st.selectbox("GPU Brand", sorted(df['GPU_company'].unique()))

# Storage section
st.markdown("### 💾 Storage")
col4, col5 = st.columns(2)

with col4:
    primary_storage = st.number_input("Primary Storage (GB)", min_value=8, max_value=2048, value=256, step=1)
    primary_storage_type = st.selectbox("Primary Storage Type", sorted(df['PrimaryStorageType'].unique()))

with col5:
    secondary_storage = st.number_input("Secondary Storage (GB)", min_value=0, max_value=2048, value=0, step=1)
    secondary_storage_type = st.selectbox("Secondary Storage Type", sorted(df['SecondaryStorageType'].unique()))

st.markdown("---")

# ---------- Predict Button ----------
if st.button("🚀 Predict Price", type="primary", use_container_width=True):
    # Build the input dictionary (must match the original dataset columns)
    user_input = {
        'Company': company,
        'Product': 'Unknown',          # not used (dropped in preprocessing)
        'TypeName': type_name,
        'Inches': inches,
        'Ram': ram,
        'OS': os_choice,
        'Weight': weight,
        'Screen': screen,
        'ScreenW': screen_w,
        'ScreenH': screen_h,
        'Touchscreen': touchscreen,
        'IPSpanel': ips,
        'RetinaDisplay': retina,
        'CPU_company': cpu_company,
        'CPU_freq': cpu_freq,
        'CPU_model': 'Unknown',        # not used (dropped)
        'PrimaryStorage': primary_storage,
        'SecondaryStorage': secondary_storage,
        'PrimaryStorageType': primary_storage_type,
        'SecondaryStorageType': secondary_storage_type,
        'GPU_company': gpu_company,
        'GPU_model': 'Unknown',        # not used (dropped)
    }

    # Preprocess and predict
    X_input = prepare_single_prediction(user_input, feature_columns)
    price_eur = model.predict(X_input)[0]
    price_inr = price_eur * 90  # rough conversion

    # Display result
    st.success("✅ Prediction Complete!")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Predicted Price (EUR)", f"€{price_eur:,.2f}")
    with col_b:
        st.metric("Predicted Price (INR)", f"₹{price_inr:,.0f}")

    st.info("💡 Note: This is an estimate based on 2017–2018 laptop data. Actual market prices may differ.")