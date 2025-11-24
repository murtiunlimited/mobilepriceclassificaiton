import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.title("📱 Mobile Price Range Predictor v2")

# -----------------------------
# Load Model + Feature Order
# -----------------------------
results_dir = Path("results")
model_path = results_dir / "model.joblib"
features_path = results_dir / "feature.csv"

try:
    model = joblib.load(model_path)
    feature_order = pd.read_csv(features_path, header=None)[0].tolist()
except:
    st.error("❌ Model files not found. Run app.py to train and save the model.")
    st.stop()

st.success("Model Loaded Successfully!")

# -----------------------------
# Create Input Form
# -----------------------------
st.header("Enter Mobile Specifications")

inputs = {}

for feature in feature_order:
    inputs[feature] = st.number_input(
        label=f"{feature}",
        min_value=0.0,
        step=1.0,
        format="%.2f"
    )

# -----------------------------
# Predict
# -----------------------------
if st.button("Predict Price Range"):
    values = [inputs[f] for f in feature_order]
    prediction = model.predict([values])[0]

    st.subheader("📊 Prediction Result")
    st.info(f"**Predicted Price Range: {prediction}**")
