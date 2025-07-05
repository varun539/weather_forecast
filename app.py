
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Load trained model
@st.cache_resource
def load_model():
    df = pd.read_csv("weather_clean.csv")
    df = df.drop(columns=["Summary", "Precip Type", "Daily Summary"])
    X = df.drop(columns=["Temperature (C)"])
    y = df["Temperature (C)"]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, X.columns

model, feature_names = load_model()

# App UI
st.title("🌦️ Temperature Predictor")
st.markdown("Enter weather conditions to predict temperature (°C)")

user_input = []
for feature in feature_names:
    val = st.slider(f"{feature}", min_value=float(0), max_value=float(1000), value=float(50))
    user_input.append(val)

input_array = np.array(user_input).reshape(1, -1)

if st.button("Predict Temperature"):
    prediction = model.predict(input_array)
    st.success(f"🌡️ Predicted Temperature: {prediction[0]:.2f} °C")
