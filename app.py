
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Load and train model
@st.cache_resource
def load_model():
    df = pd.read_csv("weatherHistory.csv")

    # Drop columns that are not needed
    df = df.drop(columns=["Summary", "Precip Type", "Daily Summary", "Formatted Date"], errors='ignore')

    # Drop missing values
    df = df.dropna()

    # Make sure only numeric columns are used
    df = df.select_dtypes(include='number')

    # Train model
    X = df.drop(columns=["Temperature (C)"])
    y = df["Temperature (C)"]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model, X.columns

# Load model and features
model, features = load_model()

# Streamlit App
st.title("🌦️ Temperature Predictor App")
st.markdown("Enter weather values below and get the predicted temperature!")

# Collect input
input_data = []
for col in features:
    value = st.slider(f"{col}", min_value=0.0, max_value=1000.0, value=50.0)
    input_data.append(value)

input_df = np.array(input_data).reshape(1, -1)

# Predict
if st.button("Predict Temperature"):
    prediction = model.predict(input_df)
    st.success(f"🌡️ Predicted Temperature: {prediction[0]:.2f} °C")
