
import streamlit as st
import pandas as pd
import joblib

from pathlib import Path

# --------------------------------
# 1. Load trained model
# --------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "models" / "random_forest_model.pkl"

model = joblib.load(MODEL_PATH)

# --------------------------------
# 2. Page configuration
# --------------------------------

st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Used Car Price Predictor")

st.write(
    "Enter your car details to estimate its selling price."
)

st.divider()

# --------------------------------
# 3. Car name and brand
# --------------------------------

st.subheader("Car Information")

car_name = st.text_input(
    "Car Brand & Model",
    placeholder="e.g. Maruti Swift Dzire VDI"
)

# Extract brand from car name
brand = car_name.strip().split()[0] if car_name.strip() else ""

# --------------------------------
# 4. Other car features
# --------------------------------

st.subheader("Car Details")

col1, col2 = st.columns(2)

with col1:

    year = st.number_input(
        "Manufacturing Year",
        min_value=1983,
        max_value=2026,
        value=2018,
        step=1
    )

    km_driven = st.number_input(
        "Kilometres Driven",
        min_value=0,
        value=50000,
        step=1000
    )

    fuel = st.selectbox(
        "Fuel Type",
        ["Diesel", "Petrol", "CNG", "LPG"]
    )

    seller_type = st.selectbox(
        "Seller Type",
        ["Individual", "Dealer", "Trustmark Dealer"]
    )

    transmission = st.selectbox(
        "Transmission",
        ["Manual", "Automatic"]
    )

with col2:

    owner = st.selectbox(
        "Owner",
        [
            "First Owner",
            "Second Owner",
            "Third Owner",
            "Fourth & Above Owner",
            "Test Drive Car"
        ]
    )

    mileage = st.number_input(
        "Mileage (kmpl)",
        min_value=0.0,
        value=20.0,
        step=0.1
    )

    engine = st.number_input(
        "Engine (CC)",
        min_value=0,
        value=1200,
        step=100
    )

    max_power = st.number_input(
        "Max Power (bhp)",
        min_value=0.0,
        value=80.0,
        step=1.0
    )

    seats = st.selectbox(
        "Number of Seats",
        [2, 4, 5, 6, 7, 8, 9, 10, 14],
        index=2
    )

# --------------------------------
# 5. Prediction
# --------------------------------

st.divider()

if st.button("Predict Selling Price", type="primary"):

    if not car_name.strip():

        st.warning("Please enter the car brand and model.")

    else:

        # Create input dataframe
        input_data = pd.DataFrame([{
            "name": car_name.strip(),
            "brand": brand,
            "year": year,
            "km_driven": km_driven,
            "fuel": fuel,
            "seller_type": seller_type,
            "transmission": transmission,
            "owner": owner,
            "mileage": mileage,
            "engine": engine,
            "max_power": max_power,
            "seats": seats
        }])

        # Generate prediction
        try:
            prediction = model.predict(input_data)[0]

            st.success(
                f"Estimated Selling Price: ₹{prediction:,.0f}"
            )

            st.caption(
                "This is an estimated market price, not a guaranteed selling price."
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")
