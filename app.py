import streamlit as st
import pandas as pd
import joblib
import json

# --- Set background image ---
def add_bg_image():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://d3rvezpmgp265q.cloudfront.net/lexusone/lexieen/hybrid-wol-953x348_tcm-3107-692187.jpg");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_image()

# --- Styled Title ---
st.markdown(
    """
    <h1 style='text-align: center; color: white; padding: 20px; background-color: rgba(0, 0, 0, 0.6); border-radius: 10px;'>
        Car Price Prediction
    </h1>
    """,
    unsafe_allow_html=True
)

# Load model and feature columns
model = joblib.load('car_price_model.pkl')
with open('feature_columns.json', 'r') as f:
    feature_columns = json.load(f)

# Load car names from CSV
car_data = pd.read_csv('car_data.csv')
car_names = sorted(car_data['CarName'].unique())

# Dropdown to select car name
car_name = st.selectbox("Select Car Name", options=["Select a car"] + list(car_names))

# Numeric inputs
wheelbase = st.number_input("Wheelbase", min_value=50.0, max_value=150.0, value=88.0)
carlength = st.number_input("Car Length", min_value=120.0, max_value=250.0, value=180.0)
carwidth = st.number_input("Car Width", min_value=50.0, max_value=100.0, value=65.0)
carheight = st.number_input("Car Height", min_value=40.0, max_value=80.0, value=55.0)
curbweight = st.number_input("Curb Weight", min_value=1000, max_value=5000, value=2500)
enginesize = st.number_input("Engine Size", min_value=50, max_value=500, value=150)
horsepower = st.number_input("Horsepower", min_value=50, max_value=500, value=120)
peakrpm = st.number_input("Peak RPM", min_value=3000, max_value=7000, value=5000)
citympg = st.number_input("City MPG", min_value=5, max_value=60, value=25)
highwaympg = st.number_input("Highway MPG", min_value=10, max_value=80, value=30)

# Categorical inputs
fueltype = st.selectbox("Fuel Type", options=['gas', 'diesel'])
aspiration = st.selectbox("Aspiration", options=['std', 'turbo'])
doornumber = st.selectbox("Door Number", options=['two', 'four'])
carbody = st.selectbox("Car Body", options=['sedan', 'hatchback', 'wagon', 'hardtop', 'convertible'])
drivewheel = st.selectbox("Drive Wheel", options=['fwd', 'rwd', '4wd'])
enginelocation = st.selectbox("Engine Location", options=['front', 'rear'])
enginetype = st.selectbox("Engine Type", options=['ohc', 'ohcf', 'ohcv', 'dohc', 'l', 'rotor'])
cylindernumber = st.selectbox("Cylinder Number", options=['two', 'three', 'four', 'five', 'six', 'eight', 'twelve'])
fuelsystem = st.selectbox("Fuel System", options=['mpfi', '2bbl', 'idi', '1bbl', 'spdi', '4bbl', 'spfi'])

if st.button("Predict Price"):
    if car_name == "Select a car":
        st.warning("Please select a car name before predicting.")
    else:
        # Build input dictionary for numeric features
        input_dict = {
            'wheelbase': wheelbase,
            'carlength': carlength,
            'carwidth': carwidth,
            'carheight': carheight,
            'curbweight': curbweight,
            'enginesize': enginesize,
            'horsepower': horsepower,
            'peakrpm': peakrpm,
            'citympg': citympg,
            'highwaympg': highwaympg
        }

        # Initialize all dummy columns with 0 if not numeric
        for col in feature_columns:
            if col not in input_dict:
                input_dict[col] = 0

        # Set 1 for selected categorical values
        input_dict[f'fueltype_{fueltype}'] = 1
        input_dict[f'aspiration_{aspiration}'] = 1
        input_dict[f'doornumber_{doornumber}'] = 1
        input_dict[f'carbody_{carbody}'] = 1
        input_dict[f'drivewheel_{drivewheel}'] = 1
        input_dict[f'enginelocation_{enginelocation}'] = 1
        input_dict[f'enginetype_{enginetype}'] = 1
        input_dict[f'cylindernumber_{cylindernumber}'] = 1
        input_dict[f'fuelsystem_{fuelsystem}'] = 1

        # Create DataFrame for prediction
        input_df = pd.DataFrame([input_dict], columns=feature_columns)

        # Predict price in USD
        prediction = model.predict(input_df)[0]

        # Convert USD to INR
        conversion_rate = 82
        price_inr = prediction * conversion_rate

        # Display car name and predicted price in INR
        st.write(f"**Car Name:** {car_name.title()}")
        st.success(f"Predicted Car Price: ₹{price_inr:,.2f} (INR)")
