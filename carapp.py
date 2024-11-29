import pandas as pd
import streamlit as slt
import numpy as np
from joblib import load
import base64
import os

# Function to encode image as base64
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Image path
background_image_path = r"C:\Users\91934\Desktop\MOVIES\blackcar.jpg"

# Streamlit settings
slt.set_page_config(page_title="CarDekho Price Prediction", layout="wide")

# Apply the background
if os.path.exists(background_image_path):
    bin_str = get_base64_of_bin_file(background_image_path)
    css_code = f"""
    <style>
    .stApp {{
        background: url("data:image/jpeg;base64,{bin_str}");
        background-size: cover;
        background-attachment: fixed;
        background-repeat: no-repeat;
        background-position: center;
    }}
    .header {{
        text-align: center;
        font-size: 40px;
        color: #2c3e50;
        font-family: 'Trebuchet MS', sans-serif;
        margin-bottom: 30px;
        text-shadow: 3px 3px 8px rgba(0, 0, 0, 0.3); /* Stronger shadow */
        border: 3px solid #000000; /* Border around the header */
        padding: 20px 40px; /* Extra padding to create space */
        border-radius: 15px; /* Rounded corners */
        background: linear-gradient(90deg, #000000); /* Gradient background */
        color: white; /* White text to contrast with the gradient */
        animation: text-fade 2s ease-in-out infinite; /* Text animation */
    }}

    @keyframes text-fade {{
        0% {{ opacity: 0.5; }}
        50% {{ opacity: 1; }}
        100% {{ opacity: 0.5; }}
    }}

    .stButton>button {{
        background-color: #4CAF50;
        color: white;
        padding: 12px 30px;
        font-size: 18px;
        border-radius: 25px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        transition: background-color 0.3s ease, transform 0.2s ease;
    }}
    .stButton>button:hover {{
        background-color: #45a049;
        transform: scale(1.05);
    }}
    .stSelectbox, .stSlider, .stTextInput {{
        font-family: 'Roboto', sans-serif;
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 10px;
        padding: 10px;
        border: 1px solid #ccc;
        transition: all 0.3s ease;
    }}
    .stSelectbox:hover, .stSlider:hover, .stTextInput:hover {{
        border-color: #4CAF50;
    }}
    </style>
    """
    slt.markdown(css_code, unsafe_allow_html=True)
else:
    slt.error("Background image not found. Please check the path.")

# Header
slt.markdown('<div class="header">🚘 𝓒𝓪𝓻 𝓓𝓱𝓮𝓴𝓸 </div>', unsafe_allow_html=True)

# Load your dataset
df = pd.read_csv(r"C:\Users\91934\Desktop\car dheko\finaldf.csv")

# Create input sections using columns for side-by-side widgets
col1, col2 = slt.columns(2)

with col1:
    st1, st2 = slt.columns(2)
    Ft = st1.selectbox("Fuel Type", ['Petrol', 'Diesel', 'Lpg', 'Cng', 'Electric'])
    Bt = st2.selectbox("Body Type", ['Hatchback', 'SUV', 'Sedan', 'MUV', 'Coupe', 'Minivans', 'Convertibles', 'Hybrids', 'Wagon', 'Pickup Trucks'])

    st3, st4 = slt.columns(2)
    Tr = st3.selectbox("Transmission", ['Manual', 'Automatic'])
    Owner = st4.selectbox("Owner", [0, 1, 2, 3, 4, 5])

    Brand = slt.selectbox("Brand", options=df['Brand'].unique())

    filtered_models = df[(df['Brand'] == Brand) & (df['body type'] == Bt) & (df['Fuel type'] == Ft)]['model'].unique()
    Model = slt.selectbox("Model", options=filtered_models)

with col2:
    st5, st6 = slt.columns(2)
    Model_year = st5.selectbox("Model Year", options=sorted(df['modelYear'].unique()))
    IV = st6.selectbox("Insurance Validity", ['Third Party insurance', 'Comprehensive', 'Third Party', 'Zero Dep', '2', '1', 'Not Available'])

    st7, st8 = slt.columns(2)
    Km = st7.number_input("Kilometers Driven", min_value=100, max_value=100000, step=1000)  # Using number_input here
    ML = st8.text_input("Mileage (km/l)", "20")  # Changed to text_input, defaulting to '20'

    st9, st10 = slt.columns(2)
    seats = st9.slider("Seats", min_value=2, max_value=10, step=1)  # Slider for number of seats
    color = st10.selectbox("Color", df['Color'].unique())

    city = slt.selectbox("City", options=df['City'].unique())

# Prediction button
Submit = slt.button("Predict")

if Submit:
    # Load the model pipeline
    with open(r"C:\Users\91934\Desktop\car dheko\model_pipeline.joblib", "rb") as file:
        pipeline = load(file)

    # Input data
    new_df = pd.DataFrame({
        'Fuel type': Ft,
        'body type': Bt,
        'transmission': Tr,
        'ownerNo': Owner,
        'Brand': Brand,
        "model": Model,
        'modelYear': Model_year,
        'Insurance Validity': IV,
        'Kms Driven': Km,
        'Mileage': float(ML),  # Convert the input mileage to float
        'Seats': seats,
        'Color': color,
        'City': city
    }, index=[0])

    # Display the selected details
    slt.subheader("Selected Features:")
    slt.write(new_df)

    # Final model prediction
    prediction = pipeline.predict(new_df)

    # Format and display the prediction properly
    predicted_price = f"{prediction[0]:,.2f}"  # Ensures exactly 2 decimal places and comma separation
    # Highlight the prediction result with custom styling
    highlighted_prediction = f"""
        <div style="background-color: #4CAF50; color: white; padding: 15px; border-radius: 10px; font-size: 18px; font-weight: bold;">
            🚘 Predicted Price for {new_df['Brand'].iloc[0]} {new_df['model'].iloc[0]}: ₹{predicted_price} lakhs
        </div>
    """

    # Display the highlighted prediction result
    slt.markdown(highlighted_prediction, unsafe_allow_html=True)

