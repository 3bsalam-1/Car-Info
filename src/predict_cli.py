"""
Car Price Prediction CLI Tool

A command-line interface for predicting car prices using a trained
Gradient Boosting model. Enter a car brand to get a predicted price
and matching vehicle from the user database.

Usage:
    python predict_cli.py
    
Then enter the car brand when prompted.
"""
import pandas as pd
import joblib
from pathlib import Path
from tabulate import tabulate

# Get the project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load model and datasets
MODEL_PATH = BASE_DIR / 'models' / 'gradient_boosting_model_v2.joblib'
USER_DATA_PATH = BASE_DIR / 'data' / 'user.csv'
MODEL_DATA_PATH = BASE_DIR / 'data' / 'model.csv'

print("Loading model and data...")
model = joblib.load(MODEL_PATH)
user_df = pd.read_csv(USER_DATA_PATH)
model_df = pd.read_csv(MODEL_DATA_PATH)

# Ask user for brand
brand_input = input("\n🚗 Enter car brand: ").strip().lower()

# Filter model data by brand (case-insensitive)
filtered_model = model_df[model_df['brand'].str.lower() == brand_input]

# Handle brand not found
if filtered_model.empty:
    print(f"❌ Brand '{brand_input}' not found in the dataset.")
    exit()

# Pick a random row from filtered model data
sample_row = filtered_model.sample(n=1).iloc[0]

# Define the feature columns used for prediction
feature_cols = [
    'fuel_type', 'engine_displacement', 'no_cylinder', 'seating_capacity',
    'transmission_type', 'fuel_tank_capacity', 'body_type',
    'max_torque_nm', 'max_torque_rpm', 'max_power_bhp', 'max_power_rp'
]

# Prepare the input for prediction
input_features = sample_row[feature_cols].values.reshape(1, -1)

# Predict the price
predicted_price = int(model.predict(input_features)[0])
print(f"\n💰 Predicted Price: ₹{predicted_price:,} INR")

# Find matching prices in user_df within a tolerance range
tolerance = 10000
matching_user_rows = user_df[
    user_df['price'].between(predicted_price - tolerance, predicted_price + tolerance)
]

# Show matching results from user.csv
if matching_user_rows.empty:
    print("\n⚠️  No matching car found at the predicted price in user database.")
    print(f"   (Searched within ±₹{tolerance:,} of predicted price)")
else:
    # Drop the unwanted columns before printing
    display_cols = matching_user_rows.drop(
        columns=['price', 'max_power_bhp', 'max_power_rp', 'model', 'brand']
    )

    # Get the first matching row
    first_match = display_cols.iloc[0]

    print("\n✅ Found matching car at the predicted price:\n")
    print(tabulate([first_match], headers='keys', tablefmt='fancy_grid', showindex=False))
