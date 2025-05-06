import pandas as pd
import pickle
import random
import joblib
from tabulate import tabulate

# Load model
model = joblib.load('gradient_boosting_model_v2.joblib')

# Load datasets
user_df = pd.read_csv('user.csv')
model_df = pd.read_csv('model.csv')

# Ask user for brand
brand_input = input("Enter car brand: ").strip().lower()

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
#print(f"\n💰 Predicted Price: {predicted_price} INR")

# Find a matching price in user_df (you can relax matching condition if needed)
matching_user_rows = user_df[user_df['price'].between(predicted_price - 10000, predicted_price + 10000)]

# Show matching results from user.csv
if matching_user_rows.empty:
    print("\n⚠️  No matching car found at the predicted price in user.csv.")
else:
    # Drop the unwanted columns before printing
    display_cols = matching_user_rows.drop(columns=['price', 'max_power_bhp', 'max_power_rp', 'model', 'brand'])
    
    # Get the first matching row
    first_match = display_cols.iloc[0]
    
    print("\n✅ Found matching car at the predicted price:\n")
    print(tabulate([first_match], headers='keys', tablefmt='fancy_grid', showindex=False))
