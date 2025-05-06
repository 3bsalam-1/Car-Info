from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import pandas as pd
import joblib
import os 
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
# Add this before or right after initializing app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["*"] to allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],)


# Load the machine learning model and datasets once at startup
model = joblib.load(os.path.join(os.path.dirname(__file__), 'gradient_boosting_model_v2.joblib'))
user_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'user.csv'))
model_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'model.csv'))

# Define the input schema using Pydantic with validation
class BrandRequest(BaseModel):
    brand: str

    # Ensure that the brand is not empty or just whitespace
    @validator("brand")
    def validate_brand(cls, v):
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Brand name cannot be empty or just whitespace.")
        return cleaned

# Define the output schema for the response
class PredictionResponse(BaseModel):
    predicted_price: int          # Price predicted by the model
    match: dict | None            # Closest matching user listing, or None if nothing reasonable

@app.post("/predict", response_model=PredictionResponse)
def predict_price(req: BrandRequest):
    # Clean and normalize the input brand
    brand_input = req.brand.strip().lower()

    # Filter the model dataset for rows matching the brand
    filtered_model = model_df[model_df['brand'].str.lower() == brand_input]

    # Return a 404 if no data found for that brand
    if filtered_model.empty:
        raise HTTPException(status_code=404, detail=f"Brand '{req.brand}' not found in the dataset.")

    # Randomly select one entry from the matching brand rows
    sample_row = filtered_model.sample(n=1).iloc[0]

    # Define the features required by the ML model
    feature_cols = [
        'fuel_type', 'engine_displacement', 'no_cylinder', 'seating_capacity',
        'transmission_type', 'fuel_tank_capacity', 'body_type',
        'max_torque_nm', 'max_torque_rpm', 'max_power_bhp', 'max_power_rp'
    ]

    # Extract and reshape the input features for prediction
    input_features = sample_row[feature_cols].values.reshape(1, -1)

    # Get the predicted price from the model
    predicted_price = int(model.predict(input_features)[0])

    # Compute absolute price difference to find the closest user listing
    user_df["price_diff"] = (user_df["price"] - predicted_price).abs()

    # Optional: filter out listings too far from prediction (e.g., > 50,000 INR difference)
    tolerance = max(predicted_price * 0.5, 20000)  # Ensure a minimum threshold

    close_matches = user_df[user_df["price_diff"] <= tolerance]

    if close_matches.empty:
        match = None
    else:
        match_row = close_matches.sort_values("price_diff").iloc[0]
        match_row = match_row.drop(labels=['price', 'max_power_bhp', 'max_power_rp', 'model', 'brand', 'price_diff'])
        match = match_row.to_dict()


    # Return both the predicted price and the closest match (if any)
    return PredictionResponse(predicted_price=predicted_price, match=match)
