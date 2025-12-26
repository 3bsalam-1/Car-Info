"""
Car Price Prediction API

A FastAPI-based REST API that predicts car prices using a Gradient Boosting
machine learning model. The API accepts a car brand and returns a predicted
price along with the closest matching user listing.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import pandas as pd
import joblib
import os
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI application
app = FastAPI(
    title="Car Price Prediction API",
    description="Predict car prices based on brand using machine learning",
    version="1.0.0"
)

# Configure CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (configure appropriately for production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load the machine learning model and datasets once at startup
MODEL_PATH = BASE_DIR / 'models' / 'gradient_boosting_model_v2.joblib'
USER_DATA_PATH = BASE_DIR / 'data' / 'user.csv'
MODEL_DATA_PATH = BASE_DIR / 'data' / 'model.csv'

model = joblib.load(MODEL_PATH)
user_df = pd.read_csv(USER_DATA_PATH)
model_df = pd.read_csv(MODEL_DATA_PATH)


# Define the input schema using Pydantic with validation
class BrandRequest(BaseModel):
    """Request model for car brand input"""
    brand: str

    @validator("brand")
    def validate_brand(cls, v):
        """Ensure that the brand is not empty or just whitespace"""
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Brand name cannot be empty or just whitespace.")
        return cleaned


# Define the output schema for the response
class PredictionResponse(BaseModel):
    """Response model containing prediction and matching listing"""
    predicted_price: int          # Price predicted by the model
    match: dict | None            # Closest matching user listing, or None if nothing reasonable


@app.get("/")
def read_root():
    """Root endpoint with API information"""
    return {
        "message": "Car Price Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Predict car price by brand",
            "/docs": "API documentation (Swagger UI)",
            "/redoc": "API documentation (ReDoc)"
        }
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_price(req: BrandRequest):
    """
    Predict car price based on brand name
    
    Args:
        req: BrandRequest object containing the car brand
        
    Returns:
        PredictionResponse containing predicted price and closest match
        
    Raises:
        HTTPException: If brand is not found in the dataset
    """
    # Clean and normalize the input brand
    brand_input = req.brand.strip().lower()

    # Filter the model dataset for rows matching the brand
    filtered_model = model_df[model_df['brand'].str.lower() == brand_input]

    # Return a 404 if no data found for that brand
    if filtered_model.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Brand '{req.brand}' not found in the dataset."
        )

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

    # Filter out listings too far from prediction (using dynamic tolerance)
    tolerance = max(predicted_price * 0.5, 20000)  # Ensure a minimum threshold
    close_matches = user_df[user_df["price_diff"] <= tolerance]

    if close_matches.empty:
        match = None
    else:
        match_row = close_matches.sort_values("price_diff").iloc[0]
        # Remove unnecessary columns from the match
        match_row = match_row.drop(labels=[
            'price', 'max_power_bhp', 'max_power_rp',
            'model', 'brand', 'price_diff'
        ])
        match = match_row.to_dict()

    # Return both the predicted price and the closest match (if any)
    return PredictionResponse(predicted_price=predicted_price, match=match)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
