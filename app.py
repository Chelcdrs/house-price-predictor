from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os
import logging
import uvicorn

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define model and scaler paths
model_path = "house_price_model.pkl"
scaler_path = "scaler.pkl"

# FastAPI setup
app = FastAPI()

# Define a request model for input data
class HouseFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

# Load the model and scaler
if not os.path.exists(model_path):
    logger.error("Model file not found!")
    raise HTTPException(status_code=500, detail="Model file not found!")
model = joblib.load(model_path)

if not os.path.exists(scaler_path):
    logger.error("Scaler file not found!")
    raise HTTPException(status_code=500, detail="Scaler file not found!")
scaler = joblib.load(scaler_path)

# Create a prediction endpoint
@app.post("/predict")
def predict_price(features: HouseFeatures):
    try:
        # Convert input data into a format suitable for prediction
        input_data = np.array([[features.MedInc, features.HouseAge, features.AveRooms,
                                features.AveBedrms, features.Population, features.AveOccup,
                                features.Latitude, features.Longitude]])
        input_scaled = scaler.transform(input_data)  # Scale the input data using the same scaler used during training
        prediction = model.predict(input_scaled)
        return {"Predicted House Price": prediction[0]}  # Return the predicted house price
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# Running the FastAPI application
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
