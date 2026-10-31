from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = Path("models/house_model.joblib")

FEATURES = [
    "area_sqft",
    "bedrooms",
    "bathrooms",
    "house_age",
]

if not MODEL_PATH.exists():
    raise RuntimeError(
        "Model not found. Run python train.py first."
    )

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="House Price Prediction API",
    description="Predict house prices using a trained Linear Regression model.",
    version="1.0.0",
)


class HouseInput(BaseModel):
    area_sqft: float = Field(gt=0, examples=[1500])
    bedrooms: int = Field(ge=1, examples=[3])
    bathrooms: int = Field(ge=1, examples=[2])
    house_age: float = Field(ge=0, examples=[5])


@app.get("/")
def home():
    return {
        "message": "House Price Prediction API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
    }


@app.post("/predict")
def predict(house: HouseInput):
    row = pd.DataFrame(
        [house.model_dump()],
        columns=FEATURES,
    )

    prediction = float(model.predict(row)[0])

    return {
        "predicted_price": round(prediction, 2),
        "currency": "INR",
    }
