from fastapi import FastAPI
import pandas as pd
import pickle
from pydantic import BaseModel
import os

# Load model
MODEL_PATH = os.path.join("models", "model.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

app = FastAPI(title="ML Inference API")

# Input schema
class InputData(BaseModel):
    feature1: float
    feature2: float

@app.get("/health")
def health_check():
    return {"status": "API is running"}

@app.post("/predict")
def predict(data: InputData):
    df = pd.DataFrame([data.dict()])
    prediction = model.predict(df)[0]
    return {"prediction": prediction}
