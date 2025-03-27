from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Load ML model
model = pickle.load(open("model.pkl", "rb"))

# Initialize FastAPI app
app = FastAPI(title="ML Model API", version="1.0")

# Request schema
class ModelInput(BaseModel):
    features: list[float]

@app.get("/")
def home():
    return {"message": "ML API is running!"}

@app.post("/predict/")
def predict(input_data: ModelInput):
    input_array = np.array(input_data.features).reshape(1, -1)
    prediction = model.predict(input_array)
    return {"prediction": prediction.tolist()}
