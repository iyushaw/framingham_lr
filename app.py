import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from joblib import load
from fastapi import FastAPI
from pydantic import BaseModel

# Define the input data schema
class InputData(BaseModel):
    gender: float
    age: float
    currentSmoker: float
    cigsPerDay: float
    BPMeds: float
    prevalentStroke: float
    prevalentHyp: float
    diabetes: float
    totChol: float
    sysBP: float
    diaBP: float
    BMI: float
    heartRate: float  # Updated field name
    glucose: float

# Load the trained model
loaded_model = load('trained_model.pkl')

# Create an instance of the FastAPI app
app = FastAPI()

# Define the prediction endpoint
@app.post("/predict")
def predict(data: InputData):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([data.dict()])

    # Make predictions using the loaded model and pipeline steps
    prediction = loaded_model.predict(input_df)
    probability = loaded_model.predict_proba(input_df)[:, 1]  # Probability of class 1

    return {"prediction": int(prediction[0]), "probability": float(probability[0])}

# if __name__ == "__main__":
#     # Run the FastAPI app
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
