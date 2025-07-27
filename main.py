from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Initialize FastAPI app
app = FastAPI(title="Iris Classifier API")

# Define input data model using Pydantic
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Load the trained model
model = joblib.load('model.pkl')
iris_species = ['setosa', 'versicolor', 'virginica']  # Class names

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Iris Classifier API!"}

# Prediction endpoint
@app.post("/predict")
async def predict(iris: IrisInput):
    try:
        # Prepare input data
        data = np.array([[iris.sepal_length, iris.sepal_width, 
                         iris.petal_length, iris.petal_width]])
        
        # Make prediction
        prediction = model.predict(data)
        predicted_species = iris_species[prediction[0]]
        
        # Return prediction and probability
        probabilities = model.predict_proba(data)[0].tolist()
        return {
            "species": predicted_species,
            "probabilities": {iris_species[i]: prob for i, prob in enumerate(probabilities)}
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")