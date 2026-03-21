import pandas as pd
import joblib
import logging
from fastapi import FastAPI
from pydantic import BaseModel

import huggingface_hub
from huggingface_hub import hf_hub_download

huggingface_hub.login(token = 'YOUR_TOKEN')

# Using model from Hugging Face Hub: https://huggingface.co/brjapon/iris-dt
# Accompanying dataset is hosted in Hugging Face under 'brjapon/iris'
model_joblib="iris_dt.joblib"
model_path = hf_hub_download(repo_id="brjapon/iris-dt",
                             filename=model_joblib,
                             repo_type="model")

model = joblib.load(model_path)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

logger.info(f"Model downloaded from: {model}")

# Uncomment the following lines to load the model from a local file instead
logger.info("Loading the pre-trained Iris model from file: ", model_joblib)
#model = joblib.load("iris_model.joblib")
logger.info("Model loaded successfully")

# Create the FastAPI application
app = FastAPI()

# Define the input data model using Pydantic
# Pydantic model for the Iris features
class IrisInput(BaseModel):
    SepalLengthCm: float
    SepalWidthCm: float
    PetalLengthCm: float
    PetalWidthCm: float

@app.post("/predict")
def predict_iris(data: IrisInput):
    """
    Predict the Iris species given measurements.
    """
    # Convert the input parameters into a Dataframe, as expected by the model
    input = pd.DataFrame([{
    "SepalLengthCm": data.SepalLengthCm,
    "SepalWidthCm":  data.SepalWidthCm,
    "PetalLengthCm": data.PetalLengthCm,
    "PetalWidthCm":  data.PetalWidthCm
    }])
    
    
    prediction =  model.predict(input)[0]

    # Convert the prediction to the string label
    if prediction == 0:
        predicted_species = 'iris-setosa'
    elif prediction == 1:
        predicted_species = 'Iris-versicolor'
    elif prediction == 2:
        predicted_species = 'Iris-virginica'
    else:
        predicted_species = 'Invalid prediction'
    
    logger.info(f"Received prediction request: {input}")
    logger.info(f"Returning prediction: {predicted_species}")

    return {
        "predicted_species": predicted_species
    }

@app.get("/")
def root():
    return {"message": "Hello, I'm serving a saved Iris model with FastAPI!"}
