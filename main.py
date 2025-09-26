from fastapi import FastAPI

# Create a FastAPI instance
app = FastAPI()

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/")
async def root():
    return {"message": "Hello World"}
#######################################

from contextlib import asynccontextmanager
import pickle
import os
from fastapi import FastAPI
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading models...")

    logistic_model_path = os.getenv("LOGISTIC_MODEL_PATH")
    rf_model_path = os.getenv("RANDOM_FOREST_MODEL_PATH")

    with open(logistic_model_path, "rb") as f:
        logreg_model = pickle.load(f)

    with open(rf_model_path, "rb") as f:
        rf_model = pickle.load(f)

    app.state.models = {
        "logistic_regression": logreg_model,
        "random_forest": rf_model
    }

    print("Models loaded successfully!")
    yield
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/models")
async def list_models():
    return {"available_models": list(app.state.models.keys())}
###########################################

from pydantic import BaseModel
from typing import Literal

from pydantic import BaseModel, Field
# donnees d'entree iris avec validation
class IrisData(BaseModel):
    sepal_length: float = Field(..., gt=0, description="Sepal length in cm, must be > 0")
    sepal_width: float = Field(..., gt=0, description="Sepal width in cm, must be > 0")
    petal_length: float = Field(..., gt=0, description="Petal length in cm, must be > 0")
    petal_width: float = Field(..., gt=0, description="Petal width in cm, must be > 0")

# Schéma pour les données d'entrée Iris
'''class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float'''

# Optionnel : pour choisir le modèle
class ModelChoice(BaseModel):
    model_name: Literal["logistic_regression", "random_forest"]

############################################
from fastapi import HTTPException

@app.post("/predict")
async def predict(data: IrisData, model_choice: ModelChoice):
    # Vérifier que le modèle existe
    model = app.state.models.get(model_choice.model_name)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")

    # Créer un tableau avec les features
    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    # Faire la prédiction
    prediction = model.predict(features).tolist()

    return {
        "model": model_choice.model_name,
        "prediction": prediction
    }
##############################################
import asyncio

@app.post("/predict/logistic_regression")
async def predict_logistic(data: IrisData):
    await asyncio.sleep(3)  # Simuler une tâche longue
    model = app.state.models.get("logistic_regression")

    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    prediction = model.predict(features).tolist()
    return {
        "model": "logistic_regression",
        "prediction": prediction
    }


@app.post("/predict/random_forest")
async def predict_rf(data: IrisData):
    await asyncio.sleep(5)  # Simuler une tâche encore plus longue
    model = app.state.models.get("random_forest")

    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    prediction = model.predict(features).tolist()
    return {
        "model": "random_forest",
        "prediction": prediction
    }
##############################################
from fastapi import BackgroundTasks
import time

# Fonction de logging simulée
def log_prediction(model_name: str, features: list, prediction: list):
    time.sleep(8)  # Simule une opération lente (écriture disque, etc.)
    with open("prediction_logs.txt", "a") as log_file:
        log_file.write(f"{model_name}, {features}, {prediction}\n")
    print("Prediction logged successfully!")

# Endpoint avec tâche en arrière-plan
@app.post("/predict/logistic_regression/background")
async def predict_logistic_background(data: IrisData, background_tasks: BackgroundTasks):
    model = app.state.models.get("logistic_regression")

    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    prediction = model.predict(features).tolist()

    # Lancer le logging en arrière-plan
    background_tasks.add_task(log_prediction, "logistic_regression", features, prediction)

    return {
        "model": "logistic_regression",
        "prediction": prediction,
        "message": "Prediction is processed, logging in background"
    }