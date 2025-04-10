from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # Exemple d'algorithme à utiliser pour le réentraînement
import pandas as pd

# Charger le modèle
MODEL_PATH = "churn_model.joblib"
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Erreur lors du chargement du modèle: {str(e)}")

# Définir l'API
app = FastAPI(title="Churn Prediction API")

# Définir le format d'entrée pour la prédiction
class ChurnInput(BaseModel):
    features: list[float]

# Définir le format d'entrée pour le réentraînement
class RetrainInput(BaseModel):
    new_data: list[list[float]]  # Liste de nouvelles données d'entraînement
    target: list[int]  # Liste des labels pour l'entraînement

# Endpoint de prédiction
@app.post("/predict")
def predict(data: ChurnInput):
    try:
        # Transformer les données d'entrée en tableau numpy
        X_input = np.array(data.features).reshape(1, -1)
        # Faire la prédiction
        prediction = model.predict(X_input)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint de réentraînement
@app.post("/retrain")
def retrain(data: RetrainInput):
    try:
        # Charger de nouvelles données d'entraînement
        new_data = np.array(data.new_data)
        target = np.array(data.target)

        # Diviser les données en jeu d'entraînement et de validation
        X_train, X_val, y_train, y_val = train_test_split(new_data, target, test_size=0.2)

        # Exemple de réentraînement du modèle avec RandomForest
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, y_train)

        # Sauvegarder le modèle réentraîné
        joblib.dump(model, MODEL_PATH)

        return {"message": "Modèle réentraîné avec succès"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
