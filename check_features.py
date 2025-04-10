import joblib

MODEL_PATH = "churn_model.joblib"
model = joblib.load(MODEL_PATH)

if hasattr(model, "feature_names_in_"):
    print("Features utilisées dans le modèle :")
    print(model.feature_names_in_)
else:
    print("Les noms des features ne sont pas stockés dans le modèle.")
