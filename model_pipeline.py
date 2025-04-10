import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import mlflow
import mlflow.sklearn
from elasticsearch import Elasticsearch


def prepare_data(file_path):
    """
    Charge et prépare les données de churn pour l'entraînement.
    Args:
        file_path (str): Chemin vers le fichier CSV
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    # Chargement des données
    data = pd.read_csv(file_path)

    # Affichage des informations sur les données
    print("\nInformations sur les données:")
    print(data.info())
    print("\nValeurs manquantes par colonne:")
    print(data.isnull().sum())

    # Gestion des valeurs manquantes et conversion de la cible
    data = data.dropna(subset=["Churn"])  # Supprimer les lignes où Churn est NaN
    data["Churn"] = data["Churn"].map({"True": 1, "False": 0, True: 1, False: 0})

    print("\nDistribution de la variable Churn après conversion:")
    print(data["Churn"].value_counts())

    # Séparation des features et de la cible
    X = data.drop(columns=["Churn"])
    y = data["Churn"]

    # Remplacement des valeurs manquantes dans les colonnes numériques
    numerical_columns = [
        "Account length",
        "Area code",
        "Number vmail messages",
        "Total day minutes",
        "Total day calls",
        "Total day charge",
        "Total eve minutes",
        "Total eve calls",
        "Total eve charge",
        "Total night minutes",
        "Total night calls",
        "Total night charge",
        "Total intl minutes",
        "Total intl calls",
        "Total intl charge",
        "Customer service calls",
    ]
    for col in numerical_columns:
        if col in X.columns:
            X[col] = X[col].fillna(X[col].median())

    # Encodage des colonnes catégorielles
    categorical_columns = ["State", "International plan", "Voice mail plan"]
    X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

    # Séparation des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """
    Entraîne le modèle Random Forest sur les données de churn.
    Args:
        X_train: Features d'entraînement
        y_train: Labels d'entraînement
    Returns:
        RandomForestClassifier: Modèle entraîné
    """
    # Définition des hyperparamètres
    n_estimators = 100
    max_depth = 10
    min_samples_split = 5
    random_state = 42

    # Enregistrement des hyperparamètres avec MLflow
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("min_samples_split", min_samples_split)
    mlflow.log_param("random_state", random_state)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Évalue le modèle sur les données de test et log les métriques avec MLflow.
    Args:
        model: Le modèle entraîné
        X_test: Features de test
        y_test: Labels de test
    Returns:
        float: Accuracy du modèle
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Calcul des métriques complémentaires
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Exemple pour envoyer la métrique 'accuracy' à Elasticsearch
    log_data = {
        "metric": "accuracy",
        "value": accuracy,
        "timestamp": "2025-03-07T10:00:00",
    }
    send_log_to_elasticsearch(log_data)

    # Enregistrement des métriques avec MLflow
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    return accuracy


def send_log_to_elasticsearch(log_data):
    """
    Envoie les logs vers Elasticsearch.
    Args:
        log_data (dict): Données du log à envoyer
    """
    es = Elasticsearch([{"host": "localhost", "port": 9200}])
    es.index(index="mlflow-metrics", body=log_data)


def save_model(model, filepath, input_example=None):
    """
    Sauvegarde le modèle avec joblib et enregistre avec MLflow.
    Args:
        model: Le modèle à sauvegarder
        filepath: Chemin vers le fichier de sauvegarde
        input_example: Exemple d'entrée pour le modèle (optionnel)
    """
    joblib.dump(model, filepath)
    if input_example is not None:
        mlflow.sklearn.log_model(
            model, artifact_path="random_forest_model", input_example=input_example
        )
    else:
        mlflow.sklearn.log_model(model, artifact_path="random_forest_model")


def load_model(filepath):
    """
    Charge un modèle de prédiction de churn sauvegardé.
    Args:
        filepath: Chemin vers le modèle sauvegardé
    Returns:
        Le modèle chargé
    """
    return joblib.load(filepath)
