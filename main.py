from model_pipeline import prepare_data, train_model, evaluate_model, save_model
import argparse
import os
import mlflow
import mlflow.sklearn
from elasticsearch import Elasticsearch

# Connexion à Elasticsearch
def connect_to_elasticsearch():
    # Utiliser 'host.docker.internal' pour accéder à Elasticsearch depuis un environnement local (non Docker)
    es = Elasticsearch([{"host": "localhost", "port": 9200,"scheme": "http"}])    
    if es.ping():
        print("Connexion à Elasticsearch réussie!")
    else:
        print("Erreur de connexion à Elasticsearch!")
    return es

# Fonction pour envoyer les logs à Elasticsearch
def send_log_to_elasticsearch(es, log_data):
    try:
        # Essayez d'envoyer les logs à Elasticsearch
        response = es.index(index="mlflow-logs", document=log_data)
        if response["result"] == "created":
            print("Log envoyé avec succès à Elasticsearch.")
        else:
            print("Erreur lors de l'envoi du log.")
    except Exception as e:
        print(f"Erreur lors de l'envoi des logs à Elasticsearch: {str(e)}")

# Fonction pour créer l'index si nécessaire
def create_index_if_not_exists(es, index_name):
    if not es.indices.exists(index=index_name):
        es.indices.create(index=index_name)
        print(f"Index '{index_name}' créé avec succès!")
    else:
        print(f"L'index '{index_name}' existe déjà.")

# Fonction principale pour préparer, entraîner et évaluer le modèle
def main():
    parser = argparse.ArgumentParser(
        description="Pipeline ML pour la prédiction du churn"
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Chemin vers le fichier de données"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="churn_model.joblib",
        help="Chemin pour sauvegarder le modèle",
    )
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Préparer les données sans entraîner le modèle",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Entraîner le modèle après préparation des données",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="churn_prediction",
        help="Nom de l'expérience MLflow",
    )
    args = parser.parse_args()

    # Set MLflow experiment
    mlflow.set_experiment(args.experiment_name)

    # Vérification de l'existence du fichier de données
    if not os.path.exists(args.data_path):
        print(f"Erreur: Le fichier {args.data_path} n'existe pas.")
        return

    try:
        # Préparation des données
        print("Préparation des données...")
        X_train, X_test, y_train, y_test = prepare_data(args.data_path)
        print("Données préparées avec succès!")
        print(f"Dimensions data_train: {X_train.shape}")

        if args.prepare:
            print("Préparation terminée. Aucun modèle entraîné.")
            return

        if args.train:
            with mlflow.start_run():
                # Log parameters regarding the dataset
                mlflow.log_param("data_path", args.data_path)
                mlflow.log_param("num_samples_train", X_train.shape[0])
                mlflow.log_param("num_features", X_train.shape[1])

                # Entraînement du modèle
                print("\nEntraînement du modèle...")
                model = train_model(X_train, y_train)
                print("Modèle entraîné avec succès!")

                # Évaluation du modèle
                print("\nÉvaluation du modèle...")
                accuracy = evaluate_model(model, X_test, y_test)
                print(f"Accuracy du modèle: {accuracy:.4f}")

                # Sauvegarde du modèle
                input_ex = X_train.iloc[0:1]  # Exemple pour log_model
                print("\nSauvegarde du modèle...")
                save_model(model, args.model_path, input_example=input_ex)
                print(f"Modèle sauvegardé dans {args.model_path}")

                mlflow.log_param("model_path", args.model_path)

                # Log feature importances
                for i, importance in enumerate(model.feature_importances_):
                    mlflow.log_metric(f"feature_importance_{i}", importance)

                # Log à Elasticsearch
                log_data = {
                    "metric": "accuracy",
                    "value": accuracy,
                    "timestamp": "2025-03-07T10:00:00",
                }

                # Connexion à Elasticsearch
                es = connect_to_elasticsearch()

                # Créer l'index si nécessaire
                create_index_if_not_exists(es, "mlflow-logs")

                # Envoyer les logs à Elasticsearch
                send_log_to_elasticsearch(es, log_data)

    except Exception as e:
        print(f"Une erreur s'est produite: {str(e)}")


if __name__ == "__main__":
    main()
