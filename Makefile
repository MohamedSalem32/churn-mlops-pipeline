# Définition de l'environnement virtuel et des fichiers nécessaires
VENV=venv
PYTHON=$(VENV)/bin/python
PIP=$(VENV)/bin/pip
REQUIREMENTS=requirements.txt
DATA_PATH=churn-mlops.csv
MODEL_PATH=churn_model.joblib

# Définition des fichiers sources pour la détection des changements
SRC=main.py model_pipeline.py test_env.py

# Création de l'environnement virtuel
.PHONY: setup
setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r $(REQUIREMENTS)

# Vérification du code (qualité, formatage, sécurité)
.PHONY: lint
lint:
	$(PIP) install black flake8 bandit
	black $(SRC)
	flake8 $(SRC) --max-line-length=100
	bandit -r $(SRC)

# Préparation des données
.PHONY: prepare
prepare:
	$(PYTHON) main.py --prepare --data_path $(DATA_PATH)

# Entraînement du modèle
.PHONY: train
train:
	$(PYTHON) main.py --train --data_path $(DATA_PATH) --model_path $(MODEL_PATH) --experiment_name churn_prediction

# Exécution des tests
.PHONY: test
test:
	$(PYTHON) -c "import sys; print('Environnement Python activé:', sys.prefix)"
	$(PYTHON) -c "import numpy; print('NumPy version:', numpy.__version__)"
	$(PYTHON) -c "import pandas; print('Pandas version:', pandas.__version__)"

# Démarrage de l'interface MLflow
.PHONY: mlflow-ui
mlflow-ui:
	mlflow ui --host 0.0.0.0 --port 5000


.PHONY: kill-ports
kill-ports:
	@echo "🔪 Killing processes on ports 5000 (MLflow) and 8000 (FastAPI)..."
	@fuser -k 5000/tcp || true
	@fuser -k 8000/tcp || true

# MLflow avec backend SQLite
.PHONY: mlflow-sqlite
mlflow-sqlite:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000

# Exécuter tout le pipeline
.PHONY: all
all: setup lint prepare train test

# Nettoyage de l'environnement (enlever les fichiers temporaires et Python)
.PHONY: clean-env
clean-env:
	rm -rf $(VENV) __pycache__

# Nettoyer les conteneurs et images Docker inutiles
.PHONY: clean-docker
clean-docker:
	docker rm -f $(shell docker ps -aq)
	docker rmi $(IMAGE_NAME):$(TAG)

# Détection des modifications et exécution automatique
.PHONY: watch
watch:
	while inotifywait -e modify -r .; do make all; done

# Serveur FastAPI
.PHONY: serve-api
serve-api:
	uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Variables Docker
IMAGE_NAME = mohamedsalem32/mohamed_salem_4ds3_mlops
TAG = latest

# Construction de l'image Docker
.PHONY: build
build:
	docker build -t $(IMAGE_NAME):$(TAG) .

# Pousser l'image sur Docker Hub
.PHONY: push
push: build
	docker push $(IMAGE_NAME):$(TAG)

# Lancer le conteneur Docker
.PHONY: run
run:
	docker run -d -p 8000:80 $(IMAGE_NAME):$(TAG)

# Rebuild et push l'image
.PHONY: rebuild
rebuild: clean-docker build push
