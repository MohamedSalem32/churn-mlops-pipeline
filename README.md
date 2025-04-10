# 🧠 Churn Prediction MLOps Pipeline

This project implements a full MLOps pipeline for customer churn prediction using:

- ✅ **Data preprocessing & model training**
- ⚙️ **Experiment tracking** with **MLflow**
- 🚀 **Model serving** via **FastAPI**
- 🐳 **Containerization** with **Docker**
- 📋 **Automation** via `Makefile`

---

## 📁 Tech Stack

- Python, scikit-learn, pandas, joblib  
- MLflow, FastAPI, Docker  
- Makefile for reproducible workflows

---

## 🚀 Quick Start

```bash
# 1. Set up virtual environment and install dependencies
make setup

# 2. Train the model and log experiment
make train

# 3. Start the FastAPI prediction service
make serve-api

# 4. Open MLflow tracking UI
make mlflow-ui
```
---

## 🐳 Docker Usage

```bash
make build     # Build Docker image
make run       # Run the container
make push      # Push image to Docker Hub
```
## ✍️ Author
- Mohamed Mohamed Salem
- 📧 mohamed.salem22032@gmail.com
- 📍 Data Science Engineer | ESPRIT | 4DS3 Option Data Science
- 📦 Docker Hub: mohamedsalem32
