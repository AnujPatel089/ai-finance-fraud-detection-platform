# 💳 AI Finance Fraud Detection Platform

Production-grade AI fraud detection platform using FastAPI, Streamlit, Docker, SHAP explainability, Gemini AI, batch transaction analytics, and machine learning.

---

# 🚀 Features

- Fraud prediction using Machine Learning
- FastAPI REST API
- Streamlit interactive dashboard
- SHAP explainability
- Gemini AI fraud explanations
- Batch CSV fraud prediction
- Fraud analytics dashboard
- Dockerized full-stack architecture

---

# 🛠️ Tech Stack

- Python
- FastAPI
- Streamlit
- Scikit-learn
- SHAP
- Plotly
- Docker
- Gemini AI
- Pandas
- NumPy

---

# 📁 Project Structure

```bash
personal-finance-ai-platform/
│
├── app/                  # Streamlit frontend
├── backend/              # FastAPI backend
├── models/               # Trained ML model
├── notebooks/            # EDA notebooks
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

# ⚙️ Setup Instructions

## 1️⃣ Clone Repository

```bash
git clone https://github.com/AnujPatel089/ai-finance-fraud-detection-platform.git
```

---

## 2️⃣ Open Project Folder

```bash
cd ai-finance-fraud-detection-platform
```

---

# 📦 Install Dependencies

## Option 1 — Local Setup

Create virtual environment:

```bash
python -m venv venv
```

Activate virtual environment:

### Windows

```bash
venv\Scripts\activate
```

Install requirements:

```bash
pip install -r requirements.txt
```

---

# 🔑 Gemini API Setup

Create `.env` file in project root:

```env
GEMINI_API_KEY=your_api_key_here
```

Get Gemini API key from:

https://ai.google.dev/

---

# 📊 Dataset Setup

Download PaySim dataset from Kaggle:

https://www.kaggle.com/datasets/ealaxi/paysim1

Place dataset inside:

```bash
data/raw/dataset.csv
```

---

# 🤖 Train Model

Run:

```bash
python backend/train_model.py
```

This will create:

```bash
models/fraud_model.pkl
```

---

# ▶️ Run Application Locally

## Start FastAPI Backend

```bash
python -m uvicorn backend.main:app --reload
```

Backend URL:

```bash
http://127.0.0.1:8000/docs
```

---

## Start Streamlit Frontend

```bash
streamlit run app/app.py
```

Frontend URL:

```bash
http://localhost:8501
```

---

# 🐳 Run with Docker

## Build Containers

```bash
docker-compose build
```

---

## Start Containers

```bash
docker-compose up
```

---

# 🌐 Application URLs

## Frontend

```bash
http://localhost:8501
```

## Backend API Docs

```bash
http://localhost:8000/docs
```

---

# 📂 Batch Prediction

Upload CSV file with columns:

```csv
step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest
```

---

# 🔍 API Endpoints

| Endpoint | Description |
|---|---|
| `/predict` | Single fraud prediction |
| `/batch_predict` | Batch CSV fraud prediction |
| `/explain` | Gemini AI explanation |
| `/shap` | SHAP explainability |

---

# 📈 Future Improvements

- PostgreSQL integration
- JWT Authentication
- Azure deployment
- MLflow MLOps
- CI/CD pipeline
- Fraud monitoring system

---

# 👨‍💻 Author

Anuj Patel

GitHub:
https://github.com/AnujPatel089