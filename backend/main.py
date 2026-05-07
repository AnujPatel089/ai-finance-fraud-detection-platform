from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

from backend.predict import predict_fraud
from backend.ai_explainer import generate_fraud_explanation

from backend.features.feature_engineering import create_features
from backend.features.behavior_features import user_behavior_features
from backend.shap_explainer import generate_shap_values
from backend.routes.batch_predict import router as batch_router

app = FastAPI(
    title="AI Personal Finance Fraud API",
    version="1.0"
)

app.include_router(batch_router)
# ==============================
# REQUEST MODEL
# ==============================

class TransactionRequest(BaseModel):

    step: int
    type: str

    amount: float

    nameOrig: str

    oldbalanceOrg: float
    newbalanceOrig: float

    nameDest: str

    oldbalanceDest: float
    newbalanceDest: float


# ==============================
# FEATURES
# ==============================

FEATURES = [
    "amount",
    "log_amount",
    "amount_to_balance_ratio",
    "sender_balance_error",
    "receiver_balance_error",
    "transaction_type_encoded",
    "high_risk_transaction",
    "sender_transaction_velocity",
    "deviation_from_avg",
    "risk_score",
    "behavior_anomaly"
]


# ==============================
# HOME ROUTE
# ==============================

@app.get("/")
def home():

    return {
        "message": "AI Personal Finance Fraud Detection API Running"
    }


# ==============================
# FRAUD PREDICTION ROUTE
# ==============================

@app.post("/predict")
def fraud_prediction(data: TransactionRequest):

    transaction = data.dict()

    result = predict_fraud(transaction)

    return result


# ==============================
# GEMINI AI EXPLANATION
# ==============================

@app.post("/explain")
def explain_transaction(data: TransactionRequest):

    transaction = data.dict()

    result = predict_fraud(transaction)

    ai_explanation = generate_fraud_explanation(
        data=transaction,
        prediction=result["fraud_prediction"],
        probability=result["fraud_probability"],
        risk_level=result["risk_level"]
    )

    return {
        "ai_explanation": ai_explanation
    }


# ==============================
# SHAP EXPLANATION
# ==============================

@app.post("/shap")
def shap_explanation(data: TransactionRequest):

    transaction = data.dict()

    df = pd.DataFrame([transaction])

    df = create_features(df)
    df = user_behavior_features(df)

    X = df[FEATURES]

    shap_values = generate_shap_values(X)

    import numpy as np

    if isinstance(shap_values, list):

        fraud_shap_values = np.array(shap_values[1][0]).flatten()

    else:

        fraud_shap_values = np.array(shap_values[0]).flatten()

    shap_result = []

    for feature, value, shap_value in zip(
        FEATURES,
        X.iloc[0].values,
        fraud_shap_values
    ):

        shap_result.append({
            "feature": feature,
            "value": float(value),
            "impact": float(np.array(shap_value).item())
        })

    shap_result = sorted(
        shap_result,
        key=lambda x: abs(x["impact"]),
        reverse=True
    )

    return {
        "shap_values": shap_result
    }