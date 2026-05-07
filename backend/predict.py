import joblib
import pandas as pd

from backend.features.feature_engineering import create_features
from backend.features.behavior_features import user_behavior_features


model = joblib.load("models/fraud_model.pkl")


def predict_fraud(input_data: dict):
    df = pd.DataFrame([input_data])

    df = create_features(df)
    df = user_behavior_features(df)

    features = [
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

    X = df[features]

    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]

    return {
        "fraud_prediction": int(prediction),
        "fraud_probability": round(float(probability), 4),
        "risk_level": "HIGH" if probability >= 0.7 else "MEDIUM" if probability >= 0.4 else "LOW"
    }