import pandas as pd
import numpy as np


def create_features(df):

    # Copy dataset
    df = df.copy()

    # -----------------------------
    # BASIC TRANSACTION FEATURES
    # -----------------------------

    df["transaction_hour"] = df["step"] % 24

    df["is_night_transaction"] = (
        (df["transaction_hour"] >= 22) |
        (df["transaction_hour"] <= 5)
    ).astype(int)

    # -----------------------------
    # AMOUNT FEATURES
    # -----------------------------

    df["log_amount"] = np.log1p(df["amount"])

    df["amount_to_balance_ratio"] = (
        df["amount"] / (df["oldbalanceOrg"] + 1)
    )

    df["amount_large_flag"] = (
        df["amount"] > df["amount"].quantile(0.95)
    ).astype(int)

    # -----------------------------
    # BALANCE DIFFERENCE FEATURES
    # -----------------------------

    df["sender_balance_error"] = (
        df["oldbalanceOrg"] -
        df["newbalanceOrig"] -
        df["amount"]
    )

    df["receiver_balance_error"] = (
        df["newbalanceDest"] -
        df["oldbalanceDest"] -
        df["amount"]
    )

    # -----------------------------
    # ZERO BALANCE FEATURES
    # -----------------------------

    df["empty_sender"] = (
        df["newbalanceOrig"] == 0
    ).astype(int)

    df["empty_receiver"] = (
        df["newbalanceDest"] == 0
    ).astype(int)

    # -----------------------------
    # TRANSACTION TYPE ENCODING
    # -----------------------------

    transaction_map = {
        "CASH_OUT": 0,
        "PAYMENT": 1,
        "TRANSFER": 2,
        "DEBIT": 3,
        "CASH_IN": 4
    }

    df["transaction_type_encoded"] = (
        df["type"].map(transaction_map)
    )

    # -----------------------------
    # HIGH RISK TYPES
    # -----------------------------

    high_risk = ["TRANSFER", "CASH_OUT"]

    df["high_risk_transaction"] = (
        df["type"].isin(high_risk)
    ).astype(int)

    # -----------------------------
    # VELOCITY FEATURES
    # -----------------------------

    df["sender_transaction_velocity"] = (
        df.groupby("nameOrig")["amount"]
        .transform("count")
    )

    # -----------------------------
    # AVERAGE USER TRANSACTION
    # -----------------------------

    avg_amount = (
        df.groupby("nameOrig")["amount"]
        .transform("mean")
    )

    df["deviation_from_avg"] = (
        df["amount"] - avg_amount
    )

    # -----------------------------
    # RISK SCORE
    # -----------------------------

    df["risk_score"] = (
        df["high_risk_transaction"] * 30 +
        df["is_night_transaction"] * 20 +
        df["amount_large_flag"] * 25 +
        df["empty_sender"] * 15 +
        df["empty_receiver"] * 10
    )

    return df