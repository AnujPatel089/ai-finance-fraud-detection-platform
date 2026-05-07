from fastapi import APIRouter, UploadFile, File
import pandas as pd
import joblib

router = APIRouter()

model = joblib.load("models/fraud_model.pkl")

@router.post("/batch_predict")
async def batch_predict(file: UploadFile = File(...)):

    df = pd.read_csv(file.file)

    # =========================
    # Feature Engineering
    # =========================

    df["balance_diff_orig"] = (
        df["oldbalanceOrg"] - df["newbalanceOrig"]
    )

    df["balance_diff_dest"] = (
        df["newbalanceDest"] - df["oldbalanceDest"]
    )

    # Encode transaction type
    df = pd.get_dummies(df, columns=["type"])

    # =========================
    # Match training columns
    # =========================

    required_cols = model.feature_names_in_

    for col in required_cols:
        if col not in df.columns:
            df[col] = 0

    df = df[required_cols]

    # =========================
    # Predictions
    # =========================

    predictions = model.predict(df)

    probabilities = model.predict_proba(df)[:, 1]

    # =========================
    # Add Results
    # =========================

    result_df = pd.DataFrame({
        "prediction": predictions,
        "fraud_probability": probabilities
    })

    final_df = pd.concat([df.reset_index(drop=True), result_df], axis=1)

    # Save results
    output_path = "batch_predictions.csv"

    final_df.to_csv(output_path, index=False)

    return {
        "message": "Batch prediction completed",
        "total_transactions": len(final_df),
        "fraud_count": int(predictions.sum()),
        "download_file": output_path,
        "records": final_df.to_dict(orient="records")
    }