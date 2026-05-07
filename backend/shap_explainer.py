import shap
import joblib
import pandas as pd

model = joblib.load("models/fraud_model.pkl")

explainer = shap.TreeExplainer(model)


def generate_shap_values(input_df):

    shap_values = explainer.shap_values(input_df)

    return shap_values