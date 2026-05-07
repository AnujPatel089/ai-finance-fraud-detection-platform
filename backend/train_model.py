from features.feature_engineering import create_features
from features.behavior_features import user_behavior_features

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE


# ==============================
# 1. LOAD DATASET
# ==============================

df = pd.read_csv("data/raw/dataset.csv")

print("Dataset loaded successfully")
print("Shape:", df.shape)


# ==============================
# 2. FEATURE ENGINEERING
# ==============================

df = create_features(df)
df = user_behavior_features(df)

print("Feature engineering completed")
print("Columns:", df.columns.tolist())


# ==============================
# 3. SELECT FEATURES
# ==============================

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
y = df["isFraud"]


# ==============================
# 4. TRAIN TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train-test split completed")
print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)


# ==============================
# 5. HANDLE IMBALANCED DATA
# ==============================

smote = SMOTE(random_state=42)

X_train_resampled, y_train_resampled = smote.fit_resample(
    X_train,
    y_train
)

print("SMOTE completed")
print("Before SMOTE:", y_train.value_counts().to_dict())
print("After SMOTE:", y_train_resampled.value_counts().to_dict())


# ==============================
# 6. TRAIN MODEL
# ==============================

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(X_train_resampled, y_train_resampled)

print("Model training completed")


# ==============================
# 7. MODEL EVALUATION
# ==============================

y_pred = model.predict(X_test)

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ==============================
# 8. SAVE MODEL
# ==============================

joblib.dump(model, "models/fraud_model.pkl")
print("Model saved successfully at models/fraud_model.pkl")

print("\nModel saved successfully as fraud_model.pkl")