import pandas as pd


def user_behavior_features(df):

    df = df.copy()

    # User average transaction
    df["user_avg_amount"] = (
        df.groupby("nameOrig")["amount"]
        .transform("mean")
    )

    # User max transaction
    df["user_max_amount"] = (
        df.groupby("nameOrig")["amount"]
        .transform("max")
    )

    # User std deviation
    df["user_std_amount"] = (
        df.groupby("nameOrig")["amount"]
        .transform("std")
    )

    # Transaction anomaly
    df["behavior_anomaly"] = (
        abs(df["amount"] - df["user_avg_amount"]) >
        2 * df["user_std_amount"]
    ).astype(int)

    return df