import os
import json
import numpy as np
import pandas as pd
import joblib

from src.data_loader import load_equity


# -----------------------------
# Load trained ML model
# -----------------------------
def load_model(ticker):
    path = os.path.join("artifacts", ticker, "model.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found for {ticker}. Run optimize first.")
    return joblib.load(path)


# -----------------------------
# Load metadata (accuracy, params)
# -----------------------------
def load_metadata(ticker):
    path = os.path.join("artifacts", ticker, "frozen_params.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def build_features(df):
    df = df.copy()

    # base features
    df['ret_1d'] = df['close'].pct_change()
    df['roll_ret'] = df['ret_1d'].rolling(20).mean()
    df['roll_vol'] = df['ret_1d'].rolling(20).std()

    # momentum
    df['mom_5'] = df['close'].pct_change(5)
    df['mom_10'] = df['close'].pct_change(10)
    df['mom_20'] = df['close'].pct_change(20)

    # realized volatility
    df['vol_5'] = df['ret_1d'].rolling(5).std()
    df['vol_10'] = df['ret_1d'].rolling(10).std()
    df['vol_20'] = df['ret_1d'].rolling(20).std()

    # long-term trend
    df['sma_50'] = df['close'].rolling(50).mean()
    df['trend_50'] = df['close'] / df['sma_50'] - 1

    # mean reversion
    df['zscore_20'] = (df['close'] - df['close'].rolling(20).mean()) / df['vol_20']

    df = df.dropna().reset_index(drop=True)
    return df



# -----------------------------
# Main prediction function
# -----------------------------
def predict(ticker, start="2010-01-01", end=None):
    # Load model + metadata
    model = load_model(ticker)
    meta = load_metadata(ticker)

    # Load data
    df = load_equity(ticker, start=start, end=end)

    # Clean duplicates
    if "date" in df.columns:
        df = df.sort_values("date").drop_duplicates(subset="date", keep="last")
    else:
        df = df.drop_duplicates()

    df = df.reset_index(drop=True)

    # Build features
    df_feat = build_features(df)

    # Extract last row of features
    feature_cols = [
        'ret_1d', 'roll_ret', 'roll_vol',
        'mom_5', 'mom_10', 'mom_20',
        'vol_5', 'vol_10', 'vol_20',
        'trend_50',
        'zscore_20'
    ]

    X_last = df_feat[feature_cols].iloc[[-1]]


    # Predict
    pred = model.predict(X_last)[0]
    prob = model.predict_proba(X_last)[0]  # [prob_down, prob_up]

    signal = "UP" if pred == 1 else "DOWN"

    return {
        "date": df_feat['date'].iloc[-1] if 'date' in df_feat.columns else None,
        "prob_up": float(prob[1]),
        "prob_down": float(prob[0]),
        "signal": signal,
        "model_accuracy": meta.get("val_accuracy", None),
        "model_params": meta.get("best_params", None)
    }


# -----------------------------
# CLI wrapper
# -----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, required=True)
    parser.add_argument("--start", type=str, default="2010-01-01")
    parser.add_argument("--end", type=str, default=None)
    args = parser.parse_args()

    out = predict(args.ticker, args.start, args.end)
    print(out)
