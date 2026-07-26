import os
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier

from src.data_loader import load_equity


# -----------------------------
# Build features
# -----------------------------
def build_features(df):
    df = df.copy()

    # base features
    df['ret_1d'] = df['close'].pct_change()
    df['roll_ret'] = df['ret_1d'].rolling(20).mean()
    df['roll_vol'] = df['ret_1d'].rolling(20).std()

    # momentum (near-term regime)
    df['mom_5'] = df['close'].pct_change(5)
    df['mom_10'] = df['close'].pct_change(10)
    df['mom_20'] = df['close'].pct_change(20)

    # realized volatility (conviction) – short horizons
    df['vol_5'] = df['ret_1d'].rolling(5).std()
    df['vol_10'] = df['ret_1d'].rolling(10).std()
    df['vol_20'] = df['ret_1d'].rolling(20).std()

    # long-term trend (alignment with equity trend)
    df['sma_50'] = df['close'].rolling(50).mean()
    df['trend_50'] = df['close'] / df['sma_50'] - 1

    # mean reversion (corrections)
    df['zscore_20'] = (df['close'] - df['close'].rolling(20).mean()) / df['vol_20']

    df = df.dropna().reset_index(drop=True)
    return df


# -----------------------------
# Build dataset (X, y)
# -----------------------------
def build_dataset(df_feat):
    feature_cols = [
        'ret_1d', 'roll_ret', 'roll_vol',
        'mom_5', 'mom_10', 'mom_20',
        'vol_5', 'vol_10', 'vol_20',
        'trend_50',
        'zscore_20'
    ]

    X = df_feat[feature_cols].copy()

    next_ret = df_feat['ret_1d'].shift(-1)
    y = np.where(next_ret > 0, 1, 0)  # 1 = UP, 0 = DOWN

    X = X.iloc[:-1].reset_index(drop=True)
    y = pd.Series(y[:-1]).reset_index(drop=True)

    return X, y, feature_cols


# -----------------------------
# Train & optimize model (GB + walk-forward)
# -----------------------------
def optimize_model(X, y):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(random_state=42))
    ])

    param_grid = {
        "clf__n_estimators": [200, 400, 800],
        "clf__learning_rate": [0.01, 0.05, 0.1],
        "clf__max_depth": [3, 4, 5],
        "clf__subsample": [0.7, 0.9, 1.0]
    }

    tscv = TimeSeriesSplit(n_splits=3)

    grid = GridSearchCV(
        pipe,
        param_grid,
        scoring="accuracy",
        cv=tscv,
        n_jobs=-1,
        verbose=0
    )

    grid.fit(X, y)

    best_model = grid.best_estimator_
    val_acc = grid.best_score_
    best_params = grid.best_params_

    return best_model, val_acc, best_params


# -----------------------------
# Save frozen model + metadata
# -----------------------------
def save_frozen_model(ticker, model, params, val_acc):
    path = os.path.join("artifacts", ticker)
    os.makedirs(path, exist_ok=True)

    meta = {
        "best_params": params,
        "val_accuracy": float(val_acc)
    }
    with open(os.path.join(path, "frozen_params.json"), "w") as f:
        json.dump(meta, f, indent=4)

    import joblib
    joblib.dump(model, os.path.join(path, "model.joblib"))


# -----------------------------
# Run optimization
# -----------------------------
def run_optimize(ticker, start="2010-01-01", end=None):
    df = load_equity(ticker, start=start, end=end)
    df_feat = build_features(df)
    X, y, _ = build_dataset(df_feat)

    model, val_acc, best_params = optimize_model(X, y)
    save_frozen_model(ticker, model, best_params, val_acc)

    return best_params, val_acc
