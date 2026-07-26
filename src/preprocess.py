# src/preprocess.py
import pandas as pd
from sklearn.preprocessing import StandardScaler

class FeatureScaler:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_cols = None

    def fit(self, df: pd.DataFrame, feature_cols):
        self.feature_cols = list(feature_cols)
        X = df[self.feature_cols].values
        self.scaler.fit(X)

    def transform(self, df: pd.DataFrame):
        if self.feature_cols is None:
            raise ValueError("Scaler not fitted: call fit() first.")
        X = df[self.feature_cols].values
        Xs = self.scaler.transform(X)
        df_scaled = df.copy()
        for i, col in enumerate(self.feature_cols):
            df_scaled[col + "_scaled"] = Xs[:, i]
        return df_scaled

    def fit_transform(self, df: pd.DataFrame, feature_cols):
        self.fit(df, feature_cols)
        return self.transform(df)
