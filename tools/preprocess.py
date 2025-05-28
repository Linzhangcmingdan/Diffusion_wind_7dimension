# tools/preprocess.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_and_clean_csv(path: str, drop_time_col: bool = True):
    df = pd.read_csv(path)
    if drop_time_col:
        df = df.iloc[:, 1:]  # Drop timestamp or first column
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df.values.astype(np.float32)

def normalize_series(data: np.ndarray):
    scaler = StandardScaler()
    data_norm = scaler.fit_transform(data)
    return data_norm.astype(np.float32), scaler

def inverse_normalize(data: np.ndarray, scaler):
    return scaler.inverse_transform(data)
