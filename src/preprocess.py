import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib


def fit_transform_save(X, out_path="outputs/scaler.joblib"):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    joblib.dump(scaler, out_path)
    return Xs


def load_scaler(path):
    return joblib.load(path)


def transform_with_scaler(X, scaler):
    return scaler.transform(X)
