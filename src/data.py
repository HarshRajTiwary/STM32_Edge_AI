import csv
import numpy as np
from sklearn.model_selection import train_test_split


def load_csv(path, target_column, test_size=0.2, random_state=42):
    import pandas as pd

    df = pd.read_csv(path)
    y = df[target_column].values
    X = df.drop(columns=[target_column]).values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return (X_train, y_train), (X_test, y_test)


def representative_data_generator_from_array(X, batch_size=1):
    def gen():
        for i in range(0, X.shape[0], batch_size):
            yield [X[i : i + batch_size].astype(np.float32)]

    return gen
