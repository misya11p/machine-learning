import numpy as np
from .._base import BaseClassifire, BaseRegressor, BaseCluster, BaseTransformer


class LinearRegression(BaseRegressor):
    def __init__(self):
        self.weights = None

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        self.weights = np.linalg.inv(X.T @ X) @ X.T @ y

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        return X @ self.weights


class Ridge(LinearRegression):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        I = np.identity(X.shape[1], dtype=np.float32)
        I[0, 0] = 0
        self.weights = np.linalg.inv(X.T @ X + self.alpha*I) @ X.T @ y
