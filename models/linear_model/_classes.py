import numpy as np
from .._base import BaseClassifire, BaseRegressor, BaseCluster, BaseTransformer


class LinearRegression(BaseRegressor):
    def _set_params(self, weights):
        weights = np.array(weights).flatten()
        self.intercept_ = weights[0]
        self.coef_ = weights[1:]

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.matrix(np.insert(X, 0, 1, axis=1))
        y = np.matrix(y).T
        weights = (X.T @ X)**(-1) @ X.T @ y
        self._set_params(weights)

    def predict(self, X: np.ndarray):
        return X @ self.coef_ + self.intercept_


class Ridge(LinearRegression):
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X, y):
        X = np.matrix(np.insert(X, 0, 1, axis=1))
        y = np.matrix(y).T
        I = np.identity(X.shape[1], dtype=np.float32)
        weights = (X.T @ X + self.alpha*I)**(-1) @ X.T @ y
        self._set_params(weights)
