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


class ElasticNet(LinearRegression):
    def __init__(
        self,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        max_iter: int = 1000,
    ):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.weights = None

    @staticmethod
    def soft_thresholding(x, threshold):
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        for _ in range(self.max_iter):
            self.weights[0] = np.mean(y - X[:, 1:] @ self.weights[1:])
            for i in range(1, n_features):
                r = y - np.delete(X, i, 1) @ np.delete(self.weights, i)
                x_i = X[:, i]
                norm = np.linalg.norm(x_i) ** 2
                norm += n_samples * self.alpha * (1 - self.l1_ratio)
                self.weights[i] = self.soft_thresholding(
                    r @ x_i / norm,
                    n_samples * self.alpha * self.l1_ratio / norm
                )


class Lasso(ElasticNet):
    def __init__(self, alpha: float = 1.0, max_iter: int = 1000):
        super().__init__(alpha=alpha, l1_ratio=1.0, max_iter=max_iter)
