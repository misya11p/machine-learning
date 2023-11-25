import numpy as np
from .._base import BaseClassifire, BaseRegressor, BaseCluster, BaseTransformer


class PCA(BaseTransformer):
    def __init__(self, n_components: int = 2):
        self.n_components = n_components

    def fit(self, X: np.ndarray) -> None:
        _, s, Vh = np.linalg.svd(X)
        self.singular_values_ = s
        self.components_ = Vh[:self.n_components]

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X @ self.components_.T
