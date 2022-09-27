from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
from collections import Counter

import numpy as np

from .knn_rs import knn_algorithm


class KNN(ABC):
    @abstractmethod
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        ...


class PythonKNN(KNN):
    def __init__(self, k: int = 3):
        self.k = k
        self._X: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> KNN:
        self._X = X
        self._y = y
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        predicted = np.zeros(X.shape[0])

        for j in range(X.shape[0]):
            closest = []
            for i in range(self._X.shape[0]):
                distance = float(np.linalg.norm(X[j] - self._X[i]))
                if len(closest) < self.k:
                    closest.append((distance, self._y[i]))
                elif distance < max(closest, key=lambda x: x[0])[0]:
                    max_position = closest.index(max(closest, key=lambda x: x[0]))
                    closest[max_position] = (distance, self._y[i])
            predicted[j] = Counter(map(lambda x: x[1], closest)).most_common(1)[0][0]
        return predicted

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)


class RustKNN(KNN):
    def __init__(self, k: int = 3):
        self.k = k

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return knn_algorithm(X, y, self.k)
