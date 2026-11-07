"""W-NCM: whitened nearest-class-mean classifier on frozen features."""
from __future__ import annotations

import numpy as np


class WNCM:
    """Cosine nearest-class-mean classifier in a shrinkage-whitened feature space.

    The head keeps running class means and a running covariance of all features seen so far.
    The covariance is shrunk towards its diagonal,
    ``Sigma = (1 - shrinkage) * Sigma + shrinkage * diag(Sigma)``, and features and class means
    are centered on the global mean and whitened with ``Sigma^{-1/2}`` before cosine scoring.
    It is refitted after every task from the features of that task only, so it stores no
    examples.
    """

    def __init__(self, feature_dim: int, shrinkage: float = 0.07) -> None:
        if not 0.0 <= shrinkage <= 1.0:
            raise ValueError(f"shrinkage must be in [0, 1], got {shrinkage}")
        self.feature_dim = int(feature_dim)
        self.shrinkage = float(shrinkage)
        self.n_samples = 0
        self.mean = np.zeros(self.feature_dim, dtype=np.float64)
        self.scatter = np.zeros((self.feature_dim, self.feature_dim), dtype=np.float64)
        self.cov = np.eye(self.feature_dim, dtype=np.float64)
        self.class_means: dict[int, np.ndarray] = {}
        self.class_counts: dict[int, int] = {}
        self.classes: list[int] = []
        self._whitening: np.ndarray | None = None

    def update(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Add the features of one task to the global and per-class statistics."""
        x = np.asarray(features).astype(np.float64, copy=False)
        y = np.asarray(labels).astype(np.int64, copy=False)
        self._update_covariance(x)
        for class_id in np.unique(y):
            c = int(class_id)
            xc = x[y == c]
            if xc.shape[0] == 0:
                continue
            if c not in self.class_means:
                self.class_means[c] = xc.mean(axis=0)
                self.class_counts[c] = int(xc.shape[0])
                self.classes.append(c)
                continue
            n_old = self.class_counts[c]
            n_new = n_old + int(xc.shape[0])
            self.class_means[c] = (self.class_means[c] * n_old + xc.sum(axis=0)) / float(n_new)
            self.class_counts[c] = n_new

    def _update_covariance(self, x: np.ndarray) -> None:
        n_new = int(x.shape[0])
        if n_new == 0:
            return
        n_old = self.n_samples
        if n_old == 0:
            self.mean = x.mean(axis=0)
            centered = x - self.mean
            self.scatter = centered.T @ centered
        else:
            old_mean = self.mean.copy()
            self.mean = (old_mean * n_old + x.sum(axis=0)) / float(n_old + n_new)
            self.scatter += (x - old_mean).T @ (x - self.mean)
        self.n_samples = n_old + n_new
        if self.n_samples > 1:
            cov = self.scatter / float(self.n_samples - 1)
        else:
            cov = np.eye(self.feature_dim, dtype=np.float64)
        if self.shrinkage > 0.0:
            cov = (1.0 - self.shrinkage) * cov + self.shrinkage * np.diag(np.diag(cov))
        self.cov = cov
        self._whitening = None

    def _whiten(self, x: np.ndarray) -> np.ndarray:
        if self._whitening is None:
            try:
                eigvals, eigvecs = np.linalg.eigh(self.cov)
                eigvals = np.maximum(eigvals, 1e-12)
                self._whitening = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            except np.linalg.LinAlgError:
                self._whitening = np.eye(self.feature_dim, dtype=np.float64)
        return (x - self.mean) @ self._whitening

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return the predicted class id of every row of ``features``."""
        x = np.asarray(features).astype(np.float64, copy=False)
        if not self.classes:
            return np.zeros(x.shape[0], dtype=np.int64)
        xw = self._whiten(x)
        pw = self._whiten(np.vstack([self.class_means[c] for c in self.classes]))
        xw = xw / (np.linalg.norm(xw, axis=1, keepdims=True) + 1e-8)
        pw = pw / (np.linalg.norm(pw, axis=1, keepdims=True) + 1e-8)
        scores = xw @ pw.T
        return np.asarray(self.classes, dtype=np.int64)[np.argmax(scores, axis=1)]
