"""Continual-learning metrics: accuracy matrix, final average accuracy and average forgetting."""
from __future__ import annotations

import numpy as np


class AccuracyMatrix:
    """Accuracies ``acc[t][i]`` on task ``i`` after training on task ``t`` (``i <= t``)."""

    def __init__(self) -> None:
        self.rows: list[list[float]] = []

    def add(self, accuracies: list[float]) -> None:
        """Record the accuracies on tasks ``0..t`` measured after training on task ``t``."""
        if len(accuracies) != len(self.rows) + 1:
            raise ValueError(f"Expected {len(self.rows) + 1} accuracies, got {len(accuracies)}.")
        self.rows.append([float(a) for a in accuracies])

    def average_accuracy(self, t: int = -1) -> float:
        """A_t: mean accuracy over tasks ``0..t`` after training on task ``t``, in percent."""
        return float(np.mean(self.rows[t])) * 100.0

    def average_forgetting(self, t: int = -1) -> float:
        """F_t: mean over tasks ``i < t`` of ``max(0, max_{i <= j < t} acc[j][i] - acc[t][i])``, in percent."""
        t = t % len(self.rows)
        if t == 0:
            return 0.0
        forgetting = [
            max(max(self.rows[j][i] for j in range(i, t)) - self.rows[t][i], 0.0) for i in range(t)
        ]
        return float(np.mean(forgetting)) * 100.0
