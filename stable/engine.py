"""Training, feature extraction and result logging shared by the two pipelines."""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from .config import Config
from .metrics import AccuracyMatrix
from .optim import STABLE

log = logging.getLogger("stable")


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: STABLE,
    head_optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_norm: float | None = None,
) -> float:
    """One epoch of cross-entropy training; returns the mean batch loss."""
    model.train()
    total = 0.0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        loss = F.cross_entropy(model(images), labels)
        optimizer.zero_grad()
        head_optimizer.zero_grad()
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        head_optimizer.step()
        value = loss.item()
        if not math.isfinite(value):
            raise FloatingPointError(f"Non-finite training loss: {value}")
        total += value
    return total / max(1, len(loader))


@torch.no_grad()
def extract_features(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Class-token features after the final norm, and labels, for every sample of ``loader``."""
    model.eval()
    features, labels = [], []
    for images, targets in loader:
        features.append(model.forward_features(images.to(device))[:, 0].cpu())
        labels.append(targets)
    return torch.cat(features).float().numpy(), torch.cat(labels).long().numpy()


class RunRecorder:
    """Logs per-epoch and per-task progress and keeps ``results.json`` up to date."""

    def __init__(self, cfg: Config, run_dir: Path, num_tasks: int) -> None:
        self.cfg = cfg
        self.path = run_dir / "results.json"
        self.num_tasks = num_tasks
        self.matrix = AccuracyMatrix()
        self.lambdas: list[float] = []

    def epoch(self, task: int, epoch: int, loss: float, optimizer: STABLE) -> None:
        s = optimizer.drift_stats()[0]
        log.info(
            f"task {task + 1}/{self.num_tasks} epoch {epoch + 1}/{self.cfg.epochs} | loss {loss:.4f} | "
            f"drift_sq {s['drift_sq']:.6f} | lambda {s['dual_lambda']:.6f} | budget_sq {s['drift_budget_sq']:.6f}"
        )

    def task(self, task: int, accuracies: list[float], optimizer: STABLE) -> None:
        self.matrix.add(accuracies)
        self.lambdas.append(optimizer.drift_stats()[0]["dual_lambda"])
        log.info(
            f"task {task + 1}/{self.num_tasks} | A_{task + 1} {self.matrix.average_accuracy():.2f}% | "
            f"F_{task + 1} {self.matrix.average_forgetting():.2f}%"
        )
        results = {
            "config": self.cfg.to_dict(),
            "tasks_completed": task + 1,
            "final_average_accuracy": self.matrix.average_accuracy(),
            "final_average_forgetting": self.matrix.average_forgetting(),
            "drift_budget_sq": optimizer.drift_stats()[0]["drift_budget_sq"],
            "accuracy_matrix": self.matrix.rows,
            "dual_lambda_end_of_task": self.lambdas,
        }
        self.path.write_text(json.dumps(results, indent=1) + "\n")

    def finish(self, optimizer: STABLE) -> None:
        log.info(f"Final average accuracy A_T: {self.matrix.average_accuracy():.2f}%")
        log.info(f"Final average forgetting F_T: {self.matrix.average_forgetting():.2f}%")
        log.info(f"Squared drift budget: {optimizer.drift_stats()[0]['drift_budget_sq']:.6f}")
        log.info(f"Results written to {self.path}")
