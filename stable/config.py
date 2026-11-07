"""Run configuration: a YAML file plus command-line overrides."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DATASET_DIRS = {"imagenet_r": "imagenet-r", "cifar100": "cifar100", "imagenet_a": "imagenet-a-top100"}


@dataclass
class Config:
    """All settings of one run. Every key of a config file must be a field of this class."""

    method: str
    dataset: str
    num_tasks: int
    epochs: int
    batch_size: int
    drift_budget_sq: float | str
    violation_memory: bool
    num_workers: int
    data_dir: str = "data"
    checkpoint: str = "checkpoints/vit_base_patch16_224.augreg_in21k.safetensors"
    output_dir: str = "outputs"
    seed: int = 42
    max_tasks: int | None = None
    lr: float = 5e-4
    weight_decay: float = 1e-4
    betas: list[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8
    dual_lr: float = 0.03
    lambda_init: float = 1.0
    deadzone: float = 0.05
    wncm_shrinkage: float = 0.07
    lora_rank: int = 8
    lora_alpha: float = 16.0
    prompt_length: int = 20
    grad_clip_norm: float | None = None

    def __post_init__(self) -> None:
        if self.method not in {"lora", "prompts"}:
            raise ValueError(f"method must be 'lora' or 'prompts', got {self.method!r}")
        if self.dataset not in DATASET_DIRS:
            raise ValueError(f"dataset must be one of {sorted(DATASET_DIRS)}, got {self.dataset!r}")
        if self.drift_budget_sq != "auto":
            self.drift_budget_sq = float(self.drift_budget_sq)
        for name in ("lr", "weight_decay", "eps", "dual_lr", "lambda_init", "deadzone", "wncm_shrinkage", "lora_alpha"):
            setattr(self, name, float(getattr(self, name)))
        self.betas = [float(b) for b in self.betas]
        if self.grad_clip_norm is not None:
            self.grad_clip_norm = float(self.grad_clip_norm)
        if self.max_tasks is not None and not 1 <= self.max_tasks <= self.num_tasks:
            raise ValueError(f"max_tasks must be in [1, {self.num_tasks}], got {self.max_tasks}")

    @property
    def data_root(self) -> Path:
        return Path(self.data_dir).expanduser() / DATASET_DIRS[self.dataset]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    """Read a YAML config and apply ``overrides`` (key -> value) on top of it."""
    with open(path) as f:
        values = yaml.safe_load(f) or {}
    values.update(overrides or {})
    known = {f.name for f in dataclasses.fields(Config)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")
    return Config(**values)


def parse_overrides(items: list[str]) -> dict[str, Any]:
    """Parse ``key=value`` strings; values are read as YAML (numbers, booleans, null, lists)."""
    overrides = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise ValueError(f"Overrides must look like key=value, got {item!r}")
        overrides[key.strip()] = yaml.safe_load(value)
    return overrides
