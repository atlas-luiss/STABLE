"""Prompt pipeline: ViT-B/16 with one shared prefix prompt in every block, trained with STABLE.

Follows the training and evaluation procedure of DualPrompt-PyTorch
(https://github.com/JH-LEE-KR/dualprompt-pytorch, Apache-2.0) with the prompt pool, keys and
G-prompts removed. Per task: re-create the head optimizer, train the prompt with STABLE and a
linear head with Adam (cross-entropy over all classes, gradient norm clipped), fit W-NCM on one
pass over the augmented training loader, and evaluate W-NCM on the test images of every task
seen so far.
"""
from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import torch

from ..config import Config
from ..data import check_same_classes, class_subset, load_split, make_loader, prompt_transforms, task_classes
from ..engine import RunRecorder, extract_features, train_epoch
from ..heads import WNCM
from ..models.prefix_vit import build_prefix_vit
from ..optim import STABLE

log = logging.getLogger("stable")


def run(cfg: Config, run_dir: Path) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_tf, test_tf = prompt_transforms()
    train_set = load_split(cfg.dataset, cfg.data_root, True, train_tf)
    test_set = load_split(cfg.dataset, cfg.data_root, False, test_tf)
    num_classes = check_same_classes(train_set, test_set)
    tasks = task_classes(num_classes, cfg.num_tasks)
    # Shuffling and worker seeds come from the global RNG, as in DualPrompt.
    loaders = [
        (
            make_loader(class_subset(train_set, c), cfg.batch_size, cfg.num_workers, shuffle=True),
            make_loader(class_subset(test_set, c), cfg.batch_size, cfg.num_workers, shuffle=False),
        )
        for c in tasks
    ]

    model, prompt_params, head_params = build_prefix_vit(cfg.checkpoint, num_classes, cfg.prompt_length)
    model.to(device)
    log.info(f"{num_classes} classes, {cfg.num_tasks} tasks | trainable: prompt "
             f"{sum(p.numel() for p in prompt_params)}, head {sum(p.numel() for p in head_params)}")

    optimizer = STABLE(
        prompt_params,
        lr=cfg.lr,
        betas=tuple(cfg.betas),
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
        drift_budget_sq=cfg.drift_budget_sq,
        dual_lr=cfg.dual_lr,
        lambda_init=cfg.lambda_init,
        deadzone=cfg.deadzone,
        violation_memory=cfg.violation_memory,
    )
    wncm = WNCM(model.num_features, cfg.wncm_shrinkage)

    num_run = cfg.max_tasks or cfg.num_tasks
    recorder = RunRecorder(cfg, run_dir, cfg.num_tasks)
    for t in range(num_run):
        train_loader = loaders[t][0]
        head_optimizer = torch.optim.Adam(
            head_params, lr=cfg.lr, betas=tuple(cfg.betas), eps=cfg.eps, weight_decay=cfg.weight_decay
        )
        optimizer.start_task()
        if t == 1 and cfg.drift_budget_sq == "auto":
            log.info(f"Calibrated squared drift budget: {optimizer.drift_stats()[0]['drift_budget_sq']:.6f}")
        for epoch in range(cfg.epochs):
            loss = train_epoch(model, train_loader, optimizer, head_optimizer, device, cfg.grad_clip_norm)
            recorder.epoch(t, epoch, loss, optimizer)

        wncm.update(*extract_features(model, train_loader, device))
        accuracies = []
        for _, test_loader in loaders[: t + 1]:
            features, labels = extract_features(model, test_loader, device)
            accuracies.append(float((wncm.predict(features) == labels).mean()))
        recorder.task(t, accuracies, optimizer)
    recorder.finish(optimizer)
