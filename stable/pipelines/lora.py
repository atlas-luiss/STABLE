"""LoRA pipeline: ViT-B/16 with LoRA (rank 8) on the attention projections, trained with STABLE.

Per task: train LoRA with STABLE and a linear head with Adam (cross-entropy over all classes),
fit W-NCM on the task's training images with the evaluation transform, and evaluate W-NCM on the
test images of all classes seen so far.
"""
from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import torch

from ..config import Config
from ..data import check_same_classes, class_subset, load_split, lora_transforms, make_loader, task_classes
from ..engine import RunRecorder, extract_features, train_epoch
from ..heads import WNCM
from ..models.lora import build_lora_vit
from ..optim import STABLE

log = logging.getLogger("stable")


def run(cfg: Config, run_dir: Path) -> None:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_tf, test_tf = lora_transforms(cfg.dataset)
    train_set = load_split(cfg.dataset, cfg.data_root, True, train_tf)
    train_eval_set = load_split(cfg.dataset, cfg.data_root, True, test_tf)
    test_set = load_split(cfg.dataset, cfg.data_root, False, test_tf)
    num_classes = check_same_classes(train_set, test_set)
    tasks = task_classes(num_classes, cfg.num_tasks)

    model, lora_params, head_params = build_lora_vit(cfg.checkpoint, num_classes, cfg.lora_rank, cfg.lora_alpha)
    model.to(device)
    log.info(f"{num_classes} classes, {cfg.num_tasks} tasks | trainable: LoRA "
             f"{sum(p.numel() for p in lora_params)}, head {sum(p.numel() for p in head_params)}")

    optimizer = STABLE(
        lora_params,
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
    head_optimizer = torch.optim.Adam(
        head_params, lr=cfg.lr, betas=tuple(cfg.betas), eps=cfg.eps, weight_decay=cfg.weight_decay
    )
    wncm = WNCM(model.num_features, cfg.wncm_shrinkage)
    generator = torch.Generator()
    generator.manual_seed(cfg.seed)

    num_run = cfg.max_tasks or cfg.num_tasks
    recorder = RunRecorder(cfg, run_dir, cfg.num_tasks)
    for t in range(num_run):
        train_loader = make_loader(
            class_subset(train_set, tasks[t]), cfg.batch_size, cfg.num_workers, shuffle=True, generator=generator
        )
        optimizer.start_task()
        if t == 1 and cfg.drift_budget_sq == "auto":
            log.info(f"Calibrated squared drift budget: {optimizer.drift_stats()[0]['drift_budget_sq']:.6f}")
        for epoch in range(cfg.epochs):
            loss = train_epoch(model, train_loader, optimizer, head_optimizer, device)
            recorder.epoch(t, epoch, loss, optimizer)

        train_eval_loader = make_loader(
            class_subset(train_eval_set, tasks[t]), cfg.batch_size, cfg.num_workers, shuffle=False
        )
        wncm.update(*extract_features(model, train_eval_loader, device))

        seen = [c for task in tasks[: t + 1] for c in task]
        test_loader = make_loader(class_subset(test_set, seen), cfg.batch_size, cfg.num_workers, shuffle=False)
        features, labels = extract_features(model, test_loader, device)
        predictions = wncm.predict(features)
        accuracies = []
        for classes in tasks[: t + 1]:
            mask = np.isin(labels, classes)
            accuracies.append(float((predictions[mask] == labels[mask]).mean()))
        recorder.task(t, accuracies, optimizer)
    recorder.finish(optimizer)
