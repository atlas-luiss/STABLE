"""STABLE: Adam with primal-dual drift control for continual learning."""
from __future__ import annotations

import math
from typing import Callable, Iterable

import numpy as np
import torch
from torch.optim import Optimizer


class STABLE(Optimizer):
    """Adam with an adaptive penalty that bounds the drift from the previous task's solution.

    STABLE solves, for each task after the first, the constrained problem

        min_theta L_t(theta)  subject to  ||theta - theta_anchor||^2 <= drift_budget_sq,

    where ``theta_anchor`` are the parameters at the start of the task. The gradient of the
    Lagrangian, ``grad L_t + 2 * lambda * (theta - theta_anchor)``, is fed to Adam, and the
    multiplier ``lambda`` is updated after every step from the measured squared drift ``g``:

    * ``g < budget * (1 - deadzone)``: multiplicative decrease,
      ``lambda <- lambda * exp(clip(dual_lr * log(g / (budget * (1 - deadzone))), -10, 0))``;
    * ``g > budget * (1 + deadzone)``: additive increase,
      ``lambda <- lambda + dual_lr * (g - budget * (1 + deadzone))``; with ``violation_memory``,
      the running mean of ``max(g - budget, 0)`` over the current task is added to the increase;
    * otherwise ``lambda`` is unchanged. ``lambda`` is clipped to ``[0, lambda_max]``.

    Call :meth:`start_task` at the beginning of every task, including the first, before the
    first :meth:`step`. It stores the anchor. Drift control is off during the first task;
    ``lambda`` is set to ``lambda_init`` at the start of the second task and carried over to the
    following tasks. Each parameter group has its own budget and multiplier.

    With ``drift_budget_sq="auto"`` the budget is calibrated from the first task: the squared
    norms of the last N updates (N = min(32, max(ceil(0.1 * T1), 12)), T1 = number of steps in
    the first task) are summed over rolling windows of 5 steps, and the budget is set to
    ``L / 5`` times the 75th percentile of these sums (taken on a log scale), where
    ``L = min(ceil(sqrt(1 / ((1 - beta1) * (1 - beta2)))), 256)`` is the Adam averaging horizon.
    If the first task has fewer than 12 steps, the budget falls back to
    :attr:`AUTO_FALLBACK_BUDGET_SQ`.

    Weight decay is decoupled and applied after the Adam step: ``theta <- theta * (1 - lr * wd)``.
    """

    AUTO_FALLBACK_BUDGET_SQ = 0.06
    CALIB_MAX_STEPS = 32
    CALIB_MIN_STEPS = 12
    CALIB_TAIL_RATIO = 0.1
    CALIB_WINDOW = 5
    CALIB_PERCENTILE = 75.0
    HORIZON_CAP = 256

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict],
        lr: float = 5e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        drift_budget_sq: float | str = "auto",
        dual_lr: float = 0.03,
        lambda_init: float = 1.0,
        deadzone: float = 0.05,
        lambda_max: float = 100.0,
        violation_memory: bool = False,
    ) -> None:
        beta1, beta2 = float(betas[0]), float(betas[1])
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not (0.0 <= beta1 < 1.0 and 0.0 <= beta2 < 1.0):
            raise ValueError(f"Invalid betas: {betas}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        auto_budget = isinstance(drift_budget_sq, str)
        if auto_budget and drift_budget_sq != "auto":
            raise ValueError(f"drift_budget_sq must be a non-negative float or 'auto', got {drift_budget_sq!r}")
        if not auto_budget and float(drift_budget_sq) < 0.0:
            raise ValueError(f"drift_budget_sq must be >= 0, got {drift_budget_sq}")
        if dual_lr < 0.0 or lambda_init < 0.0 or lambda_max <= 0.0:
            raise ValueError("dual_lr and lambda_init must be >= 0 and lambda_max > 0.")
        if not 0.0 <= deadzone < 1.0:
            raise ValueError(f"deadzone must be in [0, 1), got {deadzone}")

        defaults = dict(
            lr=lr,
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=weight_decay,
            drift_budget_sq=self.AUTO_FALLBACK_BUDGET_SQ if auto_budget else float(drift_budget_sq),
            auto_budget=auto_budget,
            dual_lr=dual_lr,
            lambda_init=lambda_init,
            deadzone=deadzone,
            lambda_max=lambda_max,
            violation_memory=bool(violation_memory),
            calibration_lr=float(lr),
            dual_lambda=0.0,
            drift_sq=0.0,
            drift_active=False,
            calibrating=False,
        )
        super().__init__(params, defaults)
        self._task_index = -1

    @classmethod
    def averaging_horizon(cls, beta1: float, beta2: float) -> int:
        """Adam averaging horizon L = min(ceil(sqrt(1 / ((1 - beta1)(1 - beta2)))), 256)."""
        return min(max(1, math.ceil(math.sqrt(1.0 / ((1.0 - beta1) * (1.0 - beta2))))), cls.HORIZON_CAP)

    @property
    def task_index(self) -> int:
        """Zero-based index of the current task (-1 before the first :meth:`start_task`)."""
        return self._task_index

    def drift_stats(self) -> list[dict[str, float]]:
        """Per parameter group: last squared drift, multiplier and squared drift budget."""
        return [
            {
                "drift_sq": float(g["drift_sq"]),
                "dual_lambda": float(g["dual_lambda"]),
                "drift_budget_sq": float(g["drift_budget_sq"]),
            }
            for g in self.param_groups
        ]

    @torch.no_grad()
    def start_task(self) -> None:
        """Mark a task boundary: finalize auto calibration, set lambda and refresh the anchors."""
        self._task_index += 1
        for group in self.param_groups:
            if self._task_index == 1 and group["auto_budget"]:
                self._calibrate_budget(group)

            group["drift_active"] = self._task_index > 0
            group["calibrating"] = False
            group["violation_sum_steps"] = 0
            group["violation_mean"] = 0.0
            if self._task_index == 0:
                group["dual_lambda"] = 0.0
                if group["auto_budget"]:
                    group["calibrating"] = True
                    group["calibration_steps"] = 0
                    group["calibration_buffer"] = []
            elif self._task_index == 1:
                group["dual_lambda"] = float(group["lambda_init"])

            for p in group["params"]:
                state = self.state[p]
                if p.requires_grad:
                    state["anchor"] = p.detach().clone(memory_format=torch.preserve_format)
                else:
                    state.pop("anchor", None)

    def _calibrate_budget(self, group: dict) -> None:
        buffer = group.get("calibration_buffer", [])
        total_steps = int(group.get("calibration_steps", 0))
        group["auto_budget"] = False
        if total_steps <= 0 or not buffer:
            return
        tail_steps = min(
            self.CALIB_MAX_STEPS,
            max(math.ceil(self.CALIB_TAIL_RATIO * total_steps), self.CALIB_MIN_STEPS, 1),
        )
        tail_steps = min(tail_steps, len(buffer))
        if tail_steps < self.CALIB_MIN_STEPS:
            group["drift_budget_sq"] = self.AUTO_FALLBACK_BUDGET_SQ
            return
        tail = np.asarray(buffer[-tail_steps:], dtype=np.float64)
        sums = np.convolve(np.maximum(tail, 0.0), np.ones(self.CALIB_WINDOW, dtype=np.float64), mode="valid")
        sums = np.maximum(sums, 1e-12)
        quantile = math.exp(float(np.percentile(np.log(sums), self.CALIB_PERCENTILE)))
        beta1, beta2 = group["betas"]
        horizon = self.averaging_horizon(beta1, beta2)
        group["drift_budget_sq"] = float(float(horizon) / float(self.CALIB_WINDOW) * quantile)

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor] | None = None):
        if self._task_index < 0:
            raise RuntimeError("Call start_task() before the first optimizer step.")
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            beta1, beta2 = group["betas"]
            eps = float(group["eps"])
            weight_decay = float(group["weight_decay"])
            budget = float(group["drift_budget_sq"])
            dual_lr = float(group["dual_lr"])
            deadzone = float(group["deadzone"])
            drift_active = bool(group["drift_active"])
            dual_lambda = max(0.0, float(group["dual_lambda"]))

            updates = []
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("STABLE does not support sparse gradients.")
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                if state.get("anchor") is None:
                    state["anchor"] = p.detach().clone(memory_format=torch.preserve_format)
                state["step"] += 1
                t = state["step"]

                grad = p.grad
                if drift_active:
                    grad = grad.add(p - state["anchor"], alpha=2.0 * dual_lambda)
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                m_hat = exp_avg / (1.0 - beta1**t)
                v_hat = exp_avg_sq / (1.0 - beta2**t)
                updates.append((p, m_hat / v_hat.sqrt().add_(eps)))

            if not updates:
                continue

            update_sq = None
            if group["calibrating"]:
                calib_lr = float(group["calibration_lr"])
                decay = 1.0 - calib_lr * weight_decay
                for p, update in updates:
                    if weight_decay != 0.0:
                        displacement = p.mul(decay - 1.0).add(update, alpha=-calib_lr * decay)
                    else:
                        displacement = update.mul(-calib_lr)
                    usq = (displacement * displacement).sum().to(torch.float64)
                    update_sq = usq if update_sq is None else update_sq + usq

            for p, update in updates:
                p.add_(update, alpha=-lr)
                if weight_decay != 0.0:
                    p.mul_(1.0 - lr * weight_decay)

            drift_sq = 0.0
            if drift_active:
                total = None
                for p in group["params"]:
                    anchor = self.state[p].get("anchor") if p in self.state else None
                    if anchor is None:
                        continue
                    delta = p - anchor
                    dsq = (delta * delta).sum().to(torch.float64)
                    total = dsq if total is None else total + dsq
                drift_sq = float(total.item()) if total is not None else 0.0
                dual_lambda = self._dual_update(group, dual_lambda, drift_sq, budget, dual_lr, deadzone)

            if group["calibrating"] and update_sq is not None:
                group["calibration_steps"] += 1
                buffer = group["calibration_buffer"]
                buffer.append(float(update_sq.item()))
                if len(buffer) > self.CALIB_MAX_STEPS:
                    del buffer[: -self.CALIB_MAX_STEPS]

            group["dual_lambda"] = dual_lambda
            group["drift_sq"] = drift_sq
        return loss

    @staticmethod
    def _dual_update(
        group: dict, dual_lambda: float, drift_sq: float, budget: float, dual_lr: float, deadzone: float
    ) -> float:
        budget_lo = budget * (1.0 - deadzone)
        budget_hi = budget * (1.0 + deadzone)
        violation_mean = 0.0
        if group["violation_memory"]:
            steps = int(group["violation_sum_steps"]) + 1
            group["violation_sum_steps"] = steps
            violation = max(drift_sq - budget, 0.0)
            violation_mean = ((steps - 1) * float(group["violation_mean"]) + violation) / steps
            group["violation_mean"] = violation_mean
        if drift_sq < budget_lo:
            ratio = max(drift_sq, 1e-12) / max(budget_lo, 1e-12)
            z = max(-10.0, min(0.0, dual_lr * math.log(ratio)))
            dual_lambda *= math.exp(z)
        elif drift_sq > budget_hi:
            if group["violation_memory"]:
                dual_lambda = max(0.0, dual_lambda + dual_lr * ((drift_sq - budget_hi) + violation_mean))
            else:
                dual_lambda = max(0.0, dual_lambda + dual_lr * (drift_sq - budget_hi))
        return min(float(group["lambda_max"]), dual_lambda)
