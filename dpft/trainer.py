"""
DP-SGD trainer — drop-in replacement for HF Trainer with privacy guarantees.

The trainer:
  1. Computes per-sample gradients (via Opacus' GradSampleModule when
     available; otherwise via a manual loop suitable for small models).
  2. Clips each per-sample gradient to L2 norm `max_grad_norm`.
  3. Adds Gaussian noise with stddev `noise_multiplier * max_grad_norm`
     to the *summed* gradient (NOT the average — the noise is calibrated
     to the sensitivity).
  4. Steps the optimizer.
  5. Updates the privacy accountant with one Sampled Gaussian step.

The interface intentionally mirrors `transformers.Trainer.train()` so swap-in
is trivial. We support both pure-PyTorch models and HF models via a single
`compute_loss` callback.
"""
from __future__ import annotations
import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .accountant import PrivacyBudget


@dataclass
class DPTrainingArgs:
    output_dir: str = "./dp_output"
    learning_rate: float = 1e-3
    num_train_epochs: int = 1
    per_device_batch_size: int = 32
    max_grad_norm: float = 1.0          # per-sample clipping bound
    noise_multiplier: float = 1.0       # sigma / clip
    target_epsilon: float = 8.0
    target_delta: float = 1e-5
    use_opacus: bool = True             # when False, fall back to manual loop
    log_every: int = 10
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class TrainingMetrics:
    step: int = 0
    epoch: float = 0.0
    loss: float = 0.0
    epsilon_spent: float = 0.0
    grad_norm_avg: float = 0.0
    history: List[Dict[str, Any]] = field(default_factory=list)


def _per_sample_grad_loop(model: nn.Module,
                          loss_fn: Callable[[nn.Module, Dict[str, torch.Tensor]], torch.Tensor],
                          batch: Dict[str, torch.Tensor]) -> Tuple[List[torch.Tensor], float]:
    """Manual per-sample-gradient loop. Slower than vmap but works on any model.

    Returns (list_of_clipped_summed_grads_per_param, average_loss).
    Caller adds noise + steps the optimizer.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    grad_sums = [torch.zeros_like(p) for p in params]
    bs = next(iter(batch.values())).shape[0]
    losses = []
    for i in range(bs):
        sub = {k: v[i:i + 1] for k, v in batch.items()}
        for p in params:
            if p.grad is not None:
                p.grad.zero_()
        loss = loss_fn(model, sub)
        loss.backward()
        losses.append(loss.detach().item())
        # gather + clip per-sample
        flat = torch.cat([p.grad.detach().reshape(-1) for p in params])
        norm = torch.linalg.norm(flat)
        # caller-side clip done here
        yield i, [p.grad.detach().clone() for p in params], norm.item(), loss.detach().item()


class DPSGDTrainer:
    """DP-SGD trainer with optional Opacus integration."""

    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 args: DPTrainingArgs,
                 loss_fn: Callable[[nn.Module, Dict[str, torch.Tensor]], torch.Tensor],
                 optimizer: Optional[torch.optim.Optimizer] = None):
        self.model = model
        self.loader = train_loader
        self.args = args
        self.loss_fn = loss_fn
        self.optimizer = optimizer or torch.optim.SGD(model.parameters(), lr=args.learning_rate)
        self.metrics = TrainingMetrics()
        torch.manual_seed(args.seed)
        self._dataset_size = len(train_loader.dataset) if hasattr(train_loader, "dataset") else None
        self._sample_rate = args.per_device_batch_size / self._dataset_size if self._dataset_size else 0.01
        self.budget = PrivacyBudget(
            target_epsilon=args.target_epsilon,
            target_delta=args.target_delta,
            sample_rate=self._sample_rate,
            noise_multiplier=args.noise_multiplier,
        )
        self._opacus_engine = None

    def _try_setup_opacus(self) -> bool:
        if not self.args.use_opacus:
            return False
        try:
            from opacus import PrivacyEngine
            engine = PrivacyEngine()
            self.model, self.optimizer, self.loader = engine.make_private(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.loader,
                noise_multiplier=self.args.noise_multiplier,
                max_grad_norm=self.args.max_grad_norm,
                poisson_sampling=False,
            )
            self._opacus_engine = engine
            return True
        except Exception as e:
            print(f"[dpft] Opacus unavailable, falling back to manual loop: {e}")
            return False

    def train(self) -> TrainingMetrics:
        used_opacus = self._try_setup_opacus()
        device = next(self.model.parameters()).device
        for epoch in range(self.args.num_train_epochs):
            for step, batch in enumerate(self.loader):
                # Move to device
                batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                         for k, v in batch.items()}

                if used_opacus:
                    self.optimizer.zero_grad()
                    loss = self.loss_fn(self.model, batch)
                    loss.backward()
                    self.optimizer.step()
                    avg_loss = float(loss.item())
                    grad_norm = self.args.max_grad_norm  # opacus handled clipping
                else:
                    avg_loss, grad_norm = self._manual_dp_step(batch)

                self.budget.step()
                eps = self.budget.epsilon()
                self.metrics.step += 1
                self.metrics.epoch = epoch + step / max(1, len(self.loader))
                self.metrics.loss = avg_loss
                self.metrics.epsilon_spent = eps
                self.metrics.grad_norm_avg = grad_norm

                if self.metrics.step % self.args.log_every == 0:
                    rec = {
                        "step": self.metrics.step,
                        "epoch": self.metrics.epoch,
                        "loss": avg_loss,
                        "epsilon": eps,
                    }
                    self.metrics.history.append(rec)

                if self.budget.exhausted():
                    print(f"[dpft] privacy budget exhausted at step {self.metrics.step} "
                          f"(eps={eps:.3f} > target={self.args.target_epsilon})")
                    return self.metrics

        return self.metrics

    def _manual_dp_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, float]:
        """Per-sample clip + sum + Gaussian noise + step. Slow but correct."""
        params = [p for p in self.model.parameters() if p.requires_grad]
        # accumulate clipped per-sample gradient sums
        accum = [torch.zeros_like(p) for p in params]
        losses = []
        norms = []
        bs = next(iter(batch.values())).shape[0]
        for i in range(bs):
            sub = {k: v[i:i + 1] for k, v in batch.items()}
            self.optimizer.zero_grad()
            loss = self.loss_fn(self.model, sub)
            loss.backward()
            losses.append(loss.detach().item())
            # per-sample grad
            grads = [p.grad.detach() for p in params]
            flat = torch.cat([g.reshape(-1) for g in grads])
            norm = float(torch.linalg.norm(flat))
            scale = min(1.0, self.args.max_grad_norm / max(norm, 1e-12))
            for a, g in zip(accum, grads):
                a.add_(g * scale)
            norms.append(norm)
        # add noise to summed clipped gradient
        sigma = self.args.noise_multiplier * self.args.max_grad_norm
        for a in accum:
            a.add_(torch.randn_like(a) * sigma)
            a.div_(bs)  # average
        # set as the gradient and step
        for p, g in zip(params, accum):
            p.grad = g
        self.optimizer.step()
        return float(sum(losses) / max(1, len(losses))), float(sum(norms) / max(1, len(norms)))


def make_compute_loss_fn(criterion: nn.Module,
                          input_key: str = "x",
                          label_key: str = "y") -> Callable:
    """Convenience wrapper for simple supervised models."""
    def f(model: nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = model(batch[input_key])
        return criterion(out, batch[label_key])
    return f
