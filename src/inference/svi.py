from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import pyro
import torch
from pyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO
from pyro.optim import ClippedAdam
from rich.progress import Progress
from torch.utils.data import DataLoader


@dataclass
class TrainerConfig:
    steps: int
    lr: float
    gradient_clip: float
    report_every: int
    checkpoint_dir: Path
    elbo: str
    eval_every: int
    lr_schedule: str = "constant"
    lr_warmup_steps: int = 0
    lr_min: float = 1e-6


class SVITrainer:
    def __init__(self, model, config: TrainerConfig) -> None:
        if config.steps <= 0 or config.lr <= 0 or config.eval_every <= 0:
            msg = "Training hyperparameters must be positive"
            raise ValueError(msg)
        self.config = config
        elbo = self._make_elbo(config.elbo)
        optim = ClippedAdam({"lr": config.lr, "clip_norm": config.gradient_clip})
        self.svi = SVI(model.model, model.guide, optim, elbo)
        self.model = model

    def _make_elbo(self, elbo_name: str):
        if elbo_name == "trace":
            return Trace_ELBO()
        if elbo_name == "meanfield":
            return TraceMeanField_ELBO()
        msg = f"Unknown ELBO type: {elbo_name}"
        raise ValueError(msg)

    def _compute_lr(self, offset: int) -> float:
        """Return learning rate for the given *offset* (1-based within run)."""
        cfg = self.config
        if cfg.lr_schedule == "constant":
            return cfg.lr
        if cfg.lr_schedule != "cosine":
            msg = f"Unknown lr_schedule: {cfg.lr_schedule}"
            raise ValueError(msg)
        warmup = cfg.lr_warmup_steps
        if offset <= warmup:
            return cfg.lr_min + (cfg.lr - cfg.lr_min) * offset / max(warmup, 1)
        progress = (offset - warmup) / max(cfg.steps - warmup, 1)
        return cfg.lr_min + 0.5 * (cfg.lr - cfg.lr_min) * (1 + math.cos(math.pi * progress))

    def _set_lr(self, lr: float) -> None:
        for opt in self.svi.optim.optim_objs.values():
            for pg in opt.param_groups:
                pg["lr"] = lr

    def _batch_stream(self, loader: DataLoader) -> Iterable[Dict[str, torch.Tensor]]:
        while True:
            for batch in loader:
                yield batch

    def fit(
        self,
        loader: DataLoader,
        eval_loader: Optional[DataLoader] = None,
        eval_fn: Optional[Callable[[object, DataLoader], Dict[str, float]]] = None,
        start_step: int = 0,
    ) -> List[Dict[str, float]]:
        path = self.config.checkpoint_dir
        path.mkdir(parents=True, exist_ok=True)
        iterator = self._batch_stream(loader)
        progress = Progress()
        task = progress.add_task("svi", total=start_step + self.config.steps)
        progress.update(task, completed=start_step)
        eval_history: List[Dict[str, float]] = []
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        with progress:
            for offset in range(1, self.config.steps + 1):
                step = start_step + offset
                batch = next(iterator)
                y = batch["y"].to(device)
                lengths = batch.get("lengths")
                controls = batch.get("controls")
                if lengths is not None:
                    lengths = lengths.to(device)
                if controls is not None:
                    controls = controls.to(device)

                loss = self.svi.step(y, lengths, controls)
                if self.config.lr_schedule != "constant":
                    self._set_lr(self._compute_lr(offset))
                progress.advance(task)
                if step % self.config.report_every == 0:
                    progress.log(f"step={step} loss={loss:.2f}")
                if (
                    eval_loader is not None
                    and eval_fn is not None
                    and step % self.config.eval_every == 0
                ):
                    metrics = eval_fn(self.model, eval_loader)
                    metrics_with_step = {"step": step, **metrics}
                    eval_history.append(metrics_with_step)
                    metrics_str = ", ".join(
                        f"{key}={value:.4f}" for key, value in metrics.items()
                    )
                    progress.log(f"eval@{step}: {metrics_str}")
                if offset == self.config.steps:
                    self._save_checkpoint(path / "final.pt", step)
        return eval_history

    def _save_checkpoint(self, target: Path, step: int) -> None:
        state = {
            "step": step,
            "params": pyro.get_param_store().get_state(),
        }
        torch.save(state, target)
