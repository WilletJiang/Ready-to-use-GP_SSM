from __future__ import annotations

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


class SVITrainer:
    def __init__(self, model, config: TrainerConfig) -> None:
        if config.steps <= 0 or config.lr <= 0 or config.eval_every <= 0:
            msg = "Training hyperparameters must be positive"
            raise ValueError(msg)
        self.config = config
        pyro.clear_param_store()
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

    def _batch_stream(self, loader: DataLoader) -> Iterable[Dict[str, torch.Tensor]]:
        while True:
            for batch in loader:
                yield batch

    def fit(
        self,
        loader: DataLoader,
        eval_loader: Optional[DataLoader] = None,
        eval_fn: Optional[Callable[[object, DataLoader], Dict[str, float]]] = None,
    ) -> List[Dict[str, float]]:
        path = self.config.checkpoint_dir
        path.mkdir(parents=True, exist_ok=True)
        iterator = self._batch_stream(loader)
        progress = Progress()
        task = progress.add_task("svi", total=self.config.steps)
        eval_history: List[Dict[str, float]] = []
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        with progress:
            for step in range(1, self.config.steps + 1):
                batch = next(iterator)
                y = batch["y"].to(device)
                lengths = batch.get("lengths")
                if lengths is not None:
                    lengths = lengths.to(device)

                loss = self.svi.step(y, lengths)
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
                if step == self.config.steps:
                    self._save_checkpoint(path / "final.pt", step)
        return eval_history

    def _save_checkpoint(self, target: Path, step: int) -> None:
        state = {
            "step": step,
            "params": pyro.get_param_store().get_state(),
        }
        torch.save(state, target)
