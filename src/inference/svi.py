from __future__ import annotations

import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import pyro
import torch
from pyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO
from pyro.optim import ClippedAdam
from rich.progress import Progress
from torch.distributions import constraints
from torch.utils.data import DataLoader

_REAL_CONSTRAINT_TYPE = type(constraints.real)
_LOWER_CHOLESKY_CONSTRAINT_TYPE = type(constraints.lower_cholesky)
_GREATER_THAN_CONSTRAINT_TYPE = type(constraints.greater_than(0.0))
_GREATER_THAN_EQ_CONSTRAINT_TYPE = type(constraints.greater_than_eq(0.0))
_LESS_THAN_CONSTRAINT_TYPE = type(constraints.less_than(0.0))
_INTERVAL_CONSTRAINT_TYPE = type(constraints.interval(0.0, 1.0))
_HALF_OPEN_INTERVAL_CONSTRAINT_TYPE = type(constraints.half_open_interval(0.0, 1.0))
_SIMPLEX_CONSTRAINT_TYPE = type(constraints.simplex)
_CHECKPOINT_VERSION = 2


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

    def _checkpoint_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

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
        device = self._checkpoint_device()

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

    def load_checkpoint(
        self,
        checkpoint_path: Path,
        *,
        device: Optional[torch.device] = None,
    ) -> int:
        checkpoint = _load_checkpoint_file(checkpoint_path)
        step = checkpoint.get("step", 0)
        if not isinstance(step, int) or step < 0:
            msg = "Checkpoint 'step' must be a non-negative integer"
            raise ValueError(msg)

        raw_param_state = checkpoint.get("params")
        if raw_param_state is None:
            msg = "Checkpoint is missing 'params'"
            raise ValueError(msg)

        target_device = device or self._checkpoint_device()
        param_state = _deserialize_param_store_state(
            raw_param_state,
            device=target_device,
        )
        pyro.get_param_store().set_state(param_state)
        if hasattr(self.model, "sync_from_param_store"):
            self.model.sync_from_param_store()

        optimizer_state = checkpoint.get("optimizer_state")
        if optimizer_state is not None:
            self.svi.optim.set_state(_move_tree_to_device(optimizer_state, target_device))
        rng_state = checkpoint.get("rng_state")
        if isinstance(rng_state, dict):
            _restore_rng_state(rng_state)
        return step

    def _save_checkpoint(self, target: Path, step: int) -> None:
        state = {
            "checkpoint_version": _CHECKPOINT_VERSION,
            "step": step,
            "params": _serialize_param_store_state(pyro.get_param_store().get_state()),
            "optimizer_state": _move_tree_to_cpu(self.svi.optim.get_state()),
            "rng_state": _capture_rng_state(),
        }
        torch.save(state, target)


def _load_checkpoint_file(path: Path) -> Dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu")
    except pickle.UnpicklingError:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        msg = "Checkpoint must be a dictionary"
        raise ValueError(msg)
    return checkpoint


def _move_tree_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _move_tree_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_tree_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_tree_to_cpu(item) for item in value)
    return value


def _move_tree_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_tree_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_tree_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_tree_to_device(item, device) for item in value)
    return value


def _capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "torch_cpu": torch.get_rng_state().cpu(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = _move_tree_to_cpu(torch.cuda.get_rng_state_all())
    return state


def _restore_rng_state(state: Dict[str, Any]) -> None:
    cpu_state = state.get("torch_cpu")
    if isinstance(cpu_state, torch.Tensor):
        torch.set_rng_state(cpu_state)
    python_state = state.get("python")
    if python_state is not None:
        random.setstate(python_state)
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def _serialize_param_store_state(state: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    if set(state.keys()) != {"params", "constraints"}:
        msg = f"Malformed ParamStore state keys: {sorted(state.keys())}"
        raise ValueError(msg)
    params = {
        name: tensor.detach().cpu()
        for name, tensor in state["params"].items()
    }
    constraints_state = {
        name: _serialize_constraint(constraint)
        for name, constraint in state["constraints"].items()
    }
    return {"params": params, "constraints": constraints_state}


def _deserialize_param_store_state(
    state: Dict[str, Any],
    *,
    device: torch.device,
) -> Dict[str, Dict[str, Any]]:
    if set(state.keys()) != {"params", "constraints"}:
        msg = f"Malformed ParamStore state keys: {sorted(state.keys())}"
        raise ValueError(msg)
    params = {}
    for name, tensor in state["params"].items():
        if not isinstance(tensor, torch.Tensor):
            msg = f"Checkpoint param '{name}' must be a tensor"
            raise ValueError(msg)
        params[name] = torch.nn.Parameter(tensor.to(device))

    raw_constraints = state["constraints"]
    constraints_state = {}
    for name, constraint in raw_constraints.items():
        if isinstance(constraint, dict):
            constraints_state[name] = _deserialize_constraint(constraint)
            continue
        constraints_state[name] = constraint
    return {"params": params, "constraints": constraints_state}


def _serialize_constraint(constraint_obj) -> Dict[str, Any]:
    if isinstance(constraint_obj, _REAL_CONSTRAINT_TYPE):
        return {"type": "real"}
    if isinstance(constraint_obj, _LOWER_CHOLESKY_CONSTRAINT_TYPE):
        return {"type": "lower_cholesky"}
    if isinstance(constraint_obj, _SIMPLEX_CONSTRAINT_TYPE):
        return {"type": "simplex"}
    if isinstance(constraint_obj, _GREATER_THAN_CONSTRAINT_TYPE):
        return {
            "type": "greater_than",
            "lower_bound": _move_tree_to_cpu(constraint_obj.lower_bound),
        }
    if isinstance(constraint_obj, _GREATER_THAN_EQ_CONSTRAINT_TYPE):
        return {
            "type": "greater_than_eq",
            "lower_bound": _move_tree_to_cpu(constraint_obj.lower_bound),
        }
    if isinstance(constraint_obj, _LESS_THAN_CONSTRAINT_TYPE):
        return {
            "type": "less_than",
            "upper_bound": _move_tree_to_cpu(constraint_obj.upper_bound),
        }
    if isinstance(constraint_obj, _INTERVAL_CONSTRAINT_TYPE):
        return {
            "type": "interval",
            "lower_bound": _move_tree_to_cpu(constraint_obj.lower_bound),
            "upper_bound": _move_tree_to_cpu(constraint_obj.upper_bound),
        }
    if isinstance(constraint_obj, _HALF_OPEN_INTERVAL_CONSTRAINT_TYPE):
        return {
            "type": "half_open_interval",
            "lower_bound": _move_tree_to_cpu(constraint_obj.lower_bound),
            "upper_bound": _move_tree_to_cpu(constraint_obj.upper_bound),
        }
    msg = f"Unsupported constraint type in checkpoint: {type(constraint_obj).__name__}"
    raise ValueError(msg)


def _deserialize_constraint(spec: Dict[str, Any]):
    constraint_type = spec.get("type")
    if constraint_type == "real":
        return constraints.real
    if constraint_type == "lower_cholesky":
        return constraints.lower_cholesky
    if constraint_type == "simplex":
        return constraints.simplex
    if constraint_type == "greater_than":
        return constraints.greater_than(spec["lower_bound"])
    if constraint_type == "greater_than_eq":
        return constraints.greater_than_eq(spec["lower_bound"])
    if constraint_type == "less_than":
        return constraints.less_than(spec["upper_bound"])
    if constraint_type == "interval":
        return constraints.interval(spec["lower_bound"], spec["upper_bound"])
    if constraint_type == "half_open_interval":
        return constraints.half_open_interval(
            spec["lower_bound"],
            spec["upper_bound"],
        )
    msg = f"Unsupported checkpoint constraint spec: {constraint_type}"
    raise ValueError(msg)
