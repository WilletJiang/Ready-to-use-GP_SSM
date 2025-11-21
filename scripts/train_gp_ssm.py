from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import pyro
import torch
import typer
import yaml
import os
from rich.console import Console
if torch.backends.mps.is_available():
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from data.timeseries import (
    TimeseriesWindowDataset,
    build_dataloader,
    generate_system_identification_sequences,
    generate_synthetic_sequences,
    split_dataset,
)
from inference.svi import SVITrainer, TrainerConfig
from models.encoder import StateEncoder
from models.gp_ssm import SparseVariationalGPSSM
from models.kernels import (
    ARDRBFKernel,
    Kernel,
    MaternKernel,
    PeriodicKernel,
    ProductKernel,
    RationalQuadraticKernel,
    SumKernel,
)
from models.transition import SparseGPTransition
from training.evaluation import evaluate_model, rollout_forecast

app = typer.Typer(add_completion=False)
console = Console()


def _load_config(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        msg = "Configuration must be a mapping"
        raise ValueError(msg)
    return data


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    pyro.util.set_rng_seed(seed)


def _cholesky_supported(device: torch.device) -> bool:
    eye = torch.eye(2, device=device)
    try:
        torch.linalg.cholesky(eye)
        return True
    except RuntimeError:
        return False


def _select_device(console: Console) -> torch.device:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        if _cholesky_supported(device):
            return device
        console.print(
            "MPS backend lacks torch.linalg.cholesky; falling back to CPU.",
            style="bold yellow",
        )
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_kernel_from_config(config: Optional[Dict[str, Any]], state_dim: int) -> Kernel:
    cfg = config or {}
    kernel_type = cfg.get("type", "ard_rbf").lower()
    jitter = cfg.get("jitter", 1e-5)
    params = cfg.get("params", {})
    if kernel_type == "ard_rbf":
        return ARDRBFKernel(input_dim=state_dim, jitter=jitter)
    if kernel_type == "matern":
        nu = params.get("nu", 1.5)
        return MaternKernel(input_dim=state_dim, nu=nu, jitter=jitter)
    if kernel_type == "rational_quadratic":
        alpha = params.get("alpha", 1.0)
        return RationalQuadraticKernel(input_dim=state_dim, jitter=jitter, alpha=alpha)
    if kernel_type == "periodic":
        period = params.get("period", 1.0)
        lengthscale = params.get("lengthscale", 1.0)
        return PeriodicKernel(
            input_dim=state_dim,
            jitter=jitter,
            period=period,
            lengthscale=lengthscale,
        )
    if kernel_type in {"sum", "product"}:
        components_cfg = cfg.get("components")
        if not isinstance(components_cfg, list) or not components_cfg:
            msg = f"{kernel_type} kernel requires a non-empty 'components' list"
            raise ValueError(msg)
        components = [
            _build_kernel_from_config(component_cfg, state_dim)
            for component_cfg in components_cfg
        ]
        if kernel_type == "sum":
            return SumKernel(kernels=components, jitter=jitter)
        return ProductKernel(kernels=components, jitter=jitter)
    msg = f"Unknown kernel type: {kernel_type}"
    raise ValueError(msg)


@app.command()
def train(
    config: Optional[Path] = typer.Option(
        None,
        exists=True,
        dir_okay=False,
        help="Path to YAML config file. 若不提供则默认使用 configs/default.yaml。",
    ),
) -> None:
    if config is None:
        project_root = Path(__file__).resolve().parent.parent
        config = project_root / "configs" / "default.yaml"
        console.print(
            f"No --config provided, using default: {config}",
            style="bold yellow",
        )

    cfg = _load_config(config)
    trainer_cfg = cfg["trainer"]
    model_cfg = cfg["model"]
    data_cfg = cfg.get("synthetic_data", cfg.get("data", {}))
    _seed_everything(trainer_cfg["seed"])
    device = _select_device(console)
    console.print(f"Using device: {device}", style="bold green")

    kernel = _build_kernel_from_config(model_cfg.get("kernel"), model_cfg["state_dim"])
    transition = SparseGPTransition(
        state_dim=model_cfg["state_dim"],
        num_inducing=model_cfg["inducing_points"],
        kernel=kernel,
    )
    encoder = StateEncoder(
        obs_dim=model_cfg["obs_dim"],
        state_dim=model_cfg["state_dim"],
        hidden_size=model_cfg["encoder_hidden"],
        num_layers=model_cfg["encoder_layers"],
    )
    model = SparseVariationalGPSSM(
        transition=transition,
        encoder=encoder,
        observation=None,
        obs_dim=model_cfg["obs_dim"],
        process_noise_init=model_cfg["process_noise_init"],
        obs_noise_init=model_cfg["obs_noise_init"],
    )
    model.to(device)
    dataset_type = data_cfg.get("type", "toy")
    if dataset_type == "toy":
        full_dataset = generate_synthetic_sequences(
            num_sequences=data_cfg["num_sequences"],
            sequence_length=data_cfg["sequence_length"],
            state_dim=model_cfg["state_dim"],
            obs_dim=model_cfg["obs_dim"],
            observation_gain=data_cfg["observation_gain"],
            process_noise=data_cfg["process_noise"],
            obs_noise=data_cfg["obs_noise"],
        )
    elif dataset_type == "system_id":
        full_dataset = generate_system_identification_sequences(
            num_sequences=data_cfg["num_sequences"],
            sequence_length=data_cfg["sequence_length"],
            state_dim=model_cfg["state_dim"],
            obs_dim=model_cfg["obs_dim"],
            dt=data_cfg["dt"],
            process_noise=data_cfg["process_noise"],
            obs_noise=data_cfg["obs_noise"],
            control_scale=data_cfg["control_scale"],
            seed=trainer_cfg["seed"],
        )
    else:
        msg = f"Unknown dataset type: {dataset_type}"
        raise ValueError(msg)

    splits_config = data_cfg.get("splits", {"train": 0.7, "val": 0.15, "test": 0.15})
    split_tuple: Tuple[float, float, float] = (
        splits_config.get("train", 0.7),
        splits_config.get("val", 0.15),
        splits_config.get("test", 0.15),
    )
    splits = split_dataset(full_dataset, split_tuple, seed=trainer_cfg["seed"])

    train_dataset = TimeseriesWindowDataset(
        sequences=splits["train"].observations,
        lengths=splits["train"].lengths,
        window_length=trainer_cfg["window_length"],
        latents=splits["train"].latents,
    )
    train_loader = build_dataloader(
        train_dataset,
        batch_size=trainer_cfg["batch_size"],
        shuffle=True,
    )
    eval_window = data_cfg.get("eval_window_length") or data_cfg["sequence_length"]
    eval_generator = torch.Generator().manual_seed(trainer_cfg["seed"])
    val_dataset = TimeseriesWindowDataset(
        sequences=splits["val"].observations,
        lengths=splits["val"].lengths,
        window_length=eval_window,
        latents=splits["val"].latents,
        generator=eval_generator,
    )
    test_generator = torch.Generator().manual_seed(trainer_cfg["seed"])
    test_dataset = TimeseriesWindowDataset(
        sequences=splits["test"].observations,
        lengths=splits["test"].lengths,
        window_length=eval_window,
        latents=splits["test"].latents,
        generator=test_generator,
    )
    val_loader = build_dataloader(val_dataset, batch_size=trainer_cfg["batch_size"], shuffle=False)
    test_loader = build_dataloader(test_dataset, batch_size=trainer_cfg["batch_size"], shuffle=False)
    trainer = SVITrainer(
        model,
        TrainerConfig(
            steps=trainer_cfg["steps"],
            lr=trainer_cfg["lr"],
            gradient_clip=trainer_cfg["gradient_clip"],
            report_every=trainer_cfg["report_every"],
            checkpoint_dir=Path(trainer_cfg["checkpoint_dir"]),
            elbo=trainer_cfg["elbo"],
            eval_every=trainer_cfg["eval_every"],
        ),
    )
    console.print("Launching training", style="bold green")
    trainer.fit(train_loader, eval_loader=val_loader, eval_fn=evaluate_model)
    console.print("Evaluating on validation split", style="bold yellow")
    val_metrics = evaluate_model(model, val_loader)
    console.print(val_metrics)
    console.print("Evaluating on test split", style="bold yellow")
    test_metrics = evaluate_model(model, test_loader)
    console.print(test_metrics)
    if splits["test"].observations.size(0) > 0:
        sample = splits["test"].observations[:1].to(device)
        sample_lengths = splits["test"].lengths[:1].to(device)
        context = max(1, min(trainer_cfg["window_length"], sample.size(1) // 2))
        forecast_horizon = data_cfg.get("forecast_horizon", 16)
        context_tensor = sample[:, :context, :]
        context_lengths = torch.full_like(sample_lengths, context)
        preds = rollout_forecast(model, context_tensor, context_lengths, forecast_horizon)
        target = sample[:, context : context + forecast_horizon, :]
        if target.size(1) < forecast_horizon:
            pad = torch.zeros(
                target.size(0),
                forecast_horizon - target.size(1),
                target.size(2),
                device=target.device,
            )
            target = torch.cat([target, pad], dim=1)
        horizon_mask = (
            torch.arange(forecast_horizon, device=preds.device).unsqueeze(0)
            < (sample_lengths - context).unsqueeze(1).to(preds.device)
        )
        mse = ((preds - target) ** 2 * horizon_mask.unsqueeze(-1)).sum()
        denom = horizon_mask.sum().clamp_min(1) * model.obs_dim
        forecast_rmse = torch.sqrt(mse / denom)
        console.print(
            {
                "forecast_context": int(context),
                "forecast_horizon": int(forecast_horizon),
                "forecast_rmse": float(forecast_rmse.item()),
            }
        )
    console.print("Finished training", style="bold blue")


if __name__ == "__main__":
    typer.run(train)
