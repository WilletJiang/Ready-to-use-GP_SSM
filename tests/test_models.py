import math

import pyro
import pytest
import torch

from data.timeseries import (
    TimeseriesWindowDataset,
    build_dataloader,
    generate_synthetic_sequences,
    generate_system_identification_sequences,
    split_dataset,
)
from inference.svi import SVITrainer, TrainerConfig
from models.encoder import StateEncoder
from models.gp_ssm import SparseVariationalGPSSM
from models.kernels import (
    ARDRBFKernel,
    MaternKernel,
    PeriodicKernel,
    ProductKernel,
    RationalQuadraticKernel,
    SumKernel,
)
from models.transition import SparseGPTransition


@pytest.mark.parametrize(
    "kernel_ctor",
    [
        lambda dim: ARDRBFKernel(input_dim=dim),
        lambda dim: MaternKernel(input_dim=dim, nu=0.5),
        lambda dim: MaternKernel(input_dim=dim, nu=1.5),
        lambda dim: MaternKernel(input_dim=dim, nu=2.5),
        lambda dim: RationalQuadraticKernel(input_dim=dim, alpha=0.8),
        lambda dim: PeriodicKernel(input_dim=dim, period=1.2, lengthscale=0.7),
        lambda dim: SumKernel(
            kernels=[ARDRBFKernel(input_dim=dim), RationalQuadraticKernel(input_dim=dim, alpha=0.5)]
        ),
        lambda dim: ProductKernel(
            kernels=[
                ARDRBFKernel(input_dim=dim),
                PeriodicKernel(input_dim=dim, period=2.0, lengthscale=0.5),
            ]
        ),
    ],
)
def test_kernel_variants_shapes(kernel_ctor) -> None:
    kernel = kernel_ctor(3)
    x = torch.randn(5, 3)
    y = torch.randn(4, 3)
    gram = kernel(x, y)
    assert gram.shape == (5, 4)
    diag = kernel.diag(x)
    assert diag.shape == (5,)
    assert torch.all(diag > 0)
    inducing = torch.randn(6, 3)
    kzz = kernel.gram(inducing)
    assert kzz.shape == (6, 6)
    assert torch.allclose(kzz, kzz.transpose(-2, -1), atol=1e-5)


def test_transition_predictive_shapes() -> None:
    kernel = ARDRBFKernel(input_dim=2)
    transition = SparseGPTransition(state_dim=2, num_inducing=4, kernel=kernel)
    x = torch.randn(7, 2)
    u = torch.randn(2, 4)
    mean, var = transition(x, u)
    assert mean.shape == var.shape == (7, 2)
    assert torch.all(var > 0)


def test_single_svi_step_runs(tmp_path) -> None:
    pyro.clear_param_store()
    torch.manual_seed(0)
    kernel = ARDRBFKernel(input_dim=2)
    transition = SparseGPTransition(state_dim=2, num_inducing=4, kernel=kernel)
    encoder = StateEncoder(obs_dim=1, state_dim=2, hidden_size=8, num_layers=1)
    model = SparseVariationalGPSSM(
        transition=transition,
        encoder=encoder,
        observation=None,
        obs_dim=1,
        process_noise_init=0.1,
        obs_noise_init=0.1,
    )
    synthetic = generate_synthetic_sequences(
        num_sequences=4,
        sequence_length=16,
        state_dim=2,
        obs_dim=1,
        observation_gain=1.0,
        process_noise=0.05,
        obs_noise=0.05,
    )
    dataset = TimeseriesWindowDataset(
        sequences=synthetic.observations,
        lengths=synthetic.lengths,
        window_length=8,
        latents=synthetic.latents,
    )
    loader = build_dataloader(dataset, batch_size=2, shuffle=False)
    trainer = SVITrainer(
        model,
        TrainerConfig(
            steps=1,
            lr=1e-3,
            gradient_clip=5.0,
            report_every=1,
            checkpoint_dir=tmp_path,
            elbo="trace",
            eval_every=1,
        ),
    )
    batch = next(iter(loader))
    loss = trainer.svi.step(batch["y"], batch.get("lengths"))
    assert math.isfinite(loss)


def test_system_identification_dataset_split() -> None:
    dataset = generate_system_identification_sequences(
        num_sequences=16,
        sequence_length=50,
        state_dim=3,
        obs_dim=2,
        dt=0.1,
        process_noise=0.01,
        obs_noise=0.02,
        control_scale=0.2,
        seed=0,
    )
    splits = split_dataset(dataset, (0.5, 0.25, 0.25), seed=0)
    assert sum(split.observations.size(0) for split in splits.values()) == 16
    train_dataset = TimeseriesWindowDataset(
        sequences=splits["train"].observations,
        lengths=splits["train"].lengths,
        window_length=30,
        latents=splits["train"].latents,
    )
    sample = train_dataset[0]
    assert sample["y"].shape[0] == 30
    assert sample["latent"] is not None
    assert sample["latent"].shape[1] == splits["train"].latents.size(-1)
