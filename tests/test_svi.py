import math

import pyro
import torch

from data.timeseries import (
    TimeseriesWindowDataset,
    build_dataloader,
    generate_synthetic_sequences,
)
from inference.svi import SVITrainer, TrainerConfig
from models.encoder import StateEncoder
from models.gp_ssm import SparseVariationalGPSSM
from models.kernels import ARDRBFKernel
from models.transition import SparseGPTransition


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


def test_markov_variational_posterior_runs(tmp_path) -> None:
    pyro.clear_param_store()
    torch.manual_seed(1)
    kernel = ARDRBFKernel(input_dim=2)
    transition = SparseGPTransition(state_dim=2, num_inducing=4, kernel=kernel)
    encoder = StateEncoder(
        obs_dim=1,
        state_dim=2,
        hidden_size=8,
        num_layers=1,
        structured=True,
        max_transition_scale=0.6,
    )
    model = SparseVariationalGPSSM(
        transition=transition,
        encoder=encoder,
        observation=None,
        obs_dim=1,
        process_noise_init=0.1,
        obs_noise_init=0.1,
        structured_q=True,
    )
    synthetic = generate_synthetic_sequences(
        num_sequences=3,
        sequence_length=10,
        state_dim=2,
        obs_dim=1,
        observation_gain=1.0,
        process_noise=0.05,
        obs_noise=0.05,
    )
    dataset = TimeseriesWindowDataset(
        sequences=synthetic.observations,
        lengths=synthetic.lengths,
        window_length=6,
        latents=synthetic.latents,
    )
    loader = build_dataloader(dataset, batch_size=3, shuffle=False)
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
