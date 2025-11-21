import math

import torch

from data.timeseries import (
    TimeseriesWindowDataset,
    build_dataloader,
    generate_synthetic_sequences,
)
from models.encoder import StateEncoder
from models.gp_ssm import SparseVariationalGPSSM
from models.kernels import ARDRBFKernel
from models.transition import SparseGPTransition
from training.evaluation import evaluate_model


def test_evaluate_model_runs_with_variable_lengths() -> None:
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
        num_sequences=3,
        sequence_length=6,
        state_dim=2,
        obs_dim=1,
        observation_gain=1.0,
        process_noise=0.05,
        obs_noise=0.05,
    )
    lengths = synthetic.lengths.clone()
    lengths[1] = 4
    dataset = TimeseriesWindowDataset(
        sequences=synthetic.observations,
        lengths=lengths,
        window_length=6,
        latents=synthetic.latents,
    )
    loader = build_dataloader(dataset, batch_size=3, shuffle=False)

    metrics = evaluate_model(model, loader)

    assert set(metrics) >= {"rmse", "nll"}
    assert math.isfinite(metrics["rmse"])
    assert math.isfinite(metrics["nll"])
