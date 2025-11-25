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
from training.evaluation import evaluate_model, rollout_forecast


class _DummyTransition(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.state_dim = 1
        self.num_inducing = 1

    def precompute(self, _u):
        return None

    def forward(self, x_prev, _u, _cache=None):
        # Identity dynamics with small state variance
        mean = x_prev
        var = torch.full_like(x_prev, 1e-3)
        return mean, var


class _DummyObservation(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(1, 1, bias=False)
        torch.nn.init.constant_(self.linear.weight, 1.0)

    def forward(self, x):
        return self.linear(x)


class _DummyEncoder:
    def __call__(self, y, lengths):
        # Deterministic posterior matching observations
        init_loc = y[torch.arange(y.size(0)), lengths - 1]
        init_scale = torch.zeros_like(init_loc)
        scale = torch.zeros_like(y)
        return type("EncOut", (), {
            "init_loc": init_loc,
            "init_scale": init_scale,
            "loc": y,
            "scale": scale,
            "trans_matrix": None,
            "trans_bias": None,
        })()


class _DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.state_dim = 1
        self.obs_dim = 1
        self.transition = _DummyTransition()
        self.observation = _DummyObservation()
        self.encoder = _DummyEncoder()
        self.q_u_loc = torch.nn.Parameter(torch.zeros(1))
        self.log_process_noise = torch.nn.Parameter(torch.log(torch.tensor([1e-4])))
        self.log_obs_noise = torch.nn.Parameter(torch.log(torch.tensor([1e-4])))

    def _process_noise(self):
        return self.log_process_noise.exp()

    def _obs_noise(self):
        return self.log_obs_noise.exp()


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


def test_rollout_forecast_sampling_returns_uncertainty_and_respects_lengths() -> None:
    torch.manual_seed(0)
    model = _DummyModel()
    # Two sequences, second is padded after length 2
    y_hist = torch.tensor([
        [[1.0], [2.0], [3.0]],
        [[10.0], [20.0], [0.0]],
    ])
    lengths = torch.tensor([3, 2])
    result = rollout_forecast(
        model,
        y_hist,
        lengths,
        steps=2,
        num_samples=64,
        return_std=True,
        return_samples=True,
    )
    mean = result["mean"]
    std = result["std"]
    samples = result["samples"]

    assert mean.shape == (2, 2, 1)
    assert std.shape == (2, 2, 1)
    assert samples.shape == (64, 2, 2, 1)
    # Last observed element for second sequence should seed the rollout
    assert torch.isclose(mean[1, 0, 0], torch.tensor(20.0), atol=0.5)
    assert torch.all(std > 0)


def test_rollout_forecast_samples_without_std_returns_mean_and_samples() -> None:
    torch.manual_seed(1)
    model = _DummyModel()
    y_hist = torch.tensor([[[0.5], [1.5]]])
    lengths = torch.tensor([2])
    result = rollout_forecast(
        model,
        y_hist,
        lengths,
        steps=1,
        num_samples=8,
        return_std=False,
        return_samples=True,
    )
    assert isinstance(result, dict)
    assert "mean" in result and "samples" in result
    assert "std" not in result
    assert result["mean"].shape == (1, 1, 1)
    assert result["samples"].shape == (8, 1, 1, 1)
