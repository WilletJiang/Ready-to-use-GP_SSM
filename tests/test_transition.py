import torch

from models.kernels import ARDRBFKernel
from models.transition import SparseGPTransition


def test_transition_predictive_shapes() -> None:
    kernel = ARDRBFKernel(input_dim=2)
    transition = SparseGPTransition(state_dim=2, num_inducing=4, kernel=kernel)
    x = torch.randn(7, 2)
    u = torch.randn(2, 4)
    mean, var = transition(x, u)
    assert mean.shape == var.shape == (7, 2)
    assert torch.all(var > 0)


def test_controlled_transition_predictive_shapes() -> None:
    kernel = ARDRBFKernel(input_dim=3)
    transition = SparseGPTransition(
        state_dim=2,
        num_inducing=4,
        kernel=kernel,
        input_dim=3,
    )
    x = torch.randn(5, 2)
    controls = torch.randn(5, 1)
    u = torch.randn(2, 4)
    mean, var = transition(x, u, controls=controls)
    assert mean.shape == var.shape == (5, 2)
    assert torch.all(var > 0)
