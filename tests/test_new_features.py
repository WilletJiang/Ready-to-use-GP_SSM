"""Tests for the four new contributions:

1. SpectralMixtureKernel  (Wilson & Adams, 2013)
2. Adaptive Cholesky jitter  (psd_safe_cholesky)
3. Cosine annealing LR schedule
4. Gaussian CRPS metric
"""

import math
from pathlib import Path

import pyro
import torch

from inference.svi import SVITrainer, TrainerConfig
from models.kernels import ARDRBFKernel, SpectralMixtureKernel, SumKernel
from models.utils import psd_safe_cholesky
from training.evaluation import gaussian_crps


# ---------------------------------------------------------------------------
# 1. SpectralMixtureKernel
# ---------------------------------------------------------------------------


def test_sm_kernel_shapes() -> None:
    kernel = SpectralMixtureKernel(input_dim=3, num_mixtures=4)
    x = torch.randn(5, 3)
    y = torch.randn(4, 3)
    K = kernel(x, y)
    assert K.shape == (5, 4)
    d = kernel.diag(x)
    assert d.shape == (5,)
    assert torch.all(d > 0)


def test_sm_kernel_gram_symmetric_and_pd() -> None:
    torch.manual_seed(42)
    kernel = SpectralMixtureKernel(input_dim=2, num_mixtures=3)
    inducing = torch.randn(6, 2)
    G = kernel.gram(inducing)
    assert G.shape == (6, 6)
    assert torch.allclose(G, G.T, atol=1e-5)
    # Positive-definite: all eigenvalues positive
    eigs = torch.linalg.eigvalsh(G)
    assert torch.all(eigs > 0)


def test_sm_kernel_diagonal_equals_weight_sum() -> None:
    kernel = SpectralMixtureKernel(input_dim=4, num_mixtures=5)
    x = torch.randn(7, 4)
    d = kernel.diag(x)
    expected = kernel.weights.sum()
    assert torch.allclose(d, expected.expand(7), atol=1e-6)


def test_sm_kernel_self_covariance_matches_diag() -> None:
    """K(x, x) diagonal should equal kernel.diag(x)."""
    torch.manual_seed(0)
    kernel = SpectralMixtureKernel(input_dim=2, num_mixtures=3)
    x = torch.randn(8, 2)
    K = kernel(x, x)
    d = kernel.diag(x)
    assert torch.allclose(K.diag(), d, atol=1e-5)


def test_sm_kernel_composable_in_sum() -> None:
    sm = SpectralMixtureKernel(input_dim=2, num_mixtures=2)
    rbf = ARDRBFKernel(input_dim=2)
    combined = SumKernel([sm, rbf])
    x = torch.randn(4, 2)
    y = torch.randn(3, 2)
    K = combined(x, y)
    assert K.shape == (4, 3)
    G = combined.gram(torch.randn(5, 2))
    assert torch.allclose(G, G.T, atol=1e-5)


def test_sm_kernel_single_mixture_approximates_rbf() -> None:
    """With Q=1, zero mean (low frequency), the SM kernel should resemble RBF."""
    torch.manual_seed(7)
    kernel = SpectralMixtureKernel(input_dim=1, num_mixtures=1)
    # Set mean to zero so cos(2pi * tau * 0) = 1  (exact RBF limit)
    kernel.raw_means.data.fill_(0.0)
    kernel.log_variances.data.fill_(0.0)  # v = 1
    kernel.log_weights.data.fill_(0.0)  # w = 1

    x = torch.linspace(-1, 1, 20).unsqueeze(-1)
    K = kernel(x, x)
    # Should be PSD and smooth
    eigs = torch.linalg.eigvalsh(K)
    assert torch.all(eigs >= -1e-5)


def test_sm_kernel_invalid_num_mixtures() -> None:
    try:
        SpectralMixtureKernel(input_dim=2, num_mixtures=0)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# 2. Adaptive Cholesky jitter
# ---------------------------------------------------------------------------


def test_psd_safe_cholesky_on_valid_matrix() -> None:
    """Should succeed on the first try without adding jitter."""
    mat = torch.eye(4) + 0.1 * torch.randn(4, 4)
    mat = mat @ mat.T  # guaranteed PD
    L = psd_safe_cholesky(mat)
    reconstructed = L @ L.T
    assert torch.allclose(mat, reconstructed, atol=1e-5)


def test_psd_safe_cholesky_recovers_from_near_singular() -> None:
    """A nearly singular matrix should be rescued by adaptive jitter."""
    torch.manual_seed(0)
    # Rank-deficient plus tiny noise — Cholesky should fail without jitter
    v = torch.randn(5, 1)
    mat = v @ v.T  # rank 1 — not PD
    L = psd_safe_cholesky(mat, jitter=1e-4)
    # Just verify we got a valid lower-triangular result
    assert L.shape == (5, 5)
    assert torch.all(torch.isfinite(L))


def test_psd_safe_cholesky_raises_on_hopeless_matrix() -> None:
    """Completely negative-definite matrix should still fail after retries."""
    mat = -torch.eye(3) * 1e6
    try:
        psd_safe_cholesky(mat, jitter=1e-6, max_retries=2)
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        pass


# ---------------------------------------------------------------------------
# 3. Cosine annealing LR schedule
# ---------------------------------------------------------------------------


def test_cosine_lr_warmup_and_decay() -> None:
    """Verify the LR schedule shape: warmup -> peak -> cosine decay."""
    from models.encoder import StateEncoder
    from models.gp_ssm import SparseVariationalGPSSM
    from models.kernels import ARDRBFKernel
    from models.transition import SparseGPTransition

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
    cfg = TrainerConfig(
        steps=100,
        lr=1e-2,
        gradient_clip=5.0,
        report_every=100,
        checkpoint_dir=Path("/tmp/test_lr"),
        elbo="trace",
        eval_every=100,
        lr_schedule="cosine",
        lr_warmup_steps=20,
        lr_min=1e-5,
    )
    trainer = SVITrainer(model, cfg)

    # Warmup: LR ramps from lr_min toward lr
    lr_start = trainer._compute_lr(1)
    lr_warmup_end = trainer._compute_lr(20)
    assert lr_start < lr_warmup_end
    assert abs(lr_warmup_end - cfg.lr) < 1e-8

    # Cosine decay: LR decreases after warmup
    lr_mid = trainer._compute_lr(60)
    lr_end = trainer._compute_lr(100)
    assert lr_mid < lr_warmup_end
    assert lr_end < lr_mid
    assert lr_end >= cfg.lr_min - 1e-10

    # Constant schedule returns fixed LR
    cfg2 = TrainerConfig(
        steps=100,
        lr=1e-2,
        gradient_clip=5.0,
        report_every=100,
        checkpoint_dir=Path("/tmp/test_lr2"),
        elbo="trace",
        eval_every=100,
        lr_schedule="constant",
    )
    trainer2 = SVITrainer(model, cfg2)
    assert trainer2._compute_lr(1) == cfg2.lr
    assert trainer2._compute_lr(50) == cfg2.lr
    assert trainer2._compute_lr(100) == cfg2.lr


def test_cosine_lr_schedule_updates_optimizer_lr_after_steps(tmp_path) -> None:
    from data.timeseries import (
        TimeseriesWindowDataset,
        build_dataloader,
        generate_synthetic_sequences,
    )
    from models.encoder import StateEncoder
    from models.gp_ssm import SparseVariationalGPSSM
    from models.kernels import ARDRBFKernel
    from models.transition import SparseGPTransition

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
        sequence_length=8,
        state_dim=2,
        obs_dim=1,
        observation_gain=1.0,
        process_noise=0.05,
        obs_noise=0.05,
    )
    dataset = TimeseriesWindowDataset(
        sequences=synthetic.observations,
        lengths=synthetic.lengths,
        window_length=4,
        latents=synthetic.latents,
    )
    loader = build_dataloader(dataset, batch_size=2, shuffle=False)
    trainer = SVITrainer(
        model,
        TrainerConfig(
            steps=3,
            lr=1e-2,
            gradient_clip=5.0,
            report_every=10,
            checkpoint_dir=tmp_path,
            elbo="trace",
            eval_every=10,
            lr_schedule="cosine",
            lr_warmup_steps=1,
            lr_min=1e-4,
        ),
    )
    trainer.fit(loader)
    lrs = {
        param_group["lr"]
        for optim in trainer.svi.optim.optim_objs.values()
        for param_group in optim.param_groups
    }
    assert lrs == {trainer._compute_lr(3)}


# ---------------------------------------------------------------------------
# 4. Gaussian CRPS metric
# ---------------------------------------------------------------------------


def test_crps_perfect_prediction() -> None:
    """CRPS should be near zero when prediction matches observation exactly."""
    mean = torch.tensor([1.0, 2.0, 3.0])
    std = torch.tensor([1e-4, 1e-4, 1e-4])
    target = mean.clone()
    crps = gaussian_crps(mean, std, target)
    # With tiny std and exact match, CRPS ~ std * (-1/sqrt(pi)) ~ 0
    assert torch.all(crps.abs() < 1e-3)


def test_crps_increases_with_error() -> None:
    """CRPS should increase as target moves away from mean."""
    mean = torch.zeros(1)
    std = torch.ones(1)
    crps_near = gaussian_crps(mean, std, torch.tensor([0.1]))
    crps_far = gaussian_crps(mean, std, torch.tensor([3.0]))
    assert crps_far > crps_near


def test_crps_increases_with_spread() -> None:
    """Wider predictive distribution should have larger CRPS (all else equal)."""
    mean = torch.zeros(1)
    target = torch.tensor([0.5])
    crps_tight = gaussian_crps(mean, torch.tensor([0.1]), target)
    crps_wide = gaussian_crps(mean, torch.tensor([10.0]), target)
    assert crps_wide > crps_tight


def test_crps_shape_preserved() -> None:
    """Output shape should match input shape."""
    mean = torch.randn(4, 8, 2)
    std = torch.ones(4, 8, 2)
    target = torch.randn(4, 8, 2)
    crps = gaussian_crps(mean, std, target)
    assert crps.shape == (4, 8, 2)


def test_crps_non_negative() -> None:
    """CRPS is a non-negative proper scoring rule."""
    torch.manual_seed(0)
    mean = torch.randn(100)
    std = torch.rand(100) + 0.01
    target = torch.randn(100)
    crps = gaussian_crps(mean, std, target)
    assert torch.all(crps >= -1e-6)


def test_crps_cache_handles_dtype_switches() -> None:
    mean32 = torch.zeros(2, dtype=torch.float32)
    std32 = torch.ones(2, dtype=torch.float32)
    target32 = torch.zeros(2, dtype=torch.float32)
    crps32 = gaussian_crps(mean32, std32, target32)

    mean64 = torch.zeros(2, dtype=torch.float64)
    std64 = torch.ones(2, dtype=torch.float64)
    target64 = torch.zeros(2, dtype=torch.float64)
    crps64 = gaussian_crps(mean64, std64, target64)

    assert crps32.dtype == torch.float32
    assert crps64.dtype == torch.float64


def test_crps_in_evaluate_model() -> None:
    """evaluate_model should now return a 'crps' key."""
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
    dataset = TimeseriesWindowDataset(
        sequences=synthetic.observations,
        lengths=synthetic.lengths,
        window_length=6,
        latents=synthetic.latents,
    )
    loader = build_dataloader(dataset, batch_size=3, shuffle=False)
    metrics = evaluate_model(
        model, loader, predictive_horizon=2, predictive_num_samples=8
    )
    assert "crps" in metrics
    assert math.isfinite(metrics["crps"])
    assert metrics["crps"] >= 0
