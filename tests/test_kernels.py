import torch

from models.kernels import (
    ARDRBFKernel,
    MaternKernel,
    PeriodicKernel,
    ProductKernel,
    RationalQuadraticKernel,
    SumKernel,
)


def test_kernel_variants_shapes() -> None:
    constructors = [
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
    ]
    for ctor in constructors:
        kernel = ctor(3)
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


def test_sum_kernel_gram_adds_single_jitter() -> None:
    torch.manual_seed(0)
    inducing = torch.randn(4, 2)
    k1 = ARDRBFKernel(input_dim=2, jitter=1e-4)
    k2 = RationalQuadraticKernel(input_dim=2, jitter=2e-4)
    combined = SumKernel([k1, k2], jitter=5e-5)
    kzz = combined.gram(inducing)
    raw = k1(inducing, inducing) + k2(inducing, inducing)
    eye = torch.eye(inducing.size(0), device=inducing.device, dtype=inducing.dtype)
    expected = raw + combined.jitter * eye
    assert torch.allclose(kzz, expected, atol=1e-6)


def test_product_kernel_gram_adds_single_jitter() -> None:
    torch.manual_seed(0)
    inducing = torch.randn(3, 2)
    k1 = ARDRBFKernel(input_dim=2, jitter=1e-4)
    k2 = PeriodicKernel(input_dim=2, jitter=2e-4, period=1.3, lengthscale=0.8)
    combined = ProductKernel([k1, k2], jitter=5e-5)
    kzz = combined.gram(inducing)
    raw = k1(inducing, inducing) * k2(inducing, inducing)
    eye = torch.eye(inducing.size(0), device=inducing.device, dtype=inducing.dtype)
    expected = raw + combined.jitter * eye
    assert torch.allclose(kzz, expected, atol=1e-6)
