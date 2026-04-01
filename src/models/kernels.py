from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn

from .utils import torch_compile


@torch_compile
def _scaled_sq_dist(x: Tensor, y: Tensor, lengthscale: Tensor) -> Tensor:
    x_scaled = x / lengthscale
    y_scaled = y / lengthscale
    x_norm = x_scaled.pow(2).sum(dim=-1, keepdim=True)
    y_norm = y_scaled.pow(2).sum(dim=-1, keepdim=True)
    cross = x_scaled @ y_scaled.transpose(-2, -1)
    sq_dist = x_norm + y_norm.transpose(-2, -1) - 2.0 * cross
    return sq_dist.clamp_min(0.0)


@torch_compile
def _rbf_forward_impl(
    x: Tensor, y: Tensor, lengthscale: Tensor, outputscale: Tensor
) -> Tensor:
    sq_dist = _scaled_sq_dist(x, y, lengthscale)
    return outputscale * torch.exp(-0.5 * sq_dist)


@torch_compile
def _rbf_gram_impl(
    inducing: Tensor, lengthscale: Tensor, outputscale: Tensor, jitter: float
) -> Tensor:
    kzz = _rbf_forward_impl(inducing, inducing, lengthscale, outputscale)
    eye = torch.eye(
        inducing.size(-2),
        device=inducing.device,
        dtype=inducing.dtype,
    )
    return kzz + jitter * eye


@dataclass
class KernelEvaluation:
    kxz: Tensor
    diag: Tensor


class Kernel(nn.Module, ABC):
    """Minimal kernel interface consumed by SparseGPTransition."""

    def __init__(self, input_dim: int, jitter: float = 1e-5) -> None:
        super().__init__()
        if input_dim <= 0:
            msg = "input_dim must be positive"
            raise ValueError(msg)
        self.input_dim = input_dim
        self.jitter = jitter

    def _validate_single(self, x: Tensor) -> None:
        if x.size(-1) != self.input_dim:
            msg = "Inputs must match kernel input_dim"
            raise ValueError(msg)

    def _validate_pair(self, x: Tensor, y: Tensor) -> None:
        self._validate_single(x)
        self._validate_single(y)

    @abstractmethod
    def forward(self, x: Tensor, y: Tensor) -> Tensor:  # pragma: no cover - interface
        """Return full covariance between x and y."""

    @abstractmethod
    def diag(self, x: Tensor) -> Tensor:  # pragma: no cover - interface
        """Return kernel diagonal for x."""

    def evaluate_cross(self, x: Tensor, inducing: Tensor) -> KernelEvaluation:
        self._validate_pair(x, inducing)
        kxz = self.forward(x, inducing)
        diag = self.diag(x)
        return KernelEvaluation(kxz=kxz, diag=diag)

    def gram(self, inducing: Tensor) -> Tensor:
        self._validate_single(inducing)
        kzz = self.forward(inducing, inducing)
        eye = torch.eye(
            inducing.size(-2),
            device=inducing.device,
            dtype=inducing.dtype,
        )
        return kzz + self.jitter * eye


class ARDRBFKernel(Kernel):
    """ARD RBF kernel with log-parameterized hyper-parameters."""

    def __init__(self, input_dim: int, jitter: float = 1e-5) -> None:
        super().__init__(input_dim=input_dim, jitter=jitter)
        self.log_lengthscale = nn.Parameter(torch.zeros(input_dim))
        self.log_outputscale = nn.Parameter(torch.zeros(1))

    @property
    def lengthscale(self) -> Tensor:
        return self.log_lengthscale.exp()

    @property
    def outputscale(self) -> Tensor:
        return self.log_outputscale.exp()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        self._validate_pair(x, y)
        return _rbf_forward_impl(x, y, self.lengthscale, self.outputscale)

    def diag(self, x: Tensor) -> Tensor:
        self._validate_single(x)
        return self.outputscale.expand(x.shape[:-1])

    def gram(self, inducing: Tensor) -> Tensor:
        self._validate_single(inducing)
        return _rbf_gram_impl(inducing, self.lengthscale, self.outputscale, self.jitter)


@torch_compile
def _matern_forward_impl(
    x: Tensor, y: Tensor, lengthscale: Tensor, outputscale: Tensor, nu: float
) -> Tensor:
    sq_dist = _scaled_sq_dist(x, y, lengthscale)
    dist = sq_dist.clamp_min(1e-12).sqrt()
    if nu == 0.5:
        kernel = torch.exp(-dist)
    elif nu == 1.5:
        scale = math.sqrt(3.0)
        kernel = (1.0 + scale * dist) * torch.exp(-scale * dist)
    elif nu == 2.5:
        scale = math.sqrt(5.0)
        kernel = (1.0 + scale * dist + 5.0 * sq_dist / 3.0) * torch.exp(-scale * dist)
    else:  # pragma: no cover - guarded in caller
        raise ValueError("Unsupported nu")
    return outputscale * kernel


class MaternKernel(Kernel):
    VALID_NU = (0.5, 1.5, 2.5)

    def __init__(self, input_dim: int, nu: float = 1.5, jitter: float = 1e-5) -> None:
        if nu not in self.VALID_NU:
            msg = f"nu must be one of {self.VALID_NU}"
            raise ValueError(msg)
        super().__init__(input_dim=input_dim, jitter=jitter)
        self.nu = nu
        self.log_lengthscale = nn.Parameter(torch.zeros(input_dim))
        self.log_outputscale = nn.Parameter(torch.zeros(1))

    @property
    def lengthscale(self) -> Tensor:
        return self.log_lengthscale.exp()

    @property
    def outputscale(self) -> Tensor:
        return self.log_outputscale.exp()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        self._validate_pair(x, y)
        return _matern_forward_impl(x, y, self.lengthscale, self.outputscale, self.nu)

    def diag(self, x: Tensor) -> Tensor:
        self._validate_single(x)
        return self.outputscale.expand(x.shape[:-1])


@torch_compile
def _rq_forward_impl(
    x: Tensor,
    y: Tensor,
    lengthscale: Tensor,
    outputscale: Tensor,
    alpha: Tensor,
) -> Tensor:
    sq_dist = _scaled_sq_dist(x, y, lengthscale)
    base = 1.0 + sq_dist / (2.0 * alpha)
    return outputscale * base.pow(-alpha)


class RationalQuadraticKernel(Kernel):
    def __init__(
        self, input_dim: int, jitter: float = 1e-5, alpha: float = 1.0
    ) -> None:
        if alpha <= 0:
            msg = "alpha must be positive"
            raise ValueError(msg)
        super().__init__(input_dim=input_dim, jitter=jitter)
        self.log_lengthscale = nn.Parameter(torch.zeros(input_dim))
        self.log_outputscale = nn.Parameter(torch.zeros(1))
        self.log_alpha = nn.Parameter(torch.tensor(math.log(alpha)))

    @property
    def lengthscale(self) -> Tensor:
        return self.log_lengthscale.exp()

    @property
    def outputscale(self) -> Tensor:
        return self.log_outputscale.exp()

    @property
    def alpha(self) -> Tensor:
        return self.log_alpha.exp()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        self._validate_pair(x, y)
        return _rq_forward_impl(x, y, self.lengthscale, self.outputscale, self.alpha)

    def diag(self, x: Tensor) -> Tensor:
        self._validate_single(x)
        return self.outputscale.expand(x.shape[:-1])


class PeriodicKernel(Kernel):
    def __init__(
        self,
        input_dim: int,
        jitter: float = 1e-5,
        period: float = 1.0,
        lengthscale: float = 1.0,
    ) -> None:
        if period <= 0:
            msg = "period must be positive"
            raise ValueError(msg)
        if lengthscale <= 0:
            msg = "lengthscale must be positive"
            raise ValueError(msg)
        super().__init__(input_dim=input_dim, jitter=jitter)
        self.log_lengthscale = nn.Parameter(
            torch.full((input_dim,), math.log(lengthscale))
        )
        self.log_outputscale = nn.Parameter(torch.zeros(1))
        self.log_period = nn.Parameter(torch.full((input_dim,), math.log(period)))

    @property
    def lengthscale(self) -> Tensor:
        return self.log_lengthscale.exp()

    @property
    def outputscale(self) -> Tensor:
        return self.log_outputscale.exp()

    @property
    def period(self) -> Tensor:
        return self.log_period.exp()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        self._validate_pair(x, y)
        two_pi = 2.0 * math.pi
        phase_x = two_pi * x / self.period
        phase_y = two_pi * y / self.period
        cos_x = torch.cos(phase_x)
        sin_x = torch.sin(phase_x)
        cos_y = torch.cos(phase_y)
        sin_y = torch.sin(phase_y)
        inv_lengthscale_sq = self.lengthscale.pow(2).clamp_min(1e-12).reciprocal()
        weighted_cos_x = cos_x * inv_lengthscale_sq
        weighted_sin_x = sin_x * inv_lengthscale_sq
        cos_term = weighted_cos_x @ cos_y.transpose(-2, -1)
        sin_term = weighted_sin_x @ sin_y.transpose(-2, -1)
        exponent = cos_term + sin_term - inv_lengthscale_sq.sum()
        return self.outputscale * torch.exp(exponent)

    def diag(self, x: Tensor) -> Tensor:
        self._validate_single(x)
        return self.outputscale.expand(x.shape[:-1])


class SumKernel(Kernel):
    def __init__(self, kernels: Sequence[Kernel], jitter: float = 1e-5) -> None:
        if not kernels:
            msg = "SumKernel requires at least one component"
            raise ValueError(msg)
        input_dim = kernels[0].input_dim
        if any(kernel.input_dim != input_dim for kernel in kernels):
            msg = "All component kernels must share input_dim"
            raise ValueError(msg)
        super().__init__(input_dim=input_dim, jitter=jitter)
        self.kernels = nn.ModuleList(kernels)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        self._validate_pair(x, y)
        components = [kernel(x, y) for kernel in self.kernels]
        return torch.stack(components, dim=0).sum(dim=0)

    def diag(self, x: Tensor) -> Tensor:
        self._validate_single(x)
        diags = [kernel.diag(x) for kernel in self.kernels]
        return torch.stack(diags, dim=0).sum(dim=0)

    def gram(self, inducing: Tensor) -> Tensor:
        self._validate_single(inducing)
        gram_terms = [kernel(inducing, inducing) for kernel in self.kernels]
        kzz = torch.stack(gram_terms, dim=0).sum(dim=0)
        eye = torch.eye(inducing.size(-2), device=inducing.device, dtype=inducing.dtype)
        return kzz + self.jitter * eye


@torch_compile
def _sm_forward_impl(
    x: Tensor,
    y: Tensor,
    weights: Tensor,
    means: Tensor,
    variances: Tensor,
) -> Tensor:
    """Spectral Mixture kernel forward (Wilson & Adams, 2013).

    Implements the product-of-dimensions form derived from Bochner's theorem:
      k(x,y) = Σ_q w_q · Π_d exp(-2π²τ_d²v_{q,d}) · cos(2πτ_d μ_{q,d})
    where τ = x - y.
    """
    tau = x.unsqueeze(-2) - y.unsqueeze(-3)  # [N, M, D]
    two_pi_sq = 2.0 * math.pi ** 2
    two_pi = 2.0 * math.pi

    # Exp component: exp(-2π² Σ_d τ_d² v_{q,d})  —  contract over D
    exp_arg = tau.pow(2) @ variances.T  # [N, M, Q]
    exp_component = torch.exp(-two_pi_sq * exp_arg)

    # Cos component: Π_d cos(2πτ_d μ_{q,d})  —  product over D
    cos_arg = two_pi * tau.unsqueeze(-2) * means  # [N, M, Q, D]
    cos_component = torch.cos(cos_arg).prod(dim=-1)  # [N, M, Q]

    return (weights * exp_component * cos_component).sum(dim=-1)


class SpectralMixtureKernel(Kernel):
    """Spectral Mixture kernel (Wilson & Adams, 2013).

    Models any stationary covariance via a mixture of Q Gaussian
    spectral densities — the most expressive kernel family justified
    by Bochner's theorem.  Each component captures a characteristic
    frequency (mean) with a bandwidth (variance) at a given magnitude
    (weight).
    """

    def __init__(
        self,
        input_dim: int,
        num_mixtures: int = 4,
        jitter: float = 1e-5,
    ) -> None:
        if num_mixtures <= 0:
            msg = "num_mixtures must be positive"
            raise ValueError(msg)
        super().__init__(input_dim=input_dim, jitter=jitter)
        self.num_mixtures = num_mixtures
        self.log_weights = nn.Parameter(torch.zeros(num_mixtures))
        # Frequencies are unconstrained: cos is even so sign is redundant,
        # and this avoids exp-overflow (CodeX review finding).
        self.raw_means = nn.Parameter(torch.randn(num_mixtures, input_dim) * 0.5)
        self.log_variances = nn.Parameter(torch.zeros(num_mixtures, input_dim))

    @property
    def weights(self) -> Tensor:
        return self.log_weights.exp()

    @property
    def means(self) -> Tensor:
        return self.raw_means

    @property
    def variances(self) -> Tensor:
        return self.log_variances.exp()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        self._validate_pair(x, y)
        return _sm_forward_impl(x, y, self.weights, self.means, self.variances)

    def diag(self, x: Tensor) -> Tensor:
        self._validate_single(x)
        return self.weights.sum().expand(x.shape[:-1])


class ProductKernel(Kernel):
    def __init__(self, kernels: Sequence[Kernel], jitter: float = 1e-5) -> None:
        if not kernels:
            msg = "ProductKernel requires at least one component"
            raise ValueError(msg)
        input_dim = kernels[0].input_dim
        if any(kernel.input_dim != input_dim for kernel in kernels):
            msg = "All component kernels must share input_dim"
            raise ValueError(msg)
        super().__init__(input_dim=input_dim, jitter=jitter)
        self.kernels = nn.ModuleList(kernels)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        self._validate_pair(x, y)
        components = [kernel(x, y) for kernel in self.kernels]
        return torch.stack(components, dim=0).prod(dim=0)

    def diag(self, x: Tensor) -> Tensor:
        self._validate_single(x)
        diags = [kernel.diag(x) for kernel in self.kernels]
        return torch.stack(diags, dim=0).prod(dim=0)

    def gram(self, inducing: Tensor) -> Tensor:
        self._validate_single(inducing)
        cov_terms = [kernel(inducing, inducing) for kernel in self.kernels]
        kzz = torch.stack(cov_terms, dim=0).prod(dim=0)
        eye = torch.eye(inducing.size(-2), device=inducing.device, dtype=inducing.dtype)
        return kzz + self.jitter * eye


__all__ = [
    "Kernel",
    "KernelEvaluation",
    "ARDRBFKernel",
    "MaternKernel",
    "RationalQuadraticKernel",
    "PeriodicKernel",
    "SpectralMixtureKernel",
    "SumKernel",
    "ProductKernel",
]
