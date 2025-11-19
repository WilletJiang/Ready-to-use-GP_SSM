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
    x_scaled = x.unsqueeze(-2) / lengthscale
    y_scaled = y.unsqueeze(-3) / lengthscale
    return (x_scaled - y_scaled).pow(2).sum(dim=-1).clamp_min(0.0)


@torch_compile
def _rbf_forward_impl(x: Tensor, y: Tensor, lengthscale: Tensor, outputscale: Tensor) -> Tensor:
    sq_dist = _scaled_sq_dist(x, y, lengthscale)
    return outputscale * torch.exp(-0.5 * sq_dist)


@torch_compile
def _rbf_gram_impl(inducing: Tensor, lengthscale: Tensor, outputscale: Tensor, jitter: float) -> Tensor:
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
def _matern_forward_impl(x: Tensor, y: Tensor, lengthscale: Tensor, outputscale: Tensor, nu: float) -> Tensor:
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
    def __init__(self, input_dim: int, jitter: float = 1e-5, alpha: float = 1.0) -> None:
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
        self.log_lengthscale = nn.Parameter(torch.full((input_dim,), math.log(lengthscale)))
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
        diff = x.unsqueeze(-2) - y.unsqueeze(-3)
        scaled = diff * (math.pi / self.period)
        sin_sq = torch.sin(scaled).pow(2)
        lengthscale_sq = self.lengthscale.pow(2)
        scaled_sum = (sin_sq / lengthscale_sq.clamp_min(1e-12)).sum(dim=-1)
        base = torch.exp(-2.0 * scaled_sum)
        return self.outputscale * base

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
        cov = self.kernels[0](x, y)
        for kernel in self.kernels[1:]:
            cov = cov + kernel(x, y)
        return cov

    def diag(self, x: Tensor) -> Tensor:
        self._validate_single(x)
        diag = self.kernels[0].diag(x)
        for kernel in self.kernels[1:]:
            diag = diag + kernel.diag(x)
        return diag

    def gram(self, inducing: Tensor) -> Tensor:
        self._validate_single(inducing)
        kzz = self.kernels[0](inducing, inducing)
        for kernel in self.kernels[1:]:
            kzz = kzz + kernel(inducing, inducing)
        eye = torch.eye(inducing.size(-2), device=inducing.device, dtype=inducing.dtype)
        return kzz + self.jitter * eye


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
        cov = self.kernels[0](x, y)
        for kernel in self.kernels[1:]:
            cov = cov * kernel(x, y)
        return cov

    def diag(self, x: Tensor) -> Tensor:
        self._validate_single(x)
        diag = self.kernels[0].diag(x)
        for kernel in self.kernels[1:]:
            diag = diag * kernel.diag(x)
        return diag

    def gram(self, inducing: Tensor) -> Tensor:
        self._validate_single(inducing)
        kzz = self.kernels[0](inducing, inducing)
        for kernel in self.kernels[1:]:
            kzz = kzz * kernel(inducing, inducing)
        eye = torch.eye(inducing.size(-2), device=inducing.device, dtype=inducing.dtype)
        return kzz + self.jitter * eye


__all__ = [
    "Kernel",
    "KernelEvaluation",
    "ARDRBFKernel",
    "MaternKernel",
    "RationalQuadraticKernel",
    "PeriodicKernel",
    "SumKernel",
    "ProductKernel",
]

