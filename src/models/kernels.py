from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor, nn
import sys
from .transition import torch_compile


@torch_compile
def _rbf_forward_impl(x: Tensor, y: Tensor, lengthscale: Tensor, outputscale: Tensor) -> Tensor:
    x_scaled = x.unsqueeze(-2) / lengthscale
    y_scaled = y.unsqueeze(-3) / lengthscale
    sq_dist = (x_scaled - y_scaled).pow(2).sum(dim=-1)
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


class ARDRBFKernel(nn.Module):
    """ARD RBF kernel with log-parameterized hyper-parameters."""

    def __init__(self, input_dim: int, jitter: float = 1e-5) -> None:
        super().__init__()
        if input_dim <= 0:
            msg = "input_dim must be positive"
            raise ValueError(msg)
        self.input_dim = input_dim
        self.log_lengthscale = nn.Parameter(torch.zeros(input_dim))
        self.log_outputscale = nn.Parameter(torch.zeros(1))
        self.jitter = jitter

    @property
    def lengthscale(self) -> Tensor:
        return self.log_lengthscale.exp()

    @property
    def outputscale(self) -> Tensor:
        return self.log_outputscale.exp()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if x.size(-1) != self.input_dim or y.size(-1) != self.input_dim:
            msg = "Inputs must match kernel input_dim"
            raise ValueError(msg)
        return _rbf_forward_impl(x, y, self.lengthscale, self.outputscale)

    def diag(self, x: Tensor) -> Tensor:
        if x.size(-1) != self.input_dim:
            msg = "Inputs must match kernel input_dim"
            raise ValueError(msg)
        return self.outputscale.expand(x.shape[:-1])

    def evaluate_cross(self, x: Tensor, inducing: Tensor) -> KernelEvaluation:
        kxz = self.forward(x, inducing)
        diag = self.diag(x)
        return KernelEvaluation(kxz=kxz, diag=diag)

    def gram(self, inducing: Tensor) -> Tensor:
        return _rbf_gram_impl(inducing, self.lengthscale, self.outputscale, self.jitter)
