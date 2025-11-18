from __future__ import annotations

from typing import Optional, Tuple

import torch
from pyro import distributions as dist
from pyro.nn import PyroModule
from torch import Tensor, nn

from .kernels import ARDRBFKernel


class SparseGPTransition(PyroModule):
    def __init__(
        self,
        state_dim: int,
        num_inducing: int,
        kernel: ARDRBFKernel,
    ) -> None:
        super().__init__()
        if state_dim <= 0 or num_inducing <= 0:
            msg = "state_dim and num_inducing must be positive"
            raise ValueError(msg)
        self.state_dim = state_dim
        self.num_inducing = num_inducing
        self.kernel = kernel
        inducing = torch.randn(num_inducing, state_dim)
        self.inducing_points = nn.Parameter(inducing)

    @property
    def u_dim(self) -> int:
        return self.state_dim * self.num_inducing

    def prior(self) -> dist.MultivariateNormal:
        kzz = self.kernel.gram(self.inducing_points)
        eye = torch.eye(self.state_dim, device=kzz.device, dtype=kzz.dtype)
        cov = torch.kron(eye, kzz)
        loc = torch.zeros(self.u_dim, device=kzz.device, dtype=kzz.dtype)
        return dist.MultivariateNormal(loc, covariance_matrix=cov)

    def _kzz_and_chol(self) -> Tuple[Tensor, Tensor]:
        kzz = self.kernel.gram(self.inducing_points)
        chol = torch.linalg.cholesky(kzz)
        return kzz, chol

    def _chol_solve(self, chol: Tensor, rhs: Tensor) -> Tensor:
        return torch.cholesky_solve(rhs, chol)

    def precompute(self, u: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Precompute Kzz cholesky and solved_u for reuse across many time steps.
        """
        kzz, chol = self._kzz_and_chol()
        solved_u = self._chol_solve(chol, u.T)
        return chol, solved_u

    def forward(
        self,
        x_prev: Tensor,
        u: Tensor,
        cache: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tensor]:
        evals = self.kernel.evaluate_cross(x_prev, self.inducing_points)
        if cache is None:
            kzz, chol = self._kzz_and_chol()
            solved_u = self._chol_solve(chol, u.T)
        else:
            chol, solved_u = cache
        mean = evals.kxz @ solved_u
        solved_kxz = self._chol_solve(chol, evals.kxz.transpose(-2, -1))
        var_scalar = evals.diag - (evals.kxz * solved_kxz.transpose(-2, -1)).sum(dim=-1)
        var_scalar = var_scalar.clamp_min(1e-6)
        var = var_scalar.unsqueeze(-1).expand_as(mean)
        return mean, var
