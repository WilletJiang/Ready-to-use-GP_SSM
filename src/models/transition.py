from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
from pyro import distributions as dist
from pyro.nn import PyroModule
from torch import Tensor, nn

from .kernels import Kernel
from .utils import torch_compile

@torch_compile
def _transition_forward_compute(
    kxz: Tensor,
    diag: Tensor,
    chol: Tensor,
    solved_u: Tensor
) -> Tuple[Tensor, Tensor]:

    mean = kxz @ solved_u
    kxz_t = kxz.transpose(-2, -1) # [..., M, N]
    v = torch.linalg.solve_triangular(chol, kxz_t, upper=False)
    quad_form = v.pow(2).sum(dim=-2)
    var_scalar = diag - quad_form
    var_scalar = var_scalar.clamp_min(1e-6)
    var = var_scalar.unsqueeze(-1).expand_as(mean)
    return mean, var

@torch_compile
def _precompute_chol_solve(chol: Tensor, u_t: Tensor) -> Tensor:
    return torch.cholesky_solve(u_t, chol)

class SparseGPTransition(PyroModule):
    def __init__(
        self,
        state_dim: int,
        num_inducing: int,
        kernel: Kernel,
    ) -> None:
        super().__init__()
        if state_dim <= 0 or num_inducing <= 0:
            msg = "state_dim and num_inducing must be positive"
            raise ValueError(msg)
        if kernel.input_dim != state_dim:
            msg = "kernel.input_dim must match state_dim"
            raise ValueError(msg)
        self.state_dim = state_dim
        self.num_inducing = num_inducing
        self.kernel = kernel
        inducing = torch.randn(num_inducing, state_dim)
        self.inducing_points = nn.Parameter(inducing)

    @property
    def u_dim(self) -> int:
        return self.state_dim * self.num_inducing

    def prior(
        self,
        return_chol: bool = False,
    ) -> Union[dist.MultivariateNormal, Tuple[dist.MultivariateNormal, Tensor]]:
        kzz, chol = self._kzz_and_chol()
        eye = torch.eye(self.state_dim, device=kzz.device, dtype=kzz.dtype)
        cov = torch.kron(eye, kzz)
        loc = torch.zeros(self.u_dim, device=kzz.device, dtype=kzz.dtype)
        mvn = dist.MultivariateNormal(loc, covariance_matrix=cov)
        if return_chol:
            return mvn, chol
        return mvn

    def _kzz_and_chol(self) -> Tuple[Tensor, Tensor]:
        kzz = self.kernel.gram(self.inducing_points)
        chol = torch.linalg.cholesky(kzz)
        return kzz, chol

    def precompute(self, u: Tensor, chol: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Precompute Kzz cholesky and solved_u for reuse across many time steps.
        """
        if chol is None:
            _, chol = self._kzz_and_chol()
        solved_u = _precompute_chol_solve(chol, u.T)
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
            solved_u = _precompute_chol_solve(chol, u.T)
        else:
            chol, solved_u = cache

        return _transition_forward_compute(evals.kxz, evals.diag, chol, solved_u)
