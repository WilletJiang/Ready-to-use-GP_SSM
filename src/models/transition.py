from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
from pyro import distributions as dist
import pyro.distributions.transforms as T
from pyro.nn import PyroModule
from torch import Tensor, nn

from .kernels import Kernel
from .utils import torch_compile


@torch_compile
def _transition_forward_compute(
    kxz: Tensor,
    diag: Tensor,
    kzz_inv: Tensor,
    alpha: Tensor,
) -> Tuple[Tensor, Tensor]:
    mean = kxz @ alpha
    proj = kxz @ kzz_inv
    quad_form = (proj * kxz).sum(dim=-1)
    var_scalar = diag - quad_form
    var_scalar = var_scalar.clamp_min(1e-6)
    var = var_scalar.unsqueeze(-1).expand_as(mean)
    return mean, var


@torch_compile
def _chol_inverse(chol: Tensor) -> Tensor:
    return torch.cholesky_inverse(chol)

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
        self.register_buffer("_eye", torch.eye(num_inducing), persistent=False)

    @property
    def u_dim(self) -> int:
        return self.state_dim * self.num_inducing

    def prior(
        self,
        return_chol: bool = False,
    ) -> Union[dist.MultivariateNormal, Tuple[dist.MultivariateNormal, Tensor]]:
        kzz, chol = self._kzz_and_chol()
        loc = torch.zeros(
            self.state_dim,
            self.num_inducing,
            device=kzz.device,
            dtype=kzz.dtype,
        )
        scale_tril = chol.unsqueeze(0).expand(self.state_dim, -1, -1)
        base = dist.MultivariateNormal(loc, scale_tril=scale_tril)
        indep = dist.Independent(base, 1)
        reshape = T.ReshapeTransform(indep.event_shape, (self.u_dim,))
        mvn = dist.TransformedDistribution(indep, reshape)
        if return_chol:
            return mvn, chol
        return mvn

    def _kzz_and_chol(self) -> Tuple[Tensor, Tensor]:
        inducing = self.inducing_points
        kzz = self.kernel(inducing, inducing)
        eye = self._eye.to(device=inducing.device, dtype=inducing.dtype)
        kzz = kzz + self.kernel.jitter * eye
        chol = torch.linalg.cholesky(kzz)
        return kzz, chol

    def precompute(self, u: Tensor, chol: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Precompute Kzz^{-1} and the projected inducing mean for reuse across many time steps.
        """
        if chol is None:
            _, chol = self._kzz_and_chol()
        kzz_inv = _chol_inverse(chol)
        alpha = kzz_inv @ u.T
        return kzz_inv, alpha

    def forward(
        self,
        x_prev: Tensor,
        u: Tensor,
        cache: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tensor]:
        evals = self.kernel.evaluate_cross(x_prev, self.inducing_points)
        if cache is None:
            kzz_inv, alpha = self.precompute(u)
        else:
            kzz_inv, alpha = cache

        return _transition_forward_compute(evals.kxz, evals.diag, kzz_inv, alpha)
