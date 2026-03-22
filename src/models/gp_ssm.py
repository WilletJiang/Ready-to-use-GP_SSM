from __future__ import annotations

from typing import Optional

import pyro
import torch
from pyro import distributions as dist, poutine
from pyro.distributions import constraints
from pyro.nn import PyroModule, PyroParam
from torch import Tensor

from .encoder import StateEncoder
from .observation import AffineObservationModel
from .transition import SparseGPTransition


class SparseVariationalGPSSM(PyroModule):
    def __init__(
        self,
        transition: SparseGPTransition,
        encoder: StateEncoder,
        observation: Optional[PyroModule],
        obs_dim: int,
        process_noise_init: float,
        obs_noise_init: float,
        structured_q: bool = False,
    ) -> None:
        super().__init__()
        if obs_dim <= 0:
            msg = "obs_dim must be positive"
            raise ValueError(msg)
        if process_noise_init <= 0 or obs_noise_init <= 0:
            msg = "Noise initializers must be positive"
            raise ValueError(msg)
        self.transition = transition
        self.encoder = encoder
        self.observation = observation or AffineObservationModel(
            state_dim=transition.state_dim,
            obs_dim=obs_dim,
        )
        self.state_dim = transition.state_dim
        self.obs_dim = obs_dim
        self.structured_q = structured_q
        self.min_noise = 1e-5
        u_dim = transition.u_dim
        eye = torch.eye(u_dim)
        self.q_u_loc = PyroParam(torch.zeros(u_dim))
        self.q_u_tril = PyroParam(eye, constraint=constraints.lower_cholesky)
        self.x0_loc = PyroParam(torch.zeros(self.state_dim))
        self.x0_log_scale = PyroParam(torch.zeros(self.state_dim))
        self.log_process_noise = PyroParam(
            torch.full((self.state_dim,), torch.log(torch.tensor(process_noise_init))),
        )
        self.log_obs_noise = PyroParam(
            torch.full((self.obs_dim,), torch.log(torch.tensor(obs_noise_init))),
        )

    def _process_noise(self) -> Tensor:
        return self.log_process_noise.exp() + self.min_noise

    def _obs_noise(self) -> Tensor:
        return self.log_obs_noise.exp() + self.min_noise

    def _x0_scale(self) -> Tensor:
        return self.x0_log_scale.exp() + self.min_noise

    def _validate_controls(self, y: Tensor, controls: Optional[Tensor]) -> None:
        if controls is None:
            if self.transition.control_dim != 0:
                msg = "controls are required when transition.control_dim > 0"
                raise ValueError(msg)
            return
        if controls.shape[:2] != y.shape[:2]:
            msg = "controls must align with y on batch and time dimensions"
            raise ValueError(msg)
        if controls.size(-1) != self.transition.control_dim:
            msg = "controls feature dimension must match transition.control_dim"
            raise ValueError(msg)

    def model(
        self,
        y: Tensor,
        lengths: Optional[Tensor] = None,
        controls: Optional[Tensor] = None,
    ) -> None:
        pyro.module("observation", self.observation)
        pyro.module("transition", self.transition)
        pyro.module("encoder", self.encoder)
        self._validate_controls(y, controls)
        batch_size, horizon, _ = y.shape
        prior_dist, chol = self.transition.prior(return_chol=True)
        u_sample = pyro.sample("u", prior_dist)
        u = u_sample.reshape(self.state_dim, self.transition.num_inducing)
        cache = self.transition.precompute(u, chol)
        init_loc = self.x0_loc.unsqueeze(0).expand(batch_size, -1)
        init_scale = self._x0_scale().unsqueeze(0).expand_as(init_loc)
        process_noise = self._process_noise()
        process_var = process_noise.pow(2)
        obs_scale = self._obs_noise()
        with pyro.plate("batch", batch_size):
            x_prev = pyro.sample(
                "x_0",
                dist.Normal(init_loc, init_scale).to_event(1),
            )
            for t in range(horizon):
                obs_loc = self.observation(x_prev)
                obs_dist = dist.Normal(obs_loc, obs_scale).to_event(1)
                obs = y[:, t, :]
                if lengths is None:
                    pyro.sample(f"y_{t}", obs_dist, obs=obs)
                else:
                    obs_mask = t < lengths
                    with poutine.mask(mask=obs_mask):
                        pyro.sample(f"y_{t}", obs_dist, obs=obs)
                control_t = controls[:, t, :] if controls is not None else None
                mean, var = self.transition(x_prev, u, cache, controls=control_t)
                total_var = var + process_var
                scale = total_var.clamp_min(self.min_noise).sqrt()
                if lengths is None:
                    x_curr = pyro.sample(
                        f"x_{t+1}",
                        dist.Normal(mean, scale).to_event(1),
                    )
                else:
                    trans_mask = t < (lengths - 1)
                    with poutine.mask(mask=trans_mask):
                        x_curr = pyro.sample(
                            f"x_{t+1}",
                            dist.Normal(mean, scale).to_event(1),
                        )
                x_prev = x_curr

    def guide(
        self,
        y: Tensor,
        lengths: Optional[Tensor] = None,
        controls: Optional[Tensor] = None,
    ) -> None:
        pyro.module("observation", self.observation)
        pyro.module("transition", self.transition)
        pyro.module("encoder", self.encoder)
        self._validate_controls(y, controls)
        pyro.sample(
            "u",
            dist.MultivariateNormal(
                self.q_u_loc,
                scale_tril=self.q_u_tril,
            ),
        )
        batch_size, horizon, _ = y.shape
        encoding = self.encoder(y, lengths)
        init_scale = encoding.init_scale
        trans_matrix = encoding.trans_matrix
        trans_bias = encoding.trans_bias
        with pyro.plate("batch", batch_size):
            x_prev = pyro.sample(
                "x_0",
                dist.Normal(encoding.init_loc, init_scale).to_event(1),
            )
            for t in range(horizon):
                if self.structured_q:
                    if trans_matrix is None or trans_bias is None:
                        msg = "Encoder must produce transition parameters when structured_q is True"
                        raise RuntimeError(msg)
                    aff = torch.einsum("bij,bj->bi", trans_matrix[:, t, :, :], x_prev)
                    mean = aff + trans_bias[:, t, :]
                else:
                    mean = encoding.loc[:, t, :]
                scale = encoding.scale[:, t, :]
                if lengths is None:
                    x_prev = pyro.sample(
                        f"x_{t+1}",
                        dist.Normal(
                            mean,
                            scale,
                        ).to_event(1),
                    )
                else:
                    trans_mask = t < (lengths - 1)
                    with poutine.mask(mask=trans_mask):
                        x_prev = pyro.sample(
                            f"x_{t+1}",
                            dist.Normal(
                                mean,
                                scale,
                            ).to_event(1),
                        )
