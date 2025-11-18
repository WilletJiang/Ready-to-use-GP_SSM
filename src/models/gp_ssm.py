from __future__ import annotations

from typing import Optional

import pyro
import torch
from pyro import distributions as dist, poutine
from pyro.distributions import constraints
from pyro.nn import PyroModule, PyroParam
from torch import Tensor

from .encoder import EncoderOutput, StateEncoder
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

    def model(self, y: Tensor, lengths: Optional[Tensor] = None) -> None:
        pyro.module("observation", self.observation)
        pyro.module("transition", self.transition)
        pyro.module("encoder", self.encoder)
        batch_size, horizon, _ = y.shape
        prior_dist, chol = self.transition.prior(return_chol=True)
        u_sample = pyro.sample("u", prior_dist)
        u = u_sample.reshape(self.state_dim, self.transition.num_inducing)
        cache = self.transition.precompute(u, chol)
        init_loc = self.x0_loc.unsqueeze(0).expand(batch_size, -1)
        init_scale = self._x0_scale().unsqueeze(0).expand_as(init_loc)
        with pyro.plate("batch", batch_size):
            x_prev = pyro.sample(
                "x_0",
                dist.Normal(init_loc, init_scale).to_event(1),
            )
            for t in range(horizon):
                mean, var = self.transition(x_prev, u, cache)
                noise_var = self._process_noise()
                total_var = var + noise_var
                scale = total_var.clamp_min(self.min_noise).sqrt()
                x_curr = pyro.sample(
                    f"x_{t+1}",
                    dist.Normal(mean, scale).to_event(1),
                )
                obs_loc = self.observation(x_curr)
                obs_scale = self._obs_noise()
                obs_dist = dist.Normal(obs_loc, obs_scale).to_event(1)
                obs = y[:, t, :]
                if lengths is None:
                    pyro.sample(f"y_{t}", obs_dist, obs=obs)
                else:
                    mask = t < lengths
                    with poutine.mask(mask=mask):
                        pyro.sample(f"y_{t}", obs_dist, obs=obs)
                x_prev = x_curr

    def guide(self, y: Tensor, lengths: Optional[Tensor] = None) -> None:
        pyro.module("observation", self.observation)
        pyro.module("transition", self.transition)
        pyro.module("encoder", self.encoder)
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
        with pyro.plate("batch", batch_size):
            pyro.sample(
                "x_0",
                dist.Normal(encoding.init_loc, init_scale).to_event(1),
            )
            for t in range(horizon):
                pyro.sample(
                    f"x_{t+1}",
                    dist.Normal(
                        encoding.loc[:, t, :],
                        encoding.scale[:, t, :],
                    ).to_event(1),
                )
