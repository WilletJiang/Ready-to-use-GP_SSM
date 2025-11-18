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

    def _diag_embed(self, diag: Tensor) -> Tensor:
        return torch.diag_embed(diag)

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
        u_sample = pyro.sample("u", self.transition.prior())
        u = u_sample.reshape(self.state_dim, self.transition.num_inducing)
        init_loc = self.x0_loc.unsqueeze(0).expand(batch_size, -1)
        init_cov = self._diag_embed(self._x0_scale().pow(2)).expand(
            batch_size,
            self.state_dim,
            self.state_dim,
        )
        with pyro.plate("batch", batch_size):
            x_prev = pyro.sample(
                "x_0",
                dist.MultivariateNormal(init_loc, covariance_matrix=init_cov),
            )
            for t in range(horizon):
                mean, var = self.transition(x_prev, u)
                cov = self._diag_embed(var + self._process_noise())
                x_curr = pyro.sample(
                    f"x_{t+1}",
                    dist.MultivariateNormal(mean, covariance_matrix=cov),
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
        init_cov = self._diag_embed(encoding.init_scale.pow(2))
        with pyro.plate("batch", batch_size):
            pyro.sample(
                "x_0",
                dist.MultivariateNormal(encoding.init_loc, covariance_matrix=init_cov),
            )
            for t in range(horizon):
                cov = self._diag_embed(encoding.scale[:, t, :].pow(2))
                pyro.sample(
                    f"x_{t+1}",
                    dist.MultivariateNormal(encoding.loc[:, t, :], covariance_matrix=cov),
                )
