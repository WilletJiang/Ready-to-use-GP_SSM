from __future__ import annotations

from torch import Tensor, nn


class AffineObservationModel(nn.Module):
    """Simple linear observation model y = C x + d."""

    def __init__(self, state_dim: int, obs_dim: int) -> None:
        super().__init__()
        if state_dim <= 0 or obs_dim <= 0:
            msg = "Dimensions must be positive"
            raise ValueError(msg)
        self.linear = nn.Linear(state_dim, obs_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)
