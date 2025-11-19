from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


@dataclass
class EncoderOutput:
    init_loc: Tensor
    init_scale: Tensor
    loc: Tensor
    scale: Tensor


class StateEncoder(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        state_dim: int,
        hidden_size: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            msg = "hidden_size must be positive"
            raise ValueError(msg)
        self.bidirectional = True
        self.gru = nn.GRU(
            input_size=obs_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
        )
        proj_dim = hidden_size * 2
        self.loc_head = nn.Linear(proj_dim, state_dim)
        self.scale_head = nn.Linear(proj_dim, state_dim)
        self.init_loc_head = nn.Linear(proj_dim, state_dim)
        self.init_scale_head = nn.Linear(proj_dim, state_dim)
        self.softplus = nn.Softplus()

    def forward(
        self,
        y: Tensor,
        lengths: Optional[Tensor],
    ) -> EncoderOutput:
        if lengths is not None:
            packed = pack_padded_sequence(
                y,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            encoded, _ = self.gru(packed)
            padded, _ = pad_packed_sequence(
                encoded,
                batch_first=True,
                total_length=y.size(1),
            )
        else:
            padded, _ = self.gru(y)
        loc = self.loc_head(padded)
        scale = self.softplus(self.scale_head(padded)) + 1e-4
        init_context = padded[:, 0, :]
        init_loc = self.init_loc_head(init_context)
        init_scale = self.softplus(self.init_scale_head(init_context)) + 1e-4
        return EncoderOutput(init_loc=init_loc, init_scale=init_scale, loc=loc, scale=scale)
