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
    trans_matrix: Optional[Tensor] = None
    trans_bias: Optional[Tensor] = None


class StateEncoder(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        state_dim: int,
        hidden_size: int,
        num_layers: int,
        structured: bool = False,
        max_transition_scale: float = 0.5,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            msg = "hidden_size must be positive"
            raise ValueError(msg)
        if max_transition_scale <= 0:
            msg = "max_transition_scale must be positive"
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
        self.structured = structured
        self.max_transition_scale = max_transition_scale
        if structured:
            self.trans_matrix_head = nn.Linear(proj_dim, state_dim * state_dim)
            self.trans_bias_head = nn.Linear(proj_dim, state_dim)

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
        trans_matrix = None
        trans_bias = None
        if self.structured:
            raw_matrix = self.trans_matrix_head(padded)
            trans_matrix = raw_matrix.view(
                raw_matrix.size(0),
                raw_matrix.size(1),
                loc.size(-1),
                loc.size(-1),
            )
            trans_matrix = trans_matrix.tanh() * self.max_transition_scale
            trans_bias = self.trans_bias_head(padded)
        return EncoderOutput(
            init_loc=init_loc,
            init_scale=init_scale,
            loc=loc,
            scale=scale,
            trans_matrix=trans_matrix,
            trans_bias=trans_bias,
        )
