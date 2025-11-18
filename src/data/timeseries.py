from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


def _sinusoidal_drift(x: Tensor, weight: Tensor) -> Tensor:
    return torch.sin(x @ weight.T)


@dataclass
class SequenceDataset:
    observations: Tensor
    lengths: Tensor
    latents: Optional[Tensor] = None


def generate_synthetic_sequences(
    num_sequences: int,
    sequence_length: int,
    state_dim: int,
    obs_dim: int,
    observation_gain: float,
    process_noise: float,
    obs_noise: float,
    device: Optional[torch.device] = None,
) -> SequenceDataset:
    if num_sequences <= 0 or sequence_length <= 0:
        msg = "Dataset dimensions must be positive"
        raise ValueError(msg)
    device = device or torch.device("cpu")
    sequences = torch.zeros(num_sequences, sequence_length, obs_dim, device=device)
    latents = torch.zeros(num_sequences, sequence_length, state_dim, device=device)
    lengths = torch.full((num_sequences,), sequence_length, dtype=torch.long, device=device)
    weight = torch.randn(state_dim, state_dim, device=device) * 0.2
    obs_matrix = torch.randn(obs_dim, state_dim, device=device) * observation_gain
    proc_noise = torch.distributions.Normal(0.0, math.sqrt(process_noise))
    obs_noise_dist = torch.distributions.Normal(0.0, math.sqrt(obs_noise))
    for n in range(num_sequences):
        x_prev = torch.randn(state_dim, device=device) * 0.1
        for t in range(sequence_length):
            drift = _sinusoidal_drift(x_prev.unsqueeze(0), weight).squeeze(0)
            x_curr = x_prev + 0.1 * drift + proc_noise.sample((state_dim,)).to(device)
            y = obs_matrix @ x_curr + obs_noise_dist.sample((obs_dim,)).to(device)
            latents[n, t] = x_curr
            sequences[n, t] = y
            x_prev = x_curr
    return SequenceDataset(observations=sequences, lengths=lengths, latents=latents)


def generate_system_identification_sequences(
    num_sequences: int,
    sequence_length: int,
    state_dim: int,
    obs_dim: int,
    dt: float,
    process_noise: float,
    obs_noise: float,
    control_scale: float,
    seed: int = 0,
    device: Optional[torch.device] = None,
) -> SequenceDataset:
    if num_sequences <= 0 or sequence_length <= 0:
        msg = "Dataset dimensions must be positive"
        raise ValueError(msg)
    device = device or torch.device("cpu")
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    obs = torch.zeros(num_sequences, sequence_length, obs_dim, device=device)
    latents = torch.zeros(num_sequences, sequence_length, state_dim, device=device)
    lengths = torch.full((num_sequences,), sequence_length, dtype=torch.long, device=device)
    lin = -0.25 * torch.eye(state_dim, device=device) + 0.05 * torch.randn(
        state_dim,
        state_dim,
        generator=generator,
        device=device,
    )
    nonlin = 0.1 * torch.randn(state_dim, state_dim, generator=generator, device=device)
    control_mat = 0.2 * torch.randn(state_dim, state_dim, generator=generator, device=device)
    obs_mat = torch.randn(obs_dim, state_dim, generator=generator, device=device)
    proc_noise = torch.distributions.Normal(0.0, math.sqrt(process_noise))
    obs_noise_dist = torch.distributions.Normal(0.0, math.sqrt(obs_noise))
    freqs = torch.rand(state_dim, generator=generator, device=device) * 0.5 + 0.2
    phases = torch.rand(state_dim, generator=generator, device=device) * 2 * math.pi
    for n in range(num_sequences):
        x_prev = torch.randn(state_dim, device=device, generator=generator) * 0.1
        for t in range(sequence_length):
            control = control_scale * torch.sin(freqs * (t * dt) + phases)
            drift = lin @ x_prev + torch.tanh(nonlin @ x_prev) + control_mat @ control
            noise = proc_noise.sample((state_dim,)).to(device)
            x_curr = x_prev + dt * drift + noise
            x_curr = torch.clamp(x_curr, -3.0, 3.0)
            y = obs_mat @ torch.tanh(x_curr) + obs_noise_dist.sample((obs_dim,)).to(device)
            latents[n, t] = x_curr
            obs[n, t] = y
            x_prev = x_curr
    return SequenceDataset(observations=obs, lengths=lengths, latents=latents)


def split_dataset(
    dataset: SequenceDataset,
    splits: Tuple[float, float, float],
    seed: int,
) -> Dict[str, SequenceDataset]:
    if not math.isclose(sum(splits), 1.0, rel_tol=1e-6):
        msg = "Splits must sum to 1.0"
        raise ValueError(msg)
    num_sequences = dataset.observations.size(0)
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_sequences, generator=generator)
    train_end = int(splits[0] * num_sequences)
    val_end = train_end + int(splits[1] * num_sequences)
    indices = {
        "train": perm[:train_end],
        "val": perm[train_end:val_end],
        "test": perm[val_end:],
    }

    def _slice(idx: Tensor) -> SequenceDataset:
        return SequenceDataset(
            observations=dataset.observations[idx],
            lengths=dataset.lengths[idx],
            latents=dataset.latents[idx] if dataset.latents is not None else None,
        )

    return {split: _slice(idx) for split, idx in indices.items()}


class TimeseriesWindowDataset(Dataset):
    def __init__(
        self,
        sequences: Tensor,
        lengths: Tensor,
        window_length: Optional[int],
        latents: Optional[Tensor] = None,
    ) -> None:
        if sequences.ndim != 3:
            msg = "Expect [N, T, D] tensor"
            raise ValueError(msg)
        self.sequences = sequences
        self.lengths = lengths
        self.latents = latents
        full_length = sequences.size(1)
        self.window_length = window_length or full_length

    def __len__(self) -> int:
        return self.sequences.size(0)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        seq = self.sequences[idx]
        length = int(self.lengths[idx].item())
        target_window = self.window_length
        pad_len = 0
        start = 0
        if length <= target_window:
            window = seq[:length]
            pad_len = target_window - length
        else:
            start = torch.randint(0, length - target_window + 1, (1,)).item()
            end = start + target_window
            window = seq[start:end]
        if pad_len > 0:
            pad = torch.zeros(pad_len, seq.size(-1), device=seq.device)
            window = torch.cat([window, pad], dim=0)
        latent_window = None
        if self.latents is not None:
            latent_seq = self.latents[idx]
            if pad_len > 0:
                latent_window = latent_seq[:length]
                latent_pad = torch.zeros(pad_len, latent_seq.size(-1), device=seq.device)
                latent_window = torch.cat([latent_window, latent_pad], dim=0)
            else:
                end = start + window.size(0)
                latent_window = latent_seq[start:end]
        return {
            "y": window,
            "length": torch.tensor(min(length, target_window), dtype=torch.long),
            "latent": latent_window,
        }


def build_dataloader(
    dataset: TimeseriesWindowDataset,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    if batch_size <= 0:
        msg = "batch_size must be positive"
        raise ValueError(msg)

    def _collate(batch):
        ys = torch.stack([item["y"] for item in batch], dim=0)
        lens = torch.stack([item["length"] for item in batch], dim=0)
        latents = None
        if batch[0]["latent"] is not None:
            latents = torch.stack([item["latent"] for item in batch], dim=0)
        result = {"y": ys, "lengths": lens}
        if latents is not None:
            result["latents"] = latents
        return result

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=_collate)
