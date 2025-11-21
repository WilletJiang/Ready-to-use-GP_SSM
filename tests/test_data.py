import pytest
import torch

from data.timeseries import (
    TimeseriesWindowDataset,
    build_dataloader,
    generate_system_identification_sequences,
)


def test_system_identification_dataset_split() -> None:
    dataset = generate_system_identification_sequences(
        num_sequences=16,
        sequence_length=50,
        state_dim=3,
        obs_dim=2,
        dt=0.1,
        process_noise=0.01,
        obs_noise=0.02,
        control_scale=0.2,
        seed=0,
    )
    splits = {
        "train": dataset,  # reuse helper; split logic tested elsewhere
    }
    train_dataset = TimeseriesWindowDataset(
        sequences=splits["train"].observations,
        lengths=splits["train"].lengths,
        window_length=30,
        latents=splits["train"].latents,
    )
    sample = train_dataset[0]
    assert sample["y"].shape[0] == 30
    assert sample["latent"] is not None
    assert sample["latent"].shape[1] == splits["train"].latents.size(-1)


def test_window_dataset_is_deterministic_with_generator() -> None:
    sequences = torch.randn(2, 12, 3)
    lengths = torch.tensor([12, 12])
    generator = torch.Generator().manual_seed(12345)
    dataset = TimeseriesWindowDataset(
        sequences=sequences,
        lengths=lengths,
        window_length=6,
        generator=generator,
    )
    first = dataset[0]["y"]
    second = dataset[0]["y"]
    assert torch.equal(first, second)


def test_window_dataset_validates_lengths() -> None:
    sequences = torch.randn(1, 5, 2)
    lengths = torch.tensor([6])
    with pytest.raises(ValueError):
        TimeseriesWindowDataset(
            sequences=sequences,
            lengths=lengths,
            window_length=4,
        )
