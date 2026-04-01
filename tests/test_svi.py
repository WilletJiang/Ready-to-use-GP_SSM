import math

import pyro
import torch

from data.timeseries import (
    TimeseriesWindowDataset,
    build_dataloader,
    generate_synthetic_sequences,
)
from inference.svi import SVITrainer, TrainerConfig
from models.encoder import StateEncoder
from models.gp_ssm import SparseVariationalGPSSM
from models.kernels import ARDRBFKernel
from models.transition import SparseGPTransition


def test_trainer_does_not_clear_param_store(tmp_path) -> None:
    pyro.clear_param_store()
    pyro.param("sentinel", torch.tensor(1.0))
    kernel = ARDRBFKernel(input_dim=2)
    transition = SparseGPTransition(state_dim=2, num_inducing=4, kernel=kernel)
    encoder = StateEncoder(obs_dim=1, state_dim=2, hidden_size=8, num_layers=1)
    model = SparseVariationalGPSSM(
        transition=transition,
        encoder=encoder,
        observation=None,
        obs_dim=1,
        process_noise_init=0.1,
        obs_noise_init=0.1,
    )
    SVITrainer(
        model,
        TrainerConfig(
            steps=1,
            lr=1e-3,
            gradient_clip=5.0,
            report_every=1,
            checkpoint_dir=tmp_path,
            elbo="trace",
            eval_every=1,
        ),
    )
    assert "sentinel" in pyro.get_param_store()


def test_fit_saves_absolute_step_when_resuming(tmp_path) -> None:
    pyro.clear_param_store()
    torch.manual_seed(0)
    kernel = ARDRBFKernel(input_dim=2)
    transition = SparseGPTransition(state_dim=2, num_inducing=4, kernel=kernel)
    encoder = StateEncoder(obs_dim=1, state_dim=2, hidden_size=8, num_layers=1)
    model = SparseVariationalGPSSM(
        transition=transition,
        encoder=encoder,
        observation=None,
        obs_dim=1,
        process_noise_init=0.1,
        obs_noise_init=0.1,
    )
    synthetic = generate_synthetic_sequences(
        num_sequences=4,
        sequence_length=8,
        state_dim=2,
        obs_dim=1,
        observation_gain=1.0,
        process_noise=0.05,
        obs_noise=0.05,
    )
    dataset = TimeseriesWindowDataset(
        sequences=synthetic.observations,
        lengths=synthetic.lengths,
        window_length=4,
        latents=synthetic.latents,
    )
    loader = build_dataloader(dataset, batch_size=2, shuffle=False)
    trainer = SVITrainer(
        model,
        TrainerConfig(
            steps=2,
            lr=1e-3,
            gradient_clip=5.0,
            report_every=1,
            checkpoint_dir=tmp_path,
            elbo="trace",
            eval_every=1,
        ),
    )
    trainer.fit(loader, start_step=7)
    state = torch.load(tmp_path / "final.pt")
    assert state["step"] == 9


def test_resume_checkpoint_restores_optimizer_state(tmp_path) -> None:
    synthetic = generate_synthetic_sequences(
        num_sequences=4,
        sequence_length=8,
        state_dim=2,
        obs_dim=1,
        observation_gain=1.0,
        process_noise=0.05,
        obs_noise=0.05,
    )
    dataset = TimeseriesWindowDataset(
        sequences=synthetic.observations,
        lengths=synthetic.lengths,
        window_length=4,
        latents=synthetic.latents,
    )
    loader = build_dataloader(dataset, batch_size=len(dataset), shuffle=False)

    pyro.clear_param_store()
    torch.manual_seed(0)
    checkpoint_kernel = ARDRBFKernel(input_dim=2)
    checkpoint_transition = SparseGPTransition(
        state_dim=2,
        num_inducing=4,
        kernel=checkpoint_kernel,
    )
    checkpoint_encoder = StateEncoder(
        obs_dim=1,
        state_dim=2,
        hidden_size=8,
        num_layers=1,
    )
    checkpoint_model = SparseVariationalGPSSM(
        transition=checkpoint_transition,
        encoder=checkpoint_encoder,
        observation=None,
        obs_dim=1,
        process_noise_init=0.1,
        obs_noise_init=0.1,
    )
    checkpoint_trainer = SVITrainer(
        checkpoint_model,
        TrainerConfig(
            steps=1,
            lr=1e-3,
            gradient_clip=5.0,
            report_every=1,
            checkpoint_dir=tmp_path / "resume_source",
            elbo="trace",
            eval_every=1,
        ),
    )
    checkpoint_trainer.fit(loader)
    checkpoint_state = torch.load(tmp_path / "resume_source" / "final.pt")

    pyro.clear_param_store()
    torch.manual_seed(999)
    resumed_kernel = ARDRBFKernel(input_dim=2)
    resumed_transition = SparseGPTransition(
        state_dim=2,
        num_inducing=4,
        kernel=resumed_kernel,
    )
    resumed_encoder = StateEncoder(
        obs_dim=1,
        state_dim=2,
        hidden_size=8,
        num_layers=1,
    )
    resumed_model = SparseVariationalGPSSM(
        transition=resumed_transition,
        encoder=resumed_encoder,
        observation=None,
        obs_dim=1,
        process_noise_init=0.1,
        obs_noise_init=0.1,
    )
    resumed_trainer = SVITrainer(
        resumed_model,
        TrainerConfig(
            steps=1,
            lr=1e-3,
            gradient_clip=5.0,
            report_every=1,
            checkpoint_dir=tmp_path / "resumed",
            elbo="trace",
            eval_every=1,
        ),
    )
    start_step = resumed_trainer.load_checkpoint(
        tmp_path / "resume_source" / "final.pt",
        device=torch.device("cpu"),
    )
    assert start_step == 1
    waiting_state = resumed_trainer.svi.optim._state_waiting_to_be_consumed
    saved_optimizer_state = checkpoint_state["optimizer_state"]
    assert waiting_state.keys() == saved_optimizer_state.keys()
    first_key = next(iter(saved_optimizer_state))
    assert (
        waiting_state[first_key]["state"][0]["step"]
        == saved_optimizer_state[first_key]["state"][0]["step"]
    )
    assert torch.allclose(
        waiting_state[first_key]["state"][0]["exp_avg"],
        saved_optimizer_state[first_key]["state"][0]["exp_avg"],
    )

    batch = next(iter(loader))
    resumed_loss = resumed_trainer.svi.step(batch["y"], batch.get("lengths"))
    assert math.isfinite(resumed_loss)
    assert not resumed_trainer.svi.optim._state_waiting_to_be_consumed


def test_load_checkpoint_supports_legacy_param_store_format(tmp_path) -> None:
    pyro.clear_param_store()
    torch.manual_seed(0)
    kernel = ARDRBFKernel(input_dim=2)
    transition = SparseGPTransition(state_dim=2, num_inducing=4, kernel=kernel)
    encoder = StateEncoder(obs_dim=1, state_dim=2, hidden_size=8, num_layers=1)
    model = SparseVariationalGPSSM(
        transition=transition,
        encoder=encoder,
        observation=None,
        obs_dim=1,
        process_noise_init=0.1,
        obs_noise_init=0.1,
    )
    synthetic = generate_synthetic_sequences(
        num_sequences=4,
        sequence_length=8,
        state_dim=2,
        obs_dim=1,
        observation_gain=1.0,
        process_noise=0.05,
        obs_noise=0.05,
    )
    dataset = TimeseriesWindowDataset(
        sequences=synthetic.observations,
        lengths=synthetic.lengths,
        window_length=4,
        latents=synthetic.latents,
    )
    loader = build_dataloader(dataset, batch_size=2, shuffle=False)
    trainer = SVITrainer(
        model,
        TrainerConfig(
            steps=1,
            lr=1e-3,
            gradient_clip=5.0,
            report_every=1,
            checkpoint_dir=tmp_path,
            elbo="trace",
            eval_every=1,
        ),
    )
    batch = next(iter(loader))
    trainer.svi.step(batch["y"], batch.get("lengths"))

    legacy_path = tmp_path / "legacy.pt"
    torch.save(
        {"step": 5, "params": pyro.get_param_store().get_state()},
        legacy_path,
    )

    pyro.clear_param_store()
    resumed_kernel = ARDRBFKernel(input_dim=2)
    resumed_transition = SparseGPTransition(
        state_dim=2,
        num_inducing=4,
        kernel=resumed_kernel,
    )
    resumed_encoder = StateEncoder(
        obs_dim=1,
        state_dim=2,
        hidden_size=8,
        num_layers=1,
    )
    resumed_model = SparseVariationalGPSSM(
        transition=resumed_transition,
        encoder=resumed_encoder,
        observation=None,
        obs_dim=1,
        process_noise_init=0.1,
        obs_noise_init=0.1,
    )
    resumed_trainer = SVITrainer(
        resumed_model,
        TrainerConfig(
            steps=1,
            lr=1e-3,
            gradient_clip=5.0,
            report_every=1,
            checkpoint_dir=tmp_path / "legacy_resume",
            elbo="trace",
            eval_every=1,
        ),
    )
    start_step = resumed_trainer.load_checkpoint(
        legacy_path,
        device=torch.device("cpu"),
    )
    assert start_step == 5
    assert pyro.get_param_store().get_state()["params"]


def test_single_svi_step_runs(tmp_path) -> None:
    pyro.clear_param_store()
    torch.manual_seed(0)
    kernel = ARDRBFKernel(input_dim=2)
    transition = SparseGPTransition(state_dim=2, num_inducing=4, kernel=kernel)
    encoder = StateEncoder(obs_dim=1, state_dim=2, hidden_size=8, num_layers=1)
    model = SparseVariationalGPSSM(
        transition=transition,
        encoder=encoder,
        observation=None,
        obs_dim=1,
        process_noise_init=0.1,
        obs_noise_init=0.1,
    )
    synthetic = generate_synthetic_sequences(
        num_sequences=4,
        sequence_length=16,
        state_dim=2,
        obs_dim=1,
        observation_gain=1.0,
        process_noise=0.05,
        obs_noise=0.05,
    )
    dataset = TimeseriesWindowDataset(
        sequences=synthetic.observations,
        lengths=synthetic.lengths,
        window_length=8,
        latents=synthetic.latents,
    )
    loader = build_dataloader(dataset, batch_size=2, shuffle=False)
    trainer = SVITrainer(
        model,
        TrainerConfig(
            steps=1,
            lr=1e-3,
            gradient_clip=5.0,
            report_every=1,
            checkpoint_dir=tmp_path,
            elbo="trace",
            eval_every=1,
        ),
    )
    batch = next(iter(loader))
    loss = trainer.svi.step(batch["y"], batch.get("lengths"))
    assert math.isfinite(loss)


def test_root_pyro_params_are_updated_by_svi(tmp_path) -> None:
    pyro.clear_param_store()
    torch.manual_seed(0)
    kernel = ARDRBFKernel(input_dim=2)
    transition = SparseGPTransition(state_dim=2, num_inducing=4, kernel=kernel)
    encoder = StateEncoder(obs_dim=1, state_dim=2, hidden_size=8, num_layers=1)
    model = SparseVariationalGPSSM(
        transition=transition,
        encoder=encoder,
        observation=None,
        obs_dim=1,
        process_noise_init=0.1,
        obs_noise_init=0.1,
    )
    before = {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
        if name
        in {
            "q_u_loc_unconstrained",
            "q_u_tril_unconstrained",
            "log_process_noise_unconstrained",
            "log_obs_noise_unconstrained",
        }
    }
    synthetic = generate_synthetic_sequences(
        num_sequences=4,
        sequence_length=8,
        state_dim=2,
        obs_dim=1,
        observation_gain=1.0,
        process_noise=0.05,
        obs_noise=0.05,
    )
    dataset = TimeseriesWindowDataset(
        sequences=synthetic.observations,
        lengths=synthetic.lengths,
        window_length=4,
        latents=synthetic.latents,
    )
    loader = build_dataloader(dataset, batch_size=2, shuffle=False)
    trainer = SVITrainer(
        model,
        TrainerConfig(
            steps=1,
            lr=1e-3,
            gradient_clip=5.0,
            report_every=1,
            checkpoint_dir=tmp_path,
            elbo="trace",
            eval_every=1,
        ),
    )
    batch = next(iter(loader))
    loss = trainer.svi.step(batch["y"], batch.get("lengths"))
    assert math.isfinite(loss)

    after = model.state_dict()
    for name, before_tensor in before.items():
        diff = (after[name] - before_tensor).abs().max().item()
        assert diff > 0.0


def test_markov_variational_posterior_runs(tmp_path) -> None:
    pyro.clear_param_store()
    torch.manual_seed(1)
    kernel = ARDRBFKernel(input_dim=2)
    transition = SparseGPTransition(state_dim=2, num_inducing=4, kernel=kernel)
    encoder = StateEncoder(
        obs_dim=1,
        state_dim=2,
        hidden_size=8,
        num_layers=1,
        structured=True,
        max_transition_scale=0.6,
    )
    model = SparseVariationalGPSSM(
        transition=transition,
        encoder=encoder,
        observation=None,
        obs_dim=1,
        process_noise_init=0.1,
        obs_noise_init=0.1,
        structured_q=True,
    )
    synthetic = generate_synthetic_sequences(
        num_sequences=3,
        sequence_length=10,
        state_dim=2,
        obs_dim=1,
        observation_gain=1.0,
        process_noise=0.05,
        obs_noise=0.05,
    )
    dataset = TimeseriesWindowDataset(
        sequences=synthetic.observations,
        lengths=synthetic.lengths,
        window_length=6,
        latents=synthetic.latents,
    )
    loader = build_dataloader(dataset, batch_size=3, shuffle=False)
    trainer = SVITrainer(
        model,
        TrainerConfig(
            steps=1,
            lr=1e-3,
            gradient_clip=5.0,
            report_every=1,
            checkpoint_dir=tmp_path,
            elbo="trace",
            eval_every=1,
        ),
    )
    batch = next(iter(loader))
    loss = trainer.svi.step(batch["y"], batch.get("lengths"))
    assert math.isfinite(loss)
