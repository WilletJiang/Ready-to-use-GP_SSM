from __future__ import annotations

import math
from typing import Dict, Optional

import torch
from torch import Tensor
from torch.distributions import MultivariateNormal, Normal

_SQRT_PI_INV = 1.0 / math.sqrt(math.pi)
_STD_NORMAL = None


def _std_normal(device: torch.device, dtype: torch.dtype) -> Normal:
    """Lazily constructed standard normal (avoids device mismatches)."""
    global _STD_NORMAL
    if _STD_NORMAL is None or _STD_NORMAL.loc.device != device:
        _STD_NORMAL = Normal(
            torch.zeros(1, device=device, dtype=dtype),
            torch.ones(1, device=device, dtype=dtype),
        )
    return _STD_NORMAL


def gaussian_crps(mean: Tensor, std: Tensor, target: Tensor) -> Tensor:
    """Closed-form CRPS for a Gaussian predictive distribution.

    CRPS(N(μ,σ²), y) = σ [z(2Φ(z)−1) + 2φ(z) − 1/√π]
    where z = (y−μ)/σ.

    Returns element-wise CRPS with the same shape as *target*.
    """
    z = (target - mean) / std.clamp_min(1e-12)
    sn = _std_normal(z.device, z.dtype)
    phi = sn.log_prob(z).exp()
    big_phi = sn.cdf(z)
    return std * (z * (2.0 * big_phi - 1.0) + 2.0 * phi - _SQRT_PI_INV)


def _mask_from_lengths(lengths: Tensor, horizon: int) -> Tensor:
    device = lengths.device
    time = torch.arange(horizon, device=device).unsqueeze(0)
    return (time < lengths.unsqueeze(1)).to(torch.bool)


def _aligned_posterior_sequence(encoding, structured_q: bool) -> tuple[Tensor, Tensor]:
    batch_size, horizon, state_dim = encoding.loc.shape
    device = encoding.loc.device
    dtype = encoding.loc.dtype
    state_mean = torch.zeros(batch_size, horizon, state_dim, device=device, dtype=dtype)
    state_scale = torch.zeros_like(state_mean)
    state_mean[:, 0, :] = encoding.init_loc
    state_scale[:, 0, :] = encoding.init_scale
    if horizon == 1:
        return state_mean, state_scale
    if structured_q:
        trans_matrix = encoding.trans_matrix
        trans_bias = encoding.trans_bias
        if trans_matrix is None or trans_bias is None:
            msg = "Encoder must produce transition parameters when structured_q is True"
            raise RuntimeError(msg)
        x_prev = encoding.init_loc
        for t in range(horizon - 1):
            aff = torch.einsum("bij,bj->bi", trans_matrix[:, t, :, :], x_prev)
            mean_next = aff + trans_bias[:, t, :]
            state_mean[:, t + 1, :] = mean_next
            state_scale[:, t + 1, :] = encoding.scale[:, t, :]
            x_prev = mean_next
    else:
        state_mean[:, 1:, :] = encoding.loc[:, : horizon - 1, :]
        state_scale[:, 1:, :] = encoding.scale[:, : horizon - 1, :]
    return state_mean, state_scale


def _sample_u_posterior(model, num_samples: int, device: torch.device) -> Tensor:
    state_dim = model.state_dim
    num_inducing = model.transition.num_inducing
    q_u_loc = model.q_u_loc.detach().to(device)
    q_u_tril = getattr(model, "q_u_tril", None)
    if q_u_tril is None:
        flat = q_u_loc.unsqueeze(0).expand(num_samples, -1)
    else:
        posterior = MultivariateNormal(q_u_loc, scale_tril=q_u_tril.detach().to(device))
        flat = posterior.rsample((num_samples,))
    return flat.reshape(num_samples, state_dim, num_inducing)


def _sample_rollout_observations(
    model,
    init_mean: Tensor,
    init_scale: Tensor,
    steps: int,
    *,
    num_samples: int,
    controls: Optional[Tensor] = None,
    sample_observations: bool = True,
) -> Tensor:
    if steps <= 0:
        msg = "steps must be positive"
        raise ValueError(msg)
    if num_samples <= 0:
        msg = "num_samples must be positive"
        raise ValueError(msg)

    batch_size = init_mean.size(0)
    obs_dim = model.obs_dim
    device = init_mean.device
    process_noise = model._process_noise().to(device)
    process_var = process_noise.pow(2)
    obs_noise = model._obs_noise().to(device)
    u_samples = _sample_u_posterior(model, num_samples, device)
    rollout_samples = []

    if controls is not None and controls.shape[:2] != (batch_size, steps):
        msg = "controls must align with rollout batch and step dimensions"
        raise ValueError(msg)

    for sample_idx in range(num_samples):
        particles = init_mean.clone()
        if init_scale is not None:
            particles = particles + torch.randn_like(particles) * init_scale
        u_sample = u_samples[sample_idx]
        cache = model.transition.precompute(u_sample)
        sample_steps = []
        for step_idx in range(steps):
            control_step = controls[:, step_idx, :] if controls is not None else None
            mean, var = model.transition(
                particles,
                u_sample,
                cache,
                controls=control_step,
            )
            total_state_std = (var + process_var).clamp_min(1e-9).sqrt()
            particles = mean + torch.randn_like(mean) * total_state_std
            obs_loc = model.observation(particles)
            if sample_observations:
                obs = obs_loc + torch.randn_like(obs_loc) * obs_noise
            else:
                obs = obs_loc
            sample_steps.append(obs)
        stacked_steps = torch.stack(sample_steps, dim=1)
        rollout_samples.append(stacked_steps)
    return torch.stack(rollout_samples, dim=0).reshape(
        num_samples, batch_size, steps, obs_dim
    )


def _split_holdout_batch(
    y: Tensor,
    lengths: Tensor,
    predictive_horizon: int,
    controls: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]:
    if predictive_horizon <= 0:
        msg = "predictive_horizon must be positive"
        raise ValueError(msg)

    target_lengths = torch.minimum(
        torch.full_like(lengths, predictive_horizon),
        (lengths - 1).clamp_min(0),
    )
    context_lengths = lengths - target_lengths
    max_context = int(context_lengths.max().item())
    max_target = int(target_lengths.max().item())
    context_y = y[:, :max_context, :]
    target_y = torch.zeros(
        y.size(0), max_target, y.size(-1), device=y.device, dtype=y.dtype
    )
    target_mask = torch.zeros(y.size(0), max_target, device=y.device, dtype=torch.bool)
    future_controls = None
    if controls is not None:
        future_controls = torch.zeros(
            controls.size(0),
            max_target,
            controls.size(-1),
            device=controls.device,
            dtype=controls.dtype,
        )

    for idx in range(y.size(0)):
        target_length = int(target_lengths[idx].item())
        if target_length == 0:
            continue
        context_length = int(context_lengths[idx].item())
        end = context_length + target_length
        target_y[idx, :target_length, :] = y[idx, context_length:end, :]
        target_mask[idx, :target_length] = True
        if future_controls is not None:
            future_controls[idx, :target_length, :] = controls[
                idx, context_length:end, :
            ]

    return context_y, context_lengths, target_y, target_mask, future_controls


def _predictive_loglikelihood(
    model,
    y: Tensor,
    lengths: Tensor,
    controls: Optional[Tensor],
    *,
    predictive_horizon: int,
    predictive_num_samples: int,
) -> tuple[float, int]:
    context_y, context_lengths, target_y, target_mask, future_controls = (
        _split_holdout_batch(
            y,
            lengths,
            predictive_horizon,
            controls=controls,
        )
    )
    valid = target_mask.any(dim=1)
    if not torch.any(valid):
        return 0.0, 0

    context_y = context_y[valid]
    context_lengths = context_lengths[valid]
    target_y = target_y[valid]
    target_mask = target_mask[valid]
    if future_controls is not None:
        future_controls = future_controls[valid]

    encoding = model.encoder(context_y, context_lengths)
    state_mean, state_scale = _aligned_posterior_sequence(
        encoding,
        getattr(model, "structured_q", False),
    )
    batch_size = context_y.size(0)
    batch_idx = torch.arange(batch_size, device=context_y.device)
    last_indices = context_lengths - 1
    init_mean = state_mean[batch_idx, last_indices]
    init_scale = state_scale[batch_idx, last_indices]
    obs_loc_samples = _sample_rollout_observations(
        model,
        init_mean,
        init_scale,
        steps=target_y.size(1),
        num_samples=predictive_num_samples,
        controls=future_controls,
        sample_observations=False,
    )
    obs_noise = model._obs_noise().to(target_y.device).view(1, 1, 1, -1)
    target_expanded = target_y.unsqueeze(0).expand(predictive_num_samples, -1, -1, -1)
    log_probs = Normal(obs_loc_samples, obs_noise).log_prob(target_expanded).sum(dim=-1)
    log_probs = log_probs * target_mask.unsqueeze(0)
    sample_log_probs = log_probs.sum(dim=-1)
    predictive_loglik = torch.logsumexp(sample_log_probs, dim=0) - math.log(
        predictive_num_samples
    )
    obs_count = int(target_mask.sum().item() * model.obs_dim)
    return float(predictive_loglik.sum().item()), obs_count


def evaluate_model(
    model,
    loader,
    *,
    predictive_horizon: int = 1,
    predictive_num_samples: int = 64,
) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    rmse_sum = 0.0
    reconstruction_nll_sum = 0.0
    crps_sum = 0.0
    obs_count = 0
    latent_rmse_sum = 0.0
    latent_count = 0
    predictive_loglik_sum = 0.0
    predictive_obs_count = 0
    obs_dim = model.obs_dim
    state_dim = model.state_dim
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    with torch.inference_mode():
        for batch in loader:
            y = batch["y"].to(device)
            lengths = batch["lengths"].to(device)
            controls = batch.get("controls")
            if controls is not None:
                controls = controls.to(device)
            encoding = model.encoder(y, lengths)
            state_mean, _ = _aligned_posterior_sequence(
                encoding,
                getattr(model, "structured_q", False),
            )
            batch_size, horizon, _ = state_mean.shape
            mask = _mask_from_lengths(lengths, horizon).to(device).unsqueeze(-1)
            flat_mean = state_mean.reshape(batch_size * horizon, state_dim)
            obs_pred = model.observation(flat_mean).reshape(
                batch_size, horizon, obs_dim
            )
            diff = y - obs_pred
            rmse_sum += (diff.pow(2) * mask).sum().item()
            obs_count += int(mask.sum().item() * obs_dim)
            obs_noise = model._obs_noise().detach().to(device)
            var = obs_noise.pow(2)
            log_norm = torch.log(2 * math.pi * var)
            reconstruction_nll = 0.5 * (diff.pow(2) / var + log_norm)
            reconstruction_nll_sum += (reconstruction_nll * mask).sum().item()
            obs_noise_expanded = obs_noise.expand_as(y)
            crps_vals = gaussian_crps(obs_pred, obs_noise_expanded, y)
            crps_sum += (crps_vals * mask).sum().item()
            if "latents" in batch:
                lat = batch["latents"].to(device)
                latent_diff = state_mean - lat
                latent_rmse_sum += (latent_diff.pow(2) * mask).sum().item()
                latent_count += int(mask.sum().item() * state_dim)
            batch_predictive_loglik, batch_predictive_obs_count = (
                _predictive_loglikelihood(
                    model,
                    y,
                    lengths,
                    controls,
                    predictive_horizon=predictive_horizon,
                    predictive_num_samples=predictive_num_samples,
                )
            )
            predictive_loglik_sum += batch_predictive_loglik
            predictive_obs_count += batch_predictive_obs_count
    if was_training:
        model.train()
    metrics = {
        "rmse": math.sqrt(rmse_sum / max(obs_count, 1)),
        "reconstruction_nll": reconstruction_nll_sum / max(obs_count, 1),
        "crps": crps_sum / max(obs_count, 1),
    }
    if predictive_obs_count > 0:
        avg_predictive_loglik = predictive_loglik_sum / predictive_obs_count
        metrics["predictive_loglik"] = avg_predictive_loglik
        metrics["predictive_nll"] = -avg_predictive_loglik
    if latent_count > 0:
        metrics["latent_rmse"] = math.sqrt(latent_rmse_sum / latent_count)
    return metrics


def rollout_forecast(
    model,
    y_hist: Tensor,
    lengths: Tensor,
    steps: int,
    controls: Optional[Tensor] = None,
    *,
    num_samples: int = 64,
    return_std: bool = True,
    moment_matching: bool = False,
    return_samples: bool = False,
):
    """Multi-step forecasting with uncertainty propagation."""

    if steps <= 0:
        msg = "steps must be positive"
        raise ValueError(msg)
    if num_samples < 0:
        msg = "num_samples must be non-negative"
        raise ValueError(msg)

    was_training = getattr(model, "training", False)
    model.eval()
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = y_hist.device

    if controls is not None and controls.shape[:2] != (y_hist.size(0), steps):
        msg = "controls must align with forecast batch and step dimensions"
        raise ValueError(msg)
    if controls is not None:
        controls = controls.to(device)

    with torch.inference_mode():
        encoding = model.encoder(y_hist, lengths)
        state_mean, state_scale = _aligned_posterior_sequence(
            encoding,
            getattr(model, "structured_q", False),
        )
        batch_size = y_hist.size(0)
        indices = (lengths - 1).clamp(min=0)
        batch_idx = torch.arange(batch_size, device=encoding.loc.device)
        init_mean = state_mean[batch_idx, indices]
        init_scale = state_scale[batch_idx, indices]
        if num_samples > 0 and not moment_matching:
            stacked = _sample_rollout_observations(
                model,
                init_mean,
                init_scale,
                steps=steps,
                num_samples=num_samples,
                controls=controls,
                sample_observations=True,
            )
            mean = stacked.mean(dim=0)
            std = stacked.std(dim=0, unbiased=False)
            result = {"mean": mean}
            if return_std:
                result["std"] = std
            if return_samples:
                result["samples"] = stacked
        else:
            process_noise = model._process_noise().to(init_mean.device)
            process_var = process_noise.pow(2)
            obs_noise = model._obs_noise().to(init_mean.device)
            u_mean = (
                model.q_u_loc.detach()
                .reshape(
                    model.state_dim,
                    model.transition.num_inducing,
                )
                .to(init_mean.device)
            )
            cache = model.transition.precompute(u_mean)
            weight = None
            linear = getattr(model.observation, "linear", None)
            if linear is not None:
                weight = linear.weight
            preds = []
            stds = [] if return_std else None
            x_prev = init_mean
            for step_idx in range(steps):
                control_step = (
                    controls[:, step_idx, :] if controls is not None else None
                )
                mean, var = model.transition(
                    x_prev, u_mean, cache, controls=control_step
                )
                total_state_var = var + process_var
                obs_loc = model.observation(mean)
                preds.append(obs_loc.unsqueeze(1))
                if return_std:
                    if weight is None:
                        obs_var = total_state_var
                    else:
                        weight_sq_t = weight.pow(2).t()
                        obs_var = total_state_var @ weight_sq_t
                    obs_var = obs_var + obs_noise.pow(2)
                    stds.append(obs_var.clamp_min(1e-9).sqrt().unsqueeze(1))
                x_prev = mean
            mean = torch.cat(preds, dim=1)
            result = {"mean": mean}
            if return_std:
                result["std"] = torch.cat(stds, dim=1)
        if was_training:
            model.train()
        return result if return_std or return_samples else result["mean"]
