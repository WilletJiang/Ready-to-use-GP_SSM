from __future__ import annotations

import math
from typing import Dict, Optional

import torch
from torch import Tensor


def _mask_from_lengths(lengths: Tensor, horizon: int) -> Tensor:
    device = lengths.device
    time = torch.arange(horizon, device=device).unsqueeze(0)
    return (time < lengths.unsqueeze(1)).to(torch.bool)


def evaluate_model(model, loader) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    rmse_sum = 0.0
    nll_sum = 0.0
    obs_count = 0
    latent_rmse_sum = 0.0
    latent_count = 0
    obs_dim = model.obs_dim
    state_dim = model.state_dim
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    with torch.no_grad():
        for batch in loader:
            y = batch["y"].to(device)
            lengths = batch["lengths"].to(device)
            encoding = model.encoder(y, lengths)
            loc = encoding.loc
            batch_size, horizon, _ = loc.shape
            mask = _mask_from_lengths(lengths, horizon).to(device).unsqueeze(-1)
            flat_loc = loc.reshape(batch_size * horizon, state_dim)
            obs_pred = model.observation(flat_loc).reshape(batch_size, horizon, obs_dim)
            diff = y - obs_pred
            rmse_sum += (diff.pow(2) * mask).sum().item()
            obs_count += mask.sum().item() * obs_dim
            obs_noise = model._obs_noise().detach().to(device)
            var = obs_noise.pow(2)
            log_norm = torch.log(2 * math.pi * var)
            nll = 0.5 * (
                diff.pow(2) / var + log_norm
            )
            nll_sum += (nll * mask).sum().item()
            if "latents" in batch:
                lat = batch["latents"].to(device)
                latent_diff = loc - lat
                latent_rmse_sum += (latent_diff.pow(2) * mask).sum().item()
                latent_count += mask.sum().item() * state_dim
    if was_training:
        model.train()
    metrics = {
        "rmse": math.sqrt(rmse_sum / max(obs_count, 1)),
        "nll": nll_sum / max(obs_count, 1),
    }
    if latent_count > 0:
        metrics["latent_rmse"] = math.sqrt(latent_rmse_sum / latent_count)
    return metrics


def rollout_forecast(
    model,
    y_hist: Tensor,
    lengths: Tensor,
    steps: int,
    *,
    num_samples: int = 64,
    return_std: bool = True,
    moment_matching: bool = False,
    return_samples: bool = False,
):
    """Multi-step forecasting with uncertainty propagation.

    Args:
        model: GPSSM model with transition/observation modules.
        y_hist: Observation history `[batch, time, obs_dim]`.
        lengths: Valid lengths for each sequence `[batch]`.
        steps: Forecast horizon (>0).
        num_samples: Number of Monte Carlo particles. If <=0, fall back to
            moment matching (only reliable for affine observation models).
        return_std: Whether to return predictive standard deviation.
        moment_matching: Force analytic (approximate) propagation even when
            `num_samples > 0`.
        return_samples: Whether to return raw samples `[S, B, steps, obs_dim]`.

    Returns:
        If `return_std` is True: dict with keys `mean`, `std`, and optionally
        `samples`. Otherwise returns only the predictive mean tensor to remain
        backward compatible.
    """

    if steps <= 0:
        msg = "steps must be positive"
        raise ValueError(msg)
    if num_samples < 0:
        msg = "num_samples must be non-negative"
        raise ValueError(msg)

    was_training = getattr(model, "training", False)
    model.eval()
    device = None
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = y_hist.device

    with torch.no_grad():
        encoding = model.encoder(y_hist, lengths)
        batch_size = y_hist.size(0)
        indices = (lengths - 1).clamp(min=0)
        batch_idx = torch.arange(batch_size, device=encoding.loc.device)
        state_dim = model.state_dim
        obs_dim = model.obs_dim

        init_mean = encoding.loc[batch_idx, indices]
        init_scale = encoding.scale[batch_idx, indices]
        process_noise = model._process_noise().to(init_mean.device)
        obs_noise = model._obs_noise().to(init_mean.device)
        u_mean = model.q_u_loc.detach().reshape(
            state_dim,
            model.transition.num_inducing,
        ).to(init_mean.device)
        cache = model.transition.precompute(u_mean)

        if num_samples > 0 and not moment_matching:
            # Monte Carlo sampling of transition + observation noise
            particles = init_mean.unsqueeze(0).expand(num_samples, -1, -1)
            if init_scale is not None:
                noise = torch.randn_like(particles) * init_scale.unsqueeze(0)
                particles = particles + noise
            samples = []
            for _ in range(steps):
                flat_particles = particles.reshape(-1, state_dim)
                mean, var = model.transition(flat_particles, u_mean, cache)
                mean = mean.view(num_samples, batch_size, state_dim)
                var = var.view_as(mean)
                total_state_std = (var + process_noise).clamp_min(1e-9).sqrt()
                particles = mean + torch.randn_like(mean) * total_state_std
                obs_loc = model.observation(particles.reshape(-1, state_dim))
                obs_loc = obs_loc.view(num_samples, batch_size, obs_dim)
                obs = obs_loc + torch.randn_like(obs_loc) * obs_noise
                samples.append(obs)
            stacked = torch.stack(samples, dim=2)  # [S, B, steps, obs_dim]
            mean = stacked.mean(dim=0)
            std = stacked.std(dim=0, unbiased=False)
            result = {"mean": mean}
            if return_std:
                result["std"] = std
            if return_samples:
                result["samples"] = stacked
        else:
            # Analytic moment matching (approximate; assumes affine observation)
            weight = None
            bias = None
            linear = getattr(model.observation, "linear", None)
            if linear is not None:
                weight = linear.weight  # [obs_dim, state_dim]
                bias = linear.bias
            preds = []
            stds = [] if return_std else None
            x_prev = init_mean
            for _ in range(steps):
                mean, var = model.transition(x_prev, u_mean, cache)
                total_state_var = var + process_noise
                obs_loc = model.observation(mean)
                preds.append(obs_loc.unsqueeze(1))
                if return_std:
                    if weight is None:
                        # Fallback: treat observation as identity mapping
                        obs_var = total_state_var
                    else:
                        weight_sq_t = weight.pow(2).t()  # [state_dim, obs_dim]
                        obs_var = total_state_var @ weight_sq_t
                    obs_var = obs_var + obs_noise.pow(2)
                    stds.append(obs_var.clamp_min(1e-9).sqrt().unsqueeze(1))
                x_prev = mean
            mean = torch.cat(preds, dim=1)
            result = {"mean": mean}
            if return_std:
                std_tensor = torch.cat(stds, dim=1)
                result["std"] = std_tensor
        if was_training:
            model.train()
        return result if return_std or return_samples else result["mean"]
