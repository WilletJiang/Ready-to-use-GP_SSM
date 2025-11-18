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
    with torch.no_grad():
        for batch in loader:
            y = batch["y"]
            lengths = batch["lengths"]
            encoding = model.encoder(y, lengths)
            loc = encoding.loc
            batch_size, horizon, _ = loc.shape
            mask = _mask_from_lengths(lengths, horizon).to(y.device).unsqueeze(-1)
            flat_loc = loc.reshape(batch_size * horizon, state_dim)
            obs_pred = model.observation(flat_loc).reshape(batch_size, horizon, obs_dim)
            diff = y - obs_pred
            rmse_sum += (diff.pow(2) * mask).sum().item()
            obs_count += mask.sum().item() * obs_dim
            obs_noise = model._obs_noise().detach().to(y.device)
            var = obs_noise.pow(2)
            log_norm = torch.log(2 * math.pi * var)
            nll = 0.5 * (
                diff.pow(2) / var + log_norm
            )
            nll_sum += (nll * mask).sum().item()
            if "latents" in batch:
                lat = batch["latents"]
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


def rollout_forecast(model, y_hist: Tensor, lengths: Tensor, steps: int) -> Tensor:
    if steps <= 0:
        msg = "steps must be positive"
        raise ValueError(msg)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        encoding = model.encoder(y_hist, lengths)
        batch_size = y_hist.size(0)
        indices = (lengths - 1).clamp(min=0)
        batch_idx = torch.arange(batch_size, device=encoding.loc.device)
        x_prev = encoding.loc[batch_idx, indices]
        u_mean = model.q_u_loc.detach().reshape(
            model.state_dim,
            model.transition.num_inducing,
        ).to(x_prev.device)
        cache = model.transition.precompute(u_mean)
        preds = []
        for _ in range(steps):
            mean, _ = model.transition(x_prev, u_mean, cache)
            obs_pred = model.observation(mean)
            preds.append(obs_pred.unsqueeze(1))
            x_prev = mean
        forecasts = torch.cat(preds, dim=1)
    if was_training:
        model.train()
    return forecasts
