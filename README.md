<div align="center">
  <h1>Gaussian Process State-Space Models</h1>
  <h3>GPSSM — Sparse Variational State-Space Modeling in PyTorch &amp; Pyro</h3>
</div>

<p align="center">
  <a href="README_CN.md">中文</a>
</p>

<p align="center">
  <a href="#1-installation">Installation</a> •
  <a href="#2-quick-start">Quick Start</a> •
  <a href="#3-problem-setting--core-concepts">Core Concepts</a> •
  <a href="#4-architecture--key-modules">Architecture</a> •
  <a href="#5-data--evaluation">Data &amp; Evaluation</a> •
  <a href="#6-extensibility">Extensibility</a> •
  <a href="#7-development--testing">Development</a> •
  <a href="#8-code-structure">Code Structure</a> •
  <a href="#9-references">References</a>
</p>

---

This repository provides a minimal, research-grade implementation of **sparse variational Gaussian process state-space models (GPSSMs)**.
The goal is convenience: generate time series, train a GPSSM, evaluate forecasting metrics, and run multi-step rollouts **without writing boilerplate** — a single CLI entry point plus a YAML config is enough.

---

## 1. Installation

We recommend an isolated virtual environment with Python ≥ 3.10. See `pyproject.toml` for the authoritative dependency list.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install "torch>=2.2.0,<2.3.0" "pyro-ppl>=1.9.1,<1.10" numpy pyyaml typer rich
pip install -e .
```

`pip install -e .` registers the package `gp-ssm` in editable mode and exposes the modules under `src/`.

For reproducing experiments or running tests, install the optional development dependencies:

```bash
pip install pytest black ruff mypy
```

---

## 2. Quick Start

### Train a model

```bash
python scripts/train_gp_ssm.py train --config configs/default.yaml
```

- The script automatically selects **MPS / CUDA / CPU** depending on availability and prints the device.
- `--config` can be any YAML file; see `configs/system_id_medium.yaml` for a system-identification setting.
- During training the script periodically reports **RMSE / NLL** on the validation split and, at the end, evaluates both validation and test sets and prints a **rollout forecast RMSE**.

### Evaluate & forecast

The training utilities expose two functions that you can reuse in your own scripts:

- `training.evaluation.evaluate_model(model, loader)`
  Computes observation RMSE, NLL, and (if latent ground truth is present) latent RMSE.
- `training.evaluation.rollout_forecast(model, y_hist, lengths, steps)`
  Uses the amortized posterior together with the GP transition mean to produce multi-step forecasts, suitable for system identification and control experiments.

You can import and reuse the model / data / inference modules directly:

```python
from models.gp_ssm import SparseVariationalGPSSM
from data.timeseries import generate_synthetic_sequences
from training.evaluation import evaluate_model, rollout_forecast
```

---

## 3. Problem Setting & Core Concepts

We focus on **nonlinear, discrete-time state-space models** of the form

$$
  x_t = f(x_{t-1}, u_{t-1}) + \varepsilon_t,\quad
  y_t = C x_t + d + \eta_t,
$$

where $f$ is an unknown nonlinear transition, modeled here as a **vector-valued Gaussian process** with an ARD RBF kernel. The state dimension is typically modest, but the dynamics and observation processes may be strongly nonlinear.

Key ideas:

- **Nonparametric dynamics.** Instead of a fixed parametric form, the transition function $f$ is given a GP prior. This yields flexible dynamics with explicit uncertainty over both states and transition function values.
- **Sparse variational GPSSM.** Following the formulation of Frigola, Chen & Rasmussen (2014), we introduce inducing variables $u = f(Z)$ at inducing locations $Z$ and optimize a variational free energy objective, trading off model capacity and computational cost while keeping the posterior tractable.
- **Amortized state inference.** Inspired by the Probabilistic Recurrent State-Space Model (PR-SSM) of Doerr et al. (2018), we use a recurrent encoder (bi-GRU) to parameterize the variational distribution $q(x_{0:T})$. This scales to long sequences and reuses an encoder across many trajectories.
- **Fast SVI training.** The combination of a sparse GP transition, amortized inference, and Pyro’s stochastic variational inference enables scalable training on batches of windowed sequences with automatic differentiation.

This implementation is intentionally **small and explicit**: it is meant as a starting point for research on GPSSMs, not as a monolithic framework.

---

## 4. Architecture & Key Modules

Core components live under `src/`:

| Component | Description |
| --- | --- |
| `models.gp_ssm.SparseVariationalGPSSM` | PyroModule that combines a GP transition, encoder, and observation head. Process / observation noise and initial state are `PyroParam`s in log-space to enforce positivity. Supports `Trace_ELBO` and `TraceMeanField_ELBO`. |
| `models.transition.SparseGPTransition` | Sparse GP transition model with ARD RBF kernel and inducing points. `prior()` returns the GP prior over inducing targets; `precompute()` caches the Kzz inverse and projected inducing mean for efficient reuse. |
| `models.kernels.*` | Minimal `Kernel` interface plus built-ins: `ARDRBFKernel`, `MaternKernel (ν ∈ {1/2, 3/2, 5/2})`, `RationalQuadraticKernel`, `PeriodicKernel`, and compositional `SumKernel` / `ProductKernel`. All are configurable from the YAML config. |
| `models.encoder.StateEncoder` | Bi-directional GRU encoder producing per-time-step mean/scale for the latent state and a separate initial state posterior; supports padded batches via `pack_padded_sequence`. |
| `models.observation.AffineObservationModel` | Default linear observation model $y_t = C x_t + d$. Easily replaceable with nonlinear decoders for richer observation models. |
| `inference.svi.SVITrainer` | Thin wrapper around Pyro’s `SVI` with `ClippedAdam`, progress reporting, configurable ELBO, and checkpoint saving. |
| `data.timeseries.TimeseriesWindowDataset` | Windowed time-series dataset with padding and length masks; pairs with `build_dataloader` for batched loading. |
| `training.evaluation` | Implements `evaluate_model` and `rollout_forecast` for standard metrics and open-loop forecasting. |

For a more narrative view of the architecture and extension hooks, see `docs/architecture.md`.

---

## 5. Data & Evaluation

### Synthetic data generators

- **Toy GPSSM-style system** — `generate_synthetic_sequences`
  Uses a sinusoidal drift in latent space plus a linear observation map. This is useful for quickly sanity-checking training and metrics.
- **System identification benchmark** — `generate_system_identification_sequences`
  Simulates control inputs, nonlinear drift (linear + tanh term), and observation noise, closer to a realistic controlled dynamical system.

Both return a `SequenceDataset` with observations, lengths, and optional latent states.

### Dataset splitting & windowing

- `split_dataset(dataset, (train, val, test), seed)` splits sequences by index with a fixed random seed for reproducibility.
- `TimeseriesWindowDataset` takes sequences and lengths and produces fixed-length windows, with zero-padding and length masks. Latent states are windowed consistently when available.
- `build_dataloader` wraps this in a `torch.utils.data.DataLoader` with a custom collate function that packs `y`, `lengths`, and optional `latents`.

### Metrics

- **RMSE** — root-mean-squared error on observations, masked by sequence lengths.
- **NLL** — analytic Gaussian negative log-likelihood under the observation noise estimated by the model.
- **Latent RMSE** — optional, when latent ground truth is present in the dataset.
- **Forecast RMSE** — computed via `rollout_forecast` in `scripts/train_gp_ssm.py` for a chosen context window and horizon.

These metrics align with the evaluation style in the original variational GPSSM and PR-SSM works, while keeping the implementation compact. 

---

## 6. Extensibility

The code is designed to be modified:

- **Kernel swaps.** Built-in options now include ARD RBF, Matérn (ν ∈ {1/2, 3/2, 5/2}), Rational Quadratic, Periodic, and Sum / Product combinators. Select them via `model.kernel`:

  ```yaml
  model:
    kernel:
      type: sum              # ard_rbf | matern | rational_quadratic | periodic | sum | product
      jitter: 1.0e-5
      components:
        - type: ard_rbf
        - type: periodic
          params:
            period: 24.0
            lengthscale: 0.5
  ```

  Custom kernels simply need to subclass `models.kernels.Kernel` and implement `forward`/`diag`; they can then be referenced with a new `type` string.
- **Custom observation / encoder.** Any PyTorch / PyroModule can be used as the observation head or encoder. For example, you can plug in a CNN-based decoder for images or a Transformer encoder for long, irregular sequences.
- **Control inputs.** To model controlled systems, extend the transition to take concatenated `(x_t, u_t)` as input. The synthetic system-identification generator already simulates control signals, which you can feed into a modified transition.
- **New experiment configs.** Add YAML files under `configs/` to define new experimental regimes (state dimension, number of inducing points, window length, noise scales, etc.). The training script reads these without changes.
- **Checkpointing & continuation.** `SVITrainer` saves a `final.pt` checkpoint containing the Pyro parameter store. You can load it via `pyro.get_param_store().load_state_dict` to resume training or run additional evaluation.

---

## 7. Development & Testing

Run the test suite:

```bash
pytest tests/test_models.py
```

This covers:

- basic kernel shape / positivity checks,
- GP transition predictive shapes,
- a single SVI training step on synthetic data, and
- dataset splitting & window extraction for the system-identification generator.

When adding new kernels, transitions, or observation models, you should mirror these tests with additional shape and stability checks.

Coding style guidelines (soft, not enforced):

- Prefer guard clauses over deeply nested conditionals.
- Make dependencies explicit (pass them in; avoid hidden global state).
- Keep components small and composable.

Issues and pull requests are welcome. If you extend the library (new kernels, transitions, observation models, or training utilities), please add a small test and, when possible, a short note in `docs/` or `configs/` so others can reuse your work.

---

## 8. Code Structure

```text
src/
  data/         # synthetic datasets and loaders
  inference/    # SVI trainer and ELBO wrappers
  models/       # GPSSM, kernels, transition, encoder, observation
  training/     # evaluation utilities (metrics, rollouts)

scripts/
  train_gp_ssm.py   # CLI entry point for training & evaluation

configs/
  *.yaml       # experiment configurations (toy, system-id, etc.)

docs/
  architecture.md   # high-level modeling and design notes

tests/
  test_models.py    # kernel, transition, dataset, and SVI smoke tests
```

---

## 9. References

 **Variational Gaussian Process State-Space Models**
   Roger Frigola, Yutian Chen, Carl E. Rasmussen.
   *Advances in Neural Information Processing Systems 27 (NeurIPS), 2014.*

 **Probabilistic Recurrent State-Space Models (PR-SSM)**
   Andreas Doerr, Christian Daniel, Martin Schiegg, Duy Nguyen-Tuong, Stefan Schaal, Marc Toussaint, Sebastian Trimpe.
   *Proceedings of the 35th International Conference on Machine Learning (ICML), PMLR 80, 2018.*

Project-internal references:

- `docs/architecture.md` — more detailed notes on modeling choices, variational guides, training loop, and extension hooks.
- `configs/` — experimental configurations for toy and system-identification settings.
- `pyproject.toml` — project metadata and dependency constraints.

---

<div align="center">
This repository is licensed under the MIT License.
</div>
