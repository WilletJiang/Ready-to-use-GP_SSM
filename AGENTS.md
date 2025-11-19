# GPSSM Coding Guidelines

This describes the coding conventions for the GPSSM project.
The goal is simple: **small, explicit, research‑grade code that is easy to extend and hard to misuse.**

---

## 1. General Principles

- Prefer **clarity over cleverness** and **guard clauses over deep nesting**.
- Fail fast on invalid inputs with explicit errors (usually `ValueError`), never silently ignore or auto‑correct unless strictly numerical (e.g. `clamp_min` for stability).
- Keep functions and modules **small and composable**; avoid “god objects”.
- Avoid hidden global state. Make dependencies explicit and injectable.
- Public APIs should be **stable and documented**; experimental or internal helpers should be clearly scoped (module‑private or underscored).

---

## 2. Project Structure & Naming

High‑level layout (see also `README.md` and `docs/architecture.md`):

- `src/data` – synthetic datasets, `SequenceDataset`, `TimeseriesWindowDataset`, and dataloader helpers.
- `src/models` – GPSSM core: kernels, transitions, encoder, observation models, and `SparseVariationalGPSSM`.
- `src/inference` – SVI trainer and ELBO wiring.
- `src/training` – evaluation utilities (metrics, rollouts).
- `scripts` – CLI entry points (currently `train_gp_ssm.py`).
- `configs` – YAML experiment configs (toy, system‑id, etc.).
- `tests` – pytest suites.

Naming rules:

- Modules / files: lower_snake_case (`timeseries.py`, `gp_ssm.py`).
- Classes: `UpperCamelCase` (`SparseVariationalGPSSM`, `StateEncoder`).
- Functions: verbs in `lower_snake_case` (`evaluate_model`, `rollout_forecast`).
- Dataclasses: used for simple structured data (`SequenceDataset`, `TrainerConfig`, `EncoderOutput`).

---

## 3. Python Style

- Target **Python ≥ 3.10**; use type hints for all new public functions / methods.
- New code should prefer:
  - **guard clauses** at the top of functions instead of nested `if` chains;
  - local helpers over inline lambdas if complexity grows;
  - explicit imports (no `from module import *`).
- One‑letter names only in these cases:
  - loop indices (`i`, `j`, `k`),
  - standard math symbols (`x`, `y`, `t`, `n`),
  - short lambdas in obvious contexts.
  Everywhere else, choose descriptive names.
- When validating arguments, raise `ValueError` with a concrete, actionable message (see existing checks in `StateEncoder`, `TimeseriesWindowDataset`, `SVITrainer`, etc.).

Formatting / linting:

- Use **black** for formatting and **ruff** for linting; do not hand‑tune style against them.
- Recommended commands before opening a PR:

  ```bash
  black src tests scripts
  ruff check src tests scripts
  mypy src
  ```

---

## 4. PyTorch & Pyro Conventions

### 4.1 Devices & dtypes

- Always move tensors to the model’s device using:

  ```python
  device = next(model.parameters()).device
  y = batch["y"].to(device)
  ```

- Respect the existing device‑selection logic in `train_gp_ssm.py` (MPS → CUDA → CPU). New code should reuse `_select_device` rather than re‑inventing device checks.

### 4.2 Randomness & reproducibility

- Use explicit seeding utilities:
  - `_seed_everything(seed: int)` in scripts (PyTorch + Pyro),
  - per‑generator seeds in data generation (see `generate_system_identification_sequences`).
- Do **not** introduce hidden global RNG usage; either:
  - take a `torch.Generator` as an argument, or
  - construct a local generator with a clear seed.

### 4.3 Models, evaluation, and training loops

- Follow the `evaluate_model` / `rollout_forecast` pattern:
  - Save `was_training = model.training` and restore it at the end.
  - Use `torch.no_grad()` for evaluation code.
  - Use length‑based masks for variable‑length sequences.
- For Pyro models:
  - register submodules with `pyro.module` inside `model` and `guide`,
  - use `PyroParam` for learnable parameters with constraints (noise scales, initial states),
  - call `pyro.clear_param_store()` in training utilities that construct fresh SVI objects.

### 4.4 Shapes & layout

Default tensor shapes:

- Time series: `[batch, time, dim]`.
- Lengths: `[batch]` (dtype `torch.long`).
- Latent states: `[batch, time, state_dim]`.
- Inducing points: `[num_inducing, state_dim]`.

New components must respect these conventions or clearly document any deviations.

---

## 5. Data & Dataloaders

- Dataset classes should:
  - validate input tensor ranks and shapes early,
  - never silently drop data,
  - return dictionaries with explicit keys (`"y"`, `"length"`, `"latent"` / `"latents"`).
- Collate functions should:
  - stack tensors along the batch dimension,
  - ensure lengths remain 1D long tensors,
  - preserve optional latent information when present.
- Windowing should follow `TimeseriesWindowDataset`:
  - window extraction along the time axis,
  - zero‑padding when sequences are shorter than the window,
  - corresponding latent windows padded identically.

If you change windowing semantics, make sure tests in `tests/` are updated to reflect the new behavior.

---

## 6. Error Handling & Numerical Stability

- Use `ValueError` for invalid configuration or inputs (negative dimensions, non‑summing splits, etc.).
- Use `RuntimeError` only for genuinely unexpected internal failures.
- For numerical safety:
  - prefer log‑space parameters for strictly positive quantities (see `log_process_noise`, `log_obs_noise`),
  - use small epsilons and `clamp_min` only where needed, and keep values centralized (e.g. `min_noise` in `SparseVariationalGPSSM`),
  - document any non‑obvious clamping or stability tricks in the code (short comment is enough).

---

## 7. Tests

We use **pytest** for all tests (see `tests/test_models.py` for patterns).

When adding or modifying functionality:

- Add tests that:
  - check shapes,
  - assert positivity / constraints where relevant,
  - verify losses are finite for small synthetic problems,
  - validate masking / length handling for variable‑length sequences.
- Prefer testing **object behavior as a whole** (e.g. full forward call, SVI step) over micro‑testing private helpers, unless there is a clear numerical concern.
- Keep tests fast; use small dimensions and short sequences unless a specific scalability property is being exercised.

Typical commands:

```bash
pytest tests/test_models.py   # core smoke tests
pytest                        # full suite
```

---

## 8. Documentation & Configs

- When changing public APIs (new arguments, return types, or behavior) or CLI entry points:
  - update `README.md` and `README_CN.md` if user‑facing behavior changes,
  - update or extend `docs/architecture.md` for architectural changes,
  - add or adjust example configs in `configs/` where appropriate.
- Keep docstrings **short and factual**; avoid narrative in code, use `docs/` or README for longer explanations.

---

## 9. Contributions

For contributions (internal or external):

- Keep pull requests focused and minimal.
- Run at least:

  ```bash
  black src tests scripts
  ruff check src tests scripts
  pytest
  ```

- If you introduce a new kernel, transition, observation model, or training utility:
  - add a small test in `tests/`,
  - add a short note in `docs/` or a new example config in `configs/` showing how to use it.

The bar is not perfection; the bar is **clear, predictable behavior** that can be relied upon and extended.

