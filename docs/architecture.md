# Architecture Notes

## Modeling Choices
- **Latent Dynamics**: Nonlinear transition `x_t = f(x_{t-1}) + \epsilon_t`, where `f` is a vector-valued GP. We adopt the sparse inducing-point variational family from Frigola et al. (2014), i.e. augment with inducing variables `u = f(Z)` and minimize the KL between `q(u)q(x_{0:T})` and the true posterior.
- **Kernel**: ARD RBF with logarithmic parametrization, ensuring strictly positive length-scales / amplitude. The kernel is shared across output dimensions; each dimension has its own inducing targets so the overall prior mean is block-diagonal.
- **Process/Observation Noise**: Diagonal covariances, learnt in log domain to keep positivity without projections.
- **Observation Head**: Default affine head `y_t = C x_t + d + \eta_t`. Swap `AffineObservationModel` for more expressive decoders if needed.

## Variational Guide
- `q(u)` is a multivariate normal with free mean and Cholesky factor, implemented via Pyro parameters.
- `q(x_{0:T})` is amortized by a bi-directional GRU encoder reading the observed sequence and emitting per-time-step mean / log-variance.
- Masked padding ensures batches of unequal length contribute only where data exist.

## Training Loop
1. Generate or load sequence data (toy sine system / system-id nonlinear controlled system), and obtain train/val/test through `split_dataset`.
2. `TimeseriesWindowDataset` is responsible for windowing and padding, with different window lengths available for training/evaluation.
3. Instantiate `SparseVariationalGPSSM` and build a `Trace_ELBO` trainer through `SVITrainer`.
4. The training loop triggers `evaluate_model` every `eval_every` steps, reporting RMSE/NLL/latent RMSE; after termination, re-evaluate on val/test.
5. `rollout_forecast` uses discounted posterior mean + GP transition mean for multi-step forward prediction, measuring the prediction performance in system identification/control scenarios.

## Complexity
- Kernel factorizations operate on `M x M` matrices; choosing `M <= 64` keeps Cholesky `O(M^3)` cost cheap relative to sequence unrolling.
- Sequence processing is linear in time horizon; random window sampling makes the method scalable to arbitrarily long streams.

## Extension Hooks
- Condition on control inputs by concatenating them into the state fed to the GP transition.
- Enable multi-start hyper-parameter training via parallel Pyro SVI instances orchestrated at the script level.
- **Variational family options**: default guide assumes independent `q(x_t)`. A structured option (`q_structure: markov`) parameterizes `q(x_t | x_{t-1}) = N(A_t x_{t-1} + b_t, diag(σ_t^2))`, producing a block-tridiagonal precision with O(T) complexity and better temporal coherence. Controlled via `model.q_structure` in configs.
