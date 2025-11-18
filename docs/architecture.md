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
1. 生成或加载序列数据（toy 正弦系统 / system-id 非线性受控系统），并通过 `split_dataset` 得到 train/val/test。
2. `TimeseriesWindowDataset` 负责窗口化和填充，训练/评估可使用不同窗口长度。
3. 实例化 `SparseVariationalGPSSM`，并通过 `SVITrainer` 构建 `Trace_ELBO` 训练器。
4. 训练循环每 `eval_every` 步触发 `evaluate_model`，报告 RMSE/NLL/latent RMSE；终止后再次在 val/test 上评估。
5. `rollout_forecast` 利用摊销后验 + GP 转移均值做多步前向预测，衡量系统辨识/控制场景的预测性能。

## Complexity
- Kernel factorizations operate on `M x M` matrices; choosing `M <= 64` keeps Cholesky `O(M^3)` cost cheap relative to sequence unrolling.
- Sequence processing is linear in time horizon; random window sampling makes the method scalable to arbitrarily long streams.

## Extension Hooks
- Replace the RBF kernel with VC-EMatérn from the SPDE note by swapping `kernel_class` in configs.
- Condition on control inputs by concatenating them into the state fed to the GP transition.
- Enable multi-start hyper-parameter training via parallel Pyro SVI instances orchestrated at the script level.
