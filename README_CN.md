<div align="center">
  <h1>Gaussian Process State-Space Models</h1>
  <h3>GPSSM —— 开箱即用的稀疏变分高斯过程状态空间模型</h3>
</div>

<p align="center">
  <a href="#1-安装">安装</a> •
  <a href="#2-快速开始">快速开始</a> •
  <a href="#3-问题设定与核心概念">问题设定与核心概念</a> •
  <a href="#4-架构与关键模块">架构与关键模块</a> •
  <a href="#5-数据与评估">数据与评估</a> •
  <a href="#6-扩展与自定义">扩展与自定义</a> •
  <a href="#7-开发与测试">开发与测试</a> •
  <a href="#8-代码结构">代码结构</a> •
  <a href="#9-参考文献">参考文献</a>
</p>

---

本仓库提供一个**研究级但尽量轻量**的稀疏变分高斯过程状态空间模型（GPSSM）实现。
目标很直接：**不要让任何人再为样板代码浪费时间** —— 生成时间序列数据、训练 GPSSM、评估指标、做多步前滚预测，全都通过一个命令行入口 + 一个 YAML 配置完成。

---

## 1. 安装

推荐使用 Python ≥ 3.10 的独立虚拟环境。完整依赖列表以 `pyproject.toml` 为准。

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install "torch>=2.2.0,<2.3.0" "pyro-ppl>=1.9.1,<1.10" numpy pyyaml typer rich
pip install -e .
```

`pip install -e .` 会以「可编辑模式」安装 `gp-ssm`，直接暴露 `src/` 里的所有模块。

如需复现实验或运行测试，可安装开发依赖：

```bash
pip install pytest black ruff mypy
```

---

## 2. 快速开始

### 训练模型

```bash
python scripts/train_gp_ssm.py train --config configs/default.yaml
```

- 脚本会自动选择可用的 **MPS / CUDA / CPU**，并在终端打印设备信息。
- `--config` 可以指向任意 YAML 文件，例如 `configs/system_id_medium.yaml` 用于更接近系统辨识的场景。
- 训练过程中每隔 `eval_every` 步会在验证集上评估 **RMSE / NLL**；训练结束后会再次在验证集和测试集上评估，并打印一次多步前滚预测的 **RMSE**。

### 评估与预测

训练相关的评估工具可以在自定义脚本中直接复用：

- `training.evaluation.evaluate_model(model, loader)`
  计算观测 RMSE、NLL，以及在提供 latent 真值时的 latent RMSE。
 - `training.evaluation.rollout_forecast(model, y_hist, lengths, steps)`
   使用摊销后验 + GP 转移均值做多步预测，适合系统辨识 / 控制类研究。

可选的结构化变分后验：

- 在 YAML 里将 `model.q_structure` 设为 `markov`，即可把独立高斯后验替换为 Markov 结构高斯（块三对角精度），在保持训练复杂度 O(T·D³) 的同时更好地刻画时间相关性；默认的 `independent` 则保持原有行为。

你也可以直接导入模型与数据模块：

```python
from models.gp_ssm import SparseVariationalGPSSM
from data.timeseries import generate_synthetic_sequences
from training.evaluation import evaluate_model, rollout_forecast
```

---

## 3. 问题设定与核心概念

我们处理的是**非线性离散时间状态空间模型**：

$$
  x_t = f(x_{t-1}, u_{t-1}) + \varepsilon_t,\quad
  y_t = C x_t + d + \eta_t,
$$

其中 $f$ 是未知的非线性转移函数，在这里用 **向量值高斯过程**（ARD RBF 核）建模。状态维度通常不大，但系统可高度非线性、且观测可能是部分可见的。

核心思想：

- **非参数化动力学**：对 $f$ 放 GP 先验，而不是固定的参数形式，从而获得灵活的非线性动力学与对函数本身的不确定性估计。
- **稀疏变分 GPSSM**：沿用 Frigola, Chen & Rasmussen (2014) 中的思路，引入诱导点 $Z$ 与诱导变量 $u = f(Z)$，通过优化变分下界进行学习，可以在模型容量与计算复杂度之间平衡，同时保持近似后验可处理。
- **摊销状态推断**：借鉴 Doerr 等人在 PR-SSM（Probabilistic Recurrent State-Space Models, 2018）中提出的摊销推断思想，使用双向 GRU 编码器来参数化 $q(x_{0:T})$，从而在长序列和批训练下保持可扩展性。
- **随机变分推断 (SVI)**：结合稀疏 GP 转移、摊销后验与 Pyro 的 SVI 接口，在长时间序列和随机小批量上高效训练。
- **可选的结构化后验**：除了独立高斯外，提供 Markov 结构化高斯后验（块三对角精度，`model.q_structure: markov`），借鉴 VGM / ESGVI / ASVI 的思路，在不增加阶数的情况下保留时间相关性。

本实现刻意保持**体积小、结构清晰**，方便在此基础上做理论或工程上的改动，而不是被一个过度工程化的框架绑死。

---

## 4. 架构与关键模块

核心代码位于 `src/` 目录：

| 组件 | 说明 |
| --- | --- |
| `models.gp_ssm.SparseVariationalGPSSM` | PyroModule 组合 GP 转移、编码器与观测头；过程/观测噪声与初始状态在对数域中作为 `PyroParam` 存储以保证正性，支持 `Trace_ELBO` 与 `TraceMeanField_ELBO`。 |
| `models.transition.SparseGPTransition` | 稀疏 GP 转移模型，使用 ARDRBF 核与诱导点；`prior()` 给出诱导变量先验；`precompute()` 预计算 Kzz 的逆与诱导均值投影以复用。 |
| `models.kernels.*` | 最小 `Kernel` 接口及现成实现：`ARDRBFKernel`、`MaternKernel (ν∈{1/2,3/2,5/2})`、`RationalQuadraticKernel`、`PeriodicKernel`，以及 `SumKernel` / `ProductKernel` 组合器，可通过 YAML 配置直接选择。 |
| `models.encoder.StateEncoder` | 双向 GRU 编码器，输出每个时间步的 mean/scale，同时给出初始状态分布；通过 `pack_padded_sequence` 支持变长序列。 |
| `models.observation.AffineObservationModel` | 默认线性观测层 $y_t = C x_t + d$，可以替换为任意非线性解码器。 |
| `inference.svi.SVITrainer` | 对 Pyro `SVI` 的轻量包装，内置 `ClippedAdam`、进度条打印、ELBO 类型选择与 checkpoint 保存。 |
| `data.timeseries.TimeseriesWindowDataset` | 支持窗口采样、补零和长度 mask 的时间序列数据集；配合 `build_dataloader` 用于批训练。 |
| `training.evaluation` | 提供 `evaluate_model` 与 `rollout_forecast`，用于标准指标与多步前滚预测。 |

更多设计细节与扩展 hooks 见 `docs/architecture.md`。

---

## 5. 数据与评估

### 数据生成

- **Toy 模型**：`generate_synthetic_sequences` 使用正弦漂移 + 线性观测，适合快速验证实现与指标计算。
- **系统辨识场景**：`generate_system_identification_sequences` 模拟控制信号、非线性 drift、过程/观测噪声，与实际控制系统更接近。

两者都返回带有观测、长度与可选 latent 的 `SequenceDataset`。

### 划分与窗口化

- `split_dataset(dataset, (train, val, test), seed)` 按比例随机划分序列，seed 可重复实验。
- `TimeseriesWindowDataset` 根据长度对序列做固定窗口采样，自动补零和生成长度 mask，latent（如存在）也会与窗口对齐。
- `build_dataloader` 将其封装为 `torch.utils.data.DataLoader`，输出包含 `y`、`lengths` 与可选 `latents` 的批次。

### 评估指标

- **RMSE**：观测值的均方根误差（按长度 mask）。
- **NLL**：在模型估计的观测噪声下的高斯负对数似然。
- **Latent RMSE**：在存在 latent 真值时的隐状态 RMSE。
- **Forecast RMSE**：使用 `rollout_forecast` 在给定 context 与 horizon 下计算的多步预测误差（在训练脚本中示例）。

这些指标与变分 GPSSM 与 PR-SSM 文献中的评估风格兼容，同时保持实现简洁。

---

## 6. 扩展与自定义

- **Kernel 替换**：内置 ARD RBF、Matérn（ν 取 1/2、3/2、5/2）、Rational Quadratic、Periodic，以及支持加法/乘法组合的 `SumKernel` / `ProductKernel`。在 `model.kernel` 中配置即可：

  ```yaml
  model:
    kernel:
      type: sum            # 可选：ard_rbf | matern | rational_quadratic | periodic | sum | product
      jitter: 1.0e-5
      components:
        - type: ard_rbf
        - type: periodic
          params:
            period: 24.0
            lengthscale: 0.5
  ```

  若需自定义，实现 `models.kernels.Kernel` 抽象类的 `forward` 与 `diag` 方法即可，并在配置里引用新的 `type` 名称。
- **Observation / Encoder 自定义**：可以用任意 PyTorch / PyroModule 作为观测头或编码器，例如图像任务中的 CNN 解码器，或长序列任务中的 Transformer 编码器。
- **控制输入建模**：在转移模型输入中拼接 `(x_t, u_t)` 即可支持受控系统；系统辨识数据生成器已经包含控制信号，可直接利用。
- **多配置实验**：在 `configs/` 下添加新的 YAML 文件来定义实验（状态维度、诱导点个数、窗口长度、噪声尺度等），训练脚本会自动读取。
- **Checkpoint 恢复**：`SVITrainer` 会在 `checkpoint_dir` 下保存 `final.pt`，可通过 `pyro.get_param_store().load_state_dict` 恢复参数并继续训练或单独评估。

---

## 7. 开发与测试

运行测试：

```bash
pytest tests/test_models.py
```

涵盖内容：

- kernel 的形状与正定性检查；
- GP 转移预测的形状检查；
- 在合成数据上的单步 SVI 训练；
- 系统辨识数据生成、划分与窗口化逻辑。

在添加新的 kernel / transition / observation 模块时，建议在 `tests/` 中增加相应的形状与稳定性测试。

风格上推荐：

- 使用 guard clause 避免深度嵌套；
- 显式传入依赖，避免隐式全局状态；
- 优先保持模块小而可组合。

欢迎大家提 Issue 与 PR。若你增加了新的 kernel / transition / 观测模型或训练工具，建议同时补充一个小测试，并在 `docs/` 或 `configs/` 中留下简短说明，方便他人复用你的工作。

---

## 8. 代码结构

```text
src/
  data/         # 合成数据与 DataLoader 构造
  inference/    # SVI 训练器与 ELBO 包装
  models/       # GPSSM、kernel、transition、encoder、observation
  training/     # 评估工具（指标与前滚预测）

scripts/
  train_gp_ssm.py   # 训练与评估的命令行入口

configs/
  *.yaml       # 实验配置（toy / system-id 等）

docs/
  architecture.md   # 建模与实现的高层说明

tests/
  test_kernels.py / test_transition.py / test_data.py / test_svi.py
```

---

## 9. 参考文献

1. **Variational Gaussian Process State-Space Models**
   Roger Frigola, Yutian Chen, Carl E. Rasmussen.
   *Advances in Neural Information Processing Systems 27 (NeurIPS), 2014.*

2. **Probabilistic Recurrent State-Space Models (PR-SSM)**
   Andreas Doerr, Christian Daniel, Martin Schiegg, Duy Nguyen-Tuong, Stefan Schaal, Marc Toussaint, Sebastian Trimpe.
   *Proceedings of the 35th International Conference on Machine Learning (ICML), PMLR 80, 2018.*

3. **Structured Variational Inference for Dynamical Systems**
   Alessandro Curi, Janis Keuper, Felipe Tobar, Joachim M. Buhmann.
   *Learning for Dynamics & Control (L4DC), 2020.*

4. **Exactly Sparse Gaussian Variational Inference (ESGVI)**
   Timothy D. Barfoot, Winston Murray.
   *arXiv:1911.08333, 2019.*

5. **Automatic Structured Variational Inference (ASVI)**
   Luca Ambrogioni, F. J. R. Ruiz, Tim Meireles, Max Welling.
   *arXiv:2002.00643, 2020.*

项目内部文档：

- `docs/architecture.md`：关于建模选择、变分推断流程与扩展 hook 的更多说明。
- `configs/`：toy 与系统辨识两类实验的配置示例。
- `pyproject.toml`：项目元数据与依赖约束。
