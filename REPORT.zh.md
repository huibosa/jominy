# Jominy 淬透性建模 — 最终报告

作者：机器学习建模阶段，2026 年 5 月。
目标：仅以钢材化学成分为输入，预测端淬试样距淬火端 9 mm 处（J9）和 15 mm 处（J15）的洛氏硬度。

---

## 1. 执行摘要

经过 7 轮实验、共 **232 次模型试验**，覆盖线性、PLS、树模型、核方法、神经网络、Stacking 与 Blend 等多个家族，最终选定的预测器为：

> **J9 = 0.70 · XGBoost + 0.30 · PLS**（凸组合）
> **δ  = 0.60 · XGBoost + 0.40 · BayesianRidge**（其中 δ = J9 − J15）
> **J15 = J9 − max(0, δ)**     （后处理保证 J9 ≥ J15）

5 折 `GroupKFold`（按 `base_heat_id` 分组）交叉验证结果：

| 目标 | 最优模型 | MAE | RMSE | R² |
|------|---------|----:|-----:|---:|
| **J9** | `blendF_xgb_pls3_w0.70` | **1.7106** | **2.2135** | **0.4808** |
| **δ**  | `blendF_delta_xgb_bayes_w0.60` | **1.0940** | 1.4363 | 0.0012 |

相对于既有生产基线 (`pls_n4_full`，MAE = 1.7617) 的提升：
- J9 MAE：−0.051（约 2.9% 相对改进）
- J9 RMSE：−0.062（约 2.7%）
- J9 R²：+0.029（绝对提升）

实用层面：65% 的预测落在实测值 ±2 HRC 之内，84% 在 ±3 HRC 内，96% 在 ±5 HRC 内。模型在 J9 分布的中段标定良好，但在两端会向均值回归约 ±3 HRC，原因在于该处仅靠化学成分无法区分这些试样。

---

## 2. 问题与数据

### 2.1 目标变量

- **J9**：水淬试样距淬火端 9 mm 处的洛氏 C 标尺硬度（HRC），是淬透性的核心指标。
- **J15**：同上，距淬火端 15 mm 处。
- **δ = J9 − J15**：硬度差，物理上恒为非负。

直接预测 J9，再用 J15 = J9 − max(0, δ) 重构 J15，是在不引入约束优化的前提下保证单调性（J9 ≥ J15）的最简方法。

### 2.2 数据集

由 `scripts/build_modeling_tables.py` 从清洗后的长表生成：

| 文件 | 行数 | 说明 |
|------|----:|------|
| `data/modeling/j9_dataset.parquet` | 566 | 所有具备 J9 实测值的试样 |
| `data/modeling/delta_dataset.parquet` | 491 | 同时具有 J9 和 J15 的子集 |

每行对应一个钢材试样，以 `炉号` 索引；`base_heat_id` 为去后缀的炉号，用作交叉验证的分组键。

### 2.3 特征（FULL_FEATURES，共 18 列）

- 13 个元素质量分数：`C, Si, Mn, P, S, Cu, Ni, Cr, V, Ti, W, Al, B`
- 5 个缺失指示变量：`V_missing, Ti_missing, W_missing, Al_missing, B_missing`

微量元素 V/Ti/W/Al/B 缺失率为 13–34%。中位数填补 + 缺失标志的编码方式，使模型可以区分"低于检测限故缺失"与"实际为低值"两种情况。

### 2.4 统计概览

数据集主要特征：

```
J9          均值 = 36.4 HRC,  标准差 = 3.1, 范围 [29.9, 45.2]
J15         均值 = 30.1 HRC,  标准差 = 3.4, 范围 [22.1, 41.7]
δ           均值 = 6.46 HRC,  标准差 = 1.45
C           均值 = 0.20,      标准差 = 0.010, 范围 [0.17, 0.22]
Mn          均值 = 0.96,      标准差 = 0.07,  范围 [0.66, 1.30]
Cr          均值 ≈ 1.13,      标准差 ≈ 0.20
```

化学成分窗口很**窄** —— 这是单一产品线的典型特征 —— 也正是限制可达精度的主要因素。

---

## 3. 方法

### 3.1 交叉验证

5 折 **GroupKFold**，分组键为 `base_heat_id`。每个基础炉号（试样组）只出现在一个验证折中，训练集与验证集不共享任何炉号。这避免了同炉但物理上不同的 `-H` / `-Z` 后缀变体之间的信息泄漏。

566 行的 J9 数据集中每个 `base_heat_id` 唯一，因此 GroupKFold 退化为普通 KFold。框架仍保持分组感知，使同一份代码能正确驱动 δ 任务，并在未来重新引入后缀试样时仍受保护。

### 3.2 评估指标

每折记录 MAE、RMSE、R²。Leaderboard 以 5 折的 `mean ± std` 表示。

**选择准则**：J9 上 `MAE_mean` 最低（主指标）；并列时按 `RMSE_mean` 决胜，再按 `MAE_std`（折间方差越小越优）。

### 3.3 复现

```bash
uv run --with pandas,pyarrow,scikit-learn,xgboost,lightgbm \
    python scripts/model_experiments.py --quick
uv run --with pandas,pyarrow,scikit-learn,xgboost \
    python scripts/model_experiments_round{2,3,4,5,7}.py
uv run --with pandas,pyarrow,scikit-learn,xgboost,lightgbm \
    python scripts/model_experiments_round6.py
uv run --with pandas,pyarrow,scikit-learn,xgboost \
    python scripts/inspect_winner.py
```

全部 232 次试验记录于 [HISTORY.md](HISTORY.md)，每个模型的精确配置都可追溯。

---

## 4. 实验过程 —— 分轮叙述

### 第 1 轮 —— 广度筛选（J9 共 37 个模型）

目标：横向覆盖各模型家族，并将框架对齐既有的 `pls_n4_full` 基线（MAE = 1.7617）。

| 家族 | 最优 | MAE | 备注 |
|------|------|----:|------|
| 线性 | `pls_n3_full` | **1.7555** | 略胜原 n=4 PLS |
| 线性 | `bayesian_ridge_full` | 1.7696 | |
| 树模型 | `rf_500_md8` | 1.7747 | RF 中最优 |
| 树模型 | `xgb_n800_lr0.02_md3` | 1.7688 | XGBoost 在梯度提升类中领先 |
| 核方法 | `svr_rbf_C1_g0.1` | 1.8055 | 核方法中最优；其余表现灾难 |
| MLP | `mlp_128x64_tanh` | 2.4864 | 在 566 行数据上严重欠/过拟合 |
| Stacking | `stack_pls_ridge_xgb` | 1.7635 | 略有帮助 |

第 1 轮领先：**`pls_n3_full`，MAE = 1.7555**。核岭回归（默认 γ = 0.1）的灾难性失败（MAE 5.78）和浅层 MLP 的失败（MAE 3.95）说明：默认参数对这份 18 维标准化数据并不合适。

### 第 2 轮 —— 特征工程与 XGBoost 调优（J9 共 28 个模型）

淬透性物理学暗示需要交互项（Cr·C、Mn·C、对数元素、碳当量等）。新增 12 个工程特征并重跑线性/PLS 家族。同时开始围绕第 1 轮领先者收紧 XGBoost 超参。

| 结果 | 结论 |
|------|------|
| `pls_n3_full+eng` | 1.7587 — 工程特征未改进 PLS |
| `bayesian_ridge_full+eng` | 1.7662 — 同上 |
| `xgb_n800_lr0.01_md3` | **1.7445** — 新领先；慢学习率胜出 |
| `xgb_n500_lr0.01_md4` | 1.7489 |
| `xgb_n1200_lr0.01_md3` | 1.7503 |

第 2 轮领先：**`xgb_n800_lr0.01_md3`，MAE = 1.7445**。

解读：树集成已经能自动发现跨元素交互，显式工程特征是冗余的。慢学习率 + 约 800 棵树是 XGBoost 在 566 行规模上具备良好泛化能力的甜点。

### 第 3 轮 —— XGBoost 正则与子采样（J9 共 17 个模型）

扫描 `(subsample, colsample_bytree)`、`(reg_alpha, reg_lambda)`，并尝试更慢的学习率。

| 本轮最佳 | MAE |
|---|---:|
| `xgb_n800_lr0.01_md3_ss0.6_cs0.6` | **1.7386** |
| `xgb_n800_lr0.01_md3_ss0.7_cs0.7` | 1.7389 |
| `xgb_n800_lr0.01_md3_l5.0_a0.0` | 1.7430 |
| `xgb_n2000_lr0.005_md3` | 1.7448 |

行子采样 0.6–0.7 与列子采样 0.6–0.7 是单一最有效的杠杆 —— 所有积极子采样配置都击败了不子采样的领先者。

### 第 4 轮 —— 精细子采样扫描 + 多种子 Bagging（J9 约 30 个模型）

| 本轮最佳 | MAE |
|---|---:|
| `xgb_v2_ss0.55_cs0.5` | **1.7263** |
| `xgb_v2_ss0.55_cs0.6` | 1.7298 |
| `xgb_v2_ss0.65_cs0.7` | 1.7323 |
| `xgb_bag_10seeds` | 1.7360 |

第 4 轮领先：**`xgb_v2_ss0.55_cs0.5`，MAE = 1.7263**。

意外的是，多种子 Bagging **未能**超越单种子领先者（1.7360 vs 1.7263）。原因：XGBoost 内部的行/列子采样已经提供了类似 Bagging 的方差降低效果，再叠加 10 种子平均反而把预测拉向数据均值。

### 第 5 轮 —— 更精细扫描与 MAE 目标函数（J9 约 30 个模型）

| 本轮最佳 | MAE |
|---|---:|
| `xgb_v3_n1500_lr0.005`（ss=0.55, cs=0.5） | **1.7260** |
| `xgb_v3_ss0.55_cs0.5` | 1.7263（持平） |
| `xgb_bag20_v3`（20 种子 Bagging） | 1.7321 |
| `xgb_mae_ss0.55_cs0.5`（MAE 目标函数） | 1.7569 |

MAE 目标函数变体在 MAE 指标上反而*更差* —— 反直觉但在小样本低信号时常见：均方误差梯度产生的估计更平滑，恰好在留出折上 MAE 也更低。

至此领先平台清晰：MAE ≈ 1.726，RMSE ≈ 2.227，单模型再调参已无显著收益。

### 第 6 轮 —— LightGBM、Stacking、凸组合 Blending（J9 共 40+ 个模型）

LightGBM 与 XGB 领先者使用同套超参，最佳为 MAE = 1.7454 —— 始终未达到 XGB 的水平。LGBM 采用按叶生长，倾向于发掘更深的交互；在小样本低信号问题上反而是劣势。

完整 sklearn `StackingRegressor`（PLS + Bayes + XGB，Ridge 元学习器）：MAE = 1.7524，加 passthrough 后变 1.857。元学习在折外预测上过度看重 XGB，passthrough 特征又给元输入增加了噪声。

XGB 领先者 + PLS_n3 的**凸组合**配以人工设定权重 w 的结果：

| 权重 (w_xgb) | MAE |
|---:|---:|
| 0.20 | 1.7349 |
| 0.40 | 1.7213 |
| 0.50 | 1.7166 |
| 0.60 | 1.7128 |
| **0.70** | **1.7106** |
| 0.80 | 1.7132 |

第 6 轮领先：**`blend_xgb0.7_pls0.3`，MAE = 1.7106**。整轮战役中首次跌破 1.72。

### 第 7 轮 —— 精细 Blend 网格（J9 / δ 共 40+ 个模型）

确认 Blend 最优在 w ∈ [0.62, 0.75] 范围内基本平坦（与最佳 MAE 差距 < 0.001）。测试不同线性搭档与 Bagging 基学习器：

| 变体 | MAE |
|------|----:|
| `blendF_xgb_pls3_w0.70` | **1.7106** |
| `blendF_xgb_ridge_w0.70` | 1.7114 |
| `blendF_xgb_bayes_w0.70` | 1.7129 |
| `blendBag10_xgb_pls3_w0.65`（10 种子 Bagging XGB 后再 Blend） | 1.7160 |

PLS_n3 搭档微胜 Ridge 与 BayesianRidge，三者都击败了 Bagging 变体。在 Blend 内部对 XGB 进行 Bagging 反而使 MAE 略微回退 —— 与第 4 轮相同的解释。

### δ 任务

δ 在 `blendF_delta_xgb_bayes_w0.60` 处达到 **MAE = 1.0940**。但 R²(δ) ≈ 0，说明模型几乎只是预测均值 —— **仅靠化学成分无法预测 δ**。这与项目早先发现一致：5–7 HRC 的 J9–J15 差值取决于晶粒尺寸与局部冷却速率，而这些信息不在特征集中。

---

## 5. J9 前 10 名榜单

来自 [HISTORY.md](HISTORY.md)，全部为 5 折交叉验证结果：

| # | 模型 | MAE_mean | MAE_std | RMSE_mean | R²_mean |
|---|------|---------:|--------:|----------:|--------:|
| 1 | `blendF_xgb_pls3_w0.70`（= `blend3_xgb0.7_pls0.3_bayes0.0`） | **1.7106** | 0.0777 | 2.2135 | 0.4808 |
| 2 | `blendF_xgb_pls3_w0.68` | 1.7107 | 0.0770 | 2.2134 | 0.4808 |
| 3 | `blendF_xgb_pls3_w0.72` | 1.7110 | 0.0781 | 2.2138 | 0.4806 |
| 4 | `blend3_xgb0.7_pls0.2_bayes0.1` | 1.7112 | 0.0781 | 2.2139 | 0.4806 |
| 5 | `blendF_xgb_pls3_w0.65` | 1.7112 | 0.0766 | 2.2134 | 0.4808 |
| 6 | `blendF_xgb_ridge_w0.70` | 1.7114 | 0.0791 | 2.2159 | 0.4795 |
| 7 | `blendF_xgb_ridge_w0.75` | 1.7114 | 0.0800 | 2.2161 | 0.4795 |
| 8 | `blendF_xgb_pls3_w0.75` | 1.7116 | 0.0785 | 2.2142 | 0.4804 |
| 9 | `blend3_xgb0.7_pls0.1_bayes0.2` | 1.7120 | 0.0785 | 2.2144 | 0.4803 |
| 10 | `blendF_xgb_pls3_w0.62` | 1.7120 | 0.0763 | 2.2136 | 0.4807 |

前 10 名全部为 Blend 或 XGB-线性三方组合。第一个非 Blend 单模型出现在第 26 名（`xgb_v3_n1500_lr0.005`，MAE = 1.7260）。

---

## 6. 最终选定模型

### 6.1 定义

```python
# 基学习器 —— 每折在完整训练集上重新拟合
xgb_j9 = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", XGBRegressor(
        n_estimators=1500, max_depth=3, learning_rate=0.005,
        subsample=0.55, colsample_bytree=0.5, reg_lambda=2.0,
        random_state=42, n_jobs=-1,
    )),
])

pls_j9 = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", PLSRegression(n_components=3, scale=False)),
])

# 凸组合
y_j9 = 0.70 * xgb_j9.predict(X) + 0.30 * pls_j9.predict(X)

# δ Blend（结构相同，线性搭档为 BayesianRidge）
y_delta = 0.60 * xgb_delta.predict(X) + 0.40 * bayes_delta.predict(X)

# 重构 J15，强制单调性
y_j15 = y_j9 - max(0.0, y_delta)
```

### 6.2 超参数依据

| 参数 | 取值 | 原因 |
|------|----:|------|
| `n_estimators` | 1500 | 慢学习率需要长拟合；继续增加收益趋平 |
| `learning_rate` | 0.005 | 在 566 行数据上更小的 lr 泛化更好 |
| `max_depth` | 3 | 4 或 5 在 CV 上过拟合 |
| `subsample` | 0.55 | 单一最有效的提升杠杆 |
| `colsample_bytree` | 0.5 | 与 subsample 协同；共同抑制记忆化 |
| `reg_lambda` | 2.0 | 在 [0.5, 10] 范围内表现稳定；第 3 轮调出 |
| `n_components`（PLS） | 3 | n=2 欠拟合，n≥5 MAE 抬升 |
| `w_xgb`（Blend） | 0.70 | [0.62, 0.75] 区间为平坦最优；0.70 为经验最佳 |

### 6.3 为什么 Blend 胜出

PLS 捕获**化学成分线性信号** —— 淬透性主要由 C、Mn、Cr、Ni 决定，且在这个窄成分窗口内这些关系基本是可加的。

XGBoost 捕获**残余非线性**：Mn 的饱和效应、Cr×C 交互、B 的阈值效应、缺失标志的分支结构。

两者**误差互补** —— XGBoost 略微过拟合高 Cr 试样而 PLS 欠拟合，反之低 Mn 试样上情况相反。平均后双方偏差都被抵消，方差不增。

固定权重的简单凸组合击败了完整 sklearn `StackingRegressor`（MAE 1.7106 vs 1.7524），原因：
- 元学习器以训练折 MSE 为依据过度看重 XGB，但 XGB 在验证折上泛化更弱。
- 元输入中的 passthrough 特征在 566 行规模上引入噪声。
- 固定权重对小数据集 stacking 中常见的元拟合不稳定问题更为鲁棒。

---

## 7. J9 最终模型的预测质量

通过用 5 折 GroupKFold 重跑 Blend，得到全部 566 个试样的折外（OOF）预测。逐行数值见 `output/modeling/predictions/blend_oof.csv`。

### 7.1 总体

| 指标 | 取值 |
|------|----:|
| MAE | 1.7108 HRC |
| RMSE | 2.2154 HRC |
| R² | 0.484 |
| 偏差（残差均值） | −0.017 HRC |
| 中位 \|误差\| | 1.40 HRC |
| 75 分位 \|误差\| | 2.40 HRC |
| 90 分位 \|误差\| | 3.54 HRC |
| 95 分位 \|误差\| | 4.33 HRC |
| 最大 \|误差\| | 8.77 HRC |
| ±1 HRC 内 | 35.9 % |
| ±2 HRC 内 | **65.5 %** |
| ±3 HRC 内 | **83.7 %** |
| ±5 HRC 内 | **96.3 %** |

### 7.2 各折一致性

| 折 | n | MAE | RMSE |
|---:|---:|----:|-----:|
| 0 | 114 | 1.829 | 2.348 |
| 1 | 113 | 1.706 | 2.096 |
| 2 | 113 | 1.608 | 2.155 |
| 3 | 113 | 1.758 | 2.248 |
| 4 | 113 | 1.652 | 2.222 |

各折一致性良好（MAE 标准差 ≈ 0.08）；折 0 最难，折 2 最易，提示存在一定炉号聚类效应而非局部灾难性失败。

### 7.3 误差随 J9 量级 —— 向均值回归的尾部偏差

| 真实 J9 区间 | n | MAE | 偏差 |
|---|---:|---:|---:|
| [30, 32) | 47 | 3.18 | **+3.17** |
| [32, 35) | 145 | 1.52 | +1.40 |
| [35, 38) | 180 | **1.26** | −0.30 |
| [38, 41) | 156 | 1.54 | −1.14 |
| [41, 45) | 37 | 3.35 | **−3.35** |

分布的中段 60% 拟合良好（35–38 区间 MAE 仅 1.26）。两端误差约为中段的两倍，且偏差量级足以对下游用途构成实际问题 —— 高 J9 处模型**系统性低估**约 3 HRC，低 J9 处则同等程度高估。

### 7.4 为什么尾部失败

观察 10 个最差预测样本：

```
炉号           C     Mn    Cr      J9    pred    err
P23700563   0.20  0.87  1.07   41.6   32.8   −8.8
P24701222   0.21  1.02  1.17   31.2   39.6   +8.4
P25602380   0.18  0.86  1.09   41.2   33.6   −7.6
P23600636   0.21  1.02  1.15   30.4   37.4   +7.0
P23704815   0.20  0.95  1.12   43.1   36.4   −6.7
P25201693   0.21  1.02  1.17   45.2   38.8   −6.4
…
```

10 个最差样本的化学成分**几乎完全相同**（C ∈ [0.18, 0.21]，Mn ∈ [0.86, 1.02]，Cr ∈ [1.07, 1.18]），但 J9 跨度达 30–45 HRC。化学特征本身就无法区分这些试样 —— 这是**特征集层面的不可约束限制**，不是模型层面的失败。

未解释方差的可能驱动因素：
- 奥氏体化温度与保温时间
- 淬火水流量与水温
- 原始奥氏体晶粒尺寸
- 硼的存在形态（固溶 vs 析出为硼化物）
- ICP 检测限以下的微量杂质

以上信息均未包含在数据集中。

---

## 8. 失败案例 —— 哪些没用，以及为什么

| 家族 | 最佳 MAE | 结论 |
|------|--------:|------|
| **核岭回归（默认 γ=0.1）** | 5.78 | γ=0.1 对 18 维标准化特征过高（经验法则：γ ≈ 1/n_features = 0.056）。调参后核岭仍只到 MAE = 2.07 —— 没有竞争力。 |
| **MLPRegressor** | 2.49（两种配置中较好者） | 566 行对 MLP 而言数据太少，无法击败正则化的线性/树模型。即使 α=0.01 配 tanh，MLP 仍过拟合。 |
| **多项式特征（degree=3）+ Ridge** | 2.40 | 特征爆炸（7 个基特征膨胀至约 1000 列）淹没 566 行训练集；R² 转负。 |
| **手工淬透性特征**（C×Cr、log_C、sumCEQ 等） | 与最佳持平（1.7587 vs 1.7555） | XGBoost 已能隐式发现这些交互。手工特征只增噪不增信。 |
| **LightGBM** | 1.7454 | 按叶生长在小样本低信号问题上是劣势。同套超参始终未追上 XGB。 |
| **MAE 目标 XGBoost**（`reg:absoluteerror`） | 1.7569 | 均方误差梯度产生更平滑的拟合，恰好在验证集上 MAE 也更低。 |
| **20 种子 Bagging XGBoost** | 1.7321 | XGB 内部的行/列子采样已提供 Bagging 式方差降低；显式 Bagging 会把预测拉向均值。 |
| **sklearn StackingRegressor**（Ridge 元学习器） | 1.7524 | 元拟合以训练折 MSE 为依据高估 XGB，忽略了它在验证折上的弱泛化。 |
| **StackingRegressor + passthrough=True** | 1.857 | passthrough 特征在小数据集上给元输入加噪声。 |
| **多项式 + ridge（degree=2）** | 1.92 | 轻微过拟合；PLS 是更合适的降维方式。 |

**通用经验：**
- 模型默认超参（核 γ、MLP 层规模、LGBM num_leaves）面向更大数据集校准；在 566 行上未调时表现极差。
- 特征工程与 stacking 实现成本不低，但在小问题上很少能击败"调好的集成 + 简单凸组合"。
- 当基学习器已自带子采样时，再做 Bagging 通常是冗余的。

---

## 9. 局限与注意事项

1. **训练数据窗口窄。** C ∈ [0.17, 0.22]，Mn ∈ [0.66, 1.30]，Cr ∈ [0.4, 1.6]。窗口外为外推，XGB 与 PLS 分量预测可能严重发散 —— 此时 Blend 会过度依赖 PLS 外推（无界），结果不可信。Web 应用通过逐元素的越界警告将该情况显式提示给用户。
2. **R²(δ) ≈ 0。** 在已用化学成分预测 J9 后，J9–J15 差值与化学成分基本无相关性。δ 模型仅勉强胜过 `mean(δ)`，应视为粗略偏移而非精确量。
3. **尾部偏差。** J9 分布两端存在系统性 ±3 HRC 偏差。模型不应用于个别异常试样（J9 < 32 或 J9 ≥ 41）的决策。
4. **标签池存在选择偏差。** 566 个有标签试样的 C 均值（0.198）远低于无标签化学池（3,498 个，C 均值 0.457），Cr 均值（1.13）高于后者（0.41）。Blend 在分布外化学成分上的预测不可信。
5. **生产管线尚未提交。** 当前生产导出（`scripts/run_baselines.py`、`scripts/select_final_model.py`）仍发布固定 v1 的 PLS_n4 / Ridge。要把 Blend 提升到生产，需要更新 `select_final_model.py` 与 `assemble_pair_predictions` 后处理链。
6. **J9 任务的 GroupKFold 退化为普通 KFold。** 因 J9 数据集中每个 `base_heat_id` 唯一，GroupKFold 与 KFold 给出相同切分。但框架保留分组感知，使后续重新引入 `-H` / `-Z` 后缀试样时仍能避免泄漏。
7. **交叉验证含超参数选择。** 所有超参均由观察 CV 分数调出。报告 MAE 在每折内对该折是无偏的，但相对真正未来留出数据偏乐观 —— 大约 0.005 MAE 可能是调参噪声。

---

## 10. 建议

### 即刻可做

- 把 Blend 推到生产，具体步骤：
  1. 在 `src/modeling/pipelines.py` 中增加 `build_j9_blend_pipeline` 与 `build_delta_blend_pipeline`。
  2. 更新 `scripts/select_final_model.py`，在保留（或替换）固定 v1 PLS / Ridge 的同时输出 Blend。
  3. 更新 `scripts/run_baselines.py`，用 Blend 刷新 `output/modeling/predictions/cv_predictions.parquet`。
- 增加集成测试，断言部署的 Blend 在留出折上的 MAE 与 1.71 的偏差不超过 0.05。

### 近期，若需进一步提精度

- 收集工艺变量（奥氏体化温度、淬火速率代理、奥氏体晶粒尺寸）。这些最有可能消除 ±3 HRC 尾部偏差。
- 提升化学测量精度：当前数值精确到 0.01 wt%，对 C、Mn 等关键元素丢失信息。
- 探究是否有任何**已有数据**（炉批、供应商、采样日期等）能区分尾部试样（J9 < 32 或 J9 ≥ 41），若有则加入为类别特征。

### 不必再做

- 在 XGBoost 上继续超参搜索。领先平台已经稳固。
- 给 Blend 加入更多线性模型。三方 Blend（PLS + Ridge + Bayes）与两方 XGB + PLS 持平；继续加线性搭档只是锁定相同信号。
- 深度学习。表格 MLP 与 TabNet 类模型在该问题规模下需要 10× 以上的数据才能击败 GBM。

---

## 附录 —— 文件清单

| 路径 | 用途 |
|------|------|
| `HISTORY.md` | 232 次模型试验完整榜单 |
| `REPORT.md` | 英文版报告 |
| `REPORT.zh.md` | 本报告 |
| `README.md` | 项目概览与快速上手 |
| `scripts/model_experiments.py` | 第 1 轮广度筛选 |
| `scripts/model_experiments_round{2..7}.py` | 迭代精修各轮 |
| `scripts/inspect_winner.py` | Blend 折外预测质量诊断 |
| `webapp/backend/train_models.py` | 将生产 Blend 持久化到 `webapp/models/*.joblib` |
| `webapp/backend/main.py` | 提供 `/api/predict` 的 FastAPI 服务 |
| `webapp/frontend/` | Vite + TypeScript 表单与结果展示界面 |
| `output/modeling/predictions/blend_oof.csv` | J9 Blend 的逐试样折外预测 |
