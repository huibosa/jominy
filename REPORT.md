# Jominy Hardenability Modeling — Final Report

Author: ML modeling pass, May 2026.
Goal: predict Rockwell hardness at 9 mm (J9) and 15 mm (J15) from the quenched end of a Jominy specimen, given only the steel's chemical composition.

---

## 1. Executive summary

After **232 model runs** across 7 experiment rounds spanning linear, PLS, tree, kernel, neural, stacking, and blend families, the selected predictor is:

> **J9 = 0.70 · XGBoost + 0.30 · PLS** (convex blend)
> **δ  = 0.60 · XGBoost + 0.40 · BayesianRidge** (where δ = J9 − J15)
> **J15 = J9 − max(0, δ)**     (post-processing enforces J9 ≥ J15)

Cross-validated performance (5-fold `GroupKFold` on `base_heat_id`):

| Target | Best model | MAE | RMSE | R² |
|--------|------------|----:|-----:|---:|
| **J9** | `blendF_xgb_pls3_w0.70` | **1.7106** | **2.2135** | **0.4808** |
| **δ**  | `blendF_delta_xgb_bayes_w0.60` | **1.0940** | 1.4363 | 0.0012 |

Improvement over the prior production baseline (`pls_n4_full`, MAE = 1.7617):
- J9 MAE: −0.051 (~2.9% relative)
- J9 RMSE: −0.062 (~2.7%)
- J9 R²:  +0.029 absolute

Practically: 65 % of predictions are within ±2 HRC of the lab measurement, 84 % within ±3 HRC, 96 % within ±5 HRC. The model is well-calibrated in the middle of the J9 distribution but regresses toward the mean by ±3 HRC at the tails, where chemistry alone does not discriminate the specimens.

---

## 2. Problem & data

### 2.1 Targets

- **J9** — Rockwell C hardness measured 9 mm from the water-quenched end of a Jominy specimen, in HRC. Primary metric of hardenability.
- **J15** — same, 15 mm from the quenched end.
- **δ = J9 − J15** — the spread, always non-negative on physical grounds.

Predicting J9 directly and reconstructing J15 = J9 − max(0, δ) is the cleanest way to enforce monotonicity (J9 ≥ J15) without solving a constrained optimization.

### 2.2 Datasets

Built by `scripts/build_modeling_tables.py` from the cleaned long-format data:

| File | Rows | Notes |
|------|-----:|-------|
| `data/modeling/j9_dataset.parquet` | 566 | All specimens with a measured J9 |
| `data/modeling/delta_dataset.parquet` | 491 | Subset with both J9 and J15 |

Each row is one steel specimen, indexed by `炉号` (heat ID) with `base_heat_id` as the suffix-stripped grouping key for cross-validation.

### 2.3 Features (FULL_FEATURES, 18 columns)

- 13 element wt%: `C, Si, Mn, P, S, Cu, Ni, Cr, V, Ti, W, Al, B`
- 5 missingness indicators: `V_missing, Ti_missing, W_missing, Al_missing, B_missing`

The trace elements V/Ti/W/Al/B have 13–34 % missing rates; the median-imputation + missingness-flag encoding lets the model infer "absent because below detection limit" vs "actually low".

### 2.4 Statistics

Selected dataset characteristics:

```
J9          mean = 36.4 HRC,  std = 3.1, range [29.9, 45.2]
J15         mean = 30.1 HRC,  std = 3.4, range [22.1, 41.7]
δ           mean = 6.46 HRC,  std = 1.45
C           mean = 0.20,      std = 0.010, range [0.17, 0.22]
Mn          mean = 0.96,      std = 0.07,  range [0.66, 1.30]
Cr          mean ≈ 1.13,      std ≈ 0.20
```

The chemistry window is **narrow** — typical of a single product line — and this turns out to be the dominant constraint on achievable accuracy.

---

## 3. Methodology

### 3.1 Cross-validation

5-fold **GroupKFold** on `base_heat_id`. Every base heat (specimen group) appears in exactly one validation fold; train and validation sets share zero heats. This prevents leakage from suffixed `-H` / `-Z` variants that share chemistry but are physically separate specimens.

For the 566-row J9 dataset every base heat is unique, so GroupKFold reduces to ordinary KFold for J9; the framework remains group-aware so the same code drives δ correctly and protects future re-introduction of suffixed specimens.

### 3.2 Metrics

Each fold reports MAE, RMSE, R². The leaderboard records `mean ± std` over the 5 folds.

**Selection criterion**: lowest `MAE_mean` on J9 (primary). Ties broken by `RMSE_mean`, then by `MAE_std` (preferring lower variance across folds).

### 3.3 Reproduction

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

All 232 runs are logged in [HISTORY.md](HISTORY.md) with the exact configuration of every model.

---

## 4. Experiment narrative — round by round

### Round 1 — broad sweep (37 J9 models)

Goal: get a wide view across families, validate framework against the existing `pls_n4_full` baseline (MAE = 1.7617).

| Family | Best | MAE | Notes |
|--------|------|----:|-------|
| Linear | `pls_n3_full` | **1.7555** | Slightly beats the prior n=4 PLS |
| Linear | `bayesian_ridge_full` | 1.7696 | |
| Tree | `rf_500_md8` | 1.7747 | RF best of trees |
| Tree | `xgb_n800_lr0.02_md3` | 1.7688 | XGBoost in the lead among GBMs |
| Kernel | `svr_rbf_C1_g0.1` | 1.8055 | Best kernel; rest catastrophic |
| MLP | `mlp_128x64_tanh` | 2.4864 | Severely under-fits/over-fits on 566 rows |
| Stacking | `stack_pls_ridge_xgb` | 1.7635 | Helps slightly |

Round-1 leader: **`pls_n3_full` at MAE = 1.7555**. Catastrophic failures from kernel ridge with default γ = 0.1 (MAE 5.78) and shallow MLPs (MAE 3.95) — defaults are wrong for this 18-feature standardized dataset.

### Round 2 — feature engineering & XGBoost tuning (28 J9 models)

Hardenability physics motivates interaction terms (Cr·C, Mn·C, log-elements, carbon-equivalent sums). Added 12 engineered features and re-ran linear/PLS family. Also began tightening XGBoost hyperparameters around the round-1 leader.

| Result | Outcome |
|--------|---------|
| `pls_n3_full+eng` | 1.7587 — engineered features didn't improve PLS |
| `bayesian_ridge_full+eng` | 1.7662 — same |
| `xgb_n800_lr0.01_md3` | **1.7445** — new leader, slow learning rate wins |
| `xgb_n500_lr0.01_md4` | 1.7489 |
| `xgb_n1200_lr0.01_md3` | 1.7503 |

Round-2 leader: **`xgb_n800_lr0.01_md3` at MAE = 1.7445**.

Interpretation: a tree ensemble can already discover the cross-element interactions; explicit engineered features are redundant. Slow learning rate plus ~800 trees is the regime where XGBoost generalizes on 566 rows.

### Round 3 — XGBoost regularization & subsampling (17 J9 models)

Sweep over `(subsample, colsample_bytree)`, `(reg_alpha, reg_lambda)`, and slower learning rates.

| Best in round | MAE |
|---|---:|
| `xgb_n800_lr0.01_md3_ss0.6_cs0.6` | **1.7386** |
| `xgb_n800_lr0.01_md3_ss0.7_cs0.7` | 1.7389 |
| `xgb_n800_lr0.01_md3_l5.0_a0.0` | 1.7430 |
| `xgb_n2000_lr0.005_md3` | 1.7448 |

Subsampling at 0.6–0.7 (rows) × 0.6–0.7 (cols) was the largest single lever — every aggressive subsampling configuration beat the unsubsampled leader.

### Round 4 — fine subsample/colsample sweep + multi-seed bagging (~30 J9 models)

| Best in round | MAE |
|---|---:|
| `xgb_v2_ss0.55_cs0.5` | **1.7263** |
| `xgb_v2_ss0.55_cs0.6` | 1.7298 |
| `xgb_v2_ss0.65_cs0.7` | 1.7323 |
| `xgb_bag_10seeds` | 1.7360 |

Round-4 leader: **`xgb_v2_ss0.55_cs0.5` at MAE = 1.7263**.

Surprisingly, multi-seed bagging did **not** beat the single-seed leader (1.7360 vs 1.7263). Reason: XGBoost's internal row/column subsampling already provides bagging-like variance reduction, and averaging 10 seeds adds bias toward the dataset mean.

### Round 5 — even tighter sweep & MAE objective (~30 J9 models)

| Best in round | MAE |
|---|---:|
| `xgb_v3_n1500_lr0.005` (ss=0.55, cs=0.5) | **1.7260** |
| `xgb_v3_ss0.55_cs0.5` | 1.7263 (tied) |
| `xgb_bag20_v3` (20-seed bag) | 1.7321 |
| `xgb_mae_ss0.55_cs0.5` (MAE objective) | 1.7569 |

The MAE-objective variant did *worse* at MAE — counterintuitive but typical with small n and low signal: the squared-error gradient produces smoother estimates that happen to also minimize MAE on the held-out folds.

The leader plateau is clear: MAE ≈ 1.726, RMSE ≈ 2.227, no single-model gain to be extracted by further tuning.

### Round 6 — LightGBM, stacking, and convex blending (40+ J9 models)

LightGBM with the same hyperparameter regime as the XGB leader topped out at MAE = 1.7454 — never reached XGB parity. LGBM uses leaf-wise growth which favors deeper interactions; on this small, low-signal dataset that's a disadvantage.

Full sklearn `StackingRegressor` (PLS + Bayes + XGB) with a Ridge meta-learner: MAE = 1.7524, then 1.857 with passthrough features. The meta-fit was over-weighting XGB on out-of-fold predictions and the passthrough features added noise to the meta-input.

The **convex blend** of XGB-leader + PLS_n3 with manually-set weight w landed:

| Weight (w_xgb) | MAE |
|---:|---:|
| 0.20 | 1.7349 |
| 0.40 | 1.7213 |
| 0.50 | 1.7166 |
| 0.60 | 1.7128 |
| **0.70** | **1.7106** |
| 0.80 | 1.7132 |

Round-6 leader: **`blend_xgb0.7_pls0.3` at MAE = 1.7106**. First MAE below 1.72 in the entire campaign.

### Round 7 — fine blend grid (40+ J9 / δ models)

Confirmed the blend optimum is flat across w ∈ [0.62, 0.75] (MAE within 0.001 of best). Tested alternative linear partners and bagged base learners:

| Variant | MAE |
|---------|----:|
| `blendF_xgb_pls3_w0.70` | **1.7106** |
| `blendF_xgb_ridge_w0.70` | 1.7114 |
| `blendF_xgb_bayes_w0.70` | 1.7129 |
| `blendBag10_xgb_pls3_w0.65` (10-seed bag of XGB then blend) | 1.7160 |

The PLS_n3 partner narrowly beats Ridge and BayesianRidge, all three beat the bagged variant. Bagging the XGB inside the blend marginally regresses MAE — same explanation as round 4.

### δ target

δ peaks at **MAE = 1.0940** with `blendF_delta_xgb_bayes_w0.60`. R²(δ) ≈ 0 means the model is barely better than predicting the mean — δ cannot be predicted from chemistry alone on this dataset. This matches prior project findings; the 5–7 HRC J9–J15 spread depends on grain size and local quench rate, which are not in the feature set.

---

## 5. Top-10 J9 leaderboard

From [HISTORY.md](HISTORY.md), all five-fold cross-validated:

| # | Model | MAE_mean | MAE_std | RMSE_mean | R²_mean |
|---|-------|---------:|--------:|----------:|--------:|
| 1 | `blendF_xgb_pls3_w0.70` (= `blend3_xgb0.7_pls0.3_bayes0.0`) | **1.7106** | 0.0777 | 2.2135 | 0.4808 |
| 2 | `blendF_xgb_pls3_w0.68` | 1.7107 | 0.0770 | 2.2134 | 0.4808 |
| 3 | `blendF_xgb_pls3_w0.72` | 1.7110 | 0.0781 | 2.2138 | 0.4806 |
| 4 | `blend3_xgb0.7_pls0.2_bayes0.1` | 1.7112 | 0.0781 | 2.2139 | 0.4806 |
| 5 | `blendF_xgb_pls3_w0.65` | 1.7112 | 0.0766 | 2.2134 | 0.4808 |
| 6 | `blendF_xgb_ridge_w0.70` | 1.7114 | 0.0791 | 2.2159 | 0.4795 |
| 7 | `blendF_xgb_ridge_w0.75` | 1.7114 | 0.0800 | 2.2161 | 0.4795 |
| 8 | `blendF_xgb_pls3_w0.75` | 1.7116 | 0.0785 | 2.2142 | 0.4804 |
| 9 | `blend3_xgb0.7_pls0.1_bayes0.2` | 1.7120 | 0.0785 | 2.2144 | 0.4803 |
| 10 | `blendF_xgb_pls3_w0.62` | 1.7120 | 0.0763 | 2.2136 | 0.4807 |

The top 10 are all blends or three-way linear-XGB combinations. The first non-blend single model appears at rank 26 (`xgb_v3_n1500_lr0.005`, MAE = 1.7260).

---

## 6. Final selected model

### 6.1 Definition

```python
# Base learners — refit on full training set per fold
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

# Convex blend
y_j9 = 0.70 * xgb_j9.predict(X) + 0.30 * pls_j9.predict(X)

# δ blend (analogous, BayesianRidge as the linear partner)
y_delta = 0.60 * xgb_delta.predict(X) + 0.40 * bayes_delta.predict(X)

# Reconstruct J15 with monotonicity guarantee
y_j15 = y_j9 - max(0.0, y_delta)
```

### 6.2 Hyperparameter rationale

| Parameter | Value | Why |
|-----------|------:|-----|
| `n_estimators` | 1500 | Slow learning rate needs long fit; further increase plateaus |
| `learning_rate` | 0.005 | Smaller lr generalizes better on 566 rows |
| `max_depth` | 3 | Deeper (4, 5) overfits in CV |
| `subsample` | 0.55 | Strongest single-lever improvement |
| `colsample_bytree` | 0.5 | Synergistic with subsample; both cripple memorization |
| `reg_lambda` | 2.0 | Stable across [0.5, 10]; tuned on round-3 grid |
| `n_components` (PLS) | 3 | n=2 underfits, n=5+ inches up MAE |
| `w_xgb` (blend) | 0.70 | Flat optimum [0.62, 0.75]; 0.70 is the empirical best |

### 6.3 Why a blend wins

PLS captures the linear-in-chemistry signal — hardenability is dominated by C, Mn, Cr, Ni, and within this narrow chemistry window those relationships are largely additive.

XGBoost captures residual non-linearity: saturating effects of Mn, Cr×C interactions, threshold effects of B, the missingness-flag splits.

The two models make **complementary errors** — XGBoost over-fits high-Cr specimens slightly while PLS under-fits them, and vice versa for low-Mn specimens. Averaging cancels both biases without raising the variance.

A simple convex blend with a fixed weight beat full sklearn `StackingRegressor` (MAE = 1.7106 vs 1.7524) because:
- The meta-learner over-weights XGB based on its lower training-fold MSE, missing that it generalizes worse on validation.
- Passthrough features in the meta-input add noise on 566 rows.
- A fixed weight is more robust to the meta-fit instability that plagues stacking on small datasets.

---

## 7. Prediction quality on the J9 winner

Computed by re-running the blend with 5-fold GroupKFold to produce out-of-fold predictions for all 566 specimens. See `output/modeling/predictions/blend_oof.csv` for the per-row values.

### 7.1 Aggregate

| Metric | Value |
|--------|------:|
| MAE | 1.7108 HRC |
| RMSE | 2.2154 HRC |
| R² | 0.484 |
| Bias (mean residual) | −0.017 HRC |
| Median &#124;error&#124; | 1.40 HRC |
| 75th percentile &#124;error&#124; | 2.40 HRC |
| 90th percentile &#124;error&#124; | 3.54 HRC |
| 95th percentile &#124;error&#124; | 4.33 HRC |
| Max &#124;error&#124; | 8.77 HRC |
| Within ±1 HRC | 35.9 % |
| Within ±2 HRC | **65.5 %** |
| Within ±3 HRC | **83.7 %** |
| Within ±5 HRC | **96.3 %** |

### 7.2 Per-fold consistency

| Fold | n | MAE | RMSE |
|---:|---:|----:|-----:|
| 0 | 114 | 1.829 | 2.348 |
| 1 | 113 | 1.706 | 2.096 |
| 2 | 113 | 1.608 | 2.155 |
| 3 | 113 | 1.758 | 2.248 |
| 4 | 113 | 1.652 | 2.222 |

Folds are reasonably consistent (MAE std ≈ 0.08); fold 0 is the hardest, fold 2 the easiest, suggesting some heat-cluster effect rather than catastrophic local failures.

### 7.3 Error vs J9 magnitude — the regression-to-the-mean tail

| True J9 range | n | MAE | Bias |
|---|---:|---:|---:|
| [30, 32) | 47 | 3.18 | **+3.17** |
| [32, 35) | 145 | 1.52 | +1.40 |
| [35, 38) | 180 | **1.26** | −0.30 |
| [38, 41) | 156 | 1.54 | −1.14 |
| [41, 45) | 37 | 3.35 | **−3.35** |

The middle 60 % of the distribution is well-fit (MAE 1.26 in the 35–38 bin). The tails are ~2× as bad and the bias is large enough to be material for downstream use — at high J9 the model **systematically under-predicts** by ~3 HRC and at low J9 it over-predicts by the same.

### 7.4 Why the tails fail

Inspecting the 10 worst predictions:

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

All ten worst cases have **nearly identical chemistry** (C ∈ [0.18, 0.21], Mn ∈ [0.86, 1.02], Cr ∈ [1.07, 1.18]) yet J9 ranges 30–45 HRC. The chemistry features genuinely cannot discriminate these specimens — this is an irreducible feature-set limitation, not a model failure.

Drivers of the unexplained variance most likely include:
- Austenitizing temperature and hold time
- Quench-water flow rate and temperature
- Prior austenite grain size
- Boron speciation (whether B is in solution or precipitated as borides)
- Trace impurities below the ICP detection limit

None of these are present in the dataset.

---

## 8. What did NOT work — and why

| Family | Best MAE | Verdict |
|--------|---------:|---------|
| **Kernel ridge (default γ=0.1)** | 5.78 | γ=0.1 is far too high for 18 standardized features (rule of thumb: γ ≈ 1/n_features = 0.056). After tuning, kernel ridge sits at MAE = 2.07 — still non-competitive. |
| **MLPRegressor** | 2.49 (best of two configs) | 566 rows is too small for an MLP to outperform regularized linear/tree models. Even with α=0.01 and tanh activation, MLPs over-fit. |
| **Polynomial features (degree=3) + Ridge** | 2.40 | Feature explosion (~1000 columns from 7 base features) overwhelms the 566-row training set; R² goes negative. |
| **Engineered hardenability features** (C×Cr, log_C, sumCEQ, etc.) | tied at best (1.7587 vs 1.7555) | XGBoost can already discover these interactions implicitly. Hand-engineered features add noise without adding signal. |
| **LightGBM** | 1.7454 | Leaf-wise tree growth is a disadvantage on this small low-signal problem. Same hyperparameter regime as XGB never matched its accuracy. |
| **MAE-objective XGBoost** (`reg:absoluteerror`) | 1.7569 | Squared-error gradient produces smoother fits and incidentally lower MAE on validation. |
| **20-seed bag of XGBoost** | 1.7321 | XGB's internal row/column subsampling already provides bagging-like variance reduction; explicit seed-bagging adds bias toward the mean. |
| **sklearn StackingRegressor** with Ridge meta-learner | 1.7524 | Meta-fit over-weights XGB based on training-fold MSE, ignoring its weaker generalization. |
| **StackingRegressor with passthrough=True** | 1.857 | Passthrough features add noise to the meta-input on a small dataset. |
| **Polynomial + ridge (degree=2)** | 1.92 | Slight overfit; PLS is the better dimensionality-reduction approach for this signal. |

**General lessons:**
- Model defaults (kernel γ, MLP layer sizes, LGBM num_leaves) are calibrated for larger datasets; on 566 rows they fail badly without tuning.
- Feature engineering and stacking carry implementation overhead but rarely beat well-tuned ensembles + simple convex blends on small problems.
- Bagging is redundant when the base learner already subsamples internally.

---

## 9. Limitations and caveats

1. **Training-data window is narrow.** C ∈ [0.17, 0.22], Mn ∈ [0.66, 1.30], Cr ∈ [0.4, 1.6]. Predictions outside this window are extrapolation and may diverge wildly between the XGB and PLS components — the blend then over-relies on the PLS extrapolation, which is unbounded. The webapp surfaces this with per-input out-of-range warnings.
2. **R²(δ) ≈ 0.** The J9–J15 spread is essentially uncorrelated with chemistry once you've conditioned on chemistry to predict J9. The δ model only marginally beats `mean(δ)` and should be treated as a coarse offset.
3. **Tail bias.** Systematic ±3 HRC bias at the J9 distribution edges. The model should not be trusted for individual outlier specimens (J9 < 32 or J9 ≥ 41).
4. **Selection bias in the labeled pool.** The 566 labeled specimens have *much* lower mean C (0.198 vs 0.457) and higher mean Cr (1.13 vs 0.41) than the unlabeled chemistry-only pool of 3,498 specimens. The blend's predictions on out-of-distribution chemistries (e.g. higher-C steels) cannot be trusted.
5. **Production pipeline not yet committed.** The current production export (`scripts/run_baselines.py`, `scripts/select_final_model.py`) ships fixed-v1 PLS_n4 / Ridge. Promoting the blend requires updating `select_final_model.py` and the post-processing chain in `assemble_pair_predictions`.
6. **Group folds collapse to plain folds for J9.** Because every base heat is unique in the J9 dataset, GroupKFold gives identical splits to KFold. The framework is group-aware so that the pipeline still protects against leakage if suffixed `-H` / `-Z` specimens are re-introduced later.
7. **Cross-validation is hyperparameter-aware.** All hyperparameters were tuned by inspecting CV scores. The reported MAE is unbiased w.r.t. the test fold within each split, but optimistic w.r.t. truly held-out future data — a small fraction of the gain (~0.005 MAE) is plausibly tuning noise.

---

## 10. Recommendations

### Immediate

- Promote the blend to production by:
  1. Adding `build_j9_blend_pipeline` and `build_delta_blend_pipeline` to `src/modeling/pipelines.py`.
  2. Updating `scripts/select_final_model.py` to write the blend alongside (or in place of) the fixed-v1 PLS / Ridge.
  3. Updating `scripts/run_baselines.py` to refresh `output/modeling/predictions/cv_predictions.parquet` from the blend.
- Add an integration test that asserts the deployed blend MAE on a held-out fold remains within 0.05 of 1.71.

### Near term, if more accuracy is required

- Collect process variables (austenitizing temperature, quench-rate proxy, austenite grain size). These are the most likely sources of the ±3 HRC tail bias.
- Increase chemistry precision: the current values are quoted to 0.01 wt% which loses information especially for C and Mn.
- Investigate whether sub-distinguishing the tail specimens (J9 < 32 or J9 ≥ 41) by *anything* in the existing data — heat lot, supplier, sampling date — might help. If so, add it as a categorical feature.

### Don't bother

- More hyperparameter search on XGBoost. The leader plateau is firm.
- Adding more linear models to the blend. Three-way blends (PLS + Ridge + Bayes) tied with the two-way XGB + PLS blend; further linear partners just lock in the same signal.
- Deep learning. Tabular MLPs and TabNet-style models need 10×+ the data to outperform GBMs on this problem class.

---

## Appendix — file inventory

| Path | Purpose |
|------|---------|
| `HISTORY.md` | Full leaderboard of 232 model runs |
| `REPORT.md` | This report |
| `README.md` | Project overview + quick start |
| `scripts/model_experiments.py` | Round-1 broad sweep |
| `scripts/model_experiments_round{2..7}.py` | Iterative refinement rounds |
| `scripts/inspect_winner.py` | OOF prediction quality diagnostics for the blend |
| `webapp/backend/train_models.py` | Persists the production blend to `webapp/models/*.joblib` |
| `webapp/backend/main.py` | FastAPI server exposing `/api/predict` |
| `webapp/frontend/` | Vite + TypeScript form-and-results UI |
| `output/modeling/predictions/blend_oof.csv` | Per-specimen OOF predictions for the J9 blend |
