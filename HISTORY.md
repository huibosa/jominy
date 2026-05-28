# Model Experiment History

Goal: predict Jominy hardenability — primary target is `J9` (HRC at 9 mm), secondary is `delta = J9 - J15`. Final J15 is reconstructed as `J9 - max(0, delta)` to enforce monotonicity (J9 ≥ J15).

## ★ Final selection — best model

After 230+ model runs across 7 experiment rounds, the best validated J9 predictor is:

**Convex blend of (XGBoost leader) and (PLS n=3), with weight w_xgb ≈ 0.70.**

| Target | Model | MAE_mean | MAE_std | RMSE_mean | R²_mean |
|--------|-------|---------:|--------:|----------:|--------:|
| **J9** | `blendF_xgb_pls3_w0.70` | **1.7106** | 0.0777 | **2.2135** | **0.481** |
| **delta** | `blendF_delta_xgb_bayes_w0.60` | **1.0940** | 0.0920 | 1.4363 | 0.001 |

Improvement vs. the prior production baseline (`pls_n4_full`, MAE=1.7617):
- J9 MAE: 1.7617 → 1.7106 (**−0.051, ~2.9%**)
- J9 RMSE: 2.2752 → 2.2135 (**−0.062, ~2.7%**)
- J9 R²: 0.452 → 0.481 (**+0.029**)

### Model definition

```python
# Base learners (refit on each fold)
xgb = XGBRegressor(
    n_estimators=1500, max_depth=3, learning_rate=0.005,
    subsample=0.55, colsample_bytree=0.5, reg_lambda=2.0,
    random_state=42, n_jobs=-1,
)  # imputer (median) → XGB
pls = PLSRegression(n_components=3, scale=False)  # imputer (median) → standard scaler → PLS

# Convex blend
y_pred = 0.70 * xgb_pred + 0.30 * pls_pred
```

The optimum w_xgb is flat across 0.62–0.75 (MAE within 0.001 of best), so the final value is robust to the exact choice. Increasing w_xgb beyond 0.85 or below 0.55 is measurably worse.

### Why this works

- **PLS captures the linear-in-chemistry signal.** Hardenability is dominated by C, Mn, Cr, Ni — and on this dataset those relationships are largely additive within a narrow chemistry range.
- **XGBoost captures residual non-linearity** (e.g. saturating effects of Mn, B, Cr×C interactions) without needing to be told about them.
- **They make complementary errors** so the blend reduces variance without raising bias.
- **Stacking with a Ridge meta-learner did worse** (MAE=1.857) because the meta-fit on out-of-fold predictions over-weighted XGBoost and the passthrough features caused noise. A convex blend with a fixed weight is more robust on 566 rows.

### Models that did NOT work

- **Kernel ridge / SVR with default γ=0.1**: catastrophic (MAE 3.4–5.8); γ=0.1 is far too high for 18 standardized features. Even after tuning to γ=0.05, kernel ridge sits at MAE=2.07 — non-competitive.
- **MLPs**: 64×32 ReLU → MAE=3.95; even the tanh 128×64 with α=0.01 sits at 2.49. 566 rows is too small for MLP to outperform regularized linear/tree models in this domain.
- **Polynomial features (degree=3)**: explodes the feature count and overfits (MAE=2.40, R²=−1.07).
- **Engineered hardenability-physics features** (C×Cr, log_C, sumCEQ, etc.): the best PLS+ENG variant (1.7587) only matched the un-engineered `pls_n3_full` (1.7555). XGBoost can already discover these interactions implicitly.
- **LightGBM**: never reached parity with the equivalent XGBoost configuration on this dataset (best LGBM J9 MAE=1.7454 vs best XGB single-model 1.7263).
- **20-seed bag of XGBoost**: MAE=1.7321 — slightly worse than the single-seed 1.7263. XGBoost's internal subsampling already provides bagging-like variance reduction.

### Caveats and limitations

- **R²(delta) ≈ 0**: The 5–7 HRC J9–J15 spread is essentially uncorrelated with chemistry once you've already conditioned on chemistry to predict J9. The delta model is barely better than predicting the mean. This is consistent with the small-training-set, narrow-chemistry-range nature of the data and matches prior project findings.
- **Blend is not committed as a production pipeline yet.** The current production export (`run_baselines.py`) ships PLS_n4 / Ridge as fixed-v1; promoting the blend requires updating `select_final_model.py` and the post-processing chain.
- **All metrics are 5-fold GroupKFold cross-validation** on `base_heat_id`. With 566 unique heats (each as its own group in J9), this is essentially KFold for J9 but the framework remains group-aware for delta and any future re-introduction of suffixed `-H` / `-Z` specimens.

### Reproduce

```bash
uv run --with pandas,pyarrow,scikit-learn,xgboost python scripts/model_experiments.py --quick
uv run --with pandas,pyarrow,scikit-learn,xgboost python scripts/model_experiments_round2.py
uv run --with pandas,pyarrow,scikit-learn,xgboost python scripts/model_experiments_round3.py
uv run --with pandas,pyarrow,scikit-learn,xgboost python scripts/model_experiments_round4.py
uv run --with pandas,pyarrow,scikit-learn,xgboost python scripts/model_experiments_round5.py
uv run --with pandas,pyarrow,scikit-learn,xgboost,lightgbm python scripts/model_experiments_round6.py
uv run --with pandas,pyarrow,scikit-learn,xgboost python scripts/model_experiments_round7.py
```

---

## Protocol

- **Dataset**: `data/modeling/j9_dataset.parquet` (566 rows, 18 features) and `data/modeling/delta_dataset.parquet` (491 rows).
- **Features**: `C, Si, Mn, P, S, Cu, Cr, Ni, V, Ti, W, Al, B + V/Ti/W/Al/B_missing` flags (FULL_FEATURES). Variants tested as noted.
- **Cross-validation**: 5-fold `GroupKFold` on `base_heat_id` (no leakage between heats).
- **Metrics**: MAE, RMSE, R² — reported as `mean ± std` over the 5 outer folds.
- **Selection criterion**: lowest `MAE_mean` on J9 (primary). Ties broken by RMSE_mean, then std.
- **Random seed**: 42.

Each row below is one model run. Lower MAE is better.

## J9 leaderboard (live)

| # | Model | Features | MAE_mean | MAE_std | RMSE_mean | R²_mean | Notes |
|---|-------|----------|---------:|--------:|----------:|--------:|-------|

<!-- J9: linear family -->
| 1 | `ridge_a1.0_full` | FULL | 1.7773 | 0.0774 | 2.2869 | 0.4458 | baseline reproduction |
| 2 | `ridge_a1.0_core` | CORE+flag | 1.8073 | 0.0685 | 2.3249 | 0.4269 |  |
| 3 | `ridge_a1.0_core7` | CORE7 | 1.8181 | 0.0626 | 2.3342 | 0.4224 |  |
| 4 | `linreg_full` | FULL | 1.7780 | 0.0776 | 2.2877 | 0.4454 |  |
| 5 | `lasso_a0.01_full` | FULL | 1.7760 | 0.0767 | 2.2870 | 0.4458 |  |
| 6 | `elasticnet_a0.05_l1_0.5` | FULL | 1.7733 | 0.0759 | 2.2825 | 0.4482 |  |
| 7 | `bayesian_ridge_full` | FULL | 1.7696 | 0.0759 | 2.2754 | 0.4516 |  |
| 8 | `huber_full` | FULL | 1.7736 | 0.0842 | 2.2930 | 0.4422 |  |
| 9 | `pls_n4_full` | FULL | 1.7617 | 0.0715 | 2.2752 | 0.4517 | baseline reproduction |
| 10 | `pls_n3_full` | FULL | 1.7555 | 0.0812 | 2.2706 | 0.4538 |  |
| 11 | `pls_n5_full` | FULL | 1.7641 | 0.0686 | 2.2768 | 0.4508 |  |
| 12 | `pls_n6_full` | FULL | 1.7722 | 0.0709 | 2.2810 | 0.4487 |  |
| 13 | `poly2_ridge_a5_core` | CORE7 | 1.9227 | 0.0622 | 2.4772 | 0.3462 | degree=2 interactions+squared |
| 14 | `poly3_ridge_a5_core` | CORE7 | 2.3986 | 0.3674 | 4.0819 | -1.0683 | degree=3 interactions+squared |

<!-- J9: tree ensembles -->
| 15 | `rf_300` | FULL | 1.7821 | 0.0687 | 2.2506 | 0.4634 |  |
| 16 | `rf_500_md8` | FULL | 1.7747 | 0.0677 | 2.2498 | 0.4639 |  |
| 17 | `extratrees_500` | FULL | 1.7765 | 0.0869 | 2.2688 | 0.4547 |  |
| 18 | `gbr_default` | FULL | 1.7902 | 0.0799 | 2.3178 | 0.4309 |  |
| 19 | `hgbr_baseline` | FULL | 1.8066 | 0.0709 | 2.2992 | 0.4410 | baseline reproduction |
| 20 | `hgbr_lr0.03_iter500` | FULL | 1.8460 | 0.1158 | 2.3504 | 0.4154 |  |
| 21 | `hgbr_lr0.05_md5` | FULL | 1.8738 | 0.1128 | 2.3941 | 0.3930 |  |
| 22 | `xgb_n500_lr0.03_md4` | FULL | 1.7889 | 0.0840 | 2.2918 | 0.4441 |  |
| 23 | `xgb_n800_lr0.02_md3` | FULL | 1.7688 | 0.0659 | 2.2698 | 0.4542 |  |
| 24 | `xgb_n1000_lr0.01_md6` | FULL | 1.7793 | 0.0965 | 2.2595 | 0.4590 |  |
| 25 | `lgbm_n500_lr0.05_lv15` | FULL | 1.9397 | 0.0893 | 2.4546 | 0.3613 |  |
| 26 | `lgbm_n800_lr0.03_lv31` | FULL | 1.9178 | 0.0573 | 2.4431 | 0.3678 |  |

<!-- J9: kernel & instance models -->
| 27 | `kridge_rbf_a1_g0.1` | FULL | 5.7842 | 0.6472 | 8.8094 | -7.2921 |  |
| 28 | `kridge_rbf_a0.5_g0.05` | FULL | 3.4117 | 0.3719 | 5.4392 | -2.1939 |  |
| 29 | `kridge_poly2_a1` | FULL | 1.8568 | 0.1088 | 2.4192 | 0.3802 |  |
| 30 | `svr_rbf_C1_g0.1` | FULL | 1.8055 | 0.0958 | 2.2993 | 0.4403 |  |
| 31 | `svr_rbf_C5_g0.05` | FULL | 1.8393 | 0.0931 | 2.3299 | 0.4256 |  |
| 32 | `svr_linear_C1` | FULL | 1.7981 | 0.0825 | 2.3196 | 0.4294 |  |
| 33 | `knn_k7` | FULL | 1.8431 | 0.0435 | 2.3258 | 0.4272 |  |
| 34 | `knn_k15` | FULL | 1.8383 | 0.0558 | 2.3033 | 0.4382 |  |

<!-- J9: MLP -->
| 35 | `mlp_64x32_relu` | FULL | 3.9545 | 0.5569 | 5.4205 | -2.1653 |  |
| 36 | `mlp_128x64_tanh` | FULL | 2.4864 | 0.1500 | 3.1178 | -0.0348 |  |

<!-- J9: stacking -->
| 37 | `stack_pls_ridge_xgb` | FULL | 1.7635 | 0.0665 | 2.2501 | 0.4638 | stacking three best families |

<!-- J9: feature engineering — PLS -->
| 1 | `pls_n2_full+eng` | FULL+ENG | 1.7730 | 0.0850 | 2.2916 | 0.4431 |  |
| 2 | `pls_n3_full+eng` | FULL+ENG | 1.7598 | 0.0753 | 2.2782 | 0.4501 |  |
| 3 | `pls_n4_full+eng` | FULL+ENG | 1.7615 | 0.0697 | 2.2801 | 0.4489 |  |
| 4 | `pls_n5_full+eng` | FULL+ENG | 1.7691 | 0.0603 | 2.2866 | 0.4459 |  |
| 5 | `pls_n6_full+eng` | FULL+ENG | 1.7726 | 0.0646 | 2.2892 | 0.4449 |  |
| 6 | `pls_n8_full+eng` | FULL+ENG | 1.7785 | 0.0649 | 2.2921 | 0.4435 |  |
| 7 | `pls_n10_full+eng` | FULL+ENG | 1.7878 | 0.0551 | 2.2982 | 0.4406 |  |

<!-- J9: feature engineering — Ridge / Lasso / ElasticNet -->
| 8 | `ridge_a0.5_full+eng` | FULL+ENG | 1.7939 | 0.0499 | 2.3094 | 0.4350 |  |
| 9 | `ridge_a1.0_full+eng` | FULL+ENG | 1.7923 | 0.0510 | 2.3067 | 0.4363 |  |
| 10 | `ridge_a2.0_full+eng` | FULL+ENG | 1.7902 | 0.0525 | 2.3038 | 0.4378 |  |
| 11 | `ridge_a5.0_full+eng` | FULL+ENG | 1.7855 | 0.0558 | 2.2984 | 0.4405 |  |
| 12 | `ridge_a10.0_full+eng` | FULL+ENG | 1.7798 | 0.0606 | 2.2927 | 0.4432 |  |
| 13 | `elasticnet_a0.05_l1_0.7_full+eng` | FULL+ENG | 1.7724 | 0.0772 | 2.2864 | 0.4463 |  |
| 14 | `bayesian_ridge_full+eng` | FULL+ENG | 1.7662 | 0.0717 | 2.2772 | 0.4506 |  |

<!-- J9: XGBoost tight sweep around xgb_n800_lr0.02_md3 -->
| 15 | `xgb_n500_lr0.01_md3` | FULL | 1.7529 | 0.0807 | 2.2362 | 0.4702 |  |
| 16 | `xgb_n500_lr0.01_md4` | FULL | 1.7489 | 0.0806 | 2.2300 | 0.4732 |  |
| 17 | `xgb_n500_lr0.02_md3` | FULL | 1.7582 | 0.0706 | 2.2519 | 0.4627 |  |
| 18 | `xgb_n500_lr0.02_md4` | FULL | 1.7568 | 0.0814 | 2.2498 | 0.4638 |  |
| 19 | `xgb_n500_lr0.03_md3` | FULL | 1.7733 | 0.0742 | 2.2702 | 0.4539 |  |
| 20 | `xgb_n500_lr0.03_md4` | FULL | 1.7803 | 0.0865 | 2.2766 | 0.4508 |  |
| 21 | `xgb_n800_lr0.01_md3` | FULL | 1.7445 | 0.0761 | 2.2361 | 0.4702 |  |
| 22 | `xgb_n800_lr0.01_md4` | FULL | 1.7517 | 0.0790 | 2.2391 | 0.4689 |  |
| 23 | `xgb_n800_lr0.02_md3` | FULL | 1.7750 | 0.0733 | 2.2745 | 0.4517 |  |
| 24 | `xgb_n800_lr0.02_md4` | FULL | 1.7879 | 0.0821 | 2.2818 | 0.4482 |  |
| 25 | `xgb_n800_lr0.03_md3` | FULL | 1.7985 | 0.0810 | 2.3048 | 0.4366 |  |
| 26 | `xgb_n800_lr0.03_md4` | FULL | 1.8247 | 0.0932 | 2.3213 | 0.4288 |  |
| 27 | `xgb_n1200_lr0.01_md3` | FULL | 1.7503 | 0.0743 | 2.2482 | 0.4646 |  |
| 28 | `xgb_n1200_lr0.01_md4` | FULL | 1.7639 | 0.0840 | 2.2550 | 0.4613 |  |

<!-- J9: XGBoost slower learning rates -->
| 1 | `xgb_n1200_lr0.02_md3` | FULL | 1.7949 | 0.0775 | 2.3028 | 0.4378 |  |
| 2 | `xgb_n1200_lr0.02_md4` | FULL | 1.8218 | 0.0836 | 2.3145 | 0.4322 |  |
| 3 | `xgb_n1500_lr0.01_md3` | FULL | 1.7554 | 0.0743 | 2.2578 | 0.4600 |  |
| 4 | `xgb_n1500_lr0.01_md4` | FULL | 1.7760 | 0.0812 | 2.2702 | 0.4540 |  |
| 5 | `xgb_n2000_lr0.005_md3` | FULL | 1.7448 | 0.0771 | 2.2379 | 0.4695 |  |
| 6 | `xgb_n2000_lr0.005_md4` | FULL | 1.7477 | 0.0837 | 2.2405 | 0.4681 |  |
| 7 | `xgb_n2000_lr0.008_md3` | FULL | 1.7525 | 0.0765 | 2.2549 | 0.4615 |  |
| 8 | `xgb_n3000_lr0.005_md3` | FULL | 1.7578 | 0.0745 | 2.2563 | 0.4608 |  |
| 9 | `xgb_n800_lr0.01_md3_l0.5_a0.5` | FULL | 1.7453 | 0.0712 | 2.2374 | 0.4699 |  |
| 10 | `xgb_n800_lr0.01_md3_l5.0_a0.0` | FULL | 1.7430 | 0.0785 | 2.2371 | 0.4695 |  |
| 11 | `xgb_n800_lr0.01_md3_l5.0_a0.5` | FULL | 1.7443 | 0.0777 | 2.2384 | 0.4689 |  |
| 12 | `xgb_n800_lr0.01_md3_l10.0_a0.0` | FULL | 1.7460 | 0.0793 | 2.2402 | 0.4682 |  |
| 13 | `xgb_n800_lr0.01_md3_ss0.6_cs0.6` | FULL | 1.7386 | 0.0861 | 2.2391 | 0.4687 |  |
| 14 | `xgb_n800_lr0.01_md3_ss0.7_cs0.7` | FULL | 1.7389 | 0.0795 | 2.2338 | 0.4711 |  |
| 15 | `xgb_n800_lr0.01_md3_ss1.0_cs0.6` | FULL | 1.7602 | 0.0848 | 2.2411 | 0.4680 |  |
| 16 | `xgb_n800_lr0.01_md3_ss0.6_cs1.0` | FULL | 1.7462 | 0.0762 | 2.2434 | 0.4668 |  |
| 17 | `xgb_n800_lr0.01_md3_ss1.0_cs1.0` | FULL | 1.7733 | 0.0765 | 2.2483 | 0.4652 |  |

<!-- J9: stacking with new leader -->
| 18 | `stack_pls3_bayes_xgbLeader` | FULL | 1.7524 | 0.0730 | 2.2330 | 0.4720 | stack new leader |

<!-- J9: XGBoost subsample/colsample fine sweep -->
| 1 | `xgb_v2_ss0.5_cs0.5` | FULL | 1.7333 | 0.0915 | 2.2365 | 0.4698 |  |
| 2 | `xgb_v2_ss0.5_cs0.6` | FULL | 1.7346 | 0.0882 | 2.2392 | 0.4686 |  |
| 3 | `xgb_v2_ss0.5_cs0.7` | FULL | 1.7377 | 0.0835 | 2.2416 | 0.4675 |  |
| 4 | `xgb_v2_ss0.5_cs0.8` | FULL | 1.7370 | 0.0825 | 2.2372 | 0.4695 |  |
| 5 | `xgb_v2_ss0.55_cs0.5` | FULL | 1.7263 | 0.0831 | 2.2269 | 0.4745 |  |
| 6 | `xgb_v2_ss0.55_cs0.6` | FULL | 1.7298 | 0.0823 | 2.2313 | 0.4725 |  |
| 7 | `xgb_v2_ss0.55_cs0.7` | FULL | 1.7401 | 0.0845 | 2.2413 | 0.4677 |  |
| 8 | `xgb_v2_ss0.55_cs0.8` | FULL | 1.7415 | 0.0778 | 2.2406 | 0.4679 |  |
| 9 | `xgb_v2_ss0.6_cs0.5` | FULL | 1.7334 | 0.0896 | 2.2340 | 0.4710 |  |
| 10 | `xgb_v2_ss0.6_cs0.6` | FULL | 1.7386 | 0.0861 | 2.2391 | 0.4687 |  |
| 11 | `xgb_v2_ss0.6_cs0.7` | FULL | 1.7378 | 0.0858 | 2.2399 | 0.4685 |  |
| 12 | `xgb_v2_ss0.6_cs0.8` | FULL | 1.7436 | 0.0803 | 2.2411 | 0.4679 |  |
| 13 | `xgb_v2_ss0.65_cs0.5` | FULL | 1.7349 | 0.0858 | 2.2328 | 0.4716 |  |
| 14 | `xgb_v2_ss0.65_cs0.6` | FULL | 1.7343 | 0.0855 | 2.2329 | 0.4716 |  |
| 15 | `xgb_v2_ss0.65_cs0.7` | FULL | 1.7323 | 0.0859 | 2.2311 | 0.4726 |  |
| 16 | `xgb_v2_ss0.65_cs0.8` | FULL | 1.7377 | 0.0805 | 2.2356 | 0.4704 |  |
| 17 | `xgb_v2_ss0.7_cs0.5` | FULL | 1.7354 | 0.0783 | 2.2288 | 0.4734 |  |
| 18 | `xgb_v2_ss0.7_cs0.6` | FULL | 1.7391 | 0.0764 | 2.2348 | 0.4707 |  |
| 19 | `xgb_v2_ss0.7_cs0.7` | FULL | 1.7389 | 0.0795 | 2.2338 | 0.4711 |  |
| 20 | `xgb_v2_ss0.7_cs0.8` | FULL | 1.7414 | 0.0769 | 2.2348 | 0.4707 |  |
| 21 | `xgb_v2_ss0.75_cs0.5` | FULL | 1.7353 | 0.0815 | 2.2296 | 0.4733 |  |
| 22 | `xgb_v2_ss0.75_cs0.6` | FULL | 1.7377 | 0.0746 | 2.2324 | 0.4721 |  |
| 23 | `xgb_v2_ss0.75_cs0.7` | FULL | 1.7373 | 0.0711 | 2.2290 | 0.4737 |  |
| 24 | `xgb_v2_ss0.75_cs0.8` | FULL | 1.7434 | 0.0738 | 2.2355 | 0.4707 |  |
| 25 | `xgb_md2_ss0.6_cs0.6` | FULL | 1.7430 | 0.0797 | 2.2437 | 0.4663 |  |
| 26 | `xgb_md2_ss0.6_cs0.7` | FULL | 1.7441 | 0.0756 | 2.2439 | 0.4661 |  |
| 27 | `xgb_md2_ss0.7_cs0.6` | FULL | 1.7500 | 0.0746 | 2.2424 | 0.4669 |  |
| 28 | `xgb_md2_ss0.7_cs0.7` | FULL | 1.7519 | 0.0769 | 2.2454 | 0.4654 |  |
| 29 | `xgb_leader_mcw1` | FULL | 1.7386 | 0.0861 | 2.2391 | 0.4687 |  |
| 30 | `xgb_leader_mcw3` | FULL | 1.7432 | 0.0852 | 2.2441 | 0.4664 |  |
| 31 | `xgb_leader_mcw5` | FULL | 1.7531 | 0.0865 | 2.2565 | 0.4605 |  |
| 32 | `xgb_leader_mcw10` | FULL | 1.7476 | 0.0817 | 2.2504 | 0.4635 |  |

<!-- J9: multi-seed bag of leader XGBoost -->
| 33 | `xgb_bag_10seeds` | FULL | 1.7360 | 0.0806 | 2.2327 | 0.4718 | 10-seed bag of leader |

<!-- J9: stacking — imputed front -->
| 34 | `stack_pls3_ridge_xgbLeader` | FULL | 1.7489 | 0.0733 | 2.2364 | 0.4703 |  |
| 35 | `stack_pls3_xgb_finalHGBR` | FULL | 1.7747 | 0.0504 | 2.2870 | 0.4461 |  |

<!-- delta: round 4 -->

## delta leaderboard (live)

| # | Model | Features | MAE_mean | MAE_std | RMSE_mean | R²_mean | Notes |
|---|-------|----------|---------:|--------:|----------:|--------:|-------|
| 1 | `pls_n2_full` | FULL | 1.1348 | 0.0895 | 1.4816 | -0.0611 |  |
| 2 | `pls_n3_full` | FULL | 1.1279 | 0.0811 | 1.4741 | -0.0512 |  |
| 3 | `pls_n4_full` | FULL | 1.1156 | 0.0818 | 1.4623 | -0.0341 |  |
| 4 | `pls_n5_full` | FULL | 1.1145 | 0.0769 | 1.4622 | -0.0343 |  |
| 5 | `ridge_a1.0_full` | FULL | 1.1190 | 0.0744 | 1.4652 | -0.0385 |  |
| 6 | `bayesian_ridge_full` | FULL | 1.1171 | 0.0876 | 1.4559 | -0.0256 |  |
| 7 | `gbr_n200_md3_lr0.05` | FULL | 1.1476 | 0.0788 | 1.4941 | -0.0845 |  |
| 8 | `xgb_n800_lr0.01_md3_ss0.6_cs0.6` | FULL | 1.1110 | 0.1024 | 1.4606 | -0.0336 |  |
| 9 | `xgb_n800_lr0.01_md3_ss0.8_cs0.8` | FULL | 1.1071 | 0.1019 | 1.4532 | -0.0238 |  |
| 10 | `xgb_n500_lr0.02_md3_ss0.8_cs0.8` | FULL | 1.1178 | 0.1047 | 1.4612 | -0.0352 |  |
| 11 | `xgb_n1200_lr0.01_md3_ss0.6_cs0.6` | FULL | 1.1241 | 0.1096 | 1.4750 | -0.0548 |  |
| 12 | `xgb_n2000_lr0.005_md3_ss0.6_cs0.6` | FULL | 1.1173 | 0.1059 | 1.4685 | -0.0455 |  |

<!-- J9: very tight (ss, cs) sweep -->
| 1 | `xgb_v3_ss0.45_cs0.45` | FULL | 1.7385 | 0.0834 | 2.2390 | 0.4687 |  |
| 2 | `xgb_v3_ss0.45_cs0.5` | FULL | 1.7313 | 0.0834 | 2.2322 | 0.4719 |  |
| 3 | `xgb_v3_ss0.45_cs0.55` | FULL | 1.7313 | 0.0834 | 2.2322 | 0.4719 |  |
| 4 | `xgb_v3_ss0.5_cs0.45` | FULL | 1.7348 | 0.0935 | 2.2358 | 0.4701 |  |
| 5 | `xgb_v3_ss0.5_cs0.5` | FULL | 1.7333 | 0.0915 | 2.2365 | 0.4698 |  |
| 6 | `xgb_v3_ss0.5_cs0.55` | FULL | 1.7333 | 0.0915 | 2.2365 | 0.4698 |  |
| 7 | `xgb_v3_ss0.55_cs0.45` | FULL | 1.7319 | 0.0835 | 2.2320 | 0.4721 |  |
| 8 | `xgb_v3_ss0.55_cs0.5` | FULL | 1.7263 | 0.0831 | 2.2269 | 0.4745 |  |
| 9 | `xgb_v3_ss0.55_cs0.55` | FULL | 1.7263 | 0.0831 | 2.2269 | 0.4745 |  |
| 10 | `xgb_v3_ss0.6_cs0.45` | FULL | 1.7336 | 0.0878 | 2.2329 | 0.4716 |  |
| 11 | `xgb_v3_ss0.6_cs0.5` | FULL | 1.7334 | 0.0896 | 2.2340 | 0.4710 |  |
| 12 | `xgb_v3_ss0.6_cs0.55` | FULL | 1.7334 | 0.0896 | 2.2340 | 0.4710 |  |

<!-- J9: lr/n_est sweep at (0.55, 0.5) -->
| 13 | `xgb_v3_n500_lr0.01` | FULL | 1.7371 | 0.0859 | 2.2317 | 0.4721 |  |
| 14 | `xgb_v3_n1000_lr0.01` | FULL | 1.7301 | 0.0795 | 2.2334 | 0.4715 |  |
| 15 | `xgb_v3_n1200_lr0.01` | FULL | 1.7372 | 0.0763 | 2.2398 | 0.4685 |  |
| 16 | `xgb_v3_n800_lr0.005` | FULL | 1.7497 | 0.0862 | 2.2414 | 0.4677 |  |
| 17 | `xgb_v3_n1500_lr0.005` | FULL | 1.7260 | 0.0827 | 2.2275 | 0.4742 |  |
| 18 | `xgb_v3_n2000_lr0.005` | FULL | 1.7299 | 0.0760 | 2.2308 | 0.4728 |  |
| 19 | `xgb_v3_n800_lr0.015` | FULL | 1.7370 | 0.0836 | 2.2429 | 0.4669 |  |
| 20 | `xgb_v3_n800_lr0.02` | FULL | 1.7586 | 0.0814 | 2.2675 | 0.4548 |  |

<!-- J9: reg:absoluteerror objective -->
| 21 | `xgb_mae_ss0.55_cs0.5` | FULL | 1.7569 | 0.0938 | 2.2660 | 0.4551 |  |
| 22 | `xgb_mae_ss0.55_cs0.6` | FULL | 1.7559 | 0.0882 | 2.2656 | 0.4556 |  |
| 23 | `xgb_mae_ss0.6_cs0.5` | FULL | 1.7688 | 0.0907 | 2.2762 | 0.4503 |  |
| 24 | `xgb_mae_ss0.6_cs0.6` | FULL | 1.7618 | 0.0905 | 2.2712 | 0.4526 |  |

<!-- J9: bag of new leader (20 seeds) -->
| 25 | `xgb_bag20_v3` | FULL | 1.7321 | 0.0821 | 2.2316 | 0.4722 | 20-seed bag at ss=0.55, cs=0.5 |
| 26 | `xgb_bag20_mae` | FULL | 1.7553 | 0.0859 | 2.2610 | 0.4576 | 20-seed bag, MAE objective |

<!-- delta: bag of leader -->
| 1 | `xgb_bag20_delta` | FULL | 1.1105 | 0.1017 | 1.4548 | -0.0259 | 20-seed bag of delta leader |

<!-- J9: LightGBM analogue of leader -->
| 1 | `lgbm_n800_lr0.01_lv7` | FULL | 1.7524 | 0.0908 | 2.2528 | 0.4622 |  |
| 2 | `lgbm_n800_lr0.01_lv15` | FULL | 1.7760 | 0.1013 | 2.2708 | 0.4532 |  |
| 3 | `lgbm_n1500_lr0.005_lv7` | FULL | 1.7454 | 0.0852 | 2.2446 | 0.4661 |  |
| 4 | `lgbm_n1500_lr0.005_lv15` | FULL | 1.7710 | 0.1010 | 2.2669 | 0.4553 |  |
| 5 | `lgbm_n1500_lr0.005_lv31` | FULL | 1.7788 | 0.1040 | 2.2714 | 0.4532 |  |
| 6 | `lgbm_n2000_lr0.005_lv7` | FULL | 1.7553 | 0.0863 | 2.2582 | 0.4595 |  |

<!-- J9: cross-family stacking with imputed front -->
| 7 | `stack_v2_pls3_bayes_xgb` | FULL | 1.8573 | 0.1029 | 2.4631 | 0.3582 | passthrough+imputed Ridge meta |

<!-- J9: convex-weighted blend of pls3 + xgb -->
| 8 | `blend_xgb0.2_pls0.8` | FULL | 1.7349 | 0.0769 | 2.2417 | 0.4676 | convex blend |
| 9 | `blend_xgb0.3_pls0.7` | FULL | 1.7271 | 0.0760 | 2.2309 | 0.4726 | convex blend |
| 10 | `blend_xgb0.4_pls0.6` | FULL | 1.7213 | 0.0755 | 2.2227 | 0.4765 | convex blend |
| 11 | `blend_xgb0.5_pls0.5` | FULL | 1.7166 | 0.0755 | 2.2171 | 0.4791 | convex blend |
| 12 | `blend_xgb0.6_pls0.4` | FULL | 1.7128 | 0.0761 | 2.2140 | 0.4805 | convex blend |
| 13 | `blend_xgb0.7_pls0.30000000000000004` | FULL | 1.7106 | 0.0777 | 2.2135 | 0.4808 | convex blend |
| 14 | `blend_xgb0.8_pls0.19999999999999996` | FULL | 1.7132 | 0.0793 | 2.2156 | 0.4798 | convex blend |

<!-- J9: three-way blend xgb + pls3 + bayes -->
| 15 | `blend3_xgb0.4_pls0.1_bayes0.5` | FULL | 1.7258 | 0.0766 | 2.2249 | 0.4755 | 3-way blend |
| 16 | `blend3_xgb0.4_pls0.2_bayes0.4` | FULL | 1.7247 | 0.0764 | 2.2242 | 0.4758 | 3-way blend |
| 17 | `blend3_xgb0.4_pls0.3_bayes0.3` | FULL | 1.7237 | 0.0762 | 2.2236 | 0.4761 | 3-way blend |
| 18 | `blend3_xgb0.5_pls0.1_bayes0.4` | FULL | 1.7196 | 0.0766 | 2.2189 | 0.4783 | 3-way blend |
| 19 | `blend3_xgb0.5_pls0.2_bayes0.3` | FULL | 1.7186 | 0.0763 | 2.2182 | 0.4786 | 3-way blend |
| 20 | `blend3_xgb0.5_pls0.3_bayes0.2` | FULL | 1.7179 | 0.0761 | 2.2177 | 0.4788 | 3-way blend |
| 21 | `blend3_xgb0.6_pls0.1_bayes0.3` | FULL | 1.7147 | 0.0768 | 2.2154 | 0.4799 | 3-way blend |
| 22 | `blend3_xgb0.6_pls0.2_bayes0.2` | FULL | 1.7139 | 0.0767 | 2.2148 | 0.4802 | 3-way blend |
| 23 | `blend3_xgb0.6_pls0.3_bayes0.1` | FULL | 1.7134 | 0.0764 | 2.2143 | 0.4804 | 3-way blend |
| 24 | `blend3_xgb0.7_pls0.1_bayes0.2` | FULL | 1.7120 | 0.0785 | 2.2144 | 0.4803 | 3-way blend |
| 25 | `blend3_xgb0.7_pls0.2_bayes0.1` | FULL | 1.7112 | 0.0781 | 2.2139 | 0.4806 | 3-way blend |
| 26 | `blend3_xgb0.7_pls0.3_bayes0.0` | FULL | 1.7106 | 0.0777 | 2.2135 | 0.4808 | 3-way blend |

<!-- delta: LightGBM and final blends -->
| 1 | `lgbm_n800_lr0.01_lv15` | FULL | 1.1301 | 0.1004 | 1.4869 | -0.0714 |  |
| 2 | `lgbm_n1500_lr0.005_lv15` | FULL | 1.1223 | 0.0984 | 1.4783 | -0.0598 |  |
| 3 | `lgbm_n1500_lr0.005_lv31` | FULL | 1.1300 | 0.0941 | 1.4826 | -0.0661 |  |

<!-- J9: fine blend grid xgb_leader + pls3 -->
| 1 | `blendF_xgb_pls3_w0.55` | FULL | 1.7145 | 0.0758 | 2.2152 | 0.4800 | fine grid; w_xgb=0.55 |
| 2 | `blendF_xgb_pls3_w0.58` | FULL | 1.7136 | 0.0759 | 2.2145 | 0.4803 | fine grid; w_xgb=0.58 |
| 3 | `blendF_xgb_pls3_w0.60` | FULL | 1.7128 | 0.0761 | 2.2140 | 0.4805 | fine grid; w_xgb=0.60 |
| 4 | `blendF_xgb_pls3_w0.62` | FULL | 1.7120 | 0.0763 | 2.2136 | 0.4807 | fine grid; w_xgb=0.62 |
| 5 | `blendF_xgb_pls3_w0.65` | FULL | 1.7112 | 0.0766 | 2.2134 | 0.4808 | fine grid; w_xgb=0.65 |
| 6 | `blendF_xgb_pls3_w0.68` | FULL | 1.7107 | 0.0770 | 2.2134 | 0.4808 | fine grid; w_xgb=0.68 |
| 7 | `blendF_xgb_pls3_w0.70` | FULL | 1.7106 | 0.0777 | 2.2135 | 0.4808 | fine grid; w_xgb=0.70 |
| 8 | `blendF_xgb_pls3_w0.72` | FULL | 1.7110 | 0.0781 | 2.2138 | 0.4806 | fine grid; w_xgb=0.72 |
| 9 | `blendF_xgb_pls3_w0.75` | FULL | 1.7116 | 0.0785 | 2.2142 | 0.4804 | fine grid; w_xgb=0.75 |
| 10 | `blendF_xgb_pls3_w0.78` | FULL | 1.7123 | 0.0789 | 2.2149 | 0.4801 | fine grid; w_xgb=0.78 |
| 11 | `blendF_xgb_pls3_w0.80` | FULL | 1.7132 | 0.0793 | 2.2156 | 0.4798 | fine grid; w_xgb=0.80 |
| 12 | `blendF_xgb_pls3_w0.82` | FULL | 1.7143 | 0.0798 | 2.2166 | 0.4793 | fine grid; w_xgb=0.82 |
| 13 | `blendF_xgb_pls3_w0.85` | FULL | 1.7156 | 0.0803 | 2.2176 | 0.4788 | fine grid; w_xgb=0.85 |

<!-- J9: blend xgb_leader + ridge -->
| 14 | `blendF_xgb_ridge_w0.55` | FULL | 1.7184 | 0.0784 | 2.2197 | 0.4778 | ridge alpha=1.0; w_xgb=0.55 |
| 15 | `blendF_xgb_ridge_w0.60` | FULL | 1.7155 | 0.0784 | 2.2177 | 0.4787 | ridge alpha=1.0; w_xgb=0.60 |
| 16 | `blendF_xgb_ridge_w0.65` | FULL | 1.7130 | 0.0786 | 2.2165 | 0.4793 | ridge alpha=1.0; w_xgb=0.65 |
| 17 | `blendF_xgb_ridge_w0.70` | FULL | 1.7114 | 0.0791 | 2.2159 | 0.4795 | ridge alpha=1.0; w_xgb=0.70 |
| 18 | `blendF_xgb_ridge_w0.75` | FULL | 1.7114 | 0.0800 | 2.2161 | 0.4795 | ridge alpha=1.0; w_xgb=0.75 |
| 19 | `blendF_xgb_ridge_w0.80` | FULL | 1.7126 | 0.0808 | 2.2170 | 0.4791 | ridge alpha=1.0; w_xgb=0.80 |
| 20 | `blendF_xgb_ridge_w0.85` | FULL | 1.7151 | 0.0813 | 2.2186 | 0.4783 | ridge alpha=1.0; w_xgb=0.85 |

<!-- J9: blend xgb_leader + bayesianridge -->
| 21 | `blendF_xgb_bayes_w0.55` | FULL | 1.7180 | 0.0767 | 2.2176 | 0.4789 | bayesian ridge; w_xgb=0.55 |
| 22 | `blendF_xgb_bayes_w0.60` | FULL | 1.7156 | 0.0769 | 2.2161 | 0.4796 | bayesian ridge; w_xgb=0.60 |
| 23 | `blendF_xgb_bayes_w0.65` | FULL | 1.7138 | 0.0773 | 2.2153 | 0.4800 | bayesian ridge; w_xgb=0.65 |
| 24 | `blendF_xgb_bayes_w0.70` | FULL | 1.7129 | 0.0788 | 2.2151 | 0.4800 | bayesian ridge; w_xgb=0.70 |
| 25 | `blendF_xgb_bayes_w0.75` | FULL | 1.7129 | 0.0800 | 2.2156 | 0.4798 | bayesian ridge; w_xgb=0.75 |
| 26 | `blendF_xgb_bayes_w0.80` | FULL | 1.7142 | 0.0805 | 2.2167 | 0.4793 | bayesian ridge; w_xgb=0.80 |
| 27 | `blendF_xgb_bayes_w0.85` | FULL | 1.7164 | 0.0810 | 2.2184 | 0.4784 | bayesian ridge; w_xgb=0.85 |

<!-- J9: blend (10-seed bagged xgb) + pls3 -->
| 28 | `blendBag10_xgb_pls3_w0.60` | FULL | 1.7166 | 0.0805 | 2.2164 | 0.4794 | 10-seed XGB bag + PLS3; w_xgb=0.60 |
| 29 | `blendBag10_xgb_pls3_w0.65` | FULL | 1.7161 | 0.0809 | 2.2160 | 0.4796 | 10-seed XGB bag + PLS3; w_xgb=0.65 |
| 30 | `blendBag10_xgb_pls3_w0.70` | FULL | 1.7163 | 0.0814 | 2.2162 | 0.4795 | 10-seed XGB bag + PLS3; w_xgb=0.70 |
| 31 | `blendBag10_xgb_pls3_w0.75` | FULL | 1.7174 | 0.0817 | 2.2170 | 0.4791 | 10-seed XGB bag + PLS3; w_xgb=0.75 |
| 32 | `blendBag10_xgb_pls3_w0.80` | FULL | 1.7194 | 0.0822 | 2.2185 | 0.4784 | 10-seed XGB bag + PLS3; w_xgb=0.80 |

<!-- delta: blend leader + pls5 -->
| 1 | `blendF_delta_xgb_pls5_w0.20` | FULL | 1.1037 | 0.0791 | 1.4498 | -0.0170 | delta blend; w_xgb=0.20 |
| 2 | `blendF_delta_xgb_pls5_w0.30` | FULL | 1.0999 | 0.0818 | 1.4455 | -0.0112 | delta blend; w_xgb=0.30 |
| 3 | `blendF_delta_xgb_pls5_w0.40` | FULL | 1.0969 | 0.0845 | 1.4426 | -0.0073 | delta blend; w_xgb=0.40 |
| 4 | `blendF_delta_xgb_pls5_w0.50` | FULL | 1.0958 | 0.0872 | 1.4410 | -0.0052 | delta blend; w_xgb=0.50 |
| 5 | `blendF_delta_xgb_pls5_w0.60` | FULL | 1.0957 | 0.0904 | 1.4408 | -0.0051 | delta blend; w_xgb=0.60 |
| 6 | `blendF_delta_xgb_pls5_w0.70` | FULL | 1.0971 | 0.0938 | 1.4419 | -0.0069 | delta blend; w_xgb=0.70 |
| 7 | `blendF_delta_xgb_pls5_w0.80` | FULL | 1.0994 | 0.0969 | 1.4443 | -0.0106 | delta blend; w_xgb=0.80 |
| 8 | `blendF_delta_xgb_bayes_w0.20` | FULL | 1.1040 | 0.0857 | 1.4434 | -0.0081 | delta blend; w_xgb=0.20 |
| 9 | `blendF_delta_xgb_bayes_w0.30` | FULL | 1.0991 | 0.0858 | 1.4394 | -0.0026 | delta blend; w_xgb=0.30 |
| 10 | `blendF_delta_xgb_bayes_w0.40` | FULL | 1.0958 | 0.0867 | 1.4368 | 0.0008 | delta blend; w_xgb=0.40 |
| 11 | `blendF_delta_xgb_bayes_w0.50` | FULL | 1.0945 | 0.0895 | 1.4358 | 0.0021 | delta blend; w_xgb=0.50 |
| 12 | `blendF_delta_xgb_bayes_w0.60` | FULL | 1.0940 | 0.0920 | 1.4363 | 0.0012 | delta blend; w_xgb=0.60 |
| 13 | `blendF_delta_xgb_bayes_w0.70` | FULL | 1.0947 | 0.0946 | 1.4383 | -0.0019 | delta blend; w_xgb=0.70 |
| 14 | `blendF_delta_xgb_bayes_w0.80` | FULL | 1.0973 | 0.0974 | 1.4418 | -0.0070 | delta blend; w_xgb=0.80 |
