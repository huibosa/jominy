# Journal - huibosa (Part 1)

> AI development session journal
> Started: 2026-05-28

---



## Session 1: Port webapp to Windows desktop app (Tauri + PyInstaller)

**Date**: 2026-05-29
**Task**: Port webapp to Windows desktop app (Tauri + PyInstaller)
**Branch**: `main`

### Summary

Wrapped the FastAPI + Vite/TypeScript Jominy hardenability predictor as a self-contained Windows installer. Backend: added sidecar entrypoint (argparse + uvicorn.run), sys._MEIPASS-aware model path resolver, RotatingFileHandler when frozen, PyInstaller main.spec with sklearn/xgboost hidden imports. Frontend: api.ts BASE URL from window.__JOMINY_API__, loading overlay waiting for backend-ready CustomEvent. Tauri scaffold (src-tauri/): free-port pick, initialization_script port injection, dev/release sidecar spawn, Windows Job Object, native menu bar (File/Quit + Help/Open Log Folder), stdout/stderr capture to tauri.log, error dialogs. CI: .github/workflows/release.yml on windows-latest runner, tauri-apps/tauri-action, minisign updater keypair.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `71848a4` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 2: Empirical formula benchmark

**Date**: 2026-06-03
**Task**: Empirical formula benchmark
**Branch**: `main`

### Summary

Ran Sheet 1 empirical formula benchmark against ridge and retrained production blend; generated metrics/report; tests and basedpyright pass.

### Main Changes

Executed `.claude/plans/empirical-formula-benchmark.md` inline in the current checkout (user declined worktree). Added `scripts/run_empirical_formula_benchmark.py`, generated empirical formula benchmark CSVs and report, verified blend retrain against stored J9 OOF predictions, and ran quality checks.

Validation:
- `uv run --with pandas,pyarrow,scikit-learn,xgboost scripts/run_empirical_formula_benchmark.py` passed; max per-fold blend J9 MAE delta vs `blend_oof.csv` = 0.000000 HRC.
- `uv run --with basedpyright basedpyright scripts/run_empirical_formula_benchmark.py` passed with 0 errors.
- `uv run --group backend-build pytest` passed: 25 tests.

Outputs:
- `scripts/run_empirical_formula_benchmark.py`
- `output/modeling/metrics/empirical_formula_per_fold.csv`
- `output/modeling/metrics/empirical_formula_monotonicity.csv`
- `output/modeling/reports/empirical_formula_comparison.md`


### Git Commits

(No commits - planning session)

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 3: README empirical benchmark results

**Date**: 2026-06-03
**Task**: README empirical benchmark results
**Branch**: `main`

### Summary

Documented empirical formula benchmark results in README and verified tests pass.

### Main Changes

Updated `README.md` with a new "Empirical formula benchmark" section summarizing the generated Sheet 1 formula comparison against the current blend and ridge baselines. Added headline J9/J15 MAE table, conclusion, output paths, pipeline entry, and reproduce command. Also corrected the pipeline note to describe `webapp/backend/train_models.py` as the desktop webapp blend export path and `select_final_model.py` as the older fixed-v1 benchmark/export path.

Validation:
- Reviewed README diff.
- `uv run --group backend-build pytest` passed: 25 tests.


### Git Commits

(No commits - planning session)

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete
