# Jominy Hardenability Webapp

FastAPI backend + TypeScript (Vite) frontend that exposes the production blend model from [HISTORY.md](../HISTORY.md) as an interactive predictor.

```
webapp/
├── backend/
│   ├── train_models.py    # Refit blend on all data, dump joblib
│   ├── predictor.py       # Load model, run prediction with input validation
│   └── main.py            # FastAPI app — /api/predict, /api/metadata, /api/health, static SPA
├── frontend/
│   ├── package.json
│   ├── vite.config.ts
│   ├── index.html
│   ├── src/
│   │   ├── main.ts        # Form + result rendering
│   │   ├── style.css
│   │   └── types.ts
│   └── dist/              # Built static bundle, served by FastAPI in production
└── models/                # Persisted artifacts (j9_blend.joblib, delta_blend.joblib, *.json)
```

## Setup

```bash
# 1. Train and persist the production models (writes webapp/models/*.joblib)
uv run --with pandas,pyarrow,scikit-learn,xgboost,joblib \
    python webapp/backend/train_models.py

# 2. Build the frontend
cd webapp/frontend && bun install && bun run build && cd -

# 3. Start the API (serves the built frontend at /)
cd webapp/backend
uv run --with fastapi,uvicorn,pandas,pyarrow,scikit-learn,xgboost,joblib \
    uvicorn main:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000/>. The API docs are at <http://127.0.0.1:8000/docs>.

## Desktop build (Windows installer)

### Local dev loop

On Linux, run the Tauri shell against the Vite dev server:

```bash
# Terminal 1 — FastAPI backend (plain Python, no bundled binary needed)
cd webapp/backend
uv run --with fastapi,uvicorn,pandas,pyarrow,scikit-learn,xgboost,joblib \
    python main.py --port 8000

# Terminal 2 — Vite frontend dev server (proxies /api to :8000)
cd webapp/frontend && bun run dev

# Terminal 3 — Tauri shell (WebKitGTK window on Linux)
cargo tauri dev
```

The Tauri window loads `http://localhost:5173`. `window.__JOMINY_API__` is injected to
`http://127.0.0.1:8000` (dev default). Predictions go through the Vite proxy.

### Releasing a Windows installer

1. **Add the signing key secret** (one-time setup):
   ```
   GitHub repo → Settings → Secrets → Actions → New repository secret
   Name:  TAURI_SIGNING_PRIVATE_KEY
   Value: contents of ~/.tauri/jominy.key
   ```

2. **Tag and push**:
   ```bash
   git tag v1.0.0
   git push origin v1.0.0
   ```

3. The `Release` GitHub Actions workflow runs on `windows-latest`, builds the PyInstaller
   sidecar, compiles Tauri, and attaches `Jominy_Setup_1.0.0.exe` and `latest.json` to
   the GitHub Release.

4. **Install**: download `Jominy_Setup_*.exe` from the release, run it (click through
   SmartScreen on first run), and launch from the Start Menu.

### Updating the app

On every launch the app checks
`https://github.com/huibosa/jominy/releases/latest/download/latest.json`.
If a newer version is found and the signature verifies, the user is prompted to update.
Accepting downloads the new installer and relaunches into it.

## Development

Run the Vite dev server alongside FastAPI (Vite proxies `/api/*` to port 8000):

```bash
# terminal 1
cd webapp/backend && uv run --with fastapi,uvicorn,... uvicorn main:app --reload

# terminal 2
cd webapp/frontend && bun run dev
```

Open <http://localhost:5173>.

## API

### `POST /api/predict`

```json
{
  "C": 0.20, "Si": 0.26, "Mn": 0.96,
  "P": 0.012, "S": 0.018, "Cu": 0.05,
  "Ni": 0.10, "Cr": 1.10,
  "V": null, "Ti": 0.005, "W": null, "Al": 0.025, "B": null
}
```

Returns:

```json
{
  "J9": 35.97,
  "J15": 29.31,
  "delta": 6.66,
  "components": {
    "j9_xgb": 35.76, "j9_pls": 36.45,
    "delta_xgb": 6.76, "delta_bayes": 6.5
  },
  "warnings": [],
  "expected_mae": {"J9": 1.7106, "delta": 1.094}
}
```

`J15` is reconstructed as `J9 - max(0, delta)` so the monotonic constraint J9 ≥ J15 always holds. Inputs whose values fall outside the training-data 1st–99th percentile are flagged in `warnings` — the prediction is still returned but should be treated as extrapolation.

### `GET /api/metadata`

Returns the feature list, feature stats (min / max / median / p01 / p99 per element), expected cross-validation metrics, and training row counts. The frontend uses this to label inputs with their typical ranges.

### `GET /api/health`

Returns `{"status": "ok"}`. Useful for liveness probes.

## Model

| Target | Predictor |
|--------|-----------|
| J9 | `0.70 · XGBoost(n=1500, lr=0.005, md=3, ss=0.55, cs=0.5, l2=2.0) + 0.30 · PLS(n=3)` |
| δ  | `0.60 · XGBoost(n=800,  lr=0.01,  md=3, ss=0.8,  cs=0.8, l2=2.0) + 0.40 · BayesianRidge` |

Cross-validated MAE (5-fold GroupKFold): **J9 = 1.71 HRC**, δ = 1.09 HRC. See HISTORY.md for the full leaderboard and selection rationale.

## Known limits

- The training data covers a narrow chemistry window (C ≈ 0.17–0.22, Cr ≈ 0.4–1.6, Mn ≈ 0.66–1.30). Predictions outside this band are extrapolation. The warning list calls this out per element.
- For specimens with chemistry near the training centroid the model is well-calibrated, but the tails (J9 < 32 or J9 ≥ 41) regress to the mean by ±3 HRC because many tail specimens have indistinguishable composition from the bulk. This is a feature limitation, not fixable in the model.
- δ has near-zero R² and should be treated as a coarse offset, not a precise quantity.
