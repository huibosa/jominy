# Quality Guidelines

> Code quality standards for backend development.

---

## Overview

Python 3.12, type-checked with Pyright (`basic` mode). Python build dependencies are managed by uv in the repo-local `.venv`; use the `backend-build` dependency group (`uv sync --group backend-build`, then `uv run --group backend-build ...`). The backend runs in two modes: imported by uvicorn (`uvicorn main:app`) and launched as a PyInstaller-bundled sidecar (`python main.py --port PORT`). Both paths must be preserved on every change.

---

## Forbidden Patterns

### Don't: Direct access to PyInstaller runtime attributes

**Problem**:
```python
# Don't do this — triggers Pyright reportAttributeAccessIssue
from pathlib import Path
import sys
return Path(sys._MEIPASS) / "models"   # ✗
```

**Why it's bad**: `sys._MEIPASS` and `sys.frozen` are injected by PyInstaller at runtime. They are not declared in the `sys` type stubs, so direct attribute access triggers `reportAttributeAccessIssue` under Pyright `basic` mode.

**Instead**:
```python
import sys

# Frozen detection — always use getattr with a default
if getattr(sys, "frozen", False):
    # Attribute access — always use getattr
    return Path(getattr(sys, "_MEIPASS")) / "models"
```

---

## Required Patterns

### Dual-mode entry point (uvicorn import + sidecar script)

FastAPI modules that need to run both as `uvicorn main:app` **and** as a directly launched script (`python main.py --port PORT`) must gate all CLI code behind `if __name__ == "__main__":`.

```python
# main.py

app = FastAPI(...)

# ... route definitions ...

if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port, log_config=None)
```

**Why**: The `if __name__` guard ensures the `argparse` / `uvicorn.run` block never executes during `import main` (the uvicorn path). Without it, the sidecar's argparse will raise `SystemExit` on import because `--port` is required but not supplied.

### PyInstaller path resolver (frozen vs dev)

Any module that loads files bundled inside a PyInstaller artifact (models, config, assets) must use a frozen-aware resolver:

```python
import sys as _sys
from pathlib import Path

def _models_dir() -> Path:
    if getattr(_sys, "frozen", False):
        # PyInstaller --onedir: _MEIPASS is the _internal/ dir next to main.exe
        return Path(getattr(_sys, "_MEIPASS")) / "models"
    # Dev mode
    return Path(__file__).resolve().parents[1] / "models"

MODELS_DIR = _models_dir()
```

Use a `_`-prefixed `sys` alias (`import sys as _sys`) when the rest of the module doesn't otherwise use `sys`, to avoid polluting the namespace.

### PyInstaller hidden imports for persisted model artifacts

When a PyInstaller sidecar loads `joblib` / pickle model artifacts, include the modules that define serialized estimator classes in `webapp/backend/main.spec` `hiddenimports`. PyInstaller cannot discover classes that are only referenced inside persisted artifacts.

For the current production blends, the required hidden imports include the sklearn pipeline/transformer/imputer/preprocessing/linear-model modules and the XGBoost sklearn wrapper used by `webapp/models/*.joblib`.

---

## Testing Requirements

- Core prediction logic (`Predictor.predict`) must be tested with the reference payload from `webapp/README.md` (`C=0.20, Si=0.26, ...`) — expected `J9 ≈ 35.97, J15 ≈ 29.31`.
- Both launch modes (`uvicorn main:app` and `python main.py --port PORT`) must return `{"status":"ok"}` from `/api/health` after startup.
- `Predictor` instantiation must not raise (verifies model file paths resolve correctly in dev mode).
- After changing `webapp/backend/main.spec`, build through uv (`uv run --project ../.. --group backend-build pyinstaller --clean -y main.spec` from `webapp/backend`), then smoke-test the frozen sidecar executable (`webapp/backend/dist/main/main.exe --host=127.0.0.1 --port <free-port>`) and verify `/api/health` returns `{"status":"ok"}`; a successful PyInstaller build alone is not enough.
- For Tauri WebView API calls, CORS must allow the actual desktop origin (`http://tauri.localhost` on Windows/Tauri 2, plus the existing dev and legacy origins). Verify with an `Origin: http://tauri.localhost` request when debugging `Failed to fetch`.
- The release sidecar must be built with `console=False` in `webapp/backend/main.spec` so Windows does not show a separate console window next to the Tauri UI. Rely on `%LOCALAPPDATA%/Jominy/logs/backend.log` for diagnostics instead.

---

## Code Review Checklist

- [ ] Any new `sys.*` attribute access uses `getattr(sys, "attr_name")` not `sys.attr_name` if the attribute is not in the stdlib stubs
- [ ] CLI-only code is inside `if __name__ == "__main__":`
- [ ] File paths that differ between dev and frozen modes go through a frozen-aware resolver function
- [ ] Joblib/pickle-loaded model classes are covered by PyInstaller `hiddenimports`
- [ ] `uv run --group backend-build ...` launch still works after the change (no import-time side effects)
