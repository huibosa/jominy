# Logging Guidelines

> How logging is done in this project.

---

## Overview

Standard library `logging` module. In **dev mode** (uvicorn import), uvicorn owns the log config — the backend does not touch `logging.root`. In **sidecar mode** (PyInstaller frozen binary), a `RotatingFileHandler` is configured at module load time and writes to `%LOCALAPPDATA%\Jominy\logs\backend.log`.

---

## Sidecar log setup

The `_configure_sidecar_logging()` function in `main.py` must:

- Be a no-op when `getattr(sys, "frozen", False)` is `False` (dev mode)
- Write to `%LOCALAPPDATA%/Jominy/logs/backend.log` with 10 MB rotation, 3 backups
- Use `encoding="utf-8"` on the handler (Windows default codepage may not be UTF-8)
- Call `logging.root.setLevel(logging.INFO)` (not per-logger) so all libraries' log output is captured

```python
from logging.handlers import RotatingFileHandler
import logging, os, sys
from pathlib import Path

def _configure_sidecar_logging() -> None:
    if not getattr(sys, "frozen", False):
        return
    log_dir = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "Jominy" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        log_dir / "backend.log",
        maxBytes=10_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.root.addHandler(handler)
    logging.root.setLevel(logging.INFO)

_configure_sidecar_logging()
```

Call at module level (before the `FastAPI()` constructor) so startup errors are captured.

---

## Log Levels

| Level | When |
|-------|------|
| `INFO` | Startup/shutdown, each prediction request (input hash + J9/J15 result) |
| `WARNING` | Out-of-range inputs (already surfaced to UI via `warnings` field) |
| `ERROR` | Unhandled exceptions in route handlers |

Do not log at `DEBUG` in production sidecar builds — it generates noise from sklearn/xgboost internals.

---

## What NOT to Log

- Full prediction input vectors (may contain commercially sensitive chemistry data)
- Any value from `os.environ` wholesale — log specific known-safe keys only
- Personal identifiers

---

## Log location (user-facing)

Logs are surfaced via `Help → Open log folder` in the desktop app, which opens `%LOCALAPPDATA%\Jominy\logs\` in Explorer. Both `backend.log` and `tauri.log` live there. Users share this folder when reporting bugs.
