# Port webapp to Windows desktop app

## Goal

Repackage the existing FastAPI + Vite/TypeScript Jominy hardenability predictor (`webapp/`) as a self-contained Windows desktop app that an end user (a metallurgist with no Python toolchain) can install and run by double-clicking an installer. Behavioral parity with the current web form for v1.0 — no new product features, only the platform port.

## What I already know

### Current webapp structure (verified)

- `webapp/backend/main.py` — FastAPI app, exposes `/api/predict`, `/api/metadata`, `/api/health`. Currently launched via `uvicorn main:app`. CORS allowlist for `localhost:5173` only.
- `webapp/backend/predictor.py` — loads `j9_blend.joblib`, `delta_blend.joblib`, `metadata.json`, `feature_stats.json` from `webapp/models/`. `MODELS_DIR = Path(__file__).resolve().parents[1] / "models"`.
- `webapp/backend/train_models.py` — refits and persists models. Not part of the runtime package.
- `webapp/frontend/` — Vite + bun + TypeScript. `package.json` confirms bun toolchain. `index.html`, `src/main.ts`, `src/style.css`, `src/types.ts`. Built bundle in `frontend/dist/`.
- `webapp/models/` — 4 artifacts: 2 joblibs (~2.6 MB total), 2 small JSON files.

### Tech stack constraints (from CLAUDE.md)

- Python 3.12 with `uv` package manager
- pandas, openpyxl, pyarrow, scikit-learn, xgboost, joblib
- bun (not npm) for JS — already in use for frontend
- Project lives at `/home/huibosa/workspaces/jominy` on Linux

### Decisions converged via /grill-me (2026-05-28)

All confirmed by user. These are **requirements, not open questions**:

| Decision | Value |
|---|---|
| **Shell** | Tauri 2.x (Rust + WebView2), not Electron |
| **Backend bundling** | PyInstaller `--onedir` (not onefile, not Nuitka, not embeddable Python) |
| **IPC** | Frontend `fetch()` to FastAPI sidecar on `127.0.0.1:<random_port>` (not `invoke()`, not stdin/stdout RPC) |
| **Port discovery** | Tauri picks a free port at startup, injects `window.__JOMINY_API__` before SPA loads |
| **Sidecar argv** | Tauri spawns sidecar with `--host=127.0.0.1 --port=PORT` (not env vars, not stdout-announced) |
| **Startup UX** | Show window immediately with form disabled + "Loading model..." overlay; frontend polls `/api/health` and enables form on 200 OK |
| **Sidecar lifecycle** | Tauri `WindowEvent::Destroyed` calls `child.kill()` + Windows Job Object so OS reaps sidecar on Tauri crash |
| **Model bundling** | Bundle `webapp/models/*` inside PyInstaller spec `datas`; resolve via `sys._MEIPASS` at runtime |
| **CORS allowlist** | Add `tauri://localhost`, `https://tauri.localhost` to existing `localhost:5173` entries |
| **Logging** | Backend `RotatingFileHandler` at `%LOCALAPPDATA%/Jominy/logs/backend.log` (10 MB × 3); Tauri pipes sidecar stdout/stderr to `tauri.log`; Help menu has "Open log folder" |
| **Windows target** | Win10 1903+ and Win11 (in-box WebView2) |
| **Installer** | NSIS .exe via `cargo tauri build`, **unsigned** for v1.0 (SmartScreen warning expected) |
| **Auto-update** | Tauri built-in updater pulling `latest.json` from GitHub Releases, public-key signed |
| **Build host** | GitHub Actions `windows-latest` runner on git tag push |
| **Dev loop** | `cargo tauri dev` against WebKitGTK on Linux for UI/Rust iteration; tag → CI → install in Win11 VM for Windows-specific validation |
| **PyInstaller spec** | Hand-written `webapp/backend/main.spec` with explicit `hiddenimports` (sklearn submodules, pandas tslibs) and `datas` (xgboost DLL + VERSION + models dir) — not `--collect-all` |
| **v1.0 scope** | Parity-only (single-composition predict form); no batch CSV / history / About panel |

## Assumptions (temporary)

- The end user is technical enough to bypass SmartScreen's "More info → Run anyway" on first launch (acceptable for internal lab use).
- WebView2 runtime is in-box on Win10 1903+ and we don't need a fallback installer for older Win10 builds.
- PyInstaller's `hiddenimports` for sklearn 1.x + xgboost 2.x on Windows are stable enough that a hand-written spec won't bitrot before the next sklearn major version.
- The bundled installer's ~190 MB total size is acceptable.
- A Windows 11 VM (or physical machine) is available to the developer for installer smoke testing — Linux-only validation does not exist.

## Open Questions

All resolved — see decisions below.

## Resolved Decisions (from user, 2026-05-29)

| # | Question | Answer |
|---|---|---|
| 1 | App identity | `productName: "jominy"`, `identifier: "com.huibosa.jominy"`, starting version `1.0.0`; repo is `https://github.com/huibosa/jominy` (public) |
| 2 | Updater hosting & keypair | GitHub Releases on `huibosa/jominy` (public repo, free); generate a fresh Tauri updater keypair as part of this task (PR 5) |
| 3 | Window chrome | Add a menu bar: `File → Quit`, `Help → Open log folder` |
| 4 | Window size | Resizable with a sensible minimum (target: `minWidth: 720, minHeight: 640`; default: `width: 900, height: 780`) |
| 5 | Startup failure UX | Modal error dialog showing the failure message + path to log file; app exits after user dismisses |
| 6 | App icon | Tauri-generated placeholder icons for v1.0 (real `.ico` in a future pass) |

## Requirements (evolving)

### Runtime behavior
- Single double-click of `Jominy_Setup_X.Y.Z.exe` installs the app, places a Start Menu entry, and registers an uninstaller.
- Launching the app shows a Tauri-native window within ~500 ms; the form renders disabled with a "Loading model..." overlay until the FastAPI sidecar's `/api/health` returns 200 (target: ≤3 s on a typical lab PC).
- Closing the window terminates the Python sidecar within ≤2 s (verified via Task Manager — no orphan `main.exe`).
- The app behaves identically to `webapp/` running locally: same form, same prediction output, same warnings, same metadata.

### Build & distribution
- A git tag matching `v*` triggers GitHub Actions on `windows-latest` to produce `Jominy_Setup_X.Y.Z.exe` and a signed `latest.json`, both attached to the GitHub Release.
- Subsequent launches detect a newer version and prompt the user to update; declining keeps the current version working.

### Code changes (concrete files)
- `webapp/backend/main.py` — add `if __name__ == "__main__":` block with argparse + `uvicorn.run(app, host=..., port=...)`. Add Tauri origins to CORS allowlist.
- `webapp/backend/predictor.py` — switch `MODELS_DIR` to a `sys._MEIPASS`-aware resolver so frozen and dev modes both work.
- `webapp/backend/main.spec` (new) — PyInstaller spec with sklearn/xgboost hidden imports and models datas.
- `webapp/frontend/src/api.ts` (new) — single module exporting a `BASE` URL read from `window.__JOMINY_API__` with a `http://localhost:8000` dev fallback. Refactor `main.ts` to import from it.
- `src-tauri/` (new) — Tauri scaffold: `tauri.conf.json`, `Cargo.toml`, `src/main.rs`, `icons/`, `build.rs`. `main.rs` picks a free port, spawns the sidecar with `--port=...`, polls `/api/health`, injects `window.__JOMINY_API__`, kills the sidecar on window close, wraps it in a Windows Job Object.
- `.github/workflows/release.yml` (new) — Windows runner; installs Rust + Python 3.12 + bun; runs `bun install && bun run build`, `pyinstaller main.spec`, `cargo tauri build --target x86_64-pc-windows-msvc`; uploads installer + signed manifest to the release.
- `webapp/README.md` — add a "Desktop build" section documenting the local dev loop and CI release flow.

## Acceptance Criteria (evolving)

- [ ] On a clean Windows 11 VM with no Python and no Rust installed, double-clicking `Jominy_Setup_*.exe` installs the app and a Start Menu shortcut.
- [ ] Launching from the Start Menu shows the window in ≤500 ms with the loading overlay; the form is interactive within ≤3 s.
- [ ] Submitting the example payload from `webapp/README.md` (`C=0.20, Si=0.26, Mn=0.96, ...`) returns `J9 ≈ 35.97, J15 ≈ 29.31` (parity with current FastAPI output).
- [ ] Closing the window kills the sidecar within 2 s (verified in Task Manager).
- [ ] Force-killing `Jominy.exe` from Task Manager also kills `main.exe` within 2 s (Job Object behavior).
- [ ] Out-of-range inputs surface the same warning text as the web app.
- [ ] `Help → Open log folder` opens `%LOCALAPPDATA%\Jominy\logs\` in Explorer; `backend.log` and `tauri.log` exist after one prediction.
- [ ] Bumping the version, tagging, pushing → CI produces `Jominy_Setup_X.Y.Z.exe` and `latest.json`. Installing the bumped version on a machine running the previous version triggers an "Update available" prompt; accepting it relaunches into the new version.
- [ ] On Linux (`cargo tauri dev`), the WebKitGTK window loads the SPA, the sidecar runs as plain `python main.py`, and a prediction completes end-to-end. (Used as the daily dev loop; not a shipping artifact.)

## Definition of Done

- [ ] All acceptance criteria pass
- [ ] PyInstaller `hiddenimports` are pinned in `main.spec`; pinned versions are recorded in a `requirements-pyinstaller.txt` next to it (or equivalent uv export)
- [ ] CI release workflow runs green end-to-end on a tagged commit
- [ ] `webapp/README.md` documents both the dev loop and the release process
- [ ] No regressions in the existing web mode: `uv run uvicorn main:app` still works against the same `predictor.py`

## Out of Scope (explicit)

- Code-signing certificate / signed installer (deferred until external distribution starts)
- Batch CSV upload, prediction history, "About" panel (deferred to v1.1)
- macOS or Linux installer artifacts (Linux remains a dev environment only)
- Hot-swap model updates separate from app updates (the Tauri updater replaces the whole bundle)
- Sentry / GlitchTip remote crash reporting (file logging is sufficient for the lab tool use case)
- Windows 7/8/10-pre-1903 support
- Persisting user preferences, recent inputs, or any state beyond logs

## Technical Approach

### Architecture

```
                  ┌─────────────────────────────────────────┐
                  │                Jominy.exe                │
                  │             (Tauri shell, Rust)          │
                  │                                          │
                  │  ┌──────────────┐    ┌────────────────┐ │
                  │  │  WebView2    │    │  sidecar mgr   │ │
                  │  │  (Edge)      │◀──▶│  (free-port,   │ │
                  │  │              │    │   spawn, kill) │ │
                  │  │  index.html  │    └────────┬───────┘ │
                  │  │  + JS bundle │             │         │
                  │  └──────┬───────┘             │         │
                  └─────────┼─────────────────────┼─────────┘
                            │                     │
                            │ HTTP fetch          │ child process
                            │ 127.0.0.1:PORT      │ + Job Object
                            ▼                     ▼
                  ┌─────────────────────────────────────────┐
                  │              main.exe                    │
                  │      (PyInstaller --onedir bundle)       │
                  │                                          │
                  │  python312.dll + uvicorn + FastAPI       │
                  │  + sklearn + xgboost + pandas            │
                  │  + j9_blend.joblib                       │
                  │  + delta_blend.joblib                    │
                  │  + feature_stats.json                    │
                  │  + metadata.json                         │
                  └─────────────────────────────────────────┘
```

### Key sequences

**Cold start**:
1. User double-clicks `Jominy.exe`
2. Tauri reads `tauri.conf.json`, opens window, loads `index.html` from the Tauri asset protocol
3. Frontend's first script tag is the injected `window.__JOMINY_API__ = 'http://127.0.0.1:PORT'`
4. In parallel, Rust shell picks a free port (bind 0, read sockname, close), spawns `main.exe --host=127.0.0.1 --port=PORT` as a sidecar, attaches it to a Windows Job Object with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`
5. Frontend mounts with form disabled + "Loading model..." overlay; polls `GET /api/health` every 200 ms with a 10 s timeout
6. On first 200, frontend removes overlay and enables the form

**Predict click**: identical to current webapp — `POST /api/predict` returns `{J9, J15, delta, components, warnings, expected_mae}`.

**Shutdown**:
- Normal close → `WindowEvent::Destroyed` → `child.kill()` → Job Object cleans up subprocesses
- Tauri crash → Job Object reaps sidecar (kernel-level guarantee)
- Force-quit via Task Manager → same Job Object guarantee

### Decision (ADR-lite)

- **Context**: The current webapp targets Linux/macOS development and lab-server deployment, but the production users are metallurgists on Windows desks who can't run `uv` commands. They need a single installer.
- **Decision**: Wrap the existing FastAPI + Vite stack in a Tauri shell with a PyInstaller-bundled Python sidecar, communicating over a localhost HTTP port. Maximum reuse of existing code; Windows-only build via GitHub Actions.
- **Consequences**:
  - Pros: 100 % backend reuse, ~95 % frontend reuse (just an `api.ts` indirection), familiar bun toolchain, ~190 MB installer, future macOS/Linux ports trivial.
  - Cons: PyInstaller spec maintenance is manual (sklearn/xgboost edge cases); SmartScreen warning until cert acquired; cold start ~1–3 s vs. instantaneous for a pure-JS app.
  - Risks: PyInstaller hidden-import bitrot at sklearn version bumps (mitigated by pinned versions + CI smoke test); Windows AV false-positives on unsigned binaries (mitigated by future signing).

## Technical Notes

### Files inspected during brainstorm
- `webapp/README.md` — confirms current build steps, model file list, API shape
- `webapp/backend/main.py` — confirms FastAPI structure, CORS allowlist, lifespan-loaded predictor
- `webapp/backend/predictor.py` — confirms `MODELS_DIR` resolution; needs `sys._MEIPASS` branch
- `webapp/frontend/` (directory listing) — confirms bun + Vite + TS setup
- `webapp/models/` (directory listing) — confirms 4 artifacts to bundle

### References
- Tauri 2.x sidecar docs: `bundle.externalBin`, `Command::new_sidecar`
- PyInstaller sklearn/xgboost recipes: well-known community patterns; sklearn submodules require explicit `hiddenimports`, xgboost requires its `lib/xgboost.dll` and `VERSION` file in `datas`
- Windows Job Objects: `CreateJobObjectW` + `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` — standard pattern for sidecar reaping
- Tauri updater: `tauri-plugin-updater`, `tauri signer generate` for keypair, `latest.json` schema

### Constraints from project
- Python 3.12, uv-managed
- bun (not npm) for frontend
- No git repo at project root (per env), but `.trellis/` and CI workflow file paths must work in the project tree regardless

## Implementation Plan (small PRs)

- **PR 1 — Backend refactor for sidecar mode**
  Add `__main__` argparse + uvicorn block to `main.py`; add Tauri origins to CORS; switch `predictor.py` to `sys._MEIPASS`-aware paths. No Tauri yet; verify the existing `uv run uvicorn` flow still works.

- **PR 2 — PyInstaller spec + frontend port indirection**
  Add `webapp/backend/main.spec` with `hiddenimports`/`datas`; lock dependency versions; add `webapp/frontend/src/api.ts` and refactor `main.ts` to import `BASE` from it. Verify `pyinstaller main.spec` produces a working `dist/main/main.exe` on Windows that responds on `--port=8000`.

- **PR 3 — Tauri scaffold + dev loop**
  `src-tauri/` with config, Cargo deps (`tauri`, `tauri-plugin-shell`, sidecar feature), `main.rs` implementing free-port pick, sidecar spawn with Job Object, port injection, health-check loop. Verify `cargo tauri dev` works on Linux.

- **PR 4 — Logging, menu, error dialogs**
  Backend `RotatingFileHandler`; Tauri stdout/stderr capture to file; Help menu with "Open log folder"; sidecar-startup-failure modal dialog (resolves Open Question 5). Window chrome/size finalized (resolves OQ 3, 4).

- **PR 5 — Updater + GitHub Actions release workflow**
  Generate Tauri updater keypair (resolves OQ 2); wire `tauri-plugin-updater`; write `.github/workflows/release.yml`; produce a v0.1.0 release as the first end-to-end test. Document the release process in `webapp/README.md`.

## Research References

(None yet — this brainstorm has been driven by the prior `/grill-me` session and direct codebase inspection. If we hit unknowns during PR 2 or PR 3, we'll spawn `trellis-research` sub-agents for specific topics, e.g., "Tauri 2 sidecar Windows Job Object pattern" or "sklearn 1.5 + xgboost 2 minimal hiddenimports list".)
