# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for the Jominy FastAPI sidecar (Windows --onedir build).

Build command (run from project root):
    pyinstaller webapp/backend/main.spec
Output: dist/main/main.exe + dist/main/_internal/
"""
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files

# ---------------------------------------------------------------------------
# Paths (resolved at spec-parse time, i.e. on the build machine)
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(SPECPATH)          # webapp/backend/
WEBAPP_ROOT = BACKEND_DIR.parent      # webapp/
MODELS_DIR  = WEBAPP_ROOT / "models"  # webapp/models/

# ---------------------------------------------------------------------------
# Hidden imports — modules that PyInstaller's static tracer misses
# ---------------------------------------------------------------------------
HIDDEN_IMPORTS = [
    # Classes referenced only by the persisted joblib model artifacts.
    # PyInstaller cannot discover these from static imports in predictor.py.
    "sklearn.pipeline",
    "sklearn.compose._column_transformer",
    "sklearn.impute._base",
    "sklearn.preprocessing._data",
    "sklearn.linear_model._bayes",
    "xgboost.core",
    "xgboost.sklearn",
    # sklearn submodules loaded lazily or via importlib
    "sklearn.utils._cython_blas",
    "sklearn.neighbors._partition_nodes",
    "sklearn.tree._utils",
    "sklearn.cross_decomposition._pls",
    # pandas C-extension datetime internals
    "pandas._libs.tslibs.np_datetime",
    "pandas._libs.tslibs.nattype",
    "pandas._libs.tslibs.timestamps",
    # scipy sparse (sklearn dependency)
    "scipy.sparse.csgraph._tools",
    "scipy.special._ufuncs_cxx",
    # uvicorn workers/protocols
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan.on",
]

# ---------------------------------------------------------------------------
# Data files — non-.py resources that must ship alongside the bundle
# ---------------------------------------------------------------------------

# xgboost ships a native DLL (xgboost.dll / libxgboost.dylib) and a VERSION
# file that it looks up at import time.  collect_data_files handles both.
xgboost_datas = collect_data_files("xgboost")

# Our serialised model artefacts
model_datas = [
    (str(MODELS_DIR / "j9_blend.joblib"),    "models"),
    (str(MODELS_DIR / "delta_blend.joblib"), "models"),
    (str(MODELS_DIR / "metadata.json"),      "models"),
    (str(MODELS_DIR / "feature_stats.json"), "models"),
]

ALL_DATAS = xgboost_datas + model_datas

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
a = Analysis(
    [str(BACKEND_DIR / "main.py")],
    pathex=[str(BACKEND_DIR)],
    binaries=[],
    datas=ALL_DATAS,
    hiddenimports=HIDDEN_IMPORTS,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Save ~30 MB: these are never used at inference time
        "tkinter",
        "matplotlib",
        "IPython",
        "jupyter",
        "notebook",
        "openpyxl",
        "pyarrow",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="main",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,      # desktop sidecar: no extra console window next to the Tauri UI
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="main",
)
