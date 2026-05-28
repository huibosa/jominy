"""FastAPI entrypoint for the Jominy hardenability predictor."""
from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from predictor import Predictor


def _configure_sidecar_logging() -> None:
    """Set up file logging when running as a PyInstaller-bundled sidecar."""
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

WEBAPP_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIST = WEBAPP_ROOT / "frontend" / "dist"

ELEMENT_FIELDS = ["C", "Si", "Mn", "P", "S", "Cu", "Ni", "Cr", "V", "Ti", "W", "Al", "B"]


class CompositionRequest(BaseModel):
    C: Annotated[float, Field(ge=0, le=2.0, description="Carbon, wt%")]
    Si: Annotated[float, Field(ge=0, le=3.0, description="Silicon, wt%")]
    Mn: Annotated[float, Field(ge=0, le=3.0, description="Manganese, wt%")]
    P: Annotated[float, Field(ge=0, le=0.1, description="Phosphorus, wt%")]
    S: Annotated[float, Field(ge=0, le=0.1, description="Sulfur, wt%")]
    Cu: Annotated[float, Field(ge=0, le=1.0, description="Copper, wt%")]
    Ni: Annotated[float, Field(ge=0, le=5.0, description="Nickel, wt%")]
    Cr: Annotated[float, Field(ge=0, le=5.0, description="Chromium, wt%")]
    V: Annotated[float | None, Field(default=None, ge=0, le=1.0, description="Vanadium, wt% (optional)")] = None
    Ti: Annotated[float | None, Field(default=None, ge=0, le=1.0, description="Titanium, wt% (optional)")] = None
    W: Annotated[float | None, Field(default=None, ge=0, le=1.0, description="Tungsten, wt% (optional)")] = None
    Al: Annotated[float | None, Field(default=None, ge=0, le=1.0, description="Aluminum, wt% (optional)")] = None
    B: Annotated[float | None, Field(default=None, ge=0, le=0.05, description="Boron, wt% (optional)")] = None


class PredictionResponse(BaseModel):
    J9: float = Field(description="Predicted Rockwell hardness at 9 mm (HRC)")
    J15: float = Field(description="Predicted Rockwell hardness at 15 mm (HRC)")
    delta: float = Field(description="Predicted J9 - J15 spread (HRC)")
    components: dict[str, float] = Field(description="Underlying base-learner predictions")
    warnings: list[str] = Field(description="Out-of-range input warnings")
    expected_mae: dict[str, float] = Field(description="Expected mean absolute error from cross-validation")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = Predictor()
    yield


app = FastAPI(
    title="Jominy Hardenability Predictor",
    description="Predict Rockwell J9 and J15 from steel chemical composition.",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow Vite dev server during development and Tauri WebView2 origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "tauri://localhost",
        "https://tauri.localhost",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/metadata")
def metadata() -> dict:
    pred: Predictor = app.state.predictor
    return {
        "features": pred.features,
        "feature_stats": pred.feature_stats,
        "expected_metrics": pred.metadata["expected_oof_metrics"],
        "j9_train_rows": pred.metadata["j9_train_rows"],
        "delta_train_rows": pred.metadata["delta_train_rows"],
        "element_fields": ELEMENT_FIELDS,
    }


@app.post("/api/predict", response_model=PredictionResponse)
def predict(req: CompositionRequest) -> PredictionResponse:
    pred: Predictor = app.state.predictor
    try:
        result = pred.predict(req.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"prediction failed: {exc}") from exc
    return PredictionResponse(**result)


# Serve the built frontend at /. If the frontend hasn't been built yet,
# the routes return a friendly message instead of 500.
if FRONTEND_DIST.exists() and (FRONTEND_DIST / "index.html").exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/")
    def root() -> FileResponse:
        return FileResponse(FRONTEND_DIST / "index.html")

    @app.get("/{path:path}")
    def spa_fallback(path: str) -> FileResponse:
        candidate = FRONTEND_DIST / path
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(FRONTEND_DIST / "index.html")
else:

    @app.get("/")
    def placeholder() -> dict:
        return {
            "status": "frontend not built",
            "next": "cd webapp/frontend && bun install && bun run build",
            "api_docs": "/docs",
        }


if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Jominy hardenability predictor backend")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port, log_config=None)
