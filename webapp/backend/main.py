"""FastAPI entrypoint for the Jominy hardenability predictor."""
from __future__ import annotations

import asyncio
import json as _json
import logging
import os
import sys
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError

from batch import (
    ParseError,
    UnsupportedFormatError,
    deduplicate_rows,
    detect_format,
    normalize_columns,
    parse_rows,
    read_dataframe,
)
from predictor import Predictor
from standards import grade_lookup


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
REQUIRED_KEYS = {"C", "Si", "Mn", "P", "S", "Cu", "Ni", "Cr"}

MAX_BATCH_BYTES = 20 * 1024 * 1024  # 20 MB hard cap

# ---------------------------------------------------------------------------
# Per-standard input bounds served to the frontend for input[min]/input[max].
# The backend CompositionRequest keeps wide limits and is not changed here;
# the standard selector is a frontend-only enforcement layer.
#
# GB/T 3077-2015 and GB/T 5216-2014 differ only in Mn and Cr upper bounds.
# V, W, Al, B are not specified in either standard — wide limits retained.
# ---------------------------------------------------------------------------
_BOUNDS_SHARED: dict[str, tuple[float, float]] = {
    "C":  (0.17, 0.23),
    "Si": (0.17, 0.37),
    "P":  (0.0,  0.035),
    "S":  (0.0,  0.035),
    "Cu": (0.0,  0.30),
    "Ni": (0.0,  0.30),
    "Ti": (0.04, 0.10),
    "V":  (0.0,  0.30),   # not in GB spec; wide limit
    "W":  (0.0,  0.20),   # not in GB spec; wide limit
    "Al": (0.0,  0.10),   # not in GB spec; wide limit
    "B":  (0.0,  0.005),  # not in GB spec; wide limit
}

STANDARD_BOUNDS: dict[str, dict[str, tuple[float, float]]] = {
    "gbt3077": {
        **_BOUNDS_SHARED,
        "Mn": (0.80, 1.10),  # GB/T 3077-2015
        "Cr": (1.00, 1.30),  # GB/T 3077-2015
    },
    "gbt5216": {
        **_BOUNDS_SHARED,
        "Mn": (0.80, 1.20),  # GB/T 5216-2014 (H-grade, wider)
        "Cr": (1.00, 1.45),  # GB/T 5216-2014 (H-grade, wider)
    },
}


# ---------------------------------------------------------------------------
# Pydantic models — single predict
# ---------------------------------------------------------------------------

class CompositionRequest(BaseModel):
    C: Annotated[float, Field(ge=0, le=0.35,  description="Carbon, wt%")]
    Si: Annotated[float, Field(ge=0, le=0.60,  description="Silicon, wt%")]
    Mn: Annotated[float, Field(ge=0, le=1.60,  description="Manganese, wt%")]
    P: Annotated[float, Field(ge=0, le=0.045, description="Phosphorus, wt%")]
    S: Annotated[float, Field(ge=0, le=0.045, description="Sulfur, wt%")]
    Cu: Annotated[float, Field(ge=0, le=0.50,  description="Copper, wt%")]
    Ni: Annotated[float, Field(ge=0, le=0.50,  description="Nickel, wt%")]
    Cr: Annotated[float, Field(ge=0, le=1.60,  description="Chromium, wt%")]
    V: Annotated[float | None, Field(default=None, ge=0, le=0.30,  description="Vanadium, wt% (optional)")] = None
    Ti: Annotated[float | None, Field(default=None, ge=0, le=0.20,  description="Titanium, wt% (optional)")] = None
    W: Annotated[float | None, Field(default=None, ge=0, le=0.20,  description="Tungsten, wt% (optional)")] = None
    Al: Annotated[float | None, Field(default=None, ge=0, le=0.10,  description="Aluminum, wt% (optional)")] = None
    B: Annotated[float | None, Field(default=None, ge=0, le=0.005, description="Boron, wt% (optional)")] = None


class PredictionResponse(BaseModel):
    J9: float = Field(description="Predicted Rockwell hardness at 9 mm (HRC)")
    J15: float = Field(description="Predicted Rockwell hardness at 15 mm (HRC)")
    delta: float = Field(description="Predicted J9 - J15 spread (HRC)")
    components: dict[str, float] = Field(description="Underlying base-learner predictions")
    warnings: list[str] = Field(description="Out-of-range input warnings")
    expected_mae: dict[str, float] = Field(description="Expected mean absolute error from cross-validation")


# ---------------------------------------------------------------------------
# Pydantic models — batch
# ---------------------------------------------------------------------------

BatchStatus = Literal["ok", "insufficient", "error", "std_fill"]


class BatchSample(BaseModel):
    id: str
    id_synthesized: bool = False
    grade: str | None = None
    composition: dict[str, float | None]    # 13 keys, None where missing
    missing_required: list[str]
    filled_elements: list[str] = []         # elements filled from GB standard (std_fill rows only)
    status: BatchStatus
    prediction: PredictionResponse | None
    error: str | None


class BatchSummary(BaseModel):
    total_rows: int       # raw spreadsheet rows (before dedup)
    deduped: int          # rows removed as duplicate 炉号
    skipped_empty: int    # all-NaN chemistry rows, dropped silently
    predicted: int        # status == "ok"
    insufficient: int
    std_fill: int         # GB-standard-filled companion rows
    errored: int


class BatchResponse(BaseModel):
    filename: str
    summary: BatchSummary
    samples: list[BatchSample]


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

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
        "http://tauri.localhost",
        "https://tauri.localhost",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

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
        "standard_bounds": STANDARD_BOUNDS,
    }


@app.post("/api/predict", response_model=PredictionResponse)
def predict(req: CompositionRequest) -> PredictionResponse:
    pred: Predictor = app.state.predictor
    try:
        result = pred.predict(req.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"prediction failed: {exc}") from exc
    return PredictionResponse(**result)


@app.post("/api/batch")
async def batch(file: UploadFile = File(...)) -> StreamingResponse:
    """Parse an XLS/XLSX file and stream predictions as Server-Sent Events.

    SSE event sequence:
      data: {"type":"start",  "filename":str, "total":int, "deduped":int}
      data: {"type":"progress","done":int, "total":int}   (repeated)
      data: {"type":"done",   "filename":str, "summary":{...}, "samples":[...]}

    On error before streaming starts: HTTP 400 / 415.
    On error mid-stream: data: {"type":"error","message":str}
    """
    content = await file.read()
    if len(content) > MAX_BATCH_BYTES:
        raise HTTPException(413, "file too large (max 20 MB)")

    try:
        fmt = detect_format(content[:64])
    except UnsupportedFormatError as exc:
        raise HTTPException(415, str(exc))

    try:
        df = read_dataframe(content, fmt)
        mapping = normalize_columns(df)
        all_rows = parse_rows(df, mapping)
        if all_rows and all(r.status == "empty" for r in all_rows):
            raise ParseError("file contains no usable data rows")
    except ParseError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(400, f"could not parse file: {exc}") from exc

    rows, deduped_count = deduplicate_rows(all_rows)
    filename = file.filename or "batch"
    pred: Predictor = app.state.predictor

    non_empty = [r for r in rows if r.status != "empty"]
    skipped_empty = len(rows) - len(non_empty)
    total = len(non_empty)

    async def generate():
        # start — announce total work so the client can render a progress bar
        yield f"data: {_json.dumps({'type':'start','filename':filename,'total':total,'deduped':deduped_count})}\n\n"

        summary_counts = {
            "total_rows": len(all_rows),
            "deduped": deduped_count,
            "skipped_empty": skipped_empty,
            "predicted": 0,
            "insufficient": 0,
            "std_fill": 0,
            "errored": 0,
        }
        samples: list[dict] = []
        # Emit ~100 progress events at most (never less than every row).
        interval = max(1, total // 100)

        try:
            for i, r in enumerate(non_empty):
                std_companion: BatchSample | None = None

                if r.status == "insufficient":
                    summary_counts["insufficient"] += 1
                    sample = BatchSample(
                        id=r.id, id_synthesized=r.id_synthesized, grade=r.grade,
                        composition=r.composition,
                        missing_required=r.missing_required,
                        status="insufficient", prediction=None, error=None,
                    )

                    # Build a GB-standard-filled companion row when grade is known.
                    if r.grade:
                        std_vals = grade_lookup(r.grade)
                        if std_vals:
                            filled_comp: dict[str, float | None] = dict(r.composition)
                            filled: list[str] = []
                            for elem, std_val in std_vals.items():
                                if filled_comp.get(elem) is None:
                                    filled_comp[elem] = std_val
                                    filled.append(elem)
                            if filled:
                                req_data = {
                                    k: v for k, v in filled_comp.items()
                                    if v is not None or k in REQUIRED_KEYS
                                }
                                try:
                                    comp_req = CompositionRequest(**req_data)
                                    result = pred.predict(comp_req.model_dump())
                                    summary_counts["std_fill"] += 1
                                    std_companion = BatchSample(
                                        id=f"{r.id} (GB)",
                                        id_synthesized=r.id_synthesized,
                                        grade=r.grade,
                                        composition={k: filled_comp.get(k) for k in ELEMENT_FIELDS},
                                        missing_required=[],
                                        filled_elements=filled,
                                        status="std_fill",
                                        prediction=PredictionResponse(**result),
                                        error=None,
                                    )
                                except Exception:
                                    pass  # silently skip if validation or prediction fails
                else:
                    try:
                        req_data = {
                            k: v for k, v in r.composition.items()
                            if v is not None or k in REQUIRED_KEYS
                        }
                        comp_req = CompositionRequest(**req_data)
                    except ValidationError as exc:
                        summary_counts["errored"] += 1
                        sample = BatchSample(
                            id=r.id, id_synthesized=r.id_synthesized, grade=r.grade,
                            composition=r.composition, missing_required=[],
                            status="error", prediction=None,
                            error=f"invalid value: {exc.errors()[0]['msg']}",
                        )
                    else:
                        try:
                            result = pred.predict(comp_req.model_dump())
                            summary_counts["predicted"] += 1
                            sample = BatchSample(
                                id=r.id, id_synthesized=r.id_synthesized, grade=r.grade,
                                composition=r.composition, missing_required=[],
                                status="ok",
                                prediction=PredictionResponse(**result), error=None,
                            )
                        except Exception as exc:
                            summary_counts["errored"] += 1
                            sample = BatchSample(
                                id=r.id, id_synthesized=r.id_synthesized, grade=r.grade,
                                composition=r.composition, missing_required=[],
                                status="error", prediction=None, error=str(exc),
                            )

                samples.append(sample.model_dump())
                if std_companion is not None:
                    samples.append(std_companion.model_dump())

                done = i + 1
                if done % interval == 0 or done == total:
                    yield f"data: {_json.dumps({'type':'progress','done':done,'total':total})}\n\n"
                    # Yield to the asyncio runloop so uvicorn actually flushes
                    # the chunk to the client; without this the synchronous
                    # predictor.predict loop starves the event loop and the
                    # browser receives all events at once at the end.
                    await asyncio.sleep(0)

            yield f"data: {_json.dumps({'type':'done','filename':filename,'summary':summary_counts,'samples':samples})}\n\n"

        except Exception as exc:
            yield f"data: {_json.dumps({'type':'error','message':str(exc)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Frontend serving
# ---------------------------------------------------------------------------

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
