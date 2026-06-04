// Batch prediction page — drop xls/xlsx → single /api/batch SSE stream → table.
// Sticky-id + sticky-J9/J15 spreadsheet view with sortable hardness columns.
//
// Windows / Tauri note
// --------------------
// WebView2 (used by Tauri on Windows) fires HTML5 dragover/drop events but
// leaves dataTransfer.files empty — a known WebView2 limitation.  We work
// around this by also listening to the native `tauri://drag-drop` window event
// which does carry the real filesystem paths.  When a path is received we POST
// it to /api/batch-path and the Python sidecar reads the file directly from
// disk, reusing the same SSE streaming pipeline.

import type { BatchResponse, BatchSample, BatchSummary, Metadata } from "./types";
import {
  ALL_ELEMENTS,
  ElementKey,
  Language,
  TRANSLATIONS,
  el,
  isOutOfRange,
} from "./shared";
import { BASE } from "./api";

// ---------------------------------------------------------------------------
// Tauri interop — minimal type surface, no @tauri-apps/api dependency
// ---------------------------------------------------------------------------

interface TauriDragDropPayload {
  paths: string[];
  position: { x: number; y: number };
}

declare global {
  interface Window {
    __TAURI__?: {
      event: {
        listen: <T>(
          event: string,
          handler: (e: { payload: T }) => void,
        ) => Promise<() => void>;
      };
    };
  }
}

function isTauri(): boolean {
  return typeof window.__TAURI__ !== "undefined";
}

// Register Tauri drag-drop listeners exactly once per page session.
// The closures capture the mutable `state` object and the stable `onChange`
// reference so they stay correct across re-renders without re-registration.
let _tauriListenersRegistered = false;

function registerTauriDragListeners(
  state: BatchState,
  onChange: () => void,
): void {
  if (!isTauri() || _tauriListenersRegistered) return;
  _tauriListenersRegistered = true;

  const listen = window.__TAURI__!.event.listen;

  const addActive = () => {
    if (state.phase.kind === "empty" || state.phase.kind === "error") {
      document.querySelector(".dropzone")?.classList.add("is-active");
    }
  };
  const removeActive = () => {
    document.querySelector(".dropzone")?.classList.remove("is-active");
  };

  void Promise.all([
    listen("tauri://drag-enter", addActive),
    listen("tauri://drag-over", addActive),
    listen("tauri://drag-leave", removeActive),
    listen<TauriDragDropPayload>("tauri://drag-drop", (e) => {
      removeActive();
      // Only accept drops when the dropzone is actually visible.
      if (state.phase.kind !== "empty" && state.phase.kind !== "error") return;
      const path = e.payload.paths?.[0];
      if (path) void handleFileByPath(path, state, onChange);
    }),
  ]);
}

async function handleFileByPath(
  path: string,
  state: BatchState,
  onChange: () => void,
): Promise<void> {
  const fileName = path.replace(/\\/g, "/").split("/").pop() ?? path;
  state.phase = { kind: "loading", fileName, total: 0, done: 0 };
  onChange();

  try {
    const res = await fetch(`${BASE}/api/batch-path`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    });
    if (!res.ok) {
      const detail = await res.text();
      state.phase = { kind: "error", message: detail || `HTTP ${res.status}` };
      onChange();
      return;
    }
    await readBatchStream(res, state, onChange);
  } catch (err) {
    state.phase = { kind: "error", message: (err as Error).message };
    onChange();
  }
}

type SortKey = "J9" | "J15" | null;
type SortDir = "asc" | "desc";

export type BatchPhase =
  | { kind: "empty" }
  | { kind: "loading"; fileName: string; total: number; done: number }
  | { kind: "loaded"; data: BatchResponse }
  | { kind: "error"; message: string };

export interface BatchState {
  phase: BatchPhase;
  sortKey: SortKey;
  sortDir: SortDir;
}

export function createBatchState(): BatchState {
  return { phase: { kind: "empty" }, sortKey: null, sortDir: "desc" };
}

function clearChildren(node: Node): void {
  while (node.firstChild) node.removeChild(node.firstChild);
}

/** Escape a single CSV cell: double-quote if it contains comma, quote, or newline. */
function csvCell(value: string): string {
  if (value.includes(",") || value.includes('"') || value.includes("\n")) {
    return `"${value.replace(/"/g, '""')}"`;
  }
  return value;
}

function fmtCell(
  value: number | null,
  decimals: number,
  outOfRange: boolean,
  missing: boolean,
): HTMLElement {
  if (missing || value === null) {
    return el("td", { class: "dim" }, ["—"]);
  }
  const cls = outOfRange ? "out-of-range" : "";
  return el("td", { class: cls }, [value.toFixed(decimals)]);
}

// ---------------------------------------------------------------------------
// SSE stream reader
// ---------------------------------------------------------------------------

/**
 * Update the loading view's progress bar in place — directly mutating DOM
 * rather than calling onChange(). Triggering a full re-render on every
 * progress event would re-run the title-block / tabs / heading fade-in
 * animations and the user would see the whole shell flicker.
 */
function updateProgressDom(done: number, total: number): void {
  const bar = document.querySelector<HTMLElement>(".batch-progress__bar");
  const label = document.querySelector<HTMLElement>(".batch-progress__label");
  if (!bar || !label) return;
  const pct = total > 0 ? Math.round((done / total) * 100) : 0;
  bar.style.width = total > 0 ? `${pct}%` : "0%";
  label.textContent = total > 0
    ? `${done.toLocaleString()} / ${total.toLocaleString()} · ${pct}%`
    : label.textContent;
}

async function readBatchStream(
  res: Response,
  state: BatchState,
  onChange: () => void,
): Promise<void> {
  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // SSE events are separated by \n\n
      const parts = buffer.split("\n\n");
      buffer = parts.pop() ?? "";

      for (const part of parts) {
        const dataLine = part.split("\n").find((l) => l.startsWith("data: "));
        if (!dataLine) continue;

        let payload: Record<string, unknown>;
        try {
          payload = JSON.parse(dataLine.slice(6)) as Record<string, unknown>;
        } catch {
          continue;
        }

        switch (payload.type) {
          case "start": {
            // Mutate state so a downstream phase change reads the right total,
            // but update DOM in place — DON'T trigger a re-render.
            const total = payload.total as number;
            if (state.phase.kind === "loading") {
              state.phase = {
                ...state.phase,
                fileName: payload.filename as string,
                total,
                done: 0,
              };
            }
            updateProgressDom(0, total);
            break;
          }

          case "progress": {
            const done = payload.done as number;
            const total = payload.total as number;
            if (state.phase.kind === "loading") {
              state.phase = { ...state.phase, done, total };
            }
            updateProgressDom(done, total);
            break;
          }

          case "done":
            state.phase = {
              kind: "loaded",
              data: {
                filename: payload.filename as string,
                summary: payload.summary as BatchSummary,
                samples: payload.samples as BatchSample[],
              },
            };
            onChange();
            break;

          case "error":
            state.phase = { kind: "error", message: payload.message as string };
            onChange();
            break;
        }
      }
    }
  } catch (err) {
    // Only override phase if we haven't already settled on a result.
    if (state.phase.kind !== "loaded") {
      state.phase = { kind: "error", message: (err as Error).message };
      onChange();
    }
  }
}

// ---------------------------------------------------------------------------
// Drop zone + file handling
// ---------------------------------------------------------------------------

function buildDropzone(
  language: Language,
  state: BatchState,
  onChange: () => void,
  errorMsg?: string,
): HTMLElement {
  const text = TRANSLATIONS[language];

  const fileInput = el("input", {
    type: "file",
    accept: ".xls,.xlsx",
    style: "display:none",
    id: "batch-file-input",
  }) as HTMLInputElement;

  const dropzone = el("div", { class: "dropzone fade-in", role: "button", tabindex: "0" }, [
    el("div", { class: "crosshair" }, [
      el("span", { class: "crosshair__h" }),
      el("span", { class: "crosshair__v" }),
      el("span", { class: "crosshair__dot" }),
    ]),
    el("h3", { class: "dropzone__title" }, [text.dropTitle]),
    el("p", { class: "dropzone__hint" }, [text.dropHint]),
    el("p", { class: "dropzone__schema" }, [
      text.dropSchema(`炉号/lh, ${ALL_ELEMENTS.join(", ")}`),
    ]),
    el("button", { type: "button", class: "dropzone__pick" }, [text.dropPick]),
    fileInput,
  ]);

  const handleFile = async (file: File) => {
    state.phase = { kind: "loading", fileName: file.name, total: 0, done: 0 };
    onChange();

    const fd = new FormData();
    fd.append("file", file);

    try {
      const res = await fetch(`${BASE}/api/batch`, { method: "POST", body: fd });
      if (!res.ok) {
        const detail = await res.text();
        state.phase = { kind: "error", message: detail || `HTTP ${res.status}` };
        onChange();
        return;
      }
      await readBatchStream(res, state, onChange);
    } catch (err) {
      state.phase = { kind: "error", message: (err as Error).message };
      onChange();
    }
  };

  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("keydown", (e) => {
    if ((e as KeyboardEvent).key === "Enter" || (e as KeyboardEvent).key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });
  fileInput.addEventListener("change", () => {
    const file = fileInput.files?.[0];
    if (file) void handleFile(file);
  });
  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("is-active");
  });
  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("is-active");
  });
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("is-active");
    // dataTransfer.files is empty in Tauri/WebView2 on Windows; the native
    // tauri://drag-drop listener (registered below) handles that path instead.
    const file = (e as DragEvent).dataTransfer?.files?.[0];
    if (file) void handleFile(file);
  });

  // On Windows the Tauri WebView2 does not populate dataTransfer.files.
  // Register native Tauri drag-drop listeners as a fallback (no-op in browser).
  registerTauriDragListeners(state, onChange);

  if (errorMsg) {
    const banner = el("div", { class: "batch-error-banner" }, [
      el("strong", {}, ["⚠ "]),
      errorMsg,
    ]);
    return el("div", { class: "batch-error-wrap fade-in" }, [banner, dropzone]);
  }

  return dropzone;
}

// ---------------------------------------------------------------------------
// Loading progress view
// ---------------------------------------------------------------------------

function buildLoadingView(
  phase: Extract<BatchPhase, { kind: "loading" }>,
  language: Language,
): HTMLElement {
  const text = TRANSLATIONS[language];
  const { fileName, done, total } = phase;
  const pct = total > 0 ? Math.round((done / total) * 100) : 0;
  const barWidth = total > 0 ? `${pct}%` : "0%";

  const label = total > 0
    ? `${done.toLocaleString()} / ${total.toLocaleString()} · ${pct}%`
    : text.batchPredicting;

  return el("div", { class: "batch-progress fade-in" }, [
    el("div", { class: "batch-progress__file" }, [
      el("span", { class: "batch-progress__icon" }, ["▣"]),
      fileName,
    ]),
    el("div", { class: "batch-progress__bar-wrap" }, [
      el("div", { class: "batch-progress__bar", style: `width: ${barWidth}` }, []),
    ]),
    el("div", { class: "batch-progress__label" }, [label]),
  ]);
}

// ---------------------------------------------------------------------------
// Table helpers
// ---------------------------------------------------------------------------

function buildHeader(language: Language, state: BatchState, onSort: (key: SortKey) => void): HTMLElement {
  const text = TRANSLATIONS[language];
  const tr = el("tr", {}, []);

  tr.append(el("th", { class: "col-id" }, [text.colId]));
  tr.append(el("th", { class: "col-grade" }, [text.colGrade]));

  for (const k of ALL_ELEMENTS) {
    tr.append(el("th", { class: "col-element" }, [k]));
  }

  const j9Marker = state.sortKey === "J9" ? (state.sortDir === "asc" ? "▲" : "▼") : "↕";
  const j15Marker = state.sortKey === "J15" ? (state.sortDir === "asc" ? "▲" : "▼") : "↕";

  const j9 = el("th", { class: "col-j9 sortable" }, [
    text.colJ9,
    el("span", { class: "sort-marker" }, [j9Marker]),
  ]);
  if (state.sortKey === "J9") j9.setAttribute("aria-sort", state.sortDir === "asc" ? "ascending" : "descending");
  j9.addEventListener("click", () => onSort("J9"));

  const j15 = el("th", { class: "col-j15 sortable" }, [
    text.colJ15,
    el("span", { class: "sort-marker" }, [j15Marker]),
  ]);
  if (state.sortKey === "J15") j15.setAttribute("aria-sort", state.sortDir === "asc" ? "ascending" : "descending");
  j15.addEventListener("click", () => onSort("J15"));

  tr.append(j9, j15);
  return tr;
}

function buildRow(sample: BatchSample, metadata: Metadata): HTMLElement {
  const stats = metadata.feature_stats;
  const isInsuf = sample.status === "insufficient";
  const isErr = sample.status === "error";
  const tr = el("tr", { "data-id": sample.id, class: isInsuf ? "row-insufficient" : "" }, []);

  const idText = sample.id_synthesized ? `${sample.id}*` : sample.id;
  tr.append(el("td", { class: isInsuf ? "col-id dim" : "col-id" }, [idText]));
  tr.append(el("td", { class: isInsuf ? "col-grade dim" : "col-grade" }, [sample.grade ?? "—"]));

  for (const k of ALL_ELEMENTS) {
    const value = sample.composition[k as ElementKey];
    const decimals = k === "B" ? 4 : 3;
    const out = isOutOfRange(value, stats[k]);
    const isMissing = isInsuf && sample.missing_required.includes(k);
    tr.append(fmtCell(value, decimals, out, isMissing));
  }

  const j9Td = el("td", { class: "col-j9" }, []);
  const j15Td = el("td", { class: "col-j15" }, []);

  if (isInsuf) {
    j9Td.classList.add("is-insuf");
    j15Td.classList.add("is-insuf");
    const tip = sample.missing_required.join(", ");
    if (tip) { j9Td.setAttribute("title", tip); j15Td.setAttribute("title", tip); }
    j9Td.append(document.createTextNode("INSUF"));
    j15Td.append(document.createTextNode("INSUF"));
  } else if (isErr) {
    j9Td.classList.add("has-error");
    j15Td.classList.add("has-error");
    if (sample.error) { j9Td.setAttribute("title", sample.error); j15Td.setAttribute("title", sample.error); }
    j9Td.append(document.createTextNode("ERR"));
    j15Td.append(document.createTextNode("ERR"));
  } else if (sample.prediction) {
    if (sample.prediction.warnings.length > 0) {
      j9Td.classList.add("has-warning");
      j15Td.classList.add("has-warning");
      const tip = sample.prediction.warnings.join("\n");
      j9Td.setAttribute("title", tip);
      j15Td.setAttribute("title", tip);
    }
    j9Td.append(document.createTextNode(sample.prediction.J9.toFixed(2)));
    j15Td.append(document.createTextNode(sample.prediction.J15.toFixed(2)));
  }

  tr.append(j9Td, j15Td);
  return tr;
}

function sortedSamples(samples: BatchSample[], state: BatchState): BatchSample[] {
  if (!state.sortKey) return samples;
  const key = state.sortKey;
  const dir = state.sortDir === "asc" ? 1 : -1;
  return [...samples].sort((a, b) => {
    const av = a.prediction ? a.prediction[key] : null;
    const bv = b.prediction ? b.prediction[key] : null;
    if (av === null && bv === null) return 0;
    if (av === null) return 1;
    if (bv === null) return -1;
    return (av - bv) * dir;
  });
}

function exportCsv(data: BatchResponse): void {
  const headers = ["heat_id", "grade", "status", "missing_required", ...ALL_ELEMENTS, "J9", "J15", "delta"];
  const rows = [headers.map(csvCell).join(",")];
  for (const s of data.samples) {
    const cells: string[] = [
      csvCell(s.id),
      csvCell(s.grade ?? ""),
      csvCell(s.status),
      csvCell(s.missing_required.join("|")),
    ];
    for (const k of ALL_ELEMENTS) {
      const v = s.composition[k as ElementKey];
      cells.push(csvCell(v === null ? "" : String(v)));
    }
    cells.push(
      csvCell(s.prediction ? s.prediction.J9.toFixed(2) : ""),
      csvCell(s.prediction ? s.prediction.J15.toFixed(2) : ""),
      csvCell(s.prediction ? s.prediction.delta.toFixed(2) : ""),
    );
    rows.push(cells.join(","));
  }
  const blob = new Blob([rows.join("\n")], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  const baseName = data.filename.replace(/\.[^.]+$/, "");
  a.href = url;
  a.download = `${baseName}-predictions.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

// ---------------------------------------------------------------------------
// Main render
// ---------------------------------------------------------------------------

export function renderBatch(
  root: HTMLElement,
  metadata: Metadata,
  state: BatchState,
  language: Language,
  onChange: () => void,
): void {
  const text = TRANSLATIONS[language];
  clearChildren(root);

  const heading = el("div", { class: "section__heading fade-in" }, [
    el("h2", {}, [text.batchHeading]),
    el("span", { class: "stamp" }, [text.batchHeadingStamp]),
  ]);
  root.append(heading);

  const phase = state.phase;

  if (phase.kind === "empty") {
    root.append(buildDropzone(language, state, onChange));
    return;
  }

  if (phase.kind === "error") {
    root.append(buildDropzone(language, state, onChange, phase.message));
    return;
  }

  if (phase.kind === "loading") {
    root.append(buildLoadingView(phase, language));
    return;
  }

  // ---------- Loaded state — strip + table ----------
  const { data } = phase;
  const { summary } = data;

  // Summary strip segments
  const summaryParts: HTMLElement[] = [];

  summaryParts.push(el("span", { class: "summary-ok" }, [
    el("strong", {}, [String(summary.predicted)]),
    ` ${text.batchOk}`,
  ]));

  if (summary.insufficient > 0) {
    summaryParts.push(el("span", { class: "summary-sep" }, [" · "]));
    summaryParts.push(el("span", { class: "summary-insuf" }, [
      el("strong", {}, [String(summary.insufficient)]),
      ` ${text.batchInsuf}`,
    ]));
  }

  if (summary.skipped_empty > 0) {
    summaryParts.push(el("span", { class: "summary-sep" }, [" · "]));
    summaryParts.push(el("span", { class: "summary-skipped" }, [
      el("strong", {}, [String(summary.skipped_empty)]),
      ` ${text.batchSkipped}`,
    ]));
  }

  if (summary.deduped > 0) {
    summaryParts.push(el("span", { class: "summary-sep" }, [" · "]));
    summaryParts.push(el("span", { class: "summary-dedup" }, [
      el("strong", {}, [String(summary.deduped)]),
      ` ${text.batchDedup}`,
    ]));
  }

  if (summary.errored > 0) {
    summaryParts.push(el("span", { class: "summary-sep" }, [" · "]));
    summaryParts.push(el("span", { class: "summary-err" }, [
      el("strong", {}, [String(summary.errored)]),
      ` ${text.batchErr}`,
    ]));
  }

  const exportBtn = el("button", { class: "btn", id: "btn-export" }, [
    el("span", {}, [text.batchExport]),
    el("span", { class: "btn__chev" }, ["↓"]),
  ]) as HTMLButtonElement;
  if (summary.predicted === 0) exportBtn.disabled = true;

  const clearBtn = el("button", { class: "btn btn--ghost", id: "btn-clear" }, [
    text.batchClear,
  ]) as HTMLButtonElement;

  const strip = el("div", { class: "batch-strip fade-in" }, [
    el("div", { class: "batch-strip__meta" }, [
      el("div", { class: "batch-strip__file" }, [data.filename]),
      el("div", { class: "batch-strip__count" }, summaryParts),
    ]),
    el("div", { class: "batch-strip__actions" }, [exportBtn, clearBtn]),
  ]);

  exportBtn.addEventListener("click", () => exportCsv(data));
  clearBtn.addEventListener("click", () => {
    state.phase = { kind: "empty" };
    state.sortKey = null;
    onChange();
  });

  const onSort = (key: SortKey) => {
    if (state.sortKey === key) {
      state.sortDir = state.sortDir === "asc" ? "desc" : "asc";
    } else {
      state.sortKey = key;
      state.sortDir = "desc";
    }
    onChange();
  };

  const thead = el("thead", {}, [buildHeader(language, state, onSort)]);
  const tbody = el("tbody", {});
  for (const sample of sortedSamples(data.samples, state)) {
    tbody.append(buildRow(sample, metadata));
  }

  const table = el("table", { class: "batch-table" }, [thead, tbody]);
  const wrap = el("div", { class: "batch-table-wrap fade-in" }, [table]);

  root.append(strip, wrap);
}
