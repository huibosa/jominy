// Batch prediction page — drop xlsx → auto-parse (mocked) → auto-predict.
// Sticky-id + sticky-J9/J15 spreadsheet view with sortable hardness columns.

import type { BatchSample, Metadata } from "./types";
import {
  ALL_ELEMENTS,
  ElementKey,
  Language,
  TRANSLATIONS,
  el,
  isOutOfRange,
  postPrediction,
} from "./shared";
import { parseXlsxMock } from "./mock-parse";

type SortKey = "J9" | "J15" | null;
type SortDir = "asc" | "desc";

export interface BatchState {
  fileName: string | null;
  samples: BatchSample[];
  sortKey: SortKey;
  sortDir: SortDir;
}

export function createBatchState(): BatchState {
  return { fileName: null, samples: [], sortKey: null, sortDir: "desc" };
}

function clearChildren(node: Node): void {
  while (node.firstChild) node.removeChild(node.firstChild);
}

function fmtCell(
  value: number | null,
  decimals: number,
  outOfRange: boolean,
): HTMLElement {
  if (value === null) {
    return el("td", { class: "dim" }, ["—"]);
  }
  const cls = outOfRange ? "out-of-range" : "";
  return el("td", { class: cls }, [value.toFixed(decimals)]);
}

function buildHeader(language: Language, state: BatchState, onSort: (key: SortKey) => void): HTMLElement {
  const text = TRANSLATIONS[language];
  const tr = el("tr", {}, []);

  // ID column (sticky left)
  tr.append(
    el("th", { class: "col-id" }, [text.colId]),
  );

  // Element columns — keep natural case (Cr, Mn, Al…)
  for (const k of ALL_ELEMENTS) {
    tr.append(el("th", { class: "col-element" }, [k]));
  }

  // J9 / J15 (sticky right) — sortable
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
  const tr = el("tr", { "data-id": sample.id }, []);

  tr.append(el("td", { class: "col-id" }, [sample.id]));

  for (const k of ALL_ELEMENTS) {
    const value = sample.composition[k as ElementKey];
    const decimals = k === "B" ? 4 : 3;
    const out = isOutOfRange(value, stats[k]);
    tr.append(fmtCell(value, decimals, out));
  }

  // J9 / J15
  const j9Td = el("td", { class: "col-j9" }, []);
  const j15Td = el("td", { class: "col-j15" }, []);
  if (sample.loading) {
    j9Td.classList.add("is-loading");
    j15Td.classList.add("is-loading");
    j9Td.append(document.createTextNode("…"));
    j15Td.append(document.createTextNode("…"));
  } else if (sample.error) {
    j9Td.classList.add("has-error");
    j15Td.classList.add("has-error");
    j9Td.setAttribute("title", sample.error);
    j15Td.setAttribute("title", sample.error);
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

function sortedSamples(state: BatchState): BatchSample[] {
  if (!state.sortKey) return state.samples;
  const key = state.sortKey;
  const dir = state.sortDir === "asc" ? 1 : -1;
  return [...state.samples].sort((a, b) => {
    const av = a.prediction ? a.prediction[key] : null;
    const bv = b.prediction ? b.prediction[key] : null;
    if (av === null && bv === null) return 0;
    if (av === null) return 1;
    if (bv === null) return -1;
    return (av - bv) * dir;
  });
}

function exportCsv(state: BatchState): void {
  if (state.samples.length === 0) return;
  const headers = ["heat_id", ...ALL_ELEMENTS, "J9", "J15", "delta"];
  const rows = [headers.join(",")];
  for (const s of state.samples) {
    const cells: string[] = [s.id];
    for (const k of ALL_ELEMENTS) {
      const v = s.composition[k as ElementKey];
      cells.push(v === null ? "" : String(v));
    }
    cells.push(
      s.prediction ? s.prediction.J9.toFixed(2) : "",
      s.prediction ? s.prediction.J15.toFixed(2) : "",
      s.prediction ? s.prediction.delta.toFixed(2) : "",
    );
    rows.push(cells.join(","));
  }
  const blob = new Blob([rows.join("\n")], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  const baseName = (state.fileName ?? "batch").replace(/\.[^.]+$/, "");
  a.href = url;
  a.download = `${baseName}-predictions.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

async function predictAll(state: BatchState, language: Language, redraw: () => void): Promise<void> {
  // Run sequentially to be polite to the local sidecar — each request is
  // ~10ms anyway. Easy to swap for `Promise.all` if the backend prefers.
  for (const sample of state.samples) {
    try {
      const result = await postPrediction(sample.composition, language);
      sample.prediction = result;
      sample.error = null;
    } catch (err) {
      sample.error = (err as Error).message;
    } finally {
      sample.loading = false;
      redraw();
    }
  }
}

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

  // ---------- Empty state — drop zone ----------
  if (state.samples.length === 0) {
    const fileInput = el("input", {
      type: "file",
      accept: ".xlsx,.xls,.csv",
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
        text.dropSchema(`炉号, ${ALL_ELEMENTS.join(", ")}`),
      ]),
      el("button", { type: "button", class: "dropzone__pick" }, [text.dropPick]),
      fileInput,
    ]);

    const handleFile = async (file: File) => {
      state.fileName = file.name;
      state.samples = await parseXlsxMock(file);
      onChange();
      // Kick off prediction after the next paint so the user sees the rows
      // appear with loading markers, then settle.
      setTimeout(() => {
        void predictAll(state, language, onChange);
      }, 0);
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
      const file = e.dataTransfer?.files?.[0];
      if (file) void handleFile(file);
    });

    root.append(dropzone);
    return;
  }

  // ---------- Loaded state — strip + table ----------
  const loadedCount = state.samples.filter((s) => !s.loading).length;
  const totalCount = state.samples.length;
  const countLabel = loadedCount < totalCount
    ? `${loadedCount}/${totalCount} ${language === "zh" ? "已完成" : "ready"}`
    : text.batchCount(totalCount);

  const exportBtn = el("button", { class: "btn", id: "btn-export" }, [
    el("span", {}, [text.batchExport]),
    el("span", { class: "btn__chev" }, ["↓"]),
  ]) as HTMLButtonElement;
  if (loadedCount === 0) exportBtn.disabled = true;

  const clearBtn = el("button", { class: "btn btn--ghost", id: "btn-clear" }, [
    text.batchClear,
  ]) as HTMLButtonElement;

  const strip = el("div", { class: "batch-strip fade-in" }, [
    el("div", { class: "batch-strip__meta" }, [
      el("div", { class: "batch-strip__file" }, [state.fileName ?? "batch.xlsx"]),
      el("div", { class: "batch-strip__count" }, [
        el("strong", {}, [String(totalCount)]),
        countLabel,
      ]),
    ]),
    el("div", { class: "batch-strip__actions" }, [exportBtn, clearBtn]),
  ]);

  exportBtn.addEventListener("click", () => exportCsv(state));
  clearBtn.addEventListener("click", () => {
    state.fileName = null;
    state.samples = [];
    state.sortKey = null;
    onChange();
  });

  // Table
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
  for (const sample of sortedSamples(state)) {
    tbody.append(buildRow(sample, metadata));
  }

  const table = el("table", { class: "batch-table" }, [thead, tbody]);
  const wrap = el("div", { class: "batch-table-wrap fade-in" }, [table]);

  root.append(strip, wrap);
}
