import "./style.css";
import type { CompositionRequest, Metadata, PredictionResponse } from "./types";
import { BASE } from "./api";

const REQUIRED_ELEMENTS = ["C", "Si", "Mn", "P", "S", "Cu", "Ni", "Cr"] as const;
const OPTIONAL_ELEMENTS = ["V", "Ti", "W", "Al", "B"] as const;
type ElementKey = (typeof REQUIRED_ELEMENTS)[number] | (typeof OPTIONAL_ELEMENTS)[number];

const SAMPLE_COMPOSITION: CompositionRequest = {
  C: 0.20,
  Si: 0.26,
  Mn: 0.96,
  P: 0.012,
  S: 0.018,
  Cu: 0.05,
  Ni: 0.10,
  Cr: 1.10,
  V: null,
  Ti: 0.005,
  W: null,
  Al: 0.025,
  B: null,
};

const ELEMENT_LABELS: Record<ElementKey, string> = {
  C: "Carbon",
  Si: "Silicon",
  Mn: "Manganese",
  P: "Phosphorus",
  S: "Sulfur",
  Cu: "Copper",
  Ni: "Nickel",
  Cr: "Chromium",
  V: "Vanadium",
  Ti: "Titanium",
  W: "Tungsten",
  Al: "Aluminum",
  B: "Boron",
};

function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  attrs: Record<string, string> = {},
  children: (Node | string)[] = [],
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else node.setAttribute(k, v);
  }
  for (const child of children) {
    node.append(typeof child === "string" ? document.createTextNode(child) : child);
  }
  return node;
}

function fmtRange(stat: { p01?: number; p99?: number } | undefined): string {
  if (!stat || stat.p01 === undefined || stat.p99 === undefined) return "";
  return `typical ${stat.p01.toFixed(3)}–${stat.p99.toFixed(3)} wt%`;
}

function buildField(
  key: ElementKey,
  optional: boolean,
  initial: number | null,
  stat?: { p01?: number; p99?: number },
): HTMLLabelElement {
  const input = el("input", {
    type: "number",
    name: key,
    step: "0.001",
    min: "0",
    "data-key": key,
  });
  if (initial !== null) input.value = String(initial);
  if (!optional) input.required = true;

  const label = el("label", { class: optional ? "optional" : "" }, [
    el("span", { class: "name" }, [`${key} — ${ELEMENT_LABELS[key]}`]),
    input,
    el("span", { class: "range" }, [fmtRange(stat)]),
  ]);
  return label;
}

function readForm(form: HTMLFormElement): CompositionRequest {
  const get = (key: string) => {
    const input = form.querySelector<HTMLInputElement>(`[data-key="${key}"]`);
    if (!input) return null;
    const val = input.value.trim();
    if (val === "") return null;
    const n = Number(val);
    return Number.isFinite(n) ? n : null;
  };
  const required = (key: string) => {
    const v = get(key);
    if (v === null) throw new Error(`${key} is required`);
    return v;
  };
  return {
    C: required("C"),
    Si: required("Si"),
    Mn: required("Mn"),
    P: required("P"),
    S: required("S"),
    Cu: required("Cu"),
    Ni: required("Ni"),
    Cr: required("Cr"),
    V: get("V"),
    Ti: get("Ti"),
    W: get("W"),
    Al: get("Al"),
    B: get("B"),
  };
}

function fillForm(form: HTMLFormElement, comp: CompositionRequest): void {
  for (const [k, v] of Object.entries(comp)) {
    const input = form.querySelector<HTMLInputElement>(`[data-key="${k}"]`);
    if (input) input.value = v === null ? "" : String(v);
  }
}

function renderResult(container: HTMLElement, result: PredictionResponse): void {
  container.innerHTML = "";
  const grid = el("div", { class: "results" }, [
    el("div", { class: "result" }, [
      el("div", { class: "label" }, ["J9"]),
      el("div", {}, [
        el("span", { class: "value" }, [result.J9.toFixed(2)]),
        el("span", { class: "unit" }, ["HRC"]),
      ]),
    ]),
    el("div", { class: "result" }, [
      el("div", { class: "label" }, ["J15"]),
      el("div", {}, [
        el("span", { class: "value" }, [result.J15.toFixed(2)]),
        el("span", { class: "unit" }, ["HRC"]),
      ]),
    ]),
    el("div", { class: "result" }, [
      el("div", { class: "label" }, ["J9 − J15"]),
      el("div", {}, [
        el("span", { class: "value" }, [result.delta.toFixed(2)]),
        el("span", { class: "unit" }, ["HRC"]),
      ]),
    ]),
  ]);
  container.append(grid);

  const expected = el("p", { class: "muted" }, [
    `Expected MAE from cross-validation: ±${result.expected_mae.J9.toFixed(2)} HRC on J9, ±${result.expected_mae.delta.toFixed(2)} HRC on δ.`,
  ]);
  expected.style.color = "var(--muted)";
  expected.style.fontSize = "0.85rem";
  expected.style.marginTop = "0.75rem";
  container.append(expected);

  if (result.warnings.length > 0) {
    const warn = el("div", { class: "warning" }, [
      el("strong", {}, ["Input outside training range"]),
      el("ul", {}, result.warnings.map((w) => el("li", {}, [w]))),
    ]);
    container.append(warn);
  }

  const breakdown = el("details", {}, [
    el("summary", {}, ["Per-component predictions"]),
    el("ul", {}, [
      el("li", {}, [`J9 from XGBoost: ${result.components.j9_xgb.toFixed(2)} HRC`]),
      el("li", {}, [`J9 from PLS: ${result.components.j9_pls.toFixed(2)} HRC`]),
      el("li", {}, [`δ from XGBoost: ${result.components.delta_xgb.toFixed(2)} HRC`]),
      el("li", {}, [`δ from BayesianRidge: ${result.components.delta_bayes.toFixed(2)} HRC`]),
    ]),
  ]);
  container.append(breakdown);
}

function showError(container: HTMLElement, message: string): void {
  container.innerHTML = "";
  container.append(el("div", { class: "error" }, [message]));
}

async function waitForBackend(): Promise<void> {
  // In desktop (Tauri) mode __JOMINY_API__ is set. Wait for the backend-ready
  // event dispatched by main.rs after the sidecar port becomes reachable.
  // In web/dev mode the variable is absent and the backend is already running.
  if (!(window as unknown as { __JOMINY_API__?: string }).__JOMINY_API__) return;
  await new Promise<void>((resolve) => {
    window.addEventListener("backend-ready", () => resolve(), { once: true });
  });
}

async function fetchMetadata(): Promise<Metadata> {
  const res = await fetch(`${BASE}/api/metadata`);
  if (!res.ok) throw new Error(`metadata: ${res.status}`);
  return (await res.json()) as Metadata;
}

async function postPrediction(req: CompositionRequest): Promise<PredictionResponse> {
  const res = await fetch(`${BASE}/api/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`prediction failed (${res.status}): ${detail}`);
  }
  return (await res.json()) as PredictionResponse;
}

async function main(): Promise<void> {
  const root = document.querySelector<HTMLDivElement>("#app");
  if (!root) return;

  const overlay = document.querySelector<HTMLDivElement>("#loading-overlay");
  await waitForBackend();
  overlay?.remove();

  let metadata: Metadata;
  try {
    metadata = await fetchMetadata();
  } catch (err) {
    root.innerHTML = `<div class="error">Cannot reach API: ${(err as Error).message}</div>`;
    return;
  }

  const stats = metadata.feature_stats;

  const requiredGrid = el("div", { class: "grid" });
  for (const k of REQUIRED_ELEMENTS) {
    requiredGrid.append(buildField(k, false, SAMPLE_COMPOSITION[k], stats[k]));
  }

  const optionalGrid = el("div", { class: "grid" });
  for (const k of OPTIONAL_ELEMENTS) {
    optionalGrid.append(buildField(k, true, SAMPLE_COMPOSITION[k], stats[k]));
  }

  const form = el("form", { id: "composition-form" }, [
    el("h2", {}, ["Required elements (wt%)"]),
    requiredGrid,
    el("h2", {}, ["Optional trace elements (wt%)"]),
    el("p", { class: "muted" }, [
      "Leave blank if not measured — the model will treat them as missing.",
    ]) as HTMLElement,
    optionalGrid,
    el("div", { class: "actions" }, [
      el("button", { type: "submit" }, ["Predict J9 / J15"]),
      el("button", { type: "button", class: "secondary", id: "btn-sample" }, ["Reset to sample"]),
    ]) as HTMLElement,
  ]);
  (form.querySelector(".muted") as HTMLElement).style.color = "var(--muted)";
  (form.querySelector(".muted") as HTMLElement).style.fontSize = "0.85rem";

  const formCard = el("div", { class: "card" }, [form]);
  const resultCard = el("div", { class: "card" }, [
    el("h2", {}, ["Prediction"]),
    el("p", { class: "muted" }, ["Submit the form to see predicted hardness."]) as HTMLElement,
  ]);
  (resultCard.querySelector(".muted") as HTMLElement).style.color = "var(--muted)";

  const resultBody = el("div", { id: "result-body" });
  resultCard.append(resultBody);

  root.append(
    el("header", {}, [
      el("h1", {}, ["Jominy Hardenability Predictor"]),
      el("p", {}, [
        `Predict Rockwell J9 / J15 from steel composition. ` +
          `Validated on ${metadata.j9_train_rows} specimens, expected MAE ` +
          `±${metadata.expected_metrics.J9.mae.toFixed(2)} HRC.`,
      ]),
    ]),
    formCard,
    resultCard,
    el("footer", {}, [
      "Model: 0.70·XGBoost + 0.30·PLS for J9, 0.60·XGBoost + 0.40·BayesianRidge for δ. ",
      el("a", { href: "/docs", target: "_blank" }, ["API docs"]),
    ]) as HTMLElement,
  );

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const button = form.querySelector("button[type='submit']") as HTMLButtonElement;
    button.disabled = true;
    try {
      const req = readForm(form as HTMLFormElement);
      const result = await postPrediction(req);
      renderResult(resultBody, result);
    } catch (err) {
      showError(resultBody, (err as Error).message);
    } finally {
      button.disabled = false;
    }
  });

  (form.querySelector("#btn-sample") as HTMLButtonElement).addEventListener("click", () => {
    fillForm(form as HTMLFormElement, SAMPLE_COMPOSITION);
  });
}

void main();
