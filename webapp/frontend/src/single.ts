// Single-sample page — restyled into the Forge & Quench system.

import type { CompositionRequest, FeatureStat, Metadata, PredictionResponse } from "./types";
import {
  ElementKey,
  FormValues,
  Language,
  StandardKey,
  TRANSLATIONS,
  el,
  fmtRange,
  getInitialStandard,
  localizedWarning,
  postPrediction,
  sampleForStandard,
  saveStandard,
} from "./shared";

// Elements with no GB 20CrMnTi composition spec — show training p01/p99 range
// instead of the wide safety-cap bounds for these four.
const ELEMENTS_NOT_IN_GB_SPEC = new Set<ElementKey>(["V", "W", "Al", "B"]);

/**
 * Format a [min, max] bounds pair as a GB spec range string.
 *   Both bounds > 0  → "min–max"  (e.g. "0.170–0.230")
 *   min == 0         → "≤ max"    (e.g. "≤ 0.035")
 * Decimal places: 4 for values whose max < 0.01 (B, P-level), 3 otherwise.
 */
function fmtBounds([lo, hi]: [number, number]): string {
  const dp = hi < 0.01 ? 4 : 3;
  const fmt = (n: number) => n.toFixed(dp);
  return lo === 0 ? `≤ ${fmt(hi)}` : `${fmt(lo)}–${fmt(hi)}`;
}

// Single-page required/optional groupings — Ti is required here (single page only).
// These are intentionally separate from the shared REQUIRED_ELEMENTS/OPTIONAL_ELEMENTS
// to avoid affecting batch page column order and CSV export.
const SINGLE_REQUIRED = [
  "C", "Si", "Mn", "P", "S", "Cu", "Ni", "Cr", "Ti",
] as const satisfies readonly ElementKey[];

const SINGLE_OPTIONAL = [
  "V", "W", "Al", "B",
] as const satisfies readonly ElementKey[];

type SingleRequiredKey = (typeof SINGLE_REQUIRED)[number];

export interface SingleState {
  standard: StandardKey;
  formValues: FormValues;
  lastResult: PredictionResponse | null;
}

export function createSingleState(): SingleState {
  const standard = getInitialStandard();
  return { standard, formValues: sampleForStandard(standard), lastResult: null };
}

function clearChildren(node: Node): void {
  while (node.firstChild) node.removeChild(node.firstChild);
}

/** Derive the input step from the element's GB-standard upper bound.
 *  Elements whose max sits below 0.01 (e.g. B: max=0.005) need 4-decimal-place
 *  granularity; everything else is fine at 3 decimal places. */
function elementStep(inputMax: number): string {
  return inputMax < 0.01 ? "0.0001" : "0.001";
}

function buildField(
  key: ElementKey,
  optional: boolean,
  initial: string,
  stat: FeatureStat | undefined,
  bounds: [number, number],
  language: Language,
): HTMLLabelElement {
  const text = TRANSLATIONS[language];
  const [inputMin, inputMax] = bounds;
  const input = el("input", {
    type: "number",
    name: key,
    step: elementStep(inputMax),
    min: String(inputMin),
    max: String(inputMax),
    placeholder: optional ? "—" : "",
    "data-key": key,
  });
  input.value = initial;
  if (!optional) input.required = true;

  const symbol = el("span", { class: "field__symbol" }, [key]);
  if (optional) symbol.append(el("span", { class: "opt-tag" }, [text.optionalTag]));

  return el(
    "label",
    { class: optional ? "field field--optional" : "field" },
    [
      el("div", { class: "field__label" }, [
        symbol,
        el("span", { class: "field__name" }, [text.elements[key]]),
      ]),
      input,
      el("span", { class: "field__range" }, [
        ELEMENTS_NOT_IN_GB_SPEC.has(key) ? fmtRange(stat, language) : fmtBounds(bounds),
      ]),
    ],
  );
}

function readFormValues(form: HTMLFormElement): FormValues {
  return Object.fromEntries(
    ([...SINGLE_REQUIRED, ...SINGLE_OPTIONAL] as ElementKey[]).map((key) => {
      const input = form.querySelector<HTMLInputElement>(`[data-key="${key}"]`);
      return [key, input?.value ?? ""];
    }),
  ) as FormValues;
}

function parseComposition(form: HTMLFormElement, language: Language): CompositionRequest {
  const text = TRANSLATIONS[language];
  const get = (key: ElementKey) => {
    const input = form.querySelector<HTMLInputElement>(`[data-key="${key}"]`);
    if (!input) return null;
    const val = input.value.trim();
    if (val === "") return null;
    const n = Number(val);
    return Number.isFinite(n) ? n : null;
  };
  // required() covers the single-page required set: C Si Mn P S Cu Ni Cr Ti.
  const required = (key: SingleRequiredKey) => {
    const value = get(key);
    if (value === null) throw new Error(text.requiredError(key));
    return value;
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
    Ti: required("Ti"),
    V: get("V"),
    W: get("W"),
    Al: get("Al"),
    B: get("B"),
  };
}

function readout(
  label: string,
  value: string,
  unit: string,
  variant?: "secondary" | "placeholder",
): HTMLElement {
  const cls = ["readout"];
  if (variant === "secondary") cls.push("readout--secondary");
  if (variant === "placeholder") cls.push("readout--placeholder");
  return el("div", { class: cls.join(" ") }, [
    el("div", { class: "readout__label" }, [label]),
    el("div", {}, [
      el("span", { class: "readout__value" }, [value]),
      el("span", { class: "readout__unit" }, [unit]),
    ]),
  ]);
}

function renderResultBody(
  container: HTMLElement,
  result: PredictionResponse | null,
  language: Language,
): void {
  const text = TRANSLATIONS[language];
  clearChildren(container);

  if (!result) {
    const placeholder = el("div", { class: "readouts" }, [
      readout("J9", "—.——", "HRC", "placeholder"),
      readout("J15", "—.——", "HRC", "placeholder"),
      readout("J9 − J15", "—.——", "HRC", "placeholder"),
    ]);
    container.append(placeholder);
    container.append(el("p", { class: "result-meta" }, [text.predictionHint]));
    return;
  }

  const grid = el("div", { class: "readouts" }, [
    readout("J9", result.J9.toFixed(2), "HRC"),
    readout("J15", result.J15.toFixed(2), "HRC"),
    readout("Δ J9−J15", result.delta.toFixed(2), "HRC", "secondary"),
  ]);
  container.append(grid);

  container.append(
    el("p", { class: "result-meta" }, [
      text.expectedMae(result.expected_mae.J9, result.expected_mae.delta),
    ]),
  );

  if (result.warnings.length > 0) {
    const warn = el("div", { class: "callout" }, [
      el("strong", {}, [text.warningTitle]),
      el(
        "ul",
        {},
        result.warnings.map((w) => el("li", {}, [localizedWarning(w, language)])),
      ),
    ]);
    container.append(warn);
  }

  container.append(
    el("details", { class: "breakdown" }, [
      el("summary", {}, [text.breakdownTitle]),
      el("ul", {}, [
        breakdownRow(text.j9Xgb, `${result.components.j9_xgb.toFixed(2)} HRC`),
        breakdownRow(text.j9Pls, `${result.components.j9_pls.toFixed(2)} HRC`),
        breakdownRow(text.deltaXgb, `${result.components.delta_xgb.toFixed(2)} HRC`),
        breakdownRow(text.deltaBayes, `${result.components.delta_bayes.toFixed(2)} HRC`),
      ]),
    ]),
  );
}

function breakdownRow(label: string, value: string): HTMLElement {
  return el("li", {}, [
    el("span", {}, [label]),
    el("span", {}, [value]),
  ]);
}

function renderError(container: HTMLElement, message: string): void {
  clearChildren(container);
  container.append(el("div", { class: "callout callout--error" }, [message]));
}

export function renderSingle(
  root: HTMLElement,
  metadata: Metadata,
  state: SingleState,
  language: Language,
  onChange: () => void,
): void {
  const text = TRANSLATIONS[language];
  const stats = metadata.feature_stats;
  const bounds = metadata.standard_bounds[state.standard];
  clearChildren(root);

  // ----- Standard toggle (type="button" — must not submit the form) -----
  const btn3077 = el("button", {
    type: "button",
    class: state.standard === "gbt3077" ? "standard-toggle__btn standard-toggle__btn--active" : "standard-toggle__btn",
    "data-standard": "gbt3077",
  }, [text.standard3077]);
  const btn5216 = el("button", {
    type: "button",
    class: state.standard === "gbt5216" ? "standard-toggle__btn standard-toggle__btn--active" : "standard-toggle__btn",
    "data-standard": "gbt5216",
  }, [text.standard5216]);

  const standardToggle = el("div", { class: "standard-toggle" }, [
    el("span", { class: "standard-toggle__label" }, [text.standardLabel]),
    btn3077,
    btn5216,
  ]);

  // ----- Form (paper) -----
  const requiredGrid = el("div", { class: "element-grid" });
  for (const k of SINGLE_REQUIRED) {
    requiredGrid.append(buildField(k, false, state.formValues[k], stats[k], bounds[k], language));
  }
  const optionalGrid = el("div", { class: "element-grid" });
  for (const k of SINGLE_OPTIONAL) {
    optionalGrid.append(buildField(k, true, state.formValues[k], stats[k], bounds[k], language));
  }

  const form = el("form", { id: "composition-form" }, [
    standardToggle,
    el("div", { class: "section" }, [
      el("div", { class: "section__heading" }, [
        el("h2", {}, [text.requiredTitle]),
        el("span", { class: "stamp" }, [text.requiredStamp]),
      ]),
      requiredGrid,
    ]),
    el("div", { class: "section" }, [
      el("div", { class: "section__heading" }, [
        el("h2", {}, [text.optionalTitle]),
        el("span", { class: "stamp" }, [text.optionalStamp]),
      ]),
      el("p", { class: "section__hint" }, [text.optionalHint]),
      optionalGrid,
    ]),
    el("div", { class: "actions" }, [
      el("button", { type: "submit", class: "btn" }, [
        el("span", {}, [text.predictButton]),
        el("span", { class: "btn__chev" }, ["▶"]),
      ]),
      el("button", { type: "button", class: "btn btn--ghost", id: "btn-sample" }, [
        text.resetButton,
      ]),
    ]),
  ]);

  const formCard = el("section", { class: "paper fade-in" }, [form]);

  // ----- Result slab -----
  const resultBody = el("div", { id: "result-body" });
  renderResultBody(resultBody, state.lastResult, language);

  const resultSlab = el("section", { class: "slab fade-in" }, [
    el("div", { class: "slab__header" }, [
      el("h2", {}, [text.predictionTitle]),
      el("span", { class: "slab__stamp" }, [text.predictionStamp]),
    ]),
    resultBody,
  ]);

  root.append(el("div", { class: "single-grid" }, [formCard, resultSlab]));

  // ----- Wiring -----
  // Standard toggle — switch bounds and re-render; keep field values (browser
  // :invalid highlight will flag anything now out-of-range).
  for (const btn of [btn3077, btn5216]) {
    btn.addEventListener("click", () => {
      state.formValues = readFormValues(form as HTMLFormElement);
      state.standard = btn.dataset["standard"] as StandardKey;
      saveStandard(state.standard);
      onChange();
    });
  }

  // Persist values into state on every keystroke, but DO NOT trigger a re-render —
  // re-rendering would clobber the focused input and make typing impossible.
  form.addEventListener("input", () => {
    state.formValues = readFormValues(form as HTMLFormElement);
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    state.formValues = readFormValues(form as HTMLFormElement);
    const button = form.querySelector("button[type='submit']") as HTMLButtonElement;
    button.disabled = true;
    try {
      const req = parseComposition(form as HTMLFormElement, language);
      state.lastResult = await postPrediction(req, language);
      renderResultBody(resultBody, state.lastResult, language);
    } catch (err) {
      renderError(resultBody, (err as Error).message);
    } finally {
      button.disabled = false;
    }
  });

  (form.querySelector("#btn-sample") as HTMLButtonElement).addEventListener("click", () => {
    state.formValues = sampleForStandard(state.standard);
    state.lastResult = null;
    onChange();
  });
}
