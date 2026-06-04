// Single-sample page — restyled into the Forge & Quench system.

import type { CompositionRequest, FeatureStat, Metadata, PredictionResponse } from "./types";
import {
  ALL_ELEMENTS,
  ElementKey,
  Language,
  OPTIONAL_ELEMENTS,
  REQUIRED_ELEMENTS,
  RequiredKey,
  SAMPLE_COMPOSITION,
  TRANSLATIONS,
  el,
  fmtRange,
  localizedWarning,
  postPrediction,
} from "./shared";

type FormValues = Record<ElementKey, string>;

export interface SingleState {
  formValues: FormValues;
  lastResult: PredictionResponse | null;
}

export function createSingleState(): SingleState {
  return { formValues: sampleFormValues(), lastResult: null };
}

function clearChildren(node: Node): void {
  while (node.firstChild) node.removeChild(node.firstChild);
}

function sampleFormValues(): FormValues {
  return Object.fromEntries(
    ALL_ELEMENTS.map((key) => {
      const value = SAMPLE_COMPOSITION[key];
      return [key, value === null ? "" : String(value)];
    }),
  ) as FormValues;
}

function buildField(
  key: ElementKey,
  optional: boolean,
  initial: string,
  stat: FeatureStat | undefined,
  language: Language,
): HTMLLabelElement {
  const text = TRANSLATIONS[language];
  const input = el("input", {
    type: "number",
    name: key,
    step: "0.001",
    min: "0",
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
      el("span", { class: "field__range" }, [fmtRange(stat, language)]),
    ],
  );
}

function readFormValues(form: HTMLFormElement): FormValues {
  return Object.fromEntries(
    ALL_ELEMENTS.map((key) => {
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
  const required = (key: RequiredKey) => {
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
    V: get("V"),
    Ti: get("Ti"),
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
  clearChildren(root);

  // ----- Form (paper) -----
  const requiredGrid = el("div", { class: "element-grid" });
  for (const k of REQUIRED_ELEMENTS) {
    requiredGrid.append(buildField(k, false, state.formValues[k], stats[k], language));
  }
  const optionalGrid = el("div", { class: "element-grid" });
  for (const k of OPTIONAL_ELEMENTS) {
    optionalGrid.append(buildField(k, true, state.formValues[k], stats[k], language));
  }

  const form = el("form", { id: "composition-form" }, [
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
  form.addEventListener("input", () => {
    state.formValues = readFormValues(form as HTMLFormElement);
    onChange();
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
    state.formValues = sampleFormValues();
    state.lastResult = null;
    onChange();
  });
}
