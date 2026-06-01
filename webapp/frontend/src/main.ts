import "./style.css";
import type { CompositionRequest, Metadata, PredictionResponse } from "./types";
import { BASE } from "./api";

const REQUIRED_ELEMENTS = ["C", "Si", "Mn", "P", "S", "Cu", "Ni", "Cr"] as const;
const OPTIONAL_ELEMENTS = ["V", "Ti", "W", "Al", "B"] as const;
type ElementKey = (typeof REQUIRED_ELEMENTS)[number] | (typeof OPTIONAL_ELEMENTS)[number];
type Language = "zh" | "en";
type FormValues = Record<ElementKey, string>;

const LANGUAGE_STORAGE_KEY = "jominy-language";
const DEFAULT_LANGUAGE: Language = "zh";

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

const TRANSLATIONS = {
  zh: {
    htmlLang: "zh-CN",
    title: "Jominy 淬透性预测器",
    subtitle: (rows: number, mae: number) =>
      `根据钢材化学成分预测洛氏硬度 J9 / J15。模型使用 ${rows} 个样本验证，J9 预期 MAE ±${mae.toFixed(2)} HRC。`,
    switchLabel: "English",
    switchAria: "Switch language to English",
    requiredTitle: "必填元素（wt%）",
    optionalTitle: "可选微量元素（wt%）",
    optionalHint: "未检测的元素请留空，模型会按缺失值处理。",
    optionalTag: "（可选）",
    typicalRange: (low: number, high: number) => `常见范围 ${low.toFixed(3)}–${high.toFixed(3)} wt%`,
    predictButton: "预测 J9 / J15",
    resetButton: "恢复示例",
    predictionTitle: "预测结果",
    predictionHint: "提交表单后显示预测硬度。",
    expectedMae: (j9: number, delta: number) =>
      `交叉验证预期 MAE：J9 ±${j9.toFixed(2)} HRC，δ ±${delta.toFixed(2)} HRC。`,
    warningTitle: "输入超出训练范围",
    breakdownTitle: "分模型预测",
    j9Xgb: "XGBoost 预测 J9",
    j9Pls: "PLS 预测 J9",
    deltaXgb: "XGBoost 预测 δ",
    deltaBayes: "BayesianRidge 预测 δ",
    apiError: "无法连接 API",
    metadataError: (status: number) => `元数据请求失败：${status}`,
    predictionError: (status: number, detail: string) => `预测请求失败（${status}）：${detail}`,
    requiredError: (key: string) => `${key} 为必填项`,
    footerPrefix: "模型：J9 使用 0.70·XGBoost + 0.30·PLS，δ 使用 0.60·XGBoost + 0.40·BayesianRidge。",
    apiDocs: "API 文档",
    elements: {
      C: "碳",
      Si: "硅",
      Mn: "锰",
      P: "磷",
      S: "硫",
      Cu: "铜",
      Ni: "镍",
      Cr: "铬",
      V: "钒",
      Ti: "钛",
      W: "钨",
      Al: "铝",
      B: "硼",
    },
  },
  en: {
    htmlLang: "en",
    title: "Jominy Hardenability Predictor",
    subtitle: (rows: number, mae: number) =>
      `Predict Rockwell J9 / J15 from steel composition. Validated on ${rows} specimens, expected MAE ±${mae.toFixed(2)} HRC.`,
    switchLabel: "中文",
    switchAria: "切换到中文",
    requiredTitle: "Required elements (wt%)",
    optionalTitle: "Optional trace elements (wt%)",
    optionalHint: "Leave blank if not measured — the model will treat them as missing.",
    optionalTag: " (opt)",
    typicalRange: (low: number, high: number) => `typical ${low.toFixed(3)}–${high.toFixed(3)} wt%`,
    predictButton: "Predict J9 / J15",
    resetButton: "Reset to sample",
    predictionTitle: "Prediction",
    predictionHint: "Submit the form to see predicted hardness.",
    expectedMae: (j9: number, delta: number) =>
      `Expected MAE from cross-validation: ±${j9.toFixed(2)} HRC on J9, ±${delta.toFixed(2)} HRC on δ.`,
    warningTitle: "Input outside training range",
    breakdownTitle: "Per-component predictions",
    j9Xgb: "J9 from XGBoost",
    j9Pls: "J9 from PLS",
    deltaXgb: "δ from XGBoost",
    deltaBayes: "δ from BayesianRidge",
    apiError: "Cannot reach API",
    metadataError: (status: number) => `metadata: ${status}`,
    predictionError: (status: number, detail: string) => `prediction failed (${status}): ${detail}`,
    requiredError: (key: string) => `${key} is required`,
    footerPrefix: "Model: 0.70·XGBoost + 0.30·PLS for J9, 0.60·XGBoost + 0.40·BayesianRidge for δ. ",
    apiDocs: "API docs",
    elements: {
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
    },
  },
} as const;

function getInitialLanguage(): Language {
  const saved = window.localStorage.getItem(LANGUAGE_STORAGE_KEY);
  return saved === "en" || saved === "zh" ? saved : DEFAULT_LANGUAGE;
}

function saveLanguage(language: Language): void {
  window.localStorage.setItem(LANGUAGE_STORAGE_KEY, language);
}

function sampleFormValues(): FormValues {
  return Object.fromEntries(
    [...REQUIRED_ELEMENTS, ...OPTIONAL_ELEMENTS].map((key) => {
      const value = SAMPLE_COMPOSITION[key];
      return [key, value === null ? "" : String(value)];
    }),
  ) as FormValues;
}

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

function fmtRange(
  stat: { p01?: number; p99?: number } | undefined,
  language: Language,
): string {
  if (!stat || stat.p01 === undefined || stat.p99 === undefined) return "";
  return TRANSLATIONS[language].typicalRange(stat.p01, stat.p99);
}

function buildField(
  key: ElementKey,
  optional: boolean,
  initial: string,
  stat: { p01?: number; p99?: number } | undefined,
  language: Language,
): HTMLLabelElement {
  const text = TRANSLATIONS[language];
  const input = el("input", {
    type: "number",
    name: key,
    step: "0.001",
    min: "0",
    "data-key": key,
  });
  input.value = initial;
  if (!optional) input.required = true;

  return el("label", { class: optional ? "optional" : "" }, [
    el("span", { class: "name" }, [
      `${key} — ${text.elements[key]}${optional ? text.optionalTag : ""}`,
    ]),
    input,
    el("span", { class: "range" }, [fmtRange(stat, language)]),
  ]);
}

function readFormValues(form: HTMLFormElement): FormValues {
  return Object.fromEntries(
    [...REQUIRED_ELEMENTS, ...OPTIONAL_ELEMENTS].map((key) => {
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
  const required = (key: (typeof REQUIRED_ELEMENTS)[number]) => {
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

function localizedWarning(warning: string, language: Language): string {
  if (language === "en") return warning;
  const match = warning.match(
    /^([A-Za-z]+)=([^ ]+) is outside the typical training range \[([^,]+), ([^\]]+)\]/,
  );
  if (!match) return warning;
  const [, element, value, low, high] = match;
  return `${element}=${value} 超出常见训练范围 [${low}, ${high}]，属于外推，预测可能不可靠。`;
}

function renderResult(container: HTMLElement, result: PredictionResponse, language: Language): void {
  const text = TRANSLATIONS[language];
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

  container.append(
    el("p", { class: "muted result-note" }, [
      text.expectedMae(result.expected_mae.J9, result.expected_mae.delta),
    ]),
  );

  if (result.warnings.length > 0) {
    const warn = el("div", { class: "warning" }, [
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
    el("details", {}, [
      el("summary", {}, [text.breakdownTitle]),
      el("ul", {}, [
        el("li", {}, [`${text.j9Xgb}: ${result.components.j9_xgb.toFixed(2)} HRC`]),
        el("li", {}, [`${text.j9Pls}: ${result.components.j9_pls.toFixed(2)} HRC`]),
        el("li", {}, [`${text.deltaXgb}: ${result.components.delta_xgb.toFixed(2)} HRC`]),
        el("li", {}, [`${text.deltaBayes}: ${result.components.delta_bayes.toFixed(2)} HRC`]),
      ]),
    ]),
  );
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

async function fetchMetadata(language: Language): Promise<Metadata> {
  const res = await fetch(`${BASE}/api/metadata`);
  if (!res.ok) throw new Error(TRANSLATIONS[language].metadataError(res.status));
  return (await res.json()) as Metadata;
}

async function postPrediction(
  req: CompositionRequest,
  language: Language,
): Promise<PredictionResponse> {
  const res = await fetch(`${BASE}/api/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(TRANSLATIONS[language].predictionError(res.status, detail));
  }
  return (await res.json()) as PredictionResponse;
}

async function main(): Promise<void> {
  const root = document.querySelector<HTMLDivElement>("#app");
  if (!root) return;

  let language = getInitialLanguage();
  let formValues = sampleFormValues();
  let lastResult: PredictionResponse | null = null;

  const setDocumentLanguage = () => {
    document.documentElement.lang = TRANSLATIONS[language].htmlLang;
    document.title = TRANSLATIONS[language].title;
  };
  setDocumentLanguage();

  const overlay = document.querySelector<HTMLDivElement>("#loading-overlay");
  await waitForBackend();
  overlay?.remove();

  let metadata: Metadata;
  try {
    metadata = await fetchMetadata(language);
  } catch (err) {
    root.append(
      el("div", { class: "error" }, [
        `${TRANSLATIONS[language].apiError}: ${(err as Error).message}`,
      ]),
    );
    return;
  }

  const render = () => {
    setDocumentLanguage();
    const text = TRANSLATIONS[language];
    const stats = metadata.feature_stats;
    root.innerHTML = "";

    const requiredGrid = el("div", { class: "grid" });
    for (const k of REQUIRED_ELEMENTS) {
      requiredGrid.append(buildField(k, false, formValues[k], stats[k], language));
    }

    const optionalGrid = el("div", { class: "grid" });
    for (const k of OPTIONAL_ELEMENTS) {
      optionalGrid.append(buildField(k, true, formValues[k], stats[k], language));
    }

    const languageButton = el("button", {
      type: "button",
      class: "language-toggle",
      id: "btn-language",
      "aria-label": text.switchAria,
    }, [text.switchLabel]);

    const form = el("form", { id: "composition-form" }, [
      el("h2", {}, [text.requiredTitle]),
      requiredGrid,
      el("h2", {}, [text.optionalTitle]),
      el("p", { class: "muted" }, [text.optionalHint]) as HTMLElement,
      optionalGrid,
      el("div", { class: "actions" }, [
        el("button", { type: "submit" }, [text.predictButton]),
        el("button", { type: "button", class: "secondary", id: "btn-sample" }, [
          text.resetButton,
        ]),
      ]) as HTMLElement,
    ]);

    const formCard = el("div", { class: "card" }, [form]);
    const resultBody = el("div", { id: "result-body" });
    if (lastResult) {
      renderResult(resultBody, lastResult, language);
    } else {
      resultBody.append(el("p", { class: "muted" }, [text.predictionHint]));
    }
    const resultCard = el("div", { class: "card" }, [
      el("h2", {}, [text.predictionTitle]),
      resultBody,
    ]);

    root.append(
      el("header", {}, [
        el("div", { class: "header-top" }, [
          el("h1", {}, [text.title]),
          languageButton,
        ]),
        el("p", {}, [
          text.subtitle(metadata.j9_train_rows, metadata.expected_metrics.J9.mae),
        ]),
      ]),
      formCard,
      resultCard,
      el("footer", {}, [
        text.footerPrefix,
        el("a", { href: "/docs", target: "_blank" }, [text.apiDocs]),
      ]) as HTMLElement,
    );

    languageButton.addEventListener("click", () => {
      formValues = readFormValues(form as HTMLFormElement);
      language = language === "zh" ? "en" : "zh";
      saveLanguage(language);
      render();
    });

    form.addEventListener("input", () => {
      formValues = readFormValues(form as HTMLFormElement);
    });

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      formValues = readFormValues(form as HTMLFormElement);
      const button = form.querySelector("button[type='submit']") as HTMLButtonElement;
      button.disabled = true;
      try {
        const req = parseComposition(form as HTMLFormElement, language);
        lastResult = await postPrediction(req, language);
        render();
      } catch (err) {
        showError(resultBody, (err as Error).message);
      } finally {
        button.disabled = false;
      }
    });

    (form.querySelector("#btn-sample") as HTMLButtonElement).addEventListener("click", () => {
      formValues = sampleFormValues();
      lastResult = null;
      render();
    });
  };

  render();
}

void main();
