// Shared building blocks for the app shell — element lists, language store,
// translations, tiny DOM helpers, API base.

import type { CompositionRequest, ElementKey, FeatureStat, Metadata, PredictionResponse } from "./types";
import { BASE } from "./api";

// Re-export ElementKey so existing consumers (`import { ElementKey } from "./shared"`)
// keep working without churn. types.ts is the authoritative source.
export type { ElementKey };

export const REQUIRED_ELEMENTS = [
  "C", "Si", "Mn", "P", "S", "Cu", "Ni", "Cr",
] as const satisfies readonly ElementKey[];

export const OPTIONAL_ELEMENTS = [
  "V", "Ti", "W", "Al", "B",
] as const satisfies readonly ElementKey[];

export const ALL_ELEMENTS = [...REQUIRED_ELEMENTS, ...OPTIONAL_ELEMENTS] as const;

export type RequiredKey = (typeof REQUIRED_ELEMENTS)[number];
export type OptionalKey = (typeof OPTIONAL_ELEMENTS)[number];
export type Language = "zh" | "en";

const LANGUAGE_STORAGE_KEY = "jominy-language";
const DEFAULT_LANGUAGE: Language = "zh";

export const SAMPLE_COMPOSITION: CompositionRequest = {
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

export const TRANSLATIONS = {
  zh: {
    htmlLang: "zh-CN",
    title: "Jominy 淬透性预测器",
    subtitleProject: "JMNY-01",
    subtitleRev: "REV.B",
    subtitleSpec: "ROCKWELL HRC · 9/15 mm",
    subtitleStats: (rows: number, mae: number) =>
      `${rows} 样本 · J9 MAE ±${mae.toFixed(2)} HRC`,
    switchLabel: "EN",
    switchAria: "Switch language to English",
    tabSingle: "单样本",
    tabBatch: "批量分析",
    requiredTitle: "必填元素",
    requiredStamp: "REQUIRED · wt%",
    optionalTitle: "可选微量元素",
    optionalStamp: "OPTIONAL · wt%",
    optionalHint: "未检测的元素请留空，模型会按缺失值处理。",
    optionalTag: "OPT",
    typicalRange: (low: number, high: number) => `${low.toFixed(3)}–${high.toFixed(3)}`,
    predictButton: "运行预测",
    resetButton: "恢复示例",
    predictionTitle: "硬度预测",
    predictionStamp: "OUTPUT · HRC",
    predictionHint: "提交化学成分以查看预测硬度。",
    expectedMae: (j9: number, delta: number) =>
      `预期 MAE：J9 ±${j9.toFixed(2)} HRC，δ ±${delta.toFixed(2)} HRC`,
    warningTitle: "输入超出训练范围",
    breakdownTitle: "分模型预测",
    j9Xgb: "XGBoost J9",
    j9Pls: "PLS J9",
    deltaXgb: "XGBoost δ",
    deltaBayes: "BayesianRidge δ",
    apiError: "无法连接 API",
    metadataError: (status: number) => `元数据请求失败：${status}`,
    predictionError: (status: number, detail: string) => `预测请求失败（${status}）：${detail}`,
    requiredError: (key: string) => `${key} 为必填项`,
    footerLeft: "J9 = 0.70·XGB + 0.30·PLS · δ = 0.60·XGB + 0.40·BayesRidge",
    apiDocs: "API 文档",
    // Batch
    batchHeading: "批量预测",
    batchHeadingStamp: "BATCH · XLS",
    dropTitle: "点击上传 XLS/XLSX",
    dropHint: "",
    dropSchema: (cols: string) => `预期表头：${cols}`,
    dropPick: "浏览文件",
    batchFile: (name: string) => name,
    batchExport: "导出 CSV",
    batchClear: "清除",
    batchPredicting: "正在预测…",
    batchOk: "OK",
    batchInsuf: "INSUF",
    batchSkipped: "SKIPPED",
    batchDedup: "去重",
    batchErr: "ERR",
    batchUploadError: (msg: string) => `上传失败：${msg}`,
    colId: "炉号 / ID",
    colGrade: "钢号",
    colJ9: "J9",
    colJ15: "J15",
    distance9: "9 mm",
    distance15: "15 mm",
    elements: {
      C: "碳", Si: "硅", Mn: "锰", P: "磷", S: "硫", Cu: "铜", Ni: "镍", Cr: "铬",
      V: "钒", Ti: "钛", W: "钨", Al: "铝", B: "硼",
    },
  },
  en: {
    htmlLang: "en",
    title: "Jominy Hardenability Predictor",
    subtitleProject: "JMNY-01",
    subtitleRev: "REV.B",
    subtitleSpec: "ROCKWELL HRC · 9/15 mm",
    subtitleStats: (rows: number, mae: number) =>
      `${rows} specimens · J9 MAE ±${mae.toFixed(2)} HRC`,
    switchLabel: "中文",
    switchAria: "切换到中文",
    tabSingle: "Single",
    tabBatch: "Batch",
    requiredTitle: "Required elements",
    requiredStamp: "REQUIRED · wt%",
    optionalTitle: "Optional trace elements",
    optionalStamp: "OPTIONAL · wt%",
    optionalHint: "Leave blank if not measured — the model treats them as missing.",
    optionalTag: "OPT",
    typicalRange: (low: number, high: number) => `${low.toFixed(3)}–${high.toFixed(3)}`,
    predictButton: "Run prediction",
    resetButton: "Reset to sample",
    predictionTitle: "Hardness prediction",
    predictionStamp: "OUTPUT · HRC",
    predictionHint: "Submit composition to see predicted hardness.",
    expectedMae: (j9: number, delta: number) =>
      `Expected MAE: J9 ±${j9.toFixed(2)} HRC, δ ±${delta.toFixed(2)} HRC`,
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
    footerLeft: "J9 = 0.70·XGB + 0.30·PLS · δ = 0.60·XGB + 0.40·BayesRidge",
    apiDocs: "API docs",
    // Batch
    batchHeading: "Batch prediction",
    batchHeadingStamp: "BATCH · XLS",
    dropTitle: "Click to upload XLS/XLSX",
    dropHint: "",
    dropSchema: (cols: string) => `Expected columns: ${cols}`,
    dropPick: "Browse files",
    batchFile: (name: string) => name,
    batchExport: "Export CSV",
    batchClear: "Clear",
    batchPredicting: "Predicting…",
    batchOk: "OK",
    batchInsuf: "INSUF",
    batchSkipped: "SKIPPED",
    batchDedup: "DEDUP",
    batchErr: "ERR",
    batchUploadError: (msg: string) => `Upload failed: ${msg}`,
    colId: "Heat / ID",
    colGrade: "Grade",
    colJ9: "J9",
    colJ15: "J15",
    distance9: "9 mm",
    distance15: "15 mm",
    elements: {
      C: "Carbon", Si: "Silicon", Mn: "Manganese", P: "Phosphorus", S: "Sulfur",
      Cu: "Copper", Ni: "Nickel", Cr: "Chromium", V: "Vanadium", Ti: "Titanium",
      W: "Tungsten", Al: "Aluminum", B: "Boron",
    },
  },
} as const;

export type TextBundle = typeof TRANSLATIONS[Language];

// ----- Language persistence -----

export function getInitialLanguage(): Language {
  const saved = window.localStorage.getItem(LANGUAGE_STORAGE_KEY);
  return saved === "en" || saved === "zh" ? saved : DEFAULT_LANGUAGE;
}

export function saveLanguage(language: Language): void {
  window.localStorage.setItem(LANGUAGE_STORAGE_KEY, language);
}

// ----- Tiny DOM helper -----

export function el<K extends keyof HTMLElementTagNameMap>(
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

// ----- Range / warning helpers -----

export function fmtRange(stat: FeatureStat | undefined, language: Language): string {
  if (!stat || stat.p01 === undefined || stat.p99 === undefined) return "";
  return TRANSLATIONS[language].typicalRange(stat.p01, stat.p99);
}

export function localizedWarning(warning: string, language: Language): string {
  if (language === "en") return warning;
  const match = warning.match(
    /^([A-Za-z]+)=([^ ]+) is outside the typical training range \[([^,]+), ([^\]]+)\]/,
  );
  if (!match) return warning;
  const [, element, value, low, high] = match;
  return `${element}=${value} 超出常见训练范围 [${low}, ${high}]，属于外推，预测可能不可靠。`;
}

/** True if `value` is outside the typical [p01, p99] range for this element. */
export function isOutOfRange(
  value: number | null,
  stat: FeatureStat | undefined,
): boolean {
  if (value === null || !stat || stat.p01 === undefined || stat.p99 === undefined) return false;
  return value < stat.p01 || value > stat.p99;
}

// ----- API -----

export async function waitForBackend(): Promise<void> {
  if (!(window as unknown as { __JOMINY_API__?: string }).__JOMINY_API__) return;
  await new Promise<void>((resolve) => {
    window.addEventListener("backend-ready", () => resolve(), { once: true });
  });
}

export async function fetchMetadata(language: Language): Promise<Metadata> {
  const res = await fetch(`${BASE}/api/metadata`);
  if (!res.ok) throw new Error(TRANSLATIONS[language].metadataError(res.status));
  return (await res.json()) as Metadata;
}

export async function postPrediction(
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
