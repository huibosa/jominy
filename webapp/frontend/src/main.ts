// Jominy — app shell.
// Owns: title block, tabs, language toggle, metadata fetch.
// Delegates to single.ts and batch.ts for the actual page bodies.

import "./style.css";
import type { Metadata } from "./types";
import {
  Language,
  TRANSLATIONS,
  el,
  fetchMetadata,
  getInitialLanguage,
  saveLanguage,
  waitForBackend,
} from "./shared";
import { BatchState, createBatchState, renderBatch } from "./batch";
import { SingleState, createSingleState, renderSingle } from "./single";

type Tab = "single" | "batch";
const TAB_STORAGE_KEY = "jominy-tab";

function getInitialTab(): Tab {
  const saved = window.localStorage.getItem(TAB_STORAGE_KEY);
  return saved === "batch" ? "batch" : "single";
}

function clearChildren(node: Node): void {
  while (node.firstChild) node.removeChild(node.firstChild);
}

interface AppState {
  language: Language;
  tab: Tab;
  metadata: Metadata;
  single: SingleState;
  batch: BatchState;
}

function buildTitleBlock(state: AppState, onLanguageToggle: () => void): HTMLElement {
  const text = TRANSLATIONS[state.language];

  const langButton = el("button", {
    type: "button",
    class: "lang-toggle",
    "aria-label": text.switchAria,
  }, [text.switchLabel]);
  langButton.addEventListener("click", onLanguageToggle);

  return el("header", { class: "title-block" }, [
    el("h1", { class: "title-block__name" }, [
      text.titleBase,
      el("span", { class: "glyph" }, [text.titleGlyph]),
    ]),
    el("div", { class: "title-block__sub" }, [
      el("div", {}, [text.subtitleSpec]),
      langButton,
    ]),
  ]);
}

function buildTabs(state: AppState, onSwitch: (tab: Tab) => void): HTMLElement {
  const text = TRANSLATIONS[state.language];

  const single = el(
    "button",
    {
      type: "button",
      class: "tab",
      "aria-selected": state.tab === "single" ? "true" : "false",
    },
    [
      el("span", { class: "tab__index" }, ["01"]),
      text.tabSingle,
    ],
  );
  single.addEventListener("click", () => onSwitch("single"));

  const batch = el(
    "button",
    {
      type: "button",
      class: "tab",
      "aria-selected": state.tab === "batch" ? "true" : "false",
    },
    [
      el("span", { class: "tab__index" }, ["02"]),
      text.tabBatch,
    ],
  );
  batch.addEventListener("click", () => onSwitch("batch"));

  return el("nav", { class: "tabs", role: "tablist" }, [single, batch]);
}

async function main(): Promise<void> {
  const root = document.querySelector<HTMLDivElement>("#app");
  if (!root) return;

  const overlay = document.querySelector<HTMLDivElement>("#loading-overlay");
  await waitForBackend();
  overlay?.remove();

  const language = getInitialLanguage();
  document.documentElement.lang = TRANSLATIONS[language].htmlLang;
  document.title = TRANSLATIONS[language].title;

  let metadata: Metadata;
  try {
    metadata = await fetchMetadata(language);
  } catch (err) {
    root.append(
      el("div", { class: "callout callout--error" }, [
        `${TRANSLATIONS[language].apiError}: ${(err as Error).message}`,
      ]),
    );
    return;
  }

  const state: AppState = {
    language,
    tab: getInitialTab(),
    metadata,
    single: createSingleState(),
    batch: createBatchState(),
  };

  const render = () => {
    document.documentElement.lang = TRANSLATIONS[state.language].htmlLang;
    document.title = TRANSLATIONS[state.language].title;
    clearChildren(root);

    const onLanguageToggle = () => {
      state.language = state.language === "zh" ? "en" : "zh";
      saveLanguage(state.language);
      render();
    };
    const onTabSwitch = (tab: Tab) => {
      state.tab = tab;
      window.localStorage.setItem(TAB_STORAGE_KEY, tab);
      render();
    };

    root.append(buildTitleBlock(state, onLanguageToggle));
    root.append(buildTabs(state, onTabSwitch));

    const page = el("main", { class: "page" }, []);
    root.append(page);

    if (state.tab === "single") {
      renderSingle(page, state.metadata, state.single, state.language, render);
    } else {
      renderBatch(page, state.metadata, state.batch, state.language, render);
    }
  };

  render();
}

void main();
