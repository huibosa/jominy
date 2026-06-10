export interface CompositionRequest {
  C: number;
  Si: number;
  Mn: number;
  P: number;
  S: number;
  Cu: number;
  Ni: number;
  Cr: number;
  V: number | null;
  Ti: number;
  W: number | null;
  Al: number | null;
  B: number | null;
}

export interface PredictionResponse {
  J9: number;
  J15: number;
  delta: number;
  components: {
    j9_xgb: number;
    j9_pls: number;
    delta_xgb: number;
    delta_bayes: number;
  };
  warnings: string[];
  expected_mae: { J9: number; delta: number };
}

export interface FeatureStat {
  min: number;
  max: number;
  median: number;
  p01?: number;
  p99?: number;
  is_flag: boolean;
}

export interface Metadata {
  features: string[];
  feature_stats: Record<string, FeatureStat>;
  expected_metrics: Record<string, { mae: number; rmse: number; r2: number }>;
  j9_train_rows: number;
  delta_train_rows: number;
  element_fields: string[];
  /** Per-standard hard limits [min, max] for each element input, keyed by StandardKey. */
  standard_bounds: Record<StandardKey, Record<string, [number, number]>>;
}

/** Identifier for the two GB standards with distinct chemistry bounds for 20CrMnTi. */
export type StandardKey = "gbt3077" | "gbt5216";

// Authoritative source for ElementKey — shared.ts imports and re-exports this.
export type ElementKey =
  | "C" | "Si" | "Mn" | "P" | "S" | "Cu" | "Ni" | "Cr"
  | "V" | "Ti" | "W" | "Al" | "B";

export type BatchStatus = "ok" | "insufficient" | "error" | "std_fill";

export interface BatchSample {
  id: string;
  id_synthesized: boolean;
  grade: string | null;
  // Record (not Partial<CompositionRequest>) so cells are `number | null`
  // never `undefined` under strict TS.
  composition: Record<ElementKey, number | null>;
  missing_required: string[];
  /** Elements whose values were filled from the GB national standard (std_fill rows only). */
  filled_elements: string[];
  status: BatchStatus;
  prediction: PredictionResponse | null;
  error: string | null;
}

export interface BatchSummary {
  total_rows: number;
  deduped: number;
  skipped_empty: number;
  predicted: number;
  insufficient: number;
  std_fill: number;
  errored: number;
}

export interface BatchResponse {
  filename: string;
  summary: BatchSummary;
  samples: BatchSample[];
}
