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
  Ti: number | null;
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
}
