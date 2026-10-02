export type ThreatLabel = "ham" | "phish" | "spam";
export type RiskLevel = "low" | "medium" | "high";
export type Agreement = "agreed" | "disagreed" | "partial" | "unavailable";
export type ModelStatus = "available" | "error" | "unavailable";
export type ModelLifecycleState =
  | "idle"
  | "loading"
  | "ready"
  | "retrying"
  | "error"
  | "unavailable";

export type Probabilities = Record<ThreatLabel, number>;
export type ModelKey = "tfidf_logreg" | "distilbert";

export type ModelPrediction = {
  status: ModelStatus;
  label: ThreatLabel | null;
  confidence: number | null;
  probabilities: Probabilities | null;
  detail: string;
};

export type ModelOutputs = Record<ModelKey, ModelPrediction>;

export type ArtifactMetadata = {
  artifact: string | null;
  model_name: string | null;
  metrics_file: string | null;
};

export type ModelManifest = Record<string, unknown>;

export type ReviewReason = string | { code?: string; message?: string; detail?: string };
export type TextSignal =
  | string
  | { name?: string; label?: string; description?: string; detail?: string };

export type InputMetadata = {
  character_count?: number;
  transformer_tokens?: number | null;
  transformer_max_tokens?: number;
  token_count?: number;
  original_token_count?: number;
  tokens_used?: number;
  max_tokens?: number;
  truncated?: boolean;
};

export type Prediction = {
  final_label: ThreatLabel;
  final_risk_level: RiskLevel;
  final_confidence: number | null;
  agreement: Agreement;
  model_outputs: ModelOutputs;
  explanation: string;
  suggested_action: string;
  artifact_metadata: ArtifactMetadata;
  model_manifests: Partial<Record<ModelKey, ModelManifest | null>>;
  final_model?: ModelKey | string | null;
  fallback_reason?: string | null;
  review_recommended?: boolean;
  review_reasons?: ReviewReason[];
  score_kind?: string | null;
  signals?: TextSignal[];
  heuristic_signals?: TextSignal[];
  input?: InputMetadata;
  input_metadata?: InputMetadata;
  token_count?: number;
  max_tokens?: number;
  input_truncated?: boolean;
};

export type ModelAvailability = {
  available?: boolean;
  ready?: boolean;
  state?: ModelLifecycleState | string | null;
  status?: string | null;
  detail?: string | null;
  manifest?: ModelManifest | null;
  metrics?: {
    test?: MetricSplit;
  } | null;
};

export type Readiness = {
  status: "ok" | "error";
  model_loaded?: boolean;
  model_path?: string | null;
  detail: string;
  ready?: boolean;
  primary_ready?: boolean;
  duel_ready?: boolean;
  service_status?: "checking" | "starting" | "ready" | "limited" | "unavailable" | string;
  models?: Partial<Record<ModelKey, ModelAvailability>>;
};

export type MetricSplit = {
  accuracy?: number;
  f1_macro?: number;
  precision_macro?: number;
  recall_macro?: number;
  per_label?: Partial<
    Record<ThreatLabel, { f1: number; precision: number; recall: number; support: number }>
  >;
};

export type Metadata = {
  app_name: string;
  labels: ThreatLabel[];
  max_text_chars: number;
  model: {
    loaded: boolean;
    artifact: string | null;
    metadata: Record<string, unknown>;
    status: string | null;
  };
  models?: Partial<Record<ModelKey, ModelAvailability>>;
  metrics: {
    model_name?: string;
    data_summary?: {
      modeling_rows?: number;
    };
    metrics?: {
      test?: MetricSplit;
      validation?: MetricSplit;
    };
  } | null;
  privacy: string;
};
