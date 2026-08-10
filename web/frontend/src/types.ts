export type ThreatLabel = "ham" | "phish" | "spam";
export type RiskLevel = "low" | "medium" | "high";
export type Agreement = "agreed" | "disagreed" | "partial" | "unavailable";
export type ModelStatus = "available" | "error" | "unavailable";

export type Probabilities = Record<ThreatLabel, number>;

export type ModelPrediction = {
  status: ModelStatus;
  label: ThreatLabel | null;
  confidence: number | null;
  probabilities: Probabilities | null;
  detail: string;
};

export type ModelKey = "tfidf_logreg" | "distilbert";
export type ModelOutputs = Record<ModelKey, ModelPrediction>;

export type ArtifactMetadata = {
  artifact: string | null;
  model_name: string | null;
  metrics_file: string | null;
};

export type ModelManifest = Record<string, unknown>;

export type Prediction = {
  final_label: ThreatLabel;
  final_risk_level: RiskLevel;
  final_confidence: number | null;
  agreement: Agreement;
  model_outputs: ModelOutputs;
  explanation: string;
  suggested_action: string;
  artifact_metadata: ArtifactMetadata;
  model_manifests: Record<ModelKey, ModelManifest | null>;
};

export type ModelAvailability = {
  available: boolean;
  status?: string | null;
  detail?: string | null;
  manifest?: ModelManifest | null;
};

export type Readiness = {
  status: "ok" | "error";
  model_loaded: boolean;
  model_path: string | null;
  detail: string;
  duel_ready?: boolean;
  models?: Record<ModelKey, ModelAvailability>;
};

export type MetricSplit = {
  accuracy?: number;
  f1_macro?: number;
  precision_macro?: number;
  recall_macro?: number;
  per_label?: Record<
    ThreatLabel,
    { f1: number; precision: number; recall: number; support: number }
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
  models?: Record<ModelKey, ModelAvailability>;
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
