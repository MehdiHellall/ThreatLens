import type {
  ModelAvailability,
  ModelKey,
  Prediction,
  Readiness,
  ReviewReason,
  TextSignal,
  ThreatLabel,
} from "./types";

export const LABELS: ThreatLabel[] = ["ham", "phish", "spam"];

export const LABEL_COPY: Record<ThreatLabel, string> = {
  ham: "Legitimate",
  phish: "Phishing",
  spam: "Spam",
};

export const MODEL_NAMES: Record<ModelKey, string> = {
  tfidf_logreg: "TF-IDF + Logistic Regression",
  distilbert: "DistilBERT",
};

export function formatPercent(value?: number | null): string {
  if (value == null || !Number.isFinite(value)) {
    return "Unavailable";
  }
  return new Intl.NumberFormat("en", {
    style: "percent",
    maximumFractionDigits: 1,
  }).format(value);
}

export function formatCount(value?: number): string {
  return value === undefined ? "Unavailable" : new Intl.NumberFormat("en").format(value);
}

export function modelIsReady(model?: ModelAvailability): boolean | undefined {
  if (!model) {
    return undefined;
  }
  if (typeof model.ready === "boolean") {
    return model.ready;
  }
  if (typeof model.available === "boolean") {
    return model.available;
  }
  if (model.state) {
    return model.state === "ready";
  }
  return undefined;
}

export type ServicePresentation = {
  label: string;
  detail: string;
  tone: "checking" | "ready" | "limited" | "unavailable";
};

export function servicePresentation(readiness: Readiness | null): ServicePresentation {
  if (!readiness) {
    return {
      label: "Checking service",
      detail: "Connecting to the analysis service.",
      tone: "checking",
    };
  }

  if (readiness.duel_ready || readiness.service_status === "ready") {
    return {
      label: "Two models ready",
      detail: readiness.detail || "Both classifiers are available.",
      tone: "ready",
    };
  }

  const modelStates = Object.values(readiness.models ?? {});
  const hasReadyModel =
    readiness.ready === true ||
    readiness.primary_ready === true ||
    readiness.model_loaded === true ||
    modelStates.some((model) => modelIsReady(model) === true);
  if (hasReadyModel || readiness.service_status === "limited") {
    return {
      label: "Limited mode",
      detail: readiness.detail || "One classifier is available while another starts.",
      tone: "limited",
    };
  }

  const isStarting =
    readiness.service_status === "starting" ||
    modelStates.some((model) => ["idle", "loading", "retrying"].includes(model.state ?? ""));
  if (isStarting) {
    return {
      label: "Models starting",
      detail: readiness.detail || "The classifiers are loading.",
      tone: "checking",
    };
  }

  return {
    label: "Service unavailable",
    detail: readiness.detail || "No classifier is currently available.",
    tone: "unavailable",
  };
}

export function selectedProbabilities(prediction: Prediction) {
  if (prediction.final_model === "tfidf_logreg") {
    return prediction.model_outputs.tfidf_logreg.probabilities;
  }
  if (prediction.final_model === "distilbert") {
    return prediction.model_outputs.distilbert.probabilities;
  }
  return prediction.model_outputs.distilbert.status === "available"
    ? prediction.model_outputs.distilbert.probabilities
    : prediction.model_outputs.tfidf_logreg.probabilities;
}

export function reviewReasonText(reason: ReviewReason): string | null {
  if (typeof reason === "string") {
    const knownReasons: Record<string, string> = {
      model_disagreement: "The classifiers reached different conclusions.",
      primary_unavailable: "The primary DistilBERT model was unavailable.",
      primary_prediction_failed: "The primary DistilBERT model could not complete this request.",
      input_truncated: "The transformer analyzed only the configured token window.",
      suspicious_text_signals:
        "A separate keyword check found a credential request combined with urgency or a contact prompt.",
      low_model_score: "The selected model score is too low for a decisive result.",
    };
    const formattedReason = knownReasons[reason] ?? reason.replaceAll("_", " ").trim();
    return formattedReason || null;
  }
  return reason.message?.trim() || reason.detail?.trim() || reason.code?.trim() || null;
}

export function signalText(signal: TextSignal): string | null {
  if (typeof signal === "string") {
    return signal.trim() || null;
  }
  return (
    signal.description?.trim() ||
    signal.detail?.trim() ||
    signal.label?.trim() ||
    signal.name?.trim() ||
    null
  );
}

export function inputSummary(prediction: Prediction) {
  const input = prediction.input_metadata ?? prediction.input;
  return {
    tokenCount:
      input?.transformer_tokens ??
      input?.original_token_count ??
      input?.token_count ??
      prediction.token_count,
    tokensUsed: input?.tokens_used,
    maxTokens: input?.transformer_max_tokens ?? input?.max_tokens ?? prediction.max_tokens,
    truncated: input?.truncated ?? prediction.input_truncated ?? false,
  };
}

export function fallbackReasonText(reason?: string | null): string | null {
  if (!reason) {
    return null;
  }
  if (reason === "primary_unavailable") {
    return "DistilBERT was unavailable, so this result uses the TF-IDF fallback.";
  }
  if (reason === "primary_prediction_failed") {
    return "DistilBERT could not complete this request, so this result uses the TF-IDF fallback.";
  }
  return reason.replaceAll("_", " ");
}
