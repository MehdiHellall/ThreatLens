import { AlertTriangle, ArrowRight, CheckCircle2, Loader2, ScanSearch, ShieldAlert } from "lucide-react";
import type { RefObject } from "react";

import {
  fallbackReasonText,
  formatPercent,
  inputSummary,
  LABEL_COPY,
  MODEL_NAMES,
  reviewReasonText,
} from "../presentation";
import type { Agreement, Prediction, RiskLevel } from "../types";

const RISK_COPY: Record<RiskLevel, string> = {
  low: "Low apparent risk",
  medium: "Medium risk",
  high: "High risk",
};

const RISK_TONE: Record<RiskLevel, string> = {
  low: "border-emerald-300/25 bg-emerald-300/10 text-emerald-100",
  medium: "border-amber-300/30 bg-amber-300/10 text-amber-100",
  high: "border-rose-300/30 bg-rose-300/10 text-rose-100",
};

const LABEL_TONE = {
  ham: "text-emerald-200",
  phish: "text-rose-200",
  spam: "text-amber-200",
};

const AGREEMENT_COPY: Record<Agreement, string> = {
  agreed: "Models agreed",
  disagreed: "Models disagreed",
  partial: "Partial result",
  unavailable: "Single-model result",
};

type VerdictPanelProps = {
  prediction: Prediction | null;
  error: string | null;
  isLoading: boolean;
  onRetry: () => void;
  resultRef: RefObject<HTMLDivElement | null>;
};

function modelName(model?: string | null): string {
  if (model === "tfidf_logreg" || model === "distilbert") {
    return MODEL_NAMES[model];
  }
  return model?.trim() || "selected model";
}

function reviewReasons(prediction: Prediction): string[] {
  const reasons = (prediction.review_reasons ?? [])
    .map(reviewReasonText)
    .filter((reason): reason is string => Boolean(reason));
  const input = inputSummary(prediction);

  if (prediction.agreement === "disagreed") {
    reasons.push("The classifiers reached different conclusions.");
  }
  if (input.truncated) {
    reasons.push("The message exceeded the transformer limit and was truncated.");
  }
  return [...new Set(reasons)];
}

export function VerdictPanel({
  prediction,
  error,
  isLoading,
  onRetry,
  resultRef,
}: VerdictPanelProps) {
  return (
    <section className="panel min-h-[28rem] p-5 sm:p-6 lg:p-7" aria-labelledby="verdict-heading">
      <div className="flex items-center gap-2">
        <ScanSearch className="h-5 w-5 text-cyan-200" aria-hidden="true" />
        <p className="eyebrow">Live analysis</p>
      </div>
      <h2 id="verdict-heading" className="mt-2 text-xl font-semibold tracking-tight text-white">
        Verdict
      </h2>

      {isLoading ? <LoadingVerdict /> : null}
      {!isLoading && error ? <ErrorVerdict error={error} onRetry={onRetry} /> : null}
      {!isLoading && !error && prediction ? (
        <PredictionVerdict prediction={prediction} resultRef={resultRef} />
      ) : null}
      {!isLoading && !error && !prediction ? <EmptyVerdict /> : null}
    </section>
  );
}

function LoadingVerdict() {
  return (
    <div className="mt-8" aria-busy="true" aria-label="Analyzing message">
      <div className="flex items-center gap-3 text-cyan-100">
        <Loader2 className="h-5 w-5 animate-spin" aria-hidden="true" />
        <p className="font-semibold">Analyzing with the available models…</p>
      </div>
      <div className="mt-8 space-y-4" aria-hidden="true">
        <div className="skeleton h-8 w-2/3" />
        <div className="skeleton h-20 w-full" />
        <div className="skeleton h-16 w-full" />
      </div>
    </div>
  );
}

function ErrorVerdict({ error, onRetry }: { error: string; onRetry: () => void }) {
  return (
    <div role="alert" className="mt-8 rounded-xl border border-rose-300/25 bg-rose-300/10 p-4">
      <div className="flex items-start gap-3">
        <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-rose-200" aria-hidden="true" />
        <div>
          <h3 className="font-semibold text-rose-100">Analysis could not finish</h3>
          <p className="mt-1 text-sm leading-6 text-rose-100/85">{error}</p>
          <button type="button" onClick={onRetry} className="secondary-button mt-4">
            Try again
            <ArrowRight className="h-4 w-4" aria-hidden="true" />
          </button>
        </div>
      </div>
    </div>
  );
}

function EmptyVerdict() {
  return (
    <div className="mt-10 flex min-h-64 flex-col items-center justify-center rounded-xl border border-dashed border-slate-600/70 bg-slate-950/20 px-6 text-center">
      <div className="grid h-12 w-12 place-items-center rounded-xl border border-cyan-300/20 bg-cyan-300/10">
        <ShieldAlert className="h-6 w-6 text-cyan-200" aria-hidden="true" />
      </div>
      <h3 className="mt-4 font-semibold text-slate-100">Your assessment will appear here</h3>
      <p className="mt-2 max-w-sm text-sm leading-6 text-slate-400">
        Paste a message or choose an example, then run a live analysis.
      </p>
    </div>
  );
}

function PredictionVerdict({
  prediction,
  resultRef,
}: {
  prediction: Prediction;
  resultRef: RefObject<HTMLDivElement | null>;
}) {
  const reasons = reviewReasons(prediction);
  const reviewRecommended =
    prediction.review_recommended === true ||
    prediction.agreement === "disagreed" ||
    inputSummary(prediction).truncated;

  return (
    <div ref={resultRef} tabIndex={-1} className="mt-6 outline-none" data-testid="verdict-result">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <span
            data-testid="prediction-risk"
            className={`inline-flex rounded-full border px-3 py-1 text-xs font-bold uppercase tracking-[0.12em] ${RISK_TONE[prediction.final_risk_level]}`}
          >
            {RISK_COPY[prediction.final_risk_level]}
          </span>
          <p
            data-testid="prediction-label"
            className={`mt-3 text-4xl font-semibold tracking-tight ${LABEL_TONE[prediction.final_label]}`}
          >
            {LABEL_COPY[prediction.final_label]}
          </p>
          <p data-testid="model-agreement" className="mt-2 text-sm font-medium text-slate-300">
            {AGREEMENT_COPY[prediction.agreement]}
          </p>
        </div>
        <div className="min-w-28 rounded-xl border border-slate-600/60 bg-slate-950/35 p-3 text-right">
          <p data-testid="prediction-confidence" className="text-2xl font-semibold text-white">
            {formatPercent(prediction.final_confidence)}
          </p>
          <p className="mt-1 text-xs font-medium text-slate-400">
            Model score
          </p>
        </div>
      </div>

      <div className="mt-6 rounded-xl border border-cyan-300/20 bg-cyan-300/[0.07] p-4">
        <p className="text-xs font-bold uppercase tracking-[0.14em] text-cyan-200">Recommended action</p>
        <p data-testid="suggested-action" className="mt-2 text-base font-medium leading-7 text-slate-50">
          {prediction.suggested_action}
        </p>
      </div>

      {reviewRecommended ? (
        <div
          data-testid="uncertainty-warning"
          role="alert"
          className="mt-4 rounded-xl border border-amber-300/30 bg-amber-300/10 p-4"
        >
          <div className="flex items-start gap-3">
            <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-amber-200" aria-hidden="true" />
            <div>
              <p className="font-semibold text-amber-100">Manual review recommended</p>
              {reasons.length ? (
                <ul className="mt-2 space-y-1 text-sm leading-6 text-amber-100/85">
                  {reasons.map((reason) => (
                    <li key={reason}>{reason}</li>
                  ))}
                </ul>
              ) : (
                <p className="mt-1 text-sm leading-6 text-amber-100/85">
                  Treat this result as a signal, not a final security decision.
                </p>
              )}
            </div>
          </div>
        </div>
      ) : (
        <div className="mt-4 flex items-start gap-2 text-sm leading-6 text-slate-400">
          <CheckCircle2 className="mt-1 h-4 w-4 shrink-0 text-emerald-200" aria-hidden="true" />
          <p>No additional review condition was reported by the service.</p>
        </div>
      )}

      {prediction.fallback_reason ? (
        <p
          data-testid="fallback-message"
          className="mt-4 rounded-lg border border-amber-300/20 bg-amber-300/[0.07] px-3 py-2 text-sm leading-6 text-amber-100"
        >
          {fallbackReasonText(prediction.fallback_reason)}
        </p>
      ) : prediction.agreement === "partial" || prediction.agreement === "unavailable" ? (
        <p
          data-testid="fallback-message"
          className="mt-4 rounded-lg border border-slate-600/60 bg-slate-950/30 px-3 py-2 text-sm leading-6 text-slate-300"
        >
          This result uses one available classifier because the other model could not complete the request.
        </p>
      ) : null}

      <div className="mt-5 border-t border-slate-700/70 pt-5">
        <p className="text-xs font-bold uppercase tracking-[0.14em] text-slate-400">Why this result</p>
        <p data-testid="prediction-explanation" className="mt-2 text-sm leading-6 text-slate-300">
          {prediction.explanation}
        </p>
        <p className="mt-3 text-xs leading-5 text-slate-500">
          Selected output: {modelName(prediction.final_model)}. Scores are model outputs and may not be calibrated probabilities.
        </p>
      </div>
    </div>
  );
}
