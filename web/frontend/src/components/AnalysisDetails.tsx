import { BarChart3, ChevronDown, Cpu, Fingerprint, ListTree } from "lucide-react";

import {
  formatPercent,
  inputSummary,
  LABEL_COPY,
  LABELS,
  MODEL_NAMES,
  modelIsReady,
  selectedProbabilities,
  signalText,
} from "../presentation";
import type {
  Metadata,
  ModelAvailability,
  ModelKey,
  ModelManifest,
  ModelPrediction,
  Prediction,
  Readiness,
} from "../types";

type AnalysisDetailsProps = {
  prediction: Prediction | null;
  readiness: Readiness | null;
  metadata: Metadata | null;
};

function manifestValue(
  manifest: ModelManifest | null | undefined,
  keys: string[],
): string | undefined {
  for (const key of keys) {
    const value = manifest?.[key];
    if (typeof value === "string" && value.trim()) {
      return value;
    }
    if (typeof value === "number") {
      return String(value);
    }
  }
  return undefined;
}

function metadataString(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value : undefined;
}

export function AnalysisDetails({ prediction, readiness, metadata }: AnalysisDetailsProps) {
  const probabilities = prediction ? selectedProbabilities(prediction) : null;
  const input = prediction ? inputSummary(prediction) : null;
  const signals = (prediction?.heuristic_signals ?? prediction?.signals ?? [])
    .map(signalText)
    .filter((signal): signal is string => Boolean(signal));

  return (
    <div className="space-y-3">
      <details className="disclosure panel" data-testid="analysis-details">
        <summary>
          <span className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5 text-cyan-200" aria-hidden="true" />
            <span>
              <span className="block font-semibold text-slate-100">Analysis details</span>
              <span className="mt-0.5 block text-xs font-normal text-slate-400">
                Scores, model comparison, and input handling
              </span>
            </span>
          </span>
          <ChevronDown className="disclosure-chevron h-5 w-5 text-slate-400" aria-hidden="true" />
        </summary>
        <div className="border-t border-slate-700/70 px-5 pb-5 pt-5 sm:px-6">
          {probabilities ? <ProbabilityBars probabilities={probabilities} /> : null}

          {prediction ? (
            <div className={probabilities ? "mt-6" : ""}>
              <h3 className="section-label">Model comparison</h3>
              <div className="mt-3 grid gap-3 sm:grid-cols-2">
                <ModelCard
                  modelKey="tfidf_logreg"
                  output={prediction.model_outputs.tfidf_logreg}
                  availability={readiness?.models?.tfidf_logreg ?? metadata?.models?.tfidf_logreg}
                  fallbackReady={readiness?.model_loaded}
                />
                <ModelCard
                  modelKey="distilbert"
                  output={prediction.model_outputs.distilbert}
                  availability={readiness?.models?.distilbert ?? metadata?.models?.distilbert}
                />
              </div>
            </div>
          ) : (
            <div>
              <h3 className="section-label">Model availability</h3>
              <div className="mt-3 grid gap-3 sm:grid-cols-2">
                <ModelCard
                  modelKey="tfidf_logreg"
                  availability={readiness?.models?.tfidf_logreg ?? metadata?.models?.tfidf_logreg}
                  fallbackReady={readiness?.model_loaded}
                />
                <ModelCard
                  modelKey="distilbert"
                  availability={readiness?.models?.distilbert ?? metadata?.models?.distilbert}
                />
              </div>
            </div>
          )}

          {prediction && (input?.tokenCount !== undefined || input?.maxTokens !== undefined) ? (
            <div className="mt-6 rounded-xl border border-slate-700/70 bg-slate-950/25 p-4">
              <div className="flex items-center gap-2">
                <ListTree className="h-4 w-4 text-cyan-200" aria-hidden="true" />
                <h3 className="section-label">Input handling</h3>
              </div>
              <p className="mt-2 text-sm leading-6 text-slate-300">
                {input.tokenCount !== undefined ? `${input.tokenCount} tokens detected. ` : ""}
                {input.tokensUsed !== undefined ? `${input.tokensUsed} tokens analyzed. ` : ""}
                {input.maxTokens !== undefined ? `Transformer limit: ${input.maxTokens}.` : ""}
              </p>
              {input.truncated ? (
                <p className="mt-2 text-sm font-medium text-amber-200">
                  Text beyond the transformer limit was not included in that model’s score.
                </p>
              ) : null}
            </div>
          ) : null}

          {signals.length ? (
            <div className="mt-6">
              <h3 className="section-label">Observed text signals</h3>
              <ul className="mt-3 flex flex-wrap gap-2">
                {signals.map((signal) => (
                  <li key={signal} className="rounded-full border border-slate-600/70 bg-slate-950/30 px-3 py-1.5 text-xs text-slate-300">
                    {signal}
                  </li>
                ))}
              </ul>
              <p className="mt-2 text-xs leading-5 text-slate-500">
                These are transparent keyword observations, not explanations of model reasoning.
              </p>
            </div>
          ) : null}
        </div>
      </details>

      <AboutModels prediction={prediction} readiness={readiness} metadata={metadata} />
    </div>
  );
}

function ProbabilityBars({ probabilities }: { probabilities: Record<string, number> }) {
  return (
    <div aria-label="Selected model scores">
      <h3 className="section-label">Selected model scores</h3>
      <div className="mt-4 space-y-3">
        {LABELS.map((label) => {
          const value = probabilities[label];
          const width = Math.max(0, Math.min(1, value)) * 100;
          return (
            <div key={label}>
              <div className="flex items-center justify-between gap-3 text-sm">
                <span className="font-medium text-slate-200">{LABEL_COPY[label]}</span>
                <span data-testid={`probability-${label}`} className="font-semibold text-slate-300">
                  {formatPercent(value)}
                </span>
              </div>
              <div className="mt-2 h-2 overflow-hidden rounded-full bg-slate-700/60">
                <div
                  className={`h-full rounded-full ${
                    label === "ham" ? "bg-emerald-300" : label === "phish" ? "bg-rose-300" : "bg-amber-300"
                  }`}
                  style={{ width: `${width}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function ModelCard({
  modelKey,
  output,
  availability,
  fallbackReady,
}: {
  modelKey: ModelKey;
  output?: ModelPrediction;
  availability?: ModelAvailability;
  fallbackReady?: boolean;
}) {
  const ready = output
    ? output.status === "available"
    : (modelIsReady(availability) ?? fallbackReady);
  const status = output
    ? output.status === "available"
      ? "Available"
      : output.status === "error"
        ? "Error"
        : "Unavailable"
    : ready === undefined
      ? availability?.state === "loading" || availability?.state === "retrying"
        ? "Starting"
        : "Checking"
      : ready
        ? "Ready"
        : "Unavailable";
  const detail = output?.detail ?? availability?.detail;

  return (
    <article
      data-testid={`model-row-${modelKey.replace("_", "-")}`}
      className="rounded-xl border border-slate-700/70 bg-slate-950/25 p-4"
    >
      <div className="flex items-start justify-between gap-3">
        <h4 className="text-sm font-semibold leading-5 text-slate-100">{MODEL_NAMES[modelKey]}</h4>
        <span
          className={`rounded-full border px-2 py-0.5 text-[0.7rem] font-bold uppercase tracking-wide ${
            ready
              ? "border-emerald-300/25 bg-emerald-300/10 text-emerald-100"
              : output?.status === "error"
                ? "border-amber-300/25 bg-amber-300/10 text-amber-100"
                : "border-slate-600 bg-slate-800 text-slate-300"
          }`}
        >
          {status}
        </span>
      </div>
      {output?.status === "available" && output.label ? (
        <div className="mt-4 flex items-end justify-between gap-3">
          <p className="font-semibold text-white">{LABEL_COPY[output.label]}</p>
          <p className="text-sm font-semibold text-slate-300">{formatPercent(output.confidence)}</p>
        </div>
      ) : null}
      {detail ? <p className="mt-3 text-xs leading-5 text-slate-400">{detail}</p> : null}
    </article>
  );
}

function AboutModels({
  prediction,
  readiness,
  metadata,
}: AnalysisDetailsProps) {
  const artifact = prediction?.artifact_metadata.artifact ?? metadata?.model.artifact;
  const baselineName =
    prediction?.artifact_metadata.model_name ??
    metadataString(metadata?.model.metadata.model_name) ??
    MODEL_NAMES.tfidf_logreg;
  const distilbertManifest =
    prediction?.model_manifests.distilbert ?? metadata?.models?.distilbert?.manifest;
  const distilbertVersion = manifestValue(distilbertManifest, [
    "model_version",
    "version",
    "base_checkpoint",
  ]);
  const recordedMetrics = metadata?.metrics?.metrics?.test;
  const recordedDistilbertMetrics = metadata?.models?.distilbert?.metrics?.test;

  return (
    <details className="disclosure panel">
      <summary>
        <span className="flex items-center gap-2">
          <Cpu className="h-5 w-5 text-emerald-200" aria-hidden="true" />
          <span>
            <span className="block font-semibold text-slate-100">About the models</span>
            <span className="mt-0.5 block text-xs font-normal text-slate-400">
              Version and evaluation provenance
            </span>
          </span>
        </span>
        <ChevronDown className="disclosure-chevron h-5 w-5 text-slate-400" aria-hidden="true" />
      </summary>
      <div className="border-t border-slate-700/70 px-5 pb-5 pt-5 sm:px-6">
        <dl className="grid gap-3 sm:grid-cols-2">
          <Detail label="Baseline" value={baselineName} />
          <Detail label="DistilBERT version" value={distilbertVersion ?? "Not reported"} />
          <Detail label="Both models ready" value={readiness?.duel_ready ? "Yes" : "No"} />
          <Detail
            label="Recorded TF-IDF test metrics"
            value={
              recordedMetrics?.accuracy !== undefined
                ? `${formatPercent(recordedMetrics.accuracy)} accuracy${
                    recordedMetrics.f1_macro !== undefined
                      ? ` · ${formatPercent(recordedMetrics.f1_macro)} macro F1`
                      : ""
                  }`
                : "Not reported"
              }
          />
          <Detail
            label="Recorded DistilBERT test metrics"
            value={
              recordedDistilbertMetrics?.accuracy !== undefined
                ? `${formatPercent(recordedDistilbertMetrics.accuracy)} accuracy${
                    recordedDistilbertMetrics.f1_macro !== undefined
                      ? ` · ${formatPercent(recordedDistilbertMetrics.f1_macro)} macro F1`
                      : ""
                  }`
                : "Not reported"
            }
          />
        </dl>
        {artifact ? (
          <div className="mt-4 flex items-start gap-2 text-xs leading-5 text-slate-500">
            <Fingerprint className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
            <p data-testid="model-artifact">Artifact: {artifact}</p>
          </div>
        ) : null}
        <p className="mt-4 text-xs leading-5 text-slate-500">
          Recorded metrics describe the packaged evaluation artifact; they are not a guarantee for an individual message.
        </p>
      </div>
    </details>
  );
}

function Detail({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-slate-700/70 bg-slate-950/25 p-3.5">
      <dt className="text-xs font-semibold uppercase tracking-wide text-slate-400">{label}</dt>
      <dd className="mt-2 break-words text-sm font-semibold text-slate-100">{value}</dd>
    </div>
  );
}
