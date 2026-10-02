import { CircleDot } from "lucide-react";

import { servicePresentation } from "../presentation";
import type { Readiness } from "../types";

const TONE_CLASSES = {
  checking: "border-slate-500/40 bg-slate-400/10 text-slate-200",
  ready: "border-emerald-300/30 bg-emerald-300/10 text-emerald-100",
  limited: "border-amber-300/30 bg-amber-300/10 text-amber-100",
  unavailable: "border-rose-300/30 bg-rose-300/10 text-rose-100",
};

export function ServiceStatus({ readiness }: { readiness: Readiness | null }) {
  const status = servicePresentation(readiness);

  return (
    <div
      role="status"
      aria-live="polite"
      title={status.detail}
      className={`inline-flex min-h-11 w-fit items-center gap-2 rounded-full border px-3.5 py-2 text-sm font-semibold ${TONE_CLASSES[status.tone]}`}
    >
      <CircleDot className="h-4 w-4" aria-hidden="true" />
      <span>{status.label}</span>
    </div>
  );
}
