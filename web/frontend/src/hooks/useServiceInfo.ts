import { useEffect, useState } from "react";

import { getMetadata, getReadiness, isAbortError } from "../api";
import type { Metadata, Readiness } from "../types";

const API_DOWN: Readiness = {
  status: "error",
  ready: false,
  model_loaded: false,
  service_status: "unavailable",
  detail: "The analysis service is unavailable. Check the service and try again.",
};

export function useServiceInfo() {
  const [readiness, setReadiness] = useState<Readiness | null>(null);
  const [metadata, setMetadata] = useState<Metadata | null>(null);

  useEffect(() => {
    let stopped = false;
    let pollTimer: number | undefined;
    let metadataLoaded = false;
    let activeController: AbortController | null = null;

    async function poll() {
      activeController = new AbortController();
      const readinessRequest = getReadiness(activeController.signal);
      const metadataRequest = metadataLoaded
        ? Promise.resolve<Metadata | null>(null)
        : getMetadata(activeController.signal);

      const [readinessResult, metadataResult] = await Promise.allSettled([
        readinessRequest,
        metadataRequest,
      ]);
      if (stopped) {
        return;
      }

      if (readinessResult.status === "fulfilled") {
        setReadiness(readinessResult.value);
      } else if (!isAbortError(readinessResult.reason)) {
        setReadiness(API_DOWN);
      }

      if (metadataResult.status === "fulfilled" && metadataResult.value) {
        metadataLoaded = true;
        setMetadata(metadataResult.value);
      }

      const current = readinessResult.status === "fulfilled" ? readinessResult.value : null;
      pollTimer = window.setTimeout(poll, current?.duel_ready ? 15_000 : 4_000);
    }

    void poll();

    return () => {
      stopped = true;
      activeController?.abort();
      if (pollTimer !== undefined) {
        window.clearTimeout(pollTimer);
      }
    };
  }, []);

  return { readiness, metadata };
}
