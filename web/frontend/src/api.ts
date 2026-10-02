import type { Metadata, Prediction, Readiness } from "./types";

const configuredApiBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim();
const API_BASE_URL = (configuredApiBaseUrl || "/api").replace(/\/$/, "");

type ApiErrorBody = {
  detail?: string | Array<{ msg?: string }>;
};

type RequestOptions = RequestInit & {
  timeoutMs?: number;
  acceptedStatuses?: number[];
};

export class ApiRequestError extends Error {
  readonly status: number | null;
  readonly retryAfterSeconds: number | null;

  constructor(
    message: string,
    options: { status?: number; retryAfterSeconds?: number | null } = {},
  ) {
    super(message);
    this.name = "ApiRequestError";
    this.status = options.status ?? null;
    this.retryAfterSeconds = options.retryAfterSeconds ?? null;
  }
}

function detailMessage(body: ApiErrorBody, fallback: string): string {
  if (typeof body.detail === "string" && body.detail.trim()) {
    return body.detail;
  }
  if (Array.isArray(body.detail)) {
    const validationMessage = body.detail.find((item) => item.msg)?.msg;
    if (validationMessage) {
      return validationMessage;
    }
  }
  return fallback;
}

function retryAfterSeconds(response: Response): number | null {
  const value = response.headers.get("retry-after");
  if (!value) {
    return null;
  }
  const seconds = Number(value);
  return Number.isFinite(seconds) && seconds >= 0 ? seconds : null;
}

async function requestJson<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const {
    timeoutMs = 10_000,
    acceptedStatuses = [],
    signal: callerSignal,
    ...init
  } = options;
  const controller = new AbortController();
  let timedOut = false;

  const abortFromCaller = () => controller.abort();
  if (callerSignal?.aborted) {
    controller.abort();
  } else {
    callerSignal?.addEventListener("abort", abortFromCaller, { once: true });
  }

  const timeout = window.setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);

  const headers = new Headers(init.headers);
  if (init.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  try {
    const response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      headers,
      signal: controller.signal,
    });
    const body = (await response.json().catch(() => ({}))) as T & ApiErrorBody;

    if (!response.ok && !acceptedStatuses.includes(response.status)) {
      throw new ApiRequestError(
        detailMessage(body, `The service returned an error (${response.status}).`),
        {
          status: response.status,
          retryAfterSeconds: retryAfterSeconds(response),
        },
      );
    }
    return body;
  } catch (error) {
    if (timedOut) {
      throw new ApiRequestError("The analysis took too long. Please try again.");
    }
    throw error;
  } finally {
    window.clearTimeout(timeout);
    callerSignal?.removeEventListener("abort", abortFromCaller);
  }
}

export async function getReadiness(signal?: AbortSignal): Promise<Readiness> {
  return requestJson<Readiness>("/v1/ready", {
    signal,
    timeoutMs: 8_000,
    acceptedStatuses: [503],
  });
}

export function getMetadata(signal?: AbortSignal): Promise<Metadata> {
  return requestJson<Metadata>("/v1/metadata", { signal, timeoutMs: 8_000 });
}

export function predictMessage(text: string, signal?: AbortSignal): Promise<Prediction> {
  return requestJson<Prediction>("/v1/predict", {
    method: "POST",
    body: JSON.stringify({ text }),
    signal,
    timeoutMs: 30_000,
  });
}

export function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}
