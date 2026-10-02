import { useEffect, useRef, useState } from "react";

import { ApiRequestError, isAbortError, predictMessage } from "./api";
import threatlensLogo from "./assets/threatlens-logo.png";
import { AnalysisDetails } from "./components/AnalysisDetails";
import { MessageEditor } from "./components/MessageEditor";
import { ServiceStatus } from "./components/ServiceStatus";
import { VerdictPanel } from "./components/VerdictPanel";
import { useServiceInfo } from "./hooks/useServiceInfo";
import { LABEL_COPY } from "./presentation";
import type { Prediction } from "./types";

function App() {
  const { readiness, metadata } = useServiceInfo();
  const [message, setMessage] = useState("");
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [announcement, setAnnouncement] = useState("");

  const requestControllerRef = useRef<AbortController | null>(null);
  const requestIdRef = useRef(0);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const submitButtonRef = useRef<HTMLButtonElement>(null);
  const resultRef = useRef<HTMLDivElement>(null);

  const textLimit = metadata?.max_text_chars ?? 5000;
  const trimmedMessage = message.trim();
  const serviceReady = readiness?.ready ?? readiness?.model_loaded ?? false;
  const canSubmit =
    trimmedMessage.length > 0 &&
    trimmedMessage.length <= textLimit &&
    !isLoading &&
    serviceReady;
  const privacyCopy =
    metadata?.privacy ??
    "Messages are processed on this demo host for the current request and are not retained by the application. Avoid submitting sensitive content.";

  useEffect(
    () => () => {
      requestControllerRef.current?.abort();
    },
    [],
  );

  function handleMessageChange(nextMessage: string) {
    requestIdRef.current += 1;
    requestControllerRef.current?.abort();
    requestControllerRef.current = null;
    setMessage(nextMessage);
    setPrediction(null);
    setError(null);
    setIsLoading(false);
    setAnnouncement("");
  }

  async function runAnalysis() {
    const submittedMessage = message.trim();
    if (!submittedMessage || submittedMessage.length > textLimit || isLoading) {
      return;
    }

    requestControllerRef.current?.abort();
    const controller = new AbortController();
    const requestId = requestIdRef.current + 1;
    requestIdRef.current = requestId;
    requestControllerRef.current = controller;
    const focusResultWhenDone = document.activeElement === submitButtonRef.current;

    setIsLoading(true);
    setError(null);
    setPrediction(null);
    setAnnouncement("Analyzing message.");

    try {
      const result = await predictMessage(submittedMessage, controller.signal);
      if (requestId !== requestIdRef.current || submittedMessage !== message.trim()) {
        return;
      }
      setPrediction(result);
      setAnnouncement(
        `Analysis complete. ${LABEL_COPY[result.final_label]}, ${result.final_risk_level} risk.`,
      );
      if (focusResultWhenDone && document.activeElement === submitButtonRef.current) {
        window.requestAnimationFrame(() => resultRef.current?.focus());
      }
    } catch (requestError) {
      if (isAbortError(requestError) || requestId !== requestIdRef.current) {
        return;
      }
      let messageText =
        requestError instanceof Error
          ? requestError.message
          : "The message could not be analyzed. Please try again.";
      if (
        requestError instanceof ApiRequestError &&
        requestError.retryAfterSeconds !== null
      ) {
        messageText += ` Try again in ${requestError.retryAfterSeconds} seconds.`;
      }
      setError(messageText);
      setAnnouncement(`Analysis failed. ${messageText}`);
    } finally {
      if (requestId === requestIdRef.current) {
        requestControllerRef.current = null;
        setIsLoading(false);
      }
    }
  }

  return (
    <div className="relative min-h-screen overflow-hidden bg-[#0b1018] text-slate-50">
      <div className="ambient ambient-one" aria-hidden="true" />
      <div className="ambient ambient-two" aria-hidden="true" />

      <main className="relative mx-auto flex min-h-screen w-full max-w-[1200px] flex-col px-4 pb-8 pt-5 sm:px-6 sm:pb-10 lg:px-8">
        <header className="flex flex-col gap-5 border-b border-slate-700/60 pb-5 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex min-w-0 items-center gap-3">
            <img
              src={threatlensLogo}
              alt=""
              width={40}
              height={40}
              className="h-10 w-10 shrink-0 object-contain [image-rendering:pixelated]"
            />
            <div className="min-w-0">
              <h1 className="text-xl font-semibold tracking-tight text-white sm:text-2xl">
                ThreatLens
              </h1>
              <p className="mt-0.5 text-xs font-medium uppercase tracking-[0.16em] text-slate-400">
                Message risk intelligence
              </p>
            </div>
          </div>
          <ServiceStatus readiness={readiness} />
        </header>

        <section className="pb-7 pt-8 sm:pb-9 sm:pt-11" aria-labelledby="page-heading">
          <p className="eyebrow">Real models · live assessment</p>
          <h2
            id="page-heading"
            className="mt-3 max-w-3xl text-3xl font-semibold tracking-[-0.035em] text-white sm:text-4xl lg:text-[2.75rem] lg:leading-[1.08]"
          >
            Check a message before you act.
          </h2>
          <p className="mt-4 max-w-2xl text-base leading-7 text-slate-300 sm:text-lg">
            Compare two classifiers, understand the recommendation, and spot when a human should take another look.
          </p>
        </section>

        <div className="grid flex-1 items-start gap-5 lg:grid-cols-[minmax(0,1.08fr)_minmax(22rem,0.92fr)] lg:gap-6">
          <MessageEditor
            message={message}
            textLimit={textLimit}
            privacyCopy={privacyCopy}
            isLoading={isLoading}
            canSubmit={canSubmit}
            serviceReady={serviceReady}
            serviceDetail={readiness?.detail}
            onMessageChange={handleMessageChange}
            onSubmit={(event) => {
              event.preventDefault();
              void runAnalysis();
            }}
            textareaRef={textareaRef}
            submitButtonRef={submitButtonRef}
          />

          <aside className="min-w-0 space-y-3">
            <VerdictPanel
              prediction={prediction}
              error={error}
              isLoading={isLoading}
              onRetry={() => void runAnalysis()}
              resultRef={resultRef}
            />
            <AnalysisDetails
              prediction={prediction}
              readiness={readiness}
              metadata={metadata}
            />
          </aside>
        </div>

        <footer className="mt-8 border-t border-slate-700/50 pt-5 text-xs leading-5 text-slate-500">
          ThreatLens supports review; it does not replace security judgment. Never follow links or share credentials solely because a message receives a low-risk result.
        </footer>
      </main>

      <p className="sr-only" aria-live="polite" aria-atomic="true">
        {announcement}
      </p>
    </div>
  );
}

export default App;
