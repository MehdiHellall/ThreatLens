import { CornerDownLeft, Eraser, Loader2, LockKeyhole, Send } from "lucide-react";
import type { FormEvent, KeyboardEvent, RefObject } from "react";

import { formatCount } from "../presentation";

const SAMPLES = [
  {
    label: "Account alert",
    text: "Security notice: We detected a new sign-in. Review the activity at https://example.com/security and confirm your password immediately.",
  },
  {
    label: "Meeting update",
    text: "Hi team, our project review has moved to Thursday at 10:30. The agenda is in the shared workspace. See you there.",
  },
  {
    label: "Special offer",
    text: "Exclusive offer: claim your complimentary reward today at https://example.com/offer before this opportunity expires.",
  },
] as const;

type MessageEditorProps = {
  message: string;
  textLimit: number;
  privacyCopy: string;
  isLoading: boolean;
  canSubmit: boolean;
  serviceReady: boolean;
  serviceDetail?: string;
  onMessageChange: (message: string) => void;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
  textareaRef: RefObject<HTMLTextAreaElement | null>;
  submitButtonRef: RefObject<HTMLButtonElement | null>;
};

export function MessageEditor({
  message,
  textLimit,
  privacyCopy,
  isLoading,
  canSubmit,
  serviceReady,
  serviceDetail,
  onMessageChange,
  onSubmit,
  textareaRef,
  submitButtonRef,
}: MessageEditorProps) {
  function loadSample(text: string) {
    onMessageChange(text);
    window.requestAnimationFrame(() => textareaRef.current?.focus());
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter" && canSubmit) {
      event.preventDefault();
      event.currentTarget.form?.requestSubmit();
    }
  }

  return (
    <section className="panel flex min-h-0 flex-col p-5 sm:p-6 lg:p-7" aria-labelledby="message-heading">
      <form className="flex h-full min-h-0 flex-col" onSubmit={onSubmit}>
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <p className="eyebrow">Message workspace</p>
            <h2 id="message-heading" className="mt-2 text-xl font-semibold tracking-tight text-white">
              What would you like to check?
            </h2>
          </div>
          <p id="message-count" className="text-sm font-medium text-slate-400">
            {formatCount(message.length)} / {formatCount(textLimit)}
          </p>
        </div>

        <div className="mt-5 flex flex-wrap gap-2" aria-label="Example messages">
          {SAMPLES.map((sample) => (
            <button
              key={sample.label}
              type="button"
              onClick={() => loadSample(sample.text)}
              aria-pressed={message === sample.text}
              className="sample-button"
            >
              {sample.label}
            </button>
          ))}
        </div>
        <p className="mt-2 text-xs leading-5 text-slate-400">
          Examples only fill the editor. Every result comes from the live models.
        </p>
        {!serviceReady ? (
          <p role="status" className="mt-3 text-sm font-medium leading-6 text-amber-200">
            {serviceDetail ?? "Analysis will be available when at least one model is ready."}
          </p>
        ) : null}

        <label htmlFor="message" className="sr-only">
          Message
        </label>
        <textarea
          ref={textareaRef}
          id="message"
          value={message}
          maxLength={textLimit}
          onChange={(event) => onMessageChange(event.target.value)}
          onKeyDown={handleKeyDown}
          aria-describedby="message-count message-privacy keyboard-hint"
          placeholder="Paste an email, text message, or notification here…"
          className="mt-4 min-h-64 flex-1 resize-y rounded-xl border border-slate-600/70 bg-[#0b111b] p-4 text-base leading-7 text-slate-50 outline-none transition placeholder:text-slate-500 hover:border-slate-500 focus:border-cyan-300/70 focus:ring-4 focus:ring-cyan-300/10 sm:min-h-80 sm:p-5"
        />

        <div className="mt-4 flex flex-col gap-4">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <button
              type="button"
              onClick={() => onMessageChange("")}
              disabled={!message && !isLoading}
              className="secondary-button"
            >
              <Eraser className="h-4 w-4" aria-hidden="true" />
              Clear
            </button>
            <div className="flex w-full items-center gap-3 sm:w-auto">
              <span
                id="keyboard-hint"
                className="hidden items-center gap-1.5 text-xs text-slate-400 sm:inline-flex"
              >
                <CornerDownLeft className="h-3.5 w-3.5" aria-hidden="true" />
                Ctrl/⌘ + Enter
              </span>
              <button
                ref={submitButtonRef}
                type="submit"
                disabled={!canSubmit}
                className="primary-button"
              >
                {isLoading ? (
                  <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
                ) : (
                  <Send className="h-4 w-4" aria-hidden="true" />
                )}
                {isLoading ? "Analyzing…" : "Analyze message"}
              </button>
            </div>
          </div>

          <div id="message-privacy" className="flex items-start gap-2 text-sm leading-6 text-slate-400">
            <LockKeyhole className="mt-1 h-4 w-4 shrink-0 text-cyan-200" aria-hidden="true" />
            <p>{privacyCopy}</p>
          </div>
        </div>
      </form>
    </section>
  );
}
