import { expect, test, type Page } from "@playwright/test";

const PHISHING_MESSAGE = "Urgent password reset required. Verify your account now.";
const PERCENTAGE = /^\d+(?:\.\d+)?%$/;
const LABEL_COPY = { ham: "Legitimate", phish: "Phishing", spam: "Spam" } as const;
const RISK_COPY = { low: "Low apparent risk", medium: "Medium risk", high: "High risk" } as const;

type ThreatLabel = "ham" | "phish" | "spam";
type ModelStatus = "available" | "error" | "unavailable";

type ModelOutput = {
  status: ModelStatus;
  label: ThreatLabel | null;
  confidence: number | null;
  probabilities: Record<ThreatLabel, number> | null;
  detail: string;
};

type DuelPrediction = {
  final_label: ThreatLabel;
  final_risk_level: "low" | "medium" | "high";
  final_confidence: number | null;
  final_model: "tfidf_logreg" | "distilbert";
  fallback_reason: string | null;
  score_kind: string;
  review_recommended: boolean;
  review_reasons: string[];
  input_metadata: {
    character_count: number;
    transformer_tokens: number | null;
    transformer_max_tokens: number;
    truncated: boolean | null;
  };
  heuristic_signals: string[];
  agreement: "agreed" | "disagreed" | "partial" | "unavailable";
  model_outputs: {
    tfidf_logreg: ModelOutput;
    distilbert: ModelOutput;
  };
  explanation: string;
  suggested_action: string;
  artifact_metadata: {
    artifact: string | null;
    model_name: string | null;
    metrics_file: string | null;
  };
  model_manifests: {
    tfidf_logreg: Record<string, unknown> | null;
    distilbert: Record<string, unknown> | null;
  };
};

const TFIDF_PHISH: ModelOutput = {
  status: "available",
  label: "phish",
  confidence: 0.91,
  probabilities: { ham: 0.03, phish: 0.91, spam: 0.06 },
  detail: "Prediction completed.",
};

const DISTILBERT_PHISH: ModelOutput = {
  status: "available",
  label: "phish",
  confidence: 0.96,
  probabilities: { ham: 0.01, phish: 0.96, spam: 0.03 },
  detail: "Prediction completed.",
};

function duelPrediction(overrides: Partial<DuelPrediction> = {}): DuelPrediction {
  return {
    final_label: "phish",
    final_risk_level: "high",
    final_confidence: 0.96,
    final_model: "distilbert",
    fallback_reason: null,
    score_kind: "uncalibrated_probability",
    review_recommended: false,
    review_reasons: [],
    input_metadata: {
      character_count: PHISHING_MESSAGE.length,
      transformer_tokens: 12,
      transformer_max_tokens: 128,
      truncated: false,
    },
    heuristic_signals: ["urgency", "credential request"],
    agreement: "agreed",
    model_outputs: {
      tfidf_logreg: TFIDF_PHISH,
      distilbert: DISTILBERT_PHISH,
    },
    explanation: "Both models classified this message as phishing.",
    suggested_action: "Do not click links or share credentials.",
    artifact_metadata: {
      artifact: "tfidf_logreg.joblib",
      model_name: "TF-IDF + Logistic Regression",
      metrics_file: "metrics.json",
    },
    model_manifests: {
      tfidf_logreg: { version: "tfidf-test" },
      distilbert: { base_checkpoint: "distilbert/distilbert-base-uncased" },
    },
    ...overrides,
  };
}

async function openApp(page: Page) {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "ThreatLens" })).toBeVisible();
}

function messageEditor(page: Page) {
  return page.getByRole("textbox", { name: "Message", exact: true });
}

async function mockPrediction(page: Page, prediction: DuelPrediction, delayMs = 0) {
  await page.route("**/v1/predict", async (route) => {
    if (route.request().method() !== "POST") {
      await route.continue();
      return;
    }
    if (delayMs) {
      await new Promise((resolve) => setTimeout(resolve, delayMs));
    }
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(prediction),
    });
  });
}

async function analyze(page: Page) {
  await messageEditor(page).fill(PHISHING_MESSAGE);
  const analyzeButton = page.getByRole("button", { name: "Analyze message" });
  await expect(analyzeButton).toBeEnabled({ timeout: 180_000 });
  await analyzeButton.click();
}

test("offers focused samples without fabricating a result", async ({ page }) => {
  await openApp(page);

  const sample = page.getByRole("button", { name: "Account alert" });
  await expect(sample).toBeVisible();
  await sample.click();

  await expect(messageEditor(page)).toHaveValue(/example\.com\/security/);
  await expect(sample).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByTestId("verdict-result")).toHaveCount(0);
  await expect(page.getByText(/examples only fill the editor/i)).toBeVisible();
});

test("runs the real artifact-backed model flow", async ({ page }) => {
  await openApp(page);
  await expect(page.getByText("Two models ready")).toBeVisible({ timeout: 180_000 });

  const responsePromise = page.waitForResponse(
    (response) =>
      response.request().method() === "POST" &&
      new URL(response.url()).pathname.endsWith("/v1/predict"),
  );
  await analyze(page);
  const apiResponse = await responsePromise;
  expect(apiResponse.ok()).toBe(true);
  const apiPrediction = (await apiResponse.json()) as DuelPrediction;

  await expect(page.getByTestId("prediction-label")).toHaveText(
    LABEL_COPY[apiPrediction.final_label],
    {
      timeout: 60_000,
    },
  );
  await expect(page.getByTestId("prediction-risk")).toHaveText(
    RISK_COPY[apiPrediction.final_risk_level],
  );
  await expect(page.getByTestId("prediction-confidence")).toHaveText(PERCENTAGE);
  await expect(page.getByTestId("model-agreement")).toContainText(/agree|disagree/i);

  await page.getByTestId("analysis-details").getByText("Analysis details").click();
  await expect(page.getByTestId("model-row-tfidf-logreg")).toContainText(
    /TF-IDF.*Logistic Regression/i,
  );
  await expect(page.getByTestId("model-row-distilbert")).toContainText(/DistilBERT/i);
  await expect(page.getByTestId("prediction-explanation")).toContainText(/model/i);
  await expect(page.getByTestId("suggested-action")).toContainText(
    /click|link|credential|sender|review|verify/i,
  );
});

test("puts disagreement and manual review beside the verdict", async ({ page }) => {
  await mockPrediction(
    page,
    duelPrediction({
      final_label: "spam",
      final_risk_level: "medium",
      final_confidence: 0.72,
      agreement: "disagreed",
      review_recommended: true,
      review_reasons: ["model_disagreement"],
      model_outputs: {
        tfidf_logreg: TFIDF_PHISH,
        distilbert: {
          status: "available",
          label: "spam",
          confidence: 0.72,
          probabilities: { ham: 0.04, phish: 0.24, spam: 0.72 },
          detail: "Prediction completed.",
        },
      },
      explanation: "The classifiers reached different conclusions.",
      suggested_action: "Treat this as spam and review it manually.",
    }),
  );
  await openApp(page);
  await analyze(page);

  await expect(page.getByTestId("prediction-label")).toHaveText("Spam");
  await expect(page.getByTestId("model-agreement")).toContainText(/disagree/i);
  await expect(page.getByTestId("uncertainty-warning")).toContainText(/manual review/i);
  await expect(page.getByTestId("uncertainty-warning")).toContainText(
    /different conclusions/i,
  );
});

test("explains a primary-model fallback without exposing raw codes", async ({ page }) => {
  await mockPrediction(
    page,
    duelPrediction({
      final_confidence: 0.91,
      final_model: "tfidf_logreg",
      fallback_reason: "primary_prediction_failed",
      agreement: "partial",
      review_recommended: true,
      review_reasons: ["primary_prediction_failed"],
      model_outputs: {
        tfidf_logreg: TFIDF_PHISH,
        distilbert: {
          status: "error",
          label: null,
          confidence: null,
          probabilities: null,
          detail: "DistilBERT could not complete this request.",
        },
      },
    }),
  );
  await openApp(page);
  await analyze(page);

  await expect(page.getByTestId("model-agreement")).toContainText(/partial/i);
  await expect(page.getByTestId("fallback-message")).toContainText(/TF-IDF fallback/i);
  await expect(page.getByTestId("fallback-message")).not.toContainText(
    "primary_prediction_failed",
  );
});

test("invalidates an in-flight result when the message changes", async ({ page }) => {
  await mockPrediction(page, duelPrediction(), 500);
  await openApp(page);
  await analyze(page);
  await messageEditor(page).fill("A different message entered while the model runs.");

  await page.waitForTimeout(700);
  await expect(page.getByTestId("verdict-result")).toHaveCount(0);
  await expect(page.getByText("Your assessment will appear here")).toBeVisible();
});

test("shows retry guidance while preserving the submitted message", async ({ page }) => {
  await page.route("**/v1/predict", async (route) => {
    await route.fulfill({
      status: 503,
      contentType: "application/json",
      headers: { "Retry-After": "5" },
      body: JSON.stringify({ detail: "The prediction service is busy." }),
    });
  });
  await openApp(page);
  await analyze(page);

  await expect(page.getByRole("alert")).toContainText(/busy/i);
  await expect(page.getByRole("alert")).toContainText(/5 seconds/i);
  await expect(messageEditor(page)).toHaveValue(PHISHING_MESSAGE);
  await expect(page.getByRole("button", { name: "Try again" })).toBeEnabled();
});

test("supports the keyboard submit shortcut", async ({ page }) => {
  await mockPrediction(page, duelPrediction());
  await openApp(page);
  const editor = messageEditor(page);
  await editor.fill(PHISHING_MESSAGE);
  await expect(page.getByRole("button", { name: "Analyze message" })).toBeEnabled({
    timeout: 180_000,
  });
  await editor.press("Control+Enter");

  await expect(page.getByTestId("prediction-label")).toHaveText("Phishing");
});

test("stays usable without horizontal overflow at release widths", async ({ page }) => {
  for (const width of [320, 390, 768, 1024, 1440]) {
    await page.setViewportSize({ width, height: 900 });
    await openApp(page);
    const hasHorizontalOverflow = await page.evaluate(
      () => document.body.scrollWidth > document.documentElement.clientWidth + 1,
    );
    expect(hasHorizontalOverflow, `unexpected horizontal overflow at ${width}px`).toBe(false);
  }
});
