import { expect, test, type Page } from "@playwright/test";

const PHISHING_MESSAGE = "Urgent password reset required verify account.";
const PERCENTAGE = /^\d+(?:\.\d+)?%$/;

type ThreatLabel = "ham" | "phish" | "spam";
type Agreement = "agreed" | "disagreed" | "partial" | "unavailable";
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
  final_confidence: number;
  agreement: Agreement;
  model_outputs: {
    tfidf_logreg: ModelOutput;
    distilbert: ModelOutput;
  };
  explanation: string;
  suggested_action: string;
  artifact_metadata: {
    artifact: string;
    model_name: string;
    metrics_file: string;
  };
  model_manifests: {
    tfidf_logreg: Record<string, unknown>;
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

function duelPrediction(
  overrides: Partial<DuelPrediction> = {},
): DuelPrediction {
  return {
    final_label: "phish",
    final_risk_level: "high",
    final_confidence: 0.96,
    agreement: "agreed",
    model_outputs: {
      tfidf_logreg: TFIDF_PHISH,
      distilbert: DISTILBERT_PHISH,
    },
    explanation:
      "TF-IDF Logistic Regression and DistilBERT both classified this message as phishing.",
    suggested_action: "Do not click links or share credentials.",
    artifact_metadata: {
      artifact: "tfidf_logreg.joblib",
      model_name: "TF-IDF + Logistic Regression",
      metrics_file: "metrics.json",
    },
    model_manifests: {
      tfidf_logreg: { version: "tfidf-test" },
      distilbert: { version: "distilbert-test" },
    },
    ...overrides,
  };
}

async function mockPrediction(page: Page, prediction: DuelPrediction) {
  await page.route("**/v1/predict", async (route) => {
    if (route.request().method() !== "POST") {
      await route.continue();
      return;
    }

    await route.fulfill({
      status: 200,
      contentType: "application/json",
      headers: { "access-control-allow-origin": "*" },
      body: JSON.stringify(prediction),
    });
  });
}

async function analyze(page: Page) {
  await page.getByLabel("Message").fill(PHISHING_MESSAGE);
  await page.getByRole("button", { name: "Analyze", exact: true }).click();
}

test.beforeEach(async ({ page }) => {
  await page.goto("/");

  await expect(page.getByRole("heading", { name: "ThreatLens" })).toBeVisible();
  await expect(page.getByText("Model ready")).toBeVisible({ timeout: 120_000 });
});

test("keeps the real classifier as the only primary action", async ({ page }) => {
  await expect(page.getByRole("button")).toHaveCount(1);
  await expect(page.getByRole("button", { name: "Analyze", exact: true })).toBeVisible();

  for (const exampleName of ["ham", "phish", "spam"]) {
    await expect(page.getByRole("button", { name: exampleName, exact: true })).toHaveCount(0);
  }

  await expect(page.getByLabel("Message")).toBeVisible();
  await expect(page.getByText(/not stored/i).first()).toBeVisible();
});

test("always presents both model rows", async ({ page }) => {
  await expect(page.getByTestId("model-row-tfidf-logreg")).toContainText(
    /TF-IDF.*Logistic Regression/i,
  );
  await expect(page.getByTestId("model-row-distilbert")).toContainText(/DistilBERT/i);
});

test("runs the real TF-IDF flow and reports DistilBERT unavailable honestly", async ({
  page,
}) => {
  const textarea = page.getByLabel("Message");
  await textarea.fill(PHISHING_MESSAGE);
  await expect(textarea).toHaveValue(PHISHING_MESSAGE);

  const predictionResponsePromise = page.waitForResponse(
    (response) =>
      response.request().method() === "POST" &&
      new URL(response.url()).pathname === "/v1/predict",
  );

  await page.getByRole("button", { name: "Analyze", exact: true }).click();
  const predictionResponse = await predictionResponsePromise;
  expect(predictionResponse.ok()).toBe(true);

  await expect(page.getByTestId("prediction-label")).toHaveText("Phishing", {
    timeout: 60_000,
  });
  await expect(page.getByTestId("prediction-risk")).toHaveText("High risk");
  await expect(page.getByTestId("prediction-confidence")).toHaveText(PERCENTAGE);
  await expect(page.getByTestId("model-agreement")).toHaveText(/unavailable/i);

  const tfidfRow = page.getByTestId("model-row-tfidf-logreg");
  await expect(tfidfRow).toContainText("Phishing");
  await expect(tfidfRow.locator("td").last()).toHaveText(PERCENTAGE);

  const distilbertRow = page.getByTestId("model-row-distilbert");
  await expect(distilbertRow).toContainText(/unavailable/i);
  await expect(distilbertRow).not.toContainText(/\d+(?:\.\d+)?%/);

  await expect(page.getByTestId("prediction-explanation")).toContainText(
    /TF-IDF|trained model/i,
  );
  await expect(page.getByTestId("suggested-action")).toContainText(
    "Do not click links or share credentials",
  );
  await expect(page.getByTestId("model-artifact")).toContainText("tfidf_logreg.joblib");
});

test("shows agreement when both real model outputs select the same label", async ({
  page,
}) => {
  await mockPrediction(page, duelPrediction());
  await analyze(page);

  await expect(page.getByTestId("model-agreement")).toHaveText(/agree/i);
  await expect(page.getByTestId("model-row-tfidf-logreg")).toContainText("Phishing");
  await expect(page.getByTestId("model-row-distilbert")).toContainText("Phishing");
  await expect(page.getByTestId("uncertainty-warning")).toHaveCount(0);
});

test("shows disagreement and uses the DistilBERT result as the final recommendation", async ({
  page,
}) => {
  await mockPrediction(
    page,
    duelPrediction({
      final_label: "spam",
      final_risk_level: "medium",
      final_confidence: 0.72,
      agreement: "disagreed",
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
      explanation:
        "TF-IDF Logistic Regression selected phishing while DistilBERT selected spam.",
      suggested_action: "Treat as spam and send this disagreement for manual review.",
    }),
  );
  await analyze(page);

  await expect(page.getByTestId("model-agreement")).toHaveText(/disagree/i);
  await expect(page.getByTestId("model-row-tfidf-logreg")).toContainText("Phishing");
  await expect(page.getByTestId("model-row-distilbert")).toContainText("Spam");
  await expect(page.getByTestId("prediction-label")).toHaveText("Spam");
  await expect(page.getByTestId("prediction-confidence")).toHaveText("72%");
  await expect(page.getByTestId("suggested-action")).toContainText("Treat as spam");
  await expect(page.getByTestId("uncertainty-warning")).toContainText(/manual review/i);
});

test("shows a partial result when DistilBERT fails during inference", async ({ page }) => {
  await mockPrediction(
    page,
    duelPrediction({
      final_confidence: 0.91,
      agreement: "partial",
      model_outputs: {
        tfidf_logreg: TFIDF_PHISH,
        distilbert: {
          status: "error",
          label: null,
          confidence: null,
          probabilities: null,
          detail: "DistilBERT inference is temporarily unavailable.",
        },
      },
      explanation:
        "TF-IDF Logistic Regression classified this message as phishing. DistilBERT could not complete inference.",
    }),
  );
  await analyze(page);

  await expect(page.getByTestId("model-agreement")).toHaveText(/partial/i);
  await expect(page.getByTestId("prediction-label")).toHaveText("Phishing");
  await expect(page.getByTestId("model-row-distilbert")).toContainText(
    /temporarily unavailable/i,
  );
  await expect(page.getByTestId("model-row-distilbert")).not.toContainText(
    /\d+(?:\.\d+)?%/,
  );
});

test("shows a useful error when the TF-IDF model is temporarily unavailable", async ({
  page,
}) => {
  const errorDetail = "Model artifact is temporarily unavailable. Try again shortly.";

  await page.route("**/v1/predict", async (route) => {
    if (route.request().method() !== "POST") {
      await route.continue();
      return;
    }

    await route.fulfill({
      status: 503,
      contentType: "application/json",
      headers: { "access-control-allow-origin": "*" },
      body: JSON.stringify({ detail: errorDetail }),
    });
  });

  await analyze(page);

  await expect(page.getByRole("alert")).toContainText(errorDetail);
  await expect(page.getByRole("button", { name: "Analyze", exact: true })).toBeEnabled();
});

test("keeps analysis available after a transient readiness failure", async ({ page }) => {
  await page.route("**/v1/ready", async (route) => {
    await route.fulfill({
      status: 503,
      contentType: "application/json",
      headers: { "access-control-allow-origin": "*" },
      body: JSON.stringify({
        status: "error",
        model_loaded: false,
        model_path: "tfidf_logreg.joblib",
        detail: "Model artifact is temporarily unavailable.",
      }),
    });
  });

  await page.reload();
  await expect(page.getByText("Model offline")).toBeVisible();
  await page.getByLabel("Message").fill(PHISHING_MESSAGE);

  await expect(page.getByRole("button", { name: "Analyze", exact: true })).toBeEnabled();
});

test("does not create horizontal overflow at the tested viewport", async ({ page }) => {
  const hasHorizontalOverflow = await page.evaluate(
    () => document.body.scrollWidth > document.documentElement.clientWidth + 1,
  );

  expect(hasHorizontalOverflow).toBe(false);
});
