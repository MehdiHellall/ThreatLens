# ThreatLens

ThreatLens is an analyst-facing web classifier for `ham`, `phish`, and `spam`. It runs a real,
artifact-backed TF-IDF + Logistic Regression model and can optionally run a real fine-tuned
DistilBERT model beside it in Model Duel Mode. If the transformer artifact is absent or invalid,
the API and UI say so and continue with TF-IDF; they never synthesize a transformer prediction.

The first screen stays focused on the classifier: readiness, one message input, one Analyze action,
the two model rows, agreement state, final recommendation, explanation, suggested action, compact
metadata, and useful errors. There are no example chips, marketing sections, or simulated claims.

## Models and artifacts

The checked-in model is `artifacts/tfidf_logreg.joblib`.

| Model | Artifact policy | Availability |
| --- | --- | --- |
| TF-IDF + Logistic Regression | Small joblib artifact is checked in | Required; readiness is based on it |
| Fine-tuned DistilBERT | Hugging Face directory or verified remote ZIP | Optional; unavailable until configured |

The current TF-IDF test metrics are `95.76%` accuracy and `95.76%` macro F1. The backend reads
them from `web/backend/assets/tfidf_logreg_metrics.json`.

Transformer weights are intentionally ignored by Git. Do not commit a generated
`artifacts/distilbert/` directory unless the repository is deliberately configured for Git LFS.
For normal deployment, keep the directory in secure artifact storage or publish the optional
checksum-labelled ZIP and configure its trusted HTTPS URL and SHA-256 digest.

## Repository map

```text
src/email_threat_detector/
  preprocessing.py           Shared text normalization
  distilbert_training.py     Leakage-safe prepare/train/export CLI
web/backend/
  settings.py                Environment-backed runtime configuration
  schemas.py                 Versioned and compatibility API schemas
  security.py                Request guards, rate limiting, and security headers
  sklearn_runtime.py         TF-IDF artifact loading and inference
  runtimes/
    distilbert_runtime.py    Validated, lazy Hugging Face inference
  explanations.py            Risk, explanation, and action policy
  main.py                    FastAPI composition and route definitions
web/frontend/                React duel interface and Playwright E2E tests
artifacts/tfidf_logreg.joblib Required sklearn artifact
tests/                       Data, runtime, API, and artifact regression tests
```

Neither backend runtime trains a model. TF-IDF and DistilBERT are loaded from completed artifacts.

## Setup

Use Python `>=3.12,<3.13`. The checked-in TF-IDF artifact is serialized for the
Python 3.12 deployment stack.

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,web,distilbert-runtime,distilbert-training]"
cd web\frontend
pnpm install --frozen-lockfile
```

The `distilbert-runtime` group is sufficient for serving an exported transformer. The
`distilbert-training` group adds the data and Hugging Face Trainer dependencies used by the CLI.

## Fine-tune and export DistilBERT

The CLI expects a CSV with `text` and `label` columns. Labels may be `ham`, `phish`, and `spam` or
their numeric IDs `0`, `1`, and `2`.

Prepare deterministic leakage-safe splits:

```powershell
.\.venv\Scripts\python.exe -m email_threat_detector.distilbert_training prepare `
  --input <source.csv> `
  --output-dir data\processed\transformer_full
```

Preparation normalizes labels and text, removes empty records, removes every normalized-text group
with conflicting labels, deduplicates normalized text, balances the three classes, creates
stratified 80/10/10 train/validation/test splits with seed 42, and verifies zero normalized-text
overlap. It writes the split CSVs plus a checksum manifest. Use `--samples-per-class N` to make a
small deterministic smoke dataset before the full run.

Older project split directories that predate per-file checksums can be adopted explicitly without
regenerating them:

```powershell
.\.venv\Scripts\python.exe -m email_threat_detector.distilbert_training adopt `
  --splits-dir data\processed\transformer_full
```

`adopt` does not trust filenames alone. It validates the legacy provenance/configuration, canonical
labels and text, reported and actual class counts, balanced 80/10/10 proportions, duplicate-free
normalized text within each split, and zero cross-split overlap before atomically adding hashes.

Train and export:

```powershell
.\.venv\Scripts\python.exe -m email_threat_detector.distilbert_training train `
  --splits-dir data\processed\transformer_full `
  --output-dir artifacts\distilbert `
  --work-dir artifacts\distilbert-training `
  --resume
```

`--resume` selects the latest complete `checkpoint-<step>` in the work directory. Re-running the
same command continues an interrupted job only when its immutable `run_contract.json` matches the
split-manifest checksum, pinned base revision, and every hyperparameter. Incomplete checkpoints are
ignored. The trainer mirrors `notebooks/02_transformer_finetuning.ipynb`:

- `distilbert/distilbert-base-uncased` pinned to revision
  `12040accade4e8a0f71eabdb258fecc2e7e948be`, fast tokenizer, truncation, and maximum length 128
- labels `{0: ham, 1: phish, 2: spam}`
- 3 epochs, batch size 16, learning rate `2e-5`, weight decay `0.01`, and seed 42
- evaluation and saving each epoch, best-model reload, and one retained checkpoint
- FP16 only when CUDA is available; CPU training uses FP32

The full 102,810-row training split takes a long time on CPU. Run the small preparation/training
smoke first, preserve the work directory between sessions, and use `--resume` after interruption.
Training needs network access the first time the base checkpoint is downloaded; inference does not.

The completed `artifacts/distilbert/` directory contains Hugging Face config, safetensors weights,
tokenizer files, `metrics.json`, and `manifest.json`. Metrics include test/per-class results,
confusion matrix, runtime, and training configuration. The manifest records completion, exact label
mapping, pinned base revision, max length, split hashes, dependency versions, provenance, hashes of
the exported files, and a stable fingerprint over that file inventory.

To build the optional remote-hosting archive from a completed export:

```powershell
.\.venv\Scripts\python.exe -m email_threat_detector.distilbert_training archive `
  --artifact-dir artifacts\distilbert `
  --output-dir artifacts\archives
```

The ZIP filename includes its SHA-256 digest, and the command writes a `.zip.sha256` sidecar. Keep
the digest with the deployment configuration and publish the archive only to a trusted HTTPS
endpoint.

## Runtime configuration

Start TF-IDF-only inference:

```powershell
$env:EMAIL_THREAT_MODEL_PATH = "artifacts\tfidf_logreg.joblib"
.\.venv\Scripts\python.exe -m uvicorn web.backend.main:app --reload --host 127.0.0.1 --port 8000
```

For a local fine-tuned transformer, set:

```powershell
$env:EMAIL_THREAT_DISTILBERT_MODEL_PATH = "artifacts\distilbert"
$env:EMAIL_THREAT_DISTILBERT_MAX_LENGTH = "128"
```

For a remotely hosted export, leave the local path unset and set all remote values:

```powershell
Remove-Item Env:EMAIL_THREAT_DISTILBERT_MODEL_PATH -ErrorAction SilentlyContinue
$env:EMAIL_THREAT_DISTILBERT_MODEL_URL = "https://example.com/models/threatlens-distilbert-<sha256>.zip"
$env:EMAIL_THREAT_DISTILBERT_MODEL_SHA256 = "<64-character-sha256>"
$env:EMAIL_THREAT_DISTILBERT_MODEL_CACHE_PATH = ".cache\threatlens\distilbert"
```

DistilBERT settings:

- `EMAIL_THREAT_DISTILBERT_MODEL_PATH`: local exported Hugging Face directory
- `EMAIL_THREAT_DISTILBERT_MODEL_URL`: optional HTTPS ZIP; mutually exclusive with the local path
- `EMAIL_THREAT_DISTILBERT_MODEL_SHA256`: required exact digest for a remote ZIP
- `EMAIL_THREAT_DISTILBERT_MODEL_CACHE_PATH`: verified remote download/extraction cache
- `EMAIL_THREAT_DISTILBERT_MAX_LENGTH`: defaults to 128 and must match the artifact manifest

The runtime verifies the complete exporter schema, pinned revision, model type, three-label mapping,
maximum length, provenance, metrics, fingerprint, and every declared artifact hash before loading.
Remote archives are stored beneath SHA-256-named children of the configured cache root, size-limited,
checksum-verified before extraction, and rejected if unsafe members try to escape the cache.
HTTP redirects are not followed; configure the final trusted HTTPS artifact URL directly.
Resolved model/tokenizer files are loaded locally. A raw base DistilBERT checkpoint or incomplete
training directory is rejected rather than treated as a trained classifier.

Existing TF-IDF controls remain available: `EMAIL_THREAT_MODEL_PATH`,
`EMAIL_THREAT_MODEL_URL`, `EMAIL_THREAT_MODEL_SHA256`, `EMAIL_THREAT_MODEL_CACHE_PATH`,
`EMAIL_THREAT_MODEL_RETRY_SECONDS`, and `EMAIL_THREAT_MAX_ARTIFACT_BYTES`. Only load artifacts from
trusted sources: joblib uses pickle internally, and model weights remain executable supply-chain
inputs even when their hashes match.

## API and Duel Mode

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/v1/live` | Process liveness; does not require a model |
| `GET` | `/v1/ready` | TF-IDF readiness plus `duel_ready` and both model states |
| `GET` | `/v1/metadata` | Sanitized metadata/manifests for exactly two models |
| `POST` | `/v1/predict` | TF-IDF inference and optional real DistilBERT inference |

`POST /v1/predict` returns `final_label`, `final_risk_level`, `final_confidence`, `agreement`,
exactly two entries under `model_outputs` (`tfidf_logreg` and `distilbert`), `explanation`,
`suggested_action`, and `model_manifests`. Each model row has a status and nullable prediction
fields, so an unavailable/error row never contains invented probabilities. The existing TF-IDF
`artifact_metadata` field remains for compatibility.

Duel policy is explicit:

| Condition | Agreement | Final decision |
| --- | --- | --- |
| Both models succeed with the same label | `agreed` | DistilBERT result |
| Both succeed with different labels | `disagreed` | DistilBERT result plus manual-review warning |
| DistilBERT is unconfigured, invalid, dependency-blocked, or cannot load | `unavailable` | TF-IDF fallback |
| A loaded DistilBERT fails for that request | `partial` | TF-IDF fallback |
| TF-IDF cannot infer | N/A | Existing `503` behavior |

The explanation names both outcomes and never averages or directly compares the models'
uncalibrated confidence values. Risk and suggested action are based on the selected final model.
Submitted message text is used only for that request and is not logged or stored by either runtime.

Compatibility endpoints remain: `/live`, `/health`, `/metadata`, and `/predict`. In particular,
legacy `/predict` remains TF-IDF-only, `/health` keeps its TF-IDF-focused response, and legacy
`/metadata` keeps its previous shape. `/v1/ready` stays HTTP 200 whenever TF-IDF inference works,
even if Duel Mode is unavailable.

## Frontend

```powershell
cd web\frontend
$env:VITE_API_BASE_URL = "http://127.0.0.1:8000"
pnpm dev
```

Open [http://127.0.0.1:5173/](http://127.0.0.1:5173/). The comparison table always shows permanent
TF-IDF and DistilBERT rows. It shows availability, labels, confidence, probabilities, agreement,
fallback state, final recommendation, and a clear warning when the models disagree.

## Docker

```powershell
Copy-Item .env.example .env
docker compose up --build
```

The backend image installs the DistilBERT runtime dependencies but does not bake generated
transformer weights into the image. Compose mounts `./artifacts` at `/app/artifacts` read-only, so
placing a valid export in `artifacts/distilbert/` makes it available to the container. With no valid
export, TF-IDF remains ready and the transformer row reports unavailable.

- Web app: [http://localhost:8080](http://localhost:8080)
- API readiness: [http://localhost:8000/v1/ready](http://localhost:8000/v1/ready)

Healthchecks call `/v1/live`; use `/v1/ready` to inspect inference and Duel readiness.

## Checks

Run Python tests with coverage and Ruff from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest --cov-fail-under=80
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
```

Run frontend lint/typecheck/build and Playwright:

```powershell
cd web\frontend
pnpm install --frozen-lockfile
pnpm lint
$env:VITE_API_BASE_URL = "http://127.0.0.1:8000"
pnpm build
pnpm test:e2e:install
pnpm test:e2e
```

The production build runs TypeScript checking. Playwright covers TF-IDF-only, unavailable,
agreement, disagreement, partial failure, mobile, and error states.

If the Docker daemon is healthy, validate and smoke-test the composed services:

```powershell
docker compose config --quiet
docker compose build
docker compose up -d
Invoke-WebRequest http://localhost:8000/v1/ready
Invoke-RestMethod -Method Post -Uri http://localhost:8000/v1/predict `
  -ContentType "application/json" -Body '{"text":"Urgent password reset required verify account."}'
Invoke-WebRequest http://localhost:8080
docker compose down
```

## Limitations

Predictions are advisory. Model disagreement is a review signal, not an invitation to hide one
output or average incompatible probabilities. Do not use ThreatLens as an unattended mail gateway,
and do not load artifacts from untrusted publishers. The displayed confidence is raw model output,
not a calibrated probability, and the low/medium/high risk mapping uses an analyst-facing heuristic
rather than a validated calibration threshold.
