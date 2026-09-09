<p align="center">
  <img src="web/frontend/src/assets/threatlens-logo.png" alt="ThreatLens pixel art shield and lens logo" width="128" height="128" />
</p>

# ThreatLens

ThreatLens is a full-stack NLP project for classifying messages as `ham`, `phish`, or `spam`.
The main model is a fine-tuned DistilBERT classifier. A TF-IDF + Logistic Regression baseline is
kept as a fallback so the app can fail honestly instead of inventing transformer predictions.

The project started as an academic notebook workflow and was finalized as a small production-style
FastAPI + React application with artifact validation, model comparison, Docker support, and tests.

## What Is Included

- Fine-tuned DistilBERT inference artifact in `artifacts/distilbert/`
- TF-IDF + Logistic Regression baseline in `artifacts/tfidf_logreg.joblib`
- FastAPI backend with versioned prediction, readiness, and metadata routes
- React frontend for analyzing one message and comparing model outputs
- Docker Compose setup for running the full app
- Pytest, Ruff, ESLint, Playwright, and CI workflows
- Academic notebooks in `notebooks/`

Large model files are stored with Git LFS. After cloning, run:

```powershell
git lfs pull
```

## Models

| Model | Role | Artifact |
| --- | --- | --- |
| Fine-tuned DistilBERT | Primary classifier | `artifacts/distilbert/` |
| TF-IDF + Logistic Regression | Baseline and fallback | `artifacts/tfidf_logreg.joblib` |

DistilBERT test metrics from the exported artifact:

- Accuracy: `98.42%`
- Macro F1: `98.42%`

TF-IDF baseline metrics:

- Accuracy: `95.76%`
- Macro F1: `95.76%`

The backend validates model manifests, labels, checksums, and artifact structure before serving.
If DistilBERT cannot be loaded, the API reports that state and falls back to TF-IDF.

## Academic Deliverables

The notebooks are kept as the original academic project deliverables:

- `notebooks/01_classical_nlp_baselines.ipynb`: classical NLP preprocessing and baseline modeling
- `notebooks/02_transformer_finetuning.ipynb`: DistilBERT fine-tuning workflow

The application code is the finalized deployment layer built from that work.

## Run The App

The simplest path is Docker Compose:

```powershell
Copy-Item .env.example .env
docker compose up --build
```

Open:

- Web app: http://localhost:8080
- API readiness: http://localhost:8000/v1/ready

If you use [Task](https://taskfile.dev/), the same flow is:

```powershell
task app
```

## Local Development

Create a Python 3.12 virtual environment and install the dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,web,distilbert-runtime,distilbert-training]"
```

With Node.js 24 and Corepack installed, enable pnpm and install frontend dependencies:

```powershell
cd web\frontend
corepack enable
pnpm install --frozen-lockfile
```

Run the backend:

```powershell
$env:EMAIL_THREAT_MODEL_PATH = "artifacts\tfidf_logreg.joblib"
$env:EMAIL_THREAT_DISTILBERT_MODEL_PATH = "artifacts\distilbert"
.\.venv\Scripts\python.exe -m uvicorn web.backend.main:app --reload --host 127.0.0.1 --port 8000
```

Run the frontend:

```powershell
cd web\frontend
$env:VITE_API_BASE_URL = "http://127.0.0.1:8000"
pnpm dev
```

Task shortcuts:

```powershell
task backend
task frontend
task test
task lint
task build
task e2e
task clean
```

## API

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/v1/live` | Process liveness |
| `GET` | `/v1/ready` | Model readiness for DistilBERT and TF-IDF |
| `GET` | `/v1/metadata` | Public model metadata and metrics |
| `POST` | `/v1/predict` | Message classification |

Compatibility routes remain available at `/live`, `/health`, `/metadata`, and `/predict`.

## Checks

Run backend checks:

```powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m pytest --basetemp "$env:TEMP\threatlens-pytest"
```

Run frontend checks:

```powershell
cd web\frontend
pnpm lint
$env:VITE_API_BASE_URL = "http://127.0.0.1:8000"
pnpm build
pnpm test:e2e
```

Run Docker smoke checks:

```powershell
docker compose config --quiet
docker compose up --build
```

Then inspect `http://localhost:8000/v1/ready` and confirm both model rows are available.

## Project Map

```text
src/email_threat_detector/    Training and preprocessing code
web/backend/                  FastAPI app and model runtimes
web/frontend/                 React interface and Playwright tests
notebooks/                    Academic notebook deliverables
artifacts/distilbert/         Final DistilBERT inference artifact
artifacts/tfidf_logreg.joblib TF-IDF baseline artifact
tests/                        Backend and artifact tests
```

## Limitations

ThreatLens is a demo classifier, not an unattended mail gateway. Predictions are advisory, model
confidence is not calibrated, and suspicious messages should still be reviewed by a human.
