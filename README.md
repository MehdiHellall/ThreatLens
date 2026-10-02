<p align="center">
  <img
    src="web/frontend/src/assets/threatlens-logo.png"
    alt="ThreatLens shield logo"
    width="96"
    height="96"
  />
</p>

# ThreatLens

ThreatLens is a polished email and message risk demo. Paste a message, run a live analysis, and
see how two real models classify it as `ham`, `spam`, or `phish`.

The app is built for a portfolio demo: one command starts the full stack, the UI is responsive, and
the result explains model disagreement instead of hiding it.

## What It Shows

- Real inference with a fine-tuned DistilBERT model and a TF-IDF Logistic Regression baseline.
- A premium React interface for testing messages quickly.
- Side-by-side model outputs, confidence, disagreement, fallback state, and review warnings.
- Safe demo samples that fill the editor without faking results.
- Dockerized FastAPI, Nginx, and React/Vite deployment.

## Tech Stack

| Area | Tools |
| --- | --- |
| Frontend | React, TypeScript, Vite, Playwright |
| Backend | FastAPI, Pydantic, Uvicorn |
| ML | DistilBERT, scikit-learn TF-IDF baseline |
| Deployment | Docker Compose, Nginx |
| Quality | Pytest, Ruff, ESLint, artifact checksum validation |

## Quick Demo

Prerequisites:

- Docker Engine with the Compose plugin
- Git LFS
- Python 3.10+ available as `python3`

If Docker is not installed, use the official guide:
<https://docs.docker.com/engine/install/ubuntu/>.

Run one command from the repository root:

```bash
python3 scripts/threatlens.py demo
```

Open:

```text
http://127.0.0.1:8080
```

Stop the app:

```bash
python3 scripts/threatlens.py down
```

The demo helper checks Docker, pulls Git LFS model artifacts when needed, builds the containers,
waits for both models, and prints the local URL.

## Demo Flow

1. Open the app.
2. Click `Account alert`.
3. Click `Analyze message`.
4. Show the verdict, confidence, model agreement/disagreement, recommended action, and details
   panel.

The most important product behavior: ThreatLens does not pretend the model is always right. If the
models disagree or the text contains credential/urgency signals, the UI asks for manual review.

## Architecture

```text
Browser :8080
  └── Nginx
        ├── React static app
        └── /api -> FastAPI :8000
                    ├── DistilBERT classifier
                    └── TF-IDF baseline
```

The Docker setup publishes only loopback ports by default:

- App: `http://127.0.0.1:8080`
- API through the frontend proxy: `/api/v1/*`
- Backend inside Docker: `:8000`

## API

| Route | Purpose |
| --- | --- |
| `GET /api/v1/live` | Process liveness |
| `GET /api/v1/ready` | Model readiness |
| `GET /api/v1/metadata` | Public model metadata |
| `POST /api/v1/predict` | Two-model message classification |

## Local Development

Install Python dependencies:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --frozen --extra dev --extra web --extra distilbert-runtime
```

Install frontend dependencies:

```bash
corepack enable
cd web/frontend
pnpm install --frozen-lockfile
cd ../..
```

Useful commands:

```bash
python3 scripts/threatlens.py backend       # FastAPI on :8000
python3 scripts/threatlens.py frontend      # Vite on :5173
python3 scripts/threatlens.py test          # Pytest + coverage
python3 scripts/threatlens.py lint          # Ruff + ESLint
python3 scripts/threatlens.py build         # Frontend production build
python3 scripts/threatlens.py e2e           # Playwright tests
python3 scripts/threatlens.py doctor        # Diagnose Docker/models
python3 scripts/threatlens.py clean         # Preview disposable cleanup
```

## Project Structure

```text
src/email_threat_detector/    Training, preprocessing, artifact validation
web/backend/                  FastAPI service and model runtimes
web/frontend/                 React/Vite client
web/frontend/e2e/             Playwright browser tests
scripts/threatlens.py         Demo, dev, cleanup, and sharing helper
tests/                        Backend and model regression tests
artifacts/                    Git LFS model artifacts
notebooks/                    Original experimentation notebooks
```

## Model Notes

The packaged DistilBERT export records about 98.42% accuracy and macro F1 on its held-out test
split. The TF-IDF baseline records about 95.76%.

Those metrics are dataset-specific. ThreatLens is a transparent portfolio demo and review aid, not
a production mail-security gateway.

## Optional Public Sharing

To share the local demo temporarily, install and sign in to Tailscale, then run:

```bash
python3 scripts/threatlens.py demo --no-build
python3 scripts/threatlens.py share
```

When finished:

```bash
python3 scripts/threatlens.py unshare
python3 scripts/threatlens.py down
```
