<p align="center">
  <img src="web/frontend/src/assets/threatlens-logo.png" alt="ThreatLens pixel shield" width="96" height="96" />
</p>

# ThreatLens

ThreatLens classifies a message as `ham`, `phish`, or `spam` with a fine-tuned
DistilBERT model and an independently loaded TF-IDF baseline. It is a focused demo and review
tool: results are advisory, scores are not calibrated probabilities, and suspicious messages
should still be reviewed by a person.

## Fastest Path on This Ubuntu Machine

The application runs in Docker, so its Python 3.12 and Node 24 environments do not change your
host Python installation. Allow at least 10 GiB free disk and roughly 4 GiB available memory for
the first build and two loaded models.

### 1. Install Docker Engine and Git LFS once

These commands target the detected Ubuntu 26.04 (`resolute`) amd64 installation and follow
[Docker's official Ubuntu instructions](https://docs.docker.com/engine/install/ubuntu/):

```bash
sudo apt update
sudo apt install ca-certificates curl git-lfs

sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

sudo tee /etc/apt/sources.list.d/docker.sources >/dev/null <<'EOF'
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: resolute
Components: stable
Architectures: amd64
Signed-By: /etc/apt/keyrings/docker.asc
EOF

sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io \
  docker-buildx-plugin docker-compose-plugin
sudo systemctl enable --now docker
sudo docker run --rm hello-world
```

For Docker commands without `sudo`:

```bash
sudo usermod -aG docker "$USER"
```

Then **sign out of Linux completely and sign back in**. Membership in the `docker` group grants
root-equivalent access; keep using `sudo docker ...` instead if you do not want that membership.
See [Docker's Linux post-install guide](https://docs.docker.com/engine/install/linux-postinstall/).

Verify the new session:

```bash
docker version
docker compose version
docker run --rm hello-world
```

### 2. Retrieve and validate the real models

From this repository:

```bash
git lfs install --local
git lfs pull
PYTHONPATH=src python3 -m email_threat_detector.artifact_validation --root .
```

The validator rejects unresolved LFS pointers, unexpected baseline bytes, and every DistilBERT
size, checksum, or manifest mismatch before any model is deserialized.

### 3. Start the verified demo

No Task installation is required:

```bash
python3 scripts/threatlens.py doctor
python3 scripts/threatlens.py demo
```

Open <http://127.0.0.1:8080>. The launcher waits up to three minutes for **both** real models and
prints useful logs if startup fails. Later starts can skip rebuilding unchanged images:

```bash
python3 scripts/threatlens.py demo --no-build
```

Everyday commands:

```bash
python3 scripts/threatlens.py logs       # follow logs
python3 scripts/threatlens.py down       # stop, preserving models and cache
python3 scripts/threatlens.py clean      # safe cleanup preview
python3 scripts/threatlens.py clean --execute
```

If [Task](https://taskfile.dev/) is already installed, the shorter equivalents are `task doctor`,
`task demo`, `task logs`, `task down`, `task clean-dry-run`, and `task clean`. `task app` remains an
alias for `task demo`.

Direct Docker fallback:

```bash
docker compose config --quiet
docker compose up --build -d
docker compose logs -f
docker compose down
```

## Share a Demo for Free

The supported public-demo path uses your existing machine and Tailscale Funnel. It costs nothing
for eligible personal, noncommercial use, but the URL works only while this machine, Docker, and
Tailscale are online.

1. Install Tailscale using its [official Linux instructions](https://tailscale.com/download/linux),
   then authenticate:

   ```bash
   curl -fsSL https://tailscale.com/install.sh | sh
   sudo tailscale up
   ```

2. Start ThreatLens, then enable its dedicated HTTPS Funnel:

   ```bash
   python3 scripts/threatlens.py demo --no-build
   python3 scripts/threatlens.py share
   ```

   The first Funnel invocation may ask you to enable Funnel in the Tailscale admin page. The
   command uses TLS termination plus PROXY protocol v2 so the app receives the visitor's source
   address without trusting browser-supplied forwarding headers.

3. Copy the `https://...ts.net` address printed by Tailscale. When the demo is over:

   ```bash
   python3 scripts/threatlens.py unshare
   python3 scripts/threatlens.py down
   ```

`unshare` removes only the ThreatLens mapping; it does not reset other Tailscale configuration.
Review existing mappings with `tailscale funnel status` before changing them.

## Architecture and API

```text
Browser :8080 ── /api ──> Nginx ──> FastAPI :8000 ──> DistilBERT
                            │                           TF-IDF fallback
                            └── React static app
```

Only loopback frontend ports are published by default. Nginx provides the same-origin `/api`
proxy, security headers, and a separate loopback-only Funnel listener. Model artifacts are mounted
read-only; remote artifact caches use a persistent Docker volume.

| Method | Browser route | Purpose |
| --- | --- | --- |
| `GET` | `/api/v1/live` | Process liveness |
| `GET` | `/api/v1/ready` | Service, primary-model, and two-model readiness |
| `GET` | `/api/v1/metadata` | Public model provenance and metrics |
| `POST` | `/api/v1/predict` | Independent two-model classification with fallback |

FastAPI retains its direct `/v1/*` and legacy compatibility routes inside the Compose network and
when run locally for development.

## Configuration

Defaults work without a `.env` file. To customize remote artifact URLs, limits, or retry settings:

```bash
cp .env.example .env
```

Use either a local model path or an HTTPS URL with its SHA-256, never both. To choose a remote
DistilBERT archive, set `EMAIL_THREAT_DISTILBERT_MODEL_PATH=` to an explicitly empty value and set
the URL plus checksum. Never commit `.env`, secrets, or machine-specific paths.

## Local Development and Checks

For non-Docker work, use the committed Python and frontend lockfiles:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --frozen --extra dev --extra web --extra distilbert-runtime
corepack enable
cd web/frontend && pnpm install --frozen-lockfile && cd ../..
```

Then use:

```bash
python3 scripts/threatlens.py backend     # terminal 1
python3 scripts/threatlens.py frontend    # terminal 2
python3 scripts/threatlens.py test
python3 scripts/threatlens.py lint
python3 scripts/threatlens.py format-check
python3 scripts/threatlens.py build
python3 scripts/threatlens.py e2e
```

Python coverage for `web/backend` must remain at least 80%. Frontend dependencies are installed
from the frozen pnpm lockfile.

## Repository Map

```text
src/email_threat_detector/    Training, preprocessing, and artifact validation
web/backend/                  FastAPI service and model runtimes
web/frontend/                 React/Vite UI and Playwright tests
scripts/threatlens.py         Doctor, demo, sharing, and safe cleanup commands
tests/                        Backend, artifact, and training tests
artifacts/                    Checksum-bound model exports stored with Git LFS
notebooks/                    Original academic deliverables
```

The exported model manifest records a balanced held-out dataset result of approximately 98.42%
accuracy/macro F1 for DistilBERT; the baseline metadata records approximately 95.76%. These are
dataset-specific historical measurements, not guarantees for new messages.

### Model behavior and demo guardrails

Natural-language smoke probes also expose an important out-of-distribution limitation: common
credential lures can be classified as `spam`, and a high-scoring `ham` result is possible when the
two models disagree. The versioned API never rewrites those model labels. It displays both outputs
and separately escalates credential requests combined with urgency or a contact prompt to manual
review, with conservative handling guidance. Treat this as a transparent demo and review aid, not
an automated mail-security gateway.

Before making production-quality claims, evaluate the frozen artifacts on a separately sourced,
recent, deduplicated phishing set and retrain or recalibrate against the failures. Keep that external
set out of training and model-selection decisions.

## Troubleshooting

- **Docker permission denied:** sign out/in after `usermod`, or prefix Docker commands with `sudo`.
- **Model is a Git LFS pointer:** run `git lfs pull`, then rerun the artifact validator.
- **Port 8080 or 8081 busy:** stop the owning service before starting this Compose project.
- **Model startup exceeds 180 seconds:** inspect `python3 scripts/threatlens.py logs`; close memory-heavy
  applications and confirm at least 4 GiB is available.
- **Public URL is offline:** confirm the computer is awake, `docker compose ps` is healthy, and
  `tailscale funnel status` lists the ThreatLens mapping.
