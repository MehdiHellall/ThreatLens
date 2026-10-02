# Repository Guidelines

## Project Structure & Module Organization

ThreatLens uses Python 3.12, FastAPI, React, and Vite. Training code lives in
`src/email_threat_detector/`, the API in `web/backend/`, and the client in
`web/frontend/src/`. Playwright scenarios are in `web/frontend/e2e/`, backend tests in `tests/`,
model exports in `artifacts/`, and academic work in `notebooks/`. Do not commit generated output.

## Build, Test, and Development Commands

Commands in `Taskfile.yml` are the preferred entry points and use the portable
`scripts/threatlens.py` helper:

- `task doctor` validates Docker, Git LFS, configuration, resources, and model integrity.
- `task demo` builds the stack, waits for both real models, and prints the local URL;
  `task app` is its compatibility alias.
- `task backend` runs FastAPI with reload on port 8000.
- `task frontend` starts Vite on port 5173.
- `task test` runs the Python test suite and coverage gate.
- `task lint` runs Ruff plus ESLint; `task format-check` checks Python formatting.
- `task build` produces the frontend bundle; `task e2e` runs Playwright.
- `task clean-dry-run` previews safe cleanup; `task clean` removes only allowlisted caches.

`task demo` is the full container smoke test. Verify `http://127.0.0.1:8080` and its
`/api/v1/ready` endpoint.

## Coding Style & Naming Conventions

Python uses 4-space indentation, Ruff formatting, double quotes, a 100-character line limit, and `snake_case` names; classes use `PascalCase`. TypeScript/React uses 2-space indentation, ESLint, `PascalCase` components, and `camelCase` functions and variables. Keep API schemas typed and preserve the versioned `/v1/*` routes. Never expose local artifact paths or internal exceptions in public responses.

## Testing Guidelines

Pytest discovers `tests/test_*.py`; add regression tests beside the affected backend behavior. Coverage targets `web/backend` and must remain at or above 80%. Playwright tests live in `web/frontend/e2e/*.spec.ts`; prefer accessible roles, labels, and stable `data-testid` selectors. Test normal predictions, model disagreement, fallback behavior, validation, and unavailable-model states.

## Commit & Pull Request Guidelines

Recent history favors short, imperative subjects, often with Conventional Commit prefixes such as `feat:`, `fix:`, `refactor:`, and `chore:`. Keep each commit focused. Pull requests should explain the user-visible effect, list verification commands, link relevant issues, and include screenshots for UI changes. Call out model-artifact, metric, environment, or deployment changes explicitly.

## Security & Configuration

Defaults require no `.env`; copy `.env.example` only for overrides. Do not commit secrets or
machine-specific paths. Treat model artifacts as untrusted input, retain checksum validation, and
review CORS, request-size, proxy-trust, and rate-limit settings when changing deployment defaults.
