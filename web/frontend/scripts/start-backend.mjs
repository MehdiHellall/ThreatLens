import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(scriptDir, "../../..");
const backendPort = process.env.E2E_BACKEND_PORT ?? "8787";
const pythonCandidates = [
  path.join(repoRoot, ".venv", "Scripts", "python.exe"),
  path.join(repoRoot, ".venv", "bin", "python"),
  "python",
];
const python = pythonCandidates.find((candidate) => candidate === "python" || existsSync(candidate));
const env = {
  ...process.env,
  EMAIL_THREAT_ALLOWED_ORIGINS: `http://127.0.0.1:${process.env.E2E_FRONTEND_PORT ?? "5174"}`,
};

if (!env.EMAIL_THREAT_MODEL_PATH && !env.EMAIL_THREAT_MODEL_URL) {
  env.EMAIL_THREAT_MODEL_PATH = path.join(repoRoot, "artifacts", "tfidf_logreg.joblib");
}

if (
  !env.EMAIL_THREAT_DISTILBERT_MODEL_PATH &&
  !env.EMAIL_THREAT_DISTILBERT_MODEL_URL
) {
  env.EMAIL_THREAT_DISTILBERT_MODEL_PATH = path.join(repoRoot, "artifacts", "distilbert");
}

const child = spawn(
  python,
  ["-m", "uvicorn", "web.backend.main:app", "--host", "127.0.0.1", "--port", backendPort],
  {
    cwd: repoRoot,
    env,
    stdio: "inherit",
  },
);

for (const signal of ["SIGINT", "SIGTERM"]) {
  process.on(signal, () => child.kill(signal));
}

child.on("exit", (code, signal) => {
  if (signal) {
    process.kill(process.pid, signal);
    return;
  }
  process.exit(code ?? 0);
});
