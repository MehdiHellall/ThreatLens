#!/usr/bin/env python3
"""Portable development and demo commands for ThreatLens."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "web" / "frontend"
DEMO_URL = "http://127.0.0.1:8080"
READY_URL = f"{DEMO_URL}/api/v1/ready"
FUNNEL_TARGET = "tcp://127.0.0.1:8081"
BASELINE_SHA256 = "d9ed306935e26c9bf7b285a861991bb3539be7089ad9d8bf798d781d01d45981"
LFS_ARTIFACT_PATHS = (
    "artifacts/tfidf_logreg.joblib",
    "artifacts/distilbert/model.safetensors",
    "artifacts/distilbert/tokenizer.json",
    "artifacts/distilbert/training_args.bin",
)
SAFE_CLEAN_PATHS = (
    ".coverage",
    ".pytest_cache",
    ".pytest-tmp",
    ".ruff_cache",
    "htmlcov",
    "web/frontend/dist",
    "web/frontend/playwright-report",
    "web/frontend/test-results",
)
CLEAN_SCAN_EXCLUDED_PARTS = frozenset({".git", ".venv", "node_modules"})


class CommandError(RuntimeError):
    """A user-actionable command failure."""


def _run(
    command: Sequence[str],
    *,
    cwd: Path = ROOT,
    check: bool = True,
    capture: bool = False,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=check,
        text=True,
        capture_output=capture,
        env=env,
    )


def _require_command(name: str, recovery: str) -> None:
    if shutil.which(name) is None:
        raise CommandError(f"Missing `{name}`. {recovery}")


def _python() -> str:
    candidates = (
        ROOT / ".venv" / "bin" / "python",
        ROOT / ".venv" / "Scripts" / "python.exe",
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return sys.executable


def _pnpm() -> str:
    command = shutil.which("pnpm")
    if command is None:
        raise CommandError("Missing `pnpm`. Enable Corepack, then run `corepack install`.")
    return command


def _available_memory_gib() -> float | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.is_file():
        return None
    values = {}
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        key, _, raw_value = line.partition(":")
        if raw_value:
            values[key] = int(raw_value.strip().split()[0])
    available_kib = values.get("MemAvailable")
    return available_kib / 1024 / 1024 if available_kib is not None else None


def _port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            listener.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _validate_artifacts() -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from email_threat_detector.artifact_validation import (  # noqa: PLC0415
        ArtifactValidationError,
        validate_artifacts,
    )

    try:
        result = validate_artifacts(ROOT)
    except ArtifactValidationError as exc:
        raise CommandError(str(exc)) from exc
    print(
        "[ok] Model artifacts: "
        f"TF-IDF {result.baseline_sha256[:12]}…, "
        f"DistilBERT {result.distilbert_fingerprint[:12]}…"
    )


def _is_lfs_pointer(path: Path) -> bool:
    try:
        header = path.read_bytes()[:128]
    except OSError:
        return False
    return header.startswith(b"version https://git-lfs.github.com/spec/v1")


def _artifacts_need_lfs_pull() -> bool:
    for relative_path in LFS_ARTIFACT_PATHS:
        path = ROOT / relative_path
        if not path.exists() or _is_lfs_pointer(path):
            return True
    return False


def _ensure_lfs_artifacts() -> None:
    if not _artifacts_need_lfs_pull():
        return
    print("[info] Fetching model artifacts with Git LFS...")
    _run(("git", "lfs", "install", "--local"), capture=True)
    _run(("git", "lfs", "pull"))


def doctor(*, require_artifacts: bool = True) -> None:
    """Check the host before doing an expensive Docker build."""
    _require_command("docker", "Install Docker Engine using the README instructions.")
    _require_command("git", "Install Git and Git LFS, then run `git lfs pull`.")
    try:
        _run(("git", "lfs", "version"), capture=True)
    except subprocess.CalledProcessError as exc:
        raise CommandError(
            "Git LFS is unavailable. Install `git-lfs`, then run `git lfs pull`."
        ) from exc

    checks = (
        (("docker", "compose", "version"), "Docker Compose plugin"),
        (("docker", "info"), "Docker daemon access"),
        (("docker", "compose", "config", "--quiet"), "Compose configuration"),
    )
    for command, label in checks:
        try:
            _run(command, capture=True)
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "unknown error").strip()
            raise CommandError(f"{label} check failed: {detail}") from exc
        print(f"[ok] {label}")

    if require_artifacts:
        _ensure_lfs_artifacts()
        _validate_artifacts()

    disk_free = shutil.disk_usage(ROOT).free / 1024**3
    memory_free = _available_memory_gib()
    print(f"[info] Free disk: {disk_free:.1f} GiB")
    if disk_free < 10:
        print("[warn] Less than 10 GiB is free; the first image build may run out of space.")
    if memory_free is not None:
        print(f"[info] Available memory: {memory_free:.1f} GiB")
        if memory_free < 4:
            print("[warn] Less than 4 GiB is currently available; close heavy applications first.")
    for port in (8080, 8081):
        if not _port_available(port):
            print(f"[info] Port {port} is already in use (possibly by this Compose project).")


def _ready_payload() -> dict[str, object]:
    with urllib.request.urlopen(READY_URL, timeout=5) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("readiness response is not an object")
    return payload


def wait_for_demo(timeout_seconds: int) -> dict[str, object]:
    deadline = time.monotonic() + timeout_seconds
    last_detail = "service has not responded"
    while time.monotonic() < deadline:
        try:
            payload = _ready_payload()
            if payload.get("duel_ready") is True:
                return payload
            last_detail = f"service_status={payload.get('service_status', 'starting')}"
        except (OSError, ValueError, json.JSONDecodeError, urllib.error.HTTPError) as exc:
            last_detail = str(exc)
        time.sleep(3)
    raise CommandError(
        f"Both models did not become ready within {timeout_seconds}s ({last_detail})."
    )


def demo(*, build: bool, timeout_seconds: int) -> None:
    doctor(require_artifacts=True)
    command = ["docker", "compose", "up", "--detach"]
    if build:
        command.append("--build")
    _run(command)
    try:
        wait_for_demo(timeout_seconds)
    except CommandError:
        _run(("docker", "compose", "ps"), check=False)
        _run(("docker", "compose", "logs", "--tail", "100"), check=False)
        raise
    print(f"\nThreatLens is ready: {DEMO_URL}")


def clean(*, execute: bool) -> None:
    targets = [ROOT / relative for relative in SAFE_CLEAN_PATHS]
    targets.extend(
        path
        for path in ROOT.rglob("__pycache__")
        if not CLEAN_SCAN_EXCLUDED_PARTS.intersection(path.relative_to(ROOT).parts)
    )
    candidates = sorted(
        {path for path in targets if path.exists()}, key=lambda path: (len(path.parts), str(path))
    )
    existing: list[Path] = []
    for path in candidates:
        if any(parent == path or parent in path.parents for parent in existing):
            continue
        existing.append(path)
    if not existing:
        print("Nothing to clean.")
        return
    verb = "Removing" if execute else "Would remove"
    for path in existing:
        if path.is_symlink():
            raise CommandError(f"Refusing to follow a symlink during cleanup: {path}")
        resolved = path.resolve()
        try:
            resolved.relative_to(ROOT.resolve())
        except ValueError as exc:
            raise CommandError(f"Refusing to clean outside the repository: {path}") from exc
        print(f"{verb}: {path.relative_to(ROOT)}")
        if execute:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    if not execute:
        print("Dry run only. Re-run with `task clean` to remove these disposable files.")


def share() -> None:
    _require_command("tailscale", "Install and sign in to Tailscale using the README guide.")
    try:
        _run(("tailscale", "status"), capture=True)
    except subprocess.CalledProcessError as exc:
        raise CommandError("Tailscale is not connected. Run `sudo tailscale up` first.") from exc
    status = _run(("tailscale", "funnel", "status", "--json"), capture=True)
    active_mapping = status.stdout.strip()
    port_443_configured = '"443"' in active_mapping or ":443" in active_mapping
    if port_443_configured and "127.0.0.1:8081" not in active_mapping:
        raise CommandError(
            "This machine already has a different Funnel mapping. "
            "Review `tailscale funnel status` before changing it."
        )
    try:
        with urllib.request.urlopen(DEMO_URL, timeout=5):
            pass
    except OSError as exc:
        raise CommandError("Start ThreatLens with `task demo` before sharing it.") from exc
    _run(
        (
            "tailscale",
            "funnel",
            "--bg",
            "--tls-terminated-tcp=443",
            "--proxy-protocol=2",
            FUNNEL_TARGET,
        )
    )
    _run(("tailscale", "funnel", "status"), check=False)


def unshare() -> None:
    _require_command("tailscale", "Install Tailscale before managing Funnel.")
    _run(
        (
            "tailscale",
            "funnel",
            "--tls-terminated-tcp=443",
            "--proxy-protocol=2",
            FUNNEL_TARGET,
            "off",
        )
    )


def local_command(name: str) -> None:
    env = os.environ.copy()
    if name == "backend":
        env.setdefault("EMAIL_THREAT_MODEL_PATH", str(ROOT / "artifacts/tfidf_logreg.joblib"))
        env.setdefault("EMAIL_THREAT_MODEL_SHA256", BASELINE_SHA256)
        env.setdefault("EMAIL_THREAT_DISTILBERT_MODEL_PATH", str(ROOT / "artifacts/distilbert"))
        env.setdefault("EMAIL_THREAT_BACKGROUND_WARMUP", "true")
        _run(
            (
                _python(),
                "-m",
                "uvicorn",
                "web.backend.main:app",
                "--reload",
                "--host",
                "127.0.0.1",
                "--port",
                "8000",
            ),
            env=env,
        )
    elif name == "frontend":
        _run((_pnpm(), "dev"), cwd=FRONTEND, env=env)
    elif name == "test":
        _run((_python(), "-m", "pytest"), env=env)
    elif name == "lint":
        _run((_python(), "-m", "ruff", "check", "."), env=env)
        _run((_pnpm(), "lint"), cwd=FRONTEND, env=env)
    elif name == "format-check":
        _run((_python(), "-m", "ruff", "format", "--check", "."), env=env)
    elif name == "build":
        _run((_pnpm(), "build"), cwd=FRONTEND, env=env)
    elif name == "e2e":
        _run((_pnpm(), "test:e2e"), cwd=FRONTEND, env=env)
    else:  # pragma: no cover - argparse prevents this branch
        raise CommandError(f"Unknown command: {name}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor", help="Check Docker, configuration, and artifacts.")
    demo_parser = subparsers.add_parser("demo", help="Build, start, and verify the real demo.")
    demo_parser.add_argument("--no-build", action="store_true")
    demo_parser.add_argument("--timeout", type=int, default=180)
    subparsers.add_parser("logs", help="Follow Compose logs.")
    subparsers.add_parser("down", help="Stop the app without deleting artifacts or caches.")
    subparsers.add_parser("share", help="Publish the running demo through Tailscale Funnel.")
    subparsers.add_parser("unshare", help="Remove only the ThreatLens Funnel mapping.")
    subparsers.add_parser("clean-dry-run", help="Preview safe cleanup without deleting anything.")
    clean_parser = subparsers.add_parser("clean", help="Clean allowlisted disposable outputs.")
    clean_parser.add_argument("--execute", action="store_true")
    for command in ("backend", "frontend", "test", "lint", "format-check", "build", "e2e"):
        subparsers.add_parser(command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "doctor":
            doctor()
        elif args.command == "demo":
            demo(build=not args.no_build, timeout_seconds=args.timeout)
        elif args.command == "logs":
            _run(("docker", "compose", "logs", "--follow", "--tail", "100"))
        elif args.command == "down":
            _run(("docker", "compose", "down"))
        elif args.command == "clean-dry-run":
            clean(execute=False)
        elif args.command == "clean":
            clean(execute=args.execute)
        elif args.command == "share":
            share()
        elif args.command == "unshare":
            unshare()
        else:
            local_command(args.command)
    except (CommandError, subprocess.CalledProcessError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
