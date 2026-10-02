"""Trusted artifact resolution, loading, and scikit-learn inference runtime."""

from __future__ import annotations

import http.client
import json
import math
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from ipaddress import ip_address
from pathlib import Path
from typing import Any, cast

import joblib

from web.backend.labels import LABEL_NAMES, normalize_label
from web.backend.schemas import ThreatLabel
from web.backend.settings import (
    DEFAULT_MAX_ARTIFACT_BYTES,
    MODEL_PATH_ENV,
    MODEL_SHA256_ENV,
    MODEL_URL_ENV,
    AppSettings,
)

PUBLIC_METADATA_KEYS = frozenset({"model_name", "metrics_file"})


@dataclass(frozen=True)
class PredictionResult:
    label: str
    probabilities: dict[str, float] | None = None


class ThreatClassifier:
    """Narrow adapter around the deployed scikit-learn pipeline."""

    def __init__(self, model: Any) -> None:
        self._model = model

    def predict_one(self, text: str) -> PredictionResult:
        label = str(self._model.predict([text])[0])
        probabilities = None
        if hasattr(self._model, "predict_proba"):
            classes = [str(value) for value in self._model.classes_]
            scores = self._model.predict_proba([text])[0]
            probabilities = {
                label_name: float(score) for label_name, score in zip(classes, scores, strict=True)
            }
        return PredictionResult(label=label, probabilities=probabilities)


def save_model(model: Any, path: str | Path, *, metadata: dict[str, Any] | None = None) -> None:
    """Persist a model and optional metadata in the established joblib envelope."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "metadata": metadata or {}}, destination)


def load_model(path: str | Path) -> tuple[Any, dict[str, Any]]:
    """Load both enveloped and legacy bare joblib artifacts."""
    payload = joblib.load(path)
    if isinstance(payload, dict) and "model" in payload:
        return payload["model"], dict(payload.get("metadata", {}))
    return payload, {}


@dataclass(frozen=True)
class ModelState:
    """Loaded model state or a clear artifact problem."""

    classifier: ThreatClassifier | None
    path: Path | None
    metadata: dict[str, object]
    error: str | None = None

    @property
    def loaded(self) -> bool:
        return self.classifier is not None and self.error is None


class ModelArtifactError(RuntimeError):
    """Raised when an artifact cannot be resolved for model loading."""


ArtifactDownloader = Callable[[str, Path], Path]
MonotonicClock = Callable[[], float]
DOWNLOAD_CHUNK_BYTES = 1024 * 1024
TERMINAL_REMOTE_ERROR_MARKERS = (
    "checksum mismatch",
    "must be a 64-character",
    "must use https",
    "is required for remote",
)


def resolve_local_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return Path.cwd() / path


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_checksum(path: Path, expected_sha256: str | None) -> None:
    if expected_sha256 is None:
        return
    actual_sha256 = file_sha256(path)
    if actual_sha256.casefold() != expected_sha256.casefold():
        raise ModelArtifactError(
            "Downloaded model artifact checksum mismatch. "
            f"Expected {expected_sha256}, got {actual_sha256}."
        )


def _is_loopback_host(host: str | None) -> bool:
    if host is None:
        return False
    if host.casefold() == "localhost":
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


def validate_remote_artifact_settings(url: str, expected_sha256: str | None) -> None:
    _validate_download_url(url)
    if expected_sha256 is None:
        raise ModelArtifactError(f"{MODEL_SHA256_ENV} is required for remote model artifacts.")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
        raise ModelArtifactError(f"{MODEL_SHA256_ENV} must be a 64-character hex SHA-256 digest.")


def _validate_download_url(url: str) -> None:
    parsed_url = urllib.parse.urlparse(url)
    if parsed_url.scheme not in {"http", "https"}:
        raise ModelArtifactError("Model artifact URL must use http or https.")
    is_loopback_http = parsed_url.scheme == "http" and _is_loopback_host(parsed_url.hostname)
    if parsed_url.scheme != "https" and not is_loopback_http:
        raise ModelArtifactError(
            "Remote model artifact URL must use https, except loopback URLs used for local tests."
        )


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args: object, **_kwargs: object) -> None:
        return None


def _open_url_no_redirect(url: str, *, timeout: int) -> Any:
    opener = urllib.request.build_opener(_NoRedirectHandler)
    return opener.open(url, timeout=timeout)


def _declared_content_length(response: Any) -> int | None:
    raw_content_length = response.headers.get("Content-Length")
    if raw_content_length is None:
        return None
    try:
        content_length = int(raw_content_length)
    except (TypeError, ValueError) as exc:
        raise ModelArtifactError("Model artifact has an invalid Content-Length header.") from exc
    if content_length < 0:
        raise ModelArtifactError("Model artifact has an invalid Content-Length header.")
    return content_length


def _download_model_artifact(
    url: str,
    destination: Path,
    max_artifact_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES,
) -> Path:
    _validate_download_url(url)

    partial_destination = destination.with_suffix(f"{destination.suffix}.partial")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with (
            _open_url_no_redirect(url, timeout=120) as response,
            partial_destination.open("wb") as file,
        ):
            declared_bytes = _declared_content_length(response)
            if declared_bytes is not None and declared_bytes > max_artifact_bytes:
                raise ModelArtifactError(
                    "Downloaded model artifact exceeds maximum artifact size "
                    f"of {max_artifact_bytes} bytes."
                )

            downloaded_bytes = 0
            while chunk := response.read(DOWNLOAD_CHUNK_BYTES):
                downloaded_bytes += len(chunk)
                if downloaded_bytes > max_artifact_bytes:
                    raise ModelArtifactError(
                        "Downloaded model artifact exceeds maximum artifact size "
                        f"of {max_artifact_bytes} bytes."
                    )
                file.write(chunk)
        partial_destination.replace(destination)
    except ModelArtifactError:
        partial_destination.unlink(missing_ok=True)
        raise
    except (OSError, http.client.HTTPException, urllib.error.URLError) as exc:
        partial_destination.unlink(missing_ok=True)
        raise ModelArtifactError(
            f"Could not download model artifact: {type(exc).__name__}"
        ) from exc
    return destination


def resolve_model_artifact(
    settings: AppSettings,
    artifact_downloader: ArtifactDownloader,
) -> tuple[Path | None, str | None]:
    if settings.model_path is not None and settings.model_url is not None:
        return (
            resolve_local_path(settings.model_path),
            f"Configure only one of {MODEL_PATH_ENV} or {MODEL_URL_ENV}.",
        )
    if settings.model_path is not None:
        model_path = resolve_local_path(settings.model_path)
        if not model_path.is_file():
            return model_path, f"Configured model artifact does not exist: {model_path.name}"
        if settings.model_sha256 is not None:
            try:
                validate_checksum(model_path, settings.model_sha256)
            except ModelArtifactError as exc:
                return model_path, str(exc)
        return model_path, None
    if settings.model_url is None:
        return None, f"{MODEL_PATH_ENV} or {MODEL_URL_ENV} must be set."

    try:
        validate_remote_artifact_settings(settings.model_url, settings.model_sha256)
    except ModelArtifactError as exc:
        return resolve_local_path(settings.model_cache_path), str(exc)

    cache_path = resolve_local_path(settings.model_cache_path)
    if cache_path.is_file():
        try:
            validate_checksum(cache_path, settings.model_sha256)
            return cache_path, None
        except ModelArtifactError:
            cache_path.unlink(missing_ok=True)

    try:
        downloaded_path = artifact_downloader(settings.model_url, cache_path)
        validate_checksum(downloaded_path, settings.model_sha256)
    except ModelArtifactError as exc:
        cache_path.unlink(missing_ok=True)
        return cache_path, str(exc)
    return cache_path, None


def load_model_state(
    settings: AppSettings,
    artifact_downloader: ArtifactDownloader = _download_model_artifact,
) -> ModelState:
    model_path, artifact_error = resolve_model_artifact(settings, artifact_downloader)
    if artifact_error is not None:
        return ModelState(classifier=None, path=model_path, metadata={}, error=artifact_error)
    if model_path is None:
        return ModelState(classifier=None, path=None, metadata={}, error="Model artifact missing.")
    try:
        model, metadata = load_model(model_path)
    except Exception as exc:  # pragma: no cover - depends on local artifact corruption.
        return ModelState(
            classifier=None,
            path=model_path,
            metadata={},
            error=f"Could not load model artifact: {type(exc).__name__}",
        )
    return ModelState(classifier=ThreatClassifier(model), path=model_path, metadata=metadata)


class ModelService:
    """Lazily resolve and load a trusted model artifact once per process."""

    def __init__(
        self,
        settings: AppSettings,
        artifact_downloader: ArtifactDownloader = _download_model_artifact,
        monotonic: MonotonicClock = time.monotonic,
    ) -> None:
        self._settings = settings
        self._artifact_downloader = artifact_downloader
        self._monotonic = monotonic
        self._lock = threading.Lock()
        self._state: ModelState | None = None
        self._last_remote_attempt_at: float | None = None
        self._warmup_thread: threading.Thread | None = None
        self._warmup_start_lock = threading.Lock()

    def get_state(self) -> ModelState:
        """Return the current model state, loading the artifact on first use."""
        if self._state is not None and (self._state.loaded or self._settings.model_url is None):
            return self._state
        with self._lock:
            if self._state is not None and (self._state.loaded or self._settings.model_url is None):
                return self._state
            if self._settings.model_url is None:
                self._state = load_model_state(self._settings, self._artifact_downloader)
                return self._state

            now = self._monotonic()
            should_retry_remote_failure = self._remote_retry_is_due(now)
            if self._state is None or should_retry_remote_failure:
                self._state = load_model_state(self._settings, self._artifact_downloader)
                if not self._state.loaded:
                    self._last_remote_attempt_at = self._monotonic()
            return self._state

    def _remote_retry_is_due(self, now: float) -> bool:
        if (
            self._state is None
            or self._state.loaded
            or self._settings.model_url is None
            or self._last_remote_attempt_at is None
            or any(
                marker in (self._state.error or "").casefold()
                for marker in TERMINAL_REMOTE_ERROR_MARKERS
            )
        ):
            return False
        return now - self._last_remote_attempt_at >= self._settings.model_retry_seconds

    def peek_state(self) -> ModelState | None:
        """Return the current state without starting or waiting for a model load."""
        state = self._state
        if state is not None and self._remote_retry_is_due(self._monotonic()):
            self.start_background_warmup()
        return state

    @property
    def loading(self) -> bool:
        return self._warmup_thread is not None and self._warmup_thread.is_alive()

    def start_background_warmup(self) -> None:
        """Start a non-blocking model load for hosted runtimes."""
        with self._warmup_start_lock:
            if self._warmup_thread is not None and self._warmup_thread.is_alive():
                return
            self._warmup_thread = threading.Thread(
                target=self.get_state,
                name="threatlens-model-warmup",
                daemon=True,
            )
            self._warmup_thread.start()


def load_metrics(path: Path) -> dict[str, object] | None:
    metrics_path = resolve_local_path(path)
    if not metrics_path.is_file():
        return None
    return cast(dict[str, object], json.loads(metrics_path.read_text(encoding="utf-8")))


def public_path(path: Path | None) -> str | None:
    return path.name if path is not None else None


def public_model_metadata(metadata: dict[str, object]) -> dict[str, object]:
    """Return only artifact metadata fields safe to expose through the browser API."""
    return {key: value for key, value in metadata.items() if key in PUBLIC_METADATA_KEYS}


def public_artifact_metadata(state: ModelState) -> dict[str, str | None]:
    """Build the non-invented deployment metadata attached to versioned results."""
    public_metadata = public_model_metadata(state.metadata)
    model_name = public_metadata.get("model_name")
    metrics_file = public_metadata.get("metrics_file")
    return {
        "artifact": public_path(state.path),
        "model_name": model_name if isinstance(model_name, str) else None,
        "metrics_file": metrics_file if isinstance(metrics_file, str) else None,
    }


def normalize_probabilities(
    probabilities: dict[str, float] | None,
) -> dict[str, float] | None:
    """Return a complete, finite canonical probability distribution or ``None``."""
    if probabilities is None:
        return None
    normalized: dict[str, float] = {}
    for raw_label, score in probabilities.items():
        try:
            label = normalize_label(raw_label)
            numeric_score = float(score)
        except (TypeError, ValueError):
            return None
        if label in normalized or not math.isfinite(numeric_score):
            return None
        if numeric_score < 0.0 or numeric_score > 1.0:
            return None
        normalized[label] = numeric_score
    if set(normalized) != set(LABEL_NAMES):
        return None
    if not math.isclose(sum(normalized.values()), 1.0, rel_tol=1e-5, abs_tol=1e-6):
        return None
    return normalized


def normalize_prediction_label(label: str) -> ThreatLabel:
    return cast(ThreatLabel, normalize_label(label))
