"""Environment-backed settings for the ThreatLens API."""

from __future__ import annotations

import os
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

MODEL_PATH_ENV = "EMAIL_THREAT_MODEL_PATH"
MODEL_URL_ENV = "EMAIL_THREAT_MODEL_URL"
MODEL_SHA256_ENV = "EMAIL_THREAT_MODEL_SHA256"
MODEL_CACHE_PATH_ENV = "EMAIL_THREAT_MODEL_CACHE_PATH"
BACKGROUND_WARMUP_ENV = "EMAIL_THREAT_BACKGROUND_WARMUP"
ALLOWED_ORIGINS_ENV = "EMAIL_THREAT_ALLOWED_ORIGINS"
RATE_LIMIT_PER_MINUTE_ENV = "EMAIL_THREAT_RATE_LIMIT_PER_MINUTE"
MAX_BODY_BYTES_ENV = "EMAIL_THREAT_MAX_BODY_BYTES"
MODEL_RETRY_SECONDS_ENV = "EMAIL_THREAT_MODEL_RETRY_SECONDS"
MAX_ARTIFACT_BYTES_ENV = "EMAIL_THREAT_MAX_ARTIFACT_BYTES"
DISTILBERT_MODEL_PATH_ENV = "EMAIL_THREAT_DISTILBERT_MODEL_PATH"
DISTILBERT_MODEL_URL_ENV = "EMAIL_THREAT_DISTILBERT_MODEL_URL"
DISTILBERT_MODEL_SHA256_ENV = "EMAIL_THREAT_DISTILBERT_MODEL_SHA256"
DISTILBERT_MODEL_CACHE_PATH_ENV = "EMAIL_THREAT_DISTILBERT_MODEL_CACHE_PATH"
DISTILBERT_MAX_LENGTH_ENV = "EMAIL_THREAT_DISTILBERT_MAX_LENGTH"

DEFAULT_METRICS_PATH = Path(__file__).resolve().parent / "assets" / "tfidf_logreg_metrics.json"
DEFAULT_MODEL_CACHE_PATH = Path("/tmp/threatlens/model.joblib")
DEFAULT_DISTILBERT_MODEL_CACHE_PATH = Path("/tmp/threatlens/distilbert")
DEFAULT_DISTILBERT_MAX_LENGTH = 128
DEFAULT_ALLOWED_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
)
MAX_TEXT_CHARS = 5_000
DEFAULT_MAX_BODY_BYTES = 16_384
DEFAULT_RATE_LIMIT_PER_MINUTE = 60
DEFAULT_MODEL_RETRY_SECONDS = 30.0
DEFAULT_MAX_ARTIFACT_BYTES = 1_073_741_824


def positive_int_from_env(name: str, default: int) -> int:
    """Read a non-negative integer, falling back for absent or malformed values."""
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default

    try:
        parsed_value = int(raw_value)
    except ValueError:
        return default
    return max(parsed_value, 0)


def non_negative_float_from_env(name: str, default: float) -> float:
    """Read a finite non-negative float, falling back for malformed values."""
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default

    try:
        parsed_value = float(raw_value)
    except ValueError:
        return default
    if not isfinite(parsed_value):
        return default
    return max(parsed_value, 0.0)


@dataclass(frozen=True)
class AppSettings:
    """Runtime settings for the API."""

    model_path: Path | None = None
    model_url: str | None = None
    model_sha256: str | None = None
    model_cache_path: Path = DEFAULT_MODEL_CACHE_PATH
    background_warmup: bool = False
    metrics_path: Path = DEFAULT_METRICS_PATH
    allowed_origins: tuple[str, ...] = DEFAULT_ALLOWED_ORIGINS
    max_text_chars: int = MAX_TEXT_CHARS
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES
    rate_limit_per_minute: int = DEFAULT_RATE_LIMIT_PER_MINUTE
    model_retry_seconds: float = DEFAULT_MODEL_RETRY_SECONDS
    max_artifact_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES
    distilbert_model_path: Path | None = None
    distilbert_model_url: str | None = None
    distilbert_model_sha256: str | None = None
    distilbert_model_cache_path: Path = DEFAULT_DISTILBERT_MODEL_CACHE_PATH
    distilbert_max_length: int = DEFAULT_DISTILBERT_MAX_LENGTH

    @classmethod
    def from_env(cls) -> AppSettings:
        """Create settings from environment variables."""
        raw_model_path = os.getenv(MODEL_PATH_ENV)
        raw_model_url = os.getenv(MODEL_URL_ENV)
        raw_model_cache_path = os.getenv(MODEL_CACHE_PATH_ENV)
        raw_origins = os.getenv(ALLOWED_ORIGINS_ENV)
        raw_distilbert_path = os.getenv(DISTILBERT_MODEL_PATH_ENV)
        raw_distilbert_url = os.getenv(DISTILBERT_MODEL_URL_ENV)
        raw_distilbert_sha256 = os.getenv(DISTILBERT_MODEL_SHA256_ENV)
        raw_distilbert_cache_path = os.getenv(DISTILBERT_MODEL_CACHE_PATH_ENV)
        origins = (
            tuple(origin.strip() for origin in raw_origins.split(",") if origin.strip())
            if raw_origins
            else DEFAULT_ALLOWED_ORIGINS
        )
        return cls(
            model_path=Path(raw_model_path).expanduser() if raw_model_path else None,
            model_url=raw_model_url.strip() if raw_model_url else None,
            model_sha256=os.getenv(MODEL_SHA256_ENV),
            model_cache_path=(
                Path(raw_model_cache_path).expanduser()
                if raw_model_cache_path
                else DEFAULT_MODEL_CACHE_PATH
            ),
            background_warmup=os.getenv(BACKGROUND_WARMUP_ENV, "").casefold()
            in {"1", "true", "yes"},
            allowed_origins=origins,
            max_body_bytes=positive_int_from_env(
                MAX_BODY_BYTES_ENV,
                DEFAULT_MAX_BODY_BYTES,
            ),
            rate_limit_per_minute=positive_int_from_env(
                RATE_LIMIT_PER_MINUTE_ENV,
                DEFAULT_RATE_LIMIT_PER_MINUTE,
            ),
            model_retry_seconds=non_negative_float_from_env(
                MODEL_RETRY_SECONDS_ENV,
                DEFAULT_MODEL_RETRY_SECONDS,
            ),
            max_artifact_bytes=positive_int_from_env(
                MAX_ARTIFACT_BYTES_ENV,
                DEFAULT_MAX_ARTIFACT_BYTES,
            ),
            distilbert_model_path=(
                Path(raw_distilbert_path).expanduser() if raw_distilbert_path else None
            ),
            distilbert_model_url=(raw_distilbert_url.strip() if raw_distilbert_url else None),
            distilbert_model_sha256=(
                raw_distilbert_sha256.strip() if raw_distilbert_sha256 else None
            ),
            distilbert_model_cache_path=(
                Path(raw_distilbert_cache_path).expanduser()
                if raw_distilbert_cache_path
                else DEFAULT_DISTILBERT_MODEL_CACHE_PATH
            ),
            distilbert_max_length=positive_int_from_env(
                DISTILBERT_MAX_LENGTH_ENV,
                DEFAULT_DISTILBERT_MAX_LENGTH,
            ),
        )
