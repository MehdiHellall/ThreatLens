from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from starlette.requests import Request

from web.backend import security
from web.backend import settings as settings_module
from web.backend.security import RequestRateLimiter
from web.backend.settings import (
    ALLOWED_ORIGINS_ENV,
    BACKGROUND_WARMUP_ENV,
    DEFAULT_ALLOWED_ORIGINS,
    MAX_BODY_BYTES_ENV,
    MODEL_CACHE_PATH_ENV,
    MODEL_PATH_ENV,
    MODEL_SHA256_ENV,
    MODEL_URL_ENV,
    RATE_LIMIT_PER_MINUTE_ENV,
    AppSettings,
    positive_int_from_env,
)


def _request(
    *,
    content_length: str | None,
    client: tuple[str, int] | None = ("203.0.113.10", 1234),
) -> Request:
    headers = [] if content_length is None else [(b"content-length", content_length.encode())]
    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/predict",
            "raw_path": b"/v1/predict",
            "query_string": b"",
            "headers": headers,
            "client": client,
            "server": ("testserver", 80),
        }
    )


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        (None, 17),
        ("", 17),
        ("not-an-integer", 17),
        ("-4", 0),
        ("23", 23),
    ],
)
def test_positive_int_from_env_falls_back_and_clamps(
    monkeypatch: pytest.MonkeyPatch,
    raw_value: str | None,
    expected: int,
) -> None:
    variable = "THREATLENS_TEST_POSITIVE_INT"
    if raw_value is None:
        monkeypatch.delenv(variable, raising=False)
    else:
        monkeypatch.setenv(variable, raw_value)

    assert positive_int_from_env(variable, 17) == expected


def test_app_settings_reads_and_normalizes_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(MODEL_PATH_ENV, "models/local.joblib")
    monkeypatch.setenv(MODEL_URL_ENV, "  https://models.example/model.joblib  ")
    monkeypatch.setenv(MODEL_SHA256_ENV, "a" * 64)
    monkeypatch.setenv(MODEL_CACHE_PATH_ENV, "cache/download.joblib")
    monkeypatch.setenv(BACKGROUND_WARMUP_ENV, "YeS")
    monkeypatch.setenv(
        ALLOWED_ORIGINS_ENV,
        " https://analyst.example, ,http://localhost:5173 ",
    )
    monkeypatch.setenv(MAX_BODY_BYTES_ENV, "4096")
    monkeypatch.setenv(RATE_LIMIT_PER_MINUTE_ENV, "0")

    settings = AppSettings.from_env()

    assert settings.model_path == Path("models/local.joblib")
    assert settings.model_url == "https://models.example/model.joblib"
    assert settings.model_sha256 == "a" * 64
    assert settings.model_cache_path == Path("cache/download.joblib")
    assert settings.background_warmup is True
    assert settings.allowed_origins == (
        "https://analyst.example",
        "http://localhost:5173",
    )
    assert settings.max_body_bytes == 4096
    assert settings.rate_limit_per_minute == 0


def test_app_settings_uses_default_origins_when_environment_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ALLOWED_ORIGINS_ENV, "")

    assert AppSettings.from_env().allowed_origins == DEFAULT_ALLOWED_ORIGINS


def test_artifact_security_settings_have_safe_defaults_and_stable_env_names() -> None:
    settings = AppSettings()

    assert settings_module.MODEL_RETRY_SECONDS_ENV == "EMAIL_THREAT_MODEL_RETRY_SECONDS"
    assert settings_module.MAX_ARTIFACT_BYTES_ENV == "EMAIL_THREAT_MAX_ARTIFACT_BYTES"
    assert settings.model_retry_seconds == 30.0
    assert settings.max_artifact_bytes == 1_073_741_824


def test_artifact_security_settings_parse_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMAIL_THREAT_MODEL_RETRY_SECONDS", "7.5")
    monkeypatch.setenv("EMAIL_THREAT_MAX_ARTIFACT_BYTES", "536870912")

    settings = AppSettings.from_env()

    assert settings.model_retry_seconds == 7.5
    assert settings.max_artifact_bytes == 536_870_912


@pytest.mark.parametrize(
    ("retry_seconds", "artifact_bytes", "expected_retry", "expected_bytes"),
    [
        ("", "", 30.0, 1_073_741_824),
        ("not-a-float", "not-an-int", 30.0, 1_073_741_824),
        ("-2.5", "-10", 0.0, 0),
    ],
)
def test_artifact_security_settings_fall_back_and_clamp(
    monkeypatch: pytest.MonkeyPatch,
    retry_seconds: str,
    artifact_bytes: str,
    expected_retry: float,
    expected_bytes: int,
) -> None:
    monkeypatch.setenv("EMAIL_THREAT_MODEL_RETRY_SECONDS", retry_seconds)
    monkeypatch.setenv("EMAIL_THREAT_MAX_ARTIFACT_BYTES", artifact_bytes)

    settings = AppSettings.from_env()

    assert settings.model_retry_seconds == expected_retry
    assert settings.max_artifact_bytes == expected_bytes


def test_rate_limiter_is_per_client_and_expires_old_requests() -> None:
    limiter = RequestRateLimiter(limit_per_minute=2)

    assert limiter.allow("analyst-a", now=100.0) is True
    assert limiter.allow("analyst-a", now=101.0) is True
    assert limiter.allow("analyst-a", now=102.0) is False
    assert limiter.allow("analyst-b", now=102.0) is True
    assert limiter.allow("analyst-a", now=161.0) is True


def test_disabled_rate_limiter_allows_every_request() -> None:
    limiter = RequestRateLimiter(limit_per_minute=0)

    assert all(limiter.allow("analyst", now=float(second)) for second in range(100))


def test_rate_limiter_bounds_client_tracking_with_lru_eviction() -> None:
    limiter = RequestRateLimiter(limit_per_minute=2, max_clients=2)

    assert limiter.allow("analyst-a", now=100.0) is True
    assert limiter.allow("analyst-b", now=101.0) is True
    assert limiter.allow("analyst-c", now=102.0) is True

    assert list(limiter._requests_by_client) == ["analyst-b", "analyst-c"]


def test_request_helpers_handle_unknown_clients_and_malformed_lengths() -> None:
    assert security._client_id(_request(content_length="10", client=None)) == "unknown"
    assert security._content_length(_request(content_length=None)) is None
    assert security._content_length(_request(content_length="invalid")) == -1
    assert security._content_length(_request(content_length="12")) == 12


def test_request_ids_accept_safe_values_and_replace_untrusted_values() -> None:
    safe_request = _request(content_length="12")
    safe_request.scope["headers"].append((b"x-request-id", b"demo-request_42"))
    unsafe_request = _request(content_length="12")
    unsafe_request.scope["headers"].append((b"x-request-id", b"bad value\nsecret"))

    assert security._request_id(safe_request) == "demo-request_42"
    generated = security._request_id(unsafe_request)
    assert len(generated) == 32
    assert generated != "bad value\nsecret"


@pytest.mark.parametrize(
    ("content_length", "expected_status", "expected_detail"),
    [
        (None, 411, "Content-Length is required"),
        ("invalid", 413, "32 bytes or fewer"),
        ("33", 413, "32 bytes or fewer"),
    ],
)
def test_prediction_guard_rejects_invalid_body_lengths(
    content_length: str | None,
    expected_status: int,
    expected_detail: str,
) -> None:
    response = security._guard_prediction_request(
        _request(content_length=content_length),
        AppSettings(max_body_bytes=32),
        RequestRateLimiter(limit_per_minute=10),
    )

    assert response is not None
    assert response.status_code == expected_status
    assert expected_detail in json.loads(response.body)["detail"]


def test_prediction_guard_allows_valid_request_then_rate_limits_client() -> None:
    limiter = RequestRateLimiter(limit_per_minute=1)
    request = _request(content_length="12")
    settings = AppSettings(max_body_bytes=32)

    assert security._guard_prediction_request(request, settings, limiter) is None
    response = security._guard_prediction_request(request, settings, limiter)

    assert response is not None
    assert response.status_code == 429
    assert "Too many prediction requests" in json.loads(response.body)["detail"]


def test_prediction_admission_is_bounded_and_recovers_after_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    middleware = security.PredictionSecurityMiddleware(
        lambda _scope, _receive, _send: None,
        settings=AppSettings(),
    )
    monkeypatch.setattr(security, "PREDICTION_QUEUE_TIMEOUT_SECONDS", 0.01)

    async def exercise() -> None:
        assert await middleware._admit_prediction() is True
        assert await middleware._admit_prediction() is False
        assert middleware._pending_predictions == 1
        await middleware._release_prediction()
        assert await middleware._admit_prediction() is True
        await middleware._release_prediction()

    asyncio.run(exercise())
    assert middleware._pending_predictions == 0
