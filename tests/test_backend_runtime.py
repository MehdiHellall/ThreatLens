from __future__ import annotations

import hashlib
import io
import urllib.error
from pathlib import Path

import joblib
import pytest
from fastapi.testclient import TestClient

from web.backend import sklearn_runtime as runtime
from web.backend.explanations import (
    confidence_for,
    explanation_for,
    matched_signals,
    risk_level_for,
    suggested_action_for,
)
from web.backend.labels import normalize_label
from web.backend.main import create_app
from web.backend.settings import AppSettings


class NoProbabilityModel:
    def predict(self, texts):
        return ["ham" for _ in texts]


class StaticModel:
    classes_ = ["ham", "phish", "spam"]

    def predict(self, texts):
        return ["spam" for _ in texts]

    def predict_proba(self, texts):
        return [[0.1, 0.2, 0.7] for _ in texts]


class StubHTTPResponse(io.BytesIO):
    def __init__(self, payload: bytes, *, content_length: int | None) -> None:
        super().__init__(payload)
        self.headers = {"Content-Length": str(content_length)} if content_length is not None else {}


def test_classifier_without_predict_proba_returns_label_only() -> None:
    result = runtime.ThreatClassifier(NoProbabilityModel()).predict_one("ordinary message")

    assert result == runtime.PredictionResult(label="ham", probabilities=None)


def test_v1_prediction_supports_artifact_without_probabilities(tmp_path: Path) -> None:
    artifact = tmp_path / "no-proba.joblib"
    runtime.save_model(NoProbabilityModel(), artifact, metadata={"model_name": "label_only"})

    response = TestClient(create_app(AppSettings(model_path=artifact))).post(
        "/v1/predict",
        json={"text": "A routine project update without suspicious language."},
    )

    assert response.status_code == 200
    assert response.json()["final_label"] == "ham"
    assert response.json()["final_risk_level"] == "low"
    assert response.json()["final_confidence"] is None
    assert response.json()["model_outputs"]["tfidf_logreg"] == {
        "status": "available",
        "label": "ham",
        "confidence": None,
        "probabilities": None,
        "detail": "Prediction completed.",
    }
    assert "with" not in response.json()["explanation"].split(". No obvious", maxsplit=1)[0]


def test_load_model_accepts_enveloped_and_legacy_bare_artifacts(tmp_path: Path) -> None:
    enveloped = tmp_path / "enveloped.joblib"
    bare = tmp_path / "bare.joblib"
    runtime.save_model(StaticModel(), enveloped, metadata={"model_name": "static"})
    joblib.dump(StaticModel(), bare)

    enveloped_model, metadata = runtime.load_model(enveloped)
    bare_model, bare_metadata = runtime.load_model(bare)

    assert isinstance(enveloped_model, StaticModel)
    assert metadata == {"model_name": "static"}
    assert isinstance(bare_model, StaticModel)
    assert bare_metadata == {}


def test_file_checksum_validation_handles_match_none_and_mismatch(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"trusted artifact")
    expected = hashlib.sha256(b"trusted artifact").hexdigest()

    runtime.validate_checksum(artifact, None)
    runtime.validate_checksum(artifact, expected.upper())

    with pytest.raises(runtime.ModelArtifactError, match="checksum mismatch"):
        runtime.validate_checksum(artifact, "0" * 64)


@pytest.mark.parametrize(
    "url",
    [
        "https://models.example/artifact.joblib",
        "http://localhost/artifact.joblib",
        "http://127.0.0.1/artifact.joblib",
    ],
)
def test_remote_artifact_validation_accepts_trusted_transports(url: str) -> None:
    runtime.validate_remote_artifact_settings(url, "a" * 64)


def test_artifact_downloader_rejects_unsupported_scheme(tmp_path: Path) -> None:
    with pytest.raises(runtime.ModelArtifactError, match="must use http or https"):
        runtime._download_model_artifact("file:///model.joblib", tmp_path / "model.joblib")


def test_artifact_downloader_removes_partial_file_after_network_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "cache" / "model.joblib"

    def fail_download(_url: str, timeout: int):
        assert timeout == 120
        destination.with_suffix(".joblib.partial").write_bytes(b"partial")
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(runtime, "_open_url_no_redirect", fail_download)

    with pytest.raises(runtime.ModelArtifactError, match="URLError"):
        runtime._download_model_artifact("https://models.example/model.joblib", destination)

    assert not destination.with_suffix(".joblib.partial").exists()


def test_artifact_downloader_rejects_declared_oversized_body(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "cache" / "model.joblib"
    response = StubHTTPResponse(b"small", content_length=11)
    monkeypatch.setattr(runtime, "_open_url_no_redirect", lambda _url, timeout: response)

    with pytest.raises(runtime.ModelArtifactError, match="exceeds maximum artifact size"):
        runtime._download_model_artifact(
            "https://models.example/model.joblib",
            destination,
            max_artifact_bytes=10,
        )

    assert not destination.exists()
    assert not destination.with_suffix(".joblib.partial").exists()


def test_artifact_downloader_rejects_oversized_stream_and_deletes_partial(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "cache" / "model.joblib"
    response = StubHTTPResponse(b"streamed body", content_length=None)
    monkeypatch.setattr(runtime, "_open_url_no_redirect", lambda _url, timeout: response)

    with pytest.raises(runtime.ModelArtifactError, match="exceeds maximum artifact size"):
        runtime._download_model_artifact(
            "https://models.example/model.joblib",
            destination,
            max_artifact_bytes=10,
        )

    assert not destination.exists()
    assert not destination.with_suffix(".joblib.partial").exists()


def test_artifact_downloader_accepts_body_at_configured_limit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "cache" / "model.joblib"
    payload = b"0123456789"
    response = StubHTTPResponse(payload, content_length=len(payload))
    monkeypatch.setattr(runtime, "_open_url_no_redirect", lambda _url, timeout: response)

    downloaded = runtime._download_model_artifact(
        "https://models.example/model.joblib",
        destination,
        max_artifact_bytes=len(payload),
    )

    assert downloaded == destination
    assert destination.read_bytes() == payload
    assert not destination.with_suffix(".joblib.partial").exists()


def test_resolve_model_artifact_reuses_valid_cache_without_downloading(tmp_path: Path) -> None:
    cache = tmp_path / "cache.joblib"
    cache.write_bytes(b"cached artifact")
    checksum = hashlib.sha256(cache.read_bytes()).hexdigest()
    downloader_called = False

    def unexpected_download(_url: str, _destination: Path) -> Path:
        nonlocal downloader_called
        downloader_called = True
        return _destination

    path, error = runtime.resolve_model_artifact(
        AppSettings(
            model_url="https://models.example/model.joblib",
            model_sha256=checksum,
            model_cache_path=cache,
        ),
        unexpected_download,
    )

    assert (path, error) == (cache, None)
    assert downloader_called is False


def test_resolve_model_artifact_replaces_invalid_cache(tmp_path: Path) -> None:
    cache = tmp_path / "cache.joblib"
    cache.write_bytes(b"stale")
    replacement = b"trusted replacement"
    checksum = hashlib.sha256(replacement).hexdigest()

    def download(_url: str, destination: Path) -> Path:
        assert not destination.exists()
        destination.write_bytes(replacement)
        return destination

    path, error = runtime.resolve_model_artifact(
        AppSettings(
            model_url="https://models.example/model.joblib",
            model_sha256=checksum,
            model_cache_path=cache,
        ),
        download,
    )

    assert (path, error) == (cache, None)
    assert cache.read_bytes() == replacement


def test_model_service_backs_off_remote_failures_before_retrying(tmp_path: Path) -> None:
    source = tmp_path / "source.joblib"
    cache = tmp_path / "cache" / "model.joblib"
    runtime.save_model(StaticModel(), source)
    checksum = runtime.file_sha256(source)
    clock = [100.0]
    download_calls = 0

    def download(_url: str, destination: Path) -> Path:
        nonlocal download_calls
        download_calls += 1
        if download_calls == 1:
            raise runtime.ModelArtifactError("temporary outage")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        return destination

    service = runtime.ModelService(
        AppSettings(
            model_url="https://models.example/model.joblib",
            model_sha256=checksum,
            model_cache_path=cache,
            model_retry_seconds=30.0,
        ),
        artifact_downloader=download,
        monotonic=lambda: clock[0],
    )

    failed = service.get_state()
    before_window = service.get_state()
    clock[0] = 129.999
    still_backing_off = service.get_state()

    assert failed.error == "temporary outage"
    assert before_window is failed
    assert still_backing_off is failed
    assert download_calls == 1

    clock[0] = 130.0
    recovered = service.get_state()

    assert recovered.loaded is True
    assert recovered.error is None
    assert download_calls == 2


def test_runtime_public_helpers_filter_private_metadata_and_unknown_labels(tmp_path: Path) -> None:
    state = runtime.ModelState(
        classifier=runtime.ThreatClassifier(StaticModel()),
        path=tmp_path / "model.joblib",
        metadata={
            "model_name": "tfidf_logreg",
            "metrics_file": "metrics.json",
            "private_path": "C:/secret/model.joblib",
        },
    )

    assert state.loaded is True
    assert runtime.public_artifact_metadata(state) == {
        "artifact": "model.joblib",
        "model_name": "tfidf_logreg",
        "metrics_file": "metrics.json",
    }
    assert runtime.public_path(None) is None
    assert runtime.normalize_probabilities(None) is None
    assert runtime.normalize_probabilities({"legit": 0.4, "scam": 0.5, "unknown": 0.1}) == {
        "ham": 0.4,
        "phish": 0.5,
    }
    assert runtime.normalize_probabilities({"unknown": 1.0}) is None
    assert runtime.normalize_prediction_label(" phishing ") == "phish"


@pytest.mark.parametrize(
    ("raw_label", "expected"),
    [
        (" HAM ", "ham"),
        ("0", "ham"),
        (0, "ham"),
        (1.0, "phish"),
        ("legit", "ham"),
        ("legitimate", "ham"),
        ("normal", "ham"),
        ("phishing", "phish"),
        ("scam", "phish"),
        ("junk", "spam"),
    ],
)
def test_label_aliases_normalize_to_canonical_names(raw_label: object, expected: str) -> None:
    assert normalize_label(raw_label) == expected


@pytest.mark.parametrize("raw_label", [None, 7, 1.5, "unrecognized"])
def test_unknown_labels_are_rejected(raw_label: object) -> None:
    with pytest.raises(ValueError, match="Label cannot|Unknown"):
        normalize_label(raw_label)


def test_explanations_cover_confidence_risk_signals_and_actions() -> None:
    assert confidence_for("ham", None) is None
    assert confidence_for("ham", {"spam": 1.0}) is None
    assert risk_level_for("ham", {"ham": 0.9}) == "low"
    assert risk_level_for("ham", {"ham": 0.4}) == "medium"
    assert risk_level_for("phish", None) == "high"
    assert risk_level_for("spam", None) == "medium"

    text = "Urgent: verify your password to claim a free prize; click https://example.test now."
    assert matched_signals(text) == [
        "urgency",
        "credential request",
        "money or prize language",
        "link or contact prompt",
    ]
    explanation = explanation_for("phish", {"phish": 0.91}, text)
    assert "91% confidence" in explanation
    assert "urgency, credential request, money or prize language" in explanation
    assert "link or contact prompt" not in explanation

    assert suggested_action_for("phish", "high").startswith("Do not click")
    assert suggested_action_for("spam", "medium").startswith("Avoid engaging")
    assert suggested_action_for("ham", "medium").startswith("Review the sender")
    assert suggested_action_for("ham", "low").startswith("Low apparent risk")
