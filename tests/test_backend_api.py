from __future__ import annotations

import hashlib
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from web.backend import main as backend_main
from web.backend.main import AppSettings, create_app
from web.backend.model import save_model
from web.backend.runtimes import distilbert_runtime

CHECKED_IN_ARTIFACT = Path(__file__).resolve().parents[1] / "artifacts" / "tfidf_logreg.joblib"
CHECKED_IN_PHISH_SAMPLE = (
    "replica watches aceyuhardepstatenjus buy replica rolex fraction price safe 15 "
    "buy 2 watches go cocktail party watch sure catch peoples attention youll class "
    "still money come replica watches shop httpgzhifgasolcn thu 07 aug 2008 041343 "
    "0000 tag heuer watches"
)


class PicklableThreatModel:
    classes_ = ["ham", "phish", "spam"]

    def predict(self, texts):
        return ["phish" for _ in texts]

    def predict_proba(self, texts):
        return [[0.03, 0.92, 0.05] for _ in texts]


class NumericClassThreatModel:
    classes_ = [0, 1, 2]

    def predict(self, texts):
        return [1 for _ in texts]

    def predict_proba(self, texts):
        return [[0.04, 0.91, 0.05] for _ in texts]


class QuietStaticFileHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return


class StubDistilBertRuntime:
    def __init__(
        self,
        prediction: distilbert_runtime.DistilBertPrediction | None = None,
        error: Exception | None = None,
    ) -> None:
        self.prediction = prediction
        self.error = error
        self.calls = 0

    def predict_one(self, _text: str) -> distilbert_runtime.DistilBertPrediction:
        self.calls += 1
        if self.error is not None:
            raise self.error
        assert self.prediction is not None
        return self.prediction


class StubDistilBertService:
    def __init__(self, state: distilbert_runtime.DistilBertState) -> None:
        self.state = state

    def get_state(self) -> distilbert_runtime.DistilBertState:
        return self.state


@pytest.fixture()
def artifact_path(tmp_path: Path) -> Path:
    path = tmp_path / "threat-model.joblib"
    save_model(
        PicklableThreatModel(),
        path,
        metadata={
            "model_name": "test_model",
            "metrics_file": "test_metrics.json",
            "local_path": str(tmp_path),
        },
    )
    return path


@pytest.fixture()
def artifact_server(tmp_path: Path):
    source_dir = tmp_path / "server"
    source_dir.mkdir()
    source_path = source_dir / "remote-model.joblib"
    save_model(PicklableThreatModel(), source_path)

    handler = partial(QuietStaticFileHandler, directory=str(source_dir))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
        yield {
            "url": f"http://127.0.0.1:{server.server_port}/remote-model.joblib",
            "sha256": digest,
        }
    finally:
        server.shutdown()
        thread.join(timeout=5)


def _client(model_path: Path) -> TestClient:
    app = create_app(AppSettings(model_path=model_path))
    return TestClient(app)


def _distilbert_state(
    *,
    prediction: distilbert_runtime.DistilBertPrediction | None = None,
    error: Exception | None = None,
    load_error: str | None = None,
) -> tuple[distilbert_runtime.DistilBertState, StubDistilBertRuntime | None]:
    if load_error is not None:
        return (
            distilbert_runtime.DistilBertState(
                runtime=None,
                path=None,
                manifest={},
                error=load_error,
            ),
            None,
        )
    runtime = StubDistilBertRuntime(prediction=prediction, error=error)
    return (
        distilbert_runtime.DistilBertState(
            runtime=runtime,
            path=Path("artifacts/distilbert"),
            manifest={
                "training_complete": True,
                "base_checkpoint": "distilbert/distilbert-base-uncased",
                "max_length": 128,
            },
            error=None,
        ),
        runtime,
    )


def _duel_client(
    artifact_path: Path,
    state: distilbert_runtime.DistilBertState,
) -> TestClient:
    return TestClient(
        create_app(
            AppSettings(model_path=artifact_path),
            distilbert_service=StubDistilBertService(state),
        )
    )


def test_health_reports_loaded_model(artifact_path: Path) -> None:
    response = _client(artifact_path).get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
        "model_path": "threat-model.joblib",
        "detail": "Model artifact loaded.",
    }


def test_live_does_not_require_model_artifact(tmp_path: Path) -> None:
    client = _client(tmp_path / "missing.joblib")

    response = client.get("/live")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "app_name": "ThreatLens",
        "detail": "API process is running.",
    }


def test_api_responses_include_security_headers(artifact_path: Path) -> None:
    response = _client(artifact_path).get("/health")

    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["Referrer-Policy"] == "no-referrer"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Cache-Control"] == "no-store"
    assert len(response.headers["X-Request-ID"]) == 32


@pytest.mark.parametrize("route", ["/predict", "/v1/predict"])
def test_predict_rejects_missing_and_empty_text(artifact_path: Path, route: str) -> None:
    client = _client(artifact_path)

    missing = client.post(route, json={})
    empty = client.post(route, json={"text": "   "})

    assert missing.status_code == 422
    assert empty.status_code == 422
    assert empty.json()["detail"] == "text must not be empty."


@pytest.mark.parametrize("route", ["/predict", "/v1/predict"])
def test_predict_rejects_oversized_request_before_prediction(
    artifact_path: Path,
    route: str,
) -> None:
    app = create_app(AppSettings(model_path=artifact_path, max_body_bytes=16))
    client = TestClient(app)

    response = client.post(route, json={"text": "verify account password"})

    assert response.status_code == 413
    assert "16 bytes or fewer" in response.json()["detail"]


def test_predict_checks_actual_body_size_when_content_length_is_incorrect(
    artifact_path: Path,
) -> None:
    app = create_app(AppSettings(model_path=artifact_path, max_body_bytes=16))
    client = TestClient(app)

    response = client.post(
        "/v1/predict",
        content=b'{"text":"verify account password"}',
        headers={"Content-Type": "application/json", "Content-Length": "1"},
    )

    assert response.status_code == 413
    assert "16 bytes or fewer" in response.json()["detail"]


@pytest.mark.parametrize("route", ["/predict", "/v1/predict"])
def test_predict_preflight_is_not_blocked_by_body_guards(
    artifact_path: Path,
    route: str,
) -> None:
    client = TestClient(
        create_app(
            AppSettings(
                model_path=artifact_path,
                allowed_origins=("http://localhost:8080",),
            )
        )
    )

    response = client.options(
        route,
        headers={
            "Origin": "http://localhost:8080",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200
    assert response.headers["Access-Control-Allow-Origin"] == "http://localhost:8080"


@pytest.mark.parametrize("route", ["/predict", "/v1/predict"])
def test_predict_rate_limits_repeated_requests(artifact_path: Path, route: str) -> None:
    app = create_app(AppSettings(model_path=artifact_path, rate_limit_per_minute=1))
    client = TestClient(app)

    first = client.post(route, json={"text": "verify account"})
    second = client.post(route, json={"text": "verify account"})

    assert first.status_code == 200
    assert second.status_code == 429
    assert "Too many prediction requests" in second.json()["detail"]
    assert second.headers["Retry-After"] == "60"


def test_predict_returns_real_classifier_result(artifact_path: Path) -> None:
    response = _client(artifact_path).post(
        "/predict",
        json={"text": "  Urgent password reset required verify account.  "},
    )

    body = response.json()

    assert response.status_code == 200
    assert body["label"] == "phish"
    assert body["probabilities"] == {"ham": 0.03, "phish": 0.92, "spam": 0.05}
    assert body["risk_level"] == "high"
    assert "model classified" in body["explanation"]
    assert "Do not click" in body["suggested_action"]


def test_checked_in_tfidf_artifact_prediction_is_characterized() -> None:
    client = _client(CHECKED_IN_ARTIFACT)

    phish = client.post("/predict", json={"text": CHECKED_IN_PHISH_SAMPLE})
    ham = client.post("/predict", json={"text": "Hey, are we still meeting for lunch tomorrow?"})
    spam = client.post(
        "/predict",
        json={
            "text": (
                "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. "
                "Text FA to 87121 to receive entry question(std txt rate)"
            )
        },
    )

    assert phish.status_code == 200
    assert phish.json()["label"] == "phish"
    assert phish.json()["probabilities"] == pytest.approx(
        {
            "ham": 0.0003345602296797188,
            "phish": 0.9973158094498464,
            "spam": 0.002349630320473904,
        },
        abs=1e-8,
    )
    assert ham.status_code == 200
    assert ham.json()["label"] == "ham"
    assert spam.status_code == 200
    assert spam.json()["label"] == "spam"


def test_v1_live_preserves_live_contract(artifact_path: Path) -> None:
    client = _client(artifact_path)

    legacy = client.get("/live")
    versioned = client.get("/v1/live")

    assert versioned.status_code == 200
    assert (
        versioned.json()
        == legacy.json()
        == {
            "status": "ok",
            "app_name": "ThreatLens",
            "detail": "API process is running.",
        }
    )


def test_v1_ready_preserves_health_contract(artifact_path: Path) -> None:
    client = _client(artifact_path)

    legacy = client.get("/health")
    versioned = client.get("/v1/ready")

    assert legacy.status_code == 200
    assert legacy.json() == {
        "status": "ok",
        "model_loaded": True,
        "model_path": "threat-model.joblib",
        "detail": "Model artifact loaded.",
    }
    assert versioned.status_code == 200
    body = versioned.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["duel_ready"] is False
    assert set(body["models"]) == {"tfidf_logreg", "distilbert"}
    assert body["models"]["tfidf_logreg"]["available"] is True
    assert body["models"]["distilbert"]["available"] is False


def test_v1_metadata_preserves_public_metadata_contract(artifact_path: Path) -> None:
    client = _client(artifact_path)

    legacy = client.get("/metadata")
    versioned = client.get("/v1/metadata")

    assert legacy.status_code == 200
    assert set(legacy.json()) == {
        "app_name",
        "labels",
        "max_text_chars",
        "model",
        "metrics",
        "privacy",
    }
    assert versioned.status_code == 200
    assert set(versioned.json()["models"]) == {"tfidf_logreg", "distilbert"}
    assert versioned.json()["models"]["tfidf_logreg"]["available"] is True
    assert versioned.json()["models"]["distilbert"]["available"] is False
    assert versioned.json()["model"] == {
        "loaded": True,
        "artifact": "threat-model.joblib",
        "metadata": {
            "metrics_file": "test_metrics.json",
            "model_name": "test_model",
        },
        "status": "ready",
    }


def test_v1_predict_has_duel_ready_shape_and_preserves_legacy_result(
    artifact_path: Path,
) -> None:
    client = _client(artifact_path)
    payload = {"text": "Urgent password reset required verify account."}

    legacy = client.post("/predict", json=payload)
    versioned = client.post("/v1/predict", json=payload)

    expected_legacy = {
        "label": "phish",
        "probabilities": {"ham": 0.03, "phish": 0.92, "spam": 0.05},
        "risk_level": "high",
        "explanation": (
            "The model classified this message as phish with a 92% model score. "
            "A separate keyword check observed: urgency, credential request; these signals "
            "are not an explanation of the model's reasoning."
        ),
        "suggested_action": (
            "Do not click links or share credentials. Verify the request through a trusted "
            "channel and report it to your security team."
        ),
    }
    assert legacy.status_code == 200
    assert legacy.json() == expected_legacy
    assert versioned.status_code == 200

    body = versioned.json()
    assert set(body) == {
        "final_label",
        "final_risk_level",
        "final_confidence",
        "final_model",
        "fallback_reason",
        "score_kind",
        "review_recommended",
        "review_reasons",
        "input_metadata",
        "heuristic_signals",
        "agreement",
        "model_outputs",
        "explanation",
        "suggested_action",
        "artifact_metadata",
        "model_manifests",
    }
    assert set(body["model_outputs"]) == {"tfidf_logreg", "distilbert"}
    assert set(body["model_outputs"]["tfidf_logreg"]) == {
        "status",
        "label",
        "confidence",
        "probabilities",
        "detail",
    }
    assert set(body["model_outputs"]["distilbert"]) == {
        "status",
        "label",
        "confidence",
        "probabilities",
        "detail",
    }
    assert body["agreement"] == "unavailable"
    assert body["model_outputs"]["distilbert"]["status"] == "unavailable"
    assert body["model_outputs"]["distilbert"]["label"] is None
    assert body["model_outputs"]["distilbert"]["confidence"] is None
    assert body["model_outputs"]["distilbert"]["probabilities"] is None
    assert body["model_outputs"]["distilbert"]["detail"]
    assert set(body["model_manifests"]) == {"tfidf_logreg", "distilbert"}
    assert body["artifact_metadata"] == {
        "artifact": "threat-model.joblib",
        "model_name": "test_model",
        "metrics_file": "test_metrics.json",
    }
    assert body["final_label"] == expected_legacy["label"]
    assert body["final_risk_level"] == expected_legacy["risk_level"]
    assert body["final_confidence"] == expected_legacy["probabilities"]["phish"]
    assert body["final_model"] == "tfidf_logreg"
    assert body["fallback_reason"] == "primary_unavailable"
    assert body["score_kind"] == "uncalibrated_probability"
    assert body["review_recommended"] is True
    assert body["review_reasons"] == ["primary_unavailable", "suspicious_text_signals"]
    assert body["input_metadata"] == {
        "character_count": len(payload["text"]),
        "transformer_tokens": None,
        "transformer_max_tokens": 128,
        "truncated": None,
    }
    assert body["heuristic_signals"] == ["urgency", "credential request"]
    assert body["model_outputs"]["tfidf_logreg"] == {
        "status": "available",
        "label": expected_legacy["label"],
        "confidence": expected_legacy["probabilities"]["phish"],
        "probabilities": expected_legacy["probabilities"],
        "detail": "Prediction completed.",
    }
    assert expected_legacy["explanation"] in body["explanation"]
    assert "DistilBERT" in body["explanation"]
    assert "unavailable" in body["explanation"].casefold()
    assert body["suggested_action"] == expected_legacy["suggested_action"]


def test_v1_duel_agreement_selects_real_distilbert_result(artifact_path: Path) -> None:
    state, _runtime = _distilbert_state(
        prediction=distilbert_runtime.DistilBertPrediction(
            label="phish",
            probabilities={"ham": 0.05, "phish": 0.88, "spam": 0.07},
        )
    )

    response = _duel_client(artifact_path, state).post(
        "/v1/predict",
        json={"text": "Urgent password reset required verify account."},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["agreement"] == "agreed"
    assert body["final_label"] == "phish"
    assert body["final_confidence"] == 0.88
    assert body["model_outputs"]["tfidf_logreg"]["label"] == "phish"
    assert body["model_outputs"]["distilbert"] == {
        "status": "available",
        "label": "phish",
        "confidence": 0.88,
        "probabilities": {"ham": 0.05, "phish": 0.88, "spam": 0.07},
        "detail": "Prediction completed.",
    }
    assert "TF-IDF" in body["explanation"]
    assert "DistilBERT" in body["explanation"]


def test_v1_predict_uses_primary_when_tfidf_is_unavailable(tmp_path: Path) -> None:
    state, _runtime = _distilbert_state(
        prediction=distilbert_runtime.DistilBertPrediction(
            label="ham",
            probabilities={"ham": 0.9, "phish": 0.04, "spam": 0.06},
            token_count=23,
            max_tokens=128,
            truncated=False,
        )
    )
    client = _duel_client(tmp_path / "missing.joblib", state)

    response = client.post("/v1/predict", json={"text": "Routine meeting update."})

    assert response.status_code == 200
    body = response.json()
    assert body["final_label"] == "ham"
    assert body["final_model"] == "distilbert"
    assert body["fallback_reason"] is None
    assert body["agreement"] == "unavailable"
    assert body["model_outputs"]["tfidf_logreg"]["status"] == "unavailable"
    assert body["model_outputs"]["distilbert"]["status"] == "available"
    assert body["input_metadata"] == {
        "character_count": 23,
        "transformer_tokens": 23,
        "transformer_max_tokens": 128,
        "truncated": False,
    }


def test_v1_predict_escalates_suspicious_ham_without_changing_model_label(
    tmp_path: Path,
) -> None:
    state, _runtime = _distilbert_state(
        prediction=distilbert_runtime.DistilBertPrediction(
            label="ham",
            probabilities={"ham": 0.98, "phish": 0.01, "spam": 0.01},
        )
    )
    client = _duel_client(tmp_path / "missing.joblib", state)

    response = client.post(
        "/v1/predict",
        json={"text": "Urgent: verify your account password at https://example.com now."},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["final_label"] == "ham"
    assert body["final_confidence"] == 0.98
    assert body["final_risk_level"] == "medium"
    assert body["review_reasons"] == ["suspicious_text_signals"]
    assert "Do not use links or provide credentials" in body["suggested_action"]


def test_v1_ready_is_healthy_when_only_primary_can_serve(tmp_path: Path) -> None:
    state, _runtime = _distilbert_state(
        prediction=distilbert_runtime.DistilBertPrediction(
            label="ham",
            probabilities={"ham": 0.9, "phish": 0.04, "spam": 0.06},
        )
    )
    client = _duel_client(tmp_path / "missing.joblib", state)

    response = client.get("/v1/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["ready"] is True
    assert body["primary_ready"] is True
    assert body["model_loaded"] is False
    assert body["duel_ready"] is False
    assert body["service_status"] == "limited"
    assert body["models"]["tfidf_logreg"]["state"] == "error"
    assert body["models"]["distilbert"]["state"] == "ready"


def test_v1_duel_disagreement_is_visible_and_distilbert_is_final(
    artifact_path: Path,
) -> None:
    state, _runtime = _distilbert_state(
        prediction=distilbert_runtime.DistilBertPrediction(
            label="spam",
            probabilities={"ham": 0.08, "phish": 0.2, "spam": 0.72},
        )
    )

    response = _duel_client(artifact_path, state).post(
        "/v1/predict",
        json={"text": "Urgent password reset required verify account."},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["agreement"] == "disagreed"
    assert body["model_outputs"]["tfidf_logreg"]["label"] == "phish"
    assert body["model_outputs"]["distilbert"]["label"] == "spam"
    assert body["final_label"] == "spam"
    assert body["final_confidence"] == 0.72
    assert body["final_risk_level"] == "medium"
    assert "spam" in body["suggested_action"].casefold()
    assert "disagree" in body["explanation"].casefold()
    assert "manual" in body["explanation"].casefold()
    assert body["review_recommended"] is True
    assert body["review_reasons"] == ["model_disagreement", "suspicious_text_signals"]


def test_v1_duel_unavailable_falls_back_without_inventing_probabilities(
    artifact_path: Path,
) -> None:
    state, _runtime = _distilbert_state(load_error="Fine-tuned artifact is not configured.")

    response = _duel_client(artifact_path, state).post(
        "/v1/predict",
        json={"text": "Urgent password reset required verify account."},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["agreement"] == "unavailable"
    assert body["final_label"] == "phish"
    assert body["final_confidence"] == 0.92
    assert body["model_outputs"]["distilbert"] == {
        "status": "unavailable",
        "label": None,
        "confidence": None,
        "probabilities": None,
        "detail": "Fine-tuned artifact is not configured.",
    }
    assert "TF-IDF" in body["explanation"]
    assert "unavailable" in body["explanation"].casefold()


def test_v1_duel_runtime_failure_is_partial_and_does_not_leak_error(
    artifact_path: Path,
) -> None:
    state, _runtime = _distilbert_state(
        error=RuntimeError("secret submitted text and internal path C:/private/model")
    )

    response = _duel_client(artifact_path, state).post(
        "/v1/predict",
        json={"text": "Urgent password reset required verify account."},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["agreement"] == "partial"
    assert body["final_label"] == "phish"
    assert body["model_outputs"]["distilbert"]["status"] == "error"
    assert body["model_outputs"]["distilbert"]["label"] is None
    assert body["final_model"] == "tfidf_logreg"
    assert body["fallback_reason"] == "primary_prediction_failed"
    assert body["review_reasons"] == [
        "primary_prediction_failed",
        "suspicious_text_signals",
    ]
    assert "secret submitted" not in str(body)
    assert "C:/private" not in str(body)


def test_v1_predict_reports_truncation_and_requires_review(artifact_path: Path) -> None:
    state, _runtime = _distilbert_state(
        prediction=distilbert_runtime.DistilBertPrediction(
            label="phish",
            probabilities={"ham": 0.05, "phish": 0.88, "spam": 0.07},
            token_count=241,
            max_tokens=128,
            truncated=True,
        )
    )

    response = _duel_client(artifact_path, state).post(
        "/v1/predict",
        json={"text": "Urgent password reset required verify account."},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["input_metadata"]["transformer_tokens"] == 241
    assert body["input_metadata"]["transformer_max_tokens"] == 128
    assert body["input_metadata"]["truncated"] is True
    assert body["review_recommended"] is True
    assert body["review_reasons"] == ["input_truncated", "suspicious_text_signals"]


def test_v1_predict_marks_low_model_score_for_review(artifact_path: Path) -> None:
    state, _runtime = _distilbert_state(
        prediction=distilbert_runtime.DistilBertPrediction(
            label="phish",
            probabilities={"ham": 0.31, "phish": 0.4, "spam": 0.29},
        )
    )

    response = _duel_client(artifact_path, state).post(
        "/v1/predict",
        json={"text": "Urgent password reset required verify account."},
    )

    assert response.status_code == 200
    assert response.json()["final_model"] == "distilbert"
    assert response.json()["review_reasons"] == ["suspicious_text_signals", "low_model_score"]


def test_v1_predict_rejects_inconsistent_primary_probabilities_and_falls_back(
    artifact_path: Path,
) -> None:
    state, _runtime = _distilbert_state(
        prediction=distilbert_runtime.DistilBertPrediction(
            label="spam",
            probabilities={"ham": 0.8, "phish": 0.1, "spam": 0.1},
        )
    )

    response = _duel_client(artifact_path, state).post(
        "/v1/predict",
        json={"text": "Urgent password reset required verify account."},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["final_model"] == "tfidf_logreg"
    assert body["fallback_reason"] == "primary_prediction_failed"
    assert body["model_outputs"]["distilbert"]["status"] == "error"


def test_v1_predict_returns_sanitized_503_when_neither_model_can_serve(
    tmp_path: Path,
) -> None:
    state, _runtime = _distilbert_state(load_error="private failure at C:/users/example/model")
    client = _duel_client(tmp_path / "missing.joblib", state)

    response = client.post("/v1/predict", json={"text": "hello"})

    assert response.status_code == 503
    assert response.json() == {"detail": "No prediction model is currently available."}
    assert "C:/users" not in response.text


def test_legacy_predict_stays_tfidf_only_and_never_invokes_distilbert(
    artifact_path: Path,
) -> None:
    state, distilbert = _distilbert_state(
        error=AssertionError("legacy prediction must not call DistilBERT")
    )
    assert distilbert is not None

    response = _duel_client(artifact_path, state).post(
        "/predict",
        json={"text": "Urgent password reset required verify account."},
    )

    assert response.status_code == 200
    assert set(response.json()) == {
        "label",
        "probabilities",
        "risk_level",
        "explanation",
        "suggested_action",
    }
    assert response.json()["label"] == "phish"
    assert distilbert.calls == 0


def test_v1_ready_and_metadata_report_available_distilbert_without_private_paths(
    artifact_path: Path,
) -> None:
    state, _runtime = _distilbert_state(
        prediction=distilbert_runtime.DistilBertPrediction(
            label="ham",
            probabilities={"ham": 0.9, "phish": 0.04, "spam": 0.06},
        )
    )
    client = _duel_client(artifact_path, state)

    ready = client.get("/v1/ready")
    metadata = client.get("/v1/metadata")

    assert ready.status_code == 200
    assert ready.json()["duel_ready"] is True
    assert ready.json()["models"]["distilbert"]["available"] is True
    assert metadata.status_code == 200
    assert set(metadata.json()["models"]) == {"tfidf_logreg", "distilbert"}
    assert metadata.json()["models"]["distilbert"]["available"] is True
    assert metadata.json()["models"]["distilbert"]["manifest"] == {
        "training_complete": True,
        "base_checkpoint": "distilbert/distilbert-base-uncased",
        "max_length": 128,
    }
    assert "artifacts/distilbert" not in str(metadata.json())


def test_predict_normalizes_numeric_class_probabilities(tmp_path: Path) -> None:
    model_path = tmp_path / "numeric.joblib"
    save_model(NumericClassThreatModel(), model_path)

    response = _client(model_path).post(
        "/predict",
        json={"text": "verify account password"},
    )

    assert response.status_code == 200
    assert response.json()["label"] == "phish"
    assert response.json()["probabilities"] == {"ham": 0.04, "phish": 0.91, "spam": 0.05}


def test_metadata_only_returns_public_artifact_metadata(artifact_path: Path) -> None:
    response = _client(artifact_path).get("/metadata")

    model = response.json()["model"]

    assert response.status_code == 200
    assert model["metadata"] == {
        "metrics_file": "test_metrics.json",
        "model_name": "test_model",
    }
    assert "local_path" not in model["metadata"]


def test_remote_model_url_is_downloaded_verified_and_cached(
    artifact_server,
    tmp_path: Path,
) -> None:
    cache_path = tmp_path / "cache" / "model.joblib"
    app = create_app(
        AppSettings(
            model_url=artifact_server["url"],
            model_sha256=artifact_server["sha256"],
            model_cache_path=cache_path,
        )
    )
    client = TestClient(app)

    response = client.post("/predict", json={"text": "verify account"})

    assert response.status_code == 200
    assert response.json()["label"] == "phish"
    assert cache_path.is_file()


def test_model_configuration_rejects_path_and_url_together(
    artifact_path: Path,
    artifact_server,
    tmp_path: Path,
) -> None:
    app = create_app(
        AppSettings(
            model_path=artifact_path,
            model_url=artifact_server["url"],
            model_sha256=artifact_server["sha256"],
            model_cache_path=tmp_path / "cache" / "model.joblib",
        )
    )
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 503
    assert "Configure only one" in response.json()["detail"]


def test_remote_model_url_requires_checksum(tmp_path: Path) -> None:
    app = create_app(
        AppSettings(
            model_url="https://example.com/remote-model.joblib",
            model_cache_path=tmp_path / "model.joblib",
        )
    )
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 503
    assert "EMAIL_THREAT_MODEL_SHA256 is required" in response.json()["detail"]


def test_remote_model_url_rejects_malformed_checksum(tmp_path: Path) -> None:
    app = create_app(
        AppSettings(
            model_url="https://example.com/remote-model.joblib",
            model_sha256="not-a-sha",
            model_cache_path=tmp_path / "model.joblib",
        )
    )
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 503
    assert "64-character hex SHA-256" in response.json()["detail"]


def test_remote_model_url_requires_https_for_non_loopback(tmp_path: Path) -> None:
    app = create_app(
        AppSettings(
            model_url="http://example.com/remote-model.joblib",
            model_sha256="0" * 64,
            model_cache_path=tmp_path / "model.joblib",
        )
    )
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 503
    assert "must use https" in response.json()["detail"]


def test_remote_model_url_rejects_checksum_mismatch(artifact_server, tmp_path: Path) -> None:
    cache_path = tmp_path / "cache" / "model.joblib"
    app = create_app(
        AppSettings(
            model_url=artifact_server["url"],
            model_sha256="0" * 64,
            model_cache_path=cache_path,
        )
    )
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 503
    assert "checksum mismatch" in response.json()["detail"]
    assert not cache_path.exists()


def test_remote_model_url_retry_recovers_after_initial_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.joblib"
    cache_path = tmp_path / "cache" / "model.joblib"
    save_model(PicklableThreatModel(), source_path)
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    calls = 0

    def fake_download_model_artifact(
        _url: str,
        destination: Path,
        max_artifact_bytes: int,
    ) -> Path:
        nonlocal calls
        assert max_artifact_bytes == 1_073_741_824
        calls += 1
        if calls == 1:
            raise backend_main.ModelArtifactError("temporary artifact outage")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source_path.read_bytes())
        return destination

    monkeypatch.setattr(
        backend_main,
        "_download_model_artifact",
        fake_download_model_artifact,
    )
    app = create_app(
        AppSettings(
            model_url="https://example.com/remote-model.joblib",
            model_sha256=digest,
            model_cache_path=cache_path,
            model_retry_seconds=0,
        )
    )
    client = TestClient(app)

    first = client.get("/health")
    second = client.get("/health")

    assert first.status_code == 503
    assert "temporary artifact outage" in first.json()["detail"]
    assert second.status_code == 200
    assert second.json()["model_loaded"] is True
    assert calls == 2


def test_missing_model_returns_health_error(tmp_path: Path) -> None:
    client = _client(tmp_path / "missing.joblib")

    health = client.get("/health")
    prediction = client.post("/predict", json={"text": "hello"})

    assert health.status_code == 503
    assert health.json()["model_loaded"] is False
    assert "does not exist" in health.json()["detail"]
    assert prediction.status_code == 503
    assert "does not exist" in prediction.json()["detail"]
