"""FastAPI application factory and versioned ThreatLens Model Duel routes."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from web.backend.explanations import (
    confidence_for,
    explanation_for,
    matched_signals,
    risk_level_for,
    suggested_action_for,
)
from web.backend.labels import LABEL_NAMES
from web.backend.runtimes.distilbert_runtime import (
    DistilBertArtifactError,
    DistilBertService,
    DistilBertState,
    public_evaluation_metrics,
    public_manifest,
)
from web.backend.schemas import (
    Agreement,
    ArtifactMetadata,
    FallbackReason,
    FinalModel,
    HealthResponse,
    InputMetadata,
    LiveResponse,
    MetadataResponse,
    ModelAvailability,
    ModelLifecycle,
    ModelPredictionOutput,
    ModelStatus,
    PredictRequest,
    PredictResponse,
    ReadyResponse,
    ReviewReason,
    ScoreKind,
    ThreatLabel,
    VersionedMetadataResponse,
    VersionedPredictResponse,
)
from web.backend.security import PredictionSecurityMiddleware
from web.backend.settings import AppSettings
from web.backend.sklearn_runtime import (
    ModelArtifactError,
    ModelService,
    ModelState,
    _download_model_artifact,
    load_metrics,
    normalize_prediction_label,
    normalize_probabilities,
    public_artifact_metadata,
    public_model_metadata,
    public_path,
)

APP_NAME = "ThreatLens"
PRIVACY_NOTICE = "Messages are analyzed for the current request only and are not stored."
LOW_MODEL_SCORE_THRESHOLD = 0.65
logger = logging.getLogger(__name__)


def _validation_message(exc: RequestValidationError) -> str:
    for error in exc.errors():
        location = tuple(error.get("loc", ()))
        if "text" not in location:
            continue
        if error.get("type") == "missing":
            return "text is required."
        message = str(error.get("msg", "Invalid text."))
        return message.removeprefix("Value error, ")
    return "Invalid request body."


def _health_response(state: ModelState) -> HealthResponse:
    if state.loaded:
        return HealthResponse(
            status="ok",
            model_loaded=True,
            model_path=public_path(state.path),
            detail="Model artifact loaded.",
        )
    return HealthResponse(
        status="error",
        model_loaded=False,
        model_path=public_path(state.path),
        detail=state.error or "Model artifact is unavailable.",
    )


def _classify(payload: PredictRequest, model_service: ModelService) -> PredictResponse:
    state = model_service.get_state()
    if not state.loaded or state.classifier is None:
        raise HTTPException(status_code=503, detail=state.error or "Model artifact is unavailable.")

    try:
        result = state.classifier.predict_one(payload.text)
        label = normalize_prediction_label(result.label)
        probabilities = normalize_probabilities(result.probabilities)
        if result.probabilities is not None and probabilities is None:
            raise ValueError("Invalid TF-IDF probabilities.")
        if probabilities is not None and probabilities[label] != max(probabilities.values()):
            raise ValueError("TF-IDF label does not match its probability distribution.")
    except Exception as exc:
        logger.warning("prediction_failed model=tfidf_logreg error_type=%s", type(exc).__name__)
        raise HTTPException(status_code=503, detail="TF-IDF prediction failed.") from None
    risk_level = risk_level_for(label, probabilities)
    return PredictResponse(
        label=label,
        probabilities=probabilities,
        risk_level=risk_level,
        explanation=explanation_for(label, probabilities, payload.text),
        suggested_action=suggested_action_for(label, risk_level),
    )


def _model_lifecycle(
    state: ModelState | DistilBertState | None,
    *,
    loading: bool,
) -> ModelLifecycle:
    if state is None:
        return "loading" if loading else "unavailable"
    if state.loaded:
        return "ready"
    if loading:
        return "retrying"
    error = (state.error or "").casefold()
    if "not configured" in error or "must be set" in error:
        return "unavailable"
    return "error"


def _availability(
    state: ModelState | DistilBertState | None,
    *,
    loading: bool,
    ready_detail: str,
    unavailable_detail: str,
) -> ModelAvailability:
    lifecycle = _model_lifecycle(state, loading=loading)
    if lifecycle == "ready":
        detail = ready_detail
    elif lifecycle == "loading":
        detail = "Model is loading."
    elif lifecycle == "retrying":
        detail = "Model is retrying after a temporary loading failure."
    else:
        detail = state.error if state is not None and state.error else unavailable_detail
    return ModelAvailability(
        available=lifecycle == "ready",
        detail=detail,
        state=lifecycle,
    )


def _versioned_ready(
    tfidf_state: ModelState | None,
    distilbert_state: DistilBertState | None,
    *,
    tfidf_loading: bool = False,
    distilbert_loading: bool = False,
) -> ReadyResponse:
    tfidf_ready = tfidf_state is not None and tfidf_state.loaded
    distilbert_ready = distilbert_state is not None and distilbert_state.loaded
    ready = tfidf_ready or distilbert_ready
    duel_ready = tfidf_ready and distilbert_ready
    loading = tfidf_loading or distilbert_loading
    if duel_ready:
        service_status = "ready"
        detail = "Both prediction models are ready."
    elif ready:
        service_status = "limited"
        detail = "Prediction is available with one model."
    elif loading:
        service_status = "starting"
        detail = "Prediction models are loading."
    else:
        service_status = "unavailable"
        detail = "No prediction model is available."
    return ReadyResponse(
        status="ok" if ready else "error",
        model_loaded=tfidf_ready,
        model_path=public_path(tfidf_state.path) if tfidf_state is not None else None,
        detail=detail,
        ready=ready,
        primary_ready=distilbert_ready,
        service_status=service_status,
        duel_ready=duel_ready,
        models={
            "tfidf_logreg": _availability(
                tfidf_state,
                loading=tfidf_loading,
                ready_detail="Model artifact loaded.",
                unavailable_detail="Model artifact is unavailable.",
            ),
            "distilbert": _availability(
                distilbert_state,
                loading=distilbert_loading,
                ready_detail="Fine-tuned model artifact loaded.",
                unavailable_detail="Fine-tuned model artifact is unavailable.",
            ),
        },
    )


def _legacy_metadata(
    settings: AppSettings,
    tfidf_state: ModelState | None,
    metrics: dict[str, object] | None,
    *,
    loading: bool = False,
) -> MetadataResponse:
    loaded = tfidf_state is not None and tfidf_state.loaded
    return MetadataResponse(
        app_name=APP_NAME,
        labels=list(LABEL_NAMES),
        max_text_chars=settings.max_text_chars,
        model={
            "loaded": loaded,
            "artifact": public_path(tfidf_state.path) if tfidf_state is not None else None,
            "metadata": public_model_metadata(tfidf_state.metadata)
            if tfidf_state is not None
            else {},
            "status": (
                "ready"
                if loaded
                else "loading"
                if loading
                else tfidf_state.error
                if tfidf_state is not None
                else "unavailable"
            ),
        },
        metrics=metrics,
        privacy=PRIVACY_NOTICE,
    )


def _versioned_metadata(
    legacy: MetadataResponse,
    tfidf_state: ModelState | None,
    distilbert_state: DistilBertState | None,
    *,
    tfidf_loading: bool = False,
    distilbert_loading: bool = False,
) -> VersionedMetadataResponse:
    tfidf_loaded = tfidf_state is not None and tfidf_state.loaded
    distilbert_loaded = distilbert_state is not None and distilbert_state.loaded
    return VersionedMetadataResponse(
        **legacy.model_dump(),
        models={
            "tfidf_logreg": {
                "available": tfidf_loaded,
                "artifact": public_path(tfidf_state.path) if tfidf_state is not None else None,
                "manifest": public_artifact_metadata(tfidf_state)
                if tfidf_state is not None
                else None,
                "status": (
                    "ready"
                    if tfidf_loaded
                    else "loading"
                    if tfidf_loading
                    else tfidf_state.error
                    if tfidf_state is not None
                    else "unavailable"
                ),
            },
            "distilbert": {
                "available": distilbert_loaded,
                "manifest": public_manifest(distilbert_state.manifest)
                if distilbert_loaded and distilbert_state is not None
                else None,
                "metrics": public_evaluation_metrics(distilbert_state)
                if distilbert_loaded and distilbert_state is not None
                else None,
                "status": (
                    "ready"
                    if distilbert_loaded
                    else "loading"
                    if distilbert_loading
                    else distilbert_state.error
                    if distilbert_state is not None
                    else "unavailable"
                ),
            },
        },
    )


def _available_output(
    label: str,
    probabilities: dict[str, float] | None,
) -> ModelPredictionOutput:
    normalized_label = normalize_prediction_label(label)
    return ModelPredictionOutput(
        status="available",
        label=normalized_label,
        confidence=confidence_for(normalized_label, probabilities),
        probabilities=probabilities,
        detail="Prediction completed.",
    )


def _empty_model_output(status: ModelStatus, detail: str) -> ModelPredictionOutput:
    return ModelPredictionOutput(
        status=status,
        label=None,
        confidence=None,
        probabilities=None,
        detail=detail,
    )


@dataclass(frozen=True)
class _PredictionAttempt:
    output: ModelPredictionOutput
    prediction: PredictResponse | None = None
    token_count: int | None = None
    max_tokens: int | None = None
    truncated: bool | None = None


def _tfidf_prediction(
    payload: PredictRequest,
    model_service: ModelService,
) -> tuple[_PredictionAttempt, ModelState]:
    state = model_service.get_state()
    if not state.loaded or state.classifier is None:
        return (
            _PredictionAttempt(
                output=_empty_model_output(
                    "unavailable",
                    state.error or "TF-IDF model artifact is unavailable.",
                )
            ),
            state,
        )
    try:
        prediction = _classify(payload, model_service)
    except HTTPException:
        return (
            _PredictionAttempt(
                output=_empty_model_output("error", "TF-IDF prediction failed for this request.")
            ),
            state,
        )
    return (
        _PredictionAttempt(
            output=_available_output(prediction.label, prediction.probabilities),
            prediction=prediction,
        ),
        state,
    )


def _distilbert_prediction(
    payload: PredictRequest,
    distilbert_service: Any,
) -> tuple[_PredictionAttempt, DistilBertState]:
    state = distilbert_service.get_state()
    if not state.loaded or state.runtime is None:
        return (
            _PredictionAttempt(
                output=_empty_model_output(
                    "unavailable",
                    state.error or "Fine-tuned DistilBERT artifact is unavailable.",
                )
            ),
            state,
        )
    try:
        result = state.runtime.predict_one(payload.text)
        label = normalize_prediction_label(result.label)
        probabilities = normalize_probabilities(result.probabilities)
        if probabilities is None or label not in probabilities:
            raise ValueError("Invalid DistilBERT probabilities.")
        if probabilities[label] != max(probabilities.values()):
            raise ValueError("DistilBERT label does not match its probability distribution.")
    except Exception as exc:
        logger.warning("prediction_failed model=distilbert error_type=%s", type(exc).__name__)
        return (
            _PredictionAttempt(
                output=_empty_model_output(
                    "error",
                    "DistilBERT prediction failed for this request.",
                )
            ),
            state,
        )
    return (
        _PredictionAttempt(
            output=_available_output(label, probabilities),
            prediction=PredictResponse(
                label=label,
                probabilities=probabilities,
                risk_level=risk_level_for(label, probabilities),
                explanation=explanation_for(label, probabilities, payload.text),
                suggested_action=suggested_action_for(
                    label,
                    risk_level_for(label, probabilities),
                ),
            ),
            token_count=getattr(result, "token_count", None),
            max_tokens=getattr(result, "max_tokens", None),
            truncated=getattr(result, "truncated", None),
        ),
        state,
    )


def _duel_explanation(
    *,
    agreement: Agreement,
    tfidf: ModelPredictionOutput,
    distilbert: ModelPredictionOutput,
    final_model: FinalModel,
    final_label: ThreatLabel,
    final_probabilities: dict[str, float] | None,
    text: str,
) -> str:
    if agreement == "agreed":
        context = f"TF-IDF and DistilBERT both classified this message as {final_label}."
    elif agreement == "disagreed":
        context = (
            f"TF-IDF classified this message as {tfidf.label}, while DistilBERT classified "
            f"it as {distilbert.label}. The models disagree; the selected result uses "
            "DistilBERT, and manual review is advised."
        )
    elif final_model == "distilbert":
        unavailable_model = "failed" if tfidf.status == "error" else "is unavailable"
        context = (
            f"DistilBERT produced the selected result; TF-IDF {unavailable_model} for this request."
        )
    else:
        unavailable_model = "failed" if distilbert.status == "error" else "is unavailable"
        context = (
            f"TF-IDF produced the fallback result because DistilBERT {unavailable_model} for "
            "this request."
        )
    return f"{context} {explanation_for(final_label, final_probabilities, text)}"


def _review_reasons(
    *,
    agreement: Agreement,
    fallback_reason: FallbackReason | None,
    confidence: float | None,
    truncated: bool | None,
    heuristic_signals: list[str],
) -> list[ReviewReason]:
    reasons: list[ReviewReason] = []
    if agreement == "disagreed":
        reasons.append("model_disagreement")
    if fallback_reason == "primary_unavailable":
        reasons.append("primary_unavailable")
    elif fallback_reason == "primary_prediction_failed":
        reasons.append("primary_prediction_failed")
    if truncated is True:
        reasons.append("input_truncated")
    signal_set = set(heuristic_signals)
    if "credential request" in signal_set and signal_set.intersection(
        {"urgency", "link or contact prompt"}
    ):
        reasons.append("suspicious_text_signals")
    if confidence is None or confidence < LOW_MODEL_SCORE_THRESHOLD:
        reasons.append("low_model_score")
    return reasons


def _duel_prediction(
    payload: PredictRequest,
    model_service: ModelService,
    distilbert_service: Any,
    *,
    distilbert_max_tokens: int,
) -> VersionedPredictResponse:
    tfidf_attempt, tfidf_state = _tfidf_prediction(payload, model_service)
    distilbert_attempt, distilbert_state = _distilbert_prediction(payload, distilbert_service)
    tfidf = tfidf_attempt.prediction
    distilbert = distilbert_attempt.prediction

    if tfidf is not None and distilbert is not None:
        agreement: Agreement = "agreed" if tfidf.label == distilbert.label else "disagreed"
    elif tfidf_attempt.output.status == "error" or distilbert_attempt.output.status == "error":
        agreement = "partial"
    else:
        agreement = "unavailable"

    if distilbert is not None:
        final_model: FinalModel = "distilbert"
        final_prediction = distilbert
        fallback_reason: FallbackReason | None = None
    elif tfidf is not None:
        final_model = "tfidf_logreg"
        final_prediction = tfidf
        fallback_reason = (
            "primary_prediction_failed"
            if distilbert_attempt.output.status == "error"
            else "primary_unavailable"
        )
    else:
        logger.warning(
            "prediction_unavailable tfidf_status=%s distilbert_status=%s",
            tfidf_attempt.output.status,
            distilbert_attempt.output.status,
        )
        raise HTTPException(
            status_code=503,
            detail="No prediction model is currently available.",
        )

    final_label = final_prediction.label
    final_probabilities = final_prediction.probabilities
    final_confidence = confidence_for(final_label, final_probabilities)
    score_kind: ScoreKind = (
        "uncalibrated_probability" if final_probabilities is not None else "label_only"
    )
    heuristic_signals = matched_signals(payload.text)
    review_reasons = _review_reasons(
        agreement=agreement,
        fallback_reason=fallback_reason,
        confidence=final_confidence,
        truncated=distilbert_attempt.truncated,
        heuristic_signals=heuristic_signals,
    )
    final_risk = risk_level_for(final_label, final_probabilities)
    if final_risk == "low" and review_reasons:
        final_risk = "medium"
    suggested_action = suggested_action_for(final_label, final_risk)
    if final_label == "ham" and "suspicious_text_signals" in review_reasons:
        suggested_action = (
            "Do not use links or provide credentials until you independently verify the sender "
            "and request through a trusted channel."
        )
    return VersionedPredictResponse(
        final_label=final_label,
        final_risk_level=final_risk,
        final_confidence=final_confidence,
        final_model=final_model,
        fallback_reason=fallback_reason,
        score_kind=score_kind,
        review_recommended=bool(review_reasons),
        review_reasons=review_reasons,
        input_metadata=InputMetadata(
            character_count=len(payload.text),
            transformer_tokens=distilbert_attempt.token_count,
            transformer_max_tokens=distilbert_attempt.max_tokens or distilbert_max_tokens,
            truncated=distilbert_attempt.truncated,
        ),
        heuristic_signals=heuristic_signals,
        agreement=agreement,
        model_outputs={
            "tfidf_logreg": tfidf_attempt.output,
            "distilbert": distilbert_attempt.output,
        },
        explanation=_duel_explanation(
            agreement=agreement,
            tfidf=tfidf_attempt.output,
            distilbert=distilbert_attempt.output,
            final_model=final_model,
            final_label=final_label,
            final_probabilities=final_probabilities,
            text=payload.text,
        ),
        suggested_action=suggested_action,
        artifact_metadata=ArtifactMetadata(**public_artifact_metadata(tfidf_state)),
        model_manifests={
            "tfidf_logreg": public_artifact_metadata(tfidf_state),
            "distilbert": public_manifest(distilbert_state.manifest)
            if distilbert_state.loaded
            else None,
        },
    )


def _readiness_state(service: Any, *, nonblocking: bool) -> tuple[Any | None, bool]:
    """Read warmup progress without making readiness wait on heavyweight model loading."""
    peek_state = getattr(service, "peek_state", None)
    if nonblocking and callable(peek_state):
        return peek_state(), bool(getattr(service, "loading", False))
    return service.get_state(), False


def create_app(
    settings: AppSettings | None = None,
    *,
    distilbert_service: Any | None = None,
) -> FastAPI:
    """Create the API with versioned duel behavior and exact legacy compatibility."""
    app_settings = settings or AppSettings.from_env()
    model_service = ModelService(
        app_settings,
        artifact_downloader=lambda url, destination: _download_model_artifact(
            url,
            destination,
            app_settings.max_artifact_bytes,
        ),
    )

    if distilbert_service is None:

        def distilbert_downloader(url: str, destination: Path) -> Path:
            try:
                return _download_model_artifact(
                    url,
                    destination,
                    app_settings.max_artifact_bytes,
                )
            except ModelArtifactError as exc:
                raise DistilBertArtifactError(str(exc)) from exc

        distilbert_service = DistilBertService(
            app_settings,
            downloader=distilbert_downloader,
        )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        if app_settings.background_warmup:
            model_service.start_background_warmup()
            start_distilbert_warmup = getattr(distilbert_service, "start_background_warmup", None)
            if callable(start_distilbert_warmup):
                start_distilbert_warmup()
        yield

    app = FastAPI(
        title="ThreatLens API",
        summary="Artifact-backed spam and phishing classification.",
        version="1.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(PredictionSecurityMiddleware, settings=app_settings)
    # Add CORS last so it wraps middleware-generated 4xx/5xx responses too.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(app_settings.allowed_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-Request-ID"],
        expose_headers=["X-Request-ID", "Retry-After"],
    )

    app.state.settings = app_settings
    app.state.model_service = model_service
    app.state.distilbert_service = distilbert_service
    app.state.metrics = load_metrics(app_settings.metrics_path)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        _request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": _validation_message(exc)})

    @app.get("/health", response_model=HealthResponse, include_in_schema=False)
    def health() -> JSONResponse | HealthResponse:
        response = _health_response(model_service.get_state())
        if not response.model_loaded:
            return JSONResponse(status_code=503, content=response.model_dump())
        return response

    @app.get("/v1/ready", response_model=ReadyResponse)
    def ready() -> JSONResponse | ReadyResponse:
        tfidf_state, tfidf_loading = _readiness_state(
            model_service,
            nonblocking=app_settings.background_warmup,
        )
        distilbert_state, distilbert_loading = _readiness_state(
            distilbert_service,
            nonblocking=app_settings.background_warmup,
        )
        response = _versioned_ready(
            tfidf_state,
            distilbert_state,
            tfidf_loading=tfidf_loading,
            distilbert_loading=distilbert_loading,
        )
        if not response.ready:
            return JSONResponse(status_code=503, content=response.model_dump())
        return response

    @app.get("/live", response_model=LiveResponse, include_in_schema=False)
    @app.get("/v1/live", response_model=LiveResponse)
    def live() -> LiveResponse:
        return LiveResponse(status="ok", app_name=APP_NAME, detail="API process is running.")

    @app.get("/metadata", response_model=MetadataResponse, include_in_schema=False)
    def metadata() -> MetadataResponse:
        tfidf_state, tfidf_loading = _readiness_state(
            model_service,
            nonblocking=app_settings.background_warmup,
        )
        return _legacy_metadata(
            app_settings,
            tfidf_state,
            app.state.metrics,
            loading=tfidf_loading,
        )

    @app.get("/v1/metadata", response_model=VersionedMetadataResponse)
    def versioned_metadata() -> VersionedMetadataResponse:
        tfidf_state, tfidf_loading = _readiness_state(
            model_service,
            nonblocking=app_settings.background_warmup,
        )
        distilbert_state, distilbert_loading = _readiness_state(
            distilbert_service,
            nonblocking=app_settings.background_warmup,
        )
        legacy = _legacy_metadata(
            app_settings,
            tfidf_state,
            app.state.metrics,
            loading=tfidf_loading,
        )
        return _versioned_metadata(
            legacy,
            tfidf_state,
            distilbert_state,
            tfidf_loading=tfidf_loading,
            distilbert_loading=distilbert_loading,
        )

    @app.post("/predict", response_model=PredictResponse, include_in_schema=False)
    def legacy_predict(payload: PredictRequest) -> PredictResponse:
        return _classify(payload, model_service)

    @app.post("/v1/predict", response_model=VersionedPredictResponse)
    def predict(payload: PredictRequest) -> VersionedPredictResponse:
        return _duel_prediction(
            payload,
            model_service,
            distilbert_service,
            distilbert_max_tokens=app_settings.distilbert_max_length,
        )

    return app


app = create_app()

__all__ = ["AppSettings", "ModelArtifactError", "app", "create_app"]
