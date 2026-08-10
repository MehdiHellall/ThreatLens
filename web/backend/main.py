"""FastAPI application factory and versioned ThreatLens Model Duel routes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from web.backend.explanations import (
    confidence_for,
    explanation_for,
    risk_level_for,
    suggested_action_for,
)
from web.backend.labels import LABEL_NAMES
from web.backend.runtimes.distilbert_runtime import (
    DistilBertArtifactError,
    DistilBertService,
    DistilBertState,
    public_manifest,
)
from web.backend.schemas import (
    Agreement,
    ArtifactMetadata,
    HealthResponse,
    LiveResponse,
    MetadataResponse,
    ModelAvailability,
    ModelPredictionOutput,
    ModelStatus,
    PredictRequest,
    PredictResponse,
    ReadyResponse,
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

    result = state.classifier.predict_one(payload.text)
    label = normalize_prediction_label(result.label)
    probabilities = normalize_probabilities(result.probabilities)
    risk_level = risk_level_for(label, probabilities)
    return PredictResponse(
        label=label,
        probabilities=probabilities,
        risk_level=risk_level,
        explanation=explanation_for(label, probabilities, payload.text),
        suggested_action=suggested_action_for(label, risk_level),
    )


def _availability(*, available: bool, detail: str) -> ModelAvailability:
    return ModelAvailability(available=available, detail=detail)


def _versioned_ready(
    tfidf_state: ModelState,
    distilbert_state: DistilBertState,
) -> ReadyResponse:
    legacy = _health_response(tfidf_state)
    return ReadyResponse(
        **legacy.model_dump(),
        duel_ready=tfidf_state.loaded and distilbert_state.loaded,
        models={
            "tfidf_logreg": _availability(
                available=tfidf_state.loaded,
                detail="Model artifact loaded."
                if tfidf_state.loaded
                else tfidf_state.error or "Model artifact is unavailable.",
            ),
            "distilbert": _availability(
                available=distilbert_state.loaded,
                detail="Fine-tuned model artifact loaded."
                if distilbert_state.loaded
                else distilbert_state.error or "Fine-tuned model artifact is unavailable.",
            ),
        },
    )


def _legacy_metadata(
    settings: AppSettings,
    tfidf_state: ModelState,
    metrics: dict[str, object] | None,
) -> MetadataResponse:
    return MetadataResponse(
        app_name=APP_NAME,
        labels=list(LABEL_NAMES),
        max_text_chars=settings.max_text_chars,
        model={
            "loaded": tfidf_state.loaded,
            "artifact": public_path(tfidf_state.path),
            "metadata": public_model_metadata(tfidf_state.metadata),
            "status": "ready" if tfidf_state.loaded else tfidf_state.error,
        },
        metrics=metrics,
        privacy=PRIVACY_NOTICE,
    )


def _versioned_metadata(
    legacy: MetadataResponse,
    tfidf_state: ModelState,
    distilbert_state: DistilBertState,
) -> VersionedMetadataResponse:
    return VersionedMetadataResponse(
        **legacy.model_dump(),
        models={
            "tfidf_logreg": {
                "available": tfidf_state.loaded,
                "artifact": public_path(tfidf_state.path),
                "manifest": public_artifact_metadata(tfidf_state),
                "status": "ready" if tfidf_state.loaded else tfidf_state.error,
            },
            "distilbert": {
                "available": distilbert_state.loaded,
                "manifest": public_manifest(distilbert_state.manifest)
                if distilbert_state.loaded
                else None,
                "status": "ready" if distilbert_state.loaded else distilbert_state.error,
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


def _empty_distilbert_output(status: ModelStatus, detail: str) -> ModelPredictionOutput:
    return ModelPredictionOutput(
        status=status,
        label=None,
        confidence=None,
        probabilities=None,
        detail=detail,
    )


def _duel_explanation(
    *,
    agreement: Agreement,
    tfidf: PredictResponse,
    final_label: ThreatLabel,
    final_probabilities: dict[str, float] | None,
    text: str,
) -> str:
    if agreement == "unavailable":
        return (
            f"{tfidf.explanation} DistilBERT is unavailable, so the final recommendation "
            "uses TF-IDF only."
        )
    if agreement == "partial":
        return (
            f"{tfidf.explanation} DistilBERT failed for this request, so this partial duel "
            "uses TF-IDF only."
        )
    if agreement == "agreed":
        return (
            f"TF-IDF and DistilBERT both classified this message as {final_label}. "
            f"{explanation_for(final_label, final_probabilities, text)}"
        )
    return (
        f"TF-IDF classified this message as {tfidf.label}, while DistilBERT classified it "
        f"as {final_label}. The models disagree; the final recommendation uses DistilBERT, "
        "and manual review is advised. "
        f"{explanation_for(final_label, final_probabilities, text)}"
    )


def _duel_prediction(
    payload: PredictRequest,
    model_service: ModelService,
    distilbert_service: Any,
) -> VersionedPredictResponse:
    tfidf = _classify(payload, model_service)
    tfidf_state = model_service.get_state()
    tfidf_output = _available_output(tfidf.label, tfidf.probabilities)
    distilbert_state = distilbert_service.get_state()

    agreement: Agreement = "unavailable"
    final_label = tfidf.label
    final_probabilities = tfidf.probabilities
    if not distilbert_state.loaded or distilbert_state.runtime is None:
        distilbert_output = _empty_distilbert_output(
            "unavailable",
            distilbert_state.error or "Fine-tuned DistilBERT artifact is unavailable.",
        )
    else:
        try:
            prediction = distilbert_state.runtime.predict_one(payload.text)
            distilbert_label = normalize_prediction_label(prediction.label)
            distilbert_probabilities = normalize_probabilities(prediction.probabilities)
            if distilbert_probabilities is None or distilbert_label not in distilbert_probabilities:
                raise ValueError("Incomplete DistilBERT probabilities.")
            distilbert_output = _available_output(
                distilbert_label,
                distilbert_probabilities,
            )
            agreement = "agreed" if distilbert_label == tfidf.label else "disagreed"
            final_label = distilbert_label
            final_probabilities = distilbert_probabilities
        except Exception:
            agreement = "partial"
            distilbert_output = _empty_distilbert_output(
                "error",
                "DistilBERT prediction failed for this request.",
            )

    final_risk = risk_level_for(final_label, final_probabilities)
    return VersionedPredictResponse(
        final_label=final_label,
        final_risk_level=final_risk,
        final_confidence=confidence_for(final_label, final_probabilities),
        agreement=agreement,
        model_outputs={
            "tfidf_logreg": tfidf_output,
            "distilbert": distilbert_output,
        },
        explanation=_duel_explanation(
            agreement=agreement,
            tfidf=tfidf,
            final_label=final_label,
            final_probabilities=final_probabilities,
            text=payload.text,
        ),
        suggested_action=suggested_action_for(final_label, final_risk),
        artifact_metadata=ArtifactMetadata(**public_artifact_metadata(tfidf_state)),
        model_manifests={
            "tfidf_logreg": public_artifact_metadata(tfidf_state),
            "distilbert": public_manifest(distilbert_state.manifest)
            if distilbert_state.loaded
            else None,
        },
    )


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

    app = FastAPI(
        title="ThreatLens API",
        summary="Artifact-backed spam and phishing classification.",
        version="1.1.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(app_settings.allowed_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    app.add_middleware(PredictionSecurityMiddleware, settings=app_settings)

    app.state.settings = app_settings
    app.state.model_service = model_service
    app.state.distilbert_service = distilbert_service
    app.state.metrics = load_metrics(app_settings.metrics_path)
    if app_settings.background_warmup:
        model_service.start_background_warmup()

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
        response = _versioned_ready(
            model_service.get_state(),
            distilbert_service.get_state(),
        )
        if not response.model_loaded:
            return JSONResponse(status_code=503, content=response.model_dump())
        return response

    @app.get("/live", response_model=LiveResponse, include_in_schema=False)
    @app.get("/v1/live", response_model=LiveResponse)
    def live() -> LiveResponse:
        return LiveResponse(status="ok", app_name=APP_NAME, detail="API process is running.")

    @app.get("/metadata", response_model=MetadataResponse, include_in_schema=False)
    def metadata() -> MetadataResponse:
        return _legacy_metadata(
            app_settings,
            model_service.get_state(),
            app.state.metrics,
        )

    @app.get("/v1/metadata", response_model=VersionedMetadataResponse)
    def versioned_metadata() -> VersionedMetadataResponse:
        tfidf_state = model_service.get_state()
        legacy = _legacy_metadata(app_settings, tfidf_state, app.state.metrics)
        return _versioned_metadata(
            legacy,
            tfidf_state,
            distilbert_service.get_state(),
        )

    @app.post("/predict", response_model=PredictResponse, include_in_schema=False)
    def legacy_predict(payload: PredictRequest) -> PredictResponse:
        return _classify(payload, model_service)

    @app.post("/v1/predict", response_model=VersionedPredictResponse)
    def predict(payload: PredictRequest) -> VersionedPredictResponse:
        return _duel_prediction(payload, model_service, distilbert_service)

    return app


app = create_app()

__all__ = ["AppSettings", "ModelArtifactError", "app", "create_app"]
