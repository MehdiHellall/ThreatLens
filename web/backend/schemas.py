"""Validated HTTP request and response schemas for ThreatLens."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from web.backend.settings import MAX_TEXT_CHARS

RiskLevel = Literal["low", "medium", "high"]
ThreatLabel = Literal["ham", "phish", "spam"]
Agreement = Literal["agreed", "disagreed", "unavailable", "partial"]
ModelStatus = Literal["available", "unavailable", "error"]
ModelLifecycle = Literal["loading", "retrying", "ready", "unavailable", "error"]
ServiceStatus = Literal["starting", "ready", "limited", "unavailable"]
FinalModel = Literal["tfidf_logreg", "distilbert"]
FallbackReason = Literal["primary_unavailable", "primary_prediction_failed"]
ScoreKind = Literal["uncalibrated_probability", "label_only"]
ReviewReason = Literal[
    "model_disagreement",
    "primary_unavailable",
    "primary_prediction_failed",
    "input_truncated",
    "suspicious_text_signals",
    "low_model_score",
]


class HealthResponse(BaseModel):
    status: Literal["ok", "error"]
    model_loaded: bool
    model_path: str | None
    detail: str


class ModelAvailability(BaseModel):
    available: bool
    detail: str
    state: ModelLifecycle


class ReadyResponse(HealthResponse):
    ready: bool
    primary_ready: bool
    service_status: ServiceStatus
    duel_ready: bool
    models: dict[str, ModelAvailability]


class LiveResponse(BaseModel):
    status: Literal["ok"]
    app_name: str
    detail: str


class MetadataResponse(BaseModel):
    app_name: str
    labels: list[str]
    max_text_chars: int
    model: dict[str, object]
    metrics: dict[str, object] | None
    privacy: str


class VersionedMetadataResponse(MetadataResponse):
    models: dict[str, dict[str, object]]


class PredictRequest(BaseModel):
    text: str = Field(..., max_length=MAX_TEXT_CHARS)

    @field_validator("text", mode="before")
    @classmethod
    def clean_text(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("text must not be empty.")
        if len(trimmed) > MAX_TEXT_CHARS:
            raise ValueError(f"text must be {MAX_TEXT_CHARS} characters or fewer.")
        return trimmed


class PredictResponse(BaseModel):
    """Legacy prediction shape retained for existing clients."""

    label: ThreatLabel
    probabilities: dict[str, float] | None
    risk_level: RiskLevel
    explanation: str
    suggested_action: str


class ModelPredictionOutput(BaseModel):
    status: ModelStatus
    label: ThreatLabel | None
    confidence: float | None
    probabilities: dict[str, float] | None
    detail: str


class ArtifactMetadata(BaseModel):
    """Real deployment metadata; unavailable artifact fields remain explicitly null."""

    artifact: str | None
    model_name: str | None
    metrics_file: str | None


class InputMetadata(BaseModel):
    """Non-sensitive facts about how the submitted text was processed."""

    character_count: int
    transformer_tokens: int | None
    transformer_max_tokens: int
    truncated: bool | None


class VersionedPredictResponse(BaseModel):
    """Versioned result containing the two honest Model Duel outputs."""

    final_label: ThreatLabel
    final_risk_level: RiskLevel
    final_confidence: float | None
    final_model: FinalModel
    fallback_reason: FallbackReason | None
    score_kind: ScoreKind
    review_recommended: bool
    review_reasons: list[ReviewReason]
    input_metadata: InputMetadata
    heuristic_signals: list[str]
    agreement: Agreement
    model_outputs: dict[str, ModelPredictionOutput]
    explanation: str
    suggested_action: str
    artifact_metadata: ArtifactMetadata
    model_manifests: dict[str, dict[str, object] | None]
