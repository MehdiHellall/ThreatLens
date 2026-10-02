"""Transparent analyst guidance derived from a model prediction."""

from __future__ import annotations

import re

from web.backend.schemas import RiskLevel

TRANSPARENT_SIGNALS = {
    "urgency": re.compile(r"\b(urgent|immediately|expires?|suspended|locked|final notice)\b", re.I),
    "credential request": re.compile(
        r"\b(password|passcode|login|verify|verification|credentials?|account)\b",
        re.I,
    ),
    "money or prize language": re.compile(
        r"\b(free|prize|winner|cash|coupon|discount|limited offer|buy now)\b",
        re.I,
    ),
    "link or contact prompt": re.compile(
        r"(https?://|www\.|\bclick\b|\breply\b|\bcall\b)",
        re.I,
    ),
}


def confidence_for(label: str, probabilities: dict[str, float] | None) -> float | None:
    """Return the probability assigned to the selected label, when available."""
    if probabilities is None:
        return None
    return probabilities.get(label)


def risk_level_for(label: str, probabilities: dict[str, float] | None) -> RiskLevel:
    """Map the existing label and confidence behavior to an analyst risk level."""
    confidence = confidence_for(label, probabilities)
    if confidence is not None and confidence < 0.55:
        return "medium"
    if label == "phish":
        return "high"
    if label == "spam":
        return "medium"
    return "low"


def matched_signals(text: str) -> list[str]:
    """Return deterministic heuristics that are separate from model attribution."""
    return [name for name, pattern in TRANSPARENT_SIGNALS.items() if pattern.search(text)]


def explanation_for(label: str, probabilities: dict[str, float] | None, text: str) -> str:
    """Describe the result without treating a raw model probability as calibrated certainty."""
    score = confidence_for(label, probabilities)
    score_text = f" with a {score:.0%} model score" if score is not None else ""
    signals = matched_signals(text)
    if signals:
        signal_text = ", ".join(signals[:3])
        return (
            f"The model classified this message as {label}{score_text}. "
            f"A separate keyword check observed: {signal_text}; these signals are not an "
            "explanation of the model's reasoning."
        )
    return (
        f"The model classified this message as {label}{score_text}. "
        "A separate keyword check found no listed signal; the model's internal reasoning is not "
        "exposed."
    )


def suggested_action_for(label: str, risk_level: RiskLevel) -> str:
    """Return conservative handling guidance for the current result."""
    if label == "phish":
        return (
            "Do not click links or share credentials. Verify the request through a trusted "
            "channel and report it to your security team."
        )
    if label == "spam":
        return (
            "Avoid links, attachments, replies, and requests for credentials. Mark it as spam "
            "or delete it if it is unsolicited."
        )
    if risk_level == "medium":
        return "Review the sender and context before acting; the model confidence is not decisive."
    return "Low apparent risk. Continue normal handling, while still checking sender context."
