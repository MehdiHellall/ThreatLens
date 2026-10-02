"""Leakage-safe DistilBERT data preparation, fine-tuning, and export."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from numbers import Integral, Real
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

from email_threat_detector.distilbert_artifacts import (
    RunContractError,
    artifact_fingerprint,
    create_artifact_archive,
    ensure_run_contract,
    find_resume_checkpoint,
)
from email_threat_detector.preprocessing import basic_text_preprocessor, normalize_text

__all__ = [
    "PreparedDataError",
    "RunContractError",
    "TrainingConfig",
    "TrainingDependencyError",
    "adopt_prepared_splits",
    "create_artifact_archive",
    "finalize_artifact_manifest",
    "find_resume_checkpoint",
    "initialize_run_contract",
    "main",
    "prepare_splits",
    "train_and_export",
    "validate_prepared_splits",
]

BASE_CHECKPOINT = "distilbert/distilbert-base-uncased"
BASE_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"
LABEL_TO_ID = {"ham": 0, "phish": 1, "spam": 2}
ID_TO_LABEL = {identifier: label for label, identifier in LABEL_TO_ID.items()}
LABEL_ALIASES = {
    "0": "ham",
    "1": "phish",
    "2": "spam",
    "legit": "ham",
    "legitimate": "ham",
    "normal": "ham",
    "phishing": "phish",
    "scam": "phish",
    "junk": "spam",
}
SPLIT_NAMES = ("train", "validation", "test")


class PreparedDataError(ValueError):
    """Raised when prepared split files cannot safely be used for training."""


class TrainingDependencyError(RuntimeError):
    """Raised when the optional Hugging Face training stack is unavailable."""


@dataclass(frozen=True)
class TrainingConfig:
    """Notebook-compatible DistilBERT fine-tuning configuration."""

    base_checkpoint: str = BASE_CHECKPOINT
    base_revision: str = BASE_REVISION
    max_length: int = 128
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    save_total_limit: int = 1
    seed: int = 42

    def __post_init__(self) -> None:
        if self.base_checkpoint != BASE_CHECKPOINT:
            raise ValueError(f"base checkpoint must be {BASE_CHECKPOINT}.")
        if self.base_revision != BASE_REVISION:
            raise ValueError(f"base revision must be {BASE_REVISION}.")
        if self.max_length <= 0:
            raise ValueError("max_length must be positive.")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write bytes instead of text so Windows cannot translate LF to CRLF after
    # the artifact checksums have been calculated.
    path.write_bytes(
        (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    )


def _normalize_label(value: object) -> str:
    if value is None or bool(pd.isna(value)):
        raise ValueError("Label cannot be empty.")
    if isinstance(value, Integral) or (
        isinstance(value, Real) and math.isfinite(float(value)) and float(value).is_integer()
    ):
        key = str(int(value))
    else:
        key = str(value).strip().casefold()
    canonical = LABEL_ALIASES.get(key, key)
    if canonical not in LABEL_TO_ID:
        raise ValueError(f"Unknown label: {value!r}")
    return canonical


def _load_source(source_path: Path) -> tuple[pd.DataFrame, int]:
    if not source_path.is_file():
        raise FileNotFoundError(f"Dataset does not exist: {source_path}")
    source = pd.read_csv(source_path)
    missing = {"text", "label"}.difference(source.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise PreparedDataError(f"Dataset is missing required column(s): {names}")

    selected = source.loc[:, ["text", "label"]]
    selected = selected.dropna(subset=["text", "label"]).copy()
    selected = selected.assign(
        text=selected["text"].map(normalize_text),
        label=selected["label"].map(_normalize_label),
    )
    cleaned = selected.loc[selected["text"].astype(bool)].reset_index(drop=True)
    return cleaned, int(len(source))


def _remove_conflicts_and_duplicates(frame: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    keyed = frame.assign(_text_key=frame["text"].map(basic_text_preprocessor))
    groups = keyed.groupby("_text_key", sort=True)["label"].agg(["size", "nunique"])
    conflict_keys = set(groups.index[groups["nunique"] > 1])
    duplicate_group_count = int((groups["size"] > 1).sum())
    retained = keyed.loc[~keyed["_text_key"].isin(conflict_keys)]
    deduplicated = retained.drop_duplicates(subset=["_text_key"], keep="first")
    result = deduplicated.loc[:, ["text", "label"]].reset_index(drop=True)
    return result, len(conflict_keys), duplicate_group_count


def _balanced(frame: pd.DataFrame, *, samples_per_class: int | None, seed: int) -> pd.DataFrame:
    counts = frame["label"].value_counts()
    missing = set(LABEL_TO_ID).difference(counts.index)
    if missing:
        names = ", ".join(sorted(missing))
        raise PreparedDataError(f"Dataset is missing required label(s): {names}")
    target = int(counts.min()) if samples_per_class is None else samples_per_class
    if target <= 0:
        raise ValueError("samples_per_class must be positive.")
    if (counts < target).any():
        raise PreparedDataError("samples_per_class exceeds at least one class count.")

    sampled = tuple(
        group.sample(n=target, random_state=seed) for _, group in frame.groupby("label", sort=True)
    )
    return pd.concat(sampled).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _split(frame: pd.DataFrame, *, seed: int) -> dict[str, pd.DataFrame]:
    try:
        train_validation, test = train_test_split(
            frame,
            test_size=0.1,
            random_state=seed,
            stratify=frame["label"],
        )
        train, validation = train_test_split(
            train_validation,
            test_size=0.1 / 0.9,
            random_state=seed,
            stratify=train_validation["label"],
        )
    except ValueError as exc:
        raise PreparedDataError(
            "Dataset is too small for deterministic stratified 80/10/10 splits."
        ) from exc
    return {
        "train": train.reset_index(drop=True),
        "validation": validation.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def _assert_no_overlap(splits: Mapping[str, pd.DataFrame]) -> None:
    normalized = {
        name: set(frame["text"].map(basic_text_preprocessor)) for name, frame in splits.items()
    }
    pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
    overlaps = {
        f"{left}/{right}": len(normalized[left] & normalized[right])
        for left, right in pairs
        if normalized[left] & normalized[right]
    }
    if overlaps:
        detail = ", ".join(f"{pair}: {count}" for pair, count in overlaps.items())
        raise PreparedDataError(f"Normalized-text overlap detected across splits ({detail}).")


def _validate_balanced_split_contract(splits: Mapping[str, pd.DataFrame]) -> None:
    for name, frame in splits.items():
        normalized = frame["text"].map(basic_text_preprocessor)
        if normalized.duplicated().any():
            raise PreparedDataError(f"Prepared {name} split contains duplicate normalized text.")
        counts = frame["label"].value_counts().to_dict()
        if set(counts) != set(LABEL_TO_ID) or len(set(counts.values())) != 1:
            raise PreparedDataError(f"Prepared {name} split is not balanced across all labels.")

    rows = {name: len(frame) for name, frame in splits.items()}
    total = sum(rows.values())
    expected_test = (total + 9) // 10
    train_validation = total - expected_test
    expected_validation = (train_validation + 8) // 9
    expected_rows = {
        "train": train_validation - expected_validation,
        "validation": expected_validation,
        "test": expected_test,
    }
    if rows != expected_rows:
        raise PreparedDataError("Prepared splits do not follow the expected 80/10/10 allocation.")


def prepare_splits(
    source_path: str | Path,
    output_dir: str | Path,
    *,
    samples_per_class: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Create balanced, deterministic, normalized-text-disjoint 80/10/10 splits."""
    source = Path(source_path)
    destination = Path(output_dir)
    if samples_per_class is not None and samples_per_class <= 0:
        raise ValueError("samples_per_class must be positive.")

    loaded, input_rows = _load_source(source)
    modeling, conflict_count, duplicate_group_count = _remove_conflicts_and_duplicates(loaded)
    modeling = _balanced(modeling, samples_per_class=samples_per_class, seed=seed)
    splits = _split(modeling, seed=seed)
    _validate_balanced_split_contract(splits)
    _assert_no_overlap(splits)

    destination.mkdir(parents=True, exist_ok=True)
    split_manifest: dict[str, dict[str, Any]] = {}
    for name in SPLIT_NAMES:
        path = destination / f"{name}.csv"
        split = splits[name].loc[:, ["text", "label"]]
        split.to_csv(path, index=False, lineterminator="\n")
        split_manifest[name] = {
            "filename": path.name,
            "rows": int(len(split)),
            "sha256": _sha256(path),
            "class_counts": {
                label: int(count)
                for label, count in split["label"].value_counts().sort_index().items()
            },
        }

    data_config = {
        "balance_classes": True,
        "random_state": seed,
        "samples_per_class": samples_per_class,
        "test_size": 0.1,
        "validation_size": 0.1,
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "dataset": {
            "filename": source.name,
            "sha256": _sha256(source),
        },
        "data_config": data_config,
        "data_summary": {
            "input_rows": input_rows,
            "clean_rows": int(len(loaded)),
            "modeling_rows": int(len(modeling)),
            "duplicate_text_groups": duplicate_group_count,
            "conflicting_text_groups_removed": conflict_count,
            "class_counts": {name: split_manifest[name]["class_counts"] for name in SPLIT_NAMES},
        },
        "splits": split_manifest,
        "notes": [
            "Labels and text were normalized before duplicate analysis.",
            "Conflicting normalized-text groups were removed before splitting.",
            "Normalized text is disjoint across train, validation, and test splits.",
        ],
    }
    _write_json(destination / "manifest.json", manifest)
    return manifest


def _validated_split_frame(path: Path, split_name: str) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        raise PreparedDataError(f"Unable to read {split_name} split.") from exc
    if list(frame.columns) != ["text", "label"]:
        raise PreparedDataError(
            f"{split_name} split must contain exactly the text and label columns."
        )
    if frame.empty or frame[["text", "label"]].isna().any().any():
        raise PreparedDataError(f"{split_name} split contains empty data.")
    try:
        canonical_labels = frame["label"].map(_normalize_label)
    except ValueError as exc:
        raise PreparedDataError(f"{split_name} split contains an invalid label.") from exc
    if not canonical_labels.equals(frame["label"]):
        raise PreparedDataError(f"{split_name} split labels are not canonical.")
    if frame["text"].map(normalize_text).ne(frame["text"]).any():
        raise PreparedDataError(f"{split_name} split text is not canonical.")
    return frame


def validate_prepared_splits(split_dir: str | Path) -> dict[str, Any]:
    """Validate split checksums, schema, labels, row counts, and leakage safety."""
    directory = Path(split_dir)
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise PreparedDataError("Prepared split manifest is missing.")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreparedDataError("Prepared split manifest is invalid.") from exc
    split_metadata = manifest.get("splits")
    if not isinstance(split_metadata, dict):
        raise PreparedDataError("Prepared split manifest has no split metadata.")

    frames: dict[str, pd.DataFrame] = {}
    for name in SPLIT_NAMES:
        metadata = split_metadata.get(name)
        if not isinstance(metadata, dict):
            raise PreparedDataError(f"Prepared split manifest is missing {name} metadata.")
        filename = metadata.get("filename", f"{name}.csv")
        if filename != f"{name}.csv":
            raise PreparedDataError(f"Prepared {name} split filename is invalid.")
        path = directory / filename
        if not path.is_file():
            raise PreparedDataError(f"Prepared {name} split is missing.")
        if metadata.get("sha256") != _sha256(path):
            raise PreparedDataError(f"Prepared {name} split checksum does not match.")
        frames[name] = _validated_split_frame(path, name)
        if metadata.get("rows") != len(frames[name]):
            raise PreparedDataError(f"Prepared {name} split row count does not match.")
        actual_counts = {
            label: int(count)
            for label, count in frames[name]["label"].value_counts().sort_index().items()
        }
        if metadata.get("class_counts") != actual_counts:
            raise PreparedDataError(f"Prepared {name} split class counts do not match.")
    data_config = manifest.get("data_config")
    summary = manifest.get("data_summary")
    if (
        not isinstance(data_config, dict)
        or data_config.get("balance_classes") is not True
        or data_config.get("test_size") != 0.1
        or data_config.get("validation_size") != 0.1
    ):
        raise PreparedDataError("Prepared split manifest does not declare balanced 80/10/10 data.")
    if not isinstance(summary, dict) or summary.get("modeling_rows") != sum(
        len(frame) for frame in frames.values()
    ):
        raise PreparedDataError("Prepared split summary row count does not match.")
    if summary.get("class_counts") != {
        name: split_metadata[name]["class_counts"] for name in SPLIT_NAMES
    }:
        raise PreparedDataError("Prepared split summary class counts do not match.")
    _validate_balanced_split_contract(frames)
    _assert_no_overlap(frames)
    return manifest


def _run_contract_payload(config: TrainingConfig, split_manifest_path: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "base_checkpoint": config.base_checkpoint,
        "base_revision": config.base_revision,
        "split_manifest_sha256": _sha256(split_manifest_path),
        "training_config": asdict(config),
    }


def initialize_run_contract(
    split_dir: str | Path,
    work_dir: str | Path,
    *,
    config: TrainingConfig | None = None,
) -> dict[str, Any]:
    """Validate inputs and atomically initialize an immutable training contract."""
    splits = Path(split_dir)
    validate_prepared_splits(splits)
    resolved_config = config or TrainingConfig()
    payload = _run_contract_payload(resolved_config, splits / "manifest.json")
    return ensure_run_contract(
        Path(work_dir) / "run_contract.json",
        payload,
        require_existing=False,
    )


def adopt_prepared_splits(split_dir: str | Path) -> dict[str, Any]:
    """Upgrade a validated legacy leakage-safe split manifest with file hashes.

    This explicit migration is intentionally separate from training. It never
    accepts arbitrary CSVs merely because they have the expected filenames.
    The legacy manifest's provenance and summary must agree with the files.
    """
    directory = Path(split_dir)
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise PreparedDataError("Legacy prepared split manifest is missing.")
    try:
        legacy = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreparedDataError("Legacy prepared split manifest is invalid.") from exc
    if isinstance(legacy.get("splits"), dict):
        return validate_prepared_splits(directory)

    dataset = legacy.get("dataset")
    summary = legacy.get("data_summary")
    if not isinstance(dataset, dict) or not isinstance(summary, dict):
        raise PreparedDataError("Legacy manifest lacks dataset provenance or a data summary.")
    data_config = summary.get("data_config", legacy.get("data_config"))
    reported_counts = summary.get("class_counts")
    if not isinstance(data_config, dict) or not isinstance(reported_counts, dict):
        raise PreparedDataError("Legacy manifest lacks preparation configuration or class counts.")
    if (
        data_config.get("balance_classes") is not True
        or data_config.get("validation_size") != 0.1
        or data_config.get("test_size") != 0.1
    ):
        raise PreparedDataError("Legacy manifest does not describe balanced 80/10/10 splits.")

    frames: dict[str, pd.DataFrame] = {}
    split_metadata: dict[str, dict[str, Any]] = {}
    for name in SPLIT_NAMES:
        path = directory / f"{name}.csv"
        if not path.is_file():
            raise PreparedDataError(f"Prepared {name} split is missing.")
        frame = _validated_split_frame(path, name)
        actual_counts = {
            label: int(count) for label, count in frame["label"].value_counts().sort_index().items()
        }
        if reported_counts.get(name) != actual_counts:
            raise PreparedDataError(f"Prepared {name} class counts disagree with the manifest.")
        frames[name] = frame
        split_metadata[name] = {
            "filename": path.name,
            "rows": int(len(frame)),
            "sha256": _sha256(path),
            "class_counts": actual_counts,
        }

    _validate_balanced_split_contract(frames)
    _assert_no_overlap(frames)
    if summary.get("modeling_rows") != sum(len(frame) for frame in frames.values()):
        raise PreparedDataError("Prepared split rows disagree with the legacy modeling row count.")

    existing_notes = legacy.get("notes")
    notes = list(existing_notes) if isinstance(existing_notes, list) else []
    upgraded: dict[str, Any] = {
        **legacy,
        "schema_version": 1,
        "data_config": dict(data_config),
        "splits": split_metadata,
        "notes": [
            *notes,
            "Legacy split files were adopted only after schema, balance, and overlap validation.",
        ],
    }
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    _write_json(temporary_manifest, upgraded)
    temporary_manifest.replace(manifest_path)
    return upgraded


def _load_training_dependencies() -> SimpleNamespace:
    try:
        import datasets
        import torch
        import transformers
    except ImportError as exc:
        raise TrainingDependencyError(
            "DistilBERT training dependencies are unavailable. Install the "
            "distilbert-training optional dependency group."
        ) from exc
    return SimpleNamespace(
        AutoTokenizer=transformers.AutoTokenizer,
        AutoModelForSequenceClassification=transformers.AutoModelForSequenceClassification,
        DataCollatorWithPadding=transformers.DataCollatorWithPadding,
        Dataset=datasets.Dataset,
        DatasetDict=datasets.DatasetDict,
        Trainer=transformers.Trainer,
        TrainingArguments=transformers.TrainingArguments,
        torch=torch,
    )


def _dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in (
        "accelerate",
        "datasets",
        "numpy",
        "pandas",
        "safetensors",
        "torch",
        "transformers",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    return versions


def _classification_metrics(label_ids: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    labels = list(range(len(LABEL_TO_ID)))
    precision, recall, f1, support = precision_recall_fscore_support(
        label_ids,
        predictions,
        labels=labels,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(label_ids, predictions)),
        "precision_macro": float(np.mean(precision)),
        "recall_macro": float(np.mean(recall)),
        "f1_macro": float(np.mean(f1)),
        "per_class": {
            ID_TO_LABEL[index]: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index in labels
        },
        "confusion_matrix": confusion_matrix(label_ids, predictions, labels=labels).tolist(),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _dataset_dict(split_dir: Path, dependencies: Any) -> Any:
    datasets: dict[str, Any] = {}
    for name in SPLIT_NAMES:
        frame = pd.read_csv(split_dir / f"{name}.csv")
        encoded = frame.assign(labels=frame["label"].map(LABEL_TO_ID).astype(int))
        datasets[name] = dependencies.Dataset.from_pandas(encoded, preserve_index=False)
    return dependencies.DatasetDict(datasets)


def _artifact_files(artifact_dir: Path) -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(artifact_dir.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        relative = path.relative_to(artifact_dir).as_posix()
        files[relative] = {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
    return files


def _normalize_exported_json_line_endings(artifact_dir: Path) -> None:
    """Make exported JSON byte-identical on Windows, macOS, and Linux."""
    for path in artifact_dir.rglob("*.json"):
        if not path.is_file() or path.name == "manifest.json":
            continue
        contents = path.read_bytes()
        normalized = contents.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        if normalized != contents:
            path.write_bytes(normalized)


def finalize_artifact_manifest(artifact_dir: str | Path) -> dict[str, Any]:
    """Upgrade a completed artifact manifest with pinned provenance and identity."""
    artifact = Path(artifact_dir)
    manifest_path = artifact / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Completed artifact manifest is missing.")
    try:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Completed artifact manifest is invalid.") from exc
    if (
        existing.get("training_complete") is not True
        or existing.get("artifact_type") != "fine_tuned_sequence_classification"
        or existing.get("base_checkpoint") != BASE_CHECKPOINT
    ):
        raise ValueError("Artifact is not a completed ThreatLens DistilBERT export.")
    actual_files = _artifact_files(artifact)
    if existing.get("files") != actual_files:
        raise ValueError("Artifact files do not match the completed manifest checksums.")
    upgraded = {
        **existing,
        "base_revision": BASE_REVISION,
        "artifact_fingerprint": artifact_fingerprint(actual_files),
    }
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    _write_json(temporary_manifest, upgraded)
    temporary_manifest.replace(manifest_path)
    return upgraded


def train_and_export(
    split_dir: str | Path,
    output_dir: str | Path,
    *,
    work_dir: str | Path,
    config: TrainingConfig | None = None,
    resume: bool = False,
    dependencies: Any | None = None,
) -> dict[str, Any]:
    """Fine-tune DistilBERT and export a manifest-backed local HF artifact."""
    resolved_config = config or TrainingConfig()
    splits = Path(split_dir)
    artifact = Path(output_dir)
    work = Path(work_dir)
    split_manifest = validate_prepared_splits(splits)
    contract_payload = _run_contract_payload(resolved_config, splits / "manifest.json")
    contract_path = work / "run_contract.json"
    if resume:
        run_contract = ensure_run_contract(
            contract_path,
            contract_payload,
            require_existing=True,
        )
        checkpoint = find_resume_checkpoint(work)
        if checkpoint is None:
            raise RunContractError("Resume requested, but no complete checkpoint is available.")
    else:
        existing_checkpoints = tuple(work.glob("checkpoint-*")) if work.is_dir() else ()
        if existing_checkpoints:
            raise RunContractError(
                "Training work directory contains checkpoints; use --resume after validation."
            )
        run_contract = ensure_run_contract(
            contract_path,
            contract_payload,
            require_existing=False,
        )
        checkpoint = None
    deps = dependencies or _load_training_dependencies()
    cuda_available = bool(deps.torch.cuda.is_available())

    dataset = _dataset_dict(splits, deps)
    tokenizer = deps.AutoTokenizer.from_pretrained(
        resolved_config.base_checkpoint,
        use_fast=True,
        revision=resolved_config.base_revision,
    )

    def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=resolved_config.max_length,
        )

    tokenized = dataset.map(tokenize, batched=True)
    tokenized = tokenized.remove_columns(["text", "label"])
    model = deps.AutoModelForSequenceClassification.from_pretrained(
        resolved_config.base_checkpoint,
        num_labels=3,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        revision=resolved_config.base_revision,
    )

    def compute_metrics(prediction: Any) -> dict[str, float]:
        predicted = np.asarray(prediction.predictions).argmax(axis=1)
        metrics = _classification_metrics(np.asarray(prediction.label_ids), predicted)
        return {
            "accuracy": metrics["accuracy"],
            "precision_macro": metrics["precision_macro"],
            "recall_macro": metrics["recall_macro"],
            "f1_macro": metrics["f1_macro"],
        }

    work.mkdir(parents=True, exist_ok=True)
    arguments = deps.TrainingArguments(
        output_dir=str(work),
        learning_rate=resolved_config.learning_rate,
        per_device_train_batch_size=resolved_config.batch_size,
        per_device_eval_batch_size=resolved_config.batch_size,
        num_train_epochs=resolved_config.epochs,
        weight_decay=resolved_config.weight_decay,
        eval_strategy=resolved_config.eval_strategy,
        save_strategy=resolved_config.save_strategy,
        load_best_model_at_end=resolved_config.load_best_model_at_end,
        save_total_limit=resolved_config.save_total_limit,
        fp16=cuda_available,
        seed=resolved_config.seed,
        data_seed=resolved_config.seed,
        report_to=[],
        dataloader_pin_memory=cuda_available,
        metric_for_best_model="eval_f1_macro",
        greater_is_better=True,
    )
    trainer = deps.Trainer(
        model=model,
        args=arguments,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=deps.DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    train_result = trainer.train(
        resume_from_checkpoint=str(checkpoint) if checkpoint is not None else None
    )
    test_result = trainer.predict(tokenized["test"])
    test_predictions = np.asarray(test_result.predictions).argmax(axis=1)
    test_metrics = _classification_metrics(
        np.asarray(test_result.label_ids),
        test_predictions,
    )

    artifact.mkdir(parents=True, exist_ok=True)
    trainer.save_model(artifact)
    tokenizer.save_pretrained(artifact)
    training_values = {
        **asdict(resolved_config),
        "fp16": cuda_available,
        "resume_from_checkpoint": checkpoint.name if checkpoint else None,
    }
    metrics: dict[str, Any] = {
        "test": {
            **test_metrics,
            "trainer_metrics": _json_safe(getattr(test_result, "metrics", {})),
        },
        "training": _json_safe(getattr(train_result, "metrics", {})),
        "training_config": training_values,
    }
    _write_json(artifact / "metrics.json", metrics)

    # Hugging Face writes some JSON files itself. Normalize all of them before
    # recording sizes and hashes so a Git checkout with `eol=lf` stays valid.
    _normalize_exported_json_line_endings(artifact)

    files = _artifact_files(artifact)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "fine_tuned_sequence_classification",
        "training_complete": True,
        "created_at": datetime.now(UTC).isoformat(),
        "base_checkpoint": resolved_config.base_checkpoint,
        "base_revision": resolved_config.base_revision,
        "max_length": resolved_config.max_length,
        "num_labels": 3,
        "id2label": {str(key): value for key, value in ID_TO_LABEL.items()},
        "label2id": LABEL_TO_ID,
        "training_config": training_values,
        "run_contract": run_contract,
        "dependency_versions": _dependency_versions(),
        "dataset": split_manifest.get("dataset", {}),
        "split_manifest_sha256": _sha256(splits / "manifest.json"),
        "splits": split_manifest.get("splits", {}),
        "files": files,
        "artifact_fingerprint": artifact_fingerprint(files),
    }
    _write_json(artifact / "manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    """Run the DistilBERT preparation or training command."""
    from email_threat_detector.distilbert_cli import main as cli_main

    return cli_main(list(argv) if argv is not None else None, api=sys.modules[__name__])


if __name__ == "__main__":
    raise SystemExit(main())
