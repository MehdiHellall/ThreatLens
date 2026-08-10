"""Validated, lazy runtime for a genuinely fine-tuned DistilBERT artifact."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import stat
import tempfile
import threading
import urllib.parse
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from ipaddress import ip_address
from pathlib import Path, PurePosixPath
from typing import Any, cast

from web.backend.schemas import ThreatLabel
from web.backend.settings import (
    DEFAULT_MAX_ARTIFACT_BYTES,
    DISTILBERT_MODEL_PATH_ENV,
    DISTILBERT_MODEL_SHA256_ENV,
    DISTILBERT_MODEL_URL_ENV,
    AppSettings,
)
from web.backend.sklearn_runtime import (
    ModelArtifactError,
    _download_model_artifact,
    file_sha256,
    resolve_local_path,
)

_UNLOADED = object()
_dependency_lock = threading.Lock()
# Public module attributes intentionally support lightweight fakes in unit tests.
torch: Any = _UNLOADED
AutoModelForSequenceClassification: Any = _UNLOADED
AutoTokenizer: Any = _UNLOADED

BASE_CHECKPOINT = "distilbert/distilbert-base-uncased"
BASE_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"
ID2LABEL = {0: "ham", 1: "phish", 2: "spam"}
LABEL2ID = {"ham": 0, "phish": 1, "spam": 2}
PUBLIC_MANIFEST_KEYS = frozenset(
    {
        "training_complete",
        "base_checkpoint",
        "base_revision",
        "max_length",
        "created_at",
        "split_manifest_sha256",
        "artifact_fingerprint",
    }
)
REQUIRED_MANIFEST_KEYS = frozenset(
    {
        "artifact_type",
        "training_complete",
        "base_checkpoint",
        "base_revision",
        "max_length",
        "id2label",
        "label2id",
        "training_config",
        "dependency_versions",
        "dataset",
        "split_manifest_sha256",
        "files",
        "artifact_fingerprint",
        "created_at",
    }
)
REQUIRED_ARTIFACT_TYPE = "fine_tuned_sequence_classification"
MAX_MANIFEST_BYTES = 2 * 1024 * 1024


class DistilBertArtifactError(RuntimeError):
    """Raised when an artifact is missing, unsafe, or not a completed fine-tune."""


@dataclass(frozen=True)
class DistilBertPrediction:
    label: ThreatLabel
    probabilities: dict[str, float]


class DistilBertRuntime:
    """Small inference-only adapter that never retains submitted message text."""

    def __init__(self, tokenizer: Any, model: Any, *, max_length: int) -> None:
        self._tokenizer = tokenizer
        self._model = model
        self._max_length = max_length
        self._inference_lock = threading.Lock()

    @classmethod
    def from_directory(cls, path: Path, *, max_length: int) -> DistilBertRuntime:
        validate_artifact_directory(path, expected_max_length=max_length)
        _load_runtime_dependencies()
        tokenizer = AutoTokenizer.from_pretrained(
            path,
            use_fast=True,
            local_files_only=True,
        )
        if getattr(tokenizer, "is_fast", True) is not True:
            raise DistilBertArtifactError("The artifact did not load with a fast tokenizer.")
        model = AutoModelForSequenceClassification.from_pretrained(
            path,
            local_files_only=True,
            use_safetensors=True,
        )
        model.eval()
        return cls(tokenizer, model, max_length=max_length)

    def predict_one(self, text: str) -> DistilBertPrediction:
        """Classify one message using exact notebook tokenization and real logits."""
        with self._inference_lock:
            encoded = self._tokenizer(
                text,
                truncation=True,
                max_length=self._max_length,
                return_tensors="pt",
            )
            with torch.no_grad():
                logits = self._model(**encoded).logits
                values = torch.softmax(logits, dim=-1)[0].tolist()

        probabilities = _validated_probabilities(values)
        winning_index = max(range(len(ID2LABEL)), key=lambda index: probabilities[ID2LABEL[index]])
        return DistilBertPrediction(
            label=cast(ThreatLabel, ID2LABEL[winning_index]),
            probabilities=probabilities,
        )


def _load_runtime_dependencies() -> None:
    """Import the optional ML stack only when a validated artifact is loaded."""
    global AutoModelForSequenceClassification, AutoTokenizer, torch
    dependencies = (torch, AutoModelForSequenceClassification, AutoTokenizer)
    if any(dependency is None for dependency in dependencies):
        raise DistilBertArtifactError(
            "A DistilBERT runtime dependency is not installed; install the runtime extra."
        )
    if all(dependency is not _UNLOADED for dependency in dependencies):
        return
    with _dependency_lock:
        dependencies = (torch, AutoModelForSequenceClassification, AutoTokenizer)
        if all(dependency is not _UNLOADED for dependency in dependencies):
            return
        try:
            import torch as imported_torch
            from transformers import (
                AutoModelForSequenceClassification as ImportedAutoModel,
            )
            from transformers import AutoTokenizer as ImportedAutoTokenizer
        except Exception as exc:  # pragma: no cover - depends on optional installation.
            torch = None
            AutoModelForSequenceClassification = None
            AutoTokenizer = None
            raise DistilBertArtifactError(
                "A DistilBERT runtime dependency is not installed; install the runtime extra."
            ) from exc
        torch = imported_torch
        AutoModelForSequenceClassification = ImportedAutoModel
        AutoTokenizer = ImportedAutoTokenizer


@dataclass(frozen=True)
class DistilBertState:
    runtime: DistilBertRuntime | Any | None
    path: Path | None
    manifest: dict[str, object]
    error: str | None = None

    @property
    def loaded(self) -> bool:
        return self.runtime is not None and self.error is None


ArtifactDownloader = Callable[[str, Path], Path]


def _validated_probabilities(values: object) -> dict[str, float]:
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise RuntimeError("DistilBERT returned an invalid probability vector.")
    scores = [float(value) for value in values]
    if any(not math.isfinite(value) or value < 0.0 for value in scores):
        raise RuntimeError("DistilBERT returned an invalid probability vector.")
    if not math.isclose(sum(scores), 1.0, rel_tol=1e-5, abs_tol=1e-6):
        raise RuntimeError("DistilBERT returned an invalid probability vector.")
    return {ID2LABEL[index]: score for index, score in enumerate(scores)}


def _read_json_object(path: Path, *, description: str) -> dict[str, object]:
    try:
        if path.stat().st_size > MAX_MANIFEST_BYTES:
            raise DistilBertArtifactError(f"DistilBERT {description} is unexpectedly large.")
        value = json.loads(path.read_text(encoding="utf-8"))
    except DistilBertArtifactError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DistilBertArtifactError(f"DistilBERT {description} is invalid.") from exc
    if not isinstance(value, dict):
        raise DistilBertArtifactError(f"DistilBERT {description} must be a JSON object.")
    return cast(dict[str, object], value)


def _normalized_id2label(value: object) -> dict[int, str] | None:
    if not isinstance(value, dict):
        return None
    try:
        return {int(key): str(label).casefold() for key, label in value.items()}
    except (TypeError, ValueError):
        return None


def _normalized_label2id(value: object) -> dict[str, int] | None:
    if not isinstance(value, dict):
        return None
    try:
        return {str(label).casefold(): int(index) for label, index in value.items()}
    except (TypeError, ValueError):
        return None


def _safe_declared_file(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if (
        relative.is_absolute()
        or not relative.parts
        or relative.as_posix() != name
        or ".." in relative.parts
        or ":" in relative.parts[0]
        or "\\" in name
    ):
        raise DistilBertArtifactError("Artifact manifest contains an unsafe file path.")
    candidate = root.joinpath(*relative.parts)
    try:
        candidate.resolve().relative_to(root.resolve())
    except (OSError, ValueError) as exc:
        raise DistilBertArtifactError("Artifact manifest contains an unsafe file path.") from exc
    if candidate.is_symlink() or not candidate.is_file():
        raise DistilBertArtifactError(f"Artifact file is missing: {relative.name}")
    return candidate


def _is_nonempty_mapping(value: object) -> bool:
    return isinstance(value, dict) and bool(value)


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None


def _is_valid_created_at(value: object) -> bool:
    if not isinstance(value, str) or len(value) > 64:
        return False
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _artifact_fingerprint(files: dict[object, object]) -> str:
    canonical = json.dumps(
        files,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def validate_artifact_directory(
    path: Path,
    *,
    expected_max_length: int,
) -> dict[str, object]:
    """Validate training provenance, model identity, labels, and every declared hash."""
    artifact_path = resolve_local_path(path)
    manifest_path = artifact_path / "manifest.json"
    if not manifest_path.is_file():
        raise DistilBertArtifactError(
            "A completed fine-tuning manifest is required; raw base checkpoints are rejected."
        )
    manifest = _read_json_object(manifest_path, description="manifest")
    if manifest.get("schema_version") != 1:
        raise DistilBertArtifactError("The artifact manifest schema version is unsupported.")
    if REQUIRED_MANIFEST_KEYS.difference(manifest):
        raise DistilBertArtifactError("The artifact manifest is missing required provenance.")
    if manifest.get("artifact_type") != REQUIRED_ARTIFACT_TYPE:
        raise DistilBertArtifactError("The artifact manifest artifact type is not fine-tuned.")
    if manifest.get("training_complete") is not True:
        raise DistilBertArtifactError("The artifact does not record completed fine-tuned training.")
    if manifest.get("base_checkpoint") != BASE_CHECKPOINT:
        raise DistilBertArtifactError(
            "The artifact base checkpoint is not the supported DistilBERT model."
        )
    if manifest.get("base_revision") != BASE_REVISION:
        raise DistilBertArtifactError("Artifact base revision is not the pinned model revision.")
    if not _is_valid_created_at(manifest.get("created_at")):
        raise DistilBertArtifactError("Artifact creation time is invalid.")
    if manifest.get("max_length") != expected_max_length:
        raise DistilBertArtifactError("Configured max_length does not match the artifact manifest.")
    if _normalized_id2label(manifest.get("id2label")) != ID2LABEL:
        raise DistilBertArtifactError("Artifact id-to-label mapping is invalid.")
    if _normalized_label2id(manifest.get("label2id")) != LABEL2ID:
        raise DistilBertArtifactError("Artifact label-to-id mapping is invalid.")
    if not _is_nonempty_mapping(manifest.get("training_config")):
        raise DistilBertArtifactError("Artifact training configuration is invalid.")
    split_manifest_sha256 = manifest.get("split_manifest_sha256")
    if not _is_sha256(split_manifest_sha256):
        raise DistilBertArtifactError("Artifact split manifest checksum is invalid.")
    if not _is_nonempty_mapping(manifest.get("dependency_versions")):
        raise DistilBertArtifactError("Artifact dependency versions are invalid.")
    if not _is_nonempty_mapping(manifest.get("dataset")):
        raise DistilBertArtifactError("Artifact dataset provenance is invalid.")

    declared_files = manifest.get("files")
    if not isinstance(declared_files, dict) or not declared_files:
        raise DistilBertArtifactError("Artifact manifest does not declare file checksums.")
    declared_names = {str(name) for name in declared_files}
    if "manifest.json" in declared_names:
        raise DistilBertArtifactError("Artifact manifest.json must not declare itself.")
    if "config.json" not in declared_names:
        raise DistilBertArtifactError("Artifact manifest does not declare config.json.")
    if "model.safetensors" not in declared_names:
        raise DistilBertArtifactError("Artifact manifest must declare safetensors model weights.")
    if "pytorch_model.bin" in declared_names or any(
        name.startswith("pytorch_model-") and name.endswith(".bin") for name in declared_names
    ):
        raise DistilBertArtifactError("Pickle-based PyTorch model weights are not accepted.")
    if not declared_names.intersection({"tokenizer.json", "vocab.txt"}):
        raise DistilBertArtifactError("Artifact manifest does not declare tokenizer files.")
    if "metrics.json" not in declared_names:
        raise DistilBertArtifactError("Artifact manifest does not declare metrics.json.")
    expected_fingerprint = manifest.get("artifact_fingerprint")
    if not _is_sha256(expected_fingerprint):
        raise DistilBertArtifactError("Artifact fingerprint is invalid.")
    if _artifact_fingerprint(declared_files).casefold() != expected_fingerprint.casefold():
        raise DistilBertArtifactError("Artifact fingerprint does not match its file inventory.")

    for raw_name, raw_entry in declared_files.items():
        name = str(raw_name)
        expected_size: int | None = None
        if isinstance(raw_entry, dict):
            digest = str(raw_entry.get("sha256", ""))
            raw_size = raw_entry.get("size_bytes")
            if not isinstance(raw_size, int) or isinstance(raw_size, bool) or raw_size < 0:
                raise DistilBertArtifactError("Artifact manifest contains an invalid file size.")
            expected_size = raw_size
        else:
            digest = str(raw_entry)
        if not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise DistilBertArtifactError("Artifact manifest contains an invalid checksum.")
        candidate = _safe_declared_file(artifact_path, name)
        if expected_size is not None and candidate.stat().st_size != expected_size:
            raise DistilBertArtifactError(f"Artifact file size mismatch: {candidate.name}")
        if file_sha256(candidate).casefold() != digest.casefold():
            raise DistilBertArtifactError(f"Artifact file checksum mismatch: {candidate.name}")

    artifact_entries = tuple(artifact_path.rglob("*"))
    if any(candidate.is_symlink() for candidate in artifact_entries):
        raise DistilBertArtifactError("Artifact directory contains a symbolic link.")
    undeclared_files = {
        candidate.relative_to(artifact_path).as_posix()
        for candidate in artifact_entries
        if candidate.is_file()
        and candidate.relative_to(artifact_path).as_posix() != "manifest.json"
    }.difference(declared_names)
    if undeclared_files:
        raise DistilBertArtifactError("Artifact directory contains files that are not declared.")

    config = _read_json_object(artifact_path / "config.json", description="model config")
    _read_json_object(artifact_path / "metrics.json", description="metrics")
    config_id2label = _normalized_id2label(config.get("id2label"))
    configured_num_labels = config.get("num_labels")
    if config.get("model_type") != "distilbert" or configured_num_labels not in {None, 3}:
        raise DistilBertArtifactError("Artifact is not a three-label DistilBERT classifier.")
    if config_id2label != ID2LABEL:
        raise DistilBertArtifactError("Model config id-to-label mapping is invalid.")
    if _normalized_label2id(config.get("label2id")) != LABEL2ID:
        raise DistilBertArtifactError("Model config label-to-id mapping is invalid.")
    return manifest


def _zip_member_is_symlink(info: zipfile.ZipInfo) -> bool:
    return stat.S_ISLNK(info.external_attr >> 16)


def safe_extract_zip(
    archive_path: Path,
    destination: Path,
    *,
    max_artifact_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES,
) -> None:
    """Extract a ZIP only after rejecting traversal, absolute paths, and symlinks."""
    if destination.exists() or destination.is_symlink():
        raise DistilBertArtifactError("Artifact extraction destination already exists.")
    destination_root = destination.resolve()
    created_destination = False
    try:
        with zipfile.ZipFile(archive_path) as archive:
            infos = archive.infolist()
            if len(infos) > 10_000:
                raise DistilBertArtifactError("Artifact ZIP contains too many files.")
            if sum(info.file_size for info in infos) > max_artifact_bytes:
                raise DistilBertArtifactError("Artifact ZIP expands beyond the safe size limit.")
            for info in infos:
                relative = PurePosixPath(info.filename.replace("\\", "/"))
                if (
                    relative.is_absolute()
                    or not relative.parts
                    or ".." in relative.parts
                    or ":" in relative.parts[0]
                    or _zip_member_is_symlink(info)
                ):
                    raise DistilBertArtifactError(
                        "Artifact ZIP contains an unsafe path traversal or link."
                    )
                target = destination_root.joinpath(*relative.parts).resolve()
                try:
                    target.relative_to(destination_root)
                except ValueError as exc:
                    raise DistilBertArtifactError(
                        "Artifact ZIP contains an unsafe path traversal."
                    ) from exc
            destination.mkdir(parents=True, exist_ok=False)
            created_destination = True
            archive.extractall(destination)
    except DistilBertArtifactError:
        if created_destination:
            shutil.rmtree(destination, ignore_errors=True)
        raise
    except (OSError, zipfile.BadZipFile) as exc:
        if created_destination:
            shutil.rmtree(destination, ignore_errors=True)
        raise DistilBertArtifactError("DistilBERT artifact ZIP is invalid.") from exc


def _is_loopback_host(host: str | None) -> bool:
    if host is None:
        return False
    if host.casefold() == "localhost":
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


def _validate_remote_settings(url: str, expected_sha256: str | None) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.hostname is None:
        raise DistilBertArtifactError("Remote DistilBERT artifact URL has no host.")
    if parsed.username is not None or parsed.password is not None:
        raise DistilBertArtifactError(
            "Remote DistilBERT artifact URL must not contain credentials."
        )
    if parsed.fragment:
        raise DistilBertArtifactError("Remote DistilBERT artifact URL must not contain a fragment.")
    loopback_http = parsed.scheme == "http" and _is_loopback_host(parsed.hostname)
    if parsed.scheme != "https" and not loopback_http:
        raise DistilBertArtifactError(
            "Remote DistilBERT artifact URL must use https, except loopback test URLs."
        )
    try:
        address = ip_address(parsed.hostname)
    except ValueError:
        address = None
    if address is not None and not address.is_global and not address.is_loopback:
        raise DistilBertArtifactError(
            "Remote DistilBERT artifact IP must be on the public network."
        )
    if expected_sha256 is None:
        raise DistilBertArtifactError(
            f"{DISTILBERT_MODEL_SHA256_ENV} is required for remote artifacts."
        )
    if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
        raise DistilBertArtifactError(
            f"{DISTILBERT_MODEL_SHA256_ENV} must be a 64-character hex SHA-256 digest."
        )


def _download_distilbert_archive(
    url: str,
    destination: Path,
    max_artifact_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES,
) -> Path:
    try:
        return _download_model_artifact(url, destination, max_artifact_bytes)
    except ModelArtifactError as exc:
        raise DistilBertArtifactError(str(exc)) from exc


def _artifact_root(extracted: Path) -> Path:
    if (extracted / "manifest.json").is_file():
        return extracted
    children = [child for child in extracted.iterdir() if child.is_dir()]
    if len(children) == 1 and (children[0] / "manifest.json").is_file():
        return children[0]
    raise DistilBertArtifactError("Extracted ZIP does not contain one DistilBERT artifact.")


def resolve_distilbert_artifact(
    settings: AppSettings,
    downloader: ArtifactDownloader | None = None,
) -> tuple[Path | None, str | None]:
    """Resolve a local directory or checksum-pinned remote ZIP into a cache directory."""
    configured_path = settings.distilbert_model_path
    configured_url = settings.distilbert_model_url
    if configured_path is not None and configured_url is not None:
        return (
            resolve_local_path(configured_path),
            f"Configure only one of {DISTILBERT_MODEL_PATH_ENV} or {DISTILBERT_MODEL_URL_ENV}.",
        )
    if configured_path is not None:
        path = resolve_local_path(configured_path)
        if not path.is_dir():
            return path, f"Configured DistilBERT artifact does not exist: {path.name}"
        return path, None
    if configured_url is None:
        return None, "Fine-tuned DistilBERT artifact is not configured."

    cache_root = resolve_local_path(settings.distilbert_model_cache_path)
    try:
        _validate_remote_settings(configured_url, settings.distilbert_model_sha256)
    except DistilBertArtifactError as exc:
        return cache_root, str(exc)

    digest = cast(str, settings.distilbert_model_sha256).casefold()
    cache_path = cache_root / f"sha256-{digest}"
    if cache_root.exists() and not cache_root.is_dir():
        return cache_path, "Configured DistilBERT cache root is not a directory."
    try:
        cache_root.mkdir(parents=True, exist_ok=True)
    except OSError:
        return cache_path, "Could not prepare the DistilBERT cache root."

    if cache_path.exists() or cache_path.is_symlink():
        if cache_path.is_symlink() or not cache_path.is_dir():
            return cache_path, "Existing checksum cache is invalid and was preserved."
        try:
            validate_artifact_directory(
                cache_path,
                expected_max_length=settings.distilbert_max_length,
            )
            return cache_path, None
        except DistilBertArtifactError:
            return cache_path, "Existing checksum cache is invalid and was preserved."

    artifact_downloader = downloader or (
        lambda url, destination: _download_distilbert_archive(
            url,
            destination,
            settings.max_artifact_bytes,
        )
    )
    staging_path: Path | None = None
    try:
        staging_path = Path(tempfile.mkdtemp(prefix=f".sha256-{digest[:16]}-", dir=cache_root))
        archive_path = staging_path / "artifact.zip"
        downloaded = artifact_downloader(configured_url, archive_path)
        if file_sha256(downloaded).casefold() != digest:
            raise DistilBertArtifactError("Downloaded DistilBERT artifact checksum mismatch.")
        extraction_path = staging_path / "extracted"
        safe_extract_zip(
            downloaded,
            extraction_path,
            max_artifact_bytes=settings.max_artifact_bytes,
        )
        artifact_root = _artifact_root(extraction_path)
        validate_artifact_directory(
            artifact_root,
            expected_max_length=settings.distilbert_max_length,
        )
        if cache_path.exists() or cache_path.is_symlink():
            try:
                validate_artifact_directory(
                    cache_path,
                    expected_max_length=settings.distilbert_max_length,
                )
            except DistilBertArtifactError:
                return cache_path, "Existing checksum cache is invalid and was preserved."
            return cache_path, None
        if artifact_root == extraction_path:
            extraction_path.replace(cache_path)
        else:
            artifact_root.replace(cache_path)
    except (DistilBertArtifactError, ModelArtifactError, OSError) as exc:
        detail = (
            str(exc)
            if isinstance(exc, (DistilBertArtifactError, ModelArtifactError))
            else "Could not cache artifact."
        )
        return cache_path, detail
    finally:
        if staging_path is not None:
            shutil.rmtree(staging_path, ignore_errors=True)
    return cache_path, None


def load_distilbert_state(
    settings: AppSettings,
    downloader: ArtifactDownloader | None = None,
) -> DistilBertState:
    path, artifact_error = resolve_distilbert_artifact(settings, downloader)
    if artifact_error is not None:
        return DistilBertState(runtime=None, path=path, manifest={}, error=artifact_error)
    if path is None:
        return DistilBertState(
            runtime=None,
            path=None,
            manifest={},
            error="Fine-tuned DistilBERT artifact is not configured.",
        )
    try:
        manifest = validate_artifact_directory(
            path,
            expected_max_length=settings.distilbert_max_length,
        )
        classifier = DistilBertRuntime.from_directory(
            path,
            max_length=settings.distilbert_max_length,
        )
    except DistilBertArtifactError as exc:
        return DistilBertState(runtime=None, path=path, manifest={}, error=str(exc))
    except Exception as exc:  # Loading errors are summarized without leaking local details.
        return DistilBertState(
            runtime=None,
            path=path,
            manifest={},
            error=f"Could not load fine-tuned DistilBERT artifact: {type(exc).__name__}.",
        )
    return DistilBertState(runtime=classifier, path=path, manifest=manifest)


class DistilBertService:
    """Thread-safe one-time lazy loader for the optional transformer artifact."""

    def __init__(
        self,
        settings: AppSettings,
        downloader: ArtifactDownloader | None = None,
    ) -> None:
        self._settings = settings
        self._downloader = downloader
        self._lock = threading.Lock()
        self._state: DistilBertState | None = None

    def get_state(self) -> DistilBertState:
        if self._state is not None:
            return self._state
        with self._lock:
            if self._state is None:
                self._state = load_distilbert_state(self._settings, self._downloader)
            return self._state


def public_manifest(manifest: dict[str, object]) -> dict[str, object]:
    """Expose provenance fields only; never local paths or file inventories."""
    safe_values = {
        "training_complete": manifest.get("training_complete") is True,
        "base_checkpoint": manifest.get("base_checkpoint") == BASE_CHECKPOINT,
        "base_revision": manifest.get("base_revision") == BASE_REVISION,
        "max_length": isinstance(manifest.get("max_length"), int)
        and not isinstance(manifest.get("max_length"), bool)
        and cast(int, manifest["max_length"]) > 0,
        "created_at": _is_valid_created_at(manifest.get("created_at")),
        "split_manifest_sha256": _is_sha256(manifest.get("split_manifest_sha256")),
        "artifact_fingerprint": _is_sha256(manifest.get("artifact_fingerprint")),
    }
    return {
        key: manifest[key] for key in PUBLIC_MANIFEST_KEYS if key in manifest and safe_values[key]
    }


__all__ = [
    "DistilBertArtifactError",
    "DistilBertPrediction",
    "DistilBertRuntime",
    "DistilBertService",
    "DistilBertState",
    "load_distilbert_state",
    "public_manifest",
    "resolve_distilbert_artifact",
    "safe_extract_zip",
    "validate_artifact_directory",
]
