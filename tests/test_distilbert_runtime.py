from __future__ import annotations

import hashlib
import json
import math
import shutil
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from web.backend.runtimes import distilbert_runtime as runtime
from web.backend.settings import AppSettings

ID2LABEL = {"0": "ham", "1": "phish", "2": "spam"}
LABEL2ID = {"ham": 0, "phish": 1, "spam": 2}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_fingerprint(files: object) -> str:
    encoded = json.dumps(
        files,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_exported_artifact(path: Path) -> dict[str, object]:
    path.mkdir(parents=True)
    files = {
        "config.json": json.dumps(
            {
                "architectures": ["DistilBertForSequenceClassification"],
                "model_type": "distilbert",
                "num_labels": 3,
                "id2label": ID2LABEL,
                "label2id": LABEL2ID,
            }
        ),
        "model.safetensors": "fine-tuned-weights-for-tests",
        "tokenizer.json": '{"version":"1.0"}',
        "tokenizer_config.json": '{"model_max_length":128}',
        "metrics.json": json.dumps(
            {
                "test": {"accuracy": 1.0},
                "training_config": {"epochs": 3, "batch_size": 16},
            }
        ),
    }
    for name, contents in files.items():
        (path / name).write_text(contents, encoding="utf-8")

    declared_files = {name: _sha256(path / name) for name in files}
    manifest: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "fine_tuned_sequence_classification",
        "training_complete": True,
        "created_at": "2026-08-09T10:00:00+00:00",
        "base_checkpoint": "distilbert/distilbert-base-uncased",
        "base_revision": runtime.BASE_REVISION,
        "max_length": 128,
        "id2label": ID2LABEL,
        "label2id": LABEL2ID,
        "training_config": {"epochs": 3, "batch_size": 16, "max_length": 128},
        "split_manifest_sha256": "a" * 64,
        "dependency_versions": {"torch": "2.13.0", "transformers": "5.14.1"},
        "dataset": {"source": "tests/fixture.csv", "rows": 30},
        "files": declared_files,
        "artifact_fingerprint": _artifact_fingerprint(declared_files),
    }
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def _zip_directory(source: Path, destination: Path) -> Path:
    with zipfile.ZipFile(destination, "w") as archive:
        for file_path in source.iterdir():
            archive.write(file_path, file_path.name)
    return destination


def _read_manifest(path: Path) -> dict[str, object]:
    return json.loads((path / "manifest.json").read_text(encoding="utf-8"))


def _write_manifest(path: Path, manifest: dict[str, object]) -> None:
    if isinstance(manifest.get("files"), dict):
        manifest["artifact_fingerprint"] = _artifact_fingerprint(manifest["files"])
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_missing_distilbert_artifact_is_honestly_unavailable(tmp_path: Path) -> None:
    missing = tmp_path / "missing-distilbert"

    state = runtime.load_distilbert_state(
        AppSettings(distilbert_model_path=missing),
    )

    assert state.loaded is False
    assert state.runtime is None
    assert state.path == missing
    assert state.error is not None
    assert "does not exist" in state.error


def test_raw_base_checkpoint_without_training_manifest_is_rejected(tmp_path: Path) -> None:
    raw_checkpoint = tmp_path / "raw-checkpoint"
    raw_checkpoint.mkdir()
    (raw_checkpoint / "config.json").write_text(
        json.dumps({"model_type": "distilbert", "num_labels": 3}),
        encoding="utf-8",
    )

    state = runtime.load_distilbert_state(
        AppSettings(distilbert_model_path=raw_checkpoint),
    )

    assert state.loaded is False
    assert state.runtime is None
    assert state.error is not None
    assert "manifest" in state.error.casefold()


def test_artifact_manifest_and_all_declared_checksums_are_validated(tmp_path: Path) -> None:
    artifact = tmp_path / "distilbert"
    expected_manifest = _write_exported_artifact(artifact)

    manifest = runtime.validate_artifact_directory(artifact, expected_max_length=128)

    assert manifest == expected_manifest

    (artifact / "tokenizer.json").write_text("tampered", encoding="utf-8")
    with pytest.raises(runtime.DistilBertArtifactError, match="checksum"):
        runtime.validate_artifact_directory(artifact, expected_max_length=128)


def test_pickle_model_weights_are_rejected_in_favor_of_safetensors(tmp_path: Path) -> None:
    artifact = tmp_path / "distilbert"
    manifest = _write_exported_artifact(artifact)
    (artifact / "model.safetensors").unlink()
    pickle_weights = artifact / "pytorch_model.bin"
    pickle_weights.write_bytes(b"untrusted-pickle-weights")
    declared_files = dict(manifest["files"])
    declared_files.pop("model.safetensors")
    declared_files["pytorch_model.bin"] = _sha256(pickle_weights)
    manifest["files"] = declared_files
    manifest["artifact_fingerprint"] = _artifact_fingerprint(declared_files)
    _write_manifest(artifact, manifest)

    with pytest.raises(runtime.DistilBertArtifactError, match="safetensors"):
        runtime.validate_artifact_directory(artifact, expected_max_length=128)


def test_training_export_checksum_objects_and_config_without_num_labels_are_accepted(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "distilbert"
    manifest = _write_exported_artifact(artifact)
    config = json.loads((artifact / "config.json").read_text(encoding="utf-8"))
    del config["num_labels"]
    (artifact / "config.json").write_text(json.dumps(config), encoding="utf-8")
    declared_names = tuple(manifest["files"])
    manifest["files"] = {
        name: {
            "sha256": _sha256(artifact / name),
            "size_bytes": (artifact / name).stat().st_size,
        }
        for name in declared_names
    }
    _write_manifest(artifact, manifest)

    assert runtime.validate_artifact_directory(artifact, expected_max_length=128) == manifest


@pytest.mark.parametrize(
    "values",
    [
        [0.5, 0.5],
        [float("nan"), 0.5, 0.5],
        [-0.1, 0.5, 0.6],
        [0.5, 0.4, 0.3],
    ],
)
def test_runtime_rejects_malformed_probability_vectors(values: list[float]) -> None:
    with pytest.raises(RuntimeError, match="invalid probability vector"):
        runtime._validated_probabilities(values)


@pytest.mark.parametrize(
    ("contents", "error_match"),
    [
        ("{", "invalid"),
        ("[]", "JSON object"),
    ],
)
def test_invalid_manifest_json_is_rejected(
    tmp_path: Path,
    contents: str,
    error_match: str,
) -> None:
    artifact = tmp_path / "distilbert"
    artifact.mkdir()
    (artifact / "manifest.json").write_text(contents, encoding="utf-8")

    with pytest.raises(runtime.DistilBertArtifactError, match=error_match):
        runtime.validate_artifact_directory(artifact, expected_max_length=128)


def test_oversized_manifest_is_rejected_before_json_parsing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "distilbert"
    artifact.mkdir()
    (artifact / "manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(runtime, "MAX_MANIFEST_BYTES", 1)

    with pytest.raises(runtime.DistilBertArtifactError, match="unexpectedly large"):
        runtime.validate_artifact_directory(artifact, expected_max_length=128)


@pytest.mark.parametrize(
    ("update", "error_match"),
    [
        ({"schema_version": 9}, "schema version"),
        ({"artifact_type": "base_checkpoint"}, "artifact type"),
        ({"base_checkpoint": "untrusted/base-model"}, "base checkpoint"),
        ({"base_revision": ""}, "base revision"),
        ({"created_at": "not-an-iso-timestamp"}, "creation time"),
        ({"label2id": {"ham": 0, "spam": 1, "phish": 2}}, "label-to-id"),
        ({"training_config": {}}, "training configuration"),
        ({"split_manifest_sha256": "invalid"}, "split manifest checksum"),
        ({"dependency_versions": {}}, "dependency versions"),
        ({"dataset": {}}, "dataset provenance"),
        ({"files": {}}, "file checksums"),
    ],
)
def test_manifest_identity_and_required_metadata_are_enforced(
    tmp_path: Path,
    update: dict[str, object],
    error_match: str,
) -> None:
    artifact = tmp_path / "distilbert"
    manifest = _write_exported_artifact(artifact)
    manifest.update(update)
    _write_manifest(artifact, manifest)

    with pytest.raises(runtime.DistilBertArtifactError, match=error_match):
        runtime.validate_artifact_directory(artifact, expected_max_length=128)


@pytest.mark.parametrize(
    ("removed_names", "error_match"),
    [
        ({"config.json"}, "config.json"),
        ({"model.safetensors"}, "model weights"),
        ({"tokenizer.json"}, "tokenizer files"),
    ],
)
def test_manifest_requires_model_config_weights_and_tokenizer(
    tmp_path: Path,
    removed_names: set[str],
    error_match: str,
) -> None:
    artifact = tmp_path / "distilbert"
    manifest = _write_exported_artifact(artifact)
    manifest["files"] = {
        name: digest
        for name, digest in dict(manifest["files"]).items()
        if name not in removed_names
    }
    _write_manifest(artifact, manifest)

    with pytest.raises(runtime.DistilBertArtifactError, match=error_match):
        runtime.validate_artifact_directory(artifact, expected_max_length=128)


def test_manifest_requires_metrics_and_declares_every_exported_file(tmp_path: Path) -> None:
    artifact = tmp_path / "distilbert"
    manifest = _write_exported_artifact(artifact)
    manifest["files"] = {
        name: digest for name, digest in dict(manifest["files"]).items() if name != "metrics.json"
    }
    _write_manifest(artifact, manifest)

    with pytest.raises(runtime.DistilBertArtifactError, match="metrics.json"):
        runtime.validate_artifact_directory(artifact, expected_max_length=128)

    manifest = _write_exported_artifact(tmp_path / "second")
    second = tmp_path / "second"
    (second / "special_tokens_map.json").write_text("{}", encoding="utf-8")
    with pytest.raises(runtime.DistilBertArtifactError, match="not declared"):
        runtime.validate_artifact_directory(second, expected_max_length=128)


def test_artifact_fingerprint_binds_canonical_declared_file_inventory(tmp_path: Path) -> None:
    artifact = tmp_path / "distilbert"
    manifest = _write_exported_artifact(artifact)
    manifest["artifact_fingerprint"] = "0" * 64
    (artifact / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(runtime.DistilBertArtifactError, match="fingerprint"):
        runtime.validate_artifact_directory(artifact, expected_max_length=128)


@pytest.mark.parametrize(
    ("config_update", "error_match"),
    [
        ({"model_type": "bert"}, "three-label DistilBERT"),
        ({"num_labels": 2}, "three-label DistilBERT"),
        ({"id2label": {"0": "ham", "1": "spam", "2": "phish"}}, "id-to-label"),
        ({"label2id": {"ham": 0, "spam": 1, "phish": 2}}, "label-to-id"),
    ],
)
def test_model_config_must_match_the_export_contract(
    tmp_path: Path,
    config_update: dict[str, object],
    error_match: str,
) -> None:
    artifact = tmp_path / "distilbert"
    manifest = _write_exported_artifact(artifact)
    config = json.loads((artifact / "config.json").read_text(encoding="utf-8"))
    config.update(config_update)
    (artifact / "config.json").write_text(json.dumps(config), encoding="utf-8")
    manifest_files = dict(manifest["files"])
    manifest_files["config.json"] = _sha256(artifact / "config.json")
    manifest["files"] = manifest_files
    _write_manifest(artifact, manifest)

    with pytest.raises(runtime.DistilBertArtifactError, match=error_match):
        runtime.validate_artifact_directory(artifact, expected_max_length=128)


def test_manifest_rejects_unsafe_and_missing_declared_files(tmp_path: Path) -> None:
    artifact = tmp_path / "distilbert"
    manifest = _write_exported_artifact(artifact)
    original_files = dict(manifest["files"])
    manifest["files"] = {"../outside.txt": "0" * 64, **original_files}
    _write_manifest(artifact, manifest)

    with pytest.raises(runtime.DistilBertArtifactError, match="unsafe file path"):
        runtime.validate_artifact_directory(artifact, expected_max_length=128)

    manifest["files"] = {"missing/vocab.txt": "0" * 64, **original_files}
    _write_manifest(artifact, manifest)
    with pytest.raises(runtime.DistilBertArtifactError, match="file is missing"):
        runtime.validate_artifact_directory(artifact, expected_max_length=128)


@pytest.mark.parametrize(
    ("manifest_update", "error_match"),
    [
        ({"training_complete": False}, "fine-tuned|training"),
        ({"max_length": 256}, "max_length"),
        ({"id2label": {"0": "ham", "1": "spam", "2": "phish"}}, "label"),
    ],
)
def test_invalid_fine_tuned_manifest_is_rejected(
    tmp_path: Path,
    manifest_update: dict[str, object],
    error_match: str,
) -> None:
    artifact = tmp_path / "distilbert"
    manifest = _write_exported_artifact(artifact)
    manifest.update(manifest_update)
    (artifact / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(runtime.DistilBertArtifactError, match=error_match):
        runtime.validate_artifact_directory(artifact, expected_max_length=128)


def test_runtime_uses_local_hf_files_eval_no_grad_exact_tokenization_and_softmax(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "distilbert"
    _write_exported_artifact(artifact)
    calls: dict[str, object] = {}

    class FakeTokenizer:
        def __call__(self, text: str, **kwargs: object) -> dict[str, object]:
            calls["tokenize"] = (text, kwargs)
            return {"input_ids": "encoded", "attention_mask": "mask"}

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, path: str | Path, **kwargs: object) -> FakeTokenizer:
            calls["tokenizer_load"] = (Path(path), kwargs)
            return FakeTokenizer()

    class FakeModel:
        def __init__(self) -> None:
            self.eval_called = False

        def eval(self) -> None:
            self.eval_called = True

        def __call__(self, **kwargs: object) -> SimpleNamespace:
            calls["model_call"] = kwargs
            return SimpleNamespace(logits="raw-logits")

    fake_model = FakeModel()

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, path: str | Path, **kwargs: object) -> FakeModel:
            calls["model_load"] = (Path(path), kwargs)
            return fake_model

    class FakeNoGrad:
        def __enter__(self) -> None:
            calls["no_grad_entered"] = True

        def __exit__(self, *_args: object) -> None:
            calls["no_grad_exited"] = True

    class FakeProbabilityRow:
        def tolist(self) -> list[float]:
            return [0.0900305732, 0.6652409558, 0.2447284711]

    class FakeTorch:
        @staticmethod
        def no_grad() -> FakeNoGrad:
            return FakeNoGrad()

        @staticmethod
        def softmax(logits: object, dim: int) -> list[FakeProbabilityRow]:
            calls["softmax"] = (logits, dim)
            return [FakeProbabilityRow()]

    monkeypatch.setattr(runtime, "AutoTokenizer", FakeAutoTokenizer)
    monkeypatch.setattr(runtime, "AutoModelForSequenceClassification", FakeAutoModel)
    monkeypatch.setattr(runtime, "torch", FakeTorch)

    classifier = runtime.DistilBertRuntime.from_directory(artifact, max_length=128)
    result = classifier.predict_one("keep this private")

    assert calls["tokenizer_load"] == (
        artifact,
        {"use_fast": True, "local_files_only": True},
    )
    assert calls["model_load"] == (
        artifact,
        {"local_files_only": True, "use_safetensors": True},
    )
    tokenized_text, tokenization_kwargs = calls["tokenize"]
    assert tokenized_text == "keep this private"
    assert tokenization_kwargs == {
        "truncation": True,
        "max_length": 128,
        "return_tensors": "pt",
    }
    assert fake_model.eval_called is True
    assert calls["no_grad_entered"] is True
    assert calls["no_grad_exited"] is True
    assert calls["softmax"] == ("raw-logits", -1)
    assert calls["model_call"] == {"input_ids": "encoded", "attention_mask": "mask"}
    assert result.label == "phish"
    assert result.probabilities == pytest.approx(
        {"ham": 0.0900305732, "phish": 0.6652409558, "spam": 0.2447284711}
    )
    assert math.isclose(sum(result.probabilities.values()), 1.0)
    assert "keep this private" not in repr(vars(classifier))


def test_optional_dependency_failure_is_reported_without_loading_a_base_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "distilbert"
    _write_exported_artifact(artifact)
    monkeypatch.setattr(runtime, "AutoTokenizer", None)
    monkeypatch.setattr(runtime, "AutoModelForSequenceClassification", None)
    monkeypatch.setattr(runtime, "torch", None)

    state = runtime.load_distilbert_state(
        AppSettings(distilbert_model_path=artifact),
    )

    assert state.loaded is False
    assert state.error is not None
    assert "dependency" in state.error.casefold()


def test_safe_zip_extraction_rejects_path_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "malicious.zip"
    destination = tmp_path / "cache"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../outside.txt", "escaped")

    with pytest.raises(runtime.DistilBertArtifactError, match="unsafe|traversal"):
        runtime.safe_extract_zip(archive_path, destination)

    assert not (tmp_path / "outside.txt").exists()


def test_safe_zip_extraction_rejects_symlinks_and_expansion_bombs(
    tmp_path: Path,
) -> None:
    symlink_archive = tmp_path / "symlink.zip"
    symlink = zipfile.ZipInfo("model-link")
    symlink.create_system = 3
    symlink.external_attr = (0o120777 << 16) | 0xA1ED
    with zipfile.ZipFile(symlink_archive, "w") as archive:
        archive.writestr(symlink, "target")

    with pytest.raises(runtime.DistilBertArtifactError, match="unsafe path traversal or link"):
        runtime.safe_extract_zip(symlink_archive, tmp_path / "symlink-cache")

    bomb_archive = tmp_path / "bomb.zip"
    with zipfile.ZipFile(bomb_archive, "w") as archive:
        archive.writestr("large.bin", b"12")

    with pytest.raises(runtime.DistilBertArtifactError, match="safe size limit"):
        runtime.safe_extract_zip(
            bomb_archive,
            tmp_path / "bomb-cache",
            max_artifact_bytes=1,
        )

    destination = tmp_path / "accepted-cache"
    runtime.safe_extract_zip(bomb_archive, destination, max_artifact_bytes=2)
    assert (destination / "large.bin").read_bytes() == b"12"


def test_safe_zip_never_removes_a_preexisting_destination(tmp_path: Path) -> None:
    archive_path = tmp_path / "invalid.zip"
    archive_path.write_text("not a zip", encoding="utf-8")
    destination = tmp_path / "cache"
    destination.mkdir()
    sentinel = destination / "sentinel.txt"
    sentinel.write_text("preserve", encoding="utf-8")

    with pytest.raises(runtime.DistilBertArtifactError, match="destination already exists"):
        runtime.safe_extract_zip(archive_path, destination)

    assert sentinel.read_text(encoding="utf-8") == "preserve"


@pytest.mark.parametrize(
    ("settings", "error_match"),
    [
        (
            AppSettings(
                distilbert_model_path=Path("local"),
                distilbert_model_url="https://models.example.test/model.zip",
                distilbert_model_sha256="0" * 64,
            ),
            "Configure only one",
        ),
        (
            AppSettings(distilbert_model_url="https://models.example.test/model.zip"),
            "SHA256 is required",
        ),
        (
            AppSettings(
                distilbert_model_url="https://models.example.test/model.zip",
                distilbert_model_sha256="bad-digest",
            ),
            "64-character hex",
        ),
        (
            AppSettings(
                distilbert_model_url="http://models.example.test/model.zip",
                distilbert_model_sha256="0" * 64,
            ),
            "must use https",
        ),
        (
            AppSettings(
                distilbert_model_url="https://operator:secret@models.example.test/model.zip",
                distilbert_model_sha256="0" * 64,
            ),
            "credentials",
        ),
        (
            AppSettings(
                distilbert_model_url="https://models.example.test/model.zip#fragment",
                distilbert_model_sha256="0" * 64,
            ),
            "fragment",
        ),
        (
            AppSettings(
                distilbert_model_url="https://169.254.169.254/model.zip",
                distilbert_model_sha256="0" * 64,
            ),
            "public network",
        ),
    ],
)
def test_distilbert_configuration_rejects_ambiguous_or_untrusted_sources(
    settings: AppSettings,
    error_match: str,
) -> None:
    _path, error = runtime.resolve_distilbert_artifact(settings)

    assert error is not None
    assert error_match in error


def test_remote_zip_is_checksum_verified_extracted_and_cached(tmp_path: Path) -> None:
    artifact = tmp_path / "source"
    _write_exported_artifact(artifact)
    source_zip = _zip_directory(artifact, tmp_path / "distilbert.zip")
    digest = _sha256(source_zip)
    cache_root = tmp_path / "cache" / "distilbert"
    calls = 0

    def downloader(_url: str, destination: Path) -> Path:
        nonlocal calls
        calls += 1
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_zip, destination)
        return destination

    settings = AppSettings(
        distilbert_model_url="https://models.example.test/distilbert.zip",
        distilbert_model_sha256=digest,
        distilbert_model_cache_path=cache_root,
    )

    first_path, first_error = runtime.resolve_distilbert_artifact(settings, downloader)
    second_path, second_error = runtime.resolve_distilbert_artifact(settings, downloader)

    assert first_error is None
    assert second_error is None
    expected_path = cache_root / f"sha256-{digest}"
    assert first_path == second_path == expected_path
    assert (expected_path / "manifest.json").is_file()
    assert calls == 1


def test_remote_cache_root_and_prior_digest_children_are_never_removed(tmp_path: Path) -> None:
    artifact = tmp_path / "source"
    _write_exported_artifact(artifact)
    first_zip = _zip_directory(artifact, tmp_path / "first.zip")
    second_zip = _zip_directory(artifact, tmp_path / "second.zip")
    with zipfile.ZipFile(second_zip, "a") as archive:
        archive.comment = b"same-artifact-new-published-archive"
    first_digest = _sha256(first_zip)
    second_digest = _sha256(second_zip)
    assert first_digest != second_digest

    cache_root = tmp_path / "shared-cache"
    cache_root.mkdir()
    sentinel = cache_root / "sentinel.txt"
    sentinel.write_text("preserve", encoding="utf-8")

    def downloader_from(source: Path):
        def download(_url: str, destination: Path) -> Path:
            shutil.copyfile(source, destination)
            return destination

        return download

    first_path, first_error = runtime.resolve_distilbert_artifact(
        AppSettings(
            distilbert_model_url="https://models.example.test/first.zip",
            distilbert_model_sha256=first_digest,
            distilbert_model_cache_path=cache_root,
        ),
        downloader_from(first_zip),
    )
    second_path, second_error = runtime.resolve_distilbert_artifact(
        AppSettings(
            distilbert_model_url="https://models.example.test/second.zip",
            distilbert_model_sha256=second_digest,
            distilbert_model_cache_path=cache_root,
        ),
        downloader_from(second_zip),
    )

    assert first_error is second_error is None
    assert first_path == cache_root / f"sha256-{first_digest}"
    assert second_path == cache_root / f"sha256-{second_digest}"
    assert first_path.is_dir()
    assert second_path.is_dir()
    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_valid_artifact_at_cache_root_cannot_override_checksum_child(tmp_path: Path) -> None:
    cache_root = tmp_path / "shared-cache"
    root_manifest = _write_exported_artifact(cache_root)
    source = tmp_path / "source"
    _write_exported_artifact(source)
    source_zip = _zip_directory(source, tmp_path / "distilbert.zip")
    digest = _sha256(source_zip)
    calls = 0

    def downloader(_url: str, destination: Path) -> Path:
        nonlocal calls
        calls += 1
        shutil.copyfile(source_zip, destination)
        return destination

    resolved, error = runtime.resolve_distilbert_artifact(
        AppSettings(
            distilbert_model_url="https://models.example.test/model.zip",
            distilbert_model_sha256=digest,
            distilbert_model_cache_path=cache_root,
        ),
        downloader,
    )

    assert error is None
    assert resolved == cache_root / f"sha256-{digest}"
    assert calls == 1
    assert _read_manifest(cache_root) == root_manifest


def test_remote_cache_root_file_is_rejected_without_modification(tmp_path: Path) -> None:
    cache_root = tmp_path / "not-a-directory"
    cache_root.write_text("preserve", encoding="utf-8")

    resolved, error = runtime.resolve_distilbert_artifact(
        AppSettings(
            distilbert_model_url="https://models.example.test/model.zip",
            distilbert_model_sha256="a" * 64,
            distilbert_model_cache_path=cache_root,
        ),
        lambda _url, destination: destination,
    )

    assert resolved == cache_root / f"sha256-{'a' * 64}"
    assert error is not None
    assert "cache root" in error.casefold()
    assert cache_root.read_text(encoding="utf-8") == "preserve"


def test_remote_extraction_uses_configured_max_artifact_size(tmp_path: Path) -> None:
    artifact = tmp_path / "source"
    _write_exported_artifact(artifact)
    source_zip = _zip_directory(artifact, tmp_path / "distilbert.zip")
    cache_root = tmp_path / "cache"

    def downloader(_url: str, destination: Path) -> Path:
        shutil.copyfile(source_zip, destination)
        return destination

    resolved, error = runtime.resolve_distilbert_artifact(
        AppSettings(
            distilbert_model_url="https://models.example.test/model.zip",
            distilbert_model_sha256=_sha256(source_zip),
            distilbert_model_cache_path=cache_root,
            max_artifact_bytes=1,
        ),
        downloader,
    )

    assert resolved == cache_root / f"sha256-{_sha256(source_zip)}"
    assert error is not None
    assert "safe size limit" in error
    assert not resolved.exists()


def test_remote_zip_accepts_one_nested_artifact_directory(tmp_path: Path) -> None:
    artifact = tmp_path / "source"
    _write_exported_artifact(artifact)
    source_zip = tmp_path / "nested-distilbert.zip"
    with zipfile.ZipFile(source_zip, "w") as archive:
        for file_path in artifact.iterdir():
            archive.write(file_path, f"distilbert/{file_path.name}")
    cache_root = tmp_path / "cache" / "distilbert"

    def downloader(_url: str, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_zip, destination)
        return destination

    resolved, error = runtime.resolve_distilbert_artifact(
        AppSettings(
            distilbert_model_url="https://models.example.test/distilbert.zip",
            distilbert_model_sha256=_sha256(source_zip),
            distilbert_model_cache_path=cache_root,
        ),
        downloader,
    )

    assert error is None
    assert resolved == cache_root / f"sha256-{_sha256(source_zip)}"
    assert resolved is not None
    assert (resolved / "manifest.json").is_file()


def test_invalid_digest_cache_is_reported_without_deleting_or_replacing_it(tmp_path: Path) -> None:
    artifact = tmp_path / "source"
    _write_exported_artifact(artifact)
    source_zip = _zip_directory(artifact, tmp_path / "distilbert.zip")
    digest = _sha256(source_zip)
    cache_root = tmp_path / "cache" / "distilbert"
    cache_path = cache_root / f"sha256-{digest}"
    cache_path.mkdir(parents=True)
    sentinel = cache_path / "sentinel.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    calls = 0

    def downloader(_url: str, destination: Path) -> Path:
        nonlocal calls
        calls += 1
        shutil.copyfile(source_zip, destination)
        return destination

    resolved, error = runtime.resolve_distilbert_artifact(
        AppSettings(
            distilbert_model_url="https://models.example.test/distilbert.zip",
            distilbert_model_sha256=digest,
            distilbert_model_cache_path=cache_root,
        ),
        downloader,
    )

    assert error is not None
    assert "existing checksum cache is invalid" in error.casefold()
    assert resolved == cache_path
    assert sentinel.read_text(encoding="utf-8") == "preserve"
    assert calls == 0


def test_remote_zip_checksum_mismatch_leaves_no_usable_cache(tmp_path: Path) -> None:
    source_zip = tmp_path / "distilbert.zip"
    with zipfile.ZipFile(source_zip, "w") as archive:
        archive.writestr("manifest.json", "{}")
    cache_root = tmp_path / "cache" / "distilbert"

    def downloader(_url: str, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_zip, destination)
        return destination

    resolved_path, error = runtime.resolve_distilbert_artifact(
        AppSettings(
            distilbert_model_url="https://models.example.test/distilbert.zip",
            distilbert_model_sha256="0" * 64,
            distilbert_model_cache_path=cache_root,
        ),
        downloader,
    )

    assert resolved_path == cache_root / f"sha256-{'0' * 64}"
    assert error is not None
    assert "checksum mismatch" in error.casefold()
    assert cache_root.is_dir()
    assert not resolved_path.exists()


def test_remote_zip_without_one_artifact_root_is_rejected_and_cleaned(tmp_path: Path) -> None:
    source_zip = tmp_path / "ambiguous.zip"
    with zipfile.ZipFile(source_zip, "w") as archive:
        archive.writestr("first/file.txt", "one")
        archive.writestr("second/file.txt", "two")
    cache_root = tmp_path / "cache" / "distilbert"

    def downloader(_url: str, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_zip, destination)
        return destination

    resolved, error = runtime.resolve_distilbert_artifact(
        AppSettings(
            distilbert_model_url="https://models.example.test/distilbert.zip",
            distilbert_model_sha256=_sha256(source_zip),
            distilbert_model_cache_path=cache_root,
        ),
        downloader,
    )

    assert resolved == cache_root / f"sha256-{_sha256(source_zip)}"
    assert error is not None
    assert "does not contain one" in error
    assert not resolved.exists()


def test_model_loading_failure_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "distilbert"
    _write_exported_artifact(artifact)

    def fail_loading(*_args: object, **_kwargs: object):
        raise OSError("C:/private/model contains secret configuration")

    monkeypatch.setattr(runtime.DistilBertRuntime, "from_directory", fail_loading)

    state = runtime.load_distilbert_state(AppSettings(distilbert_model_path=artifact))

    assert state.loaded is False
    assert state.error == "Could not load fine-tuned DistilBERT artifact: OSError."
    assert "private" not in state.error


def test_successful_model_state_preserves_validated_public_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "distilbert"
    manifest = _write_exported_artifact(artifact)
    classifier = object()
    monkeypatch.setattr(
        runtime.DistilBertRuntime,
        "from_directory",
        lambda *_args, **_kwargs: classifier,
    )

    state = runtime.load_distilbert_state(AppSettings(distilbert_model_path=artifact))

    assert state.loaded is True
    assert state.runtime is classifier
    assert state.manifest == manifest
    assert runtime.public_manifest({**manifest, "files": {"secret": "digest"}}) == {
        "artifact_fingerprint": manifest["artifact_fingerprint"],
        "training_complete": True,
        "base_checkpoint": "distilbert/distilbert-base-uncased",
        "base_revision": runtime.BASE_REVISION,
        "created_at": "2026-08-09T10:00:00+00:00",
        "max_length": 128,
        "split_manifest_sha256": "a" * 64,
    }


def test_public_manifest_omits_unvalidated_or_unsafe_values() -> None:
    assert (
        runtime.public_manifest(
            {
                "training_complete": "yes",
                "base_checkpoint": "C:/private/model",
                "base_revision": "bad",
                "created_at": "<script>alert(1)</script>",
                "max_length": -1,
                "split_manifest_sha256": "not-a-digest",
                "artifact_fingerprint": "also-not-a-digest",
            }
        )
        == {}
    )


def test_distilbert_service_loads_lazily_and_only_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = 0
    expected = runtime.DistilBertState(
        runtime=None,
        path=tmp_path / "missing",
        manifest={},
        error="not configured",
    )

    def fake_load(_settings: AppSettings, *_args: object, **_kwargs: object):
        nonlocal calls
        calls += 1
        return expected

    monkeypatch.setattr(runtime, "load_distilbert_state", fake_load)
    service = runtime.DistilBertService(AppSettings())

    assert calls == 0
    assert service.get_state() is expected
    assert service.get_state() is expected
    assert calls == 1
