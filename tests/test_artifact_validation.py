from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from email_threat_detector.artifact_validation import (
    ArtifactValidationError,
    validate_artifacts,
)


def _sha256(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def _write_fixture(root: Path) -> tuple[bytes, Path]:
    artifact_dir = root / "artifacts" / "distilbert"
    artifact_dir.mkdir(parents=True)
    baseline = b"trusted baseline"
    (root / "artifacts" / "tfidf_logreg.joblib").write_bytes(baseline)
    files: dict[str, dict[str, object]] = {}
    for name, contents in {
        "config.json": b'{"max_length":128}\n',
        "metrics.json": b'{"accuracy":0.9}\n',
        "model.safetensors": b"weights",
        "tokenizer.json": b'{"tokenizer":true}\n',
    }.items():
        (artifact_dir / name).write_bytes(contents)
        files[name] = {"sha256": _sha256(contents), "size_bytes": len(contents)}
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    manifest = {
        "artifact_fingerprint": hashlib.sha256(canonical).hexdigest(),
        "files": files,
    }
    manifest_path = artifact_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return baseline, manifest_path


def test_validate_artifacts_accepts_materialized_checksum_bound_files(tmp_path: Path) -> None:
    baseline, _ = _write_fixture(tmp_path)

    result = validate_artifacts(tmp_path, baseline_sha256=_sha256(baseline))

    assert result.baseline_sha256 == _sha256(baseline)
    assert result.distilbert_file_count == 4


@pytest.mark.parametrize("target", ["baseline", "transformer"])
def test_validate_artifacts_rejects_lfs_pointers(tmp_path: Path, target: str) -> None:
    baseline, _ = _write_fixture(tmp_path)
    pointer = (
        b"version https://git-lfs.github.com/spec/v1\n"
        b"oid sha256:0000000000000000000000000000000000000000000000000000000000000000\n"
        b"size 999\n"
    )
    if target == "baseline":
        (tmp_path / "artifacts" / "tfidf_logreg.joblib").write_bytes(pointer)
    else:
        (tmp_path / "artifacts" / "distilbert" / "model.safetensors").write_bytes(pointer)

    with pytest.raises(ArtifactValidationError, match="Git LFS pointer"):
        validate_artifacts(tmp_path, baseline_sha256=_sha256(baseline))


def test_validate_artifacts_detects_line_ending_checksum_changes(tmp_path: Path) -> None:
    baseline, _ = _write_fixture(tmp_path)
    config = tmp_path / "artifacts" / "distilbert" / "config.json"
    config.write_bytes(config.read_bytes().replace(b"\n", b"\r\n"))

    with pytest.raises(ArtifactValidationError, match="size mismatch"):
        validate_artifacts(tmp_path, baseline_sha256=_sha256(baseline))


def test_validate_artifacts_rejects_manifest_fingerprint_changes(tmp_path: Path) -> None:
    baseline, manifest_path = _write_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_fingerprint"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ArtifactValidationError, match="fingerprint"):
        validate_artifacts(tmp_path, baseline_sha256=_sha256(baseline))
