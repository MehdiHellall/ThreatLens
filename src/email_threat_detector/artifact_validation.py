"""Fast, dependency-free validation for ThreatLens demo artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

BASELINE_SHA256 = "d9ed306935e26c9bf7b285a861991bb3539be7089ad9d8bf798d781d01d45981"
LFS_HEADER = b"version https://git-lfs.github.com/spec/v1"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class ArtifactValidationError(RuntimeError):
    """Raised when checked-out model artifacts are missing or untrustworthy."""


@dataclass(frozen=True)
class ArtifactValidationResult:
    """Validated identities displayed by setup tooling and CI."""

    baseline_sha256: str
    distilbert_fingerprint: str
    distilbert_file_count: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("rb") as source:
            return source.read(len(LFS_HEADER)) == LFS_HEADER
    except OSError:
        return False


def _require_materialized_file(path: Path, *, description: str) -> None:
    if not path.is_file():
        raise ArtifactValidationError(f"Missing {description}: {path}")
    if _is_lfs_pointer(path):
        raise ArtifactValidationError(
            f"{description.capitalize()} is still a Git LFS pointer: {path}. "
            "Install Git LFS, then run `git lfs pull`."
        )


def _load_manifest(path: Path) -> dict[str, Any]:
    _require_materialized_file(path, description="DistilBERT manifest")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError("DistilBERT manifest is not valid UTF-8 JSON.") from exc
    if not isinstance(payload, dict):
        raise ArtifactValidationError("DistilBERT manifest must contain a JSON object.")
    return payload


def _validated_manifest_entry(name: object, value: object) -> tuple[str, str, int]:
    if not isinstance(name, str):
        raise ArtifactValidationError("DistilBERT manifest contains a non-string filename.")
    relative = PurePosixPath(name)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ArtifactValidationError(f"DistilBERT manifest contains an unsafe path: {name!r}")
    if not isinstance(value, Mapping):
        raise ArtifactValidationError(f"Invalid manifest entry for {name}.")
    digest = value.get("sha256")
    size = value.get("size_bytes")
    if not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None:
        raise ArtifactValidationError(f"Invalid SHA-256 in manifest entry for {name}.")
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise ArtifactValidationError(f"Invalid size in manifest entry for {name}.")
    return name, digest, size


def _artifact_fingerprint(files: Mapping[str, object]) -> str:
    encoded = json.dumps(
        files,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_artifacts(
    root: str | Path,
    *,
    baseline_sha256: str = BASELINE_SHA256,
) -> ArtifactValidationResult:
    """Validate the exact baseline and every DistilBERT manifest file."""
    repository = Path(root).resolve()
    baseline = repository / "artifacts" / "tfidf_logreg.joblib"
    _require_materialized_file(baseline, description="TF-IDF model artifact")
    actual_baseline_sha256 = _sha256(baseline)
    if actual_baseline_sha256 != baseline_sha256:
        raise ArtifactValidationError(
            "TF-IDF model checksum mismatch; refusing to deserialize an untrusted artifact."
        )

    artifact_dir = repository / "artifacts" / "distilbert"
    manifest = _load_manifest(artifact_dir / "manifest.json")
    raw_files = manifest.get("files")
    if not isinstance(raw_files, dict) or not raw_files:
        raise ArtifactValidationError("DistilBERT manifest has no file inventory.")
    expected_fingerprint = manifest.get("artifact_fingerprint")
    if (
        not isinstance(expected_fingerprint, str)
        or SHA256_PATTERN.fullmatch(expected_fingerprint) is None
        or _artifact_fingerprint(raw_files) != expected_fingerprint
    ):
        raise ArtifactValidationError("DistilBERT manifest fingerprint mismatch.")

    for raw_name, raw_entry in raw_files.items():
        name, expected_sha256, expected_size = _validated_manifest_entry(raw_name, raw_entry)
        candidate = artifact_dir.joinpath(*PurePosixPath(name).parts)
        try:
            candidate.resolve().relative_to(artifact_dir.resolve())
        except ValueError as exc:
            raise ArtifactValidationError(f"Artifact path escapes its directory: {name}") from exc
        _require_materialized_file(candidate, description=f"DistilBERT file {name}")
        if candidate.stat().st_size != expected_size:
            raise ArtifactValidationError(f"DistilBERT file size mismatch: {name}")
        if _sha256(candidate) != expected_sha256:
            raise ArtifactValidationError(f"DistilBERT file checksum mismatch: {name}")

    return ArtifactValidationResult(
        baseline_sha256=actual_baseline_sha256,
        distilbert_fingerprint=expected_fingerprint,
        distilbert_file_count=len(raw_files),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository root (default: current directory).",
    )
    args = parser.parse_args(argv)
    try:
        result = validate_artifacts(args.root)
    except ArtifactValidationError as exc:
        parser.exit(1, f"Artifact validation failed: {exc}\n")
    print(
        "Artifacts valid: "
        f"TF-IDF sha256={result.baseline_sha256}; "
        f"DistilBERT fingerprint={result.distilbert_fingerprint} "
        f"({result.distilbert_file_count} files)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
