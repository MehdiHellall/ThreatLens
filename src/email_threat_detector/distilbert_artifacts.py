"""Packaging helpers for exported DistilBERT artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class RunContractError(RuntimeError):
    """Raised when a training run cannot be safely started or resumed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_fingerprint(files: Mapping[str, Any]) -> str:
    """Hash the canonical manifest file mapping for a stable artifact identity."""
    canonical = json.dumps(
        files,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def ensure_run_contract(
    path: Path,
    expected: Mapping[str, Any],
    *,
    require_existing: bool,
) -> dict[str, Any]:
    """Atomically create an immutable run contract or verify an existing one."""
    payload = dict(expected)
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RunContractError("Existing run contract is unreadable or invalid.") from exc
        if existing != payload:
            raise RunContractError("Existing run contract does not match this training request.")
        return existing
    if require_existing:
        raise RunContractError("A matching run contract is required before resuming training.")

    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as temporary:
        temporary.write(serialized)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    try:
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            return ensure_run_contract(path, payload, require_existing=True)
    finally:
        temporary_path.unlink(missing_ok=True)
    return payload


def _has_model_weights(checkpoint: Path) -> bool:
    direct_weights = (checkpoint / "model.safetensors", checkpoint / "pytorch_model.bin")
    if any(path.is_file() and path.stat().st_size > 0 for path in direct_weights):
        return True
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = checkpoint / index_name
        if not index_path.is_file():
            continue
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = index["weight_map"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            continue
        if not isinstance(weight_map, dict) or not weight_map:
            continue
        shards = {checkpoint / str(filename) for filename in weight_map.values()}
        if all(path.is_file() and path.stat().st_size > 0 for path in shards):
            return True
    return False


def is_complete_checkpoint(checkpoint: Path) -> bool:
    """Return whether a checkpoint contains all state needed for exact resume."""
    required = (
        checkpoint / "trainer_state.json",
        checkpoint / "config.json",
        checkpoint / "optimizer.pt",
        checkpoint / "scheduler.pt",
    )
    if not all(path.is_file() and path.stat().st_size > 0 for path in required):
        return False
    if not _has_model_weights(checkpoint):
        return False
    rng_files = tuple(checkpoint.glob("rng_state*.pth"))
    if not any(path.is_file() and path.stat().st_size > 0 for path in rng_files):
        return False
    try:
        json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
        json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return True


def find_resume_checkpoint(work_dir: str | Path) -> Path | None:
    """Return the highest numbered fully restorable Hugging Face checkpoint."""
    directory = Path(work_dir)
    if not directory.is_dir():
        return None
    candidates: list[tuple[int, Path]] = []
    for path in directory.iterdir():
        if not path.is_dir() or not path.name.startswith("checkpoint-"):
            continue
        try:
            step = int(path.name.removeprefix("checkpoint-"))
        except ValueError:
            continue
        if is_complete_checkpoint(path):
            candidates.append((step, path))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def create_artifact_archive(
    artifact_dir: str | Path,
    output_dir: str | Path,
) -> tuple[Path, str]:
    """Create a deterministic, checksum-labelled ZIP suitable for HTTPS hosting."""
    source = Path(artifact_dir)
    destination = Path(output_dir)
    if not source.is_dir():
        raise FileNotFoundError(f"Artifact directory does not exist: {source}")
    files = tuple(path for path in sorted(source.rglob("*")) if path.is_file())
    if not files:
        raise ValueError("Artifact directory contains no files.")
    if any(path.is_symlink() for path in files):
        raise ValueError("Artifact archives cannot contain symbolic links.")

    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix="distilbert-",
        suffix=".tmp.zip",
        dir=destination,
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with zipfile.ZipFile(
            temporary_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as archive:
            for path in files:
                relative = path.relative_to(source).as_posix()
                info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                with path.open("rb") as input_file, archive.open(info, "w") as output_file:
                    shutil.copyfileobj(input_file, output_file, length=1024 * 1024)
        checksum = _sha256(temporary_path)
        archive_path = destination / f"distilbert-{checksum}.zip"
        temporary_path.replace(archive_path)
        archive_path.with_suffix(".zip.sha256").write_text(
            f"{checksum}  {archive_path.name}\n",
            encoding="utf-8",
        )
        return archive_path, checksum
    finally:
        temporary_path.unlink(missing_ok=True)
