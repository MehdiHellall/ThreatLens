from __future__ import annotations

from pathlib import Path

import pytest

from scripts import threatlens


def test_backend_dockerfile_installs_locked_uv_environment() -> None:
    dockerfile = (threatlens.ROOT / "web" / "backend" / "Dockerfile").read_text(encoding="utf-8")

    assert "uv sync --frozen" in dockerfile
    assert 'PATH="/app/.venv/bin:$PATH"' in dockerfile
    assert "uv export" not in dockerfile
    assert "--requirement /tmp/requirements.txt" not in dockerfile


def test_clean_is_dry_run_by_default_and_preserves_project_assets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache = tmp_path / ".pytest_cache"
    cache.mkdir()
    (cache / "state").write_text("temporary", encoding="utf-8")
    artifact = tmp_path / "artifacts" / "model.bin"
    artifact.parent.mkdir()
    artifact.write_bytes(b"model")
    dataset = tmp_path / "data" / "dataset.csv"
    dataset.parent.mkdir()
    dataset.write_text("text,label\nhello,ham\n", encoding="utf-8")
    source_cache = tmp_path / "src" / "package" / "__pycache__"
    source_cache.mkdir(parents=True)
    environment_cache = tmp_path / ".venv" / "package" / "__pycache__"
    environment_cache.mkdir(parents=True)
    monkeypatch.setattr(threatlens, "ROOT", tmp_path)

    threatlens.clean(execute=False)

    assert cache.is_dir()
    assert artifact.is_file()
    assert dataset.is_file()
    assert source_cache.is_dir()
    assert environment_cache.is_dir()

    threatlens.clean(execute=True)

    assert not cache.exists()
    assert artifact.is_file()
    assert dataset.is_file()
    assert not source_cache.exists()
    assert environment_cache.is_dir()


def test_clean_refuses_allowlisted_symlinks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    link = tmp_path / "unsafe-cache"
    link.symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(threatlens, "ROOT", tmp_path)
    monkeypatch.setattr(threatlens, "SAFE_CLEAN_PATHS", ("unsafe-cache",))

    with pytest.raises(threatlens.CommandError, match="symlink"):
        threatlens.clean(execute=True)

    assert outside.is_dir()


def test_clean_dry_run_command_does_not_remove_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache = tmp_path / ".pytest_cache"
    cache.mkdir()
    monkeypatch.setattr(threatlens, "ROOT", tmp_path)

    assert threatlens.main(["clean-dry-run"]) == 0

    assert cache.is_dir()


def test_wait_for_demo_requires_full_two_model_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads = iter(
        (
            {"ready": True, "duel_ready": False, "service_status": "limited"},
            {"ready": True, "duel_ready": True, "service_status": "ready"},
        )
    )
    monkeypatch.setattr(threatlens, "_ready_payload", lambda: next(payloads))
    monkeypatch.setattr(threatlens.time, "sleep", lambda _seconds: None)

    result = threatlens.wait_for_demo(timeout_seconds=5)

    assert result["duel_ready"] is True
