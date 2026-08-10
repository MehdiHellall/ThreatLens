from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from email_threat_detector import distilbert_training as training
from email_threat_detector.preprocessing import basic_text_preprocessor


def _write_source_dataset(path: Path) -> Path:
    rows: list[dict[str, object]] = []
    numeric_labels = {"ham": 0, "phish": 1, "spam": 2}
    for label, numeric_label in numeric_labels.items():
        rows.extend(
            {"text": f"Unique {label} message {index}", "label": numeric_label}
            for index in range(10)
        )

    rows.extend(
        [
            {"text": "  UNIQUE HAM MESSAGE 0  ", "label": "ham"},
            {"text": "Same conflicting message", "label": "spam"},
            {"text": " same   conflicting MESSAGE ", "label": "phish"},
            {"text": "   ", "label": "ham"},
            {"text": None, "label": "spam"},
        ]
    )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prepare_dataset(tmp_path: Path, output_name: str = "prepared") -> Path:
    source = _write_source_dataset(tmp_path / f"{output_name}-source.csv")
    output = tmp_path / output_name
    training.prepare_splits(source, output)
    return output


def test_training_config_matches_notebook_defaults() -> None:
    config = training.TrainingConfig()

    assert config.base_checkpoint == "distilbert/distilbert-base-uncased"
    assert config.base_revision == "12040accade4e8a0f71eabdb258fecc2e7e948be"
    assert config.max_length == 128
    assert config.epochs == 3
    assert config.batch_size == 16
    assert config.learning_rate == pytest.approx(2e-5)
    assert config.weight_decay == pytest.approx(0.01)
    assert config.eval_strategy == "epoch"
    assert config.save_strategy == "epoch"
    assert config.load_best_model_at_end is True
    assert config.save_total_limit == 1
    assert config.seed == 42


def test_training_config_rejects_an_unapproved_base_checkpoint() -> None:
    with pytest.raises(ValueError, match="base checkpoint"):
        training.TrainingConfig(
            base_checkpoint="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
        )


def test_prepare_splits_is_leakage_safe_balanced_and_deterministic(tmp_path: Path) -> None:
    source = _write_source_dataset(tmp_path / "source.csv")
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    first_manifest = training.prepare_splits(source, first_dir)
    second_manifest = training.prepare_splits(source, second_dir)

    expected_rows = {"train": 24, "validation": 3, "test": 3}
    normalized_split_texts: dict[str, set[str]] = {}
    for split_name, row_count in expected_rows.items():
        first_file = first_dir / f"{split_name}.csv"
        second_file = second_dir / f"{split_name}.csv"
        first = pd.read_csv(first_file)

        assert first_file.read_bytes() == second_file.read_bytes()
        assert len(first) == row_count
        assert first["label"].value_counts().to_dict() == {
            "ham": row_count // 3,
            "phish": row_count // 3,
            "spam": row_count // 3,
        }
        assert first_manifest["splits"][split_name] == second_manifest["splits"][split_name]
        assert first_manifest["splits"][split_name]["sha256"] == _sha256(first_file)
        normalized_split_texts[split_name] = {
            basic_text_preprocessor(text) for text in first["text"]
        }

    assert normalized_split_texts["train"].isdisjoint(normalized_split_texts["validation"])
    assert normalized_split_texts["train"].isdisjoint(normalized_split_texts["test"])
    assert normalized_split_texts["validation"].isdisjoint(normalized_split_texts["test"])
    assert all("conflicting message" not in texts for texts in normalized_split_texts.values())
    assert first_manifest["data_summary"]["conflicting_text_groups_removed"] == 1
    assert first_manifest["data_summary"]["modeling_rows"] == 30
    assert first_manifest["data_config"] == {
        "balance_classes": True,
        "random_state": 42,
        "samples_per_class": None,
        "test_size": 0.1,
        "validation_size": 0.1,
    }


def test_validate_prepared_splits_rejects_checksum_changes_and_cross_split_overlap(
    tmp_path: Path,
) -> None:
    prepared_dir = _prepare_dataset(tmp_path)
    manifest_path = prepared_dir / "manifest.json"

    manifest = training.validate_prepared_splits(prepared_dir)
    assert manifest == json.loads(manifest_path.read_text(encoding="utf-8"))

    train_file = prepared_dir / "train.csv"
    train_file.write_text(
        train_file.read_text(encoding="utf-8") + "tampered,ham\n", encoding="utf-8"
    )
    with pytest.raises(training.PreparedDataError, match="checksum"):
        training.validate_prepared_splits(prepared_dir)

    # Restore a valid preparation, then deliberately create a checksum-valid overlap.
    prepared_dir = _prepare_dataset(tmp_path, "overlap")
    manifest_path = prepared_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train = pd.read_csv(prepared_dir / "train.csv")
    validation_file = prepared_dir / "validation.csv"
    validation = pd.read_csv(validation_file)
    validation.loc[0, "text"] = train.loc[0, "text"]
    validation.to_csv(validation_file, index=False)
    manifest["splits"]["validation"]["sha256"] = _sha256(validation_file)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(training.PreparedDataError, match="overlap"):
        training.validate_prepared_splits(prepared_dir)


def test_validate_prepared_splits_enforces_balance_and_exact_allocation(tmp_path: Path) -> None:
    prepared_dir = _prepare_dataset(tmp_path)
    manifest_path = prepared_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_file = prepared_dir / "train.csv"
    train = pd.read_csv(train_file)
    train.loc[train["label"] == "ham", "label"] = "spam"
    train.to_csv(train_file, index=False)
    manifest["splits"]["train"]["sha256"] = _sha256(train_file)
    manifest["splits"]["train"]["class_counts"] = {
        label: int(count) for label, count in train["label"].value_counts().sort_index().items()
    }
    manifest["data_summary"]["class_counts"]["train"] = manifest["splits"]["train"]["class_counts"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(training.PreparedDataError, match="balanced"):
        training.validate_prepared_splits(prepared_dir)

    prepared_dir = _prepare_dataset(tmp_path, "allocation")
    manifest_path = prepared_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_file = prepared_dir / "train.csv"
    validation_file = prepared_dir / "validation.csv"
    train = pd.read_csv(train_file)
    validation = pd.read_csv(validation_file)
    moved = train.groupby("label", sort=True).head(1)
    train = train.drop(index=moved.index).reset_index(drop=True)
    validation = pd.concat([validation, moved], ignore_index=True)
    for name, path, frame in (
        ("train", train_file, train),
        ("validation", validation_file, validation),
    ):
        frame.to_csv(path, index=False)
        manifest["splits"][name] = {
            "filename": path.name,
            "rows": len(frame),
            "sha256": _sha256(path),
            "class_counts": {
                label: int(count)
                for label, count in frame["label"].value_counts().sort_index().items()
            },
        }
        manifest["data_summary"]["class_counts"][name] = manifest["splits"][name]["class_counts"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(training.PreparedDataError, match="80/10/10"):
        training.validate_prepared_splits(prepared_dir)


def _write_complete_checkpoint(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "trainer_state.json").write_text("{}", encoding="utf-8")
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"weights")
    (path / "optimizer.pt").write_bytes(b"optimizer")
    (path / "scheduler.pt").write_bytes(b"scheduler")
    (path / "rng_state.pth").write_bytes(b"rng")


def test_find_resume_checkpoint_selects_latest_complete_checkpoint(tmp_path: Path) -> None:
    for step in (2, 11):
        checkpoint = tmp_path / f"checkpoint-{step}"
        _write_complete_checkpoint(checkpoint)
    incomplete = tmp_path / "checkpoint-30"
    incomplete.mkdir()
    (incomplete / "trainer_state.json").write_text("{}", encoding="utf-8")
    (tmp_path / "checkpoint-not-a-step").mkdir()

    assert training.find_resume_checkpoint(tmp_path) == tmp_path / "checkpoint-11"
    assert training.find_resume_checkpoint(tmp_path / "missing") is None


def test_run_contract_is_immutable_and_binds_config_splits_and_revision(tmp_path: Path) -> None:
    prepared_dir = _prepare_dataset(tmp_path)
    work_dir = tmp_path / "work"
    config = training.TrainingConfig()

    contract = training.initialize_run_contract(prepared_dir, work_dir, config=config)

    assert contract == json.loads((work_dir / "run_contract.json").read_text(encoding="utf-8"))
    assert contract["base_checkpoint"] == config.base_checkpoint
    assert contract["base_revision"] == config.base_revision
    assert contract["split_manifest_sha256"] == _sha256(prepared_dir / "manifest.json")
    assert contract["training_config"] == asdict(config)
    assert training.initialize_run_contract(prepared_dir, work_dir, config=config) == contract

    with pytest.raises(training.RunContractError, match="does not match"):
        training.initialize_run_contract(
            prepared_dir,
            work_dir,
            config=replace(config, max_length=64),
        )


def test_resume_requires_matching_contract_and_complete_checkpoint(tmp_path: Path) -> None:
    prepared_dir = _prepare_dataset(tmp_path)
    work_dir = tmp_path / "work"
    _write_complete_checkpoint(work_dir / "checkpoint-12")

    with pytest.raises(training.RunContractError, match="run contract"):
        training.train_and_export(
            prepared_dir,
            tmp_path / "artifact",
            work_dir=work_dir,
            resume=True,
            dependencies=_fake_training_dependencies(),
        )

    training.initialize_run_contract(prepared_dir, work_dir)
    (work_dir / "checkpoint-12" / "optimizer.pt").unlink()
    with pytest.raises(training.RunContractError, match="complete checkpoint"):
        training.train_and_export(
            prepared_dir,
            tmp_path / "artifact",
            work_dir=work_dir,
            resume=True,
            dependencies=_fake_training_dependencies(),
        )


class FakeDataset:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame.reset_index(drop=True)
        self.column_names = list(frame.columns)

    @classmethod
    def from_pandas(cls, frame: pd.DataFrame, *, preserve_index: bool) -> FakeDataset:
        assert preserve_index is False
        return cls(frame)

    def map(self, function, *, batched: bool) -> FakeDataset:
        assert batched is True
        function({"text": self.frame["text"].tolist()})
        return self

    def remove_columns(self, columns: list[str]) -> FakeDataset:
        return FakeDataset(
            self.frame.drop(columns=[column for column in columns if column in self.frame])
        )


class FakeDatasetDict(dict[str, FakeDataset]):
    def map(self, function, *, batched: bool) -> FakeDatasetDict:
        return FakeDatasetDict(
            {name: dataset.map(function, batched=batched) for name, dataset in self.items()}
        )

    def remove_columns(self, columns: list[str]) -> FakeDatasetDict:
        return FakeDatasetDict(
            {name: dataset.remove_columns(columns) for name, dataset in self.items()}
        )


class FakeTokenizer:
    load_calls: list[tuple[str, bool, str]] = []
    tokenize_calls: list[dict[str, Any]] = []

    @classmethod
    def from_pretrained(cls, checkpoint: str, *, use_fast: bool, revision: str) -> FakeTokenizer:
        cls.load_calls.append((checkpoint, use_fast, revision))
        return cls()

    def __call__(self, texts: list[str], **kwargs: object) -> dict[str, list[list[int]]]:
        self.tokenize_calls.append({"texts": texts, **kwargs})
        return {"input_ids": [[1, 2] for _ in texts]}

    def save_pretrained(self, output_dir: str | Path) -> None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "tokenizer.json").write_text('{"fast": true}', encoding="utf-8")
        (destination / "tokenizer_config.json").write_text("{}", encoding="utf-8")


class FakeModel:
    load_calls: list[tuple[str, dict[str, object]]] = []

    @classmethod
    def from_pretrained(cls, checkpoint: str, **kwargs: object) -> FakeModel:
        cls.load_calls.append((checkpoint, kwargs))
        return cls()

    def save_pretrained(self, output_dir: str | Path, **_kwargs: object) -> None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "distilbert",
                    "id2label": {"0": "ham", "1": "phish", "2": "spam"},
                    "label2id": {"ham": 0, "phish": 1, "spam": 2},
                }
            ),
            encoding="utf-8",
        )
        (destination / "model.safetensors").write_bytes(b"fine-tuned-weights")


class FakeTrainingArguments:
    calls: list[dict[str, object]] = []

    def __init__(self, **kwargs: object) -> None:
        self.calls.append(kwargs)


class FakeTrainer:
    instances: list[FakeTrainer] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.resume_from_checkpoint: str | None = None
        self.instances.append(self)

    def train(self, *, resume_from_checkpoint: str | None = None) -> SimpleNamespace:
        self.resume_from_checkpoint = resume_from_checkpoint
        return SimpleNamespace(metrics={"train_runtime": 12.5, "train_loss": 0.25})

    def predict(self, dataset: FakeDataset) -> SimpleNamespace:
        labels = np.array([0, 1, 2], dtype=np.int64)
        assert len(dataset.frame) == len(labels)
        logits = np.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]])
        return SimpleNamespace(predictions=logits, label_ids=labels, metrics={"test_loss": 0.1})

    def save_model(self, output_dir: str | Path) -> None:
        self.kwargs["model"].save_pretrained(output_dir)


class FakeDataCollatorWithPadding:
    def __init__(self, *, tokenizer: FakeTokenizer) -> None:
        self.tokenizer = tokenizer


def _fake_training_dependencies() -> SimpleNamespace:
    return SimpleNamespace(
        AutoTokenizer=FakeTokenizer,
        AutoModelForSequenceClassification=FakeModel,
        DataCollatorWithPadding=FakeDataCollatorWithPadding,
        Dataset=FakeDataset,
        DatasetDict=FakeDatasetDict,
        Trainer=FakeTrainer,
        TrainingArguments=FakeTrainingArguments,
        torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
    )


def test_train_and_export_uses_real_hf_contract_and_writes_verifiable_artifact(
    tmp_path: Path,
) -> None:
    prepared_dir = _prepare_dataset(tmp_path)
    artifact_dir = tmp_path / "artifact"
    work_dir = tmp_path / "training-work"
    checkpoint = work_dir / "checkpoint-23"
    _write_complete_checkpoint(checkpoint)
    training.initialize_run_contract(prepared_dir, work_dir)
    FakeTokenizer.load_calls.clear()
    FakeTokenizer.tokenize_calls.clear()
    FakeModel.load_calls.clear()
    FakeTrainingArguments.calls.clear()
    FakeTrainer.instances.clear()

    training.train_and_export(
        prepared_dir,
        artifact_dir,
        work_dir=work_dir,
        config=training.TrainingConfig(),
        resume=True,
        dependencies=_fake_training_dependencies(),
    )

    assert FakeTokenizer.load_calls == [
        (
            "distilbert/distilbert-base-uncased",
            True,
            "12040accade4e8a0f71eabdb258fecc2e7e948be",
        )
    ]
    assert FakeTokenizer.tokenize_calls
    assert all(call["truncation"] is True for call in FakeTokenizer.tokenize_calls)
    assert all(call["max_length"] == 128 for call in FakeTokenizer.tokenize_calls)
    assert FakeModel.load_calls == [
        (
            "distilbert/distilbert-base-uncased",
            {
                "num_labels": 3,
                "id2label": {0: "ham", 1: "phish", 2: "spam"},
                "label2id": {"ham": 0, "phish": 1, "spam": 2},
                "revision": "12040accade4e8a0f71eabdb258fecc2e7e948be",
            },
        )
    ]

    arguments = FakeTrainingArguments.calls[0]
    assert arguments["output_dir"] == str(work_dir)
    assert arguments["learning_rate"] == pytest.approx(2e-5)
    assert arguments["per_device_train_batch_size"] == 16
    assert arguments["per_device_eval_batch_size"] == 16
    assert arguments["num_train_epochs"] == 3
    assert arguments["weight_decay"] == pytest.approx(0.01)
    assert arguments["eval_strategy"] == "epoch"
    assert arguments["save_strategy"] == "epoch"
    assert arguments["load_best_model_at_end"] is True
    assert arguments["save_total_limit"] == 1
    assert arguments["fp16"] is False
    assert arguments["seed"] == 42
    assert arguments["data_seed"] == 42
    assert "save_safetensors" not in arguments
    assert FakeTrainer.instances[0].resume_from_checkpoint == str(checkpoint)

    expected_files = {
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "metrics.json",
    }
    assert expected_files.issubset({path.name for path in artifact_dir.iterdir()})
    metrics = json.loads((artifact_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["test"]["accuracy"] == pytest.approx(1.0)
    assert metrics["test"]["per_class"].keys() == {"ham", "phish", "spam"}
    assert metrics["test"]["confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert metrics["training"]["train_runtime"] == pytest.approx(12.5)

    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["training_complete"] is True
    assert manifest["base_checkpoint"] == "distilbert/distilbert-base-uncased"
    assert manifest["base_revision"] == "12040accade4e8a0f71eabdb258fecc2e7e948be"
    assert manifest["max_length"] == 128
    assert manifest["id2label"] == {"0": "ham", "1": "phish", "2": "spam"}
    assert manifest["label2id"] == {"ham": 0, "phish": 1, "spam": 2}
    assert manifest["split_manifest_sha256"] == _sha256(prepared_dir / "manifest.json")
    for filename in expected_files:
        assert manifest["files"][filename]["sha256"] == _sha256(artifact_dir / filename)
    canonical_files = json.dumps(
        manifest["files"], sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    assert manifest["artifact_fingerprint"] == hashlib.sha256(canonical_files).hexdigest()


def test_finalize_artifact_manifest_adds_revision_and_stable_fingerprint(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "legacy-artifact"
    artifact_dir.mkdir()
    (artifact_dir / "config.json").write_text("{}", encoding="utf-8")
    (artifact_dir / "model.safetensors").write_bytes(b"weights")
    files = {
        filename: {"sha256": _sha256(artifact_dir / filename), "size_bytes": size}
        for filename, size in (("config.json", 2), ("model.safetensors", 7))
    }
    (artifact_dir / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_type": "fine_tuned_sequence_classification",
                "training_complete": True,
                "base_checkpoint": "distilbert/distilbert-base-uncased",
                "files": files,
            }
        ),
        encoding="utf-8",
    )

    manifest = training.finalize_artifact_manifest(artifact_dir)

    canonical_files = json.dumps(
        files, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    assert manifest["base_revision"] == training.BASE_REVISION
    assert manifest["artifact_fingerprint"] == hashlib.sha256(canonical_files).hexdigest()
    assert json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8")) == manifest


def test_create_artifact_archive_returns_sha256_labelled_zip(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    (artifact_dir / "config.json").write_text("{}", encoding="utf-8")
    (artifact_dir / "model.safetensors").write_bytes(b"weights")

    archive_path, checksum = training.create_artifact_archive(
        artifact_dir,
        tmp_path / "archives",
    )

    assert archive_path.exists()
    assert archive_path.suffix == ".zip"
    assert checksum == _sha256(archive_path)
    assert checksum in archive_path.stem


def test_cli_dispatches_prepare_and_train_and_validates_required_arguments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def fake_prepare(*args: object, **kwargs: object) -> dict[str, object]:
        calls.append(("prepare", args, kwargs))
        return {}

    def fake_train(*args: object, **kwargs: object) -> None:
        calls.append(("train", args, kwargs))

    def fake_init_contract(*args: object, **kwargs: object) -> dict[str, object]:
        calls.append(("init-contract", args, kwargs))
        return {}

    def fake_finalize(*args: object, **kwargs: object) -> dict[str, object]:
        calls.append(("finalize", args, kwargs))
        return {}

    monkeypatch.setattr(training, "prepare_splits", fake_prepare)
    monkeypatch.setattr(training, "train_and_export", fake_train)
    monkeypatch.setattr(training, "initialize_run_contract", fake_init_contract)
    monkeypatch.setattr(training, "finalize_artifact_manifest", fake_finalize)
    source = tmp_path / "source.csv"
    split_dir = tmp_path / "splits"
    artifact_dir = tmp_path / "artifact"
    work_dir = tmp_path / "work"

    assert (
        training.main(
            [
                "prepare",
                "--input",
                str(source),
                "--output-dir",
                str(split_dir),
                "--samples-per-class",
                "25",
            ]
        )
        == 0
    )
    assert calls.pop(0) == (
        "prepare",
        (source, split_dir),
        {"samples_per_class": 25, "seed": 42},
    )

    assert (
        training.main(
            [
                "init-contract",
                "--splits-dir",
                str(split_dir),
                "--work-dir",
                str(work_dir),
            ]
        )
        == 0
    )
    name, args, kwargs = calls.pop(0)
    assert name == "init-contract"
    assert args == (split_dir, work_dir)
    assert kwargs == {"config": training.TrainingConfig()}

    assert (
        training.main(
            [
                "finalize",
                "--artifact-dir",
                str(artifact_dir),
            ]
        )
        == 0
    )
    assert calls.pop(0) == ("finalize", (artifact_dir,), {})

    assert (
        training.main(
            [
                "train",
                "--splits-dir",
                str(split_dir),
                "--output-dir",
                str(artifact_dir),
                "--work-dir",
                str(work_dir),
                "--resume",
            ]
        )
        == 0
    )
    name, args, kwargs = calls.pop(0)
    assert name == "train"
    assert args == (split_dir, artifact_dir)
    assert kwargs["work_dir"] == work_dir
    assert kwargs["resume"] is True
    config = kwargs["config"]
    assert isinstance(config, training.TrainingConfig)
    assert config == training.TrainingConfig()

    with pytest.raises(SystemExit) as missing_prepare_args:
        training.main(["prepare"])
    assert missing_prepare_args.value.code == 2

    with pytest.raises(SystemExit) as missing_train_output:
        training.main(["train", "--splits-dir", str(split_dir)])
    assert missing_train_output.value.code == 2

    with pytest.raises(SystemExit) as invalid_sample_count:
        training.main(
            [
                "prepare",
                "--input",
                str(source),
                "--output-dir",
                str(split_dir),
                "--samples-per-class",
                "0",
            ]
        )
    assert invalid_sample_count.value.code == 2
