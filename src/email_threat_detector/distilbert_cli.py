"""Command-line routing for DistilBERT preparation and training."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

BASE_CHECKPOINT = "distilbert/distilbert-base-uncased"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _add_training_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-checkpoint", default=BASE_CHECKPOINT)
    parser.add_argument("--max-length", type=_positive_int, default=128)
    parser.add_argument("--epochs", type=_positive_int, default=3)
    parser.add_argument("--batch-size", type=_positive_int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="threatlens-distilbert",
        description="Prepare data and fine-tune the optional ThreatLens DistilBERT model.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Create leakage-safe 80/10/10 CSV splits.")
    prepare.add_argument("--input", required=True, type=Path)
    prepare.add_argument("--output-dir", required=True, type=Path)
    prepare.add_argument("--samples-per-class", type=_positive_int)
    prepare.add_argument("--seed", type=int, default=42)

    adopt = commands.add_parser(
        "adopt",
        help="Validate and checksum an existing legacy leakage-safe split directory.",
    )
    adopt.add_argument("--splits-dir", required=True, type=Path)

    archive = commands.add_parser(
        "archive",
        help="Create a checksum-labelled ZIP from a completed local artifact.",
    )
    archive.add_argument("--artifact-dir", required=True, type=Path)
    archive.add_argument("--output-dir", required=True, type=Path)

    finalize = commands.add_parser(
        "finalize",
        help="Add pinned provenance and a fingerprint to a completed legacy export.",
    )
    finalize.add_argument("--artifact-dir", required=True, type=Path)

    initialize = commands.add_parser(
        "init-contract",
        help="Initialize or verify the immutable contract for a training work directory.",
    )
    initialize.add_argument("--splits-dir", required=True, type=Path)
    initialize.add_argument("--work-dir", required=True, type=Path)
    _add_training_config_arguments(initialize)

    train = commands.add_parser("train", help="Fine-tune and export DistilBERT.")
    train.add_argument("--splits-dir", required=True, type=Path)
    train.add_argument("--output-dir", required=True, type=Path)
    train.add_argument("--work-dir", required=True, type=Path)
    train.add_argument("--resume", action="store_true")
    _add_training_config_arguments(train)
    return parser


def _training_config(api: Any, args: argparse.Namespace) -> Any:
    return api.TrainingConfig(
        base_checkpoint=args.base_checkpoint,
        max_length=args.max_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )


def main(argv: list[str] | None = None, *, api: Any) -> int:
    """Parse arguments and dispatch through the supplied training API module."""
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        api.prepare_splits(
            args.input,
            args.output_dir,
            samples_per_class=args.samples_per_class,
            seed=args.seed,
        )
        return 0
    if args.command == "adopt":
        api.adopt_prepared_splits(args.splits_dir)
        return 0
    if args.command == "archive":
        api.create_artifact_archive(args.artifact_dir, args.output_dir)
        return 0
    if args.command == "finalize":
        api.finalize_artifact_manifest(args.artifact_dir)
        return 0

    config = _training_config(api, args)
    if args.command == "init-contract":
        api.initialize_run_contract(args.splits_dir, args.work_dir, config=config)
        return 0
    api.train_and_export(
        args.splits_dir,
        args.output_dir,
        work_dir=args.work_dir,
        config=config,
        resume=args.resume,
    )
    return 0
