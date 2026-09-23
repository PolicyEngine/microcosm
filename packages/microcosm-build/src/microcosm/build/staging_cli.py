"""Country-neutral command-line options for staging telemetry."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

from microcosm.build.staging_storage import StagingRepositoryConfig

__all__ = [
    "add_staged_dataset_arguments",
    "add_staging_arguments",
    "validate_staged_dataset_arguments",
    "validate_staging_arguments",
]


def add_staging_arguments(
    parser: argparse.ArgumentParser,
    *,
    repository: StagingRepositoryConfig,
    default_upload_interval_seconds: float = 30.0,
) -> None:
    """Add staging controls using defaults supplied by a country module.

    ``default_upload_interval_seconds`` lets a long-running command choose a
    slower best-effort upload cadence: the Hub allows about 128 commits per
    hour per repository and every telemetry cycle is up to eight single-file
    commits, so a multi-hour solve at the 30-second default exhausts the
    budget and loses uploads (the UK rowwise driver runs at 300).
    """

    parser.add_argument(
        "--staging-dir",
        type=Path,
        help="Local root for version 2 staging files; defaults beside the output.",
    )
    parser.add_argument(
        "--staging-repo-id",
        default=repository.repo_id(os.environ),
        help=(
            "Access-controlled Hugging Face dataset repository for best-effort "
            "telemetry delivery."
        ),
    )
    parser.add_argument(
        "--staging-run-id",
        help="Override the staging run identifier.",
    )
    parser.add_argument(
        "--staging-candidate-id",
        help="Override the staging candidate identifier.",
    )
    parser.add_argument(
        "--staging-upload-interval-seconds",
        type=float,
        default=float(default_upload_interval_seconds),
        help="Minimum interval between best-effort progress uploads.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--staging-local-only",
        action="store_true",
        help="Write and validate staging files locally without a remote client.",
    )
    mode.add_argument(
        "--no-staging",
        action="store_true",
        help="Deliberately disable staging telemetry for this build.",
    )
    parser.add_argument(
        "--staging-read-back",
        action="store_true",
        help="After final upload, authenticate and validate the remote core files.",
    )


def validate_staging_arguments(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    """Reject contradictory or incomplete staging configuration."""

    interval = args.staging_upload_interval_seconds
    if not math.isfinite(interval) or interval < 0.0:
        parser.error(
            "--staging-upload-interval-seconds must be finite and non-negative."
        )
    if args.staging_read_back and (args.staging_local_only or args.no_staging):
        parser.error("--staging-read-back requires remote staging.")
    if not args.staging_local_only and not args.no_staging:
        if (
            not isinstance(args.staging_repo_id, str)
            or not args.staging_repo_id.strip()
        ):
            parser.error("remote staging requires a non-empty --staging-repo-id.")


def add_staged_dataset_arguments(
    parser: argparse.ArgumentParser,
    *,
    repository: StagingRepositoryConfig,
) -> None:
    """Add the staged-dataset options beside the shared staging options.

    The finished bundle follows the staging mode switch: ``--no-staging``
    keeps nothing, ``--staging-local-only`` keeps the bundle and its sidecars
    on disk, and the default uploads it to ``repository`` under
    ``staged/<run_id>/``. ``--no-staged-dataset`` runs telemetry alone: the
    bundle is neither inventoried nor uploaded.
    """

    parser.add_argument(
        "--staged-dataset-repo-id",
        default=repository.repo_id(os.environ),
        help=(
            "Private Hugging Face dataset repository that receives the finished "
            "run bundle under staged/<run_id>/ (default from "
            f"{repository.repo_id_environment_variable}, else "
            f"{repository.default_repo_id})."
        ),
    )
    parser.add_argument(
        "--no-staged-dataset",
        action="store_true",
        help=(
            "Run staging telemetry alone: the finished bundle is neither "
            "inventoried nor uploaded (--no-staging already disables both)."
        ),
    )


def validate_staged_dataset_arguments(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    """Reject staged-dataset options that contradict the staging mode."""

    if args.no_staged_dataset and args.no_staging:
        parser.error(
            "--no-staging already disables the staged dataset; "
            "drop --no-staged-dataset."
        )
    remote = not (args.no_staging or args.staging_local_only or args.no_staged_dataset)
    if remote and not str(args.staged_dataset_repo_id or "").strip():
        parser.error(
            "remote dataset staging requires a non-empty --staged-dataset-repo-id."
        )
