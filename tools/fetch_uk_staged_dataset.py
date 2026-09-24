"""Fetch a staged UK rowwise candidate bundle and verify its digests.

Downloads ``staged/<run_id>/`` from the private dataset repository into a
local directory, checking every file against the bundle's ``sha256sums.txt``,
and prints one local path per line. Scorecards, the evaluation legs and the
calibration dashboard's local-directory mode take those paths as they take
any run directory. ``--h5-only`` fetches just the dataset.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from microcosm.build.staging_dataset import StagedDatasetError, fetch_bundle
from microcosm.build.staging_storage import HuggingFaceDatasetStorage
from microcosm.build.uk_runtime.staging import (
    UK_STAGED_DATASET_PREFIX,
    UK_STAGED_DATASET_REPOSITORY,
)


def _hub_api():
    from huggingface_hub import HfApi

    return HfApi()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="The staged run id.")
    parser.add_argument(
        "--repo-id",
        default=UK_STAGED_DATASET_REPOSITORY.repo_id(os.environ),
        help=(
            "Private Hugging Face dataset repository (default from "
            f"{UK_STAGED_DATASET_REPOSITORY.repo_id_environment_variable})."
        ),
    )
    parser.add_argument(
        "--dest",
        type=Path,
        help="Destination directory (default: ./staged/<run_id>).",
    )
    parser.add_argument(
        "--revision",
        help="Repository revision to fetch (default: the current main branch).",
    )
    parser.add_argument(
        "--h5-only",
        action="store_true",
        help="Fetch only the dataset file(s).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    repo_id = str(args.repo_id).strip()
    if not repo_id:
        raise SystemExit("error: --repo-id must be non-empty")
    dest = args.dest or Path("staged") / args.run_id
    storage = HuggingFaceDatasetStorage(repo_id, api=_hub_api())
    try:
        paths = fetch_bundle(
            storage=storage,
            run_id=args.run_id,
            dest=dest,
            prefix=UK_STAGED_DATASET_PREFIX,
            revision=args.revision,
            h5_only=args.h5_only,
        )
    except StagedDatasetError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
