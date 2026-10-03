"""Upload a finished US staging run folder to the staging repository.

A build that could not upload (no write token, uploads that failed, or a
``--staging-dir``-only run) keeps its telemetry in
``<release_root>/staging/runs/<run_id>/``. Run this from any machine with a
token that can write the staging repository to put the run on the staging
dashboard:

    uv run python tools/restage_us_staging_run.py --run-dir <run folder>

It uploads the run files and the artifacts the run manifest lists, and adds
the run to ``runs.json``. It never moves ``latest_staging.json``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from microcosm.build.staging import DEFAULT_STAGING_PREFIX, restage_run

DEFAULT_REPO_ID = "policyengine/populace-us-staging"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--repo-id",
        default=os.environ.get("POPULACE_STAGING_REPO_ID") or DEFAULT_REPO_ID,
        help="Staging repository. Defaults to POPULACE_STAGING_REPO_ID or %(default)s.",
    )
    parser.add_argument("--path-prefix", default=DEFAULT_STAGING_PREFIX)
    parser.add_argument(
        "--no-index",
        action="store_true",
        help="Do not add the run to runs.json (an exact-count run, #578).",
    )
    args = parser.parse_args(argv)
    if not (args.run_dir / "run_manifest.json").is_file():
        parser.error(f"{args.run_dir} has no run_manifest.json.")
    written = restage_run(
        args.run_dir,
        repo_id=args.repo_id,
        path_prefix=args.path_prefix,
        update_index=not args.no_index,
    )
    print(f"Uploaded {len(written)} file(s) to {args.repo_id}.")


if __name__ == "__main__":
    main()
