"""CLI for contract-gated Microcosm release publication."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from microcosm.data.release import publish_release


def _staging_missing(release_dir: Path) -> bool:
    """True if the release's build manifest records no staging telemetry.

    Releases are expected to publish staging runs while building (the builder
    now stages by default); a missing block means the build predates that or
    was run with --no-staging. Non-fatal — surfaced as a warning at publish.
    """
    path = release_dir / "build_manifest.json"
    if not path.exists():
        return False
    try:
        manifest = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    return not manifest.get("staging")


def _reform_validation_skipped(release_dir: Path) -> bool:
    """True if the release carries a reform_validation.json that was built with
    out-of-sample reforms skipped (``out_of_sample_simulated`` false).

    Such a release publishes blank out-of-sample (OBBBA) rows on the dashboard,
    indistinguishable from a real result — so publishing one is refused unless
    explicitly allowed. Absent/unreadable file or a true flag → not skipped.
    """
    path = release_dir / "reform_validation.json"
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    return payload.get("out_of_sample_simulated") is False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release_dir", help="Local releases/<release_id> directory.")
    parser.add_argument(
        "--repo-id",
        default="policyengine/populace-us",
        help="Hugging Face dataset repo id.",
    )
    parser.add_argument(
        "--artifact-root",
        help=(
            "Directory holding root artifacts named by release_manifest.json, "
            "for example populace_us_2024.h5."
        ),
    )
    parser.add_argument(
        "--create-tag",
        action="store_true",
        default=True,
        help=(
            "Create a Hugging Face tag for the immutable release before any "
            "main-branch update."
        ),
    )
    parser.add_argument(
        "--no-create-tag",
        action="store_false",
        dest="create_tag",
        help=(
            "Skip creating a Hugging Face tag. Refused when the release "
            "manifest pins artifacts to the release id."
        ),
    )
    parser.add_argument(
        "--tag-name",
        help="Optional tag name override; defaults to the release id.",
    )
    parser.add_argument(
        "--extra-file",
        action="append",
        default=[],
        help="Additional release-dir file to include in the immutable release.",
    )
    parser.add_argument(
        "--updated-at",
        help="Optional latest.json timestamp override for reproducible tests.",
    )
    parser.add_argument(
        "--no-latest",
        action="store_true",
        help=(
            "Publish as a non-default release: upload files and create the "
            "immutable tag, but never touch latest.json (the default pointer)."
        ),
    )
    parser.add_argument(
        "--tag-only",
        action="store_true",
        help=(
            "Publish only the immutable tag revision, without any main-branch "
            "commit. Requires --no-latest and tag creation; used by exact-k "
            "ladder candidates."
        ),
    )
    parser.add_argument(
        "--allow-incomplete-reform-validation",
        action="store_true",
        help=(
            "Publish even if reform_validation.json was built with out-of-sample "
            "reforms skipped (out_of_sample_simulated false). Off by default so a "
            "release never silently ships blank OBBBA validation."
        ),
    )
    parser.add_argument(
        "--evidence",
        action="store_true",
        help=(
            "Publish at the EVIDENCE tier (microcosm#506): validate against the "
            "evidence release contract (which requires a non-empty "
            "known_failures block) and move latest-evidence.json instead of "
            "latest.json. Structurally never touches the certified pointer; a "
            "certified-shape release is refused under this flag."
        ),
    )
    args = parser.parse_args(argv)

    if args.tag_only and not args.no_latest:
        parser.error("--tag-only requires --no-latest.")
    if args.tag_only and not args.create_tag:
        parser.error("--tag-only requires tag creation; remove --no-create-tag.")

    if not args.allow_incomplete_reform_validation and _reform_validation_skipped(
        Path(args.release_dir)
    ):
        print(
            "refusing to publish: reform_validation.json has "
            "out_of_sample_simulated=false (built with "
            "--skip-out-of-sample-reforms), so the dashboard would show blank "
            "out-of-sample reforms. Rebuild without skipping, or pass "
            "--allow-incomplete-reform-validation to publish anyway.",
            file=sys.stderr,
        )
        return 1

    if _staging_missing(Path(args.release_dir)):
        print(
            "warning: this release's build_manifest records no staging "
            "telemetry — the build ran with --no-staging or predates "
            "staging-by-default, so it will not appear on the staging "
            "dashboard.",
            file=sys.stderr,
        )

    pointer = publish_release(
        Path(args.release_dir),
        args.repo_id,
        artifact_root=Path(args.artifact_root) if args.artifact_root else None,
        create_tag=args.create_tag,
        tag_name=args.tag_name,
        extra_files=tuple(args.extra_file),
        updated_at=args.updated_at,
        update_latest=not args.no_latest,
        tag_only=args.tag_only,
        evidence=args.evidence,
    )
    print(json.dumps(pointer, indent=2))

    # The Slack release alert now fires inside publish_release (coupled to the
    # promotion, so every publish path announces the release), warning loudly if
    # the webhook is unset. Nothing to do here.
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
