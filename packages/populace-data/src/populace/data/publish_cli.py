"""CLI for contract-gated Populace release publication."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from populace.data.release import publish_release


def _staging_undelivered(release_dir: Path) -> bool:
    """True if a build that should have staged has nothing to show for it.

    Scoped by the presence of the ``staging`` key rather than by country or
    dataset role: a builder that stages writes the key on every build, and one
    with no staging path writes none. The local-area product and any future
    build without staging are therefore untouched, with no list of exceptions
    to maintain.

    Refused, because the release never reached the staging dashboard and its
    manifest cannot show otherwise:

    - the key is present and falsy (``null``, the pre-#270 shape, or a build
      whose staging destination went away)
    - staging was enabled but nothing was ever uploaded, which is what a run
      without a write token looks like once uploads self-disable

    Allowed: no key at all, and a declared ``enabled: False`` opt-out. Skipping
    staging on purpose is a legitimate way to build, and saying so is exactly
    what makes it distinguishable from the failures above.
    """
    path = release_dir / "build_manifest.json"
    if not path.exists():
        return False
    try:
        manifest = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    if not isinstance(manifest, dict) or "staging" not in manifest:
        return False
    staging = manifest["staging"]
    if not isinstance(staging, dict) or not staging:
        return True
    if staging.get("enabled") is False:
        return False
    return not staging.get("uploads_succeeded")


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
        help="Create a Hugging Face tag for the release before updating latest.json.",
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
        help="Additional release-dir file to upload before latest.json.",
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
        "--allow-incomplete-reform-validation",
        action="store_true",
        help=(
            "Publish even if reform_validation.json was built with out-of-sample "
            "reforms skipped (out_of_sample_simulated false). Off by default so a "
            "release never silently ships blank OBBBA validation."
        ),
    )
    parser.add_argument(
        "--allow-missing-staging",
        action="store_true",
        help=(
            "Publish even if the build recorded no staging telemetry, or "
            "recorded staging that never uploaded anything. Off by default so "
            "a release that never appeared on the staging dashboard is not "
            "shipped unnoticed. A declared --no-staging build publishes "
            "without this flag."
        ),
    )
    args = parser.parse_args(argv)

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

    if not args.allow_missing_staging and _staging_undelivered(Path(args.release_dir)):
        print(
            "refusing to publish: this release's build_manifest records no "
            "delivered staging telemetry, so the build never appeared on the "
            "staging dashboard and there is no pre-publication review of it. "
            "Either the staging destination was lost mid-build (uploads "
            "self-disable after repeated failures — check the build machine's "
            "Hugging Face write token), or the manifest predates staging "
            "provenance. Rebuild with staging reaching its repo, or pass "
            "--allow-missing-staging to publish anyway. A build that declared "
            "--no-staging publishes without the flag.",
            file=sys.stderr,
        )
        return 1

    pointer = publish_release(
        Path(args.release_dir),
        args.repo_id,
        artifact_root=Path(args.artifact_root) if args.artifact_root else None,
        create_tag=args.create_tag,
        tag_name=args.tag_name,
        extra_files=tuple(args.extra_file),
        updated_at=args.updated_at,
        update_latest=not args.no_latest,
    )
    print(json.dumps(pointer, indent=2))

    # The Slack release alert now fires inside publish_release (coupled to the
    # promotion, so every publish path announces the release), warning loudly if
    # the webhook is unset. Nothing to do here.
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
