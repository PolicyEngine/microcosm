"""CLI for contract-gated Populace release publication."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from populace.data.release import publish_release
from populace.data.slack import notify_release


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
        "--allow-incomplete-reform-validation",
        action="store_true",
        help=(
            "Publish even if reform_validation.json was built with out-of-sample "
            "reforms skipped (out_of_sample_simulated false). Off by default so a "
            "release never silently ships blank OBBBA validation."
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

    pointer = publish_release(
        Path(args.release_dir),
        args.repo_id,
        artifact_root=Path(args.artifact_root) if args.artifact_root else None,
        create_tag=args.create_tag,
        tag_name=args.tag_name,
        extra_files=tuple(args.extra_file),
        updated_at=args.updated_at,
    )
    print(json.dumps(pointer, indent=2))

    # Real-time release alert, fired the moment latest.json is live. No-op
    # unless the country's SLACK_WEBHOOK_POPULACE_* env var is set, and never
    # fatal — a Slack failure must not fail an otherwise-successful publish.
    notify_release(
        args.repo_id,
        str(pointer.get("release_id", "")),
        pointer.get("updated_at"),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
