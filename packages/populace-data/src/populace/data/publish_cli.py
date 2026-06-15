"""CLI for contract-gated Populace release publication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from populace.data.release import publish_release


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
        help="Create a Hugging Face tag for the release before updating latest.json.",
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
    args = parser.parse_args(argv)

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
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
