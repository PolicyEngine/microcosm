"""Seal aggregate JSON evidence locally, without release or staging publication.

Example: uv run --no-sync python tools/build_candidate_quality.py \
    --metadata candidate.json --attachments attachments.json --out candidate-quality

The attachment map contains explicit local ``source`` paths for available JSON
artifacts, resolved relative to the attachment-map file. This tool never reads
population H5 files, computes policy results, uploads, or changes defaults.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from microcosm.build.candidate_quality import write_candidate_quality


def main(argv: list[str] | None = None) -> dict[str, str]:
    """Write content-addressed evidence and optionally candidate-only discovery."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--attachments", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-index", action="store_true")
    args = parser.parse_args(argv)
    metadata = json.loads(args.metadata.read_text())
    attachments = json.loads(args.attachments.read_text())
    for item in attachments.values():
        if item.get("status") == "available" and "source" in item:
            source = Path(item["source"])
            item["source"] = str((args.attachments.parent / source).resolve())
    reference = write_candidate_quality(
        metadata, attachments, args.out, update_index=not args.no_index
    )
    print(json.dumps(reference, sort_keys=True))
    return reference


if __name__ == "__main__":
    main()
