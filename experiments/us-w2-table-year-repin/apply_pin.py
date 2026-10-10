"""Point the US Chronicle feed pin at the rebuilt feed.

Usage: python apply_pin.py <worktree> <feed dir> <chronicle commit> <feed file name>

Rewrites ``us/chronicle_feed.json`` (commit, build text, row count, facts and
scope digests) and ``DEFAULT_FEED_NAME`` in
``tools/build_us_target_parity_manifest.py`` from the receipt
``tools/build_us_chronicle_feed.py`` wrote into ``<feed dir>``. The scope file
must already name the commit; the consumer-fact schema versions are checked
unchanged (the commit is c5e5bf8 plus one package file).
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

worktree, feed_dir, commit, feed_name = (
    Path(sys.argv[1]),
    Path(sys.argv[2]),
    sys.argv[3],
    sys.argv[4],
)
us = worktree / "packages/microcosm-build/src/microcosm/build/us"
scope_bytes = (us / "chronicle_feed_scope.json").read_bytes()
scope_sha256 = hashlib.sha256(scope_bytes).hexdigest()
assert json.loads(scope_bytes)["source_commit"] == commit
receipt = json.loads((feed_dir / "receipt.json").read_text())
facts = (feed_dir / "consumer_facts.jsonl").read_bytes()
facts_sha256 = hashlib.sha256(facts).hexdigest()
rows = [json.loads(line) for line in facts.splitlines() if line.strip()]
versions = sorted({row["schema_version"] for row in rows})
assert receipt["source_commit"] == commit, receipt
assert receipt["facts_sha256"] == facts_sha256, receipt
assert receipt["row_count"] == len(rows), receipt
assert receipt["scope_sha256"] == scope_sha256, receipt

pin_path = us / "chronicle_feed.json"
pin = json.loads(pin_path.read_text())
assert versions == pin["consumer_fact_schema_versions"], versions
pin["source_commit"] = commit
pin["build"] = (
    "tools/build_us_chronicle_feed.py --skip-artifact: one `chronicle build-bundle "
    "--year Y --source <package>` per build year in chronicle_feed_scope.json, rows "
    "kept for exactly the scope's (record set, period) pairs and sorted by "
    "aggregate_fact_key. The commit is c5e5bf8 plus only the W-2 item package's "
    "literal Tax Year 2020 labels (tag microcosm-us-feed-w2-ty2020-v1, the "
    "package file as chronicle#292 left it on main). Bare feed: `chronicle "
    "build-consumer-artifact` refuses it at this commit (994 concept_alignment "
    "rows lack `authority`); see docs/us-chronicle-feed-repin.md."
)
pin["fact_row_count"] = len(rows)
pin["facts_sha256"] = facts_sha256
pin["scope_sha256"] = scope_sha256
pin_path.write_text(json.dumps(pin, indent=2) + "\n")

tool_path = worktree / "tools/build_us_target_parity_manifest.py"
tool, count = re.subn(
    r'DEFAULT_FEED_NAME = "[^"]+"',
    f'DEFAULT_FEED_NAME = "{feed_name}"',
    tool_path.read_text(),
)
assert count == 1
tool_path.write_text(tool)
print(
    json.dumps(
        {
            key: pin[key]
            for key in (
                "source_commit",
                "fact_row_count",
                "facts_sha256",
                "scope_sha256",
            )
        },
        indent=1,
    )
)
