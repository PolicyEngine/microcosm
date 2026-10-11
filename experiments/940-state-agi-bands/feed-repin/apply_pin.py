"""Point the US Chronicle feed pin at a rebuilt feed (#940).

Usage: python apply_pin.py <worktree> <feed dir> <chronicle commit> <feed file name>
Rewrites us/chronicle_feed.json (commit, build text, row count, facts and
scope digests) and tools/build_us_target_parity_manifest.py's
DEFAULT_FEED_NAME. The scope file must already name the commit; the schema
version and its digest are checked unchanged (the commit is c5e5bf8-based).
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
assert json.loads(scope_bytes)["source_commit"] == commit
receipt = json.loads((feed_dir / "receipt.json").read_text())
facts = (feed_dir / "consumer_facts.jsonl").read_bytes()
facts_sha = hashlib.sha256(facts).hexdigest()
rows = [json.loads(line) for line in facts.splitlines() if line.strip()]
versions = sorted({row["schema_version"] for row in rows})
assert receipt["facts_sha256"] == facts_sha, receipt
assert receipt["row_count"] == len(rows), receipt
assert receipt["scope_sha256"] == hashlib.sha256(scope_bytes).hexdigest(), receipt

pin_path = us / "chronicle_feed.json"
pin = json.loads(pin_path.read_text())
assert versions == pin["consumer_fact_schema_versions"], versions
pin["source_commit"] = commit
pin["build"] = (
    "tools/build_us_chronicle_feed.py --skip-artifact: one `chronicle build-bundle "
    "--year Y --source <package>` per build year in chronicle_feed_scope.json, rows "
    "kept for exactly the scope's (record set, period) pairs and sorted by "
    "aggregate_fact_key. The commit is c5e5bf8 plus only the TY2023 Historic "
    "Table 2 state AGI-band package (tag microcosm-us-feed-940-v1, chronicle#291). "
    "Bare feed: `chronicle build-consumer-artifact` refuses it at this commit; see "
    "docs/us-chronicle-feed-repin.md."
)
pin["fact_row_count"] = len(rows)
pin["facts_sha256"] = facts_sha
pin["scope_sha256"] = hashlib.sha256(scope_bytes).hexdigest()
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
            k: pin[k]
            for k in ("source_commit", "fact_row_count", "facts_sha256", "scope_sha256")
        },
        indent=1,
    )
)
