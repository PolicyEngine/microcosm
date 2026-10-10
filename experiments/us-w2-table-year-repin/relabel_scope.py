"""Move the US Chronicle feed scope's W-2 item pairs to the tax year IRS published.

Usage: python relabel_scope.py <scope.json> <chronicle commit>
Idempotent: re-running with the same commit rewrites the same bytes.

The scope built ``soi-w2-statistics-2020`` at ``--year 2023`` for three record
sets. That package reads one pinned workbook, IRS SOI Table 4.B for Tax Year
2020 (``20in04w2all.xlsx``), so those builds stamped TY2020 cells as ty2023.
At the commit this script is given the package's labels are literal 2020, so
the three ty2023 pairs go and the two TY2020 record sets the scope lacked
(401(k) elective deferrals, designated Roth contributions) come in. The TY2020
tips pair was already in the scope.
"""

from __future__ import annotations

import json
import sys

PACKAGE = "packages/irs_soi/w2_statistics_2020"
PACKAGE_ID = "soi-w2-statistics-2020"
ITEM_RECORD_SETS = (
    "form_w2_401k_elective_deferrals",
    "form_w2_designated_roth_401k_contributions",
    "form_w2_social_security_tips",
)
TABLE_TAX_YEAR = 2020
RULE = (
    "The (record_set_id, period) pairs of the pinned feed "
    "consumer_facts_buildn_v9_4.jsonl, with the three W-2 item record sets it "
    "stamped ty2023 replaced by the TY2020 record sets of the table they are "
    "read from (IRS SOI Table 4.B, Tax Year 2020; IRS publishes no later "
    "year); each built from the named package with "
    "`chronicle build-bundle --year <build_year> --source <package>`."
)

path, commit = sys.argv[1], sys.argv[2]
assert len(commit) == 40, commit
scope = json.load(open(path))
pairs = [pair for pair in scope["pairs"] if pair["package"] != PACKAGE]
dropped = [pair for pair in scope["pairs"] if pair["package"] == PACKAGE]
assert {pair["package_id"] for pair in dropped} == {PACKAGE_ID}, dropped
for record_set in ITEM_RECORD_SETS:
    pairs.append(
        {
            "record_set_id": f"irs_soi.ty{TABLE_TAX_YEAR}.{record_set}",
            "period_type": "tax_year",
            "period_value": str(TABLE_TAX_YEAR),
            "package_id": PACKAGE_ID,
            "package": PACKAGE,
            "build_year": TABLE_TAX_YEAR,
        }
    )
pairs.sort(key=lambda e: (e["record_set_id"], e["period_type"], e["period_value"]))
runs: dict[str, set[str]] = {}
for pair in pairs:
    runs.setdefault(str(pair["build_year"]), set()).add(pair["package"])
scope["source_commit"] = commit
scope["rule"] = RULE
scope["pair_count"] = len(pairs)
scope["runs"] = {year: sorted(runs[year]) for year in sorted(runs)}
scope["pairs"] = pairs
with open(path, "w") as handle:
    handle.write(json.dumps(scope, indent=1) + "\n")
print(len(pairs), sum(len(v) for v in scope["runs"].values()))
