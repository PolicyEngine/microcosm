"""Count the restated-filter guard's refusals with the rule and with it reverted.

For each Chronicle consumer-facts feed given, compiles the US fiscal target
registry and surveys three surfaces — the whole compiled registry and the
state-level surface ``tools/build_us_acs_local_release.py::state_admin_specs``
selects in both SOI modes — under two arms:

* ``rule``: the tree as it stands, ``_restated_ledger_filter_refusal`` at both
  call sites;
* ``reverted``: ``_restated_ledger_filter_refusal`` replaced by a function that
  returns the bare key for every otherwise-unsupported ``ledger_filter_*`` key.
  That is the pre-rule behaviour of both ``_unsupported_ledger_filter_metadata``
  and ``_unsupported_soi_ledger_filters``, and so ``main``'s before this branch.

Per surface and arm it reports refused targets, refusal entries by key,
refused targets by (family, ``target_role``), and ``irs_soi`` targets the SOI
loop would skip silently. Output is aggregate target-spec counts only; no
microdata is read. It regenerates ``restated_filter_arms_receipt.json``::

    uv run python experiments/us-labelled-filter-support/\\
        measure_restated_filter_arms.py <consumer_facts.jsonl> [...] > receipt.json

Needs no engine. Each surface is compiled through the production path, so a
feed is compiled three times; the receipt records the wall time per feed
(``compile_and_select_seconds``) and the process's peak RSS
(``max_rss_bytes``, bytes on macOS). It also records the SHA-256 of this
script and the ``git status --porcelain`` of the tree it ran in, so a receipt
can be matched to the committed script that produced it.
"""

from __future__ import annotations

import collections
import hashlib
import importlib.util
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_tool(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _survey(builder, specs: list, label: str) -> dict[str, object]:
    unsupported = builder._unsupported_ledger_filter_metadata(specs)
    entries: collections.Counter[str] = collections.Counter()
    for spec_entries in unsupported.values():
        for entry in spec_entries:
            entries[entry.split("=", 1)[0]] += 1
    by_name = {spec.name: spec for spec in specs}
    by_role: collections.Counter[str] = collections.Counter()
    for name in unsupported:
        spec = by_name[name]
        by_role[f"{spec.family}|{spec.metadata.get('target_role', '')}"] += 1
    skipped = sum(
        1
        for spec in specs
        if getattr(spec, "family", "") == "irs_soi"
        and builder._unsupported_soi_ledger_filters(spec.metadata)
    )
    return {
        "label": label,
        "targets": len(specs),
        "refused_targets": len(unsupported),
        "refusal_entries": sum(entries.values()),
        "refusal_entries_by_key": dict(entries.most_common()),
        "refused_by_family_role": dict(by_role.most_common()),
        "soi_silently_skipped_targets": skipped,
    }


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2

    builder = _load_tool("build_us_fiscal_refresh_release")
    acs = _load_tool("build_us_acs_local_release")

    import microcosm.build.ledger_targets as ledger_targets
    from microcosm.build.ledger_artifact import load_ledger_consumer_artifact
    from microcosm.build.us_runtime import (
        default_congressional_district_vintage_crosswalk_path,
        load_congressional_district_vintage_crosswalk,
    )
    from microcosm.build.us_runtime.fiscal_targets import (
        compile_us_fiscal_target_registry,
    )

    # The compile must be this checkout's, not another installed tree's.
    assert Path(ledger_targets.__file__).resolve().is_relative_to(ROOT), (
        ledger_targets.__file__
    )

    rule = builder._restated_ledger_filter_refusal

    def reverted(key: str, value: str, metadata: object) -> str:
        return key

    results = []
    for feed in argv:
        started = time.time()
        artifact = load_ledger_consumer_artifact(feed)
        registry = compile_us_fiscal_target_registry(
            artifact.facts,
            target_period=acs.PERIOD,
            congressional_district_vintage_crosswalk=(
                load_congressional_district_vintage_crosswalk(
                    default_congressional_district_vintage_crosswalk_path()
                )
            ),
            age_targets=True,
        )
        surfaces = {"whole_registry": list(registry.specs)}
        del artifact, registry
        for soi_mode in ("full", "totals"):
            selected, _substitutions = acs.state_admin_specs(
                feed, ["snap", "medicaid", "soi"], soi_mode=soi_mode
            )
            surfaces[f"state_surface_{soi_mode}"] = list(selected.specs)
        result: dict[str, object] = {
            "feed": feed,
            "feed_sha256": _sha256(feed),
            "compile_and_select_seconds": round(time.time() - started, 1),
        }
        try:
            for arm, function in (("rule", rule), ("reverted", reverted)):
                builder._restated_ledger_filter_refusal = function
                result[arm] = [
                    _survey(builder, specs, label) for label, specs in surfaces.items()
                ]
        finally:
            builder._restated_ledger_filter_refusal = rule
        results.append(result)
        print(f"done {feed}", file=sys.stderr, flush=True)

    head = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.splitlines()
    payload = {
        "head": head,
        "worktree_status": status,
        "script_sha256": _sha256(__file__),
        "target_period": acs.PERIOD,
        "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "results": results,
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
