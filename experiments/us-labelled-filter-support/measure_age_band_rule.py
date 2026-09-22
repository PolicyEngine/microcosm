"""Count the restated-filter guard's refusals with and without the age-band rule.

Sibling of ``measure_restated_filter_arms.py`` for the follow-up that adds
``age`` to ``RESTATED_LEDGER_FILTER_CONCEPTS`` under the ``age_band`` rule.
For each Chronicle consumer-facts feed given, compiles the whole US fiscal
target registry the way that script does
(``compile_us_fiscal_target_registry(..., age_targets=True)`` with the
default congressional-district vintage crosswalk) and surveys it under three
arms:

* ``rule``: the tree as it stands (AGI, qualifying-child and age-band rules);
* ``without_age_band``: ``age`` removed from
  ``RESTATED_LEDGER_FILTER_CONCEPTS``, which is #969's rule
  (``us-labelled-filter-support``) and nothing else;
* ``reverted``: ``_restated_ledger_filter_refusal`` replaced by a function
  that returns the bare key for every otherwise-unsupported
  ``ledger_filter_*`` key, the pre-rule behaviour of both call sites and so
  ``main``'s.

Per arm it reports refused targets, refusal entries by key, refused targets by
(family, ``target_role``), and ``irs_soi`` targets the SOI loop would skip
silently. It also counts, independent of arm, the targets carrying a restated
age key (``ledger_filter_age_lower_bound`` / ``_upper_bound``) by
(family, ``target_role``, ``materializer``), and every ``ledger_filter_age*``
key seen, so an operator-changed key (``_inclusive`` / ``_exclusive``) or an
``age`` dimension would show up by name. Output is aggregate target-spec
counts only; no microdata is read. It regenerates
``age_band_rule_receipt.json``::

    uv run python experiments/us-labelled-filter-support/\\
        measure_age_band_rule.py <consumer_facts.jsonl> [...] > receipt.json

Needs no engine. The receipt records the SHA-256 of this script and of each
feed, the ``git rev-parse HEAD`` and ``git status --porcelain`` of the tree
it ran in, the compile time per feed and the process's peak RSS
(``max_rss_bytes``, bytes on macOS).
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
RESTATED_AGE_KEYS = ("ledger_filter_age_lower_bound", "ledger_filter_age_upper_bound")


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


def _survey(builder, specs: list) -> dict[str, object]:
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
        "refused_targets": len(unsupported),
        "refusal_entries": sum(entries.values()),
        "refusal_entries_by_key": dict(entries.most_common()),
        "refused_by_family_role": dict(by_role.most_common()),
        "soi_silently_skipped_targets": skipped,
    }


def _age_key_census(specs: list) -> dict[str, object]:
    carrying: collections.Counter[str] = collections.Counter()
    keys: collections.Counter[str] = collections.Counter()
    for spec in specs:
        for key in spec.metadata:
            if str(key).startswith("ledger_filter_age"):
                keys[str(key)] += 1
        if any(key in spec.metadata for key in RESTATED_AGE_KEYS):
            carrying[
                f"{spec.family}|{spec.metadata.get('target_role', '')}"
                f"|{spec.metadata.get('materializer', '')}"
            ] += 1
    return {
        "targets_carrying_restated_age_key": sum(carrying.values()),
        "by_family_role_materializer": dict(carrying.most_common()),
        "ledger_filter_age_keys": dict(keys.most_common()),
    }


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2

    builder = _load_tool("build_us_fiscal_refresh_release")

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
    assert builder.RESTATED_LEDGER_FILTER_CONCEPTS.get("age") == "age_band"

    rule = builder._restated_ledger_filter_refusal
    concepts = dict(builder.RESTATED_LEDGER_FILTER_CONCEPTS)
    without_age = {
        concept: name for concept, name in concepts.items() if concept != "age"
    }

    def reverted(key: str, value: str, metadata: object) -> str:
        return key

    arms = (
        ("rule", rule, concepts),
        ("without_age_band", rule, without_age),
        ("reverted", reverted, concepts),
    )
    crosswalk = load_congressional_district_vintage_crosswalk(
        default_congressional_district_vintage_crosswalk_path()
    )

    results = []
    for feed in argv:
        started = time.time()
        artifact = load_ledger_consumer_artifact(feed)
        registry = compile_us_fiscal_target_registry(
            artifact.facts,
            target_period=builder.PERIOD,
            congressional_district_vintage_crosswalk=crosswalk,
            age_targets=True,
        )
        specs = list(registry.specs)
        del artifact, registry
        result: dict[str, object] = {
            "feed": feed,
            "feed_sha256": _sha256(feed),
            "compile_seconds": round(time.time() - started, 1),
            "targets": len(specs),
            "age_key_census": _age_key_census(specs),
        }
        try:
            for arm, function, arm_concepts in arms:
                builder._restated_ledger_filter_refusal = function
                builder.RESTATED_LEDGER_FILTER_CONCEPTS = arm_concepts
                result[arm] = _survey(builder, specs)
        finally:
            builder._restated_ledger_filter_refusal = rule
            builder.RESTATED_LEDGER_FILTER_CONCEPTS = concepts
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
        "target_period": builder.PERIOD,
        "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "results": results,
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
