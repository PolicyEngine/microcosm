"""Rebuild the 09-23 release's target specs, row-aligned with the checkpoint.

The checkpoint's ``targets.json`` and ``targets_meta.parquet`` carry each
target's name, value and a parsed taxonomy, but not the ``TargetSpec``
metadata that target-loss weighting reads (``measure_mode``,
``source_measure_id``, the ledger geography, ``target_role``, the filter, the
hierarchy). The 09-23 release wrote no registry, so this recompiles one the
way its materialize stage did (``run/materialize-sup/COMMAND.json``):

* the admin families through the tool's production compile path,
  ``state_admin_surface(feed, ["snap", "medicaid", "soi"], soi_mode="state")``,
  on the release's consumer-facts feed (sha256-pinned below);
* the Census ladder populations through ``population_target_specs`` on the
  checkpoint's own ``pop_state_*`` / ``pop_cd_*`` names and values.

Then it checks that the compiled surface is the checkpoint's: the same 4,459
names (no target missing or extra, none duplicated), equal values, and the
same family and ``source_measure_id`` as ``targets.json``. It writes, beside
the checkpoint (``MANIFEST.json`` is left untouched, so earlier receipts stay
valid):

* ``target_registry.json``: the ``TargetRegistry``, spec ``i`` = CSR row ``i``;
* ``target_registry.receipt.json``: its sha256, the feed's, the compile head,
  and every check above.

Run (engine-free; reads the 164 MB feed)::

    uv run python experiments/us-acs-local-l2-basis-20260928/registry.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CHECKPOINT = Path(
    "/Users/maxghenis/PolicyEngine/_build_artifacts/acs-local-l2-basis-20260928/checkpoint"
)
RUN_0923 = Path(
    "/Users/maxghenis/PolicyEngine/_recovered/scratch-backup/893/overnight-20260923/run"
)
#: The feed, families and SOI mode the 09-23 materialize stage compiled from.
FEED = Path(
    "/Users/maxghenis/PolicyEngine/_buildh-runtime/inputs/chronicle_us_b571381/"
    "artifact/consumer_facts.jsonl"
)
FEED_SHA256 = "4d1dba8c1b6274877bf184fa6de5d99b13fc61f34709ccab1487db2b5c64a79f"
FAMILIES = ["snap", "medicaid", "soi"]
SOI_MODE = "state"
RELEASE_HEAD = "767312d60430893617af57034f62edfb4920dca3"
REGISTRY_FILENAME = "target_registry.json"
RECEIPT_FILENAME = "target_registry.receipt.json"
#: The checkpoint's values are the release's (targets_meta.parquet carries the
#: 09-23 checkpoint's targets.json values); a recompile of the same feed must
#: reproduce them bit for bit, and anything else means a different compile.
VALUE_RTOL = 0.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def compile_specs(feed: Path, population_names, population_values):
    """The admin surface and the ladder population specs, as the tool builds them."""

    sys.path.insert(0, str(REPO / "tools"))
    from build_us_acs_local_release import (
        population_target_specs,
        state_admin_surface,
    )

    surface = state_admin_surface(feed, FAMILIES, soi_mode=SOI_MODE)
    population = population_target_specs(population_names, population_values)
    return list(surface.registry.specs), list(population), surface


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--feed", type=Path, default=FEED)
    args = parser.parse_args(argv)

    from microcosm.calibrate import TargetRegistry

    started = time.time()
    checkpoint = args.checkpoint
    feed_sha = sha256(args.feed)
    if feed_sha != FEED_SHA256:
        raise SystemExit(f"feed sha256 {feed_sha} != the release's {FEED_SHA256}")
    meta = pd.read_parquet(checkpoint / "targets_meta.parquet")
    records = json.loads((checkpoint / "targets.json").read_text())
    if [r["name"] for r in records] != meta["name"].tolist():
        raise SystemExit("targets.json and targets_meta.parquet disagree on names")
    names = meta["name"].tolist()
    values = meta["value"].to_numpy(np.float64)
    is_population = meta["name"].str.startswith(("pop_state_", "pop_cd_")).to_numpy()

    admin, population, surface = compile_specs(
        args.feed,
        [names[i] for i in np.flatnonzero(is_population)],
        [float(values[i]) for i in np.flatnonzero(is_population)],
    )
    compiled = admin + population
    counts = Counter(spec.name for spec in compiled)
    duplicated = sorted(name for name, n in counts.items() if n > 1)
    by_name = {spec.name: spec for spec in compiled}
    missing = [name for name in names if name not in by_name]
    extra = sorted(set(by_name) - set(names))

    ordered = [by_name[name] for name in names if name in by_name]
    compiled_values = np.asarray([float(spec.value) for spec in ordered])
    value_check: dict = {"n_compared": len(ordered)}
    if not missing:
        relative = np.abs(compiled_values - values) / np.maximum(np.abs(values), 1.0)
        value_check.update(
            {
                "n_unequal": int((compiled_values != values).sum()),
                "max_relative_difference": float(relative.max()),
                "unequal_examples": [
                    {
                        "name": names[i],
                        "checkpoint": float(values[i]),
                        "compiled": float(compiled_values[i]),
                    }
                    for i in np.flatnonzero(relative > VALUE_RTOL)[:10]
                ],
            }
        )
    field_mismatches = []
    for record, spec in zip(records, ordered, strict=False):
        expected_family = record.get("family")
        if expected_family is not None and spec.family != expected_family:
            if not (
                record["name"].startswith("pop_")
                and expected_family == "census_population_ladder"
            ):
                field_mismatches.append(
                    {"name": record["name"], "field": "family", "spec": spec.family}
                )
        expected_measure = record.get("source_measure_id")
        actual_measure = spec.metadata.get("source_measure_id")
        if expected_measure is not None and actual_measure != expected_measure:
            field_mismatches.append(
                {
                    "name": record["name"],
                    "field": "source_measure_id",
                    "spec": actual_measure,
                    "checkpoint": expected_measure,
                }
            )
    checks = {
        "admin_specs_compiled": len(admin),
        "admin_specs_in_checkpoint": int((~is_population).sum()),
        "population_specs": len(population),
        "duplicated_names": duplicated,
        "missing_from_compile": missing[:20],
        "n_missing_from_compile": len(missing),
        "extra_in_compile": extra[:20],
        "n_extra_in_compile": len(extra),
        "values": value_check,
        "field_mismatches": field_mismatches[:20],
        "n_field_mismatches": len(field_mismatches),
    }
    ok = (
        not duplicated
        and not missing
        and not extra
        and len(admin) == int((~is_population).sum())
        and value_check.get("n_unequal", 1) == 0
        and not field_mismatches
    )
    receipt = {
        "ok": ok,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": (
            "TargetSpecs (with the metadata target-loss weighting reads) for "
            "the checkpoint's 4,459 targets, recompiled from the 09-23 "
            "release's feed; spec i is CSR row i."
        ),
        "feed": {"path": str(args.feed), "sha256": feed_sha},
        "families": FAMILIES,
        "soi_mode": SOI_MODE,
        "release_materialize_head": RELEASE_HEAD,
        "compile_head": git("rev-parse", "HEAD"),
        "compile_tree_clean": not git(
            "status", "--porcelain", "--", "tools", "packages"
        ),
        "ri_substitutions": [
            getattr(item, "__dict__", str(item)) for item in surface.ri_substitutions
        ]
        if isinstance(surface.ri_substitutions, (list, tuple))
        else str(surface.ri_substitutions),
        "soi_receipt": surface.soi_receipt,
        "checkpoint_manifest_sha256": sha256(checkpoint / "MANIFEST.json"),
        "targets_meta_sha256": sha256(checkpoint / "targets_meta.parquet"),
        "checks": checks,
        "wall_seconds": None,
    }
    if ok:
        registry = TargetRegistry(ordered, country="us")
        path = registry.to_json(checkpoint / REGISTRY_FILENAME)
        reloaded = TargetRegistry.from_json(path)
        if [spec.name for spec in reloaded.specs] != names:
            raise SystemExit("the written registry does not reload in row order")
        receipt["registry"] = {
            "path": str(path),
            "sha256": sha256(path),
            "n_specs": len(reloaded.specs),
            "family_counts": dict(Counter(spec.family for spec in reloaded.specs)),
        }
    receipt["wall_seconds"] = round(time.time() - started, 1)
    (checkpoint / RECEIPT_FILENAME).write_text(
        json.dumps(receipt, indent=1, default=str) + "\n"
    )
    print(json.dumps({"ok": ok, **checks}, indent=1, default=str)[:4000])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
