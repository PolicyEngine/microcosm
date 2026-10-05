"""How the ACS local release's loss weight moves under the national weighting.

Two real surfaces, each scored three ways:

* ``equal``: every training target weighted 1, the calibrate stage before
  this change;
* ``national_as_is``: the national release's row mapping applied to the ACS
  local specs unchanged (what a bare reuse of the national helper would do);
* ``acs_local``: the shared formula under the ACS local row mapping, which is
  what the calibrate stage now passes to ``calibrate``.

Surfaces:

1. ``release_20260923``: the 4,459 targets of the published 09-23 release
   (``--soi-mode state``, the default), every one trained. Its specs are the
   ``target_registry.json`` rebuilt from that release's feed by
   ``experiments/us-acs-local-l2-basis-20260928/registry.py`` on branch
   ``acs-local-weighted-lambda-frontier`` (microcosm#1105, commit 430491e2c).
   Its receipt, copied here as ``registry_20260923.receipt.json``, records
   that all 4,459 names and values equal the checkpoint's.
2. ``state_cd_pinned_feed``: ``--soi-mode state_cd`` compiled by the tool's
   own ``state_admin_surface`` from the feed ``chronicle_feed.json`` pins,
   plus the 09-23 release's 487 ladder population specs, with the default 10%
   district holdout assigned by the tool's ``assign_target_roles``; only the
   training rows are weighted, as in the calibrate stage.
3. ``soi_modes``: a classification receipt for every ``--soi-mode``
   (``state``, ``totals``, ``full``, ``state_cd``) compiled from the pinned
   feed with the same population specs, before any holdout. For each it
   records that the ACS local mapping classifies every row, how many rows
   take a basis override, the concept groups, and that a per-district key
   injected into every district row is refused wherever district ledger
   rows exist.

Engine-free. Run from the repository root::

    uv run python experiments/us-acs-local-target-loss-weights-20261004/distribution.py

It writes ``results.json`` beside this file and prints the tables the README
quotes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TOOLS = REPO / "tools"
DEFAULT_REGISTRY = Path(
    "/Users/maxghenis/PolicyEngine/_build_artifacts/acs-local-l2-basis-20260928/"
    "checkpoint/target_registry.json"
)
#: sha256 of the rebuilt 09-23 registry, from its receipt.
REGISTRY_SHA256 = "fd25c6002cae0c991842c81399d43a3840f12cba0294021ca8ff6718e22a209f"
DEFAULT_FEED = Path(
    "/Users/maxghenis/PolicyEngine/_buildh-runtime/inputs/consumer_facts_us_c5e5bf8.jsonl"
)
FAMILIES = ["snap", "medicaid", "soi"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def load_tool():
    sys.path.insert(0, str(TOOLS))
    path = TOOLS / "build_us_acs_local_release.py"
    spec = importlib.util.spec_from_file_location("build_us_acs_local_release", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def score(specs, lw) -> dict:
    """The three weightings of one training surface, with their shares."""

    acs = lw.US_ACS_LOCAL_TARGET_LOSS_ROW_MAPPING
    national = lw.US_FISCAL_TARGET_LOSS_ROW_MAPPING
    weights = {
        "equal": np.ones(len(specs)),
        "national_as_is": lw.target_loss_weights(specs),
        "acs_local": lw.us_acs_local_target_loss_weights(specs),
    }
    out = {}
    for name, vector in weights.items():
        mapping = national if name == "national_as_is" else acs
        out[name] = lw.target_loss_weight_distribution(
            specs, vector, row_mapping=mapping
        )
    out["acs_local"]["weights_sha256"] = lw.target_loss_weights_sha256(
        [lw.target_row_name(spec) for spec in specs], weights["acs_local"]
    )
    out["acs_local"]["mean"] = float(weights["acs_local"].mean())
    out["acs_local"]["min"] = float(weights["acs_local"].min())
    out["acs_local"]["max"] = float(weights["acs_local"].max())
    return out


def family_level_table(result: dict) -> list[dict]:
    """Rows of family x geography level: equal, national-as-is, ACS local shares."""

    def shares(distribution: dict) -> dict[tuple[str, str], float]:
        totals: dict[tuple[str, str], float] = {}
        for cell in distribution["by_family_level_basis"]:
            key = (cell["family"], cell["geography_level"])
            totals[key] = totals.get(key, 0.0) + cell["loss_share"]
        return totals

    counts: dict[tuple[str, str], int] = {}
    for cell in result["acs_local"]["by_family_level_basis"]:
        key = (cell["family"], cell["geography_level"])
        counts[key] = counts.get(key, 0) + cell["n_targets"]
    equal = shares(result["equal"])
    national = shares(result["national_as_is"])
    acs = shares(result["acs_local"])
    return [
        {
            "family": key[0],
            "geography_level": key[1],
            "n_targets": counts[key],
            "equal": equal[key],
            "national_as_is": national[key],
            "acs_local": acs[key],
        }
        for key in sorted(counts)
    ]


def print_table(title: str, rows: list[dict]) -> None:
    print(f"\n### {title}\n")
    print(
        "| Family | Level | Targets | Equal (before) | National mapping as is "
        "| ACS local mapping (after) |"
    )
    print("|---|---|---:|---:|---:|---:|")
    totals = {"n_targets": 0, "equal": 0.0, "national_as_is": 0.0, "acs_local": 0.0}
    for row in rows:
        for key in totals:
            totals[key] += row[key]
        print(
            f"| {row['family']} | {row['geography_level']} | {row['n_targets']:,} "
            f"| {row['equal']:.1%} | {row['national_as_is']:.1%} "
            f"| {row['acs_local']:.1%} |"
        )
    print(
        f"| **Total** | | {totals['n_targets']:,} | {totals['equal']:.0%} "
        f"| {totals['national_as_is']:.0%} | {totals['acs_local']:.0%} |"
    )


def state_cd_surface(tool, feed: Path, population_specs, fraction: float):
    surface = tool.state_admin_surface(feed, FAMILIES, soi_mode=tool.SOI_MODE_STATE_CD)
    specs = (*surface.registry.specs, *population_specs)
    records = [tool.cd_surface.target_record(spec) for spec in specs]
    holdout = tool.cd_surface.assign_target_roles(records, fraction=fraction)
    train = [specs[index] for index in tool.cd_surface.train_rows(records)]
    blocks = {
        spec.metadata["state_cd_parent_target_name"]
        for spec in train
        if spec.metadata.get("ledger_geography_level") == "congressional_district"
        and spec.metadata.get("state_cd_parent_target_name")
    }
    return train, {
        "soi_receipt_counts": surface.soi_receipt.get("counts"),
        "n_specs": len(specs),
        "n_train": len(train),
        "holdout": {
            key: holdout[key]
            for key in ("fraction", "eligible_units", "held_units", "held_targets")
        },
        "n_state_cd_blocks_trained": len(blocks),
    }


def mode_receipt(tool, lw, feed: Path, mode: str, population_specs) -> dict:
    """Does the ACS local mapping classify and group one SOI mode's surface?"""

    import dataclasses

    surface = tool.state_admin_surface(feed, FAMILIES, soi_mode=mode)
    specs = [*surface.registry.specs, *population_specs]
    weights = lw.us_acs_local_target_loss_weights(specs)
    distribution = lw.target_loss_weight_distribution(
        specs, weights, row_mapping=lw.US_ACS_LOCAL_TARGET_LOSS_ROW_MAPPING
    )
    district_ledger_rows = sum(
        1
        for spec in specs
        if spec.metadata.get("ledger_geography_level") == "congressional_district"
    )
    leaky = [
        dataclasses.replace(
            spec, metadata={**spec.metadata, "district_display_name": spec.name}
        )
        if spec.metadata.get("ledger_geography_level") == "congressional_district"
        else spec
        for spec in specs
    ]
    try:
        lw.us_acs_local_target_loss_weights(leaky)
        injected_key_refused = False
    except ValueError:
        injected_key_refused = True
    return {
        "n_specs": len(specs),
        "n_district_ledger_rows": district_ledger_rows,
        "n_basis_overrides": sum(
            1
            for spec in specs
            if spec.metadata.get("source_measure_id")
            in lw.ACS_LOCAL_LEDGER_BASIS_OVERRIDES
        ),
        "concept_groups": distribution["concept_groups"],
        "injected_per_district_key_refused": injected_key_refused,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--feed", type=Path, default=DEFAULT_FEED)
    parser.add_argument("--skip-state-cd", action="store_true")
    args = parser.parse_args(argv)
    started = time.time()
    tool = load_tool()
    from microcosm.build.us_runtime import target_loss_weights as lw
    from microcosm.calibrate import TargetRegistry

    registry_sha = sha256(args.registry)
    if registry_sha != REGISTRY_SHA256:
        raise SystemExit(
            f"{args.registry} is {registry_sha}, not the rebuilt 09-23 registry."
        )
    release_specs = TargetRegistry.from_json(args.registry).specs
    dirty = subprocess.run(
        ["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    results: dict = {
        "head": subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
        "tree_clean": not dirty,
        "release_20260923": {
            "registry": {"path": str(args.registry), "sha256": registry_sha},
            "n_train": len(release_specs),
            **score(release_specs, lw),
        },
    }
    rows = family_level_table(results["release_20260923"])
    results["release_20260923"]["family_level_table"] = rows
    print_table(
        "09-23 release surface (soi_mode=state), 4,459 targets, all trained", rows
    )
    if not args.skip_state_cd:
        pin = json.loads(
            (
                REPO
                / "packages/microcosm-build/src/microcosm/build/us/chronicle_feed.json"
            ).read_text()
        )
        feed_sha = sha256(args.feed)
        if feed_sha != pin["facts_sha256"]:
            raise SystemExit(f"{args.feed} is {feed_sha}, not the pinned feed.")
        population = [
            spec for spec in release_specs if spec.family == "census_population"
        ]
        train, receipt = state_cd_surface(
            tool,
            args.feed,
            population,
            tool.cd_surface.DEFAULT_STATE_CD_HOLDOUT_FRACTION,
        )
        result = {
            "feed": {"path": str(args.feed), "sha256": feed_sha},
            **receipt,
            **score(train, lw),
        }
        result["family_level_table"] = family_level_table(result)
        results["state_cd_pinned_feed"] = result
        print_table(
            f"state_cd surface, pinned feed, {receipt['n_train']:,} training targets",
            result["family_level_table"],
        )
        print(
            "\nstate_cd blocks trained:",
            receipt["n_state_cd_blocks_trained"],
            "| ACS local concept groups:",
            result["acs_local"]["concept_groups"],
            "| national-as-is:",
            result["national_as_is"]["concept_groups"],
        )
        results["soi_modes"] = {
            mode: mode_receipt(tool, lw, args.feed, mode, population)
            for mode in tool.SOI_MODES
        }
        for mode, receipt in results["soi_modes"].items():
            print(mode, receipt)
    results["wall_seconds"] = round(time.time() - started, 1)
    (HERE / "results.json").write_text(json.dumps(results, indent=1, sort_keys=True))
    print(f"\nwrote {HERE / 'results.json'} in {results['wall_seconds']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
