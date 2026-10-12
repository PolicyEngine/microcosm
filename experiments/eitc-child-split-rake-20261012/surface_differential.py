"""Release-equivalent compiles of the pinned feed under each option.

Mirrors tools/build_us_fiscal_refresh_release.py (as the #1227 evidence
scripts do): load_ledger_consumer_artifact -> compile_us_fiscal_target_registry
(target_period=PERIOD, packaged CD crosswalk, age_targets=True,
allow_unaged_dollar_targets=False) -> apply_us_medicaid_enrollment_substitutions
-> _select_target_surface(national_state | full), then the national
target-loss weights on each surface.

Options, all from one checkout that carries ``_rake_eitc_child_splits``:
  a  status quo (#1227): the rake patched to the identity;
  b  total rows := sum of their uprated bins (status-quo bins);
  c  the rake (the checkout as is);
  d  total rows dropped (status-quo bins).

Writes ``<out_dir>/surface_differential.json`` (each option's surfaces,
loss-weight shares, per-split levels, the congressional-district and Historic
Table 2 state EITC-by-children rows) and ``<out_dir>/t25_rows.csv`` (every
Table 2.5 row's value and loss weights under each option).

Run: cd <REPO> && .venv/bin/python <this> <REPO> <out_dir>
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import time
from pathlib import Path

import microcosm.build.us_runtime.fiscal_targets as fiscal_targets
from microcosm.build.ledger_artifact import load_ledger_consumer_artifact
from microcosm.build.us_runtime import (
    apply_us_medicaid_enrollment_substitutions,
    compile_us_fiscal_target_registry,
    default_congressional_district_vintage_crosswalk_path,
    load_congressional_district_vintage_crosswalk,
)
from microcosm.build.us_runtime.chronicle_feed import load_us_chronicle_feed
from microcosm.build.us_runtime.target_loss_weights import fiscal_target_loss_weights
from microcosm.calibrate import TargetRegistry

T0 = time.time()
REPO = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

assert str(REPO) in fiscal_targets.__file__, fiscal_targets.__file__
RAKE = fiscal_targets._rake_eitc_child_splits


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def load_tool(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load_tool("build_us_fiscal_refresh_release")
FEED = load_tool("build_us_target_parity_manifest").DEFAULT_FEED_PATH
pin = load_us_chronicle_feed()
artifact = load_ledger_consumer_artifact(
    FEED, expected_facts_sha256=pin.facts_sha256, expected_manifest_sha256=None
)
facts = artifact.facts
log(f"feed {FEED} sha256={artifact.facts_sha256} rows={len(facts)}")
crosswalk = load_congressional_district_vintage_crosswalk(
    default_congressional_district_vintage_crosswalk_path()
)


def is_t25(spec) -> bool:
    return fiscal_targets._is_eitc_decomposition_spec(spec)


def is_total_row(spec) -> bool:
    return (
        is_t25(spec)
        and spec.metadata.get("agi_lower_bound") == "-inf"
        and spec.metadata.get("agi_upper_bound") == "inf"
    )


def identity(specs, **_):
    return list(specs)


def totals_are_bin_sums(specs, **_):
    from dataclasses import replace

    sums: dict[tuple[str, str, str], float] = {}
    for spec in specs:
        if is_t25(spec) and not is_total_row(spec):
            key = (
                spec.metadata["ledger_layout_record_set_id"],
                spec.metadata["source_measure_id"],
                str(spec.period),
            )
            sums[key] = sums.get(key, 0.0) + spec.value
    out = []
    for spec in specs:
        if is_total_row(spec):
            key = (
                spec.metadata["ledger_layout_record_set_id"],
                spec.metadata["source_measure_id"],
                str(spec.period),
            )
            spec = replace(spec, value=sums[key])
        out.append(spec)
    return out


def drop_totals(specs, **_):
    return [spec for spec in specs if not is_total_row(spec)]


OPTIONS = {"a": identity, "b": totals_are_bin_sums, "c": RAKE, "d": drop_totals}
compiled = {}
for name, rake in OPTIONS.items():
    fiscal_targets._rake_eitc_child_splits = rake
    registry = compile_us_fiscal_target_registry(
        facts,
        target_period=builder.PERIOD,
        congressional_district_vintage_crosswalk=crosswalk,
        age_targets=True,
        allow_unaged_dollar_targets=False,
    )
    registry, _ = apply_us_medicaid_enrollment_substitutions(registry)
    surfaces = {}
    for surface in (builder.TARGET_SURFACE_NATIONAL_STATE, builder.TARGET_SURFACE_FULL):
        specs, receipt = builder._select_target_surface(registry.specs, surface)
        reg = TargetRegistry(specs, country="us")
        weights = dict(
            zip(
                [(s.name, str(s.period)) for s in reg.specs],
                fiscal_target_loss_weights(reg),
                strict=True,
            )
        )
        surfaces[surface] = {"registry": reg, "weights": weights, "receipt": receipt}
    compiled[name] = {"registry": registry, "surfaces": surfaces}
    log(
        f"option {name}: {len(registry.specs)} specs; "
        + "; ".join(
            f"{k}={len(v['registry'].specs)} v={v['registry'].version}"
            for k, v in surfaces.items()
        )
    )
fiscal_targets._rake_eitc_child_splits = RAKE

SPLITS = (
    "no_qualifying_children",
    "one_qualifying_child",
    "two_qualifying_children",
    "three_or_more_qualifying_children",
)
result = {"feed": str(FEED), "facts_sha256": artifact.facts_sha256, "options": {}}
base = compiled["a"]
base_by = {(s.name, str(s.period)): s for s in base["registry"].specs}
for name, comp in compiled.items():
    by = {(s.name, str(s.period)): s for s in comp["registry"].specs}
    changed = []
    for key in sorted(set(by) | set(base_by)):
        new, old = by.get(key), base_by.get(key)
        if new is None or old is None:
            changed.append(
                {
                    "name": key[0],
                    "old": None if old is None else old.value,
                    "new": None if new is None else new.value,
                }
            )
            continue
        if new.value != old.value or dict(new.metadata) != dict(old.metadata):
            changed.append(
                {
                    "name": key[0],
                    "old": old.value,
                    "new": new.value,
                    "ratio": new.value / old.value if old.value else None,
                    "metadata_keys_changed": sorted(
                        k
                        for k in set(new.metadata) | set(old.metadata)
                        if new.metadata.get(k) != old.metadata.get(k)
                    ),
                }
            )
    non_t25 = [
        c for c in changed if ".table_2_5.eitc_by_agi_children." not in c["name"]
    ]
    per_split = {}
    for measure in ("eitc_total", "eitc_returns"):
        for split in SPLITS:
            rs = f"irs_soi.ty2023.table_2_5.eitc_by_agi_children.{split}"
            rows = [
                s
                for s in comp["registry"].specs
                if s.metadata.get("ledger_layout_record_set_id") == rs
                and s.metadata.get("source_measure_id") == measure
            ]
            total = [s for s in rows if is_total_row(s)]
            bins = [s for s in rows if not is_total_row(s)]
            per_split[f"{measure}|{split}"] = {
                "total_row": total[0].value if total else None,
                "bins_sum": sum(s.value for s in bins),
                "total_over_bins": (total[0].value / sum(s.value for s in bins))
                if total
                else None,
                "n_bins": len(bins),
            }
    surfaces = {}
    for surface, data in comp["surfaces"].items():
        weights = data["weights"]
        base_weights = base["surfaces"][surface]["weights"]
        t25_keys = [k for k in weights if ".table_2_5.eitc_by_agi_children." in k[0]]
        t25_total_keys = [
            k
            for k in t25_keys
            if k[0].endswith((".total.eitc_total", ".total.eitc_returns"))
        ]
        common = [k for k in weights if k in base_weights]
        outside = [k for k in common if k not in set(t25_keys)]
        surfaces[surface] = {
            "version": data["registry"].version,
            "count": len(data["registry"].specs),
            "receipt": data["receipt"],
            "t25_rows": len(t25_keys),
            "t25_weight_share": sum(weights[k] for k in t25_keys)
            / sum(weights.values()),
            "t25_total_rows_weight_share": sum(weights[k] for k in t25_total_keys)
            / sum(weights.values()),
            "max_abs_weight_ratio_minus_1_outside_t25": max(
                (abs(weights[k] / base_weights[k] - 1) for k in outside), default=0.0
            ),
            "t25_weight_ratio_range_vs_a": [
                min(
                    weights[k] / base_weights[k] for k in t25_keys if k in base_weights
                ),
                max(
                    weights[k] / base_weights[k] for k in t25_keys if k in base_weights
                ),
            ],
        }
    result["options"][name] = {
        "compiled": len(comp["registry"].specs),
        "changed_vs_a": len(changed),
        "changed_vs_a_outside_t25": len(non_t25),
        "changed_rows": changed,
        "per_split": per_split,
        "surfaces": surfaces,
    }
    log(f"option {name}: changed vs a {len(changed)} (outside T2.5 {len(non_t25)})")
    for surface, s in surfaces.items():
        log(
            f"   {surface}: v={s['version']} n={s['count']} t25 rows {s['t25_rows']} share {s['t25_weight_share']:.5f} totals {s['t25_total_rows_weight_share']:.5f} outside max |dw| {s['max_abs_weight_ratio_minus_1_outside_t25']:.2e}"
        )
    for k, v in per_split.items():
        log(f"   {k}: total/bins {v['total_over_bins']}")

# Per-row detail: each T2.5 row under every option.
with (OUT / "t25_rows.csv").open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(
        [
            "name",
            "agi_lower",
            "agi_upper",
            *(
                f"{name}_{field}"
                for name in OPTIONS
                for field in (
                    "value",
                    "uprating_factor",
                    "weight_national_state",
                    "weight_full",
                )
            ),
            "c_child_split_adjustment",
        ]
    )
    by_option = {
        name: {s.name: s for s in comp["registry"].specs if is_t25(s)}
        for name, comp in compiled.items()
    }
    for row_name, spec in sorted(by_option["a"].items()):
        cells = [
            row_name,
            spec.metadata.get("agi_lower_bound"),
            spec.metadata.get("agi_upper_bound"),
        ]
        for name, comp in compiled.items():
            other = by_option[name].get(row_name)
            key = (row_name, str(spec.period))
            cells += [
                None if other is None else repr(other.value),
                None if other is None else other.metadata.get("uprating_factor"),
                comp["surfaces"][builder.TARGET_SURFACE_NATIONAL_STATE]["weights"].get(
                    key
                ),
                comp["surfaces"][builder.TARGET_SURFACE_FULL]["weights"].get(key),
            ]
        cells.append(
            by_option["c"][row_name].metadata.get("uprating_child_split_adjustment")
        )
        writer.writerow(cells)

# The other EITC-by-children family on national_state: Historic Table 2
# state rows, summed to the nation (unchanged by every option).
ht2 = {}
for s in compiled["c"]["registry"].specs:
    if ".historic_table_2.state_eitc." in s.metadata.get(
        "ledger_layout_record_set_id", ""
    ):
        measure = s.metadata["source_measure_id"]
        ht2[measure] = ht2.get(measure, 0.0) + s.value
result["ht2_state_eitc_national_sums"] = ht2
for name in result["options"]:
    del result["options"][name]["changed_rows"]

# Congressional-district US EITC-by-children rows against each option.
cd = {}
for s in compiled["c"]["registry"].specs:
    m = s.metadata
    if (
        ".congressional_district_2022." in m.get("ledger_layout_record_set_id", "")
        and m.get("ledger_geography_level") == "country"
    ):
        cd[m.get("source_measure_id")] = s.value
MAP = {
    "eitc_no_children": "no_qualifying_children",
    "eitc_one_child": "one_qualifying_child",
    "eitc_two_children": "two_qualifying_children",
    "eitc_three_or_more_children": "three_or_more_qualifying_children",
}
cmp = {}
for prefix, split in MAP.items():
    for suffix, measure in (("claims", "eitc_returns"), ("amount", "eitc_total")):
        key = f"{prefix}_{suffix}"
        if key not in cd:
            continue
        cmp[key] = {"cd_us": cd[key]}
        for name in OPTIONS:
            ps = result["options"][name]["per_split"][f"{measure}|{split}"]
            cmp[key][f"{name}_over_bins"] = cd[key] / ps["bins_sum"]
            if ps["total_row"] is not None:
                cmp[key][f"{name}_over_total"] = cd[key] / ps["total_row"]
result["cd_us_vs_t25"] = cmp
(OUT / "surface_differential.json").write_text(
    json.dumps(result, indent=1, default=str)
)
log("cd vs t25: " + json.dumps(cmp))
log("done")
