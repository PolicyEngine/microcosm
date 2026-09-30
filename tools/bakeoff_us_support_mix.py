#!/usr/bin/env python
"""Bake-off: CPS ASEC years vs ACS rows in the US calibration support.

At fixed source-household budgets, each arm keeps every household of N CPS
income years and fills the rest with ACS households, CPS location clones, or
a mix. Every arm is calibrated with the production solver and scored on
targets it never saw. See docs/us-support-mix-bakeoff.md for the design,
results and caveats.

The expensive part, running PolicyEngine-US, happens once per household, not
once per arm. ``compile`` groups the pinned target registry into concepts:
a concept is a target with its geography stripped, and every state or
district target is that concept's household value times an indicator of the
household's own state or district (the release materializer builds its
columns exactly this way; ``diffcheck`` verifies it on real rows).
``materialize`` runs the release tool's own ``_materialize_target_frame`` on
household shards for the concept specs. ``arm`` then selects rows, assembles
the sparse target matrix, calls the production ``_optimize`` and scores.

Subcommands, in order: compile, shard, materialize, diffcheck, truth, arm,
report. Every subcommand writes a JSON receipt with wall time and peak RSS.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import importlib.util
import json
import math
import pickle
import resource
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

from microcosm.build.us_runtime.support_mix import (  # noqa: E402
    SupportMixArm,
    acs_selection_order,
    arm_initial_weights,
    assemble_target_matrix,
    clone_copies,
    distinct_unit_ess,
    hash_uniform,
    kish_ess,
    plan_arm_counts,
    target_split_group_key,
    target_split_role,
)

PERIOD = 2024
FEED_SHA256 = "b85437390021777e746f507c5890305496baf5fc7f2c78ba08ddb090f4839801"
ENTITIES = ("person", "household", "tax_unit", "spm_unit", "family", "marital_unit")
GROUP_ID = {
    "tax_unit": ("tax_unit_id", "person_tax_unit_id"),
    "spm_unit": ("spm_unit_id", "person_spm_unit_id"),
    "family": ("family_id", "person_family_id"),
    "marital_unit": ("marital_unit_id", "person_marital_unit_id"),
}
ACS_SELECTION_SALT = "microcosm.us.support_mix_bakeoff.v1.acs_order"
CLONE_SALT = "microcosm.us.support_mix_bakeoff.v1.clones"
CLONE_GEO_SALT = "microcosm.us.support_mix_bakeoff.v1.clone_geography"
REPORT_SPECS = {
    "report.spm_poor_persons": {"base_variable": "person_in_poverty", "measure_mode": "sum"},
    "report.spm_poor_children": {
        "base_variable": "person_in_poverty",
        "measure_mode": "sum",
        "age_lower_bound": "0",
        "age_upper_bound": "18",
    },
}

# ------------------------------------------------------------------ utilities


def log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{stamp} {message}", flush=True)


def peak_rss_gb() -> float:
    # macOS reports ru_maxrss in bytes; Linux in kilobytes.
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / 1e9 if sys.platform == "darwin" else raw * 1024 / 1e9


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str))
    tmp.replace(path)


#: Touch this file to pause every bake-off loop between units of work (a shard
#: or an arm) — e.g. while a release stage needs the machine's memory.
PAUSE_FILE = Path("/Users/maxghenis/PolicyEngine/_bakeoff-support-mix/PAUSE")


def wait_while_paused() -> None:
    while PAUSE_FILE.exists():
        log(f"paused: {PAUSE_FILE} exists")
        time.sleep(120)


def wait_for_memory(min_available_gb: float) -> None:
    """Block until the host has ``min_available_gb`` free (a shared laptop)."""
    import psutil

    wait_while_paused()
    while psutil.virtual_memory().available / 2**30 < min_available_gb:
        log(
            f"waiting for memory: {psutil.virtual_memory().available / 2**30:.1f} GiB "
            f"available < {min_available_gb} GiB"
        )
        time.sleep(120)


def release_tool():
    name = "build_us_fiscal_refresh_release"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def acs_local_tool():
    name = "build_us_acs_local_release"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------ compile

INCOME_SOURCES = {
    "adjusted_gross_income", "employment_income", "self_employment_income",
    "capital_gains", "capital_gains_gross", "capital_gains_losses",
    "qualified_dividend_income", "ordinary_dividends", "taxable_interest_income",
    "tax_exempt_interest_income", "rent_and_royalty_net_income",
    "rent_and_royalty_net_losses", "partnership_s_corp_income",
    "partnership_s_corp_losses", "estate_income", "estate_losses",
    "taxable_pension_income", "total_pension_income", "taxable_ira_distributions",
    "taxable_social_security", "total_social_security", "unemployment_compensation",
    "business_net_profits", "business_net_losses", "non_sch_d_capital_gains",
    "count",
}


def dimension_of(family: str, metadata: dict) -> tuple[str, str]:
    """(dimension group, dimension) for one registry target."""
    if family in {"usda_snap", "cms_medicaid", "cms_aca", "ssa", "hhs_acf_tanf",
                  "hhs_acf_liheap", "cms_medicare"}:
        return "cps_native", "program_receipt"
    if family in {"state_income_tax", "jct"}:
        return "cps_native", "tax_items"
    if family == "census_population":
        return "acs_native", "demographics_pep"
    if family in {"bea", "cbo"}:
        return "cps_native", "income_components"
    if family == "federal_reserve":
        return "other", "wealth"
    if family == "irs_soi":
        source = metadata.get("source_variable", metadata.get("variable", ""))
        role = metadata.get("target_role", "")
        if source in INCOME_SOURCES and "aca" not in role:
            return "cps_native", "income_components"
        return "cps_native", "tax_items"
    return "other", family


def _concept_key(tool, spec) -> tuple[tuple, list[str]]:
    """Geography-free value key of one spec, from the release tool's helpers."""

    def bound(value):
        v = tool._as_bound(str(value))
        return "inf" if v == math.inf else ("-inf" if v == -math.inf else repr(float(v)))

    def filing(name):
        status = tool.FILING_STATUS_MAP[name]
        if status is None:
            return "ALL"
        return status if isinstance(status, str) else "|".join(sorted(status))

    def child(filter_):
        if filter_ is None:
            return None
        mask = tool._eitc_child_count_mask(np.arange(0, 6, dtype=float), filter_)
        return "".join("1" if x else "0" for x in mask)

    m = spec.metadata
    notes: list[str] = []
    if spec.family == "jct":
        return ("jct",), ["excluded:reform_simulation"]
    if spec.family == "state_income_tax":
        return ("state_income_tax",), notes
    materializer = m.get("materializer")
    if spec.family == "irs_soi":
        source = m.get("source_variable", m.get("variable"))
        if tool._unsupported_soi_ledger_filters(m):
            notes.append("excluded:unsupported_soi_filter")
        mode = "count" if source == "count" else (
            "indicator" if m.get("measure_mode") == "indicator_sum" else "amount"
        )
        key = ("irs_soi", source, mode, bound(m["agi_lower_bound"]),
               bound(m["agi_upper_bound"]), filing(m["filing_status"]),
               child(tool._soi_eitc_child_count_filter(m)),
               bool(tool._soi_requires_positive_eitc_filter(m)),
               m.get("itemized_only") == "true", m.get("taxable_only") == "true")
        return key, notes
    if materializer == "population_age":
        return ("population_age", bound(m.get("age_lower_bound", "-inf")),
                bound(m.get("age_upper_bound", "inf"))), notes
    if materializer == "policyengine_variable":
        base = tool._base_variables_from_metadata(m)
        less_than = tool._less_than_from_metadata(m)
        mode = m.get("measure_mode", "sum")
        low, high = m.get("age_lower_bound"), m.get("age_upper_bound")
        if low is not None or high is not None:
            if m.get("state_fips") or m.get("congressional_district_geoid"):
                notes.append("age_banded_geographic_spec")
            return ("pe_var_age", base, mode, bound(low if low is not None else "-inf"),
                    bound(high if high is not None else "inf")), notes
        return ("pe_var", base, mode, m.get("indicator_map_to"),
                m.get("indicator_filter_variable"),
                None if less_than is None else repr(less_than)), notes
    return ("direct", spec.measure), ["direct_measure"]


def do_compile(args) -> None:
    started = time.time()
    tool = release_tool()
    from microcosm.build.ledger_artifact import load_ledger_consumer_artifact
    from microcosm.build.us_runtime import (
        apply_us_medicaid_enrollment_substitutions,
        compile_us_fiscal_target_registry,
        default_congressional_district_vintage_crosswalk_path,
        load_congressional_district_vintage_crosswalk,
    )
    from microcosm.calibrate.registry import TargetSpec

    feed_sha = sha256_file(args.feed)
    if feed_sha != FEED_SHA256:
        raise SystemExit(f"feed sha256 {feed_sha} != pinned {FEED_SHA256}")
    artifact = load_ledger_consumer_artifact(args.feed)
    registry = compile_us_fiscal_target_registry(
        artifact.facts,
        target_period=PERIOD,
        congressional_district_vintage_crosswalk=load_congressional_district_vintage_crosswalk(
            default_congressional_district_vintage_crosswalk_path()
        ),
        age_targets=True,
    )
    registry, substitutions = apply_us_medicaid_enrollment_substitutions(registry)
    specs = tuple(registry.specs)
    national_state, surface_receipt = tool._select_target_surface(
        specs, tool.TARGET_SURFACE_NATIONAL_STATE
    )
    national_state_names = {spec.name for spec in national_state}
    loss_weights = np.asarray(tool._fiscal_target_loss_weights(registry), dtype=np.float64)
    # The release computes weights on the surface it calibrates, so the
    # national product gets its own vector over the national_state specs.
    from types import SimpleNamespace

    national_state_weights = dict(zip(
        (spec.name for spec in national_state),
        np.asarray(tool._fiscal_target_loss_weights(SimpleNamespace(specs=tuple(national_state))), dtype=np.float64),
        strict=True,
    ))

    concept_ids: dict[tuple, int] = {}
    concept_specs = []
    rows = []
    for index, spec in enumerate(specs):
        key, notes = _concept_key(tool, spec)
        excluded = next((n for n in notes if n.startswith("excluded:")), None)
        m = dict(spec.metadata)
        concept_id = -1
        measure = None
        if key == ("state_income_tax",):
            measure = "state_income_tax"
        elif excluded is None:
            if key not in concept_ids:
                concept_ids[key] = len(concept_ids)
                cid = concept_ids[key]
                stripped = {
                    k: v for k, v in m.items()
                    if k not in {"state_fips", "congressional_district_geoid",
                                 "hierarchy_parent_target_name"}
                }
                stripped["ledger_geography_level"] = "country"
                stripped["ledger_geography_id"] = "US"
                concept_measure = (
                    spec.measure if key[0] == "direct" else f"bakeoff_concept_{cid:05d}"
                )
                concept_specs.append(
                    dataclasses.replace(
                        spec,
                        name=f"bakeoff.concept.{cid:05d}",
                        measure=concept_measure,
                        metadata=stripped,
                        hierarchy=None,
                    )
                )
            concept_id = concept_ids[key]
            measure = concept_specs[concept_id].measure
        level = m.get("ledger_geography_level", "")
        group, dimension = dimension_of(spec.family, m)
        group_key = target_split_group_key(spec.name, m.get("hierarchy_parent_target_name"))
        rows.append({
            "target_index": index,
            "name": spec.name,
            "period": str(spec.period),
            "family": spec.family,
            "level": {"country": "national", "congressional_district": "cd"}.get(level, level),
            "state_fips": int(m["state_fips"]) if m.get("state_fips") else -1,
            "cd_geoid": int(m["congressional_district_geoid"]) if m.get("congressional_district_geoid") else -1,
            "value": float(spec.value),
            "measure": measure,
            "concept_id": concept_id,
            "concept_key": json.dumps(key, default=str),
            "excluded": excluded,
            "notes": ";".join(n for n in notes if not n.startswith("excluded:")),
            "in_national_state": spec.name in national_state_names,
            "loss_weight": float(loss_weights[index]),
            "loss_weight_national_state": float(national_state_weights.get(spec.name, np.nan)),
            "group_key": group_key,
            "role": target_split_role(group_key),
            "dimension_group": group,
            "dimension": dimension,
            "value_basis": tool._fiscal_target_value_basis(spec),
        })
    for name, metadata in REPORT_SPECS.items():
        concept_specs.append(
            TargetSpec(
                name=f"bakeoff.{name}",
                entity="household",
                value=1.0,
                measure=f"bakeoff_{name.replace('.', '_')}",
                source="support-mix bake-off report-only measure (never calibrated)",
                family="bakeoff_report",
                metadata={"materializer": "policyengine_variable", **metadata},
            )
        )
    targets = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    targets.to_parquet(args.out / "targets.parquet", index=False)
    with open(args.out / "concept_specs.pkl", "wb") as handle:
        pickle.dump(tuple(concept_specs), handle)
    receipt = {
        "feed": str(args.feed),
        "feed_sha256": feed_sha,
        "registry_version": registry.version,
        "n_specs": len(specs),
        "n_concepts": len(concept_ids),
        "n_report_concepts": len(REPORT_SPECS),
        "medicaid_substitutions": len(substitutions),
        "national_state_surface": surface_receipt,
        "excluded": dict(Counter(r["excluded"] for r in rows if r["excluded"])),
        "notes": dict(Counter(n for r in rows for n in r["notes"].split(";") if n)),
        "roles": dict(Counter(r["role"] for r in rows)),
        "roles_by_level": {
            level: dict(Counter(r["role"] for r in rows if r["level"] == level))
            for level in sorted({r["level"] for r in rows})
        },
        "dimension_counts": dict(Counter(f"{r['dimension_group']}/{r['dimension']}/{r['level']}" for r in rows)),
        "split_note": (
            "Mirrors the unmerged us-target-roles-holdout split (salts, \\x1f "
            "hash, group keys) without its pins; see support_mix.py."
        ),
        "wall_s": round(time.time() - started, 1),
        "peak_rss_gb": round(peak_rss_gb(), 2),
    }
    write_json(args.out / "compile.json", receipt)
    log(json.dumps({k: receipt[k] for k in ("n_specs", "n_concepts", "roles", "excluded")}))


# ------------------------------------------------------------------ shard


class _FixedStore:
    """Row-sliced reads of a pandas ``fixed``-format HDF5 frame store."""

    def __init__(self, path: Path):
        import h5py
        import tables

        self.h5 = h5py.File(path, "r")
        self.tb = tables.open_file(str(path), mode="r")
        self.layout = {}
        for entity in ENTITIES:
            group = self.h5[entity]
            blocks, i = [], 0
            while f"block{i}_items" in group:
                items = [x.decode() for x in group[f"block{i}_items"][()]]
                blocks.append((i, items, str(group[f"block{i}_values"].dtype)))
                i += 1
            self.layout[entity] = {
                "nrows": int(group["axis1"].shape[0]),
                "blocks": blocks,
                "columns": [item for _, items, _ in blocks for item in items],
            }
        self._objects: dict[tuple[str, int], np.ndarray] = {}

    def rows(self, entity: str) -> int:
        return self.layout[entity]["nrows"]

    def read(self, entity: str, start: int, stop: int, columns=None) -> pd.DataFrame:
        columns = self.layout[entity]["columns"] if columns is None else list(columns)
        out = {}
        group = self.h5[entity]
        for i, items, dtype in self.layout[entity]["blocks"]:
            wanted = [(j, c) for j, c in enumerate(items) if c in columns]
            if not wanted:
                continue
            if dtype == "object":
                key = (entity, i)
                if key not in self._objects:
                    node = self.tb.get_node(f"/{entity}/block{i}_values")
                    array = np.asarray(node[0], dtype=object)
                    self._objects[key] = array.reshape(1, -1) if array.ndim == 1 else array.T
                block = self._objects[key]
                for j, c in wanted:
                    out[c] = block[j, start:stop]
            else:
                values = group[f"block{i}_values"][start:stop, :]
                for j, c in wanted:
                    out[c] = values[:, j]
        return pd.DataFrame({c: out[c] for c in columns})

    def read_positions(self, entity: str, positions: np.ndarray) -> pd.DataFrame:
        positions = np.asarray(positions, dtype=np.int64)
        if not len(positions):
            return self.read(entity, 0, 0)
        low, high = int(positions.min()), int(positions.max()) + 1
        if high - low > 4 * len(positions) + 10_000:
            raise RuntimeError(
                f"{entity}: rows for a shard span {high - low:,} for {len(positions):,} rows; "
                "the store is not in household order."
            )
        frame = self.read(entity, low, high)
        return frame.iloc[positions - low].reset_index(drop=True)


def _write_shard(path: Path, tables: dict[str, pd.DataFrame], weights: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as handle:
        pickle.dump({"tables": tables, "household_weight": weights}, handle, protocol=5)
    tmp.replace(path)


def _row_index_columns(tables: dict[str, pd.DataFrame], source: str) -> pd.DataFrame:
    """Per-household identity, geography and ACS-native evaluation columns."""
    hh = tables["household"]
    person = tables["person"]
    positions = pd.Series(np.arange(len(hh)), index=hh["household_id"].to_numpy())
    person_position = positions.reindex(person["person_household_id"].to_numpy()).to_numpy()
    n = len(hh)

    def per_household(values) -> np.ndarray:
        return np.bincount(person_position, weights=np.asarray(values, dtype=np.float64), minlength=n)

    age = pd.to_numeric(person["age"], errors="coerce").fillna(0).to_numpy()
    male = ~pd.to_numeric(person["is_female"]).fillna(0).astype(bool).to_numpy()
    rent = pd.to_numeric(person.get("pre_subsidy_rent", 0.0), errors="coerce").fillna(0.0)
    taxes = pd.to_numeric(person.get("real_estate_taxes", 0.0), errors="coerce").fillna(0.0)
    tenure = hh["tenure_type"].astype(str).to_numpy()
    out = pd.DataFrame({
        "household_id": hh["household_id"].to_numpy(),
        "state_fips": pd.to_numeric(hh["state_fips"]).to_numpy().astype(np.int64),
        "cd_geoid": pd.to_numeric(hh["congressional_district_geoid"]).to_numpy().astype(np.int64),
        "county_fips": pd.to_numeric(hh["county_fips"]).to_numpy().astype(np.int64),
        "persons": per_household(np.ones(len(person))),
        "persons_0_17": per_household(age < 18),
        "persons_18_64": per_household((age >= 18) & (age < 65)),
        "persons_65p": per_household(age >= 65),
        "persons_male": per_household(male),
        "rent_annual": per_household(rent),
        "real_estate_taxes": per_household(taxes),
        "owner": np.isin(tenure, ["OWNED_WITH_MORTGAGE", "OWNED_OUTRIGHT"]).astype(np.float64),
        "renter": (tenure == "RENTED").astype(np.float64),
    })
    if source == "cps":
        peridnum = person["PERIDNUM"].astype(str).str[:20].to_numpy()
        if (pd.Series(peridnum).groupby(person_position).nunique() != 1).any():
            raise RuntimeError("persons of one CPS household disagree on PERIDNUM[:20] (H_IDNUM)")
        first = pd.Series(peridnum).groupby(person_position).first()
        out["unit_key"] = "cps:" + first.reindex(np.arange(n)).to_numpy().astype(str)
        year = pd.Series(pd.to_numeric(person["source_year"]).to_numpy()).groupby(person_position).first()
        out["income_year"] = year.reindex(np.arange(n)).to_numpy().astype(np.int64)
        out["source_key"] = out["income_year"].astype(str) + "|" + out["unit_key"]
        out["channel"] = hh["household_support_channel"].astype(str).to_numpy()
        out["clone_index"] = pd.to_numeric(hh["household_support_clone_index"]).to_numpy()
        out["group_quarters"] = False
    else:
        out["unit_key"] = "acs:" + hh["SERIALNO"].astype(str).to_numpy()
        out["income_year"] = -1
        out["source_key"] = out["unit_key"]
        out["channel"] = "acs"
        out["clone_index"] = 0
        out["group_quarters"] = pd.to_numeric(hh["TYPEHUGQ"], errors="coerce").fillna(1).to_numpy() != 1
    return out


def do_shard(args) -> None:
    started = time.time()
    out_dir = args.out / "shards" / args.source
    out_dir.mkdir(parents=True, exist_ok=True)
    index_parts = []
    if args.source == "cps":
        from microcosm.build.us_runtime.h5_io import read_frame_table

        with pd.HDFStore(args.h5, mode="r") as store:
            tables = {entity: read_frame_table(store, entity) for entity in ENTITIES}
        with pd.HDFStore(args.design_h5, mode="r") as store:
            design = store.select("household", columns=["household_id", "household_weight"])
        if not np.array_equal(design["household_id"].to_numpy(), tables["household"]["household_id"].to_numpy()):
            raise SystemExit("design-weight base and CPS export disagree on household ids/order")
        tables["household"] = tables["household"].drop(columns=["household_weight"])
        weights_all = design["household_weight"].to_numpy(np.float64)
        households = tables["household"]
        person_household = tables["person"]["person_household_id"].to_numpy()
        position = pd.Series(np.arange(len(households)), index=households["household_id"].to_numpy())
        person_position = position.reindex(person_household).to_numpy()
        n = len(households)
        for shard, low in enumerate(range(0, n, args.households_per_shard)):
            high = min(low + args.households_per_shard, n)
            person_mask = (person_position >= low) & (person_position < high)
            part = {"household": households.iloc[low:high].reset_index(drop=True),
                    "person": tables["person"].loc[person_mask].reset_index(drop=True)}
            for entity, (id_col, person_col) in GROUP_ID.items():
                ids = np.unique(part["person"][person_col].to_numpy())
                table = tables[entity]
                part[entity] = table.loc[table[id_col].isin(ids)].reset_index(drop=True)
            weights = weights_all[low:high]
            _write_shard(out_dir / f"shard_{shard:03d}.pkl", part, weights)
            index = _row_index_columns(part, "cps")
            index["design_weight"] = weights
            index["shard"] = shard
            index["position"] = np.arange(high - low)
            index_parts.append(index)
            log(f"cps shard {shard}: {high - low} households")
    else:
        store = _FixedStore(args.h5)
        hh_ids = store.read("household", 0, store.rows("household"), ["household_id", "household_spine"])
        acs_positions = np.flatnonzero(hh_ids["household_spine"].astype(str).to_numpy() == "acs_2024_1yr")
        if not len(acs_positions) or not np.array_equal(
            acs_positions, np.arange(acs_positions[0], acs_positions[-1] + 1)
        ):
            raise SystemExit("ACS spine rows are not one contiguous household range")
        household_position = pd.Series(np.arange(len(hh_ids)), index=hh_ids["household_id"].to_numpy())
        person_ids = store.read("person", 0, store.rows("person"),
                                ["person_household_id", *[c for _, c in GROUP_ID.values()]])
        person_position = household_position.reindex(person_ids["person_household_id"].to_numpy()).to_numpy()
        group_positions = {}
        for entity, (id_col, _) in GROUP_ID.items():
            ids = store.read(entity, 0, store.rows(entity), [id_col])[id_col].to_numpy()
            group_positions[entity] = pd.Series(np.arange(len(ids)), index=ids)
        first, last = int(acs_positions[0]), int(acs_positions[-1]) + 1
        for shard, low in enumerate(range(first, last, args.households_per_shard)):
            high = min(low + args.households_per_shard, last)
            wait_for_memory(args.min_available_gb)
            persons = np.flatnonzero((person_position >= low) & (person_position < high))
            part = {"household": store.read("household", low, high),
                    "person": store.read_positions("person", persons)}
            for entity, (_id_col, person_col) in GROUP_ID.items():
                ids = np.unique(person_ids[person_col].to_numpy()[persons])
                rows = np.sort(group_positions[entity].reindex(ids).to_numpy().astype(np.int64))
                part[entity] = store.read_positions(entity, rows)
            weights = part["household"].pop("household_weight").to_numpy(np.float64)
            _write_shard(out_dir / f"shard_{shard:03d}.pkl", part, weights)
            index = _row_index_columns(part, "acs")
            index["design_weight"] = weights
            index["shard"] = shard
            index["position"] = np.arange(high - low)
            index_parts.append(index)
            log(f"acs shard {shard}: {high - low} households, peak {peak_rss_gb():.1f} GB")
            del part
            gc.collect()
    index = pd.concat(index_parts, ignore_index=True)
    index["source"] = args.source
    index.to_parquet(args.out / f"rows_{args.source}.parquet", index=False)
    write_json(out_dir / "shard.json", {
        "source": args.source, "h5": str(args.h5), "h5_sha256": sha256_file(args.h5),
        "design_h5": str(args.design_h5) if args.design_h5 else None,
        "shards": len(index_parts), "households": len(index),
        "wall_s": round(time.time() - started, 1), "peak_rss_gb": round(peak_rss_gb(), 2),
    })


# ------------------------------------------------------------------ materialize


def _load_shard_frame(path: Path):
    from microcosm.build.serialization_dtypes import canonicalize_frame_string_dtypes
    from microcosm.frame import Frame, WeightKind, Weights
    from microcosm.frame.units import US_SCHEMA

    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    frame = Frame(
        payload["tables"],
        US_SCHEMA,
        {"household": Weights(np.asarray(payload["household_weight"]), WeightKind.CALIBRATED)},
    )
    return canonicalize_frame_string_dtypes(frame, boundary="support-mix shard load", in_place=True)


#: Engine inputs the Sep-23 ACS staging carries as all-null on ACS rows and
#: its reviewed-null register predates. ``is_spm_independent_minor_role`` has
#: a formula (``is_household_head | is_household_spouse``, spm_calculator's
#: policyengine adapter); a column that is entirely null is dropped so the
#: engine derives the role from household structure, as it did for the
#: production ACS local release, whose staging had no such column. A
#: partially null column is an error. Counted in every receipt.
FORMULA_DERIVED_WHEN_ALL_NULL = (("person", "is_spm_independent_minor_role"),)


def _drop_all_null_formula_inputs(frame):
    from microcosm.frame import Frame

    dropped = {}
    tables = {entity: frame.table(entity) for entity in frame.entities}
    for entity, column in FORMULA_DERIVED_WHEN_ALL_NULL:
        table = tables[entity]
        if column not in table.columns:
            continue
        missing = table[column].isna()
        if missing.all():
            tables[entity] = table.drop(columns=[column])
            dropped[f"{entity}.{column}"] = int(missing.sum())
        elif missing.any():
            raise RuntimeError(f"{entity}.{column} is partially null ({int(missing.sum())} rows)")
    if not dropped:
        return frame, dropped
    weights = {entity: frame.weights_for(entity) for entity in frame.weighted_entities}
    return Frame(tables, frame.schema, weights, frame.strata, mass_log=frame.mass_log), dropped


def _materialize_frame(frame, concept_specs, *, summary_path: Path | None, batch: int):
    tool = release_tool()
    acs = acs_local_tool()
    projected, dropped = acs.project_input_only(frame, period=PERIOD)
    projected, formula_derived = _drop_all_null_formula_inputs(projected)
    if summary_path is not None:
        acs.fill_reviewed_nulls(projected, summary_path, period=PERIOD)
    target_frame, registry, compilation = tool._materialize_target_frame(
        projected,
        tuple(concept_specs),
        maximum_microsim_batch_size=batch,
        refuse_population_aggregates=True,
    )
    household = target_frame.table("household")
    compilation = {**compilation, "formula_derived_inputs": formula_derived}
    return household, [spec.measure for spec in registry.specs], compilation, dropped


def acs_rank(work: Path) -> np.ndarray:
    """Each ACS row's position in the fixed salted selection order (cached,
    keyed by a digest of the row index's unit-key sequence)."""
    rows = pd.read_parquet(work / "rows_acs.parquet", columns=["unit_key"])
    digest = hashlib.sha256("\n".join(rows["unit_key"]).encode()).hexdigest()
    path, key = work / "acs_rank.npy", work / "acs_rank.key"
    if path.exists() and key.exists() and key.read_text() == digest:
        return np.load(path)
    order = acs_selection_order(rows["unit_key"].tolist(), salt=ACS_SELECTION_SALT)
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order))
    np.save(path, rank)
    key.write_text(digest)
    return rank


def do_materialize(args) -> None:
    with open(args.work / "concept_specs.pkl", "rb") as handle:
        concept_specs = pickle.load(handle)
    measures = [spec.measure for spec in concept_specs] + ["state_income_tax"]
    shard_dir = args.work / "shards" / args.source
    out_dir = args.work / "concepts" / args.source
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = pd.read_parquet(args.work / f"rows_{args.source}.parquet")
    rows["row"] = np.arange(len(rows))
    if "spm_zero_adult_unit" in rows:
        # Group-quarters records whose SPM unit has no classified adult: the
        # pinned engine refuses them, and Census puts ACS group quarters
        # outside the SPM universe (us_runtime/spm_universe_source.py).
        rows = rows[~rows["spm_zero_adult_unit"]]
    low, high = 0, len(rows)
    if args.rank_range:
        low, high = (int(x) for x in args.rank_range.split(","))
        rows["rank"] = acs_rank(args.work)[rows["row"].to_numpy()]
    shards = sorted(shard_dir.glob("shard_*.pkl"))
    if args.only:
        wanted = {int(x) for x in args.only.split(",") if x.strip()}
        shards = [s for s in shards if int(s.stem.split("_")[1]) in wanted]
    suffix = f".r{low}-{high}" if args.rank_range else ""
    for shard_path in shards:
        out = out_dir / f"{shard_path.stem}{suffix}.npz"
        if out.exists():
            continue
        shard_rows = rows[rows["shard"] == int(shard_path.stem.split("_")[1])]
        if args.rank_range:
            shard_rows = shard_rows[(shard_rows["rank"] >= low) & (shard_rows["rank"] < high)]
        if shard_rows.empty:
            continue
        wait_for_memory(args.min_available_gb)
        started = time.time()
        frame = _load_shard_frame(shard_path)
        if len(shard_rows) < frame.n("household"):
            keep = np.isin(frame.table("person")["person_household_id"].to_numpy(),
                           shard_rows["household_id"].to_numpy())
            frame = frame.select(keep)
        household, compiled, compilation, _ = _materialize_frame(
            frame, concept_specs, summary_path=args.summary, batch=args.batch
        )
        position = pd.Series(shard_rows["row"].to_numpy(), index=shard_rows["household_id"].to_numpy())
        row_positions = position.reindex(household["household_id"].to_numpy()).to_numpy()
        if np.isnan(row_positions.astype(float)).any():
            raise RuntimeError(f"{shard_path.name}: materialized households missing from the row index")
        missing = [m for m in measures if m not in household.columns]
        matrix = np.zeros((len(household), len(measures)), dtype=np.float32)
        for j, measure in enumerate(measures):
            if measure in household.columns:
                matrix[:, j] = household[measure].to_numpy(dtype=np.float32)
        csr = sp.csr_matrix(matrix)
        np.save(out.with_suffix(".rows.npy"), row_positions.astype(np.int64))
        sp.save_npz(out.with_suffix(".tmp.npz"), csr)
        out.with_suffix(".tmp.npz").replace(out)
        write_json(out.with_suffix(".json"), {
            "shard": shard_path.name, "rank_range": [low, high] if args.rank_range else None,
            "households": len(household), "nnz": int(csr.nnz),
            "missing_measures": missing, "compiled_concepts": len(compiled),
            "batching": compilation.get("target_materialization_batching"),
            "formula_derived_inputs": compilation.get("formula_derived_inputs"),
            "wall_s": round(time.time() - started, 1), "peak_rss_gb": round(peak_rss_gb(), 2),
            "cpu_s": round(resource.getrusage(resource.RUSAGE_SELF).ru_utime, 1),
        })
        log(f"{args.source} {out.name}: {len(household)} hh, nnz {csr.nnz:,}, "
            f"missing {len(missing)}, {time.time() - started:.0f}s, peak {peak_rss_gb():.1f} GB")
        del frame, household, matrix, csr
        gc.collect()
    write_json(out_dir / "measures.json", measures)


# ------------------------------------------------------------------ diffcheck


def do_diffcheck(args) -> None:
    """Differential: production per-spec columns vs concept x geography mask."""
    started = time.time()
    from microcosm.build.ledger_artifact import load_ledger_consumer_artifact
    from microcosm.build.us_runtime import (
        apply_us_medicaid_enrollment_substitutions,
        compile_us_fiscal_target_registry,
        default_congressional_district_vintage_crosswalk_path,
        load_congressional_district_vintage_crosswalk,
    )

    targets = pd.read_parquet(args.work / "targets.parquet")
    with open(args.work / "concept_specs.pkl", "rb") as handle:
        concept_specs = pickle.load(handle)
    artifact = load_ledger_consumer_artifact(args.feed)
    registry = compile_us_fiscal_target_registry(
        artifact.facts, target_period=PERIOD,
        congressional_district_vintage_crosswalk=load_congressional_district_vintage_crosswalk(
            default_congressional_district_vintage_crosswalk_path()),
        age_targets=True)
    registry, _ = apply_us_medicaid_enrollment_substitutions(registry)
    specs = registry.specs
    usable = targets[targets["excluded"].isna() & (targets["level"] != "national")]
    rng = np.random.default_rng(20260928)
    sample = []
    for _key, group in usable.groupby(["family", "level"]):
        take = min(len(group), args.per_group)
        sample.extend(rng.choice(group["target_index"].to_numpy(), take, replace=False).tolist())
    sample_specs = [specs[i] for i in sorted(sample)]
    results = {}
    previous = json.loads((args.work / "diffcheck.json").read_text())["results"] \
        if (args.work / "diffcheck.json").exists() else {}
    for source in args.sources.split(","):
        frame = _load_shard_frame(args.work / "shards" / source / "shard_000.pkl")
        hh_ids = frame.table("household")["household_id"].to_numpy()
        index = pd.read_parquet(args.work / f"rows_{source}.parquet")
        if "spm_zero_adult_unit" in index:
            flagged = index.loc[(index["shard"] == 0) & index["spm_zero_adult_unit"], "household_id"]
            hh_ids = hh_ids[~np.isin(hh_ids, flagged.to_numpy())]
        hh_ids = hh_ids[: args.households]
        person_mask = np.isin(frame.table("person")["person_household_id"].to_numpy(), hh_ids)
        frame = frame.select(person_mask)
        summary = args.summary if source == "acs" else None
        direct, _, _, _ = _materialize_frame(frame, sample_specs, summary_path=summary, batch=args.batch)
        concept, _, _, _ = _materialize_frame(frame, concept_specs, summary_path=summary, batch=args.batch)
        state = pd.to_numeric(direct["state_fips"]).to_numpy()
        cd = pd.to_numeric(direct["congressional_district_geoid"]).to_numpy()
        worst, checked, skipped = 0.0, 0, 0
        per_family = {}
        by_name = targets.set_index("name")
        for spec in sample_specs:
            row = by_name.loc[spec.name]
            if spec.measure not in direct.columns or row["measure"] not in concept.columns:
                skipped += 1
                continue
            mask = np.ones(len(direct), bool)
            if row["state_fips"] >= 0:
                mask &= state == row["state_fips"]
            if row["cd_geoid"] >= 0:
                mask &= cd == row["cd_geoid"]
            expected = direct[spec.measure].to_numpy(np.float64)
            got = concept[row["measure"]].to_numpy(np.float64) * mask
            scale = max(1.0, float(np.abs(expected).max()))
            diff = float(np.abs(expected - got).max()) / scale
            worst = max(worst, diff)
            key = f"{spec.family}/{row['level']}"
            per_family[key] = max(per_family.get(key, 0.0), diff)
            checked += 1
        results[source] = {"households": len(direct), "specs_checked": checked,
                           "specs_skipped": skipped, "max_scaled_abs_diff": worst,
                           "per_family_level": per_family}
        log(f"diffcheck {source}: {json.dumps(results[source])}")
    write_json(args.work / "diffcheck.json", {
        "results": {**previous, **results}, "sample_size": len(sample_specs),
        "wall_s": round(time.time() - started, 1), "peak_rss_gb": round(peak_rss_gb(), 2)})


# ------------------------------------------------------------------ truth

ACS_TRUTH_CELLS = {  # ACS 2024 1-year table-based summary file cells (E = estimate)
    "households": ["B25003_E001"],
    "owner": ["B25003_E002"],
    "renter": ["B25003_E003"],
    "rent_contract_monthly_aggregate": ["B25060_E001"],
    "rent_gross_monthly_aggregate": ["B25065_E001"],
    "real_estate_taxes_aggregate": ["B25090_E001"],
    "persons_in_households": ["B25008_E001"],
    "persons_total": ["B01003_E001"],
    "persons_male": ["B01001_E002"],
    "persons_0_17": [f"B01001_E{i:03d}" for i in (3, 4, 5, 6, 27, 28, 29, 30)],
    "persons_65p": [f"B01001_E{i:03d}" for i in (*range(20, 26), *range(44, 50))],
}
SUMMARY_LEVELS = {"0100000US": "national", "0400000US": "state", "0500000US": "county", "5001900US": "cd"}


def do_truth(args) -> None:
    """Long truth table from the ACS 2024 1-year table-based summary file."""
    tables = sorted({cell.split("_")[0] for cells in ACS_TRUTH_CELLS.values() for cell in cells})
    frames, digests = {}, {}
    for table in tables:
        path = args.sf_dir / f"acsdt1y2024-{table.lower()}.dat"
        digests[path.name] = sha256_file(path)
        frames[table] = pd.read_csv(path, sep="|", dtype=str).set_index("GEO_ID")
    records = []
    for measure, cells in ACS_TRUTH_CELLS.items():
        table = cells[0].split("_")[0]
        frame = frames[table]
        values = frame[cells].apply(pd.to_numeric, errors="coerce")
        total = values.sum(axis=1, min_count=len(cells))
        for geo_id, value in total.items():
            prefix, code = geo_id[:9], geo_id[9:]
            level = SUMMARY_LEVELS.get(prefix)
            if level is None or not np.isfinite(value) or value < 0:
                continue
            if level == "national":
                geo = 0
            elif level == "cd":
                district = code[2:]
                geo = int(code[:2]) * 100 + (0 if district == "98" else int(district))
            else:
                geo = int(code)
            if level in ("state", "county", "cd") and int(code[:2]) == 72:
                continue  # Puerto Rico is outside the pool
            records.append((level, geo, measure, float(value)))
    truth = pd.DataFrame(records, columns=["level", "geo", "measure", "value"])
    wide = truth.pivot_table(index=["level", "geo"], columns="measure", values="value").reset_index()
    wide["persons_18_64"] = wide["persons_total"] - wide["persons_0_17"] - wide["persons_65p"]
    wide["rent_contract_annual_aggregate"] = 12 * wide.pop("rent_contract_monthly_aggregate")
    wide["rent_gross_annual_aggregate"] = 12 * wide.pop("rent_gross_monthly_aggregate")
    truth = wide.melt(id_vars=["level", "geo"], var_name="measure", value_name="value").dropna()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    truth.to_parquet(args.out, index=False)
    write_json(args.out.with_suffix(".json"), {
        "source": "ACS 2024 1-year table-based summary file (Census), "
                  "https://www2.census.gov/programs-surveys/acs/summary_file/2024/table-based-SF/",
        "files_sha256": digests, "cells": ACS_TRUTH_CELLS,
        "rows_by_level": truth.groupby("level").size().to_dict(),
    })
    log(f"truth: {truth.groupby('level').size().to_dict()}")


def truth_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


# ------------------------------------------------------------------ arms

BUDGETS = (300_000, 600_000, 1_200_000)
YEAR_SETS = ((2024,), (2023, 2024), (2022, 2023, 2024))


def arm_grid(include_largest: bool = True) -> list[SupportMixArm]:
    arms: list[SupportMixArm] = []
    for years in YEAR_SETS:
        arms.append(SupportMixArm(budget=None, cps_income_years=years))
    for budget in BUDGETS if include_largest else BUDGETS[:2]:
        for years in YEAR_SETS:
            arms.append(SupportMixArm(budget=budget, cps_income_years=years, acs_fill_share=1.0))
            arms.append(SupportMixArm(budget=budget, cps_income_years=years, acs_fill_share=0.0))
        arms.append(SupportMixArm(budget=budget, cps_income_years=(), acs_fill_share=1.0))
    for years in ((2024,), (2022, 2023, 2024)):
        arms.append(SupportMixArm(budget=600_000, cps_income_years=years, acs_fill_share=0.5))
        for seed in (1, 2):
            arms.append(SupportMixArm(budget=300_000, cps_income_years=years, acs_fill_share=1.0, seed=seed))
    return arms


def parse_arm(label: str) -> SupportMixArm:
    budget, years, acs, seed = label.split(".")
    return SupportMixArm(
        budget=None if budget == "natural" else int(budget[1:-1]) * 1000,
        cps_income_years=() if years == "cps0" else tuple(int(y) for y in years[3:].split("-")),
        acs_fill_share=int(acs[3:]) / 100,
        seed=int(seed[1:]),
    )


class _ConceptStore:
    """Row-addressable concept values for one source (CSR, float32).

    Each materialized part carries its rows' global positions; rows never
    materialized are all-zero and flagged, and an arm refuses to use them.
    Parts must not overlap: a row materialized twice would be summed.
    """

    def __init__(self, work: Path, source: str):
        directory = work / "concepts" / source
        shard_of = pd.read_parquet(work / f"rows_{source}.parquet", columns=["shard"])["shard"].to_numpy()
        n_rows = len(shard_of)
        self.measures = json.loads((directory / "measures.json").read_text())
        rows_parts, parts = [], []
        self.materialized = np.zeros(n_rows, dtype=bool)
        for path in sorted(directory.glob("shard_*.npz")):
            if path.name.endswith(".tmp.npz"):
                continue
            part = sp.load_npz(path).tocoo()
            rows_path = path.with_suffix(".rows.npy")
            rows = (np.load(rows_path) if rows_path.exists()
                    else np.flatnonzero(shard_of == int(path.stem.split("_")[1].split(".")[0])))
            if part.shape[0] != len(rows):
                raise RuntimeError(f"{path.name}: {part.shape[0]} rows but {len(rows)} positions")
            if self.materialized[rows].any():
                raise RuntimeError(f"{path.name}: overlaps rows already materialized by another part")
            self.materialized[rows] = True
            rows_parts.append(rows[part.row])
            parts.append(part)
        self.matrix = sp.csr_matrix(
            (np.concatenate([p.data for p in parts]).astype(np.float32),
             (np.concatenate(rows_parts), np.concatenate([p.col for p in parts]))),
            shape=(n_rows, len(self.measures)),
        )


class _SparseOperator:
    """``A @ w`` with a precomputed transpose for the backward pass.

    Duck-types the torch tensor ``_optimize`` expects: it reads ``.layout``
    (not sparse CSR, so ``_apply_constraint`` calls ``matrix @ weights``) and
    ``@``. Forward and backward are one CSR SpMV each.
    """

    def __init__(self, matrix: sp.csr_matrix):
        import torch

        self.torch = torch
        self.layout = torch.strided
        self.shape = matrix.shape
        self.A = self._tensor(matrix)
        self.AT = self._tensor(matrix.T.tocsr())

    def _tensor(self, m):
        torch = self.torch
        m = m.astype(np.float32)
        return torch.sparse_csr_tensor(
            torch.from_numpy(m.indptr.astype(np.int64)),
            torch.from_numpy(m.indices.astype(np.int64)),
            torch.from_numpy(m.data),
            size=m.shape,
        )

    def __matmul__(self, weights):
        return _SpMV.apply(weights, self)


def _spmv_class():
    import torch

    class SpMV(torch.autograd.Function):
        @staticmethod
        def forward(ctx, weights, operator):
            ctx.operator = operator
            return (operator.A @ weights.unsqueeze(1)).squeeze(1)

        @staticmethod
        def backward(ctx, grad):
            return (ctx.operator.AT @ grad.unsqueeze(1)).squeeze(1), None

    return SpMV


_SpMV = None


def _assemble(values: sp.csr_matrix, geo: dict[str, np.ndarray], frame: pd.DataFrame,
              state_code: dict, cd_code: dict) -> sp.csr_matrix:
    """Targets x rows for a target frame (see ``assemble_target_matrix``)."""
    levels = frame["level"].to_numpy()
    target_geo = np.zeros(len(frame), dtype=np.int64)
    is_state, is_cd = levels == "state", levels == "cd"
    target_geo[is_state] = [state_code[x] for x in frame["state_fips"].to_numpy()[is_state]]
    target_geo[is_cd] = [cd_code[x] for x in frame["cd_geoid"].to_numpy()[is_cd]]
    return assemble_target_matrix(
        values, geo, frame["col"].to_numpy(), levels, target_geo
    ).tocsr()


def do_arm(args) -> None:
    global _SpMV
    started = time.time()
    arm = parse_arm(args.arm)
    out = args.work / "arms" / args.product / f"{arm.label}.json"
    previous = json.loads(out.read_text()) if out.exists() else None
    if previous is not None and previous.get("scored_all_levels") and not args.force:
        log(f"{out} exists")
        return
    rescore = previous is not None and out.with_suffix(".weights.npy").exists() and not args.force
    wait_for_memory(args.min_available_gb)
    import torch

    torch.set_num_threads(args.threads)
    _SpMV = _spmv_class()
    from microcosm.calibrate.solve import _optimize, default_target_loss_scales

    targets = pd.read_parquet(args.work / "targets.parquet")
    rows_cps = pd.read_parquet(args.work / "rows_cps.parquet")
    rows_acs = pd.read_parquet(args.work / "rows_acs.parquet")
    # "row" is the position in the source's concept store; it survives filters.
    rows_cps["row"] = np.arange(len(rows_cps))
    rows_acs["row"] = np.arange(len(rows_acs))
    if args.exclude_gq:
        rows_acs = rows_acs[~rows_acs["group_quarters"]].reset_index(drop=True)
    if "spm_zero_adult_unit" in rows_acs:
        rows_acs = rows_acs[~rows_acs["spm_zero_adult_unit"]].reset_index(drop=True)
    # ---------------- rows
    asec = rows_cps[rows_cps["channel"] == "asec"]
    cps_hh = asec.groupby("income_year").size().to_dict()
    if (asec.groupby("source_key").size() != 1).any() or rows_cps["unit_key"].str.endswith("nan").any():
        raise SystemExit("CPS source keys are not one ASEC row per source household")
    if rows_cps.groupby("income_year")["source_key"].nunique().to_dict() != cps_hh:
        raise SystemExit("CPS source keys disagree with ASEC household counts")
    for name, frame in (("cps", rows_cps), ("acs", rows_acs)):
        if not (frame["owner"].mean() > 0.3 and frame["renter"].mean() > 0.1):
            raise SystemExit(f"{name} tenure columns look unparsed (owner/renter shares)")
    counts = plan_arm_counts(arm, cps_households_by_year=cps_hh, acs_households=len(rows_acs))
    cps_rows = rows_cps[rows_cps["income_year"].isin(arm.cps_income_years)].reset_index(drop=True)
    source_keys = cps_rows["source_key"].drop_duplicates().to_numpy()
    copies_by_source = dict(zip(source_keys, clone_copies(
        list(source_keys), counts.clones, salt=f"{CLONE_SALT}.s{arm.seed}"), strict=True))
    cps_copies = cps_rows["source_key"].map(copies_by_source).fillna(0).to_numpy(np.int64) \
        if len(cps_rows) else np.zeros(0, np.int64)
    # One fixed order; replicate r takes the r-th disjoint block of it, so
    # replicates are independent draws and seed 0 arms are nested by budget.
    rank = acs_rank(args.work)[rows_acs["row"].to_numpy()]
    order = np.argsort(rank, kind="stable")
    block = order[arm.seed * counts.acs:(arm.seed + 1) * counts.acs]
    if len(block) < counts.acs:
        raise SystemExit(f"{arm.label}: replicate block exceeds the ACS rows available")
    acs_rows = rows_acs.iloc[np.sort(block)].reset_index(drop=True)
    total_mass = float(rows_cps.loc[rows_cps["income_year"] == 2024, "design_weight"].sum() * 3)
    cps_w, acs_w = arm_initial_weights(
        cps_design_weights=cps_rows["design_weight"].to_numpy(),
        cps_income_year=cps_rows["income_year"].to_numpy(),
        cps_copies=cps_copies,
        acs_design_weights=acs_rows["design_weight"].to_numpy(),
        counts=counts,
        total_mass=total_mass,
    )
    # Expand CPS copies; copy 0 keeps its own geography, copy k>0 draws a new
    # (district, county) within its state in proportion to ACS households.
    repeat = 1 + cps_copies
    cps_expanded = cps_rows.loc[cps_rows.index.repeat(repeat)].reset_index(drop=True)
    copy_number = np.concatenate([np.arange(r) for r in repeat]) if len(repeat) else np.zeros(0, int)
    cps_expanded["copy"] = copy_number
    w_cps = np.repeat(cps_w, repeat)
    if counts.clones:
        geo_pool = rows_acs.loc[~rows_acs["group_quarters"].astype(bool),
                                ["state_fips", "cd_geoid", "county_fips", "design_weight"]]
        for state, group in cps_expanded[cps_expanded["copy"] > 0].groupby("state_fips"):
            pool = geo_pool[geo_pool["state_fips"] == state]
            cum = np.cumsum(pool["design_weight"].to_numpy())
            draws = np.array([
                hash_uniform(f"{k}|{r}|{c}", salt=f"{CLONE_GEO_SALT}.s{arm.seed}")
                for k, r, c in zip(group["source_key"], group["row"], group["copy"], strict=True)
            ])
            pick = np.searchsorted(cum, draws * cum[-1], side="right")
            cps_expanded.loc[group.index, "cd_geoid"] = pool["cd_geoid"].to_numpy()[pick]
            cps_expanded.loc[group.index, "county_fips"] = pool["county_fips"].to_numpy()[pick]
    selected = pd.concat([cps_expanded.assign(source="cps"), acs_rows.assign(source="acs", copy=0)],
                         ignore_index=True)
    w0 = np.concatenate([w_cps, acs_w])
    # The release's base population mass repair: rescale so the weighted
    # person count equals the Census benchmark before a conserved solve.
    benchmark = float(release_tool().US_BASE_PERSON_POPULATION_BENCHMARK)
    mass_factor = benchmark / float(selected["persons"].to_numpy() @ w0)
    w0 = w0 * mass_factor
    # ---------------- concept values for the selected rows
    stores = {}
    blocks = []
    for source, frame in (("cps", cps_expanded), ("acs", acs_rows)):
        if not len(frame):
            continue
        if source not in stores:
            stores[source] = _ConceptStore(args.work, source)
        needed = frame["row"].to_numpy()
        unmaterialized = int((~stores[source].materialized[needed]).sum())
        if unmaterialized:
            raise SystemExit(f"{arm.label}: {unmaterialized} {source} rows are not materialized yet")
        blocks.append(stores[source].matrix[needed])
    measures = next(iter(stores.values())).measures
    values = sp.vstack(blocks, format="csr")
    del stores
    gc.collect()
    # ---------------- targets and lookup tables
    measure_col = {m: j for j, m in enumerate(measures)}
    usable = targets[targets["excluded"].isna() & targets["measure"].isin(measure_col)].copy()
    usable["col"] = usable["measure"].map(measure_col)
    # Both products are scored on every held-out target (national, state and
    # district); the national product trains on its surface only.
    holdout = usable[usable["role"] == "holdout"].reset_index(drop=True)
    if args.product == "national":
        usable = usable[usable["in_national_state"]]
    train = usable[usable["role"] == "train"].reset_index(drop=True)
    states = np.unique(selected["state_fips"])
    cds = np.unique(np.concatenate([selected["cd_geoid"].to_numpy(), usable.loc[usable["cd_geoid"] >= 0, "cd_geoid"].to_numpy()]))
    state_code = {s: i for i, s in enumerate(np.unique(np.concatenate([states, usable["state_fips"][usable["state_fips"] >= 0]])))}
    cd_code = {c: i for i, c in enumerate(cds)}
    geo = {"state": selected["state_fips"].map(state_code).to_numpy(),
           "cd": selected["cd_geoid"].map(cd_code).to_numpy()}

    a_train = _assemble(values, geo, train, state_code, cd_code)
    a_hold = _assemble(values, geo, holdout, state_code, cd_code)
    b_train = train["value"].to_numpy(np.float64)
    loss_w = train["loss_weight_national_state" if args.product == "national" else "loss_weight"].to_numpy(np.float64)
    # Local product: district household populations are trained, as in the
    # ACS local release (never scored).
    truth = truth_table(args.truth)
    extra_train = 0
    if args.product == "local":
        # Household population (B25008): group-quarters rows neither count
        # toward nor can be pushed by it, so CPS-only arms are not forced to
        # inflate households to stand in for dorms and nursing homes.
        cd_truth = truth[(truth["level"] == "cd") & (truth["measure"] == "persons_in_households")]
        cd_truth = cd_truth[cd_truth["geo"].isin(cd_code)]
        persons = selected["persons"].to_numpy(np.float64) * ~selected["group_quarters"].astype(bool).to_numpy()
        codes = selected["cd_geoid"].map({g: i for i, g in enumerate(cd_truth["geo"])}).fillna(-1).to_numpy().astype(np.int64)
        keep = (codes >= 0) & (persons > 0)
        pop = sp.csr_matrix((persons[keep], (codes[keep], np.flatnonzero(keep))), shape=(len(cd_truth), len(selected)))
        a_train = sp.vstack([a_train, pop], format="csr")
        b_train = np.concatenate([b_train, cd_truth["value"].to_numpy(np.float64)])
        loss_w = np.concatenate([loss_w, np.ones(len(cd_truth))])
        extra_train = len(cd_truth)
    # Drop training rows no selected household can move (zero rows).
    nonzero = np.diff(a_train.indptr) > 0
    a_train, b_train, loss_w = a_train[nonzero], b_train[nonzero], loss_w[nonzero]
    scales = default_target_loss_scales(b_train)
    solve_started = time.time()
    if rescore:
        # Re-score saved weights (a scoring change, no re-solve).
        weights = np.load(out.with_suffix(".weights.npy")).astype(np.float64)
        if len(weights) != len(w0):
            raise SystemExit(f"{arm.label}: saved weights do not match the arm's rows")
        trajectory = np.array([previous["train"]["loss_initial"], previous["train"]["loss_final"]])
    else:
        operator = _SparseOperator(a_train)
        weights, trajectory = _optimize(
            operator,
            torch.tensor(b_train, dtype=torch.float32),
            torch.tensor(loss_w, dtype=torch.float32),
            torch.tensor(scales, dtype=torch.float32),
            1.0,
            w0,
            epochs=args.epochs,
            learning_rate=0.02,
            conserve_mass=True,
            max_weight_ratio=5.0,
            l0_lambda=0.0,
            l2_lambda=0.0,
            target_records=None,
            init_mean=0.999,
            temperature=0.25,
        )
        del operator
    solve_s = time.time() - solve_started if not rescore else previous["runtime"]["solve_s"]
    gc.collect()
    # ---------------- scoring
    def errors(matrix, target_values, w):
        est = matrix @ w
        rel = np.abs(est - target_values) / np.maximum(np.abs(target_values), 1.0)
        return est, rel

    result_targets = []
    est0, rel0 = errors(a_hold, holdout["value"].to_numpy(), w0)
    est1, rel1 = errors(a_hold, holdout["value"].to_numpy(), weights)
    supported = np.diff(a_hold.indptr) > 0
    for i, row in enumerate(holdout.itertuples(index=False)):
        result_targets.append({
            "name": row.name, "family": row.family, "level": row.level,
            "dimension_group": row.dimension_group, "dimension": row.dimension,
            "loss_weight": row.loss_weight, "target": row.value,
            "estimate_initial": float(est0[i]), "estimate": float(est1[i]),
            "rel_error_initial": float(rel0[i]), "rel_error": float(rel1[i]),
            "supported": bool(supported[i]),
        })
    # ACS-native evaluation and report-only measures from row columns.
    acs_eval = []
    household_rows = ~selected["group_quarters"].astype(bool).to_numpy()
    measures_row = {
        "households": household_rows.astype(float),
        "owner": selected["owner"].to_numpy() * household_rows,
        "renter": selected["renter"].to_numpy() * household_rows,
        "rent_contract_annual_aggregate": selected["rent_annual"].to_numpy() * household_rows,
        "rent_gross_annual_aggregate": selected["rent_annual"].to_numpy() * household_rows,
        "real_estate_taxes_aggregate": selected["real_estate_taxes"].to_numpy() * household_rows * selected["owner"].to_numpy(),
        "persons_in_households": selected["persons"].to_numpy() * household_rows,
        "persons_total": selected["persons"].to_numpy(),
        "persons_0_17": selected["persons_0_17"].to_numpy(),
        "persons_18_64": selected["persons_18_64"].to_numpy(),
        "persons_65p": selected["persons_65p"].to_numpy(),
        "persons_male": selected["persons_male"].to_numpy(),
    }
    geo_cols = {"national": np.zeros(len(selected), np.int64),
                "state": selected["state_fips"].to_numpy(np.int64),
                "cd": selected["cd_geoid"].to_numpy(np.int64),
                "county": selected["county_fips"].to_numpy(np.int64)}
    for level, codes in geo_cols.items():
        level_truth = truth[truth["level"] == level]
        uniq, inverse = np.unique(codes, return_inverse=True)
        for measure, vec in measures_row.items():
            t = level_truth[level_truth["measure"] == measure]
            if t.empty:
                continue
            e0 = np.bincount(inverse, weights=vec * w0, minlength=len(uniq))
            e1 = np.bincount(inverse, weights=vec * weights, minlength=len(uniq))
            est0_map = dict(zip(uniq, e0, strict=True))
            est1_map = dict(zip(uniq, e1, strict=True))
            for geo_code, value in zip(t["geo"], t["value"], strict=True):
                if (level == "cd" and args.product == "local" and measure == "persons_in_households"):
                    continue  # trained
                acs_eval.append((level, measure, int(geo_code), float(value),
                                 float(est0_map.get(geo_code, 0.0)), float(est1_map.get(geo_code, 0.0))))
    acs_eval = pd.DataFrame(acs_eval, columns=["level", "measure", "geo", "truth", "estimate_initial", "estimate"])
    report = {}
    for name in REPORT_SPECS:
        measure = f"bakeoff_{name.replace('.', '_')}"
        if measure in measure_col:
            col = values[:, measure_col[measure]].toarray().ravel()
            report[name] = {"initial": float(col @ w0), "calibrated": float(col @ weights)}
    persons = selected["persons"].to_numpy()
    children = selected["persons_0_17"].to_numpy()
    if "report.spm_poor_persons" in report:
        report["spm_rate"] = {k: report["report.spm_poor_persons"][k] / float(persons @ w) for k, w in (("initial", w0), ("calibrated", weights))}
        report["spm_child_rate"] = {k: report["report.spm_poor_children"][k] / float(children @ w) for k, w in (("initial", w0), ("calibrated", weights))}
    # ---------------- ESS and receipt
    units = selected["unit_key"].to_numpy()
    source_units = (selected["source_key"].astype(str) + "#" + selected["copy"].astype(str)).to_numpy()
    cd_ess = []
    for _cd, idx in pd.Series(np.arange(len(selected))).groupby(selected["cd_geoid"].to_numpy()):
        cd_ess.append(distinct_unit_ess(weights[idx.to_numpy()], units[idx.to_numpy()]))
    receipt = {
        "arm": arm.label, "product": args.product, "scored_all_levels": True,
        "rescored": bool(rescore),
        "counts": dataclasses.asdict(counts),
        "rows": {"cps_physical": int(len(cps_expanded)), "acs": int(len(acs_rows)), "total": int(len(selected))},
        "distinct_households": int(pd.Series(units).nunique()),
        "ess": {
            "rows_initial": kish_ess(w0), "rows": kish_ess(weights),
            "distinct_households_initial": distinct_unit_ess(w0, units),
            "distinct_households": distinct_unit_ess(weights, units),
            "source_records": distinct_unit_ess(weights, source_units),
            "cd_distinct_median": float(np.median(cd_ess)), "cd_distinct_p10": float(np.percentile(cd_ess, 10)),
        },
        "train": {"targets": int(a_train.shape[0]), "cd_population_targets": extra_train,
                  "nnz": int(a_train.nnz),
                  "loss_initial": float(trajectory[0]), "loss_final": float(trajectory[-1]),
                  "loss_at": (previous["train"]["loss_at"] if rescore else
                              {str(e): float(trajectory[min(e, len(trajectory)) - 1]) for e in (100, 250, 500, 1000, len(trajectory))})},
        "holdout_targets": result_targets,
        "acs_native": acs_eval.to_dict(orient="records"),
        "report_only": report,
        "weights": {"mass": float(weights.sum()), "population_mass_factor": mass_factor, "max_ratio": float((weights / w0).max()),
                    "min_ratio": float((weights / w0).min())},
        "runtime": {"wall_s": round(time.time() - started, 1), "solve_s": round(solve_s, 1),
                    "epochs": args.epochs, "threads": args.threads,
                    "peak_rss_gb": round(peak_rss_gb(), 2)},
    }
    write_json(out, receipt)
    np.save(out.with_suffix(".weights.npy"), weights.astype(np.float32))
    log(f"{args.product} {arm.label}: rows {len(selected):,} train {a_train.shape[0]} "
        f"loss {trajectory[0]:.4f}->{trajectory[-1]:.4f} solve {solve_s:.0f}s peak {peak_rss_gb():.1f} GB")


def do_run_grid(args) -> None:
    """Run every missing (arm, product) receipt, one subprocess each."""
    import subprocess

    arms = [a.label for a in arm_grid(not args.no_largest)]
    if args.only:
        arms = [a for a in arms if any(token in a for token in args.only.split(","))]
    for product in args.products.split(","):
        for label in arms:
            out = args.work / "arms" / product / f"{label}.json"
            if out.exists() and json.loads(out.read_text()).get("scored_all_levels"):
                continue
            wait_for_memory(args.min_available_gb)
            command = [sys.executable, str(Path(__file__).resolve()), "arm", "--work", str(args.work),
                       "--arm", label, "--product", product, "--truth", str(args.truth),
                       "--epochs", str(args.epochs), "--threads", str(args.threads),
                       "--min-available-gb", str(args.min_available_gb)]
            log(f"run {product} {label}")
            result = subprocess.run(command, check=False)
            if result.returncode:
                log(f"FAILED {product} {label}: exit {result.returncode}")


# ------------------------------------------------------------------ report


def do_report(args) -> None:
    records, acs_records, summary = [], [], []
    for path in sorted((args.work / "arms").glob("*/*.json")):
        receipt = json.loads(path.read_text())
        base = {"arm": receipt["arm"], "product": receipt["product"]}
        arm = parse_arm(receipt["arm"])
        base.update({"budget": arm.budget or 0, "cps_years": len(arm.cps_income_years),
                     "acs_fill_share": arm.acs_fill_share, "seed": arm.seed})
        for t in receipt["holdout_targets"]:
            records.append({**base, **t})
        for a in receipt["acs_native"]:
            acs_records.append({**base, **a})
        summary.append({**base, **receipt["counts"], **{f"rows_{k}": v for k, v in receipt["rows"].items()},
                        "distinct_households": receipt["distinct_households"],
                        **{f"ess_{k}": v for k, v in receipt["ess"].items()},
                        "train_loss_final": receipt["train"]["loss_final"],
                        "train_targets": receipt["train"]["targets"],
                        **{f"runtime_{k}": v for k, v in receipt["runtime"].items()},
                        **{f"report_{k}": (v["calibrated"] if isinstance(v, dict) else v)
                           for k, v in receipt["report_only"].items()}})
    holdout = pd.DataFrame(records)
    acs_native = pd.DataFrame(acs_records)
    arms = pd.DataFrame(summary)
    args.out.mkdir(parents=True, exist_ok=True)
    arms.to_csv(args.out / "arms.csv", index=False)
    if not holdout.empty:
        holdout["capped"] = holdout["rel_error"].clip(upper=1.0)
        holdout["capped_initial"] = holdout["rel_error_initial"].clip(upper=1.0)
        # Unsupported targets (no row can move them) are scored as misses
        # (estimate 0 -> capped error 1), so every arm is averaged over the
        # same held-out set; an intersection table is written as well.
        supported_everywhere = holdout.groupby(["product", "name"])["supported"].transform("all")
        holdout[supported_everywhere].groupby(
            ["product", "arm", "dimension_group", "dimension", "level"]
        ).agg(targets=("capped", "size"), capped_mean=("capped", "mean")).reset_index().to_csv(
            args.out / "holdout_by_dimension_level_common_support.csv", index=False)
        grouped = holdout.groupby(
            ["product", "arm", "dimension_group", "dimension", "level"])
        table = grouped.apply(lambda g: pd.Series({
            "targets": len(g),
            "capped_mean": g["capped"].mean(),
            "weighted_capped_mean": np.average(g["capped"], weights=g["loss_weight"]),
            "median_rel": g["rel_error"].median(),
            "capped_mean_initial": g["capped_initial"].mean(),
            "unsupported": int((~g["supported"]).sum()),
        })).reset_index()
        table.to_csv(args.out / "holdout_by_dimension_level.csv", index=False)
        unsupported = holdout[~holdout["supported"]].groupby(["product", "arm"]).size()
        unsupported.to_csv(args.out / "holdout_unsupported.csv")
    if not acs_native.empty:
        acs_native["rel"] = (acs_native["estimate"] - acs_native["truth"]).abs() / acs_native["truth"].clip(lower=1.0)
        acs_native["rel_initial"] = (acs_native["estimate_initial"] - acs_native["truth"]).abs() / acs_native["truth"].clip(lower=1.0)
        acs_table = acs_native.groupby(["product", "arm", "level", "measure"]).agg(
            geos=("geo", "size"), mean_rel=("rel", lambda s: s.clip(upper=1.0).mean()),
            median_rel=("rel", "median"), mean_rel_initial=("rel_initial", lambda s: s.clip(upper=1.0).mean()),
        ).reset_index()
        acs_table.to_csv(args.out / "acs_native_by_level_measure.csv", index=False)
    if not holdout.empty and not acs_native.empty:
        _write_headline(args.out, arms, holdout, acs_native)
        _write_dimensions(args.out, arms, holdout, acs_native)
    log(f"report: {len(arms)} arm receipts")


DIMENSION_COLUMNS = (
    ("income_components", "state"), ("income_components", "cd"),
    ("tax_items", "state"), ("tax_items", "cd"),
    ("program_receipt", "state"), ("demographics_pep", "state"),
)


def _write_dimensions(out: Path, arms: pd.DataFrame, holdout: pd.DataFrame, acs_native: pd.DataFrame) -> None:
    """Per-dimension held-out error (seed-0 arms; replicate spread alongside)."""
    by_dim = holdout.groupby(["product", "arm", "dimension", "level"])["capped"].mean()
    acs = acs_native[acs_native["measure"].isin(HEADLINE_ACS_MEASURES)].copy()
    acs["capped"] = acs["rel"].clip(upper=1.0)
    acs = acs.groupby(["product", "arm", "level"])["capped"].mean()
    rows = []
    for record in arms.itertuples(index=False):
        row = {"product": record.product, "arm": record.arm, "budget": record.budget,
               "cps_years": record.cps_years, "acs_fill_share": record.acs_fill_share,
               "seed": record.seed, "rows": record.rows_total,
               "ess_distinct": record.ess_distinct_households}
        for dimension, level in DIMENSION_COLUMNS:
            row[f"{dimension}.{level}"] = by_dim.get((record.product, record.arm, dimension, level), np.nan)
        for level in ("state", "cd", "county"):
            row[f"acs_native.{level}"] = acs.get((record.product, record.arm, level), np.nan)
        rows.append(row)
    table = pd.DataFrame(rows).sort_values(["product", "budget", "cps_years", "acs_fill_share", "seed"])
    table.to_csv(out / "dimensions_by_arm.csv", index=False)
    replicate = table[table["arm"].str.contains("b300k") & table["acs_fill_share"].eq(1.0) & table["cps_years"].gt(0)]
    spread = replicate.groupby(["product", "cps_years"])[[c for c in table.columns if "." in c]].std()
    spread.to_csv(out / "replicate_spread.csv")


HEADLINE_ACS_MEASURES = (
    "households", "owner", "renter", "rent_contract_annual_aggregate",
    "real_estate_taxes_aggregate", "persons_in_households", "persons_0_17", "persons_65p",
)


def _write_headline(out: Path, arms: pd.DataFrame, holdout: pd.DataFrame, acs_native: pd.DataFrame) -> None:
    """One markdown table per product: held-out error by dimension and level."""
    cps = holdout[holdout["dimension_group"] == "cps_native"].groupby(["product", "arm", "level"])["capped"].mean()
    pep = holdout[holdout["dimension_group"] == "acs_native"].groupby(["product", "arm"])["capped"].mean()
    acs = acs_native[acs_native["measure"].isin(HEADLINE_ACS_MEASURES)].copy()
    acs["capped"] = acs["rel"].clip(upper=1.0)
    acs = acs.groupby(["product", "arm", "level"])["capped"].mean()
    lines = []
    for product, frame in arms.sort_values(["budget", "cps_years", "acs_fill_share", "seed"]).groupby("product", sort=False):
        lines += [f"### {product} product", "",
                  "| Arm | Rows | Distinct hh | ESS (distinct) | CPS-native nat | state | CD | PEP demog | ACS-native state | CD | county | Solve min | Peak GB |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for row in frame.itertuples(index=False):
            key = (product, row.arm)

            def cell(series, level=None, key=key):
                index = (*key, level) if level else key
                return f"{series.get(index, float('nan')):.3f}"

            lines.append(
                f"| {row.arm} | {row.rows_total:,} | {row.distinct_households:,} | {row.ess_distinct_households:,.0f} | "
                f"{cell(cps, 'national')} | {cell(cps, 'state')} | {cell(cps, 'cd')} | {cell(pep)} | "
                f"{cell(acs, 'state')} | {cell(acs, 'cd')} | {cell(acs, 'county')} | "
                f"{row.runtime_solve_s / 60:.1f} | {row.runtime_peak_rss_gb:.1f} |"
            )
        lines.append("")
    (out / "headline.md").write_text("\n".join(lines))


# ------------------------------------------------------------------ main


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("compile")
    c.add_argument("--feed", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("shard")
    s.add_argument("--source", choices=("cps", "acs"), required=True)
    s.add_argument("--h5", type=Path, required=True)
    s.add_argument("--design-h5", type=Path)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--households-per-shard", type=int, default=50_000)
    s.add_argument("--min-available-gb", type=float, default=20.0)
    m = sub.add_parser("materialize")
    m.add_argument("--work", type=Path, required=True)
    m.add_argument("--source", choices=("cps", "acs"), required=True)
    m.add_argument("--summary", type=Path)
    m.add_argument("--batch", type=int, default=5_000)
    m.add_argument("--only")
    m.add_argument("--rank-range", help="ACS only: materialize rows whose selection rank is in [LO,HI)")
    m.add_argument("--min-available-gb", type=float, default=20.0)
    d = sub.add_parser("diffcheck")
    d.add_argument("--work", type=Path, required=True)
    d.add_argument("--feed", type=Path, required=True)
    d.add_argument("--summary", type=Path, required=True)
    d.add_argument("--households", type=int, default=4_000)
    d.add_argument("--per-group", type=int, default=40)
    d.add_argument("--batch", type=int, default=5_000)
    d.add_argument("--sources", default="cps,acs")
    t = sub.add_parser("truth")
    t.add_argument("--sf-dir", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    a = sub.add_parser("arm")
    a.add_argument("--work", type=Path, required=True)
    a.add_argument("--arm", required=True)
    a.add_argument("--product", choices=("national", "local"), required=True)
    a.add_argument("--truth", type=Path, required=True)
    a.add_argument("--epochs", type=int, default=1_500)
    a.add_argument("--threads", type=int, default=6)
    a.add_argument("--exclude-gq", action="store_true")
    a.add_argument("--force", action="store_true")
    a.add_argument("--min-available-gb", type=float, default=20.0)
    g = sub.add_parser("grid")
    g.add_argument("--no-largest", action="store_true")
    rg = sub.add_parser("run-grid")
    rg.add_argument("--work", type=Path, required=True)
    rg.add_argument("--truth", type=Path, required=True)
    rg.add_argument("--products", default="national,local")
    rg.add_argument("--epochs", type=int, default=1_500)
    rg.add_argument("--threads", type=int, default=6)
    rg.add_argument("--only")
    rg.add_argument("--no-largest", action="store_true")
    rg.add_argument("--min-available-gb", type=float, default=20.0)
    r = sub.add_parser("report")
    r.add_argument("--work", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    {
        "compile": do_compile, "shard": do_shard, "materialize": do_materialize,
        "diffcheck": do_diffcheck, "truth": do_truth, "arm": do_arm, "report": do_report,
        "run-grid": do_run_grid,
        "grid": lambda a: print("\n".join(arm.label for arm in arm_grid(not a.no_largest))),
    }[args.command](args)


if __name__ == "__main__":
    main()
