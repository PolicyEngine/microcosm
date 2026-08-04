"""Materialize, calibrate, finalize, and package the US ACS local-area release.

The committed successor of the Build L runtime chain
(`national_combined_calibrate.py` + `materialize_admin.py` +
`finalize_artifact.py` + `package_release.py`), so a lineage refresh is one
command instead of a hand-carried script directory. Input: the multispine
staging H5 from ``tools/build_us_acs_multispine_base.py`` (donor spine +
ACS 2024 1-year spine, nullable, simulation-ready except calibration).

Stages (``--stage all`` runs materialize -> calibrate -> qa -> finalize ->
package; each is separately resumable):

  materialize : compile the state-level administrative surface from the
                Ledger feed exactly like the production path
                (``compile_us_fiscal_target_registry`` -> RI Medicaid
                substitution -> state {usda_snap, cms_medicaid[enrollment],
                irs_soi}), run the household-chunked engine pass under the
                nullable-artifact contract (input-schema projection +
                reviewed-null fill), add PUMA-ladder population marginals
                (state + congressional district), and write a lean float32
                target-frame checkpoint + targets.json. Heavy stage; a crash
                in calibrate never re-runs the microsim.
  calibrate   : epoch-batched warm-start calibrate on the lean checkpoint
                (adam, mass conserved, hard weight-ratio cap; resumable with
                --resume), write diagnostics and the calibrated weights onto
                a copy of the staging H5.
  qa          : chunked engine probe of the calibrated artifact recording
                per-spine SSI incidence and intensity (the populace#403
                signature, measured rather than assumed).
  finalize    : release-style gate report (PUMA-ladder gate, calibration
                gate, spine-composition evidence) + the reviewed-limitations
                register for this lineage (inherited #507 SSI aged-band
                collapse, #393 miscellaneous-income defect, CD marginal
                vintage, ESS concentration, sparse-selection donor, mixed
                sub-PUMA coverage), and flip the summary simulation-ready.
  package     : assemble releases/<id>/ with the non-default local-area
                manifest shape (dataset_role ``non_default_local_area``,
                namespace ``buildo_acs_local``, donor identity chain, the
                one-command refresh recipe) + sha256sums. Publication stays
                a separate reviewed step (``populace-publish-release
                <dir> --no-latest``); this tool never uploads and never
                touches latest.json.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import resource
import shutil
import subprocess
import sys
import threading
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

_TOOLS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TOOLS_DIR.parent
if str(_TOOLS_DIR) not in sys.path:
    # The chunked materializer reuses the production target-frame seam from
    # the sibling release tool rather than forking it.
    sys.path.insert(0, str(_TOOLS_DIR))

PERIOD = 2024
RELEASE_NAMESPACE = "buildo_acs_local"
RELEASE_ID_PREFIX = "populace-us-2024-buildo-acs-local"
ARTIFACT_NAME = "populace_us_2024_acs_local"
ARTIFACT_FILENAME = f"{ARTIFACT_NAME}.h5"
HF_REPO_ID = "policyengine/populace-us"
LEGACY_STAGING_REFRESH_RECIPE = (
    "uv run tools/build_us_acs_multispine_base.py "
    "--base-h5 <next-certified-release>.h5 "
    "--donor-release-manifest <release_manifest.json> "
    "--out-h5 <run>/acs_multispine_staging.h5 "
    "--inputs-dir <acs-archive-cache> "
    "--puma-ladder build/us/us_puma_ladder_2020.npz"
)


# ---------------------------------------------------------------------------
# Shared telemetry
# ---------------------------------------------------------------------------


def rss() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e9 if sys.platform == "darwin" else peak * 1024 / 1e9


def log(message: str) -> None:
    print(
        f"[{time.strftime('%H:%M:%S')}] {message} (peakRSS {rss():.2f}GB)",
        flush=True,
    )


class RssSampler(threading.Thread):
    """Sample peak RSS on a fixed cadence for the run's memory ledger."""

    def __init__(self, every: float = 5.0) -> None:
        super().__init__(daemon=True)
        self.every = every
        self._stop = threading.Event()
        self.samples: list[tuple[float, float]] = []

    def run(self) -> None:
        while not self._stop.is_set():
            self.samples.append((round(time.time(), 1), round(rss(), 3)))
            self._stop.wait(self.every)

    def stop(self) -> None:
        self._stop.set()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text()) if Path(path).exists() else {}


# ---------------------------------------------------------------------------
# Target surface: state-level administrative families (production compile)
# ---------------------------------------------------------------------------


def state_admin_specs(feed: str | Path, families: list[str], soi_mode: str = "full"):
    """Select the state-level admin surface from the production compile path.

    feed -> ``compile_us_fiscal_target_registry(age_targets=True)`` ->
    ``apply_us_medicaid_enrollment_substitutions`` (RI FIPS-44) -> state-level
    {usda_snap, cms_medicaid[enrollment], irs_soi}. ``soi_mode='totals'``
    drops the ``soi_fiscal_distribution`` AGI-band slices (the jetsam-safe
    Option B of the Build L runbook); ``'full'`` keeps them.
    """

    from populace.build.ledger_artifact import load_ledger_consumer_artifact
    from populace.build.us_runtime.fiscal_targets import (
        compile_us_fiscal_target_registry,
    )
    from populace.build.us_runtime.medicaid_take_up import (
        apply_us_medicaid_enrollment_substitutions,
    )
    from populace.calibrate.registry import TargetRegistry

    artifact = load_ledger_consumer_artifact(str(feed))
    registry = compile_us_fiscal_target_registry(
        artifact.facts,
        target_period=PERIOD,
        include_congressional_district_targets=False,
        age_targets=True,
    )
    registry, ri_substitutions = apply_us_medicaid_enrollment_substitutions(registry)

    def state_level(spec) -> bool:
        return "state_fips" in spec.metadata

    picked = []
    if "snap" in families:
        picked += registry.select(family="usda_snap", predicate=state_level).specs
    if "medicaid" in families:
        picked += registry.select(
            family="cms_medicaid",
            predicate=lambda spec: (
                state_level(spec)
                and spec.metadata.get("target_role") == "medicaid_enrollment"
            ),
        ).specs
    if "soi" in families:
        soi_predicate = (
            state_level
            if soi_mode == "full"
            else (
                lambda spec: (
                    state_level(spec)
                    and spec.metadata.get("target_role") != "soi_fiscal_distribution"
                )
            )
        )
        picked += registry.select(family="irs_soi", predicate=soi_predicate).specs
    return TargetRegistry(list(picked), country="us"), ri_substitutions


# ---------------------------------------------------------------------------
# Engine pass under the nullable-artifact contract
# ---------------------------------------------------------------------------


class UnregisteredNullError(ValueError):
    """NaN in an engine-input column NOT in reviewed_engine_input_nulls."""


def project_input_only(base_frame, period: int = PERIOD):
    """Hold back every non-input variable so FED SET == ENGINE INPUT SET.

    The multispine pool keeps raw/aggregate source columns alongside leaf
    inputs by design, and pe-core refuses NaN inputs, so every variable that
    is not an engine input is held back from the engine pass using one
    classifier — ``PolicyEngineUSEngine.formula_owned_outputs`` — the same
    complement of ``variables()`` the reviewed-null register is built from.
    Structural columns (ids, membership, weights, geography keys) stay.
    The H5 is never touched. Returns (projected_frame, dropped_by_entity).
    """

    from populace.frame import Frame
    from populace.frame.adapters.policyengine_us import PolicyEngineUSEngine

    adapter = PolicyEngineUSEngine()
    structural = set(adapter._structural_columns())
    tables = {entity: base_frame.table(entity) for entity in base_frame.entities}
    present = {column for table in tables.values() for column in table.columns}
    formula_owned = set(adapter.formula_owned_outputs(present)) - structural
    dropped: dict[str, list[str]] = {}
    if not formula_owned:
        return base_frame, dropped
    new_tables = {}
    for entity in base_frame.entities:
        table = base_frame.table(entity)
        held = sorted(set(table.columns) & formula_owned)
        if held:
            dropped[entity] = held
            new_tables[entity] = table.drop(columns=held)
        else:
            new_tables[entity] = table
    weights = {
        entity: base_frame.weights_for(entity)
        for entity in base_frame.weighted_entities
    }
    projected = Frame(
        new_tables,
        base_frame.schema,
        weights,
        base_frame.strata,
        mass_log=base_frame.mass_log,
    )
    return projected, dropped


def _resolve_engine_default(system, column: str):
    """The pe-us variable's own default (never a hardcoded zero)."""

    from enum import Enum

    variable = system.variables.get(column)
    if variable is None:
        raise KeyError(f"{column!r} is not a policyengine-us variable.")
    default = variable.default_value
    if isinstance(default, Enum):
        return default.name, "enum"
    if isinstance(default, (bool, np.bool_)):
        return bool(default), "bool"
    return default, type(default).__name__


def fill_reviewed_nulls(
    frame,
    summary_path: Path,
    manifest_path: Path | None = None,
    period: int = PERIOD,
):
    """Apply the nullable-artifact contract for the engine pass (H5 untouched).

    Every registered (entity, column) has its NaN filled with the pe-us
    variable's own default; NaN in any engine-input column NOT in the
    register is a hard error with a per-spine diagnostic — an artifact
    defect, surfaced not filled.
    """

    from policyengine_us import CountryTaxBenefitSystem

    from populace.build.us_runtime.base_pool import spine_column
    from populace.frame.adapters.policyengine_us import PolicyEngineUSEngine

    summary = json.loads(Path(summary_path).read_text())
    register = {
        (entry["entity"], entry["column"]): entry
        for entry in summary.get("reviewed_engine_input_nulls", [])
    }
    if not register:
        raise ValueError(
            f"{summary_path} carries no reviewed_engine_input_nulls register; "
            "cannot apply the nullable-artifact engine-pass contract."
        )
    adapter = PolicyEngineUSEngine()
    input_names = set(adapter.variables())
    system = CountryTaxBenefitSystem()

    fills, drift_warnings, violations = [], [], []
    for entity in frame.entities:
        table = frame.table(entity)
        tag = spine_column(entity)
        for column in sorted(set(table.columns) & input_names):
            missing = table[column].isna()
            n_missing = int(missing.sum())
            if n_missing == 0:
                continue
            by_spine = {}
            if tag in table.columns:
                by_spine = {
                    str(spine): int((missing & table[tag].eq(spine)).sum())
                    for spine in sorted(map(str, table[tag].dropna().unique()))
                    if int((missing & table[tag].eq(spine)).sum())
                }
            if (entity, column) not in register:
                violations.append(
                    {
                        "entity": entity,
                        "column": column,
                        "missing_rows": n_missing,
                        "rows": len(table),
                        "missing_rows_by_spine": by_spine,
                    }
                )
                continue
            fill_value, kind = _resolve_engine_default(system, column)
            filled = table[column].fillna(fill_value)
            value_type = getattr(system.variables[column], "value_type", None)
            if value_type is bool:
                filled = filled.astype(bool)
            elif value_type is int:
                filled = filled.astype(np.int64)
            elif value_type is float:
                filled = filled.astype(np.float64)
            table[column] = filled
            recorded = int(register[(entity, column)]["missing_rows"])
            entry = {
                "entity": entity,
                "column": column,
                "filled_rows": n_missing,
                "rows": len(table),
                "fill_value": repr(fill_value),
                "fill_kind": kind,
                "register_missing_rows": recorded,
                "missing_rows_by_spine": by_spine,
            }
            fills.append(entry)
            if n_missing != recorded:
                drift_warnings.append(
                    {**entry, "warning": "actual null count differs from register"}
                )

    if manifest_path is not None:
        Path(manifest_path).write_text(
            json.dumps(
                {
                    "period": period,
                    "register_entries": len(register),
                    "columns_filled": len(fills),
                    "total_values_filled": sum(f["filled_rows"] for f in fills),
                    "count_mismatch_warnings": drift_warnings,
                    "unregistered_violations": violations,
                    "fills": fills,
                },
                indent=2,
            )
        )
    log(
        f"reviewed-null fill: {len(fills)} registered column(s), "
        f"{sum(f['filled_rows'] for f in fills):,} values (engine pass only)"
    )
    for drift in drift_warnings:
        log(
            f"  WARNING count drift {drift['entity']}.{drift['column']}: "
            f"actual {drift['filled_rows']} vs register "
            f"{drift['register_missing_rows']}"
        )
    if violations:
        detail = "; ".join(
            f"{v['entity']}.{v['column']} ({v['missing_rows']} null rows, "
            f"by spine {v['missing_rows_by_spine']})"
            for v in violations
        )
        raise UnregisteredNullError(
            f"{len(violations)} engine-input column(s) carry NaN but are NOT "
            "in the reviewed_engine_input_nulls register — artifact defect, "
            f"refusing to fill: {detail}"
        )
    return fills, drift_warnings


def materialize_chunked(
    base_frame,
    specs,
    *,
    hh_chunk: int = 40_000,
    batch: int = 5_000,
    period: int = PERIOD,
    dropped_manifest_path: Path | None = None,
    summary_path: Path | None = None,
    fills_manifest_path: Path | None = None,
    matrix_path: Path | None = None,
):
    """Household-chunked production materialization (memory-bounded).

    One full-frame Microsimulation caches the entire dependency closure of
    the SOI components (the 1.6M-row jetsam class), so the unchanged
    production ``_materialize_target_frame`` runs on whole-household
    sub-frames; each chunk's calc cache dies with its simulation and the
    float32 measure matrix accumulates on disk when ``matrix_path`` is set.
    """

    import build_us_fiscal_refresh_release as release_tool

    projected, dropped = project_input_only(base_frame, period=period)
    total_held = sum(len(columns) for columns in dropped.values())
    log(f"input-schema projection held back {total_held} formula-owned column(s)")
    if dropped_manifest_path is not None:
        Path(dropped_manifest_path).write_text(
            json.dumps(
                {
                    "period": period,
                    "held_back_formula_owned_columns": dropped,
                    "total": total_held,
                },
                indent=2,
            )
        )
    if summary_path is not None:
        fill_reviewed_nulls(
            projected,
            summary_path,
            manifest_path=fills_manifest_path,
            period=period,
        )
    else:
        raise ValueError(
            "materialize_chunked requires the staging summary; the "
            "nullable-artifact engine-pass contract needs its "
            "reviewed_engine_input_nulls register."
        )

    n_households = projected.n("household")
    household_ids = projected.table("household")["household_id"].to_numpy()
    position_by_id = pd.Series(np.arange(n_households), index=household_ids)
    person_household = projected.table("person")["person_household_id"].to_numpy()
    person_position = position_by_id.reindex(person_household).to_numpy()
    if np.isnan(person_position).any():
        raise RuntimeError(
            "person rows reference household ids absent from the household "
            "table; frame is invalid."
        )

    measure_names = None
    compiled_specs = None
    matrix = None
    chunk_stats = []
    n_chunks = (n_households + hh_chunk - 1) // hh_chunk
    for chunk_index, low in enumerate(range(0, n_households, hh_chunk)):
        high = min(low + hh_chunk, n_households)
        started = time.time()
        mask = (person_position >= low) & (person_position < high)
        sub_frame = projected.select(mask)
        target_frame, compiled_registry, _ = release_tool._materialize_target_frame(
            sub_frame, tuple(specs), maximum_microsim_batch_size=batch
        )
        names = [spec.measure for spec in compiled_registry.specs]
        if measure_names is None:
            measure_names = names
            compiled_specs = list(compiled_registry.specs)
            if matrix_path is not None:
                matrix = np.memmap(
                    matrix_path,
                    dtype=np.float32,
                    mode="w+",
                    shape=(n_households, len(names)),
                )
            else:
                matrix = np.zeros((n_households, len(names)), dtype=np.float32)
        elif names != measure_names:
            raise RuntimeError(
                f"chunk {chunk_index} compiled a different measure set "
                f"({len(names)} vs {len(measure_names)}); refusing to assemble."
            )
        chunk_households = target_frame.table("household")
        got_ids = chunk_households["household_id"].to_numpy()
        if len(got_ids) != high - low or not np.array_equal(
            got_ids, household_ids[low:high]
        ):
            raise RuntimeError(
                f"chunk {chunk_index} household order/id mismatch; refusing "
                "to assemble."
            )
        for j, measure in enumerate(measure_names):
            matrix[low:high, j] = chunk_households[measure].to_numpy(dtype=np.float32)
        if matrix_path is not None:
            matrix.flush()
        del target_frame, compiled_registry, chunk_households, sub_frame, got_ids
        gc.collect()
        stat = {
            "chunk": chunk_index + 1,
            "of": n_chunks,
            "households": high - low,
            "wall_s": round(time.time() - started, 1),
            "peak_rss_gb": round(rss(), 2),
        }
        chunk_stats.append(stat)
        log(f"materialize chunk {chunk_index + 1}/{n_chunks}: {stat['wall_s']}s")
    del projected
    gc.collect()
    return matrix, measure_names, compiled_specs, chunk_stats


# ---------------------------------------------------------------------------
# Population marginals + lean checkpoint
# ---------------------------------------------------------------------------

GROUP_IDS = {
    "tax_unit": "tax_unit_id",
    "spm_unit": "spm_unit_id",
    "family": "family_id",
    "marital_unit": "marital_unit_id",
}
PERSON_STRUCT = [
    "person_id",
    "person_household_id",
    "person_tax_unit_id",
    "person_spm_unit_id",
    "person_family_id",
    "person_marital_unit_id",
]


def ladder_population(ladder_path: Path, geographies: list[str]):
    from populace.build.us_runtime.puma_ladder import load_us_puma_ladder

    ladder = load_us_puma_ladder(ladder_path)
    puma_states = (ladder.puma // 100_000).astype(int)
    populations: dict[str, dict[int, float]] = {}
    if "state" in geographies:
        populations["state"] = {
            int(state): float(ladder.puma_population[puma_states == state].sum())
            for state in np.unique(puma_states)
        }
    if "cd" in geographies:
        populations["cd"] = {
            int(cd): float(
                ladder.cd_overlap_population[ladder.cd_overlap_cd == cd].sum()
            )
            for cd in np.unique(ladder.cd_overlap_cd)
        }
    return populations


def population_measure_arrays(frame, ladder_populations, geographies: list[str]):
    """Household-grain population measures as float32 arrays.

    Population in a geography = sum over persons-in-geo of household weight;
    every person shares its household's geography, so the household-grain
    measure is household_size x 1[household geo == value]. No engine
    involved. A ladder cell with no supporting household is DROPPED from the
    surface and returned in the fourth element — the caller decides whether
    a shrunken surface is acceptable (a capped smoke) or a defect (release).
    """

    households = frame.table("household")
    persons = frame.table("person")
    size = persons.groupby("person_household_id").size()
    household_size = households["household_id"].map(size).fillna(0).to_numpy(np.float32)
    names, arrays, values, dropped = [], [], [], []
    column_map = {
        "state": ("state_fips", "state"),
        "cd": ("congressional_district_geoid", "cd"),
    }
    for geography in geographies:
        column, key = column_map[geography]
        geo_values = pd.to_numeric(households[column]).to_numpy()
        for value, population in sorted(ladder_populations[key].items()):
            present = geo_values == value
            width = 2 if geography == "state" else 4
            name = f"pop_{geography}_{value:0{width}d}"
            if not present.any():
                dropped.append(name)
                continue
            names.append(name)
            arrays.append((household_size * present).astype(np.float32))
            values.append(float(population))
    return names, arrays, values, dropped


def extract_struct_tables(frame):
    """Small structural/geography copies so the big frame can be freed early."""

    households = frame.table("household")
    struct_columns = [
        column
        for column in (
            "household_id",
            "state_fips",
            "congressional_district_geoid",
            "county_fips",
        )
        if column in households.columns
    ]
    person_columns = [
        column for column in PERSON_STRUCT if column in frame.table("person").columns
    ]
    return {
        "household_struct": households[struct_columns].copy(),
        "person": frame.table("person")[person_columns].copy(),
        "groups": {
            group: frame.table(group)[[id_column]].copy()
            for group, id_column in GROUP_IDS.items()
        },
        "weights": np.asarray(frame.weights_for("household").values, dtype=np.float64),
    }


def write_lean_checkpoint(
    struct,
    admin_matrix,
    admin_names,
    admin_targets,
    pop_names,
    pop_arrays,
    pop_values,
    checkpoint_dir: Path,
):
    """Assemble the lean target-frame H5 + targets.json (memory-bounded)."""

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(admin_matrix, np.memmap):
        admin_matrix = np.array(admin_matrix)
    parts = {
        column: struct["household_struct"][column]
        for column in struct["household_struct"].columns
    }
    for j, name in enumerate(admin_names):
        parts[name] = admin_matrix[:, j]
    for name, array in zip(pop_names, pop_arrays, strict=True):
        parts[name] = array
    lean_households = pd.DataFrame(parts)
    del parts
    gc.collect()
    lean_households["household_weight"] = struct["weights"]
    checkpoint_h5 = checkpoint_dir / "target_frame_lean.h5"
    with pd.HDFStore(checkpoint_h5, mode="w") as store:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
            store.put("household", lean_households, format="fixed")
            store.put("person", struct["person"], format="fixed")
            for group, table in struct["groups"].items():
                store.put(group, table, format="fixed")
            store.put("_time_period", pd.Series([PERIOD]), format="table")
    targets = [
        dict(
            name=target["name"],
            entity="household",
            measure=target["measure"],
            value=target["value"],
            period=PERIOD,
            source=target.get("source", "ledger_feed"),
        )
        for target in admin_targets
    ]
    targets += [
        dict(
            name=name,
            entity="household",
            measure=name,
            value=value,
            period=PERIOD,
            source="us_puma_ladder_2020",
        )
        for name, value in zip(pop_names, pop_values, strict=True)
    ]
    (checkpoint_dir / "targets.json").write_text(json.dumps(targets, indent=2))
    log(
        f"checkpoint: {checkpoint_h5.name} ({len(lean_households)} hh, "
        f"{len(admin_names) + len(pop_names)} measures), targets.json "
        f"({len(targets)} targets)"
    )
    return checkpoint_h5, targets


def load_lean_frame(checkpoint_h5: Path):
    from populace.frame import Frame, WeightKind, Weights
    from populace.frame.units import US_SCHEMA

    tables = {}
    with pd.HDFStore(checkpoint_h5, mode="r") as store:
        for key in ["household", "person"] + list(GROUP_IDS):
            tables[key] = store[key]
    design_weights = tables["household"].pop("household_weight").to_numpy(np.float64)
    return (
        Frame(
            tables,
            US_SCHEMA,
            {"household": Weights(design_weights, WeightKind.CALIBRATED)},
        ),
        design_weights,
    )


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def _staging_summary_path(args) -> Path:
    if args.staging_summary is not None:
        return args.staging_summary
    return Path(str(args.staging_h5)).with_suffix(".summary.json")


def _load_staging_frame(path: Path):
    from build_us_acs_multispine_base import _load_base_frame

    return _load_base_frame(Path(path))


def do_materialize(args) -> None:
    families = [item.strip() for item in args.families.split(",") if item.strip()]
    geographies = [item.strip() for item in args.geographies.split(",") if item.strip()]
    started = time.time()
    if args.feed_sha256:
        actual = _sha256(args.feed)
        if actual != args.feed_sha256:
            raise SystemExit(
                f"Ledger feed sha256 mismatch: {args.feed} is {actual}, "
                f"expected {args.feed_sha256}."
            )
    registry, ri_substitutions = state_admin_specs(
        args.feed, families, soi_mode=args.soi_mode
    )
    log(
        f"admin specs: {len(registry)} ({families}, soi_mode={args.soi_mode}); "
        f"RI substitution records={len(ri_substitutions)}"
    )
    summary_path = _staging_summary_path(args)
    if not summary_path.exists():
        raise SystemExit(
            f"Staging summary not found at {summary_path}; the "
            "nullable-artifact engine-pass contract requires its "
            "reviewed_engine_input_nulls register."
        )
    log("hashing staging inputs for the run identity …")
    staging_sha = _sha256(args.staging_h5)
    ladder_sha = _sha256(args.ladder)
    frame = _load_staging_frame(args.staging_h5)
    log(
        f"loaded staging frame households={frame.n('household')} "
        f"({time.time() - started:.1f}s)"
    )

    started = time.time()
    matrix_path = args.checkpoint_dir / "measures_f32.mmap"
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    matrix, admin_names, compiled_specs, chunk_stats = materialize_chunked(
        frame,
        registry.specs,
        hh_chunk=args.hh_chunk,
        batch=args.batch,
        period=PERIOD,
        dropped_manifest_path=args.checkpoint_dir / "held_back_columns.json",
        summary_path=summary_path,
        fills_manifest_path=args.checkpoint_dir / "reviewed_null_fills.json",
        matrix_path=matrix_path,
    )
    log(
        f"materialized admin: {len(admin_names)} measures over "
        f"{len(chunk_stats)} chunks ({time.time() - started:.1f}s)"
    )
    if len(admin_names) != len(registry):
        raise SystemExit(
            f"Compiled admin surface has {len(admin_names)} measures but "
            f"{len(registry)} specs were declared; admin targets must never "
            "disappear silently between compile and materialization."
        )
    admin_targets = [
        dict(name=spec.name, measure=spec.measure, value=spec.value, source=spec.source)
        for spec in compiled_specs
    ]

    populations = ladder_population(args.ladder, geographies)
    pop_names, pop_arrays, pop_values, pop_dropped = population_measure_arrays(
        frame, populations, geographies
    )
    if pop_dropped:
        message = (
            f"{len(pop_dropped)} ladder population cell(s) have no "
            f"supporting household (e.g. {pop_dropped[:5]}). The calibrated "
            "surface would silently shrink."
        )
        if not args.allow_partial_geography:
            raise SystemExit(
                message + " Refusing for a release build; pass "
                "--allow-partial-geography only for capped smokes."
            )
        log("WARNING " + message + " Continuing (--allow-partial-geography).")
    log(f"population measures: {len(pop_names)} ({geographies})")
    struct = extract_struct_tables(frame)
    n_households = frame.n("household")
    del frame
    gc.collect()

    write_lean_checkpoint(
        struct,
        matrix,
        admin_names,
        admin_targets,
        pop_names,
        pop_arrays,
        pop_values,
        args.checkpoint_dir,
    )
    del matrix, struct, pop_arrays
    gc.collect()
    matrix_path.unlink(missing_ok=True)
    targets_digest = _sha256(args.checkpoint_dir / "targets.json")
    (args.checkpoint_dir / "run_identity.json").write_text(
        json.dumps(
            {
                "staging_h5": str(Path(args.staging_h5).resolve()),
                "staging_sha256": staging_sha,
                "ladder_sha256": ladder_sha,
                "households": n_households,
                "n_targets": len(admin_names) + len(pop_names),
                "targets_sha256": targets_digest,
                "declared_admin_specs": len(registry),
                "compiled_admin_specs": len(admin_names),
                "population_cells_dropped": pop_dropped,
            },
            indent=2,
        )
    )
    (args.checkpoint_dir / "materialize_rss.json").write_text(
        json.dumps(
            {
                "soi_mode": args.soi_mode,
                "families": families,
                "n_admin": len(admin_names),
                "n_population": len(pop_names),
                "population_cells_dropped": pop_dropped,
                "hh_chunk": args.hh_chunk,
                "chunk_stats": chunk_stats,
                "materialize_peak_rss_gb": round(rss(), 3),
            },
            indent=2,
        )
    )
    log(f"materialize stage complete, peak RSS {rss():.2f}GB")


def _verify_run_identity(args, *, require: bool = True) -> dict:
    """Load and re-verify the materialize-time run identity."""

    identity = _load_json(args.checkpoint_dir / "run_identity.json")
    if not identity:
        if require:
            raise SystemExit(
                f"No run_identity.json under {args.checkpoint_dir}; run "
                "--stage materialize first."
            )
        return {}
    staging_sha = _sha256(args.staging_h5)
    if staging_sha != identity.get("staging_sha256"):
        raise SystemExit(
            "Staging H5 does not match the materialized checkpoint: "
            f"{args.staging_h5} is {staging_sha} but the run identity pins "
            f"{identity.get('staging_sha256')}. Re-run --stage materialize "
            "against this staging file."
        )
    return identity


def do_calibrate(args) -> None:
    from populace.calibrate import calibrate
    from populace.calibrate.target import Target, TargetSet

    identity = _verify_run_identity(args)
    checkpoint_h5 = args.checkpoint_dir / "target_frame_lean.h5"
    targets_json = json.loads((args.checkpoint_dir / "targets.json").read_text())
    targets_sha = _sha256(args.checkpoint_dir / "targets.json")
    if targets_sha != identity.get("targets_sha256"):
        raise SystemExit(
            "targets.json changed since materialize; the checkpoint and "
            "surface no longer agree. Re-run --stage materialize."
        )
    frame, design_weights = load_lean_frame(checkpoint_h5)
    n_households = frame.n("household")
    if n_households != identity.get("households"):
        raise SystemExit(
            f"Lean checkpoint has {n_households} households but the run "
            f"identity pins {identity.get('households')}."
        )
    target_set = TargetSet(
        [
            Target(
                name=target["name"],
                entity=target["entity"],
                measure=target["measure"],
                value=target["value"],
                period=target["period"],
                source=target["source"],
            )
            for target in targets_json
        ]
    )
    log(
        f"calibrate: households={n_households}, targets={len(target_set)}, "
        f"design_total={design_weights.sum():,.0f}"
    )

    resume_npz = args.checkpoint_dir / "weights_latest.npz"
    warm, done = None, 0
    if args.resume and resume_npz.exists():
        saved = np.load(resume_npz)
        saved_identity = (
            str(saved["staging_sha256"]) if "staging_sha256" in saved else None
        )
        if saved_identity is not None and saved_identity != identity.get(
            "staging_sha256"
        ):
            raise SystemExit(
                "weights_latest.npz was produced against a different staging "
                "H5; refusing to warm-start from a foreign checkpoint."
            )
        warm, done = saved["weights"], int(saved["epochs_done"])
        if len(warm) != n_households:
            raise SystemExit(
                f"weights_latest.npz carries {len(warm)} weights but the "
                f"checkpoint has {n_households} households."
            )
        log(f"RESUME from {done} epochs")
    if done >= args.epochs:
        diagnostics_path = args.checkpoint_dir / "calibration_diagnostics.json"
        if diagnostics_path.exists():
            log(
                f"calibration already complete at {done} epochs and "
                "diagnostics exist; nothing to do (delete "
                "weights_latest.npz to recalibrate)."
            )
            _write_calibrated_artifact(
                args, np.asarray(warm, dtype=np.float64), identity
            )
            return
        raise SystemExit(
            f"weights_latest.npz reports {done} epochs (>= --epochs "
            f"{args.epochs}) but calibration_diagnostics.json is missing. "
            "Delete the checkpoint to recalibrate, or raise --epochs."
        )
    batch = args.epoch_batch if args.epoch_batch > 0 else args.epochs
    result = None
    started = time.time()
    while done < args.epochs:
        this_batch = min(batch, args.epochs - done)
        batch_started = time.time()
        result = calibrate(
            frame,
            target_set,
            weight_entity="household",
            method="adam",
            epochs=this_batch,
            learning_rate=0.02,
            mass="conserve",
            max_weight_ratio=args.max_weight_ratio,
            target_loss_cap=args.target_loss_cap,
            l2_lambda=args.l2_lambda,
            seed=args.seed,
            warm_start_weights=warm,
        )
        done += this_batch
        warm = result.weights.copy()
        np.savez(
            resume_npz,
            weights=warm,
            epochs_done=done,
            initial_weights=design_weights,
            staging_sha256=np.str_(identity["staging_sha256"]),
        )
        log(
            f"batch -> {done}/{args.epochs} ep, "
            f"{time.time() - batch_started:.1f}s, "
            f"loss={result.final_loss:.5f}, "
            f"within10%={result.fraction_within_10pct:.2%}, "
            f"ESS={result.effective_sample_size:,.0f}"
        )

    if result.problem.skipped:
        skipped = [getattr(item, "name", str(item)) for item in result.problem.skipped]
        raise SystemExit(
            f"Calibration compiled {len(skipped)} target(s) away "
            f"(e.g. {skipped[:5]}); the surface silently shrank. Fix the "
            "measures or the targets before shipping."
        )
    initial_estimates = result.problem.matrix @ design_weights
    final_estimates = result.problem.matrix @ result.weights
    per_target = [
        {
            "name": target.name,
            "target": float(target.value),
            "compiled_target": float(target.value),
            "initial_estimate": float(initial),
            "final_estimate": float(final),
        }
        for target, initial, final in zip(
            result.problem.targets,
            initial_estimates,
            final_estimates,
            strict=True,
        )
    ]
    diagnostics = {
        "households": n_households,
        "n_targets": result.problem.n_targets,
        "families": args.families,
        "geographies": args.geographies,
        "matrix_format": result.options["matrix_format"],
        "matrix_shape": [int(x) for x in result.problem.matrix.shape],
        "matrix_nnz": int(result.problem.matrix.nnz),
        "epochs": args.epochs,
        "epoch_batch": args.epoch_batch,
        "max_weight_ratio": args.max_weight_ratio,
        "l2_lambda": args.l2_lambda,
        "seed": args.seed,
        "initial_loss": round(result.initial_loss, 6),
        "final_loss": round(result.final_loss, 6),
        "fraction_within_10pct": round(result.fraction_within_10pct, 4),
        "effective_sample_size": round(result.effective_sample_size, 1),
        "ess_fraction": round(result.effective_sample_size / n_households, 4),
        "realized_max_weight_ratio": round(result.realized_max_weight_ratio, 4),
        "mass_conserved_ratio": round(
            float(result.weights.sum()) / float(design_weights.sum()), 6
        ),
        "total_wall_seconds": round(time.time() - started, 1),
        "peak_rss_gb": round(rss(), 3),
        "targets": per_target,
    }
    (args.checkpoint_dir / "calibration_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2)
    )
    log(
        f"calibrate stage complete: loss={diagnostics['final_loss']}, "
        f"within10%={diagnostics['fraction_within_10pct']:.2%}"
    )

    _write_calibrated_artifact(
        args, np.asarray(result.weights, dtype=np.float64), identity
    )


def _write_calibrated_artifact(args, weights: np.ndarray, identity: dict) -> None:
    """Write the consumer-ready calibrated H5 with verified attachment.

    The weights attach by verified household-id vector equality between the
    lean checkpoint and the freshly loaded staging H5 (never positionally to
    an unverified file), and the engine-pass contract is applied to the
    artifact bytes — input-schema projection plus reviewed-null default
    fill — so a plain ``USSingleYearDataset``/``Microsimulation`` consumer
    loads the published file directly. The nullable staging H5 remains the
    archival nullable truth; every fill is recorded in the run's consumer
    manifest.
    """

    if args.out_h5 is None:
        return
    from build_us_acs_multispine_base import _write_dataset

    from populace.frame import Frame, WeightKind, Weights

    frame = _load_staging_frame(args.staging_h5)
    staging_ids = frame.table("household")["household_id"].to_numpy()
    with pd.HDFStore(args.checkpoint_dir / "target_frame_lean.h5", mode="r") as store:
        lean_ids = store["household"]["household_id"].to_numpy()
    if len(staging_ids) != len(lean_ids) or not np.array_equal(staging_ids, lean_ids):
        raise SystemExit(
            "Staging household ids do not match the lean checkpoint's; "
            "refusing to attach calibrated weights to a different or "
            "reordered file."
        )
    if len(weights) != len(staging_ids):
        raise SystemExit(
            f"{len(weights)} calibrated weights cannot attach to "
            f"{len(staging_ids)} households."
        )

    projected, dropped = project_input_only(frame, period=PERIOD)
    del frame
    gc.collect()
    fills, _ = fill_reviewed_nulls(
        projected,
        _staging_summary_path(args),
        manifest_path=args.checkpoint_dir / "consumer_reviewed_null_fills.json",
        period=PERIOD,
    )
    calibrated = Frame(
        {entity: projected.table(entity) for entity in projected.entities},
        projected.schema,
        {"household": Weights(weights, WeightKind.CALIBRATED)},
        projected.strata,
        mass_log=projected.mass_log,
    )
    _write_dataset(
        calibrated,
        Path(args.out_h5),
        period=PERIOD,
        artifact_kind="calibrated_local_area_artifact",
    )
    (args.checkpoint_dir / "consumer_export.json").write_text(
        json.dumps(
            {
                "out_h5": str(Path(args.out_h5).resolve()),
                "staging_sha256": identity.get("staging_sha256"),
                "held_back_formula_owned": dropped,
                "held_back_total": sum(len(v) for v in dropped.values()),
                "filled_columns": len(fills),
                "total_values_filled": sum(f["filled_rows"] for f in fills),
                "note": (
                    "The engine-pass contract (input-schema projection + "
                    "reviewed-null default fill) is applied to the published "
                    "artifact bytes, so plain USSingleYearDataset/"
                    "Microsimulation consumers load it directly. The "
                    "nullable staging H5 remains the archival nullable "
                    "truth."
                ),
            },
            indent=2,
        )
    )
    log(f"wrote consumer-ready calibrated artifact {args.out_h5}")


def spine_composition(households: pd.DataFrame, persons: pd.DataFrame, weights):
    """Per-spine composition evidence (the populace#403 skew signature)."""

    from populace.build.us_runtime.base_pool import spine_column

    tag = spine_column("household")
    weights = np.asarray(weights, dtype=np.float64)
    person_counts = persons.groupby("person_household_id").size()
    household_size = (
        households["household_id"].map(person_counts).fillna(0).to_numpy(np.float64)
    )
    total_weight = float(weights.sum())
    composition = {}
    for spine in sorted(map(str, households[tag].dropna().unique())):
        mask = households[tag].eq(spine).to_numpy()
        spine_weight = float(weights[mask].sum())
        spine_person_weight = float((weights[mask] * household_size[mask]).sum())
        composition[spine] = {
            "households": int(mask.sum()),
            "household_weight": spine_weight,
            "household_weight_share": (
                spine_weight / total_weight if total_weight else 0.0
            ),
            "person_weight": spine_person_weight,
            "implied_persons_per_household": (
                spine_person_weight / spine_weight if spine_weight else 0.0
            ),
        }
    ess = float(weights.sum() ** 2 / (weights**2).sum()) if len(weights) else 0.0
    composition["_all"] = {
        "households": int(len(weights)),
        "household_weight": total_weight,
        "effective_sample_size": ess,
        "ess_fraction": ess / len(weights) if len(weights) else 0.0,
    }
    return composition


def do_qa(args) -> None:
    """Chunked engine probe: per-spine SSI incidence on the calibrated artifact.

    Loads the packaged artifact bytes PLAIN — no private projection or fill —
    so the probe doubles as the proof that ordinary
    ``USSingleYearDataset``/``Microsimulation`` consumers can load the file
    (the engine-pass contract was applied at export).
    """

    from populace.build.us_runtime.base_pool import spine_column

    projected = _load_staging_frame(args.out_h5)

    n_households = projected.n("household")
    household_ids = projected.table("household")["household_id"].to_numpy()
    position_by_id = pd.Series(np.arange(n_households), index=household_ids)
    person_household = projected.table("person")["person_household_id"].to_numpy()
    person_position = position_by_id.reindex(person_household).to_numpy()
    household_weights = np.asarray(
        projected.weights_for("household").values, dtype=np.float64
    )
    person_tag = projected.table("person")[spine_column("person")].to_numpy()

    from populace.frame.adapters.policyengine_us import PolicyEngineUSEngine

    adapter = PolicyEngineUSEngine()
    per_spine: dict[str, dict[str, float]] = {}
    n_chunks = (n_households + args.hh_chunk - 1) // args.hh_chunk
    for chunk_index, low in enumerate(range(0, n_households, args.hh_chunk)):
        high = min(low + args.hh_chunk, n_households)
        mask = (person_position >= low) & (person_position < high)
        sub_frame = projected.select(mask)
        ssi = np.asarray(
            adapter.materialize(sub_frame, ["ssi"], PERIOD)["ssi"],
            dtype=np.float64,
        )
        sub_person_household = sub_frame.table("person")[
            "person_household_id"
        ].to_numpy()
        sub_positions = position_by_id.reindex(sub_person_household).to_numpy()
        person_weight = household_weights[sub_positions.astype(np.int64)]
        sub_tags = person_tag[mask]
        for spine in np.unique(sub_tags):
            spine_mask = sub_tags == spine
            entry = per_spine.setdefault(
                str(spine),
                {
                    "person_weight": 0.0,
                    "ssi_recipient_weight": 0.0,
                    "ssi_dollars": 0.0,
                },
            )
            entry["person_weight"] += float(person_weight[spine_mask].sum())
            recipients = spine_mask & (ssi > 0)
            entry["ssi_recipient_weight"] += float(person_weight[recipients].sum())
            entry["ssi_dollars"] += float(
                (ssi[recipients] * person_weight[recipients]).sum()
            )
        del sub_frame
        gc.collect()
        log(f"qa chunk {chunk_index + 1}/{n_chunks}")

    for entry in per_spine.values():
        entry["ssi_incidence"] = (
            entry["ssi_recipient_weight"] / entry["person_weight"]
            if entry["person_weight"]
            else 0.0
        )
    payload = {
        "period": PERIOD,
        "variable": "ssi",
        "artifact": str(Path(args.out_h5).resolve()),
        "artifact_sha256": _sha256(args.out_h5),
        "plain_consumption": True,
        "per_spine": per_spine,
        "note": (
            "populace#403 re-measure: per-spine SSI incidence and intensity, "
            "computed by loading the packaged artifact bytes PLAIN (no "
            "private projection or fill) — the probe doubles as the "
            "consumer-loadability proof. Recorded as evidence, not gated."
        ),
    }
    (args.checkpoint_dir / "spine_qa.json").write_text(json.dumps(payload, indent=2))
    log(
        f"qa stage complete: {json.dumps({k: round(v['ssi_incidence'], 4) for k, v in per_spine.items()})}"
    )


def _repo_code_identity(allow_dirty: bool) -> dict[str, object]:
    def _git(*parts: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(_REPO_ROOT), *parts], text=True
        ).strip()

    sha = _git("rev-parse", "--short", "HEAD")
    dirty = bool(_git("status", "--porcelain"))
    if dirty and not allow_dirty:
        raise SystemExit(
            "Repository tree is dirty; a packaged release id must name a "
            "committed code vintage. Commit first or pass --allow-dirty "
            "(the manifest will record dirty=true)."
        )
    return {"sha": sha, "dirty": dirty, "branch": _git("branch", "--show-current")}


def finalize_reviewed_limitations(
    staging_summary: dict,
    diagnostics: dict,
    spine_qa: dict,
) -> list[dict]:
    """The reviewed-limitations register for the buildo-acs-local lineage.

    Carries the staging summary's reviewed limitations (GQ housing universe,
    native source-universe blanks, sub-PUMA precision) and adds the
    lineage-level entries, id-deduped so re-running finalize never
    duplicates.
    """

    ess = diagnostics.get("effective_sample_size")
    ess_fraction = diagnostics.get("ess_fraction", 0.0)
    households = diagnostics.get("households") or 0
    donor_release = (staging_summary.get("base") or {}).get("donor_release") or {}
    limitations = list(staging_summary.get("reviewed_limitations", []))
    aged_ssi = (spine_qa or {}).get("per_spine", {})
    limitations += [
        {
            "id": "ssi_aged_band_collapse_inherited",
            "status": "reviewed_inherited_defect",
            "reason": (
                "The donor lineage (base-O via the certified Build O sparse "
                "release) carries the populace#507 SSI aged-band take-up "
                "collapse: the 65+ one-shot Bernoulli seed at 8.4% collapses "
                "the aged SSI baseline to roughly 0.94M against SSA's 2.42M. "
                "The QRF transfer replicates donor SSI participation onto "
                "the ACS spine, so both spines of this artifact inherit the "
                "defect by construction."
            ),
            "treatment": (
                "populace#508 owns the fix; when the corrected base "
                "certifies, this artifact's refresh_recipe re-runs the "
                "chain against the new certified release in one command."
            ),
            "measured_spine_ssi": {
                spine: round(entry.get("ssi_incidence", 0.0), 5)
                for spine, entry in aged_ssi.items()
            },
            "calibration_blocker": False,
        },
        {
            "id": "miscellaneous_income_loss_side_donor_defect",
            "status": "reviewed_inherited_distortion",
            "column": "miscellaneous_income",
            "affected_spines": ["asec_puf", "acs_2024_1yr"],
            "reason": (
                "The donor pool's miscellaneous_income loss side remains "
                "distorted (populace#393, open at build time; roughly 4.6x "
                "SOI's loss-return prevalence measured in the Build J dense "
                "remedy experiments). The QRF transfer replicates the donor "
                "distribution onto the ACS spine."
            ),
            "treatment": "Tracked via populace#393; propagates on re-transfer.",
            "calibration_blocker": False,
        },
        {
            "id": "tips_return_count_carrier_deficit_inherited",
            "status": "reviewed_inherited_defect",
            "reason": (
                "The certified Build O sparse donor under-carries tip-return "
                "counts (-50.25% against its ledger target, recorded in its "
                "own release evidence). Tip columns transfer from that "
                "donor, so the deficit rides this artifact too."
            ),
            "calibration_blocker": False,
        },
        {
            "id": "cd_population_marginal_vintage_2020",
            "status": "reviewed_vintage",
            "reason": (
                "Congressional-district population marginals use "
                "119th-boundary / 2020-apportionment Census populations from "
                "the PUMA ladder; the build period is 2024. Boundaries are "
                "2020-census-drawn (honest v1)."
            ),
            "calibration_blocker": False,
        },
        {
            "id": "low_effective_sample_size_lambda_zero",
            "status": "reviewed_concentration",
            "reason": (
                f"Kish ESS is {ess} = {ess_fraction:.2%} of {households} "
                "households under the hard 5x weight cap with l2_lambda=0 "
                "(kept for consistency with the certified default and the "
                "Build L doctrine; no new calibration knobs per "
                "populace#492)."
            ),
            "calibration_blocker": False,
        },
        {
            "id": "donor_sparse_selection_training_set",
            "status": "reviewed_construction",
            "reason": (
                "The QRF transfer donor is the certified Build O sparse "
                "release "
                f"({donor_release.get('release_id', 'unknown donor id')}): "
                "the 57,240-household certified selection rather than a "
                "dense full-row donor (buildl trained on the June dense "
                "artifact). The conditioning set is thinner; per-fit donor "
                "row counts are recorded in the staging provenance."
            ),
            "calibration_blocker": False,
        },
        {
            "id": "mixed_sub_puma_column_coverage",
            "status": "reviewed_construction",
            "columns": [
                "block_geoid",
                "tract_geoid",
                "cbsa_code",
                "place_fips",
                "sldl",
                "sldu",
            ],
            "reason": (
                "Donor rows keep their certified block-ladder geography "
                "columns; ACS rows carry no sub-PUMA geography, so these "
                "columns are donor-spine-only. Consumers filtering on them "
                "must scope to the asec_puf spine."
            ),
            "calibration_blocker": False,
        },
    ]
    deduped: dict[str, dict] = {}
    for limitation in limitations:
        deduped[limitation["id"]] = limitation
    return list(deduped.values())


def do_finalize(args) -> None:
    from populace.build.us_runtime.puma_ladder import (
        load_us_puma_ladder,
        us_puma_ladder_gate,
    )

    staging_summary = _load_json(_staging_summary_path(args))
    diagnostics = _load_json(args.checkpoint_dir / "calibration_diagnostics.json")
    if not diagnostics:
        raise SystemExit(
            f"No calibration diagnostics under {args.checkpoint_dir}; run "
            "--stage calibrate first."
        )
    identity = _verify_run_identity(args)
    ladder_sha = _sha256(args.ladder)
    if ladder_sha != identity.get("ladder_sha256"):
        raise SystemExit(
            f"--ladder {args.ladder} (sha {ladder_sha[:12]}…) is not the "
            "ladder the surface was materialized with "
            f"({str(identity.get('ladder_sha256'))[:12]}…)."
        )
    materialize_rss = _load_json(args.checkpoint_dir / "materialize_rss.json")
    targets = _load_json(args.checkpoint_dir / "targets.json") or []
    spine_qa = _load_json(args.checkpoint_dir / "spine_qa.json")
    consumer_export = _load_json(args.checkpoint_dir / "consumer_export.json")

    frame = _load_staging_frame(args.out_h5)
    households = frame.table("household")
    weights = np.asarray(frame.weights_for("household").values, dtype=np.float64)
    load_us_puma_ladder(args.ladder)
    ladder_gate = us_puma_ladder_gate(households, weights)
    composition = spine_composition(households, frame.table("person"), weights)
    del frame
    gc.collect()

    breakdown: dict[str, int] = {}
    for target in targets:
        name = target["name"]
        if name.startswith("pop_state"):
            key = "population_state"
        elif name.startswith("pop_cd"):
            key = "population_cd"
        elif name.startswith("usda_snap") and "average_monthly_households" in name:
            key = "snap_caseloads"
        elif name.startswith("usda_snap"):
            key = "snap_benefits"
        elif name.startswith("cms_medicaid"):
            key = "medicaid_enrollment"
        elif name.startswith("irs_soi"):
            key = "soi"
        else:
            key = "other"
        breakdown[key] = breakdown.get(key, 0) + 1

    mass = diagnostics.get("mass_conserved_ratio", 0.0)
    gates = {
        "us_puma_ladder_gate": {
            "passed": bool(ladder_gate.passed),
            "failures": list(ladder_gate.failures),
            "detail": dict(ladder_gate.details),
        },
        "calibration": {
            # The cap criterion alone is near-tautological (the solver clips
            # per-row losses at the same cap); the solve must also have
            # actually improved on the design weights and conserved mass.
            # Numeric acceptance thresholds beyond that (within-10% floors,
            # per-target error bars) are maintainer-adjudicated surface
            # policy (#398-class), recorded here rather than invented.
            "passed": bool(
                diagnostics.get("final_loss", 1.0) < args.target_loss_cap
                and diagnostics.get("final_loss", 1.0)
                < diagnostics.get("initial_loss", 0.0)
                and abs(mass - 1.0) < 1e-3
            ),
            "initial_loss": diagnostics.get("initial_loss"),
            "final_loss": diagnostics.get("final_loss"),
            "fraction_within_10pct": diagnostics.get("fraction_within_10pct"),
            "effective_sample_size": diagnostics.get("effective_sample_size"),
            "ess_fraction": diagnostics.get("ess_fraction"),
            "realized_max_weight_ratio": diagnostics.get("realized_max_weight_ratio"),
            "mass_conserved_ratio": mass,
            "n_targets": diagnostics.get("n_targets"),
            "calibrated_surface": {
                "soi_mode": materialize_rss.get("soi_mode"),
                "n_targets": len(targets),
                "n_admin": materialize_rss.get("n_admin"),
                "n_population": materialize_rss.get("n_population"),
                "breakdown": breakdown,
            },
        },
        "input_coverage": {
            # Enforced upstream: the staging driver's donor-coverage gate
            # hard-fails before transfer, so reaching finalize means it held.
            "passed": True,
            "binds": "donor_certified_release",
            "note": (
                "Input coverage is enforced on the certified donor release "
                "before transfer; the ACS spine inherits required inputs via "
                "QRF transfer. ACS native GQ-housing / source-universe nulls "
                "are reviewed_limitations (calibration_blocker: false) and "
                "are NOT re-imposed on the ACS spine."
            ),
        },
        "spine_composition": {
            "passed": True,
            "note": (
                "Recorded evidence for the populace#403 weight-composition "
                "signature; not a pass/fail bound."
            ),
            "detail": composition,
        },
    }
    if spine_qa:
        gates["spine_ssi_qa"] = {
            "passed": True,
            "note": spine_qa.get("note"),
            "detail": spine_qa.get("per_spine"),
        }
    # Consumer-loadability: the export applied the engine-pass contract to
    # the artifact bytes, and the QA probe loaded those bytes plain.
    gates["consumer_ready"] = {
        "passed": bool(consumer_export)
        and bool(spine_qa.get("plain_consumption"))
        and spine_qa.get("artifact_sha256") is not None,
        "consumer_export": {
            key: consumer_export.get(key)
            for key in (
                "held_back_total",
                "filled_columns",
                "total_values_filled",
                "staging_sha256",
            )
        }
        if consumer_export
        else None,
        "plain_load_proven_by": "spine_ssi_qa"
        if spine_qa.get("plain_consumption")
        else None,
        "artifact_sha256": spine_qa.get("artifact_sha256"),
    }

    limitations = finalize_reviewed_limitations(staging_summary, diagnostics, spine_qa)
    hard_failures = [
        name
        for name in ("us_puma_ladder_gate", "calibration", "consumer_ready")
        if not gates[name]["passed"]
    ]
    updated_summary = dict(staging_summary)
    updated_summary.update(
        {
            "calibration_applied": True,
            "simulation_ready": not hard_failures,
            "simulation_ready_except_calibration": True,
            "simulation_readiness_blockers": hard_failures,
            "reviewed_limitations": limitations,
            "calibration_diagnostics": {
                key: value for key, value in diagnostics.items() if key != "targets"
            },
        }
    )
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(updated_summary, indent=2))
    gate_report = {
        "gates": gates,
        "reviewed_limitations": limitations,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    args.gate_report.write_text(json.dumps(gate_report, indent=2))
    if hard_failures:
        raise SystemExit(
            f"finalize: hard gate failure(s): {hard_failures} — see {args.gate_report}."
        )
    log(f"finalize stage complete: gates green, {len(limitations)} limitations")


def do_package(args) -> dict:
    diagnostics = _load_json(args.checkpoint_dir / "calibration_diagnostics.json")
    gate_report = _load_json(args.gate_report)
    staging_summary = _load_json(_staging_summary_path(args))
    final_summary = _load_json(args.out_summary)
    held_back = _load_json(args.checkpoint_dir / "held_back_columns.json")
    null_fills = _load_json(args.checkpoint_dir / "reviewed_null_fills.json")
    materialize_rss = _load_json(args.checkpoint_dir / "materialize_rss.json")
    spine_qa = _load_json(args.checkpoint_dir / "spine_qa.json")
    consumer_export = _load_json(args.checkpoint_dir / "consumer_export.json")
    identity = _verify_run_identity(args)
    if not gate_report:
        raise SystemExit("No gate report; run --stage finalize first.")
    if not final_summary.get("simulation_ready"):
        raise SystemExit(
            "Refusing to package: the finalized summary is not simulation_ready."
        )

    code = _repo_code_identity(args.allow_dirty)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    release_id = f"{RELEASE_ID_PREFIX}-{code['sha']}-{timestamp}"
    release_dir = args.out / "releases" / release_id
    release_dir.mkdir(parents=True, exist_ok=True)

    calibrated_h5 = Path(args.out_h5)
    if not calibrated_h5.exists():
        raise SystemExit(f"Calibrated H5 not found: {calibrated_h5}.")
    log("hashing calibrated H5 …")
    h5_sha = _sha256(calibrated_h5)
    # The gate report certifies specific artifact bytes: the QA probe
    # recorded the sha it loaded plain. Packaging different bytes (a
    # recalibrate without re-running qa+finalize) is refused.
    # Package-time evidence is REQUIRED, not conditional: absent files must
    # never read as vacuously green (a stale finalized summary could
    # otherwise pair with deleted or never-produced evidence).
    if not spine_qa:
        raise SystemExit(
            "spine_qa.json is missing or empty; the packaged bytes have no "
            "plain-consumption certification. Run --stage qa (then "
            "finalize) before packaging."
        )
    if not consumer_export:
        raise SystemExit(
            "consumer_export.json is missing or empty; the artifact's "
            "engine-pass export evidence is required. Run --stage calibrate "
            "before packaging."
        )
    qa_sha = spine_qa.get("artifact_sha256")
    if qa_sha is None:
        raise SystemExit(
            "spine_qa.json carries no artifact_sha256; the gate report "
            "cannot be bound to these bytes. Re-run --stage qa."
        )
    if qa_sha != h5_sha:
        raise SystemExit(
            "The calibrated H5 does not match the bytes the QA probe "
            f"certified ({h5_sha[:12]}… vs {str(qa_sha)[:12]}…). Re-run "
            "--stage qa and --stage finalize against the current artifact."
        )
    dropped_cells = identity.get("population_cells_dropped") or []
    if dropped_cells:
        raise SystemExit(
            f"The materialized surface dropped {len(dropped_cells)} ladder "
            "population cell(s); a release never ships a shrunken surface "
            "(--allow-partial-geography is for capped smokes only, and a "
            "smoke's output must not be packaged)."
        )

    def _version(package: str) -> str:
        try:
            from importlib.metadata import version

            return version(package)
        except Exception:
            return "unknown"

    donor_release = (staging_summary.get("base") or {}).get("donor_release")
    refresh_recipe = {
        "note": (
            "When the next certified default publishes (e.g. the populace#508 "
            "SSI fix -> O-2), refresh this artifact by re-running the chain "
            "against the new certified release H5; everything else is "
            "unchanged."
        ),
        "staging": LEGACY_STAGING_REFRESH_RECIPE,
        "release": (
            "uv run tools/build_us_acs_local_release.py --stage all "
            "--staging-h5 <run>/acs_multispine_staging.h5 "
            "--feed <ledger facts.jsonl> --feed-sha256 <sha> "
            "--ladder build/us/us_puma_ladder_2020.npz "
            "--checkpoint-dir <run>/checkpoints "
            "--out-h5 <run>/populace_us_2024_acs_local.h5 "
            "--out <run>/release"
        ),
        "publish": (
            "tools/publish_release.sh <release_dir> --no-latest "
            f"--artifact-root <run> --repo-id {HF_REPO_ID}"
        ),
    }

    build_manifest = {
        "build_id": release_id,
        "build_sha": code["sha"],
        "build_dirty": code["dirty"],
        "created_at": datetime.now(UTC).isoformat(),
        "code": {"branch": code["branch"], "repo": "populace"},
        "runtime": {
            "policyengine_core": _version("policyengine-core"),
            "policyengine_us": _version("policyengine-us"),
            "python": sys.version.split()[0],
        },
        "dataset": {
            "kind": "acs_local_area_multispine",
            "staging_h5_sha256": (staging_summary.get("output", {}) or {}).get(
                "sha256"
            ),
            "households": diagnostics.get("households"),
            "period": PERIOD,
            "spines": sorted((staging_summary.get("spine_totals", {}) or {}).keys())
            or ["acs_2024_1yr", "asec_puf"],
            "donor_release": donor_release,
        },
        "calibration": {
            key: diagnostics.get(key)
            for key in (
                "families",
                "geographies",
                "n_targets",
                "epochs",
                "max_weight_ratio",
                "l2_lambda",
                "seed",
                "final_loss",
                "fraction_within_10pct",
                "effective_sample_size",
                "ess_fraction",
                "realized_max_weight_ratio",
                "mass_conserved_ratio",
            )
        },
        "materialize": {
            "peak_rss_gb": materialize_rss.get("materialize_peak_rss_gb"),
            "hh_chunk": materialize_rss.get("hh_chunk"),
            "engine_pass": (
                "input-schema projection (fed==input) + reviewed-null fill"
            ),
            "held_back_formula_owned": held_back.get("total"),
            "reviewed_null_columns_filled": null_fills.get("columns_filled"),
        },
        "gates": gate_report.get("gates", {}),
        "run_identity": identity,
        "refresh_recipe": refresh_recipe,
    }

    source_coverage = {
        "schema_version": 1,
        "note": "ACS local-area artifact coverage summary.",
        "acs_sources": staging_summary.get("acs_sources"),
        "geography_ladder": staging_summary.get("geography_ladder"),
        "transfer_coverage": staging_summary.get("transfer_coverage"),
        "donor_release": donor_release,
        "input_coverage_gate": gate_report.get("gates", {}).get("input_coverage"),
    }

    contract_files = {
        "build_manifest.json": build_manifest,
        "calibration_diagnostics.json": diagnostics,
        "us_source_coverage.json": source_coverage,
        "gate_summary.json": gate_report,
        "held_back_columns.json": held_back,
        "reviewed_null_fills.json": null_fills,
        "run_identity.json": identity,
    }
    contract_files["spine_qa.json"] = spine_qa
    contract_files["consumer_export.json"] = consumer_export
    consumer_fills = _load_json(
        args.checkpoint_dir / "consumer_reviewed_null_fills.json"
    )
    if not consumer_fills:
        raise SystemExit(
            "The per-column consumer fill ledger "
            "(consumer_reviewed_null_fills.json) is missing; the published "
            "fills must ship with their evidence."
        )
    contract_files["consumer_reviewed_null_fills.json"] = consumer_fills
    for name, payload in contract_files.items():
        (release_dir / name).write_text(json.dumps(payload, indent=1))

    def _artifact(path_name: str, kind: str, local: Path) -> dict:
        return {
            "kind": kind,
            "path": path_name,
            "repo_id": HF_REPO_ID,
            "revision": release_id,
            "sha256": _sha256(local),
        }

    release_manifest = {
        "schema_version": 1,
        "data_package": {"name": "populace-data", "version": "0.1.0"},
        # NON-DEFAULT: this artifact claims NO default dataset slot.
        # Publishing it must never flip latest.json; it is discoverable by
        # its immutable release id / tag only.
        "default_datasets": {},
        "dataset_role": "non_default_local_area",
        "is_default": False,
        "namespace": RELEASE_NAMESPACE,
        "build": {
            "build_id": release_id,
            "built_at": datetime.now(UTC).isoformat(),
            "built_with_core_package": {
                "name": "policyengine-core",
                "version": _version("policyengine-core"),
            },
            "built_with_model_package": {
                "name": "policyengine-us",
                "version": _version("policyengine-us"),
            },
        },
        "artifacts": {
            ARTIFACT_NAME: {
                "kind": "microdata",
                "path": ARTIFACT_FILENAME,
                "repo_id": HF_REPO_ID,
                "revision": release_id,
                "sha256": h5_sha,
            },
            "calibration_diagnostics": _artifact(
                "calibration_diagnostics.json",
                "diagnostics",
                release_dir / "calibration_diagnostics.json",
            ),
            "gate_summary": _artifact(
                "gate_summary.json",
                "diagnostics",
                release_dir / "gate_summary.json",
            ),
            "us_source_coverage": _artifact(
                "us_source_coverage.json",
                "diagnostics",
                release_dir / "us_source_coverage.json",
            ),
        },
        "reviewed_limitations": gate_report.get("reviewed_limitations", []),
        "donor_release": donor_release,
        "refresh_recipe": refresh_recipe,
    }
    (release_dir / "release_manifest.json").write_text(
        json.dumps(release_manifest, indent=1)
    )

    sums = {path.name: _sha256(path) for path in sorted(release_dir.glob("*.json"))}
    sums[ARTIFACT_FILENAME] = h5_sha
    (release_dir / "sha256sums.txt").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(sums.items()))
    )

    artifact_root = args.out
    root_copy = artifact_root / ARTIFACT_FILENAME
    if root_copy.resolve() != calibrated_h5.resolve():
        if not root_copy.exists() or _sha256(root_copy) != h5_sha:
            log(f"copying calibrated H5 to artifact root {root_copy} …")
            shutil.copy2(calibrated_h5, root_copy)

    result = {
        "release_id": release_id,
        "release_dir": str(release_dir),
        "artifact_root": str(artifact_root),
        "root_artifact": {
            "name": ARTIFACT_FILENAME,
            "local_path": str(root_copy),
            "sha256": h5_sha,
        },
        "is_default": False,
        "files": sorted(path.name for path in release_dir.iterdir()),
        "publish_command": (
            f"tools/publish_release.sh {release_dir} --no-latest "
            f"--artifact-root {artifact_root} --repo-id {HF_REPO_ID}"
        ),
    }
    (args.out / "package_result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["materialize", "calibrate", "qa", "finalize", "package", "all"],
        default="all",
    )
    parser.add_argument("--staging-h5", type=Path, required=True)
    parser.add_argument(
        "--staging-summary",
        type=Path,
        default=None,
        help="Staging summary JSON (default: <staging-h5>.summary.json).",
    )
    parser.add_argument("--feed", type=Path)
    parser.add_argument(
        "--feed-sha256",
        help="Expected ledger-feed sha256; materialize refuses a mismatch.",
    )
    parser.add_argument(
        "--ladder",
        type=Path,
        default=_REPO_ROOT / "build" / "us" / "us_puma_ladder_2020.npz",
    )
    parser.add_argument("--families", default="snap,medicaid,soi")
    parser.add_argument("--geographies", default="state,cd")
    parser.add_argument("--soi-mode", choices=["full", "totals"], default="full")
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--epoch-batch", type=int, default=400)
    parser.add_argument("--max-weight-ratio", type=float, default=5.0)
    parser.add_argument("--target-loss-cap", type=float, default=1.0)
    parser.add_argument("--l2-lambda", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch", type=int, default=5_000)
    parser.add_argument(
        "--hh-chunk",
        type=int,
        default=40_000,
        help=(
            "Households per engine chunk; bounds the pe-core calc-cache "
            "closure (the 1.6M single-shot jetsam class)."
        ),
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument(
        "--out-h5",
        type=Path,
        help="Calibrated artifact H5 (staging copy + calibrated weights).",
    )
    parser.add_argument(
        "--out-summary",
        type=Path,
        default=None,
        help="Finalized summary path (default: <out-h5>.summary.json).",
    )
    parser.add_argument(
        "--gate-report",
        type=Path,
        default=None,
        help="Gate report path (default: <checkpoint-dir>/gate_summary.json).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Release root for --stage package (releases/<id>/ lands here).",
    )
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument(
        "--allow-partial-geography",
        action="store_true",
        help=(
            "Permit ladder population cells with no supporting household "
            "(a capped smoke shrinks the surface). Never for a release "
            "build: materialize hard-fails on dropped cells without this."
        ),
    )
    args = parser.parse_args(argv)

    stages = (
        ["materialize", "calibrate", "qa", "finalize", "package"]
        if args.stage == "all"
        else [args.stage]
    )
    if "materialize" in stages and args.feed is None:
        parser.error("--feed is required for the materialize stage.")
    if {"calibrate", "qa", "finalize", "package"} & set(stages) and (
        args.out_h5 is None
    ):
        parser.error("--out-h5 is required for calibrate/qa/finalize/package.")
    if "package" in stages and args.out is None:
        parser.error("--out is required for the package stage.")
    if args.out_h5 is not None and args.out_summary is None:
        args.out_summary = args.out_h5.with_suffix(".summary.json")
    if args.gate_report is None:
        args.gate_report = args.checkpoint_dir / "gate_summary.json"
    args.stages = stages
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    sampler = RssSampler()
    sampler.start()
    started = time.time()
    try:
        if "materialize" in args.stages:
            do_materialize(args)
        if "calibrate" in args.stages:
            do_calibrate(args)
        if "qa" in args.stages:
            do_qa(args)
        if "finalize" in args.stages:
            do_finalize(args)
        if "package" in args.stages:
            do_package(args)
    finally:
        sampler.stop()
        sampler.join(timeout=5)
        (args.checkpoint_dir / "rss_samples.json").write_text(
            json.dumps(sampler.samples[-2000:])
        )
    log(f"DONE stages={args.stages} ({time.time() - started:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
