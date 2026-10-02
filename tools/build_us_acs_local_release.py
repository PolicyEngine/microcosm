"""Materialize, calibrate, finalize, and package the US ACS local-area release.

The committed successor of the Build L runtime chain
(`national_combined_calibrate.py` + `materialize_admin.py` +
`finalize_artifact.py` + `package_release.py`), so a lineage refresh is one
command instead of a hand-carried script directory. Input: the multispine
staging H5 from ``tools/build_us_acs_multispine_base.py`` (donor spine +
ACS 2024 1-year spine, nullable, simulation-ready except calibration).

Stages (``--stage all`` runs materialize -> calibrate -> qa -> finalize ->
package; each is separately resumable):

  materialize : compile the administrative surface from the Ledger feed
                exactly like the production path
                (``compile_us_fiscal_target_registry`` -> RI Medicaid
                substitution -> state {usda_snap, cms_medicaid[enrollment],
                irs_soi}; ``--soi-mode state`` by default -- Build O's
                state-geography SOI contract -- with ``totals``, ``full`` and
                ``state_cd`` (district SOI rows, one vintage per state
                concept) as explicit opt-ins), run the household-chunked
                engine pass under the nullable-artifact contract (input-schema
                projection + reviewed-null fill), add PUMA-ladder population
                marginals (state + congressional district), assign the CD
                holdout, and write the lean checkpoint: a structure-only H5,
                target_registry.json, a sparse (targets x households) float32
                CSR target matrix row-aligned with it (target_matrix.npz) and
                target_roles.json. ``--sample-fraction`` draws a development
                rung.
                Heavy stage; a crash in calibrate never re-runs the microsim.
  calibrate   : epoch-batched warm-start calibrate on the checkpoint's
                training targets (adam, mass conserved, hard weight-ratio cap;
                resumable with --resume), score the held-out district targets
                against the pro-rata baseline, record ESS over rows and
                distinct households, and write the calibrated weights onto a
                copy of the staging H5.
  qa          : chunked engine probe of the calibrated artifact recording
                per-spine SSI incidence and intensity (the microcosm#403
                signature, measured rather than assumed).
  finalize    : release-style gate report (PUMA-ladder gate, calibration
                gate, district ESS collapse gate -- report-only unless
                ``--district-ess-gate-blocking`` --, spine-composition
                evidence) + the reviewed-limitations
                register for this lineage (inherited #507 SSI aged-band
                collapse, #393 miscellaneous-income defect, CD marginal
                vintage, ESS concentration, sparse-selection donor, mixed
                sub-PUMA coverage), and flip the summary simulation-ready.
  package     : refuse a calibrated H5 that stores a model input the
                installed policyengine-us does not define (microcosm#1026;
                ``microcosm.data.stored_inputs``), then
                assemble releases/<id>/ with the non-default local-area
                manifest shape (dataset_role ``non_default_local_area``,
                namespace ``buildo_acs_local``, donor identity chain, the
                one-command refresh recipe) + sha256sums. Publication stays
                a separate reviewed step (``microcosm-publish-release
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
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.us_runtime.acs_local_hours import acs_local_hours_signal_gate

_TOOLS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TOOLS_DIR.parent
if str(_TOOLS_DIR) not in sys.path:
    # The chunked materializer reuses the production target-frame seam from
    # the sibling release tool rather than forking it.
    sys.path.insert(0, str(_TOOLS_DIR))

import us_acs_local_cd_surface as cd_surface  # noqa: E402 - needs tools/ on sys.path

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

#: ``--soi-mode`` values for the state-level ``irs_soi`` surface.
#:
#: ``state`` (the default; Max's ruling of 2026-09-22) is the SOI contract every
#: earlier ACS local-area release was calibrated to, Build O's and Build P's:
#: every ``irs_soi`` spec at state geography whose Ledger record set is not a
#: congressional-district file -- on the pinned feed, the three TY2022 Historic
#: Table 2 state tables (broad totals, AGI bands, EITC by qualifying
#: children). Build O selected it as ``full`` with congressional-district
#: targets switched off (``include_congressional_district_targets=False`` at
#: populace@77e2061); commit b7922b089 removed that switch, so no mode
#: reproduced it until this one.
#:
#: ``totals`` keeps only the specs whose ``target_role`` is not
#: ``soi_fiscal_distribution`` -- on the pinned feed, ACA premium tax credit
#: rows and no state AGI, income-tax or EITC total. ``full`` keeps every
#: state-bearing spec, including the congressional-district file, with both
#: vintages of each state concept. Both are explicit opt-ins.
#:
#: ``state_cd`` (explicit opt-in) binds the congressional-district file's
#: district rows on top of ``state``, keeping one vintage per state concept:
#: Historic Table 2 gives a state concept's level, the district file only
#: each district's share of it (``us_acs_local_cd_surface.state_cd_soi_surface``).
#: A hash-assigned block of its district targets is held out of calibration
#: and scored against a pro-rata baseline. The target matrix is sparse in
#: every mode. docs/us-acs-local-soi-target-surface.md has the surfaces.
SOI_MODE_STATE = "state"
SOI_MODE_TOTALS = "totals"
SOI_MODE_FULL = "full"
SOI_MODE_STATE_CD = "state_cd"
SOI_MODES = (SOI_MODE_STATE, SOI_MODE_TOTALS, SOI_MODE_FULL, SOI_MODE_STATE_CD)
DEFAULT_SOI_MODE = SOI_MODE_STATE
#: Ledger record-set specs from a congressional-district file start with this;
#: ``state`` mode excludes them, which is what Build O's switch did.
CONGRESSIONAL_DISTRICT_RECORD_SET_SPEC_PREFIX = "irs_soi.congressional_district_"


def _require_soi_mode(soi_mode: str) -> str:
    if soi_mode not in SOI_MODES:
        raise ValueError(
            f"soi_mode must be one of {SOI_MODES}, got {soi_mode!r}; an "
            "unrecognised mode must never fall through to either surface."
        )
    return soi_mode


def release_refresh_recipe(
    soi_mode: str, cd_holdout_fraction: float | None = None
) -> str:
    """The one-command release refresh, pinned to the SOI surface it built.

    The recipe names ``--soi-mode`` explicitly so re-running it reproduces
    the recorded surface even if the parser default changes again, and, for
    every ``state_cd`` build, ``--cd-holdout-fraction`` at the recorded value
    (0 included, full precision), since ``state_cd`` otherwise defaults to a
    10% holdout.
    """

    _require_soi_mode(soi_mode)
    holdout = (
        f"--cd-holdout-fraction {float(cd_holdout_fraction)!r} "
        if soi_mode == SOI_MODE_STATE_CD and cd_holdout_fraction is not None
        else ""
    )
    return (
        "uv run tools/build_us_acs_local_release.py --stage all "
        "--staging-h5 <run>/acs_multispine_staging.h5 "
        "--feed <ledger-facts.jsonl> --feed-sha256 <sha> "
        f"--soi-mode {soi_mode} "
        f"{holdout}"
        "--ladder build/us/us_puma_ladder_2020.npz "
        "--checkpoint-dir <run>/checkpoints "
        "--out-h5 <run>/populace_us_2024_acs_local.h5 "
        "--out <run>/release"
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


def soi_surface_predicate(soi_mode: str):
    """The ``irs_soi`` spec filter for one ``--soi-mode``.

    Every mode keeps only specs carrying ``state_fips`` (state rows, and the
    congressional-district rows that also carry their state).

    - ``state`` then keeps a spec only at ``ledger_geography_level ==
      "state"`` whose ``ledger_layout_record_set_spec_id`` is not a
      congressional-district file, of any ``target_role`` (AGI-band rows
      included). A spec missing either key is not selected: the mode cannot
      tell which contract it belongs to.
    - ``totals`` drops every ``soi_fiscal_distribution`` spec (the
      jetsam-safe Option B of the Build L runbook). That role is not only
      AGI-band slices: every SOI fact without a named role gets it, which at
      state level includes the all-income-range state and district rows.
    - ``full`` keeps them all.
    - ``state_cd`` is not a filter: this returns the ``full`` candidates it
      starts from, and :func:`state_admin_surface` reconciles them.
    """

    _require_soi_mode(soi_mode)
    if soi_mode == SOI_MODE_STATE_CD:
        soi_mode = SOI_MODE_FULL

    def selected(spec) -> bool:
        metadata = spec.metadata
        if "state_fips" not in metadata:
            return False
        if soi_mode == SOI_MODE_STATE:
            record_set_spec = metadata.get("ledger_layout_record_set_spec_id")
            return (
                metadata.get("ledger_geography_level") == "state"
                and isinstance(record_set_spec, str)
                and bool(record_set_spec)
                and not record_set_spec.startswith(
                    CONGRESSIONAL_DISTRICT_RECORD_SET_SPEC_PREFIX
                )
            )
        return (
            soi_mode == SOI_MODE_FULL
            or metadata.get("target_role") != "soi_fiscal_distribution"
        )

    return selected


def state_admin_specs(
    feed: str | Path, families: list[str], soi_mode: str = DEFAULT_SOI_MODE
):
    """Select the state-level admin surface from the production compile path.

    Returns ``(registry, ri_substitutions)``; :func:`state_admin_surface`
    also returns the SOI surface receipt.
    """

    surface = state_admin_surface(feed, families, soi_mode=soi_mode)
    return surface.registry, surface.ri_substitutions


class AdminSurface:
    """The compiled admin registry plus how its SOI slice was chosen."""

    def __init__(self, registry, ri_substitutions, soi_receipt: dict) -> None:
        self.registry = registry
        self.ri_substitutions = ri_substitutions
        self.soi_receipt = soi_receipt


def state_admin_surface(
    feed: str | Path, families: list[str], soi_mode: str = DEFAULT_SOI_MODE
) -> AdminSurface:
    """Select the admin surface from the production compile path.

    feed -> ``compile_us_fiscal_target_registry(age_targets=True)`` ->
    ``apply_us_medicaid_enrollment_substitutions`` (RI FIPS-44) -> state-level
    {usda_snap, cms_medicaid[enrollment], irs_soi}. The SOI slice follows
    :func:`soi_surface_predicate` for ``state`` (the default), ``totals`` and
    ``full``; ``state_cd`` adds the reconciled congressional-district rows
    (:func:`us_acs_local_cd_surface.state_cd_soi_surface`).
    """

    # Refuse an unknown mode before loading the feed and compiling the registry.
    _require_soi_mode(soi_mode)

    from microcosm.build.ledger_artifact import load_ledger_consumer_artifact
    from microcosm.build.us_runtime import (
        default_congressional_district_vintage_crosswalk_path,
        load_congressional_district_vintage_crosswalk,
    )
    from microcosm.build.us_runtime.fiscal_targets import (
        compile_us_fiscal_target_registry,
    )
    from microcosm.build.us_runtime.medicaid_take_up import (
        apply_us_medicaid_enrollment_substitutions,
    )
    from microcosm.calibrate.registry import TargetRegistry

    artifact = load_ledger_consumer_artifact(str(feed))
    crosswalk_path = default_congressional_district_vintage_crosswalk_path()
    crosswalk = load_congressional_district_vintage_crosswalk(crosswalk_path)
    registry = compile_us_fiscal_target_registry(
        artifact.facts,
        target_period=PERIOD,
        congressional_district_vintage_crosswalk=crosswalk,
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
    soi_receipt: dict = {"soi_mode": soi_mode}
    if "soi" in families:
        candidates = registry.select(
            family="irs_soi", predicate=soi_surface_predicate(soi_mode)
        ).specs
        if soi_mode == SOI_MODE_STATE_CD:
            surface = cd_surface.state_cd_soi_surface(
                candidates,
                state_surface_predicate=soi_surface_predicate(SOI_MODE_STATE),
                crosswalk=crosswalk,
                crosswalk_sha256=_sha256(crosswalk_path),
            )
            picked += list(surface.specs)
            soi_receipt.update(surface.receipt)
        else:
            picked += candidates
            soi_receipt["counts"] = {"total": len(candidates)}
    return AdminSurface(
        TargetRegistry(list(picked), country="us"), ri_substitutions, soi_receipt
    )


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

    from microcosm.frame import Frame
    from microcosm.frame.adapters.policyengine_us import PolicyEngineUSEngine

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

    from microcosm.build.us_runtime.base_pool import spine_column
    from microcosm.frame.adapters.policyengine_us import PolicyEngineUSEngine

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


class MaterializedSurface:
    """The sparse admin surface one engine pass produced."""

    def __init__(self, matrix, compiled_specs, chunk_stats, carrier_check) -> None:
        #: (compiled specs x households) CSR, float32 data, declared order.
        self.matrix = matrix
        self.compiled_specs = compiled_specs
        self.chunk_stats = chunk_stats
        #: Direct-vs-carrier evidence from the first chunk (district rows).
        self.carrier_check = carrier_check

    @property
    def names(self) -> list[str]:
        return [spec.measure for spec in self.compiled_specs]


def _check_district_rows_against_parents(
    parents, captured, target_households, measure_of, *, n_chunk: int, tally: dict
) -> None:
    """Every chunk: each stored district row against its state parent's column.

    ``parents`` comes from ``cd_surface.district_row_parents``. A strict
    (``state_cd``) block's district rows partition the parent's state, so the
    float32 values stored for them, laid side by side, must equal the parent's
    directly materialized column on every household of the chunk, bit for bit.
    A fallback parent (another mode's same-concept state row) is compared on
    the row's own households. Unlike the first-chunk row check, this covers
    every household each concept touches, so it cannot be vacuous.
    """

    strict_blocks: dict[str, list[int]] = defaultdict(list)
    fallback_rows = fallback_nonzero = 0
    for index in sorted(captured):
        if index not in parents:
            continue
        parent_name, strict = parents[index]
        if strict:
            strict_blocks[parent_name].append(index)
            continue
        positions, values = captured[index]
        direct = target_households[measure_of[parent_name]].to_numpy(dtype=np.float32)
        if not np.array_equal(values, direct[positions]):
            differing = int((values != direct[positions]).sum())
            raise RuntimeError(
                f"A stored district row differs from its same-concept state "
                f"row {parent_name} on {differing} of its household(s); the "
                "carrier split is not exact for this surface."
            )
        fallback_rows += 1
        fallback_nonzero += int(np.count_nonzero(values))
    checked = nonzero = 0
    for parent_name, rows in strict_blocks.items():
        rebuilt = np.zeros(n_chunk, dtype=np.float32)
        written = np.zeros(n_chunk, dtype=bool)
        for index in rows:
            positions, values = captured[index]
            if written[positions].any():
                raise RuntimeError(
                    f"District rows of {parent_name} overlap on a household."
                )
            written[positions] = True
            rebuilt[positions] = values
        direct = target_households[measure_of[parent_name]].to_numpy(dtype=np.float32)
        if not np.array_equal(rebuilt, direct):
            outside = np.flatnonzero(~written & (direct != 0))
            if len(outside):
                household_ids = target_households["household_id"].to_numpy()
                raise RuntimeError(
                    f"{len(outside)} household(s) (household_id "
                    f"{household_ids[outside[:5]].tolist()}) carry {parent_name} "
                    "but sit in no district of its block: their congressional "
                    "district is not one of their state's current-plan "
                    "districts, or the block is missing that district's row."
                )
            differing = int((rebuilt != direct).sum())
            raise RuntimeError(
                f"The stored district rows of {parent_name} do not rebuild its "
                f"directly materialized column ({differing} household(s)); the "
                "carrier split is not exact for this surface."
            )
        checked += 1
        nonzero += int(np.count_nonzero(direct))
    for key, value in (
        ("parent_blocks_checked", checked),
        ("parent_block_nonzero_households", nonzero),
        ("fallback_rows_checked", fallback_rows),
        ("fallback_row_nonzero_households", fallback_nonzero),
    ):
        tally[key] = tally.get(key, 0) + value


def _household_codes(release_tool, households, column: str) -> np.ndarray:
    return np.asarray(
        release_tool._integer_geography_codes(
            households[column].to_numpy(), column=column
        ),
        dtype=np.int64,
    )


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
) -> MaterializedSurface:
    """Household-chunked production materialization into a sparse matrix.

    One full-frame Microsimulation caches the entire dependency closure of
    the SOI components (the 1.6M-row jetsam class), so the unchanged
    production ``_materialize_target_frame`` runs on whole-household
    sub-frames; each chunk's calc cache dies with its simulation. Each
    chunk's measure columns go straight into a (targets x households) CSR
    matrix with float32 values (``SparseTargetAssembler``): no dense
    households x targets matrix exists at any point.

    District SOI rows are not materialized one column each. The engine pass
    materializes one geography-free carrier per distinct SOI concept, and
    each district row is its carrier restricted to the district's households
    (``us_acs_local_cd_surface.plan_carriers``). That equals the direct
    materialization because the SOI slice masks a tax unit by its household's
    state and district. The first chunk also materializes one district row
    per carrier directly and refuses any difference.
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

    plan = cd_surface.plan_carriers(specs)
    by_name = {spec.name: spec for spec in plan.declared}
    assembler = cd_surface.SparseTargetAssembler(plan.n_rows, n_households)
    compiled_names: set[str] | None = None
    carrier_check: dict[str, object] = {"district_rows": len(plan.split)}
    parents: dict[int, tuple[str, bool]] | None = None
    chunk_stats = []
    n_chunks = (n_households + hh_chunk - 1) // hh_chunk
    if n_chunks > 1:
        release_tool._assert_group_entities_nest_in_households(projected)
        release_tool._assert_medicaid_claiming_tax_units_local(projected)
    for chunk_index, low in enumerate(range(0, n_households, hh_chunk)):
        high = min(low + hh_chunk, n_households)
        started = time.time()
        mask = (person_position >= low) & (person_position < high)
        sub_frame = projected.select(mask)
        chunk_households = sub_frame.table("household")
        state_codes = district_codes = None
        if plan.split:
            state_codes = _household_codes(release_tool, chunk_households, "state_fips")
            district_codes = _household_codes(
                release_tool, chunk_households, "congressional_district_geoid"
            )
        check_specs = (
            cd_surface.carrier_check_specs(plan, district_codes)
            if chunk_index == 0 and plan.split
            else ()
        )
        target_frame, compiled_registry, _ = release_tool._materialize_target_frame(
            sub_frame,
            plan.engine_specs + check_specs,
            maximum_microsim_batch_size=batch,
            refuse_population_aggregates=True if n_chunks > 1 else None,
        )
        compiled = {spec.name for spec in compiled_registry.specs}
        engine_compiled = compiled - {spec.name for spec in check_specs}
        if compiled_names is None:
            compiled_names = engine_compiled
        elif engine_compiled != compiled_names:
            raise RuntimeError(
                f"chunk {chunk_index} compiled a different measure set "
                f"({len(engine_compiled)} vs {len(compiled_names)}); refusing "
                "to assemble."
            )
        target_households = target_frame.table("household")
        got_ids = target_households["household_id"].to_numpy()
        if len(got_ids) != high - low or not np.array_equal(
            got_ids, household_ids[low:high]
        ):
            raise RuntimeError(
                f"chunk {chunk_index} household order/id mismatch; refusing "
                "to assemble."
            )
        for index, measure in plan.direct_rows:
            if plan.declared[index].name in compiled_names:
                assembler.add_column(
                    index,
                    low,
                    target_households[measure].to_numpy(dtype=np.float64),
                    name=plan.declared[index].name,
                )
        if plan.split:
            carriers = {
                carrier: target_households[carrier].to_numpy(dtype=np.float64)
                for carrier in sorted(set(plan.carrier_of.values()))
                if carrier in compiled_names
            }
            split = tuple(row for row in plan.split if row[3] in carriers)
            index_of = {spec.name: index for index, spec in enumerate(plan.declared)}
            captured = cd_surface.split_carriers_into(
                assembler,
                cd_surface.CarrierPlan(
                    declared=plan.declared,
                    engine_specs=plan.engine_specs,
                    direct_rows=plan.direct_rows,
                    split=split,
                    carrier_of=plan.carrier_of,
                ),
                carriers,
                low=low,
                state_codes=state_codes,
                district_codes=district_codes,
                capture=frozenset(index for index, *_rest in split),
            )
            if parents is None:
                parents, unchecked = cd_surface.district_row_parents(
                    plan, compiled_names
                )
                carrier_check["rows_without_parent_check"] = len(unchecked)
            _check_district_rows_against_parents(
                parents,
                captured,
                target_households,
                {name: by_name[name].measure for name, _strict in parents.values()},
                n_chunk=high - low,
                tally=carrier_check,
            )
            if check_specs:
                checked = []
                for spec in check_specs:
                    carrier = plan.carrier_of[spec.name]
                    if spec.name not in compiled or carrier not in carriers:
                        raise RuntimeError(
                            f"carrier check spec {spec.name} or its carrier "
                            f"{carrier} did not materialize."
                        )
                    direct = target_households[spec.measure].to_numpy(dtype=np.float64)
                    positions, values = captured[index_of[spec.name]]
                    stored = np.zeros(high - low, dtype=np.float32)
                    stored[positions] = values
                    if not np.array_equal(stored, direct.astype(np.float32)):
                        differing = int((stored != direct.astype(np.float32)).sum())
                        raise RuntimeError(
                            f"District row {spec.name} materialized directly "
                            "differs from the row stored from its carrier "
                            f"({differing} household(s)); the carrier split is "
                            "not exact for this surface."
                        )
                    checked.append(
                        {
                            "target": spec.name,
                            "carrier": carrier,
                            "nonzero_households": int(np.count_nonzero(direct)),
                        }
                    )
                carrier_check.update(
                    {
                        "carriers": len(carriers),
                        "checked_district_rows": len(checked),
                        "checked_nonzero_households": sum(
                            entry["nonzero_households"] for entry in checked
                        ),
                        "compared": "stored CSR row vs direct materialization",
                        "all_equal": True,
                        "checked": checked,
                    }
                )
        del target_frame, compiled_registry, target_households, sub_frame, got_ids
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
    matrix = assembler.to_csr()
    keep = [
        index
        for index, spec in enumerate(plan.declared)
        if spec.name in compiled_names
        or plan.carrier_of.get(spec.name) in (compiled_names or set())
    ]
    compiled_specs = [by_name[plan.declared[index].name] for index in keep]
    if len(keep) != plan.n_rows:
        matrix = sparse_rows(matrix, keep)
    return MaterializedSurface(matrix, compiled_specs, chunk_stats, carrier_check)


def sparse_rows(matrix, rows):
    """The CSR rows ``rows`` of ``matrix`` (float32 data preserved)."""

    from scipy import sparse

    selected = sparse.csr_array(matrix[rows])
    selected.sort_indices()
    return selected


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
    from microcosm.build.us_runtime.puma_ladder import load_us_puma_ladder

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


def household_sizes(frame) -> np.ndarray:
    """Persons per household, aligned to the household table (float32)."""

    households = frame.table("household")
    size = frame.table("person").groupby("person_household_id").size()
    return households["household_id"].map(size).fillna(0).to_numpy(np.float32)


def population_targets(frame, ladder_populations, geographies: list[str]):
    """Household-grain population marginals as sparse target rows.

    Population in a geography = sum over persons-in-geo of household weight;
    every person shares its household's geography, so the household-grain
    measure is household_size x 1[household geo == value]. No engine
    involved. A ladder cell with no supporting household is DROPPED from the
    surface and returned in the third element: the caller decides whether a
    shrunken surface is acceptable (a capped smoke) or a defect (release).
    Returns ``(target records, CSR rows, dropped cell names)``.
    """

    return cd_surface.population_rows(
        frame.table("household"),
        household_sizes(frame),
        ladder_populations,
        geographies,
    )


def population_target_specs(names, values):
    """Declare ladder population targets with the shared diagnostics hierarchy."""

    from microcosm.build.us_runtime.congressional_district_vintage import (
        CURRENT_CONGRESSIONAL_DISTRICT_PREFIX,
    )
    from microcosm.calibrate import TargetSpec
    from microcosm.calibrate.geography_constants import US_STATE_FIPS_TO_POSTAL
    from microcosm.calibrate.hierarchy import (
        CalibrationHierarchy,
        HierarchyCategory,
        HierarchyGeography,
        HierarchyNode,
    )

    provider = HierarchyNode("census_population", "Census population")
    category = HierarchyCategory(
        "census_population.resident_population",
        "Resident population",
        provider.id,
    )
    specs = []
    for name, value in zip(names, values, strict=True):
        if name.startswith("pop_state_"):
            fips = name.removeprefix("pop_state_")
            label = US_STATE_FIPS_TO_POSTAL.get(fips, f"State {fips}")
            geography = HierarchyGeography(f"0400000US{fips}", label, "state")
        elif name.startswith("pop_cd_"):
            geoid = name.removeprefix("pop_cd_")
            state_fips, district = geoid[:2], geoid[2:]
            state = US_STATE_FIPS_TO_POSTAL.get(state_fips, state_fips)
            district_label = (
                "at-large" if district == "00" else f"district {int(district)}"
            )
            label = f"{state} congressional {district_label}"
            # The ladder's districts are the 119th Congress plan.
            geography = HierarchyGeography(
                f"{CURRENT_CONGRESSIONAL_DISTRICT_PREFIX}{geoid}",
                label,
                "congressional_district",
            )
        else:  # pragma: no cover - names originate in population_measure_arrays
            raise ValueError(f"Unknown population target name {name!r}.")
        specs.append(
            TargetSpec(
                name=name,
                entity="household",
                measure=name,
                value=value,
                period=PERIOD,
                source="US Census Bureau 2020 PUMA population ladder",
                family="census_population",
                metadata={"geography_level": geography.level},
                hierarchy=CalibrationHierarchy(
                    provider=provider,
                    category=category,
                    geography=geography,
                    dimensions=(),
                    target=HierarchyNode(name, f"{label} resident population"),
                ),
            )
        )
    return tuple(specs)


#: Household columns the lean checkpoint keeps: identity, geography, and the
#: origin tags the distinct-household ESS and spine shares need.
LEAN_HOUSEHOLD_COLUMNS = (
    "household_id",
    "state_fips",
    "congressional_district_geoid",
    "county_fips",
    "household_spine",
    "household_source_id",
)
#: The checkpoint's sparse target matrix: (targets x households) CSR with
#: float32 values, row ``i`` of which is ``target_registry.json`` spec ``i``.
TARGET_MATRIX_FILENAME = "target_matrix.npz"
#: Per-target build roles, row-aligned with the registry: train or holdout,
#: geography, sigma, and a district row's state parent and populations.
TARGET_ROLES_FILENAME = "target_roles.json"


def extract_struct_tables(frame):
    """Small structural/geography copies so the big frame can be freed early."""

    households = frame.table("household")
    struct_columns = [
        column for column in LEAN_HOUSEHOLD_COLUMNS if column in households.columns
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
    matrix,
    specs,
    roles: list[dict],
    checkpoint_dir: Path,
):
    """Write the lean checkpoint: structure H5, registry, sparse matrix, roles.

    ``target_frame_lean.h5`` holds only structure (household identity,
    geography, origin tags and design weights; person memberships; group
    ids). ``target_registry.json`` holds every target, held-out ones
    included; ``target_matrix.npz`` is a (targets x households) CSR with
    float32 values whose row ``i`` is registry spec ``i``;
    ``target_roles.json`` is row-aligned with both. Returns
    ``(h5 path, registry, digests)``.
    """

    from scipy import sparse

    from microcosm.calibrate import TargetRegistry
    from microcosm.frame import put_frame_table

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    specs = tuple(specs)
    matrix = sparse.csr_array(matrix)
    n_households = len(struct["household_struct"])
    if matrix.shape != (len(specs), n_households):
        raise ValueError(
            f"target matrix shape {matrix.shape} does not pair with "
            f"{len(specs)} targets x {n_households} households."
        )
    if [role["name"] for role in roles] != [spec.name for spec in specs]:
        raise ValueError("target roles are not row-aligned with the target specs.")
    if matrix.data.dtype != np.float32:
        matrix = sparse.csr_array(matrix, dtype=np.float32)
    lean_households = struct["household_struct"].copy()
    lean_households["household_weight"] = struct["weights"]
    checkpoint_h5 = checkpoint_dir / "target_frame_lean.h5"
    with pd.HDFStore(checkpoint_h5, mode="w") as store:
        put_frame_table(
            store,
            "household",
            lean_households,
            preferred_format="fixed",
        )
        put_frame_table(
            store,
            "person",
            struct["person"],
            preferred_format="fixed",
        )
        for group, table in struct["groups"].items():
            put_frame_table(
                store,
                group,
                table,
                preferred_format="fixed",
            )
        store.put("_time_period", pd.Series([PERIOD]), format="table")
    registry = TargetRegistry(specs, country="us")
    registry.to_json(checkpoint_dir / "target_registry.json")
    matrix_sha = cd_surface.save_target_matrix(
        checkpoint_dir / TARGET_MATRIX_FILENAME, matrix
    )
    (checkpoint_dir / TARGET_ROLES_FILENAME).write_text(json.dumps(roles, indent=1))
    log(
        f"checkpoint: {checkpoint_h5.name} ({n_households} hh), "
        f"target_registry.json ({len(registry)} targets), "
        f"{TARGET_MATRIX_FILENAME} ({matrix.shape[0]} x {matrix.shape[1]}, "
        f"nnz {matrix.nnz:,})"
    )
    return (
        checkpoint_h5,
        registry,
        {
            "target_registry_sha256": _sha256(checkpoint_dir / "target_registry.json"),
            "target_matrix_sha256": matrix_sha,
            "target_roles_sha256": _sha256(checkpoint_dir / TARGET_ROLES_FILENAME),
            "lean_h5_sha256": _sha256(checkpoint_h5),
        },
    )


def load_lean_frame(checkpoint_h5: Path):
    from microcosm.frame import Frame, WeightKind, Weights, read_frame_table
    from microcosm.frame.units import US_SCHEMA

    tables = {}
    with pd.HDFStore(checkpoint_h5, mode="r") as store:
        for key in ["household", "person"] + list(GROUP_IDS):
            tables[key] = read_frame_table(store, key)
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


def _cd_holdout_fraction(args) -> float:
    """The CD holdout fraction this run draws (0 when there is no CD surface)."""

    fraction = getattr(args, "cd_holdout_fraction", None)
    if fraction is None:
        return (
            cd_surface.DEFAULT_STATE_CD_HOLDOUT_FRACTION
            if args.soi_mode == SOI_MODE_STATE_CD
            else 0.0
        )
    return float(fraction)


def _attach_pro_rata_populations(targets: list[dict], cd_populations: dict) -> None:
    """Give each district SOI target its district and state populations.

    The pro-rata baseline allocates the state parent by these; both come
    from the PUMA ladder's 119th-plan district overlap populations, so a
    state's district shares sum to one.
    """

    state_population: dict[int, float] = {}
    for district, population in cd_populations.items():
        state = int(district) // 100
        state_population[state] = state_population.get(state, 0.0) + float(population)
    for target in targets:
        geoid = target.get("congressional_district_geoid")
        if target.get("family") != "irs_soi" or not geoid:
            continue
        district = int(geoid)
        target["cd_population"] = float(cd_populations.get(district, 0.0))
        target["state_population"] = float(state_population.get(district // 100, 0.0))


def _require_pro_rata_populations(targets: list[dict]) -> None:
    """Refuse, before any solve, a district row the baseline cannot score."""

    unpopulated = [
        target["name"]
        for target in targets
        if target.get("family") == "irs_soi"
        and target.get("congressional_district_geoid")
        and not (
            target.get("cd_population", 0) > 0 and target.get("state_population", 0) > 0
        )
    ]
    if unpopulated:
        raise SystemExit(
            f"{len(unpopulated)} district SOI target(s) have no positive "
            "ladder district or state population for the pro-rata "
            f"baseline (e.g. {unpopulated[:3]}); refusing before the solve."
        )


def do_materialize(args) -> None:
    from scipy import sparse

    families = [item.strip() for item in args.families.split(",") if item.strip()]
    geographies = [item.strip() for item in args.geographies.split(",") if item.strip()]
    holdout_fraction = _cd_holdout_fraction(args)
    if holdout_fraction and args.soi_mode != SOI_MODE_STATE_CD:
        raise SystemExit(
            f"--cd-holdout-fraction {holdout_fraction} needs --soi-mode "
            f"{SOI_MODE_STATE_CD}; no other surface binds district SOI targets."
        )
    started = time.time()
    if args.feed_sha256:
        actual = _sha256(args.feed)
        if actual != args.feed_sha256:
            raise SystemExit(
                f"Ledger feed sha256 mismatch: {args.feed} is {actual}, "
                f"expected {args.feed_sha256}."
            )
    surface = state_admin_surface(args.feed, families, soi_mode=args.soi_mode)
    registry = surface.registry
    log(
        f"admin specs: {len(registry)} ({families}, soi_mode={args.soi_mode}); "
        f"RI substitution records={len(surface.ri_substitutions)}"
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
    _require_local_hours(frame, _load_json(summary_path))
    frame, sampling = cd_surface.sample_staging_frame(
        frame, fraction=args.sample_fraction, seed=args.sample_seed
    )
    gc.collect()
    log(
        f"loaded staging frame households={frame.n('household')} "
        f"(rung {sampling['rung']}; {time.time() - started:.1f}s)"
    )

    started = time.time()
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (args.checkpoint_dir / "measures_f32.mmap").unlink(missing_ok=True)
    for name in CALIBRATION_OUTPUT_FILENAMES:
        (args.checkpoint_dir / name).unlink(missing_ok=True)
    if getattr(args, "gate_report", None) is not None:
        Path(args.gate_report).unlink(missing_ok=True)
    materialized = materialize_chunked(
        frame,
        registry.specs,
        hh_chunk=args.hh_chunk,
        batch=args.batch,
        period=PERIOD,
        dropped_manifest_path=args.checkpoint_dir / "held_back_columns.json",
        summary_path=summary_path,
        fills_manifest_path=args.checkpoint_dir / "reviewed_null_fills.json",
    )
    admin_matrix = materialized.matrix
    compiled_specs = materialized.compiled_specs
    chunk_stats = materialized.chunk_stats
    log(
        f"materialized admin: {len(compiled_specs)} measures over "
        f"{len(chunk_stats)} chunks, nnz {admin_matrix.nnz:,} "
        f"({time.time() - started:.1f}s)"
    )
    if len(compiled_specs) != len(registry):
        raise SystemExit(
            f"Compiled admin surface has {len(compiled_specs)} measures but "
            f"{len(registry)} specs were declared; admin targets must never "
            "disappear silently between compile and materialization."
        )
    admin_roles = [cd_surface.target_record(spec) for spec in compiled_specs]

    populations = ladder_population(args.ladder, sorted(set(geographies) | {"cd"}))
    if args.soi_mode == SOI_MODE_STATE_CD:
        _attach_pro_rata_populations(admin_roles, populations["cd"])
        _require_pro_rata_populations(admin_roles)
    pop_roles, pop_matrix, pop_dropped = population_targets(
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
    pop_specs = population_target_specs(
        [record["name"] for record in pop_roles],
        [record["value"] for record in pop_roles],
    )
    log(f"population measures: {len(pop_specs)} ({geographies})")
    roles = admin_roles + pop_roles
    holdout = cd_surface.assign_target_roles(roles, fraction=holdout_fraction)
    log(
        f"CD holdout: {holdout['held_units']}/{holdout['eligible_units']} units, "
        f"{holdout['held_targets']} district targets held out of calibration"
    )
    matrix = sparse.vstack([admin_matrix, pop_matrix], format="csr")
    del admin_matrix, pop_matrix
    struct = extract_struct_tables(frame)
    n_households = frame.n("household")
    del frame
    gc.collect()

    _checkpoint_h5, _registry, digests = write_lean_checkpoint(
        struct,
        matrix,
        (*compiled_specs, *pop_specs),
        roles,
        args.checkpoint_dir,
    )
    matrix_shape = [int(value) for value in matrix.shape]
    matrix_nnz = int(matrix.nnz)
    del matrix, struct
    gc.collect()
    holdout_identity = {
        key: holdout[key]
        for key in ("unit", "salt", "fraction", "held_units", "held_targets")
    }
    (args.checkpoint_dir / "run_identity.json").write_text(
        json.dumps(
            {
                "staging_h5": str(Path(args.staging_h5).resolve()),
                "staging_sha256": staging_sha,
                "ladder_sha256": ladder_sha,
                "households": n_households,
                "n_targets": len(roles),
                "target_registry_sha256": digests["target_registry_sha256"],
                "target_roles_sha256": digests["target_roles_sha256"],
                "lean_h5_sha256": digests["lean_h5_sha256"],
                "target_matrix": {
                    "file": TARGET_MATRIX_FILENAME,
                    "sha256": digests["target_matrix_sha256"],
                    "format": "csr_float32",
                    "shape": matrix_shape,
                    "nnz": matrix_nnz,
                },
                "declared_admin_specs": len(registry),
                "compiled_admin_specs": len(compiled_specs),
                "population_cells_dropped": pop_dropped,
                "soi_mode": args.soi_mode,
                "cd_holdout": holdout_identity,
                "sampling": sampling,
            },
            indent=2,
        )
    )
    (args.checkpoint_dir / "materialize_rss.json").write_text(
        json.dumps(
            {
                "soi_mode": args.soi_mode,
                "families": families,
                "n_admin": len(compiled_specs),
                "n_population": len(pop_specs),
                "population_cells_dropped": pop_dropped,
                "hh_chunk": args.hh_chunk,
                "chunk_stats": chunk_stats,
                "target_matrix": {"shape": matrix_shape, "nnz": matrix_nnz},
                "carrier_check": materialized.carrier_check,
                "soi_surface": surface.soi_receipt,
                "cd_holdout": holdout,
                "sampling": sampling,
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


#: Outputs of the calibrate stage that describe one materialization; a new
#: materialize removes them so none can outlive the surface it described.
CALIBRATION_OUTPUT_FILENAMES = (
    "weights_latest.npz",
    "calibration_summary.json",
    "calibration_diagnostics.json",
    "consumer_export.json",
    "consumer_reviewed_null_fills.json",
    "spine_qa.json",
)


#: The fixed solver settings, shared by the solve and the stamp describing it.
SOLVER_METHOD = "adam"
SOLVER_LEARNING_RATE = 0.02
SOLVER_MASS = "conserve"


def _solver_settings(args) -> dict:
    """The calibrate-stage settings a resume or reuse must share."""

    return {
        "method": SOLVER_METHOD,
        "learning_rate": SOLVER_LEARNING_RATE,
        "mass": SOLVER_MASS,
        "max_weight_ratio": args.max_weight_ratio,
        "target_loss_cap": args.target_loss_cap,
        "l2_lambda": args.l2_lambda,
        "seed": args.seed,
        "epoch_batch": args.epoch_batch,
    }


def _weights_digest(weights) -> str:
    """Digest of a calibrated household weight vector (float64 bytes)."""

    return hashlib.sha256(
        np.ascontiguousarray(weights, dtype=np.float64).tobytes()
    ).hexdigest()


def _run_identity_digest(identity: dict) -> str:
    """Content digest of a run identity (canonical JSON)."""

    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _verify_checkpoint_digests(checkpoint_dir: Path, identity: dict) -> None:
    """Refuse a registry, roles file or matrix whose bytes changed."""

    checks = (
        (
            checkpoint_dir / "target_registry.json",
            identity.get("target_registry_sha256"),
        ),
        (checkpoint_dir / TARGET_ROLES_FILENAME, identity.get("target_roles_sha256")),
        (
            checkpoint_dir / TARGET_MATRIX_FILENAME,
            (identity.get("target_matrix") or {}).get("sha256"),
        ),
    )
    if "lean_h5_sha256" in identity:
        checks += (
            (checkpoint_dir / "target_frame_lean.h5", identity["lean_h5_sha256"]),
        )
    for path, recorded in checks:
        if not path.exists() or _sha256(path) != recorded:
            raise SystemExit(
                f"{path.name} changed since materialize (or the run identity "
                "records none); the checkpoint and surface no longer agree. "
                "Re-run --stage materialize."
            )


def _require_current_artifact(
    identity: dict,
    *,
    consumer_export: dict,
    summary: dict,
    spine_qa: dict | None,
    out_h5: Path,
    out_h5_sha256: str,
    stage: str,
) -> None:
    """Refuse a calibrated H5, or evidence about it, from another calibration.

    The H5's bytes must be the ones ``consumer_export.json`` records; that
    export and the calibration summary must name the same materialization and
    the same weights; and QA evidence, when given, must be of these bytes and
    this materialization. The path is not compared: the sha binds the bytes
    wherever they are reached from.
    """

    if "target_roles_sha256" not in identity:
        return
    digest = _run_identity_digest(identity)
    if consumer_export.get("run_identity_sha256") != digest:
        raise SystemExit(
            "consumer_export.json was written for another materialization "
            f"(or records none); re-run --stage calibrate before --stage {stage}."
        )
    if consumer_export.get("out_h5_sha256") != out_h5_sha256:
        raise SystemExit(
            f"{out_h5} is not the calibrated H5 this checkpoint's calibrate stage "
            f"wrote; re-run --stage calibrate before --stage {stage}."
        )
    weights_sha = consumer_export.get("weights_sha256")
    if weights_sha is None or summary.get("weights_sha256") != weights_sha:
        raise SystemExit(
            "The calibrated H5 and calibration_summary.json describe different "
            "weights (an interrupted recalibration?); re-run --stage calibrate "
            f"before --stage {stage}."
        )
    if spine_qa is not None:
        if spine_qa.get("run_identity_sha256") != digest:
            raise SystemExit(
                "spine_qa.json was written for another materialization; re-run "
                f"--stage qa before --stage {stage}."
            )
        if spine_qa.get("artifact_sha256") != out_h5_sha256:
            raise SystemExit(
                "spine_qa.json certifies other bytes than the calibrated H5 "
                f"({str(spine_qa.get('artifact_sha256'))[:12]}… vs "
                f"{out_h5_sha256[:12]}…); re-run --stage qa before --stage "
                f"{stage}."
            )


def _require_current_calibration(identity: dict, summary: dict, *, stage: str) -> None:
    """Refuse a calibration summary from another materialization.

    A sparse-era run identity (one that records ``target_roles_sha256``)
    binds its calibration: the summary must carry that identity's digest.
    Pre-sparse checkpoints stay readable (finalize's read-only path); package
    refuses them separately (no sampling block).
    """

    if "target_roles_sha256" not in identity:
        return
    if summary.get("run_identity_sha256") != _run_identity_digest(identity):
        raise SystemExit(
            "calibration_summary.json does not belong to this checkpoint's "
            "materialization (its run identity differs or is not recorded); "
            f"re-run --stage calibrate before --stage {stage}."
        )


def load_checkpoint_surface(checkpoint_dir: Path, identity: dict | None = None):
    """The lean frame, design weights, registry, roles and matrix of a checkpoint.

    With ``identity`` (the materialize-time run identity) the registry, the
    roles and the matrix must still be the bytes materialize wrote. A
    checkpoint from before the sparse matrix (dense measure columns in the
    lean H5) is refused: re-run ``--stage materialize``.
    Returns ``(frame, design_weights, registry, roles, matrix)``.
    """

    from microcosm.calibrate import TargetRegistry

    matrix_path = checkpoint_dir / TARGET_MATRIX_FILENAME
    roles_path = checkpoint_dir / TARGET_ROLES_FILENAME
    registry_path = checkpoint_dir / "target_registry.json"
    if not matrix_path.exists() or not roles_path.exists():
        raise SystemExit(
            f"{checkpoint_dir} has no {TARGET_MATRIX_FILENAME} or "
            f"{TARGET_ROLES_FILENAME}: it predates the sparse target matrix "
            "(its measures are dense H5 columns). Re-run --stage materialize "
            "with the current tool."
        )
    if identity is not None:
        _verify_checkpoint_digests(checkpoint_dir, identity)
    registry = TargetRegistry.from_json(registry_path)
    roles = json.loads(roles_path.read_text())
    matrix = cd_surface.load_target_matrix(matrix_path)
    frame, design_weights = load_lean_frame(checkpoint_dir / "target_frame_lean.h5")
    n_households = frame.n("household")
    if identity is not None and n_households != identity.get("households"):
        raise SystemExit(
            f"Lean checkpoint has {n_households} households but the run "
            f"identity pins {identity.get('households')}."
        )
    if [role["name"] for role in roles] != [spec.name for spec in registry.specs]:
        raise SystemExit(
            f"{TARGET_ROLES_FILENAME} is not row-aligned with target_registry.json."
        )
    if matrix.shape != (len(registry), n_households):
        raise SystemExit(
            f"{TARGET_MATRIX_FILENAME} is {matrix.shape}, not "
            f"{len(registry)} targets x {n_households} households."
        )
    return frame, design_weights, registry, roles, matrix


def calibrate_surface(
    frame,
    target_set,
    *,
    epochs: int,
    epoch_batch: int,
    max_weight_ratio: float,
    target_loss_cap: float,
    l2_lambda: float,
    seed: int,
    warm: np.ndarray | None = None,
    done: int = 0,
    on_batch=None,
):
    """Epoch-batched warm-start calibration of ``target_set``.

    ``target_set`` comes from ``cd_surface.calibration_target_set``, which
    builds only the training targets, each a callable row of the checkpoint
    CSR. Each batch calls the kernel's ``calibrate``, which compiles those
    rows into its own CSR constraint matrix. Returns ``(result, epochs_done)``.
    """

    from microcosm.calibrate import calibrate

    batch = epoch_batch if epoch_batch > 0 else epochs
    result = None
    while done < epochs:
        this_batch = min(batch, epochs - done)
        batch_started = time.time()
        result = calibrate(
            frame,
            target_set,
            weight_entity="household",
            method=SOLVER_METHOD,
            epochs=this_batch,
            learning_rate=SOLVER_LEARNING_RATE,
            mass=SOLVER_MASS,
            max_weight_ratio=max_weight_ratio,
            target_loss_cap=target_loss_cap,
            l2_lambda=l2_lambda,
            seed=seed,
            warm_start_weights=warm,
        )
        done += this_batch
        warm = result.weights.copy()
        if on_batch is not None:
            on_batch(warm, done)
        log(
            f"batch -> {done}/{epochs} ep, "
            f"{time.time() - batch_started:.1f}s, "
            f"loss={result.final_loss:.5f}, "
            f"within10%={result.fraction_within_10pct:.2%}, "
            f"ESS={result.effective_sample_size:,.0f}"
        )
    return result, done


def _origin_columns(frame) -> dict:
    """The household columns ``weight_origin_summary`` groups by, if present."""

    households = frame.table("household")
    columns = {
        "spine": "household_spine",
        "source_id": "household_source_id",
        "state": "state_fips",
        "district": "congressional_district_geoid",
    }
    return {
        key: households[column].to_numpy() if column in households.columns else None
        for key, column in columns.items()
    }


def calibration_evidence(
    *,
    frame,
    roles: list[dict],
    matrix,
    design_weights: np.ndarray,
    weights: np.ndarray,
    target_loss_cap: float,
) -> dict:
    """The sparse-surface evidence the calibration summary adds.

    The CD holdout scored against the pro-rata baseline, and ESS over rows
    and distinct households with household-weight share by spine, at the
    design and the calibrated weights.
    """

    origin = _origin_columns(frame)
    return {
        "n_targets_on_surface": len(roles),
        "n_holdout_targets": len(cd_surface.holdout_rows(roles)),
        "checkpoint_matrix": {
            "format": "csr_float32",
            "shape": [int(x) for x in matrix.shape],
            "nnz": int(matrix.nnz),
        },
        "weight_origin": {
            "design": cd_surface.weight_origin_summary(design_weights, **origin),
            "calibrated": cd_surface.weight_origin_summary(weights, **origin),
        },
        "cd_holdout": cd_surface.score_cd_holdout(
            roles,
            matrix,
            design_weights=design_weights,
            final_weights=weights,
            cap=target_loss_cap,
        ),
    }


def do_calibrate(args) -> None:
    from microcosm.calibrate import write_calibration_diagnostics

    identity = _verify_run_identity(args)
    frame, design_weights, registry, roles, matrix = load_checkpoint_surface(
        args.checkpoint_dir, identity
    )
    n_households = frame.n("household")
    target_set = cd_surface.calibration_target_set(
        roles, matrix, n_households, specs=registry.specs
    )
    log(
        f"calibrate: households={n_households}, targets={len(target_set)} "
        f"trained + {len(roles) - len(target_set)} held out, "
        f"nnz={matrix.nnz:,}, design_total={design_weights.sum():,.0f}"
    )

    stamp = _run_identity_digest(identity)
    settings = _solver_settings(args)
    resume_npz = args.checkpoint_dir / "weights_latest.npz"
    warm, done = None, 0
    if args.resume and resume_npz.exists():
        saved = np.load(resume_npz)
        saved_stamp = (
            str(saved["run_identity_sha256"])
            if "run_identity_sha256" in saved
            else None
        )
        saved_settings = (
            json.loads(str(saved["solver_settings"]))
            if "solver_settings" in saved
            else None
        )
        if saved_stamp != stamp or saved_settings != settings:
            raise SystemExit(
                "weights_latest.npz was calibrated for a different "
                "materialization (staging, surface, holdout or sample) or "
                "under different solver settings, or predates the stamp; "
                "refusing to warm-start from it. Delete it to recalibrate."
            )
        warm, done = saved["weights"], int(saved["epochs_done"])
        if len(warm) != n_households:
            raise SystemExit(
                f"weights_latest.npz carries {len(warm)} weights but the "
                f"checkpoint has {n_households} households."
            )
        log(f"RESUME from {done} epochs")
    if done >= args.epochs:
        summary_path = args.checkpoint_dir / "calibration_summary.json"
        previous = _load_json(summary_path)
        if (
            previous.get("run_identity_sha256") == stamp
            and previous.get("solver_settings") == settings
            and previous.get("weights_sha256") == _weights_digest(warm)
        ):
            log(
                f"calibration already complete at {done} epochs and "
                "the calibration summary exists; nothing to do (delete "
                "weights_latest.npz to recalibrate)."
            )
            _write_calibrated_artifact(
                args, np.asarray(warm, dtype=np.float64), identity, settings
            )
            return
        raise SystemExit(
            f"weights_latest.npz reports {done} epochs (>= --epochs "
            f"{args.epochs}) but calibration_summary.json is missing or "
            "belongs to another materialization, solver settings or weights "
            "(a summary written before the weights digest). Delete "
            "weights_latest.npz or run without --resume to recalibrate, or "
            "raise --epochs."
        )

    # A solve replaces every output describing the calibration: none of the
    # previous ones may outlive it if this run stops part way.
    for name in CALIBRATION_OUTPUT_FILENAMES:
        if name != "weights_latest.npz":
            (args.checkpoint_dir / name).unlink(missing_ok=True)

    def save(weights: np.ndarray, epochs_done: int) -> None:
        np.savez(
            resume_npz,
            weights=weights,
            epochs_done=epochs_done,
            initial_weights=design_weights,
            staging_sha256=np.str_(identity["staging_sha256"]),
            run_identity_sha256=np.str_(stamp),
            solver_settings=np.str_(json.dumps(settings, sort_keys=True)),
        )

    started = time.time()
    result, done = calibrate_surface(
        frame,
        target_set,
        epochs=args.epochs,
        epoch_batch=args.epoch_batch,
        max_weight_ratio=args.max_weight_ratio,
        target_loss_cap=args.target_loss_cap,
        l2_lambda=args.l2_lambda,
        seed=args.seed,
        warm=warm,
        done=done,
        on_batch=save,
    )

    if result.problem.skipped:
        skipped = [getattr(item, "name", str(item)) for item in result.problem.skipped]
        raise SystemExit(
            f"Calibration compiled {len(skipped)} target(s) away "
            f"(e.g. {skipped[:5]}); the surface silently shrank. Fix the "
            "measures or the targets before shipping."
        )
    summary = {
        "run_identity_sha256": stamp,
        "solver_settings": settings,
        "weights_sha256": _weights_digest(result.weights),
        "households": n_households,
        "n_targets": result.problem.n_targets,
        "families": args.families,
        "geographies": args.geographies,
        "soi_mode": identity.get("soi_mode"),
        "matrix_format": result.options["matrix_format"],
        "matrix_shape": [int(x) for x in result.problem.matrix.shape],
        "matrix_nnz": int(result.problem.matrix.nnz),
        "epochs": args.epochs,
        "epoch_batch": args.epoch_batch,
        "max_weight_ratio": args.max_weight_ratio,
        "target_loss_cap": args.target_loss_cap,
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
        "sampling": identity.get("sampling"),
        **calibration_evidence(
            frame=frame,
            roles=roles,
            matrix=matrix,
            design_weights=design_weights,
            weights=np.asarray(result.weights, dtype=np.float64),
            target_loss_cap=args.target_loss_cap,
        ),
        "total_wall_seconds": round(time.time() - started, 1),
        "peak_rss_gb": round(rss(), 3),
    }
    _write_calibrated_artifact(
        args, np.asarray(result.weights, dtype=np.float64), identity, settings
    )

    holdout = summary["cd_holdout"]
    outcome = write_calibration_diagnostics(
        result,
        args.checkpoint_dir / "calibration_diagnostics.json",
        target_registry=registry,
        build={
            "dataset_role": "non_default_local_area",
            "families": args.families,
            "geographies": args.geographies,
            "soi_mode": identity.get("soi_mode"),
            "epochs": args.epochs,
            "epoch_batch": args.epoch_batch,
            "total_wall_seconds": summary["total_wall_seconds"],
            "peak_rss_gb": summary["peak_rss_gb"],
            "ess_fraction": summary["ess_fraction"],
            "mass_conserved_ratio": summary["mass_conserved_ratio"],
            "n_holdout_targets": summary["n_holdout_targets"],
            "sampling_rung": (identity.get("sampling") or {}).get("rung"),
        },
    )
    summary["calibration_diagnostics"] = (
        {
            "status": "available",
            "schema_version": outcome.schema_version,
            "sha256": outcome.sha256,
        }
        if outcome.status == "available"
        else {
            "status": "failed",
            "expected_schema_version": outcome.expected_schema_version,
            "error_code": outcome.error_code,
            "message": outcome.message,
        }
    )
    (args.checkpoint_dir / "calibration_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    log(
        f"calibrate stage complete: loss={summary['final_loss']}, "
        f"within10%={summary['fraction_within_10pct']:.2%}, "
        f"diagnostics={outcome.status}"
        + (
            f"; CD holdout {holdout['n_targets']} targets: calibrated "
            f"{holdout['calibrated']['mean_abs_rel_error']:.2%} vs pro-rata "
            f"{holdout['pro_rata_baseline']['mean_abs_rel_error']:.2%} mean "
            "abs rel error"
            if holdout.get("n_targets")
            else ""
        )
    )


def _resample_like_materialize(frame, identity: dict):
    """Re-draw the development rung materialize drew, or refuse.

    A full-rung (or pre-sampling) identity returns the frame unchanged. A
    sampled one re-draws with the recorded fraction and seed and must select
    exactly the recorded households.
    """

    sampling = identity.get("sampling") or {}
    if not sampling.get("sampled"):
        return frame
    sampled, receipt = cd_surface.sample_staging_frame(
        frame,
        fraction=float(sampling["sample_fraction"]),
        seed=int(sampling["sample_seed"]),
    )
    recorded = sampling.get("selected_household_ids_sha256")
    if receipt.get("selected_household_ids_sha256") != recorded:
        raise SystemExit(
            "Re-drawing the recorded development rung selected different "
            "households than materialize did; refusing to attach weights."
        )
    return sampled


def _write_calibrated_artifact(
    args, weights: np.ndarray, identity: dict, settings: dict
) -> None:
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

    from microcosm.frame import Frame, WeightKind, Weights

    frame = _load_staging_frame(args.staging_h5)
    _require_local_hours(frame, _load_json(_staging_summary_path(args)))
    frame = _resample_like_materialize(frame, identity)
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
                "out_h5_sha256": _sha256(args.out_h5),
                "run_identity_sha256": _run_identity_digest(identity),
                "weights_sha256": _weights_digest(weights),
                "solver_settings": settings,
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
    """Per-spine composition evidence (the microcosm#403 skew signature)."""

    from microcosm.build.us_runtime.base_pool import spine_column

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

    from microcosm.build.us_runtime.base_pool import spine_column

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

    from microcosm.frame.adapters.policyengine_us import PolicyEngineUSEngine

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
    # Recorded, not re-verified: finalize and package verify the identity.
    qa_identity = _load_json(args.checkpoint_dir / "run_identity.json")
    payload = {
        "run_identity_sha256": (
            _run_identity_digest(qa_identity) if qa_identity else None
        ),
        "period": PERIOD,
        "variable": "ssi",
        "artifact": str(Path(args.out_h5).resolve()),
        "artifact_sha256": _sha256(args.out_h5),
        "plain_consumption": True,
        "per_spine": per_spine,
        "note": (
            "microcosm#403 re-measure: per-spine SSI incidence and intensity, "
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
                "release) carries the microcosm#507 SSI aged-band take-up "
                "collapse: the 65+ one-shot Bernoulli seed at 8.4% collapses "
                "the aged SSI baseline to roughly 0.94M against SSA's 2.42M. "
                "The QRF transfer replicates donor SSI participation onto "
                "the ACS spine, so both spines of this artifact inherit the "
                "defect by construction."
            ),
            "treatment": (
                "microcosm#508 owns the fix; when the corrected base "
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
                "distorted (microcosm#393, open at build time; roughly 4.6x "
                "SOI's loss-return prevalence measured in the Build J dense "
                "remedy experiments). The QRF transfer replicates the donor "
                "distribution onto the ACS spine."
            ),
            "treatment": "Tracked via microcosm#393; propagates on re-transfer.",
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
                "microcosm#492)."
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


def state_cd_reviewed_limitations(materialize_rss: dict) -> list[dict]:
    """Reviewed limitations a ``state_cd`` surface adds to the register."""

    if materialize_rss.get("soi_mode") != SOI_MODE_STATE_CD:
        return []
    surface = materialize_rss.get("soi_surface") or {}
    holdout = materialize_rss.get("cd_holdout") or {}
    excluded = surface.get("excluded_cd_states") or {}
    exclusion = (
        " District rows of state(s) "
        + ", ".join(sorted(excluded))
        + " are excluded: "
        + "; ".join(f"{state}: {why}" for state, why in sorted(excluded.items()))
        if excluded
        else " No state's district rows are excluded for the mapping."
    )
    return [
        {
            "id": "cd_soi_117th_plan_population_crosswalk",
            "status": "reviewed_construction",
            "reason": (
                "The SOI congressional-district file (22incd.csv, TY2022) is "
                "tabulated on the 117th-Congress plan. Its district rows are "
                "mapped onto the households' 119th-plan districts by the "
                "packaged 2020-block population crosswalk (built from the "
                "block plan registry since #1043), so each 119th district "
                "target assumes returns spread with population inside every "
                "117th/119th intersection." + exclusion
            ),
            "treatment": (
                "A household 117th-plan district column "
                "(congressional_district_geoid__117th_congress, from the "
                "location v1 block draw with the plan registry attached) lets "
                "these targets bind as exact block sums with no crosswalk."
            ),
            "excluded_states": surface.get("excluded_cd_states"),
            "crosswalk": surface.get("crosswalk"),
            "calibration_blocker": False,
        },
        {
            "id": "cd_soi_one_vintage_per_state_concept",
            "status": "reviewed_construction",
            "reason": surface.get("vintage_rule_description"),
            "rebase_factor_by_measure": surface.get("rebase_factor_by_measure"),
            "level_bridges": surface.get("level_bridges"),
            "level_bridge_factor_by_measure": surface.get(
                "level_bridge_factor_by_measure"
            ),
            "dropped": surface.get("dropped"),
            "calibration_blocker": False,
        },
        {
            "id": "cd_soi_defective_district_columns_excluded",
            "status": "reviewed_exclusion",
            "reason": (
                "District-file measures that read the wrong IRS column are "
                "off the surface at both geographies (microcosm#1038)."
            ),
            "measures": surface.get("defective_cd_file_measures"),
            "calibration_blocker": False,
        },
        {
            "id": "cd_holdout_sealed",
            "status": "reviewed_construction",
            "reason": (
                f"{holdout.get('held_targets')} district SOI targets in "
                f"{holdout.get('held_units')} (state x concept family) units "
                f"(fraction {holdout.get('fraction')}, salt "
                f"{holdout.get('salt')}) never reach the calibrator; they are "
                "scored against a pro-rata baseline in calibration_summary.json "
                "(cd_holdout) and gate_summary.json (gates.cd_holdout)."
            ),
            "calibration_blocker": False,
        },
        {
            "id": "cd_soi_sigma_absent",
            "status": "reviewed_data_gap",
            "reason": (
                "The pinned facts feed carries no uncertainty for any IRS SOI "
                "fact, so no target carries sigma and the loss is unchanged "
                "(fixed-scale capped relative error)."
            ),
            "calibration_blocker": False,
        },
    ]


#: Finalize's district ESS gate. A congressional district collapses when
#: calibration leaves its Kish ESS below this share of its Kish ESS at the
#: design weights. At the release's ACS share (0.5), microcosm#1078's solves
#: (full surface and two holdout folds) leave 15-24 districts below a quarter
#: unpenalized, and none with the chi-square design-weight penalty.
DISTRICT_ESS_RELATIVE_FLOOR = 0.25
#: The absolute floor on a district's calibrated Kish ESS (0 turns it off).
#: Those solves' smallest district ESS is 7.8-11.7 unpenalized and at least
#: 18.6 penalized.
DISTRICT_ESS_FLOOR = 15.0
DISTRICT_ESS_GATE = "district_ess_collapse"


def _district_ess(weight_summary) -> dict[str, float] | None:
    """A weight summary's per-district Kish ESS as finite, nonnegative floats.

    None when the summary records none, or any value is not such a number.
    """

    by_district = (
        weight_summary.get("effective_sample_size_by_district")
        if isinstance(weight_summary, dict)
        else None
    )
    if not isinstance(by_district, dict) or not by_district:
        return None
    values = {}
    for district, ess in by_district.items():
        if isinstance(ess, bool) or not isinstance(ess, (int, float)):
            return None
        if not np.isfinite(ess) or ess < 0:
            return None
        values[str(district)] = float(ess)
    return values


def district_ess_collapse_gate(
    weight_origin: dict | None,
    *,
    relative_floor: float = DISTRICT_ESS_RELATIVE_FLOOR,
    absolute_floor: float = DISTRICT_ESS_FLOOR,
    blocking: bool = False,
) -> dict:
    """The district ESS gate entry, from the calibration summary's weight origin.

    Reads ``effective_sample_size_by_district`` at the design and the
    calibrated weights. A district collapses when its calibrated Kish ESS is
    below ``relative_floor`` times its design-weight Kish ESS, and is below
    the floor when its calibrated Kish ESS is below ``absolute_floor``; a
    floor of 0 turns its check off. ``criteria_met`` says whether no district
    does either (None when the evidence is missing or malformed).

    The release contract publishes only gates that pass, so a report-only
    entry passes whatever it measured. Under ``blocking`` the entry passes
    only when the criteria are met.
    """

    if not 0.0 <= relative_floor <= 1.0:
        raise ValueError(f"relative_floor must be in [0, 1], got {relative_floor!r}.")
    if not (np.isfinite(absolute_floor) and absolute_floor >= 0.0):
        raise ValueError(
            f"absolute_floor must be finite and >= 0, got {absolute_floor!r}."
        )
    origin = weight_origin if isinstance(weight_origin, dict) else {}
    design = _district_ess(origin.get("design"))
    calibrated = _district_ess(origin.get("calibrated"))
    gate = {
        "blocking": bool(blocking),
        "report_only": not blocking,
        "relative_floor": float(relative_floor),
        "absolute_floor": float(absolute_floor),
        "note": (
            "A congressional district collapses when its calibrated Kish ESS "
            "is below relative_floor x its design-weight Kish ESS, and is "
            "below the floor when its calibrated Kish ESS is below "
            "absolute_floor (0 turns a check off); from "
            "calibration_summary.json weight_origin."
        ),
    }
    failures = []
    if design is None or calibrated is None:
        failures.append(
            "calibration_summary.json weight_origin records no finite, "
            "nonnegative effective_sample_size_by_district at both the design "
            "and the calibrated weights"
        )
    elif set(design) != set(calibrated):
        failures.append(
            "the design and calibrated per-district ESS cover different districts"
        )
    if failures:
        return {
            **gate,
            "passed": not blocking,
            "criteria_met": None,
            "failures": failures,
        }

    rows = [
        {
            "district": district,
            "calibrated_ess": calibrated[district],
            "design_ess": design[district],
            "ratio": (
                calibrated[district] / design[district]
                if design[district] > 0
                else None
            ),
        }
        for district in sorted(design)
    ]
    collapsed = sorted(
        (
            row
            for row in rows
            if row["calibrated_ess"] < relative_floor * row["design_ess"]
        ),
        key=lambda row: (row["ratio"], row["district"]),
    )
    below_floor = sorted(
        (row for row in rows if row["calibrated_ess"] < absolute_floor),
        key=lambda row: (row["calibrated_ess"], row["district"]),
    )
    smallest = min(rows, key=lambda row: (row["calibrated_ess"], row["district"]))
    rated = [row for row in rows if row["ratio"] is not None]
    lowest = (
        min(rated, key=lambda row: (row["ratio"], row["district"])) if rated else None
    )
    criteria_met = not collapsed and not below_floor
    return {
        **gate,
        "passed": criteria_met or not blocking,
        "criteria_met": criteria_met,
        "failures": [],
        "n_districts": len(rows),
        "n_collapsed": len(collapsed),
        "n_below_floor": len(below_floor),
        "min_calibrated_ess": smallest["calibrated_ess"],
        "min_calibrated_ess_district": smallest["district"],
        "min_ess_ratio": lowest["ratio"] if lowest else None,
        "min_ess_ratio_district": lowest["district"] if lowest else None,
        "collapsed": collapsed,
        "below_floor": below_floor,
    }


def district_ess_collapse_limitation(gate: dict) -> dict:
    """The reviewed-limitations entry stating the district ESS gate's result."""

    if gate["criteria_met"] is None:
        measured = (
            "The district ESS gate could not be evaluated: "
            + "; ".join(gate["failures"])
            + "."
        )
    else:
        n = gate["n_districts"]
        checks = []
        if gate["relative_floor"] > 0:
            checks.append(
                f"below {gate['relative_floor']:g} x their design-weight Kish "
                f"ESS: {gate['n_collapsed']} of {n}"
            )
        if gate["absolute_floor"] > 0:
            checks.append(
                f"below {gate['absolute_floor']:g}: {gate['n_below_floor']} of {n}"
            )
        ratio = gate["min_ess_ratio"]
        measured = (
            (
                "Congressional districts with a calibrated Kish ESS "
                + "; ".join(checks)
                + "."
                if checks
                else f"No district ESS check is on for the {n} districts."
            )
            + f" Smallest calibrated district ESS: "
            f"{gate['min_calibrated_ess']:.1f} "
            f"({gate['min_calibrated_ess_district']})"
            + (
                f"; smallest ratio to design ESS: {ratio:.3f} "
                f"({gate['min_ess_ratio_district']})."
                if ratio is not None
                else "."
            )
        )
    if not gate["blocking"]:
        mode = (
            " The gate is report-only for this build; "
            "--district-ess-gate-blocking makes it a hard failure."
        )
    elif gate["passed"]:
        mode = " The gate is blocking for this build and passed."
    else:
        mode = (
            " The gate is blocking for this build and fails, so the build is "
            "not simulation-ready."
        )
    return {
        "id": "district_effective_sample_size_gate",
        "status": "recorded_concentration",
        "reason": measured + mode,
        "gate": DISTRICT_ESS_GATE,
        "criteria_met": gate["criteria_met"],
        "n_collapsed": gate.get("n_collapsed"),
        "n_below_floor": gate.get("n_below_floor"),
        "relative_floor": gate["relative_floor"],
        "absolute_floor": gate["absolute_floor"],
        "blocking": gate["blocking"],
        "calibration_blocker": not gate["passed"],
    }


def _require_blocking_district_ess_gate(gate_report: dict, args) -> None:
    """Under ``--district-ess-gate-blocking``, refuse a report finalize did not block on.

    A finalize run without the flag (or at other floors, or from before the
    gate existed) records a gate that passes whatever it measured, so
    packaging its report under the flag would ship what the flag refuses.
    """

    if not args.district_ess_gate_blocking:
        return
    gates = gate_report.get("gates")
    gate = gates.get(DISTRICT_ESS_GATE) if isinstance(gates, dict) else None
    if not (
        isinstance(gate, dict)
        and gate.get("blocking") is True
        and gate.get("passed") is True
        and gate.get("relative_floor") == args.district_ess_relative_floor
        and gate.get("absolute_floor") == args.district_ess_floor
    ):
        raise SystemExit(
            f"--district-ess-gate-blocking: the finalize gate report carries no "
            f"passing blocking {DISTRICT_ESS_GATE} gate at relative floor "
            f"{args.district_ess_relative_floor:g} and floor "
            f"{args.district_ess_floor:g}; re-run --stage finalize with the "
            "same flags."
        )


def _local_hours_gate(frame, staging_summary: dict):
    audit = staging_summary.get("reviewed_engine_input_nulls")
    if not isinstance(audit, list) or not all(isinstance(item, dict) for item in audit):
        raise SystemExit("Local hours gate requires the staging input-null audit.")
    return acs_local_hours_signal_gate(frame, source_null_audit=audit)


def _require_local_hours(frame, staging_summary: dict) -> None:
    gate = _local_hours_gate(frame, staging_summary)
    if not gate.passed:
        raise SystemExit("Local hours coverage failed: " + "; ".join(gate.failures))


def do_finalize(args) -> None:
    from microcosm.build.us_runtime.hours_worked import (
        US_HOURS_WORKED_POOL_OUTPUT_COLUMNS,
        us_hours_worked_signal_gate,
    )
    from microcosm.build.us_runtime.puma_ladder import (
        load_us_puma_ladder,
        us_puma_ladder_gate,
    )
    from microcosm.calibrate import TargetRegistry

    staging_summary = _load_json(_staging_summary_path(args))
    diagnostics = _load_json(args.checkpoint_dir / "calibration_summary.json")
    if not diagnostics:
        raise SystemExit(
            f"No calibration summary under {args.checkpoint_dir}; run "
            "--stage calibrate first."
        )
    identity = _verify_run_identity(args)
    _require_current_calibration(identity, diagnostics, stage="finalize")
    if "target_roles_sha256" in identity:
        _verify_checkpoint_digests(args.checkpoint_dir, identity)
    ladder_sha = _sha256(args.ladder)
    if ladder_sha != identity.get("ladder_sha256"):
        raise SystemExit(
            f"--ladder {args.ladder} (sha {ladder_sha[:12]}…) is not the "
            "ladder the surface was materialized with "
            f"({str(identity.get('ladder_sha256'))[:12]}…)."
        )
    materialize_rss = _load_json(args.checkpoint_dir / "materialize_rss.json")
    registry_path = args.checkpoint_dir / "target_registry.json"
    roles_path = args.checkpoint_dir / TARGET_ROLES_FILENAME
    if roles_path.is_file():
        targets = json.loads(roles_path.read_text())
    elif registry_path.is_file():
        registry = TargetRegistry.from_json(registry_path)
        targets = [
            {"name": spec.name, "family": spec.family} for spec in registry.specs
        ]
    else:
        # Read-only compatibility for checkpoints created before the target
        # registry became the materialize-stage artifact. Current materialize
        # runs always write target_registry.json.
        targets = _load_json(args.checkpoint_dir / "targets.json") or []
    spine_qa = _load_json(args.checkpoint_dir / "spine_qa.json")
    consumer_export = _load_json(args.checkpoint_dir / "consumer_export.json")

    # The hours gate certifies specific artifact bytes. Hash the calibrated
    # H5 before loading it, so the binding the package stage checks is the
    # bytes the gate actually evaluated, and refuse if they moved meanwhile.
    if not args.out_h5.exists():
        raise SystemExit(f"Calibrated H5 not found: {args.out_h5}.")
    hours_artifact_sha = _sha256(args.out_h5)
    _require_current_artifact(
        identity,
        consumer_export=consumer_export,
        summary=diagnostics,
        spine_qa=spine_qa or None,
        out_h5=args.out_h5,
        out_h5_sha256=hours_artifact_sha,
        stage="finalize",
    )
    frame = _load_staging_frame(args.out_h5)
    local_hours_gate = _local_hours_gate(frame, staging_summary)
    households = frame.table("household")
    weights = np.asarray(frame.weights_for("household").values, dtype=np.float64)
    load_us_puma_ladder(args.ladder)
    ladder_gate = us_puma_ladder_gate(households, weights)
    composition = spine_composition(households, frame.table("person"), weights)
    # microcosm#765: refuse to package an artifact whose usual-weekly-hours
    # surface is the engine's constant-40 default (or otherwise out of band),
    # which silently no-ops SNAP's ABAWD and general work-requirement tests.
    # weeks_worked is dropped from the pool/ACS surface, so scope the gate to
    # the two hours columns it carries.
    hours_gate = us_hours_worked_signal_gate(
        frame, required_columns=US_HOURS_WORKED_POOL_OUTPUT_COLUMNS
    )
    del frame
    gc.collect()
    if _sha256(args.out_h5) != hours_artifact_sha:
        raise SystemExit(
            "The calibrated H5 changed during hours_worked_signal validation. "
            "Re-run --stage qa and --stage finalize against the current artifact."
        )

    breakdown: dict[str, int] = {}
    for target in targets:
        name = target["name"]
        if target.get("role") == cd_surface.ROLE_HOLDOUT:
            key = "cd_holdout"
        elif name.startswith("irs_soi") and (
            target.get("geography_level") == cd_surface.GEOGRAPHY_CD
        ):
            key = "soi_congressional_district"
        elif name.startswith("pop_state"):
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
        "acs_local_hours_signal": {
            "passed": local_hours_gate.passed,
            "failures": list(local_hours_gate.failures),
            "detail": dict(local_hours_gate.details),
            "artifact_sha256": hours_artifact_sha,
        },
        "us_puma_ladder_gate": {
            "passed": bool(ladder_gate.passed),
            "failures": list(ladder_gate.failures),
            "detail": dict(ladder_gate.details),
        },
        "hours_worked_signal": {
            "passed": bool(hours_gate.passed),
            "failures": list(hours_gate.failures),
            "detail": dict(hours_gate.details),
            "artifact_sha256": hours_artifact_sha,
        },
        "calibration": {
            # The cap criterion alone is near-tautological (the solver clips
            # per-row losses at the same cap); the solve must also have
            # actually improved on the design weights and conserved mass.
            # Numeric acceptance thresholds beyond that (within-10% floors,
            # per-target error bars) are maintainer-adjudicated surface
            # policy (#398-class), recorded here rather than invented.
            "passed": bool(
                diagnostics.get("final_loss", 1.0)
                < diagnostics.get("target_loss_cap", args.target_loss_cap)
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
                "n_trained": diagnostics.get("n_targets"),
                "n_holdout": diagnostics.get("n_holdout_targets"),
                "breakdown": breakdown,
                "target_matrix": materialize_rss.get("target_matrix"),
                "rung": (identity.get("sampling") or {}).get("rung"),
            },
        },
        "cd_holdout": {
            # Report-only: held-out district targets never reach the solve;
            # the comparison with the pro-rata baseline is evidence, not a
            # bound (d487 asks whether district fidelity beats pro-rata).
            "passed": True,
            "report_only": True,
            "detail": {
                key: value
                for key, value in (diagnostics.get("cd_holdout") or {}).items()
                if key != "targets"
            },
        },
        "weight_origin": {
            "passed": True,
            "report_only": True,
            "note": (
                "Household-weight share by spine and Kish ESS over rows and "
                "over distinct households (household_spine, "
                "household_source_id), at design and calibrated weights."
            ),
            "detail": diagnostics.get("weight_origin"),
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
                "Recorded evidence for the microcosm#403 weight-composition "
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

    gates[DISTRICT_ESS_GATE] = district_ess_collapse_gate(
        diagnostics.get("weight_origin"),
        relative_floor=args.district_ess_relative_floor,
        absolute_floor=args.district_ess_floor,
        blocking=args.district_ess_gate_blocking,
    )

    limitations = finalize_reviewed_limitations(staging_summary, diagnostics, spine_qa)
    limitations += state_cd_reviewed_limitations(materialize_rss)
    limitations.append(district_ess_collapse_limitation(gates[DISTRICT_ESS_GATE]))
    hard_failures = [
        name
        for name in (
            "us_puma_ladder_gate",
            "hours_worked_signal",
            "acs_local_hours_signal",
            "calibration",
            "consumer_ready",
        )
        if not gates[name]["passed"]
    ]
    # Report-only unless --district-ess-gate-blocking, so it passes otherwise.
    if not gates[DISTRICT_ESS_GATE]["passed"]:
        hard_failures.append(DISTRICT_ESS_GATE)
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


#: Staging settings a packaged build manifest records, so a reader can see how
#: the ACS spine was built without opening the staging summary.
_STAGING_ORCHESTRATION_KEYS = (
    "max_households",
    "n_estimators",
    "max_targets_per_fit",
    "acs_share",
    "chunksize",
    "seed",
    "geography_seed",
    "donor_channel",
)


def _require_uncapped_staging(staging_summary: dict) -> dict[str, object]:
    """Refuse to package a staging run that was capped, or cannot show it was not.

    ``tools/build_us_acs_multispine_base.py --max-households`` caps the ACS
    spine for smokes. The donor spine keeps every state and congressional
    district populated, so a capped run drops no ladder population cell and
    passes every finalize gate; without this check it packages into a release
    directory with a publish command and nothing in either manifest says it
    is a smoke. A summary that does not record the cap cannot establish that
    the spine is whole, so it is refused too.
    """

    orchestration = staging_summary.get("orchestration")
    if not isinstance(orchestration, dict) or "max_households" not in orchestration:
        raise SystemExit(
            "The staging summary does not record orchestration.max_households, "
            "so packaging cannot establish that the ACS spine is uncapped. "
            "Re-run staging with the current builder; a smoke's output must "
            "not be packaged."
        )
    cap = orchestration["max_households"]
    if cap is not None:
        raise SystemExit(
            f"The staging run was capped at {cap} ACS household(s) "
            "(--max-households); a smoke's output must not be packaged."
        )
    return {key: orchestration.get(key) for key in _STAGING_ORCHESTRATION_KEYS}


def _require_recorded_soi_mode(materialize_rss: dict) -> str:
    """The SOI surface the checkpoint was materialized with, or refuse.

    Later stages never re-select targets, so the mode that counts is the one
    ``--stage materialize`` recorded, not this invocation's ``--soi-mode``.
    A checkpoint that does not record a known mode cannot say which surface
    was calibrated, and its refresh recipe could not reproduce it.
    """

    soi_mode = materialize_rss.get("soi_mode")
    if soi_mode not in SOI_MODES:
        raise SystemExit(
            f"materialize_rss.json records soi_mode={soi_mode!r}, not one of "
            f"{SOI_MODES}; packaging cannot record which SOI surface was "
            "calibrated. Re-run --stage materialize with the current tool."
        )
    return soi_mode


def _require_full_rung(identity: dict) -> dict:
    """Refuse to package a development rung, or a run that cannot show its rung.

    ``--sample-fraction`` below 1 is for development runs (DESIGN.md
    "Production US stacked spine"); a release is always full scale.
    """

    sampling = identity.get("sampling")
    if not isinstance(sampling, dict) or "sampled" not in sampling:
        raise SystemExit(
            "run_identity.json records no sampling block, so packaging cannot "
            "establish that the surface was materialized at full scale. "
            "Re-run --stage materialize with the current tool."
        )
    if sampling.get("sampled") or sampling.get("rung") != "f100":
        raise SystemExit(
            f"The checkpoint was materialized on the {sampling.get('rung')} "
            "development rung; a release is packaged only at full scale "
            "(--sample-fraction 1)."
        )
    return sampling


def _require_stored_inputs(calibrated_h5: Path) -> dict[str, object]:
    """Refuse an artifact that stores a model input the installed engine lacks.

    microcosm#1026: the ACS local-area release of 2026-09-23 records
    policyengine-us 2.2.1 as its built-with engine but stores the WIC take-up
    draw under its retired name, ``would_claim_wic``, which 2.2.1 ignores.
    :mod:`microcosm.data.stored_inputs` owns the rule and the reviewed register
    of deliberately non-variable columns. The engine is the installed
    policyengine-us, the one :func:`do_package` records as
    ``build.built_with_model_package``. Only HDF metadata is read.

    Returns the gate entry the release's gate summary records.
    """

    from microcosm.data import stored_inputs

    try:
        engine = stored_inputs.installed_us_engine()
    except ImportError as error:
        raise SystemExit(
            "Refusing to package: the installed policyengine-us cannot be "
            "imported, so the artifact cannot be checked against the engine "
            f"the release records as built-with: {error}"
        ) from error
    try:
        summary = stored_inputs.require_h5_stored_inputs(calibrated_h5, engine=engine)
    except (
        stored_inputs.StoredInputRefusalError,
        stored_inputs.StoredTableLayoutError,
    ) as error:
        raise SystemExit(f"Refusing to package: {error}") from error
    return {"passed": True, "failures": [], "engine": engine.label, **summary}


def do_package(args) -> dict:
    diagnostics = _load_json(args.checkpoint_dir / "calibration_summary.json")
    diagnostics_status = diagnostics.get("calibration_diagnostics")
    if not isinstance(diagnostics_status, dict) or diagnostics_status.get(
        "status"
    ) not in {"available", "failed"}:
        raise SystemExit(
            "calibration_summary.json has no valid calibration_diagnostics status; "
            "run --stage calibrate with the current builder."
        )
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
    # Before any release directory exists: a refused smoke leaves nothing.
    staging_orchestration = _require_uncapped_staging(staging_summary)
    soi_mode = _require_recorded_soi_mode(materialize_rss)
    _require_full_rung(identity)
    _require_current_calibration(identity, diagnostics, stage="package")
    _require_blocking_district_ess_gate(gate_report, args)
    calibrated_h5 = Path(args.out_h5)
    if not calibrated_h5.exists():
        raise SystemExit(f"Calibrated H5 not found: {calibrated_h5}.")
    # microcosm#1026: the manifest below records the installed policyengine-us
    # as build.built_with_model_package, so the artifact may store no model
    # input that engine does not define. A refused artifact leaves nothing.
    stored_inputs_gate = _require_stored_inputs(calibrated_h5)
    log("hashing calibrated H5 …")
    h5_sha = _sha256(calibrated_h5)
    # The H5 and its evidence must come from this checkpoint's calibration;
    # a refused artifact leaves no release directory behind.
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
    _require_current_artifact(
        identity,
        consumer_export=consumer_export,
        summary=diagnostics,
        spine_qa=spine_qa,
        out_h5=calibrated_h5,
        out_h5_sha256=h5_sha,
        stage="package",
    )

    code = _repo_code_identity(args.allow_dirty)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    release_id = f"{RELEASE_ID_PREFIX}-{code['sha']}-{timestamp}"
    release_dir = args.out / "releases" / release_id
    release_dir.mkdir(parents=True, exist_ok=True)

    gates = gate_report.get("gates")
    hours_gate = gates.get("hours_worked_signal") if isinstance(gates, dict) else None
    if not isinstance(hours_gate, dict) or hours_gate.get("passed") is not True:
        raise SystemExit(
            "Packaging requires a present, passing hours_worked_signal gate; "
            "an old simulation_ready summary is insufficient. Re-run --stage finalize."
        )
    if hours_gate.get("artifact_sha256") != h5_sha:
        raise SystemExit(
            "The hours_worked_signal gate is missing its artifact binding or "
            "certifies different H5 bytes. Re-run --stage finalize against the "
            "current artifact."
        )
    # Old summaries can say simulation_ready despite #765. Recheck the
    # actual artifact and the source-null evidence before packaging it, and
    # bind that result to the bytes being packaged: the finalize-time report
    # is copied into the release, so a checkpoint finalized before this gate
    # existed must not ship as if it had passed it.
    finalize_hours = gate_report.get("gates", {}).get("acs_local_hours_signal")
    if not isinstance(finalize_hours, dict) or finalize_hours.get("passed") is not True:
        raise SystemExit(
            "The finalize gate report carries no passing acs_local_hours_signal; "
            "an old simulation_ready summary is insufficient. Re-run --stage "
            "finalize."
        )
    hours_frame = _load_staging_frame(calibrated_h5)
    package_hours_gate = _local_hours_gate(hours_frame, staging_summary)
    del hours_frame
    gc.collect()
    if not package_hours_gate.passed:
        raise SystemExit(
            "Local hours coverage failed: " + "; ".join(package_hours_gate.failures)
        )
    gate_report = {
        **gate_report,
        "gates": {
            **gate_report.get("gates", {}),
            "acs_local_hours_signal": {
                "passed": True,
                "failures": [],
                "detail": dict(package_hours_gate.details),
                "artifact_sha256": h5_sha,
                "checked_at_stage": "package",
            },
            "stored_inputs": {
                **stored_inputs_gate,
                "artifact_sha256": h5_sha,
                "checked_at_stage": "package",
            },
        },
    }
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
            "When the next certified default publishes (e.g. the microcosm#508 "
            "SSI fix -> O-2), refresh this artifact by re-running the chain "
            "against the new certified release H5; everything else is "
            "unchanged."
        ),
        "staging": LEGACY_STAGING_REFRESH_RECIPE,
        "release": release_refresh_recipe(
            soi_mode, (identity.get("cd_holdout") or {}).get("fraction")
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
        "code": {"branch": code["branch"], "repo": "microcosm"},
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
            "soi_mode": soi_mode,
            "soi_surface_counts": (materialize_rss.get("soi_surface") or {}).get(
                "counts"
            ),
            "target_matrix": identity.get("target_matrix"),
            "cd_holdout": identity.get("cd_holdout"),
            "sampling": identity.get("sampling"),
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
        "staging_orchestration": staging_orchestration,
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
    if diagnostics_status["status"] == "available":
        source_diagnostics = args.checkpoint_dir / "calibration_diagnostics.json"
        if not source_diagnostics.is_file():
            raise SystemExit(
                "calibration_summary.json declares available diagnostics but "
                "calibration_diagnostics.json is missing."
            )
        if _sha256(source_diagnostics) != diagnostics_status.get("sha256"):
            raise SystemExit(
                "calibration_diagnostics.json no longer matches the validated "
                "digest recorded by the calibration stage."
            )
        shutil.copy2(source_diagnostics, release_dir / "calibration_diagnostics.json")

    def _artifact(path_name: str, kind: str, local: Path) -> dict:
        return {
            "kind": kind,
            "path": path_name,
            "repo_id": HF_REPO_ID,
            "revision": release_id,
            "sha256": _sha256(local),
        }

    artifacts = {
        ARTIFACT_NAME: {
            "kind": "microdata",
            "path": ARTIFACT_FILENAME,
            "repo_id": HF_REPO_ID,
            "revision": release_id,
            "sha256": h5_sha,
        },
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
    }
    if diagnostics_status["status"] == "available":
        artifacts["calibration_diagnostics"] = _artifact(
            "calibration_diagnostics.json",
            "diagnostics",
            release_dir / "calibration_diagnostics.json",
        )

    release_manifest = {
        "schema_version": 1,
        "data_package": {"name": "microcosm-data", "version": "0.1.0"},
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
        "calibration_diagnostics": diagnostics_status,
        "artifacts": artifacts,
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
    # Check the actual final artifact, including a reused copy or the no-copy
    # path. A changed source/copy must not inherit the earlier hours verdict.
    if _sha256(root_copy) != h5_sha:
        if root_copy.resolve() != calibrated_h5.resolve():
            # A refused copy must not sit where a good package puts its artifact.
            root_copy.unlink(missing_ok=True)
        raise SystemExit(
            "The packaged H5 no longer matches the hours_worked_signal artifact "
            "binding. Re-run --stage qa and --stage finalize against stable bytes."
        )

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
    parser.add_argument(
        "--soi-mode",
        choices=SOI_MODES,
        default=DEFAULT_SOI_MODE,
        help=(
            "SOI target surface for --stage materialize. 'state' (default) "
            "is Build O's contract: every state-geography SOI spec outside "
            "the congressional-district file. 'totals' drops every "
            "soi_fiscal_distribution spec (no state AGI, income-tax or EITC "
            "total); 'full' keeps every state-bearing spec, both vintages "
            "of each state concept included. 'state_cd' adds the district "
            "file's district rows to 'state' with one vintage per state "
            "concept, and holds a hash-assigned block of them out "
            "(docs/us-acs-local-soi-target-surface.md). Later stages use the "
            "mode the checkpoint recorded."
        ),
    )
    parser.add_argument(
        "--cd-holdout-fraction",
        type=float,
        default=None,
        help=(
            "Share of (state x SOI concept family) district-target units held "
            "out of calibration and scored against a pro-rata baseline "
            f"(state_cd only; default {cd_surface.DEFAULT_STATE_CD_HOLDOUT_FRACTION}"
            f" there, max {cd_surface.MAX_CD_HOLDOUT_FRACTION}; 0 disables)."
        ),
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=1.0,
        help=(
            "Development rung: sample whole staging households at this "
            f"fraction ({sorted(cd_surface.SAMPLE_RUNG_TOKENS)}), stratified "
            "by spine x district and normalized to each spine's full mass. "
            "Recorded in the run identity; package refuses anything below 1."
        ),
    )
    parser.add_argument(
        "--sample-seed", type=int, default=cd_surface.DEFAULT_SAMPLE_SEED
    )
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
    parser.add_argument(
        "--district-ess-relative-floor",
        type=float,
        default=DISTRICT_ESS_RELATIVE_FLOOR,
        help=(
            "Finalize's district ESS gate: a congressional district collapses "
            "when its calibrated Kish ESS is below this share of its "
            "design-weight Kish ESS (0 to 1; 0 turns the check off)."
        ),
    )
    parser.add_argument(
        "--district-ess-floor",
        type=float,
        default=DISTRICT_ESS_FLOOR,
        help=(
            "Finalize's district ESS gate: the floor on every congressional "
            "district's calibrated Kish ESS (0 turns the check off)."
        ),
    )
    parser.add_argument(
        "--district-ess-gate-blocking",
        action="store_true",
        help=(
            "Make the district ESS gate a hard failure at finalize, and make "
            "package refuse a gate report finalize did not block on. Off by "
            "default: the gate is recorded in gate_summary.json, the build "
            "manifest and the reviewed limitations, and passes."
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
    try:
        cd_surface.rung_token(args.sample_fraction)
    except ValueError as error:
        parser.error(str(error))
    if args.cd_holdout_fraction is not None and not (
        0.0 <= args.cd_holdout_fraction <= cd_surface.MAX_CD_HOLDOUT_FRACTION
    ):
        parser.error(
            "--cd-holdout-fraction must be in "
            f"[0, {cd_surface.MAX_CD_HOLDOUT_FRACTION}]."
        )
    if not 0.0 <= args.district_ess_relative_floor <= 1.0:
        parser.error("--district-ess-relative-floor must be in [0, 1].")
    if not (np.isfinite(args.district_ess_floor) and args.district_ess_floor >= 0.0):
        parser.error("--district-ess-floor must be finite and >= 0.")
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
