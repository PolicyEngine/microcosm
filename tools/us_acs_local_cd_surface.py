"""Congressional-district calibration pieces of the ACS local-area release.

``tools/build_us_acs_local_release.py`` imports this module for everything
that makes a district-level SOI surface affordable and checkable:

- **The ``state_cd`` SOI surface** (:func:`state_cd_soi_surface`): the TY2022
  congressional-district SOI file's district rows plus the state surface,
  keeping one vintage per state concept. Historic Table 2 supplies a state
  concept's level wherever it has that concept; the district file then
  supplies only each district's share of its state, and its district rows
  are rebased to sum to the Historic Table 2 parent. Concepts only the
  district file has keep the district file's own state row as their parent.
- **The sparse target matrix** (:class:`SparseTargetAssembler`): one CSR row
  per target over the household weight vector, assembled chunk by chunk.
  District SOI rows never exist as dense columns: each distinct SOI concept
  is materialized once without geography (a *carrier* column) and every
  district row is that carrier restricted to the district's households,
  which is exactly what the materializer's district mask computes, because a
  tax unit's geography is its household's.
- **The CD holdout** (:func:`assign_target_roles`): hash-assigned blocks of
  district targets that never reach the calibrator, scored against a naive
  pro-rata baseline (:func:`score_cd_holdout`).
- **Effective sample size over distinct households** and weight share by
  spine (:func:`weight_origin_summary`).
- **Sampling rungs** for development runs (:func:`sample_staging_frame`).

Everything here is engine-free; the tool supplies the engine pass.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

# ---------------------------------------------------------------------------
# Surface constants
# ---------------------------------------------------------------------------

#: Ledger record-set specs from the congressional-district SOI file.
CD_FILE_RECORD_SET_SPEC_PREFIX = "irs_soi.congressional_district_"


def _district_soi_nesting():
    """The declared choices of the ``state_cd`` surface (microcosm#1760).

    ``us/cross_grain_declarations.json`` holds them with their reasons: the
    bridge between Historic Table 2 and the district file with its bound, the
    registers of defective, unparented and excluded district measures, and
    the level bridges that stay until the feed re-pin. The names below are
    those choices under the names this module always had.
    """

    from microcosm.build.us_runtime.cross_grain_declarations import (
        us_district_soi_nesting,
    )

    return us_district_soi_nesting()


def _cross_grain():
    """The shared cross-grain operator module (its metadata key names)."""

    from microcosm.build import cross_grain

    return cross_grain


_DISTRICT_SOI_NESTING = _district_soi_nesting()
#: The metadata that a reconciled district row keeps, as every row that the
#: shared cross-grain operator reconciles does (the UK rows too): its parent
#: row, what reconciled it, the factor, and its value before.
PARENT_KEY = _cross_grain().PARENT_KEY
NESTING_KEYS = (
    PARENT_KEY,
    _cross_grain().CONTROL_KEY,
    _cross_grain().FACTOR_KEY,
    _cross_grain().VALUE_BEFORE_KEY,
)
#: How the ``state_cd`` surface resolves the two vintages of a state concept.
STATE_CD_VINTAGE_RULE = _DISTRICT_SOI_NESTING.vintage_rule
#: District-file measures that read the wrong IRS column (microcosm#1038).
STATE_CD_DEFECTIVE_CD_FILE_MEASURES: Mapping[str, str] = dict(
    _DISTRICT_SOI_NESTING.defective_lower_measures
)
#: District-file measures with no state parent in either vintage.
STATE_CD_UNPARENTED_CD_MEASURES: Mapping[str, str] = dict(
    _DISTRICT_SOI_NESTING.unparented_lower_measures
)
#: District-file-only concepts and the sibling concept that bridges their
#: state level onto the Historic Table 2 basis (until the feed re-pin).
STATE_CD_LEVEL_BRIDGES: Mapping[str, str] = dict(_DISTRICT_SOI_NESTING.level_bridges)
#: States whose district SOI rows stay off the surface, with the evidence.
STATE_CD_EXCLUDED_CD_STATES: Mapping[str, str] = dict(
    _DISTRICT_SOI_NESTING.excluded_legs
)
#: How far a state's Historic Table 2 / district-file factor may sit from its
#: measure's median across states: the declared bound of the bridge, as a
#: ratio to that median.
STATE_CD_FACTOR_BAND = _DISTRICT_SOI_NESTING.factor_band
#: The packaged crosswalk the registers above were reviewed against. A test
#: pins it, so regenerating the crosswalk forces a second look at them.
STATE_CD_REVIEWED_CROSSWALK_SHA256 = _DISTRICT_SOI_NESTING.reviewed_crosswalk_sha256

#: Relative tolerance for "district rows sum to their state parent".
RECONCILIATION_RTOL = _DISTRICT_SOI_NESTING.closure_rtol

SIGMA_BASIS_FEED = "feed_standard_error"
SIGMA_BASIS_NONE = "not_provided_by_feed"

# ---------------------------------------------------------------------------
# Holdout constants
# ---------------------------------------------------------------------------

#: Salt of the CD holdout hash. Changing it re-draws every held unit.
CD_HOLDOUT_SALT = "microcosm.us_acs_local.cd_holdout.v1"
#: The held unit: every district target of one SOI concept family in one
#: state. A single district (or a single district target) is not a fair
#: holdout: with its state total and sibling districts trained it is pinned
#: by adding up. Concept families also group measures that add up to each
#: other (EITC by number of children sums to the EITC total).
CD_HOLDOUT_UNIT = "state_x_soi_concept_family"
DEFAULT_STATE_CD_HOLDOUT_FRACTION = 0.1
MAX_CD_HOLDOUT_FRACTION = 0.5
ROLE_TRAIN = "train"
ROLE_HOLDOUT = "holdout"

# ---------------------------------------------------------------------------
# Sampling constants: the rung tokens of tools/build_us_multispine_pool.py
# (DESIGN.md "Production US stacked spine" names f001, f010 and f100).
# ---------------------------------------------------------------------------

SAMPLE_RUNG_TOKENS: Mapping[float, str] = {
    0.01: "f001",
    0.04: "f004",
    0.10: "f010",
    0.25: "f025",
    1.0: "f100",
}
DEFAULT_SAMPLE_SEED = 578

GEOGRAPHY_STATE = "state"
GEOGRAPHY_CD = "congressional_district"


def _release_tool():
    """The sibling release tool (``tools/`` is on ``sys.path`` via the ACS tool)."""

    import build_us_fiscal_refresh_release as release_tool

    return release_tool


# ---------------------------------------------------------------------------
# SOI concept identity
# ---------------------------------------------------------------------------


def _metadata(spec) -> Mapping[str, str]:
    return spec.metadata


def geography_level(spec) -> str | None:
    return _metadata(spec).get("ledger_geography_level")


def record_set_spec_id(spec) -> str:
    return str(_metadata(spec).get("ledger_layout_record_set_spec_id") or "")


def is_cd_file_spec(spec) -> bool:
    return record_set_spec_id(spec).startswith(CD_FILE_RECORD_SET_SPEC_PREFIX)


def soi_materializer_semantics(spec) -> tuple:
    """Everything the SOI slice reads from a spec, except its geography.

    ``_base_simulation_household_columns`` builds an ``irs_soi`` column from
    exactly these fields plus ``state_fips`` and
    ``congressional_district_geoid``. Two specs with equal semantics
    therefore materialize the same household column up to their geography
    masks.
    """

    tool = _release_tool()
    metadata = _metadata(spec)
    return (
        spec.family,
        spec.entity,
        spec.filter,
        metadata.get("source_variable", metadata.get("variable")),
        repr(tool._as_bound(metadata["agi_lower_bound"])),
        repr(tool._as_bound(metadata["agi_upper_bound"])),
        metadata.get("taxable_only") == "true",
        metadata.get("filing_status"),
        tool._soi_eitc_child_count_filter(metadata),
        tool._soi_requires_positive_eitc_filter(metadata),
        metadata.get("itemized_only") == "true",
        metadata.get("measure_mode") == "indicator_sum",
        tool._unsupported_soi_ledger_filters(metadata),
    )


def soi_concept_identity(spec) -> tuple[str, ...]:
    """The published concept a spec measures, independent of vintage.

    Historic Table 2 and the district file name the same concept with the
    same ``source_measure_id``; the AGI band, filing status and EITC child
    count separate the Historic Table 2 band rows from the all-income rows.
    """

    tool = _release_tool()
    metadata = _metadata(spec)
    return (
        str(metadata.get("source_measure_id")),
        repr(tool._as_bound(metadata["agi_lower_bound"])),
        repr(tool._as_bound(metadata["agi_upper_bound"])),
        str(metadata.get("filing_status", "")).lower(),
        str(tool._soi_eitc_child_count_filter(metadata)),
    )


def soi_concept_family(source_measure_id: str) -> str:
    """The holdout family of an SOI measure: its concept without count/amount.

    Every EITC measure is one family, because the per-child-count rows add up
    to the EITC total and holding one out would leave it pinned by the rest.
    """

    measure = str(source_measure_id)
    if measure.startswith("eitc"):
        return "eitc"
    for suffix in ("_amount", "_returns", "_claims", "_count"):
        if measure.endswith(suffix):
            return measure[: -len(suffix)]
    return measure


# ---------------------------------------------------------------------------
# The state_cd surface
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StateCdSurface:
    """The ``state_cd`` SOI specs and the receipt that explains them."""

    specs: tuple
    receipt: dict


def _target_nesting():
    """The US adapter to the shared cross-grain operator (microcosm#1760)."""

    from microcosm.build.us_runtime import target_nesting

    return target_nesting


def state_cd_soi_surface(
    soi_specs: Sequence,
    *,
    state_surface_predicate: Callable[[object], bool],
    crosswalk: pd.DataFrame,
    crosswalk_sha256: str | None = None,
    level_bridges: Mapping[str, str] = STATE_CD_LEVEL_BRIDGES,
    factor_band: float = STATE_CD_FACTOR_BAND,
) -> StateCdSurface:
    """Select and reconcile the ``state_cd`` SOI surface.

    The work is in
    :func:`microcosm.build.us_runtime.target_nesting.district_soi_surface`:
    the shared cross-grain operator divides each state control among its
    district rows, with the choices declared in
    ``us/cross_grain_declarations.json`` (microcosm#1760).

    Args:
        soi_specs: Every compiled ``irs_soi`` spec (the ``full`` surface).
        state_surface_predicate: The ``state`` mode's predicate; its specs
            (Historic Table 2) are the state surface.
        crosswalk: The 117th->119th CD crosswalk the compiler used; it names
            each state's source-plan and current-plan districts.
        crosswalk_sha256: Recorded in the receipt.
        level_bridges: District-file-only measure -> sibling measure whose
            Historic Table 2 / district-file ratio in the same state lifts
            the district file's state level onto the Historic Table 2 basis.
        factor_band: Largest allowed ratio between a state's Historic Table 2
            / district-file ratio and its measure's median across states
            (either direction); see ``STATE_CD_FACTOR_BAND``.

    Returns:
        The surface in a fixed order (Historic Table 2 state rows, district
        file state rows, district rows) and its receipt.

    Raises:
        ValueError: If a district concept is incomplete in a state, if its
            district rows disagree on what they materialize, if a parent's
            materialization differs from its children's, or if a parent
            cannot absorb its children (zero sum under a nonzero parent, or
            a sign conflict).
    """

    surface = _target_nesting().district_soi_surface(
        soi_specs,
        state_surface_predicate=state_surface_predicate,
        crosswalk=crosswalk,
        crosswalk_sha256=crosswalk_sha256,
        level_bridges=level_bridges,
        factor_band=factor_band,
        concept_identity=soi_concept_identity,
        materializer_semantics=soi_materializer_semantics,
        is_lower_file_spec=is_cd_file_spec,
        # Read at call time: the builder re-checks its own output with the
        # report this module holds.
        closure_report=lambda specs: state_parent_reconciliation(specs),
    )
    return StateCdSurface(specs=surface.specs, receipt=surface.receipt)


def state_parent_reconciliation(
    specs: Sequence, *, rtol: float = RECONCILIATION_RTOL
) -> list[dict]:
    """Every (state parent, district rows) block and whether it adds up.

    A block is the district specs naming one parent (``cross_grain_parent``).
    Returns one entry per block with the parent value, the district sum and
    ``ok`` (relative difference within ``rtol``).
    """

    return _target_nesting().district_block_closure(specs, rtol=rtol)


# ---------------------------------------------------------------------------
# Sparse target matrix assembly
# ---------------------------------------------------------------------------


def is_cd_soi_spec(spec) -> bool:
    return spec.family == "irs_soi" and bool(
        spec.metadata.get("congressional_district_geoid")
    )


def _without_geography(metadata: Mapping[str, str]) -> dict[str, str]:
    return {
        key: value
        for key, value in metadata.items()
        if key not in ("state_fips", "congressional_district_geoid")
    }


@dataclass(frozen=True)
class CarrierPlan:
    """How declared specs map onto what the engine pass materializes.

    ``engine_specs`` are handed to the materializer: every declared spec
    that is not a district SOI row, then one carrier per distinct district
    SOI concept. ``split`` lists, per district SOI row, its declared row
    index, state (``NO_STATE_MASK`` when the spec carries no ``state_fips``,
    which the materializer then does not mask on), district and carrier
    measure.
    """

    declared: tuple
    engine_specs: tuple
    direct_rows: tuple[tuple[int, str], ...]
    split: tuple[tuple[int, int, int, str], ...]
    carrier_of: Mapping[str, str]

    @property
    def n_rows(self) -> int:
        return len(self.declared)


CARRIER_PREFIX = "__state_cd_carrier__"
#: ``split`` state value of a district row whose spec has no ``state_fips``.
NO_STATE_MASK = -1


def plan_carriers(specs: Sequence) -> CarrierPlan:
    """Factor district SOI rows into per-concept carriers.

    Each carrier is the first district row of its concept with the state and
    district keys removed, so it materializes the concept for every
    household; a district row is the carrier masked to the district's
    households (``SparseTargetAssembler`` applies the mask).
    """

    declared = tuple(specs)
    names = [spec.name for spec in declared]
    if len(set(names)) != len(names):
        raise ValueError("Declared target specs must have unique names.")
    direct_rows: list[tuple[int, str]] = []
    split: list[tuple[int, int, int, str]] = []
    carriers: dict[tuple, object] = {}
    carrier_of: dict[str, str] = {}
    for index, spec in enumerate(declared):
        if not is_cd_soi_spec(spec):
            direct_rows.append((index, spec.measure))
            continue
        semantics = soi_materializer_semantics(spec)
        carrier = carriers.get(semantics)
        if carrier is None:
            carrier_name = f"{CARRIER_PREFIX}{len(carriers):05d}"
            # A carrier is an engine-pass column, never a calibration target:
            # it drops the hierarchy, whose target id names the district row.
            carrier = replace(
                spec,
                name=carrier_name,
                measure=carrier_name,
                metadata=_without_geography(spec.metadata),
                hierarchy=None,
            )
            carriers[semantics] = carrier
        district = str(spec.metadata["congressional_district_geoid"])
        state = spec.metadata.get("state_fips")
        split.append(
            (
                index,
                NO_STATE_MASK if state is None else int(state),
                int(district),
                carrier.measure,
            )
        )
        carrier_of[spec.name] = carrier.measure
    engine_specs = tuple(spec for spec in declared if not is_cd_soi_spec(spec)) + tuple(
        carriers.values()
    )
    return CarrierPlan(
        declared=declared,
        engine_specs=engine_specs,
        direct_rows=tuple(direct_rows),
        split=tuple(split),
        carrier_of=carrier_of,
    )


def carrier_check_specs(plan: CarrierPlan, district_codes: np.ndarray) -> tuple:
    """District rows the first chunk also materializes directly, as a check.

    One per carrier: the row whose district has the most households in the
    chunk (ties to the lower row). ``materialize_chunked`` compares each
    directly materialized row with the row the assembler stored for it and
    refuses any difference.
    """

    counts = Counter(int(code) for code in district_codes)
    best: dict[str, tuple[int, int]] = {}
    for index, _state, district, carrier in plan.split:
        score = counts.get(district, 0)
        current = best.get(carrier)
        if current is None or score > current[0]:
            best[carrier] = (score, index)
    return tuple(plan.declared[index] for _score, index in best.values())


class SparseTargetAssembler:
    """Accumulate a (targets x households) CSR matrix one chunk at a time.

    Values are stored as float32, exactly as the dense checkpoint stored its
    columns, and a zero after float32 rounding is not stored, so the matrix
    is the dense float32 matrix with its zeros removed.
    """

    def __init__(self, n_rows: int, n_households: int) -> None:
        self.n_rows = int(n_rows)
        self.n_households = int(n_households)
        self._rows: list[np.ndarray] = []
        self._cols: list[np.ndarray] = []
        self._data: list[np.ndarray] = []

    def add_column(self, row: int, low: int, values: np.ndarray, *, name: str) -> None:
        """Add one target's values for households ``low .. low+len(values)``."""

        values32 = np.asarray(values, dtype=np.float32)
        if not np.isfinite(values32).all():
            bad = int((~np.isfinite(values32)).sum())
            raise ValueError(
                f"Target {name!r} materialized {bad} non-finite household "
                "value(s); refusing to assemble the target matrix."
            )
        nonzero = np.flatnonzero(values32)
        if not len(nonzero):
            return
        self._rows.append(np.full(len(nonzero), row, dtype=np.int32))
        self._cols.append((nonzero + low).astype(np.int32))
        self._data.append(values32[nonzero])

    def add_masked(
        self,
        row: int,
        low: int,
        carrier: np.ndarray,
        positions: np.ndarray,
        *,
        name: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Add a carrier restricted to chunk ``positions`` (the district's).

        Returns the chunk positions and float32 values actually stored.
        """

        values32 = np.asarray(carrier[positions], dtype=np.float32)
        if not np.isfinite(values32).all():
            raise ValueError(
                f"Target {name!r} derives non-finite values from its carrier."
            )
        keep = values32 != 0
        stored = (positions[keep], values32[keep])
        if not keep.any():
            return stored
        self._rows.append(np.full(int(keep.sum()), row, dtype=np.int32))
        self._cols.append((positions[keep] + low).astype(np.int32))
        self._data.append(values32[keep])
        return stored

    def extend(self, other_rows: sparse.csr_array, row_offset: int) -> None:
        """Append already-assembled CSR rows at ``row_offset``."""

        coo = sparse.coo_array(other_rows)
        self._rows.append((coo.row + row_offset).astype(np.int32))
        self._cols.append(coo.col.astype(np.int32))
        self._data.append(coo.data.astype(np.float32))

    def to_csr(self) -> sparse.csr_array:
        if self._data:
            rows = np.concatenate(self._rows)
            cols = np.concatenate(self._cols)
            data = np.concatenate(self._data)
        else:
            rows = np.empty(0, dtype=np.int32)
            cols = np.empty(0, dtype=np.int32)
            data = np.empty(0, dtype=np.float32)
        self._rows, self._cols, self._data = [], [], []
        coo = sparse.coo_array(
            (data, (rows, cols)), shape=(self.n_rows, self.n_households)
        )
        stored = coo.nnz
        coo.sum_duplicates()
        if coo.nnz != stored:
            raise ValueError(
                f"{stored - coo.nnz} (target, household) cell(s) were written "
                "twice; a chunk or a district split overlapped."
            )
        matrix = sparse.csr_array(coo)
        matrix.sort_indices()
        if matrix.data.dtype != np.float32:
            raise TypeError("target matrix data must stay float32.")
        return matrix


def district_positions(district_codes: np.ndarray) -> dict[int, np.ndarray]:
    """Chunk positions of each district's households, in household order."""

    codes = np.asarray(district_codes, dtype=np.int64)
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    boundaries = np.flatnonzero(np.diff(sorted_codes)) + 1
    starts = np.concatenate(([0], boundaries))
    stops = np.concatenate((boundaries, [len(codes)]))
    return {
        int(sorted_codes[start]): order[start:stop]
        for start, stop in zip(starts, stops, strict=True)
        if stop > start
    }


def split_carriers_into(
    assembler: SparseTargetAssembler,
    plan: CarrierPlan,
    carrier_columns: Mapping[str, np.ndarray],
    *,
    low: int,
    state_codes: np.ndarray,
    district_codes: np.ndarray,
    capture: set[int] | frozenset[int] = frozenset(),
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Add every district SOI row of one chunk from its carrier column.

    Returns, for each declared row index in ``capture``, the chunk positions
    and float32 values the assembler stored for it.
    """

    positions_by_district = district_positions(district_codes)
    states = np.asarray(state_codes, dtype=np.int64)
    empty = np.empty(0, dtype=np.int64)
    captured: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for index, state, district, carrier in plan.split:
        positions = positions_by_district.get(district, empty)
        if len(positions) and state != NO_STATE_MASK:
            positions = positions[states[positions] == state]
        stored = assembler.add_masked(
            index,
            low,
            carrier_columns[carrier],
            positions,
            name=plan.declared[index].name,
        )
        if index in capture:
            captured[index] = stored
    return captured


def district_row_parents(
    plan: CarrierPlan, compiled_names: Iterable[str]
) -> tuple[dict[int, tuple[str, bool]], list[int]]:
    """Each district row's directly materialized state parent, for the check.

    Returns ``{row index: (parent name, strict)}`` and the rows with no parent,
    over the rows whose carrier compiled.

    - A ``state_cd`` row names its parent (``cross_grain_parent``),
      and its block partitions that parent's state, so the check is strict:
      the stored district rows, laid side by side, must equal the parent's
      column on every household. A ``state_cd`` row without a compiled parent
      is refused, since the check would silently skip it.
    - Any other district row falls back to a compiled state row of the same
      materializer semantics in the same state (in ``full``, the district
      file's own state total). Its block need not cover the state, so the
      check compares the row's stored values with that column on the row's
      own households only.
    """

    compiled = set(compiled_names)
    same_concept_state_row: dict[tuple, str] = {}
    for index, _measure in plan.direct_rows:
        spec = plan.declared[index]
        metadata = spec.metadata
        if (
            spec.family == "irs_soi"
            and spec.name in compiled
            and metadata.get("state_fips")
            and not metadata.get("congressional_district_geoid")
        ):
            same_concept_state_row.setdefault(
                (soi_materializer_semantics(spec), str(metadata["state_fips"])),
                spec.name,
            )
    parents: dict[int, tuple[str, bool]] = {}
    unchecked: list[int] = []
    for index, _state, _district, carrier in plan.split:
        if carrier not in compiled:
            continue  # the row is not materialized, so not on the surface
        spec = plan.declared[index]
        metadata = spec.metadata
        named = metadata.get(PARENT_KEY)
        if PARENT_KEY in metadata and named not in compiled:
            raise ValueError(
                f"state_cd district row {spec.name} has no compiled state "
                f"parent ({named!r}); its per-chunk check would be vacuous."
            )
        if named in compiled:
            parents[index] = (named, True)
            continue
        fallback = same_concept_state_row.get(
            (soi_materializer_semantics(spec), str(metadata.get("state_fips")))
        )
        if fallback is None:
            unchecked.append(index)
        else:
            parents[index] = (fallback, False)
    return parents, unchecked


#: Digits of a ladder population target's geography code, by ladder geography.
POPULATION_TARGET_CODE_DIGITS: Mapping[str, int] = {"state": 2, "cd": 4, "county": 5}


def population_target_name(geography: str, code: int) -> str:
    """A ladder population target's name: ``pop_state_06``, ``pop_cd_0601``."""

    digits = POPULATION_TARGET_CODE_DIGITS[geography]
    return f"pop_{geography}_{int(code):0{digits}d}"


def population_rows(
    households: pd.DataFrame,
    household_size: np.ndarray,
    ladder_populations: Mapping[str, Mapping[int, float]],
    geographies: Sequence[str],
    *,
    county_column: str = "county_fips",
) -> tuple[list[dict], sparse.csr_array, list[str]]:
    """Ladder population marginals as sparse rows (household size x 1[geo]).

    Returns (target records, CSR rows, dropped cell names). A ladder cell
    with no supporting household is dropped and named, as before.
    """

    if "county" in geographies:
        import us_acs_local_county_surface as county_surface

        return county_surface.population_rows(
            households, household_size, ladder_populations, geographies,
            county_column=county_column,
        )
    column_map = {
        "state": ("state_fips", "state", GEOGRAPHY_STATE),
        "cd": ("congressional_district_geoid", "cd", GEOGRAPHY_CD),
    }
    size32 = np.asarray(household_size, dtype=np.float32)
    records: list[dict] = []
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []
    dropped: list[str] = []
    for geography in geographies:
        column, key, level = column_map[geography]
        codes = pd.to_numeric(households[column]).to_numpy()
        for value, population in sorted(ladder_populations[key].items()):
            name = population_target_name(geography, value)
            present = np.flatnonzero(codes == value)
            if not len(present):
                dropped.append(name)
                continue
            present = present[size32[present] != 0]
            row = len(records)
            rows.append(np.full(len(present), row, dtype=np.int32))
            cols.append(present.astype(np.int32))
            data.append(size32[present])
            record = {
                "name": name,
                "value": float(population),
                "source": "us_puma_ladder_2020",
                "family": "census_population_ladder",
                "geography_level": level,
                "state_fips": f"{int(value) // (100 if geography == 'cd' else 1):02d}",
                "congressional_district_geoid": (
                    f"{int(value):04d}" if geography == "cd" else None
                ),
            }
            records.append(record)
    matrix = sparse.csr_array(
        (
            np.concatenate(data) if data else np.empty(0, np.float32),
            (
                np.concatenate(rows) if rows else np.empty(0, np.int32),
                np.concatenate(cols) if cols else np.empty(0, np.int32),
            ),
        ),
        shape=(len(records), len(households)),
    )
    matrix.sort_indices()
    return records, matrix, dropped


def save_target_matrix(path: Path, matrix: sparse.csr_array) -> str:
    """Write the CSR (float32 data) uncompressed; return its sha256."""

    matrix = sparse.csr_array(matrix)
    if matrix.data.dtype != np.float32:
        raise TypeError("target matrix data must be float32.")
    sparse.save_npz(path, matrix, compressed=False)
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_target_matrix(path: Path) -> sparse.csr_array:
    matrix = sparse.csr_array(sparse.load_npz(path))
    matrix.sort_indices()
    return matrix


def csr_row_measure(matrix: sparse.csr_array, row: int, n: int):
    """A callable measure returning one CSR row as a dense float64 vector.

    The calibrate kernel compiles targets through ``build_constraint_matrix``,
    which reads each target's row and keeps its nonzeros; a callable measure
    lets it read the row straight from the checkpoint CSR (the UK rowwise
    precedent, ``uk_runtime.local_rowwise._rowwise_target_set``). Only one
    dense row exists at a time. float32 values widen exactly to float64, as
    the dense checkpoint's float32 columns did.
    """

    start, stop = int(matrix.indptr[row]), int(matrix.indptr[row + 1])
    indices = matrix.indices[start:stop]
    data = matrix.data[start:stop]

    def measure(frame) -> np.ndarray:
        values = np.zeros(n, dtype=np.float64)
        values[indices] = data
        return values

    # Calibration diagnostics name a callable measure by its qualified name;
    # name the matrix row rather than this closure.
    measure.__qualname__ = f"target_matrix_row[{row}]"
    return measure


# ---------------------------------------------------------------------------
# Target records, roles and the calibration target set
# ---------------------------------------------------------------------------


#: The metadata keys of district and county rows until microcosm#1760, when
#: the keys of the shared cross-grain operator (``cross_grain_*``) replaced
#: them, and the fields of the district receipt that went with them. A saved
#: target or receipt that still holds one was written by older code. A later
#: stage that read it with this code would find no parent, no value before
#: and no factor, without an error.
RETIRED_NESTING_KEY_PREFIXES = ("state_cd_", "state_county_")
RETIRED_SURFACE_RECEIPT_FIELDS = (
    "rebase_factor_by_measure",
    "factor_band",
    "internal_consistency",
)
_OLDER_CODE = (
    "it was materialized by older code, before the keys of the shared "
    "cross-grain operator (microcosm#1760). Re-run --stage materialize with "
    "the current tool."
)


def refuse_targets_of_older_code(targets, *, where: str) -> None:
    """Refuse saved targets that carry the retired nesting keys.

    ``targets`` are role records (mappings) or specs (their metadata is
    read). ``where`` names the file, for the message.
    """

    for target in targets:
        metadata = target if isinstance(target, Mapping) else target.metadata
        retired = sorted(
            key
            for key in metadata
            if str(key).startswith(RETIRED_NESTING_KEY_PREFIXES)
        )
        if retired:
            raise SystemExit(
                f"{where} holds the metadata key {retired[0]!r}: {_OLDER_CODE}"
            )


def refuse_surface_receipt_of_older_code(surface: Mapping, *, where: str) -> None:
    """Refuse a saved district surface receipt with the retired fields."""

    retired = sorted(set(RETIRED_SURFACE_RECEIPT_FIELDS) & set(surface))
    if retired:
        raise SystemExit(f"{where} holds the field {retired[0]!r}: {_OLDER_CODE}")


def target_record(spec) -> dict:
    """The target_roles.json record for one admin spec (metadata subset)."""

    metadata = spec.metadata
    level = metadata.get("ledger_geography_level")
    se = getattr(spec, "se", None)
    record = {
        "name": spec.name,
        "entity": spec.entity,
        "period": spec.period,
        "value": float(spec.value),
        "source": spec.source or "ledger_feed",
        "family": spec.family,
        "geography_level": level,
        "state_fips": metadata.get("state_fips"),
        "congressional_district_geoid": metadata.get("congressional_district_geoid"),
        "source_measure_id": metadata.get("source_measure_id"),
        "sigma": None if se is None else float(se),
        "sigma_basis": SIGMA_BASIS_NONE if se is None else SIGMA_BASIS_FEED,
    }
    for key in NESTING_KEYS:
        if key in metadata:
            record[key] = metadata[key]
    return record


def cd_holdout_unit_key(record: Mapping) -> str | None:
    """The holdout unit of a district SOI target, or None if not eligible."""

    if record.get("family") != "irs_soi":
        return None
    if record.get("geography_level") != GEOGRAPHY_CD:
        return None
    if not record.get(PARENT_KEY):
        return None
    return (
        f"{record['state_fips']}|"
        f"{soi_concept_family(str(record.get('source_measure_id')))}"
    )


def assign_target_roles(
    records: Sequence[dict],
    *,
    fraction: float,
    salt: str = CD_HOLDOUT_SALT,
) -> dict:
    """Set ``role`` and ``holdout_unit`` on every record; return the receipt.

    Only district SOI targets with a bound state parent are eligible; every
    other target trains. A unit is held out by
    :func:`microcosm.build.holdout.hash_holdout_unit` on its key.
    """

    from microcosm.build.holdout import hash_holdout_unit

    if not 0.0 <= float(fraction) <= MAX_CD_HOLDOUT_FRACTION:
        raise ValueError(
            f"CD holdout fraction must be in [0, {MAX_CD_HOLDOUT_FRACTION}], "
            f"got {fraction!r}."
        )
    units: dict[str, bool] = {}
    for record in records:
        unit = cd_holdout_unit_key(record)
        record["holdout_unit"] = unit
        if unit is None:
            record["role"] = ROLE_TRAIN
            continue
        if unit not in units:
            units[unit] = hash_holdout_unit(unit, fraction=fraction, salt=salt)
        record["role"] = ROLE_HOLDOUT if units[unit] else ROLE_TRAIN
    held_units = sorted(unit for unit, held in units.items() if held)
    return {
        "unit": CD_HOLDOUT_UNIT,
        "salt": salt,
        "fraction": float(fraction),
        "eligible_units": len(units),
        "held_units": len(held_units),
        "held_unit_keys": held_units,
        "eligible_targets": sum(
            1 for record in records if record["holdout_unit"] is not None
        ),
        "held_targets": sum(1 for record in records if record["role"] == ROLE_HOLDOUT),
        "hash": "sha256(salt + 0x1f + unit)[:8] / 2**64 < fraction",
    }


def train_rows(records: Sequence[Mapping]) -> list[int]:
    return [
        index for index, record in enumerate(records) if record["role"] == ROLE_TRAIN
    ]


def holdout_rows(records: Sequence[Mapping]) -> list[int]:
    return [
        index for index, record in enumerate(records) if record["role"] == ROLE_HOLDOUT
    ]


def calibration_target_set(
    records: Sequence[Mapping],
    matrix: sparse.csr_array,
    n_households: int,
    *,
    specs: Sequence | None = None,
):
    """The TargetSet the calibrator sees: training targets only.

    Held-out targets are never constructed, so they cannot reach the solve.
    With ``specs`` (the checkpoint's registry specs, row-aligned with
    ``records``) each target keeps its spec's value, period, source,
    metadata and calibration hierarchy, as ``TargetSpec.to_target`` would,
    and only its measure becomes the CSR row.
    """

    from microcosm.calibrate.target import Target, TargetSet

    if matrix.shape != (len(records), n_households):
        raise ValueError(
            f"target matrix shape {matrix.shape} does not match "
            f"{len(records)} targets x {n_households} households."
        )
    if specs is not None and [spec.name for spec in specs] != [
        record["name"] for record in records
    ]:
        raise ValueError("target specs are not row-aligned with the roles.")
    targets = []
    for index in train_rows(records):
        record = records[index]
        if record["role"] != ROLE_TRAIN:  # pragma: no cover - train_rows filters
            raise AssertionError("a held-out target reached the calibration set")
        measure = csr_row_measure(matrix, index, n_households)
        if specs is None:
            targets.append(
                Target(
                    name=record["name"],
                    entity=record.get("entity", "household"),
                    measure=measure,
                    value=float(record["value"]),
                    period=record["period"],
                    source=record["source"],
                )
            )
            continue
        spec = specs[index]
        if spec.filter is not None:
            raise ValueError(
                f"{spec.name}: a filtered spec cannot take a matrix-row measure; "
                "the row already carries the filter."
            )
        targets.append(
            Target(
                name=spec.name,
                entity=spec.entity,
                measure=measure,
                value=spec.value,
                period=spec.period,
                tolerance=spec.tolerance,
                source=spec.source,
                metadata=spec.metadata,
                hierarchy=spec.hierarchy,
            )
        )
    held = {records[index]["name"] for index in holdout_rows(records)}
    leaked = held & {target.name for target in targets}
    if leaked:
        raise AssertionError(f"held-out targets reached the calibrator: {leaked}")
    return TargetSet(targets)


# ---------------------------------------------------------------------------
# Holdout scoring against the pro-rata baseline
# ---------------------------------------------------------------------------


def _error_summary(estimates, targets, *, cap: float) -> dict:
    from microcosm.calibrate import relative_error_loss

    estimates = np.asarray(estimates, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    scale = np.maximum(np.abs(targets), 1.0)
    errors = np.abs(estimates - targets) / scale
    return {
        "mean_abs_rel_error": float(errors.mean()),
        "median_abs_rel_error": float(np.median(errors)),
        "p90_abs_rel_error": float(np.quantile(errors, 0.9)),
        "max_abs_rel_error": float(errors.max()),
        "fraction_within_10pct": float((errors <= 0.10).mean()),
        "capped_loss": float(
            relative_error_loss(estimates, targets, target_loss_cap=cap)
        ),
    }


def pro_rata_baseline(records: Sequence[Mapping], rows: Sequence[int]) -> np.ndarray:
    """State parent value x district population share, for each row."""

    by_name = {record["name"]: record for record in records}
    values = []
    for index in rows:
        record = records[index]
        parent = by_name.get(record.get(PARENT_KEY))
        if parent is None:
            raise ValueError(
                f"{record['name']} has no bound state parent for the baseline."
            )
        cd_population = record.get("cd_population")
        state_population = record.get("state_population")
        if not cd_population or not state_population:
            raise ValueError(
                f"{record['name']} carries no district/state population for "
                "the pro-rata baseline."
            )
        values.append(
            float(parent["value"]) * float(cd_population) / float(state_population)
        )
    return np.asarray(values, dtype=np.float64)


def score_cd_holdout(
    records: Sequence[Mapping],
    matrix: sparse.csr_array,
    *,
    design_weights: np.ndarray,
    final_weights: np.ndarray,
    cap: float,
    receipt: Mapping | None = None,
) -> dict:
    """Held-out district targets: calibrated vs design vs pro-rata baseline."""

    rows = [
        row for row in holdout_rows(records)
        if records[row].get("geography_level") == GEOGRAPHY_CD
    ]
    base = {"report_only": True, **dict(receipt or {})}
    if not rows:
        return {**base, "n_targets": 0}
    held = sparse.csr_array(matrix[rows]).astype(np.float64)
    targets = np.asarray([records[i]["value"] for i in rows], dtype=np.float64)
    final = held @ np.asarray(final_weights, dtype=np.float64)
    design = held @ np.asarray(design_weights, dtype=np.float64)
    baseline = pro_rata_baseline(records, rows)
    scale = np.maximum(np.abs(targets), 1.0)
    final_error = np.abs(final - targets) / scale
    baseline_error = np.abs(baseline - targets) / scale
    families: dict[str, list[int]] = defaultdict(list)
    for position, index in enumerate(rows):
        families[str(records[index]["holdout_unit"]).split("|", 1)[1]].append(position)
    by_family = {}
    for family, positions in sorted(families.items()):
        idx = np.asarray(positions)
        by_family[family] = {
            "n_targets": len(positions),
            "calibrated_mean_abs_rel_error": float(final_error[idx].mean()),
            "pro_rata_mean_abs_rel_error": float(baseline_error[idx].mean()),
            "calibrated_beats_pro_rata": float(
                (final_error[idx] < baseline_error[idx]).mean()
            ),
        }
    return {
        **base,
        "n_targets": len(rows),
        "calibrated": _error_summary(final, targets, cap=cap),
        "design_weights": _error_summary(design, targets, cap=cap),
        "pro_rata_baseline": _error_summary(baseline, targets, cap=cap),
        "calibrated_beats_pro_rata_share": float((final_error < baseline_error).mean()),
        "baseline": (
            "state parent target x (district population / state population), "
            "populations from the PUMA ladder's 119th-plan district overlap"
        ),
        "by_family": by_family,
        "targets": [
            {
                "name": records[index]["name"],
                "unit": records[index]["holdout_unit"],
                "target": float(targets[position]),
                "calibrated": float(final[position]),
                "design": float(design[position]),
                "pro_rata": float(baseline[position]),
            }
            for position, index in enumerate(rows)
        ],
    }


# ---------------------------------------------------------------------------
# Effective sample size over distinct households, weight share by spine
# ---------------------------------------------------------------------------


def kish_ess(weights: np.ndarray) -> float:
    """Kish ESS; refuses negative, NaN or infinite weights (microcosm#1585)."""
    from microcosm.calibrate.solve import effective_sample_size

    return effective_sample_size(weights)


def distinct_household_weights(
    weights: np.ndarray, spine: np.ndarray, source_id: np.ndarray
) -> np.ndarray:
    """Weights summed over rows that are copies of one source household.

    A household is ``(household_spine, household_source_id)``: ACS source ids
    are the pre-offset ACS ids and collide with donor ids, and a donor
    household can appear as its native row and its PUF-detail clone.
    """

    frame = pd.DataFrame(
        {
            "spine": np.asarray(spine).astype(str),
            "source": np.asarray(source_id),
            "weight": np.asarray(weights, dtype=np.float64),
        }
    )
    return frame.groupby(["spine", "source"], sort=True)["weight"].sum().to_numpy()


def top_weight_share(weights: np.ndarray, fraction: float = 0.01) -> float:
    """Share of total weight on the heaviest ``ceil(fraction * n)`` records."""

    weights = np.asarray(weights, dtype=np.float64)
    if not len(weights) or float(weights.sum()) <= 0.0:
        return 0.0
    k = max(1, math.ceil(fraction * len(weights)))
    return float(np.sort(weights)[-k:].sum() / weights.sum())


def _distribution(values: Mapping[str, float]) -> dict[str, float | int]:
    array = np.asarray(list(values.values()), dtype=np.float64)
    if not len(array):
        return {"n": 0}
    return {
        "n": int(len(array)),
        "min": float(array.min()),
        "p10": float(np.quantile(array, 0.1)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "max": float(array.max()),
    }


def weight_origin_summary(
    weights: np.ndarray,
    *,
    spine: np.ndarray | None,
    source_id: np.ndarray | None,
    state: np.ndarray | None = None,
    district: np.ndarray | None = None,
) -> dict:
    """Concentration and origin of a household weight vector.

    Kish ESS over rows (national, per spine, per state, per district), over
    distinct households, the top-1% weight share, and household-weight share
    by spine.
    """

    from microcosm.build.area_support import effective_sample_size_by_group

    weights = np.asarray(weights, dtype=np.float64)
    summary: dict[str, object] = {
        "rows": int(len(weights)),
        "effective_sample_size_rows": kish_ess(weights),
        "top_1pct_weight_share": top_weight_share(weights, 0.01),
    }
    for label, codes in (("state", state), ("district", district)):
        if codes is None:
            continue
        by_group = effective_sample_size_by_group(weights, codes)
        summary[f"effective_sample_size_by_{label}"] = by_group
        summary[f"effective_sample_size_by_{label}_distribution"] = _distribution(
            by_group
        )
    if spine is None:
        summary["note"] = "no household_spine column; distinct ESS unavailable"
        return summary
    spine = np.asarray(spine).astype(str)
    total = float(weights.sum())
    summary["weight_share_by_spine"] = {
        str(value): float(weights[spine == value].sum() / total) if total else 0.0
        for value in sorted(set(spine))
    }
    summary["effective_sample_size_rows_by_spine"] = {
        str(value): kish_ess(weights[spine == value]) for value in sorted(set(spine))
    }
    if source_id is not None:
        distinct = distinct_household_weights(weights, spine, source_id)
        summary["distinct_households"] = int(len(distinct))
        summary["effective_sample_size_distinct_households"] = kish_ess(distinct)
        summary["distinct_key"] = ["household_spine", "household_source_id"]
    return summary


# ---------------------------------------------------------------------------
# Sampling rungs for development runs
# ---------------------------------------------------------------------------


def rung_token(fraction: float) -> str:
    for value, token in SAMPLE_RUNG_TOKENS.items():
        if math.isclose(float(fraction), value, rel_tol=0.0, abs_tol=1e-12):
            return token
    raise ValueError(
        f"sample fraction {fraction!r} is not a rung; use one of "
        f"{sorted(SAMPLE_RUNG_TOKENS)} (the stacked pool's rungs)."
    )


def _staging_strata(households: pd.DataFrame, tag: str) -> pd.Series:
    return (
        households[tag]
        .astype(str)
        .str.cat(households["congressional_district_geoid"].astype(str), sep="|cd=")
    )


def sample_staging_frame(frame, *, fraction: float, seed: int):
    """Sample whole households at a development rung.

    ``fraction == 1`` returns the frame unchanged. Otherwise
    :func:`microcosm.build.frame_sampling.sample_frame_households` draws
    ``floor(fraction * n_h)`` households in every (spine, district) stratum
    ``h``, each drawn household's weight is scaled by ``n_h / k_h`` (the
    stratum's inverse sampling rate, so a drawn stratum keeps its full
    household mass), and each spine is then scaled to its full household
    mass. A stratum that floors to zero draws loses its households; the
    receipt records how many strata and how much weight that is, and the
    per-spine scaling spreads that weight over the spine's drawn strata. At
    f001 that is material for the donor spine (its district strata are
    small), so read district-level evidence from f010 or above.
    """

    from microcosm.build.frame_sampling import (
        sample_frame_households,
        validate_sample_seed,
    )
    from microcosm.build.us_runtime.base_pool import spine_column
    from microcosm.frame import MassChange

    token = rung_token(fraction)
    validate_sample_seed(seed)
    households = frame.table("household")
    if token == "f100":
        return frame, {
            "sample_fraction": 1.0,
            "rung": token,
            "sample_seed": int(seed),
            "sampled": False,
            "households": int(len(households)),
        }
    tag = spine_column("household")
    strata = _staging_strata(households, tag)
    spine = households[tag].astype(str).to_numpy()
    full_weights = np.asarray(frame.weights_for("household").values, dtype=np.float64)
    full_mass = {
        value: float(full_weights[spine == value].sum()) for value in sorted(set(spine))
    }
    stratum_size = strata.value_counts()
    stratum_mass = pd.Series(full_weights).groupby(strata.to_numpy()).sum()
    sampled, receipt = sample_frame_households(
        frame,
        fraction=float(fraction),
        seed=int(seed),
        source_name="ACS local staging",
        unit_strata=strata.to_numpy(),
        floor_context="the ACS local development rung",
    )
    sampled_households = sampled.table("household")
    sampled_strata = _staging_strata(sampled_households, tag)
    drawn = sampled_strata.value_counts()
    inverse_rate = sampled_strata.map(stratum_size).to_numpy(
        dtype=np.float64
    ) / sampled_strata.map(drawn).to_numpy(dtype=np.float64)
    sampled_spine = sampled_households[tag].astype(str).to_numpy()
    weights = sampled.weights_for("household")
    values = np.asarray(weights.values, dtype=np.float64) * inverse_rate
    factors = {}
    for value, mass in full_mass.items():
        mask = sampled_spine == value
        sampled_mass = float(values[mask].sum())
        if sampled_mass <= 0.0:
            raise ValueError(
                f"The {token} rung drew no weight on spine {value!r}; the "
                "sample cannot stand for it."
            )
        factors[value] = mass / sampled_mass
        values[mask] *= factors[value]
    zero_draw = sorted(set(stratum_size.index) - set(drawn.index))
    new_total = float(values.sum())
    normalized = sampled.with_weights(
        "household",
        weights.with_values(values, weights.kind),
        mass=MassChange(
            factor=new_total / float(weights.total),
            reason=(
                f"ACS local {token} development rung: inverse stratum sampling "
                "rate, then per-spine normalization to the full staging "
                "household mass"
            ),
        ),
    )
    receipt = {
        **{key: value for key, value in receipt.items() if key != "strata"},
        "rung": token,
        "sample_fraction": float(fraction),
        "sample_seed": int(seed),
        "sampled": True,
        "strata": "household_spine x congressional_district_geoid",
        "n_strata": int(len(stratum_size)),
        "zero_draw_strata": len(zero_draw),
        "zero_draw_weight_share": float(
            stratum_mass.reindex(zero_draw).sum() / full_weights.sum()
        )
        if zero_draw
        else 0.0,
        "per_spine_normalization_factor": factors,
        "full_household_mass_by_spine": full_mass,
        "households": int(len(sampled_households)),
    }
    return normalized, receipt


def iter_chunks(n: int, size: int) -> Iterable[tuple[int, int]]:
    for low in range(0, n, size):
        yield low, min(low + size, n)
