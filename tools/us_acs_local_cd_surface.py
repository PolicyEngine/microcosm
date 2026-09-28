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
#: How the ``state_cd`` surface resolves the two vintages of a state concept.
STATE_CD_VINTAGE_RULE = "state_cd.historic_table_2_level_cd_file_shares.v1"

#: District-file measures that read the wrong IRS column (microcosm#1038).
#: PR #1040 excludes the same columns in the compiler; until it lands this
#: register keeps them off the ``state_cd`` surface at both geographies.
STATE_CD_DEFECTIVE_CD_FILE_MEASURES: Mapping[str, str] = {
    "limited_state_local_taxes_amount": (
        "22incd.csv column A18425 is state and local income taxes (Schedule A "
        "line 5a), not the limited SALT deduction (A18460); the district "
        "file sums to 1.98x Historic Table 2 (microcosm#1038)."
    ),
    "limited_state_local_taxes_returns": (
        "22incd.csv column N18425 counts state and local income-tax "
        "deductions, not the limited SALT deduction (microcosm#1038)."
    ),
    "premium_tax_credit_returns": (
        "22incd.csv column N85530 is the additional Medicare tax (Form 8959 "
        "line 24), not the premium tax credit (N85770) (microcosm#1038)."
    ),
}
#: District-file measures with no state parent in either vintage, so their
#: district rows could not nest in a bound state target.
STATE_CD_UNPARENTED_CD_MEASURES: Mapping[str, str] = {
    "tax_filer_individual_count": (
        "The compiler drops this measure at state and national geography "
        "(fiscal_targets._soi_reference_from_fact), so no state target exists "
        "for its district rows to reconcile to."
    ),
}
#: States whose district SOI rows stay off the surface, with the evidence.
#: Empty since #1043 rebuilt the 117th->119th crosswalk from the block plan
#: registry: the packaged crosswalk before it (sha256 c7cb040b...) carried
#: North Carolina's 2016 plan instead of the 2019 plan the 117th Congress used,
#: and 37.0% of NC's population mapped to a different 119th district, so NC's
#: district rows were held off until then.
STATE_CD_EXCLUDED_CD_STATES: Mapping[str, str] = {}
#: The packaged crosswalk the exclusions above were reviewed against. A test
#: pins it, so regenerating the crosswalk forces a second look at them.
STATE_CD_REVIEWED_CROSSWALK_SHA256 = (
    "a347303fff3fea145f758488c43cb94355df5a8cb5e512552b2ee3791d2aa233"
)

#: Relative tolerance for "district rows sum to their state parent". The
#: rebase makes it exact up to float64 rounding.
RECONCILIATION_RTOL = 1e-9

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
# Sampling constants (DESIGN.md "Production US stacked spine")
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


def _crosswalk_district_sets(crosswalk: pd.DataFrame):
    """(source districts per state, target districts per state) as SSDD."""

    source: dict[str, set[str]] = defaultdict(set)
    target: dict[str, set[str]] = defaultdict(set)
    for source_id, target_id in zip(
        crosswalk["source_geography_id"].astype(str),
        crosswalk["target_geography_id"].astype(str),
        strict=True,
    ):
        source[source_id[-4:-2]].add(source_id[-4:])
        target[target_id[-4:-2]].add(target_id[-4:])
    return dict(source), dict(target)


def _factor_summary(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "min": float(array.min()),
        "median": float(np.median(array)),
        "max": float(array.max()),
    }


def state_cd_soi_surface(
    soi_specs: Sequence,
    *,
    state_surface_predicate: Callable[[object], bool],
    crosswalk: pd.DataFrame,
    crosswalk_sha256: str | None = None,
) -> StateCdSurface:
    """Select and reconcile the ``state_cd`` SOI surface.

    Args:
        soi_specs: Every compiled ``irs_soi`` spec (the ``full`` surface).
        state_surface_predicate: The ``state`` mode's predicate; its specs
            (Historic Table 2) are the state surface.
        crosswalk: The 117th->119th CD crosswalk the compiler used; it names
            each state's source-plan and current-plan districts.
        crosswalk_sha256: Recorded in the receipt.

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

    source_districts, target_districts = _crosswalk_district_sets(crosswalk)
    at_large_source_states = sorted(
        state for state, districts in source_districts.items() if len(districts) == 1
    )
    dropped: Counter = Counter()
    dropped_examples: dict[str, str] = {}

    def drop(spec, reason: str) -> None:
        dropped[reason] += 1
        dropped_examples.setdefault(reason, spec.name)

    ht2_state: list = []
    cd_file_state: list = []
    cd_rows: list = []
    for spec in soi_specs:
        level = geography_level(spec)
        if state_surface_predicate(spec):
            ht2_state.append(spec)
        elif is_cd_file_spec(spec) and level == GEOGRAPHY_STATE:
            cd_file_state.append(spec)
        elif is_cd_file_spec(spec) and level == GEOGRAPHY_CD:
            cd_rows.append(spec)
        elif level == GEOGRAPHY_CD:
            drop(spec, "historic_table_2_at_large_proxy_district_row")
        else:
            drop(spec, f"not_state_cd_geography:{level}")

    ht2_by_key = {}
    for spec in ht2_state:
        key = (spec.metadata["state_fips"], soi_concept_identity(spec))
        if key in ht2_by_key:
            raise ValueError(
                f"Two Historic Table 2 state specs measure {key}: "
                f"{ht2_by_key[key].name} and {spec.name}."
            )
        ht2_by_key[key] = spec
    cd_file_state_by_key = {}
    for spec in cd_file_state:
        key = (spec.metadata["state_fips"], soi_concept_identity(spec))
        if key in cd_file_state_by_key:
            raise ValueError(
                f"Two district-file state specs measure {key}: "
                f"{cd_file_state_by_key[key].name} and {spec.name}."
            )
        cd_file_state_by_key[key] = spec

    kept_cd_by_key: dict[tuple, list] = defaultdict(list)
    for spec in cd_rows:
        metadata = spec.metadata
        measure = str(metadata.get("source_measure_id"))
        state = str(metadata["state_fips"])
        if measure in STATE_CD_DEFECTIVE_CD_FILE_MEASURES:
            drop(spec, f"defective_cd_file_column:{measure}")
        elif measure in STATE_CD_UNPARENTED_CD_MEASURES:
            drop(spec, f"no_state_parent:{measure}")
        elif state in at_large_source_states:
            drop(spec, "at_large_on_source_plan")
        elif state in STATE_CD_EXCLUDED_CD_STATES:
            drop(spec, f"excluded_state:{state}")
        else:
            kept_cd_by_key[(state, soi_concept_identity(spec))].append(spec)

    rebased: dict[str, object] = {}
    factors_by_measure: dict[str, list[float]] = defaultdict(list)
    parent_basis_counts: Counter = Counter()
    kept_cd_file_parents: set[str] = set()
    for key in sorted(kept_cd_by_key):
        state, identity = key
        children = sorted(
            kept_cd_by_key[key],
            key=lambda spec: spec.metadata["congressional_district_geoid"],
        )
        districts = [spec.metadata["congressional_district_geoid"] for spec in children]
        expected = sorted(target_districts.get(state, ()))
        if districts != expected:
            raise ValueError(
                f"District concept {identity} in state {state} covers "
                f"{districts}, expected the current plan's {expected}; "
                "rebasing an incomplete set would misstate every district."
            )
        semantics = {soi_materializer_semantics(spec) for spec in children}
        if len(semantics) != 1:
            raise ValueError(
                f"District rows of {identity} in state {state} materialize "
                f"differently: {sorted(map(repr, semantics))}."
            )
        (child_semantics,) = semantics
        parent = ht2_by_key.get(key)
        basis = "historic_table_2"
        if parent is None:
            parent = cd_file_state_by_key.get(key)
            basis = "cd_file_state_total"
            if parent is None:
                raise ValueError(
                    f"District concept {identity} in state {state} has no "
                    "state parent in either vintage."
                )
            kept_cd_file_parents.add(parent.name)
        if soi_materializer_semantics(parent) != child_semantics:
            raise ValueError(
                f"State parent {parent.name} materializes differently from "
                f"its district rows ({identity}, state {state})."
            )
        child_sum = math.fsum(float(spec.value) for spec in children)
        parent_value = float(parent.value)
        if child_sum == 0.0:
            if parent_value != 0.0:
                raise ValueError(
                    f"District rows of {identity} in state {state} sum to zero "
                    f"under a nonzero parent {parent.name}={parent_value}."
                )
            factor = 1.0
        else:
            factor = parent_value / child_sum
            if factor < 0.0:
                raise ValueError(
                    f"District rows of {identity} in state {state} sum to "
                    f"{child_sum}, opposite in sign to parent {parent.name}="
                    f"{parent_value}."
                )
        factors_by_measure[identity[0]].append(factor)
        parent_basis_counts[basis] += len(children)
        for child in children:
            metadata = dict(child.metadata)
            metadata.update(
                {
                    "state_cd_vintage_rule": STATE_CD_VINTAGE_RULE,
                    "state_cd_parent_target_name": parent.name,
                    "state_cd_parent_basis": basis,
                    "state_cd_parent_value": repr(parent_value),
                    "state_cd_cd_file_value": repr(float(child.value)),
                    "state_cd_cd_file_state_sum": repr(child_sum),
                    "state_cd_rebase_factor": repr(factor),
                }
            )
            value = float(child.value) * factor
            rebased[child.name] = replace(
                child,
                value=value,
                signed=value < 0.0,
                metadata=metadata,
            )

    kept_cd_file_state = []
    for spec in cd_file_state:
        measure = str(spec.metadata.get("source_measure_id"))
        key = (spec.metadata["state_fips"], soi_concept_identity(spec))
        if measure in STATE_CD_DEFECTIVE_CD_FILE_MEASURES:
            drop(spec, f"defective_cd_file_column:{measure}")
        elif key in ht2_by_key:
            drop(spec, "second_vintage_of_state_concept")
        else:
            kept_cd_file_state.append(spec)
    missing_parents = kept_cd_file_parents - {spec.name for spec in kept_cd_file_state}
    if missing_parents:
        raise ValueError(
            f"District-file state parents {sorted(missing_parents)[:5]} were "
            "dropped while their district rows were kept."
        )

    cd_specs = tuple(rebased[spec.name] for spec in cd_rows if spec.name in rebased)
    specs = (*ht2_state, *kept_cd_file_state, *cd_specs)
    with_sigma = sum(1 for spec in specs if getattr(spec, "se", None) is not None)
    receipt = {
        "vintage_rule": STATE_CD_VINTAGE_RULE,
        "vintage_rule_description": (
            "Historic Table 2 (TY2022) is the single vintage of every state "
            "concept it carries. District-file (22incd.csv) district rows "
            "keep only their within-state shares and are rebased to sum to "
            "that parent. Concepts only the district file carries keep its "
            "own state row as their parent."
        ),
        "counts": {
            "historic_table_2_state": len(ht2_state),
            "cd_file_state": len(kept_cd_file_state),
            "congressional_district": len(cd_specs),
            "congressional_district_by_parent_basis": dict(parent_basis_counts),
            "total": len(specs),
        },
        "dropped": dict(sorted(dropped.items())),
        "dropped_examples": dict(sorted(dropped_examples.items())),
        "defective_cd_file_measures": dict(STATE_CD_DEFECTIVE_CD_FILE_MEASURES),
        "unparented_cd_measures": dict(STATE_CD_UNPARENTED_CD_MEASURES),
        "excluded_cd_states": dict(STATE_CD_EXCLUDED_CD_STATES),
        "at_large_on_source_plan": at_large_source_states,
        "rebase_factor_by_measure": {
            measure: _factor_summary(values)
            for measure, values in sorted(factors_by_measure.items())
        },
        "crosswalk": {
            "source_plan": "117th_congress",
            "target_plan": "119th_congress",
            "method": "packaged 2020-block population crosswalk",
            "sha256": crosswalk_sha256,
        },
        "sigma": {
            "targets_with_sigma": with_sigma,
            "targets_without_sigma": len(specs) - with_sigma,
            "note": (
                "The pinned feed carries no uncertainty field for any fact; "
                "IRS SOI tables are administrative. targets.json records "
                "sigma wherever a spec carries one."
            ),
        },
    }
    return StateCdSurface(specs=specs, receipt=receipt)


def state_parent_reconciliation(
    specs: Sequence, *, rtol: float = RECONCILIATION_RTOL
) -> list[dict]:
    """Every (state parent, district rows) block and whether it adds up.

    A block is the district specs naming one ``state_cd_parent_target_name``.
    Returns one entry per block with the parent value, the district sum and
    ``ok`` (relative difference within ``rtol``).
    """

    by_name = {spec.name: spec for spec in specs}
    blocks: dict[str, list] = defaultdict(list)
    for spec in specs:
        parent = spec.metadata.get("state_cd_parent_target_name")
        if parent is not None:
            blocks[parent].append(spec)
    report = []
    for parent_name, children in sorted(blocks.items()):
        parent = by_name.get(parent_name)
        child_sum = math.fsum(float(spec.value) for spec in children)
        parent_value = None if parent is None else float(parent.value)
        scale = max(abs(parent_value or 0.0), 1.0)
        report.append(
            {
                "parent": parent_name,
                "parent_bound": parent is not None,
                "parent_value": parent_value,
                "district_sum": child_sum,
                "n_districts": len(children),
                "ok": parent is not None
                and abs(child_sum - parent_value) <= rtol * scale,
            }
        )
    return report


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
    chunk (ties to the lower row). The assembler compares each one with its
    carrier-derived row and refuses any difference.
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
    ) -> None:
        """Add a carrier restricted to chunk ``positions`` (the district's)."""

        values32 = np.asarray(carrier[positions], dtype=np.float32)
        if not np.isfinite(values32).all():
            raise ValueError(
                f"Target {name!r} derives non-finite values from its carrier."
            )
        keep = values32 != 0
        if not keep.any():
            return
        self._rows.append(np.full(int(keep.sum()), row, dtype=np.int32))
        self._cols.append((positions[keep] + low).astype(np.int32))
        self._data.append(values32[keep])

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
) -> None:
    """Add every district SOI row of one chunk from its carrier column."""

    positions_by_district = district_positions(district_codes)
    states = np.asarray(state_codes, dtype=np.int64)
    empty = np.empty(0, dtype=np.int64)
    for index, state, district, carrier in plan.split:
        positions = positions_by_district.get(district, empty)
        if len(positions) and state != NO_STATE_MASK:
            positions = positions[states[positions] == state]
        assembler.add_masked(
            index,
            low,
            carrier_columns[carrier],
            positions,
            name=plan.declared[index].name,
        )


def carrier_derived_column(
    carrier: np.ndarray,
    *,
    state: int,
    district: int,
    state_codes: np.ndarray,
    district_codes: np.ndarray,
) -> np.ndarray:
    """The dense column a district row derives from its carrier (float64)."""

    mask = np.asarray(district_codes, dtype=np.int64) == district
    if state != NO_STATE_MASK:
        mask &= np.asarray(state_codes, dtype=np.int64) == state
    return np.where(mask, np.asarray(carrier, dtype=np.float64), 0.0)


def population_rows(
    households: pd.DataFrame,
    household_size: np.ndarray,
    ladder_populations: Mapping[str, Mapping[int, float]],
    geographies: Sequence[str],
) -> tuple[list[dict], sparse.csr_array, list[str]]:
    """Ladder population marginals as sparse rows (household size x 1[geo]).

    Returns (target records, CSR rows, dropped cell names). A ladder cell
    with no supporting household is dropped and named, as before.
    """

    column_map = {
        "state": ("state_fips", "state", 2, GEOGRAPHY_STATE),
        "cd": ("congressional_district_geoid", "cd", 4, GEOGRAPHY_CD),
    }
    size32 = np.asarray(household_size, dtype=np.float32)
    records: list[dict] = []
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []
    dropped: list[str] = []
    for geography in geographies:
        column, key, width, level = column_map[geography]
        codes = pd.to_numeric(households[column]).to_numpy()
        for value, population in sorted(ladder_populations[key].items()):
            name = f"pop_{geography}_{value:0{width}d}"
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


def target_record(spec) -> dict:
    """The targets.json record for one admin spec (metadata subset)."""

    metadata = spec.metadata
    level = metadata.get("ledger_geography_level")
    se = getattr(spec, "se", None)
    record = {
        "name": spec.name,
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
    for key in (
        "state_cd_parent_target_name",
        "state_cd_parent_basis",
        "state_cd_rebase_factor",
        "state_cd_cd_file_value",
    ):
        if key in metadata:
            record[key] = metadata[key]
    return record


def cd_holdout_unit_key(record: Mapping) -> str | None:
    """The holdout unit of a district SOI target, or None if not eligible."""

    if record.get("family") != "irs_soi":
        return None
    if record.get("geography_level") != GEOGRAPHY_CD:
        return None
    if not record.get("state_cd_parent_target_name"):
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
        parent = by_name.get(record.get("state_cd_parent_target_name"))
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

    rows = holdout_rows(records)
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
    weights = np.asarray(weights, dtype=np.float64)
    denominator = float((weights**2).sum())
    return float(weights.sum() ** 2 / denominator) if denominator > 0 else 0.0


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


def ess_by_group(weights: np.ndarray, codes: np.ndarray) -> dict[str, float]:
    """Kish ESS of the records in each group, keyed by the group code."""

    frame = pd.DataFrame(
        {"code": np.asarray(codes).astype(str), "w": np.asarray(weights, float)}
    )
    frame["w2"] = frame["w"] ** 2
    sums = frame.groupby("code", sort=True)[["w", "w2"]].sum()
    return {
        str(code): float(row.w**2 / row.w2) if row.w2 > 0 else 0.0
        for code, row in sums.iterrows()
    }


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

    weights = np.asarray(weights, dtype=np.float64)
    summary: dict[str, object] = {
        "rows": int(len(weights)),
        "effective_sample_size_rows": kish_ess(weights),
        "top_1pct_weight_share": top_weight_share(weights, 0.01),
    }
    for label, codes in (("state", state), ("district", district)):
        if codes is None:
            continue
        by_group = ess_by_group(weights, codes)
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
        f"{sorted(SAMPLE_RUNG_TOKENS)} (DESIGN.md 'Production US stacked spine')."
    )


def sample_staging_frame(frame, *, fraction: float, seed: int):
    """Sample whole households at a rung, stratified by spine and district.

    ``fraction == 1`` returns the frame unchanged. Otherwise
    :func:`microcosm.build.frame_sampling.sample_frame_households` draws
    ``floor(fraction * n)`` households per (spine, district) stratum, and each
    spine's weights are scaled back to that spine's full household mass, so
    per-spine totals and the district mix are preserved.
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
    spine = households[tag].astype(str).to_numpy()
    strata = pd.Series(spine).str.cat(
        households["congressional_district_geoid"].astype(str).to_numpy(), sep="|cd="
    )
    full_weights = np.asarray(frame.weights_for("household").values, dtype=np.float64)
    full_mass = {
        value: float(full_weights[spine == value].sum()) for value in sorted(set(spine))
    }
    sampled, receipt = sample_frame_households(
        frame,
        fraction=float(fraction),
        seed=int(seed),
        source_name="ACS local staging",
        unit_strata=strata.to_numpy(),
        floor_context="the ACS local development rung",
    )
    sampled_households = sampled.table("household")
    sampled_spine = sampled_households[tag].astype(str).to_numpy()
    weights = sampled.weights_for("household")
    values = np.asarray(weights.values, dtype=np.float64).copy()
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
    new_total = float(values.sum())
    normalized = sampled.with_weights(
        "household",
        weights.with_values(values, weights.kind),
        mass=MassChange(
            factor=new_total / float(weights.total),
            reason=(
                f"ACS local {token} development rung: per-spine normalization "
                "to the full staging household mass"
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
        "n_strata": int(strata.nunique()),
        "per_spine_normalization_factor": factors,
        "full_household_mass_by_spine": full_mass,
        "households": int(len(sampled_households)),
    }
    return normalized, receipt


def iter_chunks(n: int, size: int) -> Iterable[tuple[int, int]]:
    for low in range(0, n, size):
        yield low, min(low + size, n)
