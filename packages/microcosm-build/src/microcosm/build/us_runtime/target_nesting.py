"""US target nesting on the shared cross-grain operator (microcosm#1760).

A target for a small area must agree with the target of the area that
contains it. Each pass here makes the rows of a lower geography level from a
control at a higher level. The arithmetic and the checks are not here:

- :func:`microcosm.build.cross_grain_controls.reconcile_rows` runs the shared
  operator (:mod:`microcosm.build.cross_grain`) on the rows of a pass and
  says what it did with each row. The operator divides each control among
  its lower rows, bounds the factor of each bridge, refuses a leg that does
  not hold its declared members, and writes the receipt.
- ``us/cross_grain_declarations.json`` declares the passes, under the key
  names of the UK file: the bridge between Historic Table 2 and the district
  file, the bridge between the state rows and the county file, the derived
  control of the county households, and the ladder population, which is one
  source at two levels.

What is here is the US side of each pass: which specs or facts are the
control and the lower rows, which rows a reviewed register keeps off the
surface, and the validation of the county facts. A reconciled row keeps the
``cross_grain_*`` metadata keys that the UK rows keep.

The module reads specs and facts only. It reads no frame and no household.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build import cross_grain_controls as controls
from microcosm.build.cross_grain import (
    MEMBERS_KEY,
    PARENT_KEY,
    CrossGrainBridge,
    CrossGrainBridgeSelector,
    CrossGrainRule,
    cross_grain_metadata,
    expand_bridge_selectors,
)
from microcosm.build.cross_grain_controls import SurfaceRow
from microcosm.build.us_runtime.cross_grain_declarations import (
    COUNTY_ACS_HOUSEHOLDS,
    COUNTY_POPULATION,
    COUNTY_SOI,
    DISTRICT_SOI,
    load_us_cross_grain_declarations,
    us_allocator_name,
    us_bridge_selector,
    us_county_soi_nesting,
    us_district_soi_nesting,
    us_pass_grains,
)

#: The file a materialize stage writes the operator's receipts to.
NESTING_RECEIPTS_FILENAME = "target_nesting.json"

# ---------------------------------------------------------------------------
# One pass on the operator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NestedSurface:
    """The specs of one nested surface, its receipt and the operator's."""

    specs: tuple
    receipt: dict
    operator_receipt: dict


def geography_id(grain: str, code: str) -> str:
    """A surface geography id that cannot collide across grains."""

    return f"{grain}:{code}"


def _own_measurement(target_id: str) -> dict[str, Any]:
    """A measurement that only this target has.

    A US target is one source's rows of one concept. Two sources have two
    measurements, so only a declared bridge ties them. One target at two
    levels is one measurement at both.
    """

    return {"measurement": {"concept": target_id, "filters": []}}


def run_pass(
    pass_id: str,
    rows: Sequence[SurfaceRow],
    *,
    bound_higher_targets: Iterable[str],
    legs: Mapping[str, str],
    bridges: Sequence[CrossGrainBridge] = (),
    expected_members: Mapping[str, Mapping[str, frozenset[str]]] | None = None,
    measurements: Mapping[str, str] | None = None,
) -> tuple[list[controls.RowOutcome], dict[str, Any]]:
    """Run one declared pass on the shared operator.

    ``legs`` maps each lower geography to the control geography it sits
    under. ``measurements`` names the measurement of a target where it is
    not the target's own. Returns the outcome of each row and the operator's
    receipt.
    """

    declarations = load_us_cross_grain_declarations()
    control_grains = tuple(declarations["control_grains"])
    allocator = us_allocator_name(pass_id)
    rule = CrossGrainRule(
        grain_precedence=tuple(declarations["grain_precedence"]),
        signature_fields=tuple(declarations["signature_fields"]),
        bridges=tuple(bridges),
        leg_of_area=legs.__getitem__,
        parent_geography_legs={
            row.geography_id: (row.geography_id,)
            for row in rows
            if row.grain in control_grains
        },
        control_grains=control_grains,
        allocator=None if allocator is None else controls.allocator_named(allocator),
    )
    outcomes, receipt = controls.reconcile_rows(
        rows,
        tuple(dict.fromkeys(bound_higher_targets)),
        {
            row.target_id: _own_measurement(
                (measurements or {}).get(row.target_id, row.target_id)
            )
            for row in rows
        },
        rule,
        expected_members=expected_members,
    )
    _record(pass_id, receipt)
    return outcomes, receipt


def bounded_selector(pass_id: str, factor_band: float) -> CrossGrainBridgeSelector:
    """The declared selector of a pass, with the bound that a caller asks for.

    ``factor_band`` is the bound as a ratio to the median factor: a leg may
    sit between ``1 / factor_band`` and ``factor_band`` of it. The declared
    band gives the declared selector. An infinite band turns the bound off.
    """

    if math.isinf(factor_band):
        return us_bridge_selector(pass_id, bounded=False)
    declared = us_bridge_selector(pass_id)
    if factor_band == 1.0 + declared.max_factor_shift:
        return declared
    return replace(
        declared,
        max_factor_shift=factor_band - 1.0,
        min_factor_shift=1.0 / factor_band - 1.0,
    )


def bridge_receipt(
    bridge_id: str,
    bridge: CrossGrainBridge | CrossGrainBridgeSelector,
    unbound: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """What a surface receipt says of its bridge: the bound, and the legs it unbound."""

    return {
        "bridge_id": bridge_id,
        "max_factor_shift": bridge.max_factor_shift,
        "min_factor_shift": bridge.min_factor_shift,
        "shift_around": bridge.shift_around,
        "out_of_bound": bridge.out_of_bound,
        "unbound": sorted(unbound, key=lambda row: (row["concept"], row["state_fips"])),
    }


# ---------------------------------------------------------------------------
# The operator's receipts
# ---------------------------------------------------------------------------

_RECEIPTS: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "us_target_nesting_receipts", default=None
)


def start_nesting_receipts() -> list[dict[str, Any]]:
    """Start a fresh record of the nesting passes that run from here on.

    A stage calls this first and writes the list with
    :func:`write_nesting_receipts` when its targets are made. Without a call,
    nothing is recorded.
    """

    receipts: list[dict[str, Any]] = []
    _RECEIPTS.set(receipts)
    return receipts


def stop_nesting_receipts() -> None:
    """End the record that :func:`start_nesting_receipts` started."""

    _RECEIPTS.set(None)


def _record(pass_id: str, receipt: Mapping[str, Any]) -> None:
    receipts = _RECEIPTS.get()
    if receipts is not None:
        receipts.append(
            {
                "pass": pass_id,
                "allocator": us_allocator_name(pass_id),
                "operator_receipt": dict(receipt),
            }
        )


def write_nesting_receipts(
    checkpoint_dir: Path, receipts: Sequence[Mapping[str, Any]]
) -> Path:
    """Write the recorded operator receipts beside the stage's other files.

    The file is a receipt only. No stage reads it, and it is not part of the
    run identity.
    """

    path = Path(checkpoint_dir) / NESTING_RECEIPTS_FILENAME
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "operator": "microcosm.build.cross_grain",
                "declarations": "us/cross_grain_declarations.json",
                "passes": list(receipts),
            },
            indent=2,
        )
    )
    return path


# ---------------------------------------------------------------------------
# State to congressional district: IRS SOI (issue row 5)
# ---------------------------------------------------------------------------

#: The sources of a district SOI target, as the declared selector names them.
STATE_SURFACE = "state_surface"
LOWER_FILE = "lower_file"
#: The lower file's own state total, where it is the parent of its districts.
LOWER_FILE_STATE = "lower_file_state"


def crosswalk_district_sets(crosswalk: pd.DataFrame):
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


def district_soi_surface(
    soi_specs: Sequence,
    *,
    state_surface_predicate: Callable[[object], bool],
    crosswalk: pd.DataFrame,
    crosswalk_sha256: str | None,
    level_bridges: Mapping[str, str],
    factor_band: float,
    concept_identity: Callable[[object], tuple[str, ...]],
    materializer_semantics: Callable[[object], tuple],
    is_lower_file_spec: Callable[[object], bool],
    closure_report: Callable[[Sequence], list[dict]],
) -> NestedSurface:
    """Select and reconcile the ``state_cd`` SOI surface.

    Historic Table 2 supplies a state concept's level wherever it has that
    concept. The district file then supplies only each district's share of
    its state. The declared bridge ``historic_table_2_state_vs_district_file``
    ties the two files, one bridge for each concept, and the operator divides
    each state row among its district rows. A state whose factor is outside
    the bridge's bound around the median across states is unbound: its
    district rows leave the surface and the receipt names them.

    Args:
        soi_specs: Every compiled ``irs_soi`` spec (the ``full`` surface).
        state_surface_predicate: The ``state`` mode's predicate; its specs
            (Historic Table 2) are the state surface.
        crosswalk: The 117th->119th CD crosswalk the compiler used; it names
            each state's source-plan and current-plan districts.
        crosswalk_sha256: Recorded in the receipt.
        level_bridges: District-file-only measure -> sibling measure (until
            the feed re-pin; see :func:`_bridged_state_parents`).
        factor_band: The bound as a ratio to the median factor: a state may
            sit between ``1 / factor_band`` and ``factor_band`` of it.
            Infinity turns the bound off.
        concept_identity: The published concept a spec measures, independent
            of vintage.
        materializer_semantics: Everything the SOI slice reads from a spec
            except its geography.
        is_lower_file_spec: Whether a spec comes from the district file.
        closure_report: Every (state parent, district rows) block of the
            finished surface and whether it adds up; a block that does not is
            refused.

    Returns:
        The surface in a fixed order (Historic Table 2 state rows, district
        file state rows, district rows), its receipt and the operator's.

    Raises:
        ValueError: If a district concept is not complete in a state, if its
            district rows disagree on what they materialize, if a parent's
            materialization differs from its children's, if a concept takes
            its parent from two sources, or if the operator refuses a block
            (rows that sum to zero under a control that is not zero, or a
            sign conflict).
    """

    nesting = us_district_soi_nesting()
    state_level, district_level = us_pass_grains(DISTRICT_SOI)
    defective = nesting.defective_lower_measures
    unparented = nesting.unparented_lower_measures
    excluded_states = nesting.excluded_legs

    source_districts, target_districts = crosswalk_district_sets(crosswalk)
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
        level = spec.metadata.get("ledger_geography_level")
        if state_surface_predicate(spec):
            ht2_state.append(spec)
        elif is_lower_file_spec(spec) and level == state_level:
            cd_file_state.append(spec)
        elif is_lower_file_spec(spec) and level == district_level:
            cd_rows.append(spec)
        elif level == district_level:
            drop(spec, "historic_table_2_at_large_proxy_district_row")
        else:
            drop(spec, f"not_state_cd_geography:{level}")

    ht2_by_key = {}
    for spec in ht2_state:
        key = (spec.metadata["state_fips"], concept_identity(spec))
        if key in ht2_by_key:
            raise ValueError(
                f"Two Historic Table 2 state specs measure {key}: "
                f"{ht2_by_key[key].name} and {spec.name}."
            )
        ht2_by_key[key] = spec
    cd_file_state_by_key = {}
    for spec in cd_file_state:
        key = (spec.metadata["state_fips"], concept_identity(spec))
        if key in cd_file_state_by_key:
            raise ValueError(
                f"Two district-file state specs measure {key}: "
                f"{cd_file_state_by_key[key].name} and {spec.name}."
            )
        cd_file_state_by_key[key] = spec

    # Which district rows enter the surface: the reviewed registers.
    kept_cd_by_key: dict[tuple, list] = defaultdict(list)
    for spec in cd_rows:
        metadata = spec.metadata
        measure = str(metadata.get("source_measure_id"))
        state = str(metadata["state_fips"])
        if measure in defective:
            drop(spec, f"defective_cd_file_column:{measure}")
        elif measure in unparented:
            drop(spec, f"no_state_parent:{measure}")
        elif state in at_large_source_states:
            drop(spec, "at_large_on_source_plan")
        elif state in excluded_states:
            drop(spec, f"excluded_state:{state}")
        else:
            kept_cd_by_key[(state, concept_identity(spec))].append(spec)

    bridged, bridge_factors, bridge_verdicts = _bridged_state_parents(
        ht2_by_key,
        cd_file_state_by_key,
        kept_cd_by_key,
        level_bridges=level_bridges,
        factor_band=factor_band,
        defective=defective,
    )

    # The surface: for each block, its state row and its district rows.
    rows: list[SurfaceRow] = []
    blocks: list[tuple] = []
    legs: dict[str, str] = {}
    expected: dict[str, dict[str, frozenset[str]]] = {}
    bound: list[str] = []
    targets: dict[str, dict[str, str]] = {}
    measurements: dict[str, str] = {}
    parent_source: dict[tuple, str] = {}
    kept_cd_file_parents: set[str] = set()
    for key in sorted(kept_cd_by_key):
        state, identity = key
        children = sorted(
            kept_cd_by_key[key],
            key=lambda spec: spec.metadata["congressional_district_geoid"],
        )
        semantics = {materializer_semantics(spec) for spec in children}
        if len(semantics) != 1:
            raise ValueError(
                f"District rows of {identity} in state {state} materialize "
                f"differently: {sorted(map(repr, semantics))}."
            )
        (child_semantics,) = semantics
        parent = ht2_by_key.get(key)
        basis, source = "historic_table_2", STATE_SURFACE
        if parent is None:
            own = cd_file_state_by_key.get(key)
            basis, source = "cd_file_state_total_bridged", LOWER_FILE_STATE
            if own is None:
                raise ValueError(
                    f"District concept {identity} in state {state} has no "
                    "state parent in either vintage."
                )
            if own.name not in bridged:
                for child in children:
                    drop(child, f"level_bridge_out_of_band:{identity[0]}")
                continue
            parent = bridged[own.name]
            kept_cd_file_parents.add(parent.name)
        if parent_source.setdefault(identity, source) != source:
            raise ValueError(
                f"District concept {identity} takes its state parent from "
                "Historic Table 2 in some states and from the district file in "
                "others; one concept has one parent source."
            )
        if materializer_semantics(parent) != child_semantics:
            raise ValueError(
                f"State parent {parent.name} materializes differently from "
                f"its district rows ({identity}, state {state})."
            )
        identity_text = "|".join(identity)
        lower_id = f"{LOWER_FILE}|{identity_text}"
        higher_id = f"{source}|{identity_text}"
        state_area = geography_id(state_level, state)
        targets[higher_id] = {
            "source": source,
            "grain": state_level,
            "identity": identity_text,
        }
        targets[lower_id] = {
            "source": LOWER_FILE,
            "grain": district_level,
            "identity": identity_text,
        }
        # The lower file's own state total has the measurement of its
        # district rows, so the operator ties them with no bridge.
        measurements[higher_id] = lower_id if source == LOWER_FILE_STATE else higher_id
        measurements[lower_id] = lower_id
        bound.append(higher_id)
        blocks.append((key, parent, basis, children, len(rows)))
        rows.append(SurfaceRow(state_level, state_area, higher_id, float(parent.value)))
        for child in children:
            area = geography_id(
                district_level, child.metadata["congressional_district_geoid"]
            )
            legs[area] = state_area
            rows.append(SurfaceRow(district_level, area, lower_id, float(child.value)))
        expected.setdefault(lower_id, {})[state_area] = frozenset(
            geography_id(district_level, district)
            for district in target_districts.get(state, ())
        )

    selector = bounded_selector(DISTRICT_SOI, factor_band)
    bridges, _unpaired = expand_bridge_selectors([selector], targets)
    outcomes, operator_receipt = run_pass(
        DISTRICT_SOI,
        rows,
        bound_higher_targets=bound,
        legs=legs,
        bridges=bridges,
        expected_members=expected,
        measurements=measurements,
    )

    rebased: dict[str, object] = {}
    factors_by_measure: dict[str, list[float]] = defaultdict(list)
    parent_basis_counts: Counter = Counter()
    unbound: list[dict] = []
    for key, parent, basis, children, start in blocks:
        state, identity = key
        block = outcomes[start + 1 : start + 1 + len(children)]
        statuses = {outcome.status for outcome in block}
        if statuses == {controls.UNBOUND}:
            unbound.append(
                {
                    "state_fips": state,
                    "concept": identity[0],
                    "parent": parent.name,
                    "factor": block[0].factor,
                }
            )
            for child in children:
                drop(child, f"rebase_out_of_band:{identity[0]}")
            continue
        if statuses != {controls.RECONCILED}:
            raise ValueError(
                f"The operator did not reconcile the district rows of "
                f"{identity} under {parent.name}."
            )
        factors_by_measure[identity[0]].append(block[0].factor)
        parent_basis_counts[basis] += len(children)
        for child, outcome in zip(children, block, strict=True):
            rebased[child.name] = replace(
                child,
                value=outcome.value,
                signed=outcome.value < 0.0,
                metadata={
                    **child.metadata,
                    **cross_grain_metadata(
                        value_before=child.value,
                        factor=outcome.factor,
                        control=outcome.control,
                        parent=parent.name,
                    ),
                },
            )

    kept_cd_file_state = []
    for spec in cd_file_state:
        measure = str(spec.metadata.get("source_measure_id"))
        key = (spec.metadata["state_fips"], concept_identity(spec))
        if measure in defective:
            drop(spec, f"defective_cd_file_column:{measure}")
        elif key in ht2_by_key:
            drop(spec, "second_vintage_of_state_concept")
        elif spec.name not in bridged:
            drop(spec, f"level_bridge_out_of_band:{measure}")
        else:
            kept_cd_file_state.append(bridged[spec.name])
    missing_parents = kept_cd_file_parents - {spec.name for spec in kept_cd_file_state}
    if missing_parents:
        raise ValueError(
            f"District-file state parents {sorted(missing_parents)[:5]} were "
            "dropped while their district rows were kept."
        )

    cd_specs = tuple(rebased[spec.name] for spec in cd_rows if spec.name in rebased)
    specs = (*ht2_state, *kept_cd_file_state, *cd_specs)
    # Fail closed on the invariant the whole surface rests on.
    unreconciled = [block for block in closure_report(specs) if not block["ok"]]
    if unreconciled:
        raise ValueError(
            f"{len(unreconciled)} district block(s) do not add up to a bound "
            f"state parent, e.g. {unreconciled[:2]}."
        )
    with_sigma = sum(1 for spec in specs if getattr(spec, "se", None) is not None)
    receipt = {
        "vintage_rule": nesting.vintage_rule,
        "vintage_rule_description": (
            "Historic Table 2 (TY2022) is the single vintage of every state "
            "concept it carries. District-file (22incd.csv) district rows "
            "keep only their within-state shares and are divided from that "
            "parent by the shared cross-grain operator. Concepts only the "
            "district file carries keep its own state row as their parent, "
            "lifted onto the Historic Table 2 basis by a sibling concept's "
            "ratio of the two files' state totals in the same state."
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
        "defective_cd_file_measures": dict(defective),
        "unparented_cd_measures": dict(unparented),
        "excluded_cd_states": dict(excluded_states),
        "at_large_on_source_plan": at_large_source_states,
        "bridge": bridge_receipt(DISTRICT_SOI, selector, unbound),
        "level_bridges": dict(level_bridges),
        "level_bridge_factor_by_measure": {
            measure: _factor_summary(values)
            for measure, values in sorted(bridge_factors.items())
        },
        "level_bridge_out_of_band": sorted(
            bridge_verdicts, key=lambda row: (row["concept"], row["state_fips"])
        ),
        "factor_by_measure": {
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
                "IRS SOI tables are administrative. target_roles.json records "
                "sigma and sigma_basis for every target, null wherever a spec "
                "carries none."
            ),
        },
    }
    return NestedSurface(
        specs=specs, receipt=receipt, operator_receipt=operator_receipt
    )


def _bridged_state_parents(
    ht2_by_key: Mapping[tuple, Any],
    cd_file_state_by_key: Mapping[tuple, Any],
    kept_cd_by_key: Mapping[tuple, Sequence],
    *,
    level_bridges: Mapping[str, str],
    factor_band: float,
    defective: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, list[float]], list[dict]]:
    """The state parents of the concepts that only the district file has.

    **Until the feed re-pin** (microcosm#1760). On the pinned feed, Historic
    Table 2 lacks six concepts that the district file has. Their state parent
    is the district file's own state row, lifted onto the Historic Table 2
    basis by a sibling concept that both files carry: the row is multiplied by
    the sibling's Historic Table 2 level over its district-file level in the
    same state. The sibling is trusted only if that ratio, and the factor of
    its own district block, sit within ``factor_band`` of their medians across
    states. After the re-pin Historic Table 2 holds the six parents, and this
    function and ``level_bridges`` go.

    Returns the lifted state rows by name, the factors by measure, and the
    verdicts that refused a sibling.
    """

    def by_identity(
        ratios: Mapping[tuple, float], *, positive_only: bool
    ) -> dict[tuple, float]:
        grouped: dict[tuple, list[float]] = defaultdict(list)
        for block_key, ratio in ratios.items():
            if math.isfinite(ratio) and (ratio > 0 or not positive_only):
                grouped[_identity_of_block(block_key)].append(ratio)
        return {
            identity: float(np.median(values)) for identity, values in grouped.items()
        }

    vintage_ratio: dict[tuple, float] = {}
    for key in ht2_by_key.keys() & cd_file_state_by_key.keys():
        higher = float(ht2_by_key[key].value)
        lower = float(cd_file_state_by_key[key].value)
        vintage_ratio[key] = (
            higher / lower if lower != 0.0 and (higher > 0) == (lower > 0) else math.nan
        )
    median_ratio = by_identity(vintage_ratio, positive_only=False)
    block_factor: dict[tuple, float] = {}
    for key, block in kept_cd_by_key.items():
        parent = ht2_by_key.get(key)
        child_sum = controls.sum_of_lower_rows(spec.value for spec in block)
        if parent is not None and child_sum != 0.0:
            block_factor[key] = float(parent.value) / child_sum
    median_factor = by_identity(block_factor, positive_only=True)
    verdicts: list[dict] = []

    def trusted(
        sibling_key: tuple, ratio: float, median: float, measure: str, verdict: str
    ) -> bool:
        relative = ratio / median if math.isfinite(ratio) and median else math.nan
        ok = math.isfinite(relative) and 1 / factor_band <= relative <= factor_band
        if not ok:
            verdicts.append(
                {
                    "state_fips": sibling_key[0],
                    "measure": measure,
                    "concept": _identity_of_block(sibling_key)[0],
                    "verdict": verdict,
                    "ratio": ratio,
                    "median_ratio": median,
                    "relative": relative,
                }
            )
        return ok

    bridged: dict[str, Any] = {}
    factors: dict[str, list[float]] = defaultdict(list)
    for key, spec in sorted(cd_file_state_by_key.items()):
        state, identity = key
        measure = identity[0]
        if key in ht2_by_key or measure in defective:
            continue
        sibling = level_bridges.get(measure)
        if sibling is None:
            raise ValueError(
                f"District-file-only concept {measure!r} has no level bridge; "
                "register a sibling measure in the level bridges of "
                "us/cross_grain_declarations.json before binding it."
            )
        sibling_key = (state, (sibling, *identity[1:]))
        ht2_sibling = ht2_by_key.get(sibling_key)
        cd_sibling = cd_file_state_by_key.get(sibling_key)
        if ht2_sibling is None or cd_sibling is None or float(cd_sibling.value) == 0:
            raise ValueError(
                f"Level bridge {measure} -> {sibling} in state {state} needs "
                "the sibling's state total in both files."
            )
        factor = float(ht2_sibling.value) / float(cd_sibling.value)
        # A sign conflict is a data defect, not a level disagreement.
        if not (math.isfinite(factor) and factor > 0.0):
            raise ValueError(
                f"Level bridge {measure} -> {sibling} in state {state} has "
                f"factor {factor}."
            )
        ok = trusted(
            sibling_key,
            factor,
            median_ratio.get(_identity_of_block(sibling_key), math.nan),
            measure,
            "sibling_vintage_ratio",
        )
        if sibling_key in block_factor:
            ok = (
                trusted(
                    sibling_key,
                    block_factor[sibling_key],
                    median_factor.get(_identity_of_block(sibling_key), math.nan),
                    measure,
                    "sibling_factor",
                )
                and ok
            )
        if not ok:
            continue
        value = float(spec.value) * factor
        bridged[spec.name] = replace(
            spec,
            value=value,
            signed=value < 0.0,
            metadata={
                **spec.metadata,
                **cross_grain_metadata(
                    value_before=spec.value,
                    factor=factor,
                    control=f"level_bridge:{sibling}",
                ),
            },
        )
        factors[measure].append(factor)
    return bridged, factors, verdicts


def district_block_closure(specs: Sequence, *, rtol: float) -> list[dict]:
    """Every (state parent, district rows) block and whether it adds up.

    A block is the district specs that name one parent
    (``cross_grain_parent``). Returns one entry per block with the parent
    value, the district sum and ``ok`` (relative difference within ``rtol``).
    """

    _state_level, district_level = us_pass_grains(DISTRICT_SOI)
    by_name = {spec.name: spec for spec in specs}
    blocks: dict[str, list] = defaultdict(list)
    for spec in specs:
        parent = spec.metadata.get(PARENT_KEY)
        if (
            parent is not None
            and spec.metadata.get("ledger_geography_level") == district_level
        ):
            blocks[parent].append(spec)
    report = []
    for parent_name, children in sorted(blocks.items()):
        parent = by_name.get(parent_name)
        child_sum = controls.sum_of_lower_rows(spec.value for spec in children)
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


def _identity_of_block(key: tuple) -> tuple:
    """The concept identity of a (state, identity) block key."""

    return key[1]


# ---------------------------------------------------------------------------
# County helpers the county passes share
# ---------------------------------------------------------------------------

#: The GEOID prefix of a county-equivalent in the ledger feeds.
COUNTY_PREFIX = "0500000US"
#: Puerto Rico (state 72) is outside the County Files. The county facts loader
#: drops its municipios with a count; a row that still reaches the county rung
#: is a defect, never a county to fit.
PUERTO_RICO_STATE = 72


def county_code_of(fact, *, source):
    """The five-digit county code of a county fact, refusing bad geography.

    The row geography is authoritative. A layout group value, when present,
    must name the same county (a FIPS/state mismatch is refused), and a
    Puerto Rico row reaching the rung is refused explicitly.
    """
    geography = fact.get("geography", {})
    if geography.get("level") != "county":
        raise ValueError(f"{source} county facts must be county rows: {geography!r}.")
    geoid = str(geography.get("id", ""))
    digits = geoid[len(COUNTY_PREFIX) :]
    if (
        not geoid.startswith(COUNTY_PREFIX)
        or len(digits) != 5
        or not (digits.isascii() and digits.isdigit())
    ):
        raise ValueError(f"Invalid county {source} geography {geoid!r}.")
    code = int(digits)
    if code // 1000 == PUERTO_RICO_STATE:
        raise ValueError(
            f"Puerto Rico {source} county fact {geoid} reached the county rung; "
            "state 72 is outside the 3,144 County Files county-equivalents and "
            "the county facts loader must drop it."
        )
    if not 1 <= code // 1000 <= 56 or code % 1000 == 0:
        raise ValueError(f"Invalid county {source} geography {geoid!r}.")
    group = (fact.get("layout") or {}).get("groupby_value_id")
    if group is not None and str(group)[-5:] != digits:
        raise ValueError(
            f"{source} county fact {geoid} names layout county {group!r}; "
            "geography and layout disagree on the county FIPS."
        )
    return code


def county_hierarchy(name, code, *, parent=None):
    """The hierarchy of a county row: its parent's, or the population one."""

    from microcosm.calibrate import (
        CalibrationHierarchy,
        HierarchyCategory,
        HierarchyGeography,
        HierarchyNode,
    )

    # One deterministic label for a county in all families. Source labels are
    # retained in metadata; no mutable county-name registry is needed.
    geography = HierarchyGeography(f"{COUNTY_PREFIX}{code}", f"County {code}", "county")
    if parent is not None and parent.hierarchy is not None:
        return replace(
            parent.hierarchy,
            geography=geography,
            target=HierarchyNode(
                name, f"{parent.hierarchy.target.label}, county {code}"
            ),
        )
    provider = HierarchyNode("census_population", "Census population")
    return CalibrationHierarchy(
        provider=provider,
        category=HierarchyCategory(
            "census_population.resident_population", "Resident population", provider.id
        ),
        geography=geography,
        dimensions=(),
        target=HierarchyNode(name, f"County {code} resident population"),
    )


def cell_members(metadata: Mapping[str, Any]) -> list[int]:
    """The county codes one county child row covers (one unless combined)."""
    members = metadata.get(MEMBERS_KEY)
    if members:
        return [int(member) for member in str(members).split(",")]
    return [int(metadata["county_fips"])]


def is_broad_state_parent(spec) -> bool:
    """A state SOI row county facts can divide: all returns, no class."""

    metadata = spec.metadata
    return (
        spec.family == "irs_soi"
        and metadata.get("ledger_geography_level") == "state"
        and str(metadata.get("filing_status", "all")).lower() == "all"
        and float(metadata.get("agi_lower_bound", "-inf")) == -math.inf
        and float(metadata.get("agi_upper_bound", "inf")) == math.inf
        and metadata.get("soi_return_universe", "all_returns") == "all_returns"
        and metadata.get("itemized_only", "false").lower() != "true"
        and metadata.get("taxable_only", "false").lower() != "true"
        and spec.filter is None
        and all(
            str(metadata.get(key, "all")).lower() == "all"
            for key in ("ledger_filter_filing_status", "ledger_filter_income_range")
        )
    )


def validated_combined_cells(combined_cells):
    """Declared combined cells as ``{absent: host}`` codes, refusing bad ones."""

    cells = {int(absent): int(host) for absent, host in dict(combined_cells).items()}
    for absent, host in cells.items():
        if absent == host or absent // 1000 != host // 1000:
            raise ValueError(
                f"Declared SOI combined cell {absent:05d}->{host:05d} must join two "
                "distinct counties of one state."
            )
        if host in cells:
            raise ValueError(
                f"Declared SOI combined cell host {host:05d} is itself merged."
            )
    return cells


def _state_label(state):
    from microcosm.calibrate.geography_constants import US_STATE_FIPS_TO_NAME

    label = US_STATE_FIPS_TO_NAME.get(f"{int(state):02d}")
    if label is None:
        raise ValueError(f"State {int(state):02d} has no Census state name.")
    return label


def _acs_hierarchy(name, source_name, *, state=None, county=None):
    from microcosm.calibrate import (
        CalibrationHierarchy,
        HierarchyCategory,
        HierarchyGeography,
        HierarchyNode,
    )
    from microcosm.calibrate.provider_labels import calibration_provider_label

    provider = HierarchyNode(source_name, calibration_provider_label("us", source_name))
    if county is not None:
        # The same label every county family gives this county
        # (county_hierarchy).
        geography = HierarchyGeography(
            f"{COUNTY_PREFIX}{county}", f"County {county}", "county"
        )
        label = f"County {county} households (ACS)"
    else:
        state_label = _state_label(state)
        geography = HierarchyGeography(
            f"0400000US{int(state):02d}", state_label, "state"
        )
        label = f"{state_label} households (ACS)"
    return CalibrationHierarchy(
        provider=provider,
        category=HierarchyCategory("census_acs.households", "Households", provider.id),
        geography=geography,
        dimensions=(),
        target=HierarchyNode(name, label),
    )


def acs_households_parent_name(state):
    return f"acs_households_state_{int(state):02d}"


# ---------------------------------------------------------------------------
# State to county: resident population (issue row 8)
# ---------------------------------------------------------------------------

#: The quantity of the ladder population. The ladder is one source at two
#: levels, so its state row and its county rows have one measurement.
POPULATION_TARGET = "census_population.resident_population"


def county_population_rows(
    state_populations: Mapping[int, float],
    overlap_counties: np.ndarray,
    overlap_values: np.ndarray,
) -> dict[int, float]:
    """County populations from ladder overlaps, reconciled to state levels.

    Sums the ladder's overlap populations by county. The operator then
    divides each state's total among its counties. A state with no county, a
    county with no state, and a state whose counties sum to zero are refused
    by the operator.
    """

    state_level, county_level = us_pass_grains(COUNTY_POPULATION)
    raw = {
        int(code): controls.sum_of_lower_rows(overlap_values[overlap_counties == code])
        for code in np.unique(overlap_counties)
    }
    rows: list[SurfaceRow] = []
    legs: dict[str, str] = {}
    for state, target in sorted(state_populations.items()):
        if not math.isfinite(target) or target <= 0:
            raise ValueError(f"State {state} has no positive county/state population.")
        rows.append(
            SurfaceRow(
                state_level,
                geography_id(state_level, f"{int(state):02d}"),
                POPULATION_TARGET,
                float(target),
            )
        )
    codes = sorted(raw)
    for code in codes:
        area = geography_id(county_level, f"{code:05d}")
        legs[area] = geography_id(state_level, f"{code // 1000:02d}")
        rows.append(SurfaceRow(county_level, area, POPULATION_TARGET, raw[code]))
    outcomes, _receipt = run_pass(
        COUNTY_POPULATION, rows, bound_higher_targets=[POPULATION_TARGET], legs=legs
    )
    counties: dict[int, float] = {}
    for code, outcome in zip(codes, outcomes[len(rows) - len(codes) :], strict=True):
        if outcome.status != controls.RECONCILED:
            raise ValueError(f"County {code:05d} population has no state parent.")
        counties[code] = outcome.value
    return counties


# ---------------------------------------------------------------------------
# State to county: IRS SOI (issue row 6)
# ---------------------------------------------------------------------------

COUNTY_FILE = "county_file"


def county_soi_surface(
    state_specs: Sequence,
    facts: Iterable[Mapping[str, Any]],
    county_populations: Mapping[int, float],
    *,
    combined_cells: Mapping[int, int],
    factor_band: float,
    soi_mode: str,
) -> NestedSurface:
    """Divide the broad state SOI rows among the counties of the county facts.

    Every state must carry county return count and AGI with broad state
    parents. Other measures need both county facts and a state parent. The
    declared bridge ``state_soi_vs_county_file`` ties each state row to the
    county facts of its measure and period, and the operator divides it. A
    county absent from a block is refused unless ``combined_cells`` declares
    it with its host present: then the host's row stands for both counties,
    and ``cross_grain_members`` lists them.

    ``factor_band`` is the bound of the bridge, as a ratio to the median
    factor of a measure across states. A state whose factor is outside it is
    one where the state row and the county file disagree: the operator
    unbinds that leg, its county rows leave the surface, the receipt names
    it, and its state row stays.
    """

    nesting = us_county_soi_nesting()
    state_level, county_level = us_pass_grains(COUNTY_SOI)
    combined = validated_combined_cells(combined_cells)
    blocks = defaultdict(dict)
    for fact in facts:
        observed = fact.get("observed_measure", {})
        if observed.get("source_name") != "irs_soi":
            continue
        code = county_code_of(fact, source="SOI")
        dimensions = fact.get("dimensions", {})
        if any(
            str(dimensions.get(key, "all")).lower() != "all"
            for key in ("filing_status", "income_range")
        ):
            raise ValueError(
                "County SOI requires all-return facts without AGI classes."
            )
        measure = observed.get("source_measure_id")
        period = str(fact.get("period", {}).get("value", ""))
        if fact.get("period", {}).get("type") != "tax_year":
            raise ValueError("County SOI facts must carry a tax_year period.")
        key = (code // 1000, measure, period)
        if code in blocks[key]:
            raise ValueError(f"Duplicate county SOI fact for {key}, {code:05d}.")
        value = float(fact["value"])
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                f"County SOI shares require finite nonnegative values: {code:05d}."
            )
        blocks[key][code] = (value, fact)
    if not blocks:
        raise ValueError(
            "County facts carry no IRS SOI county rows; pass the pinned "
            "--county-facts file (us/chronicle_county_facts.json, microcosm#922)."
        )
    parents = {}
    for spec in state_specs:
        if not is_broad_state_parent(spec):
            continue
        metadata = spec.metadata
        state = int(metadata["state_fips"])
        key = (
            state,
            metadata.get("source_measure_id"),
            str(metadata.get("source_period", "")),
        )
        if key[1] not in nesting.required_measures and key[1] not in {
            block[1] for block in blocks
        }:
            continue
        if key in parents:
            raise ValueError(f"Two state SOI parents for county concept {key}.")
        parents[key] = spec
    if not parents:
        raise ValueError(
            "No county SOI concept has an already materialized broad state parent."
        )
    for state in sorted({int(code) // 1000 for code in county_populations}):
        for measure in nesting.required_measures:
            if not any(key[:2] == (state, measure) for key in parents):
                raise ValueError(
                    f"Missing broad state SOI parent state={state:02d}, concept={measure}; "
                    "state_county requires return count and AGI in every state."
                )

    # The surface: for each state row, the county facts of its measure.
    rows: list[SurfaceRow] = []
    legs: dict[str, str] = {}
    expected: dict[str, dict[str, frozenset[str]]] = {}
    bound: list[str] = []
    targets: dict[str, dict[str, str]] = {}
    merges = defaultdict(list)
    checked = []
    for key, parent in sorted(parents.items()):
        state, measure, period = key
        counties = {
            int(code) for code in county_populations if int(code) // 1000 == state
        }
        if not counties:
            raise ValueError(
                f"State {state:02d} has a broad SOI parent for {measure} and no "
                "county in the ladder. County facts, ladder and county column "
                "must use matching boundaries."
            )
        block = blocks.get(key, {})
        absent = counties - block.keys()
        merged = {
            code: combined[code]
            for code in absent
            if code in combined
            and combined[code] in block
            and combined[code] in counties
        }
        members = {code: [code] for code in sorted(block)}
        for code, host in sorted(merged.items()):
            members[host].append(code)
        codes = sorted(block)
        if codes and controls.sum_of_lower_rows(block[code][0] for code in codes) <= 0:
            raise ValueError(
                f"County SOI block {key} has no positive share denominator."
            )
        for code in codes:
            unit = block[code][1].get("observed_measure", {}).get("unit")
            if (
                parent.metadata.get("ledger_measure_unit") is not None
                and unit != parent.metadata["ledger_measure_unit"]
            ):
                raise ValueError(
                    f"County SOI units differ from state parent {parent.name}: {unit!r}."
                )
        attributes = {"source_measure_id": str(measure), "source_period": period}
        higher_id = f"{STATE_SURFACE}|{measure}|{period}"
        lower_id = f"{COUNTY_FILE}|{measure}|{period}"
        targets[higher_id] = {
            "source": STATE_SURFACE,
            "grain": state_level,
            **attributes,
        }
        targets[lower_id] = {"source": COUNTY_FILE, "grain": county_level, **attributes}
        state_area = geography_id(state_level, f"{state:02d}")
        bound.append(higher_id)
        checked.append(
            (key, parent, counties, block, members, merged, codes, len(rows))
        )
        rows.append(SurfaceRow(state_level, state_area, higher_id, float(parent.value)))
        for code in codes:
            area = geography_id(county_level, f"{code:05d}")
            legs[area] = state_area
            rows.append(SurfaceRow(county_level, area, lower_id, block[code][0]))
        expected.setdefault(lower_id, {})[state_area] = frozenset(
            geography_id(county_level, f"{code:05d}")
            for code in counties - merged.keys()
        )

    selector = bounded_selector(COUNTY_SOI, factor_band)
    bridges, _unpaired = expand_bridge_selectors([selector], targets)
    outcomes, operator_receipt = run_pass(
        COUNTY_SOI,
        rows,
        bound_higher_targets=bound,
        legs=legs,
        bridges=bridges,
        expected_members=expected,
    )
    specs, receipts = [], []
    unbound: list[dict] = []
    unbound_counties: set[int] = set()
    for key, parent, counties, block, members, merged, codes, start in checked:
        state, measure, period = key
        block_outcomes = outcomes[start + 1 : start + 1 + len(codes)]
        statuses = {outcome.status for outcome in block_outcomes}
        if statuses == {controls.UNBOUND}:
            # The state row and the county file disagree about this state:
            # its county rows of this measure leave the surface.
            unbound.append(
                {
                    "state_fips": f"{state:02d}",
                    "concept": measure,
                    "parent": parent.name,
                    "factor": block_outcomes[0].factor,
                    "counties": len(counties),
                }
            )
            unbound_counties |= counties
            continue
        if statuses != {controls.RECONCILED}:
            raise ValueError(
                f"The operator did not reconcile the county rows of {measure} "
                f"under {parent.name}."
            )
        for code, host in sorted(merged.items()):
            merges[(host, code)].append(f"{measure}|{period}")
        state_population = controls.sum_of_lower_rows(
            county_populations[code] for code in counties
        )
        for code, outcome in zip(codes, block_outcomes, strict=True):
            raw, fact = block[code]
            county = f"{code:05d}"
            cell = sorted(members[code])
            name = f"county_{county}__{parent.name}"
            metadata = {
                **parent.metadata,
                "ledger_geography_level": "county",
                "ledger_geography_id": f"{COUNTY_PREFIX}{county}",
                "ledger_geography_vintage": str(
                    fact["geography"].get("vintage", period)
                ),
                "ledger_layout_record_set_spec_id": str(
                    fact.get("layout", {}).get("record_set_spec_id", "county_soi")
                ),
                "county_fips": county,
                "county_label": str(fact["geography"].get("name", county)),
                **cross_grain_metadata(
                    value_before=raw,
                    factor=outcome.factor,
                    control=outcome.control,
                    parent=parent.name,
                    members=(
                        [f"{member:05d}" for member in cell] if len(cell) > 1 else ()
                    ),
                ),
                "county_population": str(
                    controls.sum_of_lower_rows(
                        county_populations[member] for member in cell
                    )
                ),
                "state_population": str(state_population),
            }
            specs.append(
                replace(
                    parent,
                    name=name,
                    measure=name,
                    value=outcome.value,
                    se=None,
                    metadata=metadata,
                    hierarchy=county_hierarchy(name, county, parent=parent),
                    source=f"{parent.source}; IRS SOI county {period} within-state shares",
                )
            )
        receipts.append(
            {
                "state_fips": f"{state:02d}",
                "concept": measure,
                "parent": parent.name,
                "counties": len(counties),
                "cells": len(codes),
                "target_sum": controls.sum_of_lower_rows(
                    outcome.value for outcome in block_outcomes
                ),
            }
        )
    # Counties without a state parent cannot silently disappear. The counties
    # of an unbound leg are named in the receipt.
    covered = {int(member) for spec in specs for member in cell_members(spec.metadata)}
    missing = sorted(set(county_populations) - covered - unbound_counties)
    if missing:
        raise ValueError(f"Counties have no materialized SOI state parent: {missing}.")
    receipt = {
        "soi_mode": soi_mode,
        "county_targets": len(specs),
        "concepts": sorted({key[1] for key in parents}),
        "bridge": bridge_receipt(COUNTY_SOI, selector, unbound),
        "blocks": receipts,
        "combined_cells": [
            {
                "members": sorted((f"{host:05d}", f"{absent:05d}")),
                "host": f"{host:05d}",
                "absent_from_soi": f"{absent:05d}",
                "concepts": sorted(concepts),
                "basis": nesting.combined_cell_basis,
                "source": "https://www.irs.gov/pub/irs-soi/22incydocguide.docx",
                "own_rows_kept": "ladder population, ACS households and "
                "ACS validation rows stay per county",
            }
            for (host, absent), concepts in sorted(merges.items())
        ],
        "new_engine_columns": 0,
        "basis": "state target x county fact / state sum of county facts",
    }
    return NestedSurface(
        specs=tuple(specs), receipt=receipt, operator_receipt=operator_receipt
    )


# ---------------------------------------------------------------------------
# County to state: ACS households (issue row 7)
# ---------------------------------------------------------------------------


def county_acs_household_surface(
    values: Mapping[int, tuple[float, float | None, str]],
    county_populations: Mapping[int, float],
    *,
    period: int,
    source_name: str,
    tables: str,
    family: str,
    measure: str,
) -> NestedSurface:
    """State households parents and their county children (trained, levels).

    ``values`` is ``{county: (estimate, margin of error or None, name)}`` for
    the households measure. No source in the feed publishes the state level,
    so the control of each state is derived: the sum of its counties'
    estimates. It is a new state target. The operator then divides it among
    the counties, so they sum to it. ACS margins of error ride in metadata.
    """
    from microcosm.calibrate import TargetSpec

    state_level, county_level = us_pass_grains(COUNTY_ACS_HOUSEHOLDS)
    lower_id = f"{source_name}|{measure}"
    higher_id = f"{lower_id}#sum"
    source = (
        f"US Census Bureau {tables} B11001_001E (households), county; "
        "state = sum of its counties"
    )
    rows: list[SurfaceRow] = []
    legs: dict[str, str] = {}
    checked = []
    for state in sorted({int(code) // 1000 for code in county_populations}):
        codes = sorted(code for code in values if code // 1000 == state)
        public = [values[code][0] for code in codes]
        total = controls.sum_of_lower_rows(public)
        if not codes or total <= 0:
            raise ValueError(
                f"ACS county households for state {state:02d} have no positive "
                "share denominator."
            )
        parent_name = acs_households_parent_name(state)
        parent_hierarchy = _acs_hierarchy(parent_name, source_name, state=state)
        state_area = geography_id(state_level, f"{state:02d}")
        checked.append((state, codes, total, parent_name, parent_hierarchy, len(rows)))
        rows.append(SurfaceRow(state_level, state_area, higher_id, total))
        for code, estimate in zip(codes, public, strict=True):
            area = geography_id(county_level, f"{code:05d}")
            legs[area] = state_area
            rows.append(SurfaceRow(county_level, area, lower_id, estimate))
    bridge = CrossGrainBridge(
        bridge_id=COUNTY_ACS_HOUSEHOLDS,
        concept=f"{source_name}.{measure}",
        higher_target_ids=(higher_id,),
        lower_side=lower_id,
    )
    outcomes, operator_receipt = run_pass(
        COUNTY_ACS_HOUSEHOLDS,
        rows,
        bound_higher_targets=[higher_id],
        legs=legs,
        bridges=[bridge],
    )
    specs, receipts = [], []
    for state, codes, total, parent_name, parent_hierarchy, start in checked:
        block_outcomes = outcomes[start + 1 : start + 1 + len(codes)]
        state_population = controls.sum_of_lower_rows(
            county_populations[code] for code in codes
        )
        specs.append(
            TargetSpec(
                name=parent_name,
                entity="household",
                measure=parent_name,
                value=total,
                period=period,
                source=source,
                family=family,
                metadata={
                    "geography_level": "state",
                    "state_fips": f"{state:02d}",
                    "source_measure_id": measure,
                    "acs_state_parent_basis": "sum of county estimates",
                    "household_universe": "excludes ACS group quarters (TYPEHUGQ 2, 3)",
                    "new_state_target": "true",
                },
                hierarchy=parent_hierarchy,
            )
        )
        for code, outcome in zip(codes, block_outcomes, strict=True):
            if outcome.status != controls.RECONCILED:
                raise ValueError(
                    f"The operator did not reconcile county {code:05d} under "
                    f"{parent_name}."
                )
            county = f"{code:05d}"
            name = f"county_{county}__{parent_name}"
            estimate, moe, label = values[code]
            metadata = {
                "geography_level": "county",
                "county_fips": county,
                "state_fips": f"{state:02d}",
                "source_measure_id": measure,
                **cross_grain_metadata(
                    value_before=estimate,
                    factor=outcome.factor,
                    control=outcome.control,
                    parent=parent_name,
                ),
                "county_population": str(county_populations[code]),
                "state_population": str(state_population),
                "household_universe": "excludes ACS group quarters (TYPEHUGQ 2, 3)",
            }
            if label:
                metadata["county_label"] = label
            if moe is not None:
                metadata["acs_margin_of_error"] = str(moe)
            specs.append(
                TargetSpec(
                    name=name,
                    entity="household",
                    measure=name,
                    value=outcome.value,
                    period=period,
                    source=f"US Census Bureau {tables} B11001_001E "
                    "(households), county share of the state sum",
                    family=family,
                    metadata=metadata,
                    hierarchy=_acs_hierarchy(name, source_name, county=county),
                )
            )
        receipts.append(
            {
                "state_fips": f"{state:02d}",
                "state_target": parent_name,
                "state_households": total,
                "counties": len(codes),
                "target_sum": controls.sum_of_lower_rows(
                    outcome.value for outcome in block_outcomes
                ),
                "counties_with_moe": sum(values[code][1] is not None for code in codes),
            }
        )
    receipt = {
        "family": family,
        "source_measure_id": measure,
        "source": f"{tables} county B11001",
        "bridge": bridge_receipt(COUNTY_ACS_HOUSEHOLDS, bridge),
        "new_state_targets": len(receipts),
        "county_targets": len(specs) - len(receipts),
        "note": (
            "New state targets: each state's households target is the sum "
            "of its counties' ACS 5-year B11001_001E estimates, which moves "
            "state household counts to ACS 5-year levels (the county pilot "
            "found the release about 6% under ACS households). County rows "
            "split those state targets; group-quarters records count as "
            "no household."
        ),
        "blocks": receipts,
        "new_engine_columns": 0,
    }
    return NestedSurface(
        specs=tuple(specs), receipt=receipt, operator_receipt=operator_receipt
    )
