"""US target nesting on the shared cross-grain operator (microcosm#1760).

A target for a small area must agree with the target of the area that
contains it. This module makes the rows of a lower geography level from a
control at a higher level, for each mechanism that
``us/cross_grain_declarations.json`` declares. The arithmetic is not here:

- :func:`microcosm.build.cross_grain.apply_cross_grain_reconciliation` groups
  the rows of one quantity, finds the control of each lower leg, divides it
  among the leg's rows with the mechanism's declared allocator, checks the sum
  and writes its receipt;
- :mod:`microcosm.build.cross_grain_controls` supplies the allocators, the
  derived controls, the factor band, the completeness check and the
  self-consistency check.

What is here is the US side of each mechanism: which specs or facts are the
control and the lower rows, the refusals with their US wording, and the
metadata keys and receipts the local tools wrote before the operator made the
rows. Stage A changes no value: every surface has the bytes it had.

The module reads specs and facts only. It reads no frame and no household.
"""

from __future__ import annotations

import json
import math
import struct
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build import cross_grain_controls as controls
from microcosm.build.cross_grain import (
    CrossGrainRule,
    apply_cross_grain_reconciliation,
)
from microcosm.build.us_runtime.cross_grain_declarations import (
    DISTRICT_SOI,
    load_us_cross_grain_declarations,
    us_cross_grain_mechanism,
    us_district_soi_nesting,
)

#: The file a materialize stage writes the operator's own receipts to.
NESTING_RECEIPTS_FILENAME = "target_nesting.json"

# ---------------------------------------------------------------------------
# Blocks through the operator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NestingBlock:
    """One control and the lower rows it is divided among.

    ``concept`` and ``filters`` are the measurement signature that ties the
    control to its rows. ``lower_values`` are in the order the allocator must
    see them: a residual rule gives its residual to the first of equal
    largest rows.
    """

    concept: str
    filters: tuple[tuple[str, str], ...]
    control_geography_id: str
    control_value: float
    lower_geography_ids: tuple[str, ...]
    lower_values: tuple[float, ...]


@dataclass(frozen=True)
class NestedRows:
    """A block's rows as the operator left them, with its leg receipt."""

    values: tuple[float, ...]
    #: The factor and the lower-row total the allocator reported.
    factor: float
    lower_sum: float


def geography_id(grain: str, code: str) -> str:
    """A surface geography id that cannot collide across grains."""

    return f"{grain}:{code}"


def _signature_key(block: NestingBlock) -> str:
    return "|".join(
        [block.concept, *(f"{name}={value}" for name, value in block.filters)]
    )


def reconcile_blocks(
    mechanism_id: str, blocks: Sequence[NestingBlock]
) -> tuple[list[NestedRows], dict[str, Any]]:
    """Divide each block's control among its rows on the shared operator.

    Builds one surface for the mechanism (a control row and the lower rows of
    every block), runs :func:`apply_cross_grain_reconciliation` with the
    mechanism's declared allocator, and returns each block's rows, in block
    order, with the operator's receipt. A block is one leg: its control row
    parents exactly its own lower rows.

    Raises:
        ValueError: If two blocks share a signature and a control geography,
            if a lower row sits under two controls, if a block has no lower
            row, or if the operator refuses a block (a vanishing or
            non-finite total, a negative factor, or rows off their control).
    """

    declarations = load_us_cross_grain_declarations()
    mechanism = us_cross_grain_mechanism(mechanism_id)
    higher, lower = str(mechanism["higher_grain"]), str(mechanism["lower_grain"])
    rows: list[tuple[str, str, str, float]] = []
    signatures: dict[str, dict[str, Any]] = {}
    bound: list[str] = []
    area_legs: dict[str, str] = {}
    spans: list[tuple[str, int, int]] = []
    controls_seen: set[tuple[str, str]] = set()
    for block in blocks:
        key = _signature_key(block)
        control_id = f"{mechanism_id}:{higher}:{key}"
        lower_id = f"{mechanism_id}:{lower}:{key}"
        if (control_id, block.control_geography_id) in controls_seen:
            raise ValueError(
                f"Nesting mechanism {mechanism_id!r} has two blocks for {key!r} "
                f"at {block.control_geography_id!r}."
            )
        controls_seen.add((control_id, block.control_geography_id))
        if not block.lower_geography_ids or len(block.lower_geography_ids) != len(
            block.lower_values
        ):
            raise ValueError(
                f"Nesting mechanism {mechanism_id!r} block {key!r} at "
                f"{block.control_geography_id!r} needs one value for each of "
                "its lower rows, and at least one row."
            )
        if control_id not in signatures:
            measurement = {
                "concept": block.concept,
                "filters": [{name: value} for name, value in block.filters],
            }
            signatures[control_id] = {"measurement": measurement}
            signatures[lower_id] = {"measurement": measurement}
            bound.append(control_id)
        rows.append(
            (higher, block.control_geography_id, control_id, block.control_value)
        )
        start = len(rows)
        for area, value in zip(
            block.lower_geography_ids, block.lower_values, strict=True
        ):
            if area_legs.setdefault(area, block.control_geography_id) != (
                block.control_geography_id
            ):
                raise ValueError(
                    f"Nesting mechanism {mechanism_id!r}: lower geography "
                    f"{area!r} sits under two controls."
                )
            rows.append((lower, area, lower_id, float(value)))
        spans.append((control_id, start, len(rows)))
    if not blocks:
        receipt: dict[str, Any] = {
            "absence": f"Mechanism {mechanism_id} had no block to reconcile."
        }
        _record(mechanism_id, 0, receipt)
        return [], receipt
    surface = pd.DataFrame(
        rows, columns=["grain", "geography_id", "target_id", "value"]
    )
    rule = CrossGrainRule(
        grain_precedence=tuple(declarations["grain_precedence"]),
        signature_fields=tuple(declarations["signature_fields"]),
        bridges=(),
        leg_of_area=area_legs.__getitem__,
        parent_geography_legs={
            control_geography: (control_geography,)
            for _control_id, control_geography in sorted(controls_seen)
        },
        control_grains=tuple(declarations["control_grains"]),
        allocator=controls.allocator_named(str(mechanism["allocator"])),
    )
    reconciled, receipt = apply_cross_grain_reconciliation(
        surface, bound, signatures, rule
    )
    values = reconciled["value"].to_numpy(dtype=np.float64)
    legs = {
        (leg["higher_target_ids"][0], leg["parent_geography_id"]): leg
        for group in receipt["groups"]
        for leg in group["legs"]
    }
    nested: list[NestedRows] = []
    for block, (control_id, start, stop) in zip(blocks, spans, strict=True):
        leg = legs.get((control_id, block.control_geography_id))
        if leg is None:
            raise ValueError(
                f"Nesting mechanism {mechanism_id!r}: the operator did not "
                f"reconcile the block at {block.control_geography_id!r}."
            )
        nested.append(
            NestedRows(
                values=tuple(values[start:stop].tolist()),
                factor=leg["declared_factor"],
                lower_sum=leg["old_total"],
            )
        )
    _record(mechanism_id, len(blocks), receipt)
    return nested, receipt


# ---------------------------------------------------------------------------
# The operator's own receipts
# ---------------------------------------------------------------------------

_RECEIPTS: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "us_target_nesting_receipts", default=None
)


def start_nesting_receipts() -> list[dict[str, Any]]:
    """Start a fresh record of the nesting passes that run from here on.

    The four tool functions keep the signatures and the receipts they had, so
    the operator's own receipt cannot travel in their results. A stage that
    wants it calls this first and writes the list with
    :func:`write_nesting_receipts` when its targets are made. Without a call,
    nothing is recorded.
    """

    receipts: list[dict[str, Any]] = []
    _RECEIPTS.set(receipts)
    return receipts


def stop_nesting_receipts() -> None:
    """End the record that :func:`start_nesting_receipts` started."""

    _RECEIPTS.set(None)


def _record(mechanism_id: str, blocks: int, receipt: Mapping[str, Any]) -> None:
    receipts = _RECEIPTS.get()
    if receipts is not None:
        mechanism = us_cross_grain_mechanism(mechanism_id)
        receipts.append(
            {
                "mechanism": mechanism_id,
                "issue_row": mechanism["issue_row"],
                "allocator": mechanism["allocator"],
                "blocks": blocks,
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
# State to congressional district: IRS SOI shares (issue row 5)
# ---------------------------------------------------------------------------

#: The filters of a district SOI signature, in the order the caller's concept
#: identity gives them after the measure id.
DISTRICT_SOI_IDENTITY_FILTERS = (
    "agi_lower_bound",
    "agi_upper_bound",
    "filing_status",
    "eitc_child_count",
)


@dataclass(frozen=True)
class NestedSurface:
    """The specs of one nested surface, its receipt and the operator's."""

    specs: tuple
    receipt: dict
    operator_receipt: dict


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
    its state: the operator divides the Historic Table 2 state row among the
    district rows (``proportional_fsum``). A concept only the district file
    has keeps the district file's own state row as its control, lifted onto
    the Historic Table 2 basis by a sibling concept's ratio
    (``scaled_by_sibling_ratio``). A block whose ratio sits outside the
    factor band of its concept's median across states is dropped and
    recorded.

    Args:
        soi_specs: Every compiled ``irs_soi`` spec (the ``full`` surface).
        state_surface_predicate: The ``state`` mode's predicate; its specs
            (Historic Table 2) are the state surface.
        crosswalk: The 117th->119th CD crosswalk the compiler used; it names
            each state's source-plan and current-plan districts.
        crosswalk_sha256: Recorded in the receipt.
        level_bridges: District-file-only measure -> sibling measure.
        factor_band: Largest allowed ratio between a block's ratio and its
            concept's median across states (either direction).
        concept_identity: The published concept a spec measures, independent
            of vintage: the measure id, then the filters of
            ``DISTRICT_SOI_IDENTITY_FILTERS``.
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
        ValueError: If a district concept is incomplete in a state, if its
            district rows disagree on what they materialize, if a parent's
            materialization differs from its children's, or if a parent
            cannot absorb its children (zero sum under a nonzero parent, or
            a sign conflict).
    """

    nesting = us_district_soi_nesting()
    mechanism = us_cross_grain_mechanism(DISTRICT_SOI)
    state_level = str(mechanism["higher_grain"])
    district_level = str(mechanism["lower_grain"])
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

    # Each state concept both files carry: its Historic Table 2 / district-file
    # ratio, judged against the measure's median ratio across states.
    vintage_ratio = {
        key: controls.level_ratio(
            ht2_by_key[key].value, cd_file_state_by_key[key].value
        )
        for key in ht2_by_key.keys() & cd_file_state_by_key.keys()
    }
    median_ratio = controls.median_by_group(vintage_ratio, lambda key: key[1])
    out_of_band: list[dict] = []

    def within_band(
        key: tuple,
        ratio: float,
        median: float,
        basis: str,
        measure: str,
        *,
        verdict: str,
    ) -> bool:
        judged = controls.band_verdict(ratio, median, factor_band)
        if not judged.ok:
            out_of_band.append(
                {
                    "state_fips": key[0],
                    "measure": measure,
                    "concept": key[1][0],
                    "basis": basis,
                    "verdict": verdict,
                    "ratio": ratio,
                    "median_ratio": median,
                    "relative": judged.relative,
                }
            )
        return judged.ok

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

    # Each Historic-Table-2-parented block's rebase factor (parent over its
    # district rows' sum), judged against its concept's median across states.
    block_factor: dict[tuple, float] = {}
    for key, block in kept_cd_by_key.items():
        parent = ht2_by_key.get(key)
        if parent is None:
            continue
        ratio, _child_sum = controls.control_ratio(
            parent.value, (spec.value for spec in block)
        )
        if ratio is not None:
            block_factor[key] = ratio
    median_factor = controls.median_by_group(
        block_factor, lambda key: key[1], positive_only=True
    )

    # One basis for every state level: a concept Historic Table 2 lacks keeps
    # the district file's state row, scaled by a sibling's two levels.
    bridged: dict[str, object] = {}
    bridge_factors: dict[str, list[float]] = defaultdict(list)
    for key, spec in sorted(cd_file_state_by_key.items()):
        state, identity = key
        measure = identity[0]
        if key in ht2_by_key or measure in defective:
            continue
        sibling = level_bridges.get(measure)
        if sibling is None:
            raise ValueError(
                f"District-file-only concept {measure!r} has no level bridge; "
                "register a sibling measure in STATE_CD_LEVEL_BRIDGES before "
                "binding it."
            )
        sibling_key = (state, (sibling, *identity[1:]))
        ht2_sibling = ht2_by_key.get(sibling_key)
        cd_sibling = cd_file_state_by_key.get(sibling_key)
        if ht2_sibling is None or cd_sibling is None or float(cd_sibling.value) == 0:
            raise ValueError(
                f"Level bridge {measure} -> {sibling} in state {state} needs "
                "the sibling's state total in both files."
            )
        value, factor = controls.scaled_by_sibling_ratio(
            spec.value, ht2_sibling.value, cd_sibling.value
        )
        # A sign conflict is a data defect, not a level disagreement: refuse
        # it before the band, which only judges finite positive ratios.
        if not (math.isfinite(factor) and factor > 0.0):
            raise ValueError(
                f"Level bridge {measure} -> {sibling} in state {state} has "
                f"factor {factor}."
            )
        # The sibling is trusted for bridging only if both of the band's
        # verdicts on it pass: its two state levels, and (where it has district
        # rows) the rebase of its own district block.
        trusted = within_band(
            sibling_key,
            factor,
            median_ratio.get(sibling_key[1], math.nan),
            f"level_bridge:{measure}",
            measure,
            verdict="sibling_vintage_ratio",
        )
        if sibling_key in block_factor:
            trusted = (
                within_band(
                    sibling_key,
                    block_factor[sibling_key],
                    median_factor.get(sibling_key[1], math.nan),
                    f"level_bridge:{measure}",
                    measure,
                    verdict="sibling_rebase_factor",
                )
                and trusted
            )
        if not trusted:
            continue
        metadata = dict(spec.metadata)
        metadata.update(
            {
                "state_cd_vintage_rule": nesting.vintage_rule,
                "state_cd_cd_file_value": repr(float(spec.value)),
                "state_cd_level_bridge_sibling": ht2_sibling.name,
                "state_cd_level_bridge_cd_file_sibling": cd_sibling.name,
                "state_cd_level_bridge_factor": repr(factor),
            }
        )
        bridged[spec.name] = replace(
            spec, value=value, signed=value < 0.0, metadata=metadata
        )
        bridge_factors[measure].append(factor)

    # Every check a block must pass before the operator divides its control.
    internal_gaps: list[float] = []
    kept_cd_file_parents: set[str] = set()
    kept_blocks: list[tuple] = []
    for key in sorted(kept_cd_by_key):
        state, identity = key
        children = sorted(
            kept_cd_by_key[key],
            key=lambda spec: spec.metadata["congressional_district_geoid"],
        )
        districts = [spec.metadata["congressional_district_geoid"] for spec in children]
        expected = sorted(target_districts.get(state, ()))
        if not controls.leg_completeness(districts, expected).complete:
            raise ValueError(
                f"District concept {identity} in state {state} covers "
                f"{districts}, expected the current plan's {expected}; "
                "rebasing an incomplete set would misstate every district."
            )
        semantics = {materializer_semantics(spec) for spec in children}
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
            basis = "cd_file_state_total_bridged"
            if parent is None:
                raise ValueError(
                    f"District concept {identity} in state {state} has no "
                    "state parent in either vintage."
                )
            if parent.name not in bridged:
                for child in children:
                    drop(child, f"level_bridge_out_of_band:{identity[0]}")
                continue
            parent = bridged[parent.name]
            kept_cd_file_parents.add(parent.name)
        if materializer_semantics(parent) != child_semantics:
            raise ValueError(
                f"State parent {parent.name} materializes differently from "
                f"its district rows ({identity}, state {state})."
            )
        parent_value = float(parent.value)
        ratio, child_sum = controls.control_ratio(
            parent_value, (spec.value for spec in children)
        )
        if ratio is None:
            if parent_value != 0.0:
                raise ValueError(
                    f"District rows of {identity} in state {state} sum to zero "
                    f"under a nonzero parent {parent.name}={parent_value}."
                )
            factor = 1.0
        else:
            factor = ratio
            if factor < 0.0:
                raise ValueError(
                    f"District rows of {identity} in state {state} sum to "
                    f"{child_sum}, opposite in sign to parent {parent.name}="
                    f"{parent_value}."
                )
        # The district file must agree with itself before its shares are used.
        cd_state = cd_file_state_by_key.get(key)
        if cd_state is not None:
            cd_state_value = float(cd_state.value)
            gap = controls.sum_gap(cd_state_value, child_sum)
            internal_gaps.append(gap.relative)
            if gap.exceeds(nesting.internal_rtol):
                raise ValueError(
                    f"District-file state row {cd_state.name}={cd_state_value} "
                    f"differs from the sum of its district rows ({child_sum}) "
                    f"for {identity} in state {state}: a column or crosswalk "
                    "defect inside the district file."
                )
        # An all-zero block agrees with a zero parent whatever the median.
        if (
            basis == "historic_table_2"
            and ratio is not None
            and not within_band(
                key,
                factor,
                median_factor.get(identity, math.nan),
                "rebase",
                identity[0],
                verdict="rebase_factor",
            )
        ):
            for child in children:
                drop(child, f"rebase_out_of_band:{identity[0]}")
            continue
        kept_blocks.append(
            (key, parent, basis, parent_value, children, child_sum, factor)
        )

    # The operator divides each control among its district rows.
    nested, operator_receipt = reconcile_blocks(
        DISTRICT_SOI,
        [
            NestingBlock(
                concept=f"{children[0].family}.{identity[0]}",
                filters=tuple(
                    zip(DISTRICT_SOI_IDENTITY_FILTERS, identity[1:], strict=True)
                ),
                control_geography_id=geography_id(state_level, state),
                control_value=parent_value,
                lower_geography_ids=tuple(
                    geography_id(
                        district_level, child.metadata["congressional_district_geoid"]
                    )
                    for child in children
                ),
                lower_values=tuple(float(child.value) for child in children),
            )
            for (state, identity), _p, _b, parent_value, children, _s, _f in kept_blocks
        ],
    )
    rebased: dict[str, object] = {}
    factors_by_measure: dict[str, list[float]] = defaultdict(list)
    parent_basis_counts: Counter = Counter()
    for (key, parent, basis, parent_value, children, child_sum, factor), rows in zip(
        kept_blocks, nested, strict=True
    ):
        _state, identity = key
        if _bits(rows.lower_sum, rows.factor) != _bits(child_sum, factor):
            raise ValueError(
                f"The operator's factor for {identity} under {parent.name} "
                f"({rows.factor!r} on {rows.lower_sum!r}) is not the factor the "
                f"band judged ({factor!r} on {child_sum!r})."
            )
        factors_by_measure[identity[0]].append(rows.factor)
        parent_basis_counts[basis] += len(children)
        for child, value in zip(children, rows.values, strict=True):
            metadata = dict(child.metadata)
            metadata.update(
                {
                    "state_cd_vintage_rule": nesting.vintage_rule,
                    "state_cd_parent_target_name": parent.name,
                    "state_cd_parent_basis": basis,
                    "state_cd_parent_value": repr(parent_value),
                    "state_cd_cd_file_value": repr(float(child.value)),
                    "state_cd_cd_file_state_sum": repr(rows.lower_sum),
                    "state_cd_rebase_factor": repr(rows.factor),
                }
            )
            rebased[child.name] = replace(
                child,
                value=value,
                signed=value < 0.0,
                metadata=metadata,
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
            "keep only their within-state shares and are rebased to sum to "
            "that parent. Concepts only the district file carries keep its "
            "own state row as their parent, lifted onto the Historic Table 2 "
            "basis by a sibling concept's ratio of the two files' state "
            "totals in the same state (STATE_CD_LEVEL_BRIDGES)."
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
        "internal_consistency": {
            "rtol": nesting.internal_rtol,
            "blocks_checked": len(internal_gaps),
            "max_relative_gap": max(internal_gaps, default=0.0),
        },
        "factor_band": {
            "tolerance": factor_band,
            "relative_to": "the measure's median Historic Table 2 / "
            "district-file ratio across states",
            "out_of_band": sorted(
                out_of_band, key=lambda row: (row["concept"], row["state_fips"])
            ),
        },
        "level_bridges": dict(level_bridges),
        "level_bridge_factor_by_measure": {
            measure: _factor_summary(values)
            for measure, values in sorted(bridge_factors.items())
        },
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
                "IRS SOI tables are administrative. target_roles.json records "
                "sigma and sigma_basis for every target, null wherever a spec "
                "carries none."
            ),
        },
    }
    return NestedSurface(
        specs=specs, receipt=receipt, operator_receipt=operator_receipt
    )


def district_block_closure(specs: Sequence, *, rtol: float) -> list[dict]:
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
        child_sum = controls.sum_of_lower_rows(spec.value for spec in children)
        parent_value = None if parent is None else float(parent.value)
        report.append(
            {
                "parent": parent_name,
                "parent_bound": parent is not None,
                "parent_value": parent_value,
                "district_sum": child_sum,
                "n_districts": len(children),
                "ok": parent is not None
                and controls.sum_gap(parent_value, child_sum).within(rtol),
            }
        )
    return report


def _bits(*values: float) -> bytes:
    return b"".join(struct.pack("<d", float(value)) for value in values)
