"""Verbatim copies of the US share code before microcosm#1760 moved it.

``tools/us_acs_local_cd_surface.py`` and ``tools/us_acs_local_county_surface.py``
held this code at ``e29843dc7``: ``state_cd_soi_surface`` (the district SOI
shares), ``with_county_populations`` (the county population rows),
``state_county_soi_surface`` (the county SOI shares) and
``acs_household_surface`` (the county household rows), with the constants and
helpers they read. The arithmetic now runs on the shared cross-grain operator
(``microcosm.build.cross_grain``), through
``microcosm.build.us_runtime.target_nesting``. The equivalence tests hold the
tool functions to these copies. Nothing changed: each definition is the
tool's source text, in the tool's order, district file first.

``_release_tool`` imports the national release tool by name, as the district
tool does, so ``tools/`` must be on ``sys.path`` (loading the local release
tool puts it there).
"""

# ruff: noqa

from __future__ import annotations

import dataclasses
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd


CD_FILE_RECORD_SET_SPEC_PREFIX = "irs_soi.congressional_district_"


STATE_CD_VINTAGE_RULE = "state_cd.historic_table_2_level_cd_file_shares.v1"


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


STATE_CD_UNPARENTED_CD_MEASURES: Mapping[str, str] = {
    "tax_filer_individual_count": (
        "The compiler drops this measure at state and national geography "
        "(fiscal_targets._soi_reference_from_fact), so no state target exists "
        "for its district rows to reconcile to."
    ),
}


STATE_CD_LEVEL_BRIDGES: Mapping[str, str] = {
    "charitable_amount": "itemized_deductions_amount",
    "charitable_returns": "return_count",
    "interest_paid_deduction_amount": "itemized_deductions_amount",
    "interest_paid_deduction_returns": "return_count",
    "qualified_business_income_deduction_amount": "adjusted_gross_income",
    "qualified_business_income_deduction_returns": "return_count",
}


STATE_CD_EXCLUDED_CD_STATES: Mapping[str, str] = {}


STATE_CD_FACTOR_BAND = 1.25


STATE_CD_INTERNAL_RTOL = 1e-6


RECONCILIATION_RTOL = 1e-9


GEOGRAPHY_STATE = "state"


GEOGRAPHY_CD = "congressional_district"


def _release_tool():
    """The sibling release tool (``tools/`` is on ``sys.path`` via the ACS tool)."""

    import build_us_fiscal_refresh_release as release_tool

    return release_tool


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
    level_bridges: Mapping[str, str] = STATE_CD_LEVEL_BRIDGES,
    factor_band: float = STATE_CD_FACTOR_BAND,
) -> StateCdSurface:
    """Select and reconcile the ``state_cd`` SOI surface.

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

    # Each state concept both files carry: its Historic Table 2 / district-file
    # ratio, judged against the measure's median ratio across states.
    vintage_ratio: dict[tuple, float] = {}
    for key in ht2_by_key.keys() & cd_file_state_by_key.keys():
        ht2_value = float(ht2_by_key[key].value)
        cd_value = float(cd_file_state_by_key[key].value)
        vintage_ratio[key] = (
            ht2_value / cd_value
            if cd_value != 0.0 and (ht2_value > 0) == (cd_value > 0)
            else math.nan
        )
    ratios_by_identity: dict[tuple, list[float]] = defaultdict(list)
    for (_state, identity), ratio in vintage_ratio.items():
        if math.isfinite(ratio):
            ratios_by_identity[identity].append(ratio)
    median_ratio = {
        identity: float(np.median(values))
        for identity, values in ratios_by_identity.items()
    }
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
        relative = ratio / median if math.isfinite(ratio) and median else math.nan
        ok = math.isfinite(relative) and 1 / factor_band <= relative <= factor_band
        if not ok:
            out_of_band.append(
                {
                    "state_fips": key[0],
                    "measure": measure,
                    "concept": key[1][0],
                    "basis": basis,
                    "verdict": verdict,
                    "ratio": ratio,
                    "median_ratio": median,
                    "relative": relative,
                }
            )
        return ok

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

    # Each Historic-Table-2-parented block's rebase factor (parent over its
    # district rows' sum), judged against its concept's median across states.
    block_factor: dict[tuple, float] = {}
    for key, block in kept_cd_by_key.items():
        parent = ht2_by_key.get(key)
        child_sum = math.fsum(float(spec.value) for spec in block)
        if parent is not None and child_sum != 0.0:
            block_factor[key] = float(parent.value) / child_sum
    factors_by_identity: dict[tuple, list[float]] = defaultdict(list)
    for (_state, identity), factor in block_factor.items():
        if math.isfinite(factor) and factor > 0:
            factors_by_identity[identity].append(factor)
    median_factor = {
        identity: float(np.median(values))
        for identity, values in factors_by_identity.items()
    }

    # One basis for every state level: a concept Historic Table 2 lacks keeps
    # the district file's state row, scaled by a sibling's two levels.
    bridged: dict[str, object] = {}
    bridge_factors: dict[str, list[float]] = defaultdict(list)
    for key, spec in sorted(cd_file_state_by_key.items()):
        state, identity = key
        measure = identity[0]
        if key in ht2_by_key or measure in STATE_CD_DEFECTIVE_CD_FILE_MEASURES:
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
        factor = float(ht2_sibling.value) / float(cd_sibling.value)
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
                "state_cd_vintage_rule": STATE_CD_VINTAGE_RULE,
                "state_cd_cd_file_value": repr(float(spec.value)),
                "state_cd_level_bridge_sibling": ht2_sibling.name,
                "state_cd_level_bridge_cd_file_sibling": cd_sibling.name,
                "state_cd_level_bridge_factor": repr(factor),
            }
        )
        value = float(spec.value) * factor
        bridged[spec.name] = replace(
            spec, value=value, signed=value < 0.0, metadata=metadata
        )
        bridge_factors[measure].append(factor)

    rebased: dict[str, object] = {}
    internal_gaps: list[float] = []
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
        # The district file must agree with itself before its shares are used.
        cd_state = cd_file_state_by_key.get(key)
        if cd_state is not None:
            cd_state_value = float(cd_state.value)
            gap = abs(cd_state_value - child_sum)
            internal_gaps.append(gap / max(abs(cd_state_value), 1.0))
            if gap > STATE_CD_INTERNAL_RTOL * max(abs(cd_state_value), 1.0):
                raise ValueError(
                    f"District-file state row {cd_state.name}={cd_state_value} "
                    f"differs from the sum of its district rows ({child_sum}) "
                    f"for {identity} in state {state}: a column or crosswalk "
                    "defect inside the district file."
                )
        # An all-zero block agrees with a zero parent whatever the median.
        if (
            basis == "historic_table_2"
            and child_sum != 0.0
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
    unreconciled = [
        block for block in state_parent_reconciliation(specs) if not block["ok"]
    ]
    if unreconciled:
        raise ValueError(
            f"{len(unreconciled)} district block(s) do not add up to a bound "
            f"state parent, e.g. {unreconciled[:2]}."
        )
    with_sigma = sum(1 for spec in specs if getattr(spec, "se", None) is not None)
    receipt = {
        "vintage_rule": STATE_CD_VINTAGE_RULE,
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
        "defective_cd_file_measures": dict(STATE_CD_DEFECTIVE_CD_FILE_MEASURES),
        "unparented_cd_measures": dict(STATE_CD_UNPARENTED_CD_MEASURES),
        "excluded_cd_states": dict(STATE_CD_EXCLUDED_CD_STATES),
        "at_large_on_source_plan": at_large_source_states,
        "internal_consistency": {
            "rtol": STATE_CD_INTERNAL_RTOL,
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


COUNTY_COLUMN = "county_fips"


SOI_MODE = "state_county"


COUNTY_PREFIX = "0500000US"


COUNTY_SOI_REQUIRED_MEASURES = ("return_count", "adjusted_gross_income")


PUERTO_RICO_STATE = 72


DECLARED_SOI_COMBINED_CELLS = {15005: 15009}


SOI_COMBINED_CELL_BASIS = (
    "IRS SOI 2022 county users guide (22incydocguide.docx): items with fewer "
    "than 20 returns for a county were excluded; 22incyallnoagi.csv has no row "
    "for this county, so its SOI value is missing, not zero, and it is "
    "merged into its declared host county for the SOI families only."
)


def _codes(values, *, column: str) -> np.ndarray:
    numeric = pd.to_numeric(pd.Series(values), errors="raise").to_numpy(float)
    if not np.isfinite(numeric).all() or (numeric != np.floor(numeric)).any():
        raise ValueError(f"{column} must contain finite integer geography codes.")
    return numeric.astype(np.int64)


COUNTY_COLUMN_2023 = "county_fips_2023"


def ladder_county_overlaps(ladder, county_column=COUNTY_COLUMN):
    """``(pumas, counties, populations)`` on the boundaries of ``county_column``.

    The PUMA ladder records ``(PUMA, county)`` overlaps on 2020 counties, which
    name Connecticut's eight legacy counties. For ``county_fips_2023`` the
    overlaps are the ladder's ``(PUMA, tract)`` overlaps regrouped by each
    tract's 2023 county-equivalent
    (:func:`microcosm.build.us_runtime.county_vintage.county_fips_2023_from_tracts`):
    a Connecticut tract takes its planning region and every other tract keeps
    its county, so each PUMA's population is conserved and every county
    outside Connecticut keeps its 2020 overlaps. Any other column reads the
    2020 overlaps unchanged.
    """
    if county_column != COUNTY_COLUMN_2023:
        return (
            np.asarray(ladder.county_overlap_puma),
            np.asarray(ladder.county_overlap_county),
            np.asarray(ladder.county_overlap_population, dtype=float),
        )
    from microcosm.build.us_runtime.county_vintage import (
        county_fips_2023_from_tracts,
    )

    tracts = np.asarray(ladder.tract_overlap_tract)
    pumas = _codes(ladder.tract_overlap_puma, column="tract_overlap_puma")
    values = np.asarray(ladder.tract_overlap_population, dtype=float)
    if not len(tracts) or len(tracts) != len(pumas) or len(tracts) != len(values):
        raise ValueError(
            f"{COUNTY_COLUMN_2023} needs the ladder's tract overlaps, and this "
            "ladder has none or misaligned ones."
        )
    counties = _codes(county_fips_2023_from_tracts(tracts), column=COUNTY_COLUMN_2023)
    cells = pd.DataFrame({"puma": pumas, "county": counties, "population": values})
    grouped = cells.groupby(["puma", "county"], sort=True)["population"].sum()
    return (
        grouped.index.get_level_values("puma").to_numpy(np.int64),
        grouped.index.get_level_values("county").to_numpy(np.int64),
        grouped.to_numpy(float),
    )


def with_county_populations(
    ladder, populations, geographies, *, county_column=COUNTY_COLUMN
):
    """Aggregate ladder overlaps, then reconcile counties to state levels."""
    if "county" not in geographies:
        return populations
    result = {key: dict(value) for key, value in populations.items()}
    if "state" not in result:
        states = np.asarray(ladder.puma) // 100_000
        result["state"] = {
            int(state): float(np.asarray(ladder.puma_population)[states == state].sum())
            for state in np.unique(states)
        }
    if county_column == COUNTY_COLUMN_2023:
        _pumas, overlap_counties, overlap_values = ladder_county_overlaps(
            ladder, county_column
        )
    else:
        overlap_counties = ladder.county_overlap_county
        overlap_values = ladder.county_overlap_population
    counties = _codes(overlap_counties, column="county_overlap_county")
    values = np.asarray(overlap_values, dtype=float)
    if (
        len(counties) != len(values)
        or not np.isfinite(values).all()
        or (values < 0).any()
    ):
        raise ValueError("County overlap populations must be finite and nonnegative.")
    raw = {
        int(code): math.fsum(values[counties == code]) for code in np.unique(counties)
    }
    result["county"] = {}
    for state, target in sorted(result["state"].items()):
        block = sorted(code for code in raw if code // 1000 == int(state))
        total = math.fsum(raw[code] for code in block)
        if not block or total <= 0 or not math.isfinite(target) or target <= 0:
            raise ValueError(f"State {state} has no positive county/state population.")
        scaled = [float(target) * raw[code] / total for code in block]
        # Assign the rounding residual to the largest cell, never a small county.
        largest = max(range(len(block)), key=lambda index: scaled[index])
        scaled[largest] += float(target) - math.fsum(scaled)
        result["county"].update(zip(block, scaled, strict=True))
    unknown = sorted(set(raw) - set(result["county"]))
    if unknown:
        raise ValueError(f"County populations have no state parent: {unknown}.")
    return result


def _hierarchy(name, code, *, parent=None):
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
        return dataclasses.replace(
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


def _broad_parent(spec):
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


@dataclasses.dataclass(frozen=True)
class CountySurface:
    specs: tuple
    receipt: dict


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


def _validated_combined_cells(combined_cells):
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


def state_county_soi_surface(
    state_specs,
    facts,
    county_populations,
    *,
    combined_cells=DECLARED_SOI_COMBINED_CELLS,
):
    """Rebase complete county fact blocks onto existing broad state totals.

    Every state must carry county return count and AGI with unsliced state
    parents. Other measures require both county facts and a state parent.
    A county absent from a block is refused unless ``combined_cells`` declares
    it, with its host present: then one child row covers both counties, its
    share is the host's published value over the state's published sum, and
    ``state_county_cell_members`` lists the member counties.
    """
    combined = _validated_combined_cells(combined_cells)
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
        if not _broad_parent(spec):
            continue
        metadata = spec.metadata
        state = int(metadata["state_fips"])
        key = (
            state,
            metadata.get("source_measure_id"),
            str(metadata.get("source_period", "")),
        )
        if key[1] not in COUNTY_SOI_REQUIRED_MEASURES and key[1] not in {
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
        for measure in COUNTY_SOI_REQUIRED_MEASURES:
            if not any(key[:2] == (state, measure) for key in parents):
                raise ValueError(
                    f"Missing broad state SOI parent state={state:02d}, concept={measure}; "
                    "state_county requires return count and AGI in every state."
                )
    specs, receipts, merges = [], [], defaultdict(list)
    for key, parent in sorted(parents.items()):
        state, measure, period = key
        expected = {
            int(code) for code in county_populations if int(code) // 1000 == state
        }
        block = blocks.get(key, {})
        absent = expected - block.keys()
        merged = {
            code: combined[code]
            for code in absent
            if code in combined
            and combined[code] in block
            and combined[code] in expected
        }
        missing, extra = (
            sorted(absent - merged.keys()),
            sorted(block.keys() - expected),
        )
        if not expected or missing or extra:
            raise ValueError(
                f"Incomplete county SOI block state={state:02d}, concept={measure}, period={period}; "
                f"missing={[f'{code:05d}' for code in missing]}, unexpected={[f'{code:05d}' for code in extra]}. "
                "County facts, ladder and county column must use matching boundaries; "
                "only a declared combined SOI cell may cover an absent county."
            )
        members = {code: [code] for code in sorted(block)}
        for code, host in sorted(merged.items()):
            members[host].append(code)
            merges[(host, code)].append(f"{measure}|{period}")
        total = math.fsum(value for value, _fact in block.values())
        if total <= 0:
            raise ValueError(
                f"County SOI block {key} has no positive share denominator."
            )
        codes = sorted(block)
        shares = [block[code][0] / total for code in codes]
        largest = max(range(len(codes)), key=lambda index: shares[index])
        shares[largest] += 1.0 - math.fsum(shares)
        targets = [parent.value * share for share in shares]
        targets[largest] += parent.value - math.fsum(targets)
        state_population = math.fsum(county_populations[code] for code in expected)
        for code, share, target in zip(codes, shares, targets, strict=True):
            raw, fact = block[code]
            unit = fact.get("observed_measure", {}).get("unit")
            if (
                parent.metadata.get("ledger_measure_unit") is not None
                and unit != parent.metadata["ledger_measure_unit"]
            ):
                raise ValueError(
                    f"County SOI units differ from state parent {parent.name}: {unit!r}."
                )
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
                "state_county_parent_target_name": parent.name,
                "state_county_share": str(share),
                "state_county_source_value": str(raw),
                "county_population": str(
                    math.fsum(county_populations[member] for member in cell)
                ),
                "state_population": str(state_population),
            }
            if len(cell) > 1:
                metadata["state_county_cell_members"] = ",".join(
                    f"{member:05d}" for member in cell
                )
                metadata["state_county_cell_basis"] = SOI_COMBINED_CELL_BASIS
            specs.append(
                dataclasses.replace(
                    parent,
                    name=name,
                    measure=name,
                    value=target,
                    se=None,
                    metadata=metadata,
                    hierarchy=_hierarchy(name, county, parent=parent),
                    source=f"{parent.source}; IRS SOI county {period} within-state shares",
                )
            )
        receipts.append(
            {
                "state_fips": f"{state:02d}",
                "concept": measure,
                "parent": parent.name,
                "counties": len(expected),
                "cells": len(codes),
                "share_sum": math.fsum(shares),
                "target_sum": math.fsum(targets),
            }
        )
    # Counties without a state parent cannot silently disappear.
    covered = {int(member) for spec in specs for member in cell_members(spec.metadata)}
    missing = sorted(set(county_populations) - covered)
    if missing:
        raise ValueError(f"Counties have no materialized SOI state parent: {missing}.")
    return CountySurface(
        tuple(specs),
        {
            "soi_mode": SOI_MODE,
            "county_targets": len(specs),
            "concepts": sorted({key[1] for key in parents}),
            "blocks": receipts,
            "combined_cells": [
                {
                    "members": sorted((f"{host:05d}", f"{absent:05d}")),
                    "host": f"{host:05d}",
                    "absent_from_soi": f"{absent:05d}",
                    "concepts": sorted(concepts),
                    "basis": SOI_COMBINED_CELL_BASIS,
                    "source": "https://www.irs.gov/pub/irs-soi/22incydocguide.docx",
                    "own_rows_kept": "ladder population, ACS households and "
                    "ACS validation rows stay per county",
                }
                for (host, absent), concepts in sorted(merges.items())
            ],
            "new_engine_columns": 0,
            "basis": "state target x county fact / state sum of county facts",
        },
    )


def cell_members(metadata):
    """The county codes one county child row covers (one unless combined)."""
    members = metadata.get("state_county_cell_members")
    if members:
        return [int(member) for member in str(members).split(",")]
    return [int(metadata["county_fips"])]


ACS_SOURCE_NAME = "census_acs"


ACS_TABLES = "ACS 2020-2024 5-year"


ACS_HOUSEHOLDS_FAMILY = "acs_households"


ACS_HOUSEHOLDS_MEASURE = "B11001_001E"


def _moe_id(measure_id):
    return measure_id[:-1] + "M"


def acs_county_values(facts, measure_ids, *, required=None):
    """``{measure: {county: (estimate, moe or None, name)}}`` from ACS rows.

    Reads every record set (B01003 splits its counties across two). Trains
    and scores on estimates only: a row is an estimate exactly when its
    ``statistic`` is ``estimate`` and its id ends in ``E``; its margin of
    error (id ending ``M``) is carried when present and never required.
    Duplicate, negative or nonfinite estimates are refused. With
    ``required`` (county codes), every measure must cover exactly those
    counties.
    """
    wanted = set(measure_ids)
    moe_ids = {_moe_id(measure): measure for measure in wanted}
    estimates = {measure: {} for measure in wanted}
    margins = {measure: {} for measure in wanted}
    for fact in facts:
        observed = fact.get("observed_measure", {})
        if observed.get("source_name") != ACS_SOURCE_NAME:
            continue
        measure = observed.get("source_measure_id")
        if measure not in wanted and measure not in moe_ids:
            continue
        statistic = (fact.get("dimensions") or {}).get("statistic")
        is_estimate = measure in wanted
        if statistic != ("estimate" if is_estimate else "margin_of_error"):
            raise ValueError(
                f"ACS county fact {measure} carries statistic {statistic!r}; "
                "an id ending E is an estimate and one ending M its margin of error."
            )
        code = county_code_of(fact, source="ACS")
        value = float(fact["value"])
        target = estimates[measure] if is_estimate else margins[moe_ids[measure]]
        if code in target:
            raise ValueError(f"Duplicate ACS county fact {measure} for {code:05d}.")
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                f"ACS county fact {measure} for {code:05d} must be finite and "
                f"nonnegative, got {value!r}."
            )
        target[code] = (value, str((fact.get("geography") or {}).get("name", "")))
    result = {}
    for measure in sorted(wanted):
        values = estimates[measure]
        if required is not None:
            expected = {int(code) for code in required}
            missing = sorted(expected - values.keys())
            extra = sorted(values.keys() - expected)
            if missing or extra:
                raise ValueError(
                    f"Incomplete ACS county inventory for {measure}: "
                    f"missing={[f'{code:05d}' for code in missing][:20]} "
                    f"({len(missing)}), unexpected="
                    f"{[f'{code:05d}' for code in extra][:20]} ({len(extra)}). "
                    "County facts, ladder and county column must use matching "
                    "boundaries."
                )
        result[measure] = {
            code: (
                value,
                margins[measure][code][0] if code in margins[measure] else None,
                name,
            )
            for code, (value, name) in sorted(values.items())
        }
    return result


def _state_label(state):
    from microcosm.calibrate.geography_constants import US_STATE_FIPS_TO_NAME

    label = US_STATE_FIPS_TO_NAME.get(f"{int(state):02d}")
    if label is None:
        raise ValueError(f"State {int(state):02d} has no Census state name.")
    return label


def _acs_hierarchy(name, *, state=None, county=None):
    from microcosm.calibrate import (
        CalibrationHierarchy,
        HierarchyCategory,
        HierarchyGeography,
        HierarchyNode,
    )
    from microcosm.calibrate.provider_labels import calibration_provider_label

    provider = HierarchyNode(
        ACS_SOURCE_NAME, calibration_provider_label("us", ACS_SOURCE_NAME)
    )
    if county is not None:
        # The same label every county family gives this county (_hierarchy).
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


def acs_household_surface(facts, county_populations, *, period=2024):
    """State households parents and their county children (trained, levels).

    Each state parent is the sum of its counties' ACS B11001_001E values, a
    new state target that moves state household counts to ACS 5-year levels;
    its county children split it by county share, so they sum to it. The
    measure counts households outside ACS group quarters
    (:func:`household_universe`). ACS margins of error ride in metadata.
    """
    from microcosm.calibrate import TargetSpec

    values = acs_county_values(
        facts, [ACS_HOUSEHOLDS_MEASURE], required=county_populations
    )[ACS_HOUSEHOLDS_MEASURE]
    source = (
        f"US Census Bureau {ACS_TABLES} B11001_001E (households), county; "
        "state = sum of its counties"
    )
    specs, receipts = [], []
    for state in sorted({int(code) // 1000 for code in county_populations}):
        codes = sorted(code for code in values if code // 1000 == state)
        public = [values[code][0] for code in codes]
        total = math.fsum(public)
        if not codes or total <= 0:
            raise ValueError(
                f"ACS county households for state {state:02d} have no positive "
                "share denominator."
            )
        parent_name = acs_households_parent_name(state)
        state_population = math.fsum(county_populations[code] for code in codes)
        specs.append(
            TargetSpec(
                name=parent_name,
                entity="household",
                measure=parent_name,
                value=total,
                period=period,
                source=source,
                family=ACS_HOUSEHOLDS_FAMILY,
                metadata={
                    "geography_level": "state",
                    "state_fips": f"{state:02d}",
                    "source_measure_id": ACS_HOUSEHOLDS_MEASURE,
                    "acs_state_parent_basis": "sum of county estimates",
                    "household_universe": "excludes ACS group quarters (TYPEHUGQ 2, 3)",
                    "new_state_target": "true",
                },
                hierarchy=_acs_hierarchy(parent_name, state=state),
            )
        )
        shares = [value / total for value in public]
        largest = max(range(len(codes)), key=lambda index: shares[index])
        shares[largest] += 1.0 - math.fsum(shares)
        targets = [total * share for share in shares]
        targets[largest] += total - math.fsum(targets)
        for code, share, target in zip(codes, shares, targets, strict=True):
            county = f"{code:05d}"
            name = f"county_{county}__{parent_name}"
            estimate, moe, label = values[code]
            metadata = {
                "geography_level": "county",
                "county_fips": county,
                "state_fips": f"{state:02d}",
                "source_measure_id": ACS_HOUSEHOLDS_MEASURE,
                "state_county_parent_target_name": parent_name,
                "state_county_share": str(share),
                "state_county_source_value": str(estimate),
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
                    value=target,
                    period=period,
                    source=f"US Census Bureau {ACS_TABLES} B11001_001E "
                    "(households), county share of the state sum",
                    family=ACS_HOUSEHOLDS_FAMILY,
                    metadata=metadata,
                    hierarchy=_acs_hierarchy(name, county=county),
                )
            )
        receipts.append(
            {
                "state_fips": f"{state:02d}",
                "state_target": parent_name,
                "state_households": total,
                "counties": len(codes),
                "share_sum": math.fsum(shares),
                "target_sum": math.fsum(targets),
                "counties_with_moe": sum(values[code][1] is not None for code in codes),
            }
        )
    return CountySurface(
        tuple(specs),
        {
            "family": ACS_HOUSEHOLDS_FAMILY,
            "source_measure_id": ACS_HOUSEHOLDS_MEASURE,
            "source": f"{ACS_TABLES} county B11001",
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
        },
    )
