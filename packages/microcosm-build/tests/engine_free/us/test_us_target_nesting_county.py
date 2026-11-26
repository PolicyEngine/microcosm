"""The county rows on the shared operator give what the tool gave.

microcosm#1760 moved three county mechanisms of the ACS local release onto
the shared cross-grain operator (``microcosm.build.cross_grain``), through
``microcosm.build.us_runtime.target_nesting``: the county populations
(``with_county_populations``), the county SOI shares
(``state_county_soi_surface``) and the county household rows
(``acs_household_surface``). These tests hold the tool functions, which now
call the adapter, to verbatim copies of the code they replace
(``test_support/microcosm_build/us_target_nesting_reference.py``) on seeded
random blocks: the same specs in the same order, the same value bits, the
same metadata keys in the same order, the same receipt bytes and the same
refusal texts.

The blocks are synthetic. No test here reads the pinned county facts file
(``us/chronicle_county_facts.json``), which is not a test input on every
machine; a pin of the county rows on that file is a separate step.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import struct
from types import SimpleNamespace

import numpy as np
import pytest

from microcosm.build.us_runtime import cross_grain_declarations as declarations
from microcosm.build.us_runtime import target_nesting
from microcosm.calibrate import (
    CalibrationHierarchy,
    HierarchyCategory,
    HierarchyGeography,
    HierarchyNode,
    TargetSpec,
)
from test_support.microcosm_build import us_target_nesting_reference as reference
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")
SEEDS = range(120)
#: County SOI measures: the two every state must carry, and two optional ones.
_SOI_MEASURES = (
    "return_count",
    "adjusted_gross_income",
    "wages_salaries_amount",
    "taxable_interest_amount",
)


@pytest.fixture(scope="module")
def county():
    path = _TEST_PATHS.repository / "tools" / "build_us_acs_local_release.py"
    spec = importlib.util.spec_from_file_location(
        "build_us_acs_local_release_nesting_county", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.county_surface


def _bits(values) -> bytes:
    return b"".join(struct.pack("<d", float(value)) for value in values)


def _spec_image(spec) -> tuple:
    fields = {
        field.name: getattr(spec, field.name) for field in dataclasses.fields(spec)
    }
    return (
        struct.pack("<d", spec.value),
        list(spec.metadata.items()),
        {name: value for name, value in fields.items() if name != "value"},
    )


def _surface_image(surface) -> tuple:
    return (
        [_spec_image(spec) for spec in surface.specs],
        json.dumps(surface.receipt, indent=1),
    )


def _outcome(function, *args, **kwargs):
    try:
        result = function(*args, **kwargs)
    except ValueError as error:
        return ("refused", str(error))
    return ("result", result)


def _county_codes(rng, *, hawaii: bool) -> dict[int, float]:
    """Counties of one to four states with their ladder populations."""

    populations: dict[int, float] = {}
    # State codes with a Census name. Hawaii (15) enters only as the
    # declared combined-cell case below.
    candidates = np.setdiff1d(np.arange(1, 57), [3, 7, 14, 15, 43, 52])
    states = sorted(rng.choice(candidates, size=int(rng.integers(1, 5)), replace=False))
    for state in states:
        for index in range(int(rng.integers(1, 9))):
            populations[int(state) * 1000 + 2 * index + 1] = float(
                rng.lognormal(mean=10.0, sigma=1.5)
            )
    if hawaii:
        populations.update({15001: 200.0, 15003: 1000.0, 15005: 80.0, 15009: 160.0})
    return dict(sorted(populations.items()))


# ---------------------------------------------------------------------------
# County populations (issue row 8)
# ---------------------------------------------------------------------------


def _ladder(seed: int):
    rng = np.random.default_rng(seed)
    populations = _county_codes(rng, hawaii=False)
    overlap_county, overlap_population, overlap_puma = [], [], []
    for code in populations:
        for piece in range(int(rng.integers(1, 5))):
            overlap_county.append(code)
            overlap_puma.append(code * 100 + piece)
            value = float(rng.lognormal(mean=8.0, sigma=2.0))
            if rng.random() < 0.3:
                value = float(round(value))
            if rng.random() < 0.05:
                value = 0.0
            overlap_population.append(value)
    states = sorted({code // 1000 for code in populations})
    hazard = rng.random()
    if hazard > 0.95:
        # A county whose state has no total, or a state with no county.
        overlap_county.append(57001)
        overlap_puma.append(5700100)
        overlap_population.append(10.0)
    puma = np.asarray([state * 100_000 + 1 for state in states])
    puma_population = rng.lognormal(mean=13.0, sigma=1.0, size=len(states))
    if 0.90 < hazard <= 0.95:
        puma_population[0] = 0.0
    order = rng.permutation(len(overlap_county))
    ladder = SimpleNamespace(
        puma=puma,
        puma_population=puma_population,
        county_overlap_puma=np.asarray(overlap_puma)[order],
        county_overlap_county=np.asarray(overlap_county)[order],
        county_overlap_population=np.asarray(overlap_population)[order],
    )
    given = (
        {}
        if rng.random() < 0.5
        else {
            "state": {
                int(state): float(rng.lognormal(mean=13.0, sigma=1.0))
                for state in states
            },
            "cd": {int(states[0]) * 100 + 1: 5.0},
        }
    )
    return ladder, given


def _population_image(result) -> tuple:
    return (
        list(result),
        {
            level: (list(cells), _bits(cells.values()))
            for level, cells in result.items()
        },
    )


def test_county_populations_are_the_old_populations_on_random_ladders(county) -> None:
    outcomes = {"result": 0, "refused": 0}
    cells = 0
    for seed in SEEDS:
        ladder, given = _ladder(seed)
        new = _outcome(county.with_county_populations, ladder, given, ["county"])
        old = _outcome(reference.with_county_populations, ladder, given, ["county"])
        assert new[0] == old[0], seed
        if new[0] == "refused":
            assert new[1] == old[1], seed
        else:
            assert _population_image(new[1]) == _population_image(old[1]), seed
            cells += len(new[1]["county"])
        outcomes[new[0]] += 1
        # Without the county level the populations pass through untouched.
        assert county.with_county_populations(ladder, given, ["state", "cd"]) is given
    assert outcomes["result"] >= 90 and outcomes["refused"] >= 5
    assert cells >= 1_000


# ---------------------------------------------------------------------------
# County SOI shares (issue row 6)
# ---------------------------------------------------------------------------


def _hierarchy(name: str, state: str) -> CalibrationHierarchy:
    provider = HierarchyNode("irs_soi", "IRS Statistics of Income")
    return CalibrationHierarchy(
        provider=provider,
        category=HierarchyCategory("irs_soi.returns", "Returns", provider.id),
        geography=HierarchyGeography(f"0400000US{state}", f"State {state}", "state"),
        dimensions=(),
        target=HierarchyNode(name, f"State {state} {name}"),
    )


def _state_spec(measure, value, *, state, with_hierarchy, **metadata):
    variable = "count" if measure == "return_count" else measure
    name = f"state.{state}.{measure}"
    return TargetSpec(
        name=name,
        entity="household",
        measure=name,
        value=value,
        period=2024,
        source="fixture state SOI",
        family="irs_soi",
        metadata={
            "ledger_geography_level": "state",
            "ledger_layout_record_set_spec_id": (
                "irs_soi.historic_table_2.state_broad_totals.v1"
            ),
            "state_fips": state,
            "source_measure_id": measure,
            "source_period": "2022",
            "ledger_measure_unit": "count" if measure == "return_count" else "usd",
            "variable": variable,
            "source_variable": variable,
            "measure_mode": "sum",
            "agi_lower_bound": "-inf",
            "agi_upper_bound": "inf",
            "filing_status": "All",
            **metadata,
        },
        hierarchy=_hierarchy(name, state) if with_hierarchy else None,
    )


def _soi_fact(code: int, measure: str, value: float, *, unit=None, with_layout=True):
    fact = {
        "geography": {
            "id": f"0500000US{code:05d}",
            "level": "county",
            "name": f"County {code:05d}",
            "vintage": "2022",
        },
        "period": {"type": "tax_year", "value": 2022},
        "observed_measure": {
            "source_measure_id": measure,
            "source_name": "irs_soi",
            "unit": unit or ("count" if measure == "return_count" else "usd"),
        },
        "dimensions": {"filing_status": "all", "income_range": "all"},
        "value": value,
    }
    if with_layout:
        fact["layout"] = {
            "record_set_spec_id": "irs_soi.county_totals.v1",
            "groupby_value_id": f"{code:05d}",
        }
    return fact


def _soi_inputs(seed: int):
    """State parents, county facts and populations, with the hard cases."""

    rng = np.random.default_rng(seed)
    hawaii = rng.random() < 0.25
    populations = _county_codes(rng, hawaii=hawaii)
    states = sorted({code // 1000 for code in populations})
    measures = list(_SOI_MEASURES[:2]) + [
        measure for measure in _SOI_MEASURES[2:] if rng.random() < 0.6
    ]
    hazard = rng.random()
    parents, facts = [], []
    for state in states:
        codes = [code for code in populations if code // 1000 == state]
        for measure in measures:
            if measure not in _SOI_MEASURES[:2] and rng.random() < 0.2:
                continue  # an optional measure with no state parent here
            parents.append(
                _state_spec(
                    measure,
                    float(rng.lognormal(mean=14.0, sigma=1.0)),
                    state=f"{state:02d}",
                    with_hierarchy=rng.random() < 0.7,
                )
            )
            values = rng.lognormal(mean=9.0, sigma=2.0, size=len(codes))
            if rng.random() < 0.4:
                values = np.round(values)
            if len(codes) > 2 and rng.random() < 0.3:
                values[int(rng.integers(len(codes)))] = values.max()  # a tie
            if rng.random() < 0.2:
                values[int(rng.integers(len(codes)))] = 0.0
            if 0.86 < hazard <= 0.89 and rng.random() < 0.3:
                values[:] = 0.0  # no positive share denominator
            for code, value in zip(codes, values.tolist(), strict=True):
                if code == 15005:
                    continue  # absent from the SOI file: a declared combined cell
                unit = (
                    "returns" if 0.89 < hazard <= 0.92 and rng.random() < 0.05 else None
                )
                facts.append(
                    _soi_fact(
                        code, measure, value, unit=unit, with_layout=rng.random() < 0.9
                    )
                )
        # An AGI-band parent is no broad parent: it is passed over.
        if rng.random() < 0.3:
            parents.append(
                _state_spec(
                    "return_count",
                    7.0,
                    state=f"{state:02d}",
                    with_hierarchy=False,
                    agi_lower_bound="100000",
                )
            )
    if 0.92 < hazard <= 0.95 and facts:
        del facts[int(rng.integers(len(facts)))]  # a county absent from a block
    if 0.95 < hazard <= 0.97 and facts:
        facts.append(dict(facts[0]))  # a duplicate fact
    if hazard > 0.97:
        parents = [spec for spec in parents if "adjusted_gross" not in spec.name][:-1]
    # A fact of another source is not a county SOI fact.
    facts.append({"observed_measure": {"source_name": "census_acs"}})
    order = rng.permutation(len(facts))
    return parents, [facts[index] for index in order], populations


def _surface_outcome(function, *args, **kwargs):
    outcome = _outcome(function, *args, **kwargs)
    if outcome[0] == "refused":
        return outcome
    return ("surface", *_surface_image(outcome[1]))


def test_the_county_soi_surface_is_the_old_surface_on_random_blocks(county) -> None:
    outcomes = {"surface": 0, "refused": 0}
    refusals: set[str] = set()
    rows = combined = 0
    for seed in SEEDS:
        parents, facts, populations = _soi_inputs(seed)
        new = _surface_outcome(
            county.state_county_soi_surface, parents, facts, populations
        )
        old = _surface_outcome(
            reference.state_county_soi_surface, parents, facts, populations
        )
        assert new == old, seed
        outcomes[new[0]] += 1
        if new[0] == "refused":
            refusals.add(" ".join(new[1].split(" ")[:3]))
            continue
        receipt = json.loads(new[2])
        rows += receipt["county_targets"]
        combined += len(receipt["combined_cells"])
    assert outcomes["surface"] >= 80 and outcomes["refused"] >= 10
    assert len(refusals) >= 4
    assert rows >= 2_000 and combined >= 10


def test_a_combined_cell_other_than_the_declared_one_is_the_old_refusal(county) -> None:
    parents, facts, populations = _soi_inputs(3)
    for cells in ({1001: 1001}, {1001: 2001}, {1003: 1001, 1001: 1005}):
        new = _surface_outcome(
            county.state_county_soi_surface,
            parents,
            facts,
            populations,
            combined_cells=cells,
        )
        old = _surface_outcome(
            reference.state_county_soi_surface,
            parents,
            facts,
            populations,
            combined_cells=cells,
        )
        assert new == old and new[0] == "refused"


# ---------------------------------------------------------------------------
# County households (issue row 7)
# ---------------------------------------------------------------------------


def _acs_fact(code: int, measure: str, value, *, statistic=None, name=None):
    return {
        "geography": {
            "id": f"0500000US{code:05d}",
            "level": "county",
            **({} if name is None else {"name": name}),
            "vintage": "2024",
        },
        "observed_measure": {
            "source_measure_id": measure,
            "source_name": "census_acs",
            "unit": "count",
        },
        "dimensions": {
            "statistic": statistic
            or ("estimate" if measure.endswith("E") else "margin_of_error")
        },
        "period": {"type": "calendar_year", "value": 2024},
        "value": value,
        "layout": {"groupby_value_id": f"{code:05d}"},
    }


def _acs_inputs(seed: int):
    rng = np.random.default_rng(seed)
    populations = _county_codes(rng, hawaii=False)
    hazard = rng.random()
    facts = []
    for code in populations:
        households = float(round(rng.lognormal(mean=8.0, sigma=2.0)))
        if rng.random() < 0.1:
            households = 0.0
        if 0.90 < hazard <= 0.94:
            households = 0.0  # no positive share denominator
        name = None if rng.random() < 0.2 else f"Fixture County {code:05d}"
        facts.append(_acs_fact(code, "B11001_001E", households, name=name))
        if rng.random() < 0.7:
            facts.append(_acs_fact(code, "B11001_001M", max(1.0, households * 0.05)))
        # Another ACS measure is not the households measure.
        facts.append(_acs_fact(code, "B01003_001E", households * 2.5))
    if 0.94 < hazard <= 0.97:
        del facts[0]  # an incomplete inventory
    if 0.97 < hazard <= 0.985:
        facts.append(_acs_fact(56999, "B11001_001E", 5.0))  # outside the ladder
    if hazard > 0.985:
        # A state code with no Census name, between two states that have one.
        populations[14001] = 10.0
        populations = dict(sorted(populations.items()))
        facts.append(_acs_fact(14001, "B11001_001E", 5.0))
    order = rng.permutation(len(facts))
    return [facts[index] for index in order], populations


def test_the_county_household_surface_is_the_old_surface_on_random_blocks(
    county,
) -> None:
    outcomes = {"surface": 0, "refused": 0}
    rows = 0
    for seed in SEEDS:
        facts, populations = _acs_inputs(seed)
        period = 2024 if seed % 3 else 2026
        new = _surface_outcome(
            county.acs_household_surface, facts, populations, period=period
        )
        old = _surface_outcome(
            reference.acs_household_surface, facts, populations, period=period
        )
        assert new == old, seed
        outcomes[new[0]] += 1
        if new[0] == "surface":
            rows += json.loads(new[2])["county_targets"]
    assert outcomes["surface"] >= 90 and outcomes["refused"] >= 5
    assert rows >= 1_000


def test_household_refusals_come_in_state_order_as_before(county) -> None:
    """A state with no Census name is refused before a later state's zero sum."""

    populations = {14001: 10.0, 34001: 10.0}
    facts = [
        _acs_fact(14001, "B11001_001E", 5.0),
        _acs_fact(34001, "B11001_001E", 0.0),
    ]
    new = _surface_outcome(county.acs_household_surface, facts, populations)
    old = _surface_outcome(reference.acs_household_surface, facts, populations)
    assert new == old == ("refused", "State 14 has no Census state name.")
    # The other way round, the zero sum of the earlier state comes first.
    populations = {1001: 10.0, 14001: 10.0}
    facts = [
        _acs_fact(1001, "B11001_001E", 0.0),
        _acs_fact(14001, "B11001_001E", 5.0),
    ]
    new = _surface_outcome(county.acs_household_surface, facts, populations)
    old = _surface_outcome(reference.acs_household_surface, facts, populations)
    assert new == old and "no positive share denominator" in new[1]


# ---------------------------------------------------------------------------
# What the wrappers keep, and the operator's receipts
# ---------------------------------------------------------------------------


def test_the_helpers_the_tool_keeps_by_name_are_the_adapters(county) -> None:
    assert (
        county.COUNTY_PREFIX == target_nesting.COUNTY_PREFIX == reference.COUNTY_PREFIX
    )
    assert county.PUERTO_RICO_STATE == reference.PUERTO_RICO_STATE
    assert county._hierarchy is target_nesting.county_hierarchy
    assert county.cell_members is target_nesting.cell_members
    assert county.county_code_of is target_nesting.county_code_of
    assert county._broad_parent is target_nesting.is_broad_state_parent
    # The moved helpers give what the old ones gave.
    spec = _state_spec("return_count", 5.0, state="06", with_hierarchy=True)
    assert county._hierarchy("n", "06001", parent=spec) == reference._hierarchy(
        "n", "06001", parent=spec
    )
    assert county._hierarchy("n", "06001") == reference._hierarchy("n", "06001")
    assert county.acs_households_parent_name(6) == reference.acs_households_parent_name(
        6
    )
    for metadata in (
        {"county_fips": "15009"},
        {"county_fips": "15009", "state_county_cell_members": "15005,15009"},
    ):
        assert county.cell_members(metadata) == reference.cell_members(metadata)


def test_each_county_pass_records_the_operators_receipt(county) -> None:
    def first_accepted(inputs, function):
        """The first seeded inputs the tool accepts."""

        for seed in SEEDS:
            arguments = inputs(seed)
            if _outcome(function, *arguments)[0] == "result":
                return arguments
        raise AssertionError("no seed was accepted")

    ladder, given = first_accepted(
        _ladder, lambda *a: county.with_county_populations(*a, ["county"])
    )
    parents, facts, populations = first_accepted(
        _soi_inputs, county.state_county_soi_surface
    )
    acs_facts, acs_populations = first_accepted(
        _acs_inputs, county.acs_household_surface
    )
    recorded = target_nesting.start_nesting_receipts()
    try:
        county.with_county_populations(ladder, given, ["county"])
        soi = county.state_county_soi_surface(parents, facts, populations)
        acs = county.acs_household_surface(acs_facts, acs_populations)
    finally:
        target_nesting.stop_nesting_receipts()
    assert [(entry["mechanism"], entry["allocator"]) for entry in recorded] == [
        (declarations.COUNTY_POPULATION, "scale_residual_to_largest"),
        (declarations.COUNTY_SOI, "share_residual_to_largest"),
        (declarations.COUNTY_ACS_HOUSEHOLDS, "share_residual_to_largest"),
    ]
    for entry, blocks in zip(
        recorded[1:], (soi.receipt["blocks"], acs.receipt["blocks"]), strict=True
    ):
        legs = [
            leg
            for group in entry["operator_receipt"]["groups"]
            for leg in group["legs"]
        ]
        assert len(legs) == entry["blocks"] == len(blocks)
        assert {
            group["winning_grain"] for group in entry["operator_receipt"]["groups"]
        } == {"state"}
        # Each leg closed on its control; the old receipt's target sum is that
        # control under an exactly rounded sum.
        for leg in legs:
            assert leg["new_total"] == pytest.approx(
                leg["old_total"] * leg["declared_factor"], rel=1e-12
            )
    # The old receipts do not hold the operator's receipt.
    assert "groups" not in soi.receipt and "groups" not in acs.receipt
