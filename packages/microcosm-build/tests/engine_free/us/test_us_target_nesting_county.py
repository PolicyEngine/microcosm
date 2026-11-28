"""The county rows on the cross-grain model are the rows the tool made.

microcosm#1760 runs three county passes of the ACS local release on the
shared cross-grain operator (``microcosm.build.cross_grain``), through
``microcosm.build.us_runtime.target_nesting``: the county populations
(``with_county_populations``), the county SOI rows
(``state_county_soi_surface``) and the county household rows
(``acs_household_surface``). These tests hold the tool functions to verbatim
copies of the code they replace
(``test_support/microcosm_build/us_target_nesting_reference.py``) on seeded
random blocks: the same specs in the same order, the same value bits, the same
parent, value before and members under the operator's metadata keys, and the
same receipt. Where the old code refused, the new code refuses: with the same
text when the adapter refuses, and with the operator's text when the operator
does.

The seeded blocks are synthetic. Two tests at the end read the real inputs:
the pinned feed, the pinned PUMA ladder and the pinned county facts file
(``us/chronicle_county_facts.json``). They hold the county rows of those
inputs to the old code, and they pin the bytes of the rows. They run only
when ``MICROCOSM_US_CHRONICLE_FACTS``, ``MICROCOSM_US_PUMA_LADDER_ARTIFACT``
and ``MICROCOSM_US_CHRONICLE_COUNTY_FACTS`` name those files, because the
county facts file (922 MB) is not a test input on every machine.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import math
import os
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from microcosm.build.cross_grain import (
    CONTROL_KEY,
    MEMBERS_KEY,
    PARENT_KEY,
    VALUE_BEFORE_KEY,
)
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
def local_tool():
    path = _TEST_PATHS.repository / "tools" / "build_us_acs_local_release.py"
    spec = importlib.util.spec_from_file_location(
        "build_us_acs_local_release_nesting_county", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def county(local_tool):
    return local_tool.county_surface


def _bits(values) -> bytes:
    return b"".join(struct.pack("<d", float(value)) for value in values)


_OLD_KEYS = "state_county_"
_NEW_KEYS = "cross_grain_"


def _plain_image(spec, prefix: str) -> tuple:
    """Everything a spec holds but its nesting keys, with the value's bits."""

    fields = {
        field.name: getattr(spec, field.name) for field in dataclasses.fields(spec)
    }
    return (
        struct.pack("<d", spec.value),
        [item for item in spec.metadata.items() if not item[0].startswith(prefix)],
        {
            name: value
            for name, value in fields.items()
            if name not in ("value", "metadata")
        },
    )


def _old_nesting(spec) -> tuple | None:
    """What the old keys say: (parent, value before, members)."""

    metadata = spec.metadata
    if "state_county_parent_target_name" not in metadata:
        assert not any(key.startswith(_OLD_KEYS) for key in metadata), spec.name
        return None
    return (
        metadata["state_county_parent_target_name"],
        metadata["state_county_source_value"],
        metadata.get("state_county_cell_members"),
    )


def _new_nesting(spec) -> tuple | None:
    """The same three things from the operator's keys."""

    metadata = spec.metadata
    if CONTROL_KEY not in metadata:
        assert not any(key.startswith(_NEW_KEYS) for key in metadata), spec.name
        return None
    return (
        metadata[PARENT_KEY],
        metadata[VALUE_BEFORE_KEY],
        metadata.get(MEMBERS_KEY),
    )


def _receipt_image(receipt: dict, *, old: bool) -> str:
    """The receipt without what only one side writes.

    The old blocks hold ``share_sum``, the float sum of the shares (one, or
    one bit from it). The new receipt names its declared bridge, its bound
    and the legs it unbound under ``bridge``.
    """

    image = {key: value for key, value in receipt.items() if key != "bridge"}
    if old:
        image["blocks"] = [
            {key: value for key, value in block.items() if key != "share_sum"}
            for block in receipt["blocks"]
        ]
    return json.dumps(image, indent=1)


def _assert_same_surface(new, old, pass_id: str, seed: int) -> None:
    assert [
        (_plain_image(spec, _NEW_KEYS), _new_nesting(spec)) for spec in new.specs
    ] == [(_plain_image(spec, _OLD_KEYS), _old_nesting(spec)) for spec in old.specs], (
        seed
    )
    assert _receipt_image(new.receipt, old=False) == _receipt_image(
        old.receipt, old=True
    ), seed
    assert new.receipt["bridge"]["bridge_id"] == pass_id
    assert new.receipt["bridge"]["unbound"] == []


def _assert_same_refusal(new: tuple, old: tuple, seed: int) -> bool:
    """Both refuse. True when the text is the old text.

    The operator checks the members of a block after the adapter's own
    checks. So for inputs with two defects, where the old code named the
    block that is not complete, the new code can name the other defect.
    """

    assert new[0] == old[0] == "refused", (seed, new[1], old[1])
    if new[1] == old[1]:
        return True
    assert new[1].startswith("cross-grain ") or old[1].startswith(
        "Incomplete county SOI block"
    ), (seed, new[1], old[1])
    return False


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
    texts = {True: 0, False: 0}
    cells = 0
    for seed in SEEDS:
        ladder, given = _ladder(seed)
        new = _outcome(county.with_county_populations, ladder, given, ["county"])
        old = _outcome(reference.with_county_populations, ladder, given, ["county"])
        assert new[0] == old[0], seed
        if new[0] == "refused":
            texts[_assert_same_refusal(new, old, seed)] += 1
        else:
            assert _population_image(new[1]) == _population_image(old[1]), seed
            cells += len(new[1]["county"])
        outcomes[new[0]] += 1
        # Without the county level the populations pass through untouched.
        assert county.with_county_populations(ladder, given, ["state", "cd"]) is given
    assert outcomes["result"] >= 90 and outcomes["refused"] >= 5
    # Both kinds of refusal occur: the adapter's and the operator's.
    assert min(texts.values()) >= 1
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


def test_the_county_soi_surface_is_the_old_surface_on_random_blocks(county) -> None:
    outcomes = {"result": 0, "refused": 0}
    texts = {True: 0, False: 0}
    refusals: set[str] = set()
    rows = combined = 0
    for seed in SEEDS:
        parents, facts, populations = _soi_inputs(seed)
        # The old code had no bound, and the draws make random factors: the
        # bound is off here and has its own test below.
        new = _outcome(
            county.state_county_soi_surface,
            parents,
            facts,
            populations,
            factor_band=math.inf,
        )
        old = _outcome(reference.state_county_soi_surface, parents, facts, populations)
        assert new[0] == old[0], (seed, new[1], old[1])
        outcomes[new[0]] += 1
        if new[0] == "refused":
            texts[_assert_same_refusal(new, old, seed)] += 1
            refusals.add(" ".join(old[1].split(" ")[:3]))
            continue
        _assert_same_surface(new[1], old[1], declarations.COUNTY_SOI, seed)
        rows += new[1].receipt["county_targets"]
        combined += len(new[1].receipt["combined_cells"])
    assert outcomes["result"] >= 80 and outcomes["refused"] >= 10
    assert len(refusals) >= 4
    # A county absent from a block is the operator's refusal. The others are
    # the adapter's, with their old text.
    assert min(texts.values()) >= 2
    assert rows >= 2_000 and combined >= 10


def test_a_combined_cell_other_than_the_declared_one_is_the_old_refusal(county) -> None:
    parents, facts, populations = _soi_inputs(3)
    for cells in ({1001: 1001}, {1001: 2001}, {1003: 1001, 1001: 1005}):
        new = _outcome(
            county.state_county_soi_surface,
            parents,
            facts,
            populations,
            combined_cells=cells,
        )
        old = _outcome(
            reference.state_county_soi_surface,
            parents,
            facts,
            populations,
            combined_cells=cells,
        )
        assert new == old and new[0] == "refused"


def _banded_case(factors: dict[str, tuple[float, float]]):
    """States with two counties each; a state's factors for the two measures."""

    parents, facts, populations = [], [], {}
    for state, by_measure in factors.items():
        codes = [int(state) * 1000 + 1, int(state) * 1000 + 3]
        populations.update(dict.fromkeys(codes, 50.0))
        for measure, factor in zip(_SOI_MEASURES[:2], by_measure, strict=True):
            parents.append(
                _state_spec(measure, 100.0 * factor, state=state, with_hierarchy=False)
            )
            facts += [
                _soi_fact(codes[0], measure, 30.0),
                _soi_fact(codes[1], measure, 70.0),
            ]
    return parents, facts, dict(sorted(populations.items()))


def test_a_state_off_the_median_factor_loses_its_county_rows(county) -> None:
    """The bound of the county bridge is the district bridge's: 1.25, unbind."""

    assert county.COUNTY_SOI_FACTOR_BAND == 1.25
    # Every state is 1.12 times its county file, as an aged amount is. State
    # 05 is 2.0 times it for adjusted gross income only.
    factors = {state: (1.0, 1.12) for state in ("01", "02", "04", "05", "06")}
    factors["05"] = (1.0, 2.0)
    parents, facts, populations = _banded_case(factors)
    surface = county.state_county_soi_surface(parents, facts, populations)
    bridge = surface.receipt["bridge"]
    assert bridge["bridge_id"] == declarations.COUNTY_SOI
    assert (bridge["max_factor_shift"], bridge["min_factor_shift"]) == (0.25, -0.2)
    assert (bridge["shift_around"], bridge["out_of_bound"]) == ("median", "unbind")
    (unbound,) = bridge["unbound"]
    assert unbound == {
        "state_fips": "05",
        "concept": "adjusted_gross_income",
        "parent": "state.05.adjusted_gross_income",
        "factor": pytest.approx(2.0),
        "counties": 2,
    }
    # The leg's county rows are off the surface. Every other row is bound,
    # and a shared factor of 1.12 is no shift at all.
    by_parent: dict[str, list[float]] = {}
    for spec in surface.specs:
        by_parent.setdefault(spec.metadata[PARENT_KEY], []).append(spec.value)
    assert "state.05.adjusted_gross_income" not in by_parent
    assert len(by_parent) == 2 * 5 - 1 and len(surface.specs) == 2 * (2 * 5 - 1)
    assert by_parent["state.05.return_count"] == pytest.approx([30.0, 70.0])
    assert by_parent["state.06.adjusted_gross_income"] == pytest.approx([33.6, 78.4])
    assert surface.receipt["county_targets"] == 18
    assert len(surface.receipt["blocks"]) == 9

    # A state that is off in both measures loses all its county rows. Its
    # counties are named in the receipt, so the surface is not refused.
    factors["05"] = (0.5, 2.0)
    parents, facts, populations = _banded_case(factors)
    surface = county.state_county_soi_surface(parents, facts, populations)
    assert [
        (row["state_fips"], row["concept"])
        for row in surface.receipt["bridge"]["unbound"]
    ] == [("05", "adjusted_gross_income"), ("05", "return_count")]
    assert not [spec for spec in surface.specs if spec.metadata["state_fips"] == "05"]

    # With the bound off, every leg is divided from its state row.
    surface = county.state_county_soi_surface(
        parents, facts, populations, factor_band=math.inf
    )
    assert surface.receipt["bridge"]["unbound"] == []
    assert surface.receipt["bridge"]["max_factor_shift"] is None
    assert len(surface.specs) == 2 * 2 * 5


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
    outcomes = {"result": 0, "refused": 0}
    rows = 0
    for seed in SEEDS:
        facts, populations = _acs_inputs(seed)
        period = 2024 if seed % 3 else 2026
        new = _outcome(county.acs_household_surface, facts, populations, period=period)
        old = _outcome(
            reference.acs_household_surface, facts, populations, period=period
        )
        assert new[0] == old[0], (seed, new[1], old[1])
        outcomes[new[0]] += 1
        if new[0] == "refused":
            # Each household refusal is the adapter's, with its old text.
            assert _assert_same_refusal(new, old, seed), seed
            continue
        _assert_same_surface(new[1], old[1], declarations.COUNTY_ACS_HOUSEHOLDS, seed)
        rows += new[1].receipt["county_targets"]
    assert outcomes["result"] >= 90 and outcomes["refused"] >= 5
    assert rows >= 1_000


def test_household_refusals_come_in_state_order_as_before(county) -> None:
    """A state with no Census name is refused before a later state's zero sum."""

    populations = {14001: 10.0, 34001: 10.0}
    facts = [
        _acs_fact(14001, "B11001_001E", 5.0),
        _acs_fact(34001, "B11001_001E", 0.0),
    ]
    new = _outcome(county.acs_household_surface, facts, populations)
    old = _outcome(reference.acs_household_surface, facts, populations)
    assert new == old == ("refused", "State 14 has no Census state name.")
    # The other way round, the zero sum of the earlier state comes first.
    populations = {1001: 10.0, 14001: 10.0}
    facts = [
        _acs_fact(1001, "B11001_001E", 0.0),
        _acs_fact(14001, "B11001_001E", 5.0),
    ]
    new = _outcome(county.acs_household_surface, facts, populations)
    old = _outcome(reference.acs_household_surface, facts, populations)
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
    assert county.cell_members({"county_fips": "15009"}) == [15009]
    # The members of a combined cell, under the operator's key.
    combined = {"county_fips": "15009", MEMBERS_KEY: "15005,15009"}
    assert county.cell_members(combined) == reference.cell_members(
        {"county_fips": "15009", "state_county_cell_members": "15005,15009"}
    )


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
    assert [(entry["pass"], entry["allocator"]) for entry in recorded] == [
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
        assert len(legs) == len(blocks)
        assert {
            group["winning_grain"] for group in entry["operator_receipt"]["groups"]
        } == {"state"}
        # Each leg closed on its control.
        for leg in legs:
            assert leg["new_total"] == pytest.approx(
                leg["old_total"] * leg["declared_factor"], rel=1e-12
            )
    # The surfaces' receipts do not hold the operator's receipt.
    assert "groups" not in soi.receipt and "groups" not in acs.receipt


# ---------------------------------------------------------------------------
# The real inputs: the pinned feed, ladder and county facts
# ---------------------------------------------------------------------------

#: The household county columns: 2020 counties, and the 2023 county-equivalents
#: of the county facts (Connecticut's planning regions), which a full run uses.
_COUNTY_COLUMNS = ("county_fips", "county_fips_2023")
_COUNTY_FACTS_ENV = "MICROCOSM_US_CHRONICLE_COUNTY_FACTS"

#: The county rows on the pinned inputs: their counts, and the sha256 of the
#: rows as a registry file writes them (keys in insertion order), of the
#: populations (county code and the bits of the value) and of the receipts.
#: Measured on the code of microcosm#1760 after the county rows took the
#: operator's metadata keys.
#: ``test_the_pinned_county_rows_are_the_old_rows`` holds the same rows to
#: the code before that: the same values, bit for bit.
PINNED_COUNTY_ROWS: dict[str, object] = {
    "population_county_fips": {
        "counties": 3143,
        "sha256": "5a93db95e287a549ce4fb01126475382765e55f70a8a6e47670b63afb342c2e1",
    },
    "population_county_fips_2023": {
        "counties": 3144,
        "sha256": "636f344557d3b06b8c3363d10683df251aea099bfa7368346f3c4af294e5a2f9",
    },
    "soi": {
        "rows": 6286,
        "rows_sha256": "dce48ac26c326d3c11c01908a7a9ad3aa6fc43b88acf87037fc85271b9ddaf46",
        "receipt_sha256": "3f2e1ae71564973455aff2c882af482e2ae88ce3aa47eb653e0b434e2e439e45",
    },
    "households": {
        "rows": 3195,
        "rows_sha256": "ad642af8198354d8508e6d581a6028fc89c8beac6d9339021cdce9fee2df3430",
        "receipt_sha256": "8646315a07cb55789cb3a4d0c37fef0776d25d19f91096018a64fcc3fd73c01d",
    },
}


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _rows_sha256(specs) -> str:
    return _sha256(json.dumps([spec.to_dict() for spec in specs], indent=1))


def _population_sha256(cells) -> str:
    return _sha256(
        json.dumps([[int(code), float(value).hex()] for code, value in cells.items()])
    )


@pytest.fixture(scope="module")
def pinned_county(local_tool):
    """The three county passes on the pinned inputs, by the new and the old code."""

    paths = {}
    for name, what in (
        ("MICROCOSM_US_CHRONICLE_FACTS", "the pinned 41b2fb2 feed"),
        ("MICROCOSM_US_PUMA_LADDER_ARTIFACT", "the pinned PUMA ladder"),
        (_COUNTY_FACTS_ENV, "the pinned county facts file"),
    ):
        value = os.environ.get(name, "")
        if not value or not Path(value).exists():
            pytest.skip(f"set {name} to {what}")
        paths[name] = Path(value)
    from microcosm.build.us_runtime.county_facts import load_us_county_facts
    from microcosm.build.us_runtime.puma_ladder import load_us_puma_ladder

    county = local_tool.county_surface
    ladder_path = paths["MICROCOSM_US_PUMA_LADDER_ARTIFACT"]
    ladder = load_us_puma_ladder(ladder_path)
    states = local_tool.ladder_population(ladder_path, ["state"])
    populations = {
        column: tuple(
            module.with_county_populations(
                ladder, states, ["state", "county"], county_column=column
            )
            for module in (county, reference)
        )
        for column in _COUNTY_COLUMNS
    }
    state_specs = local_tool.state_admin_surface(
        str(paths["MICROCOSM_US_CHRONICLE_FACTS"]),
        ["snap", "medicaid", "soi"],
        soi_mode=county.SOI_MODE,
    ).registry.specs
    # The loader refuses a file that is not the pinned one.
    facts = load_us_county_facts(paths[_COUNTY_FACTS_ENV]).rows
    counties = populations["county_fips_2023"][0]["county"]
    return SimpleNamespace(
        populations=populations,
        soi=tuple(
            module.state_county_soi_surface(state_specs, facts, counties)
            for module in (county, reference)
        ),
        acs=tuple(
            module.acs_household_surface(facts, counties)
            for module in (county, reference)
        ),
    )


def test_the_pinned_county_rows_are_the_old_rows(pinned_county) -> None:
    """On the real inputs: the old rows and value bits, under the new keys."""

    for column in _COUNTY_COLUMNS:
        new, old = pinned_county.populations[column]
        assert _population_image(new) == _population_image(old), column
    _assert_same_surface(*pinned_county.soi, declarations.COUNTY_SOI, 0)
    _assert_same_surface(*pinned_county.acs, declarations.COUNTY_ACS_HOUSEHOLDS, 0)
    # The declared bound of the county bridge unbinds no leg on these inputs
    # (``_assert_same_surface`` holds ``bridge.unbound`` empty).
    # Kalawao has no IRS row: one Maui cell stands for both counties in each
    # of the two measures, so the SOI rows are two fewer than two per county.
    soi, acs = pinned_county.soi[0], pinned_county.acs[0]
    combined = [spec for spec in soi.specs if MEMBERS_KEY in spec.metadata]
    assert {spec.metadata[MEMBERS_KEY] for spec in combined} == {"15005,15009"}
    assert len(soi.specs) == 2 * 3_144 - len(combined) == 6_286
    assert len(acs.specs) == 51 + 3_144


def test_the_pinned_county_rows_have_their_pinned_bytes(pinned_county) -> None:
    soi, acs = pinned_county.soi[0], pinned_county.acs[0]
    measured = {
        **{
            f"population_{column}": {
                "counties": len(pinned_county.populations[column][0]["county"]),
                "sha256": _population_sha256(
                    pinned_county.populations[column][0]["county"]
                ),
            }
            for column in _COUNTY_COLUMNS
        },
        "soi": {
            "rows": len(soi.specs),
            "rows_sha256": _rows_sha256(soi.specs),
            "receipt_sha256": _sha256(json.dumps(soi.receipt, indent=1)),
        },
        "households": {
            "rows": len(acs.specs),
            "rows_sha256": _rows_sha256(acs.specs),
            "receipt_sha256": _sha256(json.dumps(acs.receipt, indent=1)),
        },
    }
    assert measured == PINNED_COUNTY_ROWS
