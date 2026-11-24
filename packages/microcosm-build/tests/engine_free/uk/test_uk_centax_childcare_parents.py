"""CenTax's parent tabulations as Microcosm-derived facts (microcosm#1399).

Invariants, for every input the strategies draw:

- the three ANI band shares sum to one, so the bands partition the share base
  (£50k, £200k], and every split share is below the (£80k, £150k] share;
- every committed value is its share times the baseline parent count;
- the aged edges are the 2021-22 edges times the uprating, rounded, so the
  bands stay contiguous and increasing from the aged £50k and move up with the
  uprating;
- the parent measure is 0 or 1, never 1 for a child, the same for every adult
  of a benefit unit, 0 in a unit with no eligible child, and follows its
  people when they are reordered;
- the main-source measure is 0 or 1, 0 without positive pay, never turns off
  as pay rises, and ignores the scale of all sources together;
- the region groups partition England's nine regions with the East of England
  in either group outside London;
- the diagnostic's support over FRS source households never exceeds its
  support over Microcosm households (fewer sources, a lower Kish count, a
  larger top-ten share), and equals it when each household is its own source;
- the workbook is the one Chronicle registers, with the pinned bytes;
- the contract binds exactly the resource's cells, with its edges.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from microcosm.build.uk_runtime.centax_childcare_parents import (
    CENTAX_ANI_BANDS_2022,
    CENTAX_ANI_MEASURE,
    CENTAX_BOUND_REGION_GROUPS,
    CENTAX_CHRONICLE_PACKAGE_ID,
    CENTAX_EAST_OF_ENGLAND,
    CENTAX_EMPLOYEE_MAIN_SOURCE,
    CENTAX_EMPLOYEE_SOURCE_COMPONENTS,
    CENTAX_EMPLOYEE_SOURCE_INPUTS,
    CENTAX_FACTS_RESOURCE,
    CENTAX_FILTER_CONCEPTS,
    CENTAX_PARENT_OF_ELIGIBLE_3_4,
    CENTAX_REGION_GROUP_CODES,
    CENTAX_SHARE_BASE_2022,
    CENTAX_SPLIT_BAND_2022,
    CENTAX_SPLITS,
    CENTAX_WORKBOOK_FILENAME,
    CENTAX_WORKBOOK_SHA256,
    aged_edges,
    centax_cell_filters,
    centax_cell_ids,
    centax_cell_measurement_filters,
    centax_region_codes,
    employment_main_income_source,
    load_uk_centax_childcare_parent_facts,
    model_ani_edges,
    parent_of_eligible_3_4_year_old,
    rows_sha256,
)
from microcosm.build.uk_runtime.derived_facts import (
    load_uk_derived_facts,
    uk_derived_fact_resource_pins,
    uk_derived_fact_rows,
)
from microcosm.build.uk_runtime.ledger_targets import (
    _uk_contract_targets,
    uk_contract_target_calibration_lines,
)
from test_support.paths import paths_for

ROOT = paths_for("microcosm-build").repository
_BUILDER_SPEC = importlib.util.spec_from_file_location(
    "build_uk_centax_childcare_parent_facts",
    ROOT / "tools/build_uk_centax_childcare_parent_facts.py",
)
assert _BUILDER_SPEC is not None and _BUILDER_SPEC.loader is not None
_builder = importlib.util.module_from_spec(_BUILDER_SPEC)
_BUILDER_SPEC.loader.exec_module(_builder)
_DIAGNOSTIC_SPEC = importlib.util.spec_from_file_location(
    "diagnose_uk_centax_parents", ROOT / "tools/diagnose_uk_centax_parents.py"
)
assert _DIAGNOSTIC_SPEC is not None and _DIAGNOSTIC_SPEC.loader is not None
_diagnostic = importlib.util.module_from_spec(_DIAGNOSTIC_SPEC)
_DIAGNOSTIC_SPEC.loader.exec_module(_diagnostic)

_PREFIX = "microcosm.centax_childcare_parents."
_BANDS = tuple(band for band, _, _ in CENTAX_ANI_BANDS_2022)
_SPLIT_CELLS = tuple(f"{split}_ani_80k_to_150k" for split, _ in CENTAX_SPLITS)
#: The cells of the first design that the review left as diagnostics.
_DIAGNOSTIC_CELLS = (
    "ani_0_to_50k",
    "south_excl_london_ani_80k_to_150k",
    "north_and_midlands_ani_80k_to_150k",
)
_ENGLISH_REGIONS = {f"E1200000{index}" for index in range(1, 10)}


def _resource_payload() -> dict:
    from importlib.resources import files

    return json.loads(
        files("microcosm.build.uk").joinpath(CENTAX_FACTS_RESOURCE).read_text()
    )


def _write(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / CENTAX_FACTS_RESOURCE
    path.write_text(json.dumps(payload))
    return path


# The committed resource.


def test_the_committed_resource_partitions_and_scales_its_shares() -> None:
    facts = load_uk_centax_childcare_parent_facts()
    shares = facts.shares

    assert set(shares) == set(centax_cell_ids()) == {*_BANDS, *_SPLIT_CELLS}
    assert sum(shares[band] for band in _BANDS) == pytest.approx(1.0, abs=1e-12)
    split_share = facts.parameters["split_band_share"]
    for cell in _SPLIT_CELLS:
        assert 0.0 < shares[cell] < split_share
    for cell in centax_cell_ids():
        assert facts.value(cell) == pytest.approx(
            round(shares[cell] * facts.parent_count, 3), abs=1e-9
        )
    # The bands' counts add back to the parent count the shares scale.
    assert sum(facts.value(band) for band in _BANDS) == pytest.approx(
        facts.parent_count, abs=0.01
    )


def test_the_committed_resource_ages_its_edges_contiguously() -> None:
    facts = load_uk_centax_childcare_parent_facts()
    edges = facts.parameters["model_ani_edges"]
    uprating = facts.ani_uprating

    assert edges == model_ani_edges(uprating)
    assert [edges[band][0] for band in _BANDS[1:]] == [
        edges[band][1] for band in _BANDS[:-1]
    ]
    # The bands cover the aged share base and nothing below it.
    assert [edges[_BANDS[0]][0], edges[_BANDS[-1]][1]] == aged_edges(
        *CENTAX_SHARE_BASE_2022, uprating
    )
    split = edges[_SPLIT_CELLS[0]]
    assert all(edges[cell] == split for cell in _SPLIT_CELLS)
    # The split band starts where the (£80k, £120k] band does.
    assert split[0] == edges["ani_80k_to_120k"][0]
    assert split[1] < edges["ani_120k_to_200k"][1]


def test_the_committed_resource_names_its_registration_and_baseline() -> None:
    payload = _resource_payload()
    facts = load_uk_centax_childcare_parent_facts()
    workbook = facts.inputs["workbook"]

    assert payload["rows_sha256"] == rows_sha256(payload["rows"])
    assert workbook["chronicle_package_id"] == CENTAX_CHRONICLE_PACKAGE_ID
    assert workbook["registration_kind"] == "raw_only"
    (registered,) = workbook["files"].values()
    assert registered["filename"] == CENTAX_WORKBOOK_FILENAME
    assert registered["sha256"] == CENTAX_WORKBOOK_SHA256
    assert registered["r2_key"].startswith(
        f"raw/uk/centax/{CENTAX_CHRONICLE_PACKAGE_ID}/"
    )
    assert workbook["source_page"] == (
        "https://arunadvani.com/policy.html#ChildcareCliffEdge"
    )
    # The attribution is the registration's, not typed here.
    assert payload["licence"]["attribution"] == [workbook["licence"]["attribution"]]
    assert "Flew, J." in workbook["licence"]["attribution"]
    assert len(facts.baseline["sha256"]) == 64 and facts.baseline["dataset"]
    assert payload["parameters"]["parent_count_basis"].endswith(
        f"at {facts.parameters['calibration_year']}"
    )


def test_the_committed_resource_holds_only_its_derivation() -> None:
    facts = load_uk_centax_childcare_parent_facts()
    # Vahid's review, item 8: the diagnostics-only figures left the resource.
    assert "diagnostic_comparators" not in facts.parameters
    assert "workbook_sums" not in facts.parameters
    assert {row["source"]["url"] for row in facts.rows} == {
        "https://centax.org.uk/wp-content/uploads/2026/09/"
        "AdvaniFlewPepin-HallSummers2026_Removing-the-childcare-cliff-edge.pdf"
    }
    tables = {row["source"]["source_table"] for row in facts.rows}
    assert all(
        any(f"Fig. {figure}" in table for figure in ("3.1", "3.3", "3.6", "3.7"))
        and CENTAX_CHRONICLE_PACKAGE_ID in table
        for table in tables
    )
    never = " ".join(facts.parameters["never_targets"])
    for excluded in ("3.1, 3.2 and 4.1", "elasticities", "cost"):
        assert excluded in never
    assert {row["geography"]["id"] for row in facts.rows} == {"E92000001"}
    # Item 7: the cells count the 2025-26 tax year, which the engine dates 2025.
    year = int(facts.parameters["calibration_year"])
    for row in facts.rows:
        assert row["period"] == {"type": "calendar_year", "value": str(year)}
        assert row["period_coverage"]["start_date"] == f"{year}-04-06"
        assert row["period_coverage"]["end_date"] == f"{year + 1}-04-05"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p["rows"][0].__setitem__("value", 1.0), "digest"),
        (lambda p: p["rows"].pop(), "row_count"),
        (
            lambda p: next(iter(p["inputs"]["workbook"]["files"].values())).__setitem__(
                "sha256", "0" * 64
            ),
            "pinned CenTax workbook",
        ),
        (
            lambda p: p["inputs"]["workbook"].__setitem__(
                "chronicle_package_id", "centax-another-package"
            ),
            "Chronicle registration",
        ),
        (
            lambda p: p["inputs"]["workbook"].__setitem__(
                "registration_kind", "committed"
            ),
            "raw-only",
        ),
        (lambda p: p["inputs"]["baseline"].__setitem__("sha256", ""), "baseline"),
        (
            lambda p: p["parameters"].__setitem__("ani_uprating", 1.25),
            "edges are not",
        ),
        (
            lambda p: p["parameters"]["shares"].__setitem__("ani_50k_to_80k", 0.5),
            "sum to",
        ),
        (
            lambda p: p["parameters"].__setitem__("share_base_2022", [0.0, 200_000.0]),
            "taken within",
        ),
        (lambda p: p.__setitem__("schema_version", 1), "schema-2"),
    ],
)
def test_the_loader_refuses_a_tampered_resource(tmp_path, mutate, message) -> None:
    payload = _resource_payload()
    mutate(payload)
    with pytest.raises(ValueError, match=message):
        load_uk_centax_childcare_parent_facts(_write(tmp_path, payload))


def test_the_loader_refuses_a_value_off_its_share(tmp_path) -> None:
    payload = _resource_payload()
    payload["rows"][0]["value"] = round(payload["rows"][0]["value"] + 1.0, 3)
    payload["rows_sha256"] = rows_sha256(payload["rows"])
    with pytest.raises(ValueError, match="share times the parent count"):
        load_uk_centax_childcare_parent_facts(_write(tmp_path, payload))


def test_the_loader_refuses_a_split_above_its_band(tmp_path) -> None:
    payload = _resource_payload()
    payload["parameters"]["shares"]["london_ani_80k_to_150k"] = payload["parameters"][
        "split_band_share"
    ]
    with pytest.raises(ValueError, match="exceeds"):
        load_uk_centax_childcare_parent_facts(_write(tmp_path, payload))


def test_the_loader_refuses_a_repeated_cell(tmp_path) -> None:
    payload = _resource_payload()
    payload["rows"][1] = copy.deepcopy(payload["rows"][0])
    payload["rows_sha256"] = rows_sha256(payload["rows"])
    with pytest.raises(ValueError, match="repeated"):
        load_uk_centax_childcare_parent_facts(_write(tmp_path, payload))


@settings(max_examples=40, deadline=None)
@given(st.permutations(range(6)))
def test_the_rows_digest_ignores_row_order(order) -> None:
    rows = _resource_payload()["rows"]
    assert rows_sha256([rows[i] for i in order]) == rows_sha256(rows)


# The aged edges.


@settings(max_examples=200, deadline=None)
@given(st.floats(1.0001, 1.9999, allow_nan=False), st.floats(0.0, 0.5))
def test_aged_edges_stay_contiguous_and_rise_with_the_uprating(uprating, step):
    edges = model_ani_edges(uprating)
    higher = model_ani_edges(min(uprating + step, 1.9999))

    for band, lower, upper in CENTAX_ANI_BANDS_2022:
        assert edges[band] == [round(lower * uprating), round(upper * uprating)]
        assert edges[band][0] < edges[band][1]
        assert higher[band][0] >= edges[band][0]
        assert higher[band][1] >= edges[band][1]
    assert [edges[band][0] for band in _BANDS[1:]] == [
        edges[band][1] for band in _BANDS[:-1]
    ]
    assert [edges[_BANDS[0]][0], edges[_BANDS[-1]][1]] == aged_edges(
        *CENTAX_SHARE_BASE_2022, uprating
    )
    lower, upper = CENTAX_SPLIT_BAND_2022
    assert all(
        edges[cell] == [round(lower * uprating), round(upper * uprating)]
        for cell in _SPLIT_CELLS
    )


def test_cell_filters_carry_the_edges_and_one_split_each() -> None:
    edges = model_ani_edges(1.25)
    for cell in centax_cell_ids():
        filters = centax_cell_filters(cell, edges[cell])
        assert filters[0] == {"variable": CENTAX_PARENT_OF_ELIGIBLE_3_4, "equals": 1}
        assert filters[1:3] == [
            {"variable": CENTAX_ANI_MEASURE, "operator": ">", "value": edges[cell][0]},
            {
                "variable": CENTAX_ANI_MEASURE,
                "operator": "<=",
                "value": edges[cell][1],
            },
        ]
        assert len(filters) == (3 if cell in _BANDS else 4)
        # The measurement filters are the binding filters with each variable
        # named by its concept.
        measured = centax_cell_measurement_filters(cell, edges[cell])
        assert measured == [
            {
                "concept": CENTAX_FILTER_CONCEPTS[f["variable"]],
                **{k: v for k, v in f.items() if k != "variable"},
            }
            for f in filters
        ]
    assert centax_cell_filters("london_ani_80k_to_150k", [1, 2])[-1] == {
        "variable": "region",
        "in": ["LONDON"],
    }
    # The cells the review left as diagnostics are no cells of the resource.
    for cell in _DIAGNOSTIC_CELLS:
        with pytest.raises(KeyError):
            centax_cell_filters(cell, [0, 1])


@pytest.mark.parametrize("east_in", ["south_excl_london", "north_and_midlands"])
def test_the_region_groups_partition_england_with_the_east_either_way(east_in):
    groups = centax_region_codes(east_in=east_in)
    codes = [code for group in groups.values() for code in group]
    assert sorted(codes) == sorted(_ENGLISH_REGIONS)
    assert CENTAX_EAST_OF_ENGLAND in groups[east_in]
    assert groups["london"] == CENTAX_REGION_GROUP_CODES["london"]
    assert CENTAX_BOUND_REGION_GROUPS == ("london",)
    with pytest.raises(ValueError, match="cannot sit"):
        centax_region_codes(east_in="london")


# The measures.

_people = st.integers(1, 12)


@st.composite
def _population(draw):
    n = draw(_people)
    return {
        "birth_year": np.asarray(
            draw(st.lists(st.integers(1960, 2025), min_size=n, max_size=n))
        ),
        "country": np.asarray(
            draw(
                st.lists(
                    st.sampled_from(["ENGLAND", "SCOTLAND", "WALES"]),
                    min_size=n,
                    max_size=n,
                )
            )
        ),
        "term_share_before_school": np.asarray(
            draw(st.lists(st.sampled_from([0.0, 1 / 3, 1.0]), min_size=n, max_size=n)),
            dtype=np.float32,
        ),
        "dependent_child": np.asarray(
            draw(st.lists(st.booleans(), min_size=n, max_size=n))
        ),
        "benunit_id": np.asarray(
            draw(st.lists(st.integers(0, 3), min_size=n, max_size=n))
        ),
    }


@settings(max_examples=200, deadline=None)
@given(_population(), st.randoms(use_true_random=False))
def test_the_parent_measure_marks_whole_units_of_adults(population, rng) -> None:
    parent = parent_of_eligible_3_4_year_old(**population, year=2025)
    child = population["dependent_child"]
    units = population["benunit_id"]
    dfe_age = 2025 - population["birth_year"]
    eligible = (
        child
        & (population["country"] == "ENGLAND")
        & np.isin(dfe_age, (3, 4, 5))
        & (population["term_share_before_school"] > 1 / 6)
    )

    assert set(np.unique(parent)) <= {0.0, 1.0}
    assert not parent[child].any()
    for unit in np.unique(units):
        adults = (units == unit) & ~child
        expected = 1.0 if eligible[units == unit].any() else 0.0
        assert (parent[adults] == expected).all()

    order = list(range(len(units)))
    rng.shuffle(order)
    shuffled = parent_of_eligible_3_4_year_old(
        **{name: values[order] for name, values in population.items()}, year=2025
    )
    np.testing.assert_array_equal(shuffled, parent[order])


def test_the_parent_measure_reads_the_dfe_age_and_term_share() -> None:
    # A couple with a DfE-age-5 child before school in one term, a lone
    # parent with a DfE-age-5 child already at school all year, a parent of a
    # DfE-age-2 child, and a Welsh parent of a 3-year-old.
    parent = parent_of_eligible_3_4_year_old(
        birth_year=np.asarray([1990, 1991, 2020, 1985, 2020, 1992, 2023, 1993, 2022]),
        country=np.asarray(["ENGLAND"] * 7 + ["WALES"] * 2),
        term_share_before_school=np.asarray(
            [1, 1, 1 / 3, 1, 0, 1, 1, 1, 1], dtype=np.float32
        ),
        dependent_child=np.asarray([0, 0, 1, 0, 1, 0, 1, 0, 1], dtype=bool),
        benunit_id=np.asarray([1, 1, 1, 2, 2, 3, 3, 4, 4]),
        year=2025,
    )
    assert parent.tolist() == [1, 1, 0, 0, 0, 0, 0, 0, 0]


def test_the_compared_sources_are_the_ani_components_employment_first() -> None:
    assert CENTAX_EMPLOYEE_SOURCE_INPUTS == tuple(
        name
        for sources in CENTAX_EMPLOYEE_SOURCE_COMPONENTS.values()
        for name in sources
    )
    assert CENTAX_EMPLOYEE_SOURCE_COMPONENTS["taxable_employment_income"] == (
        CENTAX_EMPLOYEE_SOURCE_INPUTS[0],
    )
    assert CENTAX_EMPLOYEE_SOURCE_INPUTS[0] == (
        "employment_income_with_benefits_in_kind_annual"
    )
    assert len(set(CENTAX_EMPLOYEE_SOURCE_INPUTS)) == len(CENTAX_EMPLOYEE_SOURCE_INPUTS)


_sources = st.lists(
    st.floats(-5_000.0, 300_000.0, allow_nan=False),
    min_size=len(CENTAX_EMPLOYEE_SOURCE_INPUTS),
    max_size=len(CENTAX_EMPLOYEE_SOURCE_INPUTS),
)


@settings(max_examples=300, deadline=None)
@given(st.lists(_sources, min_size=1, max_size=8), st.floats(0.0, 50_000.0))
def test_the_main_source_measure_is_monotone_in_pay(rows, raise_pay) -> None:
    matrix = np.asarray(rows, dtype=float)
    sources = {
        name: matrix[:, column]
        for column, name in enumerate(CENTAX_EMPLOYEE_SOURCE_INPUTS)
    }
    employee = employment_main_income_source(sources)
    pay = sources[CENTAX_EMPLOYEE_SOURCE_INPUTS[0]]

    assert set(np.unique(employee)) <= {0.0, 1.0}
    assert not employee[pay <= 0].any()
    raised = employment_main_income_source(
        {**sources, CENTAX_EMPLOYEE_SOURCE_INPUTS[0]: pay + raise_pay}
    )
    assert (raised >= employee).all()
    scaled = employment_main_income_source(
        {name: values * 3.0 for name, values in sources.items()}
    )
    np.testing.assert_array_equal(scaled, employee)


@pytest.mark.parametrize("source", CENTAX_EMPLOYEE_SOURCE_INPUTS[1:])
def test_every_other_source_can_outrank_employment(source) -> None:
    # Pay of 30,000 against 100,000 from one other source: that source is the
    # main one, whichever it is; at 100,000 each, the tie goes to employment.
    zeros = {name: np.zeros(2) for name in CENTAX_EMPLOYEE_SOURCE_INPUTS}
    sources = {
        **zeros,
        CENTAX_EMPLOYEE_SOURCE_INPUTS[0]: np.array([30_000.0, 100_000.0]),
        source: np.array([100_000.0, 100_000.0]),
    }
    assert employment_main_income_source(sources).tolist() == [0.0, 1.0]


def test_the_main_source_measure_refuses_a_missing_source() -> None:
    with pytest.raises(ValueError, match="needs"):
        employment_main_income_source(
            {"employment_income_with_benefits_in_kind_annual": np.ones(2)}
        )


# The diagnostic.


@st.composite
def _cell_people(draw):
    """People of up to a dozen households, each a copy of one source household."""

    households = draw(st.integers(1, 12))
    sized = {"min_size": households, "max_size": households}
    source = np.asarray(draw(st.lists(st.integers(0, households - 1), **sized)))
    weight = np.asarray(draw(st.lists(st.floats(0.5, 500.0), **sized)))
    members = draw(st.lists(st.integers(1, 3), **sized))
    household = np.repeat(np.arange(households), members)
    people = len(household)
    in_cell = np.asarray(
        draw(st.lists(st.booleans(), min_size=people, max_size=people))
    )
    return in_cell, weight[household], household, source[household]


@settings(max_examples=200, deadline=None)
@given(_cell_people())
def test_source_support_never_exceeds_household_support(people) -> None:
    in_cell, weight, household, source = people
    support = _diagnostic._support(in_cell, weight, household, source)
    if not in_cell.any():
        assert support["source_households"] == 0
        return
    assert support["source_households"] <= support["households"]
    assert support["kish_source_households"] <= support["kish_contributions"] * (
        1 + 1e-12
    )
    assert support["top10_source_share"] >= support["top10_household_share"] - 1e-12
    own = _diagnostic._support(in_cell, weight, household, household)
    assert own["source_households"] == own["households"]
    assert own["kish_source_households"] == pytest.approx(own["kish_contributions"])
    assert own["top10_source_share"] == pytest.approx(own["top10_household_share"])


def test_copies_of_one_source_household_count_once() -> None:
    # Households 0 and 1 copy source 7; household 2 is source 8. Over households
    # the cell's Kish count is 400^2 / (100^2 + 100^2 + 200^2) = 8/3; over
    # sources it is 400^2 / (200^2 + 200^2) = 2.
    support = _diagnostic._support(
        np.ones(3, dtype=bool),
        np.array([100.0, 100.0, 200.0]),
        np.array([0, 1, 2]),
        np.array([7, 7, 8]),
    )
    assert support["households"] == 3 and support["source_households"] == 2
    assert support["kish_contributions"] == pytest.approx(8 / 3)
    assert support["kish_source_households"] == pytest.approx(2.0)
    assert support["top10_source_share"] == pytest.approx(1.0)


def test_the_top_ten_share_counts_sources_not_households() -> None:
    # Twelve equal households, the first two copies of one source: the ten
    # heaviest households hold 10/12 of the cell, the ten heaviest of the 11
    # sources (the doubled one and nine others) hold 11/12 (review r3).
    support = _diagnostic._support(
        np.ones(12, dtype=bool),
        np.full(12, 50.0),
        np.arange(12),
        np.array([0, 0, *range(1, 11)]),
    )
    assert support["source_households"] == 11
    assert support["top10_household_share"] == pytest.approx(10 / 12)
    assert support["top10_source_share"] == pytest.approx(11 / 12)


@pytest.mark.parametrize(
    ("count", "within"),
    [(1_000.0, True), (1_019.0, True), (1_021.0, False), (979.0, False)],
)
def test_the_level_check_compares_the_parent_counts(count, within) -> None:
    # Vahid's review, item 2: a baseline that has moved from the resource's
    # count pulls the fit by the drift.
    check = _diagnostic.level_check(count, 1_000.0, 0.02)
    assert check["drift"] == pytest.approx(count / 1_000.0 - 1.0)
    assert check["within_tolerance"] is within
    with pytest.raises(ValueError, match="positive"):
        _diagnostic.level_check(count, 0.0, 0.02)


# The contract and the compile.


def test_the_contract_binds_the_resource_cells_on_the_national_line() -> None:
    facts = load_uk_centax_childcare_parent_facts()
    targets = {
        target_id: target
        for target_id, target in _uk_contract_targets(national_only=False).items()
        if target_id.startswith(_PREFIX)
    }
    specs = {
        row["dimensions"]["microcosm.centax_cell"]: row["layout"]["record_set_spec_id"]
        for row in facts.rows
    }

    assert {target_id.removeprefix(_PREFIX) for target_id in targets} == set(
        centax_cell_ids()
    )
    assert not {_PREFIX + cell for cell in _DIAGNOSTIC_CELLS} & set(targets)
    for target_id, target in targets.items():
        cell = target_id.removeprefix(_PREFIX)
        edges = facts.model_edges(cell)
        binding = target["bindings"]["policyengine"]
        assert binding["filters"] == centax_cell_filters(cell, edges)
        assert target["measurement"]["filters"] == centax_cell_measurement_filters(
            cell, edges
        )
        assert binding["value_variable"] == "person_count"
        assert target["ledger_selector"] == {
            "source_name": "microcosm",
            "source_measure_id": "parents_of_eligible_3_4_year_olds",
            "record_set_spec_id": specs[cell],
        }
        assert uk_contract_target_calibration_lines(target) == {"national"}
        assert target["family"] == "centax_childcare_parents"
        assert target["geography_levels"] == ["country"]
        variables = {f["variable"] for f in binding["filters"]}
        assert variables <= {
            CENTAX_PARENT_OF_ELIGIBLE_3_4,
            CENTAX_ANI_MEASURE,
            CENTAX_EMPLOYEE_MAIN_SOURCE,
            "gender",
            "region",
        }


def test_every_compile_reads_the_centax_resource_with_the_others() -> None:
    centax = load_uk_centax_childcare_parent_facts()
    resources = load_uk_derived_facts()

    assert resources["centax_childcare_parent_facts"].rows == centax.rows
    rows = uk_derived_fact_rows()
    assert rows[-len(centax.rows) :] == centax.rows
    pins = uk_derived_fact_resource_pins()
    assert pins[CENTAX_FACTS_RESOURCE]["rows_sha256"] == centax.rows_sha256
    keys = [row["aggregate_fact_key"] for row in rows]
    assert len(keys) == len(set(keys))


def test_the_compile_sites_read_the_merged_rows() -> None:
    sites = (
        "packages/microcosm-build/src/microcosm/build/uk_runtime/full_targets.py",
        "tools/build_uk_ledger_compile_parity_signed_differences.py",
        "tools/certify_uk_release_cut.py",
        "tools/diagnose_uk_uc_matrix.py",
        "tools/evaluate_uk_incumbent_surface.py",
    )
    for site in sites:
        text = (ROOT / site).read_text()
        assert "uk_derived_fact_rows(" in text, site
        assert "load_uk_high_value_home_stock_facts().rows" not in text, site
    generator = (ROOT / "tools/generate_uk_target_references.py").read_text()
    assert "load_uk_centax_childcare_parent_facts(args.centax_facts)" in generator


# The builder: the registration, the baseline and the arithmetic.


def _registration(tmp_path: Path, content: bytes, **overrides) -> tuple[Path, Path]:
    """A manifest registering ``content`` the way Chronicle's does, and the file."""

    workbook = tmp_path / CENTAX_WORKBOOK_FILENAME
    workbook.write_bytes(content)
    entry = {
        "filename": CENTAX_WORKBOOK_FILENAME,
        "source_url": f"https://arunadvani.com/data/{CENTAX_WORKBOOK_FILENAME}",
        "sha256": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
        "vintage": "centax_2026_09",
        "storage": {"local": False, "r2": {"key": "raw/uk/centax/x"}},
    }
    manifest = {
        "source_id": "centax",
        "package_id": CENTAX_CHRONICLE_PACKAGE_ID,
        "registration_kind": "raw_only",
        "license": "None stated",
        "files": {2026: entry},
        **overrides,
    }
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(manifest))
    return path, workbook


def test_the_builder_refuses_another_package(tmp_path) -> None:
    manifest, workbook = _registration(
        tmp_path, b"x", package_id="ons-nspl-august-2026"
    )
    with pytest.raises(ValueError, match="is not the centax-childcare"):
        _builder.open_registered_workbook(manifest, workbook)


def test_the_builder_refuses_bytes_the_manifest_does_not_pin(tmp_path) -> None:
    manifest, workbook = _registration(tmp_path, b"registered")
    workbook.write_bytes(b"something else")
    with pytest.raises(ValueError, match="size"):
        _builder.open_registered_workbook(manifest, workbook)


def test_the_builder_refuses_a_registration_of_other_bytes(tmp_path) -> None:
    # The manifest and the file agree, but not with the module's pin.
    manifest, workbook = _registration(tmp_path, b"not the CenTax workbook")
    with pytest.raises(ValueError, match="this module pins"):
        _builder.open_registered_workbook(manifest, workbook)


def _baseline(uprating: float, **overrides) -> dict:
    return {
        "schema": _builder.DIAGNOSTICS_SCHEMA,
        "year": 2025,
        "ani_uprating": uprating,
        "model_ani_edges": model_ani_edges(uprating),
        "parent_count_range": aged_edges(*CENTAX_SHARE_BASE_2022, uprating),
        "resolver": {"policyengine_uk_version": "9.9.9"},
        **overrides,
    }


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"schema": "microcosm.uk.centax_parent_diagnostics.v1"}, "not"),
        ({"year": 2024}, "at 2024"),
        ({"parent_count_range": [0, 249_543]}, "another ANI range"),
        ({"resolver": {"policyengine_uk_version": "1.0.0"}}, "installed"),
    ],
)
def test_the_builder_refuses_another_baseline(monkeypatch, overrides, message) -> None:
    monkeypatch.setattr(_builder.importlib.metadata, "version", lambda name: "9.9.9")
    _builder.check_baseline(_baseline(1.25), calibration_year=2025, uprating=1.25)
    with pytest.raises(ValueError, match=message):
        _builder.check_baseline(
            _baseline(1.25, **overrides), calibration_year=2025, uprating=1.25
        )


def test_a_fact_row_covers_the_tax_year_the_engine_dates_2025() -> None:
    row = _builder.fact_row(
        cell="london_ani_80k_to_150k",
        value=1234.5,
        period="2025",
        edges=[99_817, 187_158],
        inputs_sha256="0" * 64,
        vintage="centax_2026_09",
    )
    assert row["period"] == {"type": "calendar_year", "value": "2025"}
    assert row["period_coverage"]["start_date"] == "2025-04-06"
    assert row["period_coverage"]["end_date"] == "2026-04-05"
    assert "2025-26 tax year" in row["period_coverage"]["notes"]
    assert row["source"]["vintage"] == "centax_2026_09"


class _Sheet:
    def __init__(self, rows):
        self._rows = rows

    def iter_rows(self, *, values_only):
        assert values_only
        return iter(self._rows)


def _sheet(name: str, columns: list[list[float]]) -> _Sheet:
    header, step = _builder.SHEETS[name]
    rows = [(None, *header)]
    for index, values in enumerate(zip(*columns, strict=True), start=1):
        rows.append((None, index * step, *values))
    return _Sheet(rows)


@settings(max_examples=40, deadline=None)
@given(st.integers(0, 2**32 - 1), st.floats(1.0, 1e4), st.floats(1.0, 1e4))
def test_shares_partition_on_any_positive_workbook(seed, low, spread) -> None:
    rng = np.random.default_rng(seed)

    def column(length):
        # Positive bin counts; a lognormal draw gives spiky and flat shapes.
        return (low * rng.lognormal(0.0, np.log1p(spread) / 4, size=length)).tolist()

    workbook = {
        "Fig. 3.1": _sheet("Fig. 3.1", [column(400)]),
        "Fig. 3.3": _sheet("Fig. 3.3", [column(150), column(150)]),
        "Fig. 3.6": _sheet("Fig. 3.6", [column(300), column(300)]),
        "Fig. 3.7": _sheet("Fig. 3.7", [column(150) for _ in range(4)]),
    }
    shares, split_share = _builder.centax_shares(workbook)

    assert set(shares) == set(centax_cell_ids())
    assert sum(shares[band] for band in _BANDS) == pytest.approx(1.0, abs=1e-12)
    assert 0.0 < split_share < 1.0
    assert all(0.0 < share < 1.0 for share in shares.values())
    assert all(shares[cell] < split_share for cell in _SPLIT_CELLS)


def test_band_sums_take_bins_by_their_upper_edge() -> None:
    series = {500: 1.0, 1000: 2.0, 1500: 4.0}
    assert _builder.band_sum(series, 0.0, 1000.0) == 3.0
    assert _builder.band_sum(series, 500.0, 1500.0) == 6.0
    assert _builder.band_sum(series, 1000.0, 1000.0) == 0.0


def test_the_series_reader_refuses_a_gap_in_the_bins() -> None:
    header, _ = _builder.SHEETS["Fig. 3.1"]
    sheet = _Sheet([(None, *header), (None, 500, 1.0), (None, 1500, 1.0)])
    with pytest.raises(ValueError, match="bins are not every"):
        _builder.read_series({"Fig. 3.1": sheet}, "Fig. 3.1")
