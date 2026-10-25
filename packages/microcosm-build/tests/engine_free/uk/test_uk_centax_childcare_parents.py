"""CenTax's parent tabulations as Microcosm-derived facts (microcosm#1399).

Invariants, for every input the strategies draw:

- the four ANI band shares sum to one and the three region cells sum to the
  (£80k, £150k] share, so the bands partition (0, £200k] and the regions
  partition the split band;
- every committed value is its share times the baseline parent count;
- the aged edges are the 2021-22 edges times the uprating, rounded, so the
  bands stay contiguous and increasing and move up with the uprating;
- the parent measure is 0 or 1, never 1 for a child, the same for every adult
  of a benefit unit, 0 in a unit with no eligible child, and follows its
  people when they are reordered;
- the main-source measure is 0 or 1, 0 without positive pay, never turns off
  as pay rises, and ignores the scale of all sources together;
- the contract binds exactly the resource's cells, with its edges.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from microcosm.build.uk_runtime.centax_childcare_parents import (
    CENTAX_ANI_BANDS_2022,
    CENTAX_ANI_MEASURE,
    CENTAX_EMPLOYEE_MAIN_SOURCE,
    CENTAX_EMPLOYEE_SOURCE_INPUTS,
    CENTAX_FACTS_RESOURCE,
    CENTAX_FILTER_CONCEPTS,
    CENTAX_PARENT_OF_ELIGIBLE_3_4,
    CENTAX_REGION_GROUPS,
    CENTAX_SPLIT_BAND_2022,
    CENTAX_SPLITS,
    centax_cell_filters,
    centax_cell_ids,
    centax_cell_measurement_filters,
    employment_main_income_source,
    load_uk_centax_childcare_parent_facts,
    model_ani_edges,
    parent_of_eligible_3_4_year_old,
    rows_sha256,
)
from microcosm.build.uk_runtime.derived_facts import (
    uk_derived_fact_provenance,
    uk_derived_fact_rows,
)
from microcosm.build.uk_runtime.high_value_homes import (
    load_uk_high_value_home_stock_facts,
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

_PREFIX = "microcosm.centax_childcare_parents."
_BANDS = tuple(band for band, _, _ in CENTAX_ANI_BANDS_2022)
_SPLIT_CELLS = tuple(f"{split}_ani_80k_to_150k" for split, _ in CENTAX_SPLITS)
_REGION_CELLS = tuple(f"{group}_ani_80k_to_150k" for group in CENTAX_REGION_GROUPS)


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

    assert set(shares) == set(centax_cell_ids())
    assert sum(shares[band] for band in _BANDS) == pytest.approx(1.0, abs=1e-12)
    split_share = facts.parameters["split_band_share"]
    assert sum(shares[cell] for cell in _REGION_CELLS) == pytest.approx(
        split_share, abs=1e-12
    )
    for cell in _SPLIT_CELLS:
        assert 0.0 < shares[cell] <= split_share
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

    assert edges == model_ani_edges(facts.ani_uprating)
    assert [edges[band][0] for band in _BANDS[1:]] == [
        edges[band][1] for band in _BANDS[:-1]
    ]
    assert edges[_BANDS[0]][0] == 0
    split = edges[_SPLIT_CELLS[0]]
    assert all(edges[cell] == split for cell in _SPLIT_CELLS)
    # The split band starts where the (£80k, £120k] band does.
    assert split[0] == edges["ani_80k_to_120k"][0]
    assert split[1] < edges["ani_120k_to_200k"][1]


def test_the_committed_resource_names_its_sources_and_never_targets() -> None:
    payload = _resource_payload()
    facts = load_uk_centax_childcare_parent_facts()

    assert payload["rows_sha256"] == rows_sha256(payload["rows"])
    assert facts.inputs["workbook"]["sha256"] == (
        "8c23855745154a601064e766552c64bc2e88de51e043b4df1397c580e969354b"
    )
    assert {row["source"]["url"] for row in facts.rows} == {
        "https://centax.org.uk/wp-content/uploads/2026/09/"
        "AdvaniFlewPepin-HallSummers2026_Removing-the-childcare-cliff-edge.pdf"
    }
    tables = {row["source"]["source_table"] for row in facts.rows}
    assert all(
        any(f"Fig. {figure}" in table for figure in ("3.1", "3.3", "3.6", "3.7"))
        for table in tables
    )
    never = " ".join(facts.parameters["never_targets"])
    for excluded in ("3.1, 3.2 and 4.1", "elasticities", "cost"):
        assert excluded in never
    assert {row["geography"]["id"] for row in facts.rows} == {"E92000001"}


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p["rows"][0].__setitem__("value", 1.0), "digest"),
        (lambda p: p["rows"].pop(), "row_count"),
        (
            lambda p: p["inputs"]["workbook"].__setitem__("sha256", "0" * 64),
            "pinned CenTax workbook",
        ),
        (
            lambda p: p["parameters"].__setitem__("ani_uprating", 1.25),
            "edges are not",
        ),
        (
            lambda p: p["parameters"]["shares"].__setitem__("ani_0_to_50k", 0.5),
            "sum to",
        ),
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


def test_the_loader_refuses_a_repeated_cell(tmp_path) -> None:
    payload = _resource_payload()
    payload["rows"][1] = copy.deepcopy(payload["rows"][0])
    payload["rows_sha256"] = rows_sha256(payload["rows"])
    with pytest.raises(ValueError, match="repeated"):
        load_uk_centax_childcare_parent_facts(_write(tmp_path, payload))


@settings(max_examples=40, deadline=None)
@given(st.permutations(range(9)))
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
    with pytest.raises(KeyError):
        centax_cell_filters("ani_200k_plus", [0, 1])


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
        employment_main_income_source({"employment_income_annual": np.ones(2)})


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


def test_every_national_compile_merges_both_derived_resources() -> None:
    high_value = load_uk_high_value_home_stock_facts()
    centax = load_uk_centax_childcare_parent_facts()

    assert uk_derived_fact_rows() == (*high_value.rows, *centax.rows)
    provenance = uk_derived_fact_provenance()
    assert set(provenance) == {
        "high_value_home_stock_facts",
        "centax_childcare_parent_facts",
    }
    assert provenance["centax_childcare_parent_facts"]["rows_sha256"] == (
        centax.rows_sha256
    )
    keys = [row["aggregate_fact_key"] for row in uk_derived_fact_rows()]
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
        assert "uk_derived_fact_rows()" in text, site
        assert "load_uk_high_value_home_stock_facts().rows" not in text, site
    generator = (ROOT / "tools/generate_uk_target_references.py").read_text()
    assert "load_uk_centax_childcare_parent_facts(args.centax_facts)" in generator


# The builder's arithmetic on a synthetic workbook.


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
    shares, extra = _builder.centax_shares(workbook)
    split_share = extra["split_band_share"]

    assert set(shares) == set(centax_cell_ids())
    assert sum(shares[band] for band in _BANDS) == pytest.approx(1.0, abs=1e-12)
    assert sum(shares[cell] for cell in _REGION_CELLS) == pytest.approx(
        split_share, abs=1e-12
    )
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
