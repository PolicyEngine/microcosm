"""G8-x0 B: New Zealand's regional Census 2023 rent map.

Invariants of ``rent_bands.json`` and of ``transport.quantile_map@1`` reading it:

* merging zero-count bands conserves households, keeps bands contiguous and
  leaves no empty band (Hypothesis); an empty lowest band refuses;
* every region's bands tile zero to an open top on annual edges at 52 times a
  published weekly cut, and each reference names a distinct household count;
* on any population the map keeps zero rent at zero, preserves order within a
  region, meets each band's share within the tied-block bound, changes nothing
  on a rerun, and equals ``quantile_map`` run region by region (Hypothesis);
* the published counts compile from the pinned feed to the same shares, and the
  national Pareto alpha re-derives from the national counts.

``CENSUS`` copies the published counts so the kernel tests run without the feed.
The feed-gated test compares the copy with the pinned feed.
"""

from __future__ import annotations

import json
import os
from fractions import Fraction
from math import fsum
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from microcosm.build.ledger_artifact import load_ledger_consumer_artifact
from microcosm.build.ledger_targets import compile_ledger_target_references
from microcosm.build.tail_estimators import grouped_pareto_mle
from microcosm.build.transport.artifact_types import TARGET_SURFACE_TYPE
from microcosm.build.transport.population_kernels import (
    TRANSPORT_QUANTILE_MAP,
    _surface_bands,
)
from microcosm.build.transport.target_kernels import (
    compile_target_surface,
    parse_reference_document,
)
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from microcosm.frame.transport import quantile_map
from microcosm.graph import ArtifactInput, Node, Owned, Slice
from microcosm.graph.canonical import canonical_json
from microcosm.graph.executor import Population, _project_context
from microcosm.graph.kernel import ArtifactValue, NumericScope
from microcosm.graph.population import dtype_for_token
from test_support.microcosm_build.transport_graph import (
    COUNTRY,
    reference_document,
    reference_row,
    toy_fact,
    write_facts,
)
from test_support.microcosm_build.transport_population import digest
from test_support.paths import paths_for
from tools.generate_nz_target_references import (
    RENT_CONTRACT,
    RENT_DIMENSION,
    RENT_TAIL_BRACKET,
    RENT_TAIL_ITERATIONS,
    merge_zero_count_bands,
)

_SPEC = paths_for("microcosm-build").package / "src/microcosm/build/nz"

#: The 2023 Census counts of households by weekly rent, bands 1 to 8, per place.
CENSUS = {
    "01": (3549, 2463, 3486, 3732, 2697, 1248, 336, 147),
    "02": (28113, 9909, 16734, 28827, 42645, 37143, 19629, 22899),
    "03": (8184, 7368, 9795, 12243, 10779, 4917, 1299, 744),
    "04": (4521, 3114, 4650, 5976, 6912, 5349, 1575, 696),
    "05": (1722, 834, 1056, 927, 618, 351, 123, 78),
    "06": (4098, 2097, 2985, 3462, 2724, 1446, 465, 258),
    "07": (2277, 2148, 2751, 2559, 1425, 609, 174, 111),
    "08": (5820, 4395, 5898, 5451, 3537, 1176, 312, 249),
    "09": (8658, 4857, 7164, 9501, 10806, 8853, 5505, 6639),
    "12": (1017, 840, 855, 234, 57, 9, 0, 27),
    "13": (12450, 7692, 14568, 17946, 10554, 4218, 1239, 1464),
    "14": (4560, 3300, 4614, 4611, 3297, 2121, 1398, 2199),
    "15": (2445, 2457, 2844, 1827, 342, 75, 15, 48),
    "16": (558, 711, 810, 981, 756, 411, 72, 39),
    "17": (756, 429, 1005, 1554, 1170, 447, 90, 78),
    "18": (669, 699, 1032, 1233, 720, 165, 48, 42),
    "99": (39, 12, 3, 6, 3, 3, 0, 0),
    "NZ": (89439, 53325, 80247, 101067, 99042, 68544, 32286, 35712),
}
ALL_REGIONS = tuple(place for place in CENSUS if place != "NZ")
_SHARE_SLACK = 1e-12
_PROPERTY = settings(max_examples=40, deadline=None, database=None)
_SCHEMA = EntitySchema(group_entities=("household",))


def _document(name="rent_bands.json"):
    return json.loads((_SPEC / name).read_text())


def _name(region, code):
    return f"{RENT_CONTRACT}__r{region}__b{code}"


def _count(reference):
    region, code = reference.removeprefix(f"{RENT_CONTRACT}__r").split("__b")
    return CENSUS[region][int(code) - 1]


def _component(region):
    (bands,) = [
        row["bands"]
        for row in _document()["map_bands"]["components"]
        if row["value"] == region
    ]
    return bands


def _shares(region):
    """The declared bands with shares from the published counts."""
    rows = _component(region)
    counts = [float(_count(row["reference"])) for row in rows]
    total = fsum(counts)
    return [
        {"lower": row["lower"], "upper": row["upper"], "share": count / total}
        for row, count in zip(rows, counts, strict=True)
    ]


def _rent_rows():
    return [
        row
        for row in _document("precal_references.json")["target_references"]
        if row["metadata"].get("precal_use") == "rent_quantile_band"
    ]


# ---------------------------------------------------------------------------
# Merging zero-count bands
# ---------------------------------------------------------------------------


def _raw_bands(counts):
    return [
        {
            "reference": f"band_{index}",
            "lower": float(index),
            "upper": None if index == len(counts) - 1 else float(index + 1),
            "count": count,
        }
        for index, count in enumerate(counts)
    ]


_COUNTS = st.lists(st.integers(min_value=0, max_value=10**6), min_size=1, max_size=12)


@settings(max_examples=200, deadline=None, database=None)
@given(counts=_COUNTS)
def test_property_merging_conserves_households_and_keeps_bands_contiguous(counts):
    assume(counts[0] > 0)
    bands = _raw_bands(counts)
    snapshot = json.dumps(bands)
    kept, merges = merge_zero_count_bands(bands)
    assert json.dumps(bands) == snapshot
    assert sum(band["count"] for band in kept) == sum(counts)
    assert all(band["count"] > 0 for band in kept)
    assert kept[0]["lower"] == 0.0 and kept[-1]["upper"] is None
    for left, right in zip(kept, kept[1:], strict=False):
        assert left["upper"] == right["lower"]
    by_reference = {band["reference"]: band for band in bands}
    for band in kept:
        original = by_reference[band["reference"]]
        assert (band["lower"], band["count"]) == (original["lower"], original["count"])
    assert [merge["zero_count_reference"] for merge in merges] == [
        band["reference"] for band in bands if band["count"] == 0
    ]
    assert {merge["kept"] for merge in merges} <= {band["reference"] for band in kept}


@given(counts=_COUNTS)
def test_property_an_empty_lowest_band_has_nothing_to_merge_into(counts):
    with pytest.raises(ValueError, match="no lower band to merge into"):
        merge_zero_count_bands(_raw_bands([0, *counts]))


# ---------------------------------------------------------------------------
# The declaration
# ---------------------------------------------------------------------------


def test_every_region_tiles_zero_to_an_open_top_on_published_weekly_cuts():
    document = _document()
    assert document["weeks_per_year"] == 52
    assert document["component_column"] == "household_region_code"
    components = document["map_bands"]["components"]
    assert tuple(row["value"] for row in components) == ALL_REGIONS
    published = {0.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0}
    for component in components:
        bands = component["bands"]
        assert bands[0]["lower"] == 0.0
        assert bands[-1]["upper"] is None
        # transport.quantile_map refuses a Pareto top that starts at zero.
        assert bands[-1]["lower"] > 0
        for left, right in zip(bands, bands[1:], strict=False):
            assert left["upper"] == right["lower"] > left["lower"]
        assert {band["lower"] / 52 for band in bands} <= published
        assert all(set(band) == {"lower", "upper", "reference"} for band in bands)


def test_the_declared_regions_are_the_geography_supports_regions():
    support = _document("atomic_geography_support.json")
    assert {row["codes"]["region_code"] for row in support["rows"]} == set(ALL_REGIONS)


def test_each_band_reference_is_a_distinct_active_household_count_row():
    rows = {row["name"]: row for row in _rent_rows()}
    references = [band["reference"] for region in ALL_REGIONS for band in _component(region)]
    assert len(references) == len(set(references)) == 133
    assert set(references) == set(rows)
    for row in rows.values():
        assert row["entity"] == "household"
        assert not row.get("filter")
        assert row["family"] == "housing_costs"
        assert row["metadata"]["activation_status"] == "active"
        assert row["metadata"]["quantile_producer"] == "transport.quantile_map@1"
        assert row["ledger_selector"]["source_measure_id"] == "count"
        assert _count(row["name"]) > 0


def test_the_zero_count_merges_are_the_three_published_empty_bands():
    document = _document()
    assert document["zero_count_merges"] == [
        {"region": "12", "zero_count_reference": _name("12", 7), "kept": _name("12", 6)},
        {"region": "99", "zero_count_reference": _name("99", 7), "kept": _name("99", 6)},
        {"region": "99", "zero_count_reference": _name("99", 8), "kept": _name("99", 6)},
    ]
    assert [(band["lower"], band["upper"]) for band in _component("12")[-2:]] == [
        (31200.0, 41600.0),
        (41600.0, None),
    ]
    # The merged open top starts at $600 a week, where the alpha is fitted.
    assert [(band["lower"], band["upper"]) for band in _component("99")[-1:]] == [
        (31200.0, None)
    ]


def test_the_national_alpha_rederives_from_the_national_counts():
    document = _document()
    national = CENSUS["NZ"]
    fit = grouped_pareto_mle(
        [
            ("national_weekly_rent_band_6", 31200.0, 36400.0, float(national[5])),
            ("national_weekly_rent_band_7", 36400.0, 41600.0, float(national[6])),
            ("national_weekly_rent_band_8", 41600.0, None, float(national[7])),
        ],
        bracket=tuple(Fraction(text) for text in RENT_TAIL_BRACKET),
        iterations=RENT_TAIL_ITERATIONS,
    )
    assert document["interpolation"] == {"method": "pareto", "alpha": fit.alpha_float}
    assert fit.alpha_float > 1
    assert document["tail_fit"]["alpha_decimal"] == fit.receipt()["alpha_decimal"]
    assert document["tail_fit"]["bracket"] == list(RENT_TAIL_BRACKET)


def test_the_document_discloses_what_the_map_cannot_claim():
    disclosure = _document()["disclosure"]
    for phrase in (
        "times 52",
        "C1 rent ageing ruling remains open",
        "does not state the table's population",
        "the map does not change how many households pay rent",
    ):
        assert phrase in disclosure, phrase


# ---------------------------------------------------------------------------
# The kernel on the real declaration
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def surface(tmp_path_factory):
    """A compiled target surface holding the published counts under the real names."""
    names = [band["reference"] for region in ALL_REGIONS for band in _component(region)]
    document = reference_document(
        [reference_row(name, entity="household", measure="households") for name in names]
    )
    facts = write_facts(
        tmp_path_factory.mktemp("rent-facts") / "facts.jsonl",
        [toy_fact(name, float(_count(name)), entity="household") for name in names],
    )
    payload, _ = compile_target_surface(
        facts, document, country=COUNTRY, references_sha256=digest(document)
    )
    return ArtifactValue(
        payload=payload,
        type=TARGET_SURFACE_TYPE,
        key="a" * 64,
        producer_key="b" * 64,
        numerics=NumericScope(),
    )


def _frame(rents, regions, weights):
    ids = np.arange(1, len(rents) + 1, dtype=np.int64)
    return Frame(
        {
            "person": pd.DataFrame({"person_id": ids, "person_household_id": ids}),
            "household": pd.DataFrame(
                {
                    "household_id": ids,
                    "rent": np.asarray(rents, dtype=np.float64),
                    "household_region_code": pd.array(
                        list(regions), dtype=dtype_for_token("string")
                    ),
                }
            ),
        },
        _SCHEMA,
        {"household": Weights(np.asarray(weights, dtype=np.float64), WeightKind.DESIGN)},
    )


def _mapped(rents, regions, weights, surface):
    document = _document()
    node = Node(
        "nz.rent.map",
        "transport.quantile_map@1",
        inputs=(Slice("household", ("rent", "household_region_code")),),
        outputs=(Owned("household", "rent", "float64", rewrite=True),),
        params={
            "entity": "household",
            "column": "rent",
            "component_column": "household_region_code",
            "bands": canonical_json(document["map_bands"]).decode(),
            "interpolation": canonical_json(document["interpolation"]).decode(),
            "bands_sha256": digest(document),
        },
        artifact_inputs=(
            ArtifactInput("surface", "nz.facts.precal", "surface", TARGET_SURFACE_TYPE),
        ),
    )
    context = _project_context(
        node,
        Population.from_frame(_frame(rents, regions, weights), "nz.central"),
        key="0" * 64,
        sources={},
        tolerances={},
        numerics={},
        artifacts={"surface": surface},
    )
    return TRANSPORT_QUANTILE_MAP.run(context).columns["household", "rent"].to_numpy()


_RENT = st.one_of(st.just(0.0), st.integers(1, 150_000).map(float))


@st.composite
def populations(draw):
    """Every declared region holds a household, because the kernel requires it."""
    regions = [*ALL_REGIONS, *draw(st.lists(st.sampled_from(ALL_REGIONS), max_size=80))]
    size = len(regions)
    rents = draw(st.lists(_RENT, min_size=size, max_size=size))
    weights = draw(
        st.lists(st.floats(min_value=0.5, max_value=4.0), min_size=size, max_size=size)
    )
    return np.array(rents), np.array(regions, dtype=object), np.array(weights)


def _assert_region(rents, mapped, weights, rows):
    paying = rents > 0
    if not paying.any():
        return
    rents, mapped, weights = rents[paying], mapped[paying], weights[paying]
    order = np.argsort(rents, kind="stable")
    steps, moves = np.diff(rents[order]), np.diff(mapped[order])
    assert (moves[steps > 0] > 0).all()
    assert (moves[steps == 0] == 0).all()
    total = weights.sum()
    # A tied block is indivisible, so a band can miss its share by one block.
    largest = max(weights[rents == rent].sum() for rent in np.unique(rents)) / total
    for row in rows:
        lower = row["lower"]
        upper = np.inf if row["upper"] is None else row["upper"]
        inside = weights[(mapped > lower) & (mapped < upper)].sum() / total
        touching = weights[(mapped >= lower) & (mapped <= upper)].sum() / total
        assert inside <= row["share"] + largest + _SHARE_SLACK
        assert touching >= row["share"] - largest - _SHARE_SLACK


@_PROPERTY
@given(population=populations())
def test_property_the_map_holds_each_regions_declared_distribution(surface, population):
    rents, regions, weights = population
    mapped = _mapped(rents, regions, weights, surface)
    assert np.isfinite(mapped).all()
    assert (mapped[rents == 0] == 0).all()
    assert (mapped[rents > 0] > 0).all()
    for region in ALL_REGIONS:
        selected = regions == region
        _assert_region(
            rents[selected], mapped[selected], weights[selected], _shares(region)
        )


@_PROPERTY
@given(population=populations())
def test_property_rerunning_the_map_changes_nothing(surface, population):
    rents, regions, weights = population
    once = _mapped(rents, regions, weights, surface)
    assert _mapped(once, regions, weights, surface).tobytes() == once.tobytes()


@_PROPERTY
@given(population=populations())
def test_property_the_kernel_equals_quantile_map_run_region_by_region(surface, population):
    rents, regions, weights = population
    expected = np.empty(len(rents), dtype=np.float64)
    interpolation = _document()["interpolation"]
    for region in ALL_REGIONS:
        selected = regions == region
        expected[selected] = quantile_map(
            rents[selected],
            weights[selected],
            _shares(region),
            interpolation=interpolation,
        )
    assert _mapped(rents, regions, weights, surface).tobytes() == expected.tobytes()


def test_one_renter_in_a_region_takes_that_regions_median_band(surface):
    regions = np.array(ALL_REGIONS, dtype=object)
    size = len(regions)
    mapped = dict(
        zip(
            ALL_REGIONS,
            _mapped(np.full(size, 20_000.0), regions, np.ones(size), surface),
            strict=True,
        )
    )
    # Auckland's median household pays $500-$599 a week, West Coast's $200-$299,
    # and the 66 households outside any region under $200.
    assert 26_000.0 <= mapped["02"] < 31_200.0
    assert 10_400.0 <= mapped["12"] < 15_600.0
    assert 0.0 < mapped["99"] < 10_400.0


def test_a_declared_region_with_no_households_refuses(surface):
    """Intended: a real run fails here, early, if TA 067 draws no household."""
    regions = np.array([place for place in ALL_REGIONS if place != "99"], dtype=object)
    size = len(regions)
    with pytest.raises(ValueError, match="exactly cover"):
        _mapped(np.full(size, 12_000.0), regions, np.ones(size), surface)


# ---------------------------------------------------------------------------
# The pinned feed
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def artifact():
    path = os.environ.get("NZ_CHRONICLE_ARTIFACT")
    if not path:
        pytest.skip("NZ_CHRONICLE_ARTIFACT is unset; public NZ feed audit unavailable")
    return load_ledger_consumer_artifact(
        Path(path), feed_pin=_document("chronicle_feed.json")
    )


def test_the_published_counts_match_the_pinned_feed(artifact):
    for place, counts in CENSUS.items():
        scope = "nz" if place == "NZ" else f"rc_{place}"
        record_set = f"stats_nz.census_2023.{scope}.weekly_rent"
        cells = {
            fact["dimensions"][RENT_DIMENSION]: fact["value"]
            for fact in artifact.facts
            if fact.get("layout", {}).get("record_set_id") == record_set
            and fact["observed_measure"]["source_measure_id"] == "count"
        }
        assert sorted(cells) == list(range(1, 9)), place
        assert tuple(cells[code] for code in range(1, 9)) == counts, place


def test_the_compiled_surface_gives_each_region_its_published_shares(artifact):
    references = parse_reference_document(
        {"country": "nz", "target_references": _rent_rows()}, country="nz"
    )
    registry = compile_ledger_target_references(artifact.facts, references, country="nz")
    bands = _surface_bands(
        _document()["map_bands"], SimpleNamespace(registry=registry), unit="household"
    )
    assert set(bands) == {"components"}
    assert set(bands["components"]) == set(ALL_REGIONS)
    for region in ALL_REGIONS:
        assert bands["components"][region] == _shares(region), region
