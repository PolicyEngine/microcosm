"""Generated county partitions, reconciliation, support and holdout invariants."""

from __future__ import annotations

import copy
import importlib.util
import math
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from test_support.paths import paths_for

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

_SETTINGS = settings(max_examples=60, deadline=None)


@lru_cache
def _fixtures():
    path = (
        paths_for("microcosm-build").tests
        / "engine_free"
        / "us"
        / "test_us_acs_local_county_surface.py"
    )
    spec = importlib.util.spec_from_file_location(
        "county_surface_property_fixtures", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@st.composite
def _county_layouts(draw):
    states = [1, 6, 12, 36][: draw(st.integers(1, 4))]
    counties, raw_populations, state_populations = [], [], {}
    household_counties, household_sizes = [], []
    for state in states:
        count = draw(st.integers(1, 6))
        codes = [state * 1000 + 2 * index + 1 for index in range(count)]
        counties.extend(codes)
        raw_populations.extend(
            draw(st.lists(st.integers(1, 1_000_000), min_size=count, max_size=count))
        )
        state_populations[state] = float(draw(st.integers(1, 1_000_000_000)))
        household_counties.extend(codes)
        extra = draw(st.lists(st.sampled_from(codes), max_size=8))
        household_counties.extend(extra)
        household_sizes.extend(
            draw(
                st.lists(
                    st.integers(1, 8),
                    min_size=count + len(extra),
                    max_size=count + len(extra),
                )
            )
        )
    ladder = SimpleNamespace(
        puma=np.asarray([state * 100_000 + 1 for state in states]),
        puma_population=np.asarray([state_populations[state] for state in states]),
        county_overlap_county=np.asarray(counties),
        county_overlap_population=np.asarray(raw_populations, dtype=float),
    )
    return (
        ladder,
        state_populations,
        np.asarray(household_counties),
        np.asarray(household_sizes),
    )


@_SETTINGS
@given(_county_layouts())
def test_county_population_rows_partition_households_and_reconcile_to_states(
    layout,
) -> None:
    ladder, states, codes, sizes = layout
    county = _fixtures()._county()
    populations = county.with_county_populations(ladder, {"state": states}, ["county"])
    records, matrix, dropped = county.population_rows(
        pd.DataFrame({"county_fips": codes, "state_fips": codes // 1000}),
        sizes,
        populations,
        ["county"],
    )
    assert dropped == []
    # Exactly one county row contributes each household's size, with no
    # duplicate or omitted household membership.
    assert np.all(np.diff(matrix.tocsc().indptr) == 1)
    np.testing.assert_array_equal(np.asarray(matrix.sum(axis=0)), sizes)
    assert len(records) == len(set(codes))
    for state, target in states.items():
        values = [
            record["value"] for record in records if int(record["state_fips"]) == state
        ]
        assert math.fsum(values) == pytest.approx(target, rel=1e-12)


@_SETTINGS
@given(_county_layouts(), st.data())
def test_county_soi_shares_sum_to_one_and_preserve_parent_household_measures(
    layout, data
) -> None:
    _ladder, states, codes, _sizes = layout
    fixtures = _fixtures()
    county = fixtures._county()
    populations = {int(code): 100.0 for code in np.unique(codes)}
    parents, facts, rows = [], [], []
    for state in states:
        state_counties = sorted(code for code in populations if code // 1000 == state)
        for measure in ("return_count", "adjusted_gross_income"):
            parent_value = data.draw(st.integers(1, 1_000_000_000))
            parents.append(
                fixtures._state_spec(measure, parent_value, state=f"{state:02d}")
            )
            fact_values = data.draw(
                st.lists(
                    st.integers(1, 1_000_000),
                    min_size=len(state_counties),
                    max_size=len(state_counties),
                )
            )
            facts.extend(
                fixtures._fact(f"{code:05d}", measure, value)
                for code, value in zip(state_counties, fact_values, strict=True)
            )
            values = np.asarray(
                data.draw(
                    st.lists(
                        st.integers(-1_000_000, 1_000_000),
                        min_size=len(codes),
                        max_size=len(codes),
                    )
                ),
                dtype=np.float32,
            )
            rows.append(np.where(codes // 1000 == state, values, 0.0))
    surface = county.state_county_soi_surface(parents, facts, populations)
    state_matrix = sparse.csr_array(np.stack(rows))
    matrix = county.split_state_rows(
        parents,
        state_matrix,
        surface.specs,
        pd.DataFrame({"county_fips": codes, "state_fips": codes // 1000}),
    )
    original = {parent.name: (index, parent) for index, parent in enumerate(parents)}
    for parent_name, (parent_index, parent) in original.items():
        children = [
            (index, spec)
            for index, spec in enumerate(surface.specs)
            if spec.metadata["state_county_parent_target_name"] == parent_name
        ]
        assert len(children) == sum(
            code // 1000 == int(parent.metadata["state_fips"]) for code in populations
        )
        assert math.fsum(
            spec.value / parent.value for _index, spec in children
        ) == pytest.approx(1.0, rel=1e-12)
        assert math.fsum(spec.value for _index, spec in children) == pytest.approx(
            parent.value, rel=1e-12
        )
        np.testing.assert_array_equal(
            np.asarray(matrix[[index for index, _spec in children]].sum(axis=0)),
            state_matrix.toarray()[parent_index],
        )
        for index, spec in children:
            expected = np.where(
                codes == int(spec.metadata["county_fips"]), rows[parent_index], 0.0
            )
            np.testing.assert_array_equal(matrix[[index]].toarray()[0], expected)


@_SETTINGS
@given(_county_layouts())
def test_unsupported_counties_are_reported_and_require_an_explicit_capped_smoke(
    layout,
) -> None:
    ladder, states, codes, sizes = layout
    county = _fixtures()._county()
    populations = county.with_county_populations(ladder, {"state": states}, ["county"])
    unsupported = int(ladder.county_overlap_county[0])
    retained = codes != unsupported
    records, _matrix, dropped = county.population_rows(
        pd.DataFrame(
            {"county_fips": codes[retained], "state_fips": codes[retained] // 1000}
        ),
        sizes[retained],
        populations,
        ["county"],
    )
    expected = {f"pop_county_{code:05d}" for code in populations["county"]}
    assert {record["name"] for record in records} | set(dropped) == expected
    assert {record["name"] for record in records}.isdisjoint(dropped)
    assert dropped == [f"pop_county_{unsupported:05d}"]
    with pytest.raises((ValueError, SystemExit), match=f"{unsupported:05d}"):
        county.require_support(dropped)
    county.require_support(dropped, capped_smoke=True)


@_SETTINGS
@given(
    _county_layouts(),
    st.floats(0.0, 0.5),
    st.floats(0.0, 0.5),
    st.text(min_size=1, max_size=24),
    st.randoms(),
)
def test_county_holdout_is_deterministic_order_independent_and_nested(
    layout, first, second, salt, random
) -> None:
    _ladder, _states, codes, _sizes = layout
    county = _fixtures()._county()
    records = [
        {
            "name": f"county.{code:05d}.{measure}",
            "value": 1.0,
            "family": "irs_soi",
            "geography_level": "county",
            "state_fips": f"{code // 1000:02d}",
            "county_fips": f"{code:05d}",
            "source_measure_id": measure,
            "state_county_parent_target_name": f"state.{code // 1000}.{measure}",
        }
        for code in np.unique(codes)
        for measure in ("return_count", "adjusted_gross_income")
    ]
    low, high = sorted((first, second))
    original, shuffled, larger = (copy.deepcopy(records) for _ in range(3))
    random.shuffle(shuffled)
    county.assign_target_roles(original, fraction=low, salt=salt)
    county.assign_target_roles(shuffled, fraction=low, salt=salt)
    county.assign_target_roles(larger, fraction=high, salt=salt)

    def roles(rows):
        return {record["name"]: record["role"] for record in rows}

    assert roles(original) == roles(shuffled)

    def held(rows):
        return {record["name"] for record in rows if record["role"] == "holdout"}

    assert held(original) <= held(larger)
    # Holding out a whole state/concept block prevents the state row and
    # trained sibling counties from revealing the held-out county total.
    by_parent = {}
    for record in original:
        by_parent.setdefault(record["state_county_parent_target_name"], set()).add(
            record["role"]
        )
    assert all(len(values) == 1 for values in by_parent.values())


@_SETTINGS
@given(
    st.floats(0.0, 100_000.0, allow_nan=False, allow_infinity=False),
    st.floats(0.0, 100_000.0, allow_nan=False, allow_infinity=False),
    st.floats(0.0, 2.0, allow_nan=False, allow_infinity=False),
)
def test_reliability_grades_are_monotone_in_ess_for_fixed_error(
    first, second, error
) -> None:
    county = _fixtures()._county()
    low, high = sorted((first, second))
    rank = {"D": 0, "C": 1, "B": 2, "A": 3}
    assert (
        rank[county.reliability_grade(low, error)]
        <= rank[county.reliability_grade(high, error)]
    )
