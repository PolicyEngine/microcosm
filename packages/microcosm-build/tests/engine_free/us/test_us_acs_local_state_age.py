"""Census Vintage 2025 state age groups in the ACS local release (microcosm#1528).

Against Census estimates, state slices of the ACS local-area release counted
too many children and too few adults: the release calibrated to no age
target, and its state populations came from the 2020 Census. The ``age``
family binds fifteen Census Vintage 2025 age groups in every state and DC at
July 1, 2024 (PolicyEngine/chronicle#417), moves each state's population
target to the same estimates' total, and scales the state's congressional
districts with it, so a state's age groups, total and districts agree.

Properties these tests check for generated inputs:

- The compile fence partitions the feed. Without ``state_age_band_targets``
  the Vintage 2025 facts change no compiled target; with it the compile adds
  one target per age-group fact, at that fact's count, and changes no other.
- A state's fifteen groups tile ``[0, inf)`` and sum to its total, and the
  surface check names every state that breaks either.
- Moving to new state totals sets each state's level, keeps each district's
  share of its state, and makes each state's districts (and, on the county
  rung, its counties) sum to its total.
"""

from __future__ import annotations

import importlib.util
import json
import math
import shlex
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from microcosm.build.us_runtime.fiscal_targets import (
    US_STATE_AGE_BAND_RECORD_SET_SPEC_ID,
    compile_us_fiscal_target_registry,
    is_us_state_age_band_fact,
    us_state_age_band_population_totals,
)
from microcosm.calibrate import TargetRegistry, TargetSpec
from microcosm.calibrate.geography_constants import US_STATE_FIPS_TO_NAME
from test_support.microcosm_build.us_acs_local_reviewed_nulls import (
    load_tool_module,
)
from test_support.microcosm_build.us_fiscal_targets import (
    _dynamic_ledger_fact,
    packaged_reference_facts,
)
from test_support.paths import paths_for

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

_SETTINGS = settings(max_examples=60, deadline=None)
#: Each example compiles the packaged reference feed, so fewer examples.
_COMPILE_SETTINGS = settings(max_examples=8, deadline=None)

#: The package's fifteen age groups: value id, first age, first age after.
AGE_GROUPS = (
    *((f"age_{age}", age, age + 1) for age in range(6)),
    ("6_to_11", 6, 12),
    ("12_to_17", 12, 18),
    ("18_to_24", 18, 25),
    ("25_to_34", 25, 35),
    ("35_to_44", 35, 45),
    ("45_to_54", 45, 55),
    ("55_to_64", 55, 65),
    ("65_to_74", 65, 75),
    ("75_plus", 75, None),
)
#: The 50 states and DC, as two-digit FIPS codes.
STATES = tuple(sorted(US_STATE_FIPS_TO_NAME))


@lru_cache
def _tool():
    return load_tool_module()


def _fact(
    state: str,
    value_id: str,
    value: float,
    *,
    lower: int | None = None,
    upper: int | None = None,
    period: int = 2024,
    spec_id: str = US_STATE_AGE_BAND_RECORD_SET_SPEC_ID,
) -> dict[str, object]:
    """One fact as the Chronicle package emits it."""

    constraints = []
    if lower is not None:
        constraints.append({"variable": "age", "operator": ">=", "value": lower})
    if upper is not None:
        constraints.append({"variable": "age", "operator": "<", "value": upper})
    record_set_id = (
        f"census_pep.v2025.cy{period}.state_resident_population_age_bands.{state}"
    )
    fact = _dynamic_ledger_fact(
        source_record_id=f"{record_set_id}.{value_id}.population",
        source_name="census_pep",
        measure_id="population",
        value=value,
        period_value=period,
        geography_level="state",
        geography_id=f"0400000US{state}",
        groupby_dimension="age",
        groupby_value_id=value_id,
        layout_record_set_id=record_set_id,
        universe_constraints=constraints,
    )
    fact["layout"]["record_set_spec_id"] = spec_id
    return fact


def _state_facts(state: str, values, *, period: int = 2024) -> list[dict]:
    """One state's package facts: its total, then its fifteen groups."""

    facts = [_fact(state, "all", math.fsum(values), period=period)]
    for (value_id, lower, upper), value in zip(AGE_GROUPS, values, strict=True):
        facts.append(
            _fact(state, value_id, value, lower=lower, upper=upper, period=period)
        )
    return facts


def _values(seed: int) -> list[float]:
    return [float(1_000 * (seed + 1) + 37 * index) for index in range(15)]


def _compile(facts, **kwargs):
    return compile_us_fiscal_target_registry(
        [*packaged_reference_facts(), *facts],
        allow_unaged_dollar_targets=True,
        **kwargs,
    )


def _by_name(registry) -> dict[str, dict]:
    return {spec.name: spec.to_dict() for spec in registry.specs}


def _is_age_spec(spec) -> bool:
    return (
        spec.metadata.get("ledger_layout_record_set_spec_id")
        == US_STATE_AGE_BAND_RECORD_SET_SPEC_ID
    )


# --- Ledger facts and the compile fence -------------------------------------


def test_population_totals_read_each_states_all_ages_row() -> None:
    facts = [
        *_state_facts("06", _values(0)),
        *_state_facts("18", _values(1)),
        # Another period, and another package's total, never count.
        _fact("18", "all", 1.0, period=2023),
        _fact("18", "all", 2.0, spec_id="census_pep.v2024.state_age.v1"),
    ]

    totals = us_state_age_band_population_totals(facts, period=2024)

    assert totals == {"06": math.fsum(_values(0)), "18": math.fsum(_values(1))}
    assert sum(map(is_us_state_age_band_fact, facts)) == 2 * 16 + 1


@pytest.mark.parametrize(
    ("facts", "message"),
    [
        (
            [_fact("06", "all", 5.0), _fact("06", "all", 5.0)],
            "two Vintage 2025 all-ages totals",
        ),
        ([_fact("06", "all", 0.0)], "not a positive population"),
        ([_fact("06", "all", float("nan"))], "not a positive population"),
        (
            [
                {
                    **_fact("06", "all", 5.0),
                    "geography": {
                        "level": "country",
                        "id": "0100000US",
                        "name": "United States",
                    },
                }
            ],
            "names no US state",
        ),
    ],
)
def test_population_totals_refuse_a_bad_total(facts, message) -> None:
    with pytest.raises(ValueError, match=message):
        us_state_age_band_population_totals(facts, period=2024)


def test_compile_binds_the_age_groups_only_when_asked() -> None:
    facts = [*_state_facts("06", _values(0)), *_state_facts("18", _values(1))]

    baseline = _by_name(_compile([]))
    fenced = _by_name(_compile(facts))
    bound_registry = _compile(facts, state_age_band_targets=True)
    bound = _by_name(bound_registry)

    # Off: the facts change no target. On: every other target is unchanged.
    assert fenced == baseline
    assert {name: bound[name] for name in baseline} == baseline
    added = [spec for spec in bound_registry.specs if spec.name not in baseline]
    assert len(added) == 2 * len(AGE_GROUPS)
    assert all(_is_age_spec(spec) for spec in added)
    for spec in added:
        assert spec.family == "census_population"
        assert spec.metadata["materializer"] == "population_age"
        assert spec.metadata["measure_mode"] == "indicator_sum"
        assert spec.metadata["state_fips"] in {"06", "18"}
        assert spec.metadata["source_period"] == "2024"
    # The all-ages row carries no age bound, so it never becomes a target.
    assert not any(spec.name.endswith(".all.population") for spec in added)


@_COMPILE_SETTINGS
@given(st.lists(st.integers(1, 5_000_000), min_size=15, max_size=15))
def test_compiled_age_groups_carry_their_facts_counts(values) -> None:
    facts = _state_facts("18", [float(value) for value in values])
    counts = {fact["lineage"]["source_record_id"]: fact["value"] for fact in facts}

    specs = [
        spec
        for spec in _compile(facts, state_age_band_targets=True).specs
        if _is_age_spec(spec)
    ]

    assert len(specs) == len(AGE_GROUPS)
    for spec in specs:
        assert spec.value == counts[spec.name]
    bounds = sorted(
        (
            float(spec.metadata["age_lower_bound"]),
            float(spec.metadata["age_upper_bound"]),
        )
        for spec in specs
    )
    assert bounds == [
        (float(lower), math.inf if upper is None else float(upper))
        for _value_id, lower, upper in AGE_GROUPS
    ]


def test_state_age_surface_selects_fifteen_groups_in_every_state() -> None:
    module = _tool()
    facts = [
        fact
        for index, state in enumerate(STATES)
        for fact in _state_facts(state, _values(index))
    ]
    registry = _compile(facts, state_age_band_targets=True)

    specs, totals, receipt = module.state_age_surface(registry, facts)

    assert len(specs) == len(STATES) * len(AGE_GROUPS) == receipt["targets"] == 765
    assert totals == {
        int(state): math.fsum(_values(index)) for index, state in enumerate(STATES)
    }
    assert receipt["states"] == 51
    assert receipt["population"] == math.fsum(totals.values())
    assert receipt["record_set_spec_id"] == US_STATE_AGE_BAND_RECORD_SET_SPEC_ID


def test_state_age_surface_refuses_a_state_missing_a_group() -> None:
    module = _tool()
    facts = [
        fact
        for index, state in enumerate(STATES)
        for fact in _state_facts(state, _values(index))
        if not fact["lineage"]["source_record_id"].endswith(".18.age_3.population")
    ]
    registry = _compile(facts, state_age_band_targets=True)

    with pytest.raises(SystemExit, match="state 18 has 14 age groups"):
        module.state_age_surface(registry, facts)


# --- The surface check -------------------------------------------------------


def _spec(state, lower, upper, value, *, period="2024"):
    return SimpleNamespace(
        value=float(value),
        metadata={
            "state_fips": state,
            "age_lower_bound": str(lower),
            "age_upper_bound": "inf" if upper is None else str(upper),
            "source_period": period,
        },
    )


@st.composite
def _surfaces(draw):
    """A valid surface: in every state, fifteen groups that tile [0, inf)."""

    cuts = sorted(draw(st.sets(st.integers(1, 110), min_size=14, max_size=14)))
    edges = [0, *cuts, None]
    base = draw(st.lists(st.integers(1, 400_000), min_size=15, max_size=15))
    scale = draw(st.lists(st.integers(1, 90), min_size=51, max_size=51))
    specs, totals = [], {}
    for state, factor in zip(STATES, scale, strict=True):
        values = [count * factor for count in base]
        totals[state] = float(sum(values))
        specs += [
            _spec(state, edges[index], edges[index + 1], value)
            for index, value in enumerate(values)
        ]
    return specs, totals


@_SETTINGS
@given(_surfaces())
def test_a_partitioned_surface_has_no_problems(surface) -> None:
    specs, totals = surface
    assert _tool().state_age_surface_problems(specs, totals) == []


@_SETTINGS
@given(_surfaces(), st.data())
def test_the_surface_check_names_each_broken_state(surface, data) -> None:
    module = _tool()
    specs, totals = surface
    index = data.draw(st.integers(0, len(specs) - 1))
    state = specs[index].metadata["state_fips"]
    breakage = data.draw(
        st.sampled_from(["value", "drop", "gap", "period", "stray", "no_total"])
    )
    broken = [SimpleNamespace(**vars(spec)) for spec in specs]
    broken[index] = SimpleNamespace(
        value=specs[index].value, metadata=dict(specs[index].metadata)
    )
    broken_totals = dict(totals)
    if breakage == "value":
        broken[index].value += data.draw(
            st.integers(-50, 50).filter(lambda delta: delta != 0)
        )
        expected = f"state {state}'s age groups sum to"
    elif breakage == "drop":
        del broken[index]
        expected = f"state {state} has 14 age groups"
    elif breakage == "gap":
        lower = float(broken[index].metadata["age_lower_bound"])
        broken[index].metadata["age_lower_bound"] = str(lower + 0.5)
        expected = "do not tile [0, inf)"
    elif breakage == "period":
        broken[index].metadata["source_period"] = "2023"
        expected = f"state {state} has age groups from ['2023']"
    elif breakage == "stray":
        broken.append(_spec("72", 0, None, 1.0))
        expected = "age groups name state '72'"
    else:
        del broken_totals[state]
        expected = "50 state totals"

    problems = module.state_age_surface_problems(broken, broken_totals)

    assert any(expected in problem for problem in problems), problems
    if breakage in {"value", "drop", "gap", "period"}:
        # Only the broken state is named.
        assert all(f"state {state}" in problem for problem in problems), problems


# --- State totals, districts and counties -----------------------------------


@st.composite
def _ladders(draw):
    """Ladder state and district populations, and new state totals."""

    states = sorted(
        draw(st.sets(st.sampled_from([1, 2, 6, 11, 36, 48, 56]), min_size=1))
    )
    ladder_states, districts, totals = {}, {}, {}
    for state in states:
        count = draw(st.integers(1, 6))
        people = draw(
            st.lists(st.integers(1, 3_000_000), min_size=count, max_size=count)
        )
        for number, value in enumerate(people, start=1):
            districts[state * 100 + number] = float(value)
        ladder_states[state] = float(sum(people))
        totals[state] = float(draw(st.integers(1, 60_000_000)))
    return ladder_states, districts, totals


@_SETTINGS
@given(_ladders())
def test_new_state_totals_keep_each_districts_share(case) -> None:
    module = _tool()
    ladder_states, districts, totals = case
    populations = {
        "state": dict(ladder_states),
        "cd": dict(districts),
        "other": {7: 3.0},
    }
    before = json.dumps(populations, sort_keys=True)

    moved = module.with_state_population_totals(populations, ladder_states, totals)

    assert json.dumps(populations, sort_keys=True) == before  # inputs untouched
    assert moved["other"] == {7: 3.0}
    assert moved["state"] == totals
    assert set(moved["cd"]) == set(districts)
    for state, total in totals.items():
        block = [code for code in districts if code // 100 == state]
        assert math.fsum(moved["cd"][code] for code in block) == pytest.approx(
            total, rel=1e-12
        )
        for code in block:
            assert moved["cd"][code] > 0
            assert moved["cd"][code] / total == pytest.approx(
                districts[code] / ladder_states[state], rel=1e-9
            )


@_SETTINGS
@given(_ladders(), st.data())
def test_counties_reconcile_to_the_new_state_totals(case, data) -> None:
    module = _tool()
    ladder_states, districts, totals = case
    counties, overlaps = [], []
    for state in ladder_states:
        count = data.draw(st.integers(1, 5))
        codes = [state * 1000 + 2 * number + 1 for number in range(count)]
        counties += codes
        overlaps += data.draw(
            st.lists(st.integers(1, 2_000_000), min_size=count, max_size=count)
        )
    ladder = SimpleNamespace(
        puma=np.asarray([state * 100_000 + 1 for state in ladder_states]),
        puma_population=np.asarray(list(ladder_states.values())),
        county_overlap_county=np.asarray(counties),
        county_overlap_population=np.asarray(overlaps, dtype=float),
    )
    moved = module.with_state_population_totals(
        {"state": dict(ladder_states), "cd": dict(districts)}, ladder_states, totals
    )

    result = module.county_surface.with_county_populations(
        ladder, moved, ["state", "cd", "county"]
    )

    for state, total in totals.items():
        assert math.fsum(
            value for code, value in result["county"].items() if code // 1000 == state
        ) == pytest.approx(total, rel=1e-12)


@pytest.mark.parametrize(
    ("totals", "message"),
    [
        ({1: 36.0}, "missing \\[6\\]"),
        ({1: 36.0, 6: 110.0, 2: 5.0}, "extra \\[2\\]"),
        ({1: 36.0, 6: 0.0}, "State 6 has no positive population"),
        ({1: 36.0, 6: float("nan")}, "State 6 has no positive population"),
    ],
)
def test_new_state_totals_must_cover_the_ladders_states(totals, message) -> None:
    with pytest.raises(ValueError, match=message):
        _tool().with_state_population_totals(
            {"state": {1: 30.0, 6: 100.0}}, {1: 30.0, 6: 100.0}, totals
        )


_LADDER = SimpleNamespace(
    puma=np.asarray([100001, 600001, 600002]),
    puma_population=np.asarray([30.0, 60.0, 40.0]),
    cd_overlap_cd=np.asarray([101, 601, 601, 602]),
    cd_overlap_population=np.asarray([30.0, 25.0, 15.0, 60.0]),
)


def test_ladder_population_moves_states_and_scales_districts(monkeypatch) -> None:
    from microcosm.build.us_runtime import puma_ladder

    module = _tool()
    monkeypatch.setattr(puma_ladder, "load_us_puma_ladder", lambda *_: _LADDER)
    totals = {1: 36.0, 6: 110.0}

    before = module.ladder_population("ladder.npz", ["cd", "state"])
    after = module.ladder_population("ladder.npz", ["cd", "state"], state_totals=totals)
    districts_only = module.ladder_population("ladder.npz", ["cd"], state_totals=totals)

    assert before == {
        "state": {1: 30.0, 6: 100.0},
        "cd": {101: 30.0, 601: 40.0, 602: 60.0},
    }
    assert after["state"] == totals
    assert after["cd"] == pytest.approx({101: 36.0, 601: 44.0, 602: 66.0})
    assert set(districts_only) == {"cd"}
    assert districts_only["cd"] == after["cd"]


def test_population_targets_name_where_their_levels_come_from() -> None:
    module = _tool()
    names, values = ["pop_state_06", "pop_cd_0601"], [1.0, 1.0]

    ladder = module.population_target_specs(names, values)
    census = module.population_target_specs(names, values, census_state_totals=True)

    assert {spec.name: spec.source for spec in ladder} == {
        "pop_state_06": module.LADDER_POPULATION_SOURCE,
        "pop_cd_0601": module.LADDER_POPULATION_SOURCE,
    }
    assert {spec.name: spec.source for spec in census} == {
        "pop_state_06": module.CENSUS_STATE_POPULATION_SOURCE,
        "pop_cd_0601": module.CENSUS_SCALED_DISTRICT_POPULATION_SOURCE,
    }
    # Only the source moves: labels and hierarchy stay the release's own.
    assert [spec.hierarchy for spec in census] == [spec.hierarchy for spec in ladder]


def test_the_district_marginal_limitation_says_where_levels_come_from() -> None:
    module = _tool()
    census_receipt = module.state_population_receipt(
        SimpleNamespace(state_population_totals={1: 1.0}, age_receipt={"states": 51})
    )

    ladder = module._cd_population_marginal_limitation(None)
    census = module._cd_population_marginal_limitation(
        {"state_population": census_receipt}
    )

    assert census_receipt["source"] == module.CENSUS_STATE_POPULATION_SOURCE
    assert census_receipt["states"] == 51
    assert ladder["id"] == census["id"] == "cd_population_marginal_vintage_2020"
    assert "Vintage 2025" in census["reason"]
    assert "Vintage 2025" not in ladder["reason"]


# --- The materialize stage ---------------------------------------------------


def _materialize_parse_args(module, tmp_path, *extra):
    for filename in ("staging.h5", "feed.json", "ladder.npz"):
        (tmp_path / filename).write_bytes(b"fixture")
    (tmp_path / "staging.summary.json").write_text("{}")
    return module._parse_args(
        [
            "--stage",
            "materialize",
            "--staging-h5",
            str(tmp_path / "staging.h5"),
            "--feed",
            str(tmp_path / "feed.json"),
            "--ladder",
            str(tmp_path / "ladder.npz"),
            "--checkpoint-dir",
            str(tmp_path / "checkpoint"),
            *extra,
        ]
    )


def test_the_default_families_bind_the_state_age_groups(tmp_path) -> None:
    module = _tool()
    args = _materialize_parse_args(module, tmp_path)
    assert args.families.split(",") == ["snap", "medicaid", "soi", module.AGE_FAMILY]


@pytest.mark.parametrize(
    ("recorded", "expected"),
    [
        (None, "snap,medicaid,soi"),
        (["snap", "medicaid", "soi"], "snap,medicaid,soi"),
        (["snap", "medicaid", "soi", "age"], "snap,medicaid,soi,age"),
        ("snap,soi", "snap,soi"),
    ],
    ids=["legacy", "before-1528", "with-age", "string"],
)
def test_the_refresh_recipe_rebuilds_the_recorded_families(recorded, expected):
    """The recipe names the families, so an older release's recipe does not
    pick up the age family the default gained in microcosm#1528."""

    module = _tool()
    recipe = shlex.split(module.release_refresh_recipe("state", families=recorded))
    assert recipe[recipe.index("--families") + 1] == expected
    assert module._parse_args(recipe[3:]).families == expected


@lru_cache
def _regression_fixtures():
    path = (
        paths_for("microcosm-build").tests
        / "engine_free"
        / "us"
        / "test_us_acs_local_no_county_regression.py"
    )
    spec = importlib.util.spec_from_file_location("acs_local_regression_fixtures", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


#: Vintage 2025 totals for the fixture's two states; the ladder has 30 and 100.
_AGE_VALUES = {"01": [2.0] * 14 + [8.0], "06": [7.0] * 14 + [12.0]}


def _age_specs():
    specs = []
    for state, values in _AGE_VALUES.items():
        for (value_id, lower, upper), value in zip(AGE_GROUPS, values, strict=True):
            name = (
                "census_pep.v2025.cy2024.state_resident_population_age_bands."
                f"{state}.{value_id}.population"
            )
            specs.append(
                TargetSpec(
                    name=name,
                    entity="household",
                    measure=name,
                    value=value,
                    period=2024,
                    source="synthetic ledger",
                    family="census_population",
                    metadata={
                        "ledger_layout_record_set_spec_id": (
                            US_STATE_AGE_BAND_RECORD_SET_SPEC_ID
                        ),
                        "ledger_geography_level": "state",
                        "state_fips": state,
                        "materializer": "population_age",
                        "measure_mode": "indicator_sum",
                        "source_measure_id": "population",
                        "source_period": "2024",
                        "age_lower_bound": str(lower),
                        "age_upper_bound": "inf" if upper is None else str(upper),
                    },
                )
            )
    return specs


@pytest.mark.parametrize("geographies", ["state", "cd", "state,cd"])
def test_materialize_binds_age_groups_and_census_state_totals(
    monkeypatch, tmp_path, geographies
) -> None:
    """Run the materialize assembly with the age family on the regression fixture."""

    from microcosm.build import ledger_artifact
    from microcosm.build.us_runtime import fiscal_targets, medicaid_take_up, puma_ladder

    module = _tool()
    fixtures = _regression_fixtures()
    frame = fixtures._frame()
    age_specs = _age_specs()
    facts = [
        _fact(state, "all", math.fsum(values)) for state, values in _AGE_VALUES.items()
    ]
    monkeypatch.setattr(module, "STATE_AGE_STATE_COUNT", len(_AGE_VALUES))
    monkeypatch.setattr(
        ledger_artifact,
        "load_ledger_consumer_artifact",
        lambda *_: SimpleNamespace(facts=facts),
    )
    asked = {}

    def compile_registry(*_args, **kwargs):
        asked.update(kwargs)
        return TargetRegistry([*fixtures._input_specs(), *age_specs], country="us")

    monkeypatch.setattr(
        fiscal_targets, "compile_us_fiscal_target_registry", compile_registry
    )
    monkeypatch.setattr(
        medicaid_take_up,
        "apply_us_medicaid_enrollment_substitutions",
        lambda registry: (registry, []),
    )
    import microcosm.build.us_runtime as runtime

    monkeypatch.setattr(
        runtime,
        "load_congressional_district_vintage_crosswalk",
        lambda *_: pd.DataFrame({"source_geography_id": [], "target_geography_id": []}),
    )
    monkeypatch.setattr(
        puma_ladder,
        "load_us_puma_ladder",
        lambda *_: SimpleNamespace(
            puma=np.asarray([100001, 600001]),
            puma_population=np.asarray([30.0, 100.0]),
            cd_overlap_cd=np.asarray([101, 601, 602]),
            cd_overlap_population=np.asarray([30.0, 40.0, 60.0]),
        ),
    )
    for name in (
        "_require_local_hours",
        "_require_acs_self_employment_identity",
        "_require_local_immigration",
        "_require_local_work_disability",
        "_require_local_income_transfer",
        "_require_local_ssi_disability",
        "log",
    ):
        monkeypatch.setattr(module, name, lambda *_a, **_k: None)
    monkeypatch.setattr(module, "_load_staging_frame", lambda *_: frame)
    monkeypatch.setattr(
        module,
        "_with_local_take_up",
        lambda frame, *, seed: (frame, {"seed": seed, "assigned_sha256": "stub"}),
    )
    monkeypatch.setattr(module, "acs_local_ssi_medicaid_take_up_targets", lambda *_: {})
    monkeypatch.setattr(
        module,
        "_with_local_ssi_medicaid_take_up",
        lambda frame, *, seed, **_k: (frame, {"seed": seed, "assigned_sha256": "stub"}),
    )
    monkeypatch.setattr(module, "_write_ssi_medicaid_take_up", lambda *_: "stub")
    household_states = np.asarray([6, 6, 1, 6])

    def materialize(_frame, specs, **_kwargs):
        rows = [
            np.asarray(
                household_states == int(spec.metadata["state_fips"]), dtype=np.float32
            )
            for spec in specs
        ]
        return SimpleNamespace(
            compiled_specs=specs,
            matrix=sparse.csr_array(np.asarray(rows)),
            chunk_stats=[],
            carrier_check={},
        )

    monkeypatch.setattr(module, "materialize_chunked", materialize)
    captured = {}

    def checkpoint(_struct, _matrix, specs, roles, _directory):
        captured.update(specs=specs, roles=roles)
        return (
            None,
            None,
            dict.fromkeys(
                (
                    "target_registry_sha256",
                    "target_roles_sha256",
                    "lean_h5_sha256",
                    "target_matrix_sha256",
                ),
                "fixture",
            ),
        )

    monkeypatch.setattr(module, "write_lean_checkpoint", checkpoint)
    args = _materialize_parse_args(module, tmp_path, "--geographies", geographies)

    module.do_materialize(args)

    assert asked["state_age_band_targets"] is True
    specs = {spec.name: spec for spec in captured["specs"]}
    assert {spec.name for spec in age_specs} <= set(specs)
    expected = {}
    if "state" in geographies:
        expected |= {"pop_state_01": 36.0, "pop_state_06": 110.0}
    if "cd" in geographies:
        expected |= {"pop_cd_0101": 36.0, "pop_cd_0601": 44.0, "pop_cd_0602": 66.0}
    population = {name: spec for name, spec in specs.items() if name.startswith("pop_")}
    assert {name: spec.value for name, spec in population.items()} == pytest.approx(
        expected
    )
    assert {spec.source for spec in population.values()} <= {
        module.CENSUS_STATE_POPULATION_SOURCE,
        module.CENSUS_SCALED_DISTRICT_POPULATION_SOURCE,
    }
    for receipt in ("run_identity.json", "materialize_rss.json"):
        state_population = json.loads((args.checkpoint_dir / receipt).read_text())[
            "state_population"
        ]
        assert state_population["source"] == module.CENSUS_STATE_POPULATION_SOURCE
        assert state_population["population"] == 146.0
        assert state_population["targets"] == 30
