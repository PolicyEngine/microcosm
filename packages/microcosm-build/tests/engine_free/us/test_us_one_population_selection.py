"""Properties of the d2283 one-population selection (us_runtime/one_population_selection.py).

Invariants checked for every drawn input:

1. completeness: every CPS key is selected, its year must be named, and every
   named year must hold a key;
2. budget: CPS plus ACS households equal the budget; a short eligible set and
   a CPS side above the budget refuse rather than clamp;
3. nesting: for replicate 0, a smaller budget's ACS households are a prefix
   (in rank order) of a larger one's; replicate ``r`` takes the ``r``-th
   disjoint block of the eligible order;
4. purity: permuting rows, weights or flags (with their keys) changes neither
   the selected set, the weights per household nor the manifest digest;
5. eligibility stability: removing one ACS household changes the selection by
   at most that household and the next eligible one in rank, and survivors
   keep their order;
6. key hygiene: non-string, empty, blank, numeric-coerced and duplicate keys
   refuse on both sides;
7. weights: finite and positive, side shares equal source-household shares
   of the budget, each CPS year holds an equal share, design-weight ratios
   are kept within a year and within the ACS, and weighted persons equal the
   benchmark;
8. manifest: the digest moves with the selected key sets (ACS and CPS), the
   salt, the replicate, the rule version and the budget; it also moves with
   the other fields the manifest records (the eligibility flags, the ACS
   counts, the CPS counts by year and the input digests), and with nothing
   that permuting the input changes.

Differential checks: the ACS order equals ``support_mix.acs_selection_order``
and an independent sort; the selection equals the bake-off tool's arm path
(``acs_rank`` over every row, the zero-adult drop, a stable argsort and
``replicate_block``); the counts equal ``support_mix.plan_arm_counts``; the
unscaled weights equal ``support_mix.arm_initial_weights``; the zero-adult
flag, the group-quarters flag and both unit keys equal the bake-off tool's
row index. A golden vector pins the shared draw's bytes, and the committed
bake-off evidence pins the national arm counts.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import importlib.util
import inspect
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, assume, example, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from microcosm.build.holdout import hash_holdout_uniform  # noqa: E402
from microcosm.build.us_runtime import (  # noqa: E402
    one_population_selection as selection,
)
from microcosm.build.us_runtime import support_mix  # noqa: E402
from microcosm.build.us_runtime.demographics import (  # noqa: E402
    CENSUS_NATIONAL_AGE_BENCHMARK,
)
from microcosm.build.us_runtime.one_population_selection import (  # noqa: E402
    ACS_ORDER_SALT,
    DEFAULT_PERSON_BENCHMARK,
    RULE_VERSION,
    AcsEligibility,
    Eligibility,
    SelectionManifest,
    acs_unit_key,
    cps_unit_key,
    group_quarters,
    plan_selection,
    spm_zero_adult,
    starting_weights,
)
from microcosm.build.us_runtime.support_mix import (  # noqa: E402
    SupportMixArm,
    acs_selection_order,
    arm_initial_weights,
    plan_arm_counts,
    replicate_block,
)
from test_support.paths import paths_for  # noqa: E402

_TEST_PATHS = paths_for("microcosm-build")
_EVIDENCE = _TEST_PATHS.repository / "experiments" / "us-support-mix-bakeoff-20260930"

#: Per-year CPS source households of the bake-off (income years 2022-2024),
#: from the committed ``arms.csv`` by subtraction; checked against the file
#: in :func:`test_evidence_counts_match_the_committed_bakeoff_files`.
EVIDENCE_CPS_BY_YEAR = {2022: 56_839, 2023: 56_251, 2024: 55_762}
#: From the committed ``spm_flags.json``.
EVIDENCE_ACS_HOUSEHOLDS = 1_531_614
EVIDENCE_ZERO_ADULT = 5_246
EVIDENCE_GROUP_QUARTERS = 183_206
EVIDENCE_ELIGIBLE = 1_526_368

_RULES = (
    Eligibility(),
    Eligibility(include_group_quarters=False),
    Eligibility(drop_spm_zero_adult=False),
    Eligibility(include_group_quarters=False, drop_spm_zero_adult=False),
)
_YEARS = (2022, 2023, 2024, 2025)
_FAST = settings(max_examples=200, deadline=None)
_SLOW = settings(
    max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)


@pytest.fixture(scope="module")
def tool():
    """The bake-off driver, imported as test_bakeoff_us_support_mix_tool.py does."""
    path = _TEST_PATHS.repository / "tools" / "bakeoff_us_support_mix.py"
    spec = importlib.util.spec_from_file_location("bakeoff_us_support_mix", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------- strategies


_serials = st.text(alphabet="0123456789GHQU", min_size=1, max_size=8)
_unit_ids = st.text(alphabet="0123456789", min_size=1, max_size=6)


@dataclasses.dataclass
class Scenario:
    cps_by_year: dict[int, list[str]]
    acs_keys: list[str]
    gq: np.ndarray
    zero: np.ndarray
    rule: Eligibility
    replicate: int
    budget: int

    @property
    def cps_keys(self) -> list[str]:
        return [
            cps_unit_key(year, uid)
            for year, ids in self.cps_by_year.items()
            for uid in ids
        ]

    @property
    def n_cps(self) -> int:
        return sum(len(ids) for ids in self.cps_by_year.values())

    @property
    def eligibility(self) -> AcsEligibility:
        return self.rule.evaluate(group_quarters=self.gq, spm_zero_adult=self.zero)

    @property
    def fill(self) -> int:
        return self.budget - self.n_cps

    def plan(self, **overrides):
        args = dict(
            cps_keys=self.cps_keys,
            cps_income_years=list(self.cps_by_year),
            acs_keys=self.acs_keys,
            acs_eligible=self.eligibility,
            budget=self.budget,
            replicate=self.replicate,
        )
        args.update(overrides)
        return plan_selection(**args)


@st.composite
def cps_sets(draw, *, min_years: int = 0) -> dict[int, list[str]]:
    years = draw(
        st.lists(st.sampled_from(_YEARS), unique=True, min_size=min_years, max_size=3)
    )
    return {
        year: draw(st.lists(_unit_ids, unique=True, min_size=1, max_size=5))
        for year in years
    }


@st.composite
def acs_sets(draw, *, min_size: int = 0, max_size: int = 30):
    serials = draw(
        st.lists(_serials, unique=True, min_size=min_size, max_size=max_size)
    )
    n = len(serials)
    gq = np.array(draw(st.lists(st.booleans(), min_size=n, max_size=n)), dtype=bool)
    zero = np.array(draw(st.lists(st.booleans(), min_size=n, max_size=n)), dtype=bool)
    return [acs_unit_key(serial) for serial in serials], gq, zero


@st.composite
def scenarios(
    draw,
    *,
    replicate: int | None = None,
    rule: Eligibility | None = None,
    min_acs: int = 0,
) -> Scenario:
    cps_by_year = draw(cps_sets())
    keys, gq, zero = draw(acs_sets(min_size=min_acs))
    rule = draw(st.sampled_from(_RULES)) if rule is None else rule
    replicate = draw(st.integers(0, 2)) if replicate is None else replicate
    eligible = rule.evaluate(group_quarters=gq, spm_zero_adult=zero).n_eligible
    fill = draw(st.integers(0, eligible // (replicate + 1)))
    n_cps = sum(len(ids) for ids in cps_by_year.values())
    assume(n_cps + fill > 0)
    return Scenario(cps_by_year, keys, gq, zero, rule, replicate, n_cps + fill)


def _reference(keys, eligible, fill, replicate, salt=ACS_ORDER_SALT) -> list[str]:
    """An independent oracle: Python's sort on (draw, key), then the block."""
    ranked = sorted(
        range(len(keys)),
        key=lambda i: (hash_holdout_uniform(keys[i], salt=salt), keys[i]),
    )
    eligible_ranked = [keys[i] for i in ranked if eligible[i]]
    return eligible_ranked[replicate * fill : (replicate + 1) * fill]


def _edge(cps_by_year, n_acs, *, budget, replicate=0, zero=()) -> Scenario:
    keys = [acs_unit_key(f"X{i}") for i in range(n_acs)]
    zero_flags = np.zeros(n_acs, dtype=bool)
    zero_flags[list(zero)] = True
    return Scenario(
        cps_by_year,
        keys,
        np.zeros(n_acs, dtype=bool),
        zero_flags,
        Eligibility(),
        replicate,
        budget,
    )


#: Boundary cases the scenario properties always run, whatever is drawn: an
#: empty ACS fill, an exact fit at replicate 1, no CPS households, and every
#: ACS household ineligible.
_EDGE_SCENARIOS = (
    _edge({2023: ["1", "2"]}, 3, budget=2),
    _edge({2024: ["5"]}, 4, budget=3, replicate=1),
    _edge({}, 3, budget=2),
    _edge({2022: ["9"]}, 3, budget=1, zero=(0, 1, 2)),
)


def _with_edge_scenarios(**extra):
    """Add each :data:`_EDGE_SCENARIOS` case as an ``@example``."""

    def decorate(test):
        for scenario in _EDGE_SCENARIOS:
            test = example(scenario=scenario, **extra)(test)
        return test

    return decorate


_SALTS = st.sampled_from([ACS_ORDER_SALT, "another.salt", "x"])


# ------------------------------------------------------- 1. completeness


@_FAST
@given(scenarios(), st.data())
def test_every_cps_key_of_the_named_years_is_selected(scenario, data) -> None:
    shuffled = data.draw(st.permutations(scenario.cps_keys))
    plan = scenario.plan(cps_keys=shuffled)
    assert plan.cps_keys == tuple(sorted(scenario.cps_keys))
    assert plan.cps_income_years == tuple(sorted(scenario.cps_by_year))
    assert dict(plan.n_cps_by_income_year) == {
        year: len(ids) for year, ids in scenario.cps_by_year.items()
    }
    for key in plan.cps_keys:
        year, _sep, unit = key.partition("|cps:")
        assert int(year) in plan.cps_income_years and unit


def test_cps_keys_must_come_from_exactly_the_named_years() -> None:
    eligibility = Eligibility().evaluate(
        group_quarters=np.zeros(0, bool), spm_zero_adult=np.zeros(0, bool)
    )
    keys = [cps_unit_key(2023, "1"), cps_unit_key(2024, "2")]
    common = dict(acs_keys=[], acs_eligible=eligibility, budget=2)
    assert (
        plan_selection(cps_keys=keys, cps_income_years=[2024, 2023], **common).n_cps
        == 2
    )
    with pytest.raises(ValueError, match="not named"):
        plan_selection(cps_keys=keys, cps_income_years=[2024], **common)
    with pytest.raises(ValueError, match="no CPS households"):
        plan_selection(cps_keys=keys, cps_income_years=[2023, 2024, 2025], **common)
    with pytest.raises(ValueError, match="repeats a year"):
        plan_selection(cps_keys=keys, cps_income_years=[2023, 2024, 2024], **common)


# ------------------------------------------------------------- 2. budget


@_FAST
@given(scenario=scenarios(), salt=_SALTS)
@_with_edge_scenarios(salt=ACS_ORDER_SALT)
@_with_edge_scenarios(salt="x")
def test_the_selection_fills_the_budget_with_the_ranked_eligible_block(
    scenario, salt
) -> None:
    plan = scenario.plan(salt=salt)
    assert plan.salt == salt
    assert plan.n_cps + plan.n_acs == scenario.budget
    assert plan.n_acs == scenario.fill
    eligible = scenario.eligibility.eligible
    assert list(plan.acs_keys) == _reference(
        scenario.acs_keys, eligible, scenario.fill, scenario.replicate, salt=salt
    )
    assert all(eligible[plan.acs_positions])
    assert [scenario.acs_keys[i] for i in plan.acs_positions] == list(plan.acs_keys)
    if plan.n_acs:
        assert plan.threshold_draw == hash_holdout_uniform(plan.acs_keys[-1], salt=salt)
    else:
        assert plan.threshold_draw is None


@_FAST
@given(scenario=scenarios())
@_with_edge_scenarios()
def test_short_eligible_sets_and_oversized_cps_sides_refuse(scenario) -> None:
    eligible = scenario.eligibility.n_eligible
    too_big = scenario.n_cps + eligible // (scenario.replicate + 1) + 1
    with pytest.raises(ValueError, match="eligible ACS households"):
        scenario.plan(budget=too_big)
    if scenario.n_cps >= 2:
        with pytest.raises(ValueError, match="exceed the"):
            scenario.plan(budget=scenario.n_cps - 1)
    for bad in (0, -1, True, 2.0, "3"):
        with pytest.raises(ValueError):
            scenario.plan(budget=bad)
    for bad in (-1, True, 1.0):
        with pytest.raises(ValueError):
            scenario.plan(replicate=bad)


# ------------------------------------------------------------ 3. nesting


@_FAST
@given(scenarios(replicate=0), st.data())
def test_replicate_zero_nests_across_budgets(scenario, data) -> None:
    eligible = scenario.eligibility.n_eligible
    small = data.draw(st.integers(0, eligible))
    large = data.draw(st.integers(small, eligible))
    assume(scenario.n_cps + small > 0)
    plan_small = scenario.plan(budget=scenario.n_cps + small)
    plan_large = scenario.plan(budget=scenario.n_cps + large)
    assert plan_large.acs_keys[:small] == plan_small.acs_keys
    assert list(plan_large.acs_positions[:small]) == list(plan_small.acs_positions)


@_FAST
@given(scenarios(replicate=0, min_acs=1), st.data())
def test_later_replicates_take_disjoint_blocks(scenario, data) -> None:
    eligible = scenario.eligibility.n_eligible
    assume(eligible >= 2)
    fill = data.draw(st.integers(1, eligible // 2))
    budget = scenario.n_cps + fill
    blocks = []
    for replicate in range(eligible // fill):
        plan = scenario.plan(budget=budget, replicate=replicate)
        assert list(plan.acs_keys) == _reference(
            scenario.acs_keys, scenario.eligibility.eligible, fill, replicate
        )
        blocks.append(set(plan.acs_keys))
    for i, first in enumerate(blocks):
        for second in blocks[i + 1 :]:
            assert first.isdisjoint(second)
    with pytest.raises(ValueError, match="eligible ACS households"):
        scenario.plan(budget=budget, replicate=eligible // fill)


@_FAST
@given(scenarios(replicate=0, min_acs=2), st.data())
def test_the_plan_and_its_manifest_record_the_requested_replicate(
    scenario, data
) -> None:
    eligible = scenario.eligibility.n_eligible
    assume(eligible >= 2)
    fill = data.draw(st.integers(1, eligible // 2))
    budget = scenario.n_cps + fill
    for replicate in range(eligible // fill):
        plan = scenario.plan(budget=budget, replicate=replicate)
        assert plan.replicate == replicate
        manifest = SelectionManifest.from_plan(plan)
        assert manifest.replicate == replicate
        assert manifest.to_payload()["replicate"] == replicate


# ------------------------------------------------------------- 4. purity


def _weights_inputs(scenario: Scenario, data):
    """Per-row CPS arrays (1-3 physical rows per household) and per-ACS arrays."""
    weight = st.floats(0.5, 5_000.0, allow_nan=False, allow_infinity=False)
    rows = []
    for year, ids in scenario.cps_by_year.items():
        for uid in ids:
            for copy in range(data.draw(st.integers(1, 3))):
                rows.append(
                    (
                        cps_unit_key(year, uid),
                        copy,
                        year,
                        data.draw(weight),
                        float(data.draw(st.integers(1, 9))),
                    )
                )
    n = len(scenario.acs_keys)
    acs_w = np.array(
        data.draw(st.lists(weight, min_size=n, max_size=n)), dtype=np.float64
    )
    acs_p = np.array(
        data.draw(st.lists(st.integers(1, 9), min_size=n, max_size=n)), dtype=np.float64
    )
    return rows, acs_w, acs_p


def _weights(plan, rows, acs_w, acs_p, **kwargs):
    return starting_weights(
        plan,
        cps_design_weights=np.array([r[3] for r in rows], dtype=np.float64),
        cps_row_income_year=np.array([r[2] for r in rows], dtype=np.int64),
        acs_design_weights=acs_w,
        persons_per_cps_row=np.array([r[4] for r in rows], dtype=np.float64),
        persons_per_acs_row=acs_p,
        **kwargs,
    )


@_FAST
@given(scenarios(), st.data())
def test_selection_weights_and_manifest_ignore_input_order(scenario, data) -> None:
    rows, acs_w, acs_p = _weights_inputs(scenario, data)
    plan = scenario.plan()
    weights = _weights(plan, rows, acs_w, acs_p)

    acs_perm = np.asarray(
        data.draw(st.permutations(range(len(scenario.acs_keys)))), dtype=np.int64
    )
    row_perm = data.draw(st.permutations(range(len(rows))))
    other = dataclasses.replace(
        scenario,
        cps_by_year={
            year: data.draw(st.permutations(ids))
            for year, ids in reversed(list(scenario.cps_by_year.items()))
        },
        acs_keys=[scenario.acs_keys[i] for i in acs_perm],
        gq=scenario.gq[acs_perm] if len(acs_perm) else scenario.gq,
        zero=scenario.zero[acs_perm] if len(acs_perm) else scenario.zero,
    )
    other_plan = other.plan()
    other_rows = [rows[i] for i in row_perm]
    other_weights = _weights(
        other_plan,
        other_rows,
        acs_w[acs_perm] if len(acs_perm) else acs_w,
        acs_p[acs_perm] if len(acs_perm) else acs_p,
    )
    assert set(other_plan.acs_keys) == set(plan.acs_keys)
    assert other_plan.acs_keys == plan.acs_keys  # rank order is a function of the keys
    assert other_plan.cps_keys == plan.cps_keys
    assert (
        SelectionManifest.from_plan(other_plan).digest()
        == SelectionManifest.from_plan(plan).digest()
    )
    by_key = dict(zip(plan.acs_keys, weights.acs, strict=True))
    for key, value in zip(other_plan.acs_keys, other_weights.acs, strict=True):
        assert value == pytest.approx(by_key[key], rel=1e-12)
    by_row = {(r[0], r[1]): w for r, w in zip(rows, weights.cps, strict=True)}
    for r, value in zip(other_rows, other_weights.cps, strict=True):
        assert value == pytest.approx(by_row[(r[0], r[1])], rel=1e-12)


# --------------------------------------------- 5. eligibility stability


@_FAST
@given(scenarios(replicate=0, min_acs=1), st.data())
def test_removing_one_household_moves_at_most_it_and_the_next(scenario, data) -> None:
    plan = scenario.plan()
    drop = data.draw(st.integers(0, len(scenario.acs_keys) - 1))
    removed = scenario.acs_keys[drop]
    keep = np.array(
        [i for i in range(len(scenario.acs_keys)) if i != drop], dtype=np.int64
    )
    smaller = dataclasses.replace(
        scenario,
        acs_keys=[scenario.acs_keys[i] for i in keep],
        gq=scenario.gq[keep] if len(keep) else np.zeros(0, bool),
        zero=scenario.zero[keep] if len(keep) else np.zeros(0, bool),
    )
    if smaller.eligibility.n_eligible < scenario.fill:
        with pytest.raises(ValueError, match="eligible ACS households"):
            smaller.plan()
        return
    after = smaller.plan()
    before = set(plan.acs_keys)
    if removed not in before:
        assert after.acs_keys == plan.acs_keys
        return
    assert before - set(after.acs_keys) == {removed}
    added = set(after.acs_keys) - before
    assert len(added) == 1
    eligible = scenario.eligibility.eligible
    following = _reference(scenario.acs_keys, eligible, scenario.fill + 1, 0)[-1]
    assert added == {following}
    survivors = [key for key in plan.acs_keys if key != removed]
    assert [key for key in after.acs_keys if key in before] == survivors


# ------------------------------------------------------- 6. key hygiene


_BAD_IDENTIFIERS = (
    None,
    0,
    7,
    1.0,
    b"123",
    True,
    "",
    " ",
    "\t",
    " 12",
    "12 ",
    "1 2",
    "12\n",
    "1\x1f2",
    "123.0",
    "1.5",
    ".5",
    "1e+12",
    "1E5",
    "17E6",  # e-notation without an exponent sign is refused on purpose
    "-1",
    "+3",
    "nan",
    "NaN",
    "None",
    "<NA>",
    "inf",
    "True",
    "false",
    "a:b",
    "a|b",
)


@pytest.mark.parametrize("bad", _BAD_IDENTIFIERS)
def test_unit_key_constructors_refuse_unclean_identifiers(bad) -> None:
    with pytest.raises(ValueError):
        acs_unit_key(bad)
    with pytest.raises(ValueError):
        cps_unit_key(2024, bad)


@pytest.mark.parametrize("year", [True, 2024.0, "2024", 999, 10_000, None])
def test_cps_unit_key_refuses_a_bad_year(year) -> None:
    with pytest.raises(ValueError):
        cps_unit_key(year, "123")


def test_unit_keys_take_the_bakeoff_forms() -> None:
    assert acs_unit_key("2023HU0000001") == "acs:2023HU0000001"
    assert acs_unit_key("2013000000001") == "acs:2013000000001"
    assert cps_unit_key(2024, "00000000000000012345") == "2024|cps:00000000000000012345"
    assert cps_unit_key(np.int64(2023), "7") == "2023|cps:7"


#: Bad ACS keys and the refusal each must raise (the key check, not a later one).
_BAD_ACS_KEYS = (
    ("2023HU1", "does not start with"),
    ("acs:", "must not be empty"),
    ("acs: ", "must not be empty"),
    ("acs:12.0", "looks like a number"),
    ("acs:None", "looks like a number"),
    ("cps:1", "does not start with"),
    (None, "ACS key must be a string"),
    (3, "ACS key must be a string"),
)


def _bad_cps_keys(year: int) -> tuple[tuple[object, str], ...]:
    """Bad CPS keys for a named ``year`` and the refusal each must raise.

    Keys that parse take the named year, so the unnamed-year and empty-year
    checks cannot be what refuses them.
    """
    return (
        ("cps:1", "is not of the form"),
        (f"{year}|cps:", "is not of the form"),
        (f"{year}|cps:1.0", "looks like a number"),
        (f"{year % 100}|cps:1", "is not of the form"),
        (f"{year}|acs:1", "is not of the form"),
        (f"{year}|cps:1|2", "must not hold"),
        (f"{year}|cps:nan", "looks like a number"),
        (None, "CPS key must be a string"),
        (year, "CPS key must be a string"),
    )


@_FAST
@given(scenarios(min_acs=1), st.data())
def test_invalid_or_duplicate_keys_refuse_on_both_sides(scenario, data) -> None:
    acs_bad, acs_refusal = data.draw(st.sampled_from(_BAD_ACS_KEYS))
    position = data.draw(st.integers(0, len(scenario.acs_keys) - 1))
    keys = list(scenario.acs_keys)
    keys[position] = acs_bad
    with pytest.raises(ValueError, match=acs_refusal):
        scenario.plan(acs_keys=keys)
    if len(scenario.acs_keys) >= 2:
        keys = list(scenario.acs_keys)
        keys[position] = keys[position - 1]
        with pytest.raises(ValueError, match="repeat"):
            scenario.plan(acs_keys=keys)
    if scenario.cps_keys:
        cps = list(scenario.cps_keys)
        year = int(cps[0].partition("|")[0])
        cps_bad, cps_refusal = data.draw(st.sampled_from(_bad_cps_keys(year)))
        cps[0] = cps_bad
        with pytest.raises(ValueError, match=cps_refusal):
            scenario.plan(cps_keys=cps)
        with pytest.raises(ValueError, match="repeat"):
            scenario.plan(
                cps_keys=[*scenario.cps_keys, scenario.cps_keys[-1]],
                budget=scenario.budget + 1,
            )
    with pytest.raises(ValueError, match="eligibility flags for"):
        scenario.plan(acs_keys=scenario.acs_keys[:-1])  # flags no longer align
    with pytest.raises(ValueError, match="sequences of keys"):
        scenario.plan(acs_keys="acs:1")
    with pytest.raises(ValueError, match="sequences of keys"):
        scenario.plan(cps_keys="2024|cps:1")


def test_eligibility_must_be_an_evaluated_rule() -> None:
    with pytest.raises(ValueError, match="AcsEligibility"):
        plan_selection(
            cps_keys=[],
            cps_income_years=[],
            acs_keys=["acs:1"],
            acs_eligible=np.array([True]),
            budget=1,
        )
    with pytest.raises(ValueError, match="booleans"):
        Eligibility().evaluate(
            group_quarters=np.array([0]), spm_zero_adult=np.array([False])
        )
    with pytest.raises(ValueError, match="align"):
        Eligibility().evaluate(
            group_quarters=np.array([False]), spm_zero_adult=np.zeros(2, bool)
        )
    with pytest.raises(ValueError):
        Eligibility(include_group_quarters=1)


@_FAST
@given(acs_sets(), st.sampled_from(_RULES))
def test_eligibility_applies_its_two_flags(acs, rule) -> None:
    _keys, gq, zero = acs
    result = rule.evaluate(group_quarters=gq, spm_zero_adult=zero)
    expected = np.ones(len(gq), dtype=bool)
    if not rule.include_group_quarters:
        expected &= ~gq
    if rule.drop_spm_zero_adult:
        expected &= ~zero
    assert result.eligible.tolist() == expected.tolist()
    assert (result.n_group_quarters, result.n_spm_zero_adult) == (
        int(gq.sum()),
        int(zero.sum()),
    )
    assert not result.eligible.flags.writeable


def test_eligibility_copies_and_freezes_its_flags() -> None:
    gq = np.array([False, True, False])
    zero = np.array([True, False, False])
    result = Eligibility(include_group_quarters=False).evaluate(
        group_quarters=gq, spm_zero_adult=zero
    )
    gq[:] = False
    zero[:] = False
    assert result.eligible.tolist() == [False, False, True]
    assert (result.n_group_quarters, result.n_spm_zero_adult) == (1, 1)
    for flags in (result.group_quarters, result.spm_zero_adult, result.eligible):
        assert not flags.flags.writeable


@pytest.mark.parametrize(
    ("build", "refusal"),
    [
        (lambda: Eligibility(drop_spm_zero_adult=1), "must be a bool"),
        (lambda: Eligibility(include_group_quarters=None), "must be a bool"),
        (
            lambda: AcsEligibility(
                rule=object(),
                group_quarters=np.zeros(1, bool),
                spm_zero_adult=np.zeros(1, bool),
            ),
            "rule must be an Eligibility",
        ),
        (
            lambda: Eligibility().evaluate(
                group_quarters=np.zeros((2, 1), bool),
                spm_zero_adult=np.zeros((2, 1), bool),
            ),
            "one-dimensional",
        ),
    ],
)
def test_eligibility_refuses_bad_rules_and_flags(build, refusal) -> None:
    with pytest.raises(ValueError, match=refusal):
        build()


def test_plan_selection_refuses_bad_salts_and_freezes_its_positions() -> None:
    keys = [acs_unit_key(serial) for serial in ("1", "2", "3")]
    eligibility = Eligibility().evaluate(
        group_quarters=np.zeros(3, bool), spm_zero_adult=np.zeros(3, bool)
    )
    plan = plan_selection(
        cps_keys=[],
        cps_income_years=[],
        acs_keys=keys,
        acs_eligible=eligibility,
        budget=2,
    )
    assert not plan.acs_positions.flags.writeable
    # With no ACS keys the order never hashes, so only this check refuses an
    # empty salt before it reaches the manifest.
    no_acs = Eligibility().evaluate(
        group_quarters=np.zeros(0, bool), spm_zero_adult=np.zeros(0, bool)
    )
    common = dict(
        cps_keys=[cps_unit_key(2024, "1")],
        cps_income_years=[2024],
        acs_keys=[],
        acs_eligible=no_acs,
        budget=1,
    )
    assert plan_selection(**common).threshold_draw is None
    for salt in ("", None, 3):
        with pytest.raises(ValueError, match="salt must be a non-empty string"):
            plan_selection(**common, salt=salt)
    with pytest.raises(ValueError, match="sequences of keys"):
        plan_selection(**{**common, "cps_keys": "2024|cps:1"})
    with pytest.raises(ValueError, match="plan must be a SelectionPlan"):
        SelectionManifest.from_plan(SelectionManifest.from_plan(plan))
    with pytest.raises(ValueError, match="plan must be a SelectionPlan"):
        starting_weights(
            None,
            cps_design_weights=[],
            cps_row_income_year=[],
            acs_design_weights=[1.0, 1.0, 1.0],
            persons_per_cps_row=[],
            persons_per_acs_row=[1.0, 1.0, 1.0],
        )


# ------------------------------------------------------------ 7. weights


@_FAST
@given(scenarios(), st.data())
def test_starting_weights_follow_the_rule(scenario, data) -> None:
    rows, acs_w, acs_p = _weights_inputs(scenario, data)
    plan = scenario.plan()
    benchmark = data.draw(st.sampled_from([None, 1_000.0, 3.342e8]))
    weights = _weights(plan, rows, acs_w, acs_p, person_benchmark=benchmark)
    target = DEFAULT_PERSON_BENCHMARK if benchmark is None else benchmark
    both = np.concatenate([weights.cps, weights.acs])
    assert np.isfinite(both).all() and (both > 0).all()
    assert both.sum() == pytest.approx(weights.total_mass, rel=1e-9)
    total = weights.total_mass
    assert weights.cps.sum() / total == pytest.approx(
        plan.n_cps / plan.budget, rel=1e-9, abs=1e-12
    )
    assert weights.acs.sum() / total == pytest.approx(
        plan.n_acs / plan.budget, rel=1e-9, abs=1e-12
    )
    years = np.array([r[2] for r in rows], dtype=np.int64)
    design = np.array([r[3] for r in rows])
    shares = [weights.cps[years == year].sum() for year in plan.cps_income_years]
    for share in shares:
        assert share == pytest.approx(shares[0], rel=1e-9)
    for year in plan.cps_income_years:
        ratio = weights.cps[years == year] / design[years == year]
        assert np.allclose(ratio, ratio[0], rtol=1e-9)
    if plan.n_acs:
        ratio = weights.acs / acs_w[plan.acs_positions]
        assert np.allclose(ratio, ratio[0], rtol=1e-9)
    persons = (
        np.array([r[4] for r in rows]) @ weights.cps
        + acs_p[plan.acs_positions] @ weights.acs
    )
    assert persons == pytest.approx(target, rel=1e-9)
    assert weights.person_benchmark == target


def test_the_default_benchmark_is_the_census_age_band_sum() -> None:
    assert DEFAULT_PERSON_BENCHMARK == float(
        sum(CENSUS_NATIONAL_AGE_BENCHMARK.values())
    )
    assert DEFAULT_PERSON_BENCHMARK == 334_200_000.0


def _small_plan():
    keys = [acs_unit_key(s) for s in ("1", "2", "3", "4")]
    eligibility = Eligibility().evaluate(
        group_quarters=np.zeros(4, bool), spm_zero_adult=np.zeros(4, bool)
    )
    cps = [cps_unit_key(2023, "1"), cps_unit_key(2024, "2")]
    plan = plan_selection(
        cps_keys=cps,
        cps_income_years=[2023, 2024],
        acs_keys=keys,
        acs_eligible=eligibility,
        budget=4,
    )
    good = dict(
        cps_design_weights=np.array([1.0, 2.0, 3.0]),
        cps_row_income_year=np.array([2023, 2024, 2024]),
        acs_design_weights=np.ones(4),
        persons_per_cps_row=np.ones(3),
        persons_per_acs_row=np.ones(4),
    )
    return plan, good


@pytest.mark.parametrize(
    "change",
    [
        {"cps_design_weights": np.array([1.0, -2.0, 3.0])},
        {"cps_design_weights": np.array([1.0, np.nan, 3.0])},
        {"cps_design_weights": np.array([1.0, 2.0])},
        {"cps_row_income_year": np.array([2023, 2024, 2025])},
        {"cps_row_income_year": np.array([2024, 2024, 2024])},
        {"cps_row_income_year": np.array([2023.0, 2024.0, 2024.0])},
        {"acs_design_weights": np.ones(2)},
        {"acs_design_weights": np.array([1.0, 0.0, 1.0, 1.0])},
        {"persons_per_acs_row": np.ones(3)},
        {"persons_per_acs_row": np.array([1.0, 1.0, np.inf, 1.0])},
        {"cps_design_weights": np.ones((3, 1))},
        {"cps_row_income_year": np.array([[2023, 2024, 2024]])},
        {"persons_per_cps_row": np.zeros(3)},
        {"person_benchmark": 0.0},
        {"person_benchmark": float("nan")},
        {"person_benchmark": True},
    ],
)
def test_starting_weights_refuse_bad_inputs(change) -> None:
    plan, good = _small_plan()
    starting_weights(plan, **good)
    with pytest.raises(ValueError):
        starting_weights(plan, **{**good, **change})


# ----------------------------------------------------------- 8. manifest


@_FAST
@given(scenarios(min_acs=2), st.data())
def test_manifest_digest_moves_with_each_named_field(scenario, data) -> None:
    plan = scenario.plan()
    manifest = SelectionManifest.from_plan(
        plan, input_digests={"acs_staging": "ab" * 32}
    )
    digest = manifest.digest()
    assert (
        SelectionManifest.from_plan(
            scenario.plan(), input_digests={"acs_staging": "ab" * 32}
        ).digest()
        == digest
    )
    other_keys = hashlib.sha256(b"another key set").hexdigest()
    # The fields property 8 names (the selected key set is the ACS and the
    # CPS keys) ...
    named = {
        "selected_acs_keys_sha256": other_keys,
        "cps_keys_sha256": other_keys,
        "salt": manifest.salt + ".other",
        "replicate": manifest.replicate + 1,
        "rule_version": RULE_VERSION + ".next",
        "budget": manifest.budget + 1,
    }
    # ... and the fields the design's manifest list adds (implement.md
    # deviation 17: the digest moves with these too).
    recorded = {
        "include_group_quarters": not manifest.include_group_quarters,
        "drop_spm_zero_adult": not manifest.drop_spm_zero_adult,
        "acs_households": manifest.acs_households + 1,
        "acs_eligible": manifest.acs_eligible + 1,
        "acs_spm_zero_adult": manifest.acs_spm_zero_adult + 1,
        "acs_group_quarters": manifest.acs_group_quarters + 1,
        "n_cps": manifest.n_cps + 1,
        "n_acs": manifest.n_acs + 1,
        "input_digests": (("acs_staging", "cd" * 32),),
    }
    for name, value in {**named, **recorded}.items():
        assert dataclasses.replace(manifest, **{name: value}).digest() != digest, name
    # End to end: a different budget, salt or replicate moves the digest.
    eligible = scenario.eligibility.n_eligible
    if scenario.fill + 1 <= eligible // (scenario.replicate + 1):
        bigger = scenario.plan(budget=scenario.budget + 1)
        assert (
            SelectionManifest.from_plan(bigger).digest()
            != SelectionManifest.from_plan(plan).digest()
        )
    # One more CPS household with one more budget: the ACS selection is the
    # same, the CPS key set (and its year count) is not.
    year = next(iter(scenario.cps_by_year), 2022)
    more_cps = dataclasses.replace(
        scenario,
        cps_by_year={
            **scenario.cps_by_year,
            year: [*scenario.cps_by_year.get(year, []), "X"],
        },
        budget=scenario.budget + 1,
    )
    grown = more_cps.plan()
    assert grown.acs_keys == plan.acs_keys
    assert (
        SelectionManifest.from_plan(grown).digest()
        != SelectionManifest.from_plan(plan).digest()
    )
    salted = scenario.plan(salt="another.salt")
    assert (
        SelectionManifest.from_plan(salted).digest()
        != SelectionManifest.from_plan(plan).digest()
    )
    if scenario.fill and (scenario.replicate + 2) * scenario.fill <= eligible:
        later = scenario.plan(replicate=scenario.replicate + 1)
        assert set(later.acs_keys).isdisjoint(plan.acs_keys)
        assert (
            SelectionManifest.from_plan(later).digest()
            != SelectionManifest.from_plan(plan).digest()
        )
    # The digest records nothing input order can change (see the purity test)
    # and changes with the input digests the caller passes.
    assert (
        SelectionManifest.from_plan(
            plan, input_digests={"acs_staging": "cd" * 32}
        ).digest()
        != digest
    )


def _asymmetric_plan():
    """A plan whose manifest fields all differ, so a swap of two shows.

    Nine ACS households: two in group quarters (positions 2, 5) and one
    zero-adult household (position 7) outside them, under a rule that drops
    both, so six are eligible. Three CPS households, two in 2023 and one in
    2024. A budget of 7 takes four ACS households.
    """
    serials = [f"2023{'GQ' if i in (2, 5) else 'HU'}{i + 1:07d}" for i in range(9)]
    keys = [acs_unit_key(serial) for serial in serials]
    gq = np.zeros(9, dtype=bool)
    gq[[2, 5]] = True
    zero = np.zeros(9, dtype=bool)
    zero[7] = True
    eligibility = Eligibility(include_group_quarters=False).evaluate(
        group_quarters=gq, spm_zero_adult=zero
    )
    cps = [
        cps_unit_key(2024, "300"),
        cps_unit_key(2023, "100"),
        cps_unit_key(2023, "200"),
    ]
    plan = plan_selection(
        cps_keys=cps,
        cps_income_years=[2024, 2023],
        acs_keys=keys,
        acs_eligible=eligibility,
        budget=7,
    )
    return plan, keys, eligibility.eligible, cps


#: sha256 of the asymmetric plan's manifest JSON with input digests
#: {"z": "1" * 64, "a": "2" * 64}, computed 2026-11-17 from this module and
#: again from a hand-built payload (Python's sort on (draw, key), json.dumps
#: with sorted keys and compact separators); the two agreed. A change here
#: means the manifest is written differently, which takes a new RULE_VERSION
#: (see the comment on RULE_VERSION in one_population_selection.py).
GOLDEN_MANIFEST_SHA256 = (
    "470e29abe905481067303a018d7437ae472cce11dcc2cc722c53ec8aebfb8f77"
)


def test_manifest_json_is_canonical_and_complete() -> None:
    plan, keys, eligible, cps = _asymmetric_plan()
    manifest = SelectionManifest.from_plan(
        plan, input_digests={"z": "1" * 64, "a": "2" * 64}
    )
    selected = _reference(keys, eligible, 4, 0)
    expected = {
        "rule_version": "us.one_population.selection.v1",
        "salt": ACS_ORDER_SALT,
        "replicate": 0,
        "budget": 7,
        "eligibility": {"include_group_quarters": False, "drop_spm_zero_adult": True},
        "acs_households": 9,
        "acs_eligible": 6,
        "acs_spm_zero_adult": 1,
        "acs_group_quarters": 2,
        "cps_income_years": [2023, 2024],
        "n_by_origin": {"cps": 3, "acs": 4},
        "n_cps_by_income_year": {"2023": 2, "2024": 1},
        "threshold_draw": hash_holdout_uniform(selected[-1], salt=ACS_ORDER_SALT),
        "selected_acs_keys_sha256": hashlib.sha256(
            "\n".join(sorted(selected)).encode()
        ).hexdigest(),
        "cps_keys_sha256": hashlib.sha256("\n".join(sorted(cps)).encode()).hexdigest(),
        "input_digests": {"a": "2" * 64, "z": "1" * 64},
    }
    assert manifest.to_payload() == expected
    text = manifest.to_json()
    assert text == json.dumps(
        expected, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    assert manifest.digest() == hashlib.sha256(text.encode()).hexdigest()
    assert manifest.digest() == GOLDEN_MANIFEST_SHA256
    for bad in ({"": "1"}, {"a": ""}, {"a": "1 2"}, {"a": 1}, ["a"]):
        with pytest.raises(ValueError):
            SelectionManifest.from_plan(plan, input_digests=bad)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("budget", 7.0),  # equals 7 but is written "7.0"
        ("budget", np.int64(7)),  # not JSON-serializable
        ("budget", True),
        ("replicate", -1),
        ("acs_households", None),
        ("include_group_quarters", 0),
        ("drop_spm_zero_adult", np.bool_(True)),
        ("rule_version", ""),
        ("salt", None),
        ("cps_income_years", (2024, 2023)),
        ("cps_income_years", [2023, 2024]),
        ("cps_income_years", (2023, 2023, 2024)),
        ("n_cps_by_income_year", ((2024, 1), (2023, 2))),
        ("n_cps_by_income_year", ((2023, 2.0), (2024, 1))),
        ("n_cps_by_income_year", ((2023, 2), (2023, 1))),
        ("threshold_draw", 0),
        ("threshold_draw", 1.0),
        ("threshold_draw", -0.0),
        ("threshold_draw", float("nan")),
        ("selected_acs_keys_sha256", "AB" * 32),
        ("cps_keys_sha256", ""),
        ("input_digests", (("z", "1"), ("a", "2"))),
        ("input_digests", (("a", "1"), ("a", "2"))),
        ("input_digests", (("a", ""),)),
        ("input_digests", {"a": "1"}),
    ],
)
def test_manifest_refuses_fields_that_would_break_its_digest(name, value) -> None:
    plan, *_rest = _asymmetric_plan()
    manifest = SelectionManifest.from_plan(plan, input_digests={"a": "1"})
    assert dataclasses.replace(manifest, threshold_draw=None).threshold_draw is None
    with pytest.raises(ValueError):
        dataclasses.replace(manifest, **{name: value})


# -------------------------------------------------------- golden vector


#: (salt, key, draw) literals produced 2026-11-17 by running
#: ``microcosm.build.holdout.hash_holdout_uniform(key, salt=salt)`` from
#: microcosm main a21fa3b9b (packages/microcosm-build/src/microcosm/build/
#: holdout.py:116-132) and pasting each ``repr``. Rows 1-5 and 8 are the
#: bake-off salt with "acs:" keys (row 5 and row 8 non-ASCII). Rows 6-7 are a
#: pair whose salt+key bytes ("abc") would tie without the "\x1f" separator.
#: A real 64-bit draw tie cannot be found by search, so the key tie-break is
#: checked with a forced constant draw below.
GOLDEN_DRAWS = (
    (ACS_ORDER_SALT, "acs:2023HU0000001", 0.3433680526701613),
    (ACS_ORDER_SALT, "acs:2023GQ0000001", 0.5003082625012254),
    (ACS_ORDER_SALT, "acs:2024HU1234567", 0.24780242048649154),
    (ACS_ORDER_SALT, "acs:2013000000001", 0.7357365435373242),
    (ACS_ORDER_SALT, "acs:2023HU00000ñ1", 0.21931583899300108),
    ("ab", "c", 0.5863001957090903),
    ("a", "bc", 0.38398581048625713),
    (ACS_ORDER_SALT, "acs:日本", 0.21067514374193314),
)


@pytest.mark.parametrize(("salt", "key", "draw"), GOLDEN_DRAWS)
def test_golden_draws_pin_the_shared_helper(salt, key, draw) -> None:
    assert hash_holdout_uniform(key, salt=salt) == draw


def test_the_order_and_threshold_use_the_pinned_helper() -> None:
    """The golden vector pins the draw both the order and the threshold use."""
    assert support_mix.hash_uniform is hash_holdout_uniform
    assert selection.hash_uniform is hash_holdout_uniform


def test_golden_order_and_threshold_follow_the_pinned_draws() -> None:
    golden = [(key, draw) for salt, key, draw in GOLDEN_DRAWS if salt == ACS_ORDER_SALT]
    keys = [key for key, _draw in golden]
    eligibility = Eligibility().evaluate(
        group_quarters=np.zeros(len(keys), bool),
        spm_zero_adult=np.zeros(len(keys), bool),
    )
    plan = plan_selection(
        cps_keys=[],
        cps_income_years=[],
        acs_keys=keys,
        acs_eligible=eligibility,
        budget=3,
    )
    expected = [key for key, _draw in sorted(golden, key=lambda pair: pair[1])][:3]
    assert list(plan.acs_keys) == expected
    assert plan.threshold_draw == dict(golden)[expected[-1]] == 0.24780242048649154


def test_ties_break_by_key(monkeypatch) -> None:
    """With every draw equal, the order is the key order (no input-order leak)."""
    monkeypatch.setattr(support_mix, "hash_uniform", lambda key, *, salt: 0.5)
    monkeypatch.setattr(selection, "hash_uniform", lambda key, *, salt: 0.5)
    keys = [acs_unit_key(s) for s in ("9", "30", "2", "11", "Q1", "200", "1")]
    zero = np.array([False, False, True, False, False, False, False])
    eligibility = Eligibility().evaluate(
        group_quarters=np.zeros(len(keys), bool), spm_zero_adult=zero
    )
    plan = plan_selection(
        cps_keys=[],
        cps_income_years=[],
        acs_keys=keys,
        acs_eligible=eligibility,
        budget=4,
    )
    assert (
        list(plan.acs_keys)
        == sorted(k for k, z in zip(keys, zero, strict=True) if not z)[:4]
    )
    assert plan.threshold_draw == 0.5


# ------------------------------------------------------- differentials


def test_order_salt_is_the_bakeoff_salt(tool) -> None:
    assert ACS_ORDER_SALT == tool.ACS_SELECTION_SALT


@_FAST
@given(acs_sets(min_size=1))
def test_acs_order_equals_support_mix_order(acs) -> None:
    keys, _gq, _zero = acs
    n = len(keys)
    eligibility = Eligibility(drop_spm_zero_adult=False).evaluate(
        group_quarters=np.zeros(n, bool), spm_zero_adult=np.zeros(n, bool)
    )
    plan = plan_selection(
        cps_keys=[],
        cps_income_years=[],
        acs_keys=keys,
        acs_eligible=eligibility,
        budget=n,
    )
    order = acs_selection_order(keys, salt=ACS_ORDER_SALT)
    assert list(plan.acs_keys) == [keys[i] for i in order]
    assert list(plan.acs_positions) == list(order)
    assert list(plan.acs_keys) == _reference(keys, np.ones(n, bool), n, 0)


@_SLOW
@given(scenarios(rule=Eligibility(), min_acs=1), st.booleans(), st.data())
def test_selection_equals_the_bakeoff_arm_path(
    tool, scenario, exclude_gq, data
) -> None:
    """``acs_rank`` over every row, then do_arm's drop, argsort and block."""
    rule = Eligibility(include_group_quarters=not exclude_gq)
    scenario = dataclasses.replace(scenario, rule=rule)
    eligible = scenario.eligibility.n_eligible
    fill = data.draw(st.integers(0, eligible // (scenario.replicate + 1)))
    assume(scenario.n_cps + fill > 0)
    plan = scenario.plan(budget=scenario.n_cps + fill)

    rows_acs = pd.DataFrame(
        {
            "unit_key": scenario.acs_keys,
            "group_quarters": scenario.gq,
            "spm_zero_adult_unit": scenario.zero,
        }
    )
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        rows_acs.to_parquet(work / "rows_acs.parquet", index=False)
        rank_all = tool.acs_rank(work)
    # tools/bakeoff_us_support_mix.py do_arm, :1430-1460 at a21fa3b9b.
    rows_acs["row"] = np.arange(len(rows_acs))
    if exclude_gq:
        rows_acs = rows_acs[~rows_acs["group_quarters"]].reset_index(drop=True)
    rows_acs = rows_acs[~rows_acs["spm_zero_adult_unit"]].reset_index(drop=True)
    rank = rank_all[rows_acs["row"].to_numpy()]
    order = np.argsort(rank, kind="stable")
    block = replicate_block(order, fill, scenario.replicate)
    chosen = rows_acs.iloc[np.sort(block)]["unit_key"].tolist()
    assert set(plan.acs_keys) == set(chosen)
    assert len(plan.acs_keys) == len(chosen) == fill


#: The do_arm lines the arm-path differential above copies; do_arm needs a
#: full bake-off work dir, so the test mirrors them and this pins the mirror.
_DO_ARM_SELECTION_LINES = (
    'rows_acs["row"] = np.arange(len(rows_acs))',
    'rows_acs = rows_acs[~rows_acs["group_quarters"]].reset_index(drop=True)',
    'rows_acs = rows_acs[~rows_acs["spm_zero_adult_unit"]].reset_index(drop=True)',
    # the counts differential's acs_households is the eligible count
    "acs_households=len(rows_acs))",
    'rank = acs_rank(args.work)[rows_acs["row"].to_numpy()]',
    'order = np.argsort(rank, kind="stable")',
    "block = replicate_block(order, counts.acs, arm.seed)",
    "acs_rows = rows_acs.iloc[np.sort(block)].reset_index(drop=True)",
)


def test_the_arm_path_mirror_matches_do_arm(tool) -> None:
    source = inspect.getsource(tool.do_arm)
    for line in _DO_ARM_SELECTION_LINES:
        assert line in source, line


@_FAST
@given(scenario=scenarios())
@_with_edge_scenarios()
def test_counts_equal_plan_arm_counts(scenario) -> None:
    plan = scenario.plan()
    arm = SupportMixArm(
        budget=scenario.budget,
        cps_income_years=tuple(scenario.cps_by_year),
        acs_fill_share=1.0,
        seed=scenario.replicate,
    )
    counts = plan_arm_counts(
        arm,
        cps_households_by_year={
            year: len(ids) for year, ids in scenario.cps_by_year.items()
        },
        acs_households=scenario.eligibility.n_eligible,
    )
    assert plan.counts == counts


@_FAST
@given(scenarios(replicate=0), st.integers(1, 40))
def test_plan_refuses_whatever_plan_arm_counts_refuses(scenario, extra) -> None:
    eligible = scenario.eligibility.n_eligible
    budget = scenario.n_cps + eligible + extra  # an ACS fill above the eligible set
    arm = SupportMixArm(budget=budget, cps_income_years=tuple(scenario.cps_by_year))
    by_year = {year: len(ids) for year, ids in scenario.cps_by_year.items()}
    with pytest.raises(ValueError):
        plan_arm_counts(arm, cps_households_by_year=by_year, acs_households=eligible)
    with pytest.raises(ValueError):
        scenario.plan(budget=budget)
    if scenario.n_cps >= 2:
        arm = SupportMixArm(
            budget=scenario.n_cps - 1, cps_income_years=tuple(scenario.cps_by_year)
        )
        with pytest.raises(ValueError):
            plan_arm_counts(
                arm, cps_households_by_year=by_year, acs_households=eligible
            )
        with pytest.raises(ValueError):
            scenario.plan(budget=scenario.n_cps - 1)


@_FAST
@given(scenarios(), st.data())
def test_weights_equal_arm_initial_weights_before_the_person_scaling(
    scenario, data
) -> None:
    rows, acs_w, acs_p = _weights_inputs(scenario, data)
    plan = scenario.plan()
    weights = _weights(plan, rows, acs_w, acs_p)
    cps_w, acs_sel = arm_initial_weights(
        cps_design_weights=np.array([r[3] for r in rows], dtype=np.float64),
        cps_income_year=np.array([r[2] for r in rows], dtype=np.int64),
        cps_copies=np.zeros(len(rows), dtype=np.int64),
        acs_design_weights=acs_w[plan.acs_positions],
        counts=plan.counts,
        total_mass=float(plan.budget),
    )
    assert weights.unscaled_total_mass == float(plan.budget)
    np.testing.assert_array_equal(weights.cps, cps_w * weights.mass_factor)
    np.testing.assert_array_equal(weights.acs, acs_sel * weights.mass_factor)
    persons = (
        np.array([r[4] for r in rows]) @ cps_w + acs_p[plan.acs_positions] @ acs_sel
    )
    assert weights.mass_factor == pytest.approx(
        DEFAULT_PERSON_BENCHMARK / persons, rel=1e-12
    )


_AGES = st.one_of(
    st.sampled_from([0, 10, 14, 15, 16, 17, 18, 19, 40, 80]), st.integers(0, 95)
)


#: Ways a role, head or spouse column arrives: Python objects, 0/1 floats
#: (missing as NaN), 0/1 integers, pandas' nullable boolean and Int64.
_FLAG_DTYPES = ("object", "float64", "int64", "boolean", "Int64")


def _flag_column(values: list, dtype: str) -> pd.Series:
    """``values`` (True, False or None) as a column of ``dtype``.

    A dtype that cannot hold a missing value (int64) falls back to float64.
    """
    if dtype == "object":
        return pd.Series(values, dtype=object)
    if dtype == "boolean":
        return pd.Series(values, dtype="boolean")
    numbers_ = [None if value is None else int(value) for value in values]
    if dtype == "int64" and None in numbers_:
        dtype = "float64"
    if dtype == "float64":
        return pd.Series(
            [np.nan if value is None else float(value) for value in numbers_],
            dtype="float64",
        )
    return pd.Series(numbers_, dtype=dtype)


_FLAGS = st.sampled_from([True, False, None])


@st.composite
def spm_tables(draw):
    """Households of SPM units; the role is stored, formula-derived or broken.

    Head and spouse values may be missing, and every role, head or spouse
    column takes a drawn dtype.
    """
    n_households = draw(st.integers(1, 6))
    people = []
    for household in range(1, n_households + 1):
        if draw(st.integers(0, 9)) == 0:
            continue  # a household with no persons is never flagged
        for unit in range(draw(st.integers(1, 2))):
            for _ in range(draw(st.integers(1, 3))):
                people.append(
                    (
                        household,
                        household * 10 + unit,
                        draw(_AGES),
                        draw(_FLAGS),
                        draw(_FLAGS),
                    )
                )
    person = pd.DataFrame(
        [row[:3] for row in people],
        columns=["person_household_id", "person_spm_unit_id", "age"],
    )
    person["is_household_head"] = _flag_column(
        [row[3] for row in people], draw(st.sampled_from(_FLAG_DTYPES))
    )
    person["is_household_spouse"] = _flag_column(
        [row[4] for row in people], draw(st.sampled_from(_FLAG_DTYPES))
    )
    role_dtype = draw(st.sampled_from(_FLAG_DTYPES))
    mode = draw(
        st.sampled_from(
            [
                "formula",
                "head_only",
                "spouse_only",
                "neither",
                "stored",
                "all_null",
                "partly_null",
            ]
        )
    )
    if mode == "head_only":
        person = person.drop(columns=["is_household_spouse"])
    elif mode == "spouse_only":
        person = person.drop(columns=["is_household_head"])
    elif mode == "neither":
        person = person.drop(columns=["is_household_head", "is_household_spouse"])
    elif mode == "stored":
        person["is_spm_independent_minor_role"] = _flag_column(
            draw(st.lists(st.booleans(), min_size=len(person), max_size=len(person))),
            role_dtype,
        )
    elif mode == "all_null":
        person["is_spm_independent_minor_role"] = _flag_column(
            [None] * len(person), role_dtype
        )
    elif mode == "partly_null" and len(person) >= 2:
        values = draw(
            st.lists(st.booleans(), min_size=len(person), max_size=len(person))
        )
        values[draw(st.integers(0, len(person) - 1))] = None  # the rest keep a value
        person["is_spm_independent_minor_role"] = _flag_column(values, role_dtype)
    order = draw(st.permutations(range(1, n_households + 1)))
    household = pd.DataFrame({"household_id": list(order)})
    return person, household


@_FAST
@given(spm_tables())
def test_spm_zero_adult_equals_the_bakeoff_flag(tool, tables) -> None:
    person, household = tables
    try:
        expected = tool.spm_zero_adult_households(
            {"person": person.copy(), "household": household.copy()}
        )
    except (ValueError, TypeError):
        # The tool refuses (a nullable Int64 head with a missing value raises
        # pandas' TypeError there); the module refuses with ValueError.
        with pytest.raises(ValueError):
            spm_zero_adult(person, household)
        return
    assert spm_zero_adult(person, household).tolist() == expected.tolist()


@pytest.mark.parametrize(
    ("age", "head", "stored_role", "flagged"),
    [
        # A stored role column overrides the head-or-spouse formula.
        (16, True, False, True),
        (16, False, True, False),
        (15, False, True, False),
        (14, False, True, True),
        (17, True, None, False),  # no stored column: the formula's head counts
        # The age boundaries: 18 is an adult with no role, 15 with the role.
        (18, False, None, False),
        (18, False, False, False),
        (17, False, None, True),
        (15, True, None, False),
        (14, True, None, True),
    ],
)
def test_spm_zero_adult_reads_the_stored_role_and_the_age_boundaries(
    tool, age, head, stored_role, flagged
) -> None:
    """One person alone in a unit, beside a household with an adult."""
    person = pd.DataFrame(
        {
            "person_household_id": [1, 2],
            "person_spm_unit_id": [10, 20],
            "age": [40, age],
            "is_household_head": [True, head],
            "is_household_spouse": [False, False],
        }
    )
    if stored_role is not None:
        person["is_spm_independent_minor_role"] = [True, stored_role]
    household = pd.DataFrame({"household_id": [2, 1]})
    assert spm_zero_adult(person, household).tolist() == [flagged, False]
    expected = tool.spm_zero_adult_households(
        {"person": person, "household": household}
    )
    assert expected.tolist() == [flagged, False]


def test_spm_zero_adult_flags_a_household_with_one_adult_less_unit(tool) -> None:
    """A household with an adult unit and a unit with no adult is flagged."""
    person = pd.DataFrame(
        {
            "person_household_id": [1, 1, 2],
            "person_spm_unit_id": [10, 11, 20],
            "age": [40, 16, 40],
            "is_household_head": [True, False, True],
            "is_household_spouse": [False, False, False],
        }
    )
    household = pd.DataFrame({"household_id": [1, 2]})
    assert spm_zero_adult(person, household).tolist() == [True, False]
    expected = tool.spm_zero_adult_households(
        {"person": person, "household": household}
    )
    assert expected.tolist() == [True, False]


def test_spm_zero_adult_reads_a_0_1_role_as_booleans() -> None:
    person = pd.DataFrame(
        {
            "person_household_id": [1, 2, 3],
            "person_spm_unit_id": [10, 20, 30],
            "age": [16, 16, 40],
            "is_household_head": [False, True, True],
            "is_household_spouse": [False, False, False],
        }
    )
    household = pd.DataFrame({"household_id": [1, 2, 3]})
    as_bool = person.assign(is_spm_independent_minor_role=[True, False, False])
    as_int = person.assign(is_spm_independent_minor_role=[1, 0, 0])
    as_head_int = person.assign(
        is_household_head=[0, 1, 1], is_household_spouse=[0, 0, 0]
    )
    assert spm_zero_adult(as_bool, household).tolist() == [False, True, False]
    assert spm_zero_adult(as_int, household).tolist() == [False, True, False]
    assert spm_zero_adult(as_head_int, household).tolist() == [True, False, False]


@pytest.mark.parametrize("dtype", ["object", "float64", "boolean"])
@pytest.mark.parametrize("missing", ["head", "spouse", "both"])
def test_spm_zero_adult_reads_a_missing_head_or_spouse_as_false(
    tool, dtype, missing
) -> None:
    """A 16-year-old whose head or spouse value is missing is not in the role."""
    head = [True, None if missing in ("head", "both") else False]
    spouse = [False, None if missing in ("spouse", "both") else False]
    person = pd.DataFrame(
        {
            "person_household_id": [1, 2],
            "person_spm_unit_id": [10, 20],
            "age": [40, 16],
            "is_household_head": _flag_column(head, dtype),
            "is_household_spouse": _flag_column(spouse, dtype),
        }
    )
    household = pd.DataFrame({"household_id": [1, 2]})
    assert spm_zero_adult(person, household).tolist() == [False, True]
    expected = tool.spm_zero_adult_households(
        {"person": person, "household": household}
    )
    assert expected.tolist() == [False, True]


@pytest.mark.parametrize(
    "breakage",
    [
        "missing_age",
        "text_age",
        "infinite_age",
        "orphan_person",
        "null_unit",
        "repeated_household",
        "text_role",
        "text_head",
        "int64_head_with_missing",
    ],
)
def test_spm_zero_adult_refuses_what_the_bakeoff_would_misread(breakage) -> None:
    person = pd.DataFrame(
        {
            "person_household_id": [1, 1, 2],
            "person_spm_unit_id": [10, 10, 20],
            "age": [40.0, 10.0, 16.0],
            "is_household_head": [True, False, False],
            "is_household_spouse": [False, False, False],
        }
    )
    household = pd.DataFrame({"household_id": [1, 2]})
    assert spm_zero_adult(person, household).tolist() == [False, True]
    if breakage == "missing_age":
        person.loc[2, "age"] = np.nan
    elif breakage == "text_age":
        person["age"] = person["age"].astype(object)
        person.loc[2, "age"] = "sixteen"
    elif breakage == "infinite_age":
        person.loc[2, "age"] = np.inf  # the tool would count an adult
    elif breakage == "text_role":
        # The tool's astype(bool) reads the string "False" as true.
        person["is_spm_independent_minor_role"] = ["True", "False", "False"]
    elif breakage == "text_head":
        person["is_household_head"] = ["True", "False", "False"]
    elif breakage == "int64_head_with_missing":
        person["is_household_head"] = pd.Series([1, 0, None], dtype="Int64")
    elif breakage == "orphan_person":
        person.loc[2, "person_household_id"] = 3
    elif breakage == "null_unit":
        person["person_spm_unit_id"] = person["person_spm_unit_id"].astype(float)
        person.loc[2, "person_spm_unit_id"] = np.nan
    else:
        household = pd.DataFrame({"household_id": [1, 2, 2]})
    refusal = {
        "missing_age": "missing, non-numeric or infinite",
        "text_age": "missing, non-numeric or infinite",
        "infinite_age": "1 missing, non-numeric or infinite",
        "orphan_person": "missing from the household table",
        "null_unit": "null values",
        "repeated_household": "household_id repeats",
        "text_role": "is_spm_independent_minor_role must hold booleans",
        "text_head": "is_household_head must hold booleans",
        "int64_head_with_missing": "cannot fill with False",
    }[breakage]
    with pytest.raises(ValueError, match=refusal):
        spm_zero_adult(person, household)


@pytest.mark.parametrize(
    "column",
    ["is_spm_independent_minor_role", "is_household_head", "is_household_spouse"],
)
@pytest.mark.parametrize("text", ["0", "1"])
@pytest.mark.parametrize("layout", ["one_text_value", "string_dtype"])
def test_spm_zero_adult_refuses_numeric_text_flags(column, text, layout) -> None:
    # A numeric reading of "0" would call the 16-year-old a minor, while the
    # bake-off's astype(bool) reads any nonempty text as true.
    person = pd.DataFrame(
        {
            "person_household_id": [1, 1, 2],
            "person_spm_unit_id": [10, 10, 20],
            "age": [40.0, 10.0, 16.0],
            "is_household_head": [True, False, False],
            "is_household_spouse": [False, False, False],
        }
    )
    household = pd.DataFrame({"household_id": [1, 2]})
    if column == "is_spm_independent_minor_role":
        person[column] = [True, False, False]
    assert spm_zero_adult(person, household).tolist() == [False, True]
    if layout == "one_text_value":
        values = person[column].astype(object)
        values.iloc[2] = text
    else:
        values = pd.Series(["1", "0", text], dtype="string")
    person[column] = values
    with pytest.raises(ValueError, match=f"person.{column} must hold booleans"):
        spm_zero_adult(person, household)


@pytest.mark.parametrize(
    ("table", "column", "refusal"),
    [
        ("household", "household_id", "household table has no household_id"),
        ("person", "person_household_id", "no person_household_id"),
        ("person", "person_spm_unit_id", "no person_spm_unit_id"),
        ("person", "age", "no age column"),
        ("person", "is_household_head", "neither is_household_head"),
        ("household", "null household_id", "household.household_id has 1 null"),
    ],
)
def test_spm_zero_adult_refuses_missing_columns_and_ids(table, column, refusal) -> None:
    person = pd.DataFrame(
        {
            "person_household_id": [1, 2],
            "person_spm_unit_id": [10, 20],
            "age": [40, 30],
            "is_household_head": [True, True],
        }
    )
    household = pd.DataFrame({"household_id": [1.0, 2.0]})
    assert spm_zero_adult(person, household).tolist() == [False, False]
    if column == "null household_id":
        household.loc[1, "household_id"] = np.nan
    elif table == "person":
        person = person.drop(columns=[column])
    else:
        household = household.drop(columns=[column])
    with pytest.raises(ValueError, match=refusal):
        spm_zero_adult(person, household)


_TYPEHUGQ = st.sampled_from([1, 2, 3, 1.0, 2.0, 3.0, None, np.nan])


@st.composite
def acs_row_tables(draw):
    """Household and person tables shaped as the bake-off's ACS shards."""
    person, household = draw(spm_tables())
    household = household.sort_values("household_id").reset_index(drop=True)
    n = len(household)
    serials = draw(st.lists(_serials, unique=True, min_size=n, max_size=n))
    household = household.assign(
        state_fips=6,
        congressional_district_geoid=601,
        county_fips=6001,
        tenure_type=draw(
            st.lists(
                st.sampled_from(["RENTED", "OWNED_OUTRIGHT"]), min_size=n, max_size=n
            )
        ),
        SERIALNO=serials,
        TYPEHUGQ=pd.Series(
            draw(st.lists(_TYPEHUGQ, min_size=n, max_size=n)), dtype=object
        ),
    )
    person = person.assign(is_female=0, pre_subsidy_rent=0.0, real_estate_taxes=0.0)
    return person, household


@_SLOW
@given(acs_row_tables())
def test_acs_keys_and_flags_equal_the_bakeoff_row_index(tool, tables) -> None:
    person, household = tables
    try:
        out = tool._row_index_columns({"person": person, "household": household}, "acs")
    except (ValueError, TypeError):  # TypeError: see the zero-adult differential
        with pytest.raises(ValueError):
            spm_zero_adult(person, household)
        return
    assert out["unit_key"].tolist() == [acs_unit_key(s) for s in household["SERIALNO"]]
    assert (
        group_quarters(household["TYPEHUGQ"]).tolist() == out["group_quarters"].tolist()
    )
    assert (
        spm_zero_adult(person, household).tolist()
        == out["spm_zero_adult_unit"].tolist()
    )


@pytest.mark.parametrize("bad", ["x", "", 4, 0, 1.5, float("inf")])
def test_group_quarters_refuses_values_outside_the_pums_codes(bad) -> None:
    assert group_quarters([1, 2, 3, None, np.nan]).tolist() == [
        False,
        True,
        True,
        False,
        False,
    ]
    with pytest.raises(ValueError):
        group_quarters([1, bad])


@_SLOW
@given(
    st.lists(
        st.tuples(
            st.sampled_from([2022, 2023, 2024]),
            st.text(alphabet="0123456789", min_size=22, max_size=22),
        ),
        min_size=1,
        max_size=8,
        unique_by=lambda pair: pair[1][:20],
    )
)
def test_cps_keys_equal_the_bakeoff_source_keys(tool, households) -> None:
    n = len(households)
    household = pd.DataFrame(
        {
            "household_id": np.arange(1, n + 1),
            "state_fips": 6,
            "congressional_district_geoid": 601,
            "county_fips": 6001,
            "tenure_type": "RENTED",
            "household_support_channel": "asec",
            "household_support_clone_index": 0,
        }
    )
    person = pd.DataFrame(
        {
            "person_household_id": np.repeat(np.arange(1, n + 1), 2),
            "PERIDNUM": [
                peridnum[:20] + suffix
                for _year, peridnum in households
                for suffix in ("01", "02")
            ],
            "source_year": np.repeat([year for year, _peridnum in households], 2),
            "age": 30,
            "is_female": 0,
            "pre_subsidy_rent": 0.0,
            "real_estate_taxes": 0.0,
        }
    )
    out = tool._row_index_columns({"person": person, "household": household}, "cps")
    assert out["source_key"].tolist() == [
        cps_unit_key(year, peridnum[:20]) for year, peridnum in households
    ]


# ---------------------------------------------- committed bake-off evidence


def test_evidence_counts_match_the_committed_bakeoff_files() -> None:
    """The fixture's numbers are the bake-off's, read back from its files."""
    flags = json.loads((_EVIDENCE / "spm_flags.json").read_text())
    assert flags["acs_households"] == EVIDENCE_ACS_HOUSEHOLDS
    assert flags["spm_zero_adult_households"] == EVIDENCE_ZERO_ADULT
    assert flags["spm_zero_adult_in_group_quarters"] == EVIDENCE_ZERO_ADULT
    assert flags["group_quarters_households"] == EVIDENCE_GROUP_QUARTERS
    assert EVIDENCE_ACS_HOUSEHOLDS - EVIDENCE_ZERO_ADULT == EVIDENCE_ELIGIBLE
    with open(_EVIDENCE / "arms.csv", newline="") as handle:
        arms = [row for row in csv.DictReader(handle) if row["product"] == "national"]
    cps = {row["arm"].split(".")[1]: int(row["cps"]) for row in arms}
    assert cps["cps2024"] == EVIDENCE_CPS_BY_YEAR[2024]
    assert cps["cps2023-2024"] - cps["cps2024"] == EVIDENCE_CPS_BY_YEAR[2023]
    assert cps["cps2022-2023-2024"] - cps["cps2023-2024"] == EVIDENCE_CPS_BY_YEAR[2022]
    checked = 0
    for row in arms:
        if row["arm"].startswith("natural") or float(row["acs_fill_share"]) != 1.0:
            continue
        label = row["arm"].split(".")[1]
        years = () if label == "cps0" else tuple(int(y) for y in label[3:].split("-"))
        arm = SupportMixArm(
            budget=int(row["budget"]),
            cps_income_years=years,
            acs_fill_share=1.0,
            seed=int(row["seed"]),
        )
        counts = plan_arm_counts(
            arm,
            cps_households_by_year=EVIDENCE_CPS_BY_YEAR,
            acs_households=EVIDENCE_ELIGIBLE,
        )
        assert (counts.cps, counts.acs) == (int(row["cps"]), int(row["acs"])), row[
            "arm"
        ]
        assert counts.acs * (arm.seed + 1) <= EVIDENCE_ELIGIBLE
        checked += 1
    assert checked == 16


def test_evidence_fixture_plans_the_bakeoff_national_arms() -> None:
    """At full size: 168,852 CPS households and the bake-off's ACS eligibility."""
    cps_keys = [
        cps_unit_key(year, f"{i:020d}")
        for year, n in EVIDENCE_CPS_BY_YEAR.items()
        for i in range(n)
    ]
    acs_keys = [acs_unit_key(f"{i:013d}") for i in range(EVIDENCE_ACS_HOUSEHOLDS)]
    zero = np.zeros(EVIDENCE_ACS_HOUSEHOLDS, dtype=bool)
    gq = np.zeros(EVIDENCE_ACS_HOUSEHOLDS, dtype=bool)
    # Every zero-adult household is in group quarters, as spm_flags.json says;
    # which ones are flagged does not matter to the counts.
    zero[::291][:EVIDENCE_ZERO_ADULT] = True
    gq[zero] = True
    gq[np.flatnonzero(~zero)[: EVIDENCE_GROUP_QUARTERS - EVIDENCE_ZERO_ADULT]] = True
    eligibility = Eligibility().evaluate(group_quarters=gq, spm_zero_adult=zero)
    assert eligibility.n_eligible == EVIDENCE_ELIGIBLE
    assert eligibility.n_group_quarters == EVIDENCE_GROUP_QUARTERS
    common = dict(
        cps_keys=cps_keys,
        cps_income_years=[2022, 2023, 2024],
        acs_keys=acs_keys,
        acs_eligible=eligibility,
    )

    small = plan_selection(budget=300_000, **common)
    assert (small.n_cps, small.n_acs) == (168_852, 131_148)
    assert dict(small.n_cps_by_income_year) == EVIDENCE_CPS_BY_YEAR
    large = plan_selection(budget=1_200_000, **common)
    assert (large.n_cps, large.n_acs) == (168_852, 1_031_148)
    assert large.acs_keys[:131_148] == small.acs_keys
    assert not zero[large.acs_positions].any()
    manifest = SelectionManifest.from_plan(small)
    assert (manifest.acs_households, manifest.acs_eligible) == (
        EVIDENCE_ACS_HOUSEHOLDS,
        EVIDENCE_ELIGIBLE,
    )
    assert (manifest.acs_spm_zero_adult, manifest.acs_group_quarters) == (
        EVIDENCE_ZERO_ADULT,
        EVIDENCE_GROUP_QUARTERS,
    )
    # The threshold is near the selected share of eligible households.
    assert small.threshold_draw == pytest.approx(131_148 / EVIDENCE_ELIGIBLE, abs=0.002)
    # Seeds 1 and 2 of the 300k arm fit (3 x 131,148 <= 1,526,368); seed 1
    # of the 1.2M arm would need 2,062,296 and refuses.
    third = plan_selection(budget=300_000, replicate=2, **common)
    assert third.n_acs == 131_148 and set(third.acs_keys).isdisjoint(small.acs_keys)
    with pytest.raises(ValueError, match="2,062,296"):
        plan_selection(budget=1_200_000, replicate=1, **common)
