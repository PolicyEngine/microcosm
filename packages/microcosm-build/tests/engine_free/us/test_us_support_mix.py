"""Properties of the support-mix bake-off primitives.

Invariants checked for every drawn input:

- budget accounting: an arm's CPS, ACS and clone source households sum to
  its budget, and infeasible arms are refused rather than truncated;
- nesting: a smaller ACS selection is a prefix of a larger one, and the order
  depends only on the key set, not its input order;
- clone allocation: copies sum to the requested clones and differ by at most
  one across households;
- mass: starting weights are positive, sum to the requested total, give each
  CPS income year an equal share, and split CPS and ACS mass in proportion to
  source households; a row's copies sum back to its per-year weight;
- ESS: distinct-unit ESS never exceeds the row ESS or the number of distinct
  units, and equals the row ESS when every unit is distinct;
- assembly (differential): the sparse target matrix equals a naive dense
  construction, including cells that carry several targets;
- split: the role is a pure function of the group key, district children
  share their parent's role, vintage tokens never change a role, and the
  hash matches the pending release primitive byte for byte.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from microcosm.build.us_runtime.support_mix import (  # noqa: E402
    TARGET_SPLIT_HOLDOUT_SALT,
    TARGET_SPLIT_SEALED_SALT,
    ArmCounts,
    SupportMixArm,
    acs_selection_order,
    arm_initial_weights,
    assemble_target_matrix,
    clone_copies,
    distinct_unit_ess,
    hash_uniform,
    kish_ess,
    plan_arm_counts,
    target_split_group_key,
    target_split_role,
)

YEARS = {2022: 56_839, 2023: 56_251, 2024: 55_762}


def test_hash_uniform_matches_the_release_primitive() -> None:
    digest = hashlib.sha256(b"salt\x1fkey").digest()
    assert hash_uniform("key", salt="salt") == int.from_bytes(digest[:8], "big") / 2**64
    with pytest.raises(ValueError):
        hash_uniform("", salt="salt")
    with pytest.raises(ValueError):
        hash_uniform("key", salt="")


#: (key, salt, draw) computed 2026-09-30 by the holdout port branch's own
#: ``microcosm.build.holdout.hash_holdout_uniform`` (us-target-roles-holdout
#: working copy), so the mirror is checked against that implementation, not
#: only against a re-typed formula.
PORT_BRANCH_FIXTURES = (
    ("irs_soi.congressional_district_2022.all_returns.wv_total.net_capital_gains_returns",
     "microcosm.us.target_split.v1.sealed", 0.568527776442365),
    ("irs_soi.congressional_district_2022.all_returns.wv_total.net_capital_gains_returns",
     "microcosm.us.target_split.v1.holdout", 0.2332024720189108),
    ("census_pep.national_resident_population_age.20_to_24.population",
     "microcosm.us.target_split.v1.sealed", 0.6698038027150819),
    ("census_pep.national_resident_population_age.20_to_24.population",
     "microcosm.us.target_split.v1.holdout", 0.888848284264564),
    ("k0", "microcosm.us.target_split.v1.sealed", 0.13405297471660024),
    ("k0", "microcosm.us.target_split.v1.holdout", 0.994075647032073),
    ("usda_snap.state.06.households", "microcosm.us.target_split.v1.sealed", 0.20260546634959004),
    ("usda_snap.state.06.households", "microcosm.us.target_split.v1.holdout", 0.9762431428888598),
)


@pytest.mark.parametrize(("key", "salt", "draw"), PORT_BRANCH_FIXTURES)
def test_hash_uniform_matches_the_port_branch_fixtures(key: str, salt: str, draw: float) -> None:
    assert hash_uniform(key, salt=salt) == draw


@given(st.text(min_size=1, max_size=40), st.text(min_size=1, max_size=20))
def test_hash_uniform_is_a_unit_interval_draw(key: str, salt: str) -> None:
    u = hash_uniform(key, salt=salt)
    assert 0.0 <= u < 1.0
    assert u == hash_uniform(key, salt=salt)


@given(
    budget=st.integers(min_value=1, max_value=2_000_000),
    years=st.sets(st.sampled_from(sorted(YEARS)), max_size=3),
    share=st.floats(min_value=0.0, max_value=1.0),
)
def test_arm_counts_sum_to_the_budget_or_refuse(budget, years, share) -> None:
    arm = SupportMixArm(budget=budget, cps_income_years=tuple(years), acs_fill_share=share)
    cps = sum(YEARS[year] for year in years)
    try:
        counts = plan_arm_counts(arm, cps_households_by_year=YEARS, acs_households=1_531_614)
    except ValueError:
        fill = budget - cps
        acs = round(fill * share) if fill >= 0 else 0
        assert fill < 0 or acs > 1_531_614 or (fill - acs > 0 and cps == 0)
        return
    assert counts.total == budget
    assert counts.cps == cps
    assert min(counts.acs, counts.clones) >= 0


def test_natural_arm_has_no_fill() -> None:
    arm = SupportMixArm(budget=None, cps_income_years=(2024,))
    counts = plan_arm_counts(arm, cps_households_by_year=YEARS, acs_households=10)
    assert counts == ArmCounts(cps=55_762, acs=0, clones=0)
    assert arm.label == "natural.cps2024.acs100.s0"


@given(
    keys=st.lists(st.text(min_size=1, max_size=12), min_size=1, max_size=60, unique=True),
    data=st.data(),
)
def test_acs_selection_is_nested_and_order_free(keys, data) -> None:
    order = acs_selection_order(keys, salt="s")
    assert sorted(order.tolist()) == list(range(len(keys)))
    small = data.draw(st.integers(min_value=0, max_value=len(keys)))
    large = data.draw(st.integers(min_value=small, max_value=len(keys)))
    chosen_small = {keys[i] for i in order[:small]}
    chosen_large = {keys[i] for i in order[:large]}
    assert chosen_small <= chosen_large
    shuffled = list(reversed(keys))
    reorder = acs_selection_order(shuffled, salt="s")
    assert [shuffled[i] for i in reorder] == [keys[i] for i in order]


@given(
    keys=st.lists(st.text(min_size=1, max_size=8), min_size=1, max_size=50, unique=True),
    n_clones=st.integers(min_value=0, max_value=500),
)
def test_clone_copies_sum_and_balance(keys, n_clones) -> None:
    copies = clone_copies(keys, n_clones, salt="c")
    assert int(copies.sum()) == n_clones
    assert copies.max() - copies.min() <= 1


positive = st.floats(min_value=0.01, max_value=1e4, allow_nan=False)


@settings(max_examples=150)
@given(
    cps=st.lists(st.tuples(positive, st.sampled_from([2022, 2023, 2024]), st.integers(0, 5)),
                 min_size=1, max_size=40),
    acs=st.lists(positive, min_size=0, max_size=40),
    clones=st.integers(min_value=0, max_value=100),
    mass=st.floats(min_value=1.0, max_value=1e9),
)
def test_initial_weights_conserve_and_split_mass(cps, acs, clones, mass) -> None:
    weights = np.array([w for w, _, _ in cps])
    years = np.array([y for _, y, _ in cps])
    copies = np.array([k for _, _, k in cps])
    acs_w = np.array(acs, dtype=np.float64)
    counts = ArmCounts(cps=len(cps), acs=len(acs), clones=clones)
    cps_out, acs_out = arm_initial_weights(
        cps_design_weights=weights, cps_income_year=years, cps_copies=copies,
        acs_design_weights=acs_w, counts=counts, total_mass=mass,
    )
    expanded = cps_out * (1 + copies)
    assert (cps_out > 0).all() and (acs_out > 0).all()
    total = expanded.sum() + acs_out.sum()
    assert np.isclose(total, mass, rtol=1e-9)
    cps_share = (counts.cps + counts.clones) / counts.total
    assert np.isclose(expanded.sum(), cps_share * mass, rtol=1e-9)
    per_year = [expanded[years == y].sum() for y in np.unique(years)]
    assert np.allclose(per_year, per_year[0], rtol=1e-9)


def test_initial_weights_refuse_bad_inputs() -> None:
    counts = ArmCounts(cps=1, acs=1, clones=0)
    ok = dict(cps_design_weights=np.array([1.0]), cps_income_year=np.array([2024]),
              cps_copies=np.array([0]), acs_design_weights=np.array([1.0]),
              counts=counts, total_mass=10.0)
    arm_initial_weights(**ok)
    for key, value in (("cps_design_weights", np.array([0.0])),
                       ("acs_design_weights", np.array([np.nan])),
                       ("total_mass", -1.0), ("cps_copies", np.array([-1]))):
        with pytest.raises(ValueError):
            arm_initial_weights(**{**ok, key: value})


@given(
    rows=st.lists(st.tuples(st.floats(min_value=0.0, max_value=1e6), st.integers(0, 15)),
                  min_size=1, max_size=80)
)
def test_distinct_unit_ess_bounds(rows) -> None:
    w = np.array([r[0] for r in rows])
    units = np.array([r[1] for r in rows])
    if w.sum() == 0:
        return
    unit_ess = distinct_unit_ess(w, units)
    assert unit_ess <= kish_ess(w) * (1 + 1e-9)
    assert unit_ess <= len(np.unique(units[w > 0])) * (1 + 1e-9)
    assert np.isclose(distinct_unit_ess(w, np.arange(len(w))), kish_ess(w))


def test_group_key_normalization() -> None:
    assert target_split_group_key("irs_soi.state.ty2023.agi.06") == "irs_soi.state.agi.06"
    assert target_split_group_key("a.month2024_7.b") == "a.b"
    # District GEOIDs are 4-digit tokens and survive.
    assert target_split_group_key("irs_soi.cd.0601.v2024") == "irs_soi.cd.0601"
    assert target_split_group_key(
        "x.current_cd.0601.0123456789abcdef"
    ) == "x.current_cd.0601"
    # A district child takes its state parent's key.
    assert target_split_group_key("child.0601", "parent.06.ty2022") == "parent.06"


@given(st.lists(st.sampled_from(["ty2021", "cy2024", "v2024", "fy2023", "oep2025"]), max_size=3))
def test_vintage_tokens_never_change_a_role(vintages) -> None:
    base = "irs_soi.state.agi.band3.06"
    name = ".".join([*base.split(".")[:2], *vintages, *base.split(".")[2:]])
    assert target_split_group_key(name) == base
    assert target_split_role(target_split_group_key(name)) == target_split_role(base)


def test_role_rates_are_near_the_declared_fractions() -> None:
    roles = [target_split_role(f"k{i}") for i in range(20_000)]
    sealed = roles.count("sealed") / len(roles)
    holdout = roles.count("holdout") / len(roles)
    assert abs(sealed - 0.05) < 0.01
    assert abs(holdout - 0.095) < 0.01
    # Sealed is drawn first on its own salt.
    for i in range(200):
        key = f"k{i}"
        if hash_uniform(key, salt=TARGET_SPLIT_SEALED_SALT) < 0.05:
            assert target_split_role(key) == "sealed"
        elif hash_uniform(key, salt=TARGET_SPLIT_HOLDOUT_SALT) < 0.10:
            assert target_split_role(key) == "holdout"


def test_arm_validation() -> None:
    with pytest.raises(ValueError):
        SupportMixArm(budget=0, cps_income_years=(2024,))
    with pytest.raises(ValueError):
        SupportMixArm(budget=10, cps_income_years=(2024, 2024))
    with pytest.raises(ValueError):
        SupportMixArm(budget=10, cps_income_years=(2024,), acs_fill_share=1.5)
    arm = SupportMixArm(budget=600_000, cps_income_years=(2024, 2022), acs_fill_share=0.5, seed=2)
    assert arm.cps_income_years == (2022, 2024)
    assert arm.label == "b600k.cps2022-2024.acs050.s2"


@settings(max_examples=120, deadline=None)
@given(data=st.data())
def test_assemble_matches_naive_dense_construction(data) -> None:
    import scipy.sparse as sp

    n_rows = data.draw(st.integers(1, 30))
    n_concepts = data.draw(st.integers(1, 6))
    dense = np.array(data.draw(st.lists(
        st.lists(st.sampled_from([0.0, 0.0, 1.5, -2.0, 7.0]), min_size=n_concepts, max_size=n_concepts),
        min_size=n_rows, max_size=n_rows)))
    state = np.array(data.draw(st.lists(st.integers(0, 3), min_size=n_rows, max_size=n_rows)))
    cd = np.array(data.draw(st.lists(st.integers(0, 5), min_size=n_rows, max_size=n_rows)))
    targets = data.draw(st.lists(st.tuples(
        st.integers(0, n_concepts - 1), st.sampled_from(["national", "state", "cd"]), st.integers(0, 5)),
        min_size=1, max_size=25))
    concept = np.array([t[0] for t in targets])
    level = [t[1] for t in targets]
    geo = np.array([0 if t[1] == "national" else (t[2] % 4 if t[1] == "state" else t[2]) for t in targets])
    got = assemble_target_matrix(sp.csr_matrix(dense), {"state": state, "cd": cd}, concept, level, geo).toarray()
    expected = np.zeros((len(targets), n_rows))
    for t, (c, lv, g) in enumerate(zip(concept, level, geo, strict=True)):
        member = np.ones(n_rows, bool) if lv == "national" else ((state if lv == "state" else cd) == g)
        expected[t] = dense[:, c] * member
    np.testing.assert_array_equal(got, expected)
