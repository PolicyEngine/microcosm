"""Which rows carry each target: the spine-target diagnostic (#1729).

Every number below is made up for the test.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy import sparse

from microcosm.calibrate.spine_target_shares import (
    spine_target_shares,
    spine_target_shares_for_problem,
)

# Six records: two donor rows (heavy) and four survey rows (light).
SPINES = np.asarray(["donor", "donor", "survey", "survey", "survey", "survey"])
WEIGHTS = np.asarray([30.0, 30.0, 10.0, 10.0, 10.0, 10.0])
AREAS = np.asarray(["A", "B", "A", "A", "B", "B"])
MATRIX = np.asarray(
    [
        # Population of area A: every record in A, one person each.
        [1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        # A tax amount only the donor rows carry.
        [500.0, 300.0, 0.0, 0.0, 0.0, 0.0],
        # A net amount with losses on one survey row.
        [100.0, 0.0, -40.0, 0.0, 20.0, 0.0],
        # Nothing carries it.
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
)
NAMES = ["pop_A", "tax", "net", "empty"]


@pytest.fixture
def shares():
    return spine_target_shares(
        sparse.csr_array(MATRIX),
        WEIGHTS,
        SPINES,
        names=NAMES,
        target_scopes=[("area", "A"), None, None, ("area", "B")],
        scope_codes={"area": AREAS},
    )


def test_a_share_is_the_part_of_the_target_a_spine_carries(shares) -> None:
    assert shares.spines == ("donor", "survey")
    np.testing.assert_allclose(shares.estimates, MATRIX @ WEIGHTS)
    # Area A's population: 30 on the donor row, 20 on two survey rows.
    np.testing.assert_allclose(shares.share[0], [0.6, 0.4])
    np.testing.assert_allclose(shares.carried[0], [30.0, 20.0])
    # The donor rows carry the tax amount alone.
    np.testing.assert_allclose(shares.share[1], [1.0, 0.0])
    # Shares sum to one for every carried target.
    np.testing.assert_allclose(shares.share[:3].sum(axis=1), 1.0)
    assert np.isnan(shares.share[3]).all()


def test_the_mass_share_is_taken_in_the_scope_of_the_target(shares) -> None:
    np.testing.assert_allclose(shares.overall_mass_share, [0.6, 0.4])
    # In area A the donor row holds 30 of 50; an unscoped target reads all.
    np.testing.assert_allclose(shares.mass_share[0], [0.6, 0.4])
    np.testing.assert_allclose(shares.mass_share[1], [0.6, 0.4])
    np.testing.assert_allclose(shares.mass_share[3], [0.6, 0.4])
    # Area A's population is carried in proportion to mass: no excess.
    np.testing.assert_allclose(shares.excess[0], 0.0, atol=1e-12)
    np.testing.assert_allclose(shares.lift[0], 1.0)
    # The tax amount is over-carried by the donor rows.
    assert shares.excess[1, 0] == pytest.approx(0.4)
    assert shares.lift[1, 0] == pytest.approx(1.0 / 0.6)


def test_mixed_signs_are_shared_on_absolute_contributions(shares) -> None:
    assert shares.mixed_sign.tolist() == [False, False, True, False]
    # |100*30| donor against |-40*10| + |20*10| survey.
    np.testing.assert_allclose(shares.share[2], [3000 / 3600, 600 / 3600])
    np.testing.assert_allclose(shares.carried[2], [3000.0, -200.0])


def test_the_ranking_puts_the_targets_a_spine_over_carries_first(shares) -> None:
    ranking = shares.ranking("donor")
    assert [row["name"] for row in ranking] == ["tax", "net", "pop_A"]
    assert ranking[0]["excess"] == pytest.approx(0.4)
    assert shares.ranking("donor", top=1)[0]["name"] == "tax"
    with pytest.raises(ValueError, match="spine must be one of"):
        shares.ranking("puf")


def test_the_summary_counts_the_targets_a_spine_carries_almost_alone(shares) -> None:
    summary = shares.summary(["pop", "money", "money", "money"])
    assert [entry["group"] for entry in summary] == ["pop", "money"]
    money = summary[1]
    assert (money["n_targets"], money["n_measured"]) == (3, 2)
    assert money["spines"]["donor"]["n_carried_almost_alone"] == 1
    assert money["spines"]["survey"]["n_carried_almost_alone"] == 0
    with pytest.raises(ValueError, match="do not align"):
        shares.summary(["pop"])


def test_carriers_count_a_source_household_once() -> None:
    # The two donor rows are copies of one source household.
    sources = np.asarray(["d1", "d1", "s1", "s2", "s3", "s4"])
    by_row = spine_target_shares(MATRIX, WEIGHTS, SPINES)
    by_source = spine_target_shares(MATRIX, WEIGHTS, SPINES, source_ids=sources)
    assert by_row.carriers[1].tolist() == [2, 0]
    assert by_source.carriers[1].tolist() == [1, 0]
    # A zero-weight record is not a carrier.
    zeroed = spine_target_shares(MATRIX, [30.0, 0.0, 10, 10, 10, 10], SPINES)
    assert zeroed.carriers[1].tolist() == [1, 0]


def test_a_floor_withholds_a_target_with_few_carriers() -> None:
    shares = spine_target_shares(
        MATRIX, WEIGHTS, SPINES, names=NAMES, minimum_carriers=2
    )
    # pop_A has one donor carrier, net has one donor and one each... below 2.
    assert shares.withheld.tolist() == [True, False, True, False]
    assert np.isnan(shares.share[0]).all() and np.isnan(shares.carried[0]).all()
    assert np.isnan(shares.estimates[0])
    assert (shares.carriers[0] == -1).all()
    rows = shares.rows()
    assert set(rows[0]) == {"name", "scope", "mass_share", "withheld"}
    assert rows[1]["share"] == {"donor": 1.0, "survey": 0.0}
    # The report is strict JSON, with no NaN in it.
    report = shares.report()
    assert report["n_withheld"] == 2
    json.dumps(report, allow_nan=False)
    assert [row["name"] for row in shares.ranking("donor")] == ["tax"]


def test_the_report_names_each_target_and_its_dominant_spine(shares) -> None:
    report = shares.report()
    json.dumps(report, allow_nan=False)
    assert report["spines"] == ["donor", "survey"]
    assert report["overall_mass_share"] == pytest.approx({"donor": 0.6, "survey": 0.4})
    by_name = {row["name"]: row for row in report["targets"]}
    assert by_name["tax"]["dominant_spine"] == "donor"
    assert by_name["tax"]["dominant_share"] == 1.0
    assert by_name["pop_A"]["scope"] == {"kind": "area", "code": "A"}
    assert by_name["empty"]["dominant_spine"] is None
    assert by_name["empty"]["share"] == {"donor": None, "survey": None}
    frame = shares.frame()
    assert len(frame) == 8 and set(frame["spine"]) == {"donor", "survey"}


def test_a_compiled_problem_is_read_at_its_initial_weights() -> None:
    from types import SimpleNamespace

    problem = SimpleNamespace(
        matrix=sparse.csr_array(MATRIX),
        names=tuple(NAMES),
        initial_weights=SimpleNamespace(values=WEIGHTS),
    )
    at_start = spine_target_shares_for_problem(problem, SPINES)
    np.testing.assert_allclose(at_start.share[1], [1.0, 0.0])
    assert at_start.names == tuple(NAMES)
    # Any other weights: a solve that moved weight off the donor rows.
    moved = spine_target_shares_for_problem(
        problem, SPINES, weights=[10.0, 10.0, 20.0, 20.0, 20.0, 20.0]
    )
    assert moved.overall_mass_share[0] == pytest.approx(0.2)


@pytest.mark.parametrize(
    ("arguments", "match"),
    [
        (dict(weights=WEIGHTS[:5]), "weights have shape"),
        (dict(weights=-WEIGHTS), "finite and non-negative"),
        (dict(weights=np.zeros(6)), "positive total"),
        (dict(spines=SPINES[:5]), "spines has shape"),
        (dict(names=["only"]), "1 names for 4 targets"),
        (dict(minimum_carriers=0), "positive integer"),
        (dict(target_scopes=[None] * 3), "3 scopes for 4 targets"),
        (dict(target_scopes=[("area", "A")] + [None] * 3), "no record codes"),
    ],
)
def test_misaligned_or_invalid_inputs_are_refused(arguments, match) -> None:
    call = dict(weights=WEIGHTS, spines=SPINES)
    call.update(arguments)
    with pytest.raises(ValueError, match=match):
        spine_target_shares(MATRIX, call.pop("weights"), call.pop("spines"), **call)


def test_a_scope_no_record_is_in_has_no_mass_share() -> None:
    shares = spine_target_shares(
        MATRIX,
        WEIGHTS,
        SPINES,
        target_scopes=[("area", "Z"), None, None, None],
        scope_codes={"area": AREAS},
    )
    assert np.isnan(shares.mass_share[0]).all()
    assert np.isnan(shares.lift[0]).all()
