"""The Lifetime ISA ownership anchor to HMRC's 2024-25 payers (microcosm#1415)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.stochastic_assignment import stable_identity_uniforms
from microcosm.build.uk_runtime.was_lisa import (
    OwnershipAnchorSpec,
    OwnershipImputation,
    OwnershipModelSpec,
    WASLISAError,
    anchor_lifetime_isa_ownership,
)
from test_support.microcosm_build.uk_isa_statistics import LISA_ANCHOR_OPERATION

SPEC = OwnershipModelSpec(
    output="has_lifetime_isa",
    seed=0,
    salt="was_lisa:has_lifetime_isa",
    penalty_c=1.0,
    solver="lbfgs",
    max_iter=5000,
    age_group_lower_bounds=(0, 25, 35, 45, 55),
    log1p_predictors=(),
    indicator_predictors=(),
    count_predictors=(),
    minimum_age=18,
)
ANCHOR_PARAMETERS = {
    key: value for key, value in LISA_ANCHOR_OPERATION.items() if key != "kind"
}
ANCHOR = OwnershipAnchorSpec.from_parameters(ANCHOR_PARAMETERS)


def _ownership(n: int = 4000, seed: int = 2):
    rng = np.random.default_rng(seed)
    recipient = pd.DataFrame(
        {"person_id": np.arange(n) + 1, "age": rng.integers(0, 90, n).astype(float)}
    )
    probability = np.where(recipient["age"] >= 18, rng.uniform(0.001, 0.05, n), 0.0)
    uniforms = stable_identity_uniforms(
        recipient["person_id"].to_numpy(), seed=SPEC.seed, salt=SPEC.salt
    )
    holds = (recipient["age"].to_numpy() >= 18) & (uniforms < probability)
    ownership = OwnershipImputation(
        holds=holds, probability=probability, receipt={}, fit_weight_records=()
    )
    weights = rng.uniform(100, 300, n)
    survives = rng.random(n) < 0.95
    return recipient, ownership, weights, survives


def _counted(recipient, holds, weights, survives) -> float:
    ages = recipient["age"].to_numpy()
    counted = (ages >= 18) & (ages <= 49)
    return float(weights[holds & survives & counted].sum())


def _anchor(target, recipient, ownership, weights, survives):
    return anchor_lifetime_isa_ownership(
        ownership,
        recipient,
        spec=SPEC,
        anchor=ANCHOR,
        target_holders=target,
        person_weights=weights,
        survives_cap=survives,
    )


def test_the_anchor_reaches_the_target_without_removing_holders():
    recipient, ownership, weights, survives = _ownership()
    before = _counted(recipient, ownership.holds, weights, survives)
    target = 2.0 * before
    anchored = _anchor(target, recipient, ownership, weights, survives)
    after = _counted(recipient, anchored.holds, weights, survives)
    assert target <= after <= target + weights.max()
    assert not (ownership.holds & ~anchored.holds).any()
    ages = recipient["age"].to_numpy()
    added = anchored.holds & ~ownership.holds
    assert ((ages[added] >= 18) & (ages[added] <= 47)).all()
    receipt = anchored.receipt["anchor"]
    assert receipt["removed_holders"] == 0
    assert receipt["logit_shift"] > 0
    assert receipt["saturated"] is None
    assert receipt["holders_before"] == pytest.approx(before)
    assert receipt["holders_after"] == pytest.approx(after)


def test_holders_outside_the_counted_ages_do_not_count():
    recipient, ownership, weights, survives = _ownership()
    ages = recipient["age"].to_numpy()
    older = ownership.holds & survives & (ages > 49)
    assert older.any()
    anchored = _anchor(0.0, recipient, ownership, weights, survives)
    assert anchored.receipt["anchor"]["holders_before"] == pytest.approx(
        _counted(recipient, ownership.holds, weights, survives)
    )


def test_holders_at_or_above_the_target_are_left_as_drawn():
    recipient, ownership, weights, survives = _ownership()
    before = _counted(recipient, ownership.holds, weights, survives)
    anchored = _anchor(0.5 * before, recipient, ownership, weights, survives)
    np.testing.assert_array_equal(anchored.holds, ownership.holds)
    np.testing.assert_array_equal(anchored.probability, ownership.probability)
    receipt = anchored.receipt["anchor"]
    assert receipt["logit_shift"] == 0.0
    assert receipt["saturated"] is None
    assert receipt["holders_added_weight"] == 0.0


def test_an_unreachable_target_is_named():
    recipient, ownership, weights, survives = _ownership()
    anchored = _anchor(10.0 * weights.sum(), recipient, ownership, weights, survives)
    receipt = anchored.receipt["anchor"]
    assert receipt["saturated"] == "target_above_reach"
    assert receipt["removed_holders"] == 0
    assert not (ownership.holds & ~anchored.holds).any()


@pytest.mark.parametrize(
    "change",
    [
        {"target": "holders_from_was"},
        {"tax_year": 2023},
        {"counted_ages": {"minimum": 18, "maximum": 45}},
        {"ages": {"minimum": 40, "maximum": 30}},
    ],
)
def test_the_anchor_declaration_is_checked(change):
    with pytest.raises(WASLISAError):
        OwnershipAnchorSpec.from_parameters({**ANCHOR_PARAMETERS, **change})
