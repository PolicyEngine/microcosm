"""Properties of whole-vector retirement participation hot-deck draws."""

import numpy as np
import pandas as pd
import pytest

from microcosm.build.source_runtime import SourceRuntimeError
from microcosm.build.us_runtime.joint_retirement import (
    RETIREMENT_COLUMNS,
    draw_joint_retirement_vectors,
)


def _donors():
    vectors = np.array([
        [0, 0, 10, 20, 0],  # IRA only
        [40, 10, 0, 0, 0],  # Employer only
        [20, 5, 10, 10, 0],  # Both
        [0, 0, 20, 0, 10],  # IRA + SE pension
    ], dtype=float)
    donor = pd.DataFrame(vectors, columns=RETIREMENT_COLUMNS)
    return donor.assign(donor_id=np.arange(4), age=45, has_wages=True,
                        has_self_employment=True, rank_value=1, weight=[1., 2., 3., 4.])


def _recipients(n=1000):
    return pd.DataFrame({"person_id": np.arange(n), "age": 45,
                         "has_wages": True, "has_self_employment": True,
                         "rank_value": 1, "weight": 1.})


def _draw(donor, recipient, seed=0, positive_only=True):
    return draw_joint_retirement_vectors(donor, recipient, donor_weight="weight",
                                        recipient_weight="weight", seed=seed,
                                        age_breaks=[35, 50, 65], positive_only=positive_only)


@pytest.mark.parametrize("seed", range(10))
def test_nonnegative_complete_tuples_preserve_joint_shares_and_amounts(seed):
    donor = _donors()
    result = _draw(donor, _recipients(), seed)
    assert np.isfinite(result.to_numpy()).all()
    assert (result.to_numpy() >= 0).all()
    source_tuples = set(map(tuple, donor[list(RETIREMENT_COLUMNS)].to_numpy()))
    assert set(map(tuple, result.to_numpy())) == source_tuples
    ira = result[list(RETIREMENT_COLUMNS[2:4])].sum(axis=1) > 0
    employer = result[[*RETIREMENT_COLUMNS[:2], RETIREMENT_COLUMNS[4]]].sum(axis=1) > 0
    assert (ira & ~employer).mean() == .1
    assert (~ira & employer).mean() == .2
    assert (ira & employer).mean() == .7
    expected = np.average(donor[list(RETIREMENT_COLUMNS)], axis=0, weights=donor.weight)
    np.testing.assert_allclose(result.mean(), expected, rtol=0, atol=1e-12)


def test_row_permutation_preserves_identity_keyed_draws():
    donor, recipient = _donors(), _recipients()
    original = _draw(donor, recipient, 4)
    shuffled = _draw(donor.sample(frac=1, random_state=1),
                     recipient.sample(frac=1, random_state=2), 4)
    pd.testing.assert_frame_equal(original, shuffled.reindex(original.index))


def test_income_constraints_select_compatible_tuple_without_erasing_legs():
    recipient = _recipients(20)
    recipient["has_wages"] = False
    recipient["has_self_employment"] = False
    result = _draw(_donors(), recipient)
    np.testing.assert_allclose(result.to_numpy(), np.tile([0, 0, 10, 20, 0], (20, 1)))


def test_zero_weight_donors_are_excluded_and_puf_nonsavers_remain_joint_zeros():
    donor = _donors()
    donor.loc[0, "weight"] = 0
    donor.loc[1, list(RETIREMENT_COLUMNS)] = 0
    result = _draw(donor, _recipients(900), positive_only=False)
    assert not ((result.traditional_ira_contributions_desired == 10) &
                (result.roth_ira_contributions_desired == 20)).any()
    assert (result.sum(axis=1) == 0).mean() == pytest.approx(2/9)


@pytest.mark.parametrize("column,value", [("weight", -1),
                                         (RETIREMENT_COLUMNS[0], np.nan),
                                         (RETIREMENT_COLUMNS[1], -1)])
def test_invalid_source_fails_closed(column, value):
    donor = _donors()
    donor.loc[0, column] = value
    with pytest.raises(SourceRuntimeError, match="nonfinite|nonnegative"):
        _draw(donor, _recipients())
