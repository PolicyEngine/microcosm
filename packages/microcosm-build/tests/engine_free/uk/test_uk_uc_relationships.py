"""Tests split from packages/microcosm-build/tests/test_uk_uc_relationships.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.uk_uc_relationships import *


def test_frs_roles_distinguish_partner_from_older_dependent():
    person = pd.DataFrame(
        {
            "person_benunit_id": [20, 10, 20, 10, 20, 30],
            "is_benunit_head": [False, True, True, False, False, True],
            "is_parent": [True, True, True, False, False, False],
            "age": [36, 42, 38, 19, 18, 17],
        },
        index=[8, 6, 4, 2, 0, 9],
    )
    benunit = pd.DataFrame(
        {"benunit_id": [30, 20, 10], "dependent_children": [0, 1, 1]}
    )
    np.testing.assert_array_equal(
        frs_uc_claimant_mask(person, benunit), [True, True, True, False, False, True]
    )


def test_childless_cohabiting_partners_are_both_claimants():
    person = pd.DataFrame(
        {
            "person_benunit_id": [7, 7],
            "is_benunit_head": [True, False],
            "is_parent": [False, False],
        }
    )
    benunit = pd.DataFrame({"benunit_id": [7], "dependent_children": [0]})
    np.testing.assert_array_equal(frs_uc_claimant_mask(person, benunit), [True, True])


def test_couple_mask_uses_source_roles_and_returns_benunit_row_order():
    from microcosm.build.uk_runtime.uc_relationships import frs_uc_couple_mask

    person = pd.DataFrame(
        {
            "person_benunit_id": [20, 10, 20, 10, 30, 30],
            "is_benunit_head": [False, True, True, False, True, False],
            "is_parent": [True, True, True, False, False, False],
            "age": [36, 42, 38, 19, 26, 24],
        },
        index=[8, 6, 4, 2, 0, 9],
    )
    benunit = pd.DataFrame(
        {
            "benunit_id": [30, 20, 10],
            "dependent_children": [0, 1, 1],
            "is_married": [False, False, True],
        }
    )
    # Childless cohabitation; cohabiting parents; a lone married parent
    # whose 19-year-old dependent must not become their partner.
    np.testing.assert_array_equal(
        frs_uc_couple_mask(person, benunit), [True, True, False]
    )


@pytest.mark.parametrize("bad_role", [None, 2, "False"])
def test_invalid_roles_fail_instead_of_becoming_truthy(bad_role):
    person = pd.DataFrame(
        {"person_benunit_id": [7], "is_benunit_head": [bad_role], "is_parent": [False]}
    )
    benunit = pd.DataFrame({"benunit_id": [7], "dependent_children": [1]})
    with pytest.raises(ValueError, match="is_benunit_head"):
        frs_uc_claimant_mask(person, benunit)


def test_missing_membership_fails():
    person = pd.DataFrame(
        {"person_benunit_id": [9], "is_benunit_head": [True], "is_parent": [False]}
    )
    benunit = pd.DataFrame({"benunit_id": [7], "dependent_children": [0]})
    with pytest.raises(ValueError, match="membership"):
        frs_uc_claimant_mask(person, benunit)


def test_more_than_two_claimants_requires_source_reconciliation():
    person = pd.DataFrame(
        {
            "person_benunit_id": [7, 7, 7],
            "is_benunit_head": [True, False, False],
            "is_parent": [False, False, False],
        }
    )
    benunit = pd.DataFrame({"benunit_id": [7], "dependent_children": [0]})
    with pytest.raises(ValueError, match="one or two"):
        frs_uc_claimant_mask(person, benunit)
