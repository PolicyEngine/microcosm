"""The ACS arm of the SPM independence role (``with_acs_spm_independence_role``).

Examples and refusals. The for-all-inputs invariants (the RELSHIPP rule under
any row order, age blindness, which units lack a classified adult, and the
differential against the loader's CPS relationship encoding) are in
``test_us_acs_spm_role_properties.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime.acs_inputs import (
    ACS_SPM_ROLE_PARTITION,
    ACS_SPM_ROLE_RULE,
    map_acs_native_inputs,
    with_acs_spm_independence_role,
)
from microcosm.build.us_runtime.spm_composition import check_spm_composition
from microcosm.build.us_runtime.spm_role_source import (
    NATIVE_SPM_ROLE,
    SPM_ADULT_RULE,
    SPM_ROLE_RULE,
)
from microcosm.frame import Frame
from test_support.microcosm_build.us_acs_spm_role import acs_frame, role_by_person

# ---------------------------------------------------------------------------
# The rule, by relationship code
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "household,target,expected",
    [
        ([(20, 40)], 0, True),
        ([(20, 40), (21, 38)], 1, True),
        ([(20, 40), (23, 38)], 1, True),
        # An unmarried partner is not a family member (ASEC A_FAMREL 0).
        ([(20, 40), (22, 38)], 1, False),
        ([(20, 40), (24, 38)], 1, False),
        *[([(20, 40), (code, 30)], 1, False) for code in range(25, 37)],
        # A group-quarters record is one person, the head of its own unit.
        ([(37, 40)], 0, True),
        ([(38, 19)], 0, True),
    ],
)
def test_role_follows_relshipp(household, target, expected) -> None:
    frame = acs_frame([household])

    role = with_acs_spm_independence_role(frame).frame.table("person")[NATIVE_SPM_ROLE]

    assert role.dtype == np.dtype(bool)
    assert bool(role.iloc[target]) is expected


def test_role_ignores_age_so_a_minor_head_classifies_the_unit() -> None:
    # Two households whose only candidate adult is a 16-year-old reference
    # person (117 such ACS 2024 housing units) and a married 17-year-old.
    frame = acs_frame([[(20, 16), (28, 12)], [(20, 17), (21, 17), (25, 1)]])

    result = with_acs_spm_independence_role(frame)

    assert role_by_person(result.frame).tolist() == [True, False, True, True, False]
    assert check_spm_composition(result.frame).status == "PASS"
    assert result.provenance["housing_units_classified_only_by_role"] == 2
    assert result.provenance["independent_minor_persons"] == 3


def test_receipt_records_rule_branches_and_shares() -> None:
    frame = acs_frame(
        [
            [(20, 45), (21, 44), (25, 16), (25, 12)],
            [(20, 70)],
            [(37, 16)],
            [(38, 12)],
        ],
        typehugq=[1, 1, 2, 3],
    )

    result = with_acs_spm_independence_role(frame)

    weights = np.repeat([10.0, 20.0, 30.0, 40.0], [4, 1, 1, 1])
    role = np.asarray([1, 1, 0, 0, 1, 1, 1], dtype=bool)
    minor = np.asarray([0, 0, 1, 0, 0, 1, 0], dtype=bool)
    assert result.provenance == {
        "column": NATIVE_SPM_ROLE,
        "entity": "person",
        "source_columns": ["RELSHIPP"],
        "rule": ACS_SPM_ROLE_RULE,
        "partition": ACS_SPM_ROLE_PARTITION,
        "asec_rule": SPM_ROLE_RULE,
        "adult_rule": SPM_ADULT_RULE,
        "provenance": "acs_2024_1yr_native",
        "persons": 7,
        "spm_units": 4,
        "role_true_persons": 5,
        "role_branches": {
            "reference_person": 2,
            "spouse_of_reference_person": 1,
            "group_quarters_sole_person": 2,
        },
        "persons_aged_15_to_17": 2,
        "independent_minor_persons": 1,
        "housing_units_classified_only_by_role": 0,
        "group_quarters_units_without_classified_adult": 1,
        "weighted_role_share": pytest.approx(weights[role].sum() / weights.sum()),
        "weighted_minor_role_share": pytest.approx(
            weights[role & minor].sum() / weights[minor].sum()
        ),
    }


def test_group_quarters_child_is_counted_not_refused() -> None:
    """Intended gap: a GQ child is a one-person unit with no classified adult.

    No role can classify an under-15 as an adult; the source declares group
    quarters outside the ACS SPM universe (``spm_universe_source``), so the
    derivation counts the unit instead of refusing the spine.
    """

    result = with_acs_spm_independence_role(acs_frame([[(20, 40)], [(37, 12)]]))

    assert role_by_person(result.frame).tolist() == [True, True]
    assert result.provenance["group_quarters_units_without_classified_adult"] == 1
    composition = check_spm_composition(result.frame)
    assert composition.status == "FAIL"
    assert composition.details["n_units_without_classified_adult"] == 1


def test_input_frame_is_not_mutated() -> None:
    frame = acs_frame([[(20, 40), (25, 10)]])
    before = frame.table("person").copy()

    with_acs_spm_independence_role(frame)

    pd.testing.assert_frame_equal(frame.table("person"), before)
    assert NATIVE_SPM_ROLE not in frame.table("person")


def test_the_shared_native_mapping_never_emits_therole_by_person() -> None:
    """The stacked pool maps ACS through ``map_acs_native_inputs`` too.

    Its ASEC arm carries no role and its ACS-row rule is an open decision
    (docs/us-spm-role-stage.md §7 Q1), so the role stays out of the shared
    mapping and its operator-boundary receipt.
    """

    frame = acs_frame([[(20, 40), (21, 39)]])
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["person"]["AGEP"] = tables["person"].pop("age")
    raw = Frame(tables, frame.schema, {"household": frame.weights_for("household")})

    mapped = map_acs_native_inputs(raw)

    assert NATIVE_SPM_ROLE not in mapped.frame.table("person")
    assert NATIVE_SPM_ROLE not in mapped.native_inputs


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def _with_person_column(frame: Frame, column: str, values) -> Frame:
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["person"][column] = values
    return Frame(tables, frame.schema, {"household": frame.weights_for("household")})


def _without_person_column(frame: Frame, column: str) -> Frame:
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["person"] = tables["person"].drop(columns=[column])
    return Frame(tables, frame.schema, {"household": frame.weights_for("household")})


def test_refuses_to_overwrite_an_existingrole_by_person() -> None:
    frame = _with_person_column(acs_frame([[(20, 40)]]), NATIVE_SPM_ROLE, [True])

    with pytest.raises(ValueError, match="refuses to overwrite"):
        with_acs_spm_independence_role(frame)


@pytest.mark.parametrize("column", ["RELSHIPP", "age"])
def test_refuses_a_missing_required_column(column: str) -> None:
    frame = _without_person_column(acs_frame([[(20, 40)]]), column)

    with pytest.raises(ValueError, match=f"requires person column.*{column}"):
        with_acs_spm_independence_role(frame)


@pytest.mark.parametrize("value", [np.nan, 19, 39, 20.5])
def test_refuses_blank_or_off_domain_relshipp(value: float) -> None:
    frame = _with_person_column(
        acs_frame([[(20, 40), (25, 5)]]), "RELSHIPP", [20, value]
    )

    with pytest.raises(ValueError, match="blank or off-domain"):
        with_acs_spm_independence_role(frame)


def test_refuses_an_unobserved_age() -> None:
    frame = _with_person_column(acs_frame([[(20, 40)]]), "age", [np.nan])

    with pytest.raises(ValueError, match="observed age"):
        with_acs_spm_independence_role(frame)


@pytest.mark.parametrize(
    "households,spm_units",
    [
        # One household split into two SPM units.
        ([[(20, 40), (34, 30)]], [1, 2]),
        # One SPM unit spanning two households.
        ([[(20, 40)], [(20, 50)]], [1, 1]),
    ],
)
def test_refuses_a_non_household_partition(households, spm_units) -> None:
    frame = acs_frame(households, spm_units=spm_units)

    with pytest.raises(ValueError, match="only on the household SPM partition"):
        with_acs_spm_independence_role(frame)


@pytest.mark.parametrize(
    "household,problem",
    [
        ([(25, 10)], "not exactly one head"),
        ([(20, 40), (20, 41)], "not exactly one head"),
        ([(37, 40), (38, 41)], "not exactly one head"),
        ([(20, 40), (37, 41)], "not exactly one head"),
        ([(20, 40), (21, 40), (23, 40)], "more than one RELSHIPP 21/23 spouse"),
        ([(37, 40), (25, 10)], "group-quarters person shares the unit"),
        ([(20, 14)], "no classified adult"),
        ([(20, 14), (25, 3)], "no classified adult"),
    ],
)
def test_refuses_a_unit_that_breaks_the_household_structure(household, problem) -> None:
    with pytest.raises(ValueError, match=problem):
        with_acs_spm_independence_role(acs_frame([[(20, 40)], household]))


@pytest.mark.parametrize(
    "household,typehugq,match",
    [
        ([(20, 40)], 3, "disagree about group quarters"),
        ([(37, 40)], 1, "disagree about group quarters"),
        ([(20, 40)], 4, r"TYPEHUGQ in \{1, 2, 3\}"),
        ([(20, 40)], np.nan, r"TYPEHUGQ in \{1, 2, 3\}"),
    ],
)
def test_refuses_typehugq_that_contradicts_relshipp(household, typehugq, match) -> None:
    frame = acs_frame([[(20, 40)], household], typehugq=[1, typehugq])

    with pytest.raises(ValueError, match=match):
        with_acs_spm_independence_role(frame)
