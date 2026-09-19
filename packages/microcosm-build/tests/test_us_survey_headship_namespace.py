"""Lossless legacy evidence namespacing before source-qualified headship.

The pure transport cases issue no owner. The source case uses the maintained
tiny invented files through the genuine preparation and role qualifier.
"""

import numpy as np
import pandas as pd
import pytest
from test_us_current_asec_demographics import _demographic_arguments
from test_us_current_survey_household_roles import _frame, _qualified, _receiving

from microcosm.build.us_runtime import current_survey_household_roles as roles
from microcosm.build.us_runtime import survey_population_preparation as source

LEGACY = "legacy_prepared_is_household_head"


@pytest.mark.parametrize("dtype", ["bool", "boolean"])
def test_normalization_preserves_legacy_storage_outside_canonical_namespace(dtype):
    frame = _frame(pd.DataFrame({"person_id": np.arange(1, 4, dtype=np.int64)}))
    frame.person["A_AGE"] = np.array([35, 15, 85], dtype=np.int64)
    frame.person["is_household_head"] = pd.array(
        [True, False, pd.NA if dtype == "boolean" else False], dtype=dtype
    )
    before = source._frame_identity(frame)
    normalized = source._normalized_source_copy(frame)
    assert "is_household_head" not in normalized.person
    pd.testing.assert_series_equal(
        frame.person.is_household_head.rename(LEGACY), normalized.person[LEGACY]
    )
    assert source._frame_identity(frame) == before
    source._verify_normalized_copy(frame, normalized)


def test_reserved_legacy_name_cannot_be_supplied_by_an_input_frame():
    frame = _frame(pd.DataFrame({"person_id": np.array([1], dtype=np.int64)}))
    frame.person["A_AGE"] = np.array([35], dtype=np.int64)
    frame.person[LEGACY] = True
    with pytest.raises(ValueError, match="LEGACY_HEADSHIP_NAMESPACE_COLLISION"):
        source._normalized_source_copy(frame)


@pytest.mark.parametrize("mutation", ["value", "dtype", "canonical", "rename"])
def test_lossless_verifier_refuses_changed_or_reintroduced_headship(mutation):
    frame = _frame(pd.DataFrame({"person_id": np.arange(1, 3, dtype=np.int64)}))
    frame.person["A_AGE"] = np.array([35, 15], dtype=np.int64)
    frame.person["is_household_head"] = pd.array([True, False], dtype="boolean")
    normalized = source._normalized_source_copy(frame)
    assert LEGACY in normalized.person
    if mutation == "value":
        normalized.person.loc[0, LEGACY] = False
    elif mutation == "dtype":
        normalized.person[LEGACY] = normalized.person[LEGACY].astype(bool)
    elif mutation == "canonical":
        normalized.person["is_household_head"] = False
    else:
        normalized.person.rename(columns={LEGACY: "unsupported_alias"}, inplace=True)
    with pytest.raises(ValueError, match="NORMALIZED_"):
        source._verify_normalized_copy(frame, normalized)


def test_legacy_disagreement_remains_visible_but_cannot_issue_a_canonical_role():
    qualified = _qualified()
    receiving = _receiving(qualified.rows)
    original = receiving.person[roles.support_source_id_column("person")]
    # A false legacy reference flag contradicts original10, and original30 has
    # no qualified role. Both facts must remain explicit in the diagnostic.
    receiving.person[LEGACY] = pd.array([False] * len(original), dtype="boolean")
    before = receiving.person.copy(deep=True)
    result = roles.household_role_columns_for_population(qualified, receiving)[
        "person", roles.CANONICAL_COLUMN
    ]
    assert result[original.to_numpy() == 10].all()
    assert result[original.to_numpy() == 30].isna().all()
    audit = roles.household_role_reconciliation(qualified, receiving)
    assert not audit["binding_would_refuse"]
    legacy = audit["legacy_prepared_comparison"]
    assert legacy["column"] == LEGACY
    assert legacy["conflicting_cells"] == 2
    assert legacy["known_qualified_unbound"] == 2
    assert legacy["canonical_authority"] is False
    pd.testing.assert_frame_equal(receiving.person, before)


def test_genuine_gq_sources_keep_legacy_evidence_and_qualified_unknowns(
    tmp_path, monkeypatch
):
    prepared = source.prepare_authenticated_survey_population(
        **_demographic_arguments(tmp_path, monkeypatch)
    )
    view = prepared.checked_view()
    assert "is_household_head" not in view.frame.person
    assert LEGACY in view.frame.person
    evidence = view.receipt["origins"]["legacy_headship_namespace"]
    assert evidence["from"] == "is_household_head"
    assert evidence["to"] == LEGACY
    assert evidence["values_changed"] is False
    assert evidence["canonical_authority"] is False
    qualified = roles.qualify_current_survey_household_roles(prepared)
    rows = qualified.rows
    gq = rows[roles.UNBOUND_REASON_COLUMN].eq(5)
    assert gq.sum() == 2 and rows.loc[gq, roles.VALUE_COLUMN].isna().all()
    assert rows.loc[~gq, roles.VALUE_KNOWN_COLUMN].all()
    for original in prepared._checked()[2].source_frames:
        assert LEGACY not in original.person
    assert prepared.checked_view().frame.n("person") == 9
