"""Invented detached-value contracts; these objects confer no source authority."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from test_us_current_asec_other_disability_source import row
from test_us_current_asec_other_disability_source import values as source_values

from microcosm.build.us_runtime import (
    current_survey_other_disability_completion as values,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights


def two_clone_receiver(originals):
    """An invented receiver, not an issued host or substitute survey spine."""
    n = len(originals)
    ids = np.arange(n * 2, dtype="int64")
    source_ids = np.repeat(originals.index.to_numpy(), 2)
    origin = originals.reindex(source_ids)
    people = pd.DataFrame(
        {
            "person_id": ids,
            "age": origin.age.to_numpy(dtype="float64"),
            "unrelated_fixture_leaf": ids.astype("float64") + 20,
            values.provenance.support_source_id_column("person"): source_ids,
            values.provenance.support_clone_index_column("person"): np.tile([0, 1], n),
            values.provenance.spine_source_id_column(
                "person"
            ): origin.native_person_id.to_numpy(dtype="int64"),
            values.provenance.support_channel_column("person"): pd.array(
                origin.source.to_numpy(), dtype="string"
            ),
        }
    )
    for entity in US_SCHEMA.group_entities:
        people[US_SCHEMA.membership_column(entity)] = ids
    tables = {
        "person": people,
        **{e: pd.DataFrame({e + "_id": ids}) for e in US_SCHEMA.group_entities},
    }
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.ones(n * 2), WeightKind.IMPORTANCE)},
        pd.Series(["invented"] * (n * 2), dtype="string"),
    )


def invented_case():
    observed = source_values(
        row(DIS_VAL1="200"),
        row(DIS_YN="2", DIS_SC1="0", DIS_VAL1="0"),
        row(DIS_SC1="1", DIS_VAL1="500"),
        row(DIS_YN="1", DIS_SC1="0", DIS_VAL1="0"),
        row(A_AGE="14", DIS_YN="0", DIS_SC1="0", DIS_VAL1="0"),
    )
    originals = pd.DataFrame(
        {
            "native_person_id": [101, 102, 103, 104, 105, 800, 801],
            "source": pd.array(["asec"] * 5 + ["acs"] * 2, dtype="string"),
            "age": np.array([40, 40, 40, 40, 14, 15, 14], dtype="float64"),
        },
        index=pd.Index(
            [10, 11, 12, 13, 14, 9000, 9001], name="person_id", dtype="int64"
        ),
    )
    receiver = two_clone_receiver(originals)
    features = pd.DataFrame(
        [[15.0, 0.0, 0.0]],
        index=pd.Index([9000], name="person_id", dtype="int64"),
        columns=values.FEATURES,
    )
    matrix = values.model_input.encode_recipient_matrix(
        features, entity="person", entity_ids=features.index.to_numpy()
    )
    q = values.QualifiedOtherDisabilityCompletion(
        observed,
        observed,
        receiver,
        pd.DataFrame(index=observed.person.index),
        originals,
        features,
        matrix,
        b"{}",
        b"{}",
        {},
    )
    return q, receiver


def test_unknown_observed_zero_and_modeled_zero_remain_distinct():
    q, receiver = invented_case()
    before = values.qualified_seal(q)
    original_frame = values.source._frame_identity(receiver)
    result = values.attach_other_disability_completion(
        q, receiver, pd.Series(0.0, index=q.recipient_features.index)
    )
    columns = pd.DataFrame({name: col for (_, name), col in result.columns.items()})
    by_original = dict(
        zip(
            receiver.person.person_id,
            receiver.person[values.provenance.support_source_id_column("person")],
            strict=True,
        )
    )
    for original, amount, source_known, canonical_known, origin in (
        (10, 200, True, True, "observed"),
        (11, 0, True, True, "observed"),
        (12, 0, True, True, "observed"),
        (13, None, False, False, "source_unresolved"),
        (14, None, False, False, "outside_model_universe"),
        (9000, 0, False, True, "modeled"),
        (9001, None, False, False, "outside_model_universe"),
    ):
        selected = columns.loc[
            [pid for pid, oid in by_original.items() if oid == original]
        ]
        assert len(selected) == 2
        if amount is None:
            assert selected[values.observed.OUTPUT].isna().all()
        else:
            assert selected[values.observed.OUTPUT].eq(amount).all()
        assert (
            selected[values.observed.attached_name(values.observed.KNOWN_COLUMN)]
            .eq(source_known)
            .all()
        )
        assert selected[values.CANONICAL_KNOWN].eq(canonical_known).all()
        assert selected[values.VALUE_ORIGIN].eq(origin).all()
    assert values.qualified_seal(q) == before
    assert values.source._frame_identity(receiver) == original_frame
    assert (
        result.receipt["draws_consumed"] == 1
        and result.receipt["clone_redraw_issued"] is False
    )


def test_receiver_row_permutation_preserves_original_assignment():
    q, receiver = invented_case()
    draws = pd.Series(3.25, index=q.recipient_features.index)
    expected = values.attach_other_disability_completion(q, receiver, draws)
    receiver.person[:] = receiver.person.iloc[::-1].to_numpy()
    actual = values.attach_other_disability_completion(q, receiver, draws)
    for key in expected.columns:
        pd.testing.assert_series_equal(
            expected.columns[key].sort_index(),
            actual.columns[key].sort_index(),
            check_exact=True,
        )


@pytest.mark.parametrize(
    "change", ["clone", "native", "channel", "source", "three_clones"]
)
def test_malformed_clone_identity_refuses(change):
    q, receiver = invented_case()
    if change == "three_clones":
        receiver.person.loc[
            0, values.provenance.support_clone_index_column("person")
        ] = 2
    elif change == "channel":
        receiver.person.loc[0, values.provenance.support_channel_column("person")] = (
            "acs"
        )
    else:
        name = {
            "clone": values.provenance.support_clone_index_column,
            "native": values.provenance.spine_source_id_column,
            "source": values.provenance.support_source_id_column,
        }[change]("person")
        receiver.person.loc[0, name] += 10
    with pytest.raises(ValueError, match="CLONE_"):
        values.attach_other_disability_completion(
            q, receiver, pd.Series(0.0, index=q.recipient_features.index)
        )


def test_existing_float32_requires_exact_values_and_signed_zero():
    q, receiver = invented_case()
    receiver.person[values.observed.OUTPUT] = np.zeros(
        len(receiver.person), dtype="float32"
    )
    exact = values.attach_other_disability_completion(
        q, receiver, pd.Series(-0.0, index=q.recipient_features.index)
    )
    col = exact.columns["person", values.observed.OUTPUT]
    assert col.dtype == np.dtype("float32")
    applicable = exact.columns["person", values.MODEL_APPLICABLE].to_numpy(dtype=bool)
    assert np.signbit(col.to_numpy()[applicable]).all()
    with pytest.raises(ValueError, match="CANONICAL_PRECISION_LOSS"):
        values.attach_other_disability_completion(
            q, receiver, pd.Series(0.1, index=q.recipient_features.index)
        )


def test_unsupported_incumbent_or_report_collision_refuses_before_fit():
    q, receiver = invented_case()
    receiver.person[values.observed.OUTPUT] = np.zeros(
        len(receiver.person), dtype="int64"
    )
    with pytest.raises(ValueError, match="CANONICAL_DTYPE"):
        values.attachment_dtypes(q, receiver)
    del receiver.person[values.observed.OUTPUT]
    receiver.person[values.MODEL_APPLICABLE] = False
    with pytest.raises(ValueError, match="REPORT_ALREADY_OWNED"):
        values.attachment_dtypes(q, receiver)


@pytest.mark.parametrize("draw", [np.nan, np.inf, -1.0])
def test_invalid_draw_is_not_coerced_to_zero(draw):
    q, receiver = invented_case()
    with pytest.raises(ValueError, match="DRAW_AXIS_OR_VALUES"):
        values.attach_other_disability_completion(
            q, receiver, pd.Series(draw, index=q.recipient_features.index)
        )


def test_empty_recipient_path_does_not_require_draws():
    q, receiver = invented_case()
    # This detached fixture changes its declared ACS ages before invoking the
    # pure transport; it does not replace a live source owner or qualification.
    q.originals.loc[q.originals.source.eq("acs"), "age"] = 14
    empty = q.recipient_features.iloc[:0].copy()
    q = replace(q, recipient_features=empty, matrix=None)
    result = values.attach_other_disability_completion(
        q, receiver, pd.Series(index=empty.index, dtype="float64")
    )
    assert not result.columns["person", values.MODEL_APPLICABLE].any()
    assert result.receipt["draws_consumed"] == 0
