"""Pure observed-age identity rules over invented numeric Series only."""

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import survey_observed_age as rule


@pytest.mark.parametrize("raw_dtype", ["int64", "Int64", "uint64"])
@pytest.mark.parametrize("incumbent", ["absent", "missing", "matching", "partial"])
def test_observed_identity_preserves_top_codes_and_inputs(raw_dtype, incumbent):
    index = pd.Index([40, 30, 20, 10, 0], name="native_row")
    raw = pd.Series([0, 14, 55, 80, 85], index=index, dtype=raw_dtype)
    common = {
        "absent": None,
        "missing": pd.Series([pd.NA] * 5, index=index, dtype=object),
        "matching": pd.Series([0, 14, 55, 80, 85], index=index, dtype="Int64"),
        "partial": pd.Series([pd.NA, 14, pd.NA, 80, 85], index=index, dtype="Float64"),
    }[incumbent]
    raw_before = raw.copy(deep=True)
    common_before = None if common is None else common.copy(deep=True)
    result = rule.normalize_observed_age(raw, common)
    assert result.dtype == np.dtype("float64")
    assert result.tolist() == [0.0, 14.0, 55.0, 80.0, 85.0]
    assert result.index.identical(index) and result.name == "age"
    pd.testing.assert_series_equal(raw, raw_before)
    if common is not None:
        pd.testing.assert_series_equal(common, common_before)
    result.iloc[0] = 100
    result.index.name = "owned_output"
    pd.testing.assert_series_equal(raw, raw_before)
    if common is not None:
        pd.testing.assert_series_equal(common, common_before)


@pytest.mark.parametrize(
    "raw, reason",
    [
        (pd.Series([pd.NA], dtype="Int64"), "RAW_INTEGER_REQUIRED"),
        (pd.Series([True]), "RAW_INTEGER_REQUIRED"),
        (pd.Series([55.0]), "RAW_INTEGER_REQUIRED"),
        (pd.Series(["55"]), "RAW_INTEGER_REQUIRED"),
        (pd.Series([-1]), "RAW_REPRESENTATION"),
        (pd.Series([2**53 + 1]), "RAW_REPRESENTATION"),
        (pd.Series([2**64 - 1], dtype="uint64"), "RAW_REPRESENTATION"),
    ],
)
def test_missing_or_nonliteral_raw_age_refuses(raw, reason):
    with pytest.raises(ValueError, match=reason):
        rule.normalize_observed_age(raw)


@pytest.mark.parametrize(
    "common, reason",
    [
        (pd.Series([56]), "INCUMBENT_CONFLICT"),
        (pd.Series([55.5]), "INCUMBENT_CONFLICT"),
        (pd.Series([True]), "INCUMBENT_NUMERIC_REQUIRED"),
        (pd.Series(["55"]), "INCUMBENT_NUMERIC_REQUIRED"),
        (pd.Series([55 + 0j]), "INCUMBENT_NUMERIC_REQUIRED"),
        (pd.Series([float("inf")]), "INCUMBENT_REPRESENTATION"),
        (pd.Series([-1]), "INCUMBENT_REPRESENTATION"),
    ],
)
def test_nonmissing_common_age_must_match_without_coercion(common, reason):
    with pytest.raises(ValueError, match=reason):
        rule.normalize_observed_age(pd.Series([55]), common)


@pytest.mark.parametrize("incumbent", [2**53 + 1, 2**64 - 1])
def test_integer_incumbent_cannot_round_into_equality(incumbent):
    raw = pd.Series([2**53], dtype="uint64")
    common = pd.Series([incumbent], dtype="uint64")
    with pytest.raises(ValueError, match="INCUMBENT_REPRESENTATION"):
        rule.normalize_observed_age(raw, common)


def test_ordered_index_and_exact_representation_boundary():
    raw = pd.Series([0, 2**53], index=pd.Index([2, 1], name="row"))
    common = raw.astype(float)
    with pytest.raises(ValueError, match="ORDERED_INDEX"):
        rule.normalize_observed_age(raw, common.iloc[::-1])
    common.index = common.index.rename("different")
    with pytest.raises(ValueError, match="ORDERED_INDEX"):
        rule.normalize_observed_age(raw, common)
    result = rule.normalize_observed_age(raw)
    assert result.tolist() == [0.0, float(2**53)]


def test_nonseries_and_bound_refuse_before_numeric_allocation(monkeypatch):
    with pytest.raises(ValueError, match="RAW_SERIES"):
        rule.normalize_observed_age([55])
    with pytest.raises(ValueError, match="INCUMBENT_SERIES"):
        rule.normalize_observed_age(pd.Series([55]), [55])
    monkeypatch.setattr(rule, "MAX_ROWS", 2)
    raw = pd.Series([1, 2, 3])
    with monkeypatch.context() as local:
        local.setattr(
            pd.Series,
            "to_numpy",
            lambda *a, **kw: pytest.fail("Allocation occurred before row bound"),
        )
        with pytest.raises(ValueError, match="ROW_BOUND"):
            rule.normalize_observed_age(raw)
        with pytest.raises(ValueError, match="ROW_BOUND"):
            rule.normalize_observed_age(pd.Series([], dtype="int64"))


def test_known_negative_zero_and_rule_document_are_preserved():
    raw = pd.Series([0, 55])
    common = pd.Series([-0.0, 55.0])
    result = rule.normalize_observed_age(raw, common)
    assert result.to_numpy().tobytes() == common.to_numpy().tobytes()
    document = rule.rule_document()
    assert document["relation"] == "numeric_identity"
    assert document["temporal_adjustment"] is False
    assert document["top_code_replacement"] is False
    document["relation"] = "changed"
    assert rule.rule_document()["relation"] == "numeric_identity"


def test_row_ceiling_admits_a_full_source_channel():
    """A full-source ACS channel really is normalized, not only asserted.

    `_normalized_source_copy` normalizes the ACS and ASEC native frames
    separately, so the ceiling is met one channel at a time and the larger
    channel is ACS: 3,422,888 persons at full source. That is one int64 column,
    about 27 MiB, so the acceptance is driven for real rather than inferred.
    """
    rows = 3_422_888
    assert rule.MAX_ROWS >= rows
    raw = pd.Series(np.arange(rows, dtype="int64") % 101, name="A_AGE")
    result = rule.normalize_observed_age(raw)
    assert len(result) == rows
    assert result.dtype == np.dtype("float64")
    assert result.iloc[0] == 0.0 and result.iloc[100] == 100.0
    assert result.name == "age"


def test_row_ceiling_refuses_one_row_past_its_own_number(monkeypatch):
    """The refusal is `<= MAX_ROWS`, whatever that number is."""
    monkeypatch.setattr(rule, "MAX_ROWS", 3)
    at_ceiling = pd.Series([1, 2, 3], dtype="int64")
    assert rule.normalize_observed_age(at_ceiling).tolist() == [1.0, 2.0, 3.0]
    with pytest.raises(ValueError, match="ROW_BOUND"):
        rule.normalize_observed_age(pd.Series([1, 2, 3, 4], dtype="int64"))
