"""Invented-row contracts for the pure ASEC occupied-housing classifier.

These checks establish row semantics and immutable artifact/readback behavior.
They do not authenticate sources, construct persons, or certify a population.
"""

import hashlib
import json
import os
import signal
import struct
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import asec_housing_universe as hu

RAW_NAMES = (
    "household_id",
    "income_year",
    "H_SEQ",
    "H_HHTYPE",
    "H_LIVQRT",
    "HRHTYPE",
    "H_TENURE",
)
DERIVED_NAMES = (
    "interview_scope",
    "physical_unit",
    "household_kind",
    "tenure_subtype",
    "occupied_hu",
    "hu_tenure_class",
    "unresolved_reasons",
)


def _observations(*contexts):
    """Each explicit context is H_HHTYPE, H_LIVQRT, HRHTYPE, H_TENURE."""
    contexts = contexts or ((1, 1, 1, 1),)
    return pd.DataFrame(
        [(101 + i, 2023, 201 + i, *context) for i, context in enumerate(contexts)],
        columns=RAW_NAMES,
        dtype="int64",
    )


def _assert_derived(value, rows):
    """Compare complete, independently specified states in household order."""
    expected = np.asarray(rows, dtype="uint8")
    assert expected.shape[1] == len(DERIVED_NAMES)
    for name, column in zip(DERIVED_NAMES, expected.T, strict=True):
        actual = value.array(name)
        assert actual.dtype == np.dtype("uint8")
        np.testing.assert_array_equal(actual, column, err_msg=name)
    value.validate()


@pytest.fixture
def three_hu():
    table = _observations((1, 1, 1, 1), (1, 7, 8, 2), (1, 2, 4, 3))
    # Native keys may repeat across income years, and row order is not ID order.
    table["household_id"] = np.array([31, 7, 29], dtype="int64")
    table["income_year"] = np.array([2024, 2022, 2023], dtype="int64")
    table["H_SEQ"] = np.array([11, 11, 11], dtype="int64")
    return table


def test_three_hus_include_one_owner_and_two_renters_with_no_cash_distinct(three_hu):
    value = hu.classify_synthetic_housing_universe(three_hu)
    assert hu.RAW_COLUMNS == RAW_NAMES
    assert hu.DERIVED_COLUMNS == DERIVED_NAMES
    assert type(value) is hu.SyntheticHousingUniverse
    assert value.header_data["source_authentication"] == "synthetic_unverified"
    assert value.header_data["release_eligible"] is False
    assert value.header_data["source"] is None
    for name in RAW_NAMES:
        assert value.array(name).dtype == np.dtype("int64")
        np.testing.assert_array_equal(value.array(name), three_hu[name].to_numpy())
    _assert_derived(
        value,
        [
            (1, 1, 1, 1, 1, 1, 0),
            (1, 1, 1, 2, 1, 2, 0),
            (1, 1, 1, 3, 1, 2, 0),
        ],
    )
    assert np.count_nonzero(value.array("occupied_hu") == 1) == 3
    assert np.count_nonzero(value.array("hu_tenure_class") == 1) == 1
    assert np.count_nonzero(value.array("hu_tenure_class") == 2) == 2


def test_tenure_niu_stays_in_hu_total_as_an_unclassified_residual(three_hu):
    three_hu.loc[1, "H_TENURE"] = 0
    value = hu.classify_synthetic_housing_universe(three_hu)
    _assert_derived(
        value,
        [
            (1, 1, 1, 1, 1, 1, 0),
            (1, 1, 1, 0, 1, 3, 64),
            (1, 1, 1, 3, 1, 2, 0),
        ],
    )
    assert np.count_nonzero(value.array("occupied_hu") == 1) == 3
    assert value.array("H_TENURE").tolist() == [1, 0, 3]


def test_consistent_other_unit_gq_keeps_rows_and_raw_tenure_outside_hu():
    table = _observations((1, 1, 1, 1), (1, 8, 9, 0), (1, 12, 10, 3))
    before = table.copy(deep=True)
    value = hu.classify_synthetic_housing_universe(table)
    _assert_derived(
        value,
        [
            (1, 1, 1, 1, 1, 1, 0),
            (1, 2, 2, 0, 2, 0, 0),
            (1, 2, 2, 3, 2, 0, 0),
        ],
    )
    # The pure API retains row evidence; person retention belongs to composition.
    for name in RAW_NAMES:
        np.testing.assert_array_equal(value.array(name), before[name].to_numpy())
    pd.testing.assert_frame_equal(table, before)


@pytest.mark.parametrize(
    "context,expected",
    [
        ((1, 1, 9, 0), (1, 1, 2, 0, 0, 0, 16)),
        ((1, 12, 8, 3), (1, 2, 1, 3, 0, 0, 128)),
    ],
    ids=["hu-with-gq-kind", "other-unit-with-primary-kind"],
)
def test_disagreement_directions_are_separate_unresolved_states(context, expected):
    value = hu.classify_synthetic_housing_universe(_observations(context))
    _assert_derived(value, [expected])


def test_noninterviews_are_unestablished_and_positive_context_remains_conflicting():
    value = hu.classify_synthetic_housing_universe(
        _observations(
            (2, 1, 0, 0),
            (3, 12, 0, 0),
            (2, 1, 0, 2),  # Positive tenure alone conflicts with noninterview.
            (3, 8, 10, 0),  # Positive household kind alone also conflicts.
            (2, 1, 8, 3),
        )
    )
    _assert_derived(
        value,
        [
            (2, 1, 0, 0, 0, 0, 1),
            (3, 2, 0, 0, 0, 0, 1),
            (2, 1, 0, 2, 0, 0, 33),
            (3, 2, 2, 0, 0, 0, 33),
            (2, 1, 1, 3, 0, 0, 33),
        ],
    )


def test_zero_evidence_has_distinct_reasons_and_preserves_origin_uncertainty():
    value = hu.classify_synthetic_housing_universe(
        _observations(
            (0, 0, 0, 0),
            (0, 1, 1, 3),  # Positive answers do not establish interview scope.
            (1, 0, 1, 2),
            (1, 1, 0, 2),  # Interview + HRHTYPE zero also has context bit 32.
            (1, 0, 0, 0),
        )
    )
    _assert_derived(
        value,
        [
            (0, 0, 0, 0, 0, 0, 2),
            (0, 1, 1, 3, 0, 0, 2),
            (1, 0, 1, 2, 0, 0, 4),
            (1, 1, 0, 2, 0, 0, 40),
            (1, 0, 0, 0, 0, 0, 44),
        ],
    )
    header = value.header_data
    assert header["zero_origin_policy"] == "frozen_fillna_zero_origin_unresolved"
    assert header["zero_origin_evidence"]["original_csv_compared"] is False
    assert header["zero_origin_evidence"]["historical_execution_authenticated"] is False


@pytest.mark.parametrize(
    "field,maximum",
    [("H_HHTYPE", 3), ("H_LIVQRT", 12), ("HRHTYPE", 10), ("H_TENURE", 3)],
)
def test_outside_dictionary_bounds_refuses_instead_of_becoming_unknown(field, maximum):
    for invalid in (-1, maximum + 1):
        table = _observations()
        table.loc[0, field] = invalid
        with pytest.raises(hu.HousingUniverseRefusalError):
            hu.classify_synthetic_housing_universe(table)


@pytest.mark.parametrize(
    "field,dtype,missing",
    [
        ("H_TENURE", "float64", False),
        ("H_LIVQRT", "float64", True),
        ("HRHTYPE", "Int64", False),
        ("H_TENURE", "Int64", True),
        ("H_HHTYPE", "object", False),
        ("household_id", "bool", False),
        ("income_year", "int32", False),
        ("H_SEQ", "uint64", False),
    ],
    ids=[
        "float",
        "nan",
        "nullable",
        "nullable-na",
        "object",
        "bool",
        "int32",
        "uint64",
    ],
)
def test_every_raw_input_requires_ordinary_numpy_int64(field, dtype, missing):
    table = _observations()
    table[field] = table[field].astype(dtype)
    if missing:
        table.loc[0, field] = np.nan if dtype == "float64" else pd.NA
    with pytest.raises(hu.HousingUniverseRefusalError):
        hu.classify_synthetic_housing_universe(table)


@pytest.mark.parametrize(
    "case", ["duplicate-id", "duplicate-native-key", "nonpositive-seq", "income-year"]
)
def test_household_identity_and_native_key_contracts_refuse(case):
    table = _observations((1, 1, 1, 1), (1, 7, 8, 3))
    field, invalid_values = {
        "duplicate-id": ("household_id", [101]),
        "duplicate-native-key": ("H_SEQ", [201]),
        "nonpositive-seq": ("H_SEQ", [0, -1]),
        "income-year": ("income_year", [2021, 2025]),
    }[case]
    for invalid in invalid_values:
        candidate = table.copy(deep=True)
        candidate.loc[1, field] = invalid
        with pytest.raises(hu.HousingUniverseRefusalError):
            hu.classify_synthetic_housing_universe(candidate)


def test_missing_or_duplicate_raw_columns_refuse():
    table = _observations()
    candidates = (
        table.drop(columns="H_LIVQRT"),
        pd.concat([table, table[["H_TENURE"]]], axis=1),
    )
    for candidate in candidates:
        with pytest.raises(hu.HousingUniverseRefusalError):
            hu.classify_synthetic_housing_universe(candidate)


def test_input_values_and_both_axes_are_detached_from_issued_artifact(three_hu):
    three_hu.index = pd.Index([90, 10, 30], dtype="int64")
    before = three_hu.copy(deep=True)
    value = hu.classify_synthetic_housing_universe(three_hu)
    payload, digest = hu.encode_housing_universe(value), value.content_sha256
    three_hu.iloc[0, 6] = 0
    three_hu.index.array[0] = 999
    three_hu.columns.array[0] = "changed_id"
    for name in RAW_NAMES:
        np.testing.assert_array_equal(value.array(name), before[name].to_numpy())
    assert hu.encode_housing_universe(value) == payload
    assert value.content_sha256 == digest
    value.validate()


def test_arrays_buffers_and_nested_header_copies_cannot_change_artifact(three_hu):
    value = hu.classify_synthetic_housing_universe(three_hu)
    payload, digest = hu.encode_housing_universe(value), value.content_sha256
    assert type(value.header) is bytes
    assert type(value.buffers) is tuple
    for name in RAW_NAMES + DERIVED_NAMES:
        array = value.array(name)
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array[0] = 0
        with pytest.raises(ValueError):
            array.setflags(write=True)
    for buffer in value.buffers:
        assert type(buffer) is bytes
        with pytest.raises(TypeError):
            memoryview(buffer)[0] = 0
    with pytest.raises(FrozenInstanceError):
        value.header = b"changed"
    with pytest.raises(FrozenInstanceError):
        value.buffers = ()
    header = value.header_data
    header["release_eligible"] = True
    header["codebook"].clear()
    header["zero_origin_evidence"]["original_csv_compared"] = True
    assert value.header_data["release_eligible"] is False
    assert value.header_data["codebook"]
    assert value.header_data["zero_origin_evidence"]["original_csv_compared"] is False
    assert hu.encode_housing_universe(value) == payload
    assert value.content_sha256 == digest


def test_validate_rechecks_each_derived_buffer_after_deliberate_freeze_bypass():
    for name in DERIVED_NAMES:
        value = hu.classify_synthetic_housing_universe(_observations())
        buffers = list(value.buffers)
        index = (RAW_NAMES + DERIVED_NAMES).index(name)
        buffers[index] = bytes([buffers[index][0] ^ 1])
        # Deliberate hostile reconstruction, not an ordinary mutable API.
        object.__setattr__(value, "buffers", tuple(buffers))
        with pytest.raises(hu.HousingUniverseRefusalError):
            value.validate()
        with pytest.raises(hu.HousingUniverseRefusalError):
            hu.encode_housing_universe(value)


def test_public_constructors_and_dataclass_replace_cannot_issue_authority():
    value = hu.classify_synthetic_housing_universe(_observations())
    for cls in (hu.SyntheticHousingUniverse, hu.AuthenticatedHousingUniverse):
        with pytest.raises(hu.HousingUniverseRefusalError):
            cls(value.header, value.buffers)
    with pytest.raises(hu.HousingUniverseRefusalError):
        replace(value, header=value.header)


def test_canonical_memory_and_file_readback_use_an_independently_issued_expected(
    three_hu, tmp_path
):
    value = hu.classify_synthetic_housing_universe(three_hu)
    expected = hu.classify_synthetic_housing_universe(three_hu.copy(deep=True))
    assert expected is not value
    payload = hu.encode_housing_universe(value)
    path = tmp_path / "invented-universe.bin"
    assert hu.write_housing_universe(value, path) == value.content_sha256
    assert path.read_bytes() == payload
    for replay in (
        hu.decode_housing_universe(payload, expected=expected),
        hu.read_housing_universe(path, expected=expected),
    ):
        assert type(replay) is hu.SyntheticHousingUniverse
        assert replay.header == value.header
        assert replay.buffers == value.buffers
        assert replay.content_sha256 == value.content_sha256
        assert hu.encode_housing_universe(replay) == payload
        replay.validate()


def test_readback_requires_complete_expected_object_not_claims_or_digest(tmp_path):
    value = hu.classify_synthetic_housing_universe(_observations())
    payload = hu.encode_housing_universe(value)
    path = tmp_path / "invented-universe.bin"
    hu.write_housing_universe(value, path)
    for reader, candidate in (
        (hu.decode_housing_universe, payload),
        (hu.read_housing_universe, path),
    ):
        with pytest.raises(TypeError):
            reader(candidate)
        for invalid in (None, value.header_data, value.content_sha256):
            with pytest.raises(hu.HousingUniverseRefusalError):
                reader(candidate, expected=invalid)


@pytest.mark.parametrize("forgery", ["different-valid-answer", "derived", "header"])
def test_rechecksummed_candidate_cannot_replace_expected_evidence(forgery, tmp_path):
    value = hu.classify_synthetic_housing_universe(_observations())
    header, buffers = value.header, list(value.buffers)
    if forgery == "different-valid-answer":
        other = hu.classify_synthetic_housing_universe(_observations((1, 1, 1, 3)))
        header, buffers = other.header, list(other.buffers)
    elif forgery == "derived":
        # A legal enum code that contradicts the unchanged owner observation.
        buffers[(RAW_NAMES + DERIVED_NAMES).index("hu_tenure_class")] = b"\x02"
    else:
        claims = value.header_data
        claims["source_authentication"] = "checkpoint_bytes_verified"
        header = json.dumps(claims, sort_keys=True, separators=(",", ":")).encode()
    body = hu.MAGIC + struct.pack("<I", len(header)) + header + b"".join(buffers)
    candidate = body + hashlib.sha256(body).digest()
    assert candidate != hu.encode_housing_universe(value)
    assert hashlib.sha256(candidate[:-32]).digest() == candidate[-32:]
    if forgery == "different-valid-answer":
        # Even an internally consistent, canonically issued alternative is stale.
        assert candidate == hu.encode_housing_universe(other)
    path = tmp_path / "invented-forgery.bin"
    path.write_bytes(candidate)
    with pytest.raises(hu.HousingUniverseRefusalError):
        hu.decode_housing_universe(candidate, expected=value)
    with pytest.raises(hu.HousingUniverseRefusalError):
        hu.read_housing_universe(path, expected=value)


def test_truncation_trailing_bytes_and_transport_corruption_refuse(tmp_path):
    value = hu.classify_synthetic_housing_universe(_observations())
    payload = hu.encode_housing_universe(value)
    corrupt = bytearray(payload)
    corrupt[-33] ^= 1
    path = tmp_path / "invented-damaged-universe.bin"
    for candidate in (payload[:-1], payload + b"\x00", bytes(corrupt)):
        path.write_bytes(candidate)
        with pytest.raises(hu.HousingUniverseRefusalError):
            hu.decode_housing_universe(candidate, expected=value)
        with pytest.raises(hu.HousingUniverseRefusalError):
            hu.read_housing_universe(path, expected=value)


@pytest.mark.skipif(
    not hasattr(os, "mkfifo") or not hasattr(signal, "setitimer"),
    reason="FIFO refusal uses a bounded POSIX interval timer",
)
def test_fifo_candidate_is_refused_without_waiting_for_a_writer(tmp_path):
    value = hu.classify_synthetic_housing_universe(_observations())
    path = tmp_path / "invented-universe.fifo"
    os.mkfifo(path)

    def timeout(_signal, _frame):
        raise AssertionError("Candidate FIFO read waited for a writer")

    previous = signal.signal(signal.SIGALRM, timeout)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 3)
    try:
        with pytest.raises(hu.HousingUniverseRefusalError, match="ARTIFACT_FILE"):
            hu.read_housing_universe(path, expected=value)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
