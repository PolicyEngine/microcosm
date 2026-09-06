"""Publisher-routed source statuses on invented rows; never release evidence."""

import hashlib
import struct
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import asec_housing_status as housing


def observations(**changes):
    values = {
        "household_id": [42],
        "income_year": [2023],
        "survey_year": [2024],
        "H_SEQ": [7],
        "H_TENURE": [2],
        "HRHTYPE": [1],
        "H_HHTYPE": [1],
        "HPUBLIC": [2],
        "HLORENT": [2],
        "I_HPUBLI": [0],
        "I_HLOREN": [0],
    }
    values.update({k: [v] for k, v in changes.items()})
    return pd.DataFrame(values, dtype=np.int64)


@pytest.mark.parametrize(
    "public,lower,status,route,valid",
    [
        (1, 0, 1, 1, 1),
        (2, 1, 1, 2, 1),
        (2, 2, 2, 3, 1),
        (0, 0, 0, 0, 0),
        (2, 0, 0, 0, 0),
        (1, 1, 4, 0, 0),
        (1, 2, 4, 0, 0),
        (0, 1, 4, 0, 0),
        (0, 2, 4, 0, 0),
    ],
)
def test_all_publisher_answer_routes(public, lower, status, route, valid):
    # All three dictionaries, printed 6A-9: parent HPUBLIC; HLORENT iff parent=2.
    result = housing.classify_synthetic_housing_status(
        observations(HPUBLIC=public, HLORENT=lower)
    )
    assert result.array("status").tolist() == [status]
    assert result.array("route").tolist() == [route]
    assert result.array("receipt_valid").tolist() == [valid]
    assert result.header_data["source_authentication"] == "synthetic_unverified"
    assert result.header_data["release_eligible"] is False


@pytest.mark.parametrize("gq", [9, 10])
def test_group_quarters_is_scope_flag_not_added_receipt_exclusion(gq):
    result = housing.classify_synthetic_housing_status(
        observations(HRHTYPE=gq, HPUBLIC=1, HLORENT=0)
    )
    assert result.array("status").tolist() == [1]
    assert result.array("group_quarters").tolist() == [1]
    assert result.header_data["non_housing_unit_scope"] == "not_derived"


@pytest.mark.parametrize(
    "change,expected",
    [
        ({"H_TENURE": 1, "HPUBLIC": 0, "HLORENT": 0}, 3),
        ({"H_TENURE": 1}, 4),
        ({"H_TENURE": 0}, 0),
        ({"H_HHTYPE": 2, "HRHTYPE": 0, "H_TENURE": 0, "HPUBLIC": 0, "HLORENT": 0}, 3),
        ({"H_HHTYPE": 2, "HRHTYPE": 0}, 4),
        ({"H_HHTYPE": 1, "HRHTYPE": 0}, 4),
    ],
)
def test_universe_and_context_are_not_false_nonreceipt(change, expected):
    result = housing.classify_synthetic_housing_status(observations(**change))
    assert result.array("status").tolist() == [expected]
    assert result.array("receipt_valid").tolist() == [0]


@pytest.mark.parametrize(
    "name,bad",
    [
        ("HPUBLIC", -1),
        ("HPUBLIC", 3),
        ("HLORENT", 3),
        ("I_HPUBLI", 2),
        ("I_HLOREN", -1),
    ],
)
def test_domain_violation_refuses_before_any_status(name, bad):
    with pytest.raises(housing.HousingStatusRefusalError, match="CODE_DOMAIN"):
        housing.classify_synthetic_housing_status(observations(**{name: bad}))


def test_allocation_quality_and_zero_origin_remain_separate():
    value = housing.classify_synthetic_housing_status(observations(I_HLOREN=1))
    assert value.array("status").tolist() == [2]
    assert value.array("lower_quality").tolist() == [2]
    assert value.array("public_quality").tolist() == [1]
    assert value.array("I_HPUBLI_zero_origin").tolist() == [1]
    assert (
        value.header_data["zero_origin_policy"]
        == "frozen_fillna_zero_origin_unresolved"
    )
    evidence = value.header_data["zero_origin_evidence"]
    assert (
        evidence["inspected_sha256"]
        == "ce3dac429aad665df25c503fdcb850a4e31d89268dd73acec8343c70c7cfb647"
    )
    assert evidence["historical_execution_authenticated"] is False
    assert evidence["original_csv_compared"] is False
    conflict = housing.classify_synthetic_housing_status(
        observations(HPUBLIC=1, HLORENT=0, I_HLOREN=1)
    )
    assert conflict.array("status").tolist() == [4]
    assert conflict.array("conflicts").tolist() == [8]
    assert conflict.array("lower_quality").tolist() == [3]


def test_input_values_and_both_axes_cannot_mutate_issued_bytes():
    table = observations()
    table.index = pd.Index([15])
    result = housing.classify_synthetic_housing_status(table)
    payload = housing.encode_housing_status(result)
    table.index.array[0] = 99
    table.columns.array[0] = "changed"
    table.iloc[0, 7] = 0
    assert housing.encode_housing_status(result) == payload
    with pytest.raises(ValueError):
        result.array("status").setflags(write=True)
    result.header_data["release_eligible"] = True
    assert result.header_data["release_eligible"] is False


def test_codec_roundtrip_requires_expected_complete_bytes_not_a_checksum():
    value = housing.classify_synthetic_housing_status(observations())
    payload = housing.encode_housing_status(value)
    replay = housing.decode_housing_status(payload, expected=value)
    assert type(replay) is housing.SyntheticHousingStatus
    assert housing.encode_housing_status(replay) == payload
    # Change a valid raw answer and rechecksum under exactly the original header.
    header_size = struct.unpack_from("<I", payload, len(housing.MAGIC))[0]
    altered = bytearray(payload)
    offset = (
        len(housing.MAGIC) + 4 + header_size + 8 * housing.RAW_COLUMNS.index("HPUBLIC")
    )
    struct.pack_into("<q", altered, offset, 1)
    altered[-32:] = hashlib.sha256(altered[:-32]).digest()
    with pytest.raises(housing.HousingStatusRefusalError, match="EXPECTED_CONTENT"):
        housing.decode_housing_status(bytes(altered), expected=value)
    with pytest.raises(housing.HousingStatusRefusalError, match="EXPECTED_STATUS"):
        housing.decode_housing_status(payload, expected=value.header_data)


@pytest.mark.parametrize("dtype", ["float64", "Int64", "bool", "object"])
def test_source_status_refuses_lossy_or_nullable_raw_dtypes(dtype):
    table = observations()
    table["HPUBLIC"] = table.HPUBLIC.astype(dtype)
    with pytest.raises(housing.HousingStatusRefusalError, match="INT64"):
        housing.classify_synthetic_housing_status(table)


def test_survey_year_is_observed_income_year_plus_one():
    with pytest.raises(housing.HousingStatusRefusalError, match="SURVEY_YEAR"):
        housing.classify_synthetic_housing_status(observations(survey_year=2023))


def test_allocated_missing_parent_is_conflict_with_explicit_invalid_receipt():
    result = housing.classify_synthetic_housing_status(
        observations(HPUBLIC=0, HLORENT=0, I_HPUBLI=1)
    )
    assert result.array("status").tolist() == [4]
    assert result.array("public_quality").tolist() == [3]
    assert result.array("conflicts").tolist() == [4]
    assert result.array("receipt_valid").tolist() == [0]
    assert all(
        result.array(name).dtype == np.dtype("uint8")
        for name in housing.DERIVED_COLUMNS
    )


@pytest.mark.parametrize(
    "field,value", [("H_TENURE", 4), ("HRHTYPE", 11), ("H_HHTYPE", -1)]
)
def test_unsupported_context_cannot_become_an_unknown_status(field, value):
    with pytest.raises(housing.HousingStatusRefusalError, match="CONTEXT_DOMAIN"):
        housing.classify_synthetic_housing_status(observations(**{field: value}))


def test_public_constructors_and_dataclass_replace_do_not_issue_authority():
    value = housing.classify_synthetic_housing_status(observations())
    for cls in (housing.SyntheticHousingStatus, housing.AuthenticatedHousingStatus):
        with pytest.raises(housing.HousingStatusRefusalError, match="CONSTRUCTOR"):
            cls(value.header, value.buffers)
    with pytest.raises(housing.HousingStatusRefusalError, match="CONSTRUCTOR"):
        replace(value, header=value.header)


def test_file_roundtrip_and_transport_corruption_refusal(tmp_path):
    value = housing.classify_synthetic_housing_status(observations())
    path = tmp_path / "invented-housing.bin"
    assert housing.write_housing_status(value, path) == value.content_sha256
    assert (
        housing.read_housing_status(path, expected=value).content_sha256
        == value.content_sha256
    )
    changed = bytearray(path.read_bytes())
    changed[-33] ^= 1
    path.write_bytes(changed)
    with pytest.raises(housing.HousingStatusRefusalError, match="PAYLOAD_CHECKSUM"):
        housing.read_housing_status(path, expected=value)
