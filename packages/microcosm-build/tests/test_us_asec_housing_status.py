"""Publisher-routed source statuses on invented rows; never release evidence."""

import hashlib
import struct
from dataclasses import replace
from importlib import resources

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


def test_unknown_reasons_distinguish_each_contract_gap():
    """The three unknown routes must stay separable, not collapse to one bit."""
    bits = housing._codebook()["unknown_reason_bits"]
    for change, expected in (
        ({"HPUBLIC": 0, "HLORENT": 0}, bits["parent_answer"]),
        ({"HPUBLIC": 2, "HLORENT": 0}, bits["lower_answer"]),
        ({"H_TENURE": 0}, bits["scope"]),
    ):
        result = housing.classify_synthetic_housing_status(observations(**change))
        assert result.array("status").tolist() == [0]
        assert result.array("unknown_reasons").tolist() == [expected]
        assert result.array("receipt_valid").tolist() == [0]


def test_raw_codes_survive_conflict_and_unknown_classification():
    """The publisher's answers are retained whatever interpretation they receive."""
    for change in (
        {"HPUBLIC": 1, "HLORENT": 1},
        {"H_TENURE": 1, "HPUBLIC": 2, "HLORENT": 2},
        {"HPUBLIC": 0, "HLORENT": 0},
        {"H_TENURE": 0, "HPUBLIC": 2, "HLORENT": 1},
    ):
        table = observations(**change)
        result = housing.classify_synthetic_housing_status(table)
        assert result.array("status").tolist() != [1]
        for name in housing.RAW_COLUMNS:
            assert result.array(name).tolist() == table[name].tolist()


def test_authority_binds_both_named_publisher_documents():
    """Dictionary and questionnaire hashes and locators both reach the artifact."""
    document = housing.classify_synthetic_housing_status(observations()).header_data[
        "authority"
    ]
    assert [entry["survey_year"] for entry in document["surveys"]] == [2023, 2024, 2025]
    for entry in document["surveys"]:
        assert entry["income_year"] == entry["survey_year"] - 1
        for role in ("dictionary", "questionnaire"):
            assert len(entry[role]["sha256"]) == 64
            assert entry[role]["url"].startswith("https://www2.census.gov/")
            assert entry[role]["pages"], "every locator needs at least one page"
            for page in entry[role]["pages"]:
                assert type(page["page_index_zero_based"]) is int
                assert len(page["page_text_sha256"]) == 64
    # The 2023/24 facsimile prints 3/4 where the named dictionary encodes 1/2.
    printed = {
        entry["survey_year"]: entry["questionnaire"]["printed_answer_numbers_q85"]
        for entry in document["surveys"]
    }
    assert printed == {2023: "3/4", 2024: "3/4", 2025: "1/2"}
    assert document["facsimile_discrepancy"]["recode_inferred"] is False
    assert document["facsimile_discrepancy"]["cause"] == "unresolved"
    assert document["field_access"]["positional_parser"] == "not_used"


def test_authority_resource_is_pinned_and_tampering_refuses(monkeypatch):
    """The declared pin matches the shipped bytes and is actually enforced."""
    payload = (
        resources.files("microcosm.build.us_runtime")
        .joinpath(housing.AUTHORITY_RESOURCE)
        .read_bytes()
    )
    assert hashlib.sha256(payload).hexdigest() == housing.AUTHORITY_SHA256
    monkeypatch.setattr(housing, "AUTHORITY_SHA256", "0" * 64)
    with pytest.raises(
        housing.HousingStatusRefusalError, match="AUTHORITY_FINGERPRINT"
    ):
        housing.classify_synthetic_housing_status(observations())


def test_authority_dictionary_pins_cannot_drift_from_the_module(monkeypatch):
    """A dictionary pin changed in code without the resource refuses production."""
    year, _, pages = housing.DICTIONARIES[0]
    monkeypatch.setattr(
        housing, "DICTIONARIES", ((year, "0" * 64, pages),) + housing.DICTIONARIES[1:]
    )
    with pytest.raises(
        housing.HousingStatusRefusalError, match="AUTHORITY_DICTIONARY_PINS"
    ):
        housing.classify_synthetic_housing_status(observations())


@pytest.mark.parametrize(
    "public,lower,route,lower_quality",
    [(1, 0, 1, 0), (2, 1, 2, 1)],
)
def test_relevant_public_allocation_marks_quality_without_moving_status(
    public, lower, route, lower_quality
):
    """A relevant I_HPUBLI=1 is quality, not a different answer or a conflict."""
    quality = housing._codebook()["quality"]
    result = housing.classify_synthetic_housing_status(
        observations(HPUBLIC=public, HLORENT=lower, I_HPUBLI=1)
    )
    assert result.array("status").tolist() == [1]
    assert result.array("route").tolist() == [route]
    assert result.array("receipt_valid").tolist() == [1]
    assert result.array("conflicts").tolist() == [0]
    assert result.array("public_quality").tolist() == [quality["allocated"]]
    assert result.array("lower_quality").tolist() == [lower_quality]
    # The allocated answer itself is unchanged; allocation is a separate axis.
    assert result.array("HPUBLIC").tolist() == [public]
    assert result.array("HPUBLIC_zero_origin").tolist() == [0]


def test_each_zero_origin_mask_tracks_only_its_own_field():
    """Contract row 1/0 keeps the routed-out HLORENT zero unresolved."""
    routed_out = housing.classify_synthetic_housing_status(
        observations(HPUBLIC=1, HLORENT=0)
    )
    assert routed_out.array("status").tolist() == [1]
    assert routed_out.array("HPUBLIC_zero_origin").tolist() == [0]
    assert routed_out.array("HLORENT_zero_origin").tolist() == [1]
    assert routed_out.array("lower_quality").tolist() == [0]
    # One row where each of the four fields differs, so no mask can be shared.
    mixed = housing.classify_synthetic_housing_status(
        observations(HPUBLIC=2, HLORENT=1, I_HPUBLI=0, I_HLOREN=1)
    )
    assert [
        mixed.array(name + "_zero_origin").tolist()[0]
        for name in housing.OBSERVED_COLUMNS
    ] == [0, 0, 1, 0]
    assert (
        mixed.header_data["codebook"]["zero_origin"]["frozen_zero_origin_unresolved"]
        == 1
    )


def test_every_conflict_reason_bit_is_reachable_under_its_published_name():
    """All five published conflict bits stay separable, not one convenient bool."""
    bits = housing._codebook()["conflict_bits"]
    for change, expected in (
        ({"H_TENURE": 1}, bits["excluded_scope"]),
        ({"HPUBLIC": 1, "HLORENT": 1}, bits["parent_route"]),
        ({"HPUBLIC": 0, "HLORENT": 0, "I_HPUBLI": 1}, bits["public_allocation"]),
        ({"I_HLOREN": 1, "HPUBLIC": 1, "HLORENT": 0}, bits["lower_allocation"]),
        ({"HRHTYPE": 0}, bits["context"]),
        # Two independent reasons must accumulate rather than overwrite.
        (
            {"H_HHTYPE": 2, "HRHTYPE": 0},
            bits["excluded_scope"] | bits["context"],
        ),
    ):
        result = housing.classify_synthetic_housing_status(observations(**change))
        assert result.array("conflicts").tolist() == [expected], change
        assert result.array("status").tolist() == [4], change
        assert result.array("receipt_valid").tolist() == [0], change
