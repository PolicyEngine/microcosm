"""Invented numeric lineage controls; no H5, survey files or country imports."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

TOOL = Path(__file__).resolve().parents[3] / "tools/acs_hours_donor_preparation.py"
spec = importlib.util.spec_from_file_location("hours_preparation", TOOL)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


@pytest.fixture
def inputs():
    # Three baseline age-15 cohorts, one PUF clone, then ACS ages 14, 15, 16.
    integer = (
        "person_id",
        "person_household_id",
        "person_source_id",
        "source_year",
        "source_household_id",
        "source_row_id",
        "person_support_clone_index",
        "A_AGE",
    )
    floating = ("HRSWK", "WKSWORK", "A_HRS1", "SPORDER", "AGEP", "WAGP", "SEMP")
    person = np.zeros(
        7, dtype=[(x, "<i8") for x in integer] + [(x, "<f8") for x in floating]
    )
    person["person_id"] = [10, 20, 30, 110, 210, 211, 220]
    person["person_source_id"] = [10, 20, 30, 10, 0, 1, 2]
    person["person_household_id"] = [1, 2, 3, 11, 21, 21, 22]
    person["source_year"] = [2022, 2023, 2024, 2022, 2024, 2024, 2024]
    person["source_household_id"] = [1, 1, 1, 1, 1, 1, 2]
    person["source_row_id"] = [0, 0, 0, 0, 0, 1, 2]
    person["person_support_clone_index"] = [0, 0, 0, 1, 0, 0, 0]
    person["A_AGE"] = [15, 15, 15, 15, 14, 15, 16]
    person["HRSWK"] = [0, 12, 20, 0, np.nan, np.nan, np.nan]
    person["WKSWORK"] = [0, 8, 20, 0, np.nan, np.nan, np.nan]
    person["A_HRS1"] = [-1, 0, 40, -1, np.nan, np.nan, np.nan]
    person["SPORDER"] = [np.nan, np.nan, np.nan, np.nan, 1, 2, 1]
    person["AGEP"] = [np.nan, np.nan, np.nan, np.nan, 14, 15, 16]
    person["WAGP"] = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 100]
    person["SEMP"] = [np.nan, np.nan, np.nan, np.nan, np.nan, -50, 0]
    household = np.array(
        [(22, 0), (3, 0), (21, 0), (1, 0), (11, 1), (2, 0)],
        dtype=[("household_id", "<i8"), ("household_support_clone_index", "<i8")],
    )
    acs = np.zeros(
        3,
        dtype=[
            (x, "<i8")
            for x in (
                "parent_person_row",
                "person_id",
                "person_household_id",
                "source_year",
                "source_household_id",
                "source_row_id",
                "SPORDER",
            )
        ]
        + [(x, "<f8") for x in ("WKHP", "WKL", "FWKHP")],
    )
    acs["parent_person_row"] = [4, 5, 6]
    for name in set(acs.dtype.names) & set(person.dtype.names):
        acs[name] = person[name][4:]
    acs["WKHP"] = [np.nan, np.nan, 20]
    acs["WKL"] = [np.nan, np.nan, 1]
    acs["FWKHP"] = [np.nan, np.nan, 0]
    receipt = {
        "protocol": "acs-numeric-source-recovery-v1",
        "mode": "full",
        "parent_sha256": "a" * 64,
        "full_source_bijection": True,
        "selected_people": 3,
        "selected_households": 2,
        "parent_rows_checked_for_unique_current_ids": {"person": 7, "household": 6},
        "linkage_basis": "reviewed_sort_rank_reconstruction",
        "original_staging_revision": None,
        "puma_compared": False,
    }
    return person, household, acs, receipt


def prepare(inputs, policy="pooled_2022_2024"):
    return helper.prepare(
        *inputs[:3], recovery_receipt=inputs[3], donor_year_policy=policy
    )


def test_explicit_year_selection_excludes_acs_and_puf_and_preserves_raw(inputs):
    before = [a.tobytes() for a in inputs[:3]]
    donors, recipients, receipt = prepare(inputs)
    assert donors["person_id"].tolist() == [10, 20, 30]
    assert recipients["person_id"].tolist() == [211]
    assert recipients["parent_person_row"].tolist() == [5]
    assert np.isnan(recipients["WAGP"][0])
    assert recipients["SEMP"].tolist() == [-50]
    assert np.isnan(recipients["WKHP"][0])
    assert donors["A_HRS1"].tolist() == [-1, 0, 40]
    assert receipt["donors_by_source_year"] == {"2022": 1, "2023": 1, "2024": 1}
    assert receipt["fresh_raw_asec_join"] is False
    assert [a.tobytes() for a in inputs[:3]] == before
    only_2024, _, _ = prepare(inputs, "source_2024_only")
    assert only_2024["person_id"].tolist() == [30]
    with pytest.raises(ValueError, match="policy"):
        prepare(inputs, "automatic")


@pytest.mark.parametrize(
    "field,value",
    [
        ("person_id", 20),
        ("person_household_id", 999),
        ("person_support_clone_index", 1),
        ("source_year", 2021),
        ("HRSWK", 100),
        ("WKSWORK", 53),
        ("A_HRS1", -2),
        ("A_AGE", 100),
        ("A_AGE", 84),
        ("HRSWK", np.nan),
        ("HRSWK", 1.5),
    ],
)
def test_refuses_invalid_donor_identity_or_source_domain(inputs, field, value):
    inputs[0][field][0] = value
    with pytest.raises(ValueError):
        prepare(inputs)


def test_cohort_row_ids_may_repeat_across_years_but_not_within_clone(inputs):
    prepare(inputs)
    inputs[0]["source_year"][1] = 2022
    inputs[0]["source_household_id"][1] = 2
    with pytest.raises(ValueError, match="source-year/row/clone"):
        prepare(inputs)


def test_annual_zero_coherence_does_not_equate_reference_week_hours(inputs):
    prepare(inputs)  # Annual 20 and reference-week 40 are deliberately different.
    inputs[0]["WKSWORK"][0] = 1
    with pytest.raises(ValueError, match="annual"):
        prepare(inputs)


def test_harmonized_age_cannot_select_non_age15_source_donors(inputs):
    inputs[0]["A_AGE"][0] = 14
    donors, _, receipt = prepare(inputs)
    assert donors["person_id"].tolist() == [20, 30]
    assert receipt["baseline_asec_age_min"] == 14


def test_acs_observed_and_niu_fields_are_copied_without_target_assignment(inputs):
    inputs[2]["WKHP"][1] = 8
    inputs[2]["WKL"][1] = 1
    inputs[2]["FWKHP"][1] = 1
    _, recipients, _ = prepare(inputs)
    assert recipients["WKHP"].tolist() == [8]
    assert recipients["FWKHP"].tolist() == [1]
    assert np.isnan(recipients["WAGP"][0])


@pytest.mark.parametrize(
    "change", ["pilot", "missing_acs", "swapped_identity", "age", "mixed_household"]
)
def test_refuses_incomplete_or_misaligned_acs_evidence(inputs, change):
    person, household, acs, receipt = inputs
    if change == "pilot":
        receipt["full_source_bijection"] = False
    elif change == "missing_acs":
        acs = acs[1:]
        receipt["selected_people"] = 2
    elif change == "swapped_identity":
        acs["person_id"][:2] = acs["person_id"][:2][::-1]
    elif change == "age":
        person["AGEP"][5] = 16
    else:
        person["person_household_id"][0] = 21
    with pytest.raises(ValueError):
        prepare((person, household, acs, receipt))


def test_refuses_duplicate_households_and_object_values(inputs):
    person, household, acs, receipt = inputs
    duplicate = household.copy()
    duplicate["household_id"][0] = duplicate["household_id"][1]
    with pytest.raises(ValueError, match="household_id"):
        prepare((person, duplicate, acs, receipt))
    with pytest.raises(ValueError, match="numeric"):
        prepare(
            (
                person.astype([(n, object) for n in person.dtype.names]),
                household,
                acs,
                receipt,
            )
        )


def write_inputs(tmp_path, inputs):
    person, household, acs, receipt = inputs
    for name, array in (("person", person), ("household", household), ("acs", acs)):
        np.save(tmp_path / f"{name}.npy", array, allow_pickle=False)
    receipt["sidecar_sha256"] = helper.sha256_file(tmp_path / "acs.npy")
    receipt["sidecar_bytes"] = (tmp_path / "acs.npy").stat().st_size
    (tmp_path / "recovery.json").write_text(json.dumps(receipt))
    projection = {
        "parent_sha256": receipt["parent_sha256"],
        "person_sha256": helper.sha256_file(tmp_path / "person.npy"),
        "household_sha256": helper.sha256_file(tmp_path / "household.npy"),
    }
    (tmp_path / "projection.json").write_text(json.dumps(projection))
    return dict(
        person_path=tmp_path / "person.npy",
        household_path=tmp_path / "household.npy",
        acs_path=tmp_path / "acs.npy",
        recovery_path=tmp_path / "recovery.json",
        projection_path=tmp_path / "projection.json",
        expected_recovery_sha256=helper.sha256_file(tmp_path / "recovery.json"),
        expected_projection_sha256=helper.sha256_file(tmp_path / "projection.json"),
        donor_year_policy="pooled_2022_2024",
        output_dir=tmp_path / "out",
    )


def test_files_bind_inputs_outputs_and_refuse_overwrite(tmp_path, inputs):
    args = write_inputs(tmp_path, inputs)
    receipt = helper.prepare_files(**args)
    assert set(p.name for p in args["output_dir"].iterdir()) == {
        "asec-age15-donors.npy",
        "acs-age15-recipients.npy",
        "PREPARATION.json",
    }
    for name, pin in receipt["outputs"].items():
        output = args["output_dir"] / name
        assert helper.sha256_file(output) == pin["sha256"]
        np.load(output, allow_pickle=False)
    with pytest.raises(ValueError, match="exists"):
        helper.prepare_files(**args)


@pytest.mark.parametrize("change", ["parent", "sidecar", "projection_pin"])
def test_file_authentication_refuses_without_output(tmp_path, inputs, change):
    args = write_inputs(tmp_path, inputs)
    if change == "sidecar":
        with args["acs_path"].open("ab") as stream:
            stream.write(b"changed")
    elif change == "projection_pin":
        args["expected_projection_sha256"] = "b" * 64
    else:
        path = args["projection_path"]
        projection = json.loads(path.read_text())
        projection["parent_sha256"] = "b" * 64
        path.write_text(json.dumps(projection))
        args["expected_projection_sha256"] = helper.sha256_file(path)
    with pytest.raises(ValueError):
        helper.prepare_files(**args)
    assert not args["output_dir"].exists()
