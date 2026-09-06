"""Invented-input contract tests for the household reported-income measurement.

Every test runs the real production kernels. All source values are invented,
tiny and explicit; no genuine Census microdata is read.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
import pytest

from microcosm.build.quality_aggregates import (
    DistributionBin,
    GeographySpec,
    population_distributions,
)
from microcosm.build.us_runtime import acs_pums
from microcosm.build.us_runtime import household_reported_income as reported
from microcosm.build.us_runtime.acs_household_source_columns import (
    ACS_HOUSEHOLD_SOURCE_COLUMNS_V1,
    ACS_HOUSEHOLD_SOURCE_COLUMNS_V2,
    acs_household_source_column_contract,
    read_acs_household_income_columns,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

DECLARED_ADJINC = reported.ACS_2024_DECLARED_ADJINC


def _acs(keys, hincp, kinds, adjinc=None):
    rows = len(keys)
    return reported.measure_acs_household_reported_income(
        household_keys=keys,
        hincp=pd.array(hincp, dtype="Int64"),
        adjinc=pd.array(
            [DECLARED_ADJINC] * rows if adjinc is None else adjinc, dtype="Int64"
        ),
        housing_unit_kind=pd.array(kinds, dtype="Int64"),
    )


def _asec(keys, htotval, years):
    return reported.measure_asec_household_reported_income(
        household_keys=keys,
        htotval=pd.array(htotval, dtype="Int64"),
        income_year=pd.array(years, dtype="Int64"),
    )


def test_acs_applies_the_declared_adjustment_factor_exactly_once():
    measured = _acs(("2024HU0000001",), [100_000], [1])
    assert measured.values.tolist() == [100_000 * DECLARED_ADJINC / 1_000_000]
    assert measured.evidence["adjustment"]["applications"] == 1
    assert measured.evidence["adjustment"]["declared_adjinc"] == DECLARED_ADJINC
    assert measured.target_price_period == "calendar_2024_dollars_census_adjinc"
    assert measured.source_price_period == (
        "rolling_twelve_months_ending_in_interview_month"
    )
    assert measured.origin.observation_period == "rolling_past_twelve_months"


def test_acs_refuses_a_second_price_adjustment():
    """An already-adjusted amount and factor are both refused, separately."""

    already_adjusted = int(round(100_000 * DECLARED_ADJINC / 1_000_000))
    # The factor is the guard: a caller that pre-divided passes 1, not 1015250.
    with pytest.raises(
        reported.ReportedIncomeRefusalError,
        match="ADJINC_NOT_THE_DECLARED_2024_FACTOR",
    ):
        _acs(("2024HU0000001",), [already_adjusted], [1], adjinc=[1])
    # Re-applying the declared factor to an adjusted value is arithmetically
    # possible, so the returned artifact must state that it adjusted once and
    # feeding its own float output back in must fail the integer dtype gate.
    once = _acs(("2024HU0000001",), [100_000], [1])
    with pytest.raises(reported.ReportedIncomeRefusalError, match="HINCP_DTYPE"):
        reported.measure_acs_household_reported_income(
            household_keys=("2024HU0000001",),
            hincp=pd.Series(once.values),
            adjinc=pd.array([DECLARED_ADJINC], dtype="Int64"),
            housing_unit_kind=pd.array([1], dtype="Int64"),
        )


def test_acs_preserves_negative_net_income_and_its_declared_floor():
    measured = _acs(
        ("2024HU0000001", "2024HU0000002"),
        [-59_998, -59_999],
        [1, 1],
    )
    assert (measured.values < 0).all()
    assert measured.quality.tolist() == [
        reported.QUALITY["exact_publisher_total"],
        reported.QUALITY["declared_open_ended_bound"],
    ]
    assert measured.evidence["negative_value_rows"] == 2
    assert measured.raw_values.tolist() == [-59_998, -59_999]


def test_acs_refuses_a_loss_below_the_published_bottom_code():
    with pytest.raises(reported.ReportedIncomeRefusalError, match="HINCP_RANGE"):
        _acs(("2024HU0000001",), [-60_000], [1])


def test_acs_group_quarters_rows_are_out_of_universe_not_zero():
    measured = _acs(
        ("2024HU0000001", "2024GQ0000001", "2024GQ0000002"),
        [12_345, None, None],
        [1, 2, 3],
    )
    assert measured.availability.tolist() == [
        reported.AVAILABILITY["observed"],
        reported.AVAILABILITY["unavailable_out_of_universe"],
        reported.AVAILABILITY["unavailable_out_of_universe"],
    ]
    assert np.isnan(measured.values[1:]).all()
    assert measured.evidence["availability_counts"]["unavailable_out_of_universe"] == 2


def test_acs_refuses_an_amount_on_a_non_household_universe_row():
    with pytest.raises(
        reported.ReportedIncomeRefusalError, match="OUT_OF_UNIVERSE_AMOUNT"
    ):
        _acs(("2024GQ0000001",), [1], [2])


def test_acs_unavailable_in_universe_income_is_distinguished_from_out_of_universe():
    measured = _acs(
        ("2024HU0000001", "2024GQ0000001"),
        [None, None],
        [1, 2],
    )
    assert measured.availability.tolist() == [
        reported.AVAILABILITY["unavailable_in_universe"],
        reported.AVAILABILITY["unavailable_out_of_universe"],
    ]
    assert measured.quality.tolist() == [reported.QUALITY["not_applicable"]] * 2
    assert np.isnan(measured.values).all()


@pytest.mark.parametrize("arm", ["acs", "asec"])
def test_an_empty_household_sample_refuses(arm):
    with pytest.raises(reported.ReportedIncomeRefusalError, match="EMPTY_SAMPLE"):
        if arm == "acs":
            _acs((), [], [])
        else:
            _asec((), [], [])


def test_native_leading_zero_identifiers_join_by_exact_spelling():
    keys = ("0012345", "12345")
    measured = _acs(keys, [10, 20], [1, 1])
    aligned = reported.align_household_values(measured, ("12345", "0012345"))
    assert aligned.tolist() == [
        measured.values[1],
        measured.values[0],
    ]
    with pytest.raises(
        reported.ReportedIncomeRefusalError, match="UNKNOWN_HOUSEHOLD_KEY"
    ):
        reported.align_household_values(measured, ("012345",))


def test_integer_household_keys_refuse_before_leading_zeros_can_be_lost():
    with pytest.raises(
        reported.ReportedIncomeRefusalError, match="HOUSEHOLD_KEY_NOT_STRING"
    ):
        _acs((12345,), [10], [1])


def test_duplicate_household_keys_refuse():
    with pytest.raises(
        reported.ReportedIncomeRefusalError, match="DUPLICATE_HOUSEHOLD_KEY"
    ):
        _acs(("2024HU0000001", "2024HU0000001"), [1, 2], [1, 1])


def test_asec_identity_year_copies_the_amount_and_earlier_years_restate_once():
    measured = _asec(
        ("2024:1", "2023:1", "2022:1", "2022:2"),
        [100_000, 100_000, 100_000, 0],
        [2024, 2023, 2022, 2022],
    )
    basis = measured.evidence["price_basis"]
    assert basis["price_basis"] == "census_ccpiu_annual_2024"
    assert basis["cells"] == {"2022": "163.6", "2023": "170.0", "2024": "174.4"}
    assert basis["factors"]["2024"]["expression"] == "identity"
    assert measured.values[0] == 100_000.0
    assert measured.values[1] == 100_000.0 * (174.4 / 170.0)
    assert measured.values[2] == 100_000.0 * (174.4 / 163.6)
    # A coded zero never enters arithmetic and keeps positive-zero bytes.
    assert measured.values[3] == 0.0
    assert not np.signbit(measured.values[3])


def test_asec_keeps_negative_household_income_and_declares_no_open_ended_bound():
    measured = _asec(("2024:1",), [-999_999], [2024])
    assert measured.values.tolist() == [-999_999.0]
    assert measured.quality.tolist() == [reported.QUALITY["exact_publisher_total"]]
    with pytest.raises(reported.ReportedIncomeRefusalError, match="HTOTVAL_RANGE"):
        _asec(("2024:1",), [-1_000_000], [2024])


def test_asec_has_no_out_of_universe_state_for_its_printed_universe():
    measured = _asec(("2024:1", "2024:2"), [1, None], [2024, 2024])
    assert measured.origin.universe_as_printed == "All Households"
    assert reported.AVAILABILITY["unavailable_out_of_universe"] not in set(
        measured.availability.tolist()
    )
    assert measured.availability.tolist() == [
        reported.AVAILABILITY["observed"],
        reported.AVAILABILITY["unavailable_in_universe"],
    ]


def test_asec_refuses_an_income_year_outside_the_restatement_basis():
    with pytest.raises(reported.ReportedIncomeRefusalError, match="INCOME_YEAR_DOMAIN"):
        _asec(("2021:1",), [1], [2021])


def test_measured_artifacts_are_read_only():
    measured = _acs(("2024HU0000001",), [1], [1])
    for array in (measured.values, measured.availability, measured.quality):
        with pytest.raises(ValueError):
            array[0] = 0
    snapshot = measured.raw_values
    snapshot.iloc[0] = 999
    assert measured.raw_values.tolist() == [1]
    evidence = measured.evidence
    evidence["rows"] = 99
    assert measured.evidence["rows"] == 1


def test_the_measurement_module_imports_no_engine_or_frame_loader():
    path = Path(reported.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert imported == {
        "__future__",
        "hashlib",
        "json",
        "struct",
        "collections.abc",
        "dataclasses",
        "importlib",
        "types",
        "numpy",
        "pandas",
        "microcosm.build.quality_aggregates",
    }


def test_b19001_bins_match_the_pinned_group_metadata():
    metadata = reported.B19001_2024_METADATA
    assert metadata["targets_allowed"] is False
    assert metadata["tuning_allowed"] is False
    assert metadata["reservation"] == "reserved_same_source_holdout"
    labels = [
        item.replace("Estimate!!Total:!!", "")
        for item in (
            "Estimate!!Total:!!Less than $10,000",
            "Estimate!!Total:!!$10,000 to $14,999",
            "Estimate!!Total:!!$15,000 to $19,999",
            "Estimate!!Total:!!$20,000 to $24,999",
            "Estimate!!Total:!!$25,000 to $29,999",
            "Estimate!!Total:!!$30,000 to $34,999",
            "Estimate!!Total:!!$35,000 to $39,999",
            "Estimate!!Total:!!$40,000 to $44,999",
            "Estimate!!Total:!!$45,000 to $49,999",
            "Estimate!!Total:!!$50,000 to $59,999",
            "Estimate!!Total:!!$60,000 to $74,999",
            "Estimate!!Total:!!$75,000 to $99,999",
            "Estimate!!Total:!!$100,000 to $124,999",
            "Estimate!!Total:!!$125,000 to $149,999",
            "Estimate!!Total:!!$150,000 to $199,999",
            "Estimate!!Total:!!$200,000 or more",
        )
    ]
    spec = reported.b19001_household_income_distribution_spec("value")
    assert [item.label for item in spec.bins] == labels
    assert len(spec.bins) == len(metadata["bin_variables"]) == 16
    assert spec.bins[0] == DistributionBin("Less than $10,000", None, 10_000.0)
    assert spec.bins[-1] == DistributionBin("$200,000 or more", 200_000.0, None)


def _distribution_frame(values):
    household = pd.DataFrame(
        {
            "household_id": np.arange(1, len(values) + 1, dtype=np.int64),
            "value": np.asarray(values, dtype=np.float64),
            "national": ["us"] * len(values),
        }
    )
    person = pd.DataFrame(
        {
            "person_id": np.arange(1, len(values) + 1, dtype=np.int64),
            "person_household_id": household["household_id"].to_numpy(),
            "person_tax_unit_id": household["household_id"].to_numpy(),
            "person_spm_unit_id": household["household_id"].to_numpy(),
            "person_family_id": household["household_id"].to_numpy(),
            "person_marital_unit_id": household["household_id"].to_numpy(),
        }
    )
    tables = {
        "person": person,
        "household": household,
        "tax_unit": pd.DataFrame({"tax_unit_id": household["household_id"]}),
        "spm_unit": pd.DataFrame({"spm_unit_id": household["household_id"]}),
        "family": pd.DataFrame({"family_id": household["household_id"]}),
        "marital_unit": pd.DataFrame({"marital_unit_id": household["household_id"]}),
    }
    weights = {"household": Weights(np.ones(len(values)), WeightKind.DESIGN)}
    return Frame(tables, US_SCHEMA, weights)


def test_b19001_bin_boundaries_are_exact_at_every_cutpoint():
    # One household exactly on each cutpoint, plus a loss and the open top.
    values = [-4_000.0, 9_999.99, 10_000.0, 199_999.99, 200_000.0]
    frame = _distribution_frame(values)
    spec = reported.b19001_household_income_distribution_spec("value")
    result = population_distributions(
        frame,
        candidate_release_id="invented-fixture",
        population_sha256="0" * 64,
        period="2024",
        distributions=[spec],
        geography_specs=[
            GeographySpec("national", None, "invented", {"us": "United States"})
        ],
    )
    series = result["series"][0]
    assert series["bin_definition_id"] == reported.B19001_2024_BIN_DEFINITION_ID
    weights = {item["label"]: item["weight"] for item in series["bins"]}
    assert weights["Less than $10,000"] == 2.0
    assert weights["$10,000 to $14,999"] == 1.0
    assert weights["$150,000 to $199,999"] == 1.0
    assert weights["$200,000 or more"] == 1.0
    assert series["missing_weight"] == 0.0


def test_unavailable_income_stays_missing_mass_in_the_distribution():
    frame = _distribution_frame([np.nan, 25_000.0])
    spec = reported.b19001_household_income_distribution_spec("value")
    series = population_distributions(
        frame,
        candidate_release_id="invented-fixture",
        population_sha256="0" * 64,
        period="2024",
        distributions=[spec],
        geography_specs=[
            GeographySpec("national", None, "invented", {"us": "United States"})
        ],
    )["series"][0]
    assert series["missing_weight"] == 1.0
    assert sum(item["weight"] for item in series["bins"]) == 1.0


def test_two_arm_comparability_keeps_period_and_universe_differences_visible():
    acs = _acs(("2024HU0000001",), [50_000], [1])
    asec = _asec(("2024:1",), [50_000], [2024])
    report = reported.two_arm_comparability(acs, asec)
    axes = {item["axis"] for item in report["differences"]}
    assert axes == {
        "reference period",
        "price restatement",
        "household universe",
        "declared bounds",
    }
    assert report["shared"]["component_sum"] == (
        "never; both arms use the publisher aggregate"
    )
    assert report["arms"]["acs"]["observation_period"] == "rolling_past_twelve_months"
    assert report["arms"]["asec"]["observation_period"] == "preceding_calendar_year"
    assert json.loads(json.dumps(report)) == report
    with pytest.raises(
        reported.ReportedIncomeRefusalError, match="FIRST_ARGUMENT_MUST_BE_ACS"
    ):
        reported.two_arm_comparability(asec, acs)


def test_no_component_or_split_claim_is_made_by_either_arm():
    acs = _acs(("2024HU0000001",), [50_000], [1])
    asec = _asec(("2024:1",), [50_000], [2024])
    assert acs.evidence["source_columns"] == ["HINCP", "ADJINC", "TYPEHUGQ"]
    assert asec.evidence["source_columns"] == ["asec_HTOTVAL"]
    for measured in (acs, asec):
        text = json.dumps(measured.evidence).lower()
        assert "railroad" not in text
        assert "eight components" not in text
    assert "six of the eight" in acs.evidence["component_reconstruction"]
    assert "CAP_VAL" in asec.evidence["component_reconstruction"]


def test_version_one_roster_matches_the_executed_loader_exactly():
    assert ACS_HOUSEHOLD_SOURCE_COLUMNS_V1.required == acs_pums._HOUSEHOLD_REQUIRED
    assert ACS_HOUSEHOLD_SOURCE_COLUMNS_V1.optional == (
        acs_pums._HOUSEHOLD_STATE_COLUMNS
    )
    assert ACS_HOUSEHOLD_SOURCE_COLUMNS_V1.frame_columns == (
        acs_pums._HOUSEHOLD_FRAME_COLUMNS
    )
    assert ACS_HOUSEHOLD_SOURCE_COLUMNS_V1.added_since_previous == ()
    assert acs_household_source_column_contract(1) is ACS_HOUSEHOLD_SOURCE_COLUMNS_V1
    with pytest.raises(ValueError, match="Unknown ACS household source-column"):
        acs_household_source_column_contract(3)


def test_version_two_only_adds_the_two_income_columns():
    added = ACS_HOUSEHOLD_SOURCE_COLUMNS_V2.added_since_previous
    assert added == ("HINCP", "ADJINC")
    assert set(ACS_HOUSEHOLD_SOURCE_COLUMNS_V2.required) == set(
        ACS_HOUSEHOLD_SOURCE_COLUMNS_V1.required
    ) | set(added)
    assert ACS_HOUSEHOLD_SOURCE_COLUMNS_V2.identity() != (
        ACS_HOUSEHOLD_SOURCE_COLUMNS_V1.identity()
    )
    # The executed roster and its identity are untouched by the new version.
    assert ACS_HOUSEHOLD_SOURCE_COLUMNS_V1.identity() == (
        acs_household_source_column_contract(1).identity()
    )


def _household_row(serialno, **overrides):
    row = {
        "SERIALNO": serialno,
        "ST": "06",
        "PUMA": "00100",
        "WGTP": 10,
        "NP": 1,
        "ADJHSG": 1_000_000,
        "ADJINC": DECLARED_ADJINC,
        "TEN": 1,
        "RNTP": None,
        "GRNTP": None,
        "TAXAMT": 2_400,
        "TYPEHUGQ": 1,
        "HINCP": 50_000,
    }
    row.update(overrides)
    return row


def _household_archive(tmp_path: Path, rows) -> acs_pums.AcsPumsSource:
    household_zip = tmp_path / "csv_hus.zip"
    with ZipFile(household_zip, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("psam_husa.csv", pd.DataFrame(rows).to_csv(index=False))
    return acs_pums.AcsPumsSource(household_zip, tmp_path / "csv_pus.zip")


def test_version_two_read_keeps_blanks_missing_and_never_adjusts(tmp_path):
    source = _household_archive(
        tmp_path,
        [
            _household_row("2024HU0000002", HINCP=-59_999),
            _household_row("2024HU0000001"),
            _household_row("2024GQ0000001", TYPEHUGQ=2, WGTP=0, HINCP=None, TEN=None),
            _household_row("2024HU0000009", NP=0, HINCP=None, TEN=None),
        ],
    )
    table, receipt = read_acs_household_income_columns(source)
    assert table["SERIALNO"].tolist() == [
        "2024GQ0000001",
        "2024HU0000001",
        "2024HU0000002",
    ]
    assert table["HINCP"].tolist() == [pd.NA, 50_000, -59_999]
    assert table["HINCP"].dtype == pd.Int64Dtype()
    assert receipt["vacant_household_rows_dropped"] == 1
    assert receipt["adjustment_applied"] is False
    assert receipt["household_source_column_version"] == 2
    assert receipt["join_key"] == "SERIALNO"

    measured = reported.measure_acs_household_reported_income(
        household_keys=tuple(table["SERIALNO"]),
        hincp=table["HINCP"],
        adjinc=pd.array([DECLARED_ADJINC] * len(table), dtype="Int64"),
        housing_unit_kind=table["TYPEHUGQ"],
    )
    assert measured.availability.tolist() == [
        reported.AVAILABILITY["unavailable_out_of_universe"],
        reported.AVAILABILITY["observed"],
        reported.AVAILABILITY["observed"],
    ]


def test_origin_as_json_is_an_independent_copy():
    origin = reported.ACS_2024_HINCP_ORIGIN.as_json()
    origin["universe_as_printed"] = "rewritten"
    assert reported.ACS_2024_HINCP_ORIGIN.as_json()["universe_as_printed"] == (
        "Occupied housing units; N/A (GQ/vacant)"
    )
    pages = {
        item.get("pdf_page_1based") or item.get("printed_page")
        for item in reported.ACS_2024_HINCP_ORIGIN.as_json()["authority"]
    }
    assert pages == {15, 2, 16, 96, 98}
    assert json.loads(json.dumps(origin)) == origin


def test_version_one_frame_ignores_the_version_two_columns(tmp_path):
    """A v2-wide archive still produces exactly the executed v1 household table."""

    household_zip = tmp_path / "csv_hus.zip"
    person_zip = tmp_path / "csv_pus.zip"
    with ZipFile(household_zip, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(
            "psam_husa.csv",
            pd.DataFrame(
                [_household_row("2024HU0000001"), _household_row("2024HU0000002")]
            ).to_csv(index=False),
        )
    person = {
        "SERIALNO": "2024HU0000001",
        "SPORDER": 1,
        "RELSHIPP": 20,
        "AGEP": 40,
        "SEX": 1,
        "MAR": 5,
        "ADJINC": DECLARED_ADJINC,
        "WAGP": 50_000,
        "SEMP": 0,
        "SSP": 0,
        "SSIP": 0,
        "RETP": 0,
        "INTP": 0,
        "PWGTP": 10,
    }
    with ZipFile(person_zip, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(
            "psam_pusa.csv",
            pd.DataFrame([person, {**person, "SERIALNO": "2024HU0000002"}]).to_csv(
                index=False
            ),
        )
    frame, _receipt = acs_pums.build_acs_pums_unit_frame(
        acs_pums.AcsPumsSource(household_zip, person_zip)
    )
    columns = set(frame.table("household").columns)
    assert not columns & set(ACS_HOUSEHOLD_SOURCE_COLUMNS_V2.added_since_previous)
    assert set(ACS_HOUSEHOLD_SOURCE_COLUMNS_V1.frame_columns) <= columns

    table, _read = read_acs_household_income_columns(
        acs_pums.AcsPumsSource(household_zip, person_zip)
    )
    assert table["HINCP"].tolist() == [50_000, 50_000]


def test_version_two_read_refuses_a_vintage_it_is_not_pinned_to(tmp_path):
    with pytest.raises(ValueError, match="ACS multispine v1 is pinned"):
        acs_pums.AcsPumsSource(
            tmp_path / "csv_hus.zip", tmp_path / "csv_pus.zip", vintage=2023
        )
