"""Per-vintage reported-coverage signal gate tests (microcosm #720)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from microcosm.build.us_runtime import (
    CPS_CARRIED_PERSON_INPUTS,
    US_REPORTED_COVERAGE_PERSON_INPUTS,
    US_REPORTED_COVERAGE_VINTAGE_GATE_MIN_ROWS,
    us_reported_coverage_vintage_signal_gate,
)
from microcosm.build.us_runtime.cps_carried import _fill_health_coverage_inputs
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

_ALL_FALSE = {column: False for column in US_REPORTED_COVERAGE_PERSON_INPUTS}


def _person_table(rows: list[dict]) -> pd.DataFrame:
    records = []
    for index, row in enumerate(rows):
        record = {"A_AGE": 40}
        record.update(row)
        record.setdefault("person_id", index + 1)
        record.setdefault("person_household_id", index + 1)
        records.append(record)
    return pd.DataFrame(records)


def _us_frame(person_rows: list[dict]) -> Frame:
    return _us_frame_from_person(_person_table(person_rows))


def _us_frame_from_person(person: pd.DataFrame) -> Frame:
    person = person.copy()
    n = len(person)
    household_ids = person["person_household_id"].to_numpy()
    unique_households = np.unique(household_ids)
    person["person_tax_unit_id"] = person["person_household_id"] + 1_000
    person["person_spm_unit_id"] = person["person_household_id"] + 2_000
    person["person_family_id"] = person["person_household_id"] + 3_000
    person["person_marital_unit_id"] = np.arange(n, dtype="int64") + 4_000
    tables = {
        "person": person,
        "household": pd.DataFrame({"household_id": unique_households}),
        "tax_unit": pd.DataFrame({"tax_unit_id": unique_households + 1_000}),
        "spm_unit": pd.DataFrame({"spm_unit_id": unique_households + 2_000}),
        "family": pd.DataFrame({"family_id": unique_households + 3_000}),
        "marital_unit": pd.DataFrame(
            {"marital_unit_id": np.arange(n, dtype="int64") + 4_000}
        ),
    }
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                values=np.ones(len(unique_households), dtype=np.float64),
                kind=WeightKind.DESIGN,
            )
        },
    )


def _vintage_rows(
    source_year: int,
    *,
    reporters: tuple[str, ...] = US_REPORTED_COVERAGE_PERSON_INPUTS,
    rows: int = 3,
) -> list[dict]:
    """``rows`` persons of one vintage; the first row reports every column in
    ``reporters``, the rest report nothing."""

    out: list[dict] = []
    for index in range(rows):
        record = {"source_year": source_year, **_ALL_FALSE}
        if index == 0:
            record.update({column: True for column in reporters})
        out.append(record)
    return out


class TestContract:
    def test_the_nine_inputs_are_cps_carried_flags(self) -> None:
        assert len(US_REPORTED_COVERAGE_PERSON_INPUTS) == 9
        assert set(US_REPORTED_COVERAGE_PERSON_INPUTS) <= CPS_CARRIED_PERSON_INPUTS
        assert all(
            column == "has_esi" or column.endswith("_coverage_at_interview")
            for column in US_REPORTED_COVERAGE_PERSON_INPUTS
        )

    def test_the_nine_inputs_are_exactly_the_derived_health_coverage_flags(
        self,
    ) -> None:
        person = pd.DataFrame({"NOW_GRP": [1, 2]})
        _fill_health_coverage_inputs(person)
        derived = {column for column in person.columns if column.startswith("has_")}
        assert derived == set(US_REPORTED_COVERAGE_PERSON_INPUTS)

    def test_default_threshold_is_a_release_pool_scale(self) -> None:
        assert US_REPORTED_COVERAGE_VINTAGE_GATE_MIN_ROWS == 5_000


class TestGate:
    def test_every_vintage_with_signal_passes(self) -> None:
        frame = _us_frame(_vintage_rows(2023) + _vintage_rows(2024))
        gate = us_reported_coverage_vintage_signal_gate(frame, min_vintage_rows=3)
        assert gate.passed, gate.failures
        assert gate.name == "reported_coverage_vintage_signal"
        vintages = gate.details["vintages"]
        assert set(vintages) == {"2023", "2024"}
        for summary in vintages.values():
            assert summary["rows"] == 3
            assert summary["enforced"] is True
            assert (
                summary["reporter_counts"]["has_medicaid_health_coverage_at_interview"]
                == 1
            )

    def test_vintage_without_medicaid_reporters_fails(self) -> None:
        without_medicaid = tuple(
            column
            for column in US_REPORTED_COVERAGE_PERSON_INPUTS
            if column != "has_medicaid_health_coverage_at_interview"
        )
        frame = _us_frame(
            _vintage_rows(2022, reporters=without_medicaid) + _vintage_rows(2024)
        )
        gate = us_reported_coverage_vintage_signal_gate(frame, min_vintage_rows=3)
        assert not gate.passed
        assert len(gate.failures) == 1
        assert "has_medicaid_health_coverage_at_interview" in gate.failures[0]
        assert "vintage 2022 has 0 reporters" in gate.failures[0]
        assert "#720" in gate.failures[0]

    def test_issue_720_shape_fails_seven_flags_for_two_vintages(self) -> None:
        # Build P: ASEC 2022/2023 inputs carried only NOW_GRP and NOW_MRK, so
        # every other at-interview flag was False for those vintages.
        carried = ("has_esi", "has_marketplace_health_coverage_at_interview")
        frame = _us_frame(
            _vintage_rows(2022, reporters=carried)
            + _vintage_rows(2023, reporters=carried)
            + _vintage_rows(2024)
        )
        gate = us_reported_coverage_vintage_signal_gate(frame, min_vintage_rows=3)
        assert not gate.passed
        assert len(gate.failures) == 14
        failed_vintages = {
            failure.split("vintage ")[1].split(" ")[0] for failure in gate.failures
        }
        assert failed_vintages == {"2022", "2023"}
        assert not any("has_esi" in failure for failure in gate.failures)
        assert (
            gate.details["vintages"]["2024"]["reporter_counts"][
                "has_medicaid_health_coverage_at_interview"
            ]
            == 1
        )
        assert (
            gate.details["vintages"]["2022"]["reporter_counts"][
                "has_medicaid_health_coverage_at_interview"
            ]
            == 0
        )

    def test_small_vintage_is_recorded_but_not_enforced(self) -> None:
        frame = _us_frame(
            _vintage_rows(2022, reporters=(), rows=2) + _vintage_rows(2024)
        )
        gate = us_reported_coverage_vintage_signal_gate(frame, min_vintage_rows=3)
        assert gate.passed, gate.failures
        assert gate.details["vintages"]["2022"]["enforced"] is False
        assert gate.details["vintages"]["2024"]["enforced"] is True
        assert gate.details["min_vintage_rows"] == 3

    def test_default_threshold_skips_unit_sized_frames(self) -> None:
        frame = _us_frame(_vintage_rows(2022, reporters=()))
        gate = us_reported_coverage_vintage_signal_gate(frame)
        assert gate.passed, gate.failures
        assert gate.details["vintages"]["2022"]["enforced"] is False

    def test_missing_source_year_fails_closed(self) -> None:
        rows = [{**row} for row in _vintage_rows(2024)]
        for row in rows:
            del row["source_year"]
        gate = us_reported_coverage_vintage_signal_gate(
            _us_frame(rows), min_vintage_rows=3
        )
        assert not gate.passed
        assert gate.failures == ("person column missing: source_year.",)

    def test_null_source_year_rows_fail(self) -> None:
        rows = _vintage_rows(2024) + [{**_vintage_rows(2024)[0], "source_year": None}]
        gate = us_reported_coverage_vintage_signal_gate(
            _us_frame(rows), min_vintage_rows=3
        )
        assert not gate.passed
        assert any("1 person rows have no source year" in f for f in gate.failures)

    def test_support_channel_separates_acs_from_asec(self) -> None:
        # An ACS-spine vintage with signal must not mask an ASEC vintage
        # whose source lacked the recode: both carry source_year 2024.
        asec = [
            {**row, "person_support_channel": "asec"}
            for row in _vintage_rows(
                2024,
                reporters=("has_esi", "has_marketplace_health_coverage_at_interview"),
            )
        ]
        acs = [{**row, "person_support_channel": "acs"} for row in _vintage_rows(2024)]
        gate = us_reported_coverage_vintage_signal_gate(
            _us_frame(asec + acs), min_vintage_rows=3
        )
        assert not gate.passed
        assert gate.details["grouping"] == ["person_support_channel", "source_year"]
        assert set(gate.details["vintages"]) == {"acs/2024", "asec/2024"}
        assert len(gate.failures) == 7
        assert all("vintage asec/2024" in failure for failure in gate.failures)

    def test_null_flag_values_fail_completeness(self) -> None:
        person = _person_table(_vintage_rows(2024))
        person["has_medicaid_health_coverage_at_interview"] = pd.array(
            [True, None, False], dtype="boolean"
        )
        gate = us_reported_coverage_vintage_signal_gate(
            _us_frame_from_person(person), min_vintage_rows=3
        )
        assert not gate.passed
        assert len(gate.failures) == 1
        assert "1 null values" in gate.failures[0]
        assert (
            gate.details["vintages"]["2024"]["null_counts"][
                "has_medicaid_health_coverage_at_interview"
            ]
            == 1
        )

    def test_non_boolean_dtype_fails(self) -> None:
        person = _person_table(_vintage_rows(2024))
        person["has_esi"] = ["True", "False", "False"]
        gate = us_reported_coverage_vintage_signal_gate(
            _us_frame_from_person(person), min_vintage_rows=3
        )
        assert not gate.passed
        assert len(gate.failures) == 1
        assert "non-boolean dtype" in gate.failures[0]

    def test_missing_column_fails(self) -> None:
        rows = [{**row} for row in _vintage_rows(2024)]
        for row in rows:
            del row["has_esi"]
        gate = us_reported_coverage_vintage_signal_gate(
            _us_frame(rows), min_vintage_rows=3
        )
        assert not gate.passed
        assert gate.failures == ("person column missing: has_esi.",)
        assert gate.details == {"missing": ["has_esi"]}


class TestMechanism:
    def test_nan_source_recode_becomes_false_and_fails_the_vintage(self) -> None:
        """The #720 chain: a vintage whose source lacks ``NOW_MCAID`` derives
        ``False`` for every person, and only this gate sees it."""

        person = _person_table(
            [
                {"source_year": 2023, "NOW_GRP": 1, "NOW_MRK": 2, "NOW_MCAID": np.nan},
                {"source_year": 2023, "NOW_GRP": 2, "NOW_MRK": 1, "NOW_MCAID": np.nan},
                {"source_year": 2023, "NOW_GRP": 1, "NOW_MRK": 2, "NOW_MCAID": np.nan},
                {"source_year": 2024, "NOW_GRP": 1, "NOW_MRK": 2, "NOW_MCAID": 1},
                {"source_year": 2024, "NOW_GRP": 2, "NOW_MRK": 1, "NOW_MCAID": 2},
                {"source_year": 2024, "NOW_GRP": 1, "NOW_MRK": 2, "NOW_MCAID": 1},
            ]
        )
        for column in (
            "NOW_NONM",
            "NOW_CHAMPVA",
            "NOW_MIL",
            "NOW_VACARE",
            "NOW_OTHMT",
            "NOW_IHSFLG",
        ):
            person[column] = [np.nan] * 3 + [1, 2, 1]
        _fill_health_coverage_inputs(person)

        medicaid = person["has_medicaid_health_coverage_at_interview"]
        assert not medicaid[person["source_year"] == 2023].any()
        assert medicaid[person["source_year"] == 2024].sum() == 2
        assert person["has_esi"].sum() == 4  # NOW_GRP carried for both vintages

        gate = us_reported_coverage_vintage_signal_gate(
            _us_frame_from_person(person), min_vintage_rows=3
        )
        assert not gate.passed
        failed = {failure.split(":")[0] for failure in gate.failures}
        assert failed == set(US_REPORTED_COVERAGE_PERSON_INPUTS) - {
            "has_esi",
            "has_marketplace_health_coverage_at_interview",
        }
        assert all(
            "vintage 2023 has 0 reporters" in failure for failure in gate.failures
        )
