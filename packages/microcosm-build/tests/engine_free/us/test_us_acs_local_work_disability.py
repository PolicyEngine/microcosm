"""Native ACS disability and weeks-worked inputs for the ACS local lane
(microcosm#1021)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import acs_multispine
from microcosm.build.us_runtime.acs_inputs import map_acs_native_inputs
from microcosm.build.us_runtime.acs_local_work_disability import (
    ACS_DISABILITY_ITEMS,
    ACS_LOCAL_WORK_DISABILITY_GATE_NAME,
    ACS_NATIVE_PROVENANCE,
    WEEKS_WORKED_EXPORT_BLOCKER,
    acs_local_work_disability_signal_gate,
    map_acs_local_work_disability_inputs,
    record_acs_local_work_disability_transfer,
)
from microcosm.build.us_runtime.acs_pums import (
    ACS_2024_1YR_SPINE,
    AcsPumsSource,
)
from microcosm.build.us_runtime.acs_transfer import ASEC_PUF_DONOR_SPINE
from microcosm.build.us_runtime.base_pool import spine_column
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

TAG = spine_column("person")
_NO = {item: 2 for item in ACS_DISABILITY_ITEMS}

#: Eight ACS persons in four households of two (ages, work and SSI chosen to
#: hit each rule): a full-year worker and a 3-year-old; a part-year worker
#: with a hearing difficulty and a 10-year-old with a cognitive one (items
#: below their minimum question age are blank); a blind nonworker and an SSI
#: recipient of 45 with no difficulty; an SSI recipient of 70 with no
#: difficulty and a part-year worker of 64 with an independent-living one.
_PEOPLE: tuple[dict[str, Any], ...] = (
    {"AGEP": 40, "WKL": 1, "WKHP": 40, "WKWN": 52, "WAGP": 50_000.0, **_NO},
    {"AGEP": 3, "WAGP": np.nan, "SSIP": np.nan, "DEAR": 2, "DEYE": 2},
    {"AGEP": 30, "WKL": 1, "WKHP": 20, "WKWN": 26, "WAGP": 12_000.0, **_NO, "DEAR": 1},
    {
        "AGEP": 10,
        "WAGP": np.nan,
        "SSIP": np.nan,
        "DEAR": 2,
        "DEYE": 2,
        "DREM": 1,
        "DPHY": 2,
        "DDRS": 2,
    },
    {"AGEP": 50, "WKL": 3, **_NO, "DEYE": 1},
    {"AGEP": 45, "WKL": 2, "SSIP": 800.0, **_NO},
    {"AGEP": 70, "WKL": 3, "SSIP": 500.0, **_NO},
    {"AGEP": 64, "WKL": 1, "WKHP": 15, "WKWN": 10, "WAGP": 3_000.0, **_NO, "DOUT": 1},
)
_HOUSEHOLDS = (1, 1, 2, 2, 3, 3, 4, 4)
_RELATIONSHIPS = (20, 25, 20, 25, 20, 21, 20, 21)
_EXPECTED_DISABLED = [False, False, True, True, True, True, False, True]
_EXPECTED_BLIND = [False, False, False, False, True, False, False, False]
_EXPECTED_WEEKS = [52.0, np.nan, 26.0, np.nan, 0.0, 0.0, 0.0, 10.0]


def _raw_acs_frame(overrides: dict[int, dict[str, Any]] | None = None) -> Frame:
    records = []
    for index, person in enumerate(_PEOPLE):
        row = {
            "SEX": 1 + index % 2,
            "ADJINC": 1_000_000,
            "WAGP": 0.0,
            "SEMP": 0.0,
            "SSP": 0.0,
            "SSIP": 0.0,
            "RETP": 0.0,
            "INTP": 0.0,
            "WKHP": np.nan,
            "WKL": np.nan,
            "WKWN": np.nan,
            **{item: np.nan for item in ACS_DISABILITY_ITEMS},
            **person,
            **(overrides or {}).get(index, {}),
        }
        records.append(row)
    person = pd.DataFrame(records)
    household_ids = np.asarray([10 * h for h in _HOUSEHOLDS])
    person.insert(0, "person_id", np.arange(1, len(person) + 1))
    person["RELSHIPP"] = list(_RELATIONSHIPS)
    for entity in US_SCHEMA.entities:
        if entity != "household":
            person[f"person_{entity}_id"] = household_ids
    person["person_household_id"] = household_ids
    households = np.unique(household_ids)
    tables = {
        "person": person,
        "household": pd.DataFrame(
            {
                "household_id": households,
                "state_fips": [6, 36, 6, 36],
                "ADJHSG": 1_000_000,
                "TEN": 3,
                "RNTP": 1_000.0,
                "GRNTP": 1_200.0,
                "TAXAMT": np.nan,
            }
        ),
    }
    for entity in US_SCHEMA.entities:
        if entity not in tables:
            tables[entity] = pd.DataFrame({f"{entity}_id": households})
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                np.asarray([100.0, 200.0, 150.0, 50.0]), WeightKind.DESIGN
            )
        },
        pd.Series(ACS_2024_1YR_SPINE, index=person.index, dtype=object),
    )


def _mapped(overrides: dict[int, dict[str, Any]] | None = None) -> Frame:
    return map_acs_native_inputs(_raw_acs_frame(overrides)).frame


def _base_donor_frame() -> Frame:
    person = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4],
            "person_household_id": [1, 1, 2, 2],
            "person_tax_unit_id": [1, 1, 2, 2],
            "person_spm_unit_id": [1, 1, 2, 2],
            "person_family_id": [1, 1, 2, 2],
            "person_marital_unit_id": [1, 1, 2, 2],
            "age": [42.0, 38.0, 67.0, 12.0],
            "is_female": [False, True, True, False],
            "is_disabled": [False, True, True, False],
            "is_blind": [False, False, True, False],
            "qualified_dividend_income": [100.0, 20.0, 2_000.0, 0.0],
        }
    )
    return Frame(
        {
            "person": person,
            "household": pd.DataFrame({"household_id": [1, 2], "state_fips": [6, 36]}),
            "tax_unit": pd.DataFrame({"tax_unit_id": [1, 2]}),
            "spm_unit": pd.DataFrame({"spm_unit_id": [1, 2]}),
            "family": pd.DataFrame({"family_id": [1, 2]}),
            "marital_unit": pd.DataFrame({"marital_unit_id": [1, 2]}),
        },
        US_SCHEMA,
        {"household": Weights(np.asarray([100.0, 300.0]), WeightKind.DESIGN)},
        pd.Series(ASEC_PUF_DONOR_SPINE, index=person.index, dtype=object),
    )


_TARGETS = {
    "person": {
        "model_required_boolean": ("is_blind", "is_disabled"),
        "tax_detail": ("qualified_dividend_income",),
    }
}


def _stage(monkeypatch, tmp_path, *, work_disability_inputs: bool = True):
    raw = _raw_acs_frame()
    monkeypatch.setattr(
        acs_multispine,
        "build_acs_pums_unit_frame",
        lambda source, *, chunksize: (raw, {"spine": ACS_2024_1YR_SPINE}),
    )
    return acs_multispine.build_optional_acs_multispine(
        _base_donor_frame(),
        AcsPumsSource(tmp_path / "csv_hus.zip", tmp_path / "csv_pus.zip"),
        acs_share=0.5,
        target_families=_TARGETS,
        work_disability_inputs=work_disability_inputs,
        seed=4,
        n_estimators=2,
    )


@pytest.fixture(scope="module")
def staged(tmp_path_factory):
    with pytest.MonkeyPatch.context() as monkeypatch:
        return _stage(monkeypatch, tmp_path_factory.mktemp("acs"))


def _with_person(frame: Frame, **columns: object) -> Frame:
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    for column, values in columns.items():
        if values is None:
            tables["person"] = tables["person"].drop(columns=[column])
        else:
            tables["person"][column] = values
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
    )


def _acs_mask(frame: Frame) -> np.ndarray:
    return frame.table("person")[TAG].eq(ACS_2024_1YR_SPINE).to_numpy()


def _replace_acs(frame: Frame, column: str, acs_values) -> Frame:
    values = frame.table("person")[column].to_numpy(copy=True).astype(object)
    values[_acs_mask(frame)] = acs_values
    return _with_person(frame, **{column: values})


# --- native mapping --------------------------------------------------------


def test_native_mapping_follows_the_asec_definition() -> None:
    result = map_acs_local_work_disability_inputs(_mapped())
    person = result.frame.table("person")
    assert person["is_disabled"].tolist() == _EXPECTED_DISABLED
    assert person["is_blind"].tolist() == _EXPECTED_BLIND
    assert person["is_disabled"].dtype == np.dtype(bool)
    for column in ("is_disabled", "is_blind"):
        native = result.native_inputs[column]
        assert native["entity"] == "person"
        assert native["provenance"] == ACS_NATIVE_PROVENANCE
        assert native["observed_rows"] == 8
        assert native["missing_rows"] == 0
    assert set(ACS_DISABILITY_ITEMS) <= set(
        result.native_inputs["is_disabled"]["source_columns"]
    )
    assert "DEYE" in result.native_inputs["is_blind"]["source_columns"]
    receipt = result.receipt
    assert receipt["issue"] == "microcosm#1021"
    assert receipt["acs_persons"] == 8
    assert receipt["is_disabled"]["source"] == ACS_NATIVE_PROVENANCE
    assert receipt["is_disabled"]["true_rows"] == 5
    # Hearing, vision, cognitive (age 10) and independent living; the 45-year
    # SSI recipient is the SSI increment.
    assert receipt["is_disabled"]["difficulty_rows"] == 4
    assert receipt["is_disabled"]["ssi_increment_rows"] == 1
    assert receipt["is_blind"]["true_rows"] == 1
    # imputed_rows only exists once the transfer has been recorded.
    assert "imputed_rows" not in receipt["is_disabled"]


def test_items_below_their_minimum_question_age_read_as_no_difficulty() -> None:
    result = map_acs_local_work_disability_inputs(_mapped())
    items = result.receipt["difficulty_items"]
    assert items["DEAR"] == {
        "minimum_question_age": 0,
        "yes_rows": 1,
        "out_of_universe_rows": 0,
    }
    # The 3-year-old for the 5+ items; both children for independent living.
    for item in ("DREM", "DPHY", "DDRS"):
        assert items[item]["minimum_question_age"] == 5
        assert items[item]["out_of_universe_rows"] == 1
    assert items["DOUT"]["minimum_question_age"] == 15
    assert items["DOUT"]["out_of_universe_rows"] == 2
    person = result.frame.table("person")
    assert not bool(person.loc[1, "is_disabled"])  # 3: hearing and vision only
    assert bool(person.loc[3, "is_disabled"])  # DREM at age 10


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({3: {"DOUT": 2}}, "DOUT contradicts its minimum question age 15"),
        ({1: {"DREM": 2}}, "DREM contradicts its minimum question age 5"),
        ({0: {"DEAR": np.nan}}, "DEAR contradicts its minimum question age 0"),
        ({0: {"DPHY": 3}}, r"DPHY requires blank or integer codes within \[1, 2\]"),
        # map_acs_native_inputs accepts a blank age; this stage must not.
        ({0: {"AGEP": np.nan}}, "nonnegative AGEP"),
    ],
    ids=["coded-below-age", "coded-under-five", "blank-in-universe", "code", "age"],
)
def test_item_universe_contradictions_are_refused(overrides, match) -> None:
    with pytest.raises(ValueError, match=match):
        map_acs_local_work_disability_inputs(_mapped(overrides))


@pytest.mark.parametrize(
    "age, ssi, expected",
    [(64, 100.0, True), (65, 100.0, False), (40, 0.0, False), (40, np.nan, False)],
)
def test_ssi_rule_counts_reported_ssi_under_65_only(age, ssi, expected) -> None:
    result = map_acs_local_work_disability_inputs(
        _mapped({5: {"AGEP": age, "SSIP": ssi, "WKL": 3}})
    )
    assert bool(result.frame.table("person").loc[5, "is_disabled"]) is expected
    assert result.receipt["is_disabled"]["ssi_increment_rows"] == int(expected)


def test_is_blind_reads_the_vision_item_only() -> None:
    result = map_acs_local_work_disability_inputs(
        _mapped({0: {"DEYE": 1}, 4: {"DEYE": 2, "DPHY": 1}})
    )
    person = result.frame.table("person")
    assert person["is_blind"].tolist() == [True] + [False] * 7
    assert bool(person.loc[4, "is_disabled"])  # still disabled, via DPHY


def test_weeks_worked_follows_the_hours_universe_and_is_staging_only() -> None:
    result = map_acs_local_work_disability_inputs(_mapped())
    weeks = result.frame.table("person")["weeks_worked"].to_numpy(dtype=float)
    np.testing.assert_array_equal(weeks, np.asarray(_EXPECTED_WEEKS))
    receipt = result.receipt["weeks_worked"]
    assert receipt["source_value_rows"] == 3
    assert receipt["structural_zero_rows"] == 3
    assert receipt["source_universe_unavailable_rows"] == 2
    assert receipt["source_unresolved_rows"] == 0
    assert receipt["hours_paired_rows"] == 3
    assert receipt["exported"] is False
    assert receipt["export_blocker"] == WEEKS_WORKED_EXPORT_BLOCKER
    assert result.native_inputs["weeks_worked"]["missing_rows"] == 2


def test_veteran_and_self_care_are_documented_not_exported() -> None:
    result = map_acs_local_work_disability_inputs(_mapped())
    frame = result.frame.table("person")
    assert "is_veteran" not in frame
    assert "is_incapable_of_self_care" not in frame
    veteran = result.receipt["not_exported"]["is_veteran"]
    assert veteran["source_column"] == "ACS_MIL"
    assert veteran["coding"]["veteran"] == [2]
    assert 1 in veteran["coding"]["not_veteran"]
    assert 3 in veteran["coding"]["not_veteran"]
    assert "policyengine-us#9662" in veteran["reason"]
    self_care = result.receipt["not_exported"]["is_incapable_of_self_care"]
    assert self_care["status"] == "asec_transfer"


@pytest.mark.parametrize("column", ["WKWN", "DEAR", "SSIP"])
def test_mapper_refuses_a_source_without_its_columns(column) -> None:
    mapped = _with_person(_mapped(), **{column: None})
    with pytest.raises(ValueError, match=column):
        map_acs_local_work_disability_inputs(mapped)


def test_mapper_refuses_to_overwrite_an_existing_flag() -> None:
    mapped = _with_person(_mapped(), is_disabled=[False] * 8)
    with pytest.raises(ValueError, match="overwrite existing column 'is_disabled'"):
        map_acs_local_work_disability_inputs(mapped)


def test_recorded_transfer_counts_only_the_disability_columns() -> None:
    receipt = map_acs_local_work_disability_inputs(_mapped()).receipt
    recorded = record_acs_local_work_disability_transfer(
        receipt,
        [
            {"column": "is_disabled", "imputed_recipient_rows": 3},
            {"column": "has_esi", "imputed_recipient_rows": 8},
        ],
    )
    assert recorded["is_disabled"]["imputed_rows"] == 3
    assert recorded["is_disabled"]["transfer_entries"] == 1
    assert recorded["is_blind"]["imputed_rows"] == 0
    assert recorded["is_blind"]["transfer_entries"] == 0
    assert "imputed_rows" not in receipt["is_disabled"]  # input untouched


# --- local lane: native values survive the transfer ------------------------


def test_transfer_skips_native_complete_targets_and_leaves_asec_rows(staged):
    person = staged.frame.table("person")
    acs = person[TAG].eq(ACS_2024_1YR_SPINE)
    assert person.loc[acs, "is_disabled"].tolist() == _EXPECTED_DISABLED
    assert person.loc[acs, "is_blind"].tolist() == _EXPECTED_BLIND
    donor = _base_donor_frame().table("person")
    assert person.loc[~acs, "is_disabled"].tolist() == donor["is_disabled"].tolist()
    assert person.loc[~acs, "is_blind"].tolist() == donor["is_blind"].tolist()
    np.testing.assert_array_equal(
        person.loc[acs, "weeks_worked"].to_numpy(dtype=float),
        np.asarray(_EXPECTED_WEEKS),
    )
    assert person.loc[~acs, "weeks_worked"].isna().all()
    provenance = staged.provenance
    imputed = {item["column"] for item in provenance["imputed_inputs"]}
    assert imputed == {"qualified_dividend_income"}
    for column in ("is_disabled", "is_blind", "weeks_worked"):
        assert provenance["native_inputs"][column]["provenance"] == (
            ACS_NATIVE_PROVENANCE
        )
    receipt = provenance["acs_local_work_disability"]
    for column in ("is_disabled", "is_blind"):
        assert receipt[column]["imputed_rows"] == 0
        assert receipt[column]["transfer_entries"] == 0


def test_without_the_local_mapper_the_transfer_imputes_the_flags(monkeypatch, tmp_path):
    """The pre-#1021 lane: ACS disability came from ASEC donors."""

    result = _stage(monkeypatch, tmp_path, work_disability_inputs=False)
    imputed = {item["column"] for item in result.provenance["imputed_inputs"]}
    assert {"is_disabled", "is_blind"} <= imputed
    assert "acs_local_work_disability" not in result.provenance
    assert "weeks_worked" not in result.frame.table("person")


# --- gate ------------------------------------------------------------------


def _receipt(staged) -> dict:
    return staged.provenance["acs_local_work_disability"]


def test_gate_passes_on_the_staged_frame_and_reports_age_bands(staged) -> None:
    gate = acs_local_work_disability_signal_gate(staged.frame, receipt=_receipt(staged))
    assert gate.passed, gate.failures
    assert gate.name == ACS_LOCAL_WORK_DISABILITY_GATE_NAME
    acs = gate.details["per_spine"][ACS_2024_1YR_SPINE]
    assert acs["graded"] is True
    assert acs["native_mismatch_rows"] == {"is_disabled": 0, "is_blind": 0}
    bands = acs["shares_by_age_band"]
    assert set(bands) == {"under_18", "18_to_64", "65_and_over"}
    for band in bands.values():
        assert {"difficulty_share", "ssi_increment_share", "is_disabled_share"} <= (
            set(band)
        )
    assert bands["65_and_over"]["is_disabled_share"] == 0.0
    assert bands["18_to_64"]["ssi_increment_share"] > 0
    assert acs["ssi_increment"]["rows"] == 1
    weeks = acs["weeks_worked"]
    assert weeks["hours_unpaired_rows"] == 0
    assert weeks["worker_rows"] == 3
    assert weeks["part_year_worker_rows"] == 2
    donor = gate.details["per_spine"][ASEC_PUF_DONOR_SPINE]
    assert donor["graded"] is False
    assert "is_disabled_share_by_age_band" in donor
    assert gate.details["receipt"]["is_disabled"] == {
        "source": ACS_NATIVE_PROVENANCE,
        "imputed_rows": 0,
    }


def test_gate_refuses_a_missing_or_imputed_receipt(staged) -> None:
    missing = acs_local_work_disability_signal_gate(staged.frame, receipt=None)
    assert not missing.passed
    assert "No acs_local_work_disability staging receipt" in missing.failures[-1]

    receipt = _receipt(staged)
    imputed = {**receipt, "is_blind": {**receipt["is_blind"], "imputed_rows": 3}}
    gate = acs_local_work_disability_signal_gate(staged.frame, receipt=imputed)
    assert "receipt: the ASEC transfer imputed 3 ACS is_blind cell(s)." in (
        gate.failures
    )
    unrecorded = {**receipt, "is_disabled": {"source": ACS_NATIVE_PROVENANCE}}
    gate = acs_local_work_disability_signal_gate(staged.frame, receipt=unrecorded)
    assert "receipt: is_disabled records no transfer imputation count." in (
        gate.failures
    )
    wrong_rows = {**receipt, "acs_persons": 9}
    gate = acs_local_work_disability_signal_gate(staged.frame, receipt=wrong_rows)
    assert any("records 9 ACS person(s)" in failure for failure in gate.failures)


def test_gate_refuses_acs_flags_that_are_not_the_native_items(staged) -> None:
    flipped = list(_EXPECTED_DISABLED)
    flipped[0] = True
    frame = _replace_acs(staged.frame, "is_disabled", flipped)
    gate = acs_local_work_disability_signal_gate(frame, receipt=_receipt(staged))
    assert (
        f"{ACS_2024_1YR_SPINE}: 1 person(s) carry a is_disabled that differs from "
        "their native ACS items; the ACS value must be measured, not imputed."
    ) in gate.failures


def test_gate_refuses_the_engine_default_signature(staged) -> None:
    frame = _replace_acs(staged.frame, "is_disabled", [False] * 8)
    gate = acs_local_work_disability_signal_gate(frame, receipt=_receipt(staged))
    assert (
        f"{ACS_2024_1YR_SPINE}: is_disabled is constant False; a spine with no "
        "variation carries no disability signal."
    ) in gate.failures


def test_gate_refuses_missing_cells_and_columns(staged) -> None:
    values = staged.frame.table("person")["is_blind"].to_numpy(copy=True).astype(object)
    values[~_acs_mask(staged.frame)] = [None, False, True, False]
    frame = _with_person(staged.frame, is_blind=values)
    gate = acs_local_work_disability_signal_gate(frame, receipt=_receipt(staged))
    assert (
        f"{ASEC_PUF_DONOR_SPINE}: is_blind has 1 missing row(s); the reviewed-null "
        "fill would make them False."
    ) in gate.failures

    frame = _with_person(staged.frame, is_disabled=None)
    gate = acs_local_work_disability_signal_gate(frame, receipt=_receipt(staged))
    for spine in (ASEC_PUF_DONOR_SPINE, ACS_2024_1YR_SPINE):
        assert (
            f"{spine}: missing is_disabled; the engine default False removes "
            "every disability exemption."
        ) in gate.failures


def test_gate_requires_native_paired_weeks_at_staging_only(staged) -> None:
    receipt = _receipt(staged)
    without = _with_person(staged.frame, weeks_worked=None)
    gate = acs_local_work_disability_signal_gate(without, receipt=receipt)
    assert not gate.passed
    assert any("missing weeks_worked" in failure for failure in gate.failures)
    finalize = acs_local_work_disability_signal_gate(
        without, receipt=receipt, require_weeks_worked=False
    )
    assert finalize.passed, finalize.failures
    assert finalize.details["weeks_worked"]["evaluated"] is False
    assert WEEKS_WORKED_EXPORT_BLOCKER in finalize.details["weeks_worked"]["reason"]

    changed = list(_EXPECTED_WEEKS)
    changed[1] = 30.0
    gate = acs_local_work_disability_signal_gate(
        _replace_acs(staged.frame, "weeks_worked", changed), receipt=receipt
    )
    assert (
        f"{ACS_2024_1YR_SPINE}: 1 person(s) carry a weeks_worked that differs "
        "from their native WKWN."
    ) in gate.failures

    hours = staged.frame.table("person")["weekly_hours_worked_before_lsr"]
    acs_hours = hours.to_numpy(copy=True)[_acs_mask(staged.frame)]
    acs_hours[0] = 0.0
    gate = acs_local_work_disability_signal_gate(
        _replace_acs(staged.frame, "weekly_hours_worked_before_lsr", acs_hours),
        receipt=receipt,
    )
    assert (
        f"{ACS_2024_1YR_SPINE}: 1 person(s) have weeks worked without usual "
        "hours, or usual hours without weeks worked."
    ) in gate.failures


def test_gate_refuses_frames_without_origin_tags(staged) -> None:
    frame = _with_person(staged.frame, **{TAG: None})
    gate = acs_local_work_disability_signal_gate(frame, receipt=_receipt(staged))
    assert gate.failures == (f"Missing person origin tags: {TAG}.",)
