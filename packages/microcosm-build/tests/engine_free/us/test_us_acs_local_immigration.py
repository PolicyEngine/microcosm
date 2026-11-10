"""ACS-row immigration inputs for the retained ACS local lane (microcosm#1020)."""

from __future__ import annotations

import dataclasses
import functools

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from microcosm.build.us_runtime import acs_local_immigration
from microcosm.build.us_runtime.acs_local_immigration import (
    ACS_LOCAL_IMMIGRATION_CONTROLS_METHOD,
    ACS_LOCAL_IMMIGRATION_GATE_NAME,
    ACS_UNMAPPED_INDICATORS,
    ADJUSTMENT_LAG_YEARS,
    ENGINE_DEFAULT_YEARS_SINCE_US_ENTRY,
    FIVE_YEAR_BAR_STATUSES,
    YEARS_SINCE_US_ENTRY_COLUMN,
    _acs_cps_view,
    _acs_draw_keys,
    _cps_labor_force_status,
    _humanitarian_reconciliation,
    _status_reconciliation,
    acs_local_immigration_receipt_failures,
    acs_local_immigration_signal_gate,
    require_acs_local_immigration_donor,
    with_acs_local_immigration_inputs,
)
from microcosm.build.us_runtime.acs_pums import ACS_2024_1YR_SPINE
from microcosm.build.us_runtime.acs_transfer import ASEC_PUF_DONOR_SPINE
from microcosm.build.us_runtime.base_pool import spine_column
from microcosm.build.us_runtime.immigration import (
    _CPS_LABOR_FORCE_STATUS_CODES,
    US_IMMIGRATION_REQUIRED_SOURCE_COLUMNS,
    HumanitarianDraw,
    ImmigrationControls,
    UndocumentedControls,
    _assign_ssn_card_codes,
    _derive_immigration_status,
    _pew_unauthorized_rows,
    us_immigration_controls,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

YEARS = YEARS_SINCE_US_ENTRY_COLUMN
TAG = spine_column("person")
PERIOD = 2024
LABELS = ("ssn_card_type", "immigration_status_str")

#: A US-born citizen adult with no ACS legal-status indicator.
_ACS_BASE: dict[str, object] = {
    "CIT": 1,
    "YOEP": np.nan,
    "POBP": 6,
    "AGEP": 40,
    "A_MARITL": 7,
    "A_SPOUSE": 0,
    "WAGP": 0.0,
    "SEMP": 0.0,
    "SSP": 0.0,
    "SSIP": 0.0,
    "HINS3": 2,
    "HINS4": 2,
    "HINS5": 2,
    "HINS6": 2,
    "HINS7": 2,
    "COW": np.nan,
    "SCHG": np.nan,
    "ESR": 6,
    "ACS_MIL": 4,
}
_ACS_COLUMNS = (*_ACS_BASE, "SPORDER")
#: A US-born citizen on the donor spine, labelled by the donor release.
_ASEC_BASE: dict[str, object] = {
    "PRCITSHP": 1,
    "PEINUSYR": 0,
    # PENATVTY 57: born in the United States. The humanitarian cohorts and
    # the whole-file gate read the birth country on every row (microcosm#767).
    "PENATVTY": 57,
    "WSAL_VAL": 0.0,
    "SEMP_VAL": 0.0,
    "A_HSCOL": 0,
    "ssn_card_type": "CITIZEN",
    "immigration_status_str": "CITIZEN",
    "age": 40.0,
}
_ASEC_COLUMNS = ("PRCITSHP", "PEINUSYR")


def _noncitizen(**overrides: object) -> dict[str, object]:
    """An ACS non-citizen who arrived in 2015 as an adult, no indicators."""

    return {"CIT": 5, "YOEP": 2015, "POBP": 303, "AGEP": 35, **overrides}


def _acs_population() -> list[dict[str, object]]:
    """400 ACS persons, 7% non-citizen, every non-citizen at SPORDER 1."""

    noncitizens = (
        [
            _noncitizen(**indicator)
            for indicator in (
                {"HINS4": 1},
                {"HINS3": 1},
                {"HINS5": 1},
                {"HINS6": 1},
                {"HINS7": 1},
                {"COW": 5, "WAGP": 30_000.0},
                {"ESR": 4},
                {"ACS_MIL": 2},
            )
        ]
        + [_noncitizen(WAGP=25_000.0, COW=1, ESR=1) for _ in range(14)]
        + [_noncitizen(SCHG=15, AGEP=20, YOEP=2018) for _ in range(2)]
        + [_noncitizen(YOEP=2016) for _ in range(4)]
    )
    citizens = [{"CIT": 4, "YOEP": 2000, "POBP": 303} for _ in range(20)] + [
        {} for _ in range(352)
    ]
    rows: list[dict[str, object]] = []
    # Households of two: non-citizens first (SPORDER 1), then citizens.
    for household in range(200):
        if household < len(noncitizens):
            rows.append(noncitizens[household])
        else:
            rows.append(citizens.pop())
        rows.append(citizens.pop())
    return rows


def _asec_population() -> list[dict[str, object]]:
    """100 donor persons, 7% non-citizen, already labelled by the donor."""

    lpr = {
        "ssn_card_type": "OTHER_NON_CITIZEN",
        "immigration_status_str": "LEGAL_PERMANENT_RESIDENT",
    }
    mexico = {"PENATVTY": 303}
    return (
        [{} for _ in range(88)]
        + [{"PRCITSHP": 4, "PEINUSYR": 20, **mexico} for _ in range(5)]
        + [{"PRCITSHP": 5, "PEINUSYR": 24, **mexico, **lpr} for _ in range(3)]
        + [
            {
                # Code 28 is the 2024 ASEC's last bin (2022-2024).
                "PRCITSHP": 5,
                "PEINUSYR": 28,
                **mexico,
                "ssn_card_type": "NON_CITIZEN_VALID_EAD",
                "immigration_status_str": "LEGAL_PERMANENT_RESIDENT",
            }
        ]
        + [
            # Undocumented workers: the donor already delivers 3.3M of the
            # packaged worker control on the pooled weights.
            {
                "PRCITSHP": 5,
                "PEINUSYR": 24,
                **mexico,
                "WSAL_VAL": 20_000.0,
                "ssn_card_type": "NONE",
                "immigration_status_str": "UNDOCUMENTED",
            }
            for _ in range(3)
        ]
    )


def _frame(
    *,
    asec_rows: list[dict[str, object]] | None = None,
    acs_rows: list[dict[str, object]] | None = None,
    asec_weight: float = 1.1e6,
    acs_weight: float = 0.55e6,
) -> Frame:
    """A pooled frame: one donor person per household, ACS households of two.

    Weights make the file 330M persons, two thirds on the ACS spine, so the
    packaged Pew controls and anchor bind at realistic scale.
    """

    asec_rows = _asec_population() if asec_rows is None else asec_rows
    acs_rows = _acs_population() if acs_rows is None else acs_rows
    records: list[dict[str, object]] = []
    households: list[dict[str, object]] = []
    weights: list[float] = []
    for index, row in enumerate(asec_rows):
        household_id = index + 1
        records.append(
            {
                **{column: np.nan for column in _ACS_COLUMNS},
                **_ASEC_BASE,
                **row,
                TAG: ASEC_PUF_DONOR_SPINE,
                "person_household_id": household_id,
            }
        )
        households.append(
            {
                "household_id": household_id,
                "state_fips": 6,
                "SERIALNO": np.nan,
                spine_column("household"): ASEC_PUF_DONOR_SPINE,
            }
        )
        weights.append(asec_weight)
    for index, row in enumerate(acs_rows):
        household_id = len(asec_rows) + index // 2 + 1
        record = {
            **{column: np.nan for column in _ASEC_BASE},
            **_ACS_BASE,
            **row,
            TAG: ACS_2024_1YR_SPINE,
            "person_household_id": household_id,
            "SPORDER": index % 2 + 1,
        }
        record["age"] = float(record["AGEP"])
        records.append(record)
        if index % 2 == 0:
            households.append(
                {
                    "household_id": household_id,
                    "state_fips": 6 if (index // 2) % 2 == 0 else 36,
                    "SERIALNO": f"2024HU{household_id:07d}",
                    spine_column("household"): ACS_2024_1YR_SPINE,
                }
            )
            weights.append(acs_weight)
    person = pd.DataFrame(records)
    person.insert(0, "person_id", np.arange(1, len(person) + 1))
    household_ids = person["person_household_id"].to_numpy()
    tables = {"person": person, "household": pd.DataFrame(households)}
    for entity in US_SCHEMA.entities:
        if entity in tables:
            continue
        person[f"person_{entity}_id"] = household_ids
        tables[entity] = pd.DataFrame({f"{entity}_id": np.unique(household_ids)})
    person[["ssn_card_type", "immigration_status_str"]] = person[
        ["ssn_card_type", "immigration_status_str"]
    ].astype(object)
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.asarray(weights), WeightKind.CALIBRATED)},
    )


def _with_person(frame: Frame, person: pd.DataFrame) -> Frame:
    return Frame(
        {
            entity: person if entity == "person" else frame.table(entity)
            for entity in frame.entities
        },
        frame.schema,
        {"household": frame.weights_for("household")},
    )


def _with_household(frame: Frame, household: pd.DataFrame) -> Frame:
    return Frame(
        {
            entity: household if entity == "household" else frame.table(entity)
            for entity in frame.entities
        },
        frame.schema,
        {"household": frame.weights_for("household")},
    )


def _acs(frame: Frame) -> np.ndarray:
    return frame.table("person")[TAG].eq(ACS_2024_1YR_SPINE).to_numpy()


def _person_weights(frame: Frame) -> np.ndarray:
    return np.asarray(frame.resolve_weights("person").values, dtype=np.float64)


@functools.cache
def _packaged() -> ImmigrationControls:
    return us_immigration_controls()


def _controls(
    undocumented: UndocumentedControls | None = None,
    humanitarian: tuple[HumanitarianDraw, ...] = (),
) -> ImmigrationControls:
    return ImmigrationControls(
        undocumented=_packaged().undocumented if undocumented is None else undocumented,
        humanitarian=humanitarian,
    )


@pytest.fixture(autouse=True)
def _undocumented_controls_only(monkeypatch) -> None:
    """Run the stage and its gate on the packaged undocumented controls alone.

    The #1020/#1052 cases predate the microcosm#767 humanitarian draws, and
    every foreign-born person in their populations was born in Mexico or
    Cuba/Haiti, outside every draw's origins. The packaged draws would select
    no one and the whole-file stock bands would fail, so these cases run with
    no draws; the humanitarian cases install their own with
    :func:`_install_controls`.
    """

    _install_controls(monkeypatch, _controls())


def _install_controls(monkeypatch, controls: ImmigrationControls) -> None:
    monkeypatch.setattr(
        acs_local_immigration, "us_immigration_controls", lambda: controls
    )


def _filled(frame: Frame | None = None, *, seed: int = 0) -> Frame:
    return with_acs_local_immigration_inputs(
        _frame() if frame is None else frame, seed=seed, time_period=PERIOD
    )[0]


def _with_extra_acs_person(**overrides: object) -> Frame:
    """The default population plus one ACS person alone in a new household."""

    rows = _acs_population() + [_noncitizen(**overrides), {}]
    return _frame(acs_rows=rows)


def test_donor_rows_stay_identical_and_only_missing_cells_change() -> None:
    frame = _frame()
    before = frame.table("person").copy(deep=True)
    result, receipt = with_acs_local_immigration_inputs(
        frame, seed=0, time_period=PERIOD
    )
    after = result.table("person")
    asec = ~_acs(frame)
    for column in LABELS:
        assert after[column].notna().all()
        assert np.array_equal(
            before.loc[asec, column].to_numpy(dtype=object),
            after.loc[asec, column].to_numpy(dtype=object),
        )
    assert after[YEARS].notna().all()
    assert after[YEARS].dtype == np.float64
    # Nothing else in the person table moved.
    pd.testing.assert_frame_equal(
        before.drop(columns=list(LABELS)), after.drop(columns=[*LABELS, YEARS])
    )
    assert receipt["filled_rows"]["immigration_labels"] == int(_acs(frame).sum())
    assert receipt["filled_rows"][YEARS] == {
        ASEC_PUF_DONOR_SPINE: int(asec.sum()),
        ACS_2024_1YR_SPINE: int(_acs(frame).sum()),
    }
    # The input frame is not mutated.
    assert frame.table("person")["ssn_card_type"].isna().sum() == _acs(frame).sum()
    assert YEARS not in frame.table("person")


def test_citizenship_is_measured_from_cit() -> None:
    result = _filled()
    person = result.table("person")
    acs = _acs(result)
    cit = person["CIT"].to_numpy()
    for column in LABELS:
        labelled = person[column].eq("CITIZEN").to_numpy()
        assert (labelled[acs] == np.isin(cit[acs], [1, 2, 3, 4])).all()


@pytest.mark.parametrize(
    "indicator",
    [
        {"HINS3": 1},
        {"HINS4": 1},
        {"HINS5": 1},
        {"HINS6": 1},
        {"HINS7": 1},
        {"COW": 3},
        {"COW": 4},
        {"COW": 5},
        {"ESR": 4},
        {"ESR": 5},
        {"ACS_MIL": 1},
        {"ACS_MIL": 2},
        {"SSP": 1_200.0},
        {"SSIP": 800.0},
        {"YOEP": 1981},
    ],
)
def test_acs_legal_status_indicators_mark_other_non_citizen(indicator) -> None:
    result = _filled(_with_extra_acs_person(**indicator))
    person = result.table("person")
    assert person["ssn_card_type"].iloc[-2] == "OTHER_NON_CITIZEN"
    assert person["immigration_status_str"].iloc[-2] == "LEGAL_PERMANENT_RESIDENT"


@pytest.mark.parametrize(
    "non_indicator",
    [
        {},
        {"ACS_MIL": 3},
        {"ACS_MIL": 4},
        {"ACS_MIL": np.nan},
        {"COW": 1},
        {"COW": 2},
        {"COW": 6},
        {"ESR": np.nan},
        {"SCHG": 14},
        {"YOEP": 1982},
        {"SSP": np.nan, "SSIP": np.nan},
        {"HINS3": 2, "HINS4": 2},
    ],
)
def test_non_indicators_leave_a_non_worker_undocumented(non_indicator) -> None:
    result = _filled(_with_extra_acs_person(**non_indicator))
    person = result.table("person")
    assert person["ssn_card_type"].iloc[-2] == "NONE"
    assert person["immigration_status_str"].iloc[-2] == "UNDOCUMENTED"


@pytest.mark.parametrize("esr", [1, 2, 3])
def test_a_civilian_labor_force_esr_is_a_worker_not_an_indicator(esr) -> None:
    """ESR 1-3 is not the Armed Forces indicator (4/5): the person stays in
    the residual pool, where the kernel counts them as a worker."""

    _, before = with_acs_local_immigration_inputs(_frame(), seed=0, time_period=PERIOD)
    result, after = with_acs_local_immigration_inputs(
        _with_extra_acs_person(ESR=esr), seed=0, time_period=PERIOD
    )
    assert result.table("person")["ssn_card_type"].iloc[-2] in {
        "NONE",
        "NON_CITIZEN_VALID_EAD",
    }
    capacity = [
        receipt["controls"]["statuses"]["undocumented_workers"]["acs_capacity"]
        for receipt in (before, after)
    ]
    assert capacity[1] - capacity[0] == pytest.approx(0.55e6)


def test_cuban_haitian_class_uses_pobp_and_the_exact_entry_year() -> None:
    cuban = _filled(_with_extra_acs_person(POBP=327, HINS4=1)).table("person")
    assert cuban["immigration_status_str"].iloc[-2] == "CUBAN_HAITIAN_ENTRANT"
    haitian_1979 = _filled(_with_extra_acs_person(POBP=332, HINS4=1, YOEP=1979)).table(
        "person"
    )
    assert haitian_1979["immigration_status_str"].iloc[-2] == (
        "LEGAL_PERMANENT_RESIDENT"
    )


def test_cps_view_covers_the_kernel_inputs_and_zeros_unmapped_indicators() -> None:
    frame = _with_extra_acs_person(
        WAGP=np.nan, SEMP=np.nan, COW=np.nan, SCHG=np.nan, ESR=np.nan, ACS_MIL=np.nan
    )
    person = frame.table("person").loc[_acs(frame)]
    view, entry_year = _acs_cps_view(person, time_period=PERIOD)
    # Every raw ASEC field the kernel reads, except the bins an exact entry
    # year replaces.
    assert set(US_IMMIGRATION_REQUIRED_SOURCE_COLUMNS) - set(view) == {"PEINUSYR"}
    for column in ("PEN_SC1", "PEN_SC2", "RESNSS1", "RESNSS2", "SPM_CAPHOUSESUB"):
        assert (view[column] == 0).all()
    assert set(ACS_UNMAPPED_INDICATORS) == {
        "PEN_SC1, PEN_SC2",
        "RESNSS1, RESNSS2",
        "SPM_CAPHOUSESUB",
    }
    # Blanks are out of universe, never an indicator.
    blank = view.iloc[-2]
    assert (blank["WSAL_VAL"], blank["SEMP_VAL"]) == (0.0, 0.0)
    assert (blank["PEIO1COW"], blank["A_HSCOL"], blank["A_MJOCC"]) == (0, 0, 0)
    assert blank["A_LFSR"] == 0
    assert blank["PEAFEVER"] == 2
    # US-born rows have no entry year.
    assert np.isnan(entry_year[person["CIT"].to_numpy() == 1]).all()
    assert entry_year[-2] == 2015


@pytest.mark.parametrize(
    ("esr", "a_lfsr"),
    [(1, 1), (2, 2), (3, 3), (4, 0), (5, 0), (6, 7), (np.nan, 0)],
)
def test_esr_is_the_cps_labor_force_status(esr, a_lfsr) -> None:
    """The kernel's worker test reads A_LFSR; ACS ESR carries the same status."""

    frame = _with_extra_acs_person(ESR=esr, AGEP=35)
    person = frame.table("person").loc[_acs(frame)]
    view, _ = _acs_cps_view(person, time_period=PERIOD)
    assert view["A_LFSR"].iloc[-2] == a_lfsr
    assert "A_LFSR" in acs_local_immigration.ACS_TO_CPS_INDICATOR_MAPPING


def test_an_esr_outside_the_codebook_is_refused() -> None:
    with pytest.raises(ValueError, match="ACS ESR"):
        with_acs_local_immigration_inputs(
            _with_extra_acs_person(ESR=9), seed=0, time_period=PERIOD
        )


def _cps_row(**overrides: object) -> dict[str, object]:
    row = {
        "PRCITSHP": 5,
        "PEINUSYR": 24,
        "PENATVTY": 303,
        "A_AGE": 30,
        "A_MARITL": 7,
        "A_SPOUSE": 0,
        "A_HSCOL": 0,
        # 7: not in the labor force. Workers set 1 (employed, at work).
        "A_LFSR": 7,
        "WSAL_VAL": 0.0,
        "SEMP_VAL": 0.0,
        "MCARE": 2,
        "CAID": 2,
        "IHSFLG": 2,
        "CHAMPVA": 2,
        "MIL": 2,
        "PEN_SC1": 0,
        "PEN_SC2": 0,
        "RESNSS1": 0,
        "RESNSS2": 0,
        "SS_YN": 2,
        "SSI_YN": 2,
        "PEIO1COW": 0,
        "A_MJOCC": 0,
        "PEAFEVER": 2,
        "SPM_CAPHOUSESUB": 0.0,
    }
    row.update(overrides)
    return row


def test_exact_arrival_years_reproduce_the_peinusyr_bins() -> None:
    """The exact-year kernel path agrees with the bins on the bins' own years."""

    rows = [
        (_cps_row(PEINUSYR=5), 1972.0),  # IRCA cohort: documented
        (_cps_row(PEINUSYR=24), 2015.0),  # residual
        # DACA
        (_cps_row(PEINUSYR=19, A_AGE=29, WSAL_VAL=20_000.0, A_LFSR=1), 2005.0),
        # EAD LPR
        (_cps_row(PEINUSYR=24, A_AGE=40, WSAL_VAL=20_000.0, A_LFSR=1), 2015.0),
        (_cps_row(PENATVTY=327, CAID=1), 2015.0),  # Cuban entrant
        (_cps_row(PENATVTY=332, PEINUSYR=5, MCARE=1), 1972.0),  # pre-1980 Haiti
        (_cps_row(PRCITSHP=4, PEINUSYR=20), 2007.0),  # naturalized
        (_cps_row(PRCITSHP=1, PEINUSYR=0), np.nan),  # US-born
    ]
    person = pd.DataFrame([row for row, _ in rows])
    person.insert(0, "person_id", np.arange(1, len(person) + 1))
    arrival = np.asarray([year for _, year in rows])
    controls = _controls(
        UndocumentedControls(
            workers=0.001,
            students=0.001,
            population_anchor=1.0,
            sources={
                "undocumented_workers": "https://example.com/w",
                "undocumented_students": "https://example.com/s",
                "undocumented_population_anchor": "https://example.com/p",
            },
        )
    )
    weights = np.ones(len(person))
    binned, binned_marks = _assign_ssn_card_codes(
        person, weights, seed=0, controls=controls, time_period=PERIOD
    )
    exact, exact_marks = _assign_ssn_card_codes(
        person.drop(columns=["PEINUSYR"]),
        weights,
        seed=0,
        controls=controls,
        arrival_year=arrival,
        time_period=PERIOD,
    )
    assert binned.tolist() == exact.tolist() == [3, 0, 2, 2, 3, 3, 1, 1]
    assert binned_marks.tolist() == exact_marks.tolist() == [""] * len(rows)
    assert (
        _derive_immigration_status(
            person, binned, binned_marks, time_period=PERIOD
        ).tolist()
        == _derive_immigration_status(
            person.drop(columns=["PEINUSYR"]),
            exact,
            exact_marks,
            time_period=PERIOD,
            arrival_year=arrival,
        ).tolist()
        == [
            "LEGAL_PERMANENT_RESIDENT",
            "UNDOCUMENTED",
            "DACA",
            "LEGAL_PERMANENT_RESIDENT",
            "CUBAN_HAITIAN_ENTRANT",
            "LEGAL_PERMANENT_RESIDENT",
            "CITIZEN",
            "CITIZEN",
        ]
    )


def test_draw_keys_are_unique_across_households_sharing_sporder() -> None:
    frame = _frame()
    person = frame.table("person").loc[_acs(frame)]
    keys = _acs_draw_keys(frame.table("household"), person)
    assert keys.is_unique
    assert person["SPORDER"].nunique() == 2
    assert keys.iloc[0] == f"{ACS_2024_1YR_SPINE}:2024HU0000101:1"

    # Every residual ACS worker sits at SPORDER 1. Keyed on SPORDER alone
    # they would share one draw and spill all-or-nothing.
    result = _filled(frame)
    after = result.table("person")
    workers = _acs(result) & (after["WAGP"].fillna(0) > 0) & after["CIT"].eq(5)
    residual = after.loc[workers, "ssn_card_type"]
    assert set(residual) == {"NONE", "NON_CITIZEN_VALID_EAD", "OTHER_NON_CITIZEN"}


def test_acs_rows_target_the_donor_residual() -> None:
    frame = _frame()
    result, receipt = with_acs_local_immigration_inputs(
        frame, seed=5, time_period=PERIOD
    )
    weights = _person_weights(frame)
    acs = _acs(frame)
    national = _packaged().undocumented
    controls = receipt["controls"]
    assert controls["method"] == ACS_LOCAL_IMMIGRATION_CONTROLS_METHOD
    assert controls["acs_person_weight_share"] == pytest.approx(2 / 3)
    assert controls["donor_status_columns"] == {
        "worker": "WSAL_VAL > 0 or SEMP_VAL > 0",
        "student": "A_HSCOL == 2",
    }
    workers = controls["statuses"]["undocumented_workers"]
    assert workers["control"] == national.workers
    assert workers["donor_delivered"] == pytest.approx(3 * 1.1e6)
    assert workers["residual"] == pytest.approx(national.workers - 3 * 1.1e6)
    assert workers["acs_capacity"] == pytest.approx(14 * 0.55e6)
    person = result.table("person")
    # The kernel's worker test on ACS rows: ESR employed or unemployed, 16+.
    worker = (person["ESR"].isin([1, 2, 3]) & (person["AGEP"] >= 16)).to_numpy()
    remaining = weights[acs & worker & _pew_unauthorized_rows(person)]
    assert workers["acs_assigned"] == pytest.approx(remaining.sum())
    # The spill stops at the first record that reaches the residual.
    residual = workers["residual"]
    assert residual - weights[acs].max() < remaining.sum() <= residual
    assert workers["pooled_total"] == pytest.approx(3 * 1.1e6 + remaining.sum())
    assert abs(workers["pooled_total"] - national.workers) < weights[acs].max()
    students = controls["statuses"]["undocumented_students"]
    assert students["donor_delivered"] == 0.0
    assert students["residual"] == national.students
    assert controls["humanitarian"] == {}
    assert acs_local_immigration_receipt_failures(receipt) == []
    assert receipt["acs_composition"]["non_citizen_share"] == pytest.approx(0.07)


def _custom_controls(monkeypatch, *, workers: float, students: float):
    national = _packaged().undocumented
    controls = UndocumentedControls(
        workers=workers,
        students=students,
        population_anchor=national.population_anchor,
        sources=national.sources,
    )
    _install_controls(monkeypatch, _controls(controls))
    return controls


def _residual_review_frame() -> Frame:
    """The default frame with the last donor NONE worker also in college."""

    asec_rows = _asec_population()
    asec_rows[-1] = {**asec_rows[-1], "A_HSCOL": 2}
    return _frame(asec_rows=asec_rows)


def test_pooled_counts_meet_the_controls_not_f_plus_a_times_them(monkeypatch):
    """#1052 review: donor at ``f`` of a control plus ACS at ``A`` pools to
    ``(f + A) x control`` under the old share-scaled rule; the residual pools
    to the control.

    Units of u = 0.55M (one ACS person; a donor person is 2u). Workers:
    control 12u, donor 6u (f = 1/2), ACS share A = 2/3, ACS capacity 14u.
    Students: control 3u, donor 2u, ACS capacity 2u.
    """

    u = 0.55e6
    controls = _custom_controls(monkeypatch, workers=12 * u, students=3 * u)
    frame = _residual_review_frame()
    weights = _person_weights(frame)
    acs = _acs(frame)
    share = weights[acs].sum() / weights.sum()
    assert share == pytest.approx(2 / 3)
    result, receipt = with_acs_local_immigration_inputs(
        frame, seed=3, time_period=PERIOD
    )
    person = result.table("person")
    unauthorized = _pew_unauthorized_rows(person)
    # ACS rows: ESR employed or unemployed at 16+; donor rows carry no A_LFSR,
    # so the earnings proxy.
    worker = (person["ESR"].isin([1, 2, 3]) & (person["AGEP"] >= 16)) | (
        person["WSAL_VAL"].fillna(0) > 0
    )
    student = person["SCHG"].isin([15, 16]) | person["A_HSCOL"].eq(2)
    pooled_workers = weights[unauthorized & worker.to_numpy()].sum()
    pooled_students = weights[unauthorized & student.to_numpy()].sum()
    assert pooled_workers == controls.workers
    assert pooled_students == controls.students
    statuses = receipt["controls"]["statuses"]
    assert statuses["undocumented_workers"] == {
        "control": 12 * u,
        "donor_delivered": 6 * u,
        "donor_share_of_control": 0.5,
        "residual": 6 * u,
        "acs_capacity": 14 * u,
        "acs_assigned": 6 * u,
        "pooled_total": 12 * u,
        "pooled_relative_error": 0.0,
        "donor_excess": 0.0,
        "acs_shortfall": 0.0,
    }
    assert statuses["undocumented_students"]["residual"] == pytest.approx(u)
    assert statuses["undocumented_students"]["acs_assigned"] == pytest.approx(u)
    assert statuses["undocumented_students"]["pooled_total"] == pytest.approx(3 * u)

    # The pre-review rule targeted the ACS rows at A x control.
    acs_person = frame.table("person").loc[acs]
    view, entry_year = _acs_cps_view(acs_person, time_period=PERIOD)
    old, _ = _assign_ssn_card_codes(
        view,
        weights[acs],
        seed=3,
        controls=_controls(
            UndocumentedControls(
                workers=share * controls.workers,
                students=share * controls.students,
                population_anchor=controls.population_anchor,
                sources=controls.sources,
            )
        ),
        arrival_year=entry_year,
        time_period=PERIOD,
        draw_keys=_acs_draw_keys(frame.table("household"), acs_person),
    )
    old_workers = np.isin(view["A_LFSR"].to_numpy(), (1, 2, 3, 4))
    old_acs = weights[acs][(old == 0) & old_workers].sum()
    old_pooled = 6 * u + old_acs
    assert abs(old_pooled - (0.5 + share) * controls.workers) <= u
    assert old_pooled > controls.workers + u


def test_a_donor_over_the_control_leaves_no_acs_residual(monkeypatch) -> None:
    u = 0.55e6
    _custom_controls(monkeypatch, workers=4 * u, students=3 * u)
    result, receipt = with_acs_local_immigration_inputs(
        _residual_review_frame(), seed=3, time_period=PERIOD
    )
    workers = receipt["controls"]["statuses"]["undocumented_workers"]
    assert workers["residual"] == 0.0
    assert workers["donor_excess"] == pytest.approx(2 * u)
    assert workers["acs_assigned"] == 0.0
    person = result.table("person")
    acs = _acs(result)
    acs_workers = acs & person["ESR"].isin([1, 2, 3]).to_numpy()
    assert not _pew_unauthorized_rows(person)[acs_workers].any()
    assert acs_local_immigration_receipt_failures(receipt) == []


def test_a_residual_over_the_acs_capacity_records_the_shortfall(monkeypatch):
    u = 0.55e6
    _custom_controls(monkeypatch, workers=30 * u, students=3 * u)
    _, receipt = with_acs_local_immigration_inputs(
        _residual_review_frame(), seed=3, time_period=PERIOD
    )
    workers = receipt["controls"]["statuses"]["undocumented_workers"]
    assert workers["residual"] == pytest.approx(24 * u)
    assert workers["acs_assigned"] == pytest.approx(14 * u)
    assert workers["acs_shortfall"] == pytest.approx(10 * u)
    assert workers["pooled_relative_error"] == pytest.approx(-10 / 30)


def test_donor_status_columns_fall_back_to_harmonized_inputs() -> None:
    frame = _frame()
    person = frame.table("person").rename(
        columns={
            "WSAL_VAL": "employment_income_before_lsr",
            "SEMP_VAL": "self_employment_income_before_lsr",
        }
    )
    person["is_full_time_college_student"] = person["A_HSCOL"].eq(2)
    person = person.drop(columns="A_HSCOL")
    _, receipt = with_acs_local_immigration_inputs(
        _with_person(frame, person), seed=5, time_period=PERIOD
    )
    controls = receipt["controls"]
    assert controls["donor_status_columns"]["worker"] == (
        "employment_income_before_lsr > 0 or self_employment_income_before_lsr > 0"
    )
    assert controls["donor_status_columns"]["student"].startswith(
        "is_full_time_college_student (full time only"
    )
    assert controls["statuses"]["undocumented_workers"][
        "donor_delivered"
    ] == pytest.approx(3 * 1.1e6)


def test_a_donor_carrying_a_lfsr_is_measured_with_the_kernels_worker_test():
    """A donor with ``A_LFSR`` is measured as the kernel labels: in the labor
    force at 16+, not positive earnings last year."""

    frame = _frame()
    person = frame.table("person").copy()
    asec = ~_acs(frame)
    person["A_LFSR"] = np.where(asec, 7.0, np.nan)
    workers = np.flatnonzero(asec & person["WSAL_VAL"].gt(0).to_numpy())
    assert len(workers) == 3
    # Two employed workers, and one unemployed with no earnings last year:
    # the kernel counts all three; the earnings proxy would count two.
    person.loc[person.index[workers[:2]], "A_LFSR"] = 1.0
    person.loc[person.index[workers[2]], ["A_LFSR", "WSAL_VAL"]] = [3.0, 0.0]
    _, receipt = with_acs_local_immigration_inputs(
        _with_person(frame, person), seed=5, time_period=PERIOD
    )
    controls = receipt["controls"]
    assert controls["donor_status_columns"]["worker"] == (
        "A_LFSR in [1, 2, 3, 4] and age >= 16"
    )
    assert controls["statuses"]["undocumented_workers"][
        "donor_delivered"
    ] == pytest.approx(3 * 1.1e6)


def test_a_donor_without_worker_or_student_columns_is_refused() -> None:
    frame = _frame()
    person = frame.table("person").drop(columns=["WSAL_VAL"])
    with pytest.raises(ValueError, match="worker pair"):
        with_acs_local_immigration_inputs(
            _with_person(frame, person), seed=0, time_period=PERIOD
        )
    donor = _frame(acs_rows=[])
    with pytest.raises(ValueError, match="student column"):
        require_acs_local_immigration_donor(
            _with_person(donor, donor.table("person").drop(columns="A_HSCOL")),
            time_period=PERIOD,
        )


def _status(control: float, donor: float, assigned: float) -> dict[str, float]:
    return {
        "control": control,
        "donor_delivered": donor,
        "residual": max(0.0, control - donor),
        "acs_assigned": assigned,
        "pooled_total": donor + assigned,
    }


@pytest.mark.parametrize(
    ("controls", "match"),
    [
        (None, "controls.method"),
        ({"method": "national_rate_times_acs_person_share"}, "controls.method"),
        ({"method": ACS_LOCAL_IMMIGRATION_CONTROLS_METHOD}, "statuses is missing"),
        (
            {
                "method": ACS_LOCAL_IMMIGRATION_CONTROLS_METHOD,
                "statuses": {
                    "undocumented_workers": {
                        **_status(8.3e6, 3e6, 5e6),
                        "residual": 8.3e6 * 2 / 3,
                    },
                    "undocumented_students": _status(4.08e5, 0.0, 4.08e5),
                },
            },
            "residual",
        ),
        (
            {
                "method": ACS_LOCAL_IMMIGRATION_CONTROLS_METHOD,
                "statuses": {
                    "undocumented_workers": {
                        **_status(8.3e6, 3e6, 5e6),
                        "pooled_total": 5e6,
                    },
                    "undocumented_students": _status(4.08e5, 0.0, 4.08e5),
                },
            },
            "pooled total",
        ),
        (
            {
                "method": ACS_LOCAL_IMMIGRATION_CONTROLS_METHOD,
                "statuses": {"undocumented_workers": _status(8.3e6, 3e6, 5e6)},
            },
            "undocumented_students lacks",
        ),
    ],
    ids=[
        "no-controls",
        "share-scaled",
        "no-statuses",
        "residual-not-residual",
        "pooled-not-sum",
        "missing-status",
    ],
)
def test_receipt_failures_name_a_target_that_is_not_the_residual(controls, match):
    receipt = {"issue": "microcosm#1020"}
    if controls is not None:
        receipt["controls"] = controls
    failures = acs_local_immigration_receipt_failures(receipt)
    assert failures
    assert any(match in failure for failure in failures), failures


def _draw_entry(target: float, donor: float, assigned: float) -> dict[str, float]:
    return {
        "target": target,
        "donor_delivered": donor,
        "residual": max(0.0, target - donor),
        "acs_assigned": assigned,
        "pooled_total": donor + assigned,
    }


def _receipt_controls(humanitarian: object) -> dict[str, object]:
    return {
        "method": ACS_LOCAL_IMMIGRATION_CONTROLS_METHOD,
        "statuses": {
            "undocumented_workers": _status(9.7e6, 3e6, 6.7e6),
            "undocumented_students": _status(4.08e5, 0.0, 4.08e5),
        },
        "humanitarian": humanitarian,
    }


@pytest.mark.parametrize(
    ("humanitarian", "match"),
    [
        (None, "humanitarian is missing"),
        ({"refugee": _draw_entry(2e5, 1e5, 1e5)}, "missing ['tps:venezuela']"),
        (
            {
                "refugee": _draw_entry(2e5, 1e5, 1e5),
                "tps:venezuela": _draw_entry(6e5, 2e5, 4e5),
                "asylee": _draw_entry(1e5, 0.0, 1e5),
            },
            "unexpected ['asylee']",
        ),
        (
            {
                "refugee": {**_draw_entry(2e5, 1e5, 1e5), "residual": 2e5},
                "tps:venezuela": _draw_entry(6e5, 2e5, 4e5),
            },
            "refugee: residual",
        ),
        (
            {
                "refugee": _draw_entry(2e5, 1e5, 1e5),
                "tps:venezuela": {**_draw_entry(6e5, 2e5, 4e5), "pooled_total": 4e5},
            },
            "tps:venezuela: pooled total",
        ),
        (
            {
                "refugee": {"target": 2e5},
                "tps:venezuela": _draw_entry(6e5, 2e5, 4e5),
            },
            "humanitarian.refugee lacks",
        ),
    ],
    ids=[
        "no-humanitarian",
        "missing-draw",
        "unexpected-draw",
        "residual-not-residual",
        "pooled-not-sum",
        "non-numeric",
    ],
)
def test_receipt_failures_name_a_humanitarian_target_that_is_not_the_residual(
    monkeypatch, humanitarian, match
):
    _install_controls(monkeypatch, _controls(humanitarian=_DRAWS))
    controls = _receipt_controls(humanitarian)
    if humanitarian is None:
        del controls["humanitarian"]
    failures = acs_local_immigration_receipt_failures({"controls": controls})
    assert failures
    assert any(match in failure for failure in failures), failures
    valid = _receipt_controls(
        {
            "refugee": _draw_entry(2e5, 1e5, 1e5),
            "tps:venezuela": _draw_entry(6e5, 2e5, 4e5),
        }
    )
    assert acs_local_immigration_receipt_failures({"controls": valid}) == []


# --------------------------------------------------------------------------
# Humanitarian stocks (microcosm#767)
# --------------------------------------------------------------------------

#: One ACS person's weight; a donor person weighs 2U.
U = 0.55e6
_DRAWS = (
    HumanitarianDraw(
        category="refugee",
        origin=None,
        status="REFUGEE",
        target=2.5 * U,
        source="https://example.com/refugee",
    ),
    HumanitarianDraw(
        category="tps",
        origin="venezuela",
        status="TPS",
        target=5 * U,
        source="https://example.com/tps-venezuela",
    ),
)


def _humanitarian_frame() -> Frame:
    """The default frame plus humanitarian origins on both spines.

    The donor already carries one Venezuelan TPS holder and one Congolese
    refugee (2U each). The ACS rows add four Venezuelan-born 2022 arrivals
    with no legal-status indicator (the residual pool, so TPS candidates)
    and two Congolese-born 2023 arrivals on Medicaid (documented, so refugee
    candidates), each with a US-born partner.
    """

    asec_rows = _asec_population() + [
        {
            "PRCITSHP": 5,
            "PEINUSYR": 28,
            "PENATVTY": 373,
            "ssn_card_type": "NON_CITIZEN_VALID_EAD",
            "immigration_status_str": "TPS",
        },
        {
            "PRCITSHP": 5,
            "PEINUSYR": 28,
            "PENATVTY": 412,
            "ssn_card_type": "OTHER_NON_CITIZEN",
            "immigration_status_str": "REFUGEE",
        },
    ]
    acs_rows = _acs_population()
    for _ in range(4):
        acs_rows += [_noncitizen(POBP=373, YOEP=2022), {}]
    for _ in range(2):
        acs_rows += [_noncitizen(POBP=412, YOEP=2023, HINS4=1), {}]
    return _frame(asec_rows=asec_rows, acs_rows=acs_rows)


def _added_acs_rows(person: pd.DataFrame, birth: int) -> pd.DataFrame:
    acs = person[TAG].eq(ACS_2024_1YR_SPINE)
    return person.loc[acs & person["POBP"].eq(birth)]


def test_humanitarian_draws_target_the_donor_residual_per_draw(monkeypatch):
    """Each cited stock: donor delivered plus ACS assigned pools to it.

    TPS (Venezuela): target 5U, donor 2U, so the ACS rows draw 3U of their
    four candidates. Refugee: target 2.5U, donor 2U, so the residual 0.5U
    draws one ACS candidate (U), the first record to reach it.
    """

    _install_controls(monkeypatch, _controls(humanitarian=_DRAWS))
    result, receipt = with_acs_local_immigration_inputs(
        _humanitarian_frame(), seed=7, time_period=PERIOD
    )
    humanitarian = receipt["controls"]["humanitarian"]
    assert set(humanitarian) == {"refugee", "tps:venezuela"}
    assert humanitarian["tps:venezuela"] == {
        "category": "tps",
        "origin": "venezuela",
        "status": "TPS",
        "target": 5 * U,
        "donor_delivered": 2 * U,
        "residual": 3 * U,
        "acs_assigned": 3 * U,
        "pooled_total": 5 * U,
        "pooled_relative": 1.0,
        "donor_excess": 0.0,
        "source": "https://example.com/tps-venezuela",
    }
    refugee = humanitarian["refugee"]
    assert refugee["residual"] == pytest.approx(0.5 * U)
    assert refugee["acs_assigned"] == pytest.approx(U)
    assert refugee["pooled_total"] == pytest.approx(3 * U)
    assert acs_local_immigration_receipt_failures(receipt) == []

    person = result.table("person")
    venezuelan = _added_acs_rows(person, 373)
    tps = venezuelan["immigration_status_str"].eq("TPS")
    assert int(tps.sum()) == 3
    # Residual-pool selections hold an EAD; the one left is undocumented.
    assert venezuelan.loc[tps, "ssn_card_type"].eq("NON_CITIZEN_VALID_EAD").all()
    assert venezuelan.loc[~tps, list(LABELS)].values.tolist() == [
        ["NONE", "UNDOCUMENTED"]
    ]
    congolese = _added_acs_rows(person, 412)
    assert sorted(congolese["immigration_status_str"]) == [
        "LEGAL_PERMANENT_RESIDENT",
        "REFUGEE",
    ]
    assert congolese["ssn_card_type"].eq("OTHER_NON_CITIZEN").all()
    # TPS holders stay inside Pew's universe; refugees do not.
    unauthorized = _pew_unauthorized_rows(person)
    assert unauthorized[person.index.get_indexer(venezuelan.index)].all()
    assert not unauthorized[person.index.get_indexer(congolese.index)].any()

    gate = acs_local_immigration_signal_gate(result, time_period=PERIOD)
    assert gate.passed, gate.failures
    achieved = gate.details["file"]["details"]["humanitarian_draw_achieved"]
    assert achieved["tps:venezuela"]["population"] == pytest.approx(5 * U)
    assert achieved["refugee"]["population"] == pytest.approx(3 * U)


def test_a_donor_over_a_humanitarian_stock_leaves_no_acs_residual(monkeypatch):
    """The ACS rows only add: a donor at 2x the stock fails the whole-file
    band, and the receipt records the excess."""

    over = (
        HumanitarianDraw(
            category="refugee",
            origin=None,
            status="REFUGEE",
            target=U,
            source="https://example.com/refugee",
        ),
        _DRAWS[1],
    )
    _install_controls(monkeypatch, _controls(humanitarian=over))
    result, receipt = with_acs_local_immigration_inputs(
        _humanitarian_frame(), seed=7, time_period=PERIOD
    )
    refugee = receipt["controls"]["humanitarian"]["refugee"]
    assert refugee["residual"] == 0.0
    assert refugee["acs_assigned"] == 0.0
    assert refugee["donor_excess"] == pytest.approx(U)
    assert refugee["pooled_relative"] == pytest.approx(2.0)
    assert acs_local_immigration_receipt_failures(receipt) == []
    person = result.table("person")
    assert (
        not _added_acs_rows(person, 412)["immigration_status_str"].eq("REFUGEE").any()
    )
    gate = acs_local_immigration_signal_gate(result, time_period=PERIOD)
    assert not gate.passed
    assert any(
        failure.startswith("file: refugee: compatible emitted population")
        for failure in gate.failures
    ), gate.failures


def test_the_gate_grades_the_controls_it_is_given(monkeypatch) -> None:
    _install_controls(monkeypatch, _controls(humanitarian=_DRAWS))
    result, _ = with_acs_local_immigration_inputs(
        _humanitarian_frame(), seed=7, time_period=PERIOD
    )
    assert acs_local_immigration_signal_gate(result, time_period=PERIOD).passed
    # Without the draws, every emitted humanitarian status is unexpected.
    bare = acs_local_immigration_signal_gate(
        result, time_period=PERIOD, controls=_controls()
    )
    assert not bare.passed
    assert any("explicit zero target" in failure for failure in bare.failures)


def test_entry_clock_is_an_arrival_proxy_with_a_lag_sensitivity() -> None:
    """#1052 review: LPRs at 5, 7 and 3 years since arrival flip at lags 1+,
    3+ and never; the receipt also counts every other bar-status person."""

    acs_rows = _acs_population() + [
        _noncitizen(YOEP=2019, HINS4=1),
        {},
        _noncitizen(YOEP=2017, HINS4=1),
        {},
        _noncitizen(YOEP=2021, HINS4=1),
        {},
    ]
    frame = _frame(acs_rows=acs_rows)
    result, receipt = with_acs_local_immigration_inputs(
        frame, seed=0, time_period=PERIOD
    )
    clock = receipt["entry_clock"]
    assert clock["label"] == "arrival-based status-duration proxy"
    assert "1613(a)" in clock["caveat"]
    assert "clear the bar early" in clock["caveat"]
    for text in (
        "neither the ACS nor the ASEC donor records the date qualified status",
        "the clock runs from arrival",
        "the policyengine-us 2.2.1 tests that read the clock",
        "Medicaid (CHIP follows it) and Washington TANF read it against a "
        "five-year bar",
        "Washington RCA reads it as a window that closes",
        "wrongly exclude a recently qualified person from RCA",
        "federal SNAP status test reads the status, not this clock",
        "any SNAP effect is indirect",
        "microcosm#1093",
    ):
        assert text in clock["caveat"]
    # microcosm#1052 review: RCA reads the clock as a closing window, so the
    # long arrival clock excludes rather than admits early.
    assert "Washington TANF and RCA" not in clock["caveat"]
    assert "in Washington TANF, as a five-year bar" in clock["engine_readers"]
    assert "Washington RCA, as a closing window" in clock["engine_readers"]
    block = clock["adjustment_lag_sensitivity"]
    assert block["graded"] is False
    assert block["statuses"] == list(FIVE_YEAR_BAR_STATUSES)
    assert block["lags_years"] == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
    assert list(ADJUSTMENT_LAG_YEARS) == block["lags_years"]
    assert set(block["flips_by_lag_years"]) == {
        str(lag) for lag in ADJUSTMENT_LAG_YEARS
    }
    person = result.table("person")
    acs = _acs(result)
    weights = _person_weights(result)
    years = person[YEARS].to_numpy()
    subject = acs & person["immigration_status_str"].isin(FIVE_YEAR_BAR_STATUSES)
    subject = subject.to_numpy()
    assert block["bar_status_weight"] == pytest.approx(weights[subject].sum())
    assert block["within_bar_weight"] == pytest.approx(
        weights[subject & (years < 5)].sum()
    )
    assert block["max_years_since_us_entry"] == pytest.approx(years[subject].max())
    for lag in ADJUSTMENT_LAG_YEARS:
        expected = subject & (years >= 5) & (years < 5 + lag)
        entry = block["flips_by_lag_years"][str(lag)]
        assert entry["persons"] == int(expected.sum())
        assert entry["weight"] == pytest.approx(weights[expected].sum())
        assert entry["cumulative_share_of_bar_status_weight"] == pytest.approx(
            weights[expected].sum() / weights[subject].sum()
        )
    # The three added LPRs: rows -6, -4 and -2 of the person table.
    added = np.flatnonzero(acs)[-6::2]
    assert (
        list(person["immigration_status_str"].iloc[added])
        == ["LEGAL_PERMANENT_RESIDENT"] * 3
    )
    assert list(years[added]) == [5.0, 7.0, 3.0]
    flips = block["flips_by_lag_years"]
    lag_two_minus_one = flips["2"]["persons"] - flips["1"]["persons"]
    lag_three_minus_two = flips["3"]["persons"] - flips["2"]["persons"]
    assert flips["1"]["persons"] >= 1  # the 5-year LPR
    assert lag_two_minus_one == int(((years == 6) & subject).sum())
    assert lag_three_minus_two == int(((years == 7) & subject).sum()) >= 1


def test_lag_sensitivity_counts_a_nine_year_adjustment_lag() -> None:
    """#1052 review A2 (round 4): her example.

    An LPR who arrived in 2014 and got qualified status in 2023 has 10
    modeled years in 2024 but one qualifying year. Lags of one to five years
    miss them; every lag of six years or more counts them, and the
    cumulative share never falls as the lag grows.
    """

    acs_rows = _acs_population() + [_noncitizen(YOEP=2014, HINS4=1), {}]
    result, receipt = with_acs_local_immigration_inputs(
        _frame(acs_rows=acs_rows), seed=0, time_period=PERIOD
    )
    person = result.table("person")
    added = np.flatnonzero(_acs(result))[-2]
    assert person["immigration_status_str"].iloc[added] == "LEGAL_PERMANENT_RESIDENT"
    assert person[YEARS].iloc[added] == 10.0
    weights = _person_weights(result)
    years = person[YEARS].to_numpy()
    subject = (
        _acs(result) & person["immigration_status_str"].isin(FIVE_YEAR_BAR_STATUSES)
    ).to_numpy()
    ten = subject & (years == 10.0)
    assert ten[added]
    block = receipt["entry_clock"]["adjustment_lag_sensitivity"]
    assert block["max_years_since_us_entry"] >= 10.0
    flips = block["flips_by_lag_years"]
    for lag in ADJUSTMENT_LAG_YEARS:
        flipped = subject & (years >= 5) & (years < 5 + lag)
        assert bool(flipped[added]) == (lag >= 6)
        assert flips[str(lag)]["persons"] == int(flipped.sum())
    shares = [
        flips[str(lag)]["cumulative_share_of_bar_status_weight"]
        for lag in ADJUSTMENT_LAG_YEARS
    ]
    assert shares == sorted(shares)
    assert flips["6"]["weight"] - flips["5"]["weight"] == pytest.approx(
        weights[subject & (years == 10.0)].sum()
    )


def test_assignment_is_deterministic_in_the_seed() -> None:
    first, first_receipt = with_acs_local_immigration_inputs(
        _frame(), seed=11, time_period=PERIOD
    )
    again, again_receipt = with_acs_local_immigration_inputs(
        _frame(), seed=11, time_period=PERIOD
    )
    other_receipts = {
        with_acs_local_immigration_inputs(_frame(), seed=seed, time_period=PERIOD)[1][
            "assigned_sha256"
        ]
        for seed in (12, 13, 14)
    }
    assert first_receipt == again_receipt
    for column in (*LABELS, YEARS):
        assert first.table("person")[column].equals(again.table("person")[column])
    assert first_receipt["assigned_sha256"] not in other_receipts


def test_years_since_us_entry_on_both_spines() -> None:
    asec_rows = _asec_population() + [
        {
            "PRCITSHP": 5,
            "PEINUSYR": 28,
            "PENATVTY": 303,
            "age": 30.0,
            "ssn_card_type": "OTHER_NON_CITIZEN",
            "immigration_status_str": "LEGAL_PERMANENT_RESIDENT",
        },
        {
            "PRCITSHP": 5,
            "PEINUSYR": 29,
            "PENATVTY": 303,
            "age": 1.0,
            "ssn_card_type": "OTHER_NON_CITIZEN",
            "immigration_status_str": "LEGAL_PERMANENT_RESIDENT",
        },
        {"PRCITSHP": 2, "PEINUSYR": 0, "age": 61.0},
    ]
    acs_rows = _acs_population() + [
        _noncitizen(YOEP=2024, AGEP=1),
        {"CIT": 3, "AGEP": 12, "YOEP": 2020},
    ]
    result = _filled(_frame(asec_rows=asec_rows, acs_rows=acs_rows))
    person = result.table("person")
    years = person[YEARS].to_numpy()
    assert not np.isnan(years).any()
    by_spine = {
        spine: person[TAG].eq(spine).to_numpy()
        for spine in (ASEC_PUF_DONOR_SPINE, ACS_2024_1YR_SPINE)
    }
    asec = person.loc[by_spine[ASEC_PUF_DONOR_SPINE]]
    acs = person.loc[by_spine[ACS_2024_1YR_SPINE]]
    # US-born: age. Foreign-born: period minus the entry (or bin) year.
    native = asec["PRCITSHP"].isin([1, 2, 3])
    assert (asec.loc[native, YEARS] == asec.loc[native, "age"]).all()
    assert (asec.loc[asec["PEINUSYR"].eq(24), YEARS] == PERIOD - 2015).all()
    assert (asec.loc[asec["PEINUSYR"].eq(20), YEARS] == PERIOD - 2007).all()
    # Code 28, the 2024 ASEC's last bin (2022-2024), reads as its 2023
    # midpoint. Code 29, which only the 2026 ASEC uses (2024-2026), reads as
    # the kernel's 2024: a pooled default-pool donor is never refused for it.
    assert asec[YEARS].iloc[-3] == 1.0
    assert asec[YEARS].iloc[-2] == 0.0
    assert asec[YEARS].iloc[-1] == 61.0  # born in Puerto Rico: age
    acs_native = acs["CIT"].isin([1, 2, 3])
    assert (acs.loc[acs_native, YEARS] == acs.loc[acs_native, "AGEP"]).all()
    assert acs[YEARS].iloc[-1] == 12.0  # born abroad to US parents: age
    foreign = ~acs_native
    assert (
        acs.loc[foreign, YEARS] == PERIOD - acs.loc[foreign, "YOEP"].astype(float)
    ).all()
    assert acs[YEARS].iloc[-2] == 0.0
    assert not (years == ENGINE_DEFAULT_YEARS_SINCE_US_ENTRY).all()


def test_stored_cells_are_kept() -> None:
    frame = _frame()
    person = frame.table("person").copy()
    acs_row = int(np.flatnonzero(_acs(frame))[-1])  # a US-born ACS citizen
    person.loc[person.index[acs_row], list(LABELS)] = [
        "OTHER_NON_CITIZEN",
        "LEGAL_PERMANENT_RESIDENT",
    ]
    person[YEARS] = np.nan
    person.loc[person.index[0], YEARS] = 3.0
    result, receipt = with_acs_local_immigration_inputs(
        _with_person(frame, person), seed=0, time_period=PERIOD
    )
    after = result.table("person")
    assert after["ssn_card_type"].iloc[acs_row] == "OTHER_NON_CITIZEN"
    assert after[YEARS].iloc[0] == 3.0
    assert receipt["preserved_acs_label_rows"] == 1
    assert receipt["filled_rows"]["immigration_labels"] == int(_acs(frame).sum()) - 1


def test_a_filled_frame_passes_through_unchanged() -> None:
    filled, receipt = with_acs_local_immigration_inputs(
        _frame(), seed=0, time_period=PERIOD
    )
    again, again_receipt = with_acs_local_immigration_inputs(
        filled, seed=0, time_period=PERIOD
    )
    assert again is filled
    assert again_receipt["filled_rows"]["immigration_labels"] == 0
    assert again_receipt["assigned_sha256"] == receipt["assigned_sha256"]


@pytest.mark.parametrize(
    "column",
    [
        "CIT",
        "YOEP",
        "HINS4",
        "ACS_MIL",
        "SPORDER",
        "immigration_status_str",
        "age",
        TAG,
    ],
)
def test_missing_person_inputs_are_refused(column: str) -> None:
    frame = _frame()
    with pytest.raises(ValueError, match="ACS local immigration|draws are keyed"):
        with_acs_local_immigration_inputs(
            _with_person(frame, frame.table("person").drop(columns=column)),
            seed=0,
            time_period=PERIOD,
        )


def test_missing_serialno_is_refused() -> None:
    frame = _frame()
    stripped = _with_household(frame, frame.table("household").drop(columns="SERIALNO"))
    with pytest.raises(ValueError, match="SERIALNO"):
        with_acs_local_immigration_inputs(stripped, seed=0, time_period=PERIOD)


@pytest.mark.parametrize(
    ("acs_row", "asec_row", "match"),
    [
        (_noncitizen(YOEP=np.nan), None, "YOEP"),
        (_noncitizen(YOEP=2025), None, "YOEP"),
        ({"CIT": 7}, None, "CIT"),
        (None, {"PRCITSHP": 5, "PEINUSYR": 0}, "PEINUSYR"),
        (None, {"PRCITSHP": np.nan}, "PRCITSHP"),
    ],
)
def test_unknown_citizenship_or_entry_year_is_refused(acs_row, asec_row, match):
    acs_rows = _acs_population() + ([acs_row, {}] if acs_row else [])
    asec_rows = _asec_population() + ([asec_row] if asec_row else [])
    with pytest.raises(ValueError, match=match):
        with_acs_local_immigration_inputs(
            _frame(asec_rows=asec_rows, acs_rows=acs_rows),
            seed=0,
            time_period=PERIOD,
        )


def test_a_partial_label_pair_is_refused() -> None:
    frame = _frame()
    person = frame.table("person").copy()
    row = person.index[int(np.flatnonzero(_acs(frame))[0])]
    person.loc[row, "ssn_card_type"] = "CITIZEN"
    with pytest.raises(ValueError, match="one immigration label without the other"):
        with_acs_local_immigration_inputs(
            _with_person(frame, person), seed=0, time_period=PERIOD
        )


def _donor() -> Frame:
    return _frame(acs_rows=[])


def test_donor_preflight_accepts_a_complete_donor() -> None:
    require_acs_local_immigration_donor(_donor(), time_period=PERIOD)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (lambda person: person.drop(columns="PRCITSHP"), "PRCITSHP"),
        (lambda person: person.drop(columns="ssn_card_type"), "ssn_card_type"),
        (
            lambda person: person.assign(
                immigration_status_str=person["immigration_status_str"].where(
                    person.index > 0
                )
            ),
            "missing cells",
        ),
        (
            lambda person: person.assign(
                PEINUSYR=person["PEINUSYR"].where(~person["PRCITSHP"].eq(5), 0)
            ),
            "PEINUSYR",
        ),
        (lambda person: person.drop(columns="PENATVTY"), "PENATVTY"),
    ],
)
def test_donor_preflight_refuses_an_unusable_donor(change, match) -> None:
    donor = _donor()
    with pytest.raises(ValueError, match=match):
        require_acs_local_immigration_donor(
            _with_person(donor, change(donor.table("person").copy())),
            time_period=PERIOD,
        )


# --------------------------------------------------------------------------
# Gate
# --------------------------------------------------------------------------


def test_gate_passes_on_the_filled_surface() -> None:
    gate = acs_local_immigration_signal_gate(_filled(), time_period=PERIOD)
    assert gate.passed, gate.failures
    assert gate.name == ACS_LOCAL_IMMIGRATION_GATE_NAME
    acs = gate.details["per_spine"][ACS_2024_1YR_SPINE]
    assert acs["graded"] is True
    assert acs["cit_disagreements"] == 0
    assert acs["non_citizen_share"] == pytest.approx(0.07)
    assert set(acs["non_citizen_share_by_state"]) == {"06", "36"}
    assert gate.details["per_spine"][ASEC_PUF_DONOR_SPINE]["graded"] is False
    assert gate.details["file"]["passed"] is True


def test_gate_fails_on_the_default_filled_release_signature() -> None:
    """The 2026-09-23 release: every ACS person CITIZEN, every clock 5."""

    frame = _filled()
    acs = _acs(frame)
    person = frame.table("person").copy()
    for column in LABELS:
        person[column] = person[column].where(~acs, "CITIZEN")
    person[YEARS] = ENGINE_DEFAULT_YEARS_SINCE_US_ENTRY
    gate = acs_local_immigration_signal_gate(
        _with_person(frame, person), time_period=PERIOD
    )
    assert not gate.passed
    failures = " ".join(gate.failures)
    for column in LABELS:
        assert f"{ACS_2024_1YR_SPINE}: {column} is constant 'CITIZEN'" in failures
    for spine in (ASEC_PUF_DONOR_SPINE, ACS_2024_1YR_SPINE):
        assert f"{spine}: {YEARS} is the engine default 5 on every row" in failures
    assert "labelled against their measured CIT" in failures
    assert f"{ASEC_PUF_DONOR_SPINE}: ssn_card_type" not in failures


def test_gate_fails_on_missing_cells_and_columns() -> None:
    gate = acs_local_immigration_signal_gate(_frame(), time_period=PERIOD)
    assert not gate.passed
    failures = " ".join(gate.failures)
    assert f"{ACS_2024_1YR_SPINE}: ssn_card_type has 400 missing row(s)" in failures
    for spine in (ASEC_PUF_DONOR_SPINE, ACS_2024_1YR_SPINE):
        assert f"{spine}: missing {YEARS}" in failures
    person = _filled().table("person").copy()
    person.loc[person.index[0], YEARS] = np.nan
    missing = acs_local_immigration_signal_gate(
        _with_person(_filled(), person), time_period=PERIOD
    )
    assert missing.failures == (
        f"{ASEC_PUF_DONOR_SPINE}: {YEARS} has 1 missing row(s).",
    )


def test_gate_fails_when_labels_contradict_cit_or_leave_the_band() -> None:
    frame = _filled()
    acs = np.flatnonzero(_acs(frame))
    person = frame.table("person").copy()
    person.loc[person.index[acs[-1]], list(LABELS)] = [
        "OTHER_NON_CITIZEN",
        "LEGAL_PERMANENT_RESIDENT",
    ]
    contradicted = acs_local_immigration_signal_gate(
        _with_person(frame, person), time_period=PERIOD
    )
    assert contradicted.failures == (
        f"{ACS_2024_1YR_SPINE}: 1 person(s) are labelled against their measured "
        "CIT citizenship.",
        "file: 1 person(s) have emitted citizenship disagreeing with their source "
        "PRCITSHP/CIT evidence.",
    )
    person = frame.table("person").copy()
    person["CIT"] = person["CIT"].where(~_acs(frame), 5)
    person.loc[_acs(frame), list(LABELS)] = [
        "OTHER_NON_CITIZEN",
        "LEGAL_PERMANENT_RESIDENT",
    ]
    out_of_band = acs_local_immigration_signal_gate(
        _with_person(frame, person), time_period=PERIOD
    )
    assert any(
        failure.startswith(f"{ACS_2024_1YR_SPINE}: non-citizen weighted share 1.0000")
        for failure in out_of_band.failures
    )


def test_gate_reports_but_does_not_grade_donor_composition() -> None:
    frame = _filled()
    asec = ~_acs(frame)
    person = frame.table("person").copy()
    # A donor spine far outside the band: 47% non-citizen.
    flipped = np.flatnonzero(asec)[:40]
    person.loc[person.index[flipped], list(LABELS)] = [
        "OTHER_NON_CITIZEN",
        "LEGAL_PERMANENT_RESIDENT",
    ]
    gate = acs_local_immigration_signal_gate(
        _with_person(frame, person), time_period=PERIOD
    )
    donor = gate.details["per_spine"][ASEC_PUF_DONOR_SPINE]
    assert donor["composition"]["non_citizen_share"] == pytest.approx(0.47)
    assert "non_citizen_share" not in donor
    assert not any(
        failure.startswith(ASEC_PUF_DONOR_SPINE) for failure in gate.failures
    )


def test_gate_checks_the_whole_file_anchor() -> None:
    frame = _filled()
    tenth = Frame(
        {entity: frame.table(entity) for entity in frame.entities},
        frame.schema,
        {
            "household": Weights(
                frame.weights_for("household").values / 10, WeightKind.CALIBRATED
            )
        },
    )
    gate = acs_local_immigration_signal_gate(tenth, time_period=PERIOD)
    assert not gate.passed
    assert all(failure.startswith("file: ") for failure in gate.failures)
    assert "published anchor" in " ".join(gate.failures)


def test_gate_fails_on_missing_tags_or_unknown_spines() -> None:
    frame = _filled()
    person = frame.table("person")
    untagged = acs_local_immigration_signal_gate(
        _with_person(
            frame, person.assign(**{TAG: person[TAG].where(person.index > 0)})
        ),
        time_period=PERIOD,
    )
    assert untagged.failures == (f"Missing person origin tags: {TAG}.",)
    unknown = acs_local_immigration_signal_gate(
        _with_person(
            frame, person.assign(**{TAG: person[TAG].where(person.index > 0, "other")})
        ),
        time_period=PERIOD,
    )
    assert "unsupported spine" in " ".join(unknown.failures)


# --------------------------------------------------------------------------
# Properties (the microcosm#767 port of #1052)
# --------------------------------------------------------------------------
#
# Each property restates a mechanism the module states; the *deduced* one
# combines two of them.
#
# - ``_cps_labor_force_status``: every ACS ``ESR`` code maps onto the CPS
#   ``A_LFSR`` codebook, and a blank is code 0 ("a blank (under 16) is code
#   0"); any other value, an infinity included, is refused. Differential:
#   the mapped codes pass the kernel's worker test
#   (``_CPS_LABOR_FORCE_STATUS_CODES``) exactly for ESR 1-3, the civilian
#   labor force.
# - ``_status_reconciliation`` and ``_humanitarian_reconciliation`` against
#   ``acs_local_immigration_receipt_failures`` (differential: producer
#   versus checker): every receipt the producers can write passes the
#   checker, and each humanitarian entry splits ``target - donor`` into a
#   residual and a donor excess, at most one of them nonzero.
# - The stage, for each humanitarian draw (the ACS rows draw toward
#   ``max(0, target - donor-emitted mass)``): the receipt records exactly
#   that residual; the donor's delivered mass does not depend on the
#   target; the ACS rows emitted with the draw's status weigh exactly
#   ``acs_assigned``; a covered stock assigns nothing; a feasible residual
#   is reached and overshot by less than one ACS row (every ACS row weighs
#   U and draw keys are unique, so each tie group is one row); an
#   infeasible one takes every candidate; and (deduced: threshold
#   selection) a larger stock selects a superset of a smaller stock's rows.
#
# Independence limits: the ESR table below transcribes the ACS and CPS
# codebooks, written by the author of the mapping; the stage properties take
# each draw's candidate mass from the stage itself (an exhausting stock),
# and only the fixture test pins that mass to the rows the fixture adds.

_UNIT = settings(max_examples=100, deadline=None, database=None)
_STAGE = settings(max_examples=20, deadline=None, database=None)

#: ACS ``ESR`` to CPS ``A_LFSR``, from the two codebooks: civilian employed
#: at work and with a job keep 1 and 2; unemployed is 3 (looking); the Armed
#: Forces fall outside the CPS labor-force universe (0); not in the labor
#: force is 7.
_CODEBOOK_ESR_TO_A_LFSR = {1: 1, 2: 2, 3: 3, 4: 0, 5: 0, 6: 7}
_CIVILIAN_LABOR_FORCE_ESR = (1, 2, 3)
_MASS = st.integers(0, 10**8).map(float)
_CONTROL = st.integers(1, 10**8).map(float)
#: A humanitarian stock in quarter-person steps of U, so every sum is exact.
_STOCK = st.integers(0, 40).map(lambda quarters: quarters * U / 4)


@_UNIT
@given(
    st.lists(st.sampled_from((*_CODEBOOK_ESR_TO_A_LFSR, None)), min_size=1, max_size=40)
)
def test_every_esr_maps_onto_the_cps_labor_force_codebook(codes) -> None:
    esr = pd.Series([np.nan if code is None else float(code) for code in codes])
    status = _cps_labor_force_status(esr)
    assert status.tolist() == [
        0 if code is None else _CODEBOOK_ESR_TO_A_LFSR[code] for code in codes
    ]
    worker = np.isin(status, _CPS_LABOR_FORCE_STATUS_CODES)
    assert worker.tolist() == [code in _CIVILIAN_LABOR_FORCE_ESR for code in codes]


@_UNIT
@given(
    st.one_of(
        st.integers(-10_000, 10_000).map(float),
        st.floats(allow_nan=False),
    ).filter(lambda value: value not in _CODEBOOK_ESR_TO_A_LFSR)
)
def test_any_other_esr_is_refused(value) -> None:
    with pytest.raises(ValueError, match="ACS ESR must be one of"):
        _cps_labor_force_status(pd.Series([6.0, value]))


@_UNIT
@given(
    workers=st.tuples(_CONTROL, _MASS, _MASS, _MASS),
    students=st.tuples(_CONTROL, _MASS, _MASS, _MASS),
    draws=st.tuples(*(st.tuples(_MASS, _MASS, _MASS) for _ in _DRAWS)),
)
def test_every_receipt_the_producers_write_passes_the_checker(
    workers, students, draws
) -> None:
    humanitarian = {}
    for draw, (target, donor, assigned) in zip(_DRAWS, draws, strict=True):
        entry = _humanitarian_reconciliation(
            dataclasses.replace(draw, target=target), donor, assigned
        )
        assert entry["residual"] - entry["donor_excess"] == target - donor
        assert min(entry["residual"], entry["donor_excess"]) == 0.0
        humanitarian[draw.label] = entry
    receipt = {
        "controls": {
            "method": ACS_LOCAL_IMMIGRATION_CONTROLS_METHOD,
            "statuses": {
                "undocumented_workers": _status_reconciliation(*workers),
                "undocumented_students": _status_reconciliation(*students),
            },
            "humanitarian": humanitarian,
        }
    }
    with pytest.MonkeyPatch.context() as patch:
        _install_controls(patch, _controls(humanitarian=_DRAWS))
        assert acs_local_immigration_receipt_failures(receipt) == []


def _stage_draws(refugee: float, tps: float) -> tuple[HumanitarianDraw, ...]:
    return (
        dataclasses.replace(_DRAWS[0], target=refugee),
        dataclasses.replace(_DRAWS[1], target=tps),
    )


def _run_humanitarian(
    draws: tuple[HumanitarianDraw, ...], *, seed: int
) -> tuple[Frame, dict, list[str]]:
    with pytest.MonkeyPatch.context() as patch:
        _install_controls(patch, _controls(humanitarian=draws))
        result, receipt = with_acs_local_immigration_inputs(
            _humanitarian_frame(), seed=seed, time_period=PERIOD
        )
        failures = acs_local_immigration_receipt_failures(receipt)
    return result, receipt, failures


@functools.cache
def _humanitarian_capacity() -> dict[str, float]:
    """Each draw's ACS candidate mass: what an exhausting stock assigns."""

    _, receipt, _ = _run_humanitarian(_stage_draws(100 * U, 100 * U), seed=0)
    return {
        label: entry["acs_assigned"]
        for label, entry in receipt["controls"]["humanitarian"].items()
    }


def test_an_exhausting_stock_takes_every_added_acs_candidate() -> None:
    """The four Venezuelan and two Congolese ACS rows are the only candidates."""

    assert _humanitarian_capacity() == {
        "refugee": pytest.approx(2 * U),
        "tps:venezuela": pytest.approx(4 * U),
    }


@_STAGE
@given(refugee=_STOCK, tps=_STOCK, seed=st.integers(0, 2**16))
def test_each_humanitarian_draw_assigns_its_donor_residual(refugee, tps, seed) -> None:
    draws = _stage_draws(refugee, tps)
    result, receipt, failures = _run_humanitarian(draws, seed=seed)
    assert failures == []
    capacity = _humanitarian_capacity()
    weights = _person_weights(result)
    acs = _acs(result)
    status = result.table("person")["immigration_status_str"].to_numpy()
    for draw in draws:
        entry = receipt["controls"]["humanitarian"][draw.label]
        assert entry["target"] == draw.target
        assert entry["donor_delivered"] == pytest.approx(2 * U)
        residual = max(0.0, draw.target - entry["donor_delivered"])
        assert entry["residual"] == residual
        assigned = entry["acs_assigned"]
        assert entry["pooled_total"] == entry["donor_delivered"] + assigned
        emitted = float(weights[acs & (status == draw.status)].sum())
        assert assigned == pytest.approx(emitted)
        if residual == 0.0:
            assert assigned == 0.0
        elif residual <= capacity[draw.label]:
            assert residual <= assigned < residual + U
        else:
            assert assigned == pytest.approx(capacity[draw.label])


@_STAGE
@given(low=_STOCK, high=_STOCK, seed=st.integers(0, 2**16))
def test_a_larger_stock_selects_a_superset_of_acs_rows(low, high, seed) -> None:
    low, high = sorted((low, high))
    small, _, _ = _run_humanitarian(_stage_draws(low, low), seed=seed)
    large, _, _ = _run_humanitarian(_stage_draws(high, high), seed=seed)
    acs = _acs(small)
    for draw in _DRAWS:
        before = small.table("person")["immigration_status_str"].eq(draw.status)
        after = large.table("person")["immigration_status_str"].eq(draw.status)
        assert not (acs & before.to_numpy() & ~after.to_numpy()).any()
