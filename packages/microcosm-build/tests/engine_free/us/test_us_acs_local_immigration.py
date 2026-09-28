"""ACS-row immigration inputs for the retained ACS local lane (microcosm#1020)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime.acs_local_immigration import (
    ACS_LOCAL_IMMIGRATION_GATE_NAME,
    ACS_UNMAPPED_INDICATORS,
    ENGINE_DEFAULT_YEARS_SINCE_US_ENTRY,
    YEARS_SINCE_US_ENTRY_COLUMN,
    _acs_cps_view,
    _acs_draw_keys,
    acs_local_immigration_signal_gate,
    require_acs_local_immigration_donor,
    with_acs_local_immigration_inputs,
)
from microcosm.build.us_runtime.acs_pums import ACS_2024_1YR_SPINE
from microcosm.build.us_runtime.acs_transfer import ASEC_PUF_DONOR_SPINE
from microcosm.build.us_runtime.base_pool import spine_column
from microcosm.build.us_runtime.immigration import (
    US_IMMIGRATION_REQUIRED_SOURCE_COLUMNS,
    UndocumentedControls,
    _assign_ssn_card_codes,
    _derive_immigration_status,
    _packaged_controls,
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
    return (
        [{} for _ in range(88)]
        + [{"PRCITSHP": 4, "PEINUSYR": 20} for _ in range(5)]
        + [{"PRCITSHP": 5, "PEINUSYR": 24, **lpr} for _ in range(3)]
        + [
            {
                "PRCITSHP": 5,
                "PEINUSYR": 29,
                "ssn_card_type": "NON_CITIZEN_VALID_EAD",
                "immigration_status_str": "LEGAL_PERMANENT_RESIDENT",
            }
        ]
        + [
            {
                "PRCITSHP": 5,
                "PEINUSYR": 24,
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
        {"ESR": 1},
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
    assert blank["PEAFEVER"] == 2
    # US-born rows have no entry year.
    assert np.isnan(entry_year[person["CIT"].to_numpy() == 1]).all()
    assert entry_year[-2] == 2015


def _cps_row(**overrides: object) -> dict[str, object]:
    row = {
        "PRCITSHP": 5,
        "PEINUSYR": 24,
        "PENATVTY": 303,
        "A_AGE": 30,
        "A_MARITL": 7,
        "A_SPOUSE": 0,
        "A_HSCOL": 0,
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
        (_cps_row(PEINUSYR=19, A_AGE=29, WSAL_VAL=20_000.0), 2005.0),  # DACA
        (_cps_row(PEINUSYR=24, A_AGE=40, WSAL_VAL=20_000.0), 2015.0),  # EAD LPR
        (_cps_row(PENATVTY=327, CAID=1), 2015.0),  # Cuban entrant
        (_cps_row(PENATVTY=332, PEINUSYR=5, MCARE=1), 1972.0),  # pre-1980 Haiti
        (_cps_row(PRCITSHP=4, PEINUSYR=20), 2007.0),  # naturalized
        (_cps_row(PRCITSHP=1, PEINUSYR=0), np.nan),  # US-born
    ]
    person = pd.DataFrame([row for row, _ in rows])
    person.insert(0, "person_id", np.arange(1, len(person) + 1))
    arrival = np.asarray([year for _, year in rows])
    controls = UndocumentedControls(
        workers=0.001,
        students=0.001,
        population_anchor=1.0,
        sources={
            "undocumented_workers": "https://example.com/w",
            "undocumented_students": "https://example.com/s",
            "undocumented_population_anchor": "https://example.com/p",
        },
    )
    weights = np.ones(len(person))
    binned = _assign_ssn_card_codes(person, weights, seed=0, controls=controls)
    exact = _assign_ssn_card_codes(
        person.drop(columns=["PEINUSYR"]),
        weights,
        seed=0,
        controls=controls,
        arrival_year=arrival,
        time_period=PERIOD,
    )
    assert binned.tolist() == exact.tolist() == [3, 0, 2, 2, 3, 3, 1, 1]
    assert (
        _derive_immigration_status(person, binned, time_period=PERIOD).tolist()
        == _derive_immigration_status(
            person.drop(columns=["PEINUSYR"]),
            exact,
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


def test_controls_scale_with_the_acs_person_weight_share() -> None:
    frame = _frame()
    result, receipt = with_acs_local_immigration_inputs(
        frame, seed=5, time_period=PERIOD
    )
    weights = _person_weights(frame)
    acs = _acs(frame)
    share = weights[acs].sum() / weights.sum()
    national = _packaged_controls()
    controls = receipt["controls"]
    assert controls["acs_person_weight_share"] == pytest.approx(share)
    assert controls["acs_person_weight_share"] == pytest.approx(2 / 3)
    scaled = controls["scaled"]["undocumented_workers"]
    assert scaled == pytest.approx(national.workers * share)
    assert controls["scaled"]["undocumented_students"] == pytest.approx(
        national.students * share
    )
    person = result.table("person")
    worker = (person["WAGP"].fillna(0) > 0).to_numpy()
    remaining = weights[acs & worker & person["ssn_card_type"].eq("NONE").to_numpy()]
    # The spill stops at the first record that reaches the scaled control.
    assert scaled - weights[acs].max() < remaining.sum() <= scaled
    assert receipt["acs_composition"]["non_citizen_share"] == pytest.approx(0.07)


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
            "PEINUSYR": 29,
            "age": 30.0,
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
    assert asec[YEARS].iloc[-2] == 0.0  # code 29: 2024 arrivals
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
    gate = acs_local_immigration_signal_gate(_filled())
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
    gate = acs_local_immigration_signal_gate(_with_person(frame, person))
    assert not gate.passed
    failures = " ".join(gate.failures)
    for column in LABELS:
        assert f"{ACS_2024_1YR_SPINE}: {column} is constant 'CITIZEN'" in failures
    for spine in (ASEC_PUF_DONOR_SPINE, ACS_2024_1YR_SPINE):
        assert f"{spine}: {YEARS} is the engine default 5 on every row" in failures
    assert "labelled against their measured CIT" in failures
    assert f"{ASEC_PUF_DONOR_SPINE}: ssn_card_type" not in failures


def test_gate_fails_on_missing_cells_and_columns() -> None:
    gate = acs_local_immigration_signal_gate(_frame())
    assert not gate.passed
    failures = " ".join(gate.failures)
    assert f"{ACS_2024_1YR_SPINE}: ssn_card_type has 400 missing row(s)" in failures
    for spine in (ASEC_PUF_DONOR_SPINE, ACS_2024_1YR_SPINE):
        assert f"{spine}: missing {YEARS}" in failures
    person = _filled().table("person").copy()
    person.loc[person.index[0], YEARS] = np.nan
    missing = acs_local_immigration_signal_gate(_with_person(_filled(), person))
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
    contradicted = acs_local_immigration_signal_gate(_with_person(frame, person))
    assert contradicted.failures == (
        f"{ACS_2024_1YR_SPINE}: 1 person(s) are labelled against their measured "
        "CIT citizenship.",
    )
    person = frame.table("person").copy()
    person["CIT"] = person["CIT"].where(~_acs(frame), 5)
    person.loc[_acs(frame), list(LABELS)] = [
        "OTHER_NON_CITIZEN",
        "LEGAL_PERMANENT_RESIDENT",
    ]
    out_of_band = acs_local_immigration_signal_gate(_with_person(frame, person))
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
    gate = acs_local_immigration_signal_gate(_with_person(frame, person))
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
    gate = acs_local_immigration_signal_gate(tenth)
    assert not gate.passed
    assert all(failure.startswith("file: ") for failure in gate.failures)
    assert "published anchor" in " ".join(gate.failures)


def test_gate_fails_on_missing_tags_or_unknown_spines() -> None:
    frame = _filled()
    person = frame.table("person")
    untagged = acs_local_immigration_signal_gate(
        _with_person(frame, person.assign(**{TAG: person[TAG].where(person.index > 0)}))
    )
    assert untagged.failures == (f"Missing person origin tags: {TAG}.",)
    unknown = acs_local_immigration_signal_gate(
        _with_person(
            frame, person.assign(**{TAG: person[TAG].where(person.index > 0, "other")})
        )
    )
    assert "unsupported spine" in " ".join(unknown.failures)
