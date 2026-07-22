"""US child-disability stage tests (populace #453)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import populace.build.us_runtime.child_disability as child_disability_module
from populace.build.us_runtime import (
    SIPP_CHILD_DISABILITY_SOURCE_COLUMNS,
    SSA_SSI_AGE_0_4_CASELOAD_TARGET,
    US_CHILD_DISABILITY_AGE_0_4_SHARE_BAND,
    US_CHILD_DISABILITY_AGE_0_4_TARGET_RATE,
    US_CHILD_DISABILITY_AGE_5_14_SHARE_BAND,
    US_CHILD_DISABILITY_STAGE_NAME,
    US_STAGE_NAMES,
    load_sipp_2023_child_disability_donor,
    resolve_sipp_2023_child_disability_donor,
    us_child_disability_signal_gate,
    us_child_disability_stage_spec,
    us_child_disability_summary,
    with_us_child_disability_inputs,
)
from populace.frame import US_SCHEMA, Frame, WeightKind, Weights

TIME_PERIOD = 2024


@pytest.fixture
def sipp_child_disability_donor() -> pd.DataFrame:
    """Small SIPP-shaped model donor with an 11.1% any-item rate."""

    row = np.arange(900)
    return pd.DataFrame(
        {
            "age": 5 + (row % 10),
            "is_female": (row % 2) == 0,
            "household_income_proxy": (row % 30) * 5_000.0,
            "is_disabled": (row % 9) == 0,
            "sipp_weight": 1.0 + (row % 3) * 0.1,
        }
    )


def _frame(
    *,
    n_age_0_4: int = 4_000,
    n_age_5_14: int = 4_000,
    adult_ages: tuple[int, ...] = (15, 17, 40, 70),
) -> Frame:
    ages = np.concatenate(
        [
            np.arange(n_age_0_4) % 5,
            5 + np.arange(n_age_5_14) % 10,
            np.asarray(adult_ages, dtype=np.int64),
        ]
    )
    n = len(ages)
    ids = np.arange(1, n + 1, dtype=np.int64)
    households = ids.copy()
    person = pd.DataFrame(
        {
            "person_id": ids,
            "person_source_id": [f"source-{value}" for value in ids],
            "person_household_id": households,
            "person_tax_unit_id": households + 10_000,
            "person_spm_unit_id": households + 20_000,
            "person_family_id": households + 30_000,
            "person_marital_unit_id": households + 40_000,
            "age": ages,
            "is_female": (ids % 2) == 0,
            "employment_income_before_lsr": (ids % 40) * 2_500.0,
            "is_disabled": False,
            "adult_payload": [f"row-{value}" for value in ids],
        }
    )
    # Adult values deliberately vary; the child stage must preserve them.
    adult_mask = person["age"] >= 15
    person.loc[adult_mask, "is_disabled"] = [True, False, True, False]
    tables = {
        "person": person,
        "household": pd.DataFrame({"household_id": households}),
        "tax_unit": pd.DataFrame({"tax_unit_id": households + 10_000}),
        "spm_unit": pd.DataFrame({"spm_unit_id": households + 20_000}),
        "family": pd.DataFrame({"family_id": households + 30_000}),
        "marital_unit": pd.DataFrame({"marital_unit_id": households + 40_000}),
    }
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                # Nonuniform weights make row-share implementations fail the
                # weighted calibration and summary assertions below.
                values=np.where(ids % 2 == 0, 3.0, 1.0).astype(np.float64),
                kind=WeightKind.DESIGN,
            )
        },
    )


def _run(frame: Frame, donor: pd.DataFrame, *, seed: int = 453) -> Frame:
    return with_us_child_disability_inputs(
        frame,
        seed=seed,
        time_period=TIME_PERIOD,
        sipp_donor=donor,
    )


def _replace_person(frame: Frame, person: pd.DataFrame) -> Frame:
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["person"] = person
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
    )


def test_manifest_declares_child_disability_stage() -> None:
    spec = us_child_disability_stage_spec()
    assert spec.stage == US_CHILD_DISABILITY_STAGE_NAME
    assert spec.grain == "person"
    assert spec.outputs == ("is_disabled",)
    assert [operation.kind for operation in spec.operations] == [
        "read_table",
        "fit_weighted_logistic",
        "assign_binary_from_rate",
    ]
    assert US_STAGE_NAMES.index(US_CHILD_DISABILITY_STAGE_NAME) == (
        US_STAGE_NAMES.index("eligibility_inputs") + 1
    )


def test_sipp_resolver_prefers_explicit_then_local_then_pinned_fetch(
    tmp_path, monkeypatch
) -> None:
    explicit = tmp_path / "explicit.csv"
    local = tmp_path / "local.csv"
    fetched = tmp_path / "fetched.csv"
    local.touch()
    fetch_calls: list[bool] = []

    monkeypatch.setattr(
        child_disability_module,
        "SIPP_2023_CHILD_DISABILITY_LOCAL_PATH",
        local,
    )
    monkeypatch.setattr(
        child_disability_module,
        "fetch_sipp_2023_voluntary_filing_donor",
        lambda: fetch_calls.append(True) or fetched,
    )

    assert resolve_sipp_2023_child_disability_donor(explicit) == explicit
    assert resolve_sipp_2023_child_disability_donor() == local
    assert fetch_calls == []

    local.unlink()
    assert resolve_sipp_2023_child_disability_donor() == fetched
    assert fetch_calls == [True]


def test_small_sipp_file_builds_household_income_proxy(tmp_path) -> None:
    raw = pd.DataFrame(
        [
            {
                "SSUID": "h1",
                "ERESIDENCEID": 1,
                "PNUM": 1,
                "MONTHCODE": 12,
                "WPFINWGT": 2.0,
                "TAGE": 35,
                "ESEX": 2,
                "TPTOTINC": 1_000,
                "RSSI_YRYN": 2,
                "ECOGNIT": 2,
                "EHEARING": 2,
                "ESEEING": 2,
                "EAMBULAT": 2,
                "ESELFCARE": 2,
            },
            {
                "SSUID": "h1",
                "ERESIDENCEID": 1,
                "PNUM": 2,
                "MONTHCODE": 12,
                "WPFINWGT": 2.0,
                "TAGE": 8,
                "ESEX": 1,
                "TPTOTINC": np.nan,
                "RSSI_YRYN": 2,
                "ECOGNIT": 1,
                "EHEARING": 2,
                "ESEEING": 2,
                "EAMBULAT": 2,
                "ESELFCARE": 2,
            },
            {
                "SSUID": "h2",
                "ERESIDENCEID": 1,
                "PNUM": 1,
                "MONTHCODE": 12,
                "WPFINWGT": 1.0,
                "TAGE": 12,
                "ESEX": 2,
                "TPTOTINC": 500,
                "RSSI_YRYN": 1,
                "ECOGNIT": 2,
                "EHEARING": 2,
                "ESEEING": 2,
                "EAMBULAT": 2,
                "ESELFCARE": 2,
            },
            {
                "SSUID": "h2",
                "ERESIDENCEID": 1,
                "PNUM": 1,
                "MONTHCODE": 11,
                "WPFINWGT": 1.0,
                "TAGE": 12,
                "ESEX": 2,
                "TPTOTINC": 99_999,
                "RSSI_YRYN": 1,
                "ECOGNIT": 1,
                "EHEARING": 1,
                "ESEEING": 1,
                "EAMBULAT": 1,
                "ESELFCARE": 1,
            },
            {
                "SSUID": "h3",
                "ERESIDENCEID": 1,
                "PNUM": 1,
                "MONTHCODE": 12,
                "WPFINWGT": 0.0,
                "TAGE": 9,
                "ESEX": 1,
                "TPTOTINC": np.nan,
                "RSSI_YRYN": 2,
                "ECOGNIT": 1,
                "EHEARING": 2,
                "ESEEING": 2,
                "EAMBULAT": 2,
                "ESELFCARE": 2,
            },
        ],
        columns=SIPP_CHILD_DISABILITY_SOURCE_COLUMNS,
    )
    path = tmp_path / "pu2023.csv"
    raw.to_csv(path, sep="|", index=False)

    donor = load_sipp_2023_child_disability_donor(
        path,
        expected_sha256=None,
        expected_size_bytes=None,
        chunksize=2,
    )

    assert donor["age"].tolist() == [8.0, 12.0]
    assert donor["is_disabled"].tolist() == [True, False]
    assert donor.attrs["source_audit"]["age_5_14_rows"] == 3
    # The adult in h1 supplies the proxy.  The child-only h2 residence remains
    # zero even when the child reports TPTOTINC, preventing SSI label leakage;
    # the January row must not enter either residence aggregate.
    assert donor["household_income_proxy"].tolist() == [12_000.0, 0.0]


def test_child_disability_is_deterministic_under_seed(
    sipp_child_disability_donor: pd.DataFrame,
) -> None:
    frame = _frame()
    first = _run(frame, sipp_child_disability_donor, seed=17)
    second = _run(frame, sipp_child_disability_donor, seed=17)

    np.testing.assert_array_equal(
        first.table("person")["is_disabled"],
        second.table("person")["is_disabled"],
    )


def test_classifier_preserves_a_strong_sex_gradient() -> None:
    row = np.arange(4_000)
    within_sex = row // 2
    female = (row % 2) == 0
    donor = pd.DataFrame(
        {
            "age": 10.0,
            "is_female": female,
            "household_income_proxy": 50_000.0,
            # 20% for girls versus 2% for boys, with an 11% marginal rate.
            "is_disabled": np.where(
                female,
                (within_sex % 5) == 0,
                (within_sex % 50) == 0,
            ),
            "sipp_weight": 1.0,
        }
    )
    result = _run(_frame(n_age_5_14=8_000), donor, seed=96)
    children = result.table("person").loc[lambda table: table["age"].between(5, 14)]
    assigned_share = children.groupby("is_female")["is_disabled"].mean()

    assert assigned_share[True] > assigned_share[False] + 0.08


def test_rows_age_15_and_over_are_byte_identical(
    sipp_child_disability_donor: pd.DataFrame,
) -> None:
    frame = _frame()
    before = frame.table("person").copy(deep=True)
    result = _run(frame, sipp_child_disability_donor)
    adult = before["age"] >= 15

    pd.testing.assert_frame_equal(
        result.table("person").loc[adult],
        before.loc[adult],
        check_exact=True,
    )
    assert us_child_disability_signal_gate(result, input_frame=frame).passed


def test_age_5_14_share_lands_inside_sipp_gate_band(
    sipp_child_disability_donor: pd.DataFrame,
) -> None:
    frame = _frame()
    result = _run(frame, sipp_child_disability_donor)
    summary = us_child_disability_summary(result)
    low, high = US_CHILD_DISABILITY_AGE_5_14_SHARE_BAND
    person = result.table("person")
    band = person["age"].between(5, 14)
    weights = result.resolve_weights("person").values
    expected_weighted_share = np.average(
        person.loc[band, "is_disabled"], weights=weights[band]
    )

    assert low <= summary["age_5_14_disabled_share"] <= high
    assert summary["age_5_14_disabled_share"] == pytest.approx(expected_weighted_share)
    gate = us_child_disability_signal_gate(result, input_frame=frame)
    assert gate.passed, gate.failures


def test_age_0_4_share_tracks_explicit_anchor_rate(
    sipp_child_disability_donor: pd.DataFrame,
) -> None:
    result = _run(_frame(), sipp_child_disability_donor)
    share = us_child_disability_summary(result)["age_0_4_disabled_share"]
    low, high = US_CHILD_DISABILITY_AGE_0_4_SHARE_BAND

    assert low <= share <= high
    assert share == pytest.approx(US_CHILD_DISABILITY_AGE_0_4_TARGET_RATE, abs=0.015)
    assert US_CHILD_DISABILITY_AGE_0_4_TARGET_RATE == pytest.approx(
        0.110253387732
        * ((SSA_SSI_AGE_0_4_CASELOAD_TARGET / 17_166_422.199210) / 0.017932560598),
        abs=1e-12,
    )


def test_existing_child_true_is_never_cleared(
    sipp_child_disability_donor: pd.DataFrame,
) -> None:
    frame = _frame(n_age_0_4=20, n_age_5_14=20)
    person = frame.table("person").copy()
    child = person["age"] < 15
    existing_index = person.index[child][::3]
    person.loc[existing_index, "is_disabled"] = True
    seeded = _replace_person(frame, person)

    result = _run(seeded, sipp_child_disability_donor)

    assert result.table("person").loc[existing_index, "is_disabled"].all()


def test_signal_gate_fails_on_poisoned_child_surface(
    sipp_child_disability_donor: pd.DataFrame,
) -> None:
    frame = _frame()
    result = _run(frame, sipp_child_disability_donor)
    poisoned_person = result.table("person").copy()
    poisoned_person.loc[poisoned_person["age"].between(5, 14), "is_disabled"] = False
    poisoned = _replace_person(result, poisoned_person)

    gate = us_child_disability_signal_gate(poisoned, input_frame=frame)

    assert not gate.passed
    assert any("5-14" in failure for failure in gate.failures)


def test_signal_gate_fails_when_an_adult_row_changes(
    sipp_child_disability_donor: pd.DataFrame,
) -> None:
    frame = _frame()
    result = _run(frame, sipp_child_disability_donor)
    poisoned_person = result.table("person").copy()
    adult_index = poisoned_person.index[poisoned_person["age"] >= 15][0]
    poisoned_person.loc[adult_index, "adult_payload"] = "poisoned"
    poisoned = _replace_person(result, poisoned_person)

    gate = us_child_disability_signal_gate(poisoned, input_frame=frame)

    assert not gate.passed
    assert "age 15+: output rows differ from the stage input." in gate.failures
