"""The local hours repair must expose #765 without changing observations."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime.acs_local_hours import (
    ACS_UNDER15_ZERO_POLICY,
    acs_local_hours_signal_gate,
    acs_local_transfer_target_families,
    complete_acs_local_under15_hours,
    prepare_acs_local_hours_donor,
)
from microcosm.build.us_runtime.acs_transfer import (
    declared_acs_transfer_target_families,
)
from microcosm.build.us_runtime.support_provenance import (
    support_channel_column,
    support_clone_index_column,
    support_source_id_column,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights


def _frame(
    weekly=None, *, spines=None, raw=None, role="asec", columns=None, n=8
) -> Frame:
    ids = np.arange(1, n + 1)
    person = pd.DataFrame({"person_id": ids})
    tables = {"person": person}
    for entity in US_SCHEMA.entities:
        if entity == "person":
            continue
        person[f"person_{entity}_id"] = ids
        tables[entity] = pd.DataFrame({f"{entity}_id": ids})
    person["HRSWK"] = np.resize([40, 0, 38, 0], n)
    person["A_HRS1"] = np.resize([40, 0, 35, 0], n)
    person["WKSWORK"] = np.resize([52, 0, 50, 0], n)
    person["hours_worked_last_week"] = person["A_HRS1"]
    person["age"] = np.resize([40, 30, 35, 32], n)
    person["is_female"] = np.resize([False, True], n)
    tables["household"]["state_fips"] = 6
    if role is not None:
        for entity, table in tables.items():
            table[support_channel_column(entity)] = role
            table[support_source_id_column(entity)] = ids
            table[support_clone_index_column(entity)] = (
                table[support_channel_column(entity)].eq("puf_tax_detail").astype(int)
            )
    for column, value in (raw or {}).items():
        if value is None:
            person = person.drop(columns=column)
        else:
            person[column] = person[column].astype(float)
            person.loc[0, column] = value
    tables["person"] = person
    if weekly is not None:
        person["weekly_hours_worked_before_lsr"] = weekly
    if spines is not None:
        person["person_spine"] = spines
    for column, values in (columns or {}).items():
        person[column] = values
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.ones(n), WeightKind.CALIBRATED)},
    )


def test_local_plan_adds_usual_hours_without_changing_shared_plan() -> None:
    original = declared_acs_transfer_target_families()
    local = acs_local_transfer_target_families()
    assert local["person"]["source_operator_hours_worked"] == (
        "weekly_hours_worked_before_lsr",
    )
    assert declared_acs_transfer_target_families() == original
    assert "weeks_worked" not in {
        target for family in local["person"].values() for target in family
    }


def test_raw_donor_hours_are_derived_before_the_local_transfer() -> None:
    before = _frame()
    result, _, receipt = prepare_acs_local_hours_donor(before, seed=3, period=2024)
    assert result.table("person")["weekly_hours_worked_before_lsr"].tolist() == (
        before.table("person")["HRSWK"].tolist()
    )
    assert "weekly_hours_worked_before_lsr" not in before.table("person")
    for entity in before.entities:
        pd.testing.assert_frame_equal(
            result.table(entity)[before.table(entity).columns], before.table(entity)
        )
    np.testing.assert_array_equal(
        before.weights_for("household").values,
        result.weights_for("household").values,
    )
    assert "weeks_worked" not in result.table("person")
    assert receipt["filled_rows"] == 8


def test_source_observed_forty_is_preserved_alongside_missing_cells() -> None:
    before = _frame([40, np.nan, 38, 0, 40, 0, 38, 0])
    result, _, receipt = prepare_acs_local_hours_donor(before, seed=3, period=2024)
    assert result.table("person")["weekly_hours_worked_before_lsr"].tolist() == [
        40,
        0,
        38,
        0,
        40,
        0,
        38,
        0,
    ]
    assert receipt["filled_rows"] == 1
    assert receipt["preserved_observed_rows"] == 7


@pytest.mark.parametrize("default", [0, 40])
def test_contradictory_stored_donor_defaults_are_not_silently_overwritten(
    default,
) -> None:
    before = _frame([default] * 8)
    with pytest.raises(ValueError, match="contradict"):
        prepare_acs_local_hours_donor(before, seed=3, period=2024)
    assert before.table("person")["weekly_hours_worked_before_lsr"].eq(default).all()


@pytest.mark.parametrize("bad", [None, np.nan, float("inf"), -1])
def test_incomplete_raw_donor_is_rejected_even_when_stored_hours_have_signal(
    bad,
) -> None:
    frame = _frame([40, 0, 38, 0, 40, 0, 38, 0], raw={"HRSWK": bad})
    with pytest.raises(ValueError, match="HRSWK"):
        prepare_acs_local_hours_donor(frame, seed=3, period=2024)


@pytest.mark.parametrize(
    "column,value",
    [
        ("HRSWK", 999),
        ("HRSWK", 40.5),
        ("A_HRS1", -4),
        ("A_HRS1", 100),
        ("WKSWORK", 53),
    ],
)
def test_raw_donor_codes_respect_the_census_domain(column, value) -> None:
    with pytest.raises(ValueError, match=column):
        prepare_acs_local_hours_donor(_frame(raw={column: value}), seed=3, period=2024)


def test_declared_reference_week_niu_does_not_replace_usual_hours() -> None:
    frame = _frame(raw={"A_HRS1": -1})
    result, _, _ = prepare_acs_local_hours_donor(frame, seed=3, period=2024)
    assert result.table("person")["weekly_hours_worked_before_lsr"].iloc[0] == 40
    pd.testing.assert_series_equal(
        result.table("person")["hours_worked_last_week"],
        frame.table("person")["hours_worked_last_week"],
    )


def test_puf_role_is_not_authenticated_by_numerical_agreement_with_hrswk() -> None:
    frame = _frame([40, 0, 38, 0, 40, 0, 38, 0], role="puf_tax_detail")
    with pytest.raises(ValueError, match="no person rows"):
        prepare_acs_local_hours_donor(frame, seed=3, period=2024)


@pytest.mark.parametrize("age_column", ["age", "A_AGE"])
def test_asec_children_niu_are_excluded_without_filling_their_base_cells(age_column):
    before = _frame(
        [40, np.nan, 38, 0, 40, np.nan, 38, 0],
        columns={age_column: [40, 10, 15, np.nan] * 2},
    )
    prepared, donor, receipt = prepare_acs_local_hours_donor(
        before, seed=3, period=2024
    )
    assert prepared is before
    assert donor.person[age_column].tolist() == [40, 15, 40, 15]
    assert receipt["universe_columns"] == [age_column, "WKSWORK"]
    assert receipt["excluded_source_universe_rows"] == 4
    assert receipt["preserved_observed_rows"] == 4
    assert receipt["filled_rows"] == 0


@pytest.mark.parametrize("column,value", [("HRSWK", 0), ("WKSWORK", 0)])
def test_asec_past_year_hours_and_weeks_must_have_coherent_universes(column, value):
    with pytest.raises(ValueError, match="past-year work status"):
        prepare_acs_local_hours_donor(_frame(raw={column: value}), seed=3, period=2024)


@pytest.mark.parametrize("workyn", [0, 3, np.nan])
def test_present_workyn_requires_valid_in_universe_codes(workyn):
    frame = _frame(columns={"WORKYN": [workyn, 2, 1, 2] * 2})
    with pytest.raises(ValueError, match="WORKYN"):
        prepare_acs_local_hours_donor(frame, seed=3, period=2024)


def test_twelve_temporary_workers_preserve_observed_hours_and_raw_history():
    before = _frame(
        np.full(12, 10.0),
        n=12,
        columns={
            "A_AGE": 15,
            "HRSWK": 10,
            "WKSWORK": 4,
            "WORKYN": 2,
            "WTEMP": 1,
            "WRK_CK": 1,
        },
    )
    snapshots = {
        entity: before.table(entity).copy(deep=True) for entity in before.entities
    }
    prepared, donor, receipt = prepare_acs_local_hours_donor(
        before, seed=3, period=2024
    )
    assert prepared is before
    assert receipt["filled_rows"] == 0
    assert receipt["preserved_observed_rows"] == 12
    assert receipt["universe_columns"] == [
        "A_AGE",
        "WKSWORK",
        "WORKYN",
        "WTEMP",
        "WRK_CK",
    ]
    pd.testing.assert_frame_equal(donor.person, before.person)
    for entity, expected in snapshots.items():
        pd.testing.assert_frame_equal(before.table(entity), expected)


@pytest.mark.parametrize("final_code", [None, 1])
def test_explicit_initial_and_followup_no_refuse_positive_hours(final_code):
    columns = {"WORKYN": [2, 2, 1, 2] * 2, "WTEMP": [2, 2, 0, 2] * 2}
    if final_code is not None:
        columns["WRK_CK"] = [final_code, 2, 1, 2] * 2
    with pytest.raises(ValueError, match="past-year work status"):
        prepare_acs_local_hours_donor(_frame(columns=columns), seed=3, period=2024)


@pytest.mark.parametrize("row,final_code", [(0, 2), (1, 1)])
def test_final_work_recode_must_agree_with_hours_and_weeks(row, final_code):
    flags = np.array([1, 2, 1, 2] * 2)
    flags[row] = final_code
    with pytest.raises(ValueError, match="WRK_CK"):
        prepare_acs_local_hours_donor(
            _frame(columns={"WRK_CK": flags}), seed=3, period=2024
        )


@pytest.mark.parametrize("followup", [None, 0])
def test_initial_no_without_final_or_followup_does_not_override_hours(followup):
    columns = {"WORKYN": [2, 2, 1, 2] * 2}
    if followup is not None:
        columns["WTEMP"] = followup
    before = _frame([40, 0, 38, 0] * 2, columns=columns)
    prepared, _, receipt = prepare_acs_local_hours_donor(before, seed=3, period=2024)
    assert prepared is before
    assert receipt["preserved_observed_rows"] == 8


@pytest.mark.parametrize("column", ["WORKYN", "WTEMP"])
def test_affirmative_initial_or_followup_answer_refuses_zero_hours(column):
    columns = {"WORKYN": [1, 2, 1, 2] * 2, "WTEMP": [0, 2, 0, 2] * 2}
    columns[column][1] = 1
    with pytest.raises(ValueError, match="past-year work status"):
        prepare_acs_local_hours_donor(_frame(columns=columns), seed=3, period=2024)


@pytest.mark.parametrize(
    "column,value",
    [
        ("WTEMP", -1),
        ("WTEMP", 3),
        ("WTEMP", np.nan),
        ("WRK_CK", 0),
        ("WRK_CK", 3),
        ("WRK_CK", np.nan),
    ],
)
def test_optional_followup_and_final_recode_require_valid_raw_codes(column, value):
    values = [0, 2, 0, 2] * 2 if column == "WTEMP" else [1, 2, 1, 2] * 2
    values[0] = value
    with pytest.raises(ValueError, match=column):
        prepare_acs_local_hours_donor(
            _frame(columns={column: values}), seed=3, period=2024
        )


@pytest.mark.parametrize(
    "unresolved,recipient_age,model_children",
    [
        (False, 35, False),
        (True, 35, False),
        (True, 15, False),
        (False, 35, True),
        (True, 15, True),
    ],
)
def test_local_pipeline_uses_native_hours_then_asec_fallback_and_puf_tax_donor(
    monkeypatch, tmp_path, unresolved, recipient_age, model_children
):
    from microcosm.build.us_runtime import acs_multispine, acs_transfer
    from microcosm.build.us_runtime.acs_pums import AcsPumsSource

    # PUF hours legitimately differ from their source HRSWK. Only the ASEC
    # observation channel is qualified; the missing original cell becomes 0.
    original = _frame(
        [40, np.nan if unresolved else 0, 38, 0, 50, 5, 48, 5],
        role=["asec"] * 4 + ["puf_tax_detail"] * 4,
        columns={"taxable_interest_income": [1, 2, 3, 4, 100, 200, 300, 400]},
        raw={} if unresolved else {"HRSWK": None, "A_HRS1": None, "WKSWORK": None},
    )
    donor_calls = []

    def qualify_hours(base):
        assert unresolved, "A native-complete ACS spine must not qualify a raw donor."
        prepared, donor, receipt = prepare_acs_local_hours_donor(
            base, seed=3, period=2024
        )
        assert donor.n("person") == 4
        assert receipt["filled_rows"] == 1
        donor_calls.append(receipt)
        return prepared, donor, receipt

    before_recipient = _frame(role=None)
    tables = {
        entity: before_recipient.table(entity).copy()
        for entity in before_recipient.entities
    }
    person = tables["person"]
    person["AGEP"] = person.pop("age")
    person.loc[[2, 6], "AGEP"] = recipient_age
    person["SEX"] = person.pop("is_female").map({False: 1, True: 2})
    person["WKHP"] = [40, np.nan, np.nan if unresolved else 38, np.nan] * 2
    person["WKL"] = [1, 2, np.nan if recipient_age == 15 else 1, 3] * 2
    person["FWKHP"] = [0, 0, 1, 0] * 2
    if model_children:
        person.loc[[1, 5], "AGEP"] = 12
        person.loc[[1, 5], "WKL"] = np.nan
    recipient = Frame(
        tables,
        before_recipient.schema,
        {"household": before_recipient.weights_for("household")},
    )
    fit_targets = []

    class MeanQRF:
        def __init__(self, **kwargs):
            pass

        def fit(self, frame, predictors, targets, *, weights):
            values = frame.person[targets].mean()
            fit_targets.append((tuple(targets), len(frame.person), values.to_dict()))
            return SimpleNamespace(
                weight_kind=weights,
                predict=lambda features: pd.DataFrame(
                    {
                        column: np.full(len(features), values[column])
                        for column in targets
                    },
                    index=features.index,
                ),
            )

    monkeypatch.setattr(acs_transfer, "QRF", MeanQRF)
    monkeypatch.setattr(
        acs_multispine, "build_acs_pums_unit_frame", lambda *a, **k: (recipient, {})
    )
    # A real second transfer on the original default PUF role exercises
    # channel separation; distinct PUF interest values identify its donors.
    result = acs_multispine.build_optional_acs_multispine(
        original,
        AcsPumsSource(tmp_path / "unused-hh.zip", tmp_path / "unused-person.zip"),
        hours_donor_factory=qualify_hours,
        hours_under15_policy=ACS_UNDER15_ZERO_POLICY if model_children else None,
        target_families={"person": {"tax_detail": ("taxable_interest_income",)}},
        seed=3,
        n_estimators=1,
    )
    assert acs_local_hours_signal_gate(result.frame).passed
    assert result.provenance["fit_configuration"]["hours_donor_channel"] == (
        "asec" if unresolved else None
    )
    assert len(donor_calls) == (1 if unresolved else 0)
    assert (
        result.provenance["fit_configuration"]["resolved_donor_channel"]
        == "puf_tax_detail"
    )
    if unresolved:
        assert fit_targets[0] == (
            ("weekly_hours_worked_before_lsr",),
            4,
            {"weekly_hours_worked_before_lsr": 19.5},
        )
    assert len(fit_targets) == (2 if unresolved else 1)
    assert fit_targets[-1] == (
        ("taxable_interest_income",),
        4,
        {
            "taxable_interest_income": 250.0,
        },
    )
    for source_id, expected in zip([5, 6, 7, 8], [50, 5, 48, 5], strict=True):
        observed = result.frame.person.loc[
            result.frame.person["person_id"].eq(source_id),
            "weekly_hours_worked_before_lsr",
        ]
        assert observed.tolist() == [expected]
    assert original.person["weekly_hours_worked_before_lsr"].isna().sum() == (
        1 if unresolved else 0
    )
    acs = result.frame.person.loc[
        result.frame.person["person_spine"].eq("acs_2024_1yr")
    ]
    assert acs["weekly_hours_worked_before_lsr"].tolist() == (
        [40, 0, 19.5 if unresolved else 38, 0] * 2
    )
    assert result.provenance["native_inputs"]["weekly_hours_worked_before_lsr"][
        "missing_rows"
    ] == (2 if unresolved else 0) + (2 if model_children else 0)
    if model_children:
        assert result.provenance["hours_modeled_completion"]["modeled_rows"] == 2
        assert (
            result.provenance["hours_modeled_completion"]["provenance"]
            == "modeled_assumption"
        )
        assert result.provenance["native_inputs"]["weekly_hours_worked_before_lsr"][
            "observed_rows"
        ] == (4 if unresolved else 6)


@pytest.mark.parametrize("age", [12, np.nan])
def test_unresolved_children_or_unknown_age_refuse_before_donor_factory(
    monkeypatch, tmp_path, age
):
    from microcosm.build.us_runtime import acs_multispine
    from microcosm.build.us_runtime.acs_pums import AcsPumsSource

    recipient = _frame(role=None, columns={"AGEP": [age] * 8, "age": [age] * 8})
    monkeypatch.setattr(
        acs_multispine, "build_acs_pums_unit_frame", lambda *a, **k: (recipient, {})
    )
    # This test concerns the guard after native mapping, whose source-universe
    # behavior is tested independently with real WKHP fixtures.
    monkeypatch.setattr(
        acs_multispine,
        "map_acs_native_inputs",
        lambda frame: SimpleNamespace(frame=frame, native_inputs={}),
    )

    def unexpected_donor(_):
        pytest.fail(
            "An unsupported recipient universe must fail before qualifying raw donors."
        )

    with pytest.raises(ValueError, match="modeled-completion policy"):
        acs_multispine.build_optional_acs_multispine(
            _frame(),
            AcsPumsSource(tmp_path / "unused-hh.zip", tmp_path / "unused-person.zip"),
            hours_donor_factory=unexpected_donor,
        )


def test_under15_completion_is_explicit_and_preserves_all_observed_values():
    before = _frame(
        [np.nan, 40, 99, 0, np.nan, 38, 0, 40],
        role=None,
        columns={"age": [12, 12, 13, 14, 15, 40, 30, 35]},
    )
    unchanged, receipt = complete_acs_local_under15_hours(before)
    assert unchanged is before and receipt is None
    after, receipt = complete_acs_local_under15_hours(
        before, policy=ACS_UNDER15_ZERO_POLICY
    )
    assert after.person["weekly_hours_worked_before_lsr"].iloc[0] == 0
    assert pd.isna(after.person["weekly_hours_worked_before_lsr"].iloc[4])
    pd.testing.assert_series_equal(
        after.person["weekly_hours_worked_before_lsr"].iloc[1:],
        before.person["weekly_hours_worked_before_lsr"].iloc[1:],
    )
    assert receipt["provenance"] == "modeled_assumption"
    assert receipt["modeled_rows"] == 1
    assert receipt["modeled_rows_by_age"] == {"12": 1}
    assert receipt["earnings_unknown_rows"] == 1
    assert receipt["pre_completion_missing_rows"] == 2
    assert receipt["remaining_missing_rows"] == 1


@pytest.mark.parametrize(
    "column,value",
    [
        ("WKHP", 40),
        ("WKL", 1),
        ("WAGP", 1),
        ("SEMP", 1),
        ("SEMP", -1),
        ("employment_income_before_lsr", 1),
        ("self_employment_income_before_lsr", 1),
        ("self_employment_income_before_lsr", -1),
    ],
)
def test_under15_completion_refuses_conflicting_work_evidence(column, value):
    frame = _frame(
        [np.nan] * 8, role=None, columns={"age": [12] * 8, column: [value] * 8}
    )
    with pytest.raises(ValueError, match="work or nonzero earnings evidence"):
        complete_acs_local_under15_hours(frame, policy=ACS_UNDER15_ZERO_POLICY)


def test_under15_completion_counts_raw_earnings_blanks_despite_mapped_zeros():
    frame = _frame(
        [np.nan] * 8,
        role=None,
        columns={
            "age": [12] * 8,
            "WAGP": [np.nan] * 8,
            "SEMP": [np.nan] * 8,
            "employment_income_before_lsr": [0] * 8,
            "self_employment_income_before_lsr": [0] * 8,
        },
    )
    _, receipt = complete_acs_local_under15_hours(frame, policy=ACS_UNDER15_ZERO_POLICY)
    assert receipt["earnings_unknown_rows"] == 8
    assert receipt["earnings_columns"] == ["WAGP", "SEMP"]
    assert receipt["earnings_evidence"] == [
        {"column": "WAGP", "scope": "raw_source"},
        {"column": "SEMP", "scope": "raw_source"},
    ]


def test_under15_completion_refuses_unknown_age_and_other_spines():
    frame = _frame([np.nan] * 8, role=None, columns={"age": [np.nan] * 8})
    with pytest.raises(ValueError, match="unknown age"):
        complete_acs_local_under15_hours(frame, policy=ACS_UNDER15_ZERO_POLICY)
    with pytest.raises(ValueError, match="ACS-only"):
        complete_acs_local_under15_hours(
            _frame(spines=["asec_puf"] * 8), policy=ACS_UNDER15_ZERO_POLICY
        )


@pytest.mark.parametrize("bad_acs", [[40] * 4, [0] * 4, [np.nan] * 4])
def test_national_signal_cannot_hide_a_failed_acs_spine(bad_acs) -> None:
    frame = _frame(
        [40, 0, 38, 0, *bad_acs],
        spines=["asec_puf"] * 4 + ["acs_2024_1yr"] * 4,
    )
    gate = acs_local_hours_signal_gate(frame)
    assert not gate.passed
    assert any("acs_2024_1yr" in failure for failure in gate.failures)


def test_absent_usual_hours_fails_before_engine_defaults_can_fill_it() -> None:
    gate = acs_local_hours_signal_gate(
        _frame(spines=["asec_puf"] * 4 + ["acs_2024_1yr"] * 4)
    )
    assert not gate.passed
    assert any("weekly_hours_worked_before_lsr" in item for item in gate.failures)


def test_valid_forty_and_zero_values_in_each_spine_pass() -> None:
    frame = _frame(
        [40, 0, 38, 0, 40, 0, 38, 0],
        spines=["asec_puf"] * 4 + ["acs_2024_1yr"] * 4,
    )
    assert acs_local_hours_signal_gate(frame).passed


@pytest.mark.parametrize("default", [0, 40])
def test_recorded_source_nulls_fail_even_after_mixed_default_fill(default) -> None:
    # Both origins retain >1 value: ordinary nonconstant checks would pass.
    frame = _frame(
        [40, 0, 38, 0, 40, 0, 38, default],
        spines=["asec_puf"] * 4 + ["acs_2024_1yr"] * 4,
    )
    gate = acs_local_hours_signal_gate(
        frame,
        source_null_audit=[
            {
                "entity": "person",
                "column": "weekly_hours_worked_before_lsr",
                "missing_rows": 1,
                "missing_rows_by_spine": {"acs_2024_1yr": 1},
            }
        ],
    )
    assert not gate.passed
    assert any("before consumer filling" in failure for failure in gate.failures)
