"""ASEC-measured CDCC adult-care inputs (PolicyEngine/microcosm#451 item 1)."""

from __future__ import annotations

import importlib.util
from importlib.metadata import version

import numpy as np
import pandas as pd
import pytest

from microcosm.build.source_runtime import SourceRuntimeError
from microcosm.build.us_runtime import acs_transfer as acs_transfer_module
from microcosm.build.us_runtime import multispine_pool as multispine_pool_module
from microcosm.build.us_runtime.adult_care import (
    US_ADULT_CARE_CHILD_QUALIFYING_AGE_LIMIT,
    US_ADULT_CARE_EARNED_INCOME_SOURCES,
    US_ADULT_CARE_OUTPUT_COLUMNS,
    US_ADULT_CARE_REQUIRED_SOURCE_COLUMNS,
    derive_us_adult_care_from_manifest,
    us_adult_care_signal_gate,
    us_adult_care_stage_spec,
    with_us_adult_care_inputs,
)
from microcosm.build.us_runtime.puf_support import clone_us_frame_for_puf_support
from microcosm.build.us_runtime.source_runtime import us_source_operation_handlers
from microcosm.build.us_runtime.spine_assembly import assemble_spines
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

policyengine_us_installed = importlib.util.find_spec("policyengine_us") is not None
requires_us = pytest.mark.skipif(
    not policyengine_us_installed,
    reason="requires the policyengine-us [us] extra (build environment)",
)

_FLAG, _EXPENSE = US_ADULT_CARE_OUTPUT_COLUMNS
_DONOR_CHILDCARE = 4_000.0


def _frame() -> Frame:
    """25 people: two statute-eligible units, donor units, and filler singles.

    Unit 201: married head (earned) + spouse with measured PEDISDRS == 1 and
    no earnings — the 21(b)(1)(C) prong whose work test passes only through
    the 21(d)(2) deeming rule.
    Unit 202: single head (earned) + disabled adult dependent — 21(b)(1)(B).
    Unit 203: two working parents + a child under 13 with positive measured
    childcare — the paid-usage donor.
    Unit 204: two working parents + a child under 13 with zero childcare —
    holds the measured usage rate strictly inside (0, 1).
    Units 205+: working singles diluting the weighted flag share into the
    plausibility band.
    """

    rows: list[dict[str, object]] = []

    def _person(
        unit: int,
        role: str,
        *,
        age: float,
        pedisdrs: float = 2.0,
        employment: float = 0.0,
        student: bool = False,
    ) -> None:
        rows.append(
            {
                "person_tax_unit_id": unit,
                "tax_unit_role_input": role,
                "age": age,
                "PEDISDRS": pedisdrs,
                "employment_income_before_lsr": employment,
                "self_employment_income_before_lsr": 0.0,
                "sstb_self_employment_income_before_lsr": 0.0,
                "is_full_time_college_student": student,
                "person_support_channel": "asec",
            }
        )

    _person(201, "HEAD", age=45, employment=80_000.0)
    _person(201, "SPOUSE", age=44, pedisdrs=1.0)
    _person(202, "HEAD", age=50, employment=40_000.0)
    _person(202, "DEPENDENT", age=25, pedisdrs=1.0)
    _person(203, "HEAD", age=35, employment=60_000.0)
    _person(203, "SPOUSE", age=34, employment=30_000.0)
    _person(203, "DEPENDENT", age=6)
    _person(204, "HEAD", age=36, employment=55_000.0)
    _person(204, "SPOUSE", age=35, employment=25_000.0)
    _person(204, "DEPENDENT", age=4)
    for offset in range(15):
        _person(205 + offset, "HEAD", age=30 + offset, employment=20_000.0)

    person = pd.DataFrame(rows)
    count = len(person)
    person.insert(0, "person_id", np.arange(1, count + 1, dtype="int64"))
    person["person_household_id"] = person["person_tax_unit_id"] + 800
    person["person_spm_unit_id"] = person["person_tax_unit_id"] + 400
    person["person_family_id"] = person["person_tax_unit_id"] + 600
    person["person_marital_unit_id"] = np.arange(701, 701 + count, dtype="int64")

    tax_unit_ids = person["person_tax_unit_id"].drop_duplicates().to_numpy()
    spm_unit_ids = person["person_spm_unit_id"].drop_duplicates().to_numpy()
    household_ids = person["person_household_id"].drop_duplicates().to_numpy()
    childcare = np.where(
        spm_unit_ids == 603,
        _DONOR_CHILDCARE,
        0.0,
    )
    tables = {
        "person": person,
        "household": pd.DataFrame(
            {
                "household_id": household_ids,
                "state_fips": np.full(len(household_ids), 6),
            }
        ),
        "tax_unit": pd.DataFrame({"tax_unit_id": tax_unit_ids}),
        "spm_unit": pd.DataFrame(
            {
                "spm_unit_id": spm_unit_ids,
                "spm_unit_pre_subsidy_childcare_expenses": childcare,
            }
        ),
        "family": pd.DataFrame(
            {"family_id": person["person_family_id"].drop_duplicates().to_numpy()}
        ),
        "marital_unit": pd.DataFrame(
            {"marital_unit_id": person["person_marital_unit_id"].to_numpy()}
        ),
    }
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                np.ones(len(household_ids), dtype=np.float64),
                WeightKind.DESIGN,
            )
        },
    )


def test_stage_manifest_pins_measured_flag_and_documented_donor_proxy() -> None:
    spec = us_adult_care_stage_spec()

    assert spec.stage == "adult_care_inputs"
    assert spec.survey == "Census CPS ASEC"
    assert spec.grain == "person"
    assert spec.outputs == US_ADULT_CARE_OUTPUT_COLUMNS
    assert spec.nonnegative_outputs == (_EXPENSE,)
    assert [operation.kind for operation in spec.operations] == [
        "read_table",
        "derive_adult_care_inputs",
    ]
    assert spec.operations[1].parameters == {
        "self_care_difficulty_source": "PEDISDRS",
        "childcare_expense_source": "spm_unit_pre_subsidy_childcare_expenses",
        "full_time_student_source": "is_full_time_college_student",
        "support_channel_source": "person_support_channel",
        "child_qualifying_age_limit": US_ADULT_CARE_CHILD_QUALIFYING_AGE_LIMIT,
        "earned_income_sources": list(US_ADULT_CARE_EARNED_INCOME_SOURCES),
        "seed_from_build_config": True,
        "output_flag": _FLAG,
        "output_expenses": _EXPENSE,
    }
    # The source-decision receipts must stay in the shipped declaration.
    assert "PEDISDRS" in spec.notes
    assert "ESELFCARE" in spec.notes
    assert "TDPCAREAMT" in spec.notes
    assert "former household member" in spec.notes
    assert "21(d)(2)" in spec.notes

    handlers = us_source_operation_handlers()
    assert handlers["derive_adult_care_inputs"] is derive_us_adult_care_from_manifest


def test_flag_is_measured_and_expenses_bind_the_statute_structure() -> None:
    result = with_us_adult_care_inputs(_frame(), seed=7, time_period=2024)
    person = result.table("person")

    # The flag is exactly the measured PEDISDRS == 1 signal.
    assert person[_FLAG].dtype == bool
    np.testing.assert_array_equal(
        person[_FLAG].to_numpy(), person["PEDISDRS"].to_numpy() == 1.0
    )

    # The measured usage rate is 0.5 (one positive donor of two), so the
    # weight-targeted prefix selects exactly one of the two eligible units,
    # and the level draw reproduces the only positive donor value.
    expenses = person[_EXPENSE].to_numpy()
    positive = expenses > 0.0
    assert positive.sum() == 1
    assert expenses[positive][0] == pytest.approx(_DONOR_CHILDCARE)
    carrier = person.loc[positive].iloc[0]
    assert bool(carrier[_FLAG])
    assert int(carrier["person_tax_unit_id"]) in (201, 202)

    gate = us_adult_care_signal_gate(result)
    assert gate.passed, gate.failures


def test_real_pool_chain_exposes_nullable_boolean_to_auto_transfer_donor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def source_peer(*, spine: str) -> Frame:
        base = _frame()
        tables = {entity: base.table(entity).copy() for entity in base.entities}
        person = tables["person"]
        person.drop(columns=["person_support_channel"], inplace=True)
        person["is_female"] = False
        if spine == "asec":
            person["PERIDNUM"] = [f"asec-{index}" for index in person.index]
        else:
            person.drop(
                columns=[
                    "PEDISDRS",
                    "is_full_time_college_student",
                    "sstb_self_employment_income_before_lsr",
                    "tax_unit_role_input",
                ],
                inplace=True,
            )
            tables["spm_unit"].drop(
                columns=["spm_unit_pre_subsidy_childcare_expenses"],
                inplace=True,
            )
        return Frame(
            tables,
            base.schema,
            {"household": base.weights_for("household")},
            base.strata,
        )

    assembled = assemble_spines(
        {"asec": source_peer(spine="asec"), "acs": source_peer(spine="acs")},
        household_mass_shares={"asec": 0.5, "acs": 0.5},
    )
    cloned = clone_us_frame_for_puf_support(assembled)
    completed = multispine_pool_module._run_source_operator_chain(
        cloned,
        phase="post_clone",
        operator_names=("with_us_adult_care_inputs",),
        operators={
            "with_us_adult_care_inputs": lambda frame: with_us_adult_care_inputs(
                frame,
                seed=0,
                time_period=2024,
            )
        },
    )
    assert completed.receipt["operator_order"] == ["with_us_adult_care_inputs"]

    fit_donor, role = acs_transfer_module.resolve_acs_donor_channel(
        completed.frame,
        acs_transfer_module.ACS_DONOR_CHANNEL_AUTO,
    )
    assert role == "puf_tax_detail"
    fit_person = fit_donor.table("person")
    assert set(fit_person["person_support_clone_index"]) == {1}

    flag = fit_person[_FLAG]
    assert flag.dtype == object
    assert pd.api.types.infer_dtype(flag, skipna=True) == "boolean"
    source_channel = fit_person["person_support_channel"]
    assert flag.loc[source_channel.eq("asec")].notna().all()
    assert flag.loc[source_channel.eq("acs")].isna().all()

    monkeypatch.setattr(
        acs_transfer_module,
        "_engine_variable_metadata",
        lambda _target: None,
    )
    acs_transfer_module._validate_donor_targets(
        fit_donor,
        entity="person",
        targets=US_ADULT_CARE_OUTPUT_COLUMNS,
    )
    complete = acs_transfer_module._complete_target_mask(
        fit_person,
        targets=US_ADULT_CARE_OUTPUT_COLUMNS,
    )
    assert int(complete.sum()) == int(source_channel.eq("asec").sum())
    encodings = acs_transfer_module._complete_case_target_encodings(
        fit_person,
        targets=US_ADULT_CARE_OUTPUT_COLUMNS,
        complete=complete,
    )
    assert encodings[_FLAG].kind == "boolean"
    np.testing.assert_array_equal(encodings[_FLAG].support, np.array([0.0, 1.0]))
    assert encodings[_EXPENSE].kind == "continuous"
    assert encodings[_EXPENSE].support is None


def test_derivation_is_deterministic_per_seed() -> None:
    first = with_us_adult_care_inputs(_frame(), seed=7, time_period=2024)
    second = with_us_adult_care_inputs(_frame(), seed=7, time_period=2024)
    pd.testing.assert_series_equal(
        first.table("person")[_EXPENSE], second.table("person")[_EXPENSE]
    )
    pd.testing.assert_series_equal(
        first.table("person")[_FLAG], second.table("person")[_FLAG]
    )


def test_taxpayer_alone_is_never_their_own_qualifying_individual() -> None:
    frame = _frame()
    person = frame.table("person")
    # Turn the married-spouse carrier into an unmarried disabled head: the
    # spouse prong must stop qualifying, leaving only unit 202 eligible.
    spouse_row = person.index[person["tax_unit_role_input"] == "SPOUSE"][0]
    assert int(person.loc[spouse_row, "person_tax_unit_id"]) == 201
    person.loc[spouse_row, "person_tax_unit_id"] = 299
    person.loc[spouse_row, "tax_unit_role_input"] = "HEAD"
    person.loc[spouse_row, "person_household_id"] = 1_099
    person.loc[spouse_row, "person_spm_unit_id"] = 699
    person.loc[spouse_row, "person_family_id"] = 899
    frame.table("tax_unit").loc[len(frame.table("tax_unit"))] = {"tax_unit_id": 299}
    frame.table("spm_unit").loc[len(frame.table("spm_unit"))] = {
        "spm_unit_id": 699,
        "spm_unit_pre_subsidy_childcare_expenses": 0.0,
    }
    frame.table("family").loc[len(frame.table("family"))] = {"family_id": 899}
    households = frame.table("household")
    households.loc[len(households)] = {"household_id": 1_099, "state_fips": 6}
    rebuilt = Frame(
        {entity: frame.table(entity) for entity in frame.entities},
        US_SCHEMA,
        {
            "household": Weights(
                np.ones(len(frame.table("household")), dtype=np.float64),
                WeightKind.DESIGN,
            )
        },
    )

    result = with_us_adult_care_inputs(rebuilt, seed=7, time_period=2024)
    person_out = result.table("person")
    expense_units = set(
        person_out.loc[person_out[_EXPENSE] > 0.0, "person_tax_unit_id"].tolist()
    )
    assert 299 not in expense_units
    assert 201 not in expense_units
    # The remaining statute-eligible unit must actually receive the expense:
    # exclusion of the invalid units may not silently zero the surface.
    assert expense_units == {202}


def test_deeming_never_rescues_a_unit_whose_only_earner_is_the_disabled_spouse() -> (
    None
):
    frame = _frame()
    person = frame.table("person")
    # Invert unit 201: the incapacitated spouse becomes the sole earner. The
    # engine deems only the eligible spouse and keeps the healthy spouse's
    # actual zero, so min_head_spouse_earned stays 0 and the unit cannot
    # bind; it must not consume the seeded selection prefix.
    head_row = person.index[
        (person["person_tax_unit_id"] == 201)
        & (person["tax_unit_role_input"] == "HEAD")
    ][0]
    spouse_row = person.index[
        (person["person_tax_unit_id"] == 201)
        & (person["tax_unit_role_input"] == "SPOUSE")
    ][0]
    person.loc[head_row, "employment_income_before_lsr"] = 0.0
    person.loc[spouse_row, "employment_income_before_lsr"] = 40_000.0

    result = with_us_adult_care_inputs(frame, seed=7, time_period=2024)
    person_out = result.table("person")
    expense_units = set(
        person_out.loc[person_out[_EXPENSE] > 0.0, "person_tax_unit_id"].tolist()
    )
    assert 201 not in expense_units
    assert expense_units == {202}


def test_student_spouse_deeming_admits_dependent_prong_units() -> None:
    frame = _frame()
    person = frame.table("person")
    # A working head, a non-earning full-time-student spouse, and a disabled
    # adult dependent: the engine's 21(d)(2) floor covers the student spouse,
    # so the unit binds and must be selectable.
    base = int(person["person_id"].max())
    student_unit = pd.DataFrame(
        [
            {
                "person_id": base + 1,
                "person_tax_unit_id": 310,
                "tax_unit_role_input": "HEAD",
                "age": 40.0,
                "PEDISDRS": 2.0,
                "employment_income_before_lsr": 50_000.0,
                "self_employment_income_before_lsr": 0.0,
                "sstb_self_employment_income_before_lsr": 0.0,
                "is_full_time_college_student": False,
                "person_support_channel": "asec",
                "person_household_id": 1_110,
                "person_spm_unit_id": 710,
                "person_family_id": 910,
                "person_marital_unit_id": 810,
            },
            {
                "person_id": base + 2,
                "person_tax_unit_id": 310,
                "tax_unit_role_input": "SPOUSE",
                "age": 39.0,
                "PEDISDRS": 2.0,
                "employment_income_before_lsr": 0.0,
                "self_employment_income_before_lsr": 0.0,
                "sstb_self_employment_income_before_lsr": 0.0,
                "is_full_time_college_student": True,
                "person_support_channel": "asec",
                "person_household_id": 1_110,
                "person_spm_unit_id": 710,
                "person_family_id": 910,
                "person_marital_unit_id": 811,
            },
            {
                "person_id": base + 3,
                "person_tax_unit_id": 310,
                "tax_unit_role_input": "DEPENDENT",
                "age": 24.0,
                "PEDISDRS": 1.0,
                "employment_income_before_lsr": 0.0,
                "self_employment_income_before_lsr": 0.0,
                "sstb_self_employment_income_before_lsr": 0.0,
                "is_full_time_college_student": False,
                "person_support_channel": "asec",
                "person_household_id": 1_110,
                "person_spm_unit_id": 710,
                "person_family_id": 910,
                "person_marital_unit_id": 812,
            },
        ]
    )
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["person"] = pd.concat([person, student_unit], ignore_index=True)
    tables["tax_unit"].loc[len(tables["tax_unit"])] = {"tax_unit_id": 310}
    tables["spm_unit"].loc[len(tables["spm_unit"])] = {
        "spm_unit_id": 710,
        "spm_unit_pre_subsidy_childcare_expenses": 0.0,
    }
    tables["family"].loc[len(tables["family"])] = {"family_id": 910}
    tables["household"].loc[len(tables["household"])] = {
        "household_id": 1_110,
        "state_fips": 6,
    }
    for extra in (810, 811, 812):
        tables["marital_unit"].loc[len(tables["marital_unit"])] = {
            "marital_unit_id": extra
        }
    rebuilt = Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                np.ones(len(tables["household"]), dtype=np.float64),
                WeightKind.DESIGN,
            )
        },
    )

    # With three eligible units and the 0.5 measured rate, the prefix takes
    # two: whichever two are selected, the student-deemed unit must be
    # selectable, and every carrier must be a qualifying person. Run a few
    # seeds and require unit 310 to appear for at least one.
    selected_by_seed = {}
    for seed in (0, 1, 2, 3):
        result = with_us_adult_care_inputs(rebuilt, seed=seed, time_period=2024)
        person_out = result.table("person")
        positive = person_out[_EXPENSE] > 0.0
        assert person_out.loc[positive, _FLAG].all()
        selected_by_seed[seed] = set(
            person_out.loc[positive, "person_tax_unit_id"].tolist()
        )
        assert selected_by_seed[seed] <= {201, 202, 310}
    assert any(310 in selected for selected in selected_by_seed.values())


def test_expense_assignment_is_invariant_to_person_row_order() -> None:
    baseline = with_us_adult_care_inputs(_frame(), seed=7, time_period=2024)
    source = _frame()
    tables = {entity: source.table(entity).copy() for entity in source.entities}
    order = np.random.default_rng(123).permutation(len(tables["person"]))
    tables["person"] = tables["person"].iloc[order].reset_index(drop=True)
    shuffled_frame = Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                np.ones(len(tables["household"]), dtype=np.float64),
                WeightKind.DESIGN,
            )
        },
    )

    shuffled = with_us_adult_care_inputs(shuffled_frame, seed=7, time_period=2024)

    baseline_by_person = dict(
        zip(
            baseline.table("person")["person_id"].tolist(),
            baseline.table("person")[_EXPENSE].tolist(),
            strict=True,
        )
    )
    shuffled_by_person = dict(
        zip(
            shuffled.table("person")["person_id"].tolist(),
            shuffled.table("person")[_EXPENSE].tolist(),
            strict=True,
        )
    )
    assert baseline_by_person == shuffled_by_person


def test_fails_closed_without_measured_sources() -> None:
    missing_flag_source = _frame()
    missing_flag_source.table("person").drop(columns=["PEDISDRS"], inplace=True)
    with pytest.raises(ValueError, match="cannot heal"):
        with_us_adult_care_inputs(missing_flag_source, seed=0, time_period=2024)

    missing_donor = _frame()
    missing_donor.table("spm_unit").drop(
        columns=["spm_unit_pre_subsidy_childcare_expenses"], inplace=True
    )
    with pytest.raises(ValueError, match="cannot heal"):
        with_us_adult_care_inputs(missing_donor, seed=0, time_period=2024)

    missing_student = _frame()
    missing_student.table("person").drop(
        columns=["is_full_time_college_student"], inplace=True
    )
    with pytest.raises(ValueError, match="cannot heal"):
        with_us_adult_care_inputs(missing_student, seed=0, time_period=2024)

    missing_channel = _frame()
    missing_channel.table("person").drop(
        columns=["person_support_channel"], inplace=True
    )
    with pytest.raises(ValueError, match="cannot heal"):
        with_us_adult_care_inputs(missing_channel, seed=0, time_period=2024)


def test_degenerate_measured_usage_rate_fails_closed() -> None:
    frame = _frame()
    spm = frame.table("spm_unit")
    # Making every kid-donor unit a payer collapses the measured usage rate
    # to 1.0, which must refuse rather than silently saturate.
    spm.loc[spm["spm_unit_id"] == 604, "spm_unit_pre_subsidy_childcare_expenses"] = (
        1_000.0
    )
    with pytest.raises(SourceRuntimeError, match="degenerate"):
        with_us_adult_care_inputs(frame, seed=0, time_period=2024)


def test_signal_gate_rejects_missing_default_and_invented_surfaces() -> None:
    valid = with_us_adult_care_inputs(_frame(), seed=7, time_period=2024)
    assert us_adult_care_signal_gate(valid).passed

    missing = with_us_adult_care_inputs(_frame(), seed=7, time_period=2024)
    missing.table("person").drop(columns=[_EXPENSE], inplace=True)
    assert not us_adult_care_signal_gate(missing).passed

    zeroed = with_us_adult_care_inputs(_frame(), seed=7, time_period=2024)
    zeroed.table("person")[_EXPENSE] = 0.0
    gate = us_adult_care_signal_gate(zeroed)
    assert not gate.passed
    assert any("structural zero" in failure for failure in gate.failures)

    unflagged_carrier = with_us_adult_care_inputs(_frame(), seed=7, time_period=2024)
    person = unflagged_carrier.table("person")
    unflagged_row = person.index[~person[_FLAG]][0]
    person.loc[unflagged_row, _EXPENSE] = 500.0
    gate = us_adult_care_signal_gate(unflagged_carrier)
    assert not gate.passed
    assert any("without the qualifying flag" in failure for failure in gate.failures)

    flagless = with_us_adult_care_inputs(_frame(), seed=7, time_period=2024)
    flagless.table("person")[_FLAG] = False
    assert not us_adult_care_signal_gate(flagless).passed


def test_signal_gate_certifies_statute_structure_and_value_sanity() -> None:
    # A null in the flag column must fail rather than coerce.
    nan_flag = with_us_adult_care_inputs(_frame(), seed=7, time_period=2024)
    person = nan_flag.table("person")
    person[_FLAG] = person[_FLAG].astype(object)
    person.loc[person.index[0], _FLAG] = np.nan
    gate = us_adult_care_signal_gate(nan_flag)
    assert not gate.passed
    assert any("non-boolean" in failure for failure in gate.failures)

    # The flag must match the measured PEDISDRS identity while the raw
    # source is still on the frame.
    drifted = with_us_adult_care_inputs(_frame(), seed=7, time_period=2024)
    person = drifted.table("person")
    unflagged_row = person.index[~person[_FLAG]][0]
    person.loc[unflagged_row, _FLAG] = True
    gate = us_adult_care_signal_gate(drifted)
    assert not gate.passed
    assert any("PEDISDRS" in failure for failure in gate.failures)

    # An expense on a disabled-but-unmarried head is not a section 21
    # qualifying person and must fail the structure certificate even though
    # the person carries the flag. (Drop PEDISDRS so only structure judges.)
    misplaced = with_us_adult_care_inputs(_frame(), seed=7, time_period=2024)
    person = misplaced.table("person")
    person.drop(columns=["PEDISDRS"], inplace=True)
    carrier_row = person.index[person[_EXPENSE] > 0.0][0]
    lone_head_row = person.index[
        (person["person_tax_unit_id"] >= 205)
        & (person["tax_unit_role_input"] == "HEAD")
    ][0]
    person.loc[lone_head_row, _FLAG] = True
    person.loc[lone_head_row, _EXPENSE] = person.loc[carrier_row, _EXPENSE]
    person.loc[carrier_row, _EXPENSE] = 0.0
    gate = us_adult_care_signal_gate(misplaced)
    assert not gate.passed
    assert any("qualifying persons" in failure for failure in gate.failures)

    # Two carriers in one unit break the single-carrier invariant.
    doubled = with_us_adult_care_inputs(_frame(), seed=7, time_period=2024)
    person = doubled.table("person")
    carrier_row = person.index[person[_EXPENSE] > 0.0][0]
    carrier_unit = person.loc[carrier_row, "person_tax_unit_id"]
    other_row = person.index[
        (person["person_tax_unit_id"] == carrier_unit) & (person.index != carrier_row)
    ][0]
    person.loc[other_row, _FLAG] = True
    person.loc[other_row, "PEDISDRS"] = 1.0
    person.loc[other_row, _EXPENSE] = 250.0
    gate = us_adult_care_signal_gate(doubled)
    assert not gate.passed
    assert any("more than one expense carrier" in failure for failure in gate.failures)

    # Corrupted magnitudes fail the plausibility ceiling.
    corrupt = with_us_adult_care_inputs(_frame(), seed=7, time_period=2024)
    person = corrupt.table("person")
    carrier_row = person.index[person[_EXPENSE] > 0.0][0]
    person.loc[carrier_row, _EXPENSE] = 1e300
    gate = us_adult_care_signal_gate(corrupt)
    assert not gate.passed
    assert any("plausibility ceiling" in failure for failure in gate.failures)

    # Without the role column the structure cannot be certified, so a healed
    # surface cannot pass on bands alone.
    unstructured = with_us_adult_care_inputs(_frame(), seed=7, time_period=2024)
    unstructured.table("person").drop(
        columns=["tax_unit_role_input", "PEDISDRS"], inplace=True
    )
    gate = us_adult_care_signal_gate(unstructured)
    assert not gate.passed
    assert any("certificate sources missing" in failure for failure in gate.failures)


def test_release_frame_without_raw_source_preserves_valid_surface() -> None:
    built = with_us_adult_care_inputs(_frame(), seed=7, time_period=2024)
    built.table("person").drop(columns=["PEDISDRS"], inplace=True)

    healed = with_us_adult_care_inputs(
        built,
        seed=7,
        time_period=2024,
        allow_existing_without_source=True,
    )

    pd.testing.assert_frame_equal(healed.table("person"), built.table("person"))


def test_required_source_columns_are_the_measured_set() -> None:
    assert US_ADULT_CARE_REQUIRED_SOURCE_COLUMNS == (
        "PEDISDRS",
        "spm_unit_pre_subsidy_childcare_expenses",
        "is_full_time_college_student",
        "person_support_channel",
    )


@requires_us
def test_policyengine_1_764_6_cdcc_adult_care_contract_and_live_binding() -> None:
    from policyengine_core.reforms import Reform
    from policyengine_us import CountryTaxBenefitSystem, Simulation

    assert version("policyengine-us") == "1.764.6"
    variables = CountryTaxBenefitSystem().variables
    flag = variables[_FLAG]
    assert flag.is_input_variable()
    assert flag.entity.key == "person"
    assert flag.value_type is bool
    expense = variables[_EXPENSE]
    assert expense.is_input_variable()
    assert expense.entity.key == "person"
    assert expense.value_type is float

    situation = {
        "people": {
            "head": {
                "age": {"2024": 45},
                "employment_income": {"2024": 80_000},
            },
            "spouse": {
                "age": {"2024": 44},
                _FLAG: {"2024": True},
                _EXPENSE: {"2024": 3_000},
            },
        },
        "tax_units": {
            "tax_unit": {
                "members": ["head", "spouse"],
                "filing_status": {"2024": "JOINT"},
            }
        },
        "spm_units": {"spm_unit": {"members": ["head", "spouse"]}},
        "households": {
            "household": {
                "members": ["head", "spouse"],
                "state_code": {"2024": "TX"},
            }
        },
    }

    baseline = Simulation(situation=situation)
    # 21(b)(1)(C) + 21(d)(2): the incapacitated spouse is the qualifying
    # individual and is deemed the earnings floor, so the credit binds.
    assert baseline.calculate("count_cdcc_eligible", 2024)[0] == 1
    assert baseline.calculate("cdcc_relevant_expenses", 2024)[0] == pytest.approx(
        3_000.0
    )
    assert baseline.calculate("cdcc", 2024)[0] > 0.0

    class NeutralizeAdultCare(Reform):
        def apply(self) -> None:
            self.neutralize_variable(_EXPENSE)

    neutralized = Simulation(situation=situation, reform=NeutralizeAdultCare)
    assert neutralized.calculate("cdcc_relevant_expenses", 2024)[0] == 0.0
    assert neutralized.calculate("cdcc", 2024)[0] == 0.0
    assert (
        neutralized.calculate("income_tax", 2024)[0]
        > baseline.calculate("income_tax", 2024)[0]
    )


def test_shipped_cdcc_adult_care_probe() -> None:
    from microcosm.build.us_runtime.release_input_coverage import (
        us_release_input_coverage_required_columns,
        us_release_reform_coverage_probes,
    )

    probe = next(
        probe
        for probe in us_release_reform_coverage_probes()
        if probe.id == "cdcc_adult_care_expense_neutralization"
    )
    assert probe.neutralized_variable == _EXPENSE
    assert probe.binding_inputs == US_ADULT_CARE_OUTPUT_COLUMNS
    assert probe.budget_measure == "income_tax"
    assert probe.period == 2024
    assert probe.effect_direction == "baseline_minus_reform"
    assert probe.expected_sign == "negative"
    assert probe.min_abs_effect >= 1_000_000.0
    assert "PolicyEngine/microcosm#451" in probe.issue

    required = us_release_input_coverage_required_columns()
    for column in US_ADULT_CARE_OUTPUT_COLUMNS:
        assert column in required


def test_implausible_donor_knots_are_refused_with_receipt() -> None:
    """microcosm#567 base-P3: two measured ASEC childcare values ($730k/$360k)
    were latent donor knots in every build; the grid shift from the tail
    clones let 12 draws enter the >$250k top-tail interpolation region
    (three between the ceiling and $360k, nine at or above $360k). Donors
    above the plausibility
    ceiling are refused as knots (measured frame values untouched), with a
    receipt; clean donors pass through unchanged."""
    import numpy as np

    from microcosm.build.us_runtime.adult_care import (
        _EXPENSE_PLAUSIBILITY_CEILING,
        _screen_implausible_donors,
    )

    childcare = np.asarray([500.0, 12_000.0, 360_000.0, 730_000.0, 80_000.0])
    weight = np.asarray([10.0, 20.0, 30.0, 40.0, 50.0])
    level_mask = np.asarray([True, True, True, True, True])

    clean_mask, receipt = _screen_implausible_donors(childcare, weight, level_mask)
    assert clean_mask.tolist() == [True, True, False, False, True]
    assert receipt == {
        "count": 2,
        "values": [360_000.0, 730_000.0],
        "weight": 70.0,
        "excluded_weight_share": 70.0 / 150.0,
        "retained_maximum": 80_000.0,
        "ceiling": _EXPENSE_PLAUSIBILITY_CEILING,
    }

    # Already-excluded rows (outside the level mask) are not double-counted.
    masked = np.asarray([True, True, False, True, True])
    clean_mask2, receipt2 = _screen_implausible_donors(childcare, weight, masked)
    assert clean_mask2.tolist() == [True, True, False, False, True]
    assert receipt2["count"] == 1
    assert receipt2["values"] == [730_000.0]

    # A clean pool is untouched with an empty receipt.
    clean = np.asarray([500.0, 12_000.0, 80_000.0])
    ones = np.asarray([1.0, 1.0, 1.0])
    all_true = np.asarray([True, True, True])
    same_mask, empty = _screen_implausible_donors(clean, ones, all_true)
    assert same_mask.tolist() == [True, True, True]
    assert empty["count"] == 0 and empty["values"] == []


def test_poisoned_donor_is_screened_end_to_end(capsys) -> None:
    """microcosm#573 review blocker: the screen must be bound at its
    production call site. A donor unit with an implausible measured
    childcare value flows through with_us_adult_care_inputs: outputs stay
    bounded by the CLEAN donor maximum, the measured childcare column is
    untouched (including the poisoned value), incidence survives (the
    usage rate is computed before the screen), and the receipt is
    emitted."""
    rows: list[dict[str, object]] = []

    def _person(unit, role, *, age, pedisdrs=2.0, employment=0.0):
        rows.append(
            {
                "person_tax_unit_id": unit,
                "tax_unit_role_input": role,
                "age": age,
                "PEDISDRS": pedisdrs,
                "employment_income_before_lsr": employment,
                "self_employment_income_before_lsr": 0.0,
                "sstb_self_employment_income_before_lsr": 0.0,
                "is_full_time_college_student": False,
                "person_support_channel": "asec",
            }
        )

    # Statute-eligible recipients (disabled spouse via 21(d)(2) deeming).
    for unit in (301, 302):
        _person(unit, "HEAD", age=45, employment=80_000.0)
        _person(unit, "SPOUSE", age=44, pedisdrs=1.0)
    # Donor units with children and earnings: 303 POISONED ($600k), 304
    # clean paid ($14k), 305 zero-childcare (keeps usage inside (0, 1)).
    for unit in (303, 304, 305):
        _person(unit, "HEAD", age=35, employment=60_000.0)
        _person(unit, "SPOUSE", age=34, employment=30_000.0)
        _person(unit, "DEPENDENT", age=6)
    for offset in range(15):
        _person(306 + offset, "HEAD", age=30 + offset, employment=20_000.0)

    person = pd.DataFrame(rows)
    person.insert(0, "person_id", np.arange(1, len(person) + 1, dtype="int64"))
    person["person_household_id"] = person["person_tax_unit_id"] + 800
    person["person_spm_unit_id"] = person["person_tax_unit_id"] + 400
    person["person_family_id"] = person["person_tax_unit_id"] + 600
    person["person_marital_unit_id"] = np.arange(701, 701 + len(person), dtype="int64")

    spm_unit_ids = person["person_spm_unit_id"].drop_duplicates().to_numpy()
    poisoned_childcare = np.where(
        spm_unit_ids == 703,
        600_000.0,
        np.where(spm_unit_ids == 704, 14_000.0, 0.0),
    )
    household_ids = person["person_household_id"].drop_duplicates().to_numpy()
    tax_unit_ids = person["person_tax_unit_id"].drop_duplicates().to_numpy()
    tables = {
        "person": person,
        "household": pd.DataFrame(
            {
                "household_id": household_ids,
                "state_fips": np.full(len(household_ids), 6),
            }
        ),
        "tax_unit": pd.DataFrame({"tax_unit_id": tax_unit_ids}),
        "spm_unit": pd.DataFrame(
            {
                "spm_unit_id": spm_unit_ids,
                "spm_unit_pre_subsidy_childcare_expenses": poisoned_childcare,
            }
        ),
        "family": pd.DataFrame(
            {"family_id": person["person_family_id"].drop_duplicates().to_numpy()}
        ),
        "marital_unit": pd.DataFrame(
            {"marital_unit_id": person["person_marital_unit_id"].to_numpy()}
        ),
    }
    frame = Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                np.full(len(household_ids), 100.0),
                WeightKind.DESIGN,
            )
        },
    )

    result = with_us_adult_care_inputs(frame, seed=7, time_period=2024)
    emitted = capsys.readouterr().out
    assert "refused 1 implausible donor knot" in emitted
    assert "600,000" in emitted or "600000" in emitted

    out_person = result.table("person")
    expenses = out_person["pre_subsidy_care_expenses"].to_numpy(dtype=np.float64)
    positive = expenses[expenses > 0.0]
    # Incidence survives at the PRE-screen usage rate: two paid of three
    # child-bearing donor units -> usage 2/3, which selects BOTH eligible
    # units (a screening-before-usage mutation drops usage to 1/3 and
    # selects only one, failing this count).
    assert positive.size == 2
    # Outputs are bounded by the CLEAN donor maximum, far under the ceiling.
    assert float(expenses.max()) <= 14_000.0
    # The measured childcare column is untouched, poisoned value included.
    out_spm = result.table("spm_unit")
    assert (
        out_spm.set_index("spm_unit_id").loc[
            703, "spm_unit_pre_subsidy_childcare_expenses"
        ]
        == 600_000.0
    )
