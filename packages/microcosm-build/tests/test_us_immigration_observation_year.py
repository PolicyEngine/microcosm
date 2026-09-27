"""Explicit source observation years; invented rule/transfer inputs only."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import test_us_acs_transfer as transfer_tests
import test_us_immigration as rule_tests

from microcosm.build.us_runtime import acs_transfer as transfer
from microcosm.build.us_runtime import immigration as rules
from microcosm.frame import Frame

YEAR = "immigration_source_observation_year"
PAIR = rules.US_IMMIGRATION_OUTPUT_COLUMNS
FAMILIES = {"person": {"source_operator_immigration": PAIR}}


def _replace_person(frame, person):
    return Frame(
        {e: person if e == "person" else frame.table(e).copy() for e in frame.entities},
        frame.schema,
        {e: frame.weights_for(e) for e in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def _mixed_boundary():
    frame = rule_tests._us_frame(
        [rule_tests._noncitizen(A_AGE=33, PEINUSYR=20) for _ in range(2)]
    )
    person = frame.person.copy()
    person[YEAR] = [2025, 2024]
    person["age"] = 33
    person["ssn_card_type"] = "NON_CITIZEN_VALID_EAD"
    person["immigration_status_str"] = ["DACA", "LEGAL_PERMANENT_RESIDENT"]
    for name, value in {"CIT": 5, "POBP": 303, "YOEP": 2007}.items():
        person[name] = [np.nan, value]
    for name in ("PRCITSHP", "PENATVTY", "PEINUSYR", "A_AGE"):
        person[name] = person[name].astype(float)
        person.loc[1, name] = np.nan
    return _replace_person(frame, person)


def test_both_arrival_helpers_use_the_same_row_year():
    person = _mixed_boundary().person
    profile = rules._source_aware_immigration_profile(
        person, time_period=2024, observation_year_column=YEAR
    )
    assert profile.age_at_entry.tolist() == [15, 16]
    asec = person.iloc[:1]
    _, arrival, age_at_entry = rules._arrival_profile(
        asec, time_period=2024, observation_year_column=YEAR
    )
    assert arrival.tolist() == [2007]
    assert age_at_entry.tolist() == [15]
    _, _, legacy = rules._arrival_profile(asec, time_period=2024)
    assert legacy.tolist() == [16]


@pytest.mark.parametrize(
    "value",
    [
        None,
        pd.NA,
        np.nan,
        np.inf,
        True,
        np.bool_(False),
        2024.5,
        "2025",
        0,
        1899,
        10000,
    ],
)
def test_explicit_years_refuse_missing_boolean_fractional_or_coerced(value):
    person = _mixed_boundary().person.copy()
    person[YEAR] = pd.Series([value, 2024], dtype=object)
    with pytest.raises(ValueError, match="observation years"):
        rules._source_aware_immigration_profile(
            person, time_period=2024, observation_year_column=YEAR
        )


@pytest.mark.parametrize("column", ["absent", "", True])
def test_explicit_year_column_must_resolve(column):
    with pytest.raises(ValueError, match="observation.year"):
        rules._source_aware_immigration_profile(
            _mixed_boundary().person, time_period=2024, observation_year_column=column
        )


def test_unmapped_year_join_is_refused():
    person = _mixed_boundary().person.copy()
    person[YEAR] = person.person_id.map({1: 2025})
    with pytest.raises(ValueError, match="observation years"):
        rules._source_aware_immigration_profile(
            person, time_period=2024, observation_year_column=YEAR
        )


def test_acs_entry_bound_is_its_own_observation_year():
    person = _mixed_boundary().person.copy()
    person.loc[1, "YOEP"] = 2025
    with pytest.raises(rules.SourceRuntimeError, match="YOEP"):
        rules._source_aware_immigration_profile(
            person, time_period=2025, observation_year_column=YEAR
        )


def test_row_year_alignment_preserves_arbitrary_ids_and_index_order():
    person = _mixed_boundary().person.copy()
    person["person_id"] = pd.Series([2**90 + 1, 2**90 + 2], dtype=object)
    person.index = pd.Index(["asec:17", "acs:17"], name="source_namespace")
    result = rules._source_aware_immigration_profile(
        person.iloc[::-1], time_period=2024, observation_year_column=YEAR
    )
    assert result.age_at_entry.tolist() == [16, 15]
    assert person.person_id.tolist() == [2**90 + 1, 2**90 + 2]


@pytest.mark.parametrize("code,year", [(27, 2021), (28, 2023)])
def test_current_asec_arrival_bins_remain_exact(code, year):
    # Official 2025 dictionary 6C-6 and cpsmar25.pdf 6C-6 both list 27/28,
    # not 29. Their code-28 endpoints differ; 2023 is an approximation for both.
    person = rule_tests._person_table(
        [rule_tests._noncitizen(PEINUSYR=code, **{YEAR: 2025})]
    )
    profile = rules._source_aware_immigration_profile(
        person, time_period=2024, observation_year_column=YEAR
    )
    _, arrival, _ = rules._arrival_profile(
        person, time_period=2024, observation_year_column=YEAR
    )
    assert profile.arrival_year.tolist() == arrival.tolist() == [year]


def test_2025_asec_does_not_invent_an_unpublished_arrival_code_29():
    person = rule_tests._person_table(
        [rule_tests._noncitizen(PEINUSYR=29, **{YEAR: 2025})]
    )
    with pytest.raises(rules.SourceRuntimeError, match="reviewed ASEC code domain"):
        rules._source_aware_immigration_profile(
            person, time_period=2024, observation_year_column=YEAR
        )


@pytest.mark.parametrize(
    "source,category,origin,codes,expected",
    [
        ("asec", "refugee", None, [459, 461], [True, False]),
        ("acs", "refugee", None, [459, 461], [True, False]),
        ("acs", "tps", "other_designated", [451, 464], [True, False]),
    ],
)
def test_reviewed_named_countries_do_not_expand_residual_groups(
    source, category, origin, codes, expected
):
    # https://www2.census.gov/programs-surveys/cps/techdocs/cpsmar25.pdf Appendix J:
    # 459 Zaire/DR Congo, 461 Zimbabwe. ACS2024CodeLists.xls Place of Birth:
    # 451 Sudan; 464 groups South Sudan with Tunisia and Western Sahara.
    frame = rule_tests._us_frame(
        [rule_tests._noncitizen(PENATVTY=code, PEINUSYR=28) for code in codes]
    )
    person = frame.person.copy()
    person["ssn_card_type"] = "OTHER_NON_CITIZEN"
    person["immigration_status_str"] = category.upper()
    person[YEAR] = 2025 if source == "asec" else 2024
    if source == "acs":
        person[["PRCITSHP", "PENATVTY", "PEINUSYR"]] = np.nan
        person["CIT"], person["POBP"], person["YOEP"] = 5, codes, 2022
    draw = rules.HumanitarianDraw(
        category=category,
        origin=origin,
        status=category.upper(),
        target=1.0,
        source="https://example.com/invented",
    )
    assert (
        rules.us_immigration_humanitarian_draw_mask(
            _replace_person(frame, person), draw, observation_year_column=YEAR
        ).tolist()
        == expected
    )


def test_wrapper_manifest_handler_and_assignment_share_explicit_year(monkeypatch):
    stage = rule_tests._stage_spec(workers=0.1, students=100, anchor=1)
    monkeypatch.setattr(rules, "us_immigration_stage_spec", lambda: stage)
    frame = rule_tests._us_frame(
        [rule_tests._noncitizen(A_AGE=33, PEINUSYR=20, A_LFSR=1, **{YEAR: 2025})]
    )
    explicit = rules.with_us_immigration_inputs(
        frame, seed=5, time_period=2024, observation_year_column=YEAR
    )
    scalar = rules.with_us_immigration_inputs(frame, seed=5, time_period=2025)
    legacy = rules.with_us_immigration_inputs(frame, seed=5, time_period=2024)
    pd.testing.assert_frame_equal(explicit.person, scalar.person)
    assert explicit.person.immigration_status_str.tolist() == ["DACA"]
    assert legacy.person.immigration_status_str.tolist() == ["LEGAL_PERMANENT_RESIDENT"]
    assert explicit.person.A_AGE.tolist() == [33]


def test_immutable_boundary_survives_mixed_reconciliation():
    frame = _mixed_boundary()
    kwargs = dict(
        weights=frame.resolve_weights("person").values,
        mutable_rows=np.array([False, True]),
        seed=0,
        controls=transfer_tests._no_humanitarian_controls(),
    )
    result, receipt = rules.reconcile_us_immigration_humanitarian_transfer(
        frame.person, observation_year_column=YEAR, **kwargs
    )
    assert result.immigration_status_str.tolist() == [
        "DACA",
        "LEGAL_PERMANENT_RESIDENT",
    ]
    assert receipt["observation_years"] == {
        "mode": "per_row",
        "column": YEAR,
        "row_counts": {"2024": 1, "2025": 1},
    }
    with pytest.raises(ValueError, match="Immutable ASEC"):
        rules.reconcile_us_immigration_humanitarian_transfer(frame.person, **kwargs)


def test_manifest_humanitarian_assignment_uses_row_years(monkeypatch):
    stage = rule_tests._stage_spec(
        workers=100,
        students=100,
        anchor=1,
        humanitarian=rule_tests._humanitarian_block(deportation_withheld=1),
    )
    monkeypatch.setattr(rules, "us_immigration_stage_spec", lambda: stage)
    frame = rule_tests._us_frame(
        [
            rule_tests._noncitizen(A_AGE=33, PEINUSYR=20, CAID=1, **{YEAR: year})
            for year in (2025, 2024)
        ]
    )
    output = rules.with_us_immigration_inputs(
        frame, seed=0, time_period=2024, observation_year_column=YEAR
    )
    assert output.person.immigration_status_str.tolist() == [
        "LEGAL_PERMANENT_RESIDENT",
        "DEPORTATION_WITHHELD",
    ]


def test_candidate_and_selection_replay_honor_year_boundary():
    frame = _mixed_boundary()
    person = frame.person.copy()
    person["ssn_card_type"] = "OTHER_NON_CITIZEN"
    person["immigration_status_str"] = "DEPORTATION_WITHHELD"
    frame = _replace_person(frame, person)
    draw = rules.HumanitarianDraw(
        category="deportation_withheld",
        origin=None,
        status="DEPORTATION_WITHHELD",
        target=1.0,
        source="https://example.com/invented",
    )
    assert rules.us_immigration_humanitarian_draw_mask(
        frame, draw, observation_year_column=YEAR
    ).tolist() == [False, True]
    controls = replace(transfer_tests._no_humanitarian_controls(), humanitarian=(draw,))
    masks = rules.us_immigration_humanitarian_transfer_selection_masks(
        frame,
        mutable_rows=np.ones(2, dtype=bool),
        seed=0,
        controls=controls,
        observation_year_column=YEAR,
    )
    assert masks[draw.label].tolist() == [False, True]
    baseline = frame.person.copy()
    baseline["immigration_status_str"] = "LEGAL_PERMANENT_RESIDENT"
    emitted, receipt = rules.reconcile_us_immigration_humanitarian_transfer(
        baseline,
        weights=frame.resolve_weights("person").values,
        mutable_rows=np.ones(2, dtype=bool),
        seed=0,
        controls=controls,
        observation_year_column=YEAR,
    )
    bound = rules.us_immigration_humanitarian_transfer_selection_masks(
        _replace_person(frame, emitted),
        mutable_rows=np.ones(2, dtype=bool),
        seed=0,
        controls=controls,
        observation_year_column=YEAR,
        reconciliation_receipt=receipt,
    )
    np.testing.assert_array_equal(bound[draw.label], masks[draw.label])


@pytest.mark.parametrize("change", ["weights", "mutable_rows", "seed"])
def test_selection_replay_receipt_refuses_changed_numerical_inputs(change):
    frame = _mixed_boundary()
    weights = frame.resolve_weights("person").values
    mutable = np.array([False, True])
    controls = transfer_tests._no_humanitarian_controls()
    result, receipt = rules.reconcile_us_immigration_humanitarian_transfer(
        frame.person,
        weights=weights,
        mutable_rows=mutable,
        seed=13,
        controls=controls,
        observation_year_column=YEAR,
    )
    replay = _replace_person(frame, result)
    kwargs = dict(
        mutable_rows=mutable,
        seed=13,
        controls=controls,
        observation_year_column=YEAR,
        reconciliation_receipt=receipt,
    )
    assert (
        rules.us_immigration_humanitarian_transfer_selection_masks(replay, **kwargs)
        == {}
    )
    assert receipt["time_period"] is None  # Explicit years, not a claimed scalar 2024.
    if change == "weights":
        from microcosm.frame import Weights

        replay = Frame(
            {e: replay.table(e) for e in replay.entities},
            replay.schema,
            {
                "household": Weights(
                    np.array([2.0, 1.0]), frame.weights_for("household").kind
                )
            },
            replay.strata,
            mass_log=replay.mass_log,
            metadata=replay.metadata,
        )
    elif change == "mutable_rows":
        kwargs["mutable_rows"] = ~mutable
    else:
        kwargs["seed"] = 14
    with pytest.raises(ValueError, match="receipt disagrees"):
        rules.us_immigration_humanitarian_transfer_selection_masks(replay, **kwargs)


def test_composition_gate_uses_explicit_year_and_omission_is_legacy():
    frame = rule_tests._composition_frame(humanitarian={"DACA": 1})
    person = frame.person.copy()
    person[YEAR] = 2025
    person.loc[person.immigration_status_str.eq("DACA"), ["A_AGE", "PEINUSYR"]] = [
        33,
        20,
    ]
    frame = _replace_person(frame, person)
    explicit = rules.us_immigration_composition_gate(
        frame, controls=rule_tests._plausible_controls(), observation_year_column=YEAR
    )
    assert explicit.passed, explicit.failures
    scalar = rules.us_immigration_composition_gate(
        frame, controls=rule_tests._plausible_controls(), time_period=2025
    )
    assert scalar.passed, scalar.failures
    legacy = rules.us_immigration_composition_gate(
        frame, controls=rule_tests._plausible_controls()
    )
    assert not legacy.passed
    assert any("DACA cohort" in error for error in legacy.failures)


@pytest.mark.parametrize("source", ["asec", "acs"])
def test_no_arrival_evidence_features_do_not_reveal_observation_year(source):
    person = rule_tests._us_frame([{"PRCITSHP": 1, "PENATVTY": 57, "PEINUSYR": 0}])
    if source == "acs":
        people = person.person.copy()
        people[["PRCITSHP", "PENATVTY", "PEINUSYR"]] = np.nan
        people["CIT"], people["POBP"], people["YOEP"] = 1, 6, np.nan
        person = _replace_person(person, people)
    frames = [
        _replace_person(person, person.person.assign(**{YEAR: year}))
        for year in (2024, 2025)
    ]
    features = [
        rules.us_immigration_evidence_features(frame, observation_year_column=YEAR)
        for frame in frames
    ]
    pd.testing.assert_frame_equal(features[0], features[1], check_exact=True)
    assert features[0].arrival_year.tolist() == [0.0]


def test_execution_contract_binds_arrival_feature_values(monkeypatch):
    before = transfer.acs_transfer_execution_contract_identity(targets=PAIR)
    assert before["schema_version"] == 4
    features = before["immigration_evidence_features"]
    assert features["no_arrival_evidence_value"] == 0
    assert features["asec_arrival_year_midpoints"]["28"] == 2023
    monkeypatch.setattr(
        rules, "_ARRIVAL_YEAR_MIDPOINTS", {**rules._ARRIVAL_YEAR_MIDPOINTS, 28: 2024}
    )
    changed = transfer.acs_transfer_execution_contract_identity(targets=PAIR)
    assert changed["sha256"] != before["sha256"]
    assert (
        changed["immigration_evidence_features"]["asec_arrival_year_midpoints"]["28"]
        == 2024
    )


def _transfer_inputs():
    donor = transfer_tests._with_columns(
        transfer_tests._donor_frame(),
        "person",
        {
            YEAR: [2025] * 8,
            "ssn_card_type": ["CITIZEN"] * 6 + ["NONE"] * 2,
            "immigration_status_str": ["CITIZEN"] * 6 + ["UNDOCUMENTED"] * 2,
        },
    )
    recipient = transfer_tests._with_columns(
        transfer_tests._recipient_frame(), "person", {YEAR: [2024] * 6}
    )
    return donor, recipient


def test_transfer_donor_projection_and_contract_bind_explicit_year():
    donor, _ = _transfer_inputs()
    assert (
        YEAR
        in transfer.acs_transfer_donor_requirements(
            donor, FAMILIES, observation_year_column=YEAR
        )["person"]
    )
    scalar = transfer.acs_transfer_execution_contract_identity(targets=PAIR)
    explicit = transfer.acs_transfer_execution_contract_identity(
        targets=PAIR, observation_year_column=YEAR
    )
    assert scalar["sha256"] != explicit["sha256"]
    assert "immigration_observation_year" not in scalar


@pytest.mark.parametrize(
    "change",
    [
        "donor_year",
        "recipient_year",
        "column_name",
        "person_namespace",
        "legacy_mode",
        "arrival_midpoint",
        "feature_version",
    ],
)
def test_bank_refuses_stale_year_identity_without_caller_summary(
    monkeypatch, tmp_path, change
):
    transfer_tests._lock_bank_fixture_threads(monkeypatch)
    monkeypatch.setattr(
        rules, "us_immigration_controls", transfer_tests._no_humanitarian_controls
    )
    donor, recipient = _transfer_inputs()

    def run(donor, recipient, bank=None, column=YEAR):
        return transfer.transfer_acs_inputs(
            recipient,
            donor,
            target_families=FAMILIES,
            donor_channel=None,
            seed=37,
            n_estimators=1,
            target_bank=bank,
            observation_year_column=column,
        )

    monolithic = run(donor, recipient)
    cold_bank = transfer_tests._bank_store(tmp_path / "bank")
    cold = run(donor, recipient, cold_bank)
    transfer_tests._assert_transfer_results_exact(cold, monolithic)
    warm_bank = transfer_tests._bank_store(tmp_path / "bank")
    transfer_tests._assert_transfer_results_exact(
        run(donor, recipient, warm_bank), monolithic
    )
    assert warm_bank.receipt()["targets"]["0"]["source"] == "checkpoint"
    column = YEAR
    if change == "donor_year":
        donor = transfer_tests._with_columns(donor, "person", {YEAR: [2024] * 8})
    elif change == "recipient_year":
        recipient = transfer_tests._with_columns(
            recipient, "person", {YEAR: [2025] * 6}
        )
    elif change == "column_name":
        column = "same_year_different_contract"
        donor = transfer_tests._with_columns(donor, "person", {column: [2025] * 8})
        recipient = transfer_tests._with_columns(
            recipient, "person", {column: [2024] * 6}
        )
    elif change == "person_namespace":
        recipient = transfer_tests._with_columns(
            recipient, "person", {"person_id": [2**90 + i for i in range(6)]}
        )
    elif change == "arrival_midpoint":
        monkeypatch.setattr(
            rules,
            "_ARRIVAL_YEAR_MIDPOINTS",
            {**rules._ARRIVAL_YEAR_MIDPOINTS, 28: 2024},
        )
    elif change == "feature_version":
        monkeypatch.setattr(rules, "_EVIDENCE_FEATURE_VERSION", "invented-next-version")
    else:
        column = None
    stale_bank = transfer_tests._bank_store(tmp_path / "bank")
    changed = run(donor, recipient, stale_bank, column)
    target = stale_bank.receipt()["targets"]["0"]
    assert target["source"] != "checkpoint"
    assert target["load_status"] == "invalid_rebuild"
    assert "pattern" in target["invalid_checkpoint"]["message"].lower()
    transfer_tests._assert_transfer_results_exact(
        changed, run(donor, recipient, column=column)
    )


@pytest.mark.parametrize("banked", [False, True])
def test_transfer_immutable_asec_boundary_and_acs_surface(
    monkeypatch, tmp_path, banked
):
    transfer_tests._lock_bank_fixture_threads(monkeypatch)
    monkeypatch.setattr(
        rules, "us_immigration_controls", transfer_tests._no_humanitarian_controls
    )
    donor, _ = _transfer_inputs()
    recipient = transfer_tests._mixed_immigration_recipient()
    person = recipient.person.copy()
    person[YEAR] = [2025, 2024, 2024, 2024, 2024, 2024]
    person.loc[
        person.index[0],
        ["age", "PENATVTY", "PEINUSYR", "ssn_card_type", "immigration_status_str"],
    ] = [33, 303, 20, "NON_CITIZEN_VALID_EAD", "DACA"]
    recipient = _replace_person(recipient, person)
    result = transfer.transfer_acs_inputs(
        recipient,
        donor,
        target_families=FAMILIES,
        donor_channel=None,
        seed=37,
        n_estimators=1,
        target_bank=transfer_tests._bank_store(tmp_path / "bank") if banked else None,
        observation_year_column=YEAR,
    )
    assert result.frame.person.iloc[0]["immigration_status_str"] == "DACA"
    assert result.imputed_inputs[0].reconciliation["observation_years"][
        "row_counts"
    ] == {"2024": 5, "2025": 1}
    assert all(
        YEAR not in pattern.predictors
        for item in result.imputed_inputs
        for pattern in item.patterns
    )


def test_row_year_and_feature_identity_do_not_change_pattern_seeds(monkeypatch):
    transfer_tests._lock_bank_fixture_threads(monkeypatch)
    monkeypatch.setattr(
        rules, "us_immigration_controls", transfer_tests._no_humanitarian_controls
    )
    donor, recipient = _transfer_inputs()
    kwargs = dict(target_families=FAMILIES, donor_channel=None, seed=37, n_estimators=1)
    scalar = transfer.transfer_acs_inputs(recipient, donor, **kwargs)
    explicit = transfer.transfer_acs_inputs(
        recipient, donor, observation_year_column=YEAR, **kwargs
    )

    def seeds(result):
        return [
            pattern.seed for item in result.imputed_inputs for pattern in item.patterns
        ]

    assert seeds(scalar) == seeds(explicit)
