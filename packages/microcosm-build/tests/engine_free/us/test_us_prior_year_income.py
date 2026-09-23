"""Tests split from packages/microcosm-build/tests/test_us_prior_year_income.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_prior_year_income import *


def test_stage_manifest_pins_archived_join_signedness_and_puf_qrf() -> None:
    spec = us_prior_year_income_stage_spec()

    assert spec.stage == "prior_year_income"
    assert spec.outputs == US_PRIOR_YEAR_INCOME_OUTPUT_COLUMNS
    assert spec.nonnegative_outputs == ("employment_income_last_year",)
    operations = {operation.kind: operation for operation in spec.operations}
    assert tuple(operations) == (
        "read_table",
        "derive_prior_year_income",
        "impute_prior_year_income_to_puf_support",
    )
    derive = operations["derive_prior_year_income"].parameters
    assert derive["person_id"] == "PERIDNUM"
    assert derive["prior_year_offset"] == -1
    assert derive["employment_allocation_flag"] == "I_ERNVAL"
    assert derive["self_employment_allocation_flag"] == "I_SEVAL"
    assert derive["unallocated_flag"] == 0
    assert derive["sentinels"] == [-1, -9999]
    assert derive["fallback_to_current"] is True
    assert derive["no_prior_artifact"] == "leave_defaults"
    qrf = operations["impute_prior_year_income_to_puf_support"].parameters
    assert qrf["predictors"] == [
        "age",
        "is_male",
        "has_esi",
        "tax_unit_is_joint",
        "tax_unit_count_dependents",
        "employment_income",
        "self_employment_income",
        "social_security",
    ]
    assert qrf["outputs"] == [
        "employment_income_last_year",
        "self_employment_income_last_year",
    ]
    assert qrf["max_train_samples"] == 5_000
    assert qrf["n_estimators"] == 100
    assert qrf["weight"] == "person_weight"
    assert all(
        url.startswith(
            "https://github.com/PolicyEngine/policyengine-" + "us-data/blob/"
            "42ed5d45c56df80d754fbe24cce21cfeb8d05cbe/"
        )
        for url in (
            PRIOR_YEAR_INCOME_ARCHIVED_DERIVATION_URL,
            PRIOR_YEAR_INCOME_ARCHIVED_PUF_OUTPUTS_URL,
            PRIOR_YEAR_INCOME_ARCHIVED_PUF_IMPUTATION_URL,
            PRIOR_YEAR_INCOME_ARCHIVED_PUF_SPLICE_URL,
            PRIOR_YEAR_INCOME_ARCHIVED_FORMULA_OUTPUT_URL,
            PRIOR_YEAR_INCOME_ARCHIVED_FINALIZER_URL,
        )
    )


def test_adjacent_join_matches_flags_sentinels_fallback_and_signed_losses() -> None:
    result = with_us_prior_year_income_inputs(
        _source_frame(), seed=0, time_period=2024
    ).table("person")

    assert result["previous_year_income_available"].tolist() == [
        False,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
    ]
    assert result["employment_income_last_year"].tolist() == [
        0,
        100,
        200,
        10,
        400,
        300,
        500,
        600,
    ]
    assert result["self_employment_income_last_year"].tolist() == [
        0,
        -20,
        30,
        20,
        -5,
        0,
        50,
        60,
    ]


def test_valid_zero_prior_values_are_available_not_fallbacks() -> None:
    frame = _frame(
        pd.DataFrame(
            {
                "source_year": [2023, 2024],
                "PERIDNUM": ["0000000000000000000001"] * 2,
                "WSAL_VAL": [0, 50_000],
                "SEMP_VAL": [0, 10_000],
                "I_ERNVAL": [0, 0],
                "I_SEVAL": [0, 0],
            }
        )
    )

    person = with_us_prior_year_income_inputs(frame, seed=0, time_period=2024).table(
        "person"
    )

    assert person["previous_year_income_available"].tolist() == [False, True]
    assert person["employment_income_last_year"].tolist() == [0, 0]
    assert person["self_employment_income_last_year"].tolist() == [0, 0]


def test_existing_default_outputs_rederive_when_raw_sources_remain() -> None:
    frame = _source_frame()
    person = frame.table("person").copy()
    person["employment_income_last_year"] = 0.0
    person["self_employment_income_last_year"] = 0.0
    person["previous_year_income_available"] = False
    stale = module._replace_person_table(frame, person)

    result = with_us_prior_year_income_inputs(stale, seed=0, time_period=2024).table(
        "person"
    )

    assert result["previous_year_income_available"].any()
    assert result["self_employment_income_last_year"].tolist() != [0.0] * len(result)


def test_join_rejects_duplicate_source_year_person_key() -> None:
    frame = _source_frame()
    person = frame.table("person").copy()
    duplicate = person.iloc[[0]].copy()
    duplicate["person_id"] = 999
    duplicate["person_household_id"] = 999
    duplicate["person_tax_unit_id"] = 999
    duplicate["person_spm_unit_id"] = 999
    duplicate["person_family_id"] = 999
    duplicate["person_marital_unit_id"] = 999
    operation = us_prior_year_income_stage_spec().operations[1]

    with pytest.raises(SourceRuntimeError, match="duplicate key"):
        derive_us_prior_year_income_from_manifest(
            pd.concat([person, duplicate], ignore_index=True), operation, None
        )


def test_join_refuses_missing_allocation_source() -> None:
    person = _source_frame().table("person").drop(columns=["I_SEVAL"])
    operation = us_prior_year_income_stage_spec().operations[1]

    with pytest.raises(SourceRuntimeError, match="I_SEVAL"):
        derive_us_prior_year_income_from_manifest(person, operation, None)


def test_puf_support_joint_qrf_is_weighted_signed_and_drops_formula_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "QRF", _QRF)
    _QRF.calls.clear()
    direct = with_us_prior_year_income_inputs(_source_frame(), seed=7, time_period=2024)
    expanded = clone_us_frame_for_puf_support(direct)

    result = with_us_prior_year_income_inputs(expanded, seed=7, time_period=2024)
    person = result.table("person")
    assert "employment_income_last_year" not in person
    channel = person["person_support_channel"].astype(str)
    puf_values = person.loc[
        channel == PUF_TAX_DETAIL_SUPPORT_CHANNEL,
        "self_employment_income_last_year",
    ].to_numpy()
    assert puf_values[:2].tolist() == [125.0, -25.0]
    asec_values = person.loc[
        channel == BASE_ASEC_SUPPORT_CHANNEL,
        "self_employment_income_last_year",
    ].to_numpy()
    assert asec_values.tolist() == [0, -20, 30, 20, -5, 0, 50, 60]

    assert len(_QRF.calls) == 1
    call = _QRF.calls[0]
    assert call["targets"] == [
        "employment_income_last_year",
        "self_employment_income_last_year",
    ]
    assert call["predictors"] == list(module._PUF_PREDICTORS)
    assert np.all(np.asarray(call["weights"]) > 0)
    assert call["kwargs"] == {"n_estimators": 100, "seed": 7}


def test_puf_qrf_rejects_zero_weight_capped_training_sample() -> None:
    n_asec = 5_001
    sampled = (
        pd.DataFrame(index=np.arange(n_asec)).sample(n=5_000, random_state=0).index
    )
    omitted = int(next(iter(set(range(n_asec)) - set(sampled))))
    weights = np.zeros(n_asec + 1, dtype=np.float64)
    weights[omitted] = 1.0
    person = pd.DataFrame(
        {
            "person_support_channel": [BASE_ASEC_SUPPORT_CHANNEL] * n_asec
            + [PUF_TAX_DETAIL_SUPPORT_CHANNEL],
            "person_weight": weights,
            "employment_income_last_year": np.ones(n_asec + 1),
            "self_employment_income_last_year": np.ones(n_asec + 1),
        }
    )
    for predictor in module._PUF_PREDICTORS:
        person[module._PUF_PREDICTOR_PREFIX + predictor] = 1.0
    operation = us_prior_year_income_stage_spec().operations[2]

    with pytest.raises(SourceRuntimeError, match="sampled training weights"):
        impute_us_prior_year_income_to_puf_support_from_manifest(
            person, operation, None
        )


def test_signal_gate_accepts_signed_source_signal_and_rejects_defaults() -> None:
    passing = us_prior_year_income_signal_gate(_signal_frame())
    assert passing.passed, passing.failures
    assert passing.details["self_employment_income_last_year_negative_rows"] > 0

    frame = _signal_frame()
    person = frame.table("person").copy()
    person["previous_year_income_available"] = False
    failing = us_prior_year_income_signal_gate(
        module._replace_person_table(frame, person)
    )
    assert not failing.passed
    assert "availability" in " ".join(failing.failures)


def test_sampled_rung_scales_only_prior_year_availability_floor() -> None:
    frame = _with_stack_manifest(
        _signal_frame_with_availability_rows(6),
        {"version": 4, "sample_fraction": 0.25},
    )

    gate = us_prior_year_income_signal_gate(frame)

    assert gate.passed, gate.failures
    assert gate.details["previous_year_income_available_share"] == pytest.approx(
        0.04101010101010102
    )
    assert gate.details["previous_year_income_available_share_band"] == [0.05, 0.50]
    assert gate.details[
        "previous_year_income_available_sampled_match_survival_factor"
    ] == pytest.approx(0.25)
    assert gate.details[
        "previous_year_income_available_applied_floor"
    ] == pytest.approx(0.0125)
    assert gate.details["previous_year_income_available_applied_share_band"] == [
        0.0125,
        0.50,
    ]
    assert gate.details["self_employment_income_last_year_nonzero_share_band"] == [
        0.01,
        0.25,
    ]


def test_sampled_rung_preserves_applied_floor_and_authored_upper_bound() -> None:
    below_floor = us_prior_year_income_signal_gate(
        _with_stack_manifest(
            _signal_frame_with_availability_rows(1),
            {"version": 4, "sample_fraction": 0.25},
        )
    )
    above_upper = us_prior_year_income_signal_gate(
        _with_stack_manifest(
            _signal_frame_with_availability_rows(60),
            {"version": 4, "sample_fraction": 0.25},
        )
    )

    assert any("outside [0.012500, 0.500000]" in row for row in below_floor.failures)
    assert any("outside [0.012500, 0.500000]" in row for row in above_upper.failures)


def test_full_rung_gate_manifest_is_byte_identical_to_legacy_gate() -> None:
    frame = _signal_frame_with_availability_rows(6)
    legacy = us_prior_year_income_signal_gate(frame)
    full_rung = us_prior_year_income_signal_gate(
        _with_stack_manifest(
            frame,
            {"version": 4, "sample_fraction": 1.0},
        )
    )

    def manifest_bytes(gate) -> bytes:
        return json.dumps(
            GateReport((gate,)).to_manifest(),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

    assert full_rung == legacy
    assert manifest_bytes(full_rung) == manifest_bytes(legacy)
    assert not any(
        "applied" in key or "survival_factor" in key for key in full_rung.details
    )


def test_legacy_acs_only_sampling_does_not_scale_asec_match_floor() -> None:
    frame = _signal_frame_with_availability_rows(6)
    legacy = us_prior_year_income_signal_gate(frame)
    pilot = us_prior_year_income_signal_gate(
        _with_stack_manifest(
            frame,
            {"version": 1, "acs_sample_fraction": 0.25},
        )
    )

    assert not legacy.passed
    assert pilot == legacy


@pytest.mark.parametrize(
    "manifest",
    [
        {"version": 4},
        {"version": 4, "sample_fraction": True},
        {"version": 4, "sample_fraction": 0.0},
        {"version": 3, "sample_fraction": 0.25},
        "malformed",
    ],
)
def test_signal_gate_rejects_malformed_stacked_sampling_metadata(
    manifest: object,
) -> None:
    with pytest.raises(ValueError, match="prior-year-income availability"):
        us_prior_year_income_signal_gate(
            _with_stack_manifest(_signal_frame(), manifest)
        )


def test_clone_availability_checks_all_assembled_clones_and_legacy_pairs() -> None:
    assembled = _frame(
        pd.DataFrame(
            {
                "person_source_id": [10, 10, 10, 20, 20, 20],
                "person_spine_source_id": [1, 1, 1, 2, 2, 2],
                "person_support_channel": ["acs"] * 6,
                "person_support_clone_index": [0, 1, 2, 0, 1, 2],
                "self_employment_income_last_year": [10, 10, 10, -5, -5, -5],
                "previous_year_income_available": [True, True, False] + [False] * 3,
            }
        )
    )
    assembled_summary = module.us_prior_year_income_summary(assembled)
    assert assembled_summary["clone_availability_mismatches"] == 1
    assembled_gate = us_prior_year_income_signal_gate(assembled)
    assert any("1 source person" in failure for failure in assembled_gate.failures)

    legacy = _frame(
        pd.DataFrame(
            {
                "person_source_id": [10, 10, 20, 20],
                "person_support_channel": [
                    BASE_ASEC_SUPPORT_CHANNEL,
                    PUF_TAX_DETAIL_SUPPORT_CHANNEL,
                ]
                * 2,
                "self_employment_income_last_year": [10, 10, -5, -5],
                "previous_year_income_available": [True, True, False, False],
            }
        )
    )
    legacy_summary = module.us_prior_year_income_summary(legacy)
    assert legacy_summary["clone_availability_mismatches"] == 0


def test_source_reconciliation_detects_plausible_but_wrong_asec_carry() -> None:
    derived = with_us_prior_year_income_inputs(
        _source_frame(), seed=0, time_period=2024
    )
    passing = us_prior_year_income_source_reconciliation_gate(derived)
    assert passing.passed, passing.failures

    person = derived.table("person").copy()
    person.loc[1, "self_employment_income_last_year"] = 30.0
    corrupted = module._replace_person_table(derived, person)
    failing = us_prior_year_income_source_reconciliation_gate(corrupted)
    assert not failing.passed
    assert failing.details["mismatch_counts"] == {
        "employment_income_last_year": 0,
        "self_employment_income_last_year": 1,
        "previous_year_income_available": 0,
    }


def test_release_contract_promotes_both_persisted_inputs_without_wage_formula() -> None:
    manifest = load_release_input_coverage_manifest()
    for column in US_PRIOR_YEAR_INCOME_PERSISTED_OUTPUT_COLUMNS:
        assert column in RESTORED_REFERENCE_ECPS_REQUIRED_INPUTS
        assert column in US_RELEASE_REQUIRED_PERSON_SOURCE_COLUMNS
        assert column in manifest.required_columns
        assert column not in manifest.reviewed_exclusions
    assert US_PRIOR_YEAR_INCOME_NONCONSTANT_PERSON_COLUMNS == (
        "self_employment_income_last_year",
        "previous_year_income_available",
    )
    assert "employment_income_last_year" not in manifest.required_columns

    probe = next(
        probe
        for probe in us_release_reform_coverage_probes()
        if probe.id == "prior_year_self_employment_neutralization"
    )
    assert probe.neutralized_variable == "self_employment_income_last_year"
    assert probe.parameter_changes == {}
    assert probe.binding_inputs == ("self_employment_income_last_year",)
    assert probe.budget_measure == "tax_unit_earned_income_last_year"
    assert probe.effect_direction == "baseline_minus_reform"
    assert probe.expected_sign == "positive"
