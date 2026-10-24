"""Tests split from packages/microcosm-build/tests/test_us_other_health_insurance.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_other_health_insurance import *


def test_archive_urls_pin_exact_derivation_outputs_predictors_and_splice() -> None:
    commit = "42ed5d45c56df80d754fbe24cce21cfeb8d05cbe"
    urls = (
        OTHER_HEALTH_INSURANCE_ARCHIVED_DERIVATION_URL,
        OTHER_HEALTH_INSURANCE_ARCHIVED_PUF_OUTPUTS_URL,
        OTHER_HEALTH_INSURANCE_ARCHIVED_PUF_PREDICTORS_URL,
        OTHER_HEALTH_INSURANCE_ARCHIVED_PUF_IMPUTATION_URL,
        OTHER_HEALTH_INSURANCE_ARCHIVED_PUF_SPLICE_URL,
    )
    assert all(commit in url for url in urls)
    assert OTHER_HEALTH_INSURANCE_ARCHIVED_DERIVATION_URL.endswith(
        "datasets/cps/cps.py#L828-L944"
    )
    assert OTHER_HEALTH_INSURANCE_ARCHIVED_PUF_OUTPUTS_URL.endswith(
        "datasets/cps/extended_cps.py#L135-L194"
    )
    assert OTHER_HEALTH_INSURANCE_ARCHIVED_PUF_PREDICTORS_URL.endswith(
        "datasets/cps/extended_cps.py#L234-L248"
    )
    assert OTHER_HEALTH_INSURANCE_ARCHIVED_PUF_IMPUTATION_URL.endswith(
        "datasets/cps/extended_cps.py#L639-L745"
    )
    assert OTHER_HEALTH_INSURANCE_ARCHIVED_PUF_SPLICE_URL.endswith(
        "datasets/cps/extended_cps.py#L1014-L1076"
    )


def test_stage_manifest_pins_residual_and_joint_puf_qrf() -> None:
    spec = us_other_health_insurance_stage_spec()

    assert spec.stage == "other_health_insurance_premiums"
    assert spec.survey == "Census CPS ASEC"
    assert spec.source == "https://www.census.gov/programs-surveys/cps.html"
    assert spec.grain == "person"
    assert spec.outputs == US_OTHER_HEALTH_INSURANCE_STAGE_OUTPUT_COLUMNS
    assert spec.outputs == (
        *US_OTHER_HEALTH_INSURANCE_OUTPUT_COLUMNS,
        *US_SE_HEALTH_ATTRIBUTION_OUTPUT_COLUMNS,
    )
    assert spec.nonnegative_outputs == (
        *US_OTHER_HEALTH_INSURANCE_OUTPUT_COLUMNS,
        US_SE_HEALTH_ATTRIBUTION_OUTPUT_COLUMNS[0],
    )
    assert US_OTHER_HEALTH_INSURANCE_NONCONSTANT_PERSON_COLUMNS == (_OTHER,)
    assert [operation.kind for operation in spec.operations] == [
        "read_table",
        "derive_other_health_insurance_premiums",
        "impute_other_health_insurance_premiums_to_puf_support",
        "attribute_self_employed_health_premiums",
    ]
    assert spec.operations[0].parameters == {
        "table": "person",
        "weight": "person_weight",
    }
    assert spec.operations[1].parameters == {
        "reported_source": _REPORTED,
        "chip_premium_source": "chip_premium",
        "marketplace_net_premium_source": "marketplace_net_premium",
        "medicaid_premium_source": "medicaid_premium",
        "output": _OTHER,
    }
    assert spec.operations[2].parameters == {
        "predictors": list(_PREDICTORS),
        "max_train_samples": 5_000,
        "n_estimators": 100,
        "seed_from_build_config": True,
        "weight": "person_weight",
    }
    assert spec.operations[3].parameters == {
        "reported_source": _REPORTED,
        "self_employment_income_sources": list(
            US_SE_HEALTH_SELF_EMPLOYMENT_INCOME_SOURCES
        ),
        "medicare_age_source": "age",
        "medicare_age_threshold": US_SE_HEALTH_MEDICARE_AGE_THRESHOLD,
        "medicare_ssdi_source": "social_security_disability",
        "employer_coverage_source": "has_esi",
        "output_premiums": US_SE_HEALTH_ATTRIBUTION_OUTPUT_COLUMNS[0],
        "output_self_employed_flag": US_SE_HEALTH_ATTRIBUTION_OUTPUT_COLUMNS[1],
    }
    assert "PUF-only residual-order exceedances remain diagnostics" in spec.notes

    handlers = us_source_operation_handlers()
    assert (
        handlers["derive_other_health_insurance_premiums"]
        is derive_us_other_health_insurance_from_manifest
    )
    assert (
        handlers["impute_other_health_insurance_premiums_to_puf_support"]
        is impute_us_other_health_insurance_to_puf_support_from_manifest
    )
    assert (
        handlers["attribute_self_employed_health_premiums"]
        is attribute_us_se_health_premiums_from_manifest
    )


def test_asec_residual_is_exact_nonnegative_and_does_not_mutate_source() -> None:
    source = pd.DataFrame(
        {
            _REPORTED: [0.0, 100.0, 500.0, 1_000.0],
            "chip_premium": [0.0, 80.0, 100.0, 100.0],
            "marketplace_net_premium": [0.0, 30.0, 200.0, 200.0],
            "medicaid_premium": [0.0, 10.0, 300.0, 300.0],
        }
    )
    original = source.copy(deep=True)

    result = derive_us_other_health_insurance_from_asec(source)

    assert result[_OTHER].tolist() == [0.0, 0.0, 0.0, 400.0]
    pd.testing.assert_frame_equal(source, original)


@pytest.mark.parametrize(
    ("column", "bad_value", "message"),
    [
        (_REPORTED, np.nan, "nonnumeric or nonfinite"),
        ("chip_premium", np.inf, "nonnumeric or nonfinite"),
        ("marketplace_net_premium", -1.0, "negative"),
        ("medicaid_premium", -1.0, "negative"),
    ],
)
def test_asec_residual_fails_closed_on_invalid_sources(
    column: str,
    bad_value: float,
    message: str,
) -> None:
    source = pd.DataFrame(
        {
            _REPORTED: [1_000.0],
            "chip_premium": [0.0],
            "marketplace_net_premium": [0.0],
            "medicaid_premium": [0.0],
        }
    )
    source.loc[0, column] = bad_value

    with pytest.raises(SourceRuntimeError, match=message):
        derive_us_other_health_insurance_from_asec(source)


def test_with_inputs_preserves_measured_asec_and_carries_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "PolicyEngineUSEngine", _ZeroPremiumEngine)
    frame = _frame()
    original = frame.table("person").copy(deep=True)

    result = with_us_other_health_insurance_inputs(frame, seed=0, time_period=2024)

    pd.testing.assert_frame_equal(frame.table("person"), original)
    np.testing.assert_allclose(result.table("person")[_REPORTED], _REPORTED_VALUES)
    np.testing.assert_allclose(result.table("person")[_OTHER], _REPORTED_VALUES)
    gate = us_other_health_insurance_signal_gate(result)
    assert gate.passed, gate.failures


def test_tax_unit_premiums_are_allocated_wholly_to_first_person() -> None:
    frame = _frame()
    person = frame.table("person")
    person.loc[1, "person_tax_unit_id"] = person.loc[0, "person_tax_unit_id"]
    frame.table("tax_unit").drop(index=[1], inplace=True)

    allocated = module._tax_unit_values_on_first_person(
        frame,
        np.arange(1.0, frame.n("tax_unit") + 1.0) * 100.0,
        variable="chip_premium",
    )

    assert allocated[0] == 100.0
    assert allocated[1] == 0.0
    assert np.count_nonzero(allocated) == frame.n("tax_unit")


def test_modeled_premiums_batch_by_household_and_realign_tax_units(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _frame()
    person_tax_unit_ids = np.asarray(
        [210, 201, 209, 202, 208, 203, 207, 204, 206, 205],
        dtype=np.int64,
    )
    frame.table("person")["person_tax_unit_id"] = person_tax_unit_ids
    calls: list[np.ndarray] = []

    class IdPremiumEngine:
        def materialize(
            self,
            batch: Frame,
            variables: list[str],
            *,
            period: int,
        ) -> dict[str, np.ndarray]:
            assert period == 2024
            assert variables == list(
                module.US_OTHER_HEALTH_INSURANCE_MODELED_PREMIUM_VARIABLES
            )
            tax_unit_ids = batch.table("tax_unit")["tax_unit_id"].to_numpy()
            calls.append(tax_unit_ids.copy())
            factor = tax_unit_ids.astype(np.float64) - 200.0
            return {
                "chip_premium": factor,
                "marketplace_net_premium": factor * 2.0,
                "medicaid_premium": factor * 3.0,
            }

    monkeypatch.setattr(module, "PolicyEngineUSEngine", IdPremiumEngine)

    result = with_us_other_health_insurance_inputs(
        frame,
        seed=0,
        time_period=2024,
        maximum_microsim_batch_size=3,
    )

    assert [len(call) for call in calls] == [3, 3, 3, 1]
    assert set(np.concatenate(calls)) == set(range(201, 211))
    expected_modeled = (person_tax_unit_ids - 200.0) * 6.0
    np.testing.assert_allclose(
        result.table("person")[_OTHER],
        np.maximum(_REPORTED_VALUES - expected_modeled, 0.0),
    )
    gate = us_other_health_insurance_signal_gate(result)
    assert gate.passed, gate.failures


def test_joint_qrf_replaces_only_puf_and_does_not_invent_clipping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "PolicyEngineUSEngine", _ZeroPremiumEngine)
    expanded = clone_us_frame_for_puf_support(_frame())
    original = expanded.table("person").copy(deep=True)
    calls: dict[str, object] = {}

    class FakeFitted:
        def predict(self, test: pd.DataFrame, **kwargs) -> pd.DataFrame:
            calls["test"] = test.copy()
            reported = _REPORTED_VALUES.copy()
            other = reported.copy()
            other[1] = 1_500.0
            return pd.DataFrame(
                {_REPORTED: reported, _OTHER: other},
                index=test.index,
            )

    class FakeQRF:
        def __init__(self, **kwargs: object) -> None:
            calls["init"] = kwargs

        def fit(
            self,
            training: pd.DataFrame,
            predictors: list[str],
            targets: list[str],
            *,
            weights: np.ndarray,
        ) -> FakeFitted:
            calls["training"] = training.copy()
            calls["predictors"] = predictors
            calls["targets"] = targets
            calls["weights"] = weights.copy()
            return FakeFitted()

    monkeypatch.setattr(module, "QRF", FakeQRF)

    result = with_us_other_health_insurance_inputs(
        expanded,
        seed=17,
        time_period=2024,
    )

    assert calls["init"] == {"n_estimators": 100, "seed": 17}
    assert calls["predictors"] == list(_PREDICTORS)
    assert calls["targets"] == list(US_OTHER_HEALTH_INSURANCE_OUTPUT_COLUMNS)
    training = calls["training"]
    assert isinstance(training, pd.DataFrame)
    assert list(training.columns) == [
        *_PREDICTORS,
        *US_OTHER_HEALTH_INSURANCE_OUTPUT_COLUMNS,
    ]
    asec_mask = original["person_support_channel"] == "asec"
    np.testing.assert_allclose(
        calls["weights"],
        expanded.resolve_weights("person").values[asec_mask],
    )

    person = result.table("person")
    puf_mask = person["person_support_channel"] == "puf_tax_detail"
    np.testing.assert_allclose(person.loc[~puf_mask, _OTHER], _REPORTED_VALUES)
    assert person.loc[puf_mask, _OTHER].iloc[1] == 1_500.0
    assert person.loc[puf_mask, _REPORTED].iloc[1] == 1_000.0
    gate = us_other_health_insurance_signal_gate(result)
    assert gate.passed, gate.failures
    assert gate.details["other_exceeds_reported_by_channel"] == {
        "asec": 0,
        "puf_tax_detail": 1,
    }
    pd.testing.assert_frame_equal(expanded.table("person"), original)


def test_signal_gate_rejects_missing_default_and_measured_identity_violation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "PolicyEngineUSEngine", _ZeroPremiumEngine)
    valid = with_us_other_health_insurance_inputs(_frame(), seed=0, time_period=2024)
    assert us_other_health_insurance_signal_gate(valid).passed

    missing = with_us_other_health_insurance_inputs(_frame(), seed=0, time_period=2024)
    missing.table("person").drop(columns=[_OTHER], inplace=True)
    assert not us_other_health_insurance_signal_gate(missing).passed

    default = with_us_other_health_insurance_inputs(_frame(), seed=0, time_period=2024)
    default.table("person")[_OTHER] = 0.0
    assert not us_other_health_insurance_signal_gate(default).passed

    invalid = with_us_other_health_insurance_inputs(_frame(), seed=0, time_period=2024)
    invalid.table("person").loc[0, _OTHER] = 1.0
    gate = us_other_health_insurance_signal_gate(invalid)
    assert not gate.passed
    assert any("measured ASEC" in failure for failure in gate.failures)


def test_se_attribution_is_deterministic_reported_identity_with_guards() -> None:
    source = _attribution_source(
        reported=[5_000.0, 5_000.0, 5_000.0, 5_000.0, 0.0, 5_000.0, 5_000.0],
        self_employment=[30_000.0, 0.0, 30_000.0, 30_000.0, 30_000.0, 0.0, 30_000.0],
        sstb=[0.0, 0.0, 0.0, 0.0, 0.0, 12_000.0, 0.0],
        age=[40.0, 40.0, 65.0, 40.0, 40.0, 40.0, 40.0],
        ssdi=[0.0, 0.0, 0.0, 9_000.0, 0.0, 0.0, 0.0],
        esi=[False, False, False, False, False, False, True],
    )
    original = source.copy(deep=True)

    result = attribute_us_se_health_premiums(source)

    # Attribution copies the reported premium exactly for self-employed
    # people outside the Medicare proxy and outside measured employer
    # coverage (the 162(l)(2)(B) subsidized-plan exclusion proxy), and never
    # invents premium mass.
    assert result[_PREMIUMS_OUTPUT].tolist() == [
        5_000.0,  # self-employed, under 65, no SSDI, no employer coverage
        0.0,  # no self-employment income
        0.0,  # Medicare age proxy
        0.0,  # SSDI proxy
        0.0,  # self-employed with no reported premium
        5_000.0,  # SSTB self-employment income binds the same chain
        0.0,  # measured employer-sponsored coverage excludes the months
    ]
    assert result[_FLAG_OUTPUT].tolist() == [
        True,
        False,
        True,
        True,
        True,
        True,
        True,
    ]
    assert result[_FLAG_OUTPUT].dtype == bool
    pd.testing.assert_frame_equal(source, original)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda source: source.drop(
                columns=["sstb_self_employment_income_before_lsr"]
            ),
            "sstb_self_employment_income_before_lsr",
        ),
        (
            lambda source: source.drop(columns=["age"]),
            "age",
        ),
        (
            lambda source: source.drop(columns=["social_security_disability"]),
            "social_security_disability",
        ),
        (
            lambda source: source.assign(self_employment_income_before_lsr=[np.nan]),
            "nonnumeric or nonfinite",
        ),
        (
            lambda source: source.assign(age=[np.inf]),
            "nonnumeric or nonfinite",
        ),
        (
            lambda source: source.drop(columns=["has_esi"]),
            "has_esi",
        ),
        (
            lambda source: source.assign(has_esi=["True"]),
            "object dtype",
        ),
    ],
)
def test_se_attribution_fails_closed_on_missing_or_invalid_sources(
    mutate,
    message: str,
) -> None:
    source = _attribution_source(reported=[5_000.0], self_employment=[30_000.0])

    with pytest.raises(SourceRuntimeError, match=message):
        attribute_us_se_health_premiums(mutate(source))


def test_se_attribution_allows_negative_self_employment_losses() -> None:
    # Schedule C losses are real measured signal: the mask is strictly
    # positive net income, so loss rows keep the flag off without failing.
    source = _attribution_source(
        reported=[5_000.0, 5_000.0],
        self_employment=[-4_000.0, 3_000.0],
        sstb=[0.0, -8_000.0],
    )

    result = attribute_us_se_health_premiums(source)

    assert result[_PREMIUMS_OUTPUT].tolist() == [0.0, 0.0]
    assert result[_FLAG_OUTPUT].tolist() == [False, False]


def test_with_inputs_materializes_se_attribution_and_carries_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "PolicyEngineUSEngine", _ZeroPremiumEngine)

    result = with_us_other_health_insurance_inputs(_frame(), seed=0, time_period=2024)

    person = result.table("person")
    expected_premiums = np.where(
        (_SELF_EMPLOYMENT_VALUES > 0.0) & ~person["has_esi"].to_numpy(),
        _REPORTED_VALUES,
        0.0,
    )
    np.testing.assert_allclose(person[_PREMIUMS_OUTPUT], expected_premiums)
    assert person[_FLAG_OUTPUT].tolist() == (_SELF_EMPLOYMENT_VALUES > 0.0).tolist()
    assert person[_FLAG_OUTPUT].dtype == bool
    gate = us_other_health_insurance_signal_gate(result)
    assert gate.passed, gate.failures


def test_signal_gate_rejects_broken_or_absent_se_attribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "PolicyEngineUSEngine", _ZeroPremiumEngine)

    missing = with_us_other_health_insurance_inputs(_frame(), seed=0, time_period=2024)
    missing.table("person").drop(columns=[_PREMIUMS_OUTPUT], inplace=True)
    gate = us_other_health_insurance_signal_gate(missing)
    assert not gate.passed

    structural_zero = with_us_other_health_insurance_inputs(
        _frame(), seed=0, time_period=2024
    )
    structural_zero.table("person")[_PREMIUMS_OUTPUT] = 0.0
    gate = us_other_health_insurance_signal_gate(structural_zero)
    assert not gate.passed
    assert any(_PREMIUMS_OUTPUT in failure for failure in gate.failures)

    invented = with_us_other_health_insurance_inputs(_frame(), seed=0, time_period=2024)
    invented.table("person").loc[0, _PREMIUMS_OUTPUT] = 123.0
    gate = us_other_health_insurance_signal_gate(invented)
    assert not gate.passed
    assert any("attribution identity" in failure for failure in gate.failures)

    unflagged = with_us_other_health_insurance_inputs(
        _frame(), seed=0, time_period=2024
    )
    unflagged.table("person")[_FLAG_OUTPUT] = False
    gate = us_other_health_insurance_signal_gate(unflagged)
    assert not gate.passed


def test_signal_gate_rejects_malformed_flag_and_nonfinite_identity_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "PolicyEngineUSEngine", _ZeroPremiumEngine)

    # A null in the flag column must fail rather than coerce (astype(bool)
    # would silently read NaN as True).
    nan_flag = with_us_other_health_insurance_inputs(_frame(), seed=0, time_period=2024)
    person = nan_flag.table("person")
    person[_FLAG_OUTPUT] = person[_FLAG_OUTPUT].astype(object)
    carrier_row = person.index[person[_FLAG_OUTPUT].astype(bool)][0]
    person.loc[carrier_row, _FLAG_OUTPUT] = np.nan
    gate = us_other_health_insurance_signal_gate(nan_flag)
    assert not gate.passed
    assert any("non-boolean" in failure for failure in gate.failures)

    # A nonfinite identity source on a noncarrier must fail rather than be
    # silently read as not-self-employed.
    nan_income = with_us_other_health_insurance_inputs(
        _frame(), seed=0, time_period=2024
    )
    person = nan_income.table("person")
    noncarrier_row = person.index[~person[_FLAG_OUTPUT]][0]
    person.loc[noncarrier_row, "self_employment_income_before_lsr"] = np.nan
    gate = us_other_health_insurance_signal_gate(nan_income)
    assert not gate.passed
    assert any("nonfinite" in failure for failure in gate.failures)

    # Numeric-string flags coerce cleanly through to_numeric but the engine
    # parses the nonempty string "0" as True, so a string-typed column on a
    # healed artifact must fail outright.
    string_flag = with_us_other_health_insurance_inputs(
        _frame(), seed=0, time_period=2024
    )
    person = string_flag.table("person")
    person[_FLAG_OUTPUT] = np.where(person[_FLAG_OUTPUT], "1", "0")
    gate = us_other_health_insurance_signal_gate(string_flag)
    assert not gate.passed
    assert any("non-boolean" in failure for failure in gate.failures)


def test_legacy_two_column_surface_is_rebuilt_or_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "PolicyEngineUSEngine", _ZeroPremiumEngine)
    legacy = with_us_other_health_insurance_inputs(_frame(), seed=0, time_period=2024)
    legacy.table("person").drop(
        columns=list(US_SE_HEALTH_ATTRIBUTION_OUTPUT_COLUMNS), inplace=True
    )

    # With the measured reported premium still present, the stage re-runs in
    # full instead of passing the pre-attribution surface through untouched.
    rebuilt = with_us_other_health_insurance_inputs(
        legacy,
        seed=0,
        time_period=2024,
        allow_existing_without_source=True,
    )
    person = rebuilt.table("person")
    for column in US_SE_HEALTH_ATTRIBUTION_OUTPUT_COLUMNS:
        assert column in person
    assert us_other_health_insurance_signal_gate(rebuilt).passed

    # Without the measured source, a legacy two-column surface no longer
    # carries the full signal, so the heal path refuses it.
    unsourced = with_us_other_health_insurance_inputs(
        _frame(), seed=0, time_period=2024
    )
    unsourced_person = unsourced.table("person")
    unsourced_person.drop(
        columns=list(US_SE_HEALTH_ATTRIBUTION_OUTPUT_COLUMNS), inplace=True
    )
    unsourced_person.rename(columns={_REPORTED: "retired_reported"}, inplace=True)

    with pytest.raises(ValueError, match="cannot heal"):
        with_us_other_health_insurance_inputs(
            unsourced,
            seed=0,
            time_period=2024,
            allow_existing_without_source=True,
        )


def test_esi_exclusion_pins_complete_hermetic_source_unavailability_evidence() -> None:
    payload = json.loads(
        files("microcosm.build.us").joinpath("ecps_parity_known_gaps.json").read_text()
    )
    entry = payload["known_gaps"]["employer_sponsored_insurance_premiums"]
    evidence = entry["evidence"]

    assert entry["reason"].startswith("SOURCE UNAVAILABILITY WITH EVIDENCE:")
    assert evidence["classification"] == "source_unavailability"
    assert evidence["retired_derivation"] == {
        "repository_owner": "PolicyEngine",
        "repository_name_parts": ["policyengine-", "us-data"],
        "commit": "42ed5d45c56df80d754fbe24cce21cfeb8d05cbe",
        "path_parts": ["policyengine_", "us_data", "datasets", "cps", "cps.py"],
        "lines": "197-271,1575-1581",
    }
    assert evidence["optional_source_fields"]["lines"] == "13-55"
    assert evidence["required_columns"] == [
        "NOW_OWNGRP",
        "NOW_HIPAID",
        "NOW_GRPFTYP",
        "PHIP_VAL",
    ]
    evidence_hashes = {
        item["filename"]: item["sha256"] for item in evidence["hermetic_inputs"]
    }
    build_summary = json.loads(
        (ROOT / "experiments/build_j_recert/base_j.summary.json").read_text()
    )
    recorded_hashes = {
        Path(item["path"]).name: item["sha256"]
        for item in build_summary["base_source"]["sources"]
    }
    assert evidence_hashes == recorded_hashes
    assert set(evidence_hashes) == {
        "census_cps_2022.h5",
        "census_cps_2023.h5",
        "census_cps_2024.h5",
    }
    for item in evidence["hermetic_inputs"]:
        assert item["present_columns"] == ["PHIP_VAL"]
        assert item["missing_columns"] == [
            "NOW_OWNGRP",
            "NOW_HIPAID",
            "NOW_GRPFTYP",
        ]
    assert "buildj_base.sh lines 65-69" in evidence["hermetic_build_contract"]
    assert "base_j.summary.json lines 55-75" in evidence["hermetic_build_contract"]
    build_script = (ROOT / "experiments/build_j_recert/buildj_base.sh").read_text()
    for year in (2022, 2023, 2024):
        assert f'--asec-h5 {year}="$USD/census_cps_{year}.h5"' in build_script

    esi_stage = US_SOURCE_MANIFEST.stage_map()["meps_esi_premiums"]
    assignment = next(
        operation
        for operation in esi_stage.operations
        if operation.kind == "assign_by_plan_type"
    )
    assert assignment.parameters["inputs"] == [
        "NOW_OWNGRP",
        "NOW_HIPAID",
        "NOW_GRPFTYP",
        "PHIP_VAL",
    ]
    assert assignment.parameters["source_unavailable_when_missing"] == [
        "NOW_OWNGRP",
        "NOW_HIPAID",
        "NOW_GRPFTYP",
    ]
    forbidden_proxies = {"has_esi", "is_esi_dependent", "tax_unit_size", "state_fips"}
    assert forbidden_proxies.isdisjoint(assignment.parameters["inputs"])


def test_shipped_se_health_premium_neutralization_probe() -> None:
    from microcosm.build.us_runtime.release_input_coverage import (
        us_release_input_coverage_required_columns,
        us_release_reform_coverage_probes,
    )

    probe = next(
        probe
        for probe in us_release_reform_coverage_probes()
        if probe.id == "self_employed_health_premium_neutralization"
    )
    assert probe.neutralized_variable == _PREMIUMS_OUTPUT
    assert probe.binding_inputs == US_SE_HEALTH_ATTRIBUTION_OUTPUT_COLUMNS
    assert probe.budget_measure == "income_tax"
    assert probe.period == 2024
    assert probe.effect_direction == "baseline_minus_reform"
    assert probe.expected_sign == "negative"
    assert probe.min_abs_effect >= 100_000_000.0
    assert "PolicyEngine/microcosm#451" in probe.issue

    required = us_release_input_coverage_required_columns()
    for column in US_SE_HEALTH_ATTRIBUTION_OUTPUT_COLUMNS:
        assert column in required
