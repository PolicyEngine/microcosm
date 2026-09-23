"""Tests split from packages/microcosm-build/tests/test_us_housing_inputs.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_housing_inputs import *


def test_stage_manifest_pins_exact_archived_sources_and_two_qrfs() -> None:
    spec = us_housing_inputs_stage_spec()

    assert spec.stage == "acs_rent"
    assert spec.outputs == US_HOUSING_INPUTS_OUTPUT_COLUMNS
    assert [operation.kind for operation in spec.operations] == [
        "derive_housing_tenure_inputs",
        "read_acs_rent_donor",
        "fit_weighted_acs_rent_qrf",
        "impute_housing_assistance_to_puf_support",
    ]
    rent_fit = spec.operations[2]
    assert tuple(rent_fit.parameters["predictors"]) == module.ACS_RENT_PREDICTORS
    assert rent_fit.parameters["max_train_samples"] == 10_000
    assert rent_fit.parameters["shared_sample_targets"] == [
        "rent",
        "real_estate_taxes",
    ]
    assert rent_fit.parameters["per_target_initial_cap"] == 5_000
    assert rent_fit.parameters["weight"] == "household_weight"
    puf_fit = spec.operations[3]
    assert puf_fit.parameters["max_train_samples"] == 5_000
    assert puf_fit.parameters["reduction"] == "value_from_first_person"
    assert puf_fit.parameters["take_up_output"] == (
        "takes_up_housing_assistance_if_eligible"
    )
    direct = spec.operations[0]
    assert direct.parameters["housing_take_up_assignment"] == (
        "equal_to_measured_receipt"
    )
    assert all(
        url.startswith(
            "https://github.com/PolicyEngine/" + "policyengine-" + "us-data/blob/"
            "42ed5d45c56df80d754fbe24cce21cfeb8d05cbe/"
        )
        for url in (
            HOUSING_INPUTS_ARCHIVED_CPS_RENT_URL,
            HOUSING_INPUTS_ARCHIVED_CPS_SPM_URL,
            HOUSING_INPUTS_ARCHIVED_ACS_DERIVATION_URL,
            HOUSING_INPUTS_ARCHIVED_PUF_IMPUTATION_URL,
            HOUSING_TAKE_UP_ARCHIVED_DERIVATION_URL,
            HOUSING_TAKE_UP_ARCHIVED_PARAMETER_URL,
            HOUSING_TAKE_UP_ARCHIVED_HUD_ETL_URL,
        )
    )
    artifact = next(artifact for artifact in spec.artifacts if artifact.get("sha256"))
    assert artifact["sha256"] == ACS_2022_RENT_ARTIFACT_SHA256
    assert HOUSING_TAKE_UP_ARCHIVED_DERIVATION_URL.endswith(
        "/datasets/cps/cps.py#L664-L682"
    )
    assert HOUSING_TAKE_UP_ARCHIVED_PARAMETER_URL.endswith(
        "/parameters/take_up/housing_assistance.yaml#L1-L15"
    )
    assert HOUSING_TAKE_UP_ARCHIVED_HUD_ETL_URL.endswith(
        "/db/etl_housing_assistance.py#L35-L169"
    )


def test_direct_asec_mappings_are_exact() -> None:
    result = derive_us_housing_inputs(_frame())

    household = result.table("household")
    assert household["tenure_type"].tolist()[:6] == [
        "RENTED",
        "RENTED",
        "RENTED",
        "RENTED",
        "NONE",
        "OWNED_WITH_MORTGAGE",
    ]
    spm = result.table("spm_unit")
    assert spm["receives_housing_assistance"].tolist().count(True) == 1
    pd.testing.assert_series_equal(
        spm["takes_up_housing_assistance_if_eligible"],
        spm["receives_housing_assistance"],
        check_names=False,
    )
    assert spm["spm_unit_tenure_type"].tolist()[:3] == [
        "RENTER",
        "OWNER_WITH_MORTGAGE",
        "OWNER_WITHOUT_MORTGAGE",
    ]


def test_direct_mapping_rejects_inconsistent_spm_replicas() -> None:
    frame = _frame()
    extra = frame.table("person").iloc[[0]].copy()
    extra["person_id"] = 999
    extra["SPM_CAPHOUSESUB"] = 0.0
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["person"] = pd.concat([tables["person"], extra], ignore_index=True)
    broken = Frame(tables, frame.schema, {"household": frame.weights_for("household")})

    with pytest.raises(ValueError, match="constant within its SPM unit"):
        derive_us_housing_inputs(broken)


def test_materialization_places_rent_only_on_renter_heads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "QRF", _RentQRF)

    result = with_us_housing_inputs(
        _frame(),
        seed=0,
        time_period=2024,
        acs_rent_donor=_donor(),
    )

    rent = result.table("person")["pre_subsidy_rent"]
    assert rent.tolist()[:6] == [12_000.0] * 4 + [0.0, 0.0]
    gate = us_housing_inputs_signal_gate(result)
    assert gate.passed, gate.failures


def test_rent_fit_replays_archived_joint_target_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    donor = _donor(12_000)
    positions = np.arange(len(donor))
    donor["rent_is_allocated"] = positions >= 6_000
    donor["real_estate_taxes_is_allocated"] = positions < 6_000
    captured: dict[str, object] = {}

    class Fitted:
        def predict(self, test: pd.DataFrame, **kwargs) -> pd.DataFrame:
            return pd.DataFrame({"rent": np.zeros(len(test))}, index=test.index)

    class QRF:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def fit(
            self,
            training: pd.DataFrame,
            *,
            predictors: list[str],
            targets: list[str],
            weights: str,
        ) -> Fitted:
            captured["training"] = training.copy()
            captured["predictors"] = predictors
            captured["targets"] = targets
            captured["weights"] = weights
            return Fitted()

    monkeypatch.setattr(module, "QRF", QRF)
    module.impute_us_pre_subsidy_rent(derive_us_housing_inputs(_frame()), donor, seed=0)

    # The retired shared cap first draws 5,000 rent rows and 5,000 disjoint
    # real-estate-tax rows.  Only the rent-filtered half trains this first
    # chained target; sampling 10,000 rent rows directly would violate parity.
    assert len(captured["training"]) == 5_000
    assert captured["targets"] == ["rent"]
    assert captured["weights"] == "household_weight"


def test_signal_gate_preserves_archived_qrf_rent_outside_renter_household(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "QRF", _RentQRF)
    result = with_us_housing_inputs(
        _frame(), seed=0, time_period=2024, acs_rent_donor=_donor()
    )
    result.table("person").loc[5, "pre_subsidy_rent"] = 1.0

    gate = us_housing_inputs_signal_gate(result)

    assert gate.passed, gate.failures
    assert gate.details["positive_rent_nonrenter"] == 1


def test_puf_half_reimputes_only_housing_assistance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "QRF", _RentQRF)
    direct = with_us_housing_inputs(
        _frame(), seed=0, time_period=2024, acs_rent_donor=_donor()
    )
    expanded = clone_us_frame_for_puf_support(direct)
    calls: dict[str, object] = {}

    class Fitted:
        def predict(self, test: pd.DataFrame, **kwargs) -> pd.DataFrame:
            values = np.zeros(len(test), dtype=np.float64)
            values[1] = 1.0
            return pd.DataFrame(
                {"receives_housing_assistance": values}, index=test.index
            )

    class QRF:
        def __init__(self, **kwargs: object) -> None:
            calls["init"] = kwargs

        def fit(
            self,
            training: pd.DataFrame,
            predictors: list[str],
            targets: list[str],
            **kwargs: object,
        ) -> Fitted:
            calls["training"] = training.copy()
            calls["predictors"] = predictors
            calls["targets"] = targets
            calls["fit_kwargs"] = kwargs
            return Fitted()

    monkeypatch.setattr(module, "QRF", QRF)
    result = impute_us_housing_assistance_to_puf_support(expanded, seed=7)

    spm = result.table("spm_unit")
    channel = spm[support_channel_column("spm_unit")]
    asec = channel == BASE_ASEC_SUPPORT_CHANNEL
    puf = channel == PUF_TAX_DETAIL_SUPPORT_CHANNEL
    assert spm.loc[asec, "receives_housing_assistance"].sum() == 1
    assert spm.loc[puf, "receives_housing_assistance"].sum() == 1
    assert not np.array_equal(
        spm.loc[asec, "receives_housing_assistance"].to_numpy(),
        spm.loc[puf, "receives_housing_assistance"].to_numpy(),
    )
    pd.testing.assert_series_equal(
        spm["takes_up_housing_assistance_if_eligible"],
        spm["receives_housing_assistance"],
        check_names=False,
    )
    assert calls["targets"] == ["receives_housing_assistance"]
    assert len(calls["predictors"]) == 8
    assert "pre_subsidy_rent" not in calls["targets"]
    for entity, column in (
        ("person", "pre_subsidy_rent"),
        ("spm_unit", "spm_unit_tenure_type"),
        ("household", "tenure_type"),
    ):
        source = expanded.table(entity)[column]
        actual = result.table(entity)[column]
        pd.testing.assert_series_equal(actual, source)

    result.table("spm_unit").loc[puf, "receives_housing_assistance"] = False
    collapsed_gate = us_housing_inputs_signal_gate(result)
    assert not collapsed_gate.passed
    assert any("PUF-tax-detail" in failure for failure in collapsed_gate.failures)


@pytest.mark.parametrize("bad_value", [None, np.nan, 2, -1, "true"])
def test_signal_gate_rejects_missing_or_invalid_take_up(bad_value: object) -> None:
    result = derive_us_housing_inputs(_frame())
    result.table("spm_unit")["takes_up_housing_assistance_if_eligible"] = result.table(
        "spm_unit"
    )["takes_up_housing_assistance_if_eligible"].astype(object)
    result.table("spm_unit").loc[0, "takes_up_housing_assistance_if_eligible"] = (
        bad_value
    )
    result.table("person")["pre_subsidy_rent"] = np.where(
        result.table("person")["is_household_head"],
        1.0,
        0.0,
    )

    gate = us_housing_inputs_signal_gate(result)

    assert not gate.passed
    assert any("takes_up_housing_assistance" in failure for failure in gate.failures)


def test_acs_loader_aligns_entities_collapses_tenure_and_marks_allocations(
    tmp_path: Path,
) -> None:
    path = tmp_path / "acs_2022.h5"
    _write_tiny_acs(path)

    donor = load_acs_2022_rent_donor(path, expected_sha256=None)

    assert len(donor) == 3
    assert donor["household_size"].tolist() == [2.0, 1.0, 1.0]
    assert donor["tenure_type"].tolist() == [
        "RENTED",
        "OWNED_WITH_MORTGAGE",
        "OWNED_WITH_MORTGAGE",
    ]
    assert donor["state_code_str"].tolist() == ["06", "36", "48"]
    assert donor["rent"].tolist() == [12_000.0, 0.0, 18_000.0]
    assert donor["rent_is_allocated"].tolist() == [False, True, False]
    assert donor["real_estate_taxes"].tolist() == [0.0, 8_000.0, 4_000.0]
    assert donor["real_estate_taxes_is_allocated"].tolist() == [False, False, True]
    assert donor["household_weight"].tolist() == [100.0, 200.0, 300.0]


def test_acs_loader_rejects_wrong_artifact_hash(tmp_path: Path) -> None:
    path = tmp_path / "acs_2022.h5"
    _write_tiny_acs(path)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_acs_2022_rent_donor(path, expected_sha256="0" * 64)


def test_acs_loader_retains_zero_weight_heads_for_archived_sampling(
    tmp_path: Path,
) -> None:
    path = tmp_path / "acs_2022.h5"
    _write_tiny_acs(path)
    with h5py.File(path, "a") as h5:
        h5["household_weight"][1] = 0.0

    donor = load_acs_2022_rent_donor(path, expected_sha256=None)

    assert len(donor) == 3
    assert donor["household_weight"].tolist() == [100.0, 0.0, 300.0]


def test_release_wiring_promotes_all_five_inputs_and_probes() -> None:
    manifest = load_release_input_coverage_manifest()
    for column in US_HOUSING_INPUTS_OUTPUT_COLUMNS:
        assert column in RESTORED_REFERENCE_ECPS_REQUIRED_INPUTS
        assert column in manifest.required_columns
        assert column not in manifest.reviewed_exclusions
    assert set(US_HOUSING_NONCONSTANT_PERSON_COLUMNS) <= set(
        US_RELEASE_REQUIRED_PERSON_SOURCE_COLUMNS
    )
    assert set(US_HOUSING_NONCONSTANT_SPM_UNIT_COLUMNS) <= set(
        US_RELEASE_REQUIRED_SPM_UNIT_SOURCE_COLUMNS
    )
    assert set(US_HOUSING_NONCONSTANT_HOUSEHOLD_COLUMNS) <= set(
        US_RELEASE_REQUIRED_HOUSEHOLD_NONCONSTANT_SOURCE_COLUMNS
    )
    probe = next(
        probe
        for probe in us_release_reform_coverage_probes()
        if probe.id == "pre_subsidy_rent_neutralization"
    )
    assert probe.neutralized_variable == "pre_subsidy_rent"
    assert probe.binding_inputs == ("pre_subsidy_rent",)
    assert probe.budget_measure == "snap"
    assert probe.effect_direction == "baseline_minus_reform"
    assert probe.expected_sign == "positive"
    take_up_probe = next(
        probe
        for probe in us_release_reform_coverage_probes()
        if probe.id == "housing_assistance_take_up_neutralization"
    )
    assert take_up_probe.neutralized_variable == (
        "takes_up_housing_assistance_if_eligible"
    )
    assert take_up_probe.binding_inputs == ("takes_up_housing_assistance_if_eligible",)
    assert take_up_probe.budget_measure == "housing_assistance"
    assert take_up_probe.min_abs_effect == 100_000_000.0
    contract = load_take_up_contract().program_map()[
        "takes_up_housing_assistance_if_eligible"
    ]
    assert contract.populace_treatment == "out_of_scope"
    assert contract.rate == {"status": "not_used_measured_source"}
