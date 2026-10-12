"""Tests split from packages/microcosm-build/tests/test_us_retirement_distributions.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_retirement_distributions import *


def test_stage_manifest_pins_direct_mapping_and_retired_qrf() -> None:
    spec = us_retirement_distributions_stage_spec()

    assert spec.stage == "retirement_distributions"
    assert spec.grain == "person"
    assert tuple(spec.outputs) == _OUTPUTS
    assert [operation.kind for operation in spec.operations] == [
        "read_table",
        "derive_retirement_distributions",
        "impute_retirement_distributions_to_puf_support",
    ]
    assert spec.operations[1].parameters["output_by_account_code"] == {
        "1": "taxable_401k_distributions",
        "2": "taxable_403b_distributions",
        "3": "tax_exempt_ira_distributions",
        "4": "taxable_ira_distributions",
        "5": "keogh_distributions",
        "6": "taxable_sep_distributions",
    }
    impute = spec.operations[2].parameters
    assert impute["predictors"] == [
        "age",
        "is_male",
        "has_esi",
        "tax_unit_is_joint",
        "tax_unit_count_dependents",
        "employment_income",
        "self_employment_income",
        "social_security",
    ]
    assert impute["max_train_samples"] == 5_000
    assert impute["n_estimators"] == 100
    assert RETIREMENT_DISTRIBUTIONS_ARCHIVED_DERIVATION_URL.endswith(
        "cps.py#L1448-L1481"
    )
    assert RETIREMENT_DISTRIBUTIONS_ARCHIVED_PARAMETERS_URL.endswith(
        "imputation_parameters.yaml#L10-L15"
    )


def test_handler_is_registered() -> None:
    handlers = us_source_operation_handlers()
    assert (
        handlers["derive_retirement_distributions"]
        is derive_us_retirement_distributions_from_manifest
    )
    assert (
        handlers["impute_retirement_distributions_to_puf_support"]
        is module.impute_us_retirement_distributions_to_puf_support_from_manifest
    )


def test_direct_mapping_sums_all_four_measured_slots() -> None:
    person = _person_source()
    person.loc[0, ["DST_SC1_YNG", "DST_VAL1_YNG"]] = [1, 25.0]
    person.loc[0, ["DST_SC2_YNG", "DST_VAL2_YNG"]] = [1, 75.0]

    result = _derive(person)

    np.testing.assert_array_equal(
        result[list(_OUTPUTS)].to_numpy(),
        np.asarray(
            [
                [250.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 200.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 300.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 400.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 500.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 600.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
    )


@pytest.mark.parametrize("missing", US_RETIREMENT_DISTRIBUTION_REQUIRED_SOURCE_COLUMNS)
def test_direct_mapping_fails_closed_without_each_source_column(missing: str) -> None:
    with pytest.raises(SourceRuntimeError, match=missing):
        _derive(_person_source().drop(columns=[missing]))


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("DST_SC1", np.nan, "account codes"),
        ("DST_SC1", 8, "account codes"),
        ("DST_SC1", 1.5, "account codes"),
        ("DST_VAL1", np.nan, "nonnegative amounts"),
        ("DST_VAL1", -1, "nonnegative amounts"),
    ],
)
def test_direct_mapping_rejects_invalid_source_values(
    column: str,
    value: float,
    message: str,
) -> None:
    person = _person_source()
    person[column] = person[column].astype(float)
    person.loc[0, column] = value

    with pytest.raises(SourceRuntimeError, match=message):
        _derive(person)


def test_direct_mapping_rejects_amount_on_not_in_universe_slot() -> None:
    person = _person_source()
    person.loc[7, "DST_VAL1"] = 1.0
    with pytest.raises(SourceRuntimeError, match="NIU code 0"):
        _derive(person)


def test_wrong_operation_and_missing_table_fail_closed() -> None:
    wrong = SourceStageSpec.from_mapping(
        {
            "stage": "test",
            "survey": "test",
            "source": "https://example.com",
            "grain": "person",
            "operations": [{"kind": "derive"}],
            "outputs": list(_OUTPUTS),
        }
    ).operations[0]
    with pytest.raises(SourceRuntimeError, match="unexpected operation"):
        derive_us_retirement_distributions_from_manifest(_person_source(), wrong, None)
    with pytest.raises(SourceRuntimeError, match="person table"):
        derive_us_retirement_distributions_from_manifest(None, _operation(), None)


def test_frame_integration_gate_and_idempotence() -> None:
    result = with_us_retirement_distribution_inputs(_frame(), seed=0, time_period=2024)
    gate = us_retirement_distributions_signal_gate(result)

    assert gate.passed, gate.failures
    assert all(value == 0 for value in gate.details["source_mismatches"].values())
    assert (
        with_us_retirement_distribution_inputs(result, seed=0, time_period=2024)
        is result
    )


def test_stacked_gate_validates_physical_source_and_reconciles_direct_role() -> None:
    stacked = _stacked_frame()

    gate = us_retirement_distributions_signal_gate(stacked)

    assert gate.passed, gate.failures
    assert gate.details["source_rows"] == 8
    assert gate.details["source_reconciliation_rows"] == 4
    assert all(value == 0 for value in gate.details["source_mismatches"].values())

    person = stacked.table("person")
    asec_puf_role = person["person_support_channel"].eq("asec") & person[
        "person_support_clone_index"
    ].eq(1)
    person.loc[person.index[asec_puf_role][0], "DST_SC1"] = np.nan
    with pytest.raises(SourceRuntimeError, match="account codes"):
        us_retirement_distributions_summary(stacked)


def test_puf_half_uses_qrf_and_asec_half_remains_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    direct = with_us_retirement_distribution_inputs(_frame(), seed=0, time_period=2024)
    expanded = clone_us_frame_for_puf_support(direct)
    expanded_person = expanded.table("person")
    puf_mask = expanded_person["person_support_channel"] == "puf_tax_detail"
    # The PUF tax-detail stage owns this leaf; the retirement stage must not
    # replace it with either the copied ASEC value or a QRF prediction.
    puf_indices = expanded_person.index[puf_mask]
    expanded_person.loc[puf_mask, "taxable_ira_distributions"] = 0.0
    expanded_person.loc[puf_indices[3], "taxable_ira_distributions"] = 999.0
    calls: dict[str, object] = {}

    class FakeFitted:
        def predict(self, test: pd.DataFrame, **kwargs) -> pd.DataFrame:
            calls["test"] = test.copy()
            return pd.DataFrame(
                0.0,
                index=test.index,
                columns=list(_PUF_QRF_OUTPUTS),
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
    result = with_us_retirement_distribution_inputs(
        expanded,
        seed=7,
        time_period=2024,
        force_puf_imputation=True,
    )

    assert calls["init"] == {"n_estimators": 100, "seed": 7}
    assert len(calls["training"]) == len(direct.table("person"))
    assert len(calls["test"]) == len(direct.table("person"))
    assert calls["targets"] == list(_PUF_QRF_OUTPUTS)
    person = result.table("person")
    asec = person[person["person_support_channel"] == "asec"]
    puf = person[person["person_support_channel"] == "puf_tax_detail"]
    np.testing.assert_array_equal(
        asec[list(_OUTPUTS)].to_numpy(),
        direct.table("person")[list(_OUTPUTS)].to_numpy(),
    )
    assert not puf[list(_PUF_QRF_OUTPUTS)].to_numpy().any()
    np.testing.assert_array_equal(
        puf["tax_exempt_ira_distributions"].to_numpy(),
        direct.table("person")["tax_exempt_ira_distributions"].to_numpy(),
    )
    np.testing.assert_array_equal(
        puf["taxable_ira_distributions"].to_numpy(),
        [0.0, 0.0, 0.0, 999.0, 0.0, 0.0, 0.0, 0.0],
    )
    gate = us_retirement_distributions_signal_gate(result)
    assert gate.passed, gate.failures


def test_completed_puf_surface_survives_narrowed_support_without_refit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    direct = with_us_retirement_distribution_inputs(_frame(), seed=0, time_period=2024)
    expanded = clone_us_frame_for_puf_support(direct)

    class ZeroFitted:
        def predict(self, test: pd.DataFrame, **kwargs) -> pd.DataFrame:
            return pd.DataFrame(
                0.0,
                index=test.index,
                columns=list(_PUF_QRF_OUTPUTS),
            )

    class ZeroQRF:
        def __init__(self, **kwargs: object) -> None:
            pass

        def fit(self, *args: object, **kwargs: object) -> ZeroFitted:
            return ZeroFitted()

    monkeypatch.setattr(module, "QRF", ZeroQRF)
    completed = with_us_retirement_distribution_inputs(
        expanded,
        seed=7,
        time_period=2024,
        force_puf_imputation=True,
    )

    # Model frozen-support recovery by removing one all-zero PUF household.
    # Frame.select deliberately preserves the surviving person index, so the
    # selected frame also covers the non-RangeIndex path that changed the
    # historical 5,000-row donor sample.
    person = completed.table("person")
    puf_zero = person["person_support_channel"].eq("puf_tax_detail") & ~person[
        list(_PUF_QRF_OUTPUTS)
    ].any(axis=1)
    drop_index = person.index[puf_zero][0]
    selected = completed.select(person.index != drop_index)
    before = us_retirement_distributions_signal_gate(selected)
    assert before.passed, before.failures
    before_share = before.details["nonzero_shares"]["keogh_distributions"]
    assert 0.0000001 <= before_share <= 0.005
    before_values = selected.table("person")[list(_OUTPUTS)].copy()
    before_keogh_carriers = int((before_values["keogh_distributions"] > 0).sum())

    class UnexpectedQRF:
        def __init__(self, **kwargs: object) -> None:
            raise AssertionError("a completed retirement surface must not be refit")

    monkeypatch.setattr(module, "QRF", UnexpectedQRF)
    result = with_us_retirement_distribution_inputs(
        selected,
        seed=7,
        time_period=2024,
    )

    assert result is selected
    after = us_retirement_distributions_signal_gate(result)
    assert after.passed, after.failures
    assert after.details["nonzero_shares"]["keogh_distributions"] == before_share
    pd.testing.assert_frame_equal(
        result.table("person")[list(_OUTPUTS)],
        before_values,
    )
    assert (
        result.table("person")["keogh_distributions"].gt(0).sum()
        == before_keogh_carriers
    )

    # If support selection removes every rare carrier from one leaf, preserve
    # the completed surface and fail closed at the gate instead of refitting.
    # Removing the measured carrier rows keeps source reconciliation valid and
    # isolates the degeneration/prevalence checks this regression owns.
    keogh_carriers = result.table("person")["keogh_distributions"].gt(0)
    assert keogh_carriers.any()
    selected_away = result.select(~keogh_carriers)
    degenerate = with_us_retirement_distribution_inputs(
        selected_away,
        seed=7,
        time_period=2024,
    )
    assert degenerate is selected_away
    degenerate_gate = us_retirement_distributions_signal_gate(degenerate)
    assert not degenerate_gate.passed
    assert degenerate_gate.failures == (
        "keogh_distributions: degenerate with 1 distinct value(s).",
        "keogh_distributions: weighted nonzero share 0.00000000 outside "
        "[0.00000010, 0.00500000].",
    )
    assert degenerate_gate.details["source_mismatches"]["keogh_distributions"] == 0


@pytest.mark.parametrize("missing", _OUTPUTS)
def test_incomplete_puf_surface_is_consume_only_and_fails_at_the_signal_gate(
    monkeypatch: pytest.MonkeyPatch,
    missing: str,
) -> None:
    direct = with_us_retirement_distribution_inputs(_frame(), seed=0, time_period=2024)
    incomplete = clone_us_frame_for_puf_support(direct)
    incomplete.table("person").drop(columns=[missing], inplace=True)

    class UnexpectedQRF:
        def __init__(self, **kwargs: object) -> None:
            raise AssertionError("an incomplete support surface must not be refit")

    monkeypatch.setattr(module, "QRF", UnexpectedQRF)
    result = with_us_retirement_distribution_inputs(
        incomplete,
        seed=7,
        time_period=2024,
    )

    assert result is incomplete
    gate = us_retirement_distributions_signal_gate(result)
    assert not gate.passed
    assert gate.failures == (f"person columns missing: {[missing]}.",)
    assert gate.details == {"missing": [missing]}


def test_gate_rejects_a_default_or_source_divergent_leaf() -> None:
    result = with_us_retirement_distribution_inputs(_frame(), seed=0, time_period=2024)
    result.table("person")["keogh_distributions"] = 0.0
    gate = us_retirement_distributions_signal_gate(result)
    assert not gate.passed
    assert any("keogh_distributions" in failure for failure in gate.failures)


def test_all_sha_locked_asec_artifacts_carry_exact_source_signal() -> None:
    pytest.importorskip("tables", exc_type=ModuleNotFoundError)
    expected = {
        "census_cps_2022.h5": (
            "7ccca976284bb47815d84460cc4f75a0a65d26d7754ab0a0f417de351b3d474e",
            4,
            44_000.0,
        ),
        "census_cps_2023.h5": (
            "cb57817327799f42b741caed5f9be94d04021c2e6809c1ad7bd0686da5428d88",
            5,
            130_040.0,
        ),
        "census_cps_2024.h5": (
            "ec36604cb735a660b51b0b2f90be27d803b5878f3464fb30d0eacead59c1260d",
            4,
            166_600.0,
        ),
    }
    summary = json.loads(
        (ROOT / "experiments/build_j_recert/base_j.summary.json").read_text()
    )
    paths = {
        Path(item["path"]).name: Path(item["path"])
        for item in summary["base_source"]["sources"]
    }
    if not all(paths[name].is_file() for name in expected):
        pytest.skip("SHA-locked ASEC artifacts are not mounted")

    for name, (digest, positives, total) in expected.items():
        path = paths[name]
        assert _sha256(path) == digest
        person = load_asec_h5_tables(path)["person"]
        assert set(US_RETIREMENT_DISTRIBUTION_REQUIRED_SOURCE_COLUMNS) <= set(
            person.columns
        )
        derived = _derive(person)["keogh_distributions"]
        assert int((derived > 0).sum()) == positives
        assert float(derived.sum()) == total


def test_shipped_keogh_neutralization_probe() -> None:
    probe = next(
        probe
        for probe in us_release_reform_coverage_probes()
        if probe.id == "keogh_distribution_neutralization"
    )
    assert probe.neutralized_variable == "keogh_distributions"
    assert probe.binding_inputs == ("keogh_distributions",)
    assert probe.budget_measure == "income_tax"
    assert probe.period == 2024
    assert probe.effect_direction == "baseline_minus_reform"
    assert probe.expected_sign == "positive"
    assert probe.min_abs_effect >= 1_000_000.0
