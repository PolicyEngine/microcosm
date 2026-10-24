"""Tests split from packages/microcosm-build/tests/test_us_energy_subsidy.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_energy_subsidy import *


def test_archived_urls_are_immutable_and_pin_both_retired_steps() -> None:
    assert f"/blob/{_ARCHIVED_COMMIT}/" in (ENERGY_SUBSIDY_ARCHIVED_CPS_DERIVATION_URL)
    assert f"/blob/{_ARCHIVED_COMMIT}/" in (ENERGY_SUBSIDY_ARCHIVED_PUF_IMPUTATION_URL)
    assert ENERGY_SUBSIDY_ARCHIVED_CPS_DERIVATION_URL.endswith(
        "datasets/cps/cps.py#L1612-L1622"
    )
    assert ENERGY_SUBSIDY_ARCHIVED_PUF_IMPUTATION_URL.endswith(
        "datasets/cps/extended_cps.py#L639-L739"
    )


def test_stage_manifest_pins_exact_source_qrf_and_reduction_contract() -> None:
    spec = us_energy_subsidy_stage_spec()

    assert US_ENERGY_SUBSIDY_STAGE_NAME == "energy_subsidy"
    assert spec.stage == US_ENERGY_SUBSIDY_STAGE_NAME
    assert spec.grain == "person"
    assert tuple(spec.outputs) == US_ENERGY_SUBSIDY_OUTPUT_COLUMNS
    assert tuple(spec.nonnegative_outputs) == US_ENERGY_SUBSIDY_OUTPUT_COLUMNS
    assert [operation.kind for operation in spec.operations] == [
        "read_table",
        "derive_energy_subsidy",
        "impute_energy_subsidy_to_puf_support",
    ]
    assert spec.operations[0].parameters == {
        "table": "person",
        "weight": "person_weight",
    }
    impute = spec.operations[2]
    assert tuple(impute.parameters["predictors"]) == (
        "age",
        "is_male",
        "has_esi",
        "tax_unit_is_joint",
        "tax_unit_count_dependents",
        "employment_income",
        "self_employment_income",
        "social_security",
    )
    assert impute.parameters["max_train_samples"] == 5_000
    assert impute.parameters["n_estimators"] == 100
    assert impute.parameters["seed_from_build_config"] is True
    assert impute.parameters["weight"] == "person_weight"
    assert impute.parameters["reduction"] == "value_from_first_person"


def test_direct_source_is_exact_and_replicated() -> None:
    result = _derive(_person_source())

    assert result[_OUTPUT].tolist() == [600.0, 600.0, 0.0, 0.0, 0.0, 0.0]


@pytest.mark.parametrize("missing", US_ENERGY_SUBSIDY_REQUIRED_SOURCE_COLUMNS)
def test_direct_source_fails_closed_when_required_source_is_missing(
    missing: str,
) -> None:
    with pytest.raises(SourceRuntimeError, match=missing):
        _derive(_person_source().drop(columns=[missing]))


@pytest.mark.parametrize(
    ("bad_value", "message"),
    [(np.nan, "nonfinite"), (np.inf, "nonfinite"), (-1.0, "negative")],
)
def test_direct_source_rejects_invalid_values(
    bad_value: float,
    message: str,
) -> None:
    person = _person_source()
    person.loc[[0, 1], "SPM_ENGVAL"] = bad_value

    with pytest.raises(SourceRuntimeError, match=message):
        _derive(person)


def test_direct_source_rejects_inconsistent_unit_replicas() -> None:
    person = _person_source()
    person.loc[1, "SPM_ENGVAL"] = 599.0

    with pytest.raises(SourceRuntimeError, match="disagrees within replicated"):
        _derive(person)


def test_with_input_materializes_first_person_spm_unit_values() -> None:
    result = with_us_energy_subsidy_input(_frame(), seed=0, time_period=2024)

    assert result.table("spm_unit")[_OUTPUT].tolist() == [
        600.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    gate = us_energy_subsidy_signal_gate(result)
    assert gate.passed, gate.failures
    assert us_energy_subsidy_summary(result)["positive_share_band"] == [0.01, 0.2]


def test_existing_output_is_recomputed_from_measured_source() -> None:
    frame = _frame()
    frame.table("spm_unit")[_OUTPUT] = [111.0, 222.0, 333.0, 444.0, 555.0]

    result = with_us_energy_subsidy_input(frame, seed=0, time_period=2024)

    assert result.table("spm_unit")[_OUTPUT].tolist() == [
        600.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]


def test_production_release_can_preserve_only_a_valid_existing_surface() -> None:
    materialized = with_us_energy_subsidy_input(_frame(), seed=0, time_period=2024)
    tables = {
        entity: materialized.table(entity).copy() for entity in materialized.entities
    }
    tables["person"] = tables["person"].drop(columns=["SPM_ENGVAL"])
    release = Frame(
        tables,
        materialized.schema,
        {
            entity: materialized.weights_for(entity)
            for entity in materialized.weighted_entities
        },
        materialized.strata,
    )

    with pytest.raises(ValueError, match="cannot heal.*without measured"):
        with_us_energy_subsidy_input(release, seed=0, time_period=2024)

    assert (
        with_us_energy_subsidy_input(
            release,
            seed=0,
            time_period=2024,
            allow_existing_without_source=True,
        )
        is release
    )

    release.table("spm_unit")[_OUTPUT] = 0.0
    with pytest.raises(ValueError, match="cannot heal.*without measured"):
        with_us_energy_subsidy_input(
            release,
            seed=0,
            time_period=2024,
            allow_existing_without_source=True,
        )


def test_puf_half_uses_weighted_qrf_and_first_person_spm_reduction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    direct = with_us_energy_subsidy_input(_frame(), seed=0, time_period=2024)
    expanded = clone_us_frame_for_puf_support(direct)
    calls: dict[str, object] = {}

    class FakeFitted:
        def predict(self, test: pd.DataFrame, **kwargs) -> pd.DataFrame:
            calls["test"] = test.copy()
            return pd.DataFrame(
                {_OUTPUT: np.arange(100.0, 100.0 + len(test))},
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

    result = with_us_energy_subsidy_input(expanded, seed=7, time_period=2024)

    assert calls["init"] == {"n_estimators": 100, "seed": 7}
    assert len(calls["training"]) == 6
    assert len(calls["test"]) == 6
    assert calls["targets"] == [_OUTPUT]
    spm = result.table("spm_unit")
    asec = spm[spm["spm_unit_support_channel"] == "asec"]
    puf = spm[spm["spm_unit_support_channel"] == "puf_tax_detail"]
    assert asec[_OUTPUT].tolist() == [600.0, 0.0, 0.0, 0.0, 0.0]
    # The first two PUF people share an SPM unit, so prediction 101 is ignored.
    assert puf[_OUTPUT].tolist() == [100.0, 102.0, 103.0, 104.0, 105.0]


def test_puf_qrf_caps_training_at_5000_and_keeps_aligned_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operation = next(
        operation
        for operation in us_energy_subsidy_stage_spec().operations
        if operation.kind == "impute_energy_subsidy_to_puf_support"
    )
    predictors = tuple(operation.parameters["predictors"])
    asec_rows = 5_010
    puf_rows = 2
    frame = pd.DataFrame(
        {
            "person_support_channel": ["asec"] * asec_rows
            + ["puf_tax_detail"] * puf_rows,
            "person_weight": np.arange(1.0, asec_rows + puf_rows + 1.0),
            _OUTPUT: np.tile([0.0, 500.0], (asec_rows + puf_rows + 1) // 2)[
                : asec_rows + puf_rows
            ],
            **{
                f"energy_subsidy_predictor_{predictor}": np.arange(
                    asec_rows + puf_rows, dtype=np.float64
                )
                for predictor in predictors
            },
        }
    )
    calls: dict[str, object] = {}

    class FakeFitted:
        def predict(self, test: pd.DataFrame, **kwargs) -> pd.DataFrame:
            calls["test_rows"] = len(test)
            return pd.DataFrame({_OUTPUT: np.zeros(len(test))}, index=test.index)

    class FakeQRF:
        def __init__(self, **kwargs: object) -> None:
            calls["init"] = kwargs

        def fit(
            self,
            training: pd.DataFrame,
            predictor_names: list[str],
            targets: list[str],
            *,
            weights: np.ndarray,
        ) -> FakeFitted:
            calls["training_rows"] = len(training)
            calls["weights"] = weights.copy()
            calls["training_index"] = training.index.to_numpy()
            return FakeFitted()

    monkeypatch.setattr(module, "QRF", FakeQRF)
    context = SourceRuntimeContext(
        config=SourceRuntimeConfig(seed=19, target_year=2024),
        tables={},
    )

    impute_us_energy_subsidy_to_puf_support_from_manifest(frame, operation, context)

    assert calls["init"] == {"n_estimators": 100, "seed": 19}
    assert calls["training_rows"] == 5_000
    assert calls["test_rows"] == 2
    np.testing.assert_allclose(
        calls["weights"],
        frame.loc[calls["training_index"], "person_weight"].to_numpy(),
    )


def test_signal_gate_rejects_missing_default_invalid_and_channel_drift() -> None:
    valid = with_us_energy_subsidy_input(_frame(), seed=0, time_period=2024)
    for values in (
        None,
        [0.0] * 5,
        [600.0, 0.0, -1.0, 0.0, 0.0],
        [600.0, 0.0, np.nan, 0.0, 0.0],
        [600.0] * 5,
    ):
        candidate = _frame()
        if values is not None:
            candidate.table("spm_unit")[_OUTPUT] = values
        assert not us_energy_subsidy_signal_gate(candidate).passed

    assert us_energy_subsidy_signal_gate(valid).passed

    valid.table("spm_unit")["spm_unit_support_channel"] = "asec"
    channel_gate = us_energy_subsidy_signal_gate(valid)
    assert not channel_gate.passed
    assert any("asec positive share" in failure for failure in channel_gate.failures)
    assert any(
        "puf_tax_detail positive share" in failure for failure in channel_gate.failures
    )


def test_all_sha_locked_asec_artifacts_carry_exact_source_signal() -> None:
    pytest.importorskip("tables", exc_type=ModuleNotFoundError)
    expected = {
        "census_cps_2022.h5": (
            "7ccca976284bb47815d84460cc4f75a0a65d26d7754ab0a0f417de351b3d474e",
            4_740,
            3_439_573.0,
            2_083,
            1_442_882.0,
        ),
        "census_cps_2023.h5": (
            "cb57817327799f42b741caed5f9be94d04021c2e6809c1ad7bd0686da5428d88",
            4_562,
            3_078_344.0,
            1_995,
            1_270_213.0,
        ),
        "census_cps_2024.h5": (
            "ec36604cb735a660b51b0b2f90be27d803b5878f3464fb30d0eacead59c1260d",
            4_297,
            2_866_931.0,
            1_955,
            1_246_009.0,
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

    for name, expected_values in expected.items():
        digest, positive_rows, person_total, positive_units, unit_total = (
            expected_values
        )
        path = paths[name]
        assert _sha256(path) == digest
        raw_person = load_asec_h5_tables(path)["person"]
        assert {"SPM_ID", "SPM_ENGVAL"} <= set(raw_person.columns)
        source = raw_person[["SPM_ID", "SPM_ENGVAL"]].rename(
            columns={"SPM_ID": "person_spm_unit_id"}
        )
        derived = _derive(source)
        values = derived[_OUTPUT]
        assert int((values > 0.0).sum()) == positive_rows
        assert float(values.sum()) == person_total
        by_unit = derived.groupby("person_spm_unit_id", sort=False)[_OUTPUT]
        assert int((by_unit.first() > 0.0).sum()) == positive_units
        assert float(by_unit.first().sum()) == unit_total
        np.testing.assert_array_equal(by_unit.min(), by_unit.max())
