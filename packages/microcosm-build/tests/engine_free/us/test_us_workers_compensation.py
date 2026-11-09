"""Tests split from packages/microcosm-build/tests/test_us_workers_compensation.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_workers_compensation import *


def test_archived_sources_are_sha_and_line_pinned_and_sources_available() -> None:
    commit = "42ed5d45c56df80d754fbe24cce21cfeb8d05cbe"
    urls = (
        WORKERS_COMPENSATION_ARCHIVED_DERIVATION_URL,
        WORKERS_COMPENSATION_ARCHIVED_SOURCE_COLUMNS_URL,
        WORKERS_COMPENSATION_ARCHIVED_PUF_OUTPUTS_URL,
        WORKERS_COMPENSATION_ARCHIVED_PUF_IMPUTATION_URL,
    )
    assert all(commit in url for url in urls)
    assert WORKERS_COMPENSATION_ARCHIVED_DERIVATION_URL.endswith(
        "datasets/cps/cps.py#L1559-L1571"
    )
    assert WORKERS_COMPENSATION_ARCHIVED_SOURCE_COLUMNS_URL.endswith(
        "datasets/cps/census_cps.py#L306-L381"
    )
    assert WORKERS_COMPENSATION_ARCHIVED_PUF_OUTPUTS_URL.endswith(
        "datasets/cps/extended_cps.py#L135-L194"
    )
    assert WORKERS_COMPENSATION_ARCHIVED_PUF_IMPUTATION_URL.endswith(
        "datasets/cps/extended_cps.py#L639-L745"
    )
    assert US_WORKERS_COMPENSATION_REQUIRED_SOURCE_COLUMNS == ("WC_VAL",)


def test_stage_manifest_pins_direct_wc_val_formula_and_one_output_qrf() -> None:
    spec = us_workers_compensation_stage_spec()

    assert spec.stage == "workers_compensation_input"
    assert spec.survey == "Census CPS ASEC"
    assert spec.grain == "person"
    assert tuple(spec.outputs) == (_OUTPUT,)
    assert tuple(spec.nonnegative_outputs) == (_OUTPUT,)
    assert [operation.kind for operation in spec.operations] == [
        "read_table",
        "derive_workers_compensation",
        "impute_workers_compensation_to_puf_support",
    ]
    assert spec.operations[0].parameters == {
        "table": "person",
        "weight": "person_weight",
    }
    assert spec.operations[1].parameters == {
        "source": "WC_VAL",
        "output": _OUTPUT,
    }
    assert spec.operations[2].parameters == {
        "predictors": list(_PREDICTORS),
        "max_train_samples": 5_000,
        "n_estimators": 100,
        "seed_from_build_config": True,
        "weight": "person_weight",
    }

    handlers = us_source_operation_handlers()
    assert (
        handlers["derive_workers_compensation"]
        is derive_us_workers_compensation_from_manifest
    )
    assert (
        handlers["impute_workers_compensation_to_puf_support"]
        is impute_us_workers_compensation_to_puf_support_from_manifest
    )


def test_direct_formula_carries_wc_val_and_preserves_topcodes() -> None:
    source = pd.DataFrame({"WC_VAL": [0.0, 99_999.0, 500.0, 800.0, 300.0]})
    original = source.copy(deep=True)

    result = _derive(source)

    assert result[_OUTPUT].tolist() == [0.0, 99_999.0, 500.0, 800.0, 300.0]
    pd.testing.assert_frame_equal(source, original)


def test_direct_formula_does_not_substitute_disability_code_one_slots() -> None:
    source = pd.DataFrame(
        {
            "WC_VAL": [1_000.0, 0.0, 2_000.0],
            "DIS_VAL1": [1_000.0, 5_000.0, 0.0],
            "DIS_SC1": [1, 1, 0],
            "DIS_VAL2": [0.0, 4_000.0, 8_000.0],
            "DIS_SC2": [0, 1, 1],
        }
    )

    result = _derive(source)

    assert result[_OUTPUT].tolist() == [1_000.0, 0.0, 2_000.0]


@pytest.mark.parametrize("missing", US_WORKERS_COMPENSATION_REQUIRED_SOURCE_COLUMNS)
def test_direct_formula_fails_closed_when_source_is_missing(missing: str) -> None:
    with pytest.raises(SourceRuntimeError, match=missing):
        _derive(_person_source().drop(columns=[missing]))


@pytest.mark.parametrize(
    ("column", "bad_value", "message"),
    [
        ("WC_VAL", np.nan, "nonfinite"),
        ("WC_VAL", np.inf, "nonfinite"),
        ("WC_VAL", -1.0, "negative"),
    ],
)
def test_direct_formula_rejects_invalid_sources(
    column: str,
    bad_value: float,
    message: str,
) -> None:
    person = _person_source()
    person.loc[0, column] = bad_value

    with pytest.raises(SourceRuntimeError, match=message):
        _derive(person)


def test_with_inputs_materializes_exact_asec_values_without_mutation() -> None:
    frame = _frame()
    original = frame.table("person").copy(deep=True)

    result = with_us_workers_compensation(frame, seed=0, time_period=2024)

    pd.testing.assert_frame_equal(frame.table("person"), original)
    expected = original["WC_VAL"]
    np.testing.assert_allclose(result.table("person")[_OUTPUT], expected)
    gate = us_workers_compensation_signal_gate(result)
    assert gate.passed, gate.failures


def test_release_requires_opt_in_to_preserve_valid_existing_surface() -> None:
    materialized = with_us_workers_compensation(_frame(), seed=0, time_period=2024)
    tables = {
        entity: materialized.table(entity).copy() for entity in materialized.entities
    }
    tables["person"] = tables["person"].drop(
        columns=list(US_WORKERS_COMPENSATION_REQUIRED_SOURCE_COLUMNS)
    )
    release = Frame(
        tables,
        materialized.schema,
        {
            entity: materialized.weights_for(entity)
            for entity in materialized.weighted_entities
        },
        materialized.strata,
        mass_log=materialized.mass_log,
    )

    with pytest.raises(ValueError, match="cannot heal.*without measured"):
        with_us_workers_compensation(release, seed=0, time_period=2024)

    assert (
        with_us_workers_compensation(
            release,
            seed=0,
            time_period=2024,
            allow_existing_without_source=True,
        )
        is release
    )


def test_puf_half_uses_one_output_qrf_and_preserves_asec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    direct = with_us_workers_compensation(_frame(), seed=0, time_period=2024)
    expanded = clone_us_frame_for_puf_support(direct)
    original = expanded.table("person").copy(deep=True)
    calls: dict[str, object] = {}

    class FakeFitted:
        def predict(self, test: pd.DataFrame, **kwargs) -> pd.DataFrame:
            calls["test"] = test.copy()
            predicted = np.zeros(len(test), dtype=np.float64)
            predicted[0] = 7_200.0
            return pd.DataFrame({_OUTPUT: predicted}, index=test.index)

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
            calls["fit_count"] = int(calls.get("fit_count", 0)) + 1
            calls["training"] = training.copy()
            calls["predictors"] = predictors
            calls["targets"] = targets
            calls["weights"] = weights.copy()
            return FakeFitted()

    monkeypatch.setattr(module, "QRF", FakeQRF)

    result = with_us_workers_compensation(expanded, seed=7, time_period=2024)

    assert calls["fit_count"] == 1
    assert calls["init"] == {"n_estimators": 100, "seed": 7}
    assert calls["predictors"] == list(_PREDICTORS)
    assert calls["targets"] == [_OUTPUT]
    training = calls["training"]
    assert isinstance(training, pd.DataFrame)
    assert list(training.columns) == [*_PREDICTORS, _OUTPUT]
    asec_mask = original["person_support_channel"] == "asec"
    expected_weights = expanded.resolve_weights("person").values[asec_mask]
    np.testing.assert_allclose(calls["weights"], expected_weights)

    person = result.table("person")
    asec = person[person["person_support_channel"] == "asec"]
    puf = person[person["person_support_channel"] == "puf_tax_detail"]
    assert asec[_OUTPUT].tolist() == [3_600.0, *([0.0] * 99)]
    assert puf[_OUTPUT].tolist() == [7_200.0, *([0.0] * 99)]
    pd.testing.assert_frame_equal(expanded.table("person"), original)
    gate = us_workers_compensation_signal_gate(result)
    assert gate.passed, gate.failures


def test_puf_qrf_caps_training_at_5000_and_keeps_weights_aligned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operation = us_workers_compensation_stage_spec().operations[2]
    asec_rows = 5_010
    puf_rows = 2
    rows = asec_rows + puf_rows
    frame = pd.DataFrame(
        {
            "person_support_channel": ["asec"] * asec_rows
            + ["puf_tax_detail"] * puf_rows,
            "person_weight": np.arange(1.0, rows + 1.0),
            _OUTPUT: np.tile([0.0, 500.0], (rows + 1) // 2)[:rows],
            **{
                f"workers_compensation_predictor_{predictor}": np.arange(
                    rows, dtype=np.float64
                )
                for predictor in _PREDICTORS
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
            predictors: list[str],
            targets: list[str],
            *,
            weights: np.ndarray,
        ) -> FakeFitted:
            calls["training_rows"] = len(training)
            calls["training_index"] = training.index.to_numpy()
            calls["weights"] = weights.copy()
            return FakeFitted()

    monkeypatch.setattr(module, "QRF", FakeQRF)
    context = SourceRuntimeContext(
        config=SourceRuntimeConfig(seed=19, target_year=2024),
        tables={},
    )

    impute_us_workers_compensation_to_puf_support_from_manifest(
        frame,
        operation,
        context,
    )

    assert calls["init"] == {"n_estimators": 100, "seed": 19}
    assert calls["training_rows"] == 5_000
    assert calls["test_rows"] == 2
    np.testing.assert_allclose(
        calls["weights"],
        frame.loc[calls["training_index"], "person_weight"].to_numpy(),
    )


def test_signal_gate_rejects_missing_default_and_invalid_surfaces() -> None:
    valid = with_us_workers_compensation(_frame(), seed=0, time_period=2024)
    assert us_workers_compensation_signal_gate(valid).passed

    candidates: list[Frame] = []
    for replacement in (
        None,
        np.zeros(100),
        np.asarray([-1.0, *([0.0] * 99)]),
        np.asarray([np.nan, *([0.0] * 99)]),
    ):
        candidate = with_us_workers_compensation(_frame(), seed=0, time_period=2024)
        if replacement is None:
            candidate.table("person").drop(columns=[_OUTPUT], inplace=True)
        else:
            candidate.table("person")[_OUTPUT] = replacement
        candidates.append(candidate)

    assert all(
        not us_workers_compensation_signal_gate(candidate).passed
        for candidate in candidates
    )


@pytest.mark.parametrize("dead_channel", ["asec", "puf_tax_detail"])
def test_signal_gate_rejects_either_dead_support_channel(dead_channel: str) -> None:
    direct = with_us_workers_compensation(_frame(), seed=0, time_period=2024)
    expanded = clone_us_frame_for_puf_support(direct)
    assert us_workers_compensation_signal_gate(expanded).passed

    channel = expanded.table("person")["person_support_channel"]
    expanded.table("person").loc[channel == dead_channel, _OUTPUT] = 0.0
    gate = us_workers_compensation_signal_gate(expanded)

    assert not gate.passed
    assert any(dead_channel in failure for failure in gate.failures)


def test_signal_gate_reconciles_physical_asec_rows_in_stacked_pool() -> None:
    frame = _stacked_workers_compensation_frame()

    gate = us_workers_compensation_signal_gate(frame)

    assert gate.passed, gate.failures
    assert gate.details["source_invalid"] == 0
    assert gate.details["source_mismatch_count"] == 0

    person = frame.table("person")
    asec_clone = person["person_support_channel"].eq("asec") & person[
        "person_support_clone_index"
    ].eq(1)
    person.loc[asec_clone.idxmax(), _OUTPUT] = 1.0
    transferred_clone = us_workers_compensation_signal_gate(frame)
    assert transferred_clone.passed, transferred_clone.failures
    assert transferred_clone.details["source_mismatch_count"] == 0

    asec_native = person["person_support_channel"].eq("asec") & person[
        "person_support_clone_index"
    ].eq(0)
    person.loc[asec_native.idxmax(), _OUTPUT] = 1.0
    mismatch = us_workers_compensation_signal_gate(frame)
    assert not mismatch.passed
    assert mismatch.details["source_mismatch_count"] == 1

    person.loc[asec_native.idxmax(), _OUTPUT] = person.loc[
        asec_native.idxmax(), "WC_VAL"
    ]
    person.loc[asec_clone.idxmax(), "WC_VAL"] = np.nan
    invalid_source = us_workers_compensation_signal_gate(frame)
    assert not invalid_source.passed
    assert invalid_source.details["source_invalid"] == 1


def test_signal_gate_preserves_legacy_asec_puf_source_scope() -> None:
    direct = with_us_workers_compensation(_frame(), seed=0, time_period=2024)
    legacy = clone_us_frame_for_puf_support(direct)
    person = legacy.table("person")
    puf = person["person_support_channel"].eq("puf_tax_detail")
    person.loc[puf, "WC_VAL"] = np.nan

    gate = us_workers_compensation_signal_gate(legacy)

    assert gate.passed, gate.failures
    assert gate.details["source_invalid"] == 0
    assert gate.details["source_mismatch_count"] == 0


def test_release_promotion_plan_builders_and_probe_are_wired() -> None:
    from microcosm.build.us_runtime import US_DONORS, US_STAGE_NAMES

    manifest = load_release_input_coverage_manifest()
    assert _OUTPUT in RESTORED_REFERENCE_ECPS_REQUIRED_INPUTS
    assert _OUTPUT in manifest.required_columns
    assert _OUTPUT not in manifest.reviewed_exclusions
    assert _OUTPUT in US_RELEASE_REQUIRED_PERSON_SOURCE_COLUMNS
    assert module.US_WORKERS_COMPENSATION_STAGE_NAME in US_DONORS
    assert module.US_WORKERS_COMPENSATION_STAGE_NAME in US_STAGE_NAMES

    probe = next(
        probe
        for probe in us_release_reform_coverage_probes()
        if probe.id == "workers_compensation_snap_exclusion"
    )
    assert probe.binding_inputs == (_OUTPUT,)
    assert probe.budget_measure == "snap"
    assert probe.effect_direction == "reform_minus_baseline"
    assert probe.expected_sign == "positive"
    assert probe.min_abs_effect == 10_000_000.0
    sources = probe.parameter_changes["gov.usda.snap.income.sources.unearned"][
        "2024-01-01.2024-12-31"
    ]
    assert _OUTPUT not in sources
    assert "disability_benefits" in sources

    support_builder = (ROOT / "tools/build_us_puf_support_base.py").read_text()
    fiscal_builder = (ROOT / "tools/build_us_fiscal_refresh_release.py").read_text()
    cache_driver = (ROOT / "experiments/build_j_recert/buildj_base.sh").read_text()
    assert support_builder.count("with_us_workers_compensation(") == 4
    assert "us_workers_compensation_signal_gate(" in support_builder
    assert "us_workers_compensation_signal_gate(" in fiscal_builder
    assert f'"{_OUTPUT}"' in cache_driver
