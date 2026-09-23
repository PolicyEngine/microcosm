"""Tests split from packages/microcosm-build/tests/test_us_ssi_disability_criteria.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_ssi_disability_criteria import *


def test_reported_ssi_anchor_coalesces_native_asec_and_harmonized_acs() -> None:
    person = pd.DataFrame(
        {
            "SSI_VAL": [1_200.0, np.nan, np.nan],
            "ssi_reported": [np.nan, 900.0, np.nan],
        }
    )

    values = module._reported_ssi_anchor(
        person,
        age=np.asarray([40.0, 50.0, 10.0]),
    )

    np.testing.assert_array_equal(values, [1_200.0, 900.0, 0.0])
    assert pd.isna(person.loc[2, "ssi_reported"])


def test_reported_ssi_anchor_refuses_an_adult_universe_blank() -> None:
    person = pd.DataFrame(
        {
            "SSI_VAL": [np.nan],
            "ssi_reported": [np.nan],
        }
    )

    with pytest.raises(ValueError, match="blank only below"):
        module._reported_ssi_anchor(person, age=np.asarray([40.0]))


def test_archived_coordinates_exact_predictors_and_pinned_artifact() -> None:
    assert US_SSI_DISABILITY_CRITERIA_STAGE_NAME == "ssi_disability_criteria"
    assert US_SSI_DISABILITY_CRITERIA_OUTPUT_COLUMNS == (
        "meets_ssi_disability_criteria",
    )
    assert US_SSI_DISABILITY_CRITERIA_NONCONSTANT_PERSON_COLUMNS == (
        "meets_ssi_disability_criteria",
    )
    assert SIPP_2023_SSI_DISABILITY_DONOR_REVISION == (
        "21280dca5995e978d706740a8a4b9b7860cfd7b6"
    )
    assert SIPP_2023_SSI_DISABILITY_DONOR_SHA256 == (
        "5c30439e365fc26483318ef61d1d8f4bb2f0e9d6bb47c22c06756a7698733ee2"
    )
    assert SIPP_2023_SSI_DISABILITY_DONOR_SIZE_BYTES == 3_726_010_471
    assert SIPP_2023_SSI_DISABILITY_DONOR_REVISION in (
        SIPP_2023_SSI_DISABILITY_DONOR_URL
    )
    assert SSI_DISABILITY_ARCHIVED_SIPP_URL.endswith("datasets/sipp/sipp.py#L63-L105")
    assert SSI_DISABILITY_ARCHIVED_CPS_URL.endswith("datasets/cps/cps.py#L2853-L2886")
    assert SSI_DISABILITY_ARCHIVED_SOURCE_IMPUTE_URL.endswith(
        "calibration/source_impute.py#L869-L990"
    )
    assert SSI_DISABILITY_ARCHIVED_EXTENDED_CPS_URL.endswith(
        "datasets/cps/extended_cps.py#L392-L424"
    )
    assert len(SIPP_SSI_DISABILITY_MODEL_PREDICTORS) == 19
    assert SIPP_SSI_DISABILITY_MODEL_PREDICTORS == (
        "age",
        "is_female",
        "is_married",
        "employment_income",
        "interest_income",
        "dividend_income",
        "rental_income",
        "bank_account_assets",
        "stock_assets",
        "bond_assets",
        "count_under_18",
        *SIPP_SSI_DISABILITY_DIFFICULTY_PREDICTORS,
        "social_security_disability",
        "has_disability_income",
    )


def test_stage_manifest_pins_exact_runtime_contract() -> None:
    spec = us_ssi_disability_criteria_stage_spec()

    assert spec.grain == "person"
    assert spec.outputs == US_SSI_DISABILITY_CRITERIA_OUTPUT_COLUMNS
    assert [operation.kind for operation in spec.operations] == [
        "read_table",
        "fit_weighted_qrf",
    ]
    assert dict(spec.operations[0].parameters) == SIPP_SSI_DISABILITY_READ_PARAMETERS
    assert dict(spec.operations[1].parameters) == SIPP_SSI_DISABILITY_FIT_PARAMETERS
    assert SIPP_SSI_DISABILITY_FIT_PARAMETERS["training_sample_seed"] == (
        8_386_123_572_872_638_692
    )
    assert SIPP_SSI_DISABILITY_FIT_PARAMETERS["model_seed"] == 42
    assert SIPP_SSI_DISABILITY_FIT_PARAMETERS["seed_from_build_config"] is False


def test_loader_applies_observation_allocation_and_financial_screens(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        module,
        "_ssi_policy_screen_values",
        lambda _year: {
            "individual_resource_limit": 2_000.0,
            "couple_resource_limit": 3_000.0,
            "individual_fbr": 943.0,
            "couple_fbr": 1_415.0,
            "general_exclusion": 20.0,
            "earned_exclusion": 65.0,
            "earned_share_excluded": 0.5,
            "non_blind_sga": 1_550.0,
        },
    )
    rows = [
        _source_row("month11", 1, month=11),
        # A positive observed label survives otherwise failing finances.
        _source_row(
            "positive",
            1,
            received_ssi=1,
            reason=1,
            assets=100_000,
            monthly_earnings=10_000,
        ),
        # Nonrecipients do not need an observed reason.
        _source_row("negative", 1, reason=np.nan),
        _source_row("high_assets", 1, assets=10_000),
        _source_row(
            "allocated_reason",
            1,
            received_ssi=1,
            reason=1,
            reason_allocation=2,
        ),
        _source_row("allocated_receipt", 1, receipt_allocation=2),
        _source_row("aged_negative", 1, age=70),
        # $1,600 monthly passes countable-income but fails nonblind SGA.
        _source_row("sga", 1, monthly_earnings=1_600),
        _source_row(
            "blind_sga",
            1,
            monthly_earnings=1_600,
            difficulty_seeing=True,
        ),
        # A reported aged reason is a clean false label for an under-65 row.
        _source_row("aged_reason", 1, received_ssi=1, reason=2),
    ]
    donor = load_sipp_2023_ssi_disability_donor(
        _write_source(tmp_path, rows),
        expected_size_bytes=None,
        chunksize=3,
    )

    assert len(donor) == 4
    assert donor[_OUTPUT].tolist() == [True, False, False, False]
    assert donor["difficulty_seeing"].tolist() == [False, False, True, False]
    audit = donor.attrs["source_audit"]
    assert audit["december_rows"] == 9
    assert audit["training_rows"] == 4
    assert audit["positive_rows"] == 1
    assert audit["negative_rows"] == 3
    assert audit["pinned_transform"] is False


def test_loader_rejects_missing_allocation_flags(tmp_path: Path) -> None:
    path = _write_source(tmp_path, [_source_row("one", 1)])
    source = pd.read_csv(path, sep="|").drop(columns=["ASSI_BRSN"])
    source.to_csv(path, sep="|", index=False)

    with pytest.raises(ValueError, match="ASSI_BRSN"):
        load_sipp_2023_ssi_disability_donor(
            path,
            expected_size_bytes=None,
        )


def test_pinned_full_file_audit_contract_is_exact() -> None:
    assert module._PINNED_DECEMBER_ROWS == 39_513
    assert module._PINNED_TRAINING_ROWS == 9_346
    assert module._PINNED_POSITIVE_ROWS == 577
    assert module._PINNED_NEGATIVE_ROWS == 8_769
    assert module._PINNED_WEIGHT_SUM == pytest.approx(88_690_359.47893329)
    assert module._PINNED_POSITIVE_WEIGHT_SUM == pytest.approx(4_937_167.914119501)
    assert module._PINNED_WEIGHTED_TRUE_SHARE == pytest.approx(0.05566746987074994)
    assert module._PINNED_RESAMPLE_UNIQUE_SOURCE_ROWS == 5_314
    assert module._PINNED_RESAMPLE_POSITIVE_ROWS == 524
    assert module._PINNED_RESAMPLE_TRUE_SHARE == pytest.approx(0.05606676653113631)


def test_imputer_uses_exact_weighted_replacement_draw_and_fixed_model_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    donor = _donor(120)
    monkeypatch.setattr(module, "QRF", _FakeQRF)

    first = impute_us_ssi_disability_criteria(_frame(), donor, seed=1)
    first_fit = _FakeQRF.instances[-1]
    second = impute_us_ssi_disability_criteria(_frame(), donor, seed=999)
    second_fit = _FakeQRF.instances[-1]

    probability = donor["household_weight"].to_numpy(dtype=np.float64).copy()
    probability /= probability.sum()
    expected_positions = np.random.default_rng(8_386_123_572_872_638_692).choice(
        120, size=120, replace=True, p=probability
    )
    expected_ages = donor.iloc[expected_positions]["age"].to_numpy()
    assert first_fit.n_estimators == 100
    assert first_fit.seed == 42
    assert first_fit.weights == "none"
    np.testing.assert_array_equal(first_fit.training["age"], expected_ages)
    np.testing.assert_array_equal(second_fit.training["age"], expected_ages)
    np.testing.assert_array_equal(first, second)


def test_receiver_uses_complete_income_components_and_archived_signal_screen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "QRF", _FakeQRF)
    result = impute_us_ssi_disability_criteria(_frame(), _donor(), seed=7)
    receiver = _FakeQRF.instances[-1].receiver
    assert receiver is not None

    np.testing.assert_array_equal(
        receiver["interest_income"],
        np.arange(20, dtype=np.float64) + 2.0,
    )
    np.testing.assert_array_equal(
        receiver["dividend_income"],
        np.arange(20, dtype=np.float64) * 3.0 + 4.0,
    )
    # Person 1 is preserved by direct reported SSI despite no model/signal;
    # person 2 has both the positive model draw and a difficulty. Person 3 has
    # a positive model draw but negative SSDI, which is not a disability signal.
    assert np.flatnonzero(result.to_numpy()).tolist() == [0, 1]


def test_asec_reporter_anchor_is_not_copied_to_puf_and_rows_predict_separately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "QRF", _FakeQRF)
    expanded = clone_us_frame_for_puf_support(_frame())
    person = expanded.table("person")
    puf = person["person_support_channel"].astype(str).eq("puf_tax_detail")
    source_three = person["person_source_id"].eq(3)
    # Give only the PUF row of source person 3 a positive model draw and signal.
    person.loc[puf & source_three, "bank_account_assets"] = 100.0
    person.loc[puf & source_three, "PEDISDRS"] = 1

    result = impute_us_ssi_disability_criteria(expanded, _donor(), seed=7)
    assert len(_FakeQRF.predict_receivers) == 2
    assert _FakeQRF.predict_start_offsets == [0, 0]
    assert [len(receiver) for receiver in _FakeQRF.predict_receivers] == [20, 20]
    assert [
        expanded.table("person")
        .loc[receiver.index, "person_support_channel"]
        .unique()
        .tolist()
        for receiver in _FakeQRF.predict_receivers
    ] == [["asec"], ["puf_tax_detail"]]
    rows = pd.DataFrame(
        {
            "source": person["person_source_id"].to_numpy(),
            "channel": person["person_support_channel"].astype(str).to_numpy(),
            "value": result.to_numpy(),
        }
    )
    reporter = rows[rows["source"] == 1].set_index("channel")["value"]
    source_three_values = rows[rows["source"] == 3].set_index("channel")["value"]

    assert bool(reporter["asec"])
    assert not bool(reporter["puf_tax_detail"])
    assert not bool(source_three_values["asec"])
    assert bool(source_three_values["puf_tax_detail"])


def test_support_validation_allows_puf_only_source_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "QRF", _FakeQRF)
    expanded = clone_us_frame_for_puf_support(_frame())
    person = expanded.table("person")
    puf = person["person_support_channel"].astype(str).eq("puf_tax_detail")
    person.loc[puf, "person_source_id"] += 10_000

    result = impute_us_ssi_disability_criteria(expanded, _donor(), seed=0)

    assert len(result) == len(person)


def test_support_validation_rejects_unknown_or_missing_channel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "QRF", _FakeQRF)
    expanded = clone_us_frame_for_puf_support(_frame())
    expanded.table("person").loc[0, "person_support_channel"] = "mystery"

    with pytest.raises(ValueError, match="unsupported support channel"):
        impute_us_ssi_disability_criteria(expanded, _donor(), seed=0)


def test_wrapper_heals_stale_output_and_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "QRF", _FakeQRF)
    monkeypatch.setattr(module, "us_ssi_disability_criteria_stage_spec", lambda: None)
    stale = _replace_person(_frame(), **{_OUTPUT: np.ones(20, dtype=bool)})

    healed = with_us_ssi_disability_criteria(
        stale,
        seed=123,
        time_period=2024,
        sipp_donor=_donor(),
    )
    twice = with_us_ssi_disability_criteria(
        healed,
        seed=999,
        time_period=2024,
        sipp_donor=_donor(),
    )

    assert healed.table("person")[_OUTPUT].sum() == 2
    assert twice is healed


def test_signal_gate_requires_each_channel_but_allows_clone_divergence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "QRF", _FakeQRF)
    expanded = clone_us_frame_for_puf_support(_frame())
    person = expanded.table("person")
    puf = person["person_support_channel"].astype(str).eq("puf_tax_detail")
    source_three = person["person_source_id"].eq(3)
    person.loc[puf & source_three, "bank_account_assets"] = 100.0
    person.loc[puf & source_three, "PEDISDRS"] = 1
    values = impute_us_ssi_disability_criteria(expanded, _donor(), seed=0)
    valid = _replace_person(expanded, **{_OUTPUT: values.to_numpy()})

    summary = us_ssi_disability_criteria_summary(valid)
    gate = us_ssi_disability_criteria_signal_gate(valid)
    assert gate.passed, gate.failures
    assert summary["clone_divergence_source_people"] == 2
    assert summary["channels"]["asec"]["unique_count"] == 2
    assert summary["channels"]["puf_tax_detail"]["unique_count"] == 2

    dead_values = values.copy()
    dead_values.loc[puf.to_numpy()] = False
    dead = _replace_person(valid, **{_OUTPUT: dead_values.to_numpy()})
    dead_gate = us_ssi_disability_criteria_signal_gate(dead)
    assert not dead_gate.passed
    assert any("puf_tax_detail" in failure for failure in dead_gate.failures)

    implausible_values = values.copy()
    implausible_values.loc[puf.to_numpy()] = True
    first_puf = int(np.flatnonzero(puf.to_numpy())[0])
    implausible_values.iloc[first_puf] = False
    implausible = _replace_person(
        valid,
        **{_OUTPUT: implausible_values.to_numpy()},
    )
    implausible_gate = us_ssi_disability_criteria_signal_gate(implausible)
    assert not implausible_gate.passed
    assert any(
        "puf_tax_detail" in failure and "plausibility band" in failure
        for failure in implausible_gate.failures
    )


def test_stacked_clone_divergence_diagnostic_checks_clone_two() -> None:
    stacked = _replace_person(
        _frame(3),
        **{
            "person_source_id": np.asarray([10, 10, 10]),
            "person_spine_source_id": np.asarray([1, 1, 1]),
            "person_support_channel": np.asarray(["acs", "acs", "acs"]),
            "person_support_clone_index": np.asarray([0, 1, 2]),
            _OUTPUT: np.asarray([False, False, True]),
        },
    )

    summary = us_ssi_disability_criteria_summary(stacked)

    assert summary["clone_divergence_source_people"] == 1


def test_summary_checks_harmonized_ssi_on_native_role() -> None:
    expanded = clone_us_frame_for_puf_support(_frame())
    person = expanded.table("person")
    person["ssi_reported"] = np.nan
    native = person["person_support_channel"].astype(str).eq("asec")
    source_two = person["person_source_id"].eq(2)
    person.loc[native & source_two, "SSI_VAL"] = np.nan
    person.loc[native & source_two, "ssi_reported"] = 900.0
    preserved_existing_anchor = (native & person["person_source_id"].eq(1)).to_numpy()
    invalid = _replace_person(
        expanded,
        **{_OUTPUT: preserved_existing_anchor},
    )

    summary = us_ssi_disability_criteria_summary(invalid)
    gate = us_ssi_disability_criteria_signal_gate(invalid)

    assert summary["reporter_anchor_mismatches"] == 1
    assert any(
        "native-role SSI reporter anchor" in failure for failure in gate.failures
    )


def test_gate_requires_complete_support_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "QRF", _FakeQRF)
    expanded = clone_us_frame_for_puf_support(_frame())
    values = impute_us_ssi_disability_criteria(expanded, _donor(), seed=0)
    tables = {entity: expanded.table(entity).copy() for entity in expanded.entities}
    tables["person"][_OUTPUT] = values.to_numpy()
    tables["person"] = tables["person"].drop(columns=["person_source_id"])
    without_source_id = Frame(
        tables,
        expanded.schema,
        {entity: expanded.weights_for(entity) for entity in expanded.weighted_entities},
        expanded.strata,
        mass_log=expanded.mass_log,
    )

    gate = us_ssi_disability_criteria_signal_gate(without_source_id)
    assert not gate.passed
    assert gate.details["support_provenance_missing"] is True
    assert any("provenance" in failure for failure in gate.failures)

    tables = {entity: expanded.table(entity).copy() for entity in expanded.entities}
    tables["person"][_OUTPUT] = values.to_numpy()
    tables["person"] = tables["person"].drop(columns=["person_support_channel"])
    without_channel = Frame(
        tables,
        expanded.schema,
        {entity: expanded.weights_for(entity) for entity in expanded.weighted_entities},
        expanded.strata,
        mass_log=expanded.mass_log,
    )
    channel_gate = us_ssi_disability_criteria_signal_gate(without_channel)
    assert not channel_gate.passed
    assert channel_gate.details["support_provenance_missing"] is True
    assert any(
        "support channel 'asec' is missing" in failure
        for failure in channel_gate.failures
    )
    assert any(
        "support channel 'puf_tax_detail' is missing" in failure
        for failure in channel_gate.failures
    )
