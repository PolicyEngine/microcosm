"""Tests split from packages/microcosm-build/tests/test_us_scf_wealth.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_scf_wealth import *


def test_stage_spec_loads_and_declares_outputs() -> None:
    spec = us_scf_wealth_stage_spec()
    assert isinstance(spec, SourceStageSpec)
    assert spec.stage == US_SCF_WEALTH_STAGE_NAME
    for column in (
        *US_SCF_FINANCIAL_ASSET_OUTPUT_COLUMNS,
        *US_SCF_NET_WORTH_OUTPUT_COLUMNS,
    ):
        assert column in spec.outputs


def test_output_columns_are_the_ssi_countable_resource_leaves() -> None:
    assert US_SCF_FINANCIAL_ASSET_OUTPUT_COLUMNS == (
        "bank_account_assets",
        "stock_assets",
        "bond_assets",
    )


def test_stock_target_sums_stocks_and_nmmf() -> None:
    assert SCF_FINANCIAL_ASSET_TARGET_COMPONENTS["stock_assets"] == ("stocks", "nmmf")
    assert SCF_FINANCIAL_ASSET_TARGET_COMPONENTS["bank_account_assets"] == ("liq",)
    assert SCF_FINANCIAL_ASSET_TARGET_COMPONENTS["bond_assets"] == ("bond",)


def test_net_worth_target_is_the_direct_signed_scf_anchor() -> None:
    assert SCF_NET_WORTH_TARGET_COMPONENTS == {"net_worth": ("networth",)}
    assert US_SCF_NET_WORTH_OUTPUT_COLUMNS == ("net_worth",)
    assert US_SCF_WEALTH_NONCONSTANT_HOUSEHOLD_COLUMNS == ("net_worth",)

    spec = us_scf_wealth_stage_spec()
    derivation = next(
        operation
        for operation in spec.operations
        if operation.kind == "derive"
        and "net_worth_anchor" in operation.parameters["outputs"]
    )
    assert derivation.parameters["net_worth_source_column"] == "networth"
    fit = next(
        operation
        for operation in spec.operations
        if operation.kind == "fit_weighted_qrf"
    )
    assert fit.parameters["net_worth_target"] == "networth"
    assert fit.parameters["net_worth_signed"] is True
    assert "utils/asset_imputation.py lines 15-19" in spec.notes
    assert "calibration/source_impute.py lines 1324-1338" in spec.notes


def test_load_donor_derives_targets_predictors_and_weight(tmp_path) -> None:
    raw = _raw_scf_summary()
    path = tmp_path / "rscfp2022.dta"
    raw.to_stata(path, write_index=False)
    donor = load_scf_2022_financial_asset_donor(path)
    for column in (
        *US_SCF_FINANCIAL_ASSET_OUTPUT_COLUMNS,
        *US_SCF_NET_WORTH_OUTPUT_COLUMNS,
        *SCF_WEALTH_PREDICTORS,
        _DONOR_WEIGHT_COLUMN,
    ):
        assert column in donor.columns
    # stock_assets is the stocks + nmmf sum.
    expected_stock = raw["stocks"].to_numpy() + raw["nmmf"].to_numpy()
    np.testing.assert_allclose(
        np.sort(donor["stock_assets"].to_numpy()), np.sort(expected_stock), rtol=1e-6
    )
    # All weights positive; targets non-negative.
    assert (donor[_DONOR_WEIGHT_COLUMN] > 0).all()
    for column in US_SCF_FINANCIAL_ASSET_OUTPUT_COLUMNS:
        assert (donor[column] >= 0).all()
    # Unlike the three SSI leaves, net worth is a signed direct carry.
    np.testing.assert_allclose(donor["net_worth"], raw["networth"], rtol=1e-6)
    assert (donor["net_worth"] < 0).any()
    assert (donor["net_worth"] > 0).any()


def test_load_donor_missing_column_raises(tmp_path) -> None:
    raw = _raw_scf_summary().drop(columns=["liq"])
    path = tmp_path / "rscfp2022.dta"
    raw.to_stata(path, write_index=False)
    with pytest.raises(ValueError, match="missing required column"):
        load_scf_2022_financial_asset_donor(path)


def test_replace_sentinels_zeroes_scf_missing_codes() -> None:
    series = pd.Series([-1.0, -7.0, -8.0, -9.0, 5_000.0, 0.0])
    cleaned = _replace_sentinels(series).to_numpy()
    np.testing.assert_array_equal(cleaned, [0.0, 0.0, 0.0, 0.0, 5_000.0, 0.0])


def test_recipient_cps_race_mapping() -> None:
    person = pd.DataFrame(
        {
            "PRDTRACE": [1, 2, 4, 3, 1, 5],
            "PRDTHSP": [0, 0, 0, 0, 2, 0],
        }
    )
    race = _recipient_cps_race(person)
    # White, Black, Asian, Other(race 3), Hispanic-overrides, Other(race 5).
    np.testing.assert_array_equal(race, [1.0, 2.0, 4.0, 7.0, 3.0, 7.0])


def test_household_head_mask_is_one_per_household_at_lowest_line() -> None:
    person = pd.DataFrame(
        {
            "person_household_id": [5, 5, 5, 9, 9],
            "A_LINENO": [3, 1, 2, 2, 1],
        }
    )
    mask = _household_head_mask(person)
    # Head of hh 5 is the A_LINENO==1 row (index 1); head of hh 9 is index 4.
    np.testing.assert_array_equal(mask, [False, True, False, False, True])
    assert mask.sum() == 2


def test_impute_head_carries_and_zeroes_non_heads() -> None:
    person = _person_rows(60)
    donor = _donor_table()
    result = impute_us_scf_financial_assets(person, donor, seed=42, n_estimators=20)
    assert list(result.columns) == list(US_SCF_FINANCIAL_ASSET_OUTPUT_COLUMNS)
    assert len(result) == len(person)
    head_mask = _household_head_mask(person)
    for column in US_SCF_FINANCIAL_ASSET_OUTPUT_COLUMNS:
        values = result[column].to_numpy()
        assert (values >= 0).all()  # non-negative
        # Every non-head person carries exactly $0 (head-carry).
        assert np.all(values[~head_mask] == 0.0)
    # Heads carry signal: bank assets are not all zero on the heads.
    assert result["bank_account_assets"].to_numpy()[head_mask].sum() > 0


def test_impute_is_deterministic_for_a_seed() -> None:
    person = _person_rows(40)
    donor = _donor_table()
    a = impute_us_scf_financial_assets(person, donor, seed=7, n_estimators=15)
    b = impute_us_scf_financial_assets(person, donor, seed=7, n_estimators=15)
    for column in US_SCF_FINANCIAL_ASSET_OUTPUT_COLUMNS:
        np.testing.assert_array_equal(a[column].to_numpy(), b[column].to_numpy())


def test_sipp_scf_blend_is_deterministic_for_fixed_seed(
    sipp_scf_blend_case: dict[str, object],
) -> None:
    person = sipp_scf_blend_case["person"]
    blended = sipp_scf_blend_case["blended"]
    assert isinstance(person, pd.DataFrame)
    assert isinstance(blended, pd.DataFrame)
    repeated = impute_us_sipp_scf_financial_assets(
        person,
        sipp_scf_blend_case["scf_donor"],
        sipp_scf_blend_case["sipp_donor"],
        seed=int(sipp_scf_blend_case["seed"]),
        time_period=TIME_PERIOD,
        n_estimators=int(sipp_scf_blend_case["n_estimators"]),
    )
    np.testing.assert_array_equal(blended.to_numpy(), repeated.to_numpy())
    assert (
        blended.attrs[FINANCIAL_ASSET_BLEND_AUDIT_KEY]
        == repeated.attrs[FINANCIAL_ASSET_BLEND_AUDIT_KEY]
    )


def test_seeded_household_source_share_is_50_50_within_tolerance(
    sipp_scf_blend_case: dict[str, object],
) -> None:
    blended = sipp_scf_blend_case["blended"]
    assert isinstance(blended, pd.DataFrame)
    audit = blended.attrs[FINANCIAL_ASSET_BLEND_AUDIT_KEY]
    assert audit["scf_probability"] == FINANCIAL_ASSET_SOURCE_SCF_PROBABILITY
    assert audit["scf_household_count"] > 0
    assert audit["sipp_household_count"] > 0
    assert 0.4 <= audit["scf_household_share"] <= 0.6

    household_ids = np.arange(1, 401)
    selected = financial_asset_source_is_scf(
        household_ids,
        seed=23,
        time_period=TIME_PERIOD,
    )
    reversed_selected = financial_asset_source_is_scf(
        household_ids[::-1],
        seed=23,
        time_period=TIME_PERIOD,
    )
    np.testing.assert_array_equal(selected, reversed_selected[::-1])
    assert 0.4 <= selected.mean() <= 0.6


def test_sipp_draws_have_lower_median_bank_assets_and_raise_low_tail(
    sipp_scf_blend_case: dict[str, object],
) -> None:
    blended = sipp_scf_blend_case["blended"]
    assert isinstance(blended, pd.DataFrame)
    audit = blended.attrs[FINANCIAL_ASSET_BLEND_AUDIT_KEY]
    assert audit["sipp_selected_bank_median"] < audit["scf_selected_bank_median"]
    assert audit["blended_bank_low_tail_share"] > audit["scf_only_bank_low_tail_share"]


def test_blend_uses_one_complete_source_vector_and_reference_person_carry(
    sipp_scf_blend_case: dict[str, object],
) -> None:
    person = sipp_scf_blend_case["person"]
    blended = sipp_scf_blend_case["blended"]
    scf = sipp_scf_blend_case["scf"]
    sipp = sipp_scf_blend_case["sipp"]
    assert isinstance(person, pd.DataFrame)
    assert isinstance(blended, pd.DataFrame)
    assert isinstance(scf, pd.DataFrame)
    assert isinstance(sipp, pd.DataFrame)
    head_mask = _household_head_mask(person)
    selected_scf = financial_asset_source_is_scf(
        person.loc[head_mask, "person_household_id"],
        seed=int(sipp_scf_blend_case["seed"]),
        time_period=TIME_PERIOD,
    )
    source_mask = np.zeros(len(person), dtype=bool)
    source_mask[np.flatnonzero(head_mask)] = selected_scf
    expected = sipp.copy()
    expected.loc[source_mask, list(US_SCF_FINANCIAL_ASSET_OUTPUT_COLUMNS)] = scf.loc[
        source_mask, list(US_SCF_FINANCIAL_ASSET_OUTPUT_COLUMNS)
    ].to_numpy()
    np.testing.assert_array_equal(blended.to_numpy(), expected.to_numpy())
    assert (
        blended.loc[~head_mask, list(US_SCF_FINANCIAL_ASSET_OUTPUT_COLUMNS)]
        .eq(0.0)
        .all(axis=None)
    )


def test_impute_missing_donor_column_raises() -> None:
    person = _person_rows(10)
    donor = _donor_table().drop(columns=["bond_assets"])
    with pytest.raises(ValueError, match="donor table missing column"):
        impute_us_scf_financial_assets(person, donor, seed=0, n_estimators=10)


def test_impute_net_worth_is_signed_finite_and_household_aligned() -> None:
    person = _person_rows(200)
    household = _us_frame(person).table("household").iloc[::-1].copy()
    result = impute_us_scf_net_worth(
        person,
        household,
        _donor_table(),
        seed=42,
        n_estimators=20,
    )

    assert result.name == "net_worth"
    assert result.index.equals(household.index)
    assert np.isfinite(result.to_numpy()).all()
    assert (result > 0).any()
    assert (result < 0).any()
    assert result.nunique() > 1


def test_with_inputs_writes_asset_and_net_worth_columns() -> None:
    frame = _us_frame(_person_rows(60))
    donor = _donor_table()
    out = with_us_scf_wealth_inputs(
        frame, seed=42, time_period=TIME_PERIOD, scf_donor=donor
    )
    person = out.table("person")
    for column in US_SCF_FINANCIAL_ASSET_OUTPUT_COLUMNS:
        assert column in person.columns
        assert person[column].to_numpy().dtype == np.float64
    assert person["bank_account_assets"].to_numpy().sum() > 0
    net_worth = out.table("household")["net_worth"]
    assert net_worth.to_numpy().dtype == np.float64
    assert net_worth.nunique() > 1
    assert (net_worth < 0).any()


def test_carry_signal_tolerates_a_single_constant_leaf() -> None:
    # Bond holdings are ~97% zero in the donor: a healthy draw on a small
    # frame can produce an all-zero bond column. That must NOT read as an
    # engine-default surface (per-leaf nonconstancy made pass-through
    # platform-dependent — the #510 CI failure). Only an all-leaves-constant
    # surface forces re-imputation.
    frame = _us_frame(_person_rows(60))
    donor = _donor_table()
    once = with_us_scf_wealth_inputs(
        frame, seed=42, time_period=TIME_PERIOD, scf_donor=donor
    )

    def _with_person(base, mutate):
        tables = {entity: base.table(entity).copy() for entity in base.entities}
        mutate(tables["person"])
        return Frame(
            tables,
            base.schema,
            {entity: base.weights_for(entity) for entity in base.weighted_entities},
        )

    def _zero_bond(person):
        person["bond_assets"] = 0.0

    degenerate_bond = _with_person(once, _zero_bond)
    twice = with_us_scf_wealth_inputs(
        degenerate_bond, seed=99, time_period=TIME_PERIOD, scf_donor=donor
    )
    np.testing.assert_array_equal(
        degenerate_bond.table("person")["bank_account_assets"].to_numpy(),
        twice.table("person")["bank_account_assets"].to_numpy(),
    )

    def _zero_all(person):
        for column in US_SCF_FINANCIAL_ASSET_OUTPUT_COLUMNS:
            person[column] = 0.0

    redrawn = with_us_scf_wealth_inputs(
        _with_person(once, _zero_all),
        seed=99,
        time_period=TIME_PERIOD,
        scf_donor=donor,
    )
    assert redrawn.table("person")["bank_account_assets"].to_numpy().sum() > 0

    def _distinct_constants(person):
        person["bank_account_assets"] = 5.0
        person["stock_assets"] = 3.0
        person["bond_assets"] = 0.0

    # Three DISTINCT constant leaves are still an untrustworthy surface:
    # flattened cross-leaf uniqueness would wave it through.
    reimputed = with_us_scf_wealth_inputs(
        _with_person(once, _distinct_constants),
        seed=99,
        time_period=TIME_PERIOD,
        scf_donor=donor,
    )
    bank = reimputed.table("person")["bank_account_assets"].to_numpy()
    assert np.unique(bank).size >= 2


def test_with_inputs_is_idempotent_when_signal_present() -> None:
    frame = _us_frame(_person_rows(60))
    donor = _donor_table()
    once = with_us_scf_wealth_inputs(
        frame, seed=42, time_period=TIME_PERIOD, scf_donor=donor
    )
    twice = with_us_scf_wealth_inputs(
        once, seed=99, time_period=TIME_PERIOD, scf_donor=donor
    )
    # Passing through untouched: the second call (different seed) does not
    # re-impute because the surface already carries signal.
    np.testing.assert_array_equal(
        once.table("person")["bank_account_assets"].to_numpy(),
        twice.table("person")["bank_account_assets"].to_numpy(),
    )
    np.testing.assert_array_equal(
        once.table("household")["net_worth"].to_numpy(),
        twice.table("household")["net_worth"].to_numpy(),
    )


def test_with_inputs_reimputes_when_bank_assets_constant() -> None:
    person = _person_rows(60)
    person["bank_account_assets"] = 0.0  # the engine-default landmine
    person["stock_assets"] = 0.0
    person["bond_assets"] = 0.0
    frame = _us_frame(person)
    donor = _donor_table()
    out = with_us_scf_wealth_inputs(
        frame, seed=42, time_period=TIME_PERIOD, scf_donor=donor
    )
    assert out.table("person")["bank_account_assets"].to_numpy().sum() > 0


def test_with_inputs_sipp_donor_heals_scf_only_surface_and_preserves_audit(
    monkeypatch,
) -> None:
    person = _person_rows(60)
    head_mask = _household_head_mask(person)
    for offset, column in enumerate(US_SCF_FINANCIAL_ASSET_OUTPUT_COLUMNS, start=1):
        person[column] = np.where(head_mask, np.arange(len(person)) + offset, 0.0)
    net_worth = np.linspace(10_000.0, 100_000.0, 60)
    net_worth[:6] *= -1.0
    frame = _us_frame(person, household_extra={"net_worth": net_worth})
    calls: list[tuple[int, int]] = []

    def fake_blend(
        recipient,
        scf_donor,
        sipp_donor,
        *,
        seed,
        time_period,
        n_estimators=100,
    ):
        del scf_donor, sipp_donor, n_estimators
        calls.append((seed, time_period))
        result = pd.DataFrame(
            {
                column: np.where(head_mask, 100.0 + position, 0.0)
                for position, column in enumerate(US_SCF_FINANCIAL_ASSET_OUTPUT_COLUMNS)
            },
            index=recipient.index,
        )
        result.attrs[FINANCIAL_ASSET_BLEND_AUDIT_KEY] = {
            "schema_version": 1,
            "seed": seed,
            "time_period": time_period,
            "scf_probability": FINANCIAL_ASSET_SOURCE_SCF_PROBABILITY,
        }
        return result

    monkeypatch.setattr(
        scf_wealth_runtime,
        "impute_us_sipp_scf_financial_assets",
        fake_blend,
    )
    out = with_us_scf_wealth_inputs(
        frame,
        seed=23,
        time_period=TIME_PERIOD,
        scf_donor=pd.DataFrame(),
        sipp_donor=pd.DataFrame(),
    )

    assert calls == [(23, TIME_PERIOD)]
    assert out.table("person").attrs[FINANCIAL_ASSET_BLEND_AUDIT_KEY]["seed"] == 23
    assert (out.table("person").loc[~head_mask, "bank_account_assets"] == 0.0).all()
    repeated = with_us_scf_wealth_inputs(
        out,
        seed=23,
        time_period=TIME_PERIOD,
        scf_donor=pd.DataFrame(),
        sipp_donor=pd.DataFrame(),
    )
    assert repeated is out
    assert calls == [(23, TIME_PERIOD)]


def test_with_inputs_heals_nonfinite_net_worth_without_redrawing_assets() -> None:
    once = with_us_scf_wealth_inputs(
        _us_frame(_person_rows(60)),
        seed=42,
        time_period=TIME_PERIOD,
        scf_donor=_donor_table(),
    )
    tables = {entity: once.table(entity).copy() for entity in once.entities}
    original_assets = tables["person"]["bank_account_assets"].to_numpy().copy()
    tables["household"].loc[tables["household"].index[0], "net_worth"] = np.inf
    damaged = Frame(
        tables,
        once.schema,
        {entity: once.weights_for(entity) for entity in once.weighted_entities},
    )

    healed = with_us_scf_wealth_inputs(
        damaged,
        seed=99,
        time_period=TIME_PERIOD,
        scf_donor=_donor_table(),
    )

    np.testing.assert_array_equal(
        healed.table("person")["bank_account_assets"].to_numpy(), original_assets
    )
    assert np.isfinite(healed.table("household")["net_worth"]).all()


def test_signal_gate_passes_on_imputed_surface() -> None:
    frame = _us_frame(_person_rows(200))
    donor = _donor_table()
    out = with_us_scf_wealth_inputs(
        frame, seed=42, time_period=TIME_PERIOD, scf_donor=donor
    )
    gate = us_scf_wealth_signal_gate(out)
    assert gate.passed, gate.failures


def test_signal_gate_observes_seeded_sipp_scf_blend(
    sipp_scf_blend_case: dict[str, object],
) -> None:
    person = sipp_scf_blend_case["person"]
    blended = sipp_scf_blend_case["blended"]
    assert isinstance(person, pd.DataFrame)
    assert isinstance(blended, pd.DataFrame)
    base = _us_frame(person)
    tables = {entity: base.table(entity).copy() for entity in base.entities}
    for column in US_SCF_FINANCIAL_ASSET_OUTPUT_COLUMNS:
        tables["person"][column] = blended[column].to_numpy(dtype=np.float64)
    tables["person"].attrs[FINANCIAL_ASSET_BLEND_AUDIT_KEY] = dict(
        blended.attrs[FINANCIAL_ASSET_BLEND_AUDIT_KEY]
    )
    net_worth = np.linspace(10_000.0, 250_000.0, len(tables["household"]))
    net_worth[:24] *= -1.0
    tables["household"]["net_worth"] = net_worth
    frame = Frame(
        tables,
        base.schema,
        {entity: base.weights_for(entity) for entity in base.weighted_entities},
    )

    gate = us_scf_wealth_signal_gate(frame, require_sipp_blend=True)
    assert gate.passed, gate.failures
    assert gate.details["financial_asset_blend"]["scf_household_count"] > 0
    assert gate.details["financial_asset_blend"]["sipp_household_count"] > 0

    failed_tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    failed_audit = dict(failed_tables["person"].attrs[FINANCIAL_ASSET_BLEND_AUDIT_KEY])
    failed_audit["scf_only_bank_low_tail_share"] = failed_audit[
        "blended_bank_low_tail_share"
    ]
    failed_tables["person"].attrs[FINANCIAL_ASSET_BLEND_AUDIT_KEY] = failed_audit
    no_low_tail_gain = Frame(
        failed_tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
    )
    failed_gate = us_scf_wealth_signal_gate(
        no_low_tail_gain,
        require_sipp_blend=True,
    )
    assert not failed_gate.passed
    assert any(
        "low-tail share did not increase" in failure for failure in failed_gate.failures
    )


def test_signal_gate_fails_when_columns_missing() -> None:
    frame = _us_frame(_person_rows(4))
    gate = us_scf_wealth_signal_gate(frame)
    assert not gate.passed
    assert any("missing" in f for f in gate.failures)


def test_signal_gate_fails_on_constant_zero_surface() -> None:
    person = _person_rows(20)
    person["bank_account_assets"] = 0.0
    person["stock_assets"] = 0.0
    person["bond_assets"] = 0.0
    frame = _us_frame(
        person,
        household_extra={"net_worth": np.zeros(20, dtype=np.float64)},
    )
    gate = us_scf_wealth_signal_gate(frame)
    assert not gate.passed
    assert any("constant" in f or "nonzero share" in f for f in gate.failures)


def test_signal_gate_requires_signed_net_worth_support() -> None:
    out = with_us_scf_wealth_inputs(
        _us_frame(_person_rows(200)),
        seed=42,
        time_period=TIME_PERIOD,
        scf_donor=_donor_table(),
    )
    tables = {entity: out.table(entity).copy() for entity in out.entities}
    tables["household"]["net_worth"] = (
        np.abs(tables["household"]["net_worth"].to_numpy()) + 1.0
    )
    positive_only = Frame(
        tables,
        out.schema,
        {entity: out.weights_for(entity) for entity in out.weighted_entities},
    )

    gate = us_scf_wealth_signal_gate(positive_only)
    assert not gate.passed
    assert any("net_worth negative share" in failure for failure in gate.failures)


def test_summary_reports_shares_and_bands() -> None:
    frame = _us_frame(_person_rows(200))
    out = with_us_scf_wealth_inputs(
        frame, seed=42, time_period=TIME_PERIOD, scf_donor=_donor_table()
    )
    summary = us_scf_wealth_summary(out)
    assert 0.0 <= summary["bank_account_assets_nonzero_share"] <= 1.0
    assert "bank_nonzero_share_band" in summary
    assert summary["unique_counts"]["bank_account_assets"] >= 2
    assert summary["net_worth_unique_count"] >= 2
    assert 0.0 < summary["net_worth_negative_share"] < 1.0
    assert 0.0 < summary["net_worth_positive_share"] < 1.0


def test_fetch_returns_cached_file_without_network(tmp_path) -> None:
    cached = tmp_path / "rscfp2022.dta"
    cached.write_bytes(b"stub")
    # A pre-existing non-empty cache file is returned as-is (no network call).
    # Verification is disabled so the stub is trusted; the point of this test is
    # the no-network cache-return path, not the pin.
    result = fetch_scf_2022_summary_extract(
        cache_dir=tmp_path,
        expected_member_sha256=None,
        expected_zip_sha256=None,
    )
    assert result == cached
    assert result.read_bytes() == b"stub"


def test_fetch_reuses_cache_only_when_member_sha_matches(tmp_path) -> None:
    from microcosm.build.us_runtime.scf_wealth import _sha256_hexdigest

    cached = tmp_path / "rscfp2022.dta"
    cached.write_bytes(b"pinned-payload")
    pinned = _sha256_hexdigest(b"pinned-payload")
    # A cache hit whose digest matches the pin is returned without a network
    # call (any network attempt in this offline test would raise).
    result = fetch_scf_2022_summary_extract(
        cache_dir=tmp_path,
        expected_member_sha256=pinned,
        expected_zip_sha256=None,
    )
    assert result == cached
    assert result.read_bytes() == b"pinned-payload"
