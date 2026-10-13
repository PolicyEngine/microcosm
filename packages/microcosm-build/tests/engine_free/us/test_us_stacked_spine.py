"""Tests split from packages/microcosm-build/tests/test_us_stacked_spine.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_stacked_spine import *


def test_stacked_assembly_is_deterministic_and_floor_exact() -> None:
    first = assemble_stacked_spine(
        _asec_source(),
        _acs_source(),
        acs_sample_fraction=0.25,
        acs_sample_seed=578,
    )
    repeated = assemble_stacked_spine(
        _asec_source(),
        _acs_source(),
        acs_sample_fraction=0.25,
        acs_sample_seed=578,
    )

    sample = first.receipt["acs_sample"]
    assert sample["eligible_household_count"] == 10
    assert sample["requested_household_count"] == 2
    assert sample["realized_household_count"] == 2
    assert sample["exact_count_rule"] == "floor(fraction * eligible)"
    assert first.receipt["acs_sample_fraction"] == 0.25
    assert first.receipt["acs_sample_seed"] == 578
    assert (
        first.receipt["acs_sample"]["selected_household_ids_sha256"]
        == repeated.receipt["acs_sample"]["selected_household_ids_sha256"]
    )
    for entity in first.frame.entities:
        assert_frame_equal(first.frame.table(entity), repeated.frame.table(entity))
    np.testing.assert_array_equal(
        first.frame.weights_for("household").values,
        repeated.frame.weights_for("household").values,
    )

    manifest = first.frame.metadata[STACKED_SPINE_MANIFEST_KEY]
    assert manifest["acs_sample_fraction"] == 0.25
    assert manifest["acs_sample_seed"] == 578
    assert validate_stacked_spine_frame(
        first.frame,
        boundary="determinism fixture",
    )


def test_stacked_assembly_seed_and_fraction_bind_identity() -> None:
    base = assemble_stacked_spine(
        _asec_source(),
        _acs_source(),
        acs_sample_fraction=0.25,
        acs_sample_seed=578,
    )
    changed_seed = assemble_stacked_spine(
        _asec_source(),
        _acs_source(),
        acs_sample_fraction=0.25,
        acs_sample_seed=579,
    )
    wider_fraction = assemble_stacked_spine(
        _asec_source(),
        _acs_source(),
        acs_sample_fraction=0.55,
        acs_sample_seed=578,
    )

    assert (
        base.receipt["acs_sample"]["selected_household_ids_sha256"]
        != changed_seed.receipt["acs_sample"]["selected_household_ids_sha256"]
    )
    assert wider_fraction.receipt["acs_sample"]["realized_household_count"] == 5


def test_production_sampling_is_uniform_and_composition_preserving() -> None:
    asec = _asec_source()
    acs = _acs_source()
    result = assemble_stacked_spine(
        asec,
        acs,
        sample_fraction=0.5,
        sample_seed=578,
    )

    assert result.receipt["version"] == 4
    assert result.receipt["sample_fraction"] == 0.5
    assert result.receipt["sample_seed"] == 578
    samples = result.receipt["survey_samples"]
    assert samples["asec"]["realized_household_count"] == 1
    assert samples["acs"]["realized_household_count"] == 5
    for channel, source in (("asec", asec), ("acs", acs)):
        sample = samples[channel]
        assert sample["fraction"] == 0.5
        assert sample["seed"] == 578
        assert np.isclose(
            sample["normalized_household_mass"],
            source.weights_for("household").total,
        )

    # The rung changes row counts, not either arm's population meaning.
    assert np.isclose(
        result.frame.weights_for("household").total,
        asec.weights_for("household").total,
    )
    validate_stacked_spine_frame(result.frame, boundary="production sample fixture")


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("sample_fraction", 1.0, "sample fraction does not match"),
        ("sample_seed", 579, "sample seed does not match"),
    ),
)
def test_production_sampling_identity_mutations_fail_closed(
    field: str,
    value: object,
    match: str,
) -> None:
    result = assemble_stacked_spine(
        _asec_source(),
        _acs_source(),
        sample_fraction=0.5,
        sample_seed=578,
    )
    manifest = deepcopy(result.receipt)
    manifest[field] = value
    tampered = Frame(
        {entity: result.frame.table(entity) for entity in result.frame.entities},
        result.frame.schema,
        {
            entity: result.frame.weights_for(entity)
            for entity in result.frame.weighted_entities
        },
        result.frame.strata,
        mass_log=result.frame.mass_log,
        metadata={
            **result.frame.metadata,
            STACKED_SPINE_MANIFEST_KEY: manifest,
        },
    )

    with pytest.raises(ValueError, match=match):
        validate_stacked_spine_frame(tampered, boundary="mutated sample identity")


def test_production_and_legacy_sampling_controls_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="exactly one complete sampling control"):
        assemble_stacked_spine(
            _asec_source(),
            _acs_source(),
            acs_sample_fraction=0.5,
            acs_sample_seed=578,
            sample_fraction=0.5,
            sample_seed=578,
        )


@pytest.mark.parametrize(
    ("mutate", "match"),
    (
        (
            lambda manifest: manifest["acs_sample"].__setitem__(
                "realized_household_count", 3
            ),
            "realized household count",
        ),
        (
            lambda manifest: manifest["acs_sample"].__setitem__(
                "selected_household_ids_sha256", "0" * 64
            ),
            "selection digest",
        ),
        (
            lambda manifest: manifest.__setitem__("acs_sample_fraction", 0.35),
            "sample fraction does not match",
        ),
        (
            lambda manifest: manifest.__setitem__("acs_sample_seed", "578"),
            "acs_sample_seed",
        ),
        (
            lambda manifest: manifest.pop("acs_sample"),
            "sample receipt",
        ),
    ),
)
def test_stacked_manifest_mutations_fail_closed(mutate, match) -> None:
    result = assemble_stacked_spine(
        _asec_source(),
        _acs_source(),
        acs_sample_fraction=0.25,
        acs_sample_seed=578,
    )
    stacked = result.frame
    manifest = {
        key: (
            {
                nested_key: (
                    dict(nested_value)
                    if isinstance(nested_value, dict)
                    else nested_value
                )
                for nested_key, nested_value in value.items()
            }
            if isinstance(value, dict)
            else value
        )
        for key, value in result.receipt.items()
    }
    mutate(manifest)
    tampered = Frame(
        {entity: stacked.table(entity) for entity in stacked.entities},
        stacked.schema,
        {entity: stacked.weights_for(entity) for entity in stacked.weighted_entities},
        stacked.strata,
        mass_log=stacked.mass_log,
        metadata={**stacked.metadata, STACKED_SPINE_MANIFEST_KEY: manifest},
    )

    with pytest.raises(ValueError, match=match):
        validate_stacked_spine_frame(tampered, boundary="tampered fixture")


def test_sample_acs_households_takes_whole_lineages() -> None:
    acs = _acs_source()
    sampled, receipt = sample_acs_households(acs, fraction=0.55, seed=7)

    assert receipt["requested_household_count"] == 5
    selected_households = set(sampled.table("household")["household_id"].tolist())
    person = sampled.table("person")
    assert set(person["person_household_id"]) == selected_households
    full_person = acs.table("person")
    for household_id in selected_households:
        expected = int((full_person["person_household_id"] == household_id).sum())
        actual = int((person["person_household_id"] == household_id).sum())
        assert actual == expected
    for group in ("tax_unit", "spm_unit", "family", "marital_unit"):
        assert set(sampled.table(group)[f"{group}_id"]) == set(
            person[f"person_{group}_id"]
        )


def test_sample_floor_zero_fails_closed() -> None:
    with pytest.raises(ValueError, match="floors to zero"):
        sample_acs_households(_acs_source(), fraction=0.05, seed=1)


def test_sample_rejects_provenance_carrying_source() -> None:
    stacked = assemble_stacked_spine(
        _asec_source(),
        _acs_source(),
        acs_sample_fraction=1.0,
        acs_sample_seed=0,
    ).frame
    with pytest.raises(ValueError, match="already carries support provenance"):
        sample_acs_households(stacked, fraction=0.5, seed=1)


def test_weight_harmonization_matches_share_math() -> None:
    asec = _asec_source()
    acs = _acs_source()
    result = assemble_stacked_spine(
        asec,
        acs,
        acs_sample_fraction=0.25,
        acs_sample_seed=578,
    )
    stacked = result.frame
    anchor_mass = float(asec.weights_for("household").total)

    household = stacked.table("household")
    weights = stacked.weights_for("household").values
    channel = household[support_channel_column("household")]
    asec_mass = float(weights[channel.eq("asec").to_numpy()].sum())
    acs_mass = float(weights[channel.eq("acs").to_numpy()].sum())
    assert np.isclose(asec_mass, 0.5 * anchor_mass, rtol=1e-12)
    assert np.isclose(acs_mass, 0.5 * anchor_mass, rtol=1e-12)
    assert float(stacked.weights_for("household").total) == anchor_mass

    harmonization = result.receipt["weight_harmonization"]
    sampled_mass = result.receipt["acs_sample"]["sampled_household_mass"]
    assert np.isclose(
        harmonization["acs"]["scale_factor"],
        0.5 * anchor_mass / sampled_mass,
        rtol=1e-12,
    )
    assert harmonization["asec"]["incoming_mass"] == anchor_mass
    assert np.isclose(
        harmonization["asec"]["scale_factor"],
        0.5,
        rtol=1e-12,
    )

    acs_weights = weights[channel.eq("acs").to_numpy()]
    source_ids = household.loc[
        channel.eq("acs").to_numpy(),
        spine_source_id_column("household"),
    ].to_numpy()
    incoming_by_id = dict(
        zip(
            acs.table("household")["household_id"].tolist(),
            acs.weights_for("household").values.tolist(),
            strict=True,
        )
    )
    incoming = np.asarray(
        [incoming_by_id[int(value)] for value in source_ids],
        dtype=np.float64,
    )
    np.testing.assert_allclose(
        acs_weights,
        incoming * harmonization["acs"]["scale_factor"],
        rtol=1e-9,
    )


def test_weight_harmonization_receipts_use_the_live_selected_anchor() -> None:
    result = assemble_stacked_spine(
        _asec_source(),
        _acs_source(),
        acs_sample_fraction=1.0,
        acs_sample_seed=0,
        mass_anchor_channel="acs",
    )

    assert result.frame.weights_for("household").total == 550.0
    harmonization = result.receipt["weight_harmonization"]
    for channel in ("asec", "acs"):
        assert harmonization[channel]["declared_allocation"] == 275.0
        assert harmonization[channel]["allocated_mass"] == 275.0

    metadata = {
        **result.frame.metadata,
        STACKED_SPINE_MANIFEST_KEY: deepcopy(result.receipt),
    }
    metadata[STACKED_SPINE_MANIFEST_KEY]["weight_harmonization"]["asec"][
        "declared_allocation"
    ] = 200.0
    forged = Frame(
        {entity: result.frame.table(entity) for entity in result.frame.entities},
        result.frame.schema,
        {
            entity: result.frame.weights_for(entity)
            for entity in result.frame.weighted_entities
        },
        result.frame.strata,
        mass_log=result.frame.mass_log,
        metadata=metadata,
    )
    with pytest.raises(
        ValueError,
        match=(
            "declared 'asec' allocation 200.0 differs from share 0.5 times "
            "live anchor mass 550.0"
        ),
    ):
        validate_stacked_spine_frame(forged, boundary="forged allocation")


def test_fraction_one_matches_plain_assembly() -> None:
    asec = _asec_source()
    acs = _acs_source()
    result = assemble_stacked_spine(
        asec,
        acs,
        acs_sample_fraction=1.0,
        acs_sample_seed=123,
    )
    plain = assemble_spines(
        {"asec": _asec_source(), "acs": _acs_source()},
        household_mass_shares=dict(DEFAULT_STACKED_HOUSEHOLD_MASS_SHARES),
        mass_anchor_channel="asec",
    )

    sample = result.receipt["acs_sample"]
    assert sample["realized_household_count"] == sample["eligible_household_count"]
    for entity in plain.entities:
        assert_frame_equal(result.frame.table(entity), plain.table(entity))
    np.testing.assert_array_equal(
        result.frame.weights_for("household").values,
        plain.weights_for("household").values,
    )


def test_selection_digest_uses_raw_spine_ids_under_collision_remap() -> None:
    asec = _source_frame(
        household_ids=[101, 102],
        weights=[300.0, 100.0],
        extra_person_columns={"asec_detail_income": 40.0},
        stratum="asec_2024",
    )
    result = assemble_stacked_spine(
        asec,
        _acs_source(),
        acs_sample_fraction=1.0,
        acs_sample_seed=0,
    )

    household = result.frame.table("household")
    channel = household[support_channel_column("household")]
    clone_index = household[support_clone_index_column("household")]
    native_acs = channel.eq("acs") & clone_index.eq(0)
    remapped = household.loc[native_acs, "household_id"].to_numpy()
    raw = household.loc[native_acs, spine_source_id_column("household")].to_numpy()
    assert not np.array_equal(np.sort(remapped), np.sort(raw))
    assert validate_stacked_spine_frame(
        result.frame,
        boundary="collision fixture",
    )


def test_s_corp_universe_zero_materializes_native_rows_with_exact_receipt() -> None:
    frame = _s_corp_universe_fixture()
    input_person = frame.table("person").copy(deep=True)
    donor = pd.DataFrame({"s_corp_income": [0.0, -0.0, 0.0]})

    materialized, receipt = (
        stacked_spine_module._materialize_us_puf_s_corp_universe_zero(frame, donor)
    )

    clone_column = support_clone_index_column("person")
    clone_index = input_person[clone_column]
    native = clone_index.eq(0)
    assert input_person.loc[native, "s_corp_income"].isna().all()
    assert materialized.table("person")["s_corp_income"].eq(0.0).all()
    assert receipt["rule"] == (
        stacked_spine_module.us_puf_s_corp_universe_zero_rule_identity()
    )
    assert receipt["status"] == "materialized"
    assert receipt["donor_rows_verified"] == 3
    assert receipt["native_rows_materialized"] == int(native.sum())
    assert receipt["produced_rows_verified"] == int((clone_index > 0).sum())
    assert receipt["person_rows"] == len(input_person)
    assert receipt["person_rows_by_clone_role"] == {
        str(role): int(clone_index.eq(role).sum()) for role in (0, 1, 2)
    }
    assert receipt["post_materialization_nonfinite_rows"] == 0
    assert receipt["post_materialization_nonzero_rows"] == 0
    assert len(receipt["donor_values_sha256"]) == 64
    assert len(receipt["person_values_sha256"]) == 64
    assert receipt["sha256"] == stacked_spine_module._canonical_sha256(
        {key: value for key, value in receipt.items() if key != "sha256"}
    )


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("donor_nonfinite", r"donor precondition failed: 1 nonfinite"),
        ("donor_nonzero", r"donor precondition failed: 1 nonzero"),
        ("native_preexisting", r"native precondition failed: 1 native cell"),
        ("clone_nonfinite", r"clone precondition failed: 1 nonfinite"),
        ("tail_nonzero", r"clone precondition failed: 1 nonzero"),
    ),
)
def test_s_corp_universe_zero_fails_closed(
    mutation: str,
    match: str,
) -> None:
    frame = _s_corp_universe_fixture()
    donor = pd.DataFrame({"s_corp_income": [0.0, 0.0]})
    person = frame.table("person")
    clone_index = person[support_clone_index_column("person")]
    if mutation == "donor_nonfinite":
        donor.loc[0, "s_corp_income"] = np.nan
    elif mutation == "donor_nonzero":
        donor.loc[0, "s_corp_income"] = 1.0
    elif mutation == "native_preexisting":
        person.loc[person.index[clone_index.eq(0)][0], "s_corp_income"] = 0.0
    elif mutation == "clone_nonfinite":
        person.loc[person.index[clone_index.eq(1)][0], "s_corp_income"] = np.nan
    else:
        person.loc[person.index[clone_index.eq(2)][0], "s_corp_income"] = 1.0

    with pytest.raises(ValueError, match=match):
        stacked_spine_module._materialize_us_puf_s_corp_universe_zero(frame, donor)


def test_stacked_primary_applies_s_corp_universe_rule_after_qrf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    def impute(frame: Frame, *_args: object, **kwargs: object) -> Frame:
        events.append("primary_qrf")
        receipts = kwargs["predictor_universe_receipts"]
        assert isinstance(receipts, list)
        receipts.append({"fixture": "recipient-universe"})
        person = frame.table("person").copy(deep=True)
        clone_index = person[support_clone_index_column("person")]
        person["s_corp_income"] = np.where(clone_index.eq(0), np.nan, 0.0)
        tables = {entity: frame.table(entity) for entity in frame.entities}
        tables["person"] = person
        return Frame(
            tables,
            frame.schema,
            {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
            frame.strata,
            mass_log=frame.mass_log,
            metadata=frame.metadata,
        )

    materialize = stacked_spine_module._materialize_us_puf_s_corp_universe_zero

    def materialize_after_qrf(
        frame: Frame,
        donor: pd.DataFrame,
    ) -> tuple[Frame, dict[str, object]]:
        events.append("s_corp_universe_zero")
        return materialize(frame, donor)

    monkeypatch.setattr(
        stacked_spine_module,
        "impute_us_puf_tax_detail_support",
        impute,
    )
    monkeypatch.setattr(
        stacked_spine_module,
        "_materialize_us_puf_s_corp_universe_zero",
        materialize_after_qrf,
    )

    result = stacked_spine_module._run_stacked_puf_pass_without_tail_for_test(
        _late_primary_entry(_stacked_gap_fixture()),
        pd.DataFrame({"s_corp_income": [0.0, 0.0]}),
        clone_attachment_fraction=1.0,
        clone_attachment_seed=578,
        predictors=(),
        person_outputs=("s_corp_income",),
        tax_unit_outputs=(),
    )

    assert events == ["primary_qrf", "s_corp_universe_zero"]
    assert result.frame.table("person")["s_corp_income"].eq(0.0).all()
    assert result.receipt["s_corp_income_universe_zero"]["status"] == "materialized"
    assert result.receipt["doctrines"]["whole_pool_output_universes"] == {
        "person.s_corp_income": (
            stacked_spine_module.us_puf_s_corp_universe_zero_rule_identity()
        )
    }


def test_finalize_preserve_nulls_keeps_unowned_cells_null() -> None:
    cloned = _cloned_stacked_fixture()
    predictions, donor = _finalize_fixture_predictions(cloned)
    before_person = cloned.table("person").copy(deep=True)

    finalized = finalize_us_puf_tax_detail_predictions(
        cloned,
        donor,
        predictions.copy(),
        person_outputs=("taxable_interest_income",),
        tax_unit_outputs=("health_savings_account_ald",),
        absent_cells=PUF_ABSENT_CELLS_PRESERVE_NULLS,
    )

    person = finalized.table("person")
    channel = person[support_channel_column("person")]
    clone_index = person[support_clone_index_column("person")]
    native_acs = channel.eq("acs") & clone_index.eq(0)
    native_asec = channel.eq("asec") & clone_index.eq(0)
    detail = clone_index.eq(1)
    assert person.loc[native_acs, "taxable_interest_income"].isna().all()
    pd.testing.assert_series_equal(
        person.loc[native_asec, "taxable_interest_income"],
        before_person.loc[native_asec.to_numpy(), "taxable_interest_income"],
        check_names=False,
    )
    assert person.loc[detail, "taxable_interest_income"].notna().all()

    tax_unit = finalized.table("tax_unit")
    tax_unit_clone = tax_unit[support_clone_index_column("tax_unit")]
    assert tax_unit.loc[tax_unit_clone.eq(0), "health_savings_account_ald"].isna().all()
    assert (
        tax_unit.loc[tax_unit_clone.eq(1), "health_savings_account_ald"].notna().all()
    )


def test_finalize_preserve_nulls_materializes_registry_boolean_outputs() -> None:
    cloned = _cloned_stacked_fixture()
    registry = stacked_spine_module.CANONICAL_ORIGIN_BATTERY_METRIC_REGISTRY
    boolean_outputs = tuple(
        column
        for column in puf_support_module.PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS
        if registry.get(("person", "puf_tax_itemization", column, 0))
        == "boolean_incidence"
    )
    assert set(boolean_outputs) == set(US_QBI_BOOLEAN_OUTPUT_COLUMNS)

    tax_unit = cloned.table("tax_unit")
    detail_tax_units = tax_unit[support_clone_index_column("tax_unit")].eq(1)
    predictions = pd.DataFrame(
        {
            column: np.ones(int(detail_tax_units.sum()), dtype=np.float64)
            for column in boolean_outputs
        },
        index=tax_unit.index[detail_tax_units],
    )
    donor = pd.DataFrame(
        {column: np.asarray([0.0, 1.0], dtype=np.float64) for column in boolean_outputs}
        | {"weight": np.ones(2, dtype=np.float64)}
    )

    finalized = finalize_us_puf_tax_detail_predictions(
        cloned,
        donor,
        predictions,
        person_outputs=boolean_outputs,
        tax_unit_outputs=(),
        absent_cells=PUF_ABSENT_CELLS_PRESERVE_NULLS,
    )

    person = finalized.table("person")
    detail_people = person[support_clone_index_column("person")].eq(1)
    for column in boolean_outputs:
        values = person[column]
        assert pd.api.types.is_bool_dtype(values.dtype), (column, values.dtype)
        assert values.loc[~detail_people].isna().all()
        assert values.loc[detail_people].notna().all()


def test_finalize_preserve_nulls_rejects_numeric_boolean_materialization() -> None:
    cloned = _cloned_stacked_fixture()
    column = US_QBI_BOOLEAN_OUTPUT_COLUMNS[0]
    person = cloned.table("person")
    person[column] = np.zeros(len(person), dtype=np.float64)
    tax_unit = cloned.table("tax_unit")
    detail_tax_units = tax_unit[support_clone_index_column("tax_unit")].eq(1)
    predictions = pd.DataFrame(
        {column: np.ones(int(detail_tax_units.sum()), dtype=np.float64)},
        index=tax_unit.index[detail_tax_units],
    )
    donor = pd.DataFrame(
        {column: np.asarray([0.0, 1.0]), "weight": np.ones(2, dtype=np.float64)}
    )

    with pytest.raises(
        TypeError,
        match=(
            rf"PUF boolean output {column!r} must contain only physical boolean "
            r"values.*dtype float64.*builtins\.float"
        ),
    ):
        finalize_us_puf_tax_detail_predictions(
            cloned,
            donor,
            predictions,
            person_outputs=(column,),
            tax_unit_outputs=(),
            absent_cells=PUF_ABSENT_CELLS_PRESERVE_NULLS,
        )


def test_finalize_legacy_zero_fill_reproduces_the_audited_defect() -> None:
    """Pin the run-7 boundary: legacy finalization reads absence as zero."""

    cloned = _cloned_stacked_fixture()
    predictions, donor = _finalize_fixture_predictions(cloned)

    finalized = finalize_us_puf_tax_detail_predictions(
        cloned,
        donor,
        predictions.copy(),
        person_outputs=("taxable_interest_income",),
        tax_unit_outputs=("health_savings_account_ald",),
    )

    person = finalized.table("person")
    channel = person[support_channel_column("person")]
    clone_index = person[support_clone_index_column("person")]
    native_acs = channel.eq("acs") & clone_index.eq(0)
    assert person.loc[native_acs, "taxable_interest_income"].eq(0.0).all()
    tax_unit = finalized.table("tax_unit")
    tax_unit_clone = tax_unit[support_clone_index_column("tax_unit")]
    assert (
        tax_unit.loc[tax_unit_clone.eq(0), "health_savings_account_ald"].eq(0.0).all()
    )


def test_finalize_legacy_zero_fill_matches_run_7_protocol_5_byte_pin() -> None:
    """Pin every legacy run-7 output byte, not only its zero-region meaning."""

    cloned = _cloned_stacked_fixture()
    predictions, donor = _finalize_fixture_predictions(cloned)
    finalized = finalize_us_puf_tax_detail_predictions(
        cloned,
        donor,
        predictions.copy(),
        person_outputs=("taxable_interest_income",),
        tax_unit_outputs=("health_savings_account_ald",),
    )

    actual = (
        finalized.table("person")["taxable_interest_income"].to_numpy(copy=True),
        finalized.table("tax_unit")["health_savings_account_ald"].to_numpy(copy=True),
    )
    actual_bytes = pickle.dumps(actual, protocol=5)
    assert actual_bytes[:2] == b"\x80\x05"
    assert hashlib.sha256(actual_bytes).hexdigest() == (
        "f2dc86f630a41c7b0eb060d7cd630bfadda423ae26c4311403116e3e6cb7720e"
    )


def test_preserve_nulls_sparsification_never_rewrites_native_rows() -> None:
    cloned = _cloned_stacked_fixture()
    predictions, donor = _finalize_fixture_predictions(cloned)
    donor = donor.assign(
        taxable_interest_income=[100.0, 0.0, 0.0, 0.0],
    )
    before_person = cloned.table("person").copy(deep=True)

    finalized = finalize_us_puf_tax_detail_predictions(
        cloned,
        donor,
        predictions.copy(),
        person_outputs=("taxable_interest_income",),
        tax_unit_outputs=("health_savings_account_ald",),
        absent_cells=PUF_ABSENT_CELLS_PRESERVE_NULLS,
    )

    person = finalized.table("person")
    clone_index = person[support_clone_index_column("person")]
    channel = person[support_channel_column("person")]
    native = clone_index.eq(0)
    native_asec = native & channel.eq("asec")
    pd.testing.assert_series_equal(
        person.loc[native_asec, "taxable_interest_income"],
        before_person.loc[native_asec.to_numpy(), "taxable_interest_income"],
        check_names=False,
    )
    assert (
        person.loc[native & channel.eq("acs"), "taxable_interest_income"].isna().all()
    )
    detail_units = (
        person.loc[clone_index.eq(1)]
        .groupby("person_tax_unit_id", sort=False)["taxable_interest_income"]
        .sum()
    )
    assert (detail_units == 0.0).any()


def test_strict_recipient_predictors_fail_closed_on_absence() -> None:
    cloned = _cloned_stacked_fixture()
    donor = pd.DataFrame(
        {
            "employment_income": [45_000.0, 8_000.0],
            "taxable_interest_income": [120.0, 30.0],
            "weight": [1.0, 1.0],
        }
    )

    with pytest.raises(ValueError, match="puf_predictor_employment_income"):
        prepare_us_puf_tax_detail_chain_inputs(
            cloned,
            donor,
            predictors=("puf_predictor_employment_income",),
            person_outputs=("taxable_interest_income",),
            tax_unit_outputs=(),
            require_complete_recipient_predictors=True,
        )

    legacy = prepare_us_puf_tax_detail_chain_inputs(
        cloned,
        donor,
        predictors=("puf_predictor_employment_income",),
        person_outputs=("taxable_interest_income",),
        tax_unit_outputs=(),
    )
    assert not legacy.recipient_features.isna().any().any()
    tax_unit = cloned.table("tax_unit")
    detail_mask = tax_unit[support_clone_index_column("tax_unit")].eq(1).to_numpy()
    acs_detail = (
        tax_unit.loc[detail_mask, support_channel_column("tax_unit")]
        .eq("acs")
        .to_numpy()
    )
    zero_filled = legacy.recipient_features.loc[
        acs_detail, "puf_predictor_employment_income"
    ]
    assert zero_filled.eq(0.0).all()


def test_strict_recipient_predictors_apply_exact_acs_age_universe() -> None:
    cloned = _cloned_acs_earnings_universe_fixture()
    before = cloned.table("person")[
        ["employment_income_before_lsr", "self_employment_income_before_lsr"]
    ].copy(deep=True)
    donor = pd.DataFrame(
        {
            "employment_income": [45_000.0, 8_000.0],
            "self_employment_income": [1_000.0, 0.0],
            "taxable_interest_income": [120.0, 30.0],
            "weight": [1.0, 1.0],
        }
    )

    prepared = prepare_us_puf_tax_detail_chain_inputs(
        cloned,
        donor,
        predictors=(
            "puf_predictor_employment_income",
            "puf_predictor_self_employment_income",
        ),
        person_outputs=("taxable_interest_income",),
        tax_unit_outputs=(),
        require_complete_recipient_predictors=True,
    )

    receipt = prepared.recipient_predictor_universe
    assert receipt["structurally_absent_person_rows"] == 4
    assert receipt["affected_tax_unit_rows"] == 2
    assert receipt["mixed_universe_tax_unit_rows"] == 1
    assert receipt["empty_universe_tax_unit_rows"] == 1
    assert receipt["raw_pums_source_cells_mutated"] is False
    assert receipt["mapped_person_cells_materialized"] is True
    assert len(receipt["sha256"]) == 64
    employment = prepared.recipient_features["puf_predictor_employment_income"]
    self_employment = prepared.recipient_features[
        "puf_predictor_self_employment_income"
    ]
    recipient_tax_units = cloned.table("tax_unit").loc[
        prepared.recipient_features.index
    ]
    acs_recipient = recipient_tax_units[support_channel_column("tax_unit")].eq("acs")
    assert employment.loc[acs_recipient].eq(0.0).sum() == 1
    assert self_employment.loc[acs_recipient].eq(0.0).sum() == 1
    assert employment.notna().all()
    assert self_employment.notna().all()
    assert receipt["predictor_source_mapping"] == {
        "puf_predictor_employment_income": {
            "entity": "person",
            "columns": ["employment_income_before_lsr"],
        },
        "puf_predictor_self_employment_income": {
            "entity": "person",
            "columns": ["self_employment_income_before_lsr"],
        },
    }
    pd.testing.assert_frame_equal(
        cloned.table("person")[before.columns],
        before,
    )


def test_acs_universe_raw_blanks_survive_checkpoint_round_trip(
    tmp_path: Path,
) -> None:
    frame = _cloned_acs_earnings_universe_fixture()
    person = frame.table("person")
    raw_columns = ["WAGP", "SEMP"]
    mapped_columns = [
        "employment_income_before_lsr",
        "self_employment_income_before_lsr",
    ]
    raw_before = person[raw_columns].copy(deep=True)
    mapped_before = person[mapped_columns].copy(deep=True)
    assert raw_before.isna().sum().eq(8).all()
    assert mapped_before.loc[raw_before.isna().any(axis=1)].eq(0.0).all().all()

    checkpoint_path = tmp_path / "acs-earnings-universe.frame.h5"
    write_frame_checkpoint(checkpoint_path, frame)
    restored = load_frame_checkpoint(checkpoint_path).frame.table("person")

    pd.testing.assert_frame_equal(
        restored[raw_columns],
        raw_before,
        check_dtype=True,
        check_exact=True,
    )
    pd.testing.assert_frame_equal(
        restored[mapped_columns],
        mapped_before,
        check_dtype=True,
        check_exact=True,
    )


def test_acs_universe_application_rejects_unreceipted_preexisting_zero() -> None:
    applied = _cloned_acs_earnings_universe_fixture()

    with pytest.raises(
        ValueError,
        match=("acs_2024_pums_wagp_age_15_plus.*unreceipted_preexisting_mapped_rows"),
    ):
        apply_acs_pums_earnings_universe_zeros(
            applied,
            boundary="reapplied ACS universe fixture",
        )


def test_strict_recipient_predictors_reject_cross_grain_source_collision() -> None:
    cloned = _cloned_acs_earnings_universe_fixture()
    cloned.table("tax_unit")["employment_income"] = 777.0
    donor = pd.DataFrame(
        {
            "employment_income": [45_000.0, 8_000.0],
            "taxable_interest_income": [120.0, 30.0],
            "weight": [1.0, 1.0],
        }
    )

    with pytest.raises(ValueError, match="ambiguous across entity grains"):
        prepare_us_puf_tax_detail_chain_inputs(
            cloned,
            donor,
            predictors=("puf_predictor_employment_income",),
            person_outputs=("taxable_interest_income",),
            tax_unit_outputs=(),
            require_complete_recipient_predictors=True,
        )


def test_strict_recipient_predictors_reject_infinite_feature() -> None:
    cloned = _cloned_acs_earnings_universe_fixture()
    person = cloned.table("person")
    candidate = (
        person[support_channel_column("person")].eq("acs")
        & person[support_clone_index_column("person")].eq(1)
        & person["age"].ge(15.0)
    )
    row = person.index[candidate][0]
    person.loc[row, "employment_income_before_lsr"] = np.inf
    person.loc[row, "WAGP"] = np.inf
    donor = pd.DataFrame(
        {
            "employment_income": [45_000.0, 8_000.0],
            "taxable_interest_income": [120.0, 30.0],
            "weight": [1.0, 1.0],
        }
    )

    with pytest.raises(ValueError, match="nonfinite values.*employment"):
        prepare_us_puf_tax_detail_chain_inputs(
            cloned,
            donor,
            predictors=("puf_predictor_employment_income",),
            person_outputs=("taxable_interest_income",),
            tax_unit_outputs=(),
            require_complete_recipient_predictors=True,
        )


@pytest.mark.parametrize(
    ("raw_source", "predictor"),
    [
        ("WAGP", "puf_predictor_employment_income"),
        ("SEMP", "puf_predictor_self_employment_income"),
    ],
)
def test_strict_recipient_predictors_require_raw_acs_universe_authority(
    raw_source: str,
    predictor: str,
) -> None:
    cloned = _cloned_acs_earnings_universe_fixture()
    cloned.table("person").drop(columns=[raw_source], inplace=True)
    donor_source = predictor.removeprefix("puf_predictor_")
    donor = pd.DataFrame(
        {
            donor_source: [45_000.0, 8_000.0],
            "taxable_interest_income": [120.0, 30.0],
            "weight": [1.0, 1.0],
        }
    )

    with pytest.raises(ValueError, match="raw_source_authority_missing"):
        prepare_us_puf_tax_detail_chain_inputs(
            cloned,
            donor,
            predictors=(predictor,),
            person_outputs=("taxable_interest_income",),
            tax_unit_outputs=(),
            require_complete_recipient_predictors=True,
        )


@pytest.mark.parametrize(
    ("age", "value", "message"),
    [
        (
            15.0,
            np.nan,
            "missing values before coercion.*acs_2024_pums_wagp_age_15_plus",
        ),
        (
            12.0,
            1.0,
            "acs_2024_pums_wagp_age_15_plus.*out_of_universe_mapped_nonzero_rows=1",
        ),
    ],
)
def test_strict_recipient_predictors_reject_acs_universe_mismatch(
    age: float,
    value: float,
    message: str,
) -> None:
    cloned = _cloned_acs_earnings_universe_fixture()
    person = cloned.table("person")
    candidate = (
        person[support_channel_column("person")].eq("acs")
        & person[support_clone_index_column("person")].eq(1)
        & person["age"].eq(12.0)
    )
    row = person.index[candidate][0]
    person.loc[row, "age"] = age
    person.loc[row, "employment_income_before_lsr"] = value
    person.loc[row, "WAGP"] = value
    donor = pd.DataFrame(
        {
            "employment_income": [45_000.0, 8_000.0],
            "taxable_interest_income": [120.0, 30.0],
            "weight": [1.0, 1.0],
        }
    )

    with pytest.raises(ValueError, match=message):
        prepare_us_puf_tax_detail_chain_inputs(
            cloned,
            donor,
            predictors=("puf_predictor_employment_income",),
            person_outputs=("taxable_interest_income",),
            tax_unit_outputs=(),
            require_complete_recipient_predictors=True,
        )


@pytest.mark.parametrize(
    ("minimum_age", "row_age", "value"),
    [
        (14, 14.0, np.nan),
        (16, 15.0, 100.0),
    ],
)
def test_acs_universe_age_rule_mutations_fail_closed_by_rule_id(
    monkeypatch: pytest.MonkeyPatch,
    minimum_age: int,
    row_age: float,
    value: float,
) -> None:
    frame = _cloned_acs_earnings_universe_fixture()
    person = frame.table("person")
    candidate = person[support_channel_column("person")].eq("acs") & person[
        support_clone_index_column("person")
    ].eq(1)
    row = person.index[candidate][0]
    person.loc[row, "age"] = row_age
    person.loc[row, "employment_income_before_lsr"] = value
    person.loc[row, "WAGP"] = value
    monkeypatch.setattr(
        universe_module,
        "ACS_PUMS_EARNINGS_MINIMUM_AGE",
        minimum_age,
    )
    donor = pd.DataFrame(
        {
            "employment_income": [45_000.0, 8_000.0],
            "taxable_interest_income": [120.0, 30.0],
            "weight": [1.0, 1.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="acs_2024_pums_wagp_age_15_plus",
    ):
        prepare_us_puf_tax_detail_chain_inputs(
            frame,
            donor,
            predictors=("puf_predictor_employment_income",),
            person_outputs=("taxable_interest_income",),
            tax_unit_outputs=(),
            require_complete_recipient_predictors=True,
        )


def test_strict_recipient_predictors_reject_null_filing_status_before_coercion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing filing status is terminal before status-code conversion."""

    cloned = _cloned_stacked_fixture()
    tax_unit = cloned.table("tax_unit")
    tax_unit["filing_status_input"] = "SINGLE"
    puf_mask = tax_unit[support_clone_index_column("tax_unit")].eq(1)
    tax_unit.loc[tax_unit.index[puf_mask][0], "filing_status_input"] = np.nan
    donor = pd.DataFrame(
        {
            "puf_predictor_filing_status_code": [1.0, 2.0],
            "taxable_interest_income": [120.0, 30.0],
            "weight": [1.0, 1.0],
        }
    )

    def reject_early_coercion(_values: object) -> np.ndarray:
        raise AssertionError("filing-status coercion ran before recipient validation")

    monkeypatch.setattr(
        puf_support_module,
        "_filing_status_codes",
        reject_early_coercion,
    )
    with pytest.raises(
        ValueError,
        match=r"puf_predictor_filing_status_code.*1.*recipient rows",
    ):
        prepare_us_puf_tax_detail_chain_inputs(
            cloned,
            donor,
            predictors=("puf_predictor_filing_status_code",),
            person_outputs=("taxable_interest_income",),
            tax_unit_outputs=(),
            require_complete_recipient_predictors=True,
        )


def test_production_entrypoints_take_no_authority_parameters() -> None:
    production_entrypoints = (
        gap_fill_stacked_spine,
        transfer_stacked_post_puf_inputs,
        stacked_completeness_gate,
        by_origin_battery,
    )
    authority_parameter_tokens = {
        "authority",
        "canonical",
        "declared",
        "metric",
        "metrics",
        "plan",
        "profile",
        "registry",
        "support",
        "surface",
    }

    for entrypoint in production_entrypoints:
        authority_parameters = {
            parameter
            for parameter in inspect.signature(entrypoint).parameters
            if authority_parameter_tokens.intersection(parameter.split("_"))
        }
        assert not authority_parameters, (
            f"{entrypoint.__name__} exposes caller-controlled authority "
            f"parameter(s): {sorted(authority_parameters)}"
        )


def test_canonical_gap_fill_rejects_nondefault_target_fit_width() -> None:
    with pytest.raises(
        ValueError,
        match="Canonical stacked gap fill requires max_targets_per_fit=8",
    ):
        gap_fill_stacked_spine(
            _stacked_gap_fixture(),
            max_targets_per_fit=1,
        )


def test_canonical_authority_objects_are_deeply_immutable() -> None:
    plan = stacked_spine_module.CANONICAL_STACKED_GAP_FILL_PLAN
    post_puf_surface = stacked_spine_module.CANONICAL_STACKED_POST_PUF_TRANSFER_SURFACE
    puf_producer_surface = (
        stacked_spine_module.CANONICAL_STACKED_POST_PUF_PUF_PRODUCER_SURFACE
    )
    source_producer_surface = (
        stacked_spine_module.CANONICAL_STACKED_POST_PUF_SOURCE_PRODUCER_SURFACE
    )
    surface = stacked_spine_module.CANONICAL_STACKED_DECLARED_SURFACE
    registry = stacked_spine_module.CANONICAL_ORIGIN_BATTERY_METRIC_REGISTRY
    joint_registry = stacked_spine_module.CANONICAL_ORIGIN_BATTERY_JOINT_METRIC_REGISTRY
    profile = stacked_spine_module.CANONICAL_ORIGIN_BATTERY_SUPPORT_PROFILE
    calibration = (
        stacked_spine_module._CANONICAL_STACKED_AUTHORITY.post_transfer_calibration
    )

    assert isinstance(plan, tuple)
    with pytest.raises(TypeError):
        plan[0].target_families["person"]["model_required_numeric"] = ()
    with pytest.raises(TypeError):
        post_puf_surface["person"]["puf_tax_itemization"] = ()
    with pytest.raises(TypeError):
        puf_producer_surface["person"]["puf_tax_itemization"] = ()
    with pytest.raises(TypeError):
        source_producer_surface["person"]["model_required_boolean"] = ()
    with pytest.raises(TypeError):
        surface["person"]["model_required_numeric"] = ()
    with pytest.raises(TypeError):
        registry[("person", "puf_tax_itemization", "taxable_interest_income", 0)] = (
            "rare_incidence"
        )
    with pytest.raises(TypeError):
        joint_registry[
            (
                "person",
                "source_operator_immigration",
                ("ssn_card_type", "immigration_status_str"),
                0,
            )
        ] = "categorical_tvd"
    with pytest.raises(FrozenInstanceError):
        profile.min_effective_support = 50
    with pytest.raises(TypeError):
        calibration["scope"] = {}
    with pytest.raises(TypeError):
        calibration["scope"]["reference"] = "forged"


def test_canonical_metric_registry_covers_the_declared_134_target_split() -> None:
    surface = stacked_spine_module.CANONICAL_STACKED_DECLARED_SURFACE
    registry = stacked_spine_module.CANONICAL_ORIGIN_BATTERY_METRIC_REGISTRY
    surface_targets = {
        (entity, family, target, 0)
        for entity, families in surface.items()
        for family, targets in families.items()
        for target in targets
    }

    assert len(surface_targets) == 134
    assert Counter(entity for entity, _family, _target, _clone in surface_targets) == {
        "person": 114,
        "tax_unit": 12,
        "spm_unit": 8,
    }
    assert (
        len(
            {
                (entity, family)
                for entity, families in surface.items()
                for family in families
            }
        )
        == 31
    )
    assert set(registry) == surface_targets
    assert Counter(registry.values()) == {
        "monetary_sign_separated": 79,
        "boolean_incidence": 51,
        "categorical_tvd": 4,
    }
    assert (
        registry[("person", "puf_tax_itemization", "taxable_interest_income", 0)]
        == "monetary_sign_separated"
    )
    gap_targets = {
        (entity, family, target, 0)
        for entity, families in (
            stacked_spine_module.CANONICAL_STACKED_GAP_FILL_SURFACE.items()
        )
        for family, targets in families.items()
        for target in targets
    }
    post_puf_targets = {
        (entity, family, target, 0)
        for entity, families in (
            stacked_spine_module.CANONICAL_STACKED_POST_PUF_TRANSFER_SURFACE.items()
        )
        for family, targets in families.items()
        for target in targets
    }
    puf_producer_targets = {
        (entity, family, target, 0)
        for entity, families in (
            stacked_spine_module.CANONICAL_STACKED_POST_PUF_PUF_PRODUCER_SURFACE.items()
        )
        for family, targets in families.items()
        for target in targets
    }
    source_producer_targets = {
        (entity, family, target, 0)
        for entity, families in (
            stacked_spine_module.CANONICAL_STACKED_POST_PUF_SOURCE_PRODUCER_SURFACE.items()
        )
        for family, targets in families.items()
        for target in targets
    }
    assert len(gap_targets) == 48
    assert len(post_puf_targets) == 70
    assert len(puf_producer_targets) == 43
    assert len(source_producer_targets) == 29
    assert len(puf_producer_targets & source_producer_targets) == 2
    assert puf_producer_targets | source_producer_targets == post_puf_targets
    assert gap_targets.isdisjoint(post_puf_targets)
    assert len(gap_targets | post_puf_targets) == 118
    assert gap_targets | post_puf_targets < surface_targets
    assert not {
        "bank_account_assets",
        "bond_assets",
        "stock_assets",
    } & {target for _entity, _family, target, _clone in surface_targets}


def test_canonical_metric_registry_drives_checkpoint_round_trip(
    tmp_path: Path,
) -> None:
    frame = _canonical_registry_checkpoint_frame()
    first_path = tmp_path / "registry-first.h5"
    second_path = tmp_path / "registry-second.h5"

    write_frame_checkpoint(first_path, frame)
    loaded = load_frame_checkpoint(first_path).frame
    write_frame_checkpoint(second_path, loaded)

    assert first_path.read_bytes() == second_path.read_bytes()
    registry = stacked_spine_module.CANONICAL_ORIGIN_BATTERY_METRIC_REGISTRY
    nullable_booleans = _transferred_registry_boolean_targets()
    assert Counter(registry.values()) == {
        "monetary_sign_separated": 79,
        "boolean_incidence": 51,
        "categorical_tvd": 4,
    }
    for (entity, _family, column, _clone_index), metric in registry.items():
        expected = frame.table(entity)[column]
        observed = loaded.table(entity)[column]
        pd.testing.assert_series_equal(
            observed,
            expected,
            check_dtype=True,
            check_exact=True,
        )
        if metric == "boolean_incidence":
            if (entity, column) in nullable_booleans:
                assert observed.dtype == pd.BooleanDtype()
                assert observed.isna().sum() == 1
            else:
                assert observed.dtype == np.dtype(np.bool_)
                assert not observed.isna().any()
        elif metric == "monetary_sign_separated":
            assert observed.dtype == np.dtype(np.float64)
        elif column in {"immigration_status_str", "ssn_card_type"}:
            assert observed.dtype == CANONICAL_STRING_DTYPE
        else:
            assert observed.dtype == np.dtype(np.float64)
            assert observed.isna().sum() == 1
    for column in ("is_female", "is_household_head"):
        observed = loaded.person[column]
        assert observed.dtype == pd.BooleanDtype()
        assert not observed.isna().any()


def test_checkpoint_boundary_extension_dtype_inventory_is_exact() -> None:
    registry = stacked_spine_module.CANONICAL_ORIGIN_BATTERY_METRIC_REGISTRY
    transferred_registry = _transferred_registry_boolean_targets()
    source_native = {
        ("person", "is_female"),
        ("person", "is_household_head"),
    }
    transferred = transferred_registry | source_native
    expected = {
        (
            "person",
            "attends_eligible_educational_institution_for_american_opportunity_credit",
        ),
        ("person", "business_is_sstb"),
        ("person", "estate_income_would_be_qualified"),
        ("person", "farm_operations_income_would_be_qualified"),
        ("person", "farm_rent_income_would_be_qualified"),
        ("person", "has_american_opportunity_credit_1098_t_or_exception"),
        ("person", "has_american_opportunity_credit_institution_ein"),
        ("person", "has_champva_health_coverage_at_interview"),
        ("person", "has_esi"),
        ("person", "has_indian_health_service_coverage_at_interview"),
        ("person", "has_marketplace_health_coverage_at_interview"),
        ("person", "has_medicaid_health_coverage_at_interview"),
        ("person", "has_non_marketplace_direct_purchase_health_coverage_at_interview"),
        ("person", "has_other_means_tested_health_coverage_at_interview"),
        ("person", "has_tricare_health_coverage_at_interview"),
        ("person", "has_va_health_coverage_at_interview"),
        ("person", "is_blind"),
        ("person", "is_disabled"),
        ("person", "is_enrolled_at_least_half_time_for_american_opportunity_credit"),
        ("person", "is_female"),
        ("person", "is_full_time_college_student"),
        ("person", "is_household_head"),
        ("person", "is_incapable_of_self_care"),
        ("person", "is_pregnant"),
        ("person", "is_pursuing_credential_for_american_opportunity_credit"),
        ("person", "is_separated"),
        ("person", "is_surviving_spouse"),
        ("person", "partnership_s_corp_income_would_be_qualified"),
        ("person", "previous_year_income_available"),
        ("person", "receives_wic"),
        ("person", "rental_income_would_be_qualified"),
        ("person", "self_employment_income_would_be_qualified"),
        ("person", "sstb_self_employment_income_would_be_qualified"),
        ("person", "takes_up_medicare_if_eligible"),
        ("person", "takes_up_wic_if_eligible"),
        ("spm_unit", "receives_tanf"),
        ("spm_unit", "receives_housing_assistance"),
        ("spm_unit", "receives_snap"),
        ("spm_unit", "takes_up_housing_assistance_if_eligible"),
    }
    assert len(transferred_registry) == 37
    assert transferred == expected
    assert len(transferred) == 39

    terminal_registry = {
        (entity, column)
        for (entity, _family, column, _clone_index), metric in registry.items()
        if metric == "boolean_incidence"
    }
    seeded_numpy_booleans = terminal_registry - transferred_registry
    assert seeded_numpy_booleans == {
        ("person", "takes_up_basic_health_program_if_eligible"),
        ("person", "takes_up_chip_if_eligible"),
        ("person", "takes_up_early_head_start_if_eligible"),
        ("person", "takes_up_head_start_if_eligible"),
        ("person", "takes_up_medicaid_if_eligible"),
        ("person", "takes_up_ssi_if_eligible"),
        ("spm_unit", "takes_up_snap_if_eligible"),
        ("spm_unit", "takes_up_tanf_if_eligible"),
        ("tax_unit", "takes_up_aca_if_eligible"),
        ("tax_unit", "takes_up_ca_premium_subsidy_if_eligible"),
        ("tax_unit", "takes_up_co_premium_assistance_if_eligible"),
        ("tax_unit", "takes_up_dc_ptc"),
        ("tax_unit", "takes_up_eitc"),
        ("tax_unit", "takes_up_nm_premium_assistance_if_eligible"),
    }
    assert len(seeded_numpy_booleans) == 14
    assembled_strings = {
        ("person", "PERIDNUM"),
        ("person", "source_person_id"),
        ("person", "tax_unit_role_input"),
        ("person", "person_support_channel"),
        ("household", "SERIALNO"),
        ("household", "ST"),
        ("household", "PUMA"),
        ("household", "puma_geoid"),
        ("household", "puma"),
        ("household", "tenure_type"),
        ("household", "household_support_channel"),
        ("tax_unit", "filing_status_input"),
        ("tax_unit", "tax_unit_support_channel"),
        ("spm_unit", "spm_unit_tenure_type"),
        ("spm_unit", "spm_unit_support_channel"),
        ("family", "family_support_channel"),
        ("marital_unit", "marital_unit_support_channel"),
    }
    transferred_strings = assembled_strings | {
        ("person", "immigration_status_str"),
        ("person", "ssn_card_type"),
    }
    boundary_inventory = {
        "assembled": {
            "boolean": frozenset(),
            "string": frozenset(assembled_strings),
        },
        "transferred": {
            "boolean": frozenset(transferred),
            "string": frozenset(transferred_strings),
        },
        "simulated": {
            "boolean": frozenset(transferred),
            "string": frozenset(transferred_strings),
        },
    }
    assert {
        stage: {family: len(columns) for family, columns in families.items()}
        for stage, families in boundary_inventory.items()
    } == {
        "assembled": {"boolean": 0, "string": 17},
        "transferred": {"boolean": 39, "string": 19},
        "simulated": {"boolean": 39, "string": 19},
    }
    assert sum(map(len, boundary_inventory["assembled"].values())) == 17
    assert sum(map(len, boundary_inventory["transferred"].values())) == 58
    assert boundary_inventory["transferred"] == boundary_inventory["simulated"]


def test_registry_drives_every_late_callback_dtype_family_check() -> None:
    registry = stacked_spine_module.CANONICAL_ORIGIN_BATTERY_METRIC_REGISTRY
    by_column = {
        (entity, column): metric
        for (entity, _family, column, _clone_index), metric in registry.items()
    }
    assert len(by_column) == len(registry) == 134

    representative = {
        "monetary_sign_separated": pd.Series([1.0, pd.NA], dtype="Float64"),
        "boolean_incidence": pd.Series([True, pd.NA], dtype="boolean"),
        "categorical_tvd": pd.Series([1, pd.NA], dtype="Int64"),
    }
    wrong = {
        "monetary_sign_separated": pd.Series([True], dtype=bool),
        "boolean_incidence": pd.Series([1.0], dtype=np.float64),
        "categorical_tvd": pd.Series([True], dtype=bool),
    }
    for metric in registry.values():
        assert stacked_spine_module._late_output_matches_metric_family(
            representative[metric],
            metric,
        )
        assert not stacked_spine_module._late_output_matches_metric_family(
            wrong[metric],
            metric,
        )
    assert stacked_spine_module._late_output_matches_metric_family(
        pd.Series(["NON_CITIZEN", pd.NA], dtype="string"),
        "categorical_tvd",
    )
    assert stacked_spine_module._late_output_matches_metric_family(
        pd.Series(pd.Categorical(["A", "B"])),
        "categorical_tvd",
    )
    assert not stacked_spine_module._late_output_matches_metric_family(
        pd.Series([True], dtype=object),
        "boolean_incidence",
    )

    registered_occurrences = [
        (contract.name, output.entity, output.column, by_column[key])
        for contract in (
            stacked_spine_module.CANONICAL_US_LATE_PRODUCER_REGISTRY.values()
        )
        for output in contract.outputs
        if (key := (output.entity, output.column)) in by_column
    ]
    assert len(registered_occurrences) == 163
    assert Counter(
        metric for _producer, _entity, _column, metric in registered_occurrences
    ) == {
        "monetary_sign_separated": 120,
        "boolean_incidence": 37,
        "categorical_tvd": 6,
    }
    unique_late_targets = {
        (entity, column): metric
        for _producer, entity, column, metric in registered_occurrences
    }
    assert len(unique_late_targets) == 90
    assert Counter(unique_late_targets.values()) == {
        "monetary_sign_separated": 67,
        "boolean_incidence": 20,
        "categorical_tvd": 3,
    }


def test_explicit_test_seams_reject_the_canonical_authority() -> None:
    authority = stacked_spine_module._production_stacked_authority()
    frame = _stacked_gap_fixture()

    with pytest.raises(ValueError, match="NON-CANONICAL test authority"):
        stacked_spine_module._gap_fill_stacked_spine_with_test_authority(
            frame,
            authority=authority,
        )
    with pytest.raises(ValueError, match="NON-CANONICAL test authority"):
        stacked_spine_module._stacked_completeness_gate_with_test_authority(
            frame,
            authority=authority,
        )
    with pytest.raises(ValueError, match="NON-CANONICAL test authority"):
        stacked_spine_module._by_origin_battery_with_test_authority(
            frame,
            authority=authority,
        )


def test_stacked_assembly_binds_native_acs_group_quarters_lineage() -> None:
    stacked = _stacked_gap_fixture()
    receipt = stacked.metadata[STACKED_SPINE_MANIFEST_KEY]["acs_native_group_quarters"]

    assert receipt["version"] == 1
    assert receipt["household_count"] == 1
    assert receipt["person_count"] == 1
    assert len(receipt["household_spine_source_ids_sha256"]) == 64
    assert len(receipt["person_spine_lineages_sha256"]) == 64
    validate_stacked_spine_frame(stacked, boundary="native ACS GQ receipt fixture")


@pytest.mark.parametrize(
    ("from_kind", "to_kind"),
    ((1, 2), (2, 1)),
)
def test_native_acs_group_quarters_reclassification_fails_closed(
    from_kind: int,
    to_kind: int,
) -> None:
    stacked = _stacked_gap_fixture()
    household = stacked.table("household").copy()
    channel = household[support_channel_column("household")].astype(str)
    kind = pd.to_numeric(household["TYPEHUGQ"], errors="coerce")
    row = household.index[channel.eq("acs") & kind.eq(from_kind)][0]
    household.loc[row, "TYPEHUGQ"] = to_kind
    household.loc[row, "tenure_type"] = np.nan if to_kind in (2, 3) else "RENTED"
    tables = {entity: stacked.table(entity) for entity in stacked.entities}
    tables["household"] = household
    corrupted = Frame(
        tables,
        stacked.schema,
        {entity: stacked.weights_for(entity) for entity in stacked.weighted_entities},
        stacked.strata,
        mass_log=stacked.mass_log,
        metadata=stacked.metadata,
    )

    with pytest.raises(
        ValueError,
        match="native ACS group-quarters household lineage differs",
    ):
        validate_stacked_spine_frame(
            corrupted,
            boundary="reclassified ACS GQ fixture",
        )


def test_acs_group_quarters_reclassification_across_clone_roles_fails_closed() -> None:
    attached = clone_us_frame_for_puf_support(
        _stacked_gap_fixture(),
        clone_attachment_fraction=1.0,
        clone_attachment_seed=578,
    )
    household = attached.table("household").copy()
    channel = household[support_channel_column("household")].astype(str)
    clone = household[support_clone_index_column("household")]
    kind = pd.to_numeric(household["TYPEHUGQ"], errors="coerce")
    row = household.index[channel.eq("acs") & clone.eq(1) & kind.isin((2, 3))][0]
    household.loc[row, "TYPEHUGQ"] = 1
    tables = {entity: attached.table(entity) for entity in attached.entities}
    tables["household"] = household
    corrupted = Frame(
        tables,
        attached.schema,
        {entity: attached.weights_for(entity) for entity in attached.weighted_entities},
        attached.strata,
        mass_log=attached.mass_log,
        metadata=attached.metadata,
    )

    with pytest.raises(
        ValueError,
        match="classification differs across clone roles",
    ):
        validate_stacked_spine_frame(
            corrupted,
            boundary="reclassified ACS GQ clone fixture",
        )


def test_acs_clone_cannot_substitute_group_quarters_support_lineage() -> None:
    attached = clone_us_frame_for_puf_support(
        _stacked_gap_fixture(),
        clone_attachment_fraction=1.0,
        clone_attachment_seed=578,
    )
    household = attached.table("household").copy()
    channel = household[support_channel_column("household")].astype(str)
    clone = household[support_clone_index_column("household")]
    kind = pd.to_numeric(household["TYPEHUGQ"], errors="coerce")
    gq_row = household.index[channel.eq("acs") & clone.eq(1) & kind.isin((2, 3))][0]
    housing_unit_row = household.index[channel.eq("acs") & clone.eq(1) & kind.eq(1)][0]
    household.loc[
        housing_unit_row,
        support_source_id_column("household"),
    ] = household.loc[gq_row, support_source_id_column("household")]
    household.loc[housing_unit_row, "TYPEHUGQ"] = 2
    household.loc[housing_unit_row, "tenure_type"] = np.nan
    tables = {entity: attached.table(entity) for entity in attached.entities}
    tables["household"] = household
    corrupted = Frame(
        tables,
        attached.schema,
        {entity: attached.weights_for(entity) for entity in attached.weighted_entities},
        attached.strata,
        mass_log=attached.mass_log,
        metadata=attached.metadata,
    )

    with pytest.raises(
        ValueError,
        match=("full-clone identity failed|support/raw household lineage pairs differ"),
    ):
        validate_stacked_spine_frame(
            corrupted,
            boundary="substituted ACS GQ clone fixture",
        )


def test_acs_housing_unit_cannot_substitute_group_quarters_raw_identity() -> None:
    filled = _gap_fill_with_test_authority(
        _stacked_gap_fixture(),
        plan=_GAP_FILL_TEST_PLAN,
        seed=578,
        n_estimators=10,
    ).frame
    household = filled.table("household").copy()
    person = filled.table("person").copy()
    channel = household[support_channel_column("household")].astype(str)
    clone = household[support_clone_index_column("household")]
    kind = pd.to_numeric(household["TYPEHUGQ"], errors="coerce")
    acs_native = channel.eq("acs") & clone.eq(0)
    person_counts = person["person_household_id"].value_counts()
    one_person_households = household["household_id"].map(person_counts).eq(1)
    housing_unit_row = household.index[acs_native & kind.eq(1) & one_person_households][
        0
    ]
    gq_row = household.index[acs_native & kind.isin((2, 3))][0]
    housing_unit_id = household.loc[housing_unit_row, "household_id"]
    gq_id = household.loc[gq_row, "household_id"]
    housing_unit_person = person.index[
        person["person_household_id"].eq(housing_unit_id)
    ][0]
    gq_person = person.index[person["person_household_id"].eq(gq_id)][0]

    for column in ("TYPEHUGQ", "tenure_type", spine_source_id_column("household")):
        housing_unit_value = household.loc[housing_unit_row, column]
        household.loc[housing_unit_row, column] = household.loc[gq_row, column]
        household.loc[gq_row, column] = housing_unit_value
    housing_unit_person_source = person.loc[
        housing_unit_person, spine_source_id_column("person")
    ]
    person.loc[housing_unit_person, spine_source_id_column("person")] = person.loc[
        gq_person, spine_source_id_column("person")
    ]
    person.loc[gq_person, spine_source_id_column("person")] = housing_unit_person_source
    housing_unit_rent = person.loc[housing_unit_person, "pre_subsidy_rent"]
    person.loc[housing_unit_person, "pre_subsidy_rent"] = np.nan
    person.loc[gq_person, "pre_subsidy_rent"] = housing_unit_rent

    tables = {entity: filled.table(entity) for entity in filled.entities}
    tables["household"] = household
    tables["person"] = person
    corrupted = Frame(
        tables,
        filled.schema,
        {entity: filled.weights_for(entity) for entity in filled.weighted_entities},
        filled.strata,
        mass_log=filled.mass_log,
        metadata=filled.metadata,
    )

    with pytest.raises(
        ValueError,
        match="support/raw/classification mapping differs",
    ):
        validate_stacked_spine_frame(
            corrupted,
            boundary="substituted native ACS GQ identity fixture",
        )


def test_acs_clone_person_cannot_substitute_group_quarters_parent() -> None:
    filled = _gap_fill_with_test_authority(
        _stacked_gap_fixture(),
        plan=_GAP_FILL_TEST_PLAN,
        seed=578,
        n_estimators=10,
    ).frame
    attached = clone_us_frame_for_puf_support(
        filled,
        clone_attachment_fraction=1.0,
        clone_attachment_seed=578,
    )
    household = attached.table("household")
    person = attached.table("person").copy()
    household_channel = household[support_channel_column("household")].astype(str)
    household_clone = household[support_clone_index_column("household")]
    kind = pd.to_numeric(household["TYPEHUGQ"], errors="coerce")
    clone_one = household_channel.eq("acs") & household_clone.eq(1)
    person_counts = person["person_household_id"].value_counts()
    one_person_households = household["household_id"].map(person_counts).eq(1)
    housing_unit_id = household.loc[
        clone_one & kind.eq(1) & one_person_households,
        "household_id",
    ].iloc[0]
    gq_id = household.loc[clone_one & kind.isin((2, 3)), "household_id"].iloc[0]
    housing_unit_person = person.index[
        person["person_household_id"].eq(housing_unit_id)
    ][0]
    gq_person = person.index[person["person_household_id"].eq(gq_id)][0]

    person.loc[housing_unit_person, "person_household_id"] = gq_id
    person.loc[gq_person, "person_household_id"] = housing_unit_id
    housing_unit_rent = person.loc[housing_unit_person, "pre_subsidy_rent"]
    person.loc[housing_unit_person, "pre_subsidy_rent"] = np.nan
    person.loc[gq_person, "pre_subsidy_rent"] = housing_unit_rent
    tables = {entity: attached.table(entity) for entity in attached.entities}
    tables["person"] = person
    corrupted = Frame(
        tables,
        attached.schema,
        {entity: attached.weights_for(entity) for entity in attached.weighted_entities},
        attached.strata,
        mass_log=attached.mass_log,
        metadata=attached.metadata,
    )

    with pytest.raises(
        ValueError,
        match="person support/raw/parent/classification lineages differ",
    ):
        validate_stacked_spine_frame(
            corrupted,
            boundary="substituted clone ACS GQ parent fixture",
        )


def test_gap_fill_plan_covers_declared_families_exactly() -> None:
    plan = stacked_gap_fill_plan()
    assert [direction.name for direction in plan] == [
        "asec_survey_to_acs",
        "asec_housing_to_acs",
    ]
    survey, housing = plan
    assert survey.recipient_channel == "acs"
    assert survey.donor_channel == "asec"
    assert housing.recipient_channel == "acs"
    assert housing.donor_channel == "asec"
    assert set(housing.target_families) == {"person"}
    assert housing.target_families["person"] == {"housing": ("pre_subsidy_rent",)}

    declared = stacked_spine_module.CANONICAL_STACKED_GAP_FILL_SURFACE
    recombined: dict[str, dict[str, tuple[str, ...]]] = {}
    for direction in plan:
        for entity, families in direction.target_families.items():
            for family, targets in families.items():
                recombined.setdefault(entity, {})[family] = tuple(targets)
    assert recombined == {
        entity: {family: tuple(targets) for family, targets in families.items()}
        for entity, families in declared.items()
    }
    early_keys = {
        (entity, family, target)
        for entity, families in recombined.items()
        for family, targets in families.items()
        for target in targets
    }
    late_keys = {
        (entity, family, target)
        for entity, families in (
            stacked_spine_module.CANONICAL_STACKED_POST_PUF_TRANSFER_SURFACE.items()
        )
        for family, targets in families.items()
        for target in targets
    }
    full_keys = {
        (entity, family, target)
        for entity, families in pool_transfer_target_families().items()
        for family, targets in families.items()
        for target in targets
    }
    assert early_keys.isdisjoint(late_keys)
    assert early_keys | late_keys == full_keys


def test_gap_fill_validator_accepts_canonical_calibration_evidence() -> None:
    stacked_spine_module.validate_stacked_gap_fill_receipt(
        _canonical_gap_fill_calibration_receipt(),
        boundary="canonical early calibration evidence control",
    )


def test_gap_fill_qrf_binding_excludes_unassigned_batched_targets() -> None:
    receipt = _canonical_gap_fill_calibration_receipt()
    direction = next(
        item
        for item in stacked_spine_module.CANONICAL_STACKED_GAP_FILL_PLAN
        if item.name == "asec_survey_to_acs"
    )
    family = "puf_tax_itemization"
    targets = direction.target_families["person"][family]
    target = "taxable_interest_income"
    key = f"person/{family}/{target}"
    target_receipt = receipt["directions"][direction.name]["targets"][key]
    legacy_counts = {
        "authorized_null_rows": 1,
        "imputed_rows": 1,
        "unmodeled_rows": 0,
        "residual_null_rows": 0,
    }
    target_receipt.update(legacy_counts)

    batch_targets = targets[
        : acs_transfer_module.DEFAULT_ACS_TRANSFER_MAX_TARGETS_PER_FIT
    ]
    required_predictors, _optional_predictors = (
        stacked_spine_module._acs_pattern_predictor_authority(
            entity="person",
            family_targets=batch_targets,
        )
    )
    record = AcsImputedInput(
        column=target,
        entity="person",
        family=f"{family}__batch_1",
        donor_spine="synthetic_batched_gap_validator_fixture",
        donor_channel=None,
        predictors=required_predictors,
        seed=0,
        weight_kind="design",
        patterns=(
            AcsTransferPattern(
                name=acs_transfer_module._pattern_name(0, ()),
                observed_optional_predictors=(),
                predictors=required_predictors,
                seed=0,
                weight_kind="design",
                donor_rows=1,
                recipient_rows=1,
                target_regimes=tuple(
                    (model_target, "positive_only")
                    for model_target in acs_transfer_module._model_target_names(
                        batch_targets
                    )
                ),
            ),
        ),
        imputed_recipient_rows=1,
    )
    target_receipt["qrf_pattern_evidence"] = (
        stacked_spine_module._acs_imputed_pattern_evidence(record)
    )

    with pytest.raises(
        ValueError,
        match="undeclared ACS QRF pattern evidence.*taxable_interest_income",
    ):
        stacked_spine_module.validate_stacked_gap_fill_receipt(
            receipt,
            boundary="unassigned batched QRF evidence",
        )

    target_receipt.pop("qrf_pattern_evidence")
    stacked_spine_module.validate_stacked_gap_fill_receipt(
        receipt,
        boundary="unassigned legacy target receipt",
    )
    assert target_receipt == legacy_counts


def test_gap_fill_validator_rejects_unassigned_legacy_count_tampering() -> None:
    receipt = _canonical_gap_fill_calibration_receipt()
    target_receipt = receipt["directions"]["asec_survey_to_acs"]["targets"][
        "person/puf_tax_itemization/taxable_interest_income"
    ]
    target_receipt.update(
        {
            "authorized_null_rows": 0,
            "imputed_rows": 1,
            "unmodeled_rows": 0,
            "residual_null_rows": 99,
        }
    )

    with pytest.raises(ValueError, match="ACS transfer row-count"):
        stacked_spine_module.validate_stacked_gap_fill_receipt(
            receipt,
            boundary="forged unassigned early transfer counts",
        )


def test_gap_fill_validator_rejects_unassigned_legacy_count_stripping() -> None:
    receipt = _canonical_gap_fill_calibration_receipt()
    target_receipt = receipt["directions"]["asec_survey_to_acs"]["targets"][
        "person/puf_tax_itemization/taxable_interest_income"
    ]
    for field in stacked_spine_module._ACS_TRANSFER_ROW_COUNT_FIELDS:
        target_receipt.pop(field)

    with pytest.raises(ValueError, match="ACS transfer row-count schema is invalid"):
        stacked_spine_module.validate_stacked_gap_fill_receipt(
            receipt,
            boundary="stripped unassigned early transfer counts",
        )


def test_gap_fill_validator_rejects_qrf_regime_evidence_tampering() -> None:
    receipt, direction_name, key, _entity, _family, _targets = (
        _canonical_gap_fill_receipt_with_pattern_evidence()
    )
    stacked_spine_module.validate_stacked_gap_fill_receipt(
        receipt,
        boundary="signed early QRF pattern evidence control",
    )

    forged = deepcopy(receipt)
    forged["directions"][direction_name]["targets"][key]["qrf_pattern_evidence"][
        "patterns"
    ][0]["target_regimes"][0]["regime"] = "negative_only"
    with pytest.raises(ValueError, match="QRF pattern evidence SHA-256 mismatch"):
        stacked_spine_module.validate_stacked_gap_fill_receipt(
            forged,
            boundary="tampered signed early QRF pattern evidence",
        )


@pytest.mark.parametrize(
    ("mutation", "error_match"),
    (
        ("pattern_count", "evidence header is invalid"),
        ("pattern_order", "name is not derived"),
        ("recipient_rows", "recipient-row accounting is invalid"),
        ("donor_rows", "metadata is invalid"),
        ("weight_kind", "record binding is invalid"),
        ("predictors", "outside canonical transfer authority"),
        ("pattern_name", "name is not derived"),
        ("model_target", "target order"),
        ("record_family", "record binding is invalid"),
        ("record_family_in_range", "record binding is invalid"),
        ("record_family_out_of_range", "record binding is invalid"),
        ("record_target", "record binding is invalid"),
    ),
)
def test_gap_fill_validator_rejects_rehashed_qrf_pattern_structure_mutations(
    mutation: str,
    error_match: str,
) -> None:
    receipt, direction_name, key, _entity, family, _targets = (
        _canonical_gap_fill_receipt_with_pattern_evidence()
    )
    evidence = receipt["directions"][direction_name]["targets"][key][
        "qrf_pattern_evidence"
    ]
    patterns = evidence["patterns"]
    if mutation == "pattern_count":
        evidence["pattern_count"] += 1
    elif mutation == "pattern_order":
        patterns.reverse()
    elif mutation == "recipient_rows":
        patterns[0]["recipient_rows"] += 1
    elif mutation == "donor_rows":
        patterns[0]["donor_rows"] = 0
    elif mutation == "weight_kind":
        evidence["record"]["weight_kind"] = "fabricated"
        for pattern in patterns:
            pattern["weight_kind"] = "fabricated"
    elif mutation == "predictors":
        patterns[0]["predictors"].append("fabricated_predictor")
    elif mutation == "pattern_name":
        patterns[0]["name"] = "pattern_00_00000000"
    elif mutation == "model_target":
        patterns[0]["target_regimes"][0]["model_target"] = "fabricated_target"
    elif mutation == "record_family":
        evidence["record"]["family"] = f"{family}__batch_forged"
    elif mutation == "record_family_in_range":
        evidence["record"]["family"] = f"{family}__batch_1"
    elif mutation == "record_family_out_of_range":
        evidence["record"]["family"] = f"{family}__batch_99"
    else:
        assert mutation == "record_target"
        evidence["record"]["column"] = "fabricated_target"
    unsigned = dict(evidence)
    unsigned.pop("sha256")
    evidence["sha256"] = stacked_spine_module._canonical_sha256(unsigned)

    with pytest.raises(ValueError, match=error_match):
        stacked_spine_module.validate_stacked_gap_fill_receipt(
            receipt,
            boundary=f"rehashed {mutation} QRF evidence",
        )


def test_receipt_only_qrf_validation_does_not_claim_seed_or_regime_replay() -> None:
    receipt, direction_name, key, _entity, _family, _targets = (
        _canonical_gap_fill_receipt_with_pattern_evidence()
    )
    evidence = receipt["directions"][direction_name]["targets"][key][
        "qrf_pattern_evidence"
    ]
    evidence["record"]["seed"] = 123
    for pattern in evidence["patterns"]:
        pattern["seed"] += 123
        pattern["target_regimes"][0]["regime"] = "negative_only"
    unsigned = dict(evidence)
    unsigned.pop("sha256")
    evidence["sha256"] = stacked_spine_module._canonical_sha256(unsigned)

    # Donor values and the top-level transfer seed are deliberately absent at
    # this boundary. The receipt-only validator checks placement/vocabulary;
    # the enclosing persisted manifest or late execution signature authenticates
    # the reported values, as the late-signature mutation test below proves.
    stacked_spine_module.validate_stacked_gap_fill_receipt(
        receipt,
        boundary="receipt-only reported seed and regime scope",
    )


@pytest.mark.parametrize(
    ("mutation", "error_match"),
    (
        (
            "stripped_direction_summary",
            "stripped or misbound calibration summary evidence",
        ),
        ("stripped_target_evidence", "owner selection is misbound"),
        ("deleted_target_receipt", "target surface is non-canonical"),
    ),
)
def test_gap_fill_validator_rejects_stripped_calibration_evidence(
    mutation: str,
    error_match: str,
) -> None:
    receipt = _canonical_gap_fill_calibration_receipt()
    forged = deepcopy(receipt)
    direction = next(
        value
        for value in forged["directions"].values()
        if value["post_transfer_calibration"]["target_count"] > 0
    )
    if mutation == "stripped_direction_summary":
        direction.pop("post_transfer_calibration")
    else:
        target_key = direction["post_transfer_calibration"]["targets"][0]
        if mutation == "deleted_target_receipt":
            direction["targets"].pop(target_key)
        else:
            direction["targets"][target_key].pop("post_transfer_calibration")

    with pytest.raises(ValueError, match=error_match):
        stacked_spine_module.validate_stacked_gap_fill_receipt(
            forged,
            boundary=f"{mutation} regression",
        )


def test_every_declared_direction_producer_precedes_its_activation_check() -> None:
    receipt = stacked_gap_fill_producer_schedule_receipt()
    assert receipt["status"] == "all_producers_precede_activation"
    assert receipt["direction_count"] == 2
    assert receipt["target_count"] == 48
    assert [direction["target_count"] for direction in receipt["directions"]] == [
        47,
        1,
    ]

    schedule = stacked_spine_module._build_gap_fill_producer_schedule(
        stacked_spine_module.CANONICAL_STACKED_GAP_FILL_SURFACE
    )
    rent_record = next(
        record for record in schedule if record.target == "pre_subsidy_rent"
    )
    late_schedule = tuple(
        replace(
            record,
            producer_stage=stacked_spine_module._POST_GAP_FILL_STAGE,
        )
        if record == rent_record
        else record
        for record in schedule
    )
    failures = stacked_spine_module._gap_fill_producer_precedence_failures(
        stacked_gap_fill_plan(),
        late_schedule,
    )
    assert len(failures) == 1
    assert "asec_housing_to_acs/person/housing/pre_subsidy_rent" in failures[0]
    assert "does not precede activation stage" in failures[0]


def test_gap_fill_producer_guard_rejects_unknown_execution_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schedule = stacked_spine_module._build_gap_fill_producer_schedule(
        stacked_spine_module.CANONICAL_STACKED_GAP_FILL_SURFACE
    )
    rent_record = next(
        record for record in schedule if record.target == "pre_subsidy_rent"
    )
    contract = stacked_spine_module.POOL_OPERATOR_CONTRACTS[rent_record.operator]
    monkeypatch.setattr(
        stacked_spine_module,
        "POOL_OPERATOR_CONTRACTS",
        {
            **stacked_spine_module.POOL_OPERATOR_CONTRACTS,
            rent_record.operator: replace(
                contract,
                execution_scope="bogus_source",
            ),
        },
    )

    mutated = stacked_spine_module._build_gap_fill_producer_schedule(
        stacked_spine_module.CANONICAL_STACKED_GAP_FILL_SURFACE
    )
    failures = stacked_spine_module._gap_fill_producer_precedence_failures(
        stacked_gap_fill_plan(),
        mutated,
    )

    assert len(failures) == 3
    assert all(
        "unknown execution scope 'bogus_source'" in failure for failure in failures
    )
    assert any(
        "asec_housing_to_acs/person/housing/pre_subsidy_rent" in failure
        for failure in failures
    )


def test_housing_activation_requires_asec_rent_producer_to_run_first() -> None:
    produced = _stacked_gap_fixture()
    tables = {entity: produced.table(entity) for entity in produced.entities}
    tables["person"] = tables["person"].drop(columns=["pre_subsidy_rent"])
    unproduced = Frame(
        tables,
        produced.schema,
        {entity: produced.weights_for(entity) for entity in produced.weighted_entities},
        produced.strata,
        mass_log=produced.mass_log,
        metadata=produced.metadata,
    )
    housing = _GAP_FILL_TEST_PLAN[1]

    with pytest.raises(
        ValueError,
        match=(
            r"asec_housing_to_acs/person/housing/pre_subsidy_rent: "
            r"declared gap-fill target column is absent"
        ),
    ):
        stacked_spine_module._verify_gap_fill_activation_authority(
            unproduced,
            direction=housing,
        )

    assert stacked_spine_module._verify_gap_fill_activation_authority(
        produced,
        direction=housing,
    ) == {
        ("person", "pre_subsidy_rent"): {
            "authorized_null_rows": 11,
            "recipient_rows": 11,
            "donor_rows": 6,
        }
    }


def test_gap_fill_fills_both_directions_with_authority_receipts() -> None:
    stacked = _stacked_gap_fixture()
    result = _gap_fill_with_test_authority(
        stacked,
        plan=_GAP_FILL_TEST_PLAN,
        seed=578,
        n_estimators=10,
    )

    person = result.frame.table("person")
    channel = person[support_channel_column("person")]
    acs_rows = channel.eq("acs")
    asec_rows = channel.eq("asec")
    gq_household_ids = result.frame.table("household").loc[
        pd.to_numeric(
            result.frame.table("household")["TYPEHUGQ"], errors="coerce"
        ).isin((2, 3)),
        "household_id",
    ]
    acs_gq_rows = acs_rows & person["person_household_id"].isin(gq_household_ids)
    for column in ("unemployment_compensation", "is_disabled"):
        assert person.loc[acs_rows, column].notna().all()
    assert person.loc[acs_rows & ~acs_gq_rows, "pre_subsidy_rent"].notna().all()
    assert person.loc[acs_gq_rows, "pre_subsidy_rent"].isna().all()

    before_person = stacked.table("person")
    for column in ("unemployment_compensation", "is_disabled"):
        pd.testing.assert_series_equal(
            person.loc[asec_rows, column],
            before_person.loc[asec_rows.to_numpy(), column],
            check_names=False,
        )
    pd.testing.assert_series_equal(
        person.loc[asec_rows, "pre_subsidy_rent"],
        before_person.loc[asec_rows.to_numpy(), "pre_subsidy_rent"],
        check_names=False,
    )

    directions = result.receipt["directions"]
    survey = directions["asec_survey_to_acs"]
    assert survey["donor_selection"] == "owner_projection_of_native_donor_rows"
    assert survey["resolved_donor_channel"] is None
    unemployment = survey["targets"][
        "person/model_required_numeric/unemployment_compensation"
    ]
    assert unemployment["authorized_null_rows"] == int(acs_rows.sum())
    assert unemployment["imputed_rows"] == int(acs_rows.sum())
    assert unemployment["residual_null_rows"] == 0
    qrf_evidence = unemployment["qrf_pattern_evidence"]
    assert qrf_evidence["pattern_count"] == len(qrf_evidence["patterns"])
    assert [pattern["name"] for pattern in qrf_evidence["patterns"]] == [
        f"pattern_{index:02d}_{pattern['name'].rsplit('_', 1)[1]}"
        for index, pattern in enumerate(qrf_evidence["patterns"])
    ]
    assert all(
        pattern["target_regimes"]
        == [
            {
                "model_target": "unemployment_compensation",
                "regime": "zero_inflated_positive",
            }
        ]
        for pattern in qrf_evidence["patterns"]
    )
    qrf_payload = dict(qrf_evidence)
    assert qrf_payload.pop("sha256") == stacked_spine_module._canonical_sha256(
        qrf_payload
    )
    forged_unemployment = deepcopy(unemployment)
    forged_unemployment["qrf_pattern_evidence"]["patterns"][0]["target_regimes"][0][
        "regime"
    ] = "negative_only"
    with pytest.raises(ValueError, match="QRF pattern evidence SHA-256 mismatch"):
        stacked_spine_module._validate_acs_imputed_pattern_evidence(
            forged_unemployment,
            expected_entity="person",
            expected_family="model_required_numeric",
            expected_target="unemployment_compensation",
            expected_family_targets=("unemployment_compensation",),
            boundary="tampered ordinary early transfer receipt",
        )
    housing = directions["asec_housing_to_acs"]
    rent = housing["targets"]["person/housing/pre_subsidy_rent"]
    assert rent["imputed_rows"] == int((acs_rows & ~acs_gq_rows).sum())
    assert rent["residual_null_rows"] == int(acs_gq_rows.sum()) == 1
    assert rent["unmodeled_rows"] == 1
    absence = rent["recipient_absence_authority"]
    assert absence["rule_id"] == "acs_native_group_quarters_without_housing_unit"
    assert absence["status"] == "exact_structural_absence"
    assert absence["rows"] == 1
    assert absence["unexpected_null_rows"] == 0
    assert absence["structural_rows_filled"] == 0

    survey_transfer = result.transfer_results["asec_survey_to_acs"]
    native_predictor_used = any(
        "__acs_transfer_interest_dividend_rental_income"
        in pattern.observed_optional_predictors
        for record in survey_transfer.imputed_inputs
        for pattern in record.patterns
    )
    assert native_predictor_used


def test_gap_fill_outcome_rejects_forged_residual_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transfer cannot receipt unmodeled rows after filling every hole."""

    transfer = stacked_spine_module.transfer_acs_inputs

    def forge_unmodeled_rows(*args: object, **kwargs: object) -> object:
        result = transfer(*args, **kwargs)
        return replace(
            result,
            imputed_inputs=tuple(
                replace(record, unmodeled_recipient_rows=1)
                if record.column == "unemployment_compensation"
                else record
                for record in result.imputed_inputs
            ),
        )

    monkeypatch.setattr(
        stacked_spine_module,
        "transfer_acs_inputs",
        forge_unmodeled_rows,
    )
    with pytest.raises(
        ValueError,
        match=(
            "residual-null equation failed: residual_null_rows=0 != unmodeled_rows=1"
        ),
    ) as error:
        _gap_fill_with_test_authority(
            _stacked_gap_fixture(),
            plan=_GAP_FILL_TEST_PLAN,
            seed=578,
            n_estimators=10,
        )
    assert (
        "activation accounting equation failed: authorized_null_rows=11 "
        "!= imputed_rows=11 + unmodeled_rows=1" in str(error.value)
    )


def test_gap_fill_rejects_honestly_accounted_undeclared_residual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accounting alone cannot authorize a residual that starves later gates."""

    transfer = stacked_spine_module.transfer_acs_inputs

    def leave_one_unmodeled(*args: object, **kwargs: object) -> object:
        result = transfer(*args, **kwargs)
        person = result.frame.table("person")
        recipient = person[support_channel_column("person")].eq("acs")
        row = person.index[recipient][0]
        person.loc[row, "unemployment_compensation"] = np.nan
        return replace(
            result,
            imputed_inputs=tuple(
                replace(
                    record,
                    imputed_recipient_rows=record.imputed_recipient_rows - 1,
                    unmodeled_recipient_rows=1,
                )
                if record.column == "unemployment_compensation"
                else record
                for record in result.imputed_inputs
            ),
        )

    monkeypatch.setattr(
        stacked_spine_module,
        "transfer_acs_inputs",
        leave_one_unmodeled,
    )

    with pytest.raises(
        ValueError,
        match=(
            "undeclared gap-fill residual is forbidden; "
            "unmodeled_rows=1, residual_null_rows=1"
        ),
    ):
        _gap_fill_with_test_authority(
            _stacked_gap_fixture(),
            plan=_GAP_FILL_TEST_PLAN,
            seed=578,
            n_estimators=10,
        )


def test_gap_fill_rejects_signed_zero_donor_byte_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Donor identity distinguishes equal-valued IEEE-754 signed zeros."""

    stacked = _stacked_gap_fixture()
    person = stacked.table("person")
    channel_column = support_channel_column("person")
    donor_rows = person[channel_column].astype(str).eq("asec")
    donor_index = person.index[
        donor_rows & person["unemployment_compensation"].eq(0.0)
    ][0]
    person.loc[donor_index, "unemployment_compensation"] = -0.0
    assert np.signbit(person.loc[donor_index, "unemployment_compensation"])

    transfer = stacked_spine_module.transfer_acs_inputs

    def flip_donor_signed_zero(*args: object, **kwargs: object) -> object:
        result = transfer(*args, **kwargs)
        result_person = result.frame.table("person").copy(deep=True)
        assert np.signbit(result_person.loc[donor_index, "unemployment_compensation"])
        result_person.loc[donor_index, "unemployment_compensation"] = 0.0
        tables = {
            entity: result.frame.table(entity) for entity in result.frame.entities
        }
        tables["person"] = result_person
        changed = Frame(
            tables,
            result.frame.schema,
            {
                entity: result.frame.weights_for(entity)
                for entity in result.frame.weighted_entities
            },
            result.frame.strata,
            mass_log=result.frame.mass_log,
            metadata=result.frame.metadata,
        )
        return replace(result, frame=changed)

    monkeypatch.setattr(
        stacked_spine_module,
        "transfer_acs_inputs",
        flip_donor_signed_zero,
    )
    with pytest.raises(
        ValueError,
        match=(
            r"asec_survey_to_acs/person/model_required_numeric/"
            r"unemployment_compensation: donor byte identity failed.*"
            r"canonical donor payload changed"
        ),
    ):
        _gap_fill_with_test_authority(
            stacked,
            plan=(_GAP_FILL_TEST_PLAN[0],),
            seed=578,
            n_estimators=10,
        )


def test_donor_byte_identity_canonicalizes_semantic_strings() -> None:
    index = pd.Index([7, 3], name="donor_row")
    object_strings = pd.Series(
        ["RENTED", None],
        index=index,
        name="tenure_type",
        dtype=object,
    )
    canonical_strings = object_strings.astype(CANONICAL_STRING_DTYPE)

    assert stacked_spine_module._canonical_donor_series_payload(
        object_strings,
        boundary="object-string donor identity",
    ) == stacked_spine_module._canonical_donor_series_payload(
        canonical_strings,
        boundary="canonical-string donor identity",
    )

    unchanged_controls = (
        pd.Series([True, False], name="native_bool", dtype=bool),
        pd.Series([True, pd.NA], name="nullable_bool", dtype="boolean"),
        pd.Series(
            pd.Categorical(["a", None], categories=["a", "b"]),
            name="native_category",
        ),
        pd.Series([None, np.nan], name="all_null_object", dtype=object),
    )
    for control in unchanged_controls:
        assert stacked_spine_module._canonical_donor_series_payload(
            control,
            boundary=f"{control.name} donor identity before",
        ) == stacked_spine_module._canonical_donor_series_payload(
            control.copy(deep=True),
            boundary=f"{control.name} donor identity after",
        )

    mixed = pd.Series(
        ["RENTED", 1],
        name="mixed_tenure_type",
        dtype=object,
    )
    with pytest.raises(TypeError, match="semantic strings cannot mix"):
        stacked_spine_module._canonical_donor_series_payload(
            mixed,
            boundary="mixed-object donor identity",
        )


def test_donor_byte_identity_ignores_string_object_alias_topology() -> None:
    shared = "".join(["dynamically", "-", "allocated", "-", "donor"])
    separate = [
        "".join(["dynamically", "-", "allocated", "-", "donor"]) for _ in range(2)
    ]
    before = pd.Series([shared, shared], name="native_label", dtype=object)
    after = pd.Series(separate, name="native_label", dtype=object)
    assert before.iloc[0] is before.iloc[1]
    assert after.iloc[0] is not after.iloc[1]

    assert stacked_spine_module._canonical_donor_series_payload(
        before,
        boundary="aliased-string donor identity before",
    ) == stacked_spine_module._canonical_donor_series_payload(
        after,
        boundary="aliased-string donor identity after",
    )


def test_donor_byte_identity_accepts_semantic_boolean_object_scalars() -> None:
    semantic_booleans = pd.Series(
        [True, np.bool_(False), None],
        name="is_disabled",
        dtype=object,
    )

    assert stacked_spine_module._canonical_donor_series_payload(
        semantic_booleans,
        boundary="semantic-boolean donor identity before",
    ) == stacked_spine_module._canonical_donor_series_payload(
        semantic_booleans.copy(deep=True),
        boundary="semantic-boolean donor identity after",
    )


def test_gap_fill_activation_authority_fails_closed_on_donor_nulls() -> None:
    stacked = _stacked_gap_fixture()
    person = stacked.table("person").copy()
    channel = person[support_channel_column("person")]
    poke = person.index[channel.eq("asec")][1]
    person.loc[poke, "unemployment_compensation"] = np.nan
    tables = {entity: stacked.table(entity) for entity in stacked.entities}
    tables["person"] = person
    poked = Frame(
        tables,
        stacked.schema,
        {entity: stacked.weights_for(entity) for entity in stacked.weighted_entities},
        stacked.strata,
        mass_log=stacked.mass_log,
        metadata=stacked.metadata,
    )

    with pytest.raises(ValueError, match="donors must observe"):
        _gap_fill_with_test_authority(poked, plan=_GAP_FILL_TEST_PLAN, seed=578)


@pytest.mark.parametrize(
    ("recipient_channel", "donor_channel", "role", "missing_channel"),
    (
        ("acx_typo", "asec", "recipient", "acx_typo"),
        ("acs", "asec_typo", "donor", "asec_typo"),
    ),
)
def test_gap_fill_activation_authority_rejects_nonlive_declared_channel(
    recipient_channel: str,
    donor_channel: str,
    role: str,
    missing_channel: str,
) -> None:
    plan = (
        GapFillDirection(
            name="nonlive_channel",
            recipient_channel=recipient_channel,
            donor_channel=donor_channel,
            target_families={"person": {"complete": ("age",)}},
        ),
    )

    with pytest.raises(
        ValueError,
        match=f"declared {role} channel {missing_channel!r} has no live rows",
    ):
        _gap_fill_with_test_authority(_stacked_gap_fixture(), plan=plan, seed=578)


def test_gap_fill_fails_closed_on_missing_target_column() -> None:
    stacked = _stacked_gap_fixture()
    plan = (
        GapFillDirection(
            name="asec_survey_to_acs",
            recipient_channel="acs",
            donor_channel="asec",
            target_families={
                "person": {"model_required_numeric": ("veterans_benefits",)}
            },
        ),
    )
    with pytest.raises(ValueError, match="veterans_benefits.*absent"):
        _gap_fill_with_test_authority(stacked, plan=plan, seed=578)


def test_gap_fill_rejects_cloned_frames() -> None:
    cloned = clone_us_frame_for_puf_support(_stacked_gap_fixture())
    with pytest.raises(ValueError, match="before clone operators"):
        _gap_fill_with_test_authority(cloned, plan=_GAP_FILL_TEST_PLAN, seed=578)


def test_bounded_transfer_group_remaps_canonical_producer_roles() -> None:
    group = next(
        group
        for group in stacked_spine_module.CANONICAL_US_LATE_TRANSFER_GROUPS
        if group.family == "puf_tax_itemization__batch_2"
    )

    puf_roles = stacked_spine_module._producer_role_surface_for_group(
        group.target_families,
        stacked_spine_module.CANONICAL_STACKED_POST_PUF_PUF_PRODUCER_SURFACE,
    )
    source_roles = stacked_spine_module._producer_role_surface_for_group(
        group.target_families,
        stacked_spine_module.CANONICAL_STACKED_POST_PUF_SOURCE_PRODUCER_SURFACE,
    )

    assert "qualified_tuition_expenses" in puf_roles["person"][group.family]
    assert "traditional_ira_contributions_desired" in puf_roles["person"][group.family]
    assert source_roles == {
        "person": {
            group.family: ("traditional_ira_contributions_desired",),
        }
    }


def test_late_readiness_rejects_object_typed_nonfinite_numeric_input() -> None:
    frame = _post_puf_transfer_fixture()
    person = frame.table("person").copy()
    person["late_numeric"] = pd.Series(
        np.resize(np.asarray([np.inf, "bad"], dtype=object), len(person)),
        index=person.index,
        dtype=object,
    )
    tables = {entity: frame.table(entity) for entity in frame.entities}
    tables["person"] = person
    poisoned = Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )
    requirement = ProducerInput(
        "person",
        "late_numeric",
        "asec_source",
        "transfer:fixture",
        alternatives=(
            (
                ProducerInputColumn(
                    "person",
                    "late_numeric",
                    "finite_numeric",
                ),
            ),
        ),
    )
    contract = ProducerContract(
        "source:fixture",
        "post_clone_source",
        (requirement,),
        (),
    )

    unfilled, invalid = stacked_spine_module._late_input_readiness_rows(
        poisoned,
        contract,
    )

    assert unfilled[requirement] == 0
    assert invalid[requirement] == int(
        person[support_channel_column("person")].astype(str).eq("asec").sum()
    )
    with pytest.raises(
        ValueError,
        match=r"person\.late_numeric.*transfer:fixture",
    ):
        stacked_spine_module.run_producer_when_ready(
            contract,
            lambda: pytest.fail("invalid numeric input reached callback"),
            unfilled_rows=unfilled,
            invalid_rows=invalid,
            absence_receipts={},
        )


def test_typehugq_accepts_exact_1688_asec_structural_null_rows() -> None:
    frame, contract, requirement = _typehugq_cross_origin_readiness_fixture()
    household = frame.table("household")
    support_channel = household[support_channel_column("household")].astype(str)
    asec_rows = support_channel.eq("asec")
    acs_rows = support_channel.eq("acs")

    assert int(asec_rows.sum()) == 1_688
    assert int(acs_rows.sum()) == 10
    assert int(household.loc[asec_rows, "TYPEHUGQ"].isna().sum()) == 1_688
    assert int(household.loc[acs_rows, "TYPEHUGQ"].isna().sum()) == 0
    assert requirement.required_scope == "acs_source"
    assert requirement.tolerated_absence_receipts == ()

    unfilled, invalid = stacked_spine_module._late_input_readiness_rows(
        frame,
        contract,
    )
    absence = stacked_spine_module._late_declared_absence_receipts(
        contract,
        unfilled,
        invalid_rows=invalid,
    )

    assert unfilled == {requirement: 0}
    assert invalid == {requirement: 0}
    assert absence == {}
    assert (
        stacked_spine_module.run_producer_when_ready(
            contract,
            lambda: "ran",
            unfilled_rows=unfilled,
            invalid_rows=invalid,
            absence_receipts=absence,
        )
        == "ran"
    )


def test_typehugq_still_refuses_one_missing_acs_row() -> None:
    frame, contract, requirement = _typehugq_cross_origin_readiness_fixture()
    household = frame.table("household").copy()
    support_channel = household[support_channel_column("household")].astype(str)
    acs_row = household.index[support_channel.eq("acs")][0]
    household.loc[acs_row, "TYPEHUGQ"] = np.nan
    tables = {entity: frame.table(entity) for entity in frame.entities}
    tables["household"] = household
    missing = Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )

    unfilled, invalid = stacked_spine_module._late_input_readiness_rows(
        missing,
        contract,
    )
    absence = stacked_spine_module._late_declared_absence_receipts(
        contract,
        unfilled,
        invalid_rows=invalid,
    )
    invoked = False

    def callback() -> None:
        nonlocal invoked
        invoked = True

    assert unfilled == {requirement: 1}
    assert invalid == {requirement: 0}
    assert absence == {}
    with pytest.raises(
        ValueError,
        match=(
            r"(?s)primary_puf_qrf.*"
            r"household\.@effective:validated_structure:TYPEHUGQ.*"
            r"1 unfilled.*acs_source.*post_clone_input_surface.*"
            r"tolerated absence receipts=\[\]"
        ),
    ):
        stacked_spine_module.run_producer_when_ready(
            contract,
            callback,
            unfilled_rows=unfilled,
            invalid_rows=invalid,
            absence_receipts=absence,
        )
    assert invoked is False


def test_canonical_transfer_rejects_nonfinite_optional_numeric_as_invalid() -> None:
    contract = stacked_spine_module.CANONICAL_US_LATE_PRODUCER_REGISTRY[
        "transfer:person/adult_care"
    ]
    complete = _fill_late_contract_surface(
        _post_puf_transfer_fixture(),
        contracts=(contract,),
        include_outputs=False,
    )
    person = complete.table("person").copy()
    person.loc[person.index[0], "employment_income_before_lsr"] = np.inf
    tables = {entity: complete.table(entity) for entity in complete.entities}
    tables["person"] = person
    poisoned = Frame(
        tables,
        complete.schema,
        {entity: complete.weights_for(entity) for entity in complete.weighted_entities},
        complete.strata,
        mass_log=complete.mass_log,
        metadata=complete.metadata,
    )
    requirement = next(
        item
        for item in contract.inputs
        if item.column == "@effective:optional_employment_income"
    )

    unfilled, invalid = stacked_spine_module._late_input_readiness_rows(
        poisoned,
        contract,
    )
    absence = stacked_spine_module._late_declared_absence_receipts(
        contract,
        unfilled,
        invalid_rows=invalid,
    )

    assert unfilled[requirement] == 0
    assert invalid[requirement] == 1
    assert requirement.tolerated_absence_receipts[0] not in absence
    with pytest.raises(
        ValueError,
        match=(
            r"(?s)transfer:person/adult_care.*"
            r"person\.@effective:optional_employment_income.*1 invalid.*"
            r"post_clone_input_surface"
        ),
    ):
        stacked_spine_module.run_producer_when_ready(
            contract,
            lambda: pytest.fail("invalid predictor reached transfer callback"),
            unfilled_rows=unfilled,
            invalid_rows=invalid,
            absence_receipts=absence,
        )


def test_person_transfer_refuses_invalid_peer_grain_provenance_before_fit() -> None:
    contract = stacked_spine_module.CANONICAL_US_LATE_PRODUCER_REGISTRY[
        "transfer:person/adult_care"
    ]
    complete = _fill_late_contract_surface(
        _post_puf_transfer_fixture(),
        contracts=(contract,),
        include_outputs=False,
    )
    family = complete.table("family").copy()
    family["family_support_clone_index"] = family["family_support_clone_index"].astype(
        float
    )
    family.loc[family.index[0], "family_support_clone_index"] = np.inf
    tables = {entity: complete.table(entity) for entity in complete.entities}
    tables["family"] = family
    poisoned = Frame(
        tables,
        complete.schema,
        {entity: complete.weights_for(entity) for entity in complete.weighted_entities},
        complete.strata,
        mass_log=complete.mass_log,
        metadata=complete.metadata,
    )
    direct = next(
        item
        for item in contract.inputs
        if item.entity == "family"
        and item.column == "family_support_clone_index"
        and item.producing_stage == stacked_spine_module.US_LATE_PRIMARY_PUF_STAGE
    )

    unfilled, invalid = stacked_spine_module._late_input_readiness_rows(
        poisoned,
        contract,
    )

    assert unfilled[direct] == 0
    assert invalid[direct] == 1
    with pytest.raises(
        ValueError,
        match=(
            r"(?s)transfer:person/adult_care.*"
            r"family\.family_support_clone_index.*1 invalid.*primary_puf_qrf"
        ),
    ):
        stacked_spine_module.run_producer_when_ready(
            contract,
            lambda: pytest.fail("peer-grain poison reached transfer fit"),
            unfilled_rows=unfilled,
            invalid_rows=invalid,
            absence_receipts={},
        )


@pytest.mark.parametrize(
    ("metadata_key", "producing_stage"),
    (
        ("us_spine_assembly_manifest", "post_clone_input_surface"),
        ("us_stacked_spine_manifest", "post_clone_input_surface"),
        ("us_puf_clone_attachment_manifest", "primary_puf_qrf"),
    ),
)
def test_transfer_refuses_missing_validation_metadata_before_fit(
    metadata_key: str,
    producing_stage: str,
) -> None:
    contract = stacked_spine_module.CANONICAL_US_LATE_PRODUCER_REGISTRY[
        "transfer:person/adult_care"
    ]
    complete = _fill_late_contract_surface(
        _post_puf_transfer_fixture(),
        contracts=(contract,),
        include_outputs=False,
    )
    missing = Frame(
        {entity: complete.table(entity) for entity in complete.entities},
        complete.schema,
        {entity: complete.weights_for(entity) for entity in complete.weighted_entities},
        complete.strata,
        mass_log=complete.mass_log,
        metadata={
            key: value
            for key, value in complete.metadata.items()
            if key != metadata_key
        },
    )
    requirement = next(
        item
        for item in contract.inputs
        if item.producing_stage == producing_stage
        and any(
            column.entity == "frame" and column.column == f"@{metadata_key}"
            for alternative in item.alternatives
            for column in alternative
        )
    )

    unfilled, invalid = stacked_spine_module._late_input_readiness_rows(
        missing,
        contract,
    )

    assert unfilled[requirement] == 1
    assert invalid[requirement] == 0
    with pytest.raises(
        ValueError,
        match=rf"(?s)transfer:person/adult_care.*{producing_stage}",
    ):
        stacked_spine_module.run_producer_when_ready(
            contract,
            lambda: pytest.fail("missing metadata reached transfer fit"),
            unfilled_rows=unfilled,
            invalid_rows=invalid,
            absence_receipts={},
        )


def test_late_table_content_digest_is_byte_stable_for_typed_scalar_vector() -> None:
    table = _late_table_digest_vector()

    first = stacked_spine_module._late_table_values_sha256(table)
    second = stacked_spine_module._late_table_values_sha256(table.copy(deep=True))

    assert (
        first
        == second
        == ("da35b24dd68ac7a8917e27c37c81a44d8ab4fbc4888539e29999f904ce741254")
    )
    assert len(first) == 64
    assert set(first) <= set("0123456789abcdef")


@pytest.mark.parametrize(
    ("left", "right"),
    (
        ([True], [1]),
        ([1], [1.0]),
        ([None], [""]),
        (["ab", "c"], ["a", "bc"]),
        ([0.0], [-0.0]),
    ),
)
def test_late_table_content_digest_domain_separates_object_scalars(
    left: list[object],
    right: list[object],
) -> None:
    left_table = pd.DataFrame({"value": pd.Series(left, dtype=object)})
    right_table = pd.DataFrame({"value": pd.Series(right, dtype=object)})

    assert stacked_spine_module._late_table_values_sha256(
        left_table
    ) != stacked_spine_module._late_table_values_sha256(right_table)


def test_late_table_content_digest_canonicalizes_null_float_payloads() -> None:
    ordinary_nan = np.array([np.nan], dtype=np.float64)
    alternate_nan = np.array([0x7FF8_0000_0000_0001], dtype="<u8").view("<f8")
    ordinary = pd.DataFrame({"value": ordinary_nan})
    alternate = pd.DataFrame({"value": alternate_nan})

    assert (
        ordinary["value"].to_numpy().tobytes()
        != alternate["value"].to_numpy().tobytes()
    )
    assert stacked_spine_module._late_table_values_sha256(
        ordinary
    ) == stacked_spine_module._late_table_values_sha256(alternate)


def test_late_table_content_digest_normalizes_serialized_string_dtype() -> None:
    index = pd.Index([9, 4], dtype=np.int64, name="row_id")
    object_strings = pd.DataFrame(
        {
            "label": pd.Series(
                ["RENTED", None],
                index=index,
                dtype=object,
            )
        },
        index=index,
    )
    canonical_strings = pd.DataFrame(
        {
            "label": pd.Series(
                ["RENTED", pd.NA],
                index=index,
                dtype=CANONICAL_STRING_DTYPE,
            )
        },
        index=index,
    )

    assert stacked_spine_module._late_table_values_sha256(
        object_strings,
        normalize_strings=True,
    ) == stacked_spine_module._late_table_values_sha256(
        canonical_strings,
        normalize_strings=True,
    )
    assert stacked_spine_module._late_table_values_sha256(
        object_strings,
        normalize_strings=False,
    ) != stacked_spine_module._late_table_values_sha256(
        canonical_strings,
        normalize_strings=False,
    )


def test_late_table_content_digest_binds_dtype_index_and_order() -> None:
    base = pd.DataFrame(
        {"value": np.array([1, 2], dtype=np.int32)},
        index=pd.Index([4, 8], name="row_id"),
    )
    digest = stacked_spine_module._late_table_values_sha256(base)
    variants = (
        base.astype({"value": np.int64}),
        base.iloc[::-1],
        base.rename_axis("different_index"),
        base.rename(columns={"value": "different_column"}),
    )

    assert all(
        stacked_spine_module._late_table_values_sha256(variant) != digest
        for variant in variants
    )


def test_late_primary_worker_authentication_ignores_audit_alias_relocation() -> None:
    binding = _fixture_primary_execution_config_binding()
    worker = binding["qrf"]["worker_execution"]
    original_semantic_identity = deepcopy(worker["semantic_identity"])
    original_semantic_sha256 = worker["semantic_identity_sha256"]
    relocated_interpreter = (
        "/Users/maxghenis/PolicyEngine/_worktrees/microcosm-c26-build/.venv/bin/python"
    )

    worker["audit_aliases"] = {
        "sys_executable": relocated_interpreter,
        "sys_prefix": str(Path(relocated_interpreter).parents[1]),
        "argv_template_0": relocated_interpreter,
    }

    _validate_fixture_primary_execution_config(binding)
    assert worker["semantic_identity"] == original_semantic_identity
    assert worker["semantic_identity_sha256"] == original_semantic_sha256


def test_late_primary_worker_audit_paths_must_be_absolute() -> None:
    binding = _fixture_primary_execution_config_binding()
    binding["qrf"]["worker_execution"]["audit_aliases"]["sys_executable"] = (
        "relative/python"
    )

    with pytest.raises(ValueError, match="worker binding is malformed"):
        _validate_fixture_primary_execution_config(binding)


def test_late_primary_worker_authentication_rejects_rehashed_semantic_change() -> None:
    binding = _fixture_primary_execution_config_binding()
    worker = binding["qrf"]["worker_execution"]
    semantic_identity = worker["semantic_identity"]
    semantic_identity["interpreter"]["bytes_sha256"] = "0" * 64
    worker["semantic_identity_sha256"] = stacked_spine_module._canonical_sha256(
        semantic_identity
    )

    with pytest.raises(ValueError, match="semantic worker identity changed"):
        _validate_fixture_primary_execution_config(binding)


@pytest.mark.usefixtures("live_worker_identity")
def test_worker_transitive_source_identity_includes_package_initializers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker_source = tmp_path / "worker.py"
    package_dir = tmp_path / "fixture" / "pkg"
    package_dir.mkdir(parents=True)
    package_init = package_dir / "__init__.py"
    child_source = package_dir / "child.py"
    worker_source.write_text("import fixture.pkg.child\n", encoding="utf-8")
    package_init.write_text("PACKAGE_VALUE = 1\n", encoding="utf-8")
    child_source.write_text("CHILD_VALUE = 1\n", encoding="utf-8")
    source_index = {
        worker_identity_module.PRIMARY_QRF_WORKER_MODULE: worker_source,
        "fixture.pkg": package_init,
        "fixture.pkg.child": child_source,
    }
    monkeypatch.setattr(
        worker_identity_module,
        "_module_source_index",
        lambda: source_index,
    )

    worker_before, imports_before, _ = worker_identity_module._worker_source_identity()
    package_init.write_text("PACKAGE_VALUE = 2\n", encoding="utf-8")
    worker_after, imports_after, _ = worker_identity_module._worker_source_identity()

    assert worker_after == worker_before
    assert imports_after != imports_before


@pytest.mark.usefixtures("live_worker_identity")
def test_worker_source_index_stays_inside_installed_namespace_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site_packages = tmp_path / "site-packages"
    namespace_root = site_packages / "microcosm"
    worker_source = namespace_root / "build" / "us_runtime" / "puf_qrf_worker.py"
    neighboring_package = site_packages / "packaging" / "__init__.py"
    worker_source.parent.mkdir(parents=True)
    neighboring_package.parent.mkdir(parents=True)
    worker_source.write_text("import packaging\n", encoding="utf-8")
    neighboring_package.write_text("NEIGHBOR = True\n", encoding="utf-8")
    monkeypatch.setattr(
        worker_identity_module.importlib.util,
        "find_spec",
        lambda name: (
            SimpleNamespace(submodule_search_locations=(namespace_root,))
            if name == "microcosm"
            else None
        ),
    )

    index = worker_identity_module._module_source_index()

    assert index == {
        worker_identity_module.PRIMARY_QRF_WORKER_MODULE: worker_source.resolve()
    }
    internal, external = worker_identity_module._source_imports(
        worker_identity_module.PRIMARY_QRF_WORKER_MODULE,
        worker_source,
        worker_source.read_bytes(),
        index=index,
    )
    assert internal == set()
    assert external == {"packaging"}


@pytest.mark.usefixtures("live_worker_identity")
def test_worker_source_index_accepts_a_byte_identical_shadow_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An installed wheel beside its own checkout (or lib64 beside lib) is the
    same code; the first root on the import path wins."""
    installed, checkout = _two_namespace_roots(tmp_path, monkeypatch)
    for root in (installed, checkout):
        (root / "build" / "us_runtime" / "puf_qrf_worker.py").write_text(
            "import packaging\n", encoding="utf-8"
        )
    (checkout / "build" / "__init__.py").write_text("", encoding="utf-8")

    index = worker_identity_module._module_source_index()

    assert (
        index[worker_identity_module.PRIMARY_QRF_WORKER_MODULE]
        == (installed / "build" / "us_runtime" / "puf_qrf_worker.py").resolve()
    )
    assert index["microcosm.build"] == (checkout / "build" / "__init__.py").resolve()


@pytest.mark.usefixtures("live_worker_identity")
def test_worker_source_index_rejects_a_shadow_root_with_different_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed, checkout = _two_namespace_roots(tmp_path, monkeypatch)
    (installed / "build" / "us_runtime" / "puf_qrf_worker.py").write_text(
        "import packaging\n", encoding="utf-8"
    )
    (checkout / "build" / "us_runtime" / "puf_qrf_worker.py").write_text(
        "import packaging  # edited\n", encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="Duplicate source modules .* differs"):
        worker_identity_module._module_source_index()


@pytest.mark.parametrize("components", (2, 3))
@pytest.mark.usefixtures("live_worker_identity")
def test_canonical_pyvenv_config_accepts_uv_major_minor_and_triplet_versions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    components: int,
) -> None:
    """uv 0.12 writes ``version_info = 3.13``; older uv wrote the triplet. Both
    canonicalize to the running interpreter's triplet."""
    running = worker_identity_module.sys.version_info
    _pyvenv_prefix(
        tmp_path, monkeypatch, ".".join(str(part) for part in running[:components])
    )

    config = worker_identity_module._canonical_pyvenv_config()

    assert config["version"] == list(running[:3])
    assert config["uv_version"] == "0.12.9"


@pytest.mark.parametrize(
    ("version", "message"),
    (
        ("9.9.9", "does not match the running interpreter"),
        ("9.9", "does not match the running interpreter"),
        ("3", "major.minor or major.minor.micro"),
        ("3.13.11.0", "major.minor or major.minor.micro"),
        ("3.13rc1", "is not numeric"),
    ),
)
@pytest.mark.usefixtures("live_worker_identity")
def test_canonical_pyvenv_config_rejects_versions_that_are_not_the_interpreter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    version: str,
    message: str,
) -> None:
    _pyvenv_prefix(tmp_path, monkeypatch, version)

    with pytest.raises(RuntimeError, match=message):
        worker_identity_module._canonical_pyvenv_config()


def test_primary_qrf_worker_identity_memo_reuses_graph_and_clear_recomputes(
    _worker_identity_memo_inputs,
) -> None:
    _source, calls = _worker_identity_memo_inputs
    first = worker_identity_module.primary_qrf_worker_semantic_identity()
    second = worker_identity_module.primary_qrf_worker_semantic_identity()
    assert second is first
    assert second["interpreter"] is first["interpreter"]
    assert second["environment"] is first["environment"]
    assert len(calls) == 1

    worker_identity_module.clear_primary_qrf_worker_identity_cache()
    third = worker_identity_module.primary_qrf_worker_semantic_identity()
    assert third == first
    assert third is not first
    assert third["interpreter"] is not first["interpreter"]
    assert len(calls) == 2


@pytest.mark.parametrize(
    "name", ("POPULACE_FIT_N_JOBS", "POPULACE_FIT_PREDICT_WORKERS")
)
def test_primary_qrf_worker_identity_memo_binds_environment(
    _worker_identity_memo_inputs,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    _source, calls = _worker_identity_memo_inputs
    monkeypatch.setenv(name, "1")
    first = worker_identity_module.primary_qrf_worker_semantic_identity()
    monkeypatch.setenv(name, "3")
    second = worker_identity_module.primary_qrf_worker_semantic_identity()
    assert second is not first
    assert first["environment"]["semantic_controls"][name]["resolved"] == 1
    assert second["environment"]["semantic_controls"][name]["resolved"] == 3
    assert worker_identity_module._canonical_sha256(
        first
    ) != worker_identity_module._canonical_sha256(second)
    assert len(calls) == 2
    monkeypatch.setenv(name, "1")
    assert worker_identity_module.primary_qrf_worker_semantic_identity() is first
    assert len(calls) == 2


@pytest.mark.parametrize(
    "name", ("POPULACE_FIT_N_JOBS", "POPULACE_FIT_PREDICT_WORKERS")
)
@pytest.mark.parametrize("value", ("0", "01", "invalid"))
def test_primary_qrf_worker_identity_memo_refuses_invalid_environment(
    _worker_identity_memo_inputs,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
) -> None:
    monkeypatch.setenv(name, "1")
    worker_identity_module.primary_qrf_worker_semantic_identity()
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        worker_identity_module.primary_qrf_worker_semantic_identity()


def test_primary_qrf_worker_identity_memo_binds_lock(
    _worker_identity_memo_inputs,
) -> None:
    first = worker_identity_module.primary_qrf_worker_semantic_identity()
    legacy = worker_identity_module.primary_qrf_worker_semantic_identity(
        uv_lock_sha256=worker_identity_module.LEGACY_CAMPAIGN_UV_LOCK_SHA256
    )
    assert first["uv_lock_sha256"] == worker_identity_module.APPROVED_UV_LOCK_SHA256
    assert (
        legacy["uv_lock_sha256"]
        == worker_identity_module.LEGACY_CAMPAIGN_UV_LOCK_SHA256
    )
    assert legacy != first
    assert worker_identity_module.primary_qrf_worker_semantic_identity() is first
    for invalid in ("0" * 64, "malformed", []):
        with pytest.raises(ValueError, match="lock"):
            worker_identity_module.primary_qrf_worker_semantic_identity(
                uv_lock_sha256=invalid
            )


def test_primary_qrf_worker_execution_binding_isolates_semantic_memo(
    _worker_identity_memo_inputs,
) -> None:
    memo = worker_identity_module.primary_qrf_worker_semantic_identity()
    expected = deepcopy(memo)
    binding = worker_identity_module.primary_qrf_worker_execution_binding()
    semantic = binding["semantic_identity"]
    semantic["interpreter"]["runtime_binary"]["bytes_sha256"] = "0" * 64
    semantic["argv_template"].append("--changed")
    semantic["environment"]["overrides"]["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "1"
    fresh = worker_identity_module.primary_qrf_worker_execution_binding()
    assert memo == expected
    assert fresh["semantic_identity"] == expected
    assert fresh["semantic_identity"] is not memo
    assert fresh["semantic_identity"]["interpreter"] is not memo["interpreter"]


def test_worker_identity_session_prime_matches_pipeline_binding(
    _session_primary_qrf_worker_identities,
) -> None:
    key = (
        None,
        worker_identity_module.os.environ.get("POPULACE_FIT_N_JOBS"),
        worker_identity_module.os.environ.get("POPULACE_FIT_PREDICT_WORKERS"),
    )
    memo = worker_identity_module._PRIMARY_QRF_WORKER_IDENTITY_CACHE
    assert key in memo
    first = memo[key]
    binding = _fixture_primary_execution_config_binding()["qrf"]["worker_execution"]
    assert binding["semantic_identity"] == first
    assert worker_identity_module.primary_qrf_worker_semantic_identity() is first
    assert first == _session_primary_qrf_worker_identities[key]


def test_live_worker_identity_namespace_source_mutation_recomputes(
    _session_primary_qrf_worker_identities,
    live_worker_identity,
    _worker_identity_memo_inputs,
) -> None:
    # The session fixture attests first; the opt-out must then remove it rather
    # than restoring that identity over the temporary namespace used below.
    assert _session_primary_qrf_worker_identities
    assert not worker_identity_module._PRIMARY_QRF_WORKER_IDENTITY_CACHE
    source, calls = _worker_identity_memo_inputs
    before = worker_identity_module.primary_qrf_worker_semantic_identity()
    source.write_text("# changed fixture worker\n", encoding="utf-8")
    assert worker_identity_module.primary_qrf_worker_semantic_identity() is before

    live_worker_identity()
    after = worker_identity_module.primary_qrf_worker_semantic_identity()
    assert after is not before
    assert (
        after["worker_module"]["transitive_imports_sha256"]
        != before["worker_module"]["transitive_imports_sha256"]
    )
    assert (
        after["transitive_environment_code_sha256"]
        != before["transitive_environment_code_sha256"]
    )
    assert len(calls) == 2


@pytest.mark.usefixtures("live_worker_identity")
def test_worker_import_trace_refuses_empty_result() -> None:
    with pytest.raises(RuntimeError, match="no readable namespace roots"):
        worker_identity_module._validated_worker_import_trace(
            {
                "module_origins": {},
                "opened_files": (),
                "namespace_roots": (),
            }
        )


@pytest.mark.parametrize("cache_location", ["source_tree", "inherited_prefix"])
@pytest.mark.parametrize("execution_path", ["identity_probe", "worker_module"])
@pytest.mark.usefixtures("live_worker_identity")
def test_worker_identity_ignores_valid_header_stale_bytecode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cache_location: str,
    execution_path: str,
) -> None:
    import importlib.util
    import marshal
    import os
    import struct
    import subprocess

    trace = _fixture_worker_import_trace(tmp_path)
    worker_source = Path(
        trace["module_origins"][worker_identity_module.PRIMARY_QRF_WORKER_MODULE]
    )
    (worker_source.parent.parent / "__init__.py").write_text("", encoding="utf-8")
    marker = tmp_path / "worker-executed.txt"
    worker_source.write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('source')\n",
        encoding="utf-8",
    )
    source_bytes = worker_source.read_bytes()
    source_stat = worker_source.stat()
    poisoned_code = compile(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('stale cache')\n",
        str(worker_source),
        "exec",
        optimize=sys.flags.optimize,
    )
    inherited_prefix = (
        tmp_path / "inherited-cache" if cache_location == "inherited_prefix" else None
    )
    with monkeypatch.context() as cache_path_context:
        cache_path_context.setattr(
            sys,
            "pycache_prefix",
            None if inherited_prefix is None else str(inherited_prefix),
        )
        cache_path = Path(importlib.util.cache_from_source(str(worker_source)))
    cache_path.parent.mkdir(parents=True)
    # Python accepts this timestamp cache: its source mtime and size are valid,
    # although the executable code object is unrelated to the source bytes.
    cache_bytes = (
        importlib.util.MAGIC_NUMBER
        + struct.pack(
            "<III", 0, int(source_stat.st_mtime) & 0xFFFFFFFF, len(source_bytes)
        )
        + marshal.dumps(poisoned_code)
    )
    cache_path.write_bytes(cache_bytes)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "namespace"))
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    if inherited_prefix is None:
        monkeypatch.delenv("PYTHONPYCACHEPREFIX", raising=False)
    else:
        monkeypatch.setenv("PYTHONPYCACHEPREFIX", str(inherited_prefix))

    control = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import {worker_identity_module.PRIMARY_QRF_WORKER_MODULE}",
        ],
        env=dict(os.environ),
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert control.returncode == 0, control.stderr
    assert marker.read_text(encoding="utf-8") == "stale cache"
    marker.unlink()

    if execution_path == "identity_probe":
        observed = worker_identity_module._clean_worker_import_trace()
        assert str(worker_source) in observed["opened_files"]
        assert str(cache_path) not in observed["opened_files"]
    else:
        with (
            worker_identity_module.primary_qrf_worker_launch_environment() as overrides
        ):
            worker = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    worker_identity_module.PRIMARY_QRF_WORKER_MODULE,
                ],
                env={**os.environ, **overrides},
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
        assert worker.returncode == 0, worker.stderr

    assert marker.read_text(encoding="utf-8") == "source"
    assert worker_source.read_bytes() == source_bytes
    assert cache_path.read_bytes() == cache_bytes


@pytest.mark.parametrize("suffix", [".pyc", ".pyo"])
@pytest.mark.usefixtures("live_worker_identity")
def test_worker_identity_namespace_trace_refuses_bytecode_substitution(
    tmp_path: Path,
    suffix: str,
) -> None:
    trace = _fixture_worker_import_trace(tmp_path)
    source = Path(
        trace["module_origins"][worker_identity_module.PRIMARY_QRF_WORKER_MODULE]
    )
    cache_path = source.with_suffix(suffix)
    cache_path.write_bytes(b"unexpected executable cache")
    trace["opened_files"] = (str(cache_path),)

    with pytest.raises(RuntimeError, match="unexpectedly read bytecode"):
        worker_identity_module._worker_package_resource_rows(trace)


@pytest.mark.usefixtures("live_worker_identity")
def test_worker_identity_probe_cleans_cache_prefix_on_launch_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inherited_prefix = tmp_path / "inherited-cache"
    inherited_prefix.mkdir()
    (inherited_prefix / "keep.txt").write_text("caller cache", encoding="utf-8")
    monkeypatch.setenv("PYTHONPYCACHEPREFIX", str(inherited_prefix))
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "0")
    observed_prefixes: list[Path] = []

    def fail_launch(
        _argv: list[str], *, env: dict[str, str], **_kwargs: object
    ) -> None:
        prefix = Path(env["PYTHONPYCACHEPREFIX"])
        assert prefix != inherited_prefix
        assert prefix.is_absolute()
        assert prefix.is_dir()
        assert list(prefix.iterdir()) == []
        assert env["PYTHONDONTWRITEBYTECODE"] == "1"
        observed_prefixes.append(prefix)
        raise OSError("fixture launch failure")

    monkeypatch.setattr(worker_identity_module.subprocess, "run", fail_launch)
    with pytest.raises(RuntimeError, match="clean worker import could not run"):
        worker_identity_module._clean_worker_import_trace()

    assert len(observed_prefixes) == 1
    assert not observed_prefixes[0].exists()
    assert worker_identity_module.os.environ["PYTHONPYCACHEPREFIX"] == str(
        inherited_prefix
    )
    assert worker_identity_module.os.environ["PYTHONDONTWRITEBYTECODE"] == "0"
    assert (inherited_prefix / "keep.txt").read_text(encoding="utf-8") == "caller cache"


@pytest.mark.usefixtures("live_worker_identity")
def test_primary_qrf_worker_identity_binds_loaded_runtime_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_worker_identity_static_closure(monkeypatch)
    runtime = tmp_path / "libpython-fixture.so"
    runtime.write_bytes(b"fixture runtime version one\n")
    # ``raising=False`` makes this a behavioral red test on the vulnerable
    # baseline: that implementation ignores the future loaded-runtime seam and
    # therefore returns the same identity after the library changes.
    monkeypatch.setattr(
        worker_identity_module,
        "_loaded_python_runtime_binary",
        lambda: ("shared_library", runtime),
        raising=False,
    )
    monkeypatch.setattr(
        worker_identity_module,
        "_clean_worker_import_trace",
        lambda *_args, **_kwargs: _fixture_worker_import_trace(tmp_path),
        raising=False,
    )

    before = worker_identity_module.primary_qrf_worker_semantic_identity(
        uv_lock_sha256=worker_identity_module.APPROVED_UV_LOCK_SHA256
    )
    runtime.write_bytes(b"fixture runtime version two\n")
    worker_identity_module.clear_primary_qrf_worker_identity_cache()
    after = worker_identity_module.primary_qrf_worker_semantic_identity(
        uv_lock_sha256=worker_identity_module.APPROVED_UV_LOCK_SHA256
    )

    assert before["interpreter"].get("runtime_binary") != after["interpreter"].get(
        "runtime_binary"
    )


@pytest.mark.usefixtures("live_worker_identity")
def test_primary_qrf_worker_identity_binds_imported_stdlib_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_worker_identity_static_closure(monkeypatch)
    stdlib = tmp_path / "stdlib"
    stdlib.mkdir()
    argparse_source = stdlib / "argparse.py"
    argparse_source.write_text("FIXTURE_VALUE = 1\n", encoding="utf-8")
    site_packages = tmp_path / "site-packages"
    monkeypatch.setattr(
        worker_identity_module,
        "_loaded_python_runtime_binary",
        lambda: ("shared_library", tmp_path / "unchanged-libpython.so"),
        raising=False,
    )
    (tmp_path / "unchanged-libpython.so").write_bytes(b"unchanged runtime\n")
    monkeypatch.setattr(
        worker_identity_module,
        "_clean_worker_import_trace",
        lambda *_args, **_kwargs: _fixture_worker_import_trace(
            tmp_path,
            module_origins={"argparse": str(argparse_source)},
        ),
        raising=False,
    )
    original_get_paths = worker_identity_module.sysconfig.get_paths

    def fixture_get_paths(*args: object, **kwargs: object) -> dict[str, str]:
        paths = dict(original_get_paths(*args, **kwargs))
        paths.update(
            {
                "stdlib": str(stdlib),
                "platstdlib": str(stdlib),
                "purelib": str(site_packages),
                "platlib": str(site_packages),
            }
        )
        return paths

    monkeypatch.setattr(
        worker_identity_module.sysconfig, "get_paths", fixture_get_paths
    )

    before = worker_identity_module.primary_qrf_worker_semantic_identity(
        uv_lock_sha256=worker_identity_module.APPROVED_UV_LOCK_SHA256
    )
    argparse_source.write_text("FIXTURE_VALUE = 2\n", encoding="utf-8")
    worker_identity_module.clear_primary_qrf_worker_identity_cache()
    after = worker_identity_module.primary_qrf_worker_semantic_identity(
        uv_lock_sha256=worker_identity_module.APPROVED_UV_LOCK_SHA256
    )

    assert before["interpreter"].get("stdlib_imports_sha256") != after[
        "interpreter"
    ].get("stdlib_imports_sha256")


@pytest.mark.usefixtures("live_worker_identity")
def test_worker_identity_refuses_unapproved_torch_backend_provider_before_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_worker_identity_static_closure(
        monkeypatch,
        stub_installed_distributions=False,
    )
    provider = SimpleNamespace(
        metadata={"Name": "fixture-torch-backend"},
        version="1.0",
        requires=(),
        files=(),
        entry_points=(
            worker_identity_module.metadata.EntryPoint(
                name="fixture_backend",
                value="fixture_backend:register",
                group="torch.backends",
            ),
        ),
    )
    monkeypatch.setattr(
        worker_identity_module.metadata,
        "distributions",
        lambda: (provider,),
    )
    monkeypatch.setattr(
        worker_identity_module.metadata,
        "packages_distributions",
        lambda: {},
    )

    def unexpected_import_trace(*_args: object, **_kwargs: object) -> object:
        pytest.fail("clean worker import ran before torch backend provider refusal")

    monkeypatch.setattr(
        worker_identity_module,
        "_clean_worker_import_trace",
        unexpected_import_trace,
        raising=False,
    )

    with pytest.raises(RuntimeError, match=r"torch\.backends"):
        worker_identity_module.primary_qrf_worker_semantic_identity(
            uv_lock_sha256=worker_identity_module.APPROVED_UV_LOCK_SHA256
        )


@pytest.mark.usefixtures("live_worker_identity")
def test_worker_identity_refuses_duplicate_backend_provider_distribution_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry_point = worker_identity_module.metadata.EntryPoint(
        name="fixture_backend",
        value="fixture_backend:register",
        group="torch.backends",
    )
    distributions = tuple(
        SimpleNamespace(
            metadata={"Name": "fixture-torch-backend"},
            version=version,
            requires=(),
            files=(),
            entry_points=entry_points_for_distribution,
        )
        for version, entry_points_for_distribution in (
            ("1.0", ()),
            ("2.0", (entry_point,)),
        )
    )
    monkeypatch.setattr(
        worker_identity_module.metadata,
        "distributions",
        lambda: distributions,
    )
    monkeypatch.setattr(
        worker_identity_module.metadata,
        "packages_distributions",
        lambda: {"fixture_backend": ["fixture-torch-backend"]},
    )

    with pytest.raises(RuntimeError, match="duplicate installed distribution identity"):
        worker_identity_module._installed_distributions_record_sha256(
            ("fixture_backend",)
        )


@pytest.mark.usefixtures("live_worker_identity")
def test_worker_transitive_source_identity_binds_actual_imported_package_resource(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_name = "soi_table_2_1_interest_components_ty2015.json"
    trace = worker_identity_module._clean_worker_import_trace()
    before = worker_identity_module._worker_package_resource_rows(trace)
    stdlib_rows = worker_identity_module._worker_stdlib_import_rows(trace)
    assert any(row["module"] == "argparse" for row in stdlib_rows)
    target_rows = [
        row for row in before if str(row.get("resource", "")).endswith(target_name)
    ]
    assert len(target_rows) == 1
    target_resource = target_rows[0]["resource"]
    target_sha256 = target_rows[0]["sha256"]
    _stub_worker_identity_static_closure(monkeypatch)
    runtime = tmp_path / "libpython-fixture.so"
    runtime.write_bytes(b"fixture runtime\n")
    monkeypatch.setattr(
        worker_identity_module,
        "_loaded_python_runtime_binary",
        lambda: ("shared_library", runtime),
    )
    monkeypatch.setattr(
        worker_identity_module,
        "_clean_worker_import_trace",
        lambda: trace,
    )
    identity_before = worker_identity_module.primary_qrf_worker_semantic_identity(
        uv_lock_sha256=worker_identity_module.APPROVED_UV_LOCK_SHA256
    )
    original_read_bytes = Path.read_bytes

    def changed_resource_bytes(path: Path) -> bytes:
        raw = original_read_bytes(path)
        if path.name == target_name:
            return raw + b"\n"
        return raw

    monkeypatch.setattr(Path, "read_bytes", changed_resource_bytes)
    worker_identity_module.clear_primary_qrf_worker_identity_cache()
    after = worker_identity_module._worker_package_resource_rows(trace)
    identity_after = worker_identity_module.primary_qrf_worker_semantic_identity(
        uv_lock_sha256=worker_identity_module.APPROVED_UV_LOCK_SHA256
    )
    changed_target = next(row for row in after if row["resource"] == target_resource)

    assert changed_target["sha256"] != target_sha256
    assert worker_identity_module._canonical_sha256(after) != (
        worker_identity_module._canonical_sha256(before)
    )
    assert (
        identity_after["worker_module"]["transitive_imports_sha256"]
        != (identity_before["worker_module"]["transitive_imports_sha256"])
    )
    assert worker_identity_module._canonical_sha256(identity_after) != (
        worker_identity_module._canonical_sha256(identity_before)
    )


@pytest.mark.parametrize(
    "semantic_change",
    (
        "interpreter_bytes",
        "runtime_binary",
        "stdlib_imports",
        "implementation",
        "version",
        "abi",
        "cache_tag",
        "pyvenv_implementation",
        "pyvenv_version",
        "pyvenv_include_system",
        "pyvenv_uv",
        "worker_module",
        "worker_source",
        "transitive_source",
        "argv_1",
        "argv_2",
        "argv_3",
        "argv_4",
        "argv_5",
        "argv_6",
        "uv_lock",
        "distribution_record",
        "environment_code",
        "fit_jobs",
        "predict_workers",
        "torch_backend_autoload",
        "python_dont_write_bytecode",
        "python_pycache_prefix",
    ),
)
def test_late_primary_worker_authentication_rejects_semantic_matrix(
    semantic_change: str,
) -> None:
    binding = _fixture_primary_execution_config_binding()
    worker = binding["qrf"]["worker_execution"]
    semantic = worker["semantic_identity"]
    refresh_environment_code = False
    if semantic_change == "interpreter_bytes":
        semantic["interpreter"]["bytes_sha256"] = "0" * 64
    elif semantic_change == "runtime_binary":
        semantic["interpreter"]["runtime_binary"]["bytes_sha256"] = "0" * 64
    elif semantic_change == "stdlib_imports":
        semantic["interpreter"]["stdlib_imports_sha256"] = "0" * 64
    elif semantic_change == "implementation":
        semantic["interpreter"]["implementation"] = "changed"
    elif semantic_change == "version":
        semantic["interpreter"]["version"] = [99, 0, 0]
    elif semantic_change == "abi":
        semantic["interpreter"]["abi"]["soabi"] = "changed"
    elif semantic_change == "cache_tag":
        semantic["interpreter"]["cache_tag"] = "changed"
    elif semantic_change == "pyvenv_implementation":
        semantic["interpreter"]["pyvenv_cfg"]["implementation"] = "changed"
    elif semantic_change == "pyvenv_version":
        semantic["interpreter"]["pyvenv_cfg"]["version"] = [99, 0, 0]
    elif semantic_change == "pyvenv_include_system":
        pyvenv = semantic["interpreter"]["pyvenv_cfg"]
        pyvenv["include_system_site_packages"] = not pyvenv[
            "include_system_site_packages"
        ]
    elif semantic_change == "pyvenv_uv":
        semantic["interpreter"]["pyvenv_cfg"]["uv_version"] = "changed"
    elif semantic_change == "worker_module":
        semantic["worker_module"]["name"] = "changed.worker"
    elif semantic_change == "worker_source":
        semantic["worker_module"]["source_sha256"] = "0" * 64
        refresh_environment_code = True
    elif semantic_change == "transitive_source":
        semantic["worker_module"]["transitive_imports_sha256"] = "0" * 64
        refresh_environment_code = True
    elif semantic_change.startswith("argv_"):
        semantic["argv_template"][int(semantic_change.removeprefix("argv_"))] = (
            "changed"
        )
    elif semantic_change == "uv_lock":
        semantic["uv_lock_sha256"] = "0" * 64
    elif semantic_change == "distribution_record":
        semantic["installed_distributions_record_sha256"] = "0" * 64
        refresh_environment_code = True
    elif semantic_change == "environment_code":
        semantic["transitive_environment_code_sha256"] = "0" * 64
    elif semantic_change == "fit_jobs":
        semantic["environment"]["semantic_controls"]["POPULACE_FIT_N_JOBS"][
            "resolved"
        ] = 1
    elif semantic_change == "predict_workers":
        predict = semantic["environment"]["semantic_controls"][
            "POPULACE_FIT_PREDICT_WORKERS"
        ]
        predict["resolved"] = int(predict["resolved"]) + 1
    elif semantic_change == "python_dont_write_bytecode":
        semantic["environment"]["overrides"]["PYTHONDONTWRITEBYTECODE"] = "0"
    elif semantic_change == "python_pycache_prefix":
        semantic["environment"]["overrides"]["PYTHONPYCACHEPREFIX"] = "/existing/cache"
    else:
        semantic["environment"]["overrides"]["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "1"
    if refresh_environment_code:
        semantic["transitive_environment_code_sha256"] = (
            worker_identity_module._canonical_sha256(
                {
                    "worker_module_source_sha256": semantic["worker_module"][
                        "source_sha256"
                    ],
                    "transitive_imports_sha256": semantic["worker_module"][
                        "transitive_imports_sha256"
                    ],
                    "installed_distributions_record_sha256": semantic[
                        "installed_distributions_record_sha256"
                    ],
                }
            )
        )
    worker["semantic_identity_sha256"] = worker_identity_module._canonical_sha256(
        semantic
    )

    with pytest.raises(ValueError, match="worker (binding|identity)"):
        _validate_fixture_primary_execution_config(binding)


def test_portable_worker_identity_does_not_mutate_origin_battery_receipt() -> None:
    values = np.asarray([True, True, True, False, False, False], dtype=bool)
    frame = _battery_frame({"has_esi": (values, values.copy())})
    assert frame.n("household") == 12
    surface = {"person": {"model_required_boolean": ("has_esi",)}}
    authority = stacked_spine_module._make_test_stacked_authority(
        declared_surface=surface,
        metric_registry={
            ("person", "model_required_boolean", "has_esi", 0): ("boolean_incidence")
        },
        support_profile=replace(
            stacked_spine_module.CANONICAL_ORIGIN_BATTERY_SUPPORT_PROFILE,
            min_effective_support=5,
        ),
    )
    before = stacked_spine_module._by_origin_battery_with_test_authority(
        frame,
        authority=authority,
    )
    before_receipt = stacked_spine_module._json_ready(
        {
            "name": before.name,
            "passed": before.passed,
            "failures": list(before.failures),
            "details": before.details,
        }
    )

    config = _fixture_primary_execution_config_binding()
    config["qrf"]["worker_execution"]["audit_aliases"]["sys_prefix"] = (
        "/relocated/worktree/.venv"
    )
    _validate_fixture_primary_execution_config(config)
    after = stacked_spine_module._by_origin_battery_with_test_authority(
        frame,
        authority=authority,
    )
    after_receipt = stacked_spine_module._json_ready(
        {
            "name": after.name,
            "passed": after.passed,
            "failures": list(after.failures),
            "details": after.details,
        }
    )

    label = "person/model_required_boolean/has_esi[clone_0]"
    comparison = after.details["comparisons"][label]
    assert before_receipt == after_receipt
    assert stacked_spine_module._canonical_sha256(before_receipt) == (
        stacked_spine_module._canonical_sha256(after_receipt)
    )
    assert after.passed is True
    assert after.failures == ()
    assert comparison["status"] == "tested"
    assert comparison["nonzero_rows"] == {"asec": 3, "acs": 3}
    assert comparison["asec_incidence"] == 0.5
    assert comparison["acs_incidence"] == 0.5
    assert comparison["incidence_ratio_acs_over_asec"] == 1.0


def test_legacy_worker_mismatch_paths_reproduce_sealed_stop_case() -> None:
    sealed_interpreter = (
        "/Users/maxghenis/PolicyEngine/_worktrees/microcosm-c26-build/.venv/bin/python"
    )
    replay_interpreter = "/private/tmp/microcosm-c27-rootcause/.venv/bin/python"
    common_argv = [
        "-m",
        "microcosm.build.us_runtime.puf_qrf_worker",
        "--checkpoint-dir",
        "{checkpoint_dir}",
        "--target-index",
        "{target_index}",
    ]
    sealed_worker = {
        "module": "microcosm.build.us_runtime.puf_qrf_worker",
        "argv_template": [sealed_interpreter, *common_argv],
        "interpreter": {
            "executable": sealed_interpreter,
            "resolved_executable": (
                "/Users/maxghenis/.local/share/uv/python/"
                "cpython-3.14.4-macos-aarch64-none/bin/python3.14"
            ),
            "implementation": "cpython",
            "cache_tag": "cpython-314",
            "version": [3, 14, 4],
        },
        "environment": {
            "policy": "inherit_parent_environment_with_bound_fit_controls",
            "overrides": {},
            "semantic_controls": {
                "POPULACE_FIT_N_JOBS": {"configured": None, "resolved": -1},
                "POPULACE_FIT_PREDICT_WORKERS": {
                    "configured": "18",
                    "resolved": 18,
                    "resolution": "environment_override",
                },
            },
            "bound_names": [
                "POPULACE_FIT_N_JOBS",
                "POPULACE_FIT_PREDICT_WORKERS",
            ],
        },
    }
    replay_worker = deepcopy(sealed_worker)
    replay_worker["argv_template"][0] = replay_interpreter
    replay_worker["interpreter"]["executable"] = replay_interpreter

    mismatch_paths = stacked_spine_module._legacy_worker_execution_mismatch_paths(
        sealed_worker,
        replay_worker,
    )

    assert set(mismatch_paths) == {
        "argv_template[0]",
        "interpreter.executable",
    }


@pytest.mark.usefixtures("live_worker_identity")
def test_late_primary_resources_bind_donor_content_and_execution_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("POPULACE_FIT_N_JOBS", raising=False)
    monkeypatch.setenv("POPULACE_FIT_PREDICT_WORKERS", "2")
    donor = pd.DataFrame(
        {"income": np.array([10.0, 20.0], dtype=np.float64)},
        index=pd.Index([3, 9], name="donor_id"),
    )
    common = {
        "primary_qrf_checkpoint_identity_sha256": "a" * 64,
        "clone_attachment_fraction": 1.0,
        "clone_attachment_seed": 578,
        "seed": 0,
        "n_estimators": 100,
        "fit_records_enabled": True,
        "tail_bound_diagnostics_enabled": True,
    }

    baseline = stacked_spine_module.stacked_late_primary_resource_receipts(
        donor,
        **common,
    )
    changed_donor = donor.copy()
    changed_donor.iloc[0, 0] = 11.0
    donor_variant = stacked_spine_module.stacked_late_primary_resource_receipts(
        changed_donor,
        **common,
    )
    config_variant = stacked_spine_module.stacked_late_primary_resource_receipts(
        donor,
        **{**common, "clone_attachment_seed": 579},
    )
    monkeypatch.setenv("POPULACE_FIT_PREDICT_WORKERS", "3")
    environment_variant = stacked_spine_module.stacked_late_primary_resource_receipts(
        donor,
        **common,
    )
    monkeypatch.setenv("POPULACE_FIT_PREDICT_WORKERS", "2")
    monkeypatch.setenv("FIXTURE_UNRELATED_SECRET", "must-not-enter-identity")
    unrelated_environment_variant = (
        stacked_spine_module.stacked_late_primary_resource_receipts(
            donor,
            **common,
        )
    )

    assert set(baseline) == {
        "tax_unit.@puf_donor_tax_units",
        "tax_unit.@primary_qrf_checkpoint",
        "tax_unit.@primary_puf_execution_config",
    }
    donor_receipt = baseline["tax_unit.@puf_donor_tax_units"]
    assert (
        donor_receipt["binding"]["table_content_sha256"]
        != (
            donor_variant["tax_unit.@puf_donor_tax_units"]["binding"][
                "table_content_sha256"
            ]
        )
    )
    assert (
        donor_receipt["rows"] == donor_variant["tax_unit.@puf_donor_tax_units"]["rows"]
    )
    assert (
        baseline["tax_unit.@primary_puf_execution_config"]["binding_sha256"]
        != config_variant["tax_unit.@primary_puf_execution_config"]["binding_sha256"]
    )
    qrf = baseline["tax_unit.@primary_puf_execution_config"]["binding"]["qrf"]
    assert qrf["predictors"] == list(
        stacked_spine_module.PUF_TAX_DETAIL_DEFAULT_PREDICTORS
    )
    assert qrf["person_outputs"] == list(
        stacked_spine_module.PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS
    )
    assert qrf["tax_unit_outputs"] == list(
        stacked_spine_module.PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS
    )
    assert qrf["invocation_mode"] == {
        "predictors": "canonical_default",
        "person_outputs": "canonical_default",
        "tax_unit_outputs": "canonical_default",
    }
    execution = baseline["tax_unit.@primary_puf_execution_config"]["binding"]
    assert execution["schema_version"] == 5
    assert execution["clone_attachment"]["support_channels"] == [
        stacked_spine_module.BASE_ASEC_SUPPORT_CHANNEL,
        stacked_spine_module.PUF_TAX_DETAIL_SUPPORT_CHANNEL,
    ]
    assert execution["capital_gains_tail"]["spec"] == (
        stacked_spine_module.puf_capital_gains_tail_spec_identity()
    )
    assert execution["qrf"]["tail_bound_quantiles"] == {
        "non_sch_d_capital_gains": 0.999
    }
    assert execution["doctrines"]["whole_pool_output_universes"] == {
        "person.s_corp_income": {
            "rule_id": "puf_tax_detail_s_corp_income_universe_zero_v1",
            "schema_version": 1,
            "entity": "person",
            "column": "s_corp_income",
            "coverage_scope": "whole_pool",
            "materialized_value": 0.0,
            "source_semantics": (
                "puf_combined_partnership_s_corp_carried_by_partnership_income"
            ),
            "donor_precondition": "finite_exact_zero",
            "puf_clone_precondition": "finite_exact_zero",
            "native_precondition": "all_null",
            "assignment": "explicit_array_assignment",
        }
    }
    worker = execution["qrf"]["worker_execution"]
    assert set(worker) == {
        "schema_version",
        "semantic_identity",
        "semantic_identity_sha256",
        "audit_aliases",
    }
    assert worker["schema_version"] == 1
    semantic = worker["semantic_identity"]
    assert semantic["worker_module"]["name"] == (
        "microcosm.build.us_runtime.puf_qrf_worker"
    )
    interpreter = semantic["interpreter"]
    assert set(interpreter) == {
        "bytes_sha256",
        "runtime_binary",
        "stdlib_imports_sha256",
        "implementation",
        "version",
        "abi",
        "cache_tag",
        "pyvenv_cfg",
    }
    assert interpreter["runtime_binary"]["kind"] in {
        "shared_library",
        "statically_linked_executable",
    }
    assert len(interpreter["runtime_binary"]["bytes_sha256"]) == 64
    assert len(interpreter["stdlib_imports_sha256"]) == 64
    assert semantic["argv_template"] == [
        worker_identity_module.PRIMARY_QRF_INTERPRETER_PLACEHOLDER,
        "-m",
        "microcosm.build.us_runtime.puf_qrf_worker",
        "--checkpoint-dir",
        "{checkpoint_dir}",
        "--target-index",
        "{target_index}",
    ]
    assert worker["semantic_identity_sha256"] == (
        worker_identity_module._canonical_sha256(semantic)
    )
    assert worker["audit_aliases"] == {
        "sys_executable": str(Path(sys.executable)),
        "sys_prefix": str(Path(sys.prefix)),
        "argv_template_0": str(Path(sys.executable)),
    }
    environment = semantic["environment"]
    assert environment["policy"] == (
        "inherit_parent_environment_with_bound_fit_controls_and_forced_overrides"
    )
    assert environment["overrides"] == {
        "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPYCACHEPREFIX": "{empty_pycache_dir}",
    }
    assert environment["bound_names"] == [
        "POPULACE_FIT_N_JOBS",
        "POPULACE_FIT_PREDICT_WORKERS",
        "TORCH_DEVICE_BACKEND_AUTOLOAD",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONPYCACHEPREFIX",
    ]
    assert environment["semantic_controls"] == {
        "POPULACE_FIT_N_JOBS": {"configured": None, "resolved": -1},
        "POPULACE_FIT_PREDICT_WORKERS": {
            "configured": "2",
            "resolved": 2,
            "resolution": "environment_override",
        },
    }
    assert execution["capital_gains_tail"]["soi_e19200_agi_bands"]["asset_sha256"]
    assert execution["capital_gains_tail"]["concentration_gate"] == {
        "schema_version": 2,
        "selection_quantile": 0.995,
        "selection_comparison": "strictly_greater_than",
        "reference_quantile": 0.999,
        "recipient_capital_gains_topcode": 1_999_998.0,
        "positive_mass_five_x_target": 1_270_900_000_000.0,
        "worsening_share_tolerance": 1e-9,
        "ordered_recipient_agi_proxy_columns": [
            "employment_income_before_lsr",
            "self_employment_income_before_lsr",
            "taxable_interest_income",
            "qualified_dividend_income",
            "non_qualified_dividend_income",
            "short_term_capital_gains",
            "long_term_capital_gains_before_response",
        ],
        "ordered_joint_vector_columns": [
            "short_term_capital_gains",
            "long_term_capital_gains_before_response",
            "long_term_capital_gains_on_collectibles",
            "non_sch_d_capital_gains",
            "unrecaptured_section_1250_gain",
        ],
        "recipient_owned_candidate_overlap": sorted(
            set(stacked_spine_module.PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS)
            - set(PUF_CAPITAL_GAINS_TAIL_TAX_UNIT_COLUMNS)
        ),
        "top_k": 100,
        "max_top_share": 0.75,
        "min_nonzero_records": 500,
        "reviewed_exclusions": {},
    }
    assert execution["audit_sinks"] == {
        "fit_records": "enabled",
        "tail_bound_diagnostics": "enabled",
        "recipient_predictor_universe": "required_receipt",
    }
    assert (
        baseline["tax_unit.@primary_puf_execution_config"]["binding_sha256"]
        != environment_variant["tax_unit.@primary_puf_execution_config"][
            "binding_sha256"
        ]
    )
    assert (
        baseline["tax_unit.@primary_puf_execution_config"]["binding_sha256"]
        == unrelated_environment_variant["tax_unit.@primary_puf_execution_config"][
            "binding_sha256"
        ]
    )


@pytest.mark.parametrize(
    ("name", "replacement"),
    (
        ("PUF_CAPITAL_GAINS_TAIL_QUANTILE", 0.994),
        ("PUF_CAPITAL_GAINS_TAIL_REFERENCE_QUANTILE", 0.998),
        ("PUF_CAPITAL_GAINS_TAIL_ASEC_CAPITAL_GAINS_TOPCODE", 2_000_000.0),
        ("PUF_CAPITAL_GAINS_TAIL_POSITIVE_MASS_FIVE_X_TARGET", 1.0),
        ("PUF_CAPITAL_GAINS_TAIL_WORSENING_SHARE_TOLERANCE", 2e-9),
        ("PUF_CAPITAL_GAINS_TAIL_CONCENTRATION_TOP_K", 101),
        ("PUF_CAPITAL_GAINS_TAIL_CONCENTRATION_MAX_TOP_SHARE", 0.74),
        ("PUF_CAPITAL_GAINS_TAIL_CONCENTRATION_MIN_NONZERO_RECORDS", 501),
        (
            "_RECIPIENT_AGI_PROXY_COLUMNS",
            tuple(reversed(tail_module._RECIPIENT_AGI_PROXY_COLUMNS)),
        ),
        (
            "_JOINT_VECTOR_COLUMNS",
            tuple(reversed(tail_module._JOINT_VECTOR_COLUMNS)),
        ),
        ("_RECIPIENT_OWNED_CANDIDATE_OVERLAP", frozenset()),
    ),
)
def test_late_primary_resource_identity_binds_every_tail_control(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    replacement: object,
) -> None:
    # In-memory tail controls change the resource config, while the worker's
    # installed source and startup environment remain identical in both calls.
    worker = _fixture_primary_execution_config_binding()["qrf"]["worker_execution"]
    monkeypatch.setattr(
        stacked_spine_module,
        "_late_primary_qrf_worker_execution_binding",
        lambda: deepcopy(worker),
    )
    donor = pd.DataFrame({"fixture_donor": [1.0]})
    common = {
        "primary_qrf_checkpoint_identity_sha256": "a" * 64,
        "clone_attachment_fraction": 1.0,
        "clone_attachment_seed": 578,
        "seed": 0,
        "n_estimators": 100,
        "fit_records_enabled": True,
        "tail_bound_diagnostics_enabled": True,
    }
    baseline = stacked_spine_module.stacked_late_primary_resource_receipts(
        donor,
        **common,
    )
    monkeypatch.setattr(tail_module, name, replacement)
    changed = stacked_spine_module.stacked_late_primary_resource_receipts(
        donor,
        **common,
    )

    assert (
        baseline["tax_unit.@primary_puf_execution_config"]["binding_sha256"]
        != changed["tax_unit.@primary_puf_execution_config"]["binding_sha256"]
    )


def test_stacked_primary_reuses_one_resolved_tail_input_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, agi_bands = tail_module.resolve_puf_capital_gains_tail_execution_inputs()
    monkeypatch.setattr(
        stacked_spine_module,
        "resolve_puf_capital_gains_tail_execution_inputs",
        lambda: (spec, agi_bands),
    )

    def impute(frame: Frame, *_args: object, **kwargs: object) -> Frame:
        receipts = kwargs["predictor_universe_receipts"]
        assert isinstance(receipts, list)
        receipts.append({"fixture": "recipient-universe"})
        return frame

    observed: dict[str, object] = {}

    def transfer(
        _frame: Frame,
        _donor: pd.DataFrame,
        **kwargs: object,
    ) -> tuple[Frame, dict[str, object]]:
        observed.update(kwargs)
        raise RuntimeError("tail snapshot observed")

    monkeypatch.setattr(
        stacked_spine_module, "impute_us_puf_tax_detail_support", impute
    )
    monkeypatch.setattr(
        stacked_spine_module, "transfer_puf_capital_gains_tail", transfer
    )

    with pytest.raises(RuntimeError, match="tail snapshot observed"):
        stacked_spine_module.run_stacked_puf_pass(
            _late_primary_entry(_stacked_gap_fixture()),
            pd.DataFrame({"fixture_donor": [1.0]}),
            clone_attachment_fraction=1.0,
            clone_attachment_seed=578,
        )

    assert observed["spec"] is spec
    assert observed["agi_bands"] is agi_bands


def test_stacked_primary_qrf_refuses_unbound_surface_and_missing_audit_sink() -> None:
    donor = pd.DataFrame({"fixture_donor": [1.0]})
    resources = stacked_spine_module.stacked_late_primary_resource_receipts(
        donor,
        primary_qrf_checkpoint_identity_sha256="a" * 64,
        clone_attachment_fraction=1.0,
        clone_attachment_seed=578,
        seed=0,
        n_estimators=100,
        fit_records_enabled=True,
        tail_bound_diagnostics_enabled=True,
    )
    binding = stacked_spine_module.stacked_late_primary_checkpoint_input_binding(
        resources
    )
    assert binding["schema_version"] == 2
    frame = _late_primary_entry(_stacked_gap_fixture())

    with pytest.raises(ValueError, match=r"canonical predictor/output surface"):
        stacked_spine_module._run_stacked_puf_pass_without_tail_for_test(
            frame,
            donor,
            clone_attachment_fraction=1.0,
            clone_attachment_seed=578,
            person_outputs=("taxable_interest_income",),
            fit_records=[],
            tail_bound_diagnostics=[],
            primary_qrf_checkpoint_dir=Path("fixture-primary-qrf"),
            primary_qrf_input_binding=binding,
        )

    with pytest.raises(
        ValueError,
        match=r"declared audit sink\(s\).*fit_records",
    ):
        stacked_spine_module._run_stacked_puf_pass_without_tail_for_test(
            frame,
            donor,
            clone_attachment_fraction=1.0,
            clone_attachment_seed=578,
            tail_bound_diagnostics=[],
            primary_qrf_checkpoint_dir=Path("fixture-primary-qrf"),
            primary_qrf_input_binding=binding,
        )


def test_late_primary_resource_rejects_shallow_receipt_before_callback() -> None:
    contract = stacked_spine_module.CANONICAL_US_LATE_PRODUCER_REGISTRY[
        stacked_spine_module.US_LATE_PRIMARY_PUF_STAGE
    ]
    initial = _fill_late_contract_surface(
        _stacked_gap_fixture(),
        contracts=(contract,),
        include_outputs=False,
    )
    resources = stacked_spine_module.stacked_late_primary_resource_receipts(
        pd.DataFrame({"income": [10.0]}),
        primary_qrf_checkpoint_identity_sha256="a" * 64,
        clone_attachment_fraction=1.0,
        clone_attachment_seed=578,
        seed=0,
        n_estimators=100,
        fit_records_enabled=True,
        tail_bound_diagnostics_enabled=True,
    )
    shallow = resources["tax_unit.@puf_donor_tax_units"]
    shallow["binding"] = {
        "resource_kind": "puf_donor_tax_units",
        "schema_version": 1,
    }
    shallow["binding_sha256"] = stacked_spine_module._canonical_sha256(
        shallow["binding"]
    )

    with pytest.raises(
        ValueError,
        match=(
            r"(?s)primary_puf_qrf.*@effective:puf_donor.*"
            r"post_clone_input_surface"
        ),
    ):
        stacked_spine_module.run_stacked_late_producer_dag(
            initial,
            primary_puf_producer=lambda _frame: pytest.fail(
                "shallow resource receipt reached primary callback"
            ),
            primary_resource_receipts=resources,
        )


@pytest.mark.usefixtures("live_worker_identity")
def test_stacked_worker_first_torch_import_forces_autoload_off(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Exercise the stacked launch in a child with no preloaded parent packages."""

    import subprocess

    import microcosm.build.us_runtime.puf_qrf_chain as chain_module

    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "1")
    inherited_prefix = tmp_path / "inherited-pycache"
    inherited_prefix.mkdir()
    monkeypatch.setenv("PYTHONPYCACHEPREFIX", str(inherited_prefix))
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "")
    worker_binding = _fixture_primary_execution_config_binding()["qrf"][
        "worker_execution"
    ]
    monkeypatch.setattr(
        stacked_spine_module,
        "_late_primary_qrf_worker_execution_binding",
        lambda: deepcopy(worker_binding),
    )
    checkpoint_identity = "a" * 64
    checkpoint_dir = tmp_path / checkpoint_identity
    donor = pd.DataFrame({"fixture_donor": [1.0]})

    def initialize(_frame: Frame, _donor: pd.DataFrame, root: Path, **_kwargs) -> None:
        root.mkdir(parents=True)
        (root / stacked_spine_module.PRIMARY_QRF_MANIFEST_FILENAME).write_text("{}")

    monkeypatch.setattr(
        stacked_spine_module, "initialize_primary_puf_qrf_chain", initialize
    )
    monkeypatch.setattr(
        stacked_spine_module,
        "primary_puf_qrf_recipient_predictor_universe_receipt",
        lambda _root: {"fixture": "recipient-universe"},
    )
    monkeypatch.setattr(
        stacked_spine_module,
        "finalize_primary_puf_qrf_chain",
        lambda frame, _root, **_kwargs: (
            frame,
            frame.resolve_weights("tax_unit").kind,
        ),
    )
    # Keep the real chain's subprocess argv/environment assembly; replace only
    # the model/checkpoint work, which the first-import probe never reaches.
    monkeypatch.setattr(
        chain_module,
        "_load_manifest",
        lambda _root: {"target_order": ["fixture"], "initial_state": {}},
    )
    monkeypatch.setattr(
        chain_module, "QRFChainState", SimpleNamespace(from_dict=lambda _raw: None)
    )
    monkeypatch.setattr(
        chain_module, "_load_target_checkpoint", lambda *_args, **_kwargs: (None, None)
    )
    observed: list[dict[str, object]] = []
    launch_prefixes: list[Path] = []
    script = r"""
import builtins
import json
import os
import runpy
import sys

assert "torch" not in sys.modules
assert "microcosm.build" not in sys.modules
original_import = builtins.__import__

def record_first_torch_import(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        print("FIRST_TORCH_IMPORT=" + json.dumps({
            "autoload": os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD"),
            "worker_loaded": "microcosm.build.us_runtime.puf_qrf_worker" in sys.modules,
            "pycache_prefix": sys.pycache_prefix,
            "dont_write_bytecode": sys.flags.dont_write_bytecode,
        }), flush=True)
        raise SystemExit(0)
    return original_import(name, *args, **kwargs)

builtins.__import__ = record_first_torch_import
sys.argv = [sys.argv[1], *sys.argv[2:]]
runpy.run_module("microcosm.build.us_runtime.puf_qrf_worker", run_name="__main__")
raise AssertionError("worker never imported torch")
"""

    def launch_probe(argv: list[str], **kwargs):
        assert argv[1:3] == ["-m", worker_identity_module.PRIMARY_QRF_WORKER_MODULE]
        prefix = Path(kwargs["env"]["PYTHONPYCACHEPREFIX"])
        assert prefix.is_absolute() and prefix.is_dir()
        assert prefix != inherited_prefix
        assert not list(prefix.iterdir())
        launch_prefixes.append(prefix)
        completed = subprocess.run(
            [argv[0], "-c", script, *argv[2:]],
            **kwargs,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert completed.returncode == 0, completed.stderr
        for line in completed.stdout.splitlines():
            if line.startswith("FIRST_TORCH_IMPORT="):
                observed.append(json.loads(line.removeprefix("FIRST_TORCH_IMPORT=")))
        return completed

    monkeypatch.setattr(chain_module, "subprocess", SimpleNamespace(run=launch_probe))
    resources = stacked_spine_module.stacked_late_primary_resource_receipts(
        donor,
        primary_qrf_checkpoint_identity_sha256=checkpoint_identity,
        clone_attachment_fraction=1.0,
        clone_attachment_seed=578,
        seed=0,
        n_estimators=100,
        fit_records_enabled=True,
        tail_bound_diagnostics_enabled=True,
    )
    stacked_spine_module._run_stacked_puf_pass_without_tail_for_test(
        _late_primary_entry(_stacked_gap_fixture()),
        donor,
        clone_attachment_fraction=1.0,
        clone_attachment_seed=578,
        fit_records=[],
        tail_bound_diagnostics=[],
        primary_qrf_checkpoint_dir=checkpoint_dir,
        primary_qrf_input_binding=(
            stacked_spine_module.stacked_late_primary_checkpoint_input_binding(
                resources
            )
        ),
    )

    assert len(launch_prefixes) == 1
    assert observed == [
        {
            "autoload": "0",
            "worker_loaded": False,
            "pycache_prefix": str(launch_prefixes[0]),
            "dont_write_bytecode": 1,
        }
    ]
    assert not launch_prefixes[0].exists()
    assert inherited_prefix.is_dir()
    assert stacked_spine_module.os.environ["PYTHONPYCACHEPREFIX"] == str(
        inherited_prefix
    )
    assert stacked_spine_module.os.environ["PYTHONDONTWRITEBYTECODE"] == ""
    assert (
        worker_binding["semantic_identity"]["environment"]["overrides"][
            "TORCH_DEVICE_BACKEND_AUTOLOAD"
        ]
        == "0"
    )
    assert stacked_spine_module.os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] == "1"


def test_stacked_primary_qrf_refuses_stale_bound_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint_identity = "a" * 64
    checkpoint_dir = tmp_path / checkpoint_identity
    donor = pd.DataFrame({"fixture_donor": [1.0]})

    def initialize(_frame: Frame, _donor: pd.DataFrame, root: Path, **_kwargs) -> None:
        root.mkdir(parents=True)
        (root / stacked_spine_module.PRIMARY_QRF_MANIFEST_FILENAME).write_text(
            "{}",
            encoding="utf-8",
        )

    monkeypatch.setattr(
        stacked_spine_module,
        "initialize_primary_puf_qrf_chain",
        initialize,
    )
    monkeypatch.setattr(
        stacked_spine_module,
        "primary_puf_qrf_recipient_predictor_universe_receipt",
        lambda _root: {"fixture": "recipient-universe"},
    )
    monkeypatch.setattr(
        stacked_spine_module,
        "run_primary_puf_qrf_chain",
        lambda _root, **_kwargs: None,
    )
    monkeypatch.setattr(
        stacked_spine_module,
        "finalize_primary_puf_qrf_chain",
        lambda frame, _root, **_kwargs: (
            frame,
            frame.resolve_weights("tax_unit").kind,
        ),
    )

    def binding(bound_donor: pd.DataFrame) -> dict[str, object]:
        resources = stacked_spine_module.stacked_late_primary_resource_receipts(
            bound_donor,
            primary_qrf_checkpoint_identity_sha256=checkpoint_identity,
            clone_attachment_fraction=1.0,
            clone_attachment_seed=578,
            seed=0,
            n_estimators=100,
            fit_records_enabled=True,
            tail_bound_diagnostics_enabled=True,
        )
        return stacked_spine_module.stacked_late_primary_checkpoint_input_binding(
            resources
        )

    with pytest.raises(
        ValueError,
        match=r"checkpoint directory identity differs.*directory='b{64}'.*bound='a{64}'",
    ):
        stacked_spine_module._run_stacked_puf_pass_without_tail_for_test(
            _late_primary_entry(_stacked_gap_fixture()),
            donor,
            clone_attachment_fraction=1.0,
            clone_attachment_seed=578,
            fit_records=[],
            tail_bound_diagnostics=[],
            primary_qrf_checkpoint_dir=tmp_path / ("b" * 64),
            primary_qrf_input_binding=binding(donor),
        )

    stacked_spine_module._run_stacked_puf_pass_without_tail_for_test(
        _late_primary_entry(_stacked_gap_fixture()),
        donor,
        clone_attachment_fraction=1.0,
        clone_attachment_seed=578,
        fit_records=[],
        tail_bound_diagnostics=[],
        primary_qrf_checkpoint_dir=checkpoint_dir,
        primary_qrf_input_binding=binding(donor),
    )
    assert (checkpoint_dir / "late-producer-input-binding.json").is_file()

    changed_donor = donor.copy()
    changed_donor.iloc[0, 0] = 2.0
    with pytest.raises(ValueError, match="refusing stale predictions"):
        stacked_spine_module._run_stacked_puf_pass_without_tail_for_test(
            _late_primary_entry(_stacked_gap_fixture()),
            changed_donor,
            clone_attachment_fraction=1.0,
            clone_attachment_seed=578,
            fit_records=[],
            tail_bound_diagnostics=[],
            primary_qrf_checkpoint_dir=checkpoint_dir,
            primary_qrf_input_binding=binding(changed_donor),
        )


def test_relocated_checkpoint_resume_reports_the_retained_sidecar_digest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Resume across a worker relocation keeps the sidecar's bytes, so the
    receipt must report the sidecar's own authenticated digest, not the live
    binding's, which hashes the current audit aliases."""
    checkpoint_identity = "a" * 64
    checkpoint_dir = tmp_path / checkpoint_identity
    donor = pd.DataFrame({"fixture_donor": [1.0]})

    def initialize(_frame: Frame, _donor: pd.DataFrame, root: Path, **_kwargs) -> None:
        root.mkdir(parents=True)
        (root / stacked_spine_module.PRIMARY_QRF_MANIFEST_FILENAME).write_text(
            "{}",
            encoding="utf-8",
        )

    monkeypatch.setattr(
        stacked_spine_module, "initialize_primary_puf_qrf_chain", initialize
    )
    monkeypatch.setattr(
        stacked_spine_module,
        "primary_puf_qrf_recipient_predictor_universe_receipt",
        lambda _root: {"fixture": "recipient-universe"},
    )
    monkeypatch.setattr(
        stacked_spine_module, "run_primary_puf_qrf_chain", lambda _root, **_kwargs: None
    )
    monkeypatch.setattr(
        stacked_spine_module,
        "finalize_primary_puf_qrf_chain",
        lambda frame, _root, **_kwargs: (frame, frame.resolve_weights("tax_unit").kind),
    )

    def binding(bound_donor: pd.DataFrame) -> dict[str, object]:
        resources = stacked_spine_module.stacked_late_primary_resource_receipts(
            bound_donor,
            primary_qrf_checkpoint_identity_sha256=checkpoint_identity,
            clone_attachment_fraction=1.0,
            clone_attachment_seed=578,
            seed=0,
            n_estimators=100,
            fit_records_enabled=True,
            tail_bound_diagnostics_enabled=True,
        )
        return stacked_spine_module.stacked_late_primary_checkpoint_input_binding(
            resources
        )

    def run(input_binding: dict[str, object]):
        return stacked_spine_module._run_stacked_puf_pass_without_tail_for_test(
            _late_primary_entry(_stacked_gap_fixture()),
            donor,
            clone_attachment_fraction=1.0,
            clone_attachment_seed=578,
            fit_records=[],
            tail_bound_diagnostics=[],
            primary_qrf_checkpoint_dir=checkpoint_dir,
            primary_qrf_input_binding=input_binding,
        )

    initialized = run(binding(donor)).receipt["primary_puf_qrf"]
    sidecar_path = checkpoint_dir / "late-producer-input-binding.json"
    sidecar_bytes = sidecar_path.read_bytes()
    sidecar_digest = json.loads(sidecar_bytes)["sha256"]
    assert initialized["resume_status"] == "initialized"
    assert initialized["input_binding_sha256"] == sidecar_digest

    real_binding = stacked_spine_module._late_primary_qrf_worker_execution_binding

    def relocated_binding() -> dict[str, object]:
        relocated = copy.deepcopy(real_binding())
        relocated["audit_aliases"]["sys_executable"] = "/relocated/venv/bin/python"
        relocated["audit_aliases"]["sys_prefix"] = "/relocated/venv"
        return relocated

    monkeypatch.setattr(
        stacked_spine_module,
        "_late_primary_qrf_worker_execution_binding",
        relocated_binding,
    )
    live_binding = binding(donor)
    assert live_binding["sha256"] != sidecar_digest, "aliases must move the live digest"

    resumed = run(live_binding).receipt["primary_puf_qrf"]

    assert resumed["resume_status"] == "resumed"
    assert sidecar_path.read_bytes() == sidecar_bytes
    assert resumed["input_binding_sha256"] == sidecar_digest


def test_late_transfer_rejects_identityless_bank_before_dispatch() -> None:
    group = stacked_spine_module.CANONICAL_US_LATE_TRANSFER_GROUPS[0]

    with pytest.raises(ValueError, match="target-bank identity"):
        stacked_spine_module._late_transfer_resource_receipts(
            group_name=group.name,
            entity=group.entity,
            family=group.family,
            targets=group.targets,
            seed=0,
            n_estimators=100,
            max_targets_per_fit=(
                stacked_spine_module.DEFAULT_ACS_TRANSFER_MAX_TARGETS_PER_FIT
            ),
            target_bank=object(),
        )


def test_late_transfer_resources_bind_all_callback_controls() -> None:
    group = stacked_spine_module.CANONICAL_US_LATE_TRANSFER_GROUPS[0]
    resources = stacked_spine_module._late_transfer_resource_receipts(
        group_name=group.name,
        entity=group.entity,
        family=group.family,
        targets=group.targets,
        seed=0,
        n_estimators=100,
        max_targets_per_fit=(
            stacked_spine_module.DEFAULT_ACS_TRANSFER_MAX_TARGETS_PER_FIT
        ),
        target_bank=None,
    )
    model = resources[f"{group.entity}.@late_transfer_model_config"]["binding"]

    assert model["donor_spine"] == stacked_spine_module.ASEC_PUF_DONOR_SPINE
    assert model["donor_channel"] is None
    assert model["donor_selection"] == ("all_rows_from_post_puf_asec_origin_projection")
    assert model["donor_projection"] == {
        "support_channel": stacked_spine_module.BASE_ASEC_SUPPORT_CHANNEL,
        "support_clone_index": stacked_spine_module.PUF_TAX_DETAIL_CLONE_INDEX,
    }
    assert model["schema_version"] == 3
    assert model["transfer_execution_contract"] == (
        acs_transfer_module.acs_transfer_execution_contract_identity(
            targets=group.targets,
            derive_schedule_d=False,
        )
    )
    assert model["transfer_execution_contract"]["post_transfer_structure"][
        "schedule_d_capital_gain_distributions"
    ] == {
        "enabled": False,
        "source": "long_term_capital_gains_before_response",
        "exclusive_with": "non_sch_d_capital_gains",
        "output": "schedule_d_capital_gain_distributions",
        "preserve_preexisting_nonnull": True,
        "share_asset": None,
    }


def test_late_transfer_refuses_a_stale_bound_execution_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = next(
        item
        for item in stacked_spine_module.CANONICAL_US_LATE_TRANSFER_GROUPS
        if item.name == "transfer:person/adult_care"
    )
    bound = acs_transfer_module.acs_transfer_execution_contract_identity(
        targets=group.targets,
        derive_schedule_d=False,
    )
    changed_codes = dict(acs_transfer_module._TENURE_CODES)
    changed_codes["OWN"] = 9.0
    monkeypatch.setattr(acs_transfer_module, "_TENURE_CODES", changed_codes)

    with pytest.raises(
        ValueError,
        match="runtime execution contract differs from its bound input",
    ):
        acs_transfer_module.transfer_acs_inputs(
            _post_puf_transfer_fixture(),
            _post_puf_transfer_fixture(),
            target_families=group.target_families,
            donor_spine=stacked_spine_module.ASEC_PUF_DONOR_SPINE,
            donor_channel=None,
            derive_schedule_d=False,
            execution_contract=bound,
        )


def test_late_source_resources_bind_all_callback_controls(tmp_path: Path) -> None:
    allow_existing_operators = {
        "with_us_child_support_inputs",
        "with_us_disability_benefits",
        "with_us_workers_compensation",
        "with_us_childcare_inputs",
        "with_us_adult_care_inputs",
        "with_us_energy_subsidy_input",
    }
    for operator in multispine_pool_module.POOL_POST_CLONE_SOURCE_OPERATOR_ORDER:
        producer = f"source:{operator}"
        resources = stacked_spine_module._late_source_resource_receipts(
            producer_name=producer
        )
        assert set(resources) == {"person.@post_clone_source_execution_config"}
        binding = resources["person.@post_clone_source_execution_config"]["binding"]
        assert binding["schema_version"] == 3
        assert binding["operator"] == operator
        assert binding["phase"] == multispine_pool_module._POST_CLONE_PHASE
        assert binding["operator_registry"] == list(
            multispine_pool_module.POOL_POST_CLONE_SOURCE_OPERATOR_ORDER
        )
        contract = multispine_pool_module.POOL_OPERATOR_CONTRACTS[operator]
        assert binding["operator_contract"] == {
            "family": contract.family,
            "phases": list(contract.phases),
            "mechanism": contract.mechanism,
            "execution_scope": contract.execution_scope,
        }
        assert binding["declared_output_family"]
        assert binding["seed"] == multispine_pool_module.POOL_RANDOM_SEED
        assert binding["time_period"] == (
            None
            if operator == "impute_us_housing_assistance_to_puf_support"
            else multispine_pool_module.POOL_TIME_PERIOD
        )
        assert binding["force_puf_imputation"] is (
            True if operator == "with_us_retirement_distribution_inputs" else None
        )
        expected_sidecars = {}
        if operator == "with_us_weeks_unemployed":
            expected_sidecars = {"asec_2023_source": {"mode": "not_supplied"}}
        if operator == "with_us_education_inputs":
            expected_sidecars = {"asec_education_source": {"mode": "not_supplied"}}
        assert binding["external_sidecars"] == expected_sidecars
        assert binding["allow_existing_without_source"] is (
            multispine_pool_module.POOL_SOURCE_ALLOW_EXISTING_WITHOUT_SOURCE
            if operator in allow_existing_operators
            else None
        )
        assert binding["housing_assistance_qrf"] == (
            {
                "n_estimators": multispine_pool_module.POOL_HOUSING_ASSISTANCE_N_ESTIMATORS,
                "max_train_samples": (
                    multispine_pool_module.POOL_HOUSING_ASSISTANCE_MAX_TRAIN_SAMPLES
                ),
            }
            if operator == "impute_us_housing_assistance_to_puf_support"
            else None
        )
        source_stage = binding["source_stage_spec"]
        if operator == "impute_us_housing_assistance_to_puf_support":
            assert source_stage is None
        else:
            assert (
                source_stage["stage_name"]
                == (source_stage["resolved_stage_spec"]["stage"])
            )
            assert source_stage["runtime_stage_spec_verified"] is True
            assert set(source_stage["runtime_stage_spec_resolver"]) == {
                "module",
                "callable",
            }
            assert source_stage["resolved_stage_spec_sha256"] == (
                stacked_spine_module._canonical_sha256(
                    source_stage["resolved_stage_spec"]
                )
            )

    source_asset = stacked_spine_module.files("microcosm.build.us").joinpath(
        "source_stages.json"
    )
    changed_payload = json.loads(source_asset.read_text(encoding="utf-8"))
    stage = next(
        item
        for item in changed_payload["stages"]
        if item["stage"] == "adult_care_inputs"
    )
    stage["notes"] = f"{stage.get('notes', '')} identity-regression"
    changed_asset = tmp_path / "source_stages.json"
    changed_asset.write_text(json.dumps(changed_payload), encoding="utf-8")
    baseline = stacked_spine_module._late_source_stage_spec_binding(
        "with_us_adult_care_inputs"
    )
    changed = stacked_spine_module._late_source_stage_spec_binding(
        "with_us_adult_care_inputs",
        resource=changed_asset,
    )
    assert baseline["asset_sha256"] != changed["asset_sha256"]
    assert (
        baseline["resolved_stage_spec_sha256"]
        != (changed["resolved_stage_spec_sha256"])
    )


def test_source_resource_refuses_live_callback_control_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(multispine_pool_module, "POOL_RANDOM_SEED", 913)

    with pytest.raises(ValueError, match="late source execution config changed"):
        stacked_spine_module._late_source_resource_receipts(
            producer_name="source:with_us_adult_care_inputs"
        )


def test_source_resource_binding_does_not_introspect_injected_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def injected_runner(*_args: object, **_kwargs: object) -> PoolStageOutput:
        raise AssertionError("resource construction must not execute the runner")

    monkeypatch.setattr(
        multispine_pool_module,
        "_run_source_operator_chain",
        injected_runner,
    )
    resources = stacked_spine_module._late_source_resource_receipts(
        producer_name="source:with_us_adult_care_inputs"
    )
    binding = resources["person.@post_clone_source_execution_config"]["binding"]
    family = multispine_pool_module.POOL_OPERATOR_CONTRACTS[
        "with_us_adult_care_inputs"
    ].family
    assert binding["declared_output_family"] == {
        entity: sorted(columns)
        for entity, columns in sorted(
            multispine_pool_module.PRE_ASSEMBLY_OPERATOR_OUTPUT_FAMILIES[family].items()
        )
    }


def test_source_resource_refuses_runtime_stage_helper_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = stacked_spine_module.importlib.import_module(
        "microcosm.build.us_runtime.adult_care"
    )
    stage_spec = runtime.us_adult_care_stage_spec()
    monkeypatch.setattr(
        runtime,
        "us_adult_care_stage_spec",
        lambda: replace(stage_spec, notes=f"{stage_spec.notes} drift"),
    )

    with pytest.raises(ValueError, match="runtime SourceStageSpec differs"):
        stacked_spine_module._late_source_resource_receipts(
            producer_name="source:with_us_adult_care_inputs"
        )


def test_late_source_finalizer_resources_bind_all_callback_controls() -> None:
    resources = stacked_spine_module._late_source_finalizer_resource_receipts()
    assert set(resources) == {"person.@source_finalizer_execution_config"}
    binding = resources["person.@source_finalizer_execution_config"]["binding"]
    assert binding["schema_version"] == 2
    assert binding == stacked_spine_module._late_source_finalizer_execution_binding()
    assert binding["source_operator_registry"] == list(
        multispine_pool_module.POOL_POST_CLONE_SOURCE_OPERATOR_ORDER
    )
    assert binding["deferred_transfer_inputs"] == (
        multispine_pool_module.POOL_DEFERRED_TRANSFER_INPUTS
    )


def test_source_finalizer_resource_refuses_live_doctrine_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        multispine_pool_module,
        "POOL_DEFERRED_TRANSFER_STATUS",
        "unreceipted_runtime_drift",
    )

    with pytest.raises(ValueError, match="late source-finalizer config changed"):
        stacked_spine_module._late_source_finalizer_resource_receipts()


def test_primary_refuses_missing_universe_receipt_before_callback() -> None:
    contract = stacked_spine_module.CANONICAL_US_LATE_PRODUCER_REGISTRY[
        stacked_spine_module.US_LATE_PRIMARY_PUF_STAGE
    ]
    initial = _fill_late_contract_surface(
        _stacked_gap_fixture(),
        contracts=(contract,),
        include_outputs=False,
    )
    resources = stacked_spine_module.stacked_late_primary_resource_receipts(
        pd.DataFrame({"fixture_donor": [1.0]}),
        primary_qrf_checkpoint_identity_sha256="a" * 64,
        clone_attachment_fraction=1.0,
        clone_attachment_seed=578,
        seed=0,
        n_estimators=100,
        fit_records_enabled=True,
        tail_bound_diagnostics_enabled=True,
    )
    unfilled, invalid = stacked_spine_module._late_input_readiness_rows(
        initial,
        contract,
        available_input_receipts=resources,
    )

    with pytest.raises(
        ValueError,
        match=(
            r"(?s)primary_puf_qrf.*"
            r"frame\.@acs_pums_earnings_universe_application.*1 unfilled.*"
            r"acs_pums_earnings_universe"
        ),
    ):
        stacked_spine_module.run_producer_when_ready(
            contract,
            lambda: pytest.fail("universe-less frame reached primary callback"),
            unfilled_rows=unfilled,
            invalid_rows=invalid,
            absence_receipts={},
        )


def test_primary_tax_unit_passthrough_requires_finite_or_declared_absence() -> None:
    contract = stacked_spine_module.CANONICAL_US_LATE_PRODUCER_REGISTRY[
        stacked_spine_module.US_LATE_PRIMARY_PUF_STAGE
    ]
    initial = _fill_late_contract_surface(
        _late_primary_entry(_stacked_gap_fixture()),
        contracts=(contract,),
        include_outputs=False,
    )
    column = stacked_spine_module.PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS[0]
    requirement = next(
        item
        for item in contract.inputs
        if item.column == f"@effective:tax_unit_output_passthrough:{column}"
    )
    resources = stacked_spine_module.stacked_late_primary_resource_receipts(
        pd.DataFrame({"fixture_donor": [1.0]}),
        primary_qrf_checkpoint_identity_sha256="a" * 64,
        clone_attachment_fraction=1.0,
        clone_attachment_seed=578,
        seed=0,
        n_estimators=100,
        fit_records_enabled=True,
        tail_bound_diagnostics_enabled=True,
    )

    absent_tables = {
        entity: initial.table(entity).copy() for entity in initial.entities
    }
    absent_tables["tax_unit"].drop(columns=column, inplace=True)
    absent = Frame(
        absent_tables,
        initial.schema,
        {entity: initial.weights_for(entity) for entity in initial.weighted_entities},
        initial.strata,
        mass_log=initial.mass_log,
        metadata=initial.metadata,
    )
    unfilled, invalid = stacked_spine_module._late_input_readiness_rows(
        absent,
        contract,
        available_input_receipts=resources,
    )
    absence_receipts = stacked_spine_module._late_declared_absence_receipts(
        contract,
        unfilled,
        invalid_rows=invalid,
    )
    receipt_id = requirement.tolerated_absence_receipts[0]
    assert unfilled[requirement] == len(absent.table("tax_unit"))
    assert invalid[requirement] == 0
    assert absence_receipts[receipt_id]["status"] == "declared_absence"
    assert (
        stacked_spine_module.run_producer_when_ready(
            contract,
            lambda: "ran",
            unfilled_rows=unfilled,
            invalid_rows=invalid,
            absence_receipts=absence_receipts,
        )
        == "ran"
    )

    invalid_tables = {
        entity: initial.table(entity).copy() for entity in initial.entities
    }
    invalid_tables["tax_unit"][column] = invalid_tables["tax_unit"][column].astype(
        object
    )
    invalid_tables["tax_unit"].loc[invalid_tables["tax_unit"].index[0], column] = (
        "not-numeric"
    )
    invalid_frame = Frame(
        invalid_tables,
        initial.schema,
        {entity: initial.weights_for(entity) for entity in initial.weighted_entities},
        initial.strata,
        mass_log=initial.mass_log,
        metadata=initial.metadata,
    )
    unfilled, invalid = stacked_spine_module._late_input_readiness_rows(
        invalid_frame,
        contract,
        available_input_receipts=resources,
    )
    absence_receipts = stacked_spine_module._late_declared_absence_receipts(
        contract,
        unfilled,
        invalid_rows=invalid,
    )
    assert unfilled[requirement] == 0
    assert invalid[requirement] == 1
    assert receipt_id not in absence_receipts
    with pytest.raises(
        ValueError,
        match=(
            rf"(?s)primary_puf_qrf.*tax_unit_output_passthrough:{column}.*"
            r"1 invalid.*post_clone_input_surface"
        ),
    ):
        stacked_spine_module.run_producer_when_ready(
            contract,
            lambda: pytest.fail("invalid tax-unit passthrough reached callback"),
            unfilled_rows=unfilled,
            invalid_rows=invalid,
            absence_receipts=absence_receipts,
        )


def test_universe_resource_binds_exact_contract_and_scope() -> None:
    resources = stacked_spine_module._late_acs_earnings_universe_resource_receipts()
    receipt = resources["person.@acs_pums_earnings_universe_execution_config"]
    binding = receipt["binding"]
    assert binding["schema_version"] == 2
    assert binding["runtime_identity_owner"] == (
        "microcosm.build.us_runtime.acs_income_universe"
    )
    assert binding["ordered_mapped_columns"] == [
        "employment_income_before_lsr",
        "self_employment_income_before_lsr",
    ]
    assert binding["person_scope_mode"] == "whole_frame_acs_channel"
    assert binding["contract_identity"] == (
        stacked_spine_module.acs_pums_earnings_universe_contract_identity()
    )


def test_universe_resource_refuses_live_contract_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed = deepcopy(universe_module.acs_pums_earnings_universe_contract_identity())
    changed["minimum_age"] = 14
    monkeypatch.setattr(
        universe_module,
        "acs_pums_earnings_universe_contract_identity",
        lambda: changed,
    )

    with pytest.raises(ValueError, match="late ACS earnings-universe contract changed"):
        stacked_spine_module._late_acs_earnings_universe_resource_receipts()


def test_universe_raw_authority_binds_present_column_with_structural_nulls() -> None:
    contract = stacked_spine_module.CANONICAL_US_LATE_PRODUCER_REGISTRY[
        stacked_spine_module.US_LATE_ACS_EARNINGS_UNIVERSE_STAGE
    ]
    resources = stacked_spine_module._late_acs_earnings_universe_resource_receipts()
    initial = _late_universe_entry_fixture()
    unfilled, invalid = stacked_spine_module._late_input_readiness_rows(
        initial,
        contract,
        available_input_receipts=resources,
    )
    wagp = next(
        requirement
        for requirement in contract.inputs
        if requirement.column == "@effective:raw_source:WAGP"
    )
    assert unfilled[wagp] == 0
    assert invalid[wagp] == 0

    evidence = stacked_spine_module._late_declared_input_evidence(
        initial,
        contract,
        available_input_receipts=resources,
        unfilled_rows=unfilled,
        invalid_rows=invalid,
    )
    baseline_sha256 = stacked_spine_module._canonical_sha256(evidence)

    changed_person = initial.table("person").copy()
    eligible = changed_person[support_channel_column("person")].eq(
        "acs"
    ) & pd.to_numeric(changed_person["age"], errors="raise").ge(15)
    changed_person.loc[changed_person.index[eligible][0], "WAGP"] = 2.0
    changed = Frame(
        {
            entity: changed_person if entity == "person" else initial.table(entity)
            for entity in initial.entities
        },
        initial.schema,
        {entity: initial.weights_for(entity) for entity in initial.weighted_entities},
        initial.strata,
        mass_log=initial.mass_log,
        metadata=initial.metadata,
    )
    changed_unfilled, changed_invalid = stacked_spine_module._late_input_readiness_rows(
        changed,
        contract,
        available_input_receipts=resources,
    )
    changed_evidence = stacked_spine_module._late_declared_input_evidence(
        changed,
        contract,
        available_input_receipts=resources,
        unfilled_rows=changed_unfilled,
        invalid_rows=changed_invalid,
    )
    assert stacked_spine_module._canonical_sha256(changed_evidence) != baseline_sha256

    absent_person = initial.table("person").drop(columns=["WAGP"])
    absent = Frame(
        {
            entity: absent_person if entity == "person" else initial.table(entity)
            for entity in initial.entities
        },
        initial.schema,
        {entity: initial.weights_for(entity) for entity in initial.weighted_entities},
        initial.strata,
        mass_log=initial.mass_log,
        metadata=initial.metadata,
    )
    absent_unfilled, absent_invalid = stacked_spine_module._late_input_readiness_rows(
        absent,
        contract,
        available_input_receipts=resources,
    )
    assert absent_unfilled[wagp] > 0
    with pytest.raises(
        ValueError,
        match=r"(?s)acs_pums_earnings_universe.*raw_source:WAGP.*post_clone_input_surface",
    ):
        stacked_spine_module.run_producer_when_ready(
            contract,
            lambda: pytest.fail("missing WAGP reached universe callback"),
            unfilled_rows=absent_unfilled,
            invalid_rows=absent_invalid,
            absence_receipts={},
        )


def test_real_late_executor_follows_canonical_order_and_finalizes_sources_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, events, finalizer_calls = _run_real_late_executor_fixture(monkeypatch)
    schedule = stacked_spine_module.CANONICAL_US_LATE_PRODUCER_SCHEDULE

    assert events == schedule.order
    assert finalizer_calls == 1
    universe_row = result.receipt["execution"][0]
    assert universe_row["producer"] == (
        stacked_spine_module.US_LATE_ACS_EARNINGS_UNIVERSE_STAGE
    )
    assert universe_row["declared_absence_receipts"]
    produced_person = result.primary_puf_result.frame.table("person")
    produced_child = produced_person[
        produced_person[support_channel_column("person")].eq("acs")
        & produced_person["age"].lt(15)
    ]
    assert (
        produced_child[
            [
                "employment_income_before_lsr",
                "self_employment_income_before_lsr",
            ]
        ]
        .eq(0.0)
        .all()
        .all()
    )
    assert events.index("transfer:person/puf_tax_itemization__batch_5") < events.index(
        "source:with_us_adult_care_inputs"
    )
    assert (
        result.transition_authority_sha256
        == result.frame.metadata[
            stacked_spine_module.US_LATE_PRODUCER_TRANSITION_AUTHORITY_KEY
        ]["sha256"]
    )
    stacked_spine_module.validate_stacked_late_producer_receipt(
        result.receipt,
        boundary="executor regression",
        frame=result.frame,
        expected_transition_authority_sha256=result.transition_authority_sha256,
    )


def test_late_executor_signature_rejects_qrf_regime_evidence_tampering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _events, _finalizer_calls = _run_real_late_executor_fixture(monkeypatch)
    forged = deepcopy(dict(result.receipt))
    target_receipt = next(
        target
        for row in forged["execution"]
        if row["kind"] == "late_transfer"
        for target in row["producer_receipt"]["targets"].values()
        if "qrf_pattern_evidence" in target
    )
    target_receipt["qrf_pattern_evidence"]["patterns"][0]["target_regimes"][0][
        "regime"
    ] = "negative_only"

    with pytest.raises(ValueError, match="callback-receipt SHA-256 mismatch"):
        stacked_spine_module.validate_stacked_late_producer_receipt(
            forged,
            boundary="tampered signed QRF regime evidence",
        )


def test_post_puf_validator_rejects_unassigned_legacy_count_tampering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _events, _finalizer_calls = _run_real_late_executor_fixture(monkeypatch)
    transfer = deepcopy(dict(result.receipt["post_puf_transfer"]))
    target_key, target_receipt = next(
        (key, target)
        for group in transfer["groups"].values()
        for key, target in group["targets"].items()
        if "qrf_pattern_evidence" not in target
    )
    target = target_key.rsplit("/", 1)[1]
    aggregate_receipt = next(
        receipt
        for key, receipt in transfer["targets"].items()
        if key.rsplit("/", 1)[1] == target
    )
    for receipt in (target_receipt, aggregate_receipt):
        receipt["imputed_rows"] = "forged"

    with pytest.raises(ValueError, match="ACS transfer row-count"):
        stacked_spine_module.validate_stacked_post_puf_transfer_receipt(
            transfer,
            boundary="forged unassigned late transfer counts",
            frame=result.frame,
        )


def test_late_executor_signature_rejects_generation_only_calibration_tampering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _events, _finalizer_calls = _run_real_late_executor_fixture(monkeypatch)
    forged = deepcopy(dict(result.receipt))
    owner = next(
        target_receipt["post_transfer_calibration"]
        for row in forged["execution"]
        if row["kind"] == "late_transfer"
        for target_receipt in row["producer_receipt"]["targets"].values()
        if target_receipt.get("post_transfer_calibration") is not None
    )
    calibration = owner["calibration"]
    calibration["scope"]["input_values_sha256"] = "0" * 64
    unsigned = dict(calibration)
    unsigned.pop("sha256")
    calibration["sha256"] = stacked_spine_module._canonical_sha256(unsigned)

    with pytest.raises(ValueError, match="callback-receipt SHA-256 mismatch"):
        stacked_spine_module.validate_stacked_late_producer_receipt(
            forged,
            boundary="tampered generation-only calibration evidence",
            frame=result.frame,
            expected_transition_authority_sha256=(result.transition_authority_sha256),
        )


@pytest.mark.parametrize("mutation", ("carrier", "amount"))
def test_late_transfer_rejects_rehashed_diagnostics_detached_from_live_output(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    result, _events, _finalizer_calls = _run_real_late_executor_fixture(monkeypatch)
    transfer = deepcopy(dict(result.receipt["post_puf_transfer"]))
    group_receipt = next(
        group
        for group in transfer["groups"].values()
        if any(key.endswith("/disability_benefits") for key in group["targets"])
    )
    group_key = next(
        key for key in group_receipt["targets"] if key.endswith("/disability_benefits")
    )
    aggregate_key = next(
        key for key in transfer["targets"] if key.endswith("/disability_benefits")
    )
    target_receipts = (
        group_receipt["targets"][group_key],
        transfer["targets"][aggregate_key],
    )
    for target_receipt in target_receipts:
        calibration = target_receipt["post_transfer_calibration"]["calibration"]
        if mutation == "carrier":
            carrier = calibration["carrier"]
            recipient_total = calibration["weights"]["recipient_total"]
            forged_mass = carrier["after_positive_mass"] / 2.0
            carrier["before_positive_mass"] = forged_mass
            carrier["after_positive_mass"] = forged_mass
            carrier["before_positive_share"] = forged_mass / recipient_total
            carrier["after_positive_share"] = forged_mass / recipient_total
            carrier["residual_after_minus_target"] = (
                forged_mass - carrier["target_positive_mass"]
            )
            carrier["absolute_residual"] = abs(carrier["residual_after_minus_target"])
        else:
            amount = calibration["amount"]
            forged_quantiles = [
                value + 100.0 for value in amount["reference_quantiles"]
            ]
            amount["reference_quantiles"] = forged_quantiles
            amount["recipient_before_quantiles"] = forged_quantiles.copy()
            amount["recipient_after_quantiles"] = forged_quantiles.copy()
            amount["qed_before"] = 0.0
            amount["qed_after"] = 0.0
            for anchor, value in zip(
                amount["anchor_rows"], forged_quantiles, strict=True
            ):
                anchor["reference_value"] = value
        unsigned = dict(calibration)
        unsigned.pop("sha256")
        calibration["sha256"] = stacked_spine_module._canonical_sha256(unsigned)

    with pytest.raises(ValueError, match="diagnostics do not match the live output"):
        stacked_spine_module.validate_stacked_post_puf_transfer_receipt(
            transfer,
            boundary=f"rehashed {mutation} live-diagnostic forgery",
            frame=result.frame,
        )


@pytest.mark.parametrize(
    "mutation",
    ("reference_scope", "output_values", "weights"),
)
def test_late_transfer_rejects_rehashed_context_detached_from_live_output(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    result, _events, _finalizer_calls = _run_real_late_executor_fixture(monkeypatch)
    transfer = deepcopy(dict(result.receipt["post_puf_transfer"]))
    group_receipt = next(
        group
        for group in transfer["groups"].values()
        if any(key.endswith("/disability_benefits") for key in group["targets"])
    )
    group_key = next(
        key for key in group_receipt["targets"] if key.endswith("/disability_benefits")
    )
    aggregate_key = next(
        key for key in transfer["targets"] if key.endswith("/disability_benefits")
    )
    owners = [
        group_receipt["targets"][group_key]["post_transfer_calibration"],
        transfer["targets"][aggregate_key]["post_transfer_calibration"],
    ]
    seen: set[int] = set()
    for owner in owners:
        if id(owner) in seen:
            continue
        seen.add(id(owner))
        calibration = owner["calibration"]
        context = owner["context_binding"]
        if mutation == "reference_scope":
            forged_count = owner["reference_rows"] + 1
            owner["reference_rows"] = forged_count
            calibration["scope"]["reference_rows"] = forged_count
            calibration["scope"]["reference_rows_sha256"] = "0" * 64
            context["scope"]["reference_rows"] = forged_count
            context["scope"]["reference_rows_sha256"] = "0" * 64
            context["live_output"]["reference_rows"] = forged_count
        elif mutation == "output_values":
            calibration["scope"]["output_values_sha256"] = "0" * 64
            context["scope"]["output_values_sha256"] = "0" * 64
        else:
            calibration["weights"]["sha256"] = "0" * 64
            context["weights_sha256"] = "0" * 64
        unsigned = dict(calibration)
        unsigned.pop("sha256")
        calibration["sha256"] = stacked_spine_module._canonical_sha256(unsigned)

    with pytest.raises(ValueError, match="match the live output"):
        stacked_spine_module.validate_stacked_post_puf_transfer_receipt(
            transfer,
            boundary=f"rehashed {mutation} live-context forgery",
            frame=result.frame,
        )


def test_late_executor_authority_binds_every_transfer_bank_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, _events, _finalizer_calls = _run_real_late_executor_fixture(
        monkeypatch,
        bank_identity_sha256="a" * 64,
    )
    second, _events, _finalizer_calls = _run_real_late_executor_fixture(
        monkeypatch,
        bank_identity_sha256="b" * 64,
    )
    assert first.transition_authority_sha256 != second.transition_authority_sha256
    transfer_rows = [
        row for row in first.receipt["execution"] if row["kind"] == "late_transfer"
    ]
    assert len(transfer_rows) == 19
    for row in transfer_rows:
        available = row["available_input_receipts"]
        assert len(available) == 2
        bank = next(
            receipt
            for key, receipt in available.items()
            if key.endswith(".@late_transfer_target_bank")
        )
        assert bank["binding"] == {
            "resource_kind": "late_transfer_target_bank",
            "schema_version": 1,
            "mode": "identity_bound_checkpoint",
            "identity_sha256": "a" * 64,
        }


def test_late_executor_rejects_primary_callback_resource_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(
        ValueError,
        match="primary callback receipt disagrees with its declared resources",
    ):
        _run_real_late_executor_fixture(
            monkeypatch,
            bound_clone_attachment_seed=579,
        )


def test_universe_receipt_excludes_out_of_scope_asec_earnings_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline, _events, _finalizer_calls = _run_real_late_executor_fixture(monkeypatch)
    changed, _events, _finalizer_calls = _run_real_late_executor_fixture(
        monkeypatch,
        asec_earnings_delta=1.0,
    )
    baseline_row = baseline.receipt["execution"][0]
    changed_row = changed.receipt["execution"][0]

    assert baseline_row["input_surface_sha256"] == changed_row["input_surface_sha256"]
    assert (
        baseline_row["producer_receipt_sha256"]
        == changed_row["producer_receipt_sha256"]
    )


@pytest.mark.parametrize(
    "column",
    (
        "person_tax_unit_id",
        "person_support_clone_index",
        "person_source_id",
    ),
)
def test_universe_receipt_affecting_acs_identity_changes_input_surface(
    column: str,
) -> None:
    contract = stacked_spine_module.CANONICAL_US_LATE_PRODUCER_REGISTRY[
        stacked_spine_module.US_LATE_ACS_EARNINGS_UNIVERSE_STAGE
    ]
    resources = stacked_spine_module._late_acs_earnings_universe_resource_receipts()

    def identities(frame: Frame) -> tuple[str, str]:
        unfilled, invalid = stacked_spine_module._late_input_readiness_rows(
            frame,
            contract,
            available_input_receipts=resources,
        )
        evidence = stacked_spine_module._late_declared_input_evidence(
            frame,
            contract,
            available_input_receipts=resources,
            unfilled_rows=unfilled,
            invalid_rows=invalid,
        )
        application = stacked_spine_module._materialize_stacked_acs_earnings_universe(
            frame
        )
        return (
            stacked_spine_module._canonical_sha256(evidence),
            stacked_spine_module._canonical_sha256(application.receipt),
        )

    baseline = _late_universe_entry_fixture()
    changed = _late_universe_entry_fixture()
    person = changed.table("person")
    structural_row = person.index[
        person[support_channel_column("person")].eq("acs") & person["age"].lt(15)
    ][0]
    if column == "person_tax_unit_id":
        previous_id = int(person.loc[structural_row, column])
        replacement_id = previous_id - 1
        person.loc[person[column].eq(previous_id), column] = replacement_id
        tax_unit = changed.table("tax_unit")
        tax_unit.loc[tax_unit["tax_unit_id"].eq(previous_id), "tax_unit_id"] = (
            replacement_id
        )
    else:
        person.loc[structural_row, column] = (
            int(person.loc[structural_row, column]) + 10_000
        )

    baseline_input, baseline_receipt = identities(baseline)
    changed_input, changed_receipt = identities(changed)
    assert changed_input != baseline_input
    assert changed_receipt != baseline_receipt


def test_late_receipt_rejects_internally_consistent_forgery_against_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _events, _finalizer_calls = _run_real_late_executor_fixture(monkeypatch)
    forged = deepcopy(dict(result.receipt))
    forged["input_frame_sha256"] = "0" * 64
    previous = stacked_spine_module._late_execution_genesis_sha256(
        producer_schedule_sha256=forged["producer_schedule"]["payload_sha256"],
        input_frame_sha256=forged["input_frame_sha256"],
    )
    for row in forged["execution"]:
        row["previous_execution_sha256"] = previous
        row.pop("sha256")
        row["sha256"] = stacked_spine_module._canonical_sha256(row)
        previous = row["sha256"]
    forged["execution_chain_sha256"] = previous
    forged.pop("sha256")
    forged["sha256"] = stacked_spine_module._canonical_sha256(forged)
    forged_authority = stacked_spine_module._late_producer_transition_authority_receipt(
        forged
    )
    forged_frame = Frame(
        {entity: result.frame.table(entity) for entity in result.frame.entities},
        result.frame.schema,
        {
            entity: result.frame.weights_for(entity)
            for entity in result.frame.weighted_entities
        },
        result.frame.strata,
        mass_log=result.frame.mass_log,
        metadata={
            **result.frame.metadata,
            stacked_spine_module.US_LATE_PRODUCER_TRANSITION_AUTHORITY_KEY: (
                forged_authority
            ),
        },
    )

    with pytest.raises(
        ValueError,
        match="independently carried late-producer transition authority",
    ):
        stacked_spine_module.validate_stacked_late_producer_receipt(
            forged,
            boundary="forged executor regression",
            frame=forged_frame,
            expected_transition_authority_sha256=(result.transition_authority_sha256),
        )


def test_late_receipt_rejects_live_output_content_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _events, _finalizer_calls = _run_real_late_executor_fixture(monkeypatch)
    tables = {entity: result.frame.table(entity) for entity in result.frame.entities}
    person = tables["person"].copy()
    target = "sstb_self_employment_income_before_lsr"
    person.loc[person.index[0], target] = float(person.loc[person.index[0], target]) + 1
    tables["person"] = person
    drifted = Frame(
        tables,
        result.frame.schema,
        {
            entity: result.frame.weights_for(entity)
            for entity in result.frame.weighted_entities
        },
        result.frame.strata,
        mass_log=result.frame.mass_log,
        metadata=result.frame.metadata,
    )

    with pytest.raises(ValueError, match="output digest does not match the live frame"):
        stacked_spine_module.validate_stacked_late_producer_receipt(
            result.receipt,
            boundary="drifted executor regression",
            frame=drifted,
            expected_transition_authority_sha256=(result.transition_authority_sha256),
        )


def test_late_receipt_rejects_forged_absent_required_virtual_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _events, _finalizer_calls = _run_real_late_executor_fixture(monkeypatch)
    forged = deepcopy(dict(result.receipt))
    primary = next(
        row
        for row in forged["execution"]
        if row["producer"] == stacked_spine_module.US_LATE_PRIMARY_PUF_STAGE
    )
    config_key = "tax_unit.@primary_puf_execution_config"
    config_input = next(
        item
        for item in primary["declared_inputs"]
        if item["column"] == "@effective:primary_puf_execution_config"
    )
    config_evidence = config_input["evidence"]
    config_column = config_evidence["alternatives"][0][0]
    config_column["status"] = "absent"
    config_column["content_sha256"] = stacked_spine_module._canonical_sha256(
        {"absent": True}
    )
    config_evidence["sha256"] = stacked_spine_module._canonical_sha256(
        {"alternatives": config_evidence["alternatives"]}
    )
    del primary["available_input_receipts"][config_key]
    _rehash_late_receipt_after_fixture_mutation(forged)

    with pytest.raises(
        ValueError,
        match="inconsistent counts or content identity",
    ):
        stacked_spine_module.validate_stacked_late_producer_receipt(
            forged,
            boundary="forged required virtual input",
        )


def test_late_receipt_rejects_virtual_evidence_receipt_digest_disagreement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _events, _finalizer_calls = _run_real_late_executor_fixture(monkeypatch)
    forged = deepcopy(dict(result.receipt))
    primary = next(
        row
        for row in forged["execution"]
        if row["producer"] == stacked_spine_module.US_LATE_PRIMARY_PUF_STAGE
    )
    donor_input = next(
        item
        for item in primary["declared_inputs"]
        if item["column"] == "@effective:puf_donor"
    )
    donor_evidence = donor_input["evidence"]
    donor_evidence["alternatives"][0][0]["content_sha256"] = "0" * 64
    donor_evidence["sha256"] = stacked_spine_module._canonical_sha256(
        {"alternatives": donor_evidence["alternatives"]}
    )
    _rehash_late_receipt_after_fixture_mutation(forged)

    with pytest.raises(ValueError, match="disagrees with its declared content"):
        stacked_spine_module.validate_stacked_late_producer_receipt(
            forged,
            boundary="forged virtual input digest",
        )


def test_late_receipt_recomputes_readiness_from_physical_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _events, _finalizer_calls = _run_real_late_executor_fixture(monkeypatch)
    forged = deepcopy(dict(result.receipt))
    primary = next(
        row
        for row in forged["execution"]
        if row["producer"] == stacked_spine_module.US_LATE_PRIMARY_PUF_STAGE
    )
    filing_status = next(
        item
        for item in primary["declared_inputs"]
        if item["column"] == "@effective:filing_status"
    )
    physical = filing_status["evidence"]["alternatives"][0][0]
    physical["missing_rows"] = 1
    filing_status["evidence"]["sha256"] = stacked_spine_module._canonical_sha256(
        {"alternatives": filing_status["evidence"]["alternatives"]}
    )
    _rehash_late_receipt_after_fixture_mutation(forged)

    with pytest.raises(ValueError, match="readiness counts disagree"):
        stacked_spine_module.validate_stacked_late_producer_receipt(
            forged,
            boundary="forged readiness counts",
        )


def test_late_receipt_rejects_completed_absent_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _events, _finalizer_calls = _run_real_late_executor_fixture(monkeypatch)
    forged = deepcopy(dict(result.receipt))
    universe = forged["execution"][0]
    universe["output_surface"][0]["status"] = "absent"
    universe["output_surface_sha256"] = stacked_spine_module._canonical_sha256(
        universe["output_surface"]
    )
    _rehash_late_receipt_after_fixture_mutation(forged)

    with pytest.raises(ValueError, match="completed with absent output"):
        stacked_spine_module.validate_stacked_late_producer_receipt(
            forged,
            boundary="forged absent output",
        )


def test_late_receipt_rejects_rehashed_output_scope_cardinality_forgery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _events, _finalizer_calls = _run_real_late_executor_fixture(monkeypatch)
    forged = deepcopy(dict(result.receipt))
    universe = forged["execution"][0]
    output = next(item for item in universe["output_surface"] if "scope_rows" in item)
    output["scope_rows"] = 0
    universe["output_surface_sha256"] = stacked_spine_module._canonical_sha256(
        universe["output_surface"]
    )
    _rehash_late_receipt_after_fixture_mutation(forged)

    with pytest.raises(ValueError, match="scope cardinality"):
        stacked_spine_module.validate_stacked_late_producer_receipt(
            forged,
            boundary="forged output scope rows",
        )


def test_late_receipt_rejects_rehashed_duplicate_physical_input_forgery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _events, _finalizer_calls = _run_real_late_executor_fixture(monkeypatch)
    forged = deepcopy(dict(result.receipt))
    primary = next(
        row
        for row in forged["execution"]
        if row["producer"] == stacked_spine_module.US_LATE_PRIMARY_PUF_STAGE
    )
    hits = [
        (declared_input, column)
        for declared_input in primary["declared_inputs"]
        for alternative in declared_input["evidence"]["alternatives"]
        for column in alternative
        if column["entity"] == "person" and column["column"] == "person_id"
    ]
    assert len(hits) > 1
    declared_input, column = hits[1]
    column["content_sha256"] = "0" * 64
    declared_input["evidence"]["sha256"] = stacked_spine_module._canonical_sha256(
        {"alternatives": declared_input["evidence"]["alternatives"]}
    )
    _rehash_late_receipt_after_fixture_mutation(forged)

    with pytest.raises(ValueError, match="inconsistent physical input evidence"):
        stacked_spine_module.validate_stacked_late_producer_receipt(
            forged,
            boundary="forged duplicate input evidence",
        )


def test_late_receipt_rejects_detached_source_receipt_output_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _events, _finalizer_calls = _run_real_late_executor_fixture(monkeypatch)
    forged = deepcopy(dict(result.receipt))
    source = next(
        row for row in forged["execution"] if row["kind"] == "post_clone_source"
    )
    output = next(
        item
        for item in source["output_surface"]
        if item["column"].startswith("@source_receipt:")
    )
    output["content_sha256"] = "0" * 64
    source["output_surface_sha256"] = stacked_spine_module._canonical_sha256(
        source["output_surface"]
    )
    _rehash_late_receipt_after_fixture_mutation(forged)

    with pytest.raises(ValueError, match="source-receipt output digest"):
        stacked_spine_module.validate_stacked_late_producer_receipt(
            forged,
            boundary="forged detached source receipt",
        )


def test_post_puf_transfer_preserves_complete_asec_source_producers() -> None:
    frame = _post_puf_transfer_fixture()
    surface = {"person": {"model_required_boolean": ("is_pregnant",)}}
    authority = stacked_spine_module._make_test_stacked_authority(
        declared_surface=surface,
        gap_fill_plan=(),
        post_puf_transfer_surface=surface,
    )
    before = frame.table("person")["is_pregnant"].copy(deep=True)
    result = stacked_spine_module._transfer_stacked_post_puf_inputs_with_test_authority(
        frame,
        authority=authority,
        seed=578,
        n_estimators=10,
    )

    person = result.frame.table("person")
    producer_rows = person[support_channel_column("person")].astype(str).eq("asec")
    assert person["is_pregnant"].notna().all()
    pd.testing.assert_series_equal(
        person.loc[producer_rows, "is_pregnant"],
        before.loc[producer_rows],
    )
    receipt = result.receipt["targets"]["person/model_required_boolean/is_pregnant"]
    assert receipt["producer_roles"] == ["asec_source"]
    assert receipt["producer_rows"] == int(producer_rows.sum())
    assert receipt["authorized_null_rows"] == int((~producer_rows).sum())
    assert receipt["imputed_rows"] == int((~producer_rows).sum())
    assert receipt["unmodeled_rows"] == 0
    assert receipt["residual_null_rows"] == 0
    assert "qrf_pattern_evidence" not in receipt
    record = next(
        item
        for item in result.transfer_result.imputed_inputs
        if item.column == "is_pregnant"
    )
    assert all(not pattern.target_regimes for pattern in record.patterns)
    structural = receipt["structural_policy"]
    assert structural == record.structural_receipt
    assert structural["status"] == "verified"
    assert structural["source_person_key"] == "person_source_id"
    assert structural["qrf_draw_rows"] < receipt["imputed_rows"]
    assert structural["final_domain_violation_rows"] == 0
    assert structural["final_clone_disagreement_source_persons"] == 0
    assert (
        person.groupby("person_source_id", sort=False)["is_pregnant"]
        .nunique()
        .eq(1)
        .all()
    )


def test_complete_pregnancy_surface_retains_zero_imputation_structural_receipt() -> (
    None
):
    frame = _post_puf_transfer_fixture()
    person = frame.table("person").copy()
    recipient_rows = person[support_channel_column("person")].astype(str).eq("acs")
    person.loc[recipient_rows, "is_pregnant"] = False
    tables = {entity: frame.table(entity) for entity in frame.entities}
    tables["person"] = person
    complete = Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )
    surface = {"person": {"model_required_boolean": ("is_pregnant",)}}
    authority = stacked_spine_module._make_test_stacked_authority(
        declared_surface=surface,
        gap_fill_plan=(),
        post_puf_transfer_surface=surface,
    )

    result = stacked_spine_module._transfer_stacked_post_puf_inputs_with_test_authority(
        complete,
        authority=authority,
        seed=578,
        n_estimators=10,
    )

    receipt = result.receipt["targets"]["person/model_required_boolean/is_pregnant"]
    assert receipt["authorized_null_rows"] == 0
    assert receipt["imputed_rows"] == 0
    assert receipt["structural_policy"]["status"] == "verified"
    record = next(
        item
        for item in result.transfer_result.imputed_inputs
        if item.column == "is_pregnant"
    )
    assert record.imputed_recipient_rows == 0
    assert record.structural_receipt == receipt["structural_policy"]


def test_post_puf_transfer_refuses_invalid_pregnancy_source_producer() -> None:
    frame = _post_puf_transfer_fixture()
    person = frame.table("person").copy()
    source_rows = person[support_channel_column("person")].astype(str).eq("asec")
    ineligible = source_rows & ~(
        person["is_female"].astype(bool)
        & person["age"].between(15, 44, inclusive="both")
    )
    donor_ineligible = ineligible & person[support_clone_index_column("person")].eq(1)
    person.loc[person.index[donor_ineligible][0], "is_pregnant"] = True
    tables = {entity: frame.table(entity) for entity in frame.entities}
    tables["person"] = person
    invalid = Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )
    surface = {"person": {"model_required_boolean": ("is_pregnant",)}}
    authority = stacked_spine_module._make_test_stacked_authority(
        declared_surface=surface,
        gap_fill_plan=(),
        post_puf_transfer_surface=surface,
    )

    with pytest.raises(
        ValueError,
        match=r"preexisting donor domain violation",
    ):
        stacked_spine_module._transfer_stacked_post_puf_inputs_with_test_authority(
            invalid,
            authority=authority,
            seed=578,
            n_estimators=10,
        )


def test_late_calibration_owner_mutates_only_acs_clone_zero_transfer_cells() -> None:
    frame = _post_puf_transfer_fixture()
    person = frame.table("person")
    channel = person[support_channel_column("person")].astype(str)
    clone_index = pd.to_numeric(
        person[support_clone_index_column("person")],
        errors="raise",
    )
    reference_rows = channel.eq("asec") & clone_index.eq(0)
    recipient_rows = channel.eq("acs") & clone_index.eq(0)
    target = "child_support_received"

    transferred_person = person.copy(deep=True)
    transferred_person[target] = 0.0
    transferred_person.loc[reference_rows, target] = np.resize(
        np.asarray([0.0, 100.0, 250.0]),
        int(reference_rows.sum()),
    )
    transferred_person.loc[recipient_rows, target] = np.resize(
        np.asarray([0.0, 10.0, 20.0, 30.0]),
        int(recipient_rows.sum()),
    )
    before_person = transferred_person.copy(deep=True)
    before_person.loc[recipient_rows, target] = np.nan

    def rebuild(person_table: pd.DataFrame) -> Frame:
        tables = {entity: frame.table(entity) for entity in frame.entities}
        tables["person"] = person_table
        return Frame(
            tables,
            frame.schema,
            {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
            frame.strata,
            mass_log=frame.mass_log,
            metadata=frame.metadata,
        )

    transferred = rebuild(transferred_person)
    calibrated, receipts = (
        stacked_spine_module._apply_stacked_post_transfer_calibrations(
            rebuild(before_person),
            transferred,
            target_families={
                "person": {
                    "source_operator_child_support": (target,),
                }
            },
            stage="late_transfer",
        )
    )
    calibrated_person = calibrated.table("person")
    pd.testing.assert_series_equal(
        calibrated_person.loc[~recipient_rows, target],
        transferred_person.loc[~recipient_rows, target],
        check_exact=True,
    )
    receipt = receipts[f"person/source_operator_child_support/{target}"]
    assert receipt["stage"] == "late_transfer"
    assert receipt["reference_rows"] == int(reference_rows.sum())
    assert receipt["recipient_rows"] == int(recipient_rows.sum())
    assert receipt["mutable_rows"] == int(recipient_rows.sum())
    assert receipt["calibration"]["invariants"]["immutable_bytes_preserved"]


@pytest.mark.parametrize(
    "clone_index",
    (
        pytest.param(0, id="source_nondonor"),
        pytest.param(1, id="model_donor"),
    ),
)
def test_post_puf_transfer_rejects_incomplete_source_producer(
    clone_index: int,
) -> None:
    frame = _post_puf_transfer_fixture()
    person = frame.table("person").copy()
    source_producer_rows = person[support_channel_column("person")].astype(str).eq(
        "asec"
    ) & person[support_clone_index_column("person")].eq(clone_index)
    person.loc[person.index[source_producer_rows][0], "is_pregnant"] = pd.NA
    tables = {entity: frame.table(entity) for entity in frame.entities}
    tables["person"] = person
    incomplete = Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )
    surface = {"person": {"model_required_boolean": ("is_pregnant",)}}
    authority = stacked_spine_module._make_test_stacked_authority(
        declared_surface=surface,
        gap_fill_plan=(),
        post_puf_transfer_surface=surface,
    )

    with pytest.raises(
        ValueError,
        match=r"post_puf_transfer/person/model_required_boolean/is_pregnant:.*"
        r"upstream producers must observe every producer-owned target",
    ):
        stacked_spine_module._transfer_stacked_post_puf_inputs_with_test_authority(
            incomplete,
            authority=authority,
            seed=578,
            n_estimators=10,
        )


def test_post_puf_transfer_preserves_every_live_puf_clone_producer() -> None:
    frame = _post_puf_puf_producer_fixture()
    surface = {"person": {"puf_tax_itemization": ("educator_expense",)}}
    authority = stacked_spine_module._make_test_stacked_authority(
        declared_surface=surface,
        gap_fill_plan=(),
        post_puf_transfer_surface=surface,
    )
    before = frame.table("person")["educator_expense"].copy(deep=True)

    result = stacked_spine_module._transfer_stacked_post_puf_inputs_with_test_authority(
        frame,
        authority=authority,
        seed=578,
        n_estimators=10,
    )

    person = result.frame.table("person")
    producer_rows = pd.to_numeric(
        person[support_clone_index_column("person")], errors="raise"
    ).gt(0)
    assert person["educator_expense"].notna().all()
    pd.testing.assert_series_equal(
        person.loc[producer_rows, "educator_expense"],
        before.loc[producer_rows],
    )
    receipt = result.receipt["targets"]["person/puf_tax_itemization/educator_expense"]
    assert receipt["producer_roles"] == ["puf_clone"]
    assert receipt["producer_rows"] == int(producer_rows.sum())
    assert receipt["authorized_null_rows"] == int((~producer_rows).sum())
    assert receipt["imputed_rows"] == int((~producer_rows).sum())
    assert receipt["unmodeled_rows"] == 0
    assert receipt["residual_null_rows"] == 0


def test_post_puf_transfer_rejects_incomplete_acs_puf_clone_producer() -> None:
    frame = _post_puf_puf_producer_fixture()
    person = frame.table("person").copy()
    corrupted_rows = person[support_channel_column("person")].astype(str).eq(
        "acs"
    ) & person[support_clone_index_column("person")].eq(1)
    person.loc[person.index[corrupted_rows][0], "educator_expense"] = np.nan
    tables = {entity: frame.table(entity) for entity in frame.entities}
    tables["person"] = person
    incomplete = Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )
    surface = {"person": {"puf_tax_itemization": ("educator_expense",)}}
    authority = stacked_spine_module._make_test_stacked_authority(
        declared_surface=surface,
        gap_fill_plan=(),
        post_puf_transfer_surface=surface,
    )

    with pytest.raises(
        ValueError,
        match=r"post_puf_transfer/person/puf_tax_itemization/educator_expense:.*"
        r"upstream producers must observe every producer-owned target",
    ):
        stacked_spine_module._transfer_stacked_post_puf_inputs_with_test_authority(
            incomplete,
            authority=authority,
            seed=578,
            n_estimators=10,
        )


def test_gap_fill_banks_per_target_via_608_store(tmp_path) -> None:
    identity = {"pilot": "stacked-gap-fill", "seed": 578}
    banks = {
        "asec_survey_to_acs": AcsTransferTargetBankStore(
            tmp_path / "survey",
            identity=identity,
        ),
        "asec_housing_to_acs": AcsTransferTargetBankStore(
            tmp_path / "housing",
            identity=identity,
        ),
    }
    first = _gap_fill_with_test_authority(
        _stacked_gap_fixture(),
        plan=_GAP_FILL_TEST_PLAN,
        seed=578,
        n_estimators=10,
        target_banks=banks,
    )
    survey_files = sorted((tmp_path / "survey" / "targets").glob("*.h5"))
    housing_files = sorted((tmp_path / "housing" / "targets").glob("*.h5"))
    assert len(survey_files) == 2
    assert len(housing_files) == 1

    resumed_banks = {
        "asec_survey_to_acs": AcsTransferTargetBankStore(
            tmp_path / "survey",
            identity=identity,
        ),
        "asec_housing_to_acs": AcsTransferTargetBankStore(
            tmp_path / "housing",
            identity=identity,
        ),
    }
    second = _gap_fill_with_test_authority(
        _stacked_gap_fixture(),
        plan=_GAP_FILL_TEST_PLAN,
        seed=578,
        n_estimators=10,
        target_banks=resumed_banks,
    )
    for column in ("unemployment_compensation", "is_disabled", "pre_subsidy_rent"):
        pd.testing.assert_series_equal(
            first.frame.table("person")[column],
            second.frame.table("person")[column],
        )
    survey_receipt = resumed_banks["asec_survey_to_acs"].receipt()
    assert survey_receipt["targets"]


@pytest.mark.parametrize(
    "use_target_bank",
    [False, True],
    ids=("ordinary", "banked"),
)
def test_gap_fill_scopes_qrf_evidence_off_wide_unassigned_family(
    tmp_path: Path,
    use_target_bank: bool,
) -> None:
    stacked = _stacked_gap_fixture()
    canonical_direction = next(
        direction
        for direction in stacked_spine_module.CANONICAL_STACKED_GAP_FILL_PLAN
        if direction.name == "asec_survey_to_acs"
    )
    puf_targets = canonical_direction.target_families["person"]["puf_tax_itemization"]
    person = stacked.table("person").copy()
    channel = person[support_channel_column("person")].astype(str)
    donor_rows = channel.eq("asec")
    for position, target in enumerate(puf_targets, start=1):
        values = pd.Series(np.nan, index=person.index, dtype=np.float64)
        values.loc[donor_rows] = np.arange(1, int(donor_rows.sum()) + 1) + position
        person[target] = values
    tables = {entity: stacked.table(entity) for entity in stacked.entities}
    tables["person"] = person
    frame = Frame(
        tables,
        stacked.schema,
        {entity: stacked.weights_for(entity) for entity in stacked.weighted_entities},
        stacked.strata,
        mass_log=stacked.mass_log,
        metadata=stacked.metadata,
    )
    direction = GapFillDirection(
        name="asec_survey_to_acs",
        recipient_channel="acs",
        donor_channel="asec",
        target_families={
            "person": {
                "model_required_numeric": ("unemployment_compensation",),
                "puf_tax_itemization": puf_targets,
            }
        },
    )
    target_banks = (
        {
            direction.name: AcsTransferTargetBankStore(
                tmp_path / "survey",
                identity={"regression": "scoped-wide-gap-fill"},
            )
        }
        if use_target_bank
        else None
    )

    result = _gap_fill_with_test_authority(
        frame,
        plan=(direction,),
        seed=578,
        n_estimators=1,
        target_banks=target_banks,
    )

    receipts = result.receipt["directions"][direction.name]["targets"]
    taxable_key = "person/puf_tax_itemization/taxable_interest_income"
    canonical_receipt = _canonical_gap_fill_calibration_receipt()
    canonical_targets = canonical_receipt["directions"][direction.name]["targets"]
    canonical_targets[taxable_key] = deepcopy(receipts[taxable_key])
    stacked_spine_module.validate_stacked_gap_fill_receipt(
        canonical_receipt,
        boundary=(
            f"{'banked' if use_target_bank else 'ordinary'} generated "
            "wide-family receipt"
        ),
    )

    records = {
        record.column: record
        for record in result.transfer_results[direction.name].imputed_inputs
    }
    taxable = records["taxable_interest_income"]
    unemployment = records["unemployment_compensation"]
    assert taxable.family == "puf_tax_itemization__batch_1"
    assert all(not pattern.target_regimes for pattern in taxable.patterns)
    assert all(
        tuple(target for target, _regime in pattern.target_regimes)
        == ("unemployment_compensation",)
        for pattern in unemployment.patterns
    )
    assert "qrf_pattern_evidence" not in receipts[taxable_key]
    assert (
        "qrf_pattern_evidence"
        in receipts["person/model_required_numeric/unemployment_compensation"]
    )


def test_clone_attachment_is_seeded_exact_and_pair_weighted() -> None:
    stacked = _stacked_gap_fixture()
    attached = clone_us_frame_for_puf_support(
        stacked,
        clone_attachment_fraction=0.5,
        clone_attachment_seed=578,
    )
    repeated = clone_us_frame_for_puf_support(
        stacked,
        clone_attachment_fraction=0.5,
        clone_attachment_seed=578,
    )
    changed_seed = clone_us_frame_for_puf_support(
        stacked,
        clone_attachment_fraction=0.5,
        clone_attachment_seed=579,
    )

    manifest = attached.metadata[PUF_CLONE_ATTACHMENT_MANIFEST_KEY]
    assert manifest["eligible_household_count"] == 14
    assert manifest["requested_household_count"] == 7
    assert manifest["realized_household_count"] == 7
    assert manifest["exact_count_rule"] == "floor(fraction * eligible)"
    assert (
        manifest["selected_household_source_ids_sha256"]
        == repeated.metadata[PUF_CLONE_ATTACHMENT_MANIFEST_KEY][
            "selected_household_source_ids_sha256"
        ]
    )
    assert (
        manifest["selected_household_source_ids_sha256"]
        != changed_seed.metadata[PUF_CLONE_ATTACHMENT_MANIFEST_KEY][
            "selected_household_source_ids_sha256"
        ]
    )

    household = attached.table("household")
    clone_index = household[support_clone_index_column("household")]
    assert int(clone_index.eq(0).sum()) == 14
    assert int(clone_index.eq(1).sum()) == 7
    weights = attached.weights_for("household").values
    assert np.isclose(
        float(weights.sum()),
        float(stacked.weights_for("household").total),
        rtol=1e-12,
    )
    source_column = household.columns[household.columns.str.endswith("_source_id")][0]
    attached_ids = set(
        household.loc[clone_index.eq(1), source_column].astype(int).tolist()
    )
    incoming = dict(
        zip(
            stacked.table("household")[source_column].astype(int).tolist(),
            stacked.weights_for("household").values.tolist(),
            strict=True,
        )
    )
    for row, weight in zip(household.itertuples(index=False), weights, strict=True):
        source_id = int(getattr(row, source_column))
        expected = (
            incoming[source_id] / 2.0
            if source_id in attached_ids
            else incoming[source_id]
        )
        assert np.isclose(weight, expected, rtol=1e-12)

    assert validate_puf_clone_attachment(attached, boundary="attachment fixture")


def test_clone_attachment_fraction_one_matches_full_clone() -> None:
    stacked = _stacked_gap_fixture()
    full = clone_us_frame_for_puf_support(stacked)
    attached = clone_us_frame_for_puf_support(
        stacked,
        clone_attachment_fraction=1.0,
        clone_attachment_seed=0,
    )

    assert attached.schema == full.schema
    assert attached.entities == full.entities
    assert attached.links == full.links
    assert attached.weighted_entities == full.weighted_entities
    for entity in full.entities:
        assert_frame_equal(
            attached.table(entity),
            full.table(entity),
            check_exact=True,
        )
    for link in full.links:
        assert_frame_equal(
            attached.link(link),
            full.link(link),
            check_exact=True,
        )
    for entity in full.weighted_entities:
        attached_weights = attached.weights_for(entity)
        full_weights = full.weights_for(entity)
        assert attached_weights.kind == full_weights.kind
        assert attached_weights.values.dtype == full_weights.values.dtype
        assert attached_weights.values.shape == full_weights.values.shape
        assert attached_weights.values.tobytes(
            order="C"
        ) == full_weights.values.tobytes(order="C")
    pd.testing.assert_series_equal(attached.strata, full.strata, check_exact=True)
    assert attached.mass_log == full.mass_log
    assert PUF_CLONE_ATTACHMENT_MANIFEST_KEY not in full.metadata
    assert PUF_CLONE_ATTACHMENT_MANIFEST_KEY not in attached.metadata
    assert attached.metadata == full.metadata
    receipt = validate_puf_clone_attachment(
        attached,
        boundary="fraction-one identity fixture",
        expected_fraction=1.0,
        expected_seed=0,
    )
    assert receipt["authority_form"] == "full_clone_identity_no_manifest"
    assert receipt["eligible_household_count"] == 14
    assert receipt["realized_household_count"] == 14
    with pytest.raises(ValueError, match="clone attachment manifest.*is absent"):
        validate_puf_clone_attachment(
            attached,
            boundary="fraction-one identity without declared expectation",
        )


def test_full_clone_identity_validation_fails_closed() -> None:
    attached = clone_us_frame_for_puf_support(
        _stacked_gap_fixture(),
        clone_attachment_fraction=1.0,
        clone_attachment_seed=0,
    )
    tables = {entity: attached.table(entity) for entity in attached.entities}
    tables.update({link: attached.link(link) for link in attached.links})
    weights = {
        entity: attached.weights_for(entity) for entity in attached.weighted_entities
    }
    household_weights = attached.weights_for("household")
    tampered_values = household_weights.values.copy()
    household_clone = attached.table("household")[
        support_clone_index_column("household")
    ]
    first_detail = int(np.flatnonzero(household_clone.eq(1).to_numpy())[0])
    tampered_values[first_detail] += 1.0
    weights["household"] = Weights(tampered_values, household_weights.kind)
    tampered = Frame(
        tables,
        attached.schema,
        weights,
        attached.strata,
        mass_log=attached.mass_log,
        metadata=attached.metadata,
    )
    with pytest.raises(ValueError, match="full-clone identity failed"):
        validate_puf_clone_attachment(
            tampered,
            boundary="tampered full clone",
            expected_fraction=1.0,
            expected_seed=0,
        )

    asymmetric_metadata = Frame(
        tables,
        attached.schema,
        {entity: attached.weights_for(entity) for entity in attached.weighted_entities},
        attached.strata,
        mass_log=attached.mass_log,
        metadata={
            **attached.metadata,
            PUF_CLONE_ATTACHMENT_MANIFEST_KEY: {"unexpected": True},
        },
    )
    with pytest.raises(ValueError, match="full-clone metadata symmetry failed"):
        validate_puf_clone_attachment(
            asymmetric_metadata,
            boundary="metadata-asymmetric full clone",
            expected_fraction=1.0,
            expected_seed=0,
        )


def test_clone_attachment_configuration_fails_closed() -> None:
    stacked = _stacked_gap_fixture()
    with pytest.raises(ValueError, match="provided together"):
        clone_us_frame_for_puf_support(stacked, clone_attachment_fraction=0.5)
    with pytest.raises(ValueError, match="assembled frame"):
        clone_us_frame_for_puf_support(
            _asec_gap_source(),
            clone_attachment_fraction=0.5,
            clone_attachment_seed=1,
        )
    with pytest.raises(ValueError, match="floors to zero"):
        clone_us_frame_for_puf_support(
            stacked,
            clone_attachment_fraction=0.01,
            clone_attachment_seed=1,
        )


def test_clone_attachment_manifest_mutation_fails_closed() -> None:
    attached = clone_us_frame_for_puf_support(
        _stacked_gap_fixture(),
        clone_attachment_fraction=0.5,
        clone_attachment_seed=578,
    )
    manifest = {
        key: value
        for key, value in attached.metadata[PUF_CLONE_ATTACHMENT_MANIFEST_KEY].items()
    }
    manifest["realized_household_count"] = 8
    manifest["requested_household_count"] = 8
    tampered = Frame(
        {entity: attached.table(entity) for entity in attached.entities},
        attached.schema,
        {entity: attached.weights_for(entity) for entity in attached.weighted_entities},
        attached.strata,
        mass_log=attached.mass_log,
        metadata={
            **attached.metadata,
            PUF_CLONE_ATTACHMENT_MANIFEST_KEY: manifest,
        },
    )
    with pytest.raises(ValueError, match="violates floor"):
        validate_puf_clone_attachment(tampered, boundary="tampered attachment")


def test_stacked_validator_rejects_an_unreceipted_clone_role() -> None:
    attached = clone_us_frame_for_puf_support(
        _stacked_gap_fixture(),
        clone_attachment_fraction=1.0,
        clone_attachment_seed=578,
    )
    tables = {entity: attached.table(entity).copy() for entity in attached.entities}
    for entity, table in tables.items():
        clone_column = support_clone_index_column(entity)
        table.loc[table[clone_column].eq(1), clone_column] = 3
    relabeled = Frame(
        tables,
        attached.schema,
        {entity: attached.weights_for(entity) for entity in attached.weighted_entities},
        attached.strata,
        mass_log=attached.mass_log,
        metadata=attached.metadata,
    )

    with pytest.raises(ValueError, match="unreceipted stacked clone roles.*3"):
        validate_stacked_spine_frame(
            relabeled,
            boundary="unreceipted clone-role fixture",
        )


def test_stacked_validator_rejects_a_partial_attachment_receipt() -> None:
    attached = clone_us_frame_for_puf_support(
        _stacked_gap_fixture(),
        clone_attachment_fraction=0.5,
        clone_attachment_seed=578,
    )
    canonical = attached.metadata[PUF_CLONE_ATTACHMENT_MANIFEST_KEY]
    malformed = Frame(
        {entity: attached.table(entity) for entity in attached.entities},
        attached.schema,
        {entity: attached.weights_for(entity) for entity in attached.weighted_entities},
        attached.strata,
        mass_log=attached.mass_log,
        metadata={
            **attached.metadata,
            PUF_CLONE_ATTACHMENT_MANIFEST_KEY: {
                "realized_household_count": canonical["realized_household_count"],
                "selected_household_source_ids_sha256": canonical[
                    "selected_household_source_ids_sha256"
                ],
            },
        },
    )

    with pytest.raises(ValueError, match="clone attachment manifest is malformed"):
        validate_stacked_spine_frame(
            malformed,
            boundary="partial attachment receipt fixture",
        )


def test_run_stacked_puf_pass_imputes_only_the_attached_arm() -> None:
    gap_filled = _gap_fill_with_test_authority(
        _stacked_gap_fixture(),
        plan=_GAP_FILL_TEST_PLAN,
        seed=578,
        n_estimators=10,
    ).frame
    gap_filled = _late_primary_entry(gap_filled)
    donor = pd.DataFrame(
        {
            "employment_income": [45_000.0, 8_000.0, 70_000.0, 22_000.0],
            "taxable_interest_income": [120.0, 30.0, 900.0, 0.0],
            "weight": [1.0, 1.0, 1.0, 1.0],
        }
    )
    result = stacked_spine_module._run_stacked_puf_pass_without_tail_for_test(
        gap_filled,
        donor,
        clone_attachment_fraction=0.5,
        clone_attachment_seed=578,
        predictors=("puf_predictor_employment_income",),
        person_outputs=("taxable_interest_income",),
        tax_unit_outputs=(),
        seed=578,
        n_estimators=10,
    )

    person = result.frame.table("person")
    channel = person[support_channel_column("person")].astype(str)
    clone_index = person[support_clone_index_column("person")]
    assert person.loc[clone_index.eq(1), "taxable_interest_income"].notna().all()
    assert (
        person.loc[clone_index.eq(0) & channel.eq("acs"), "taxable_interest_income"]
        .isna()
        .all()
    )
    by_origin = result.receipt["recipient_person_rows_by_origin"]
    assert set(by_origin) == {"asec", "acs"}
    assert all(count > 0 for count in by_origin.values())
    assert result.receipt["doctrines"]["absent_cells"] == "preserve_nulls"
    universe = result.receipt["primary_puf_qrf"]["recipient_predictor_universe"]
    assert universe["rules"]["employment_income_before_lsr"]["source_column"] == "WAGP"
    assert universe["raw_pums_source_cells_mutated"] is False
    assert len(universe["sha256"]) == 64

    with pytest.raises(ValueError, match="clone attachment"):
        stacked_spine_module._run_stacked_puf_pass_without_tail_for_test(
            result.frame,
            donor,
            clone_attachment_fraction=0.5,
            clone_attachment_seed=578,
        )


def test_run_stacked_puf_pass_receipts_raw_child_universe_application() -> None:
    gap_filled = _gap_fill_with_test_authority(
        _stacked_gap_fixture(),
        plan=_GAP_FILL_TEST_PLAN,
        seed=578,
        n_estimators=10,
    ).frame
    person = gap_filled.table("person")
    acs = person[support_channel_column("person")].eq("acs")
    groups = list(
        person.loc[acs].groupby("person_tax_unit_id", sort=False).groups.values()
    )
    mixed_group = list(next(group for group in groups if len(group) > 1))
    all_child_group = list(next(group for group in groups if len(group) == 1))
    structural_rows = [mixed_group[0], *all_child_group]
    person.loc[structural_rows, "age"] = 12.0
    for column in (
        "employment_income_before_lsr",
        "self_employment_income_before_lsr",
        "WAGP",
        "SEMP",
    ):
        person.loc[structural_rows, column] = np.nan
    donor = pd.DataFrame(
        {
            "employment_income": [45_000.0, 8_000.0, 70_000.0, 22_000.0],
            "self_employment_income": [1_000.0, 0.0, 5_000.0, 200.0],
            "taxable_interest_income": [120.0, 30.0, 900.0, 0.0],
            "weight": [1.0, 1.0, 1.0, 1.0],
        }
    )

    gap_filled = _late_primary_entry(gap_filled)
    result = stacked_spine_module._run_stacked_puf_pass_without_tail_for_test(
        gap_filled,
        donor,
        clone_attachment_fraction=1.0,
        clone_attachment_seed=578,
        predictors=(
            "puf_predictor_employment_income",
            "puf_predictor_self_employment_income",
        ),
        person_outputs=("taxable_interest_income",),
        tax_unit_outputs=(),
        seed=578,
        n_estimators=10,
    )

    receipt = result.receipt["acs_earnings_universe_application"]
    assert receipt["structurally_absent_person_rows"] == 2
    assert receipt["affected_tax_unit_rows"] == 2
    assert receipt["mixed_universe_tax_unit_rows"] == 1
    assert receipt["empty_universe_tax_unit_rows"] == 1
    assert receipt["mapped_universe_zero_cells"] == 4
    assert receipt["raw_pums_source_cells_mutated"] is False
    assert receipt["mapped_person_cells_materialized"] is True
    output_person = result.frame.table("person")
    output_child = output_person[support_channel_column("person")].eq(
        "acs"
    ) & output_person["age"].lt(15)
    assert int(output_child.sum()) == 4
    assert (
        output_person.loc[
            output_child,
            ["employment_income_before_lsr", "self_employment_income_before_lsr"],
        ]
        .eq(0.0)
        .all()
        .all()
    )
    assert output_person.loc[output_child, ["WAGP", "SEMP"]].isna().all().all()


def test_run_stacked_puf_pass_fraction_one_receipts_out_of_frame_identity() -> None:
    gap_filled = _gap_fill_with_test_authority(
        _stacked_gap_fixture(),
        plan=_GAP_FILL_TEST_PLAN,
        seed=578,
        n_estimators=10,
    ).frame
    gap_filled = _late_primary_entry(gap_filled)
    donor = pd.DataFrame(
        {
            "employment_income": [45_000.0, 8_000.0, 70_000.0, 22_000.0],
            "taxable_interest_income": [120.0, 30.0, 900.0, 0.0],
            "weight": [1.0, 1.0, 1.0, 1.0],
        }
    )
    result = stacked_spine_module._run_stacked_puf_pass_without_tail_for_test(
        gap_filled,
        donor,
        clone_attachment_fraction=1.0,
        clone_attachment_seed=0,
        predictors=("puf_predictor_employment_income",),
        person_outputs=("taxable_interest_income",),
        tax_unit_outputs=(),
        seed=578,
        n_estimators=10,
    )

    assert PUF_CLONE_ATTACHMENT_MANIFEST_KEY not in result.frame.metadata
    attachment = result.receipt["clone_attachment"]
    assert attachment["authority_form"] == "full_clone_identity_no_manifest"
    assert attachment["clone_attachment_fraction"] == 1.0
    assert attachment["clone_attachment_seed"] == 0
    assert attachment["eligible_household_count"] == 14
    assert attachment["realized_household_count"] == 14


def test_run_stacked_puf_pass_applies_clone_two_capital_gains_tail() -> None:
    gap_filled = _gap_fill_with_test_authority(
        _stacked_gap_fixture(),
        plan=_GAP_FILL_TEST_PLAN,
        seed=578,
        n_estimators=10,
    ).frame
    tables = {entity: gap_filled.table(entity) for entity in gap_filled.entities}
    person = tables["person"].copy()
    for column in PUF_CAPITAL_GAINS_TAIL_PERSON_COLUMNS:
        person[column] = 0.0
    tables["person"] = person
    tax_unit = tables["tax_unit"].copy()
    tax_unit["filing_status_input"] = "SINGLE"
    for column in PUF_CAPITAL_GAINS_TAIL_TAX_UNIT_COLUMNS:
        tax_unit[column] = 0.0
    tables["tax_unit"] = tax_unit
    gap_filled = Frame(
        tables,
        gap_filled.schema,
        {
            entity: gap_filled.weights_for(entity)
            for entity in gap_filled.weighted_entities
        },
        gap_filled.strata,
        mass_log=gap_filled.mass_log,
        metadata=gap_filled.metadata,
    )
    gap_filled = _late_primary_entry(gap_filled)
    donor = pd.DataFrame(
        {
            "tax_unit_id": [10, 20, 1_000_001],
            "employment_income": [45_000.0, 8_000.0, 70_000.0],
            "taxable_interest_income": [120.0, 30.0, 900.0],
            "health_savings_account_ald": [0.0, 500.0, 1_000.0],
            "weight": [996.0, 3.0, 1.0],
            "filing_status_code": [1.0, 1.0, 1.0],
            PUF_DONOR_SOURCE_ADJUSTED_GROSS_INCOME_COLUMN: [
                100_000.0,
                5_000_000.0,
                10_000_000.0,
            ],
            "short_term_capital_gains": [0.0, -10_000_000_000.0, 5_000_000_000.0],
            "long_term_capital_gains_before_response": [
                100_000.0,
                100_000_000_000.0,
                75_000_000_000.0,
            ],
            "long_term_capital_gains_on_collectibles": [
                0.0,
                2_000_000_000.0,
                1_000_000_000.0,
            ],
            "non_sch_d_capital_gains": [0.0, 3_000_000_000.0, 0.0],
            "unrecaptured_section_1250_gain": [
                0.0,
                4_000_000_000.0,
                250_000_000.0,
            ],
        }
    )

    result = run_stacked_puf_pass(
        gap_filled,
        donor,
        clone_attachment_fraction=0.5,
        clone_attachment_seed=578,
        predictors=("puf_predictor_employment_income",),
        person_outputs=("taxable_interest_income",),
        tax_unit_outputs=("health_savings_account_ald",),
        seed=578,
        n_estimators=10,
    )

    assert result.receipt["tail_status"] == "applied"
    tail = result.receipt["puf_capital_gains_tail_transfer"]
    assert tail["record_count"] == 2
    for entity in result.frame.entities:
        assert 2 in set(
            result.frame.table(entity)[support_clone_index_column(entity)].astype(int)
        )

    attachment = validate_puf_clone_attachment(
        result.frame,
        boundary="stacked tail descendant fixture",
        expected_fraction=0.5,
        expected_seed=578,
    )
    assert attachment["version"] == 2
    assert (
        attachment["post_attachment_transform"]["tail_manifest_sha256"]
        == tail["manifest_sha256"]
    )
    tail_household = result.frame.table("household")
    tail_household_clone = tail_household[support_clone_index_column("household")].eq(2)
    live_source_channels = {
        str(channel): int(count)
        for channel, count in sorted(
            tail_household.loc[
                tail_household_clone,
                support_channel_column("household"),
            ]
            .astype(str)
            .value_counts()
            .items()
        )
    }
    assert tail["clone"]["support_role"] == "puf_tax_detail"
    assert tail["clone"]["source_channels"] == live_source_channels
    assert "support_channel" not in tail["clone"]
    assert tail["late_overlap_ownership"] == dict(us_late_overlap_ownership_receipt())

    preservation = stacked_spine_module.assert_stacked_tail_cells_preserved(
        result.frame,
        tail,
    )
    assert preservation["passed"] is True
    assert preservation["tail_owned_cell_count"] == 14
    assert (
        preservation["overlap_ownership_sha256"]
        == tail["late_overlap_ownership"]["sha256"]
    )

    forged_ownership = deepcopy(tail)
    forged_receipt = forged_ownership["late_overlap_ownership"]
    forged_receipt["ownership"][0]["final_owner"] = "forged_owner"
    receipt_payload = dict(forged_receipt)
    receipt_payload.pop("sha256")
    forged_receipt["sha256"] = stacked_spine_module._canonical_sha256(receipt_payload)
    manifest_payload = dict(forged_ownership)
    manifest_payload.pop("manifest_sha256")
    forged_ownership["manifest_sha256"] = stacked_spine_module._canonical_sha256(
        manifest_payload
    )
    with pytest.raises(ValueError, match="overlap ownership"):
        stacked_spine_module.assert_stacked_tail_cells_preserved(
            result.frame,
            forged_ownership,
        )

    terminal_gates = (
        stacked_completeness_gate(result.frame, tail_manifest=tail),
        by_origin_battery(result.frame, tail_manifest=tail),
    )
    for gate in terminal_gates:
        terminal_support = gate.details["puf_capital_gains_tail_support"]
        assert terminal_support["tail_manifest_sha256"] == tail["manifest_sha256"]
        assert terminal_support["recipient_support"] == tail["recipient_support"]

    tampered_details = deepcopy(terminal_gates[0].details)
    tampered_terminal = tampered_details["puf_capital_gains_tail_support"]
    tampered_support = tampered_terminal["recipient_support"]
    tampered_support["strata"][0]["observed_count"] += 1
    support_payload = dict(tampered_support)
    support_payload.pop("sha256")
    tampered_support["sha256"] = stacked_spine_module._canonical_sha256(support_payload)
    terminal_payload = dict(tampered_terminal)
    terminal_payload.pop("sha256")
    tampered_terminal["sha256"] = stacked_spine_module._canonical_sha256(
        terminal_payload
    )
    with pytest.raises(ValueError, match="recipient-support candidate count"):
        stacked_spine_module._validate_stacked_gate_manifest_details(
            terminal_gates[0].name,
            tampered_details,
            passed=terminal_gates[0].passed,
        )

    multi_person_record = next(
        record
        for record in tail["records"]
        if result.frame.table("person")["person_tax_unit_id"]
        .eq(int(record["tail_tax_unit_id"]))
        .sum()
        > 1
    )
    person = result.frame.table("person").copy()
    noncarrier = person["person_tax_unit_id"].eq(
        int(multi_person_record["tail_tax_unit_id"])
    ) & ~person["person_id"].eq(int(multi_person_record["tail_person_id"]))
    person.loc[noncarrier, "short_term_capital_gains"] = 123_456_789.0
    noncarrier_tampered = Frame(
        {
            **{entity: result.frame.table(entity) for entity in result.frame.entities},
            "person": person,
        },
        result.frame.schema,
        {
            entity: result.frame.weights_for(entity)
            for entity in result.frame.weighted_entities
        },
        result.frame.strata,
        mass_log=result.frame.mass_log,
        metadata=result.frame.metadata,
    )
    with pytest.raises(
        ValueError,
        match="cell person.short_term_capital_gains",
    ):
        stacked_spine_module.assert_stacked_tail_cells_preserved(
            noncarrier_tampered,
            tail,
        )

    first_record = tail["records"][0]
    household = result.frame.table("household")
    weights = result.frame.weights_for("household")
    changed_weights = weights.values.copy()
    recipient_position = int(
        np.flatnonzero(
            household["household_id"]
            .eq(int(first_record["recipient_household_id"]))
            .to_numpy()
        )[0]
    )
    tail_position = int(
        np.flatnonzero(
            household["household_id"]
            .eq(int(first_record["tail_household_id"]))
            .to_numpy()
        )[0]
    )
    shift = min(0.5, changed_weights[recipient_position] / 2.0)
    changed_weights[recipient_position] -= shift
    changed_weights[tail_position] += shift
    weight_tampered = Frame(
        {entity: result.frame.table(entity) for entity in result.frame.entities},
        result.frame.schema,
        {"household": Weights(changed_weights, weights.kind)},
        result.frame.strata,
        mass_log=result.frame.mass_log,
        metadata=result.frame.metadata,
    )
    assert validate_puf_clone_attachment(
        weight_tampered,
        boundary="conservative tail-weight mutation",
        expected_fraction=0.5,
        expected_seed=578,
    )
    with pytest.raises(ValueError, match="clone-2 household weight"):
        stacked_spine_module.assert_stacked_tail_cells_preserved(
            weight_tampered,
            tail,
        )

    tax_unit = result.frame.table("tax_unit").copy()
    tail_tax_unit = tax_unit["tax_unit_id"].eq(int(first_record["tail_tax_unit_id"]))
    tax_unit.loc[
        tail_tax_unit,
        "puf_capital_gains_tail_transfer_weight",
    ] += 1.0
    provenance_tampered = Frame(
        {
            **{entity: result.frame.table(entity) for entity in result.frame.entities},
            "tax_unit": tax_unit,
        },
        result.frame.schema,
        {
            entity: result.frame.weights_for(entity)
            for entity in result.frame.weighted_entities
        },
        result.frame.strata,
        mass_log=result.frame.mass_log,
        metadata=result.frame.metadata,
    )
    with pytest.raises(ValueError, match="transfer-weight provenance"):
        stacked_spine_module.assert_stacked_tail_cells_preserved(
            provenance_tampered,
            tail,
        )

    prepared, receipt = stacked_spine_module.prepare_stacked_tail_derivation(
        result.frame
    )
    if "schedule_d_capital_gain_distributions" in prepared.table("person"):
        clone_two = prepared.table("person")[support_clone_index_column("person")].eq(2)
        assert (
            prepared.table("person")
            .loc[clone_two, "schedule_d_capital_gain_distributions"]
            .isna()
            .all()
        )
    assert receipt["cleared_rows"] == int(
        prepared.table("person")[support_clone_index_column("person")].eq(2).sum()
    )
    stacked_spine_module.assert_stacked_tail_cells_preserved(prepared, tail)


def test_tail_preservation_pairs_clones_by_assembly_unique_source_id() -> None:
    """Raw ASEC/ACS IDs may collide without confusing clone parentage."""

    def overlapping_source(stratum: str, income_shift: float) -> Frame:
        source = _source_frame(
            household_ids=list(range(11, 21)),
            weights=[100.0] * 10,
            stratum=stratum,
        )
        tables = {entity: source.table(entity).copy() for entity in source.entities}
        person = tables["person"]
        person["employment_income_before_lsr"] = np.linspace(
            10_000.0 + income_shift,
            100_000.0 + income_shift,
            len(person),
        )
        person["WAGP"] = person["employment_income_before_lsr"]
        person["self_employment_income_before_lsr"] = 0.0
        person["SEMP"] = 0.0
        for column in PUF_CAPITAL_GAINS_TAIL_PERSON_COLUMNS:
            person[column] = 0.0
        tax_unit = tables["tax_unit"]
        tax_unit["filing_status_input"] = "SINGLE"
        for column in PUF_CAPITAL_GAINS_TAIL_TAX_UNIT_COLUMNS:
            tax_unit[column] = 0.0
        household = tables["household"]
        household["TYPEHUGQ"] = 1
        household["tenure_type"] = "RENTED"
        return Frame(
            tables,
            US_SCHEMA,
            {"household": source.weights_for("household")},
            source.strata,
        )

    stacked = assemble_stacked_spine(
        overlapping_source("asec_2024", 0.0),
        overlapping_source("acs_2024_1yr", 500.0),
        sample_fraction=1.0,
        sample_seed=578,
    ).frame
    for entity in ("person", "tax_unit"):
        table = stacked.table(entity)
        assert table[spine_source_id_column(entity)].duplicated(keep=False).all()
        assert table[support_source_id_column(entity)].is_unique

    donor = pd.DataFrame(
        {
            "tax_unit_id": [10, 20, 1_000_001],
            "employment_income": [45_000.0, 8_000.0, 70_000.0],
            "health_savings_account_ald": [0.0, 500.0, 1_000.0],
            "weight": [996.0, 3.0, 1.0],
            "filing_status_code": [1.0, 1.0, 1.0],
            PUF_DONOR_SOURCE_ADJUSTED_GROSS_INCOME_COLUMN: [
                100_000.0,
                5_000_000.0,
                10_000_000.0,
            ],
            "short_term_capital_gains": [0.0, -10_000_000_000.0, 5_000_000_000.0],
            "long_term_capital_gains_before_response": [
                100_000.0,
                100_000_000_000.0,
                75_000_000_000.0,
            ],
            "long_term_capital_gains_on_collectibles": [
                0.0,
                2_000_000_000.0,
                1_000_000_000.0,
            ],
            "non_sch_d_capital_gains": [0.0, 3_000_000_000.0, 0.0],
            "unrecaptured_section_1250_gain": [
                0.0,
                4_000_000_000.0,
                250_000_000.0,
            ],
        }
    )
    result = run_stacked_puf_pass(
        _late_primary_entry(stacked),
        donor,
        clone_attachment_fraction=1.0,
        clone_attachment_seed=578,
        predictors=("puf_predictor_employment_income",),
        person_outputs=(),
        tax_unit_outputs=("health_savings_account_ald",),
        seed=578,
        n_estimators=2,
    )

    tail = result.receipt["puf_capital_gains_tail_transfer"]
    assert tail["record_count"] == 2
    assert (
        validate_puf_clone_attachment(
            result.frame,
            boundary="full stacked tail descendant",
            expected_fraction=1.0,
            expected_seed=578,
        )["version"]
        == 2
    )
    assert stacked_spine_module.assert_stacked_tail_cells_preserved(
        result.frame,
        tail,
    )["passed"]


def test_completeness_gate_passes_on_filled_and_proven_surface() -> None:
    gap_filled = _gap_fill_with_test_authority(
        _stacked_gap_fixture(),
        plan=_GAP_FILL_TEST_PLAN,
        seed=578,
        n_estimators=10,
    ).frame
    surface = {
        "person": {
            "model_required_numeric": ("unemployment_compensation",),
            "model_required_boolean": ("is_disabled",),
            "housing": ("pre_subsidy_rent",),
        }
    }
    result = _completeness_with_test_authority(
        gap_filled,
        declared_surface=surface,
        declared_gap_fill_plan=_GAP_FILL_TEST_PLAN,
    )
    assert result.passed
    assert result.details["declared_targets"] == 3
    statuses = {
        label: receipt["status"] for label, receipt in result.details["targets"].items()
    }
    assert set(statuses.values()) == {"complete", "proven_absent"}
    rent = result.details["targets"]["person/housing/pre_subsidy_rent"]
    assert rent["recipient_absence_authority"]["rows"] == 1
    assert rent["proven"]["acs/clone_0"]["structural_absence_rule_id"] == (
        "acs_native_group_quarters_without_housing_unit"
    )


def test_structural_rent_absence_covers_every_clone_role_and_battery_scope() -> None:
    gap_filled = _gap_fill_with_test_authority(
        _stacked_gap_fixture(),
        plan=_GAP_FILL_TEST_PLAN,
        seed=578,
        n_estimators=10,
    ).frame
    attached = clone_us_frame_for_puf_support(
        gap_filled,
        clone_attachment_fraction=1.0,
        clone_attachment_seed=578,
    )
    surface = {"person": {"housing": ("pre_subsidy_rent",)}}
    housing_plan = (_GAP_FILL_TEST_PLAN[1],)
    completeness = _completeness_with_test_authority(
        attached,
        declared_surface=surface,
        declared_gap_fill_plan=housing_plan,
    )
    assert completeness.passed, completeness.failures
    target = completeness.details["targets"]["person/housing/pre_subsidy_rent"]
    assert set(target["proven"]) == {"acs/clone_0", "acs/clone_1"}
    assert target["recipient_absence_authority"]["by_origin_role"] == {
        "acs/clone_0": 1,
        "acs/clone_1": 1,
    }

    authority = stacked_spine_module._make_test_stacked_authority(
        declared_surface=surface,
        gap_fill_plan=housing_plan,
        metric_registry={
            (
                "person",
                "housing",
                "pre_subsidy_rent",
                0,
            ): "monetary_sign_separated"
        },
    )
    battery = stacked_spine_module._by_origin_battery_with_test_authority(
        attached,
        authority=authority,
    )
    comparison = battery.details["comparisons"][
        "person/housing/pre_subsidy_rent[clone_0]"
    ]
    assert comparison["status"] != "null_in_scope"
    assert comparison["recipient_absence_authority"]["rows_excluded_from_scope"] == 1
    assert not any(
        "exact structural-absence" in failure for failure in battery.failures
    )


def test_structural_rent_absence_rejects_a_non_gq_recipient_null() -> None:
    gap_filled = _gap_fill_with_test_authority(
        _stacked_gap_fixture(),
        plan=_GAP_FILL_TEST_PLAN,
        seed=578,
        n_estimators=10,
    ).frame
    person = gap_filled.table("person").copy()
    channel = person[support_channel_column("person")].astype(str)
    household = gap_filled.table("household")
    gq_households = set(
        household.loc[
            pd.to_numeric(household["TYPEHUGQ"], errors="coerce").isin((2, 3)),
            "household_id",
        ]
    )
    non_gq = channel.eq("acs") & ~person["person_household_id"].isin(gq_households)
    person.loc[person.index[non_gq][0], "pre_subsidy_rent"] = np.nan
    tables = {entity: gap_filled.table(entity) for entity in gap_filled.entities}
    tables["person"] = person
    corrupted = Frame(
        tables,
        gap_filled.schema,
        {
            entity: gap_filled.weights_for(entity)
            for entity in gap_filled.weighted_entities
        },
        gap_filled.strata,
        mass_log=gap_filled.mass_log,
        metadata=gap_filled.metadata,
    )
    result = _completeness_with_test_authority(
        corrupted,
        declared_surface={"person": {"housing": ("pre_subsidy_rent",)}},
        declared_gap_fill_plan=(_GAP_FILL_TEST_PLAN[1],),
    )
    assert not result.passed
    assert any(
        "exact structural-absence equation failed" in failure
        and "unexpected_null_rows=1" in failure
        for failure in result.failures
    )


def test_completeness_gate_names_a_silently_missing_family() -> None:
    """The run-7 catcher: a declared family with no columns fails by name."""

    surface = {
        "person": {
            "puf_tax_itemization": (
                "taxable_interest_income",
                "qualified_dividend_income",
            ),
        }
    }
    stacked = _stacked_gap_fixture()
    tables = {entity: stacked.table(entity) for entity in stacked.entities}
    person = tables["person"].drop(columns=["taxable_interest_income"])
    tables["person"] = person
    dropped = Frame(
        tables,
        stacked.schema,
        {entity: stacked.weights_for(entity) for entity in stacked.weighted_entities},
        stacked.strata,
        mass_log=stacked.mass_log,
        metadata=stacked.metadata,
    )

    result = _completeness_with_test_authority(
        dropped,
        declared_surface=surface,
        declared_gap_fill_plan=_GAP_FILL_TEST_PLAN,
    )
    assert not result.passed
    missing = [
        failure
        for failure in result.failures
        if "puf_tax_itemization/taxable_interest_income" in failure
        and "missing" in failure
    ]
    assert missing
    unproven = [
        failure
        for failure in result.failures
        if "qualified_dividend_income" in failure and "authority proof" in failure
    ]
    assert unproven


def test_completeness_gate_requires_source_role_proofs_for_nulls() -> None:
    stacked = _stacked_gap_fixture()
    surface = {"person": {"puf_tax_itemization": ("taxable_interest_income",)}}

    unproven = _completeness_with_test_authority(
        stacked,
        declared_surface=surface,
        declared_gap_fill_plan=(),
    )
    assert not unproven.passed
    assert any(
        "acs/clone_0" in failure and "authority proof" in failure
        for failure in unproven.failures
    )

    proven = _completeness_with_test_authority(
        stacked,
        declared_surface=surface,
        declared_gap_fill_plan=(),
        absence_proofs=(
            AbsenceProof(
                entity="person",
                column="taxable_interest_income",
                channel="acs",
                clone_index=0,
                reason="pending asec_survey_to_acs gap-fill",
            ),
        ),
    )
    assert proven.passed
    receipt = proven.details["targets"][
        "person/puf_tax_itemization/taxable_interest_income"
    ]
    assert receipt["status"] == "proven_absent"
    assert receipt["proven"]["acs/clone_0"]["reason"] == (
        "pending asec_survey_to_acs gap-fill"
    )


def test_completeness_gate_rejects_wildcard_for_declared_donor_hole() -> None:
    stacked = _stacked_gap_fixture()
    surface = {"person": {"model_required_numeric": ("unemployment_compensation",)}}
    wildcard = AbsenceProof(
        entity="person",
        column="unemployment_compensation",
        channel="*",
        clone_index=0,
        reason="WILDCARD LAUNDER",
    )

    laundered = _completeness_with_test_authority(
        stacked,
        declared_surface=surface,
        declared_gap_fill_plan=_GAP_FILL_TEST_PLAN,
        absence_proofs=(wildcard,),
    )
    assert not laundered.passed
    assert any(
        "origin-exact authority proof" in failure
        and "asec_survey_to_acs" in failure
        and "donor 'asec'" in failure
        for failure in laundered.failures
    )

    exact = _completeness_with_test_authority(
        stacked,
        declared_surface=surface,
        declared_gap_fill_plan=_GAP_FILL_TEST_PLAN,
        absence_proofs=(replace(wildcard, channel="acs", reason="DECLARED EXACT"),),
    )
    assert exact.passed
    proof = exact.details["targets"][
        "person/model_required_numeric/unemployment_compensation"
    ]["proven"]["acs/clone_0"]
    assert proof["authority_form"] == "origin_exact_recipient"
    assert proof["declared_direction"] == "asec_survey_to_acs"
    assert proof["declared_donor_channel"] == "asec"


def test_completeness_gate_empty_plan_cannot_launder_canonical_target() -> None:
    surface = {"person": {"model_required_numeric": ("unemployment_compensation",)}}
    authority = stacked_spine_module._make_test_stacked_authority(
        declared_surface=surface,
        gap_fill_plan=(),
    )
    result = stacked_spine_module._stacked_completeness_gate_with_test_authority(
        _stacked_gap_fixture(),
        authority=authority,
        absence_proofs=(
            AbsenceProof(
                entity="person",
                column="unemployment_compensation",
                channel="*",
                clone_index=0,
                reason="EMPTY PLAN LAUNDER",
            ),
        ),
    )

    assert not result.passed
    assert any(
        "person/model_required_numeric/unemployment_compensation" in failure
        and "canonical gap-fill plan" in failure
        and "recipient 'acs'" in failure
        and "donor 'asec'" in failure
        and "wildcard authority is forbidden" in failure
        for failure in result.failures
    )


def test_completeness_gate_cannot_launder_post_puf_transfer_target() -> None:
    frame = _post_puf_transfer_fixture()
    surface = {"person": {"model_required_boolean": ("is_pregnant",)}}
    authority = stacked_spine_module._make_test_stacked_authority(
        declared_surface=surface,
        gap_fill_plan=(),
        post_puf_transfer_surface=surface,
    )
    result = stacked_spine_module._stacked_completeness_gate_with_test_authority(
        frame,
        authority=authority,
        absence_proofs=tuple(
            AbsenceProof(
                entity="person",
                column="is_pregnant",
                channel="*",
                clone_index=clone_index,
                reason="POST-PUF WILDCARD LAUNDER",
            )
            for clone_index in (0, 1)
        ),
    )

    assert not result.passed
    assert any(
        "person/model_required_boolean/is_pregnant" in failure
        and "zero-residual post-PUF transfer contract" in failure
        and "absence proofs are forbidden" in failure
        for failure in result.failures
    )


def test_completeness_gate_empty_surface_is_terminal() -> None:
    authority = stacked_spine_module._make_test_stacked_authority(
        declared_surface={},
        gap_fill_plan=(),
        metric_registry={},
    )
    result = stacked_spine_module._stacked_completeness_gate_with_test_authority(
        _stacked_gap_fixture(),
        authority=authority,
    )

    assert not result.passed
    assert result.details["declared_targets"] == 0
    assert any(
        "declared stacked surface contains zero targets" in failure
        for failure in result.failures
    )


def test_completeness_gate_rejects_origin_exact_proof_for_declared_donor() -> None:
    stacked = _stacked_gap_fixture()
    person = stacked.table("person").copy()
    channel = person[support_channel_column("person")].astype(str)
    donor_index = person.index[channel.eq("asec")][0]
    person.loc[donor_index, "unemployment_compensation"] = np.nan
    tables = {entity: stacked.table(entity) for entity in stacked.entities}
    tables["person"] = person
    donor_hole = Frame(
        tables,
        stacked.schema,
        {entity: stacked.weights_for(entity) for entity in stacked.weighted_entities},
        stacked.strata,
        mass_log=stacked.mass_log,
        metadata=stacked.metadata,
    )
    surface = {"person": {"model_required_numeric": ("unemployment_compensation",)}}
    authority = stacked_spine_module._make_test_stacked_authority(
        declared_surface=surface,
        gap_fill_plan=_GAP_FILL_TEST_PLAN,
    )
    result = stacked_spine_module._stacked_completeness_gate_with_test_authority(
        donor_hole,
        authority=authority,
        absence_proofs=(
            AbsenceProof(
                entity="person",
                column="unemployment_compensation",
                channel="asec",
                clone_index=0,
                reason="DONOR-ORIGIN LAUNDER",
            ),
            AbsenceProof(
                entity="person",
                column="unemployment_compensation",
                channel="acs",
                clone_index=0,
                reason="DECLARED RECIPIENT",
            ),
        ),
    )

    assert not result.passed
    assert any(
        "person/model_required_numeric/unemployment_compensation" in failure
        and "origin-exact authority proof is valid only for declared recipient 'acs'"
        in failure
        and "declared donor 'asec'" in failure
        for failure in result.failures
    )
    target = result.details["targets"][
        "person/model_required_numeric/unemployment_compensation"
    ]
    assert "asec/clone_0" not in target["proven"]


def test_completeness_receipts_bind_live_authority_per_target() -> None:
    frame = _battery_frame(
        {
            "taxable_interest_income": (
                np.asarray([100.0] * 8),
                np.asarray([100.0] * 11),
            )
        }
    )
    canonical = stacked_completeness_gate(frame)
    assert canonical.passed, canonical.failures
    assert canonical.details["declared_targets"] == 134
    authority = canonical.details["authority"]
    assert authority["authority_form"] == "CANONICAL"
    assert authority["canonical"] is True
    battery = by_origin_battery(frame)
    assert battery.passed, battery.failures
    assert battery.details["authority"] == authority
    canonical_manifest = GateReport((canonical, battery)).to_manifest()
    assert canonical_manifest["passed"] is True
    assert (
        stacked_spine_module._authority_receipt(
            stacked_spine_module._production_stacked_authority()
        )
        == authority
    )
    plan_sha256 = authority["components"]["gap_fill_plan"]["sha256"]
    post_puf_sha256 = authority["components"]["post_puf_transfer_surface"]["sha256"]
    surface_sha256 = authority["components"]["declared_surface"]["sha256"]
    for receipt in canonical.details["targets"].values():
        assert receipt["authority_form"] == "observed_complete"
        assert receipt["plan_sha256"] == plan_sha256
        assert receipt["post_puf_surface_sha256"] == post_puf_sha256
        assert receipt["surface_sha256"] == surface_sha256

    stacked = _stacked_gap_fixture()
    person = stacked.table("person").copy()
    person["test_unplanned_absence"] = np.nan
    tables = {entity: stacked.table(entity) for entity in stacked.entities}
    tables["person"] = person
    custom_frame = Frame(
        tables,
        stacked.schema,
        {entity: stacked.weights_for(entity) for entity in stacked.weighted_entities},
        stacked.strata,
        mass_log=stacked.mass_log,
        metadata=stacked.metadata,
    )
    custom_authority = stacked_spine_module._make_test_stacked_authority(
        declared_surface={"person": {"test_only": ("test_unplanned_absence",)}},
        gap_fill_plan=(),
    )
    custom = stacked_spine_module._stacked_completeness_gate_with_test_authority(
        custom_frame,
        authority=custom_authority,
        absence_proofs=(
            AbsenceProof(
                entity="person",
                column="test_unplanned_absence",
                channel="*",
                clone_index=0,
                reason="test-only unplanned surface",
            ),
        ),
    )
    assert custom.passed, custom.failures
    custom_top = custom.details["authority"]
    assert custom_top["authority_form"] == "NON-CANONICAL"
    custom_target = custom.details["targets"]["person/test_only/test_unplanned_absence"]
    assert custom_target["authority_form"] == "wildcard_no_declared_donor_plan"
    assert (
        custom_target["plan_sha256"]
        == custom_top["components"]["gap_fill_plan"]["sha256"]
    )
    assert (
        custom_target["surface_sha256"]
        == custom_top["components"]["declared_surface"]["sha256"]
    )
    with pytest.raises(
        ValueError,
        match="non-canonical stacked authority is forbidden",
    ):
        stacked_spine_module._validate_production_authority_receipt(
            custom_top,
            boundary="test production artifact",
        )
    with pytest.raises(ValueError, match="production manifest emission is forbidden"):
        GateReport((custom,)).to_manifest()
    custom.details["authority"]["production_manifest_permitted"] = True
    with pytest.raises(ValueError, match="production manifest emission is forbidden"):
        GateReport((custom,)).to_manifest()
    forged_receipt = deepcopy(custom_top)
    forged_receipt.update(
        {
            "authority_id": "us_stacked_spine_authority",
            "authority_form": "CANONICAL",
            "declared_authority_form": "CANONICAL",
            "canonical": True,
            "canonical_identity": True,
            "canonical_content": True,
            "integrity_valid": True,
            "digest_matches_declared": True,
            "production_manifest_permitted": True,
            "declared_sha256": forged_receipt["sha256"],
        }
    )
    for component in forged_receipt["components"].values():
        component["declared_sha256"] = component["sha256"]
        component["digest_matches_declared"] = True
    forged_result = replace(custom, details={"authority": forged_receipt})
    with pytest.raises(ValueError, match="production manifest emission is forbidden"):
        GateReport((forged_result,)).to_manifest()


def test_artifact_battery_uses_canonical_formulas_without_assembly_metadata() -> None:
    frame = _battery_frame(
        {
            "taxable_interest_income": (
                np.asarray([100.0] * 8),
                np.asarray([100.0] * 11),
            )
        }
    )
    canonical = by_origin_battery(frame)
    assert canonical.passed, canonical.failures
    metadata_free = Frame(
        {entity: frame.table(entity) for entity in frame.entities},
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
    )

    with pytest.raises(ValueError, match="assembly manifest"):
        by_origin_battery(metadata_free)

    evidence = by_origin_battery_artifact_evidence(metadata_free)

    assert evidence.passed == canonical.passed
    assert evidence.failures == canonical.failures
    assert evidence.details["authority"] == canonical.details["authority"]
    assert evidence.details["tolerances"] == canonical.details["tolerances"]
    canonical_comparisons = deepcopy(canonical.details["comparisons"])
    evidence_comparisons = deepcopy(evidence.details["comparisons"])
    artifact_absence_receipts = []
    for comparisons in (canonical_comparisons, evidence_comparisons):
        for comparison in comparisons.values():
            receipt = comparison.pop("recipient_absence_authority", None)
            if comparisons is evidence_comparisons and isinstance(receipt, Mapping):
                artifact_absence_receipts.append(receipt)
    assert evidence_comparisons == canonical_comparisons
    assert len(artifact_absence_receipts) == 1
    assert artifact_absence_receipts[0]["assembly_manifest_authenticated"] is False


def test_self_digested_partial_authority_cannot_forge_production_identity() -> None:
    surface = {"person": {"test_only": ("unemployment_compensation",)}}
    forged = stacked_spine_module._make_stacked_authority(
        authority_id="us_stacked_spine_authority",
        version=1,
        gap_fill_plan=(),
        post_puf_transfer_surface={},
        post_puf_puf_producer_surface={},
        post_puf_source_producer_surface={},
        declared_surface=surface,
        metric_registry={
            ("person", "test_only", "unemployment_compensation", 0): (
                "monetary_sign_separated"
            )
        },
        support_profile=(stacked_spine_module.CANONICAL_ORIGIN_BATTERY_SUPPORT_PROFILE),
        declared_form="CANONICAL",
    )
    result = stacked_spine_module._stacked_completeness_gate_evaluate(
        _stacked_gap_fixture(),
        authority=forged,
        production=True,
        absence_proofs=(),
    )

    assert not result.passed
    assert result.details["authority"]["canonical_identity"] is False
    assert result.details["authority"]["canonical_content"] is False
    assert any(
        "canonical stacked authority identity mismatch" in failure
        for failure in result.failures
    )
    with pytest.raises(ValueError, match="production manifest emission is forbidden"):
        GateReport((result,)).to_manifest()


@pytest.mark.parametrize("stale_version", (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11))
def test_self_consistent_stale_stacked_authority_versions_are_rejected(
    stale_version: int,
) -> None:
    canonical = stacked_spine_module._production_stacked_authority()
    stale = stacked_spine_module._make_stacked_authority(
        authority_id=canonical.authority_id,
        version=stale_version,
        gap_fill_plan=canonical.gap_fill_plan,
        post_puf_transfer_surface=canonical.post_puf_transfer_surface,
        post_puf_puf_producer_surface=canonical.post_puf_puf_producer_surface,
        post_puf_source_producer_surface=canonical.post_puf_source_producer_surface,
        declared_surface=canonical.declared_surface,
        metric_registry=canonical.metric_registry,
        joint_metric_registry=canonical.joint_metric_registry,
        support_profile=canonical.support_profile,
        post_transfer_calibration=canonical.post_transfer_calibration,
        declared_form="CANONICAL",
    )
    stale_receipt = stacked_spine_module._authority_receipt(stale)

    assert stacked_spine_module.stacked_spine_authority_receipt()["version"] == 12
    assert stale_receipt["version"] == stale_version
    assert stale_receipt["integrity_valid"] is True
    assert stale_receipt["digest_matches_declared"] is True
    assert stale_receipt["canonical_content"] is False
    with pytest.raises(
        ValueError,
        match="non-canonical stacked authority is forbidden",
    ):
        stacked_spine_module._validate_production_authority_receipt(
            stale_receipt,
            boundary=f"stale authority v{stale_version}",
        )


def test_stacked_authority_binds_import_validated_late_producer_schedule() -> None:
    receipt = stacked_spine_module.stacked_spine_authority_receipt()
    component = receipt["components"]["late_producer_schedule"]

    assert receipt["version"] == 12
    assert component["producer_count"] == 38
    assert component["schedule_sha256"] == (
        stacked_spine_module.CANONICAL_US_LATE_PRODUCER_SCHEDULE.sha256
    )
    assert component["identity"]["status"] == "derived_and_import_validated"
    assert component["digest_matches_declared"] is True


def test_stacked_authority_binds_post_transfer_calibration_policy() -> None:
    receipt = stacked_spine_module.stacked_spine_authority_receipt()
    component = receipt["components"]["post_transfer_calibration"]

    assert component["target_count"] == 9
    assert component["identity"] == (
        stacked_spine_module.post_transfer_calibration_policy_identity()
    )
    assert component["digest_matches_declared"] is True


def test_rebound_late_producer_schedule_invalidates_production_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    live = dict(stacked_spine_module.us_late_producer_schedule_receipt())
    live["schedule_sha256"] = "0" * 64
    monkeypatch.setattr(
        stacked_spine_module,
        "us_late_producer_schedule_receipt",
        lambda: live,
    )

    authority = stacked_spine_module._production_stacked_authority()
    receipt = stacked_spine_module._authority_receipt(authority)

    assert receipt["canonical"] is False
    assert (
        receipt["components"]["late_producer_schedule"]["digest_matches_declared"]
        is False
    )
    with pytest.raises(ValueError, match="non-canonical stacked authority"):
        stacked_spine_module._validate_production_authority_receipt(
            receipt,
            boundary="rebound late producer schedule",
        )


def test_rebound_post_transfer_calibration_invalidates_production_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    live = dict(stacked_spine_module.post_transfer_calibration_policy_identity())
    live["tampered"] = True
    monkeypatch.setattr(
        stacked_spine_module,
        "post_transfer_calibration_policy_identity",
        lambda: live,
    )

    authority = stacked_spine_module._production_stacked_authority()
    receipt = stacked_spine_module._authority_receipt(authority)

    assert receipt["canonical"] is False
    assert (
        receipt["components"]["post_transfer_calibration"]["digest_matches_declared"]
        is False
    )
    with pytest.raises(ValueError, match="non-canonical stacked authority"):
        stacked_spine_module._validate_production_authority_receipt(
            receipt,
            boundary="rebound post-transfer calibration",
        )


def test_rebound_live_calibration_registry_is_noncanonical_and_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical_policy = (
        post_transfer_calibration_runtime.post_transfer_calibration_policy_identity()
    )
    monkeypatch.setattr(
        post_transfer_calibration_runtime,
        "POST_TRANSFER_CALIBRATION_SPECS",
        {},
    )

    live_policy = (
        post_transfer_calibration_runtime.post_transfer_calibration_policy_identity()
    )
    assert live_policy["targets"] == []
    assert live_policy["sha256"] != canonical_policy["sha256"]

    authority = stacked_spine_module._production_stacked_authority()
    receipt = stacked_spine_module._authority_receipt(authority)
    assert receipt["canonical"] is False
    assert receipt["production_manifest_permitted"] is False
    assert receipt["components"]["post_transfer_calibration"]["target_count"] == 0
    with pytest.raises(ValueError, match="non-canonical stacked authority"):
        stacked_spine_module._validate_production_authority_receipt(
            receipt,
            boundary="rebound empty post-transfer calibration registry",
        )


def test_rebound_anchor_aliases_cannot_replace_captured_canonical_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    surface = {"person": {"test_only": ("unemployment_compensation",)}}
    forged = stacked_spine_module._make_stacked_authority(
        authority_id="us_stacked_spine_authority",
        version=1,
        gap_fill_plan=(),
        post_puf_transfer_surface={},
        post_puf_puf_producer_surface={},
        post_puf_source_producer_surface={},
        declared_surface=surface,
        metric_registry={
            ("person", "test_only", "unemployment_compensation", 0): (
                "monetary_sign_separated"
            )
        },
        support_profile=(stacked_spine_module.CANONICAL_ORIGIN_BATTERY_SUPPORT_PROFILE),
        declared_form="CANONICAL",
    )
    rebound = {
        "_CANONICAL_STACKED_DECLARED_SURFACE_ANCHOR": forged.declared_surface,
        "_CANONICAL_STACKED_GAP_FILL_PLAN_ANCHOR": forged.gap_fill_plan,
        "_CANONICAL_STACKED_POST_PUF_TRANSFER_SURFACE_ANCHOR": (
            forged.post_puf_transfer_surface
        ),
        "_CANONICAL_ORIGIN_BATTERY_METRIC_REGISTRY_ANCHOR": forged.metric_registry,
        "_CANONICAL_ORIGIN_BATTERY_SUPPORT_PROFILE_ANCHOR": (forged.support_profile),
        "_CANONICAL_STACKED_AUTHORITY_ANCHOR": forged,
        "_STACKED_DECLARED_SURFACE": forged.declared_surface,
        "_STACKED_GAP_FILL_PLAN": forged.gap_fill_plan,
        "_STACKED_POST_PUF_TRANSFER_SURFACE": forged.post_puf_transfer_surface,
        "_BATTERY_METRIC_REGISTRY": forged.metric_registry,
        "_BATTERY_SUPPORT_PROFILE": forged.support_profile,
    }
    for name, value in rebound.items():
        monkeypatch.setattr(stacked_spine_module, name, value)

    result = stacked_completeness_gate(_stacked_gap_fixture())

    assert not result.passed
    assert result.details["authority"]["canonical"] is False
    assert result.details["authority"]["canonical_identity"] is False
    assert any(
        "canonical stacked authority identity mismatch" in failure
        for failure in result.failures
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "support_threshold",
        "surface_count",
        "direction_count",
        "post_puf_target_count",
        "target_binding",
    ),
)
def test_stacked_manifest_rejects_pre_emission_nested_receipt_mutation(
    mutation: str,
) -> None:
    frame = _battery_frame(
        {
            "taxable_interest_income": (
                np.asarray([100.0] * 8),
                np.asarray([100.0] * 11),
            )
        }
    )
    result = stacked_completeness_gate(frame)
    assert result.passed, result.failures
    authority = result.details["authority"]
    if mutation == "support_threshold":
        authority["components"]["support_profile"]["min_effective_support"] = 50
    elif mutation == "surface_count":
        authority["components"]["declared_surface"]["target_count"] = 0
    elif mutation == "direction_count":
        authority["components"]["gap_fill_plan"]["direction_count"] = 0
    elif mutation == "post_puf_target_count":
        authority["components"]["post_puf_transfer_surface"]["target_count"] = 0
    else:
        target = next(iter(result.details["targets"].values()))
        target["authority_form"] = "wildcard_no_declared_donor_plan"
        target["plan_sha256"] = "0" * 64

    with pytest.raises(
        ValueError,
        match="details changed after evaluation.*manifest emission is forbidden",
    ):
        GateReport((result,)).to_manifest()


@pytest.mark.parametrize("replacement", ({}, None))
def test_stacked_manifest_requires_the_authority_receipt(
    replacement: object,
) -> None:
    frame = _battery_frame(
        {
            "taxable_interest_income": (
                np.asarray([100.0] * 8),
                np.asarray([100.0] * 11),
            )
        }
    )
    canonical = stacked_completeness_gate(frame)
    details = deepcopy(dict(canonical.details))
    if replacement is None:
        details["authority"] = None
    else:
        details.pop("authority")
    missing = replace(canonical, details=details)

    with pytest.raises(
        ValueError,
        match="no stacked authority receipt.*manifest emission is forbidden",
    ):
        GateReport((missing,)).to_manifest()


def test_fresh_gate_result_cannot_graft_canonical_authority_onto_test_surface() -> None:
    frame = _stacked_gap_fixture()
    person = frame.table("person").copy()
    person["test_unplanned_absence"] = np.nan
    tables = {entity: frame.table(entity) for entity in frame.entities}
    tables["person"] = person
    custom_frame = Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )
    test_authority = stacked_spine_module._make_test_stacked_authority(
        declared_surface={"person": {"test_only": ("test_unplanned_absence",)}},
        gap_fill_plan=(),
    )
    custom = stacked_spine_module._stacked_completeness_gate_with_test_authority(
        custom_frame,
        authority=test_authority,
        absence_proofs=(
            AbsenceProof(
                entity="person",
                column="test_unplanned_absence",
                channel="*",
                clone_index=0,
                reason="test-only wildcard",
            ),
        ),
    )
    canonical = stacked_completeness_gate(
        _battery_frame(
            {
                "taxable_interest_income": (
                    np.asarray([100.0] * 8),
                    np.asarray([100.0] * 11),
                )
            }
        )
    )
    grafted_details = deepcopy(dict(custom.details))
    grafted_details["authority"] = deepcopy(canonical.details["authority"])
    grafted = replace(custom, details=grafted_details)

    with pytest.raises(
        ValueError,
        match="must declare exactly 134 targets.*manifest emission is forbidden",
    ):
        GateReport((grafted,)).to_manifest()


def test_fresh_gate_result_cannot_forge_a_donor_origin_proof() -> None:
    frame = _battery_frame(
        {
            "taxable_interest_income": (
                np.asarray([100.0] * 8),
                np.asarray([100.0] * 11),
            )
        }
    )
    canonical = stacked_completeness_gate(frame)
    details = deepcopy(dict(canonical.details))
    authority = details["authority"]
    binding = {
        "authority_sha256": authority["sha256"],
        "plan_sha256": authority["components"]["gap_fill_plan"]["sha256"],
        "post_puf_surface_sha256": authority["components"]["post_puf_transfer_surface"][
            "sha256"
        ],
        "surface_sha256": authority["components"]["declared_surface"]["sha256"],
    }
    label = "person/puf_tax_itemization/taxable_interest_income"
    details["targets"][label] = {
        "status": "proven_absent",
        "null_rows": 1,
        "authority_form": "origin_exact_recipient",
        **binding,
        "proven": {
            "asec/clone_0": {
                "null_rows": 1,
                "reason": "DONOR-ORIGIN FORGERY",
                "authority_form": "origin_exact_recipient",
                **binding,
                "declared_direction": "asec_survey_to_acs",
                "declared_donor_channel": "asec",
                "declared_recipient_channel": "acs",
            }
        },
        "unproven": {},
    }
    forged = replace(canonical, details=details)

    with pytest.raises(
        ValueError,
        match="asec/clone_0 proof is not recipient-exact.*emission is forbidden",
    ):
        GateReport((forged,)).to_manifest()


def test_fresh_gate_result_cannot_forge_structural_rent_absence() -> None:
    frame = _battery_frame(
        {
            "taxable_interest_income": (
                np.asarray([100.0] * 8),
                np.asarray([100.0] * 11),
            )
        }
    )
    canonical = stacked_completeness_gate(frame)
    assert canonical.passed, canonical.failures
    details = deepcopy(dict(canonical.details))
    rent = details["targets"]["person/housing/pre_subsidy_rent"]
    rent["recipient_absence_authority"]["reason"] = "FORGED STRUCTURAL ABSENCE"
    forged = replace(canonical, details=details)

    with pytest.raises(
        ValueError,
        match="structural-absence doctrine mismatch.*emission is forbidden",
    ):
        GateReport((forged,)).to_manifest()


@pytest.mark.parametrize(
    "evaluate",
    (stacked_completeness_gate, by_origin_battery),
    ids=("completeness", "battery"),
)
def test_canonical_structural_receipt_cannot_be_grafted_between_evaluations(
    evaluate: object,
) -> None:
    columns = {
        "taxable_interest_income": (
            np.asarray([100.0] * 8),
            np.asarray([100.0] * 11),
        )
    }
    no_gq = evaluate(_battery_frame(columns))
    one_gq = evaluate(_battery_frame(columns, acs_group_quarters=True))
    assert no_gq.passed, no_gq.failures
    assert one_gq.passed, one_gq.failures

    # Every field in the replacement receipt is canonical: it came from a
    # real evaluation of the same gate and authority.  It still cannot stand
    # in for the evaluator's evidence about another frame.
    with pytest.raises(TypeError, match="init=False"):
        replace(
            no_gq,
            details=deepcopy(dict(one_gq.details)),
            _stacked_authority_seal=b"caller-recomputed-seal",
        )
    forged = replace(no_gq, details=deepcopy(dict(one_gq.details)))
    with pytest.raises(
        ValueError,
        match="was not sealed by its evaluator.*emission is forbidden",
    ):
        GateReport((forged,)).to_manifest()


def test_fresh_battery_result_cannot_forge_canonical_coverage_receipts() -> None:
    frame = _battery_frame(
        {
            "taxable_interest_income": (
                np.asarray([100.0] * 8),
                np.asarray([100.0] * 11),
            )
        }
    )
    canonical = by_origin_battery(frame)
    details = deepcopy(dict(canonical.details))
    details["declared_target_count"] = 1
    details["registered_target_count"] = 1
    details["comparisons"] = {
        "person/test_only/fake[clone_0]": {
            "status": "tested",
            "metric": "rare_incidence",
        }
    }
    forged = replace(canonical, details=details)

    with pytest.raises(
        ValueError,
        match="coverage receipt must bind all 134 targets.*emission is forbidden",
    ):
        GateReport((forged,)).to_manifest()


def test_fresh_battery_result_requires_structural_scope_receipt() -> None:
    frame = _battery_frame(
        {
            "taxable_interest_income": (
                np.asarray([100.0] * 8),
                np.asarray([100.0] * 11),
            )
        }
    )
    canonical = by_origin_battery(frame)
    assert canonical.passed, canonical.failures
    details = deepcopy(dict(canonical.details))
    rent_label = "person/housing/pre_subsidy_rent[clone_0]"
    details["comparisons"][rent_label].pop("recipient_absence_authority")
    forged = replace(canonical, details=details)

    with pytest.raises(
        ValueError,
        match="must carry canonical recipient-absence authority.*emission is forbidden",
    ):
        GateReport((forged,)).to_manifest()


def test_fresh_battery_result_cannot_relabel_a_canonical_metric() -> None:
    frame = _battery_frame(
        {
            "taxable_interest_income": (
                np.asarray([100.0] * 8),
                np.asarray([100.0] * 11),
            )
        }
    )
    canonical = by_origin_battery(frame)
    details = deepcopy(dict(canonical.details))
    label = "person/puf_tax_itemization/taxable_interest_income[clone_0]"
    details["comparisons"][label]["metric"] = "rare_incidence"
    forged = replace(canonical, details=details)

    with pytest.raises(
        ValueError,
        match=(
            "taxable_interest_income.*canonical metric "
            "'monetary_sign_separated'.*emission is forbidden"
        ),
    ):
        GateReport((forged,)).to_manifest()


def test_noncanonical_stacked_receipt_cannot_escape_under_a_renamed_gate() -> None:
    authority = stacked_spine_module._make_test_stacked_authority(
        declared_surface={"person": {"test_only": ("unemployment_compensation",)}},
        gap_fill_plan=(),
    )
    custom = stacked_spine_module._stacked_completeness_gate_with_test_authority(
        _stacked_gap_fixture(),
        authority=authority,
    )
    renamed = replace(custom, name="renamed_stacked_completeness")

    with pytest.raises(
        ValueError,
        match="unrecognized gate name.*manifest emission is forbidden",
    ):
        GateReport((renamed,)).to_manifest()


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("authority_form", "NON-CANONICAL"),
        ("declared_authority_form", "NON-CANONICAL"),
        ("digest_matches_declared", False),
    ),
)
def test_stripped_noncanonical_receipt_cannot_escape_under_a_renamed_gate(
    field: str,
    value: object,
) -> None:
    stripped = GateResult(
        name="renamed_stacked_completeness",
        passed=True,
        details={"authority": {field: value}},
    )

    with pytest.raises(
        ValueError,
        match="unrecognized gate name.*manifest emission is forbidden",
    ):
        GateReport((stripped,)).to_manifest()


def test_stripped_nine_component_authority_cannot_escape_under_a_renamed_gate() -> None:
    authority = stacked_spine_module.stacked_spine_authority_receipt()
    components = deepcopy(dict(authority["components"]))
    assert set(components) == {
        "gap_fill_plan",
        "post_puf_transfer_surface",
        "declared_surface",
        "metric_registry",
        "joint_metric_registry",
        "support_profile",
        "puf_capital_gains_tail_support_contract",
        "late_producer_schedule",
        "post_transfer_calibration",
    }
    stripped = GateResult(
        name="renamed_stacked_battery",
        passed=True,
        details={"authority": {"components": components}},
    )

    with pytest.raises(
        ValueError,
        match="unrecognized gate name.*manifest emission is forbidden",
    ):
        GateReport((stripped,)).to_manifest()


def test_completeness_gate_wildcard_proof_covers_every_origin() -> None:
    attached = clone_us_frame_for_puf_support(
        _stacked_gap_fixture(),
        clone_attachment_fraction=1.0,
        clone_attachment_seed=0,
    )
    person = attached.table("person").copy()
    person["health_savings_account_ald_person_carrier"] = np.nan
    clone_mask = person[support_clone_index_column("person")].eq(1)
    person.loc[clone_mask, "health_savings_account_ald_person_carrier"] = 100.0
    tables = {entity: attached.table(entity) for entity in attached.entities}
    tables["person"] = person
    frame = Frame(
        tables,
        attached.schema,
        {entity: attached.weights_for(entity) for entity in attached.weighted_entities},
        attached.strata,
        mass_log=attached.mass_log,
        metadata=attached.metadata,
    )
    surface = {
        "person": {
            "puf_tax_itemization": ("health_savings_account_ald_person_carrier",)
        }
    }

    unproven = _completeness_with_test_authority(
        frame,
        declared_surface=surface,
        declared_gap_fill_plan=(),
    )
    assert not unproven.passed

    proven = _completeness_with_test_authority(
        frame,
        declared_surface=surface,
        declared_gap_fill_plan=(),
        absence_proofs=(
            AbsenceProof(
                entity="person",
                column="health_savings_account_ald_person_carrier",
                channel="*",
                clone_index=0,
                reason="base arm carries no PUF detail; engine default applies",
            ),
        ),
    )
    assert proven.passed
    wildcard_receipt = proven.details["targets"][
        "person/puf_tax_itemization/health_savings_account_ald_person_carrier"
    ]["proven"]["asec/clone_0"]
    assert wildcard_receipt["authority_form"] == "wildcard_no_declared_donor_plan"


@pytest.mark.parametrize("clone_role", (0, 1))
def test_completeness_rejects_nonfinite_values_on_every_non_tail_clone_role(
    clone_role: int,
) -> None:
    base = _battery_frame(
        {
            "taxable_interest_income": (
                np.asarray([100.0] * 8),
                np.asarray([100.0] * 11),
            )
        }
    )
    attached = clone_us_frame_for_puf_support(
        base,
        clone_attachment_fraction=1.0,
        clone_attachment_seed=578,
    )
    tables = {entity: attached.table(entity).copy() for entity in attached.entities}
    person = tables["person"]
    clone_column = support_clone_index_column("person")
    invalid = person[clone_column].eq(clone_role)
    person.loc[invalid, "taxable_interest_income"] = np.inf
    frame = Frame(
        tables,
        attached.schema,
        {entity: attached.weights_for(entity) for entity in attached.weighted_entities},
        attached.strata,
        mass_log=attached.mass_log,
        metadata=attached.metadata,
    )

    result = stacked_completeness_gate(frame)

    assert not result.passed
    label = "person/puf_tax_itemization/taxable_interest_income"
    receipt = result.details["targets"][label]
    assert receipt["status"] == "invalid_values"
    assert receipt["invalid_rows"] == int(invalid.sum())
    assert receipt["invalidity"] == {"non_finite_rows": int(invalid.sum())}
    assert set(receipt["invalid_by_origin_role"]) == {
        f"asec/clone_{clone_role}",
        f"acs/clone_{clone_role}",
    }
    assert any(
        label in failure
        and "monetary_sign_separated" in failure
        and "non_finite_rows" in failure
        for failure in result.failures
    )
    json.dumps(GateReport((result,)).to_manifest(), allow_nan=False)


def test_terminal_gate_requires_manifest_for_live_clone_two_rows() -> None:
    attached = clone_us_frame_for_puf_support(
        _battery_frame(
            {
                "taxable_interest_income": (
                    np.asarray([100.0] * 8),
                    np.asarray([100.0] * 11),
                )
            }
        ),
        clone_attachment_fraction=1.0,
        clone_attachment_seed=578,
    )
    tables = {entity: attached.table(entity).copy() for entity in attached.entities}
    for entity, table in tables.items():
        clone_column = support_clone_index_column(entity)
        table.loc[table[clone_column].eq(1), clone_column] = 2
    frame = Frame(
        tables,
        attached.schema,
        {entity: attached.weights_for(entity) for entity in attached.weighted_entities},
        attached.strata,
        mass_log=attached.mass_log,
        metadata=attached.metadata,
    )

    with pytest.raises(ValueError, match="clone-2 rows require the bound"):
        stacked_completeness_gate(frame)


@pytest.mark.parametrize(
    ("asec_values", "acs_values", "expected_invalid"),
    (
        (np.full(8, np.inf), np.full(11, np.inf), 19),
        (np.full(8, np.inf), np.full(11, 100.0), 8),
    ),
)
def test_battery_rejects_nonfinite_values_before_comparison(
    asec_values: np.ndarray,
    acs_values: np.ndarray,
    expected_invalid: int,
) -> None:
    frame = _battery_frame({"taxable_interest_income": (asec_values, acs_values)})

    result = by_origin_battery(frame)

    assert not result.passed
    label = "person/puf_tax_itemization/taxable_interest_income[clone_0]"
    receipt = result.details["comparisons"][label]
    assert receipt == {
        "status": "invalid_values",
        "metric": "monetary_sign_separated",
        "invalid_rows": expected_invalid,
        "invalidity": {"non_finite_rows": expected_invalid},
    }
    assert any(
        label in failure and "non_finite_rows" in failure for failure in result.failures
    )
    json.dumps(GateReport((result,)).to_manifest(), allow_nan=False)


def test_quantile_envelope_rejects_nonfinite_inputs_defensively() -> None:
    with pytest.raises(ValueError, match="require finite values"):
        stacked_spine_module._battery_quantile_envelope_distance(
            np.asarray([np.inf]),
            np.asarray([np.inf]),
        )


def test_battery_boolean_incidence_is_declared_not_dispatched() -> None:
    frame = _battery_frame(
        {
            "matched_flag": (
                np.asarray([1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0]),
                np.asarray([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]),
            ),
            "object_backed_flag": (
                np.asarray(
                    [True, True, True, False, False, False, False, False], dtype=object
                ),
                np.asarray([True] + [False] * 10, dtype=object),
            ),
        }
    )
    registry = (
        OriginBatterySpec(
            entity="person",
            family="model_required_boolean",
            column_metrics={
                "matched_flag": "boolean_incidence",
                "object_backed_flag": "boolean_incidence",
            },
        ),
    )
    result = _battery_with_test_authority(
        frame, registry=_complete_battery_registry(*registry)
    )

    assert not result.passed
    matched = result.details["comparisons"][
        "person/model_required_boolean/matched_flag[clone_0]"
    ]
    assert matched["status"] == "tested"
    assert not any("matched_flag" in failure for failure in result.failures)
    assert any(
        "object_backed_flag" in failure and "incidence ratio" in failure
        for failure in result.failures
    )


def test_battery_joint_immigration_tvd_preserves_dependence_guarantee() -> None:
    frame = _battery_frame(
        {
            "ssn_card_type": (
                np.asarray(["A"] * 4 + ["B"] * 4, dtype=object),
                np.asarray(["A"] * 4 + ["B"] * 4, dtype=object),
            ),
            "immigration_status_str": (
                np.asarray(["X"] * 4 + ["Y"] * 4, dtype=object),
                np.asarray(["Y"] * 4 + ["X"] * 4, dtype=object),
            ),
        }
    )

    result = by_origin_battery(frame)

    assert not result.passed
    for column in ("ssn_card_type", "immigration_status_str"):
        marginal = result.details["comparisons"][
            f"person/source_operator_immigration/{column}[clone_0]"
        ]
        assert marginal["total_variation_distance"] == 0.0
    joint_label = (
        "person/source_operator_immigration/"
        "joint[ssn_card_type,immigration_status_str][clone_0]"
    )
    assert result.details["comparisons"][joint_label]["total_variation_distance"] == 1.0
    assert any(joint_label in failure for failure in result.failures)


def test_battery_rejects_registry_omitting_declared_champva_before_comparisons() -> (
    None
):
    champva = "has_champva_health_coverage_at_interview"
    target = ("person", "model_required_boolean", champva, 0)
    registry = dict(stacked_spine_module.CANONICAL_ORIGIN_BATTERY_METRIC_REGISTRY)
    assert registry.pop(target) == "boolean_incidence"
    authority = stacked_spine_module._make_test_stacked_authority(
        metric_registry=registry,
    )
    result = stacked_spine_module._by_origin_battery_with_test_authority(
        _battery_frame({champva: (np.ones(8), np.zeros(11))}),
        authority=authority,
    )

    assert not result.passed
    assert result.details["tested_comparisons"] == 0
    label = f"person/model_required_boolean/{champva}[clone_0]"
    assert result.details["missing_declared_targets"] == [label]
    assert f"missing declared battery target {label}." in result.failures


def test_battery_taxable_interest_metric_cannot_be_relabelled_rare_incidence() -> None:
    frame = _battery_frame(
        {
            "taxable_interest_income": (
                np.arange(1.0, 9.0),
                np.arange(101.0, 112.0),
            )
        }
    )
    registry = dict(stacked_spine_module.CANONICAL_ORIGIN_BATTERY_METRIC_REGISTRY)
    target = ("person", "puf_tax_itemization", "taxable_interest_income", 0)
    registry[target] = "rare_incidence"
    authority = stacked_spine_module._make_test_stacked_authority(
        metric_registry=registry,
    )
    result = stacked_spine_module._by_origin_battery_with_test_authority(
        frame,
        authority=authority,
    )

    assert not result.passed
    assert result.details["tested_comparisons"] == 0
    metric_receipt = result.details["authority"]["components"]["metric_registry"]
    assert metric_receipt["target_count"] == 134
    assert any(
        "person/puf_tax_itemization/taxable_interest_income[clone_0]" in failure
        and "authoritative metric 'monetary_sign_separated'" in failure
        and "got 'rare_incidence'" in failure
        for failure in result.failures
    )


def test_gap_fill_plan_digest_binds_direction_and_channels() -> None:
    frame = _battery_frame(
        {
            "taxable_interest_income": (
                np.asarray([100.0] * 8),
                np.asarray([100.0] * 11),
            )
        }
    )
    canonical = stacked_completeness_gate(frame)
    canonical_sha256 = canonical.details["authority"]["components"]["gap_fill_plan"][
        "sha256"
    ]
    plan = stacked_spine_module.CANONICAL_STACKED_GAP_FILL_PLAN
    altered = (
        replace(
            plan[0],
            name="rerouted_gap_fill",
            recipient_channel="asec",
            donor_channel="acs",
        ),
        *plan[1:],
    )
    authority = stacked_spine_module._make_test_stacked_authority(
        gap_fill_plan=altered,
    )
    result = stacked_spine_module._stacked_completeness_gate_with_test_authority(
        frame,
        authority=authority,
    )

    assert not result.passed
    receipt = result.details["authority"]
    assert receipt["authority_form"] == "NON-CANONICAL"
    assert receipt["components"]["gap_fill_plan"]["sha256"] != canonical_sha256
    assert any("canonical gap-fill direction mismatch" in f for f in result.failures)


def test_battery_sign_separated_catches_the_one_sided_hole() -> None:
    """The run-7 signature: ample support, hollow ACS leg -> terminal."""

    frame = _battery_frame(
        {
            "sentinel_amount": (
                np.asarray([500.0, 0.0, 1_200.0, 0.0, 800.0, 0.0, 2_000.0, 650.0]),
                np.zeros(11),
            ),
        }
    )
    registry = (
        OriginBatterySpec(
            entity="person",
            family="puf_tax_itemization",
            column_metrics={"sentinel_amount": "monetary_sign_separated"},
        ),
    )
    result = _battery_with_test_authority(
        frame, registry=_complete_battery_registry(*registry)
    )

    assert not result.passed
    assert any(
        "sentinel_amount" in failure and "positive-leg incidence ratio" in failure
        for failure in result.failures
    )


def test_battery_sign_separated_passes_matching_legs() -> None:
    frame = _battery_frame(
        {
            "signed_amount": (
                np.asarray(
                    [
                        100.0,
                        -100.0,
                        200.0,
                        -200.0,
                        300.0,
                        -300.0,
                        400.0,
                        -400.0,
                        500.0,
                        -500.0,
                        600.0,
                        -600.0,
                    ]
                ),
                np.asarray(
                    [
                        105.0,
                        -105.0,
                        210.0,
                        -210.0,
                        315.0,
                        -315.0,
                        420.0,
                        -420.0,
                        525.0,
                        -525.0,
                        630.0,
                        -630.0,
                    ]
                ),
            ),
        }
    )
    registry = (
        OriginBatterySpec(
            entity="person",
            family="puf_tax_itemization",
            column_metrics={"signed_amount": "monetary_sign_separated"},
        ),
    )
    result = _battery_with_test_authority(
        frame, registry=_complete_battery_registry(*registry)
    )
    assert result.passed, result.failures
    record = result.details["comparisons"][
        "person/puf_tax_itemization/signed_amount[clone_0]"
    ]
    assert record["legs"]["positive"]["quantile_envelope_distance"] <= 0.25
    assert record["legs"]["negative"]["quantile_envelope_distance"] <= 0.25


def test_battery_support_awareness_and_dead_comparisons() -> None:
    frame = _battery_frame(
        {
            "rare_flag": (
                np.asarray([0.0] * 8),
                np.asarray([0.0] * 11),
            ),
        }
    )
    dead = _battery_with_test_authority(
        frame,
        registry=_complete_battery_registry(
            OriginBatterySpec(
                entity="person",
                family="take_up",
                column_metrics={"rare_flag": "rare_incidence"},
            ),
        ),
    )
    assert not dead.passed
    assert any("dead" in failure for failure in dead.failures)
    assert dead.details["support_profile"]["profile_id"] == (
        "us_stacked_origin_battery_support"
    )
    assert dead.details["support_profile"]["min_effective_support"] == 5
    assert len(dead.details["support_profile"]["sha256"]) == 64


def test_battery_rebound_support_profile_with_stale_digest_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _battery_frame(
        {
            "taxable_interest_income": (
                np.asarray([100.0] * 8),
                np.asarray([100.0] * 11),
            )
        }
    )
    rebound = replace(
        stacked_spine_module.CANONICAL_ORIGIN_BATTERY_SUPPORT_PROFILE,
        min_effective_support=50,
    )
    monkeypatch.setattr(stacked_spine_module, "_BATTERY_SUPPORT_PROFILE", rebound)
    result = by_origin_battery(frame)

    assert not result.passed
    assert result.details["tested_comparisons"] == 0
    assert any(
        "support profile live-content digest mismatch" in failure
        for failure in result.failures
    )
    profile = result.details["authority"]["components"]["support_profile"]
    assert profile["sha256"] == (
        "7ffd25d0bc4c7cca1a12b61171d8d433094a60fc56cf5a099564598841252af9"
    )
    assert profile["sha256"] != profile["declared_sha256"]


def test_battery_rejects_caller_controlled_support_threshold() -> None:
    with pytest.raises(TypeError, match="min_effective_support"):
        OriginBatterySpec(
            entity="person",
            family="take_up",
            column_metrics={"rare_flag": "rare_incidence"},
            min_effective_support=50,
        )


def test_battery_categorical_tvd_and_null_scope() -> None:
    frame = _battery_frame(
        {
            "category_field": (
                np.asarray(["A", "A", "A", "B", "B", "B", "A", "B"], dtype=object),
                np.asarray(
                    ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B", "A"],
                    dtype=object,
                ),
            ),
            "leaky_field": (
                np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
                np.asarray([1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1.0, 2.0]),
            ),
        }
    )
    result = _battery_with_test_authority(
        frame,
        registry=_complete_battery_registry(
            OriginBatterySpec(
                entity="person",
                family="model_required_discrete",
                column_metrics={
                    "category_field": "categorical_tvd",
                    "leaky_field": "monetary_sign_separated",
                },
            )
        ),
    )
    assert not result.passed
    assert any(
        "leaky_field" in failure and "null value(s)" in failure
        for failure in result.failures
    )
    category = result.details["comparisons"][
        "person/model_required_discrete/category_field[clone_0]"
    ]
    assert category["total_variation_distance"] <= 0.25
    assert not any("category_field" in failure for failure in result.failures)


def test_battery_spec_rejects_undeclared_metric_kinds() -> None:
    with pytest.raises(ValueError, match="unknown metric kind"):
        OriginBatterySpec(
            entity="person",
            family="take_up",
            column_metrics={"anything": "dtype_dispatch"},
        )


def test_pilot_configuration_is_the_ratified_ten_percent() -> None:
    assert STACKED_PILOT_ACS_SAMPLE_FRACTION == 0.10
    assert STACKED_PILOT_ACS_SAMPLE_SEED == 578


def test_end_to_end_stack_gap_fill_puf_pass_gates_and_battery(tmp_path) -> None:
    """The pilot pipeline end to end: the ACS tax-detail hole is closed.

    Run 7's failure signature was a hollow ACS income surface: the ACS spine
    carried ~zero taxable-interest incidence against ASEC's 45% because the
    PUF family silently skipped.  This walks the revised architecture at
    fixture scale — stack, banked cross-origin gap-fill with native ACS
    predictors, seeded clone attachment, one doctrine-mode PUF pass, the
    completeness gate, and the terminal by-origin battery — and proves the
    sentinel is healthy on ACS-origin rows.
    """

    stacked = assemble_stacked_spine(
        _asec_e2e_source(),
        _acs_e2e_source(),
        acs_sample_fraction=0.5,
        acs_sample_seed=578,
    ).frame

    banks = {
        direction.name: AcsTransferTargetBankStore(
            tmp_path / direction.name,
            identity={"lane": "stacked-e2e", "direction": direction.name},
        )
        for direction in _E2E_GAP_FILL_PLAN
    }
    gap_filled = _gap_fill_with_test_authority(
        stacked,
        plan=_E2E_GAP_FILL_PLAN,
        seed=578,
        n_estimators=12,
        target_banks=banks,
    )

    donor = pd.DataFrame(
        {
            "employment_income": 25_000.0 + 6_000.0 * np.arange(8),
            "taxable_interest_income": [
                1_300.0,
                0.0,
                1_500.0,
                1_800.0,
                0.0,
                2_100.0,
                1_650.0,
                1_950.0,
            ],
            "health_savings_account_ald": [
                500.0,
                0.0,
                750.0,
                1_000.0,
                250.0,
                0.0,
                800.0,
                600.0,
            ],
            "weight": [1.0] * 8,
        }
    )
    passed = stacked_spine_module._run_stacked_puf_pass_without_tail_for_test(
        _late_primary_entry(gap_filled.frame),
        donor,
        clone_attachment_fraction=0.5,
        clone_attachment_seed=578,
        predictors=(
            "puf_predictor_employment_income",
            "puf_predictor_taxable_interest_income",
        ),
        person_outputs=("taxable_interest_income",),
        tax_unit_outputs=("health_savings_account_ald",),
        seed=578,
        n_estimators=12,
    )

    # The audit's failure mode is now a named terminal error, not a silent
    # zero-fill: without gap-fill the strict doctrine refuses the PUF pass.
    with pytest.raises(ValueError, match="puf_predictor_taxable_interest_income"):
        stacked_spine_module._run_stacked_puf_pass_without_tail_for_test(
            _late_primary_entry(stacked),
            donor,
            clone_attachment_fraction=0.5,
            clone_attachment_seed=578,
            predictors=(
                "puf_predictor_employment_income",
                "puf_predictor_taxable_interest_income",
            ),
            person_outputs=("taxable_interest_income",),
            tax_unit_outputs=("health_savings_account_ald",),
            seed=578,
            n_estimators=12,
        )

    declared_surface = {
        "person": {
            "puf_tax_itemization": ("taxable_interest_income",),
            "model_required_numeric": ("unemployment_compensation",),
            "model_required_boolean": ("is_disabled",),
            "housing": ("pre_subsidy_rent",),
        },
        "tax_unit": {
            "puf_tax_itemization": ("health_savings_account_ald",),
        },
    }
    completeness = _completeness_with_test_authority(
        passed.frame,
        declared_surface=declared_surface,
        declared_gap_fill_plan=_E2E_GAP_FILL_PLAN,
    )
    assert completeness.passed, completeness.failures

    # Drop-a-family mutation: the gate names the vanished family.
    mutated_surface = {
        "person": {
            **declared_surface["person"],
            "puf_tax_itemization": (
                "taxable_interest_income",
                "salt_refund_income",
            ),
        },
        "tax_unit": declared_surface["tax_unit"],
    }
    mutated = _completeness_with_test_authority(
        passed.frame,
        declared_surface=mutated_surface,
        declared_gap_fill_plan=_E2E_GAP_FILL_PLAN,
    )
    assert not mutated.passed
    assert any(
        "salt_refund_income" in failure and "missing" in failure
        for failure in mutated.failures
    )

    registry = (
        OriginBatterySpec(
            entity="person",
            family="puf_tax_itemization",
            column_metrics={"taxable_interest_income": "monetary_sign_separated"},
        ),
        OriginBatterySpec(
            entity="person",
            family="model_required_numeric",
            column_metrics={"unemployment_compensation": "monetary_sign_separated"},
        ),
        OriginBatterySpec(
            entity="person",
            family="model_required_boolean",
            column_metrics={"is_disabled": "boolean_incidence"},
        ),
        OriginBatterySpec(
            entity="person",
            family="housing",
            column_metrics={"pre_subsidy_rent": "monetary_sign_separated"},
        ),
    )
    battery = _battery_with_test_authority(
        _with_declared_battery_defaults(
            passed.frame,
            preserve=frozenset(
                {
                    ("person", "taxable_interest_income"),
                    ("person", "unemployment_compensation"),
                    ("person", "is_disabled"),
                    ("person", "pre_subsidy_rent"),
                }
            ),
        ),
        registry=_complete_battery_registry(*registry),
    )
    assert battery.passed, battery.failures

    sentinel = battery.details["comparisons"][
        "person/puf_tax_itemization/taxable_interest_income[clone_0]"
    ]
    assert sentinel["status"] == "tested"
    positive = sentinel["legs"]["positive"]
    assert positive["acs_incidence"] > 0.0
    assert 0.8 <= positive["incidence_ratio_acs_over_asec"] <= 1.25
    detail = passed.frame.table("person")
    detail_clone = detail[support_clone_index_column("person")].eq(1)
    detail_channel = detail[support_channel_column("person")].astype(str)
    for origin in ("asec", "acs"):
        origin_detail = detail.loc[
            detail_clone & detail_channel.eq(origin),
            "taxable_interest_income",
        ]
        assert origin_detail.notna().all()
        assert (origin_detail > 0.0).any()
