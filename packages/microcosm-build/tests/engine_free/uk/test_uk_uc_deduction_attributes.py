"""Tests split from packages/microcosm-build/tests/test_uk_uc_deduction_attributes.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.uk_uc_deduction_attributes import *


def test_resource_is_well_formed_and_carries_pinned_provenance() -> None:
    resource = json.loads(RESOURCE.read_text(encoding="utf-8"))

    validate_uc_deduction_resource(resource)
    assert resource["source"]["ods_sha256"] == (
        "307ec8fa49a1f1e23db3c59f1282e16609de03444e501824e204b4151e2e5c9b"
    )
    assert resource["source"]["chronicle_package_id"] == (
        "dwp-uc-deductions-march-2025-february-2026"
    )
    assert resource["source"]["chronicle_candidate"] is True
    assert (
        tuple(row["name"] for row in resource["latent_rate_distribution"]["bands"])
        == UC_DEDUCTION_BANDS
    )
    assert (
        tuple(row["name"] for row in resource["type_combination"]["shares"])
        == UC_DEDUCTION_COMBINATIONS
    )
    assert set(resource["region_incidence_factor"]["factors"]) == (UC_DEDUCTION_REGIONS)


def test_mapping_matches_engine_formula_on_synthetic_draws() -> None:
    resource = load_uc_deduction_distributions()
    draws = np.asarray([0.0235, 0.05, 0.1, 0.15, 0.5], dtype=np.float32).astype(
        np.float64
    )
    regions = np.asarray(["UNKNOWN"] * len(draws))
    rates = map_uniform_to_banded_rate(draws, regions, resource)
    combinations = map_uniform_to_categorical(
        np.asarray([0.0, 0.33, 0.5, 0.95, 0.2], dtype=np.float32),
        gate=rates > 0.0,
        resource=resource,
    )

    np.testing.assert_allclose(
        rates,
        np.asarray(
            [
                draws[0] / 0.047 * 0.05,
                0.05,
                0.05 + (draws[2] - 0.077) / 0.07 * 0.05,
                0.1,
                0.0,
            ]
        ),
        atol=1e-7,
    )
    assert combinations.tolist() == [
        "ADVANCE_ONLY",
        "THIRD_PARTY_ONLY",
        "GOVERNMENT_ONLY",
        "ALL_THREE",
        "NONE",
    ]


def test_stage_is_identity_keyed_and_float32_rounded_with_enum_names() -> None:
    resource = load_uc_deduction_distributions()
    original = _frame(512)
    transform = UKUCDeductionAttributesStageTransform(stage=_stage(), resource=resource)
    assigned = transform(original)
    expected_draw = stable_identity_uniforms(
        original.table("benunit")["benunit_id"],
        seed=0,
        salt="uc_deduction_random_draw",
    ).astype(np.float32)
    expected_draw = np.minimum(expected_draw, np.float32(FLOAT32_UNIFORM_MAX))
    np.testing.assert_array_equal(
        assigned.table("benunit")["uc_deduction_random_draw"],
        expected_draw.astype(np.float64),
    )
    assert assigned.table("benunit")["uc_deduction_combination"].map(type).eq(str).all()
    assert set(assigned.table("benunit")["uc_deduction_combination"]) <= {
        "NONE",
        *UC_DEDUCTION_COMBINATIONS,
    }

    ids = original.table("benunit")["benunit_id"].to_numpy()
    permuted_ids = ids[::-1]
    original_draws = _identity_float32_uniforms(ids, seed=0, salt="permutation")
    permuted_draws = _identity_float32_uniforms(
        permuted_ids, seed=0, salt="permutation"
    )
    assert dict(zip(ids, original_draws, strict=True)) == dict(
        zip(permuted_ids, permuted_draws, strict=True)
    )


def test_stage_refuses_operation_drift_and_incoherent_resource() -> None:
    stage_mapping = _stage_mapping()
    stage_mapping["operations"][2]["unexpected"] = True
    transform = UKUCDeductionAttributesStageTransform(
        stage=SourceStageSpec.from_mapping(stage_mapping),
        resource=load_uc_deduction_distributions(),
    )
    with pytest.raises(ValueError, match="declaration drifted"):
        transform(_frame(16))

    resource = copy.deepcopy(load_uc_deduction_distributions())
    resource["type_combination"]["shares"][0]["share"] = -1
    with pytest.raises(ValueError, match="finite and positive"):
        validate_uc_deduction_resource(resource)


def test_stage_receipt_passes_latent_attribute_health_gate() -> None:
    transform = UKUCDeductionAttributesStageTransform(
        stage=_stage(), resource=load_uc_deduction_distributions()
    )
    transform(_frame(8192, region_names=("UNKNOWN",)))
    evidence = transform.checkpoint_metadata()["evidence"]

    result = uk_stage_health_gate(
        evidence=evidence,
        stage="uc_deduction_attributes",
        check="latent_attribute_realization",
        parameters={},
    )

    assert result.passed, result.failures
    assert evidence["coherence_violation_count"] == 0


def test_persisted_draws_are_not_the_engine_fallback_and_stay_exported() -> None:
    """The engine reads the persisted draws; it does not recompute them.

    microcosm's draws are identity-keyed blake2b uniforms, the engine's fallback
    is splitmix64 of ``benunit_id``: they agree on essentially no row, which is
    fine only because the engine consumes the persisted columns. The design
    therefore rests on the two draw columns staying on the export surface, so
    that is pinned here alongside the divergence.
    """

    from microcosm.build.uk_runtime.terminal_gates import (
        UK_ALLOWED_EXTRA_EXPORT_COLUMNS,
    )

    ids = np.arange(1, 4097, dtype=np.int64)
    ours = _identity_float32_uniforms(ids, seed=0, salt="uc_deduction_random_draw")
    engine_fallback = _engine_splitmix64_uniform(ids, salt=0)
    agreement = np.mean(np.isclose(ours, engine_fallback, atol=1e-6))
    assert agreement < 1e-3, agreement

    for column in ("uc_deduction_random_draw", "uc_deduction_type_random_draw"):
        assert f"benunit.{column}" in UK_ALLOWED_EXTRA_EXPORT_COLUMNS
        assert column in _stage().outputs
