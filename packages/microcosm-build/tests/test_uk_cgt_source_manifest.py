"""Contract tests for the UK capital gains stage manifest and family."""

from __future__ import annotations

from microcosm.build.country_spec import load_country_spec
from microcosm.build.uk_runtime.cgt_imputation import (
    UK_CGT_IMPUTATION_SEED,
    UK_CGT_SPINE_MASS_CONSERVATION_REASON,
    UK_CGT_SPINE_STAGE_NAME,
    UK_CGT_TAXABLE_INCOME_PROXY_COMPONENTS,
)
from microcosm.build.uk_runtime.hmrc_capital_gains import (
    HMRC_CGT_BUILD_PERIOD,
    HMRC_CGT_CONDITIONING_RECORD_SETS,
    HMRC_CGT_CONDITIONING_RESOURCE,
    HMRC_CGT_SOURCE_VINTAGE,
)
from microcosm.build.uk_runtime.release_input_coverage import (
    load_uk_release_input_coverage_manifest,
)


def _stage():
    """The spine's CGT amounts stage: since microcosm#823 the only one."""
    spec = load_country_spec("uk")
    assert spec.sources is not None
    return spec.sources.stage_map()[UK_CGT_SPINE_STAGE_NAME]


def _operations() -> dict[str, dict]:
    return {op.kind: dict(op.parameters) for op in _stage().operations}


def test_manifest_names_the_resource_the_code_reads() -> None:
    """One provenance, declared once: the manifest repeats the module's roster."""
    artifacts = {artifact["role"]: artifact for artifact in _stage().artifacts}
    assert "published_fact_surface" not in artifacts
    surface = artifacts["cgt_conditioning_facts"]
    verify = _operations()["verify_vendored_fact_resource"]

    assert surface["resource"] == HMRC_CGT_CONDITIONING_RESOURCE
    assert surface["runtime_sha256_required"] is True
    assert verify["artifact_role"] == "cgt_conditioning_facts"
    assert verify["resource"] == HMRC_CGT_CONDITIONING_RESOURCE
    assert verify["feed_pin"] == "chronicle_feed.json"
    assert verify["record_sets"] == list(HMRC_CGT_CONDITIONING_RECORD_SETS)
    assert verify["source_vintage"] == HMRC_CGT_SOURCE_VINTAGE
    assert verify["mapped_build_period"] == int(HMRC_CGT_BUILD_PERIOD)


def test_manifest_operations_match_the_stage_implementation() -> None:
    operations = _operations()

    assert _stage().stage == UK_CGT_SPINE_STAGE_NAME
    proxy = operations["taxable_income_proxy"]
    assert tuple(proxy["components"]) == UK_CGT_TAXABLE_INCOME_PROXY_COMPONENTS
    draws = operations["within_band_draws"]
    assert draws["seed_base"] == UK_CGT_IMPUTATION_SEED
    assert draws["deterministic"] is True
    verify = operations["verify_vendored_fact_resource"]
    assert verify["require_before_source_read"] is True
    assert verify["fail_on_mismatch"] is True


def test_band_facts_stay_fenced_from_calibration() -> None:
    fence = _operations()["classify_cgt_band_facts_with_reviewed_fence"]

    assert fence["calibration_permitted"] is False
    assert fence["fenced_fact_count"] == 76
    assert fence["fact_fence_id"]


def test_receipt_reason_matches_the_stage_constant() -> None:
    """The gate matches on the exact string, so one source of truth."""
    receipt = _operations()["record_mass_conservation_receipt"]

    assert receipt["reason"] == UK_CGT_SPINE_MASS_CONSERVATION_REASON
    assert receipt["declared_factor"] == 1.0


def test_family_coverage_carries_the_stage_as_required_at_build() -> None:
    manifest = load_uk_release_input_coverage_manifest()
    family = manifest.family_coverage[UK_CGT_SPINE_STAGE_NAME]

    assert family["status"] == "required_at_build"
    assert family["source_manifest"] == "source_stages.json"
    assert family["calibration_permitted"] is False
    assert family["outputs"] == ["capital_gains"]
    assert family["output_weight_kind"] == "importance"
    assert (
        family["required_mass_change_reason"] == UK_CGT_SPINE_MASS_CONSERVATION_REASON
    )
    assert UK_CGT_SPINE_STAGE_NAME in manifest.required_build_stages


def test_the_june_path_cgt_family_is_retired() -> None:
    """One CGT gains family, the spine's (microcosm#823)."""
    manifest = load_uk_release_input_coverage_manifest()
    assert "hmrc_cgt_gains" not in manifest.family_coverage
    assert "hmrc_cgt_gains" not in manifest.required_build_stages
    assert "hmrc_cgt_gains" not in load_country_spec("uk").sources.stage_map()


def test_every_source_stage_family_requires_its_declared_receipt() -> None:
    """A column-writing stage records a conservation receipt under its module
    constant and the contract requires exactly it: the positive "household
    mass unchanged" assertion, never an invented reason (review finding on
    the first release cut) and never no assertion at all."""
    from microcosm.build.uk_runtime.etb_services import (
        UK_ETB_SERVICES_MASS_CONSERVATION_REASON,
    )
    from microcosm.build.uk_runtime.etb_vat import UK_ETB_VAT_MASS_CONSERVATION_REASON
    from microcosm.build.uk_runtime.lcfs_consumption import (
        UK_LCFS_CONSUMPTION_MASS_CONSERVATION_REASON,
    )
    from microcosm.build.uk_runtime.regional_uprating import (
        UK_REGIONAL_PROPERTY_UPRATING_MASS_CONSERVATION_REASON,
    )
    from microcosm.build.uk_runtime.was_wealth import (
        UK_WAS_WEALTH_MASS_CONSERVATION_REASON,
    )

    manifest = load_uk_release_input_coverage_manifest()
    expected = {
        "was_wealth": UK_WAS_WEALTH_MASS_CONSERVATION_REASON,
        "lcfs_consumption": UK_LCFS_CONSUMPTION_MASS_CONSERVATION_REASON,
        "etb_vat": UK_ETB_VAT_MASS_CONSERVATION_REASON,
        "etb_services": UK_ETB_SERVICES_MASS_CONSERVATION_REASON,
        "regional_property_uprating": (
            UK_REGIONAL_PROPERTY_UPRATING_MASS_CONSERVATION_REASON
        ),
    }
    for name, reason in expected.items():
        family = manifest.family_coverage[name]
        assert family["required_mass_change_reason"] == reason, name
        assert family["mass_change_semantics"] == "mass_conserving", name
        assert family["output_weight_kind"] == "importance", name
    for name, family in manifest.family_coverage.items():
        assert str(family["required_mass_change_reason"]).strip(), name
