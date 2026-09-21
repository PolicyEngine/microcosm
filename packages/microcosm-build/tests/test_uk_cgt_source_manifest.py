"""Contract tests for the UK capital gains stage manifest and family."""

from __future__ import annotations

from microcosm.build.country_spec import load_country_spec
from microcosm.build.uk_runtime.cgt_asset_type import (
    UK_CGT_ASSET_TYPE_MASS_CONSERVATION_REASON,
)
from microcosm.build.uk_runtime.cgt_imputation import (
    UK_CGT_IMPUTATION_SEED,
    UK_CGT_SPINE_MASS_CONSERVATION_REASON,
    UK_CGT_SPINE_STAGE_NAME,
    UK_CGT_TAXABLE_INCOME_PROXY_COMPONENTS,
)
from microcosm.build.uk_runtime.cgt_structure import (
    CGT_CLONE_MASS_CHANGE_REASON,
    CGT_DONOR_MASS_CHANGE_REASON,
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
from microcosm.build.uk_runtime.salary_sacrifice import (
    SALSAC_MASS_CHANGE_REASON,
)
from microcosm.build.uk_runtime.student_loans import (
    STUDENT_LOANS_MASS_CHANGE_REASON,
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


def test_stages_without_a_declared_receipt_require_none() -> None:
    """A column-writing stage that leaves the weights untouched records no
    mass receipt, so the contract demands none; the weight-kind check still
    covers it. The first release cut failed five families on a reason the
    generator had invented."""
    manifest = load_uk_release_input_coverage_manifest()
    for name in (
        "was_wealth",
        "lcfs_consumption",
        "etb_vat",
        "etb_services",
        "regional_property_uprating",
    ):
        family = manifest.family_coverage[name]
        assert "required_mass_change_reason" not in family, name
        assert family["mass_change_semantics"] == "weights_pass_through", name
        assert family["output_weight_kind"] == "importance", name


def test_the_shipped_family_contracts_pass_the_terminal_gate_shape() -> None:
    """The real manifest's family entries against a compliant final frame.

    The wiring first shipped a family whose declared weight kind and
    mass-change coupling would have failed every real build at the terminal
    gate — invisible because no test drove the shipped manifest through the
    family diagnostics. This drives exactly that path.
    """
    from types import SimpleNamespace

    from microcosm.build.uk_runtime.release_input_coverage import (
        _family_build_state_diagnostics,
    )
    from microcosm.frame import MassChangeRecord, WeightKind

    manifest = load_uk_release_input_coverage_manifest()
    spi_reason = str(
        manifest.family_coverage["hmrc_spi_income"]["required_mass_change_reason"]
    )
    compliant = SimpleNamespace(
        household_weight_kind=WeightKind.IMPORTANCE,
        time_period="2024",
        mass_log=(
            MassChangeRecord(
                entity="household",
                old_total=100.0,
                new_total=100.0,
                declared_factor=1.0,
                reason=spi_reason,
            ),
            MassChangeRecord(
                entity="household",
                old_total=100.0,
                new_total=100.0,
                declared_factor=1.0,
                reason=CGT_CLONE_MASS_CHANGE_REASON,
            ),
            MassChangeRecord(
                entity="household",
                old_total=100.0,
                new_total=110.0,
                declared_factor=None,
                reason=CGT_DONOR_MASS_CHANGE_REASON,
            ),
            MassChangeRecord(
                entity="household",
                old_total=100.0,
                new_total=100.0,
                declared_factor=1.0,
                reason=UK_CGT_SPINE_MASS_CONSERVATION_REASON,
            ),
            MassChangeRecord(
                entity="household",
                old_total=100.0,
                new_total=100.0,
                declared_factor=1.0,
                reason=UK_CGT_ASSET_TYPE_MASS_CONSERVATION_REASON,
            ),
            MassChangeRecord(
                entity="household",
                old_total=100.0,
                new_total=100.0,
                declared_factor=1.0,
                reason=SALSAC_MASS_CHANGE_REASON,
            ),
            MassChangeRecord(
                entity="household",
                old_total=100.0,
                new_total=100.0,
                declared_factor=1.0,
                reason=STUDENT_LOANS_MASS_CHANGE_REASON,
            ),
        ),
    )

    _, failures = _family_build_state_diagnostics(compliant, manifest)
    assert failures == []

    missing_receipt = SimpleNamespace(
        household_weight_kind=WeightKind.IMPORTANCE,
        time_period="2024",
        mass_log=compliant.mass_log[:1],
    )
    _, failures = _family_build_state_diagnostics(missing_receipt, manifest)
    assert any("hmrc_cgt_gains_spine" in failure for failure in failures)

    wrong_kind = SimpleNamespace(
        household_weight_kind=WeightKind.DESIGN,
        time_period="2024",
        mass_log=compliant.mass_log,
    )
    _, failures = _family_build_state_diagnostics(wrong_kind, manifest)
    assert any(
        "hmrc_cgt_gains_spine" in failure and "kind" in failure for failure in failures
    )
