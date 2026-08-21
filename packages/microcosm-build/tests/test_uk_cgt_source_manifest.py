"""Contract tests for the UK capital gains source manifest."""

from __future__ import annotations

import json
from pathlib import Path

from microcosm.build.uk_runtime.cgt_imputation import (
    UK_CGT_IMPUTATION_SEED,
    UK_CGT_IMPUTATION_STAGE_NAME,
    UK_CGT_MASS_CONSERVATION_REASON,
    UK_CGT_TAXABLE_INCOME_PROXY_COMPONENTS,
)
from microcosm.build.uk_runtime.cgt_structure import (
    CGT_CLONE_MASS_CHANGE_REASON,
    CGT_DONOR_MASS_CHANGE_REASON,
)
from microcosm.build.uk_runtime.hmrc_capital_gains import (
    HMRC_CGT_JOINT_ODS_SHA256,
    HMRC_CGT_JOINT_ODS_SIZE_BYTES,
    HMRC_CGT_JOINT_ODS_URL,
    HMRC_CGT_JOINT_SHEET_NAMES,
)
from microcosm.build.uk_runtime.release_input_coverage import (
    load_uk_release_input_coverage_manifest,
)

_MANIFEST_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "microcosm"
    / "build"
    / "uk"
    / "cgt_source_stages.json"
)


def _stage() -> dict:
    payload = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    stages = payload["stages"]
    assert len(stages) == 1
    return stages[0]


def test_manifest_pins_the_artifact_the_code_pins() -> None:
    """One provenance, declared once: the manifest repeats the module's pin."""
    surface = {artifact["role"]: artifact for artifact in _stage()["artifacts"]}[
        "published_fact_surface"
    ]

    assert surface["locator"] == HMRC_CGT_JOINT_ODS_URL
    assert surface["sha256"] == HMRC_CGT_JOINT_ODS_SHA256
    assert surface["size_bytes"] == HMRC_CGT_JOINT_ODS_SIZE_BYTES
    assert surface["sheets"] == list(HMRC_CGT_JOINT_SHEET_NAMES.values())
    assert surface["runtime_sha256_required"] is True


def test_manifest_operations_match_the_stage_implementation() -> None:
    operations = {operation["kind"]: operation for operation in _stage()["operations"]}

    assert _stage()["stage"] == UK_CGT_IMPUTATION_STAGE_NAME
    proxy = operations["taxable_income_proxy"]
    assert tuple(proxy["components"]) == UK_CGT_TAXABLE_INCOME_PROXY_COMPONENTS
    draws = operations["within_band_draws"]
    assert draws["seed_base"] == UK_CGT_IMPUTATION_SEED
    assert draws["deterministic"] is True
    verify = operations["verify_pinned_cgt_ods"]
    assert verify["require_before_source_read"] is True


def test_band_facts_stay_fenced_from_calibration() -> None:
    fence = {operation["kind"]: operation for operation in _stage()["operations"]}[
        "classify_cgt_band_facts_with_reviewed_fence"
    ]

    assert fence["calibration_permitted"] is False
    assert fence["fenced_fact_count"] == 76
    assert fence["fact_fence_id"]


def test_receipt_reason_matches_the_stage_constant() -> None:
    """The gate matches on the exact string, so one source of truth."""
    receipt = {operation["kind"]: operation for operation in _stage()["operations"]}[
        "record_mass_conservation_receipt"
    ]

    assert receipt["reason"] == UK_CGT_MASS_CONSERVATION_REASON
    assert receipt["declared_factor"] == 1.0


def test_family_coverage_carries_the_stage_as_required_at_build() -> None:
    manifest = load_uk_release_input_coverage_manifest()
    family = manifest.family_coverage[UK_CGT_IMPUTATION_STAGE_NAME]

    assert family["status"] == "required_at_build"
    assert family["source_manifest"] == _MANIFEST_PATH.name
    assert family["calibration_permitted"] is False
    assert family["outputs"] == ["capital_gains"]
    assert family["output_weight_kind"] == "importance"
    assert family["required_mass_change_reason"] == UK_CGT_MASS_CONSERVATION_REASON
    assert UK_CGT_IMPUTATION_STAGE_NAME in manifest.required_build_stages


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
    e5_reason = str(
        manifest.family_coverage["was_wealth"]["required_mass_change_reason"]
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
                reason=UK_CGT_MASS_CONSERVATION_REASON,
            ),
            MassChangeRecord(
                entity="household",
                old_total=100.0,
                new_total=100.0,
                declared_factor=1.0,
                reason=e5_reason,
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
    assert any("hmrc_cgt_gains" in failure for failure in failures)

    wrong_kind = SimpleNamespace(
        household_weight_kind=WeightKind.DESIGN,
        time_period="2024",
        mass_log=compliant.mass_log,
    )
    _, failures = _family_build_state_diagnostics(wrong_kind, manifest)
    assert any(
        "hmrc_cgt_gains" in failure and "kind" in failure for failure in failures
    )
