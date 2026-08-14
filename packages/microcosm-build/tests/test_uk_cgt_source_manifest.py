"""Contract tests for the UK capital gains source manifest."""

from __future__ import annotations

import json
from pathlib import Path

from microcosm.build.uk_runtime.cgt_imputation import (
    UK_CGT_IMPUTATION_SEED,
    UK_CGT_IMPUTATION_STAGE_NAME,
    UK_CGT_TAXABLE_INCOME_PROXY_COMPONENTS,
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


def test_family_coverage_carries_the_stage_as_required_at_build() -> None:
    manifest = load_uk_release_input_coverage_manifest()
    family = manifest.family_coverage[UK_CGT_IMPUTATION_STAGE_NAME]

    assert family["status"] == "required_at_build"
    assert family["source_manifest"] == _MANIFEST_PATH.name
    assert family["calibration_permitted"] is False
    assert family["outputs"] == ["capital_gains"]
    assert UK_CGT_IMPUTATION_STAGE_NAME in manifest.required_build_stages
