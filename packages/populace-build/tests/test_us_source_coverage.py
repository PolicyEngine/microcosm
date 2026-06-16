import json

import pytest

from populace.build.gates import GateReport
from populace.build.us.source_coverage import (
    LEDGER_US_SOURCE_COVERAGE_CONTRACT_COMMIT,
    hard_target_package_aliases,
    source_gap_family_ids,
    us_source_coverage_diagnostics,
    us_source_coverage_gate,
    validation_only_family_ids,
    write_us_source_coverage_diagnostics,
)


def test_us_source_coverage_snapshot_has_expected_roles() -> None:
    assert len(LEDGER_US_SOURCE_COVERAGE_CONTRACT_COMMIT) == 40
    assert "ssa-ssi-table-7b1-2024" in hard_target_package_aliases()
    assert "cms-aca-oep-state-level-2025" in hard_target_package_aliases()
    assert "census_cps_spm" in validation_only_family_ids()
    assert "usda_wic" in source_gap_family_ids()


def test_us_source_coverage_gate_passes_when_hard_aliases_are_active() -> None:
    result = us_source_coverage_gate(
        active_target_aliases=hard_target_package_aliases(),
    )
    assert result.passed
    assert result.name == "us_source_coverage"
    assert result.details["source_gaps"]["hud_assisted_housing"]
    assert result.details["coverage_summary"]["hard_target"] == {
        "families": 10,
        "package_aliases": len(hard_target_package_aliases()),
        "covered_package_aliases": len(hard_target_package_aliases()),
        "missing_package_aliases": 0,
        "reviewed_excluded_package_aliases": 0,
    }


def test_us_source_coverage_gate_names_missing_hard_aliases() -> None:
    active_aliases = tuple(
        alias
        for alias in hard_target_package_aliases()
        if alias != "ssa-ssi-table-7b1-2024"
    )
    result = us_source_coverage_gate(
        active_target_aliases=active_aliases,
    )
    assert not result.passed
    assert result.details["missing_hard_targets"] == ["ssa-ssi-table-7b1-2024"]
    assert any(
        failure
        == (
            "social_security_ssi/ssa-ssi-table-7b1-2024: hard-target source alias "
            "is not active and has no reviewed exclusion."
        )
        for failure in result.failures
    )


def test_us_source_coverage_gate_allows_reviewed_hard_exclusion() -> None:
    active_aliases = tuple(
        alias
        for alias in hard_target_package_aliases()
        if alias != "ssa-ssi-table-7b1-2024"
    )
    result = us_source_coverage_gate(
        active_target_aliases=active_aliases,
        reviewed_exclusions={
            "ssa-ssi-table-7b1-2024": "SSI amount target is validation-only in this release"
        },
    )
    assert result.passed
    assert result.details["reviewed_exclusions"] == {
        "ssa-ssi-table-7b1-2024": "SSI amount target is validation-only in this release"
    }
    assert (
        result.details["coverage_summary"]["hard_target"][
            "reviewed_excluded_package_aliases"
        ]
        == 1
    )


def test_us_source_coverage_gate_requires_reviewed_exclusion_reasons() -> None:
    active_aliases = tuple(
        alias
        for alias in hard_target_package_aliases()
        if alias != "ssa-ssi-table-7b1-2024"
    )
    with pytest.raises(TypeError, match="mapping from alias to reason"):
        us_source_coverage_gate(
            active_target_aliases=active_aliases,
            reviewed_exclusions=("ssa-ssi-table-7b1-2024",),
        )


def test_us_source_coverage_gate_blocks_cps_spm_hard_target() -> None:
    result = us_source_coverage_gate(
        active_target_aliases=hard_target_package_aliases(),
        active_target_families=("census_cps_spm",),
    )
    assert not result.passed
    assert any("census_cps_spm" in failure for failure in result.failures)


def test_us_source_coverage_gate_blocks_validation_alias() -> None:
    result = us_source_coverage_gate(
        active_target_aliases=(
            *hard_target_package_aliases(),
            "census-acs-s2201-congressional-district-snap-2024",
        ),
    )
    assert not result.passed
    assert result.details["validation_only_activated"] == ["snap_local_proxy"]


def test_us_source_coverage_manifest_reports_source_gap_families() -> None:
    result = us_source_coverage_gate(
        active_target_aliases=hard_target_package_aliases(),
    )
    manifest = GateReport((result,)).to_manifest()
    gate = manifest["gates"]["us_source_coverage"]
    assert gate["passed"]
    assert gate["details"]["source_gap_families"]["usda_wic"] == {
        "label": "WIC participation and benefits",
        "missing_source_packages": ("USDA FNS WIC program data",),
    }


def test_us_source_coverage_diagnostics_artifact_contains_summary(
    tmp_path,
) -> None:
    payload = us_source_coverage_diagnostics(
        active_target_aliases=hard_target_package_aliases(),
    )
    path = write_us_source_coverage_diagnostics(
        payload,
        tmp_path / "us_source_coverage.json",
    )
    written = json.loads(path.read_text())

    assert written["classification"] == "release_gate"
    assert written["source_contract"] == {
        "name": "us_source_coverage",
        "ledger_commit": LEDGER_US_SOURCE_COVERAGE_CONTRACT_COMMIT,
    }
    assert written["gate"] == {
        "name": "us_source_coverage",
        "passed": True,
        "failures": [],
    }
    assert written["coverage_summary"]["source_gap"] == {
        "families": 6,
        "missing_source_packages": 11,
    }
    assert written["source_gap_families"]["usda_wic"] == {
        "label": "WIC participation and benefits",
        "missing_source_packages": ["USDA FNS WIC program data"],
    }
