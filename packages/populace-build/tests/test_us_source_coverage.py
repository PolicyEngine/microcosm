from populace.build.gates import source_coverage_gate
from populace.build.us.source_coverage import (
    ARCH_US_POVERTY_CONTRACT_COMMIT,
    US_POVERTY_NONFILER_SOURCE_COVERAGE,
    hard_target_package_aliases,
    source_gap_family_ids,
    validation_only_family_ids,
)


def test_us_poverty_source_coverage_snapshot_has_expected_roles() -> None:
    assert len(ARCH_US_POVERTY_CONTRACT_COMMIT) == 40
    assert "ssa-ssi-table-7b1-2024" in hard_target_package_aliases()
    assert "cms-aca-oep-state-level-2025" in hard_target_package_aliases()
    assert "census_cps_spm" in validation_only_family_ids()
    assert "usda_wic" in source_gap_family_ids()


def test_us_poverty_source_coverage_gate_passes_when_hard_aliases_are_active() -> None:
    result = source_coverage_gate(
        US_POVERTY_NONFILER_SOURCE_COVERAGE,
        active_target_aliases=hard_target_package_aliases(),
    )
    assert result.passed
    assert result.details["source_gaps"]["hud_assisted_housing"]


def test_us_poverty_source_coverage_gate_blocks_cps_spm_hard_target() -> None:
    result = source_coverage_gate(
        US_POVERTY_NONFILER_SOURCE_COVERAGE,
        active_target_aliases=hard_target_package_aliases(),
        active_target_families=("census_cps_spm",),
    )
    assert not result.passed
    assert any("census_cps_spm" in failure for failure in result.failures)
