"""Contract tests for the candidate-time UK local gate scope."""

from microcosm.build import load_country_spec
from microcosm.build.uk_runtime.calibration_run import (
    UK_LOCAL_GATE_SCOPE,
    uk_local_gate_scope_exclusions,
    uk_scoped_gate_manifest,
)


def test_local_scope_is_the_declared_six_gate_terminal_battery() -> None:
    assert UK_LOCAL_GATE_SCOPE == (
        "uk_local_geography_ladder_post_calibration",
        "uk_local_area_support",
        "uk_local_target_fit",
        "uk_local_per_family_fit",
        "uk_local_weight_ratio",
        "uk_local_weight_ess",
    )
    manifest = uk_scoped_gate_manifest(
        UK_LOCAL_GATE_SCOPE,
        phases=("terminal",),
        policy_suffix="local_candidate",
    )
    assert tuple(entry.id for entry in manifest.gates) == UK_LOCAL_GATE_SCOPE
    assert {entry.phase for entry in manifest.gates} == {"terminal"}
    assert {
        entry.id for entry in manifest.gates if entry.criticality == "diagnostic"
    } == {
        "uk_local_target_fit",
        "uk_local_per_family_fit",
        "uk_local_weight_ratio",
        "uk_local_weight_ess",
    }


def test_local_area_support_parameters_pin_both_grains_and_all_floors() -> None:
    entries = {entry.id: entry for entry in load_country_spec("uk").gates.gates}
    area_support = entries["uk_local_area_support"]
    assert area_support.gate == "area_support"
    assert dict(area_support.parameters) == {
        "crosswalk_resource": "local_area_crosswalk.json",
        "geography_levels": ("constituency", "local_authority"),
        "minimum_rows": 50,
        "minimum_effective_sample_size": 50.0,
        "minimum_distinct_sources": 50,
    }


def test_local_scope_exclusions_classify_every_other_uk_gate() -> None:
    declared = {entry.id for entry in load_country_spec("uk").gates.gates}
    exclusions = uk_local_gate_scope_exclusions()
    assert set(exclusions) | set(UK_LOCAL_GATE_SCOPE) == declared
    assert not set(exclusions) & set(UK_LOCAL_GATE_SCOPE)
    assert all(exclusions.values())
