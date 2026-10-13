"""Tests split from packages/microcosm-build/tests/test_us_survivor_benefits_exclusion.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_survivor_benefits_exclusion import *


def test_exclusion_pins_exact_archived_person_source_and_qrf_dependency() -> None:
    entry = _entry()
    evidence = entry["evidence"]

    assert entry["reason"].startswith("SOURCE UNAVAILABILITY WITH EVIDENCE:")
    assert evidence["classification"] == "source_unavailability"
    assert evidence["retired_derivation"] == {
        "repository_owner": "PolicyEngine",
        "repository_name_parts": ["policyengine-", "us-data"],
        "commit": "42ed5d45c56df80d754fbe24cce21cfeb8d05cbe",
        "path_parts": [
            "policyengine_",
            "us_data",
            "datasets",
            "cps",
            "cps.py",
        ],
        "lines": "1495",
    }
    assert evidence["required_person_source_declaration"]["lines"] == ("39-58,306-359")
    assert evidence["puf_clone_qrf"]["lines"] == "140-194,639-745"
    assert evidence["puf_clone_qrf"]["target_line"] == 167
    assert evidence["no_independent_puf_source"] == {
        "commit": "42ed5d45c56df80d754fbe24cce21cfeb8d05cbe",
        "path_parts": [
            "policyengine_",
            "us_data",
            "calibration",
            "puf_impute.py",
        ],
        "tax_detail_target_lines": "90-149,158-198",
        "survivor_benefits_occurrences": 0,
    }
    assert evidence["required_person_columns"] == ["SRVS_VAL"]
    assert evidence["aggregate_only_columns"] == {
        "family": ["FSURVAL", "FINC_SUR"],
        "household": ["HSURVAL", "HSUR_YN"],
    }
    assert "synthesize" in evidence["semantic_rejection"]


def test_exclusion_pins_all_sha_locked_hermetic_inputs() -> None:
    evidence = _entry()["evidence"]
    evidence_hashes = {
        item["filename"]: item["sha256"] for item in evidence["hermetic_inputs"]
    }
    build_summary = json.loads(
        (ROOT / "experiments/build_j_recert/base_j.summary.json").read_text()
    )
    recorded_hashes = {
        Path(item["path"]).name: item["sha256"]
        for item in build_summary["base_source"]["sources"]
    }

    assert evidence_hashes == recorded_hashes
    assert set(evidence_hashes) == {
        "census_cps_2022.h5",
        "census_cps_2023.h5",
        "census_cps_2024.h5",
    }
    for item in evidence["hermetic_inputs"]:
        assert item["missing_person_columns"] == ["SRVS_VAL"]
        assert item["present_family_columns"] == ["FSURVAL", "FINC_SUR"]
        assert item["present_household_columns"] == ["HSURVAL", "HSUR_YN"]
        assert item["positive_multi_person_families"] > 0

    assert (
        "HSURVAL equals the household sum of FSURVAL" in evidence["aggregate_identity"]
    )

    build_script = (ROOT / "experiments/build_j_recert/buildj_base.sh").read_text()
    for year in (2022, 2023, 2024):
        assert f'--asec-h5 {year}="$USD/census_cps_{year}.h5"' in build_script
    assert "buildj_base.sh lines 65-69" in evidence["hermetic_build_contract"]
    assert "base_j.summary.json lines 55-75" in evidence["hermetic_build_contract"]


def test_generated_release_manifest_preserves_the_evidenced_exclusion() -> None:
    reason = load_release_input_coverage_manifest().reviewed_exclusions[
        "survivor_benefits"
    ]
    assert reason == _entry()["reason"]
    assert reason.startswith("SOURCE UNAVAILABILITY WITH EVIDENCE:")
