"""Tests split from packages/microcosm-build/tests/test_us_financial_assistance_exclusion.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_financial_assistance_exclusion import *


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
        "lines": "1493-1496",
    }
    assert evidence["required_person_source_declaration"]["lines"] == ("39-58,306-359")
    assert evidence["puf_clone_qrf"]["lines"] == "140-194,639-745"
    assert evidence["puf_clone_qrf"]["target_line"] == 166
    assert evidence["no_independent_puf_source"] == {
        "commit": "42ed5d45c56df80d754fbe24cce21cfeb8d05cbe",
        "path_parts": [
            "policyengine_",
            "us_data",
            "calibration",
            "puf_impute.py",
        ],
        "tax_detail_target_lines": "90-149,158-198",
        "financial_assistance_occurrences": 0,
    }
    assert evidence["required_person_columns"] == ["FIN_VAL"]
    assert evidence["missing_person_context_columns"] == ["FIN_YN", "I_FINVAL"]


def test_exclusion_pins_all_sha_locked_hermetic_inputs_and_grain() -> None:
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
        assert item["missing_person_columns"] == ["FIN_VAL", "FIN_YN", "I_FINVAL"]
        assert item["present_household_columns"] == ["HFINVAL", "HFIN_YN"]
        assert item["present_family_columns"] == ["FFINVAL", "FINC_FIN"]
        assert item["positive_multi_person_families"] > 0

    dictionary = evidence["official_variable_dictionary"]
    assert dictionary["url"] == (
        "https://api.census.gov/data/2024/cps/asec/mar/variables.html"
    )
    assert dictionary["FIN_VAL"]["entity"] == "person"
    assert dictionary["HFINVAL"]["entity"] == "household"
    assert dictionary["FFINVAL"]["entity"] == "family"
    substitutes = evidence["semantic_non_substitutes"]
    assert "synthesize" in substitutes["rejection"]
    assert "recipient person" in substitutes["HFINVAL"]
    assert "recipient person" in substitutes["FFINVAL"]

    build_script = (ROOT / "experiments/build_j_recert/buildj_base.sh").read_text()
    for year in (2022, 2023, 2024):
        assert f'--asec-h5 {year}="$USD/census_cps_{year}.h5"' in build_script
    assert "buildj_base.sh lines 65-69" in evidence["hermetic_build_contract"]
    assert "base_j.summary.json lines 55-75" in evidence["hermetic_build_contract"]
    asec_pool = (
        ROOT / "packages/microcosm-build/src/microcosm/build/us_runtime/asec_pool.py"
    ).read_text()
    assert 'pd.HDFStore(path, mode="r")' in asec_pool


def test_generated_release_manifest_preserves_the_evidenced_exclusion() -> None:
    reason = load_release_input_coverage_manifest().reviewed_exclusions[
        "financial_assistance"
    ]
    assert reason == _entry()["reason"]
    assert reason.startswith("SOURCE UNAVAILABILITY WITH EVIDENCE:")
