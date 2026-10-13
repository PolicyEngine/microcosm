"""Tests split from packages/microcosm-build/tests/test_us_dc_ptc_take_up_exclusion.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_dc_ptc_take_up_exclusion import *


def test_exclusion_pins_archived_random_derivation_and_model_relative_rate() -> None:
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
        "lines": "565-599",
        "method": "seeded Bernoulli draw below a scalar dc_ptc_rate",
    }
    assert evidence["retired_rate_parameter"] == {
        "commit": "42ed5d45c56df80d754fbe24cce21cfeb8d05cbe",
        "path_parts": [
            "policyengine_",
            "us_data",
            "parameters",
            "take_up",
            "dc_ptc.yaml",
        ],
        "lines": "1-11",
        "value": 0.32,
        "administrative_claim_count": 37133,
        "comment_model_estimate": 131791388,
        "denominator_description": "PolicyEngine DC PTC value estimate",
        "external_eligible_population_denominator": False,
        "dimensionally_consistent_participation_rate": False,
        "arithmetically_consistent_with_value": False,
    }
    assert evidence["retired_randomness"]["lines"] == "5-28"
    assert evidence["retired_randomness"]["key"] == "takes_up_dc_ptc"
    assert evidence["missing_claim_column_patterns"] == [
        "takes_up_dc_ptc",
        "dc_ptc",
        "schedule_h",
        "property_tax_credit",
    ]
    assert evidence["archived_asec_tax_columns"]["lines"] == "251-262"
    assert evidence["archived_puf_credit_mapping"]["lines"] == "704-719"
    assert evidence["archived_puf_credit_mapping"]["other_credits_source"] == ("P08000")
    assert "synthesize" in evidence["semantic_non_substitutes"]["rejection"]


def test_current_take_up_contract_rejects_the_model_relative_rate() -> None:
    evidence = _entry()["evidence"]
    program = load_take_up_contract().program_map()["takes_up_dc_ptc"]

    assert evidence["current_take_up_contract"]["treatment"] == "rate_unsourced"
    assert evidence["current_take_up_contract"]["rate_status"] == "model_relative"
    assert program.populace_treatment == "rate_unsourced"
    assert program.rate["status"] == "model_relative"
    assert "policyengine" in str(program.rate["detail"]).lower()
    assert "takes_up_dc_ptc" not in {
        seeded.variable for seeded in seeded_take_up_programs()
    }


def test_exclusion_pins_all_sha_locked_hermetic_inputs() -> None:
    evidence = _entry()["evidence"]
    build_summary = json.loads(
        (ROOT / "experiments/build_j_recert/base_j.summary.json").read_text()
    )
    recorded_asec_hashes = {
        Path(item["path"]).name: item["sha256"]
        for item in build_summary["base_source"]["sources"]
    }
    evidence_asec_hashes = {
        item["filename"]: item["sha256"] for item in evidence["hermetic_asec_inputs"]
    }

    assert evidence_asec_hashes == recorded_asec_hashes
    assert evidence["processed_puf"]["filename"] == Path(build_summary["puf_h5"]).name
    assert evidence["processed_puf"]["sha256"] == build_summary["puf_sha256"]
    for item in evidence["hermetic_asec_inputs"]:
        assert item["missing_columns"] == ["takes_up_dc_ptc"]
        assert item["present_generic_tax_amount_columns"] == [
            "STATETAX_A",
            "STATETAX_B",
            "SPM_STTAX",
        ]
    assert evidence["processed_puf"]["missing_arrays"] == [
        "takes_up_dc_ptc",
        "state_fips",
        "state_code",
        "state_code_str",
    ]
    dictionary = evidence["official_census_variable_dictionary"]
    assert dictionary["STATETAX_A"] == "State income tax liability, after credits"
    assert dictionary["STATETAX_B"] == "State income tax liability, before credits"
    assert dictionary["SPM_STTAX"] == "SPM unit's state tax"


def test_generated_release_manifest_preserves_the_evidenced_exclusion() -> None:
    reason = load_release_input_coverage_manifest().reviewed_exclusions[
        "takes_up_dc_ptc"
    ]
    assert reason == _entry()["reason"]
    assert reason.startswith("SOURCE UNAVAILABILITY WITH EVIDENCE:")
