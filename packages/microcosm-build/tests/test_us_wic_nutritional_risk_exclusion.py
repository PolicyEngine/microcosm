"""Evidence contract for the irreducible WIC nutritional-risk exclusion."""

from __future__ import annotations

import importlib.util
import inspect
import json
from hashlib import sha256
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path

import pytest

from microcosm.build.us_runtime.asec_pool import load_asec_h5_tables
from microcosm.build.us_runtime.release_input_coverage import (
    load_release_input_coverage_manifest,
)

ROOT = Path(__file__).resolve().parents[3]
policyengine_us_installed = importlib.util.find_spec("policyengine_us") is not None
requires_us = pytest.mark.skipif(
    not policyengine_us_installed,
    reason="requires the policyengine-us [us] extra (build environment)",
)


def _entry() -> dict[str, object]:
    payload = json.loads(
        files("microcosm.build.us").joinpath("ecps_parity_known_gaps.json").read_text()
    )
    return payload["known_gaps"]["is_wic_at_nutritional_risk"]


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wic_like_columns(columns) -> list[str]:
    return sorted(
        column
        for column in columns
        if any(token in str(column).upper() for token in ("WIC", "NUTRI", "RISK"))
    )


def test_exclusion_pins_exact_archived_stochastic_derivation() -> None:
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
        "lines": "684-701",
        "method": "receives_wic OR category-specific seeded Bernoulli draw",
    }
    assert evidence["retired_risk_rates"]["lines"] == "1-13"
    assert evidence["retired_risk_rates"]["values"] == {
        "PREGNANT": 0.913,
        "POSTPARTUM": 0.933,
        "BREASTFEEDING": 0.889,
        "INFANT": 0.95,
        "CHILD": 0.752,
        "NONE": 0,
    }
    assert evidence["category_rate_source"]["lines"] == "38-50"
    assert evidence["retired_receipt_mapping"] == {
        "commit": "42ed5d45c56df80d754fbe24cce21cfeb8d05cbe",
        "path_parts": [
            "policyengine_",
            "us_data",
            "datasets",
            "cps",
            "cps.py",
        ],
        "lines": "1553-1559",
        "mapping": "receives_wic = person.WICYN == 1",
    }
    assert evidence["retired_source_declaration"]["lines"] == "293-365"
    assert evidence["retired_source_declaration"]["available_wic_fields"] == [
        "SPM_WICVAL",
        "WICYN",
    ]


def test_exclusion_pins_official_assessment_semantics_and_no_substitute() -> None:
    evidence = _entry()["evidence"]

    census = evidence["official_census_dictionary"]
    assert census["field"] == "WICYN"
    assert census["definition"] == "Who received WIC?"
    assert census["universe"] == "Adult female"
    assert "on behalf of a child" in census["questionnaire_context"]
    assert census["codes"] == {
        "0": "Not in universe",
        "1": "Received WIC",
        "2": "Did not receive WIC",
    }
    assert census["url"].endswith("/cpsmar24.pdf")

    certification = evidence["official_certification_source"]
    assert certification["url"].startswith("https://www.fns.usda.gov/")
    assert "competent professional authority" in certification["finding"]
    assert "state administrative records" in certification["finding"]
    assert "inference" not in certification["finding"]
    assert "Therefore" in certification["record_linkage_inference"]
    assert (
        "neither an assessment field nor a link key"
        in certification["record_linkage_inference"]
    )

    current_method = evidence["current_fns_estimation_method"]
    assert current_method["lines"] == "PDF page 84 (printed report page 70)"
    assert current_method["adopted_when"] == "while producing the CY2020 estimates"
    assert current_method["applied_period"] == (
        "revised CY2016-CY2021 estimates in the cited report"
    )
    assert current_method["risk_adjustment"] == 1.0
    assert current_method["scope"] == "all participant categories"

    substitutes = evidence["semantic_non_substitutes"]
    assert "does not identify the assessed person" in substitutes["WICYN"]
    assert "nonreceipt also does not prove" in substitutes["WICYN"]
    assert "neither an assessed person nor a negative" in substitutes["SPM_WICVAL"]
    assert "synthesize" in substitutes["rejection"]


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
        assert item["person_wic_columns"] == ["SPM_WICVAL", "WICYN"]
        assert item["household_wic_columns"] == ["HRNUMWIC", "HRWICYN"]
        assert sum(item["wicyn_counts"].values()) > 100_000
        assert item["wicyn_counts"]["1"] > 1_000

    build_script = (ROOT / "experiments/build_j_recert/buildj_base.sh").read_text()
    for year in (2022, 2023, 2024):
        assert f'--asec-h5 {year}="$USD/census_cps_{year}.h5"' in build_script
    assert "buildj_base.sh lines 65-69" in evidence["hermetic_build_contract"]
    assert "base_j.summary.json lines 55-75" in evidence["hermetic_build_contract"]
    asec_pool = (
        ROOT / "packages/microcosm-build/src/microcosm/build/us_runtime/asec_pool.py"
    ).read_text()
    assert 'pd.HDFStore(path, mode="r")' in asec_pool


def test_mounted_artifacts_contain_receipt_but_no_risk_assessment() -> None:
    pytest.importorskip("tables")
    evidence = _entry()["evidence"]
    build_summary = json.loads(
        (ROOT / "experiments/build_j_recert/base_j.summary.json").read_text()
    )
    paths = {
        Path(item["path"]).name: Path(item["path"])
        for item in build_summary["base_source"]["sources"]
    }
    if not all(path.is_file() for path in paths.values()):
        pytest.skip("SHA-locked ASEC artifacts are not mounted in this environment")

    for item in evidence["hermetic_inputs"]:
        path = paths[item["filename"]]
        assert _sha256(path) == item["sha256"]
        tables = load_asec_h5_tables(path)
        assert _wic_like_columns(tables["person"].columns) == sorted(
            item["person_wic_columns"]
        )
        assert _wic_like_columns(tables["household"].columns) == sorted(
            item["household_wic_columns"]
        )
        observed_counts = {
            str(int(code)): int(count)
            for code, count in tables["person"]["WICYN"].value_counts().items()
        }
        assert observed_counts == item["wicyn_counts"]


@requires_us
def test_policyengine_1_764_6_defaults_risk_true_and_uses_it_for_eligibility() -> None:
    from policyengine_us import CountryTaxBenefitSystem

    assert version("policyengine-us") == "1.764.6"
    system = CountryTaxBenefitSystem()
    variable = system.variables["is_wic_at_nutritional_risk"]
    assert variable.is_input_variable()
    assert variable.entity.key == "person"
    assert str(variable.definition_period).lower() == "month"
    assert variable.value_type is bool
    assert variable.default_value is True

    eligibility_formula = system.variables["is_wic_eligible"].get_formula("2024-01")
    assert eligibility_formula is not None
    assert 'person("is_wic_at_nutritional_risk", period)' in inspect.getsource(
        eligibility_formula
    )


def test_generated_release_manifest_preserves_the_evidenced_exclusion() -> None:
    reason = load_release_input_coverage_manifest().reviewed_exclusions[
        "is_wic_at_nutritional_risk"
    ]
    assert reason == _entry()["reason"]
    assert reason.startswith("SOURCE UNAVAILABILITY WITH EVIDENCE:")
