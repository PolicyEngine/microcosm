"""Evidence contract for the irreducible financial-assistance exclusion."""

from __future__ import annotations

import importlib.util
import json
from hashlib import sha256
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest

from populace.build.us_runtime.release_input_coverage import (
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
        files("populace.build.us").joinpath("ecps_parity_known_gaps.json").read_text()
    )
    return payload["known_gaps"]["financial_assistance"]


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        ROOT / "packages/populace-build/src/populace/build/us_runtime/asec_pool.py"
    ).read_text()
    assert 'pd.HDFStore(path, mode="r")' in asec_pool


@requires_us
def test_sha_locked_artifact_schemas_match_the_recorded_absence() -> None:
    from policyengine_us.data import USSingleYearDataset

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
        dataset = USSingleYearDataset(file_path=str(path))
        assert set(item["missing_person_columns"]).isdisjoint(dataset.person.columns)
        assert set(item["present_household_columns"]) <= set(dataset.household.columns)
        assert set(item["present_family_columns"]) <= set(dataset.family.columns)
        positive_multi_person_families = int(
            ((dataset.family["FFINVAL"] > 0) & (dataset.family["FPERSONS"] > 1)).sum()
        )
        assert positive_multi_person_families == item["positive_multi_person_families"]
        family_amount_by_household = dataset.family.groupby("FH_SEQ", sort=False)[
            "FFINVAL"
        ].sum()
        household_amount = dataset.household.set_index("H_SEQ")["HFINVAL"]
        np.testing.assert_array_equal(
            family_amount_by_household.reindex(
                household_amount.index,
                fill_value=0,
            ).to_numpy(),
            household_amount.to_numpy(),
        )


@requires_us
def test_policyengine_1_764_6_requires_a_person_year_input() -> None:
    from policyengine_us import CountryTaxBenefitSystem

    assert version("policyengine-us") == "1.764.6"
    variable = CountryTaxBenefitSystem().variables["financial_assistance"]
    assert variable.is_input_variable()
    assert variable.entity.key == "person"
    assert str(variable.definition_period).lower() == "year"
    assert variable.documentation == (
        "Cash financial assistance from outside the household."
    )


def test_generated_release_manifest_preserves_the_evidenced_exclusion() -> None:
    reason = load_release_input_coverage_manifest().reviewed_exclusions[
        "financial_assistance"
    ]
    assert reason == _entry()["reason"]
    assert reason.startswith("SOURCE UNAVAILABILITY WITH EVIDENCE:")
