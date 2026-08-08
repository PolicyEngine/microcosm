"""Evidence contract for irreducible Early Head Start source unavailability."""

from __future__ import annotations

import importlib.util
import inspect
import json
import os
from hashlib import sha256
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path

import h5py
import pandas as pd
import pytest

from microcosm.build.us_runtime.asec_pool import load_asec_h5_tables
from microcosm.build.us_runtime.release_input_coverage import (
    load_release_input_coverage_manifest,
)
from microcosm.build.us_runtime.sipp_head_start import (
    SIPP_2023_HEAD_START_DONOR_REVISION,
    SIPP_2023_HEAD_START_DONOR_SHA256,
    SIPP_2023_HEAD_START_DONOR_SIZE_BYTES,
)
from microcosm.build.us_runtime.take_up_contract import load_take_up_contract

ROOT = Path(__file__).resolve().parents[3]
_RUN_LARGE_SOURCE_AUDITS = os.environ.get("POPULACE_RUN_LARGE_SOURCE_AUDITS") == "1"
policyengine_us_installed = importlib.util.find_spec("policyengine_us") is not None
requires_us = pytest.mark.skipif(
    not policyengine_us_installed,
    reason="requires the policyengine-us [us] extra (build environment)",
)


def _entry() -> dict[str, object]:
    payload = json.loads(
        files("microcosm.build.us").joinpath("ecps_parity_known_gaps.json").read_text()
    )
    return payload["known_gaps"]["takes_up_early_head_start_if_eligible"]


def _build_summary() -> dict[str, object]:
    return json.loads(
        (ROOT / "experiments/build_j_recert/base_j.summary.json").read_text()
    )


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_recorded_artifact(recorded_path: str, filename: str) -> Path | None:
    """Resolve a Build-J input without silently substituting another artifact."""

    candidates = (
        Path(recorded_path).expanduser(),
        Path.home()
        / "PolicyEngine"
        / ("policyengine-" + "us-data")
        / ("policyengine_" + "us_data")
        / "storage"
        / filename,
    )
    return next((path for path in candidates if path.is_file()), None)


def _sipp_snapshot() -> Path:
    return (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / ("models--policyengine--policyengine-" + "us-data")
        / "snapshots"
        / SIPP_2023_HEAD_START_DONOR_REVISION
        / "pu2023.csv"
    )


def test_exclusion_pins_exact_archived_scalar_draw_evidence() -> None:
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
        "lines": "582-583,640-648",
    }
    assert evidence["retired_rate"] == {
        "commit": "42ed5d45c56df80d754fbe24cce21cfeb8d05cbe",
        "path_parts": [
            "policyengine_",
            "us_data",
            "parameters",
            "take_up",
            "early_head_start.yaml",
        ],
        "lines": "1-9",
        "value": 0.09,
        "citation_scope": (
            "NIEER research chart; no ACF administrative participation fact"
        ),
    }
    assert evidence["retired_randomness"] == {
        "commit": "42ed5d45c56df80d754fbe24cce21cfeb8d05cbe",
        "path_parts": [
            "policyengine_",
            "us_data",
            "utils",
            "randomness.py",
        ],
        "lines": "5-28",
        "key": "takes_up_early_head_start_if_eligible",
    }
    assert evidence["required_person_observation"] == (
        "Individual Early Head Start enrollment covering infants, toddlers, "
        "and pregnant participants"
    )
    assert "synthesize" in entry["reason"]


def test_exclusion_pins_exact_locked_artifact_absence_claims() -> None:
    evidence = _entry()["evidence"]
    missing = [
        "takes_up_early_head_start_if_eligible",
        "takes_up_head_start_if_eligible",
        "is_enrolled_in_head_start",
    ]
    assert evidence["hermetic_asec_inputs"] == [
        {
            "filename": "census_cps_2022.h5",
            "sha256": (
                "7ccca976284bb47815d84460cc4f75a0a65d26d7754ab0a0f417de351b3d474e"
            ),
            "missing_columns": missing,
        },
        {
            "filename": "census_cps_2023.h5",
            "sha256": (
                "cb57817327799f42b741caed5f9be94d04021c2e6809c1ad7bd0686da5428d88"
            ),
            "missing_columns": missing,
        },
        {
            "filename": "census_cps_2024.h5",
            "sha256": (
                "ec36604cb735a660b51b0b2f90be27d803b5878f3464fb30d0eacead59c1260d"
            ),
            "missing_columns": missing,
        },
    ]
    assert evidence["processed_puf"] == {
        "filename": "puf_2024.h5",
        "release": "policyengine/irs-soi-puf/1.8.0",
        "sha256": ("7669f5b5281f20080e77204f9bd4aabfad0aa101fa283e22caf9ba8d61d4d6df"),
        "missing_columns": missing,
    }

    summary = _build_summary()
    recorded_asec_hashes = {
        Path(item["path"]).name: item["sha256"]
        for item in summary["base_source"]["sources"]
    }
    asserted_asec_hashes = {
        item["filename"]: item["sha256"] for item in evidence["hermetic_asec_inputs"]
    }
    assert asserted_asec_hashes == recorded_asec_hashes
    assert evidence["processed_puf"]["filename"] == Path(summary["puf_h5"]).name
    assert evidence["processed_puf"]["sha256"] == summary["puf_sha256"]


@pytest.mark.parametrize(
    "filename",
    ["census_cps_2022.h5", "census_cps_2023.h5", "census_cps_2024.h5"],
)
def test_mounted_sha_locked_asec_schema_has_no_person_enrollment_signal(
    filename: str,
) -> None:
    evidence = _entry()["evidence"]
    item = next(
        record
        for record in evidence["hermetic_asec_inputs"]
        if record["filename"] == filename
    )
    summary_item = next(
        record
        for record in _build_summary()["base_source"]["sources"]
        if Path(record["path"]).name == filename
    )
    path = _resolve_recorded_artifact(summary_item["path"], filename)
    if path is None:
        pytest.skip(f"SHA-locked {filename} artifact is not mounted")

    assert _sha256(path) == item["sha256"]
    tables = load_asec_h5_tables(path)
    all_columns = {str(column) for table in tables.values() for column in table.columns}
    assert set(item["missing_columns"]).isdisjoint(all_columns)


def test_mounted_sha_locked_puf_schema_has_no_person_enrollment_signal() -> None:
    evidence = _entry()["evidence"]
    item = evidence["processed_puf"]
    summary = _build_summary()
    path = _resolve_recorded_artifact(summary["puf_h5"], item["filename"])
    if path is None:
        pytest.skip("SHA-locked processed PUF artifact is not mounted")

    assert _sha256(path) == item["sha256"]
    with h5py.File(path, mode="r") as puf:
        assert set(item["missing_columns"]).isdisjoint(puf.keys())


def test_pinned_sipp_evidence_has_no_early_head_start_domain_substitute() -> None:
    evidence = _entry()["evidence"]
    sipp = evidence["pinned_sipp"]

    assert sipp == {
        "revision": "21280dca5995e978d706740a8a4b9b7860cfd7b6",
        "sha256": ("5c30439e365fc26483318ef61d1d8f4bb2f0e9d6bb47c22c06756a7698733ee2"),
        "direct_education_item": (
            "EEDHEADST is nursery/preschool only and has zero reported positives "
            "under age 3"
        ),
        "child_care_scope": (
            "RDAYHS, RHEADST, and RNURHS cover ages 2-7 and do not distinguish "
            "Early Head Start"
        ),
        "pregnancy_signal": "none",
        "census_user_note": (
            "https://www.census.gov/programs-surveys/sipp/tech-documentation/"
            "user-notes/2023-usernotes/2023-small-inconsist-child-care.html"
        ),
    }
    assert sipp["revision"] == SIPP_2023_HEAD_START_DONOR_REVISION
    assert sipp["sha256"] == SIPP_2023_HEAD_START_DONOR_SHA256

    snapshot = _sipp_snapshot()
    if not snapshot.is_file():
        pytest.skip("the independently pinned full SIPP artifact is not mounted")
    assert snapshot.stat().st_size == SIPP_2023_HEAD_START_DONOR_SIZE_BYTES
    with snapshot.open("r", encoding="utf-8") as stream:
        columns = set(stream.readline().rstrip("\n").split("|"))
    assert {
        "MONTHCODE",
        "TAGE",
        "EEDHEADST",
        "AEDHEADST",
        "RDAYHS",
        "RHEADST",
        "RNURHS",
    } <= columns
    normalized = {column.upper().replace("_", "") for column in columns}
    assert not any("EARLYHEADSTART" in column for column in normalized)
    assert not any(
        "PREG" in column and ("HEADST" in column or column.endswith("EHS"))
        for column in normalized
    )


@pytest.mark.skipif(
    not _RUN_LARGE_SOURCE_AUDITS,
    reason="set POPULACE_RUN_LARGE_SOURCE_AUDITS=1 for the 3.7 GB SIPP scan",
)
def test_opt_in_full_sipp_scan_confirms_early_head_start_domain_absence() -> None:
    snapshot = _sipp_snapshot()
    if not snapshot.is_file():
        pytest.skip("the independently pinned full SIPP artifact is not mounted")
    assert _sha256(snapshot) == SIPP_2023_HEAD_START_DONOR_SHA256

    direct_positive_under_3 = 0
    observed_child_care_rows: dict[str, list[pd.DataFrame]] = {
        "RDAYHS": [],
        "RHEADST": [],
        "RNURHS": [],
    }
    allocation_status = {
        "RDAYHS": "ARDAYHS",
        "RHEADST": "ARHEADST",
        "RNURHS": "ARNURHS",
    }
    for chunk in pd.read_csv(
        snapshot,
        sep="|",
        usecols=[
            "SSUID",
            "PNUM",
            "MONTHCODE",
            "TAGE",
            "EEDHEADST",
            "AEDHEADST",
            *observed_child_care_rows,
            *allocation_status.values(),
        ],
        chunksize=100_000,
        low_memory=False,
    ):
        age = pd.to_numeric(chunk["TAGE"], errors="coerce")
        direct_positive_under_3 += int(
            (
                age.lt(3)
                & pd.to_numeric(chunk["AEDHEADST"], errors="coerce").eq(1)
                & pd.to_numeric(chunk["EEDHEADST"], errors="coerce").eq(1)
            ).sum()
        )
        for column, status_column in allocation_status.items():
            # Annual child-care recodes repeat on person-month rows. Status 1
            # is as reported; reduce to the first represented month per source
            # person before checking the documented age-2--7 respondent scope,
            # so a within-year birthday does not introduce a spurious age 8.
            observed = pd.to_numeric(chunk[status_column], errors="coerce").eq(
                1
            ) & pd.to_numeric(chunk[column], errors="coerce").isin([1, 2])
            observed_child_care_rows[column].append(
                chunk.loc[
                    observed,
                    ["SSUID", "PNUM", "MONTHCODE", "TAGE"],
                ].copy()
            )

    assert direct_positive_under_3 == 0
    for column, parts in observed_child_care_rows.items():
        observed = pd.concat(parts, ignore_index=True)
        first_person_month = observed.sort_values(
            "MONTHCODE", kind="mergesort"
        ).drop_duplicates(["SSUID", "PNUM"], keep="first")
        ages = pd.to_numeric(first_person_month["TAGE"], errors="coerce").dropna()
        assert len(ages), f"{column} has no directly observed source people"
        assert int(ages.min()) >= 2
        assert int(ages.max()) <= 7


def test_administrative_counts_are_semantic_non_substitutes() -> None:
    substitutes = _entry()["evidence"]["administrative_non_substitutes"]
    assert substitutes == {
        "cumulative_enrollment": (
            "Counts every participant served at any point in the program year, "
            "including turnover replacements; it is not point-in-time take-up "
            "and supplies no person identity."
        ),
        "funded_enrollment": (
            "Capacity or federally supported slots, not observed occupancy or "
            "person identity."
        ),
        "source": (
            "https://headstart.gov/sites/default/files/pdf/"
            "service-snapshot-ehs-2023-2024.pdf"
        ),
    }


def test_take_up_contract_retains_source_unavailable_treatment() -> None:
    program = load_take_up_contract().program_map()[
        "takes_up_early_head_start_if_eligible"
    ]
    assert program.populace_treatment == "rate_unsourced"
    assert program.rate["status"] == "source_unavailable"
    assert "locked individual-level source" in program.raw["followup"]
    assert "synthesize" in program.raw["notes"]


@requires_us
def test_policyengine_1_764_6_requires_under_3_or_pregnant_person_input() -> None:
    from policyengine_us import CountryTaxBenefitSystem

    assert version("policyengine-us") == "1.764.6"
    system = CountryTaxBenefitSystem()
    evidence = _entry()["evidence"]["policyengine_variable"]
    assert evidence == {
        "version": "1.764.6",
        "entity": "person",
        "definition_period": "year",
        "value_type": "bool",
        "default": True,
        "eligibility_domain": (
            "age < 3 or is_pregnant, plus Head Start income or categorical eligibility"
        ),
    }
    variable = system.variables["takes_up_early_head_start_if_eligible"]
    assert variable.is_input_variable()
    assert variable.entity.key == "person"
    assert str(variable.definition_period).lower() == "year"
    assert variable.value_type is bool
    assert variable.default_value is True

    formula = system.variables["is_early_head_start_eligible"].get_formula("2024")
    assert formula is not None
    source = inspect.getsource(formula)
    assert "age < p.early_head_start.age_limit" in source
    assert "is_pregnant" in source
    assert system.parameters("2024").gov.hhs.head_start.early_head_start.age_limit == 3


def test_generated_release_manifest_preserves_evidenced_exclusion() -> None:
    reason = load_release_input_coverage_manifest().reviewed_exclusions[
        "takes_up_early_head_start_if_eligible"
    ]
    assert reason == _entry()["reason"]
    assert reason.startswith("SOURCE UNAVAILABILITY WITH EVIDENCE:")
