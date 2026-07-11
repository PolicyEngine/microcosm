"""UK release input-coverage manifest derivation and candidate evidence."""

from __future__ import annotations

import importlib.util
import json
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]
_GENERATOR = _REPO_ROOT / "tools" / "build_uk_release_input_coverage_manifest.py"
_UK_PACKAGE = "populace.build.uk"


def _resource(name: str) -> dict:
    return json.loads(files(_UK_PACKAGE).joinpath(name).read_text(encoding="utf-8"))


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "build_uk_release_input_coverage_manifest", _GENERATOR
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_candidate_evidence_is_sha_pinned_and_covers_reference() -> None:
    reference = _resource("efrs_parity_reference.json")
    gaps = _resource("efrs_parity_known_gaps.json")
    evidence = gaps["candidate_evidence"]
    assert evidence["source"]["sha256"] == (
        "f17306ccb2aad7ff0130be3589b560afb2e2a12a943570911cd0c77f07934833"
    )
    assert evidence["source"]["revision"] == (
        "populace-uk-2023-dd68c73-4aa4b14-20260619T023711Z"
    )
    assert set(evidence["nondefault_shares"]) == set(reference["nonzero_shares"])
    assert all(float(share) > 0 for share in evidence["nondefault_shares"].values())
    # Nonzero is not the gate criterion for default-True flags or zero-weight
    # support rows. These pin the default-aware extraction rather than a naive
    # nonzero comparison.
    assert evidence["nonzero_shares"]["household_owns_tv"] == 0.951073
    assert evidence["nondefault_shares"]["household_owns_tv"] == 0.048927
    assert evidence["nonzero_shares"]["household_weight"] == 0.626224
    assert evidence["nondefault_shares"]["household_weight"] == 1.0
    assert evidence["nondefault_shares"]["employment_income"] == 0.478282


def test_initial_known_gap_register_is_honestly_empty() -> None:
    gaps = _resource("efrs_parity_known_gaps.json")
    assert gaps["known_gaps"] == {}
    assert gaps["candidate_evidence"]["missing_columns"] == []
    assert gaps["candidate_evidence"]["default_only_columns"] == []
    assert gaps["exclusion_policy"]["reason"] == (
        "not yet ported from enhanced FRS pipeline — pending review"
    )
    assert gaps["exclusion_policy"]["tracking_note"].strip()


def test_committed_manifest_matches_regeneration() -> None:
    generator = _load_generator()
    committed = _resource("release_input_coverage_manifest.json")
    assert generator.build_manifest() == committed


def test_initial_manifest_requires_every_populated_reference_input() -> None:
    reference = _resource("efrs_parity_reference.json")
    manifest = _resource("release_input_coverage_manifest.json")
    assert manifest["counts"] == {
        "required": 132,
        "reviewed_exclusion": 0,
        "total": 132,
    }
    assert set(manifest["columns"]) == set(reference["nonzero_shares"])
    assert {entry["status"] for entry in manifest["columns"].values()} == {"required"}
    assert manifest["schema_version"] == 2
    assert manifest["effective_mass_coverage"] == {
        "weight_source": "household_weight",
        "minimum_nondefault_mass_share": 1e-6,
        "reviewed_on": "2026-07-11",
        "rationale": (
            "One part per million rejects zero-weight support and numerical "
            "dust while remaining about 100 times below the rarest populated "
            "record share in the pinned enhanced-FRS reference."
        ),
    }


def test_generator_assigns_exact_reason_and_tracking_note_to_a_real_gap() -> None:
    generator = _load_generator()
    reference = _resource("efrs_parity_reference.json")
    evidence = _resource("efrs_parity_known_gaps.json")["candidate_evidence"]
    tampered = json.loads(json.dumps(evidence))
    tampered["nondefault_shares"]["dividend_income"] = 0.0
    gaps = generator.build_known_gaps(tampered, reference=reference)
    entry = gaps["known_gaps"]["dividend_income"]
    assert entry["reason"] == (
        "not yet ported from enhanced FRS pipeline — pending review"
    )
    assert entry["tracking_note"].strip()


def test_candidate_signal_helpers_reject_null_and_blank_only_columns() -> None:
    generator = _load_generator()
    nulls = pd.Series([np.nan, None], dtype=object)
    blanks = pd.Series(["", "  "], dtype=object)
    for column in (nulls, blanks):
        assert generator._nonzero_share(column) == 0.0
        assert generator._nondefault_share(column, 0.0) == 0.0
    assert generator._nondefault_share(pd.Series([0, None], dtype=object), 0.0) == 0.0
