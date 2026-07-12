"""Integrity tests for the pinned UK enhanced-FRS parity reference."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from importlib.resources import files
from pathlib import Path

import pytest

from populace.build.uk_runtime.parity_reference import (
    EFRS_PARITY_REFERENCE_RESOURCE,
    load_efrs_parity_reference,
)

_UK_PACKAGE = "populace.build.uk"
_REPO_ROOT = Path(__file__).resolve().parents[3]
_GENERATOR = _REPO_ROOT / "tools" / "build_uk_efrs_parity_reference.py"
_FORMULA_OWNED_PERSISTED_OVERRIDES = {
    "current_education",
    "is_benunit_head",
    "is_enhanced_disabled_for_benefits",
    "is_household_head",
    "is_in_non_advanced_education",
    "is_married",
    "is_severely_disabled_for_benefits",
    "marital_status",
    "property_wealth",
    "rail_subsidy_spending",
    "state_pension_reported",
    "student_loan_repayments",
    "tax_free_savings_income",
}


def _resource(name: str) -> Path:
    return Path(str(files(_UK_PACKAGE).joinpath(name)))


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "build_uk_efrs_parity_reference",
        _GENERATOR,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cached_incumbent_path(source) -> Path | None:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    resolved = try_to_load_from_cache(
        repo_id=source.repo_id,
        filename=source.filename,
        revision=source.revision,
        repo_type=source.repo_type,
    )
    if isinstance(resolved, str):
        return Path(resolved)
    return None


class TestEfrsParityReference:
    def test_reference_loads_with_populated_layers(self) -> None:
        reference = load_efrs_parity_reference()
        assert reference.schema_version == 3
        assert len(reference.nonzero_shares) == 145
        assert len(reference.populated_layers) == 145
        assert all(share > 0.0 for share in reference.nonzero_shares.values())
        assert set(reference.input_entities) == set(reference.nonzero_shares)
        assert set(reference.input_entities.values()) == {
            "person",
            "benunit",
            "household",
        }

    def test_source_provenance_is_complete_and_immutable(self) -> None:
        source = load_efrs_parity_reference().source
        assert source.repo_id == "policyengine/policyengine-uk-data-private"
        assert source.repo_type == "model"
        assert source.filename == "enhanced_frs_2023_24.h5"
        assert source.revision == "655dd07e4bb9c777b00dac044949611f1feb824f"
        assert source.sha256 == (
            "584ae33d80ca0431254610a3f8254d132da73477d31966d6446282861ecae50d"
        )
        assert source.size_bytes == 125_434_652
        assert source.url.endswith(f"/{source.revision}/{source.filename}")
        assert source.period == "2023"
        assert source.vintage == "2023_24"

    def test_recorded_sha256_is_lowercase_hex(self) -> None:
        sha = load_efrs_parity_reference().source.sha256
        assert len(sha) == 64
        int(sha, 16)
        assert sha == sha.lower()

    def test_reference_json_content_matches_loader(self) -> None:
        raw = json.loads(_resource(EFRS_PARITY_REFERENCE_RESOURCE).read_text())
        reference = load_efrs_parity_reference()
        assert set(raw["nonzero_shares"]) == set(reference.nonzero_shares)
        assert raw["input_entities"] == dict(reference.input_entities)
        assert raw["source"]["sha256"] == reference.source.sha256

    def test_reference_requires_exact_entity_surface(self, tmp_path) -> None:
        payload = json.loads(_resource(EFRS_PARITY_REFERENCE_RESOURCE).read_text())
        payload["input_entities"].pop("employment_income")
        bad = tmp_path / "bad_reference.json"
        bad.write_text(json.dumps(payload))

        with pytest.raises(ValueError, match="must exactly cover"):
            load_efrs_parity_reference(str(bad))

    def test_all_zero_reference_inputs_are_documented_not_required(self) -> None:
        raw = json.loads(_resource(EFRS_PARITY_REFERENCE_RESOURCE).read_text())
        assert raw["engine"]["zero_share_input_columns_excluded"] == [
            "disabled_students_allowance_eligible_expenses",
            "incapacity_benefit_reported",
            "is_in_approved_training",
        ]
        assert not (
            set(raw["engine"]["zero_share_input_columns_excluded"])
            & set(raw["nonzero_shares"])
        )

    def test_uk_loader_migration_inputs_are_covered(self) -> None:
        raw = json.loads(_resource(EFRS_PARITY_REFERENCE_RESOURCE).read_text())
        aliases = raw["engine"]["h5_input_aliases"]
        assert aliases == {
            "capital_gains": "capital_gains_before_response",
            "employee_pension_contributions": (
                "employee_pension_contributions_reported"
            ),
            "employment_income": "employment_income_before_lsr",
        }
        assert set(aliases) <= set(raw["nonzero_shares"])

    def test_formula_owned_persisted_overrides_are_hard_covered(self) -> None:
        raw = json.loads(_resource(EFRS_PARITY_REFERENCE_RESOURCE).read_text())
        included = set(
            raw["engine"]["formula_owned_persisted_overrides_included"]
        )
        assert included == _FORMULA_OWNED_PERSISTED_OVERRIDES
        assert included <= set(raw["nonzero_shares"])
        assert "formula_owned_export_columns_excluded" not in raw["engine"]

    def test_empty_reference_is_refused(self, tmp_path) -> None:
        payload = json.loads(_resource(EFRS_PARITY_REFERENCE_RESOURCE).read_text())
        payload["nonzero_shares"] = {}
        bad = tmp_path / "bad_reference.json"
        bad.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="vacuous"):
            load_efrs_parity_reference(str(bad))

    def test_nonfinite_share_is_refused(self, tmp_path) -> None:
        payload = json.loads(_resource(EFRS_PARITY_REFERENCE_RESOURCE).read_text())
        payload["nonzero_shares"]["age"] = float("nan")
        bad = tmp_path / "bad_reference.json"
        bad.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="outside"):
            load_efrs_parity_reference(str(bad))

    def test_recorded_sha_matches_cached_incumbent_bytes(self) -> None:
        source = load_efrs_parity_reference().source
        cached = _cached_incumbent_path(source)
        if cached is None:
            pytest.skip("pinned licensed eFRS artifact is not in the local HF cache")
        digest = hashlib.sha256(cached.read_bytes()).hexdigest()
        assert cached.stat().st_size == source.size_bytes
        assert digest == source.sha256

    def test_cached_reference_regeneration_matches_committed_surface(self) -> None:
        source = load_efrs_parity_reference().source
        cached = _cached_incumbent_path(source)
        if cached is None:
            # A content-addressed blob can still be supplied explicitly even
            # when the HF revision-to-filename cache mapping is unavailable.
            candidate = (
                Path.home()
                / ".cache"
                / "huggingface"
                / "hub"
                / "models--policyengine--policyengine-uk-data-private"
                / "blobs"
                / source.sha256
            )
            cached = candidate if candidate.is_file() else None
        if cached is None:
            pytest.skip("pinned licensed eFRS artifact is not in the local HF cache")

        regenerated = _load_generator().build_reference(cached)
        committed = json.loads(
            _resource(EFRS_PARITY_REFERENCE_RESOURCE).read_text(encoding="utf-8")
        )
        assert regenerated == committed

    def test_implicit_resolution_requires_revision_mapping(
        self,
        monkeypatch,
    ) -> None:
        import huggingface_hub

        generator = _load_generator()

        class DownloadAttemptedError(RuntimeError):
            pass

        monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda **_: None)

        def fail_download(**kwargs):
            assert kwargs["repo_id"] == generator.SOURCE_REPO_ID
            assert kwargs["filename"] == generator.SOURCE_FILENAME
            assert kwargs["revision"] == generator.SOURCE_REVISION
            raise DownloadAttemptedError

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", fail_download)

        with pytest.raises(DownloadAttemptedError):
            generator.resolve_source_h5()
