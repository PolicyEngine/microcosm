"""Adapter boundary: lazy imports, protocol conformance, export contracts.

These tests run without ``policyengine_us`` installed — the load-bearing
claim is that the adapter module imports and constructs without the engine,
and fails with a precise error only when an engine-backed method is called.
"""

import importlib.util
import json

import pytest

from populace.frame import US_SCHEMA, ExportContract, RulesEngine
from populace.frame.adapters.policyengine_us import PolicyEngineUSEngine

_POLICYENGINE_INSTALLED = importlib.util.find_spec("policyengine_us") is not None


class TestLazyImport:
    def test_adapter_constructs_without_the_engine(self) -> None:
        adapter = PolicyEngineUSEngine()
        assert isinstance(adapter, RulesEngine)

    def test_entity_schema_needs_no_engine(self) -> None:
        assert PolicyEngineUSEngine().entity_schema() == US_SCHEMA

    @pytest.mark.skipif(
        _POLICYENGINE_INSTALLED, reason="policyengine-us is installed here"
    )
    def test_engine_methods_name_the_extra_when_missing(self) -> None:
        adapter = PolicyEngineUSEngine()
        with pytest.raises(ImportError, match=r"populace-frame\[policyengine\]"):
            adapter.variable_entity("employment_income")


class TestExportContract:
    def test_empty_contract_has_no_constraints(self) -> None:
        contract = ExportContract.empty()
        assert contract.required == ()
        assert contract.forbidden == ()
        assert contract.optional == ()
        assert contract.formula_owned_excluded == ()

    def test_from_path_parses_sections_and_ignores_metadata(self, tmp_path) -> None:
        manifest = {
            "_description": "ignored documentation",
            "required": ["person_id", "household_weight"],
            "forbidden": ["spm_unit_net_income"],
            "optional": ["ecps_bookkeeping"],
            "formula_owned_excluded": ["eitc"],
        }
        path = tmp_path / "contract.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        contract = ExportContract.from_path(path)
        assert contract.required == ("person_id", "household_weight")
        assert contract.forbidden == ("spm_unit_net_income",)
        assert contract.optional == ("ecps_bookkeeping",)
        assert contract.formula_owned_excluded == ("eitc",)

    def test_from_path_accepts_ecps_internal_optional_key(self, tmp_path) -> None:
        manifest = {"ecps_internal_optional": ["spm_unit_pre_subsidy_childcare"]}
        path = tmp_path / "contract.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        contract = ExportContract.from_path(path)
        assert contract.optional == ("spm_unit_pre_subsidy_childcare",)

    def test_adapter_returns_its_contract(self) -> None:
        contract = ExportContract(
            required=("person_id",),
            forbidden=(),
            optional=(),
            formula_owned_excluded=(),
        )
        adapter = PolicyEngineUSEngine(contract=contract)
        assert adapter.export_contract() is contract
