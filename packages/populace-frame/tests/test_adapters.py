"""Adapter boundary: lazy imports, protocol conformance, export contracts.

These tests run without ``policyengine_us`` installed — the load-bearing
claim is that the adapter module imports and constructs without the engine,
and fails with a precise error only when an engine-backed method is called.
"""

import importlib.util
import json

import numpy as np
import pandas as pd
import pytest

from populace.frame import (
    US_SCHEMA,
    ExportContract,
    Frame,
    RulesEngine,
    WeightKind,
    Weights,
)
from populace.frame.adapters.policyengine_us import PolicyEngineUSEngine

_POLICYENGINE_INSTALLED = importlib.util.find_spec("policyengine_us") is not None


def _us_bundle(household_weight_column=None, weight_values=(1500.0, 900.0)):
    """A minimal US-schema bundle (no engine needed for _engine_tables)."""
    person = pd.DataFrame(
        {
            "person_id": [1, 2, 3],
            "person_household_id": [1, 1, 2],
            "person_tax_unit_id": [1, 1, 2],
            "person_spm_unit_id": [1, 1, 2],
            "person_family_id": [1, 1, 2],
            "person_marital_unit_id": [1, 1, 2],
            "age": [40.0, 38.0, 33.0],
        }
    )
    household = pd.DataFrame({"household_id": [1, 2]})
    if household_weight_column is not None:
        household["household_weight"] = household_weight_column
    group = lambda name: pd.DataFrame({f"{name}_id": [1, 2]})  # noqa: E731
    tables = {
        "person": person,
        "household": household,
        "tax_unit": group("tax_unit"),
        "spm_unit": group("spm_unit"),
        "family": group("family"),
        "marital_unit": group("marital_unit"),
    }
    weights = {
        "household": Weights(
            values=np.array(weight_values), kind=WeightKind.CALIBRATED
        )
    }
    return Frame(tables, US_SCHEMA, weights)


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


class TestEngineTablesWeights:
    """C1: typed weights are authoritative for the exported household_weight."""

    def test_typed_weights_overwrite_a_stale_household_weight_column(self) -> None:
        # A leftover household_weight column [999, 999] must NOT win over the
        # bundle's calibrated typed weights [1500, 900].
        bundle = _us_bundle(household_weight_column=[999.0, 999.0])
        tables = PolicyEngineUSEngine()._engine_tables(bundle)
        assert tables["household"]["household_weight"].tolist() == [1500.0, 900.0]

    def test_household_weight_is_materialized_when_no_column_present(self) -> None:
        bundle = _us_bundle(household_weight_column=None)
        tables = PolicyEngineUSEngine()._engine_tables(bundle)
        assert tables["household"]["household_weight"].tolist() == [1500.0, 900.0]
