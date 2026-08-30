"""The RulesEngine behavioral contract, parameterized over adapters.

One suite, every adapter: each case supplies an adapter, a bundle it
understands, and a computed variable to exercise, and the tests assert the
*protocol's* promises — conformance, metadata coherence, input enumeration,
row-aligned materialization, weight authority on export, and round-trip
persistence. Engine-specific arithmetic (exact liabilities, statutory
values) belongs to each adapter's own test module; everything here must
hold for policyengine-us and the Axiom engine alike, and for every adapter
after them — a new engine earns its place by passing this file unchanged.

Cases skip when their engine is not installed: policyengine-us runs in CI;
the Axiom engine (not yet on PyPI) runs wherever its dense extension is
built.
"""

import importlib.util
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.frame import (
    US_SCHEMA,
    Frame,
    RulesEngine,
    VariableMetadata,
    WeightKind,
    Weights,
)
from microcosm.frame.adapters.axiom import (
    BE_SCHEMA,
    AxiomEngine,
    AxiomEntityTableDataset,
)
from microcosm.frame.adapters.policyengine_us import PolicyEngineUSEngine

_POLICYENGINE_INSTALLED = importlib.util.find_spec("policyengine_us") is not None
_AXIOM_INSTALLED = importlib.util.find_spec("axiom_rules_engine") is not None
if _AXIOM_INSTALLED:
    from axiom_rules_engine.dense import NativeCompiledDenseProgram

    _AXIOM_DENSE = NativeCompiledDenseProgram is not None
else:
    _AXIOM_DENSE = False

_FIXTURE_MODULE = (
    Path(__file__).parent / "fixtures/rulespec-xx/xx/policies/axiom_toy_country.yaml"
)


@dataclass(frozen=True)
class AdapterCase:
    """Everything the shared contract needs to exercise one adapter.

    Attributes:
        make_adapter: Fresh adapter instance.
        make_bundle: A bundle the adapter's engine can compute over, with
            household weights ``[1500.0, 900.0]``.
        computed_variables: Formula-owned variable names to materialize
            (at least one; spanning more than one entity where the engine
            supports it).
        known_input: An input variable the engine must enumerate.
        period: The period materialization runs against.
        reload_tables: Read the adapter's written dataset back as
            entity-name -> table.
    """

    make_adapter: Callable[[], RulesEngine]
    make_bundle: Callable[[], Frame]
    computed_variables: tuple[str, ...]
    known_input: str
    period: int
    reload_tables: Callable[[Path], Mapping[str, pd.DataFrame]]


def _us_bundle() -> Frame:
    person = pd.DataFrame(
        {
            "person_id": [1, 2, 3],
            "person_household_id": [1, 1, 2],
            "person_tax_unit_id": [1, 1, 2],
            "person_spm_unit_id": [1, 1, 2],
            "person_family_id": [1, 1, 2],
            "person_marital_unit_id": [1, 1, 2],
            "age": [40.0, 38.0, 33.0],
            # The true engine input: employment_income itself is
            # formula-owned (derived from the pre-LSR value), so a bundle
            # carrying it would be rejected at export.
            "employment_income_before_lsr": [70_000.0, 30_000.0, 48_000.0],
        }
    )
    group = lambda name: pd.DataFrame({f"{name}_id": [1, 2]})  # noqa: E731
    tables = {
        "person": person,
        "household": pd.DataFrame({"household_id": [1, 2], "state_fips": [6, 36]}),
        "tax_unit": group("tax_unit"),
        "spm_unit": group("spm_unit"),
        "family": group("family"),
        "marital_unit": group("marital_unit"),
    }
    weights = {
        "household": Weights(values=np.array([1500.0, 900.0]), kind=WeightKind.DESIGN)
    }
    return Frame(tables, US_SCHEMA, weights)


def _us_reload(path: Path) -> Mapping[str, pd.DataFrame]:
    from policyengine_us.data import USSingleYearDataset

    dataset = USSingleYearDataset(file_path=str(path))
    return {name: getattr(dataset, name) for name in dataset.table_names}


def _axiom_bundle() -> Frame:
    person = pd.DataFrame(
        {
            "person_id": [1, 2, 3],
            "person_household_id": [1, 1, 2],
            "toy_taxable_income": [5_000.0, 10_000.0, 20_000.0],
            "toy_is_exempt": [False, False, False],
            "toy_child_count": [0, 2, 1],
        }
    )
    household = pd.DataFrame(
        {"household_id": [1, 2], "toy_household_rent": [7_200.0, 4_800.0]}
    )
    weights = {
        "household": Weights(values=np.array([1500.0, 900.0]), kind=WeightKind.DESIGN)
    }
    return Frame({"person": person, "household": household}, BE_SCHEMA, weights)


def _axiom_reload(path: Path) -> Mapping[str, pd.DataFrame]:
    return AxiomEntityTableDataset(file_path=path).tables


CASES = [
    pytest.param(
        AdapterCase(
            make_adapter=PolicyEngineUSEngine,
            make_bundle=_us_bundle,
            computed_variables=("income_tax", "household_net_income"),
            known_input="employment_income_before_lsr",
            period=2024,
            reload_tables=_us_reload,
        ),
        id="policyengine-us",
        marks=pytest.mark.skipif(
            not _POLICYENGINE_INSTALLED, reason="policyengine-us is not installed"
        ),
    ),
    pytest.param(
        AdapterCase(
            make_adapter=lambda: AxiomEngine(_FIXTURE_MODULE),
            make_bundle=_axiom_bundle,
            computed_variables=("toy_income_tax", "toy_housing_allowance"),
            known_input="toy_taxable_income",
            period=2025,
            reload_tables=_axiom_reload,
        ),
        id="axiom",
        marks=pytest.mark.skipif(
            not _AXIOM_DENSE,
            reason="axiom_rules_engine dense extension is not installed",
        ),
    ),
]


@pytest.fixture(params=CASES)
def case(request) -> AdapterCase:
    return request.param


class TestProtocolConformance:
    def test_adapter_satisfies_the_protocol(self, case) -> None:
        assert isinstance(case.make_adapter(), RulesEngine)

    def test_entity_schema_matches_the_bundle(self, case) -> None:
        adapter = case.make_adapter()
        schema = adapter.entity_schema()
        assert set(schema.entities) == set(case.make_bundle().entities)


class TestMetadataResolution:
    def test_computed_variables_resolve_to_declared_entities(self, case) -> None:
        adapter = case.make_adapter()
        schema = adapter.entity_schema()
        for name in case.computed_variables:
            metadata = adapter.variable_metadata(name)
            assert isinstance(metadata, VariableMetadata)
            assert metadata.name == name
            assert metadata.entity in schema.entities

    def test_unknown_variables_are_refused_by_name(self, case) -> None:
        with pytest.raises(ValueError, match="definitely_not_a_variable"):
            case.make_adapter().variable_metadata("definitely_not_a_variable")


class TestInputEnumeration:
    def test_inputs_are_sorted_unique_and_exclude_computed(self, case) -> None:
        adapter = case.make_adapter()
        names = adapter.variables()
        assert names, "an engine with no inputs cannot host a dataset"
        assert list(names) == sorted(set(names))
        assert case.known_input in names
        for computed in case.computed_variables:
            assert computed not in names


class TestMaterialization:
    def test_arrays_are_row_aligned_to_their_entity(self, case) -> None:
        adapter = case.make_adapter()
        bundle = case.make_bundle()
        results = adapter.materialize(
            bundle, list(case.computed_variables), period=case.period
        )
        assert set(results) == set(case.computed_variables)
        for name, values in results.items():
            entity = adapter.variable_metadata(name).entity
            assert values.shape == (bundle.n(entity),)
            assert np.all(np.isfinite(np.asarray(values, dtype=float)))


class TestExport:
    def test_written_dataset_round_trips_with_authoritative_weights(
        self, case, tmp_path
    ) -> None:
        adapter = case.make_adapter()
        bundle = case.make_bundle()
        path = tmp_path / "contract.h5"
        adapter.write_dataset(bundle, path, period=case.period)
        tables = case.reload_tables(path)
        weights = tables["household"]["household_weight"]
        assert weights.tolist() == [1500.0, 900.0]

    def test_stale_weight_columns_never_survive_export(self, case, tmp_path) -> None:
        adapter = case.make_adapter()
        bundle = case.make_bundle()
        household = bundle.table("household").copy()
        household["household_weight"] = [999.0, 999.0]
        tables = {name: bundle.table(name) for name in bundle.entities}
        tables["household"] = household
        stale = Frame(
            tables,
            adapter.entity_schema(),
            {"household": bundle.weights_for("household")},
        )
        path = tmp_path / "stale.h5"
        adapter.write_dataset(stale, path, period=case.period)
        reloaded = case.reload_tables(path)
        assert reloaded["household"]["household_weight"].tolist() == [1500.0, 900.0]
