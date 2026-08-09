"""Axiom adapter behavior: lazy import, metadata, materialize, export.

The no-engine tests run everywhere (the load-bearing claim is that the
adapter module imports and constructs without ``axiom_rules_engine``, and
fails with a precise error only when an engine-backed method is called).
Engine-backed tests skip unless the package *and* its native dense extension
are importable — the engine is not on PyPI, so CI runs without it; build it
locally from an axiom-rules-engine checkout to exercise them.

``TestBelgianPilotSlice`` additionally needs a rulespec-be checkout, pointed
at by the ``POPULACE_RULESPEC_BE`` environment variable — it materializes
the real CIR 1992 article 130 rate scale and checks hand-computed 2025
liabilities.
"""

import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.frame import (
    EntitySchema,
    ExportContract,
    Frame,
    RulesEngine,
    WeightKind,
    Weights,
)
from microcosm.frame.adapters.axiom import (
    BE_SCHEMA,
    AxiomEngine,
    AxiomEntityTableDataset,
    _period_bounds,
)

_ENGINE_INSTALLED = importlib.util.find_spec("axiom_rules_engine") is not None
if _ENGINE_INSTALLED:
    from axiom_rules_engine.dense import NativeCompiledDenseProgram

    _DENSE_AVAILABLE = NativeCompiledDenseProgram is not None
else:
    _DENSE_AVAILABLE = False
_TABLES_INSTALLED = importlib.util.find_spec("tables") is not None

needs_engine = pytest.mark.skipif(
    not _DENSE_AVAILABLE,
    reason="axiom_rules_engine (with the dense native extension) is not installed",
)
needs_tables = pytest.mark.skipif(
    not _TABLES_INSTALLED,
    reason="pytables (microcosm-frame[axiom]) is not installed",
)

FIXTURE_MODULE = Path(__file__).parent / "fixtures" / "axiom_toy_country.yaml"
RULESPEC_BE = os.environ.get("POPULACE_RULESPEC_BE")


def _toy_bundle(
    incomes=(5_000.0, 10_000.0, 20_000.0),
    exempt=(False, False, False),
    children=(0, 2, 1),
    rents=(7_200.0, 4_800.0),
) -> Frame:
    """Three persons in two households carrying the fixture's inputs."""
    person = pd.DataFrame(
        {
            "person_id": [1, 2, 3],
            "person_household_id": [1, 1, 2],
            "toy_taxable_income": list(incomes),
            "toy_is_exempt": list(exempt),
            "toy_child_count": list(children),
        }
    )
    household = pd.DataFrame(
        {"household_id": [1, 2], "toy_household_rent": list(rents)}
    )
    weights = {
        "household": Weights(values=np.array([1500.0, 900.0]), kind=WeightKind.DESIGN)
    }
    return Frame({"person": person, "household": household}, BE_SCHEMA, weights)


class TestLazyImport:
    def test_adapter_constructs_without_the_engine(self) -> None:
        adapter = AxiomEngine(FIXTURE_MODULE)
        assert isinstance(adapter, RulesEngine)

    def test_entity_schema_needs_no_engine(self) -> None:
        assert AxiomEngine(FIXTURE_MODULE).entity_schema() == BE_SCHEMA

    def test_export_contract_needs_no_engine(self) -> None:
        contract = ExportContract(
            required=("person_id",),
            forbidden=(),
            optional=(),
            formula_owned_excluded=(),
        )
        adapter = AxiomEngine(FIXTURE_MODULE, contract=contract)
        assert adapter.export_contract() is contract

    @pytest.mark.skipif(_ENGINE_INSTALLED, reason="axiom engine is installed here")
    def test_engine_methods_describe_installation_when_missing(self) -> None:
        adapter = AxiomEngine(FIXTURE_MODULE)
        with pytest.raises(ImportError, match="axiom-rules-engine"):
            adapter.variable_metadata("toy_income_tax")


class TestConstruction:
    def test_rejects_unknown_arithmetic(self) -> None:
        with pytest.raises(ValueError, match="arithmetic"):
            AxiomEngine(FIXTURE_MODULE, arithmetic="float32")

    def test_rejects_entity_names_outside_the_schema(self) -> None:
        with pytest.raises(ValueError, match="undeclared frame entit"):
            AxiomEngine(
                FIXTURE_MODULE,
                entity_names={"person": "Person", "tax_unit": "TaxUnit"},
            )

    def test_default_entity_names_capitalize(self) -> None:
        adapter = AxiomEngine(FIXTURE_MODULE)
        assert adapter._entity_names == {"person": "Person", "household": "Household"}


class TestPeriodBounds:
    def test_year_as_int_and_str(self) -> None:
        assert _period_bounds(2025) == ("2025-01-01", "2025-12-31", "calendar_year")
        assert _period_bounds("2025") == ("2025-01-01", "2025-12-31", "calendar_year")

    def test_month(self) -> None:
        assert _period_bounds("2025-02") == ("2025-02-01", "2025-02-28", "month")

    def test_rejects_other_shapes(self) -> None:
        with pytest.raises(ValueError, match="Unsupported period"):
            _period_bounds("2025-Q1")
        with pytest.raises(ValueError, match="Invalid month"):
            _period_bounds("2025-13")


@needs_engine
class TestVariableMetadata:
    @pytest.fixture(scope="class")
    def adapter(self) -> AxiomEngine:
        return AxiomEngine(FIXTURE_MODULE)

    def test_person_variable(self, adapter) -> None:
        meta = adapter.variable_metadata("toy_income_tax")
        assert meta.entity == "person"
        assert meta.dtype == "float"
        assert meta.period == "year"

    def test_household_variable(self, adapter) -> None:
        meta = adapter.variable_metadata("toy_housing_allowance")
        assert meta.entity == "household"
        assert meta.period == "year"

    def test_month_period_variable(self, adapter) -> None:
        assert adapter.variable_metadata("toy_monthly_benefit").period == "month"

    def test_unknown_variable_is_named(self, adapter) -> None:
        with pytest.raises(ValueError, match="not_a_variable"):
            adapter.variable_metadata("not_a_variable")

    def test_input_variable_refuses_fabricated_metadata(self, adapter) -> None:
        with pytest.raises(ValueError, match="input variable"):
            adapter.variable_metadata("toy_taxable_income")

    def test_variables_lists_inputs_not_outputs(self, adapter) -> None:
        names = adapter.variables()
        assert "toy_taxable_income" in names
        assert "toy_is_exempt" in names
        assert "toy_household_rent" in names  # household-scope input
        assert "toy_income_tax" not in names  # derived output
        assert names == sorted(names)


@needs_engine
class TestMaterialize:
    @pytest.fixture(scope="class")
    def adapter(self) -> AxiomEngine:
        return AxiomEngine(FIXTURE_MODULE)

    def test_person_values_row_aligned_and_hand_computed(self, adapter) -> None:
        bundle = _toy_bundle()
        results = adapter.materialize(bundle, ["toy_income_tax"], period=2025)
        # 5,000 * 10% = 500; 10,000 * 10% = 1,000;
        # 10,000 * 10% + 10,000 * 25% = 3,500.
        np.testing.assert_allclose(
            results["toy_income_tax"], [500.0, 1_000.0, 3_500.0]
        )

    def test_bool_column_drives_the_exemption_predicate(self, adapter) -> None:
        bundle = _toy_bundle(exempt=(True, False, True))
        results = adapter.materialize(bundle, ["toy_income_tax"], period=2025)
        np.testing.assert_allclose(results["toy_income_tax"], [0.0, 1_000.0, 0.0])

    def test_household_values_align_to_the_household_table(self, adapter) -> None:
        bundle = _toy_bundle()
        results = adapter.materialize(
            bundle, ["toy_income_tax", "toy_housing_allowance"], period=2025
        )
        assert results["toy_income_tax"].shape == (bundle.n("person"),)
        assert results["toy_housing_allowance"].shape == (bundle.n("household"),)
        np.testing.assert_allclose(
            results["toy_housing_allowance"], [1_200.0, 0.0]
        )

    def test_integer_column_feeds_count_inputs(self, adapter) -> None:
        bundle = _toy_bundle(children=(0, 2, 1))
        results = adapter.materialize(bundle, ["toy_monthly_benefit"], period=2025)
        np.testing.assert_allclose(results["toy_monthly_benefit"], [0.0, 200.0, 100.0])

    def test_f64_arithmetic_matches_decimal(self) -> None:
        fast = AxiomEngine(FIXTURE_MODULE, arithmetic="f64")
        bundle = _toy_bundle()
        results = fast.materialize(bundle, ["toy_income_tax"], period=2025)
        np.testing.assert_allclose(
            results["toy_income_tax"], [500.0, 1_000.0, 3_500.0]
        )

    def test_wrong_bundle_entities_are_refused(self, adapter) -> None:
        person = pd.DataFrame(
            {"person_id": [1], "person_family_id": [1], "toy_taxable_income": [1.0]}
        )
        family = pd.DataFrame({"family_id": [1]})
        bundle = Frame(
            {"person": person, "family": family},
            EntitySchema(group_entities=("family",)),
            {"family": Weights(values=np.array([1.0]), kind=WeightKind.DESIGN)},
        )
        with pytest.raises(ValueError, match="requires the schema entities"):
            adapter.materialize(bundle, ["toy_income_tax"], period=2025)

    def test_object_column_is_refused_with_the_column_named(self, adapter) -> None:
        bundle = _toy_bundle()
        person = bundle.table("person").copy()
        person["toy_taxable_income"] = person["toy_taxable_income"].astype(object)
        broken = Frame(
            {"person": person, "household": bundle.table("household")},
            BE_SCHEMA,
            {"household": bundle.weights_for("household")},
        )
        with pytest.raises(ValueError, match="toy_taxable_income"):
            adapter.materialize(broken, ["toy_income_tax"], period=2025)


@needs_engine
@needs_tables
class TestWriteDataset:
    def test_round_trips_and_carries_household_weight(self, tmp_path) -> None:
        adapter = AxiomEngine(FIXTURE_MODULE)
        bundle = _toy_bundle()
        path = tmp_path / "toy.h5"
        adapter.write_dataset(bundle, path, period=2025)
        reloaded = AxiomEntityTableDataset(file_path=path)
        assert reloaded.time_period == 2025
        assert reloaded.household["household_weight"].tolist() == [1500.0, 900.0]
        assert reloaded.person["toy_taxable_income"].tolist() == [
            5_000.0,
            10_000.0,
            20_000.0,
        ]

    def test_typed_weights_overwrite_a_stale_weight_column(self, tmp_path) -> None:
        adapter = AxiomEngine(FIXTURE_MODULE)
        bundle = _toy_bundle()
        household = bundle.table("household").copy()
        household["household_weight"] = [999.0, 999.0]
        stale = Frame(
            {"person": bundle.table("person"), "household": household},
            BE_SCHEMA,
            {"household": bundle.weights_for("household")},
        )
        path = tmp_path / "stale.h5"
        adapter.write_dataset(stale, path, period=2025)
        reloaded = AxiomEntityTableDataset(file_path=path)
        assert reloaded.household["household_weight"].tolist() == [1500.0, 900.0]

    def test_formula_owned_column_blocks_the_write(self, tmp_path) -> None:
        adapter = AxiomEngine(FIXTURE_MODULE)
        bundle = _toy_bundle()
        person = bundle.table("person").copy()
        person["toy_income_tax"] = [0.0, 0.0, 0.0]  # persisted engine output
        poisoned = Frame(
            {"person": person, "household": bundle.table("household")},
            BE_SCHEMA,
            {"household": bundle.weights_for("household")},
        )
        path = tmp_path / "poisoned.h5"
        with pytest.raises(ValueError, match="toy_income_tax"):
            adapter.write_dataset(poisoned, path, period=2025)
        assert not path.exists()

    def test_missing_required_column_blocks_the_write(self, tmp_path) -> None:
        contract = ExportContract(
            required=("definitely_absent",),
            forbidden=(),
            optional=(),
            formula_owned_excluded=(),
        )
        adapter = AxiomEngine(FIXTURE_MODULE, contract=contract)
        path = tmp_path / "missing.h5"
        with pytest.raises(ValueError, match="definitely_absent"):
            adapter.write_dataset(_toy_bundle(), path, period=2025)
        assert not path.exists()

    def test_defaults_broadcast_onto_the_owning_table(self, tmp_path) -> None:
        contract = ExportContract(
            required=("toy_default_flag",),
            forbidden=(),
            optional=(),
            formula_owned_excluded=(),
        )
        adapter = AxiomEngine(
            FIXTURE_MODULE, contract=contract, defaults={"toy_default_flag": 0.0}
        )
        path = tmp_path / "defaulted.h5"
        adapter.write_dataset(_toy_bundle(), path, period=2025)
        reloaded = AxiomEntityTableDataset(file_path=path)
        assert reloaded.person["toy_default_flag"].tolist() == [0.0, 0.0, 0.0]

    def test_closed_contract_rejects_unexpected_columns(self, tmp_path) -> None:
        contract = ExportContract(
            required=(),
            forbidden=(),
            optional=(),
            formula_owned_excluded=(),
            closed=True,
        )
        adapter = AxiomEngine(FIXTURE_MODULE, contract=contract)
        path = tmp_path / "closed.h5"
        with pytest.raises(ValueError, match="unexpected"):
            adapter.write_dataset(_toy_bundle(), path, period=2025)
        assert not path.exists()

    def test_forbidden_column_blocks_the_write(self, tmp_path) -> None:
        contract = ExportContract(
            required=(),
            forbidden=("toy_child_count",),
            optional=(),
            formula_owned_excluded=(),
        )
        adapter = AxiomEngine(FIXTURE_MODULE, contract=contract)
        path = tmp_path / "forbidden.h5"
        with pytest.raises(ValueError, match="toy_child_count"):
            adapter.write_dataset(_toy_bundle(), path, period=2025)
        assert not path.exists()

    def test_contract_formula_owned_exclusion_blocks_non_engine_column(
        self, tmp_path
    ) -> None:
        # legacy_output is not a derived rule of the module, so only the
        # contract's formula_owned_excluded list can catch it — the exact
        # case the shared contract tests never exercise.
        bundle = _toy_bundle()
        person = bundle.person.copy()
        person["legacy_output"] = [1.0, 0.0, 1.0]
        rebuilt = Frame(
            {"person": person, "household": bundle.table("household")},
            BE_SCHEMA,
            {"household": bundle.weights_for("household")},
        )
        contract = ExportContract(
            required=(),
            forbidden=(),
            optional=(),
            formula_owned_excluded=("legacy_output",),
        )
        adapter = AxiomEngine(FIXTURE_MODULE, contract=contract)
        path = tmp_path / "contract_formula_blocked.h5"
        with pytest.raises(ValueError, match="legacy_output"):
            adapter.write_dataset(rebuilt, path, period=2025)
        assert not path.exists()

    def test_formula_owned_exclusion_applies_under_a_closed_contract(
        self, tmp_path
    ) -> None:
        # Listing the column as optional in a closed contract must not
        # readmit it: formula_owned_excluded wins.
        bundle = _toy_bundle()
        person = bundle.person.copy()
        person["legacy_output"] = [1.0, 0.0, 1.0]
        rebuilt = Frame(
            {"person": person, "household": bundle.table("household")},
            BE_SCHEMA,
            {"household": bundle.weights_for("household")},
        )
        contract = ExportContract(
            required=(),
            forbidden=(),
            optional=(
                "toy_taxable_income",
                "toy_is_exempt",
                "toy_child_count",
                "toy_household_rent",
                "legacy_output",
            ),
            formula_owned_excluded=("legacy_output",),
            closed=True,
        )
        adapter = AxiomEngine(FIXTURE_MODULE, contract=contract)
        path = tmp_path / "closed_formula_blocked.h5"
        with pytest.raises(ValueError, match="legacy_output"):
            adapter.write_dataset(rebuilt, path, period=2025)
        assert not path.exists()


class TestAxiomEntityTableDataset:
    def test_requires_exactly_one_construction_mode(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="tables and time_period"):
            AxiomEntityTableDataset()
        with pytest.raises(ValueError, match="not both"):
            AxiomEntityTableDataset(
                tables={"person": pd.DataFrame({"person_id": [1]})},
                time_period=2025,
                file_path=tmp_path / "x.h5",
            )

    def test_missing_file_is_named(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match="nothing.h5"):
            AxiomEntityTableDataset(file_path=tmp_path / "nothing.h5")

    @needs_tables
    def test_save_and_reload_without_an_engine(self, tmp_path) -> None:
        tables = {
            "person": pd.DataFrame({"person_id": [1, 2], "age": [40.0, 8.0]}),
            "household": pd.DataFrame({"household_id": [1]}),
        }
        path = tmp_path / "plain.h5"
        AxiomEntityTableDataset(tables=tables, time_period=2025).save(path)
        reloaded = AxiomEntityTableDataset(file_path=path)
        assert reloaded.time_period == 2025
        assert reloaded.person["age"].tolist() == [40.0, 8.0]
        assert reloaded.table("household")["household_id"].tolist() == [1]

    def test_unknown_attribute_is_an_attribute_error(self, tmp_path) -> None:
        dataset = AxiomEntityTableDataset(
            tables={"person": pd.DataFrame({"person_id": [1]})}, time_period=2025
        )
        with pytest.raises(AttributeError, match="tax_unit"):
            _ = dataset.tax_unit


@needs_engine
@pytest.mark.skipif(
    RULESPEC_BE is None,
    reason="set POPULACE_RULESPEC_BE to a rulespec-be checkout to run",
)
class TestBelgianPilotSlice:
    """The microcosm#260 smoke test: real CIR 1992 rules, hand-computed values.

    2025 article 130 brackets: 25% to 16,320; 40% to 28,800; 45% to 49,840;
    50% above. Hand computation for taxable incomes 10,000 / 30,000 / 60,000:

    - 10,000 * 25% = 2,500
    - 16,320 * 25% + (30,000 - 16,320) * 40% = 4,080 + 5,472 ... exceeds the
      28,800 threshold, so: 4,080 + 12,480 * 40% + 1,200 * 45% = 9,612
    - 4,080 + 4,992 + 21,040 * 45% + 10,160 * 50% = 23,620
    """

    @pytest.fixture(scope="class")
    def adapter(self) -> AxiomEngine:
        module = (
            Path(RULESPEC_BE)
            / "be/statutes/income_tax/individual/rate_scale.yaml"
        )
        return AxiomEngine(module)

    def _be_bundle(self) -> Frame:
        person = pd.DataFrame(
            {
                "person_id": [1, 2, 3],
                "person_household_id": [1, 1, 2],
                "belgium_pit_taxable_income": [10_000.0, 30_000.0, 60_000.0],
            }
        )
        household = pd.DataFrame({"household_id": [1, 2]})
        weights = {
            "household": Weights(
                values=np.array([1200.0, 800.0]), kind=WeightKind.DESIGN
            )
        }
        return Frame({"person": person, "household": household}, BE_SCHEMA, weights)

    def test_article_130_base_tax_reproduces_hand_computed_values(
        self, adapter
    ) -> None:
        results = adapter.materialize(
            self._be_bundle(), ["belgium_pit_article_130_base_tax"], period=2025
        )
        np.testing.assert_allclose(
            results["belgium_pit_article_130_base_tax"],
            [2_500.0, 9_612.0, 23_620.0],
        )

    def test_metadata_resolves_the_statutory_variable(self, adapter) -> None:
        meta = adapter.variable_metadata("belgium_pit_article_130_base_tax")
        assert meta.entity == "person"
        assert meta.dtype == "float"
        assert meta.period == "year"

    def test_taxable_income_is_an_input(self, adapter) -> None:
        assert "belgium_pit_taxable_income" in adapter.variables()
