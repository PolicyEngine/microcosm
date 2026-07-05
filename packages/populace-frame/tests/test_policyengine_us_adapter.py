"""PolicyEngine-US adapter behavior against the real engine.

The whole module skips when ``policyengine_us`` is not installed (it is not
a workspace dependency; install via ``populace-frame[policyengine]`` to run
these).
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("policyengine_us")

from populace.frame import (  # noqa: E402
    US_SCHEMA,
    ExportContract,
    Frame,
    WeightKind,
    Weights,
    wsum,
)
from populace.frame.adapters.policyengine_us import PolicyEngineUSEngine  # noqa: E402


@pytest.fixture(scope="module")
def adapter() -> PolicyEngineUSEngine:
    return PolicyEngineUSEngine()


@pytest.fixture
def us_bundle() -> Frame:
    """Two households (a married couple, a single filer) with full US linkage."""
    person = pd.DataFrame(
        {
            "person_id": [1, 2, 3],
            "person_household_id": [1, 1, 2],
            "person_tax_unit_id": [1, 1, 2],
            "person_spm_unit_id": [1, 1, 2],
            "person_family_id": [1, 1, 2],
            "person_marital_unit_id": [1, 1, 2],
            "age": [40.0, 38.0, 33.0],
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


class TestVariableMetadata:
    def test_entity_dtype_period(self, adapter) -> None:
        meta = adapter.variable_metadata("employment_income")
        assert meta.entity == "person"
        assert meta.dtype == "float"
        assert meta.period == "year"
        assert adapter.variable_metadata("household_net_income").entity == "household"

    def test_enum_variable_reports_str_dtype(self, adapter) -> None:
        assert adapter.variable_metadata("filing_status").dtype == "str"

    def test_unknown_variable_is_named(self, adapter) -> None:
        with pytest.raises(ValueError, match="not_a_variable"):
            adapter.variable_metadata("not_a_variable")

    def test_variables_lists_inputs_not_outputs(self, adapter) -> None:
        names = adapter.variables()
        assert "employment_income_before_lsr" in names
        assert "employment_income" not in names  # adds-owned output
        assert "partnership_income" in names
        assert "s_corp_income" in names
        assert "partnership_self_employment_net_earnings" in names
        assert "partnership_s_corp_income" not in names
        assert "in_nyc" not in names
        assert "ssi" not in names  # formula-owned output
        assert "spm_unit_capped_work_childcare_expenses" not in names
        assert "income_tax" not in names  # formula-owned output

    def test_default_values_cover_scalar_bool_and_enum_inputs(self, adapter) -> None:
        defaults = adapter.default_values(
            [
                "weekly_hours_worked_before_lsr",
                "takes_up_snap_if_eligible",
                "ssn_card_type",
                "employment_income_before_lsr",
            ]
        )
        assert defaults["weekly_hours_worked_before_lsr"] == 40
        assert defaults["takes_up_snap_if_eligible"] is True
        # Enum defaults come back as the stored member name.
        assert defaults["ssn_card_type"] == "CITIZEN"
        assert defaults["employment_income_before_lsr"] == 0

    def test_default_values_skip_unknown_and_formula_owned_names(self, adapter) -> None:
        defaults = adapter.default_values(
            ["HRSWK", "not_a_variable", "income_tax", "snap"]
        )
        assert defaults == {}

    def test_formula_owned_outputs_flags_computed_names(self, adapter) -> None:
        # A requested output set mixing engine input leaves with formula-owned
        # aggregates: only the formula-owned names come back, so a caller can
        # reject them without maintaining a hand-written blocklist.
        flagged = adapter.formula_owned_outputs(
            [
                "employment_income_before_lsr",  # input leaf
                "qualified_dividend_income",  # input leaf
                "income_tax",  # formula (direct)
                "ssi",  # formula (mapping)
                "employment_income",  # adds-owned aggregate
                "self_employed_pension_contribution_ald",  # formula-owned ALD
            ]
        )
        assert flagged == {
            "income_tax",
            "ssi",
            "employment_income",
            "self_employed_pension_contribution_ald",
        }

    def test_formula_owned_outputs_flags_compat_aggregates(self, adapter) -> None:
        # The dividend/social-security/partnership aggregates the adapter refuses
        # to persist even when a published wheel still reports them as inputs.
        flagged = adapter.formula_owned_outputs(
            [
                "dividend_income",
                "social_security",
                "partnership_s_corp_income",
                "employment_income_before_lsr",
            ]
        )
        assert flagged == {
            "dividend_income",
            "social_security",
            "partnership_s_corp_income",
        }

    def test_formula_owned_outputs_ignores_unknown_names(self, adapter) -> None:
        # A name the engine does not know cannot be classified as formula-owned
        # here; the export/enum guards own unknown columns.
        assert adapter.formula_owned_outputs(["not_a_variable"]) == set()

    def test_formula_owned_outputs_is_disjoint_from_inputs(self, adapter) -> None:
        # The derived rejection set and the input surface partition the engine's
        # variables: nothing the engine accepts as an input is ever flagged.
        inputs = set(adapter.variables())
        flagged = adapter.formula_owned_outputs(inputs)
        assert flagged == set()


class TestMaterialize:
    def test_materializes_row_aligned_arrays(self, adapter, us_bundle) -> None:
        results = adapter.materialize(
            us_bundle,
            ["employment_income", "household_net_income"],
            period=2024,
        )
        assert results["employment_income"].shape == (us_bundle.n("person"),)
        np.testing.assert_allclose(
            results["employment_income"], [70_000.0, 30_000.0, 48_000.0]
        )
        assert results["household_net_income"].shape == (us_bundle.n("household"),)


class TestWriteDataset:
    def test_round_trips_and_carries_household_weight(
        self, adapter, us_bundle, tmp_path
    ) -> None:
        from policyengine_us.data import USSingleYearDataset

        path = tmp_path / "bundle.h5"
        adapter.write_dataset(us_bundle, path, period=2024)
        reloaded = USSingleYearDataset(file_path=str(path))
        assert reloaded.household["household_weight"].tolist() == [1500.0, 900.0]
        assert reloaded.person["employment_income_before_lsr"].tolist() == [
            70_000.0,
            30_000.0,
            48_000.0,
        ]

    def test_missing_required_column_blocks_the_write(
        self, us_bundle, tmp_path
    ) -> None:
        contract = ExportContract(
            required=("definitely_absent_column",),
            forbidden=(),
            optional=(),
            formula_owned_excluded=(),
        )
        gated = PolicyEngineUSEngine(contract=contract)
        path = tmp_path / "gated.h5"
        with pytest.raises(ValueError, match="definitely_absent_column"):
            gated.write_dataset(us_bundle, path, period=2024)
        assert not path.exists()

    def test_forbidden_column_blocks_the_write(self, us_bundle, tmp_path) -> None:
        contract = ExportContract(
            required=(),
            forbidden=("employment_income_before_lsr",),
            optional=(),
            formula_owned_excluded=(),
        )
        gated = PolicyEngineUSEngine(contract=contract)
        path = tmp_path / "gated.h5"
        with pytest.raises(ValueError, match="employment_income_before_lsr"):
            gated.write_dataset(us_bundle, path, period=2024)
        assert not path.exists()

    def test_formula_owned_columns_block_the_write(
        self, adapter, us_bundle, tmp_path
    ) -> None:
        person = us_bundle.person.copy()
        person["ssi"] = [12_000.0, 0.0, 0.0]
        rebuilt = Frame(
            {
                name: (person if name == "person" else us_bundle.table(name))
                for name in us_bundle.entities
            },
            US_SCHEMA,
            {"household": us_bundle.weights_for("household")},
        )

        path = tmp_path / "formula_blocked.h5"
        with pytest.raises(ValueError, match="ssi"):
            adapter.write_dataset(rebuilt, path, period=2024)
        assert not path.exists()

    def test_adds_owned_columns_block_the_write(
        self, adapter, us_bundle, tmp_path
    ) -> None:
        person = us_bundle.person.copy()
        person["employment_income"] = [70_000.0, 30_000.0, 48_000.0]
        rebuilt = Frame(
            {
                name: (person if name == "person" else us_bundle.table(name))
                for name in us_bundle.entities
            },
            US_SCHEMA,
            {"household": us_bundle.weights_for("household")},
        )

        path = tmp_path / "adds_owned_blocked.h5"
        with pytest.raises(ValueError, match="employment_income"):
            adapter.write_dataset(rebuilt, path, period=2024)
        assert not path.exists()

    def test_redundant_dividend_total_blocks_the_write(
        self, adapter, us_bundle, tmp_path
    ) -> None:
        person = us_bundle.person.copy()
        person["qualified_dividend_income"] = [100.0, 0.0, 40.0]
        person["non_qualified_dividend_income"] = [25.0, 0.0, 10.0]
        person["dividend_income"] = [125.0, 0.0, 50.0]
        rebuilt = Frame(
            {
                name: (person if name == "person" else us_bundle.table(name))
                for name in us_bundle.entities
            },
            US_SCHEMA,
            {"household": us_bundle.weights_for("household")},
        )

        path = tmp_path / "formula_blocked.h5"
        with pytest.raises(ValueError, match="dividend_income"):
            adapter.write_dataset(rebuilt, path, period=2024)
        assert not path.exists()

    def test_dividend_leaf_inputs_round_trip_without_total(
        self, adapter, us_bundle, tmp_path
    ) -> None:
        from policyengine_us.data import USSingleYearDataset

        person = us_bundle.person.copy()
        person["qualified_dividend_income"] = [100.0, 0.0, 40.0]
        person["non_qualified_dividend_income"] = [25.0, 0.0, 10.0]
        rebuilt = Frame(
            {
                name: (person if name == "person" else us_bundle.table(name))
                for name in us_bundle.entities
            },
            US_SCHEMA,
            {"household": us_bundle.weights_for("household")},
        )

        path = tmp_path / "dividend_leaves.h5"
        adapter.write_dataset(rebuilt, path, period=2024)
        reloaded = USSingleYearDataset(file_path=str(path))
        assert reloaded.person["qualified_dividend_income"].tolist() == [
            100.0,
            0.0,
            40.0,
        ]
        assert reloaded.person["non_qualified_dividend_income"].tolist() == [
            25.0,
            0.0,
            10.0,
        ]
        assert "dividend_income" not in reloaded.person.columns
        assert "ordinary_dividend_income" not in reloaded.person.columns

    def test_redundant_social_security_total_blocks_the_write(
        self, adapter, us_bundle, tmp_path
    ) -> None:
        person = us_bundle.person.copy()
        person["social_security_retirement"] = [10_000.0, 0.0, 5_000.0]
        person["social_security_disability"] = [0.0, 2_000.0, 0.0]
        person["social_security_survivors"] = [0.0, 0.0, 0.0]
        person["social_security_dependents"] = [0.0, 0.0, 0.0]
        person["social_security"] = [10_000.0, 2_000.0, 5_000.0]
        rebuilt = Frame(
            {
                name: (person if name == "person" else us_bundle.table(name))
                for name in us_bundle.entities
            },
            US_SCHEMA,
            {"household": us_bundle.weights_for("household")},
        )

        path = tmp_path / "social_security_aggregate.h5"
        with pytest.raises(ValueError, match="social_security"):
            adapter.write_dataset(rebuilt, path, period=2024)
        assert not path.exists()

    def test_partnership_s_corp_leaf_inputs_round_trip(
        self, adapter, us_bundle, tmp_path
    ) -> None:
        from policyengine_us.data import USSingleYearDataset

        person = us_bundle.person.copy()
        person["partnership_income"] = [1_500.0, 0.0, 400.0]
        person["s_corp_income"] = [500.0, 0.0, 100.0]
        person["partnership_self_employment_net_earnings"] = [900.0, 0.0, 250.0]
        rebuilt = Frame(
            {
                name: (person if name == "person" else us_bundle.table(name))
                for name in us_bundle.entities
            },
            US_SCHEMA,
            {"household": us_bundle.weights_for("household")},
        )

        path = tmp_path / "partnership_s_corp_inputs.h5"
        adapter.write_dataset(rebuilt, path, period=2024)
        reloaded = USSingleYearDataset(file_path=str(path))
        assert reloaded.person["partnership_income"].tolist() == [1_500.0, 0.0, 400.0]
        assert reloaded.person["s_corp_income"].tolist() == [500.0, 0.0, 100.0]
        assert "partnership_s_corp_income" not in reloaded.person.columns
        assert reloaded.person["partnership_self_employment_net_earnings"].tolist() == [
            900.0,
            0.0,
            250.0,
        ]

    def test_partnership_s_corp_aggregate_blocks_the_write(
        self, adapter, us_bundle, tmp_path
    ) -> None:
        person = us_bundle.person.copy()
        person["partnership_s_corp_income"] = [2_000.0, 0.0, 500.0]
        rebuilt = Frame(
            {
                name: (person if name == "person" else us_bundle.table(name))
                for name in us_bundle.entities
            },
            US_SCHEMA,
            {"household": us_bundle.weights_for("household")},
        )

        path = tmp_path / "partnership_s_corp_aggregate.h5"
        with pytest.raises(ValueError, match="partnership_s_corp_income"):
            adapter.write_dataset(rebuilt, path, period=2024)
        assert not path.exists()

    def test_enum_domain_column_blocks_raw_source_codes(
        self, adapter, us_bundle, tmp_path
    ) -> None:
        person = us_bundle.person.copy()
        person["race"] = [0, 1, 10]
        rebuilt = Frame(
            {
                name: (person if name == "person" else us_bundle.table(name))
                for name in us_bundle.entities
            },
            US_SCHEMA,
            {"household": us_bundle.weights_for("household")},
        )

        path = tmp_path / "enum_blocked.h5"
        with pytest.raises(ValueError, match="enum-domain.*race"):
            adapter.write_dataset(rebuilt, path, period=2024)
        assert not path.exists()

    def test_enum_domain_names_round_trip_for_true_input(
        self, adapter, us_bundle, tmp_path
    ) -> None:
        from policyengine_us.data import USSingleYearDataset

        household = us_bundle.table("household").copy()
        household["tenure_type"] = ["RENTED", "OWNED_WITH_MORTGAGE"]
        rebuilt = Frame(
            {
                name: (household if name == "household" else us_bundle.table(name))
                for name in us_bundle.entities
            },
            US_SCHEMA,
            {"household": us_bundle.weights_for("household")},
        )

        path = tmp_path / "enum_valid.h5"
        adapter.write_dataset(rebuilt, path, period=2024)
        reloaded = USSingleYearDataset(file_path=str(path))
        assert reloaded.household["tenure_type"].tolist() == [
            "RENTED",
            "OWNED_WITH_MORTGAGE",
        ]

    def test_in_nyc_formula_owned_column_blocks_the_write(
        self, adapter, us_bundle, tmp_path
    ) -> None:
        household = us_bundle.table("household").copy()
        household["in_nyc"] = [False, True]
        rebuilt = Frame(
            {
                name: (household if name == "household" else us_bundle.table(name))
                for name in us_bundle.entities
            },
            US_SCHEMA,
            {"household": us_bundle.weights_for("household")},
        )

        path = tmp_path / "formula_blocked.h5"
        with pytest.raises(ValueError, match="in_nyc"):
            adapter.write_dataset(rebuilt, path, period=2024)
        assert not path.exists()

    def test_spm_unit_formula_owned_column_blocks_the_write(
        self, adapter, us_bundle, tmp_path
    ) -> None:
        spm_unit = us_bundle.table("spm_unit").copy()
        spm_unit["spm_unit_capped_work_childcare_expenses"] = [3_000.0, 0.0]
        rebuilt = Frame(
            {
                name: (spm_unit if name == "spm_unit" else us_bundle.table(name))
                for name in us_bundle.entities
            },
            US_SCHEMA,
            {"household": us_bundle.weights_for("household")},
        )

        path = tmp_path / "formula_blocked.h5"
        with pytest.raises(ValueError, match="spm_unit_capped_work_childcare_expenses"):
            adapter.write_dataset(rebuilt, path, period=2024)
        assert not path.exists()

    def test_contract_listed_formula_owned_columns_still_block_the_write(
        self, us_bundle, tmp_path
    ) -> None:
        person = us_bundle.person.copy()
        person["ssi"] = [12_000.0, 0.0, 0.0]
        rebuilt = Frame(
            {
                name: (person if name == "person" else us_bundle.table(name))
                for name in us_bundle.entities
            },
            US_SCHEMA,
            {"household": us_bundle.weights_for("household")},
        )
        adapter = PolicyEngineUSEngine(
            contract=ExportContract(
                required=(),
                forbidden=(),
                optional=(),
                formula_owned_excluded=("ssi",),
            )
        )

        path = tmp_path / "formula_blocked.h5"
        with pytest.raises(ValueError, match="ssi"):
            adapter.write_dataset(rebuilt, path, period=2024)
        assert not path.exists()

    def test_contract_formula_owned_exclusion_blocks_non_engine_column(
        self, us_bundle, tmp_path
    ) -> None:
        person = us_bundle.person.copy()
        person["legacy_formula_output"] = [1.0, 0.0, 1.0]
        rebuilt = Frame(
            {
                name: (person if name == "person" else us_bundle.table(name))
                for name in us_bundle.entities
            },
            US_SCHEMA,
            {"household": us_bundle.weights_for("household")},
        )
        adapter = PolicyEngineUSEngine(
            contract=ExportContract(
                required=(),
                forbidden=(),
                optional=(),
                formula_owned_excluded=("legacy_formula_output",),
            )
        )

        path = tmp_path / "contract_formula_blocked.h5"
        with pytest.raises(ValueError, match="legacy_formula_output"):
            adapter.write_dataset(rebuilt, path, period=2024)
        assert not path.exists()

    def test_future_formula_does_not_block_current_period_input(
        self, adapter, us_bundle, tmp_path
    ) -> None:
        from policyengine_us.data import USSingleYearDataset

        person = us_bundle.person.copy()
        person["weeks_worked"] = [52, 26, 40]
        rebuilt = Frame(
            {
                name: (person if name == "person" else us_bundle.table(name))
                for name in us_bundle.entities
            },
            US_SCHEMA,
            {"household": us_bundle.weights_for("household")},
        )

        path = tmp_path / "future_formula_input.h5"
        adapter.write_dataset(rebuilt, path, period=2024)

        reloaded = USSingleYearDataset(file_path=str(path))
        assert reloaded.person["weeks_worked"].tolist() == [52, 26, 40]

    def test_closed_contract_blocks_unexpected_columns(
        self, us_bundle, tmp_path
    ) -> None:
        person = us_bundle.person.copy()
        person["internal_bookkeeping"] = [1.0, 0.0, 1.0]
        rebuilt = Frame(
            {
                name: (person if name == "person" else us_bundle.table(name))
                for name in us_bundle.entities
            },
            US_SCHEMA,
            {"household": us_bundle.weights_for("household")},
        )
        gated = PolicyEngineUSEngine(
            contract=ExportContract(
                required=(),
                forbidden=(),
                optional=("age", "employment_income", "state_fips"),
                formula_owned_excluded=(),
                closed=True,
            )
        )

        path = tmp_path / "closed.h5"
        with pytest.raises(ValueError, match="internal_bookkeeping"):
            gated.write_dataset(rebuilt, path, period=2024)
        assert not path.exists()

    def test_defaults_broadcast_onto_owning_entity(self, us_bundle, tmp_path) -> None:
        from policyengine_us.data import USSingleYearDataset

        contract = ExportContract(
            required=("takes_up_snap_if_eligible",),
            forbidden=(),
            optional=(),
            formula_owned_excluded=(),
        )
        defaulted = PolicyEngineUSEngine(
            contract=contract, defaults={"takes_up_snap_if_eligible": True}
        )
        path = tmp_path / "defaulted.h5"
        defaulted.write_dataset(us_bundle, path, period=2024)
        reloaded = USSingleYearDataset(file_path=str(path))
        meta = PolicyEngineUSEngine().variable_metadata("takes_up_snap_if_eligible")
        assert meta.entity == "spm_unit"
        assert reloaded.spm_unit["takes_up_snap_if_eligible"].all()

    def test_non_h5_path_is_rejected(self, adapter, us_bundle, tmp_path) -> None:
        with pytest.raises(ValueError, match=r"\.h5"):
            adapter.write_dataset(us_bundle, tmp_path / "bundle.csv", period=2024)


class TestEndToEnd:
    def test_weighted_accounting_matches_materialized_engine_totals(
        self, adapter, us_bundle
    ) -> None:
        """Materialized variables drop back onto the bundle and aggregate."""
        results = adapter.materialize(us_bundle, ["employment_income"], period=2024)
        person = us_bundle.person.copy()
        person["employment_income_2024"] = results["employment_income"]
        rebuilt = Frame(
            {
                name: (person if name == "person" else us_bundle.table(name))
                for name in us_bundle.entities
            },
            US_SCHEMA,
            {"household": us_bundle.weights_for("household")},
        )
        # 1500*(70k + 30k) + 900*48k = 193,200,000
        assert wsum(rebuilt, "employment_income_2024") == 193_200_000.0


class TestTakeUpContract:
    """Take-up variable discovery and classification (populace #312)."""

    def test_take_up_variables_are_discovered(self, adapter) -> None:
        names = adapter.take_up_variables()
        # Known take-up flags in the pinned engine.
        assert "takes_up_snap_if_eligible" in names
        assert "takes_up_eitc" in names
        assert "takes_up_tanf_if_eligible" in names
        # All discovered names carry the take-up marker.
        assert all(
            name.startswith("takes_up") or "take_up_seed" in name for name in names
        )
        assert names == sorted(names)

    def test_contract_covers_every_take_up_variable(self, adapter) -> None:
        contract = adapter.take_up_contract()
        assert set(contract) == set(adapter.take_up_variables())

    def test_data_seeded_flag_is_classified(self, adapter) -> None:
        # SNAP take-up has no formula and a True default -> data_seeded, and is
        # consumed by the snap formula.
        info = adapter.take_up_contract()["takes_up_snap_if_eligible"]
        assert info["engine_class"] == "data_seeded"
        assert info["default"] is True
        assert info["entity"] == "spm_unit"
        assert info["value_type"] == "bool"
        assert not info["engine_computed"]
        assert "snap" in info["consumers"]

    def test_enrolled_adds_consumer_is_detected(self, adapter) -> None:
        # Medicaid take-up is read through medicaid_enrolled's `adds`, not a
        # formula body -- the classifier must still see it as consumed.
        info = adapter.take_up_contract()["takes_up_medicaid_if_eligible"]
        assert "medicaid_enrolled" in info["consumers"]
        assert info["engine_class"] == "data_seeded"

    def test_no_take_up_flag_is_dead_in_pinned_engine(self, adapter) -> None:
        contract = adapter.take_up_contract()
        dead = [n for n, info in contract.items() if info["engine_class"] == "dead"]
        assert dead == [], f"unexpected dead take-up flags: {dead}"
