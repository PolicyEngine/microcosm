"""PolicyEngine-US adapter behavior against the real engine.

The whole module skips when ``policyengine_us`` is not installed (it is not
a workspace dependency; install via ``microframe[policyengine]`` to run
these).
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("policyengine_us")

from microframe import (  # noqa: E402
    US_SCHEMA,
    ExportContract,
    WeightedBundle,
    WeightKind,
    Weights,
    wsum,
)
from microframe.adapters.policyengine_us import PolicyEngineUSEngine  # noqa: E402


@pytest.fixture(scope="module")
def adapter() -> PolicyEngineUSEngine:
    return PolicyEngineUSEngine()


@pytest.fixture
def us_bundle() -> WeightedBundle:
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
            "employment_income": [70_000.0, 30_000.0, 48_000.0],
        }
    )
    group = lambda name: pd.DataFrame({f"{name}_id": [1, 2]})  # noqa: E731
    tables = {
        "person": person,
        "household": pd.DataFrame(
            {"household_id": [1, 2], "state_fips": [6, 36]}
        ),
        "tax_unit": group("tax_unit"),
        "spm_unit": group("spm_unit"),
        "family": group("family"),
        "marital_unit": group("marital_unit"),
    }
    weights = {
        "household": Weights(
            values=np.array([1500.0, 900.0]), kind=WeightKind.DESIGN
        )
    }
    return WeightedBundle(tables, US_SCHEMA, weights)


class TestVariableMetadata:
    def test_variable_entity(self, adapter) -> None:
        assert adapter.variable_entity("employment_income") == "person"
        assert adapter.variable_entity("household_net_income") == "household"

    def test_variable_dtype(self, adapter) -> None:
        assert adapter.variable_dtype("employment_income") is float

    def test_unknown_variable_is_named(self, adapter) -> None:
        with pytest.raises(ValueError, match="not_a_variable"):
            adapter.variable_entity("not_a_variable")


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
        assert reloaded.person["employment_income"].tolist() == [
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
            forbidden=("employment_income",),
            optional=(),
            formula_owned_excluded=(),
        )
        gated = PolicyEngineUSEngine(contract=contract)
        path = tmp_path / "gated.h5"
        with pytest.raises(ValueError, match="employment_income"):
            gated.write_dataset(us_bundle, path, period=2024)
        assert not path.exists()

    def test_defaults_broadcast_onto_owning_entity(
        self, us_bundle, tmp_path
    ) -> None:
        from policyengine_us.data import USSingleYearDataset

        contract = ExportContract(
            required=("snap_take_up_seed",),
            forbidden=(),
            optional=(),
            formula_owned_excluded=(),
        )
        defaulted = PolicyEngineUSEngine(
            contract=contract, defaults={"snap_take_up_seed": 0.5}
        )
        path = tmp_path / "defaulted.h5"
        defaulted.write_dataset(us_bundle, path, period=2024)
        reloaded = USSingleYearDataset(file_path=str(path))
        entity = PolicyEngineUSEngine().variable_entity("snap_take_up_seed")
        assert (getattr(reloaded, entity)["snap_take_up_seed"] == 0.5).all()

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
        rebuilt = WeightedBundle(
            {
                name: (person if name == "person" else us_bundle.table(name))
                for name in us_bundle.entities
            },
            US_SCHEMA,
            {"household": us_bundle.weights_for("household")},
        )
        # 1500*(70k + 30k) + 900*48k = 193,200,000
        assert wsum(rebuilt, "employment_income_2024") == 193_200_000.0
