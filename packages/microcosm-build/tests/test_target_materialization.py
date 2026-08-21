import numpy as np

from microcosm.build.target_materialization import materialize_target_bindings
from microcosm.calibrate import TargetRegistry, TargetSpec


class StubAdapter:
    def __init__(self):
        self.tables = {
            "person": {
                "income": np.array([10.0, 20.0, 30.0]),
                "age": np.array([16.0, 30.0, 40.0]),
                "capital_gains": np.array([0.0, 5_000.0, 20_000.0]),
                "baseline_tax": np.array([1.0, 2.0, 3.0]),
            },
            "household": {
                "affected": np.array([1.0, 0.0, 1.0]),
                "children": np.array([3.0, 2.0, 4.0]),
            },
        }

    def column(self, entity, variable):
        return self.tables[entity][variable]

    def set_column(self, entity, variable, values):
        self.tables.setdefault(entity, {})[variable] = np.asarray(values)

    def parameter(self, name, period):
        assert name == "cgt.aea"
        assert period == 2025
        return 6_000.0

    def counterfactual_delta(self, binding, period):
        assert binding["zeroed_input"] == "salary_sacrifice"
        assert period == 2025
        return np.array([0.0, -1.0, -2.0])


def test_prepared_column_path_materializes_filtered_values():
    registry = TargetRegistry(
        [
            TargetSpec(
                name="adult_income",
                entity="person",
                measure="adult_income_measure",
                value=50.0,
                source="test",
                metadata={"contract_target_id": "adult_income"},
            )
        ],
        country="uk",
    )
    contract = {
        "adult_income": {
            "bindings": {
                "policyengine": {
                    "value_variable": "income",
                    "filters": [{"variable": "age", "operator": ">=", "value": 18}],
                }
            }
        }
    }
    adapter = StubAdapter()

    result = materialize_target_bindings(adapter, registry, contract, period=2025)

    assert result.skipped == ()
    assert adapter.tables["person"]["adult_income_measure"].tolist() == [
        0.0,
        20.0,
        30.0,
    ]


def test_generic_provider_kinds_materialize_expected_columns():
    registry = TargetRegistry(
        [
            TargetSpec(
                name="cgt_taxpayers",
                entity="person",
                measure="cgt_taxpayer_measure",
                value=1.0,
                source="test",
                metadata={"contract_target_id": "cgt_taxpayers"},
            ),
            TargetSpec(
                name="affected_children",
                entity="household",
                measure="affected_children_measure",
                value=7.0,
                source="test",
                metadata={"contract_target_id": "affected_children"},
            ),
            TargetSpec(
                name="salary_sacrifice",
                entity="person",
                measure="salary_sacrifice_delta",
                value=-3.0,
                source="test",
                signed=True,
                metadata={"contract_target_id": "salary_sacrifice"},
            ),
        ],
        country="uk",
    )
    contract = {
        "cgt_taxpayers": {
            "bindings": {
                "policyengine": {
                    "kind": "parameter_gated_threshold",
                    "gate_parameter": "cgt.aea",
                    "gate_comparison": ">",
                    "gated_variable": "capital_gains",
                    "value_variable": "person_count",
                }
            }
        },
        "affected_children": {
            "bindings": {
                "policyengine": {
                    "kind": "baseline_flag_crosstab",
                    "from_entity": "household",
                    "affected_flag_variable": "affected",
                    "count_of": "children",
                }
            }
        },
        "salary_sacrifice": {
            "bindings": {
                "policyengine": {
                    "kind": "input_substitution_counterfactual",
                    "zeroed_input": "salary_sacrifice",
                    "folded_into": "employment_income",
                    "output_variable": "income_tax",
                    "output_delta": "baseline_minus_reform",
                }
            }
        },
    }
    adapter = StubAdapter()

    result = materialize_target_bindings(adapter, registry, contract, period=2025)

    assert result.skipped == ()
    assert adapter.tables["person"]["cgt_taxpayer_measure"].tolist() == [0.0, 0.0, 1.0]
    assert adapter.tables["household"]["affected_children_measure"].tolist() == [
        3.0,
        0.0,
        4.0,
    ]
    assert adapter.tables["person"]["salary_sacrifice_delta"].tolist() == [
        0.0,
        -1.0,
        -2.0,
    ]


def test_missing_materialization_inputs_are_reported_as_skips():
    registry = TargetRegistry(
        [
            TargetSpec(
                name="missing",
                entity="person",
                measure="missing_measure",
                value=1.0,
                source="test",
                metadata={"contract_target_id": "missing"},
            )
        ],
        country="uk",
    )
    contract = {
        "missing": {"bindings": {"policyengine": {"value_variable": "not_present"}}}
    }

    result = materialize_target_bindings(StubAdapter(), registry, contract, period=2025)

    assert len(result.skipped) == 1
    assert result.skipped[0].name == "missing"
    assert result.report()["skipped_count"] == 1


def test_count_aliases_and_in_predicates_materialize_prepared_columns():
    registry = TargetRegistry(
        [
            TargetSpec(
                name="affected_households",
                entity="household",
                measure="affected_households_measure",
                value=2.0,
                source="test",
                metadata={"contract_target_id": "affected_households"},
            )
        ],
        country="uk",
    )
    contract = {
        "affected_households": {
            "bindings": {
                "policyengine": {
                    "value_variable": "household_count",
                    "filters": [
                        {
                            "variable": "children",
                            "operator": "in",
                            "value": [3.0, 4.0],
                        }
                    ],
                }
            }
        }
    }
    adapter = StubAdapter()

    result = materialize_target_bindings(adapter, registry, contract, period=2025)

    assert result.skipped == ()
    assert adapter.tables["household"]["affected_households_measure"].tolist() == [
        1.0,
        0.0,
        1.0,
    ]
