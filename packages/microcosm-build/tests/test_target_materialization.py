import numpy as np
import pandas as pd
import pytest

from microcosm.build.target_materialization import (
    MeasureResolutionError,
    assert_calibration_input_finite,
    materialize_target_bindings,
    resolve_target_measures,
)
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
        # Counts are adapter concerns (the UK adapter's convention): a
        # *_count variable is the all-ones indicator over the entity.
        if variable in {"person_count", "household_count", f"{entity}_count"}:
            first = next(iter(self.tables[entity].values()))
            return np.ones(len(first), dtype=float)
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


class ResolutionAdapter:
    def __init__(self, source):
        self.source = source
        self.tables = {entity: table.copy() for entity, table in source.items()}

    def column(self, entity, variable):
        if variable not in self.tables[entity]:
            raise KeyError(f"{entity}.{variable}")
        return np.asarray(self.tables[entity][variable])

    def set_column(self, entity, variable, values):
        self.tables[entity][variable] = np.asarray(values)


class StubMeasureProvider:
    contract_targets = {
        "needs_input_a": {
            "bindings": {"policyengine": {"value_variable": "input_a"}}
        },
        "needs_input_a_and_b": {
            "bindings": {
                "policyengine": {"value_expression": "input_a + input_b"}
            }
        },
        "needs_input_a_again": {
            "bindings": {"policyengine": {"value_variable": "input_a"}}
        },
    }

    def __init__(self):
        self.calls = []

    def knows(self, entity, variable):
        return (entity, variable) in {
            ("person", "input_a"),
            ("person", "input_b"),
        }

    def compute(self, entity, variable):
        self.calls.append((entity, variable))
        values = {
            ("person", "input_a"): np.array([1.0, 2.0, 3.0]),
            ("person", "input_b"): np.array([10.0, 20.0, 30.0]),
        }[(entity, variable)]
        return values, f"stub:{entity}.{variable}"

    def receipt(self):
        return {"provider": "stub"}


def _resolution_source():
    return {
        "person": pd.DataFrame(
            {
                "person_id": [1, 2, 3],
                "base": [5.0, 6.0, 7.0],
            }
        )
    }


def _resolution_registry(*names):
    specs = []
    for name in names:
        specs.append(
            TargetSpec(
                name=name,
                entity="person",
                measure=f"{name}_measure",
                value=1.0,
                source="test",
                metadata={"contract_target_id": name},
            )
        )
    return TargetRegistry(specs, country="uk")


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


def test_resolve_target_measures_converges_over_multiple_rounds_without_frame_mutation():
    source = _resolution_source()
    before = {entity: table.copy(deep=True) for entity, table in source.items()}
    provider = StubMeasureProvider()

    resolution = resolve_target_measures(
        lambda: ResolutionAdapter(source),
        _resolution_registry("needs_input_a_and_b"),
        provider,
        period=2025,
    )

    assert set(resolution.measure_inputs) == {
        ("person", "input_a"),
        ("person", "input_b"),
    }
    assert len(resolution.receipt["rounds"]) == 3
    assert resolution.receipt["attached"] == {
        "person.input_a": "stub:person.input_a",
        "person.input_b": "stub:person.input_b",
    }
    for entity, table in source.items():
        pd.testing.assert_frame_equal(table, before[entity])


def test_resolve_target_measures_computes_shared_missing_key_once_per_round():
    source = _resolution_source()
    provider = StubMeasureProvider()

    resolve_target_measures(
        lambda: ResolutionAdapter(source),
        _resolution_registry("needs_input_a", "needs_input_a_again"),
        provider,
        period=2025,
    )

    assert provider.calls == [("person", "input_a")]


def test_resolve_target_measures_raises_when_provided_key_still_fails():
    class IgnoringAdapter(ResolutionAdapter):
        def column(self, entity, variable):
            if variable == "input_a":
                raise KeyError(f"{entity}.{variable}")
            return super().column(entity, variable)

    provider = StubMeasureProvider()

    with pytest.raises(
        MeasureResolutionError, match="remained unmaterializable"
    ) as error:
        resolve_target_measures(
            lambda: IgnoringAdapter(_resolution_source()),
            _resolution_registry("needs_input_a"),
            provider,
            period=2025,
        )

    assert error.value.receipt["attached"] == {
        "person.input_a": "stub:person.input_a"
    }
    assert error.value.receipt["skips"][-1]["name"] == "needs_input_a"


def test_resolve_target_measures_raises_on_no_progress_and_max_rounds():
    provider = StubMeasureProvider()

    with pytest.raises(MeasureResolutionError, match="does not know"):
        resolve_target_measures(
            lambda: ResolutionAdapter(_resolution_source()),
            TargetRegistry(
                [
                    TargetSpec(
                        name="unknown",
                        entity="person",
                        measure="unknown_measure",
                        value=1.0,
                        source="test",
                        metadata={"contract_target_id": "unknown"},
                    )
                ],
                country="uk",
            ),
            provider,
            period=2025,
            contract_targets={
                "unknown": {
                    "bindings": {
                        "policyengine": {"value_variable": "not_provider_known"}
                    }
                }
            },
        )

    with pytest.raises(MeasureResolutionError, match="did not converge"):
        resolve_target_measures(
            lambda: ResolutionAdapter(_resolution_source()),
            _resolution_registry("needs_input_a"),
            provider,
            period=2025,
            max_rounds=0,
        )


def test_assert_calibration_input_finite_reports_every_nan_float_column():
    class FrameLike:
        entities = ("person", "household")

        def __init__(self):
            self._tables = {
                "person": pd.DataFrame(
                    {
                        "ok": [1.0, 2.0],
                        "bad": [np.nan, 3.0],
                        "also_bad": [np.nan, np.nan],
                        "label": ["x", None],
                    }
                ),
                "household": pd.DataFrame({"bad": [np.nan]}),
            }

        def table(self, entity):
            return self._tables[entity]

    with pytest.raises(ValueError) as error:
        assert_calibration_input_finite(FrameLike())

    message = str(error.value)
    assert "person.bad: 1" in message
    assert "person.also_bad: 2" in message
    assert "household.bad: 1" in message
    assert "person.label" not in message


def test_assert_calibration_input_finite_noops_on_clean_frame():
    class CleanFrame:
        entities = ("person",)

        def table(self, entity):
            assert entity == "person"
            return pd.DataFrame({"value": [1.0], "label": ["x"]})

    assert_calibration_input_finite(CleanFrame())


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


def _banded_registry(*, count_measure: bool = True) -> TargetRegistry:
    """Three adjacent income bands over one contract target."""

    specs = []
    for lower, label in ((0, "0"), (20, "20"), (40, "40")):
        specs.append(
            TargetSpec(
                name=f"band_{label}",
                entity="person",
                measure=f"income_band_{label}",
                value=1.0,
                source="test",
                family="hmrc_spi",
                metadata={
                    "contract_target_id": "spi.income_by_band",
                    "ledger_filter_total_income_lower_bound": str(lower),
                },
            )
        )
    return TargetRegistry(specs, country="uk")


_BANDED_CONTRACT = {
    "spi.income_by_band": {
        "bindings": {
            "policyengine": {
                "value_variable": "person_count",
                "groupby_variable": "income",
                "from_entity": "person",
            }
        }
    }
}


def test_bands_slice_the_population_and_partition_it():
    adapter = StubAdapter()
    registry = _banded_registry()

    result = materialize_target_bindings(
        adapter, registry, _BANDED_CONTRACT, period=2025
    )

    assert result.skipped == ()
    # income is [10, 20, 30]; bands are [0,20), [20,40), [40,inf).
    assert list(adapter.tables["person"]["income_band_0"]) == [1.0, 0.0, 0.0]
    assert list(adapter.tables["person"]["income_band_20"]) == [0.0, 1.0, 1.0]
    assert list(adapter.tables["person"]["income_band_40"]) == [0.0, 0.0, 0.0]
    # Every record lands in exactly one band: the bands partition the surface.
    total = sum(
        adapter.tables["person"][f"income_band_{label}"]
        for label in ("0", "20", "40")
    )
    assert list(total) == [1.0, 1.0, 1.0]


def test_adjacent_bands_are_not_identical():
    # The regression that would have caught the unsliced-measure defect:
    # before banding was implemented every band returned the same unsliced
    # column, so two adjacent bands compared equal.
    adapter = StubAdapter()
    materialize_target_bindings(
        adapter, _banded_registry(), _BANDED_CONTRACT, period=2025
    )

    assert not np.array_equal(
        adapter.tables["person"]["income_band_0"],
        adapter.tables["person"]["income_band_20"],
    )


def test_published_range_labels_band_in_model_units():
    adapter = StubAdapter()
    # Monthly published bands, annual model values: "£1.01 to £2.00" x12
    # covers [12.12, 24.12) and the sibling opens at 24.12.
    registry = TargetRegistry(
        [
            TargetSpec(
                name="award_low",
                entity="person",
                measure="award_low",
                value=1.0,
                source="test",
                family="dwp_universal_credit",
                metadata={
                    "contract_target_id": "uc.award_bands",
                    "ledger_filter_family_type": "Single, no children",
                    "ledger_filter_monthly_award_bands": "£1.01 to £2.00",
                },
            ),
            TargetSpec(
                name="award_high",
                entity="person",
                measure="award_high",
                value=1.0,
                source="test",
                family="dwp_universal_credit",
                metadata={
                    "contract_target_id": "uc.award_bands",
                    "ledger_filter_monthly_award_bands": "£2.01 to £3.00",
                },
            ),
        ],
        country="uk",
    )
    contract = {
        "uc.award_bands": {
            "bindings": {
                "policyengine": {
                    "value_variable": "person_count",
                    "groupby_variable": "income",
                    "from_entity": "person",
                    "band_period_factor": 12,
                }
            }
        }
    }

    result = materialize_target_bindings(adapter, registry, contract, period=2025)

    assert result.skipped == ()
    # income is [10, 20, 30]: only 20 falls inside [12.12, 24.12).
    assert list(adapter.tables["person"]["award_low"]) == [0.0, 1.0, 0.0]
    assert list(adapter.tables["person"]["award_high"]) == [0.0, 0.0, 1.0]


def test_unreadable_band_is_skipped_not_silently_unsliced():
    adapter = StubAdapter()
    registry = TargetRegistry(
        [
            TargetSpec(
                name="mystery_band",
                entity="person",
                measure="mystery_band",
                value=1.0,
                source="test",
                family="hmrc_spi",
                metadata={
                    "contract_target_id": "spi.income_by_band",
                    "ledger_filter_total_income_band": "not a range",
                },
            )
        ],
        country="uk",
    )

    result = materialize_target_bindings(
        adapter, registry, _BANDED_CONTRACT, period=2025
    )

    assert [skip.name for skip in result.skipped] == ["mystery_band"]
    assert "no readable band edge" in result.skipped[0].reason
    assert "mystery_band" not in adapter.tables["person"]


def test_entity_count_binding_yields_the_unit_indicator():
    # An entity-count value_variable with no count_of column counts records:
    # the shape every two-child-limit household reference needs. Those
    # references previously put a prose label in count_of, which the provider
    # then looked up as a column and failed on.
    adapter = StubAdapter()
    registry = TargetRegistry(
        [
            TargetSpec(
                name="affected",
                entity="household",
                measure="affected_households",
                value=1.0,
                source="test",
                family="dwp_two_child_limit",
                metadata={"contract_target_id": "tcl.households"},
            )
        ],
        country="uk",
    )
    contract = {
        "tcl.households": {
            "bindings": {
                "policyengine": {
                    "kind": "baseline_flag_crosstab",
                    "affected_flag_variable": "affected",
                    "value_variable": "household_count",
                    "from_entity": "household",
                }
            }
        }
    }

    result = materialize_target_bindings(adapter, registry, contract, period=2025)

    assert result.skipped == ()
    assert list(adapter.tables["household"]["affected_households"]) == [1.0, 0.0, 1.0]


def test_crosstab_counts_a_real_value_variable_per_record():
    adapter = StubAdapter()
    registry = TargetRegistry(
        [
            TargetSpec(
                name="children",
                entity="household",
                measure="affected_children",
                value=1.0,
                source="test",
                family="dwp_two_child_limit",
                metadata={"contract_target_id": "tcl.children"},
            )
        ],
        country="uk",
    )
    contract = {
        "tcl.children": {
            "bindings": {
                "policyengine": {
                    "kind": "baseline_flag_crosstab",
                    "affected_flag_variable": "affected",
                    "count_of": "affected_children",
                    "value_variable": "children",
                    "from_entity": "household",
                }
            }
        }
    }

    materialize_target_bindings(adapter, registry, contract, period=2025)

    # children is [3, 2, 4], masked by affected [1, 0, 1].
    assert list(adapter.tables["household"]["affected_children"]) == [3.0, 0.0, 4.0]
