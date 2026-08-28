import json
from pathlib import Path

import numpy as np

from microcosm.build.uk_runtime.ledger_targets import (
    compile_uk_target_registry,
    materialize_uk_ledger_targets,
)
from microcosm.calibrate import TargetRegistry, TargetSpec

FIXTURE_FEED_ROWS = (
    Path(__file__).parent / "fixtures" / "uk_target_reference_feed_rows.jsonl"
)


class StubUKAdapter:
    """Seven people across three households, with distinct child concepts.

    The affected flag and the child counts are deliberately distinct here:
    household 0 holds three affected children but four children in all, and
    household 2 holds two of each. A household-grain read of the boolean
    (1/0/1) can therefore be distinguished from both the affected-child counts
    (3/0/2) and the all-child counts (4/0/2).
    """

    def __init__(self):
        person_household = np.array([0, 0, 0, 0, 1, 2, 2])
        child_flags = np.array([True, True, True, False, False, True, True])
        self.tables = {
            "person": {
                "capital_gains": np.array(
                    [0.0, 0.0, 0.0, 5_000.0, 0.0, 0.0, 20_000.0]
                ),
                "person_household_id": person_household,
                "uc_is_child_limit_affected": child_flags,
                "is_child": np.array(
                    [True, True, True, True, False, True, True]
                ),
                "pip": np.array([0.0, 0.0, 0.0, 100.0, 100.0, 0.0, 0.0]),
            },
            "household": {
                "household_id": np.array([0, 1, 2]),
                # The household-grain flag is the any-collapse: "this
                # household contains at least one flagged child".
                "uc_is_child_limit_affected": np.array([1.0, 0.0, 1.0]),
            },
        }
    def column(self, entity, variable):
        return self.tables[entity][variable]

    def set_column(self, entity, variable, values):
        self.tables.setdefault(entity, {})[variable] = np.asarray(values)

    def entity_reduction(self, reduction):
        assert reduction["reduce"] == "sum"
        assert reduction["entity"] == "person"
        variable = reduction["variable"]
        values = self.tables["person"][variable].astype(float)
        households = self.tables["person"]["person_household_id"]
        return np.asarray(
            [
                values[households == household_id].sum()
                for household_id in self.tables["household"]["household_id"]
            ],
            dtype=float,
        )

    def household_condition(self, condition):
        assert condition["variable"] == "pip"
        assert condition["entity"] == "person"
        assert condition["reduce"] == "sum"
        assert condition["operator"] == ">"
        return self.entity_reduction(condition) > float(condition["value"])

    def parameter(self, name, period):
        assert name == "gov.hmrc.cgt.annual_exempt_amount"
        assert period == 2025
        return 6_000.0

    def counterfactual_delta(self, binding, period):
        return np.zeros(3)


def _fixture_rows() -> list[dict]:
    return [
        json.loads(line)
        for line in FIXTURE_FEED_ROWS.read_text().splitlines()
        if line.strip()
    ]


def test_compile_uk_target_registry_compiles_fixture_subset():
    result = compile_uk_target_registry(_fixture_rows(), target_period=2025)
    names = {spec.name for spec in result.registry.specs}

    assert {
        "obr.income_tax",
        "dwp.uc.two_child_limit.households_affected",
        "slc.repayments.england_plan_2",
        "hmrc.cgt.gains_total",
        "hmrc.cgt.taxpayers_total",
    } <= names
    assert result.unsupported


def test_materialize_uk_ledger_targets_with_stub_adapter():
    registry = TargetRegistry(
        [
            TargetSpec(
                name="hmrc.cgt.taxpayers_total",
                entity="person",
                measure="hmrc/cgt_taxpayers",
                value=378_000.0,
                source="test",
                metadata={"contract_target_id": "hmrc.cgt.taxpayers_total"},
            ),
            TargetSpec(
                name="dwp.uc.two_child_limit.children_affected",
                entity="household",
                measure="dwp/uc/two_child_limit/children_affected",
                value=1.0,
                source="test",
                metadata={
                    "contract_target_id": "dwp.uc.two_child_limit.children_affected"
                },
            ),
            TargetSpec(
                name="dwp.uc.two_child_limit.children_claimant_pip",
                entity="household",
                measure="dwp/uc/two_child_limit/adult_pip_children",
                value=1.0,
                source="test",
                metadata={
                    "contract_target_id": (
                        "dwp.uc.two_child_limit.children_claimant_pip"
                    )
                },
            ),
        ],
        country="uk",
    )
    adapter = StubUKAdapter()

    result = materialize_uk_ledger_targets(adapter, registry, period=2025)

    assert result.skipped == ()
    assert adapter.tables["person"]["hmrc/cgt_taxpayers"].tolist() == [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    # Child counts, not household indicators: the declared value_reduction
    # sums the flag over each household's people. A boolean any-collapse
    # would have published [1.0, 0.0, 1.0] against a count target.
    assert adapter.tables["household"][
        "dwp/uc/two_child_limit/children_affected"
    ].tolist() == [3.0, 0.0, 2.0]
    # Sheet 04B's claimant-PIP row counts every child in affected households
    # satisfying the PIP condition, not only the children carrying the
    # affected flag. Household 0 therefore contributes four, not three.
    assert adapter.tables["household"][
        "dwp/uc/two_child_limit/adult_pip_children"
    ].tolist() == [4.0, 0.0, 0.0]


def _composition_frame():
    """Two households: one lone parent with a child, one older couple."""

    import pandas as pd

    from microcosm.frame import EntitySchema, Frame, WeightKind, Weights

    return Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_id": np.arange(5),
                    "person_benunit_id": [0, 0, 1, 2, 2],
                    "person_household_id": [0, 0, 0, 1, 1],
                    "is_child": [0.0, 1.0, 0.0, 0.0, 0.0],
                    "age": [40.0, 8.0, 30.0, 70.0, 72.0],
                }
            ),
            "benunit": pd.DataFrame(
                {
                    "benunit_id": np.arange(3),
                    "family_type": ["LONE_PARENT", "SINGLE", "COUPLE_NO_CHILDREN"],
                }
            ),
            "household": pd.DataFrame({"household_id": np.arange(2)}),
        },
        EntitySchema(group_entities=("benunit", "household")),
        {"household": Weights(np.array([10.0, 20.0]), WeightKind.DESIGN)},
        metadata={"time_period": "2025"},
    )


def _uc_composition_frame():
    """Two multibenunit households that distinguish UC claim composition."""

    import pandas as pd

    from microcosm.frame import EntitySchema, Frame, WeightKind, Weights

    return Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_id": np.arange(6),
                    "person_benunit_id": [0, 0, 1, 2, 2, 3],
                    "person_household_id": [0, 0, 0, 1, 1, 1],
                    "is_child": [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                }
            ),
            "benunit": pd.DataFrame(
                {
                    "benunit_id": np.arange(4),
                    "family_type": [
                        "LONE_PARENT",
                        "SINGLE",
                        "LONE_PARENT",
                        "SINGLE",
                    ],
                    "universal_credit": [100.0, 0.0, 0.0, 100.0],
                    "num_children": [1, 0, 1, 0],
                }
            ),
            "household": pd.DataFrame({"household_id": np.arange(2)}),
        },
        EntitySchema(group_entities=("benunit", "household")),
        {"household": Weights(np.array([10.0, 20.0]), WeightKind.DESIGN)},
        metadata={"time_period": "2025"},
    )


def test_uc_composition_materializes_at_benunit_grain():
    from microcosm.build.uk_runtime.ledger_targets import UKFrameTargetAdapter

    registry = TargetRegistry(
        [
            TargetSpec(
                name="dwp.uc.households_children_1",
                entity="benunit",
                measure="dwp/uc/claimants_with_1_children",
                value=1.0,
                source="test",
                metadata={
                    "contract_target_id": "dwp.uc.households_children_1"
                },
            ),
            TargetSpec(
                name="dwp.uc.households_single_no_children",
                entity="benunit",
                measure="dwp/uc/claimants_single_no_children",
                value=1.0,
                source="test",
                metadata={
                    "contract_target_id": "dwp.uc.households_single_no_children"
                },
            ),
        ],
        country="uk",
    )
    adapter = UKFrameTargetAdapter(_uc_composition_frame())

    result = materialize_uk_ledger_targets(adapter, registry, period=2025)

    assert result.skipped == ()
    assert adapter.tables["benunit"][
        "dwp/uc/claimants_with_1_children"
    ].tolist() == [1.0, 0.0, 0.0, 0.0]
    # The non-UC single sharing household 0 with the claimant fails its own
    # filters. The childless UC single in household 1 counts even though the
    # other benunit in that dwelling contains a child.
    assert adapter.tables["benunit"][
        "dwp/uc/claimants_single_no_children"
    ].tolist() == [0.0, 0.0, 0.0, 1.0]


def test_household_condition_reduces_person_level_conditions():
    # Person-level conditions collapse through person_household_id. Building
    # the group-membership column here would ask for "person_person_id",
    # which cannot exist — the failure that excluded every ONS
    # household-composition reference from calibration.
    from microcosm.build.uk_runtime.ledger_targets import UKFrameTargetAdapter

    adapter = UKFrameTargetAdapter(_composition_frame())

    children = adapter.household_condition(
        {
            "variable": "is_child",
            "entity": "person",
            "reduce": "sum",
            "operator": ">=",
            "value": 1,
        }
    )

    assert list(children) == [True, False]


def test_household_condition_still_reduces_group_entities():
    from microcosm.build.uk_runtime.ledger_targets import UKFrameTargetAdapter

    adapter = UKFrameTargetAdapter(_composition_frame())

    single = adapter.household_condition(
        {
            "variable": "family_type",
            "entity": "benunit",
            "reduce": "any",
            "operator": "==",
            "value": "SINGLE",
        }
    )

    assert list(single) == [True, False]
