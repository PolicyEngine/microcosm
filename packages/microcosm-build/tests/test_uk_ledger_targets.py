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
    def __init__(self):
        self.tables = {
            "person": {
                "capital_gains": np.array([0.0, 5_000.0, 20_000.0]),
            },
            "household": {
                "uc_is_child_limit_affected": np.array([1.0, 0.0, 1.0]),
                "children_count": np.array([3.0, 2.0, 4.0]),
            },
        }

    def column(self, entity, variable):
        return self.tables[entity][variable]

    def set_column(self, entity, variable, values):
        self.tables.setdefault(entity, {})[variable] = np.asarray(values)

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
        ],
        country="uk",
    )
    adapter = StubUKAdapter()

    result = materialize_uk_ledger_targets(adapter, registry, period=2025)

    assert result.skipped == ()
    assert adapter.tables["person"]["hmrc/cgt_taxpayers"].tolist() == [0.0, 0.0, 1.0]
    assert adapter.tables["household"][
        "dwp/uc/two_child_limit/children_affected"
    ].tolist() == [3.0, 0.0, 4.0]


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
