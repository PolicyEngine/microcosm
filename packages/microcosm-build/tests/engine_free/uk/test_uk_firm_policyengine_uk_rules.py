"""The policyengine-uk firm rules adapter, with an injected simulation factory.

No test imports policyengine-uk: a fake factory stands in for the
``Microsimulation`` and records what the adapter hands it. The schema snapshot
(``tests/fixtures/uk/policyengine_uk_firm_inputs.json``) says which firm
variables policyengine-uk has and which carry a formula.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from functools import cache
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.firm_policyengine_uk_rules import (
    POLICYENGINE_UK_FIRM_INPUT_COLUMNS,
    POLICYENGINE_UK_FIRM_LIABILITY_VARIABLE,
    POLICYENGINE_UK_FIRM_PULL_REQUEST,
    POLICYENGINE_UK_FIRM_REGISTERED_VARIABLE,
    PolicyEngineUKFirmRules,
    require_policyengine_uk_firm_rules,
)
from microcosm.build.uk_runtime.firm_rules import (
    UKFirmRuleOutputs,
    align_rule_outputs,
    rule_period_for_vintage,
)
from microcosm.build.uk_runtime.firm_table import build_candidate_rules_table
from test_support.paths import paths_for

SNAPSHOT = (
    paths_for("microcosm-build").tests
    / "fixtures"
    / "uk"
    / "policyengine_uk_firm_inputs.json"
)

MODULE = "microcosm.build.uk_runtime.firm_policyengine_uk_rules"


@cache
def snapshot() -> dict:
    return json.loads(SNAPSHOT.read_text(encoding="utf-8"))


def candidates() -> pd.DataFrame:
    firms = pd.DataFrame(
        {
            "firm_id": [70, 3, 512, 44, 8],
            "sic_code": [47, 62, 1, 86, 64],
            "annual_turnover_k": [120.0, 2_500.0, 30.0, 0.0, 61_000.0],
            "annual_input_k": [70.0, 3_100.0, 12.0, 4.0, 20_000.0],
            "employment": [3, 20, 0, 1, 450],
        }
    )
    return build_candidate_rules_table(firms)


def paper_liability(table: pd.DataFrame) -> np.ndarray:
    return 0.20 * table["firm_standard_rated_supplies"].to_numpy(np.float64) - table[
        "firm_input_vat"
    ].to_numpy(np.float64)


@dataclass
class FakeSimulation:
    table: pd.DataFrame
    liability: np.ndarray
    firm_id: np.ndarray
    reads: list = field(default_factory=list)

    def calculate(self, variable_name: str, period: int):
        self.reads.append((variable_name, period))
        if variable_name == POLICYENGINE_UK_FIRM_LIABILITY_VARIABLE:
            return self.liability
        if variable_name == "firm_id":
            return self.firm_id
        raise KeyError(variable_name)


@dataclass
class FakeFactory:
    """Records each call and answers with the paper formula unless told to
    return something else."""

    liability: np.ndarray | None = None
    firm_id: np.ndarray | None = None
    calls: list = field(default_factory=list)

    def __call__(self, table: pd.DataFrame, period: int) -> FakeSimulation:
        simulation = FakeSimulation(
            table=table,
            liability=(
                paper_liability(table) if self.liability is None else self.liability
            ),
            firm_id=(
                table["firm_id"].to_numpy() if self.firm_id is None else self.firm_id
            ),
        )
        self.calls.append((table, period, simulation))
        return simulation


def test_the_variables_it_reads_are_policyengine_uk_firm_formulas() -> None:
    variables = snapshot()["firm_variables"]
    for name in (
        POLICYENGINE_UK_FIRM_LIABILITY_VARIABLE,
        POLICYENGINE_UK_FIRM_REGISTERED_VARIABLE,
    ):
        assert name in variables
        assert variables[name]["has_formula"] is True
    assert POLICYENGINE_UK_FIRM_LIABILITY_VARIABLE == "firm_vat_liability_if_registered"
    assert POLICYENGINE_UK_FIRM_REGISTERED_VARIABLE == "firm_vat_registered"


def test_the_factory_gets_only_formula_less_firm_inputs() -> None:
    factory = FakeFactory()
    table = candidates().assign(
        firm_weight=1.0, **{POLICYENGINE_UK_FIRM_LIABILITY_VARIABLE: 0.0}
    )
    PolicyEngineUKFirmRules(simulation_factory=factory).evaluate(table, period=2024)

    (handed, _, _), *rest = factory.calls
    assert not rest
    inputs = set(snapshot()["firm_input_names"])
    assert tuple(handed.columns) == POLICYENGINE_UK_FIRM_INPUT_COLUMNS
    assert set(handed.columns) <= inputs
    # A column naming a formula variable would override the rule it reads.
    assert POLICYENGINE_UK_FIRM_LIABILITY_VARIABLE not in handed.columns


def test_the_period_is_forwarded_and_firm_ids_keep_their_rows() -> None:
    factory = FakeFactory()
    table = candidates().iloc[[3, 0, 4, 1, 2]]
    period = rule_period_for_vintage("2024-25")
    outputs = PolicyEngineUKFirmRules(simulation_factory=factory).evaluate(
        table, period=period
    )

    handed, handed_period, simulation = factory.calls[0]
    # The tax year's start year, passed through unchanged.
    assert handed_period == period == 2024
    assert handed["firm_id"].tolist() == table["firm_id"].tolist()
    assert {read for read, _ in simulation.reads} == {
        POLICYENGINE_UK_FIRM_LIABILITY_VARIABLE,
        "firm_id",
    }
    assert {at for _, at in simulation.reads} == {period}

    assert isinstance(outputs, UKFirmRuleOutputs)
    assert outputs.firm_id.tolist() == table["firm_id"].tolist()
    np.testing.assert_allclose(
        outputs.vat_liability_if_registered, paper_liability(table)
    )
    by_id = align_rule_outputs(outputs, candidates()["firm_id"])
    np.testing.assert_allclose(by_id, paper_liability(candidates()))


def test_provenance_names_the_rules_version_variable_and_period() -> None:
    outputs = PolicyEngineUKFirmRules(simulation_factory=FakeFactory()).evaluate(
        candidates(), period=2024
    )
    provenance = dict(outputs.provenance)
    assert provenance["rules"] == "policyengine_uk"
    assert provenance["variable"] == POLICYENGINE_UK_FIRM_LIABILITY_VARIABLE
    assert provenance["period"] == "2024"
    assert provenance["policyengine_uk_version"]


@pytest.mark.parametrize(
    "liability",
    [
        np.zeros(4),
        np.zeros(6),
        np.array([1.0, np.nan, 3.0, 4.0, 5.0]),
        np.array([1.0, 2.0, np.inf, 4.0, 5.0]),
    ],
    ids=["short", "long", "nan", "inf"],
)
def test_a_bad_liability_from_the_factory_raises(liability) -> None:
    rules = PolicyEngineUKFirmRules(simulation_factory=FakeFactory(liability=liability))
    with pytest.raises(ValueError, match=POLICYENGINE_UK_FIRM_LIABILITY_VARIABLE):
        rules.evaluate(candidates(), period=2024)


def test_firm_ids_out_of_order_in_the_simulation_raise() -> None:
    table = candidates()
    rules = PolicyEngineUKFirmRules(
        simulation_factory=FakeFactory(firm_id=table["firm_id"].to_numpy()[::-1])
    )
    with pytest.raises(ValueError, match="firm_id order"):
        rules.evaluate(table, period=2024)


def test_a_table_without_a_candidate_column_raises() -> None:
    rules = PolicyEngineUKFirmRules(simulation_factory=FakeFactory())
    with pytest.raises(ValueError, match="firm_input_vat"):
        rules.evaluate(candidates().drop(columns="firm_input_vat"), period=2024)


def fake_system(*, entities=("person", "benunit", "household", "firm"), variables=()):
    firm = SimpleNamespace(key="firm")
    return SimpleNamespace(
        entities=[SimpleNamespace(key=key) for key in entities],
        variables={name: SimpleNamespace(entity=firm) for name in variables},
    )


@pytest.mark.parametrize(
    ("system", "detail"),
    [
        (
            fake_system(entities=("person", "benunit", "household")),
            "no Firm entity",
        ),
        (
            fake_system(variables=("firm_id",)),
            f"no {POLICYENGINE_UK_FIRM_LIABILITY_VARIABLE} variable",
        ),
    ],
    ids=["no-firm-entity", "no-liability-variable"],
)
def test_a_policyengine_uk_without_firm_rules_names_the_pull_request(
    system, detail
) -> None:
    with pytest.raises(RuntimeError) as error:
        require_policyengine_uk_firm_rules(system)
    message = str(error.value)
    assert detail in message
    assert POLICYENGINE_UK_FIRM_PULL_REQUEST in message
    assert POLICYENGINE_UK_FIRM_PULL_REQUEST == "PolicyEngine/policyengine-uk#2940"


def test_a_dataset_without_a_firm_table_names_the_pull_request() -> None:
    system = fake_system(variables=("firm_id", POLICYENGINE_UK_FIRM_LIABILITY_VARIABLE))
    require_policyengine_uk_firm_rules(system)

    def old_dataset(person, benunit, household, fiscal_year=None):
        del person, benunit, household, fiscal_year

    def new_dataset(person, benunit, household, firm=None, fiscal_year=None):
        del person, benunit, household, firm, fiscal_year

    require_policyengine_uk_firm_rules(system, new_dataset)
    with pytest.raises(RuntimeError, match="takes no firm table.*#2940"):
        require_policyengine_uk_firm_rules(system, old_dataset)


def test_importing_the_adapter_leaves_policyengine_uk_unimported() -> None:
    script = (
        "import sys\n"
        f"import {MODULE}\n"
        "from microcosm.build.uk_runtime.firm_policyengine_uk_rules import "
        "PolicyEngineUKFirmRules\n"
        "PolicyEngineUKFirmRules()\n"
        "leaked = sorted(m for m in sys.modules "
        "if m.split('.')[0] in ('policyengine_uk', 'policyengine_core'))\n"
        "assert not leaked, leaked\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
