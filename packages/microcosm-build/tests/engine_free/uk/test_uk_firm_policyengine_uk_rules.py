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
    POLICYENGINE_UK_FIRM_INFORMATIONAL_VARIABLES,
    POLICYENGINE_UK_FIRM_INPUT_COLUMNS,
    POLICYENGINE_UK_FIRM_LIABILITY_VARIABLE,
    POLICYENGINE_UK_FIRM_PULL_REQUEST,
    POLICYENGINE_UK_FIRM_REGISTERED_VARIABLE,
    POLICYENGINE_UK_FIRM_TAX_VARIABLES,
    POLICYENGINE_UK_FIRM_TAXES,
    PolicyEngineUKFirmRules,
    evaluate_uk_firm_taxes,
    require_policyengine_uk_firm_rules,
    weighted_uk_firm_tax_totals,
)
from microcosm.build.uk_runtime.firm_rules import (
    UKFirmRuleOutputs,
    align_rule_outputs,
    rule_period_for_vintage,
)
from microcosm.build.uk_runtime.firm_table import (
    UK_FIRM_MICROCOSM_ONLY_COLUMNS,
    UK_FIRM_POLICYENGINE_UK_INPUT_STATUS,
    build_candidate_rules_table,
)
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


# evaluate_uk_firm_taxes: every firm tax from the exported firm table.


def export_table() -> pd.DataFrame:
    """Rows shaped like the extended ``uk_firms.parquet`` export, with
    microcosm-only columns after the policyengine-uk ones."""

    return pd.DataFrame(
        {
            "firm_id": [5, 2, 9],
            "firm_weight": [10.0, 2.5, 1.0],
            "firm_turnover": [100_000.0, 2_000_000.0, 50_000.0],
            "firm_input_costs": [40_000.0, 900_000.0, 10_000.0],
            "firm_employees": [2, 30, 0],
            "firm_sic_section": ["G", "J", "A"],
            "firm_standard_rated_supplies": [100_000.0, 2_000_000.0, 0.0],
            "firm_exempt_supplies": [0.0, 0.0, 50_000.0],
            "firm_vat_voluntarily_registered": [False, False, False],
            "firm_input_vat": [8_000.0, 180_000.0, 2_000.0],
            "firm_legal_form": ["COMPANY", "COMPANY", "SOLE_TRADER"],
            "firm_region": ["LONDON", "NORTH_WEST", "WALES"],
            "firm_in_england": [True, True, False],
            "firm_is_public_body": [False, False, False],
            "in_frame": [True, True, True],
            "vat_scope": [True, True, False],
            "vat_registered": [True, True, False],
            "base_weight": [9.0, 3.0, 1.0],
            "data_vintage": ["2024-25"] * 3,
            "source_firm_id": [1, 2, 3],
            "source_firm_key": ["a", "b", "c"],
            "ons_legal_status": [1, 1, 2],
            "ons_region_code": ["E12000007", "E12000002", "W92000004"],
        }
    )


def tax_value(name: str, firm_id: np.ndarray) -> np.ndarray:
    """A distinct, deterministic value per variable and firm."""

    offset = 1_000.0 * (1 + POLICYENGINE_UK_FIRM_TAX_VARIABLES.index(name))
    return offset + firm_id.astype(np.float64)


@dataclass
class FakeTaxSimulation:
    table: pd.DataFrame
    overrides: dict
    reads: list = field(default_factory=list)

    def calculate(self, variable_name: str, period: int):
        self.reads.append((variable_name, period))
        if variable_name in self.overrides:
            return self.overrides[variable_name]
        firm_id = self.table["firm_id"].to_numpy()
        if variable_name == "firm_id":
            return firm_id
        if variable_name in POLICYENGINE_UK_FIRM_TAX_VARIABLES:
            # float32, as policyengine-uk stores firm amounts.
            return tax_value(variable_name, firm_id).astype(np.float32)
        raise KeyError(variable_name)


@dataclass
class FakeTaxFactory:
    overrides: dict = field(default_factory=dict)
    calls: list = field(default_factory=list)

    def __call__(self, table: pd.DataFrame, period: int) -> FakeTaxSimulation:
        simulation = FakeTaxSimulation(table=table, overrides=self.overrides)
        self.calls.append((table, period, simulation))
        return simulation


def test_the_tax_variables_are_policyengine_uk_firm_gbp_formulas() -> None:
    assert POLICYENGINE_UK_FIRM_TAXES == (
        "firm_vat",
        "firm_employer_ni",
        "apprenticeship_levy",
        "corporation_tax",
        "bank_surcharge",
        "firm_business_rates",
    )
    assert POLICYENGINE_UK_FIRM_INFORMATIONAL_VARIABLES == (
        "employment_allowance",
        "firm_auto_enrolment_employer_contribution",
    )
    assert POLICYENGINE_UK_FIRM_TAX_VARIABLES == (
        *POLICYENGINE_UK_FIRM_TAXES,
        *POLICYENGINE_UK_FIRM_INFORMATIONAL_VARIABLES,
    )
    variables = snapshot()["firm_variables"]
    for name in POLICYENGINE_UK_FIRM_TAX_VARIABLES:
        assert variables[name]["has_formula"] is True, name
        assert variables[name]["unit"] == "GBP", name
        assert variables[name]["value_type"] == "float", name
        assert variables[name]["definition_period"] == "year", name


def test_only_policyengine_uk_firm_inputs_reach_the_factory() -> None:
    factory = FakeTaxFactory()
    # A column named after an output would override the rule that computes it.
    table = export_table().assign(firm_vat=0.0, corporation_tax=0.0)
    with pytest.warns(UserWarning, match=r"drops firm columns \['firm_vat'\]"):
        evaluate_uk_firm_taxes(table, period=2024, simulation_factory=factory)

    (handed, period, simulation), *rest = factory.calls
    assert not rest
    assert period == 2024
    inputs = set(UK_FIRM_POLICYENGINE_UK_INPUT_STATUS)
    assert set(handed.columns) <= inputs
    assert set(handed.columns) <= set(snapshot()["firm_input_names"])
    assert not set(handed.columns) & set(UK_FIRM_MICROCOSM_ONLY_COLUMNS)
    assert not set(handed.columns) & set(POLICYENGINE_UK_FIRM_TAX_VARIABLES)
    # Every policyengine-uk input in the table is kept, in the table's order.
    assert list(handed.columns) == [c for c in table.columns if c in inputs]
    assert "firm_region" in handed.columns
    assert "firm_legal_form" in handed.columns
    assert list(handed.index) == list(range(len(table)))
    assert {at for _, at in simulation.reads} == {2024}
    assert {read for read, _ in simulation.reads} == {
        "firm_id",
        *POLICYENGINE_UK_FIRM_TAX_VARIABLES,
    }


def test_an_input_the_export_leaves_unset_passes_through_when_given() -> None:
    table = export_table()
    unset = sorted(
        name
        for name, status in UK_FIRM_POLICYENGINE_UK_INPUT_STATUS.items()
        if status == "not_set" and name not in table.columns
    )
    assert unset
    factory = FakeTaxFactory()
    table = table.assign(**{unset[0]: [0.0, 1.0, 0.0]})
    evaluate_uk_firm_taxes(table, period=2024, simulation_factory=factory)
    handed = factory.calls[0][0]
    assert handed[unset[0]].tolist() == [0.0, 1.0, 0.0]


def test_taxes_are_keyed_by_firm_id_in_input_order() -> None:
    table = export_table().iloc[[2, 0, 1]].set_axis([7, 3, 11])
    taxes = evaluate_uk_firm_taxes(
        table, period=2024, simulation_factory=FakeTaxFactory()
    )

    assert list(taxes.columns) == [
        "firm_id",
        "firm_weight",
        *POLICYENGINE_UK_FIRM_TAX_VARIABLES,
    ]
    assert list(taxes.index) == [0, 1, 2]
    assert taxes["firm_id"].dtype == np.int64
    assert taxes["firm_id"].tolist() == [9, 5, 2]
    assert taxes["firm_weight"].tolist() == [1.0, 10.0, 2.5]
    for name in POLICYENGINE_UK_FIRM_TAX_VARIABLES:
        assert taxes[name].dtype == np.float64
        np.testing.assert_array_equal(
            taxes[name].to_numpy(), tax_value(name, np.array([9, 5, 2]))
        )


def test_a_table_without_weights_returns_no_weight_column() -> None:
    taxes = evaluate_uk_firm_taxes(
        export_table().drop(columns="firm_weight"),
        period=2024,
        simulation_factory=FakeTaxFactory(),
        variables=("firm_vat", "corporation_tax"),
    )
    assert list(taxes.columns) == ["firm_id", "firm_vat", "corporation_tax"]
    with pytest.raises(ValueError, match="firm_weight"):
        weighted_uk_firm_tax_totals(taxes)


def test_a_table_without_firm_id_raises() -> None:
    with pytest.raises(ValueError, match="firm_id"):
        evaluate_uk_firm_taxes(
            export_table().drop(columns="firm_id"),
            period=2024,
            simulation_factory=FakeTaxFactory(),
        )


@pytest.mark.parametrize(
    "variables",
    [(), ("firm_vat", "firm_vat"), ("firm_vat", "firm_weight"), ("firm_id",)],
    ids=["empty", "repeated", "weight", "id"],
)
def test_bad_variable_lists_raise(variables) -> None:
    with pytest.raises(ValueError):
        evaluate_uk_firm_taxes(
            export_table(),
            period=2024,
            simulation_factory=FakeTaxFactory(),
            variables=variables,
        )


def test_firm_ids_out_of_order_in_the_tax_simulation_raise() -> None:
    reversed_ids = export_table()["firm_id"].to_numpy()[::-1]
    factory = FakeTaxFactory(overrides={"firm_id": reversed_ids})
    with pytest.raises(ValueError, match="firm_id order"):
        evaluate_uk_firm_taxes(export_table(), period=2024, simulation_factory=factory)


@pytest.mark.parametrize(
    "values",
    [
        np.zeros(2),
        np.zeros(4),
        np.zeros((3, 1)),
        np.array([1.0, np.nan, 3.0]),
        np.array([1.0, 2.0, -np.inf]),
    ],
    ids=["short", "long", "2d", "nan", "inf"],
)
def test_a_bad_tax_from_the_factory_raises(values) -> None:
    factory = FakeTaxFactory(overrides={"corporation_tax": values})
    with pytest.raises(ValueError, match="corporation_tax"):
        evaluate_uk_firm_taxes(export_table(), period=2024, simulation_factory=factory)


def test_weighted_totals_multiply_by_firm_weight() -> None:
    taxes = pd.DataFrame(
        {
            "firm_id": np.array([1, 2, 3], dtype=np.int64),
            "firm_weight": [2.0, 0.5, 10.0],
            "firm_vat": [100.0, -40.0, 3.0],
            "corporation_tax": [0.0, 1_000.0, 25.0],
        }
    )
    totals = weighted_uk_firm_tax_totals(taxes)
    assert list(totals) == ["firm_vat", "corporation_tax"]
    assert totals["firm_vat"] == pytest.approx(2.0 * 100 - 0.5 * 40 + 10.0 * 3)
    assert totals["corporation_tax"] == pytest.approx(0.5 * 1_000 + 10.0 * 25)
    assert all(isinstance(value, float) for value in totals.values())


def test_weighted_totals_of_evaluated_taxes() -> None:
    table = export_table()
    taxes = evaluate_uk_firm_taxes(
        table, period=2024, simulation_factory=FakeTaxFactory()
    )
    totals = weighted_uk_firm_tax_totals(taxes)
    assert list(totals) == list(POLICYENGINE_UK_FIRM_TAX_VARIABLES)
    weight = table["firm_weight"].to_numpy()
    for name, total in totals.items():
        expected = float(tax_value(name, table["firm_id"].to_numpy()) @ weight)
        assert total == pytest.approx(expected)


def test_a_policyengine_uk_without_a_tax_variable_names_it_and_the_pull_request() -> (
    None
):
    system = fake_system(
        variables=("firm_id", *POLICYENGINE_UK_FIRM_TAX_VARIABLES[:-1])
    )
    # The VAT-only check is unchanged.
    with pytest.raises(RuntimeError, match=POLICYENGINE_UK_FIRM_LIABILITY_VARIABLE):
        require_policyengine_uk_firm_rules(system)
    missing = POLICYENGINE_UK_FIRM_TAX_VARIABLES[-1]
    with pytest.raises(RuntimeError) as error:
        require_policyengine_uk_firm_rules(
            system, variables=("firm_id", *POLICYENGINE_UK_FIRM_TAX_VARIABLES)
        )
    message = str(error.value)
    assert f"no {missing} variable" in message
    assert POLICYENGINE_UK_FIRM_PULL_REQUEST in message
    require_policyengine_uk_firm_rules(
        system, variables=("firm_id", *POLICYENGINE_UK_FIRM_TAX_VARIABLES[:-1])
    )


def test_importing_the_tax_reader_leaves_policyengine_uk_unimported() -> None:
    script = (
        "import sys\n"
        "from microcosm.build.uk_runtime.firm_policyengine_uk_rules import "
        "evaluate_uk_firm_taxes, weighted_uk_firm_tax_totals\n"
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
