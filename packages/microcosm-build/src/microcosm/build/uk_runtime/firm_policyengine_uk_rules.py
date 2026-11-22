"""policyengine-uk as the VAT rules for the UK firm generator.

policyengine-uk's firm entity is the rules home for Microcosm's UK firms.
``PolicyEngineUKFirmRules`` implements ``firm_rules.UKFirmRules``: it loads
the pre-calibration candidate table (``firm_table.build_candidate_rules_table``)
as a policyengine-uk firm table beside a one-person stub household, runs a
``Microsimulation`` and reads ``firm_vat_liability_if_registered`` for each
firm. The Axiom RuleSpec evaluator stays an optional cross-check through
``firm_rules.LegacyVATEvaluatorRules``; it is no longer the rules home.

Only VAT is read so far. ``POLICYENGINE_UK_FIRM_REGISTERED_VARIABLE`` names
``firm_vat_registered``, which a later step reconciles with the generator's
registration flags; this adapter does not read it yet. No other firm tax
(employer NIC, corporation tax, business rates) is read.

policyengine-uk is imported only inside the default simulation factory, so
importing this module needs no engine. The firm entity and its VAT variables
come with PolicyEngine/policyengine-uk#2940; a policyengine-uk without them,
such as the release Microcosm's ``uk`` extra pins today, makes the default
factory raise a ``RuntimeError`` naming that pull request.

Firm order: policyengine-uk builds its firm population from the firm table's
``firm_id`` column in the order the table gives its rows, and returns firm
variables in that order. The default factory hands over the candidate table in
its own order and reads ``firm_id`` back, refusing a simulation whose firm ids
differ from the table's, so each liability is keyed by the input ``firm_id``.

Period convention. A firm data vintage is evaluated in the period
``firm_rules.rule_period_for_vintage`` gives it, the tax year's start year:
2024-25 is period 2024 and 2023-24 is period 2023. This adapter passes that
period through unchanged, as the dataset's ``fiscal_year`` and as the
simulation period. It matches how policyengine-uk labels a year, by its
financial year's start:

- a dataset's ``fiscal_year`` is the year its financial year starts
  (``UKSingleYearDataset`` warns that "FRS 2024/25 data is 2024");
- ``CountryTaxBenefitSystem`` runs ``convert_to_fiscal_year_parameters`` over
  ``gov``, which writes each year's 30 April value (or, for
  ``fiscal_year_blend`` leaves, the 6 April to 5 April average) at 1 January
  of that year. Period ``Y`` therefore reads the parameters of financial year
  ``Y``-``Y+1``, not those in force on 1 January ``Y``: period 2023 reads the
  £85,000 VAT registration threshold, period 2024 the £90,000 one that took
  effect on 1 April 2024, and period 2025 the 2025-26 values.

The VAT registration threshold parameter carries no
``preserve_calendar_dates``, so it is annualised this way too. Reading period
2025 for the 2024-25 vintage would give 2025-26 parameters: the same for VAT
(20% and £90,000 in both years), but not for employer NIC, whose rate and
threshold changed in April 2025, which is why the period is the start year.
"""

from __future__ import annotations

import importlib.metadata
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd

from microcosm.build.uk_runtime.firm_rules import UKFirmRuleOutputs
from microcosm.build.uk_runtime.firm_table import UK_FIRM_CANDIDATE_RULES_COLUMNS

#: The firm variable the adapter reads: the VAT a firm would owe if
#: registered, in pounds per year (``firm_vat`` charges it when registered).
POLICYENGINE_UK_FIRM_LIABILITY_VARIABLE = "firm_vat_liability_if_registered"

#: policyengine-uk's VAT registration test, for a later reconciliation with
#: the generator's registration flags. Not read yet.
POLICYENGINE_UK_FIRM_REGISTERED_VARIABLE = "firm_vat_registered"

#: The pull request that adds the firm entity and its VAT variables.
POLICYENGINE_UK_FIRM_PULL_REQUEST = "PolicyEngine/policyengine-uk#2940"

#: The policyengine-uk entity key of firms.
POLICYENGINE_UK_FIRM_ENTITY = "firm"

#: Variables the adapter needs from policyengine-uk.
POLICYENGINE_UK_FIRM_REQUIRED_VARIABLES = (
    "firm_id",
    POLICYENGINE_UK_FIRM_LIABILITY_VARIABLE,
)

#: Candidate-table columns handed to policyengine-uk as firm inputs, in order.
#: All are formula-less policyengine-uk firm variables; any other column of the
#: table is dropped, so it cannot override a rule.
POLICYENGINE_UK_FIRM_INPUT_COLUMNS: tuple[str, ...] = tuple(
    UK_FIRM_CANDIDATE_RULES_COLUMNS
)

#: The provenance version recorded when policyengine-uk is not installed (an
#: injected factory, for example).
POLICYENGINE_UK_VERSION_UNKNOWN = "not installed"


class UKFirmSimulation(Protocol):
    """What the adapter reads from a simulation: a firm variable's values."""

    def calculate(self, variable_name: str, period: int) -> Any:
        """Values of ``variable_name`` in ``period``, one per firm."""


#: Builds a simulation over a firm input table for a period.
UKFirmSimulationFactory = Callable[[pd.DataFrame, int], UKFirmSimulation]


def _missing_firm_support_message(detail: str) -> str:
    return (
        f"The installed policyengine-uk cannot evaluate UK firm VAT: {detail}. "
        f"It needs the firm entity and {POLICYENGINE_UK_FIRM_LIABILITY_VARIABLE} "
        f"from {POLICYENGINE_UK_FIRM_PULL_REQUEST}; install a policyengine-uk "
        "that includes it."
    )


def require_policyengine_uk_firm_rules(
    tax_benefit_system: Any,
    dataset_class: Any = None,
) -> None:
    """Raise ``RuntimeError`` unless policyengine-uk can evaluate firm VAT.

    Checks that ``tax_benefit_system`` has the firm entity and
    ``POLICYENGINE_UK_FIRM_REQUIRED_VARIABLES`` as firm variables and, when
    given, that ``dataset_class`` (``UKSingleYearDataset``) takes a ``firm``
    table. The error names PolicyEngine/policyengine-uk#2940.
    """

    entity_keys = {
        getattr(entity, "key", None)
        for entity in getattr(tax_benefit_system, "entities", ())
    }
    if POLICYENGINE_UK_FIRM_ENTITY not in entity_keys:
        raise RuntimeError(_missing_firm_support_message("it has no Firm entity"))
    variables = getattr(tax_benefit_system, "variables", {})
    for name in POLICYENGINE_UK_FIRM_REQUIRED_VARIABLES:
        variable = variables.get(name)
        if variable is None:
            raise RuntimeError(
                _missing_firm_support_message(f"it has no {name} variable")
            )
        if getattr(variable.entity, "key", None) != POLICYENGINE_UK_FIRM_ENTITY:
            raise RuntimeError(
                _missing_firm_support_message(f"{name} is not a firm variable")
            )
    if dataset_class is not None:
        import inspect

        try:
            parameters = inspect.signature(dataset_class).parameters
        except (TypeError, ValueError):
            parameters = {}
        if "firm" not in parameters:
            raise RuntimeError(
                _missing_firm_support_message(
                    f"{getattr(dataset_class, '__name__', dataset_class)} "
                    "takes no firm table"
                )
            )


def _stub_household_tables() -> dict[str, pd.DataFrame]:
    """One adult in one benefit unit and one household, as policyengine-uk's
    firm tests use. Firms are root entities with no members, so the household
    side only has to load."""

    return {
        "person": pd.DataFrame(
            {
                "person_id": [1],
                "person_benunit_id": [1],
                "person_household_id": [1],
                "age": [40],
                "employment_income": [0.0],
            }
        ),
        "benunit": pd.DataFrame({"benunit_id": [1]}),
        "household": pd.DataFrame(
            {
                "household_id": [1],
                "household_weight": [1.0],
                "region": ["LONDON"],
                "rent": [0.0],
                "tenure_type": ["OWNED_OUTRIGHT"],
                "council_tax": [0.0],
            }
        ),
    }


def policyengine_uk_firm_simulation(
    firm_table: pd.DataFrame, period: int
) -> UKFirmSimulation:
    """A policyengine-uk ``Microsimulation`` over ``firm_table`` in ``period``.

    The default ``PolicyEngineUKFirmRules`` factory. Imports policyengine-uk
    here, checks it has the firm rules (``require_policyengine_uk_firm_rules``)
    and builds a ``UKSingleYearDataset`` labelled ``fiscal_year=period`` with
    a one-person stub household and ``firm_table`` as its firm table.
    """

    try:
        from policyengine_uk import CountryTaxBenefitSystem, Microsimulation
        from policyengine_uk.data import UKSingleYearDataset
    except ImportError as error:
        raise RuntimeError(
            _missing_firm_support_message(f"importing it failed ({error})")
        ) from error
    require_policyengine_uk_firm_rules(CountryTaxBenefitSystem(), UKSingleYearDataset)
    dataset = UKSingleYearDataset(
        **_stub_household_tables(),
        firm=firm_table,
        fiscal_year=int(period),
    )
    return Microsimulation(dataset=dataset)


def policyengine_uk_version() -> str:
    """The installed policyengine-uk version, or a marker when absent."""

    try:
        return importlib.metadata.version("policyengine-uk")
    except importlib.metadata.PackageNotFoundError:
        return POLICYENGINE_UK_VERSION_UNKNOWN


def _firm_values(
    simulation: UKFirmSimulation, variable: str, period: int, n_firms: int
) -> np.ndarray:
    values = simulation.calculate(variable, period)
    values = np.asarray(values)
    if values.shape != (n_firms,):
        raise ValueError(
            f"policyengine-uk returned {variable} with shape {values.shape} "
            f"for {n_firms} firms."
        )
    return values


@dataclass(frozen=True)
class PolicyEngineUKFirmRules:
    """``UKFirmRules`` that run policyengine-uk's firm entity.

    ``evaluate`` hands the candidate table's
    ``POLICYENGINE_UK_FIRM_INPUT_COLUMNS`` to ``simulation_factory`` with the
    period, reads ``firm_vat_liability_if_registered`` (pounds per year,
    negative for a repayment) and keys it by the input ``firm_id``. The
    simulation's own ``firm_id`` must match the input order.

    ``simulation_factory`` defaults to ``policyengine_uk_firm_simulation``;
    tests inject their own.
    """

    simulation_factory: UKFirmSimulationFactory | None = None
    name: str = "policyengine_uk"

    def evaluate(self, firms: pd.DataFrame, *, period: int) -> UKFirmRuleOutputs:
        missing = [
            column
            for column in POLICYENGINE_UK_FIRM_INPUT_COLUMNS
            if column not in firms.columns
        ]
        if missing:
            raise ValueError(f"UK firm candidate table misses columns {missing}.")
        period = int(period)
        firm_table = firms.loc[:, list(POLICYENGINE_UK_FIRM_INPUT_COLUMNS)].reset_index(
            drop=True
        )
        firm_id = firm_table["firm_id"].to_numpy(np.int64)
        factory = self.simulation_factory or policyengine_uk_firm_simulation
        simulation = factory(firm_table, period)

        liability = _firm_values(
            simulation, POLICYENGINE_UK_FIRM_LIABILITY_VARIABLE, period, len(firm_id)
        ).astype(np.float64)
        if not np.all(np.isfinite(liability)):
            raise ValueError(
                f"policyengine-uk returned a non-finite "
                f"{POLICYENGINE_UK_FIRM_LIABILITY_VARIABLE}."
            )
        simulated_ids = _firm_values(simulation, "firm_id", period, len(firm_id))
        if not np.array_equal(simulated_ids.astype(np.int64), firm_id):
            raise ValueError(
                "policyengine-uk's firm_id order differs from the candidate "
                "table's, so its liabilities cannot be keyed by firm_id."
            )

        return UKFirmRuleOutputs(
            firm_id=firm_id,
            vat_liability_if_registered=liability,
            provenance=self._provenance(period),
        )

    def _provenance(self, period: int) -> Mapping[str, str]:
        return {
            "rules": self.name,
            "policyengine_uk_version": policyengine_uk_version(),
            "variable": POLICYENGINE_UK_FIRM_LIABILITY_VARIABLE,
            "period": str(period),
            "output": "vat_liability_if_registered",
        }


__all__ = [
    "POLICYENGINE_UK_FIRM_INPUT_COLUMNS",
    "POLICYENGINE_UK_FIRM_LIABILITY_VARIABLE",
    "POLICYENGINE_UK_FIRM_PULL_REQUEST",
    "POLICYENGINE_UK_FIRM_REGISTERED_VARIABLE",
    "POLICYENGINE_UK_FIRM_REQUIRED_VARIABLES",
    "PolicyEngineUKFirmRules",
    "UKFirmSimulation",
    "UKFirmSimulationFactory",
    "policyengine_uk_firm_simulation",
    "policyengine_uk_version",
    "require_policyengine_uk_firm_rules",
]
