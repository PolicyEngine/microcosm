"""policyengine-uk as the firm tax rules for the UK firm generator.

policyengine-uk's firm entity is the rules home for Microcosm's UK firms.
``PolicyEngineUKFirmRules`` implements ``firm_rules.UKFirmRules``: it loads
the pre-calibration candidate table (``firm_table.build_candidate_rules_table``)
as a policyengine-uk firm table beside a one-person stub household, runs a
``Microsimulation`` and reads ``firm_vat_liability_if_registered`` for each
firm. The Axiom RuleSpec evaluator stays an optional cross-check through
``firm_rules.LegacyVATEvaluatorRules``; it is no longer the rules home.

The generator's rules seam reads only VAT. ``POLICYENGINE_UK_FIRM_REGISTERED_VARIABLE``
names ``firm_vat_registered``, which a later step reconciles with the
generator's registration flags; the seam does not read it yet.

``evaluate_uk_firm_taxes`` reads every firm tax from the exported firm table
(``to_policyengine_uk_firm_table``, ``uk_firms.parquet``):
``POLICYENGINE_UK_FIRM_TAX_VARIABLES`` lists firm VAT, employer NIC (after the
Employment Allowance), the apprenticeship levy, corporation tax, the bank
surcharge and business rates, then two informational outputs that are not
additional taxes: the Employment Allowance (already netted out of
``firm_employer_ni``) and the employer's automatic enrolment minimum
contribution (an employment cost). It hands policyengine-uk only the table's
policyengine-uk firm inputs (keys of
``firm_table.UK_FIRM_POLICYENGINE_UK_INPUT_STATUS``), so microcosm-only
columns such as ``vat_registered`` are dropped and policyengine-uk applies its
own registration test; any other ``firm_*`` column that is not a known input
is dropped with a warning (an output, or an input newer than the committed
policyengine-uk snapshot). Inputs the export does not set (paybill, profit,
rateable value, capital expenditure, R&D, region, legal form, the bank flag)
take policyengine-uk's defaults, so corporation tax, the bank surcharge,
business rates, employer NIC, the levy and auto-enrolment are zero.
``weighted_uk_firm_tax_totals`` sums each output by ``firm_weight``.

policyengine-uk is imported only inside the default simulation factory, so
importing this module needs no engine. The firm entity and its VAT variables
and the other firm taxes come with PolicyEngine/policyengine-uk#2940; a
policyengine-uk without them,
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
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd

from microcosm.build.firms import weighted_totals
from microcosm.build.uk_runtime.firm_rules import UKFirmRuleOutputs
from microcosm.build.uk_runtime.firm_table import (
    UK_FIRM_CANDIDATE_RULES_COLUMNS,
    UK_FIRM_MICROCOSM_ONLY_COLUMNS,
    UK_FIRM_POLICYENGINE_UK_INPUT_STATUS,
)

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

#: Firm taxes ``evaluate_uk_firm_taxes`` reads: Firm-entity GBP-per-year
#: formula variables of policyengine-uk (PolicyEngine/policyengine-uk#2940),
#: each with its file under ``policyengine_uk/variables/gov/``:
#:
#: - ``firm_vat``: hmrc/vat/firm/firm_vat.py, net VAT, zero when not
#:   registered, negative for a repayment;
#: - ``firm_employer_ni``: hmrc/national_insurance/employer/firm/, secondary
#:   Class 1 after the Employment Allowance;
#: - ``apprenticeship_levy``: hmrc/apprenticeship_levy/;
#: - ``corporation_tax``: hmrc/corporation_tax/, after the R&D expenditure
#:   credit and without the bank surcharge;
#: - ``bank_surcharge``: hmrc/bank_surcharge/;
#: - ``firm_business_rates``: hmrc/business_rates/firm/, England only.
POLICYENGINE_UK_FIRM_TAXES: tuple[str, ...] = (
    "firm_vat",
    "firm_employer_ni",
    "apprenticeship_levy",
    "corporation_tax",
    "bank_surcharge",
    "firm_business_rates",
)

#: Firm-entity GBP outputs read beside the taxes for information. Not
#: additional taxes: ``employment_allowance``
#: (hmrc/national_insurance/employment_allowance/) is already subtracted in
#: ``firm_employer_ni``, and ``firm_auto_enrolment_employer_contribution``
#: (dwp/auto_enrolment/) is an employment cost. Do not add them to the taxes.
POLICYENGINE_UK_FIRM_INFORMATIONAL_VARIABLES: tuple[str, ...] = (
    "employment_allowance",
    "firm_auto_enrolment_employer_contribution",
)

#: Everything ``evaluate_uk_firm_taxes`` reads by default, taxes first.
POLICYENGINE_UK_FIRM_TAX_VARIABLES: tuple[str, ...] = (
    *POLICYENGINE_UK_FIRM_TAXES,
    *POLICYENGINE_UK_FIRM_INFORMATIONAL_VARIABLES,
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


def _missing_firm_support_message(
    detail: str,
    variables: Sequence[str] = POLICYENGINE_UK_FIRM_REQUIRED_VARIABLES,
) -> str:
    needed = ", ".join(name for name in variables if name != "firm_id")
    what = (
        "VAT"
        if tuple(variables) == POLICYENGINE_UK_FIRM_REQUIRED_VARIABLES
        else "taxes"
    )
    return (
        f"The installed policyengine-uk cannot evaluate UK firm {what}: "
        f"{detail}. It needs the firm entity and {needed} "
        f"from {POLICYENGINE_UK_FIRM_PULL_REQUEST}; install a policyengine-uk "
        "that includes it."
    )


def require_policyengine_uk_firm_rules(
    tax_benefit_system: Any,
    dataset_class: Any = None,
    *,
    variables: Sequence[str] = POLICYENGINE_UK_FIRM_REQUIRED_VARIABLES,
) -> None:
    """Raise ``RuntimeError`` unless policyengine-uk can evaluate firm rules.

    Checks that ``tax_benefit_system`` has the firm entity and ``variables``
    (by default ``POLICYENGINE_UK_FIRM_REQUIRED_VARIABLES``, what the VAT path
    reads) as firm variables and, when given, that ``dataset_class``
    (``UKSingleYearDataset``) takes a ``firm`` table. The error names the
    missing variable and PolicyEngine/policyengine-uk#2940.
    """

    variables = tuple(variables)

    entity_keys = {
        getattr(entity, "key", None)
        for entity in getattr(tax_benefit_system, "entities", ())
    }
    if POLICYENGINE_UK_FIRM_ENTITY not in entity_keys:
        raise RuntimeError(
            _missing_firm_support_message("it has no Firm entity", variables)
        )
    system_variables = getattr(tax_benefit_system, "variables", {})
    for name in variables:
        variable = system_variables.get(name)
        if variable is None:
            raise RuntimeError(
                _missing_firm_support_message(f"it has no {name} variable", variables)
            )
        if getattr(variable.entity, "key", None) != POLICYENGINE_UK_FIRM_ENTITY:
            raise RuntimeError(
                _missing_firm_support_message(
                    f"{name} is not a firm variable", variables
                )
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
                    "takes no firm table",
                    variables,
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

    return _policyengine_uk_firm_simulation(
        firm_table, period, POLICYENGINE_UK_FIRM_REQUIRED_VARIABLES
    )


def _policyengine_uk_firm_simulation(
    firm_table: pd.DataFrame, period: int, variables: Sequence[str]
) -> UKFirmSimulation:
    variables = tuple(variables)
    try:
        from policyengine_uk import CountryTaxBenefitSystem, Microsimulation
        from policyengine_uk.data import UKSingleYearDataset
    except ImportError as error:
        raise RuntimeError(
            _missing_firm_support_message(f"importing it failed ({error})", variables)
        ) from error
    require_policyengine_uk_firm_rules(
        CountryTaxBenefitSystem(), UKSingleYearDataset, variables=variables
    )
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


#: Columns of a firm table ``evaluate_uk_firm_taxes`` may hand to
#: policyengine-uk: its formula-less firm inputs.
_POLICYENGINE_UK_FIRM_INPUTS = frozenset(
    {"firm_id", "firm_weight", *UK_FIRM_POLICYENGINE_UK_INPUT_STATUS}
)


def _default_tax_simulation_factory(
    variables: Sequence[str],
) -> UKFirmSimulationFactory:
    required = ("firm_id", *variables)

    def factory(firm_table: pd.DataFrame, period: int) -> UKFirmSimulation:
        return _policyengine_uk_firm_simulation(firm_table, period, required)

    return factory


def evaluate_uk_firm_taxes(
    firm_table: pd.DataFrame,
    *,
    period: int,
    simulation_factory: UKFirmSimulationFactory | None = None,
    variables: Sequence[str] = POLICYENGINE_UK_FIRM_TAX_VARIABLES,
) -> pd.DataFrame:
    """Each firm's policyengine-uk taxes in ``period``, keyed by ``firm_id``.

    ``firm_table`` is the policyengine-uk-named export
    (``firm_table.to_policyengine_uk_firm_table``, ``uk_firms.parquet``). Its
    columns that are policyengine-uk firm inputs (keys of
    ``UK_FIRM_POLICYENGINE_UK_INPUT_STATUS``), in the table's column order,
    go to ``simulation_factory`` with ``period``; every other column
    (microcosm-only ones, or one named after an output) is dropped, so it
    cannot override a rule. ``firm_id`` is required.

    Returns one row per input row, in input order: ``firm_id`` (int64),
    ``firm_weight`` (float64) when the table has it, then one float64 column
    per name in ``variables`` (pounds per year). Raises ``ValueError`` if a
    variable comes back with the wrong shape or a non-finite value, or if the
    simulation's ``firm_id`` order differs from the table's.

    ``simulation_factory`` defaults to a policyengine-uk ``Microsimulation``
    (as ``policyengine_uk_firm_simulation``) that first checks every name in
    ``variables`` is a firm variable, raising a ``RuntimeError`` that names
    the missing one and PolicyEngine/policyengine-uk#2940.
    """

    variables = tuple(variables)
    if not variables:
        raise ValueError("evaluate_uk_firm_taxes needs at least one variable.")
    if len(set(variables)) != len(variables):
        raise ValueError(f"UK firm tax variables {list(variables)} repeat a name.")
    clashing = [name for name in variables if name in ("firm_id", "firm_weight")]
    if clashing:
        raise ValueError(f"UK firm tax variables cannot include {clashing}.")
    if "firm_id" not in firm_table.columns:
        raise ValueError("UK firm table misses column firm_id.")

    period = int(period)
    inputs = [
        column
        for column in firm_table.columns
        if column in _POLICYENGINE_UK_FIRM_INPUTS
    ]
    unknown = [
        column
        for column in firm_table.columns
        if str(column).startswith("firm_")
        and column not in _POLICYENGINE_UK_FIRM_INPUTS
        and column not in UK_FIRM_MICROCOSM_ONLY_COLUMNS
    ]
    if unknown:
        warnings.warn(
            f"evaluate_uk_firm_taxes drops firm columns {unknown}: they are not "
            "policyengine-uk firm inputs in UK_FIRM_POLICYENGINE_UK_INPUT_STATUS "
            "(an output, or an input newer than the committed snapshot).",
            stacklevel=2,
        )
    handed = firm_table.loc[:, inputs].reset_index(drop=True)
    firm_id = handed["firm_id"].to_numpy(np.int64)
    factory = simulation_factory or _default_tax_simulation_factory(variables)
    simulation = factory(handed, period)

    simulated_ids = _firm_values(simulation, "firm_id", period, len(firm_id))
    if not np.array_equal(simulated_ids.astype(np.int64), firm_id):
        raise ValueError(
            "policyengine-uk's firm_id order differs from the firm table's, "
            "so its taxes cannot be keyed by firm_id."
        )
    columns: dict[str, np.ndarray] = {"firm_id": firm_id}
    if "firm_weight" in handed.columns:
        columns["firm_weight"] = handed["firm_weight"].to_numpy(np.float64)
    for name in variables:
        values = _firm_values(simulation, name, period, len(firm_id)).astype(np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"policyengine-uk returned a non-finite {name}.")
        columns[name] = values
    return pd.DataFrame(columns)


def weighted_uk_firm_tax_totals(taxes: pd.DataFrame) -> dict[str, float]:
    """Weighted total of each output of ``evaluate_uk_firm_taxes``.

    Returns ``sum(value * firm_weight)`` for every column other than
    ``firm_id`` and ``firm_weight``, in pounds per year. Each total stands
    alone: the informational outputs
    (``POLICYENGINE_UK_FIRM_INFORMATIONAL_VARIABLES``) are not taxes to add.
    """

    return weighted_totals(taxes)


__all__ = [
    "POLICYENGINE_UK_FIRM_INFORMATIONAL_VARIABLES",
    "POLICYENGINE_UK_FIRM_INPUT_COLUMNS",
    "POLICYENGINE_UK_FIRM_LIABILITY_VARIABLE",
    "POLICYENGINE_UK_FIRM_PULL_REQUEST",
    "POLICYENGINE_UK_FIRM_REGISTERED_VARIABLE",
    "POLICYENGINE_UK_FIRM_REQUIRED_VARIABLES",
    "POLICYENGINE_UK_FIRM_TAXES",
    "POLICYENGINE_UK_FIRM_TAX_VARIABLES",
    "PolicyEngineUKFirmRules",
    "UKFirmSimulation",
    "UKFirmSimulationFactory",
    "evaluate_uk_firm_taxes",
    "policyengine_uk_firm_simulation",
    "policyengine_uk_version",
    "require_policyengine_uk_firm_rules",
    "weighted_uk_firm_tax_totals",
]
