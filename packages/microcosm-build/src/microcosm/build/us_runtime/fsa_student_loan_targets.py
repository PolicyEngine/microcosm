"""Federal Student Aid portfolio targets for federal student loan balances.

microcosm#1074 calibrates the person balances the SCF education-debt stage
imputes (``federal_student_loan_balance`` and
``federal_parent_plus_loan_balance``, microcosm#1073) to Federal Student Aid's
federal student loan portfolio tables, which Chronicle stores as published
(PolicyEngine/chronicle#310). Chronicle keeps each table's own level; the
alignment across tables, the treatment of FSA's "Not Reported" and "Other"
rows and the reference period are Microcosm's, and live here.

What binds. One person's federal balance is ``own + Parent PLUS``; a borrower
is a person whose federal balance is positive. FSA's borrower tables count a
Parent PLUS borrower as the parent, so the person holding the loan is the one
counted, which is how the stage assigns those balances.

- **By state** (``portfolio_by_location``): the balance and the borrower count
  of each of the 50 states and DC.
- **By borrower age** (``portfolio_by_age``): balance and borrowers in each
  age band, on the person's age.
- **By debt size** (``portfolio_by_debt_size``): balance and borrowers in each
  band of the person's federal balance (lower edge inclusive, upper exclusive,
  as FSA's Definitions sheet states).
- **By loan type** (``portfolio_by_loan_type``): the Parent PLUS balance, and
  the balance of every other loan type, against the two person columns. FSA
  files a consolidated Parent PLUS loan under "Consolidation", so its Parent
  PLUS row is a lower bound on the stage's concept (every loan for a child's
  education, consolidated or not).

Repayment plans, the portfolio summary and the loan-type recipient counts stay
diagnostic only: :data:`FSA_DIAGNOSTIC_TABLES` and the row rules in
:func:`plan_fsa_student_loan_targets` give each reason.

Reference period. The build's period is the H5's (2024), but the balances are
stocks that the student-debt analysis reads as they stand (its reference year
is 2026), and the stage's SCF draws are 2025 survey-date dollars. So the
targets are FSA's levels at one reference date, unaged:
``period_alignment = stock_at_source_reference_date``, which
:mod:`microcosm.build.us_runtime.target_aging` recognises as a declared
alignment. The reference date is the latest quarter that every bound table
publishes (FY2026 Q2, 31 March 2026, at chronicle#347): the location workbook
publishes only that date, and holding the other tables to it keeps every family
a statement about one portfolio on one day. Later quarters of the age,
debt-size and loan-type tables stay in the receipt.

One control total. FSA's borrower tables do not agree exactly with one another
or with its recipient tables, and the location table holds rows the host
cannot carry:

- "Not Reported" (FFEL and Perkins loans whose commercial lenders need not
  report an address) is allocated to every reported location in proportion to
  its reported balance, or its reported borrowers for the count.
- Puerto Rico and "Other" (an address outside the states and territories the
  table lists) are outside the host's universe, which is the 50 states and DC
  that the CPS samples. They leave, with their share of "Not Reported".

The 51 allocated state levels sum to the in-scope control. The age and
debt-size bands, and the two loan-type rows, bind as their table's shares of
that control (``control x row / partition``, the microcosm#940 pattern), so all
four families describe one in-scope total. A table's own "not reported" age row
is in no partition, which allocates it across the reported bands the same way.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.ledger_targets import LedgerTargetReference
from microcosm.calibrate import (
    CalibrationHierarchySeed,
    HierarchyCategory,
    HierarchyNode,
    TargetRegistry,
    TargetSpec,
    calibration_provider_label,
    calibration_variable_label,
)
from microcosm.calibrate.geography_constants import US_STATE_FIPS_TO_POSTAL

__all__ = [
    "FSA_BALANCE_COLUMNS",
    "FSA_BALANCE_LOWER_BOUND_KEY",
    "FSA_BALANCE_UPPER_BOUND_KEY",
    "FSA_BOUND_TABLES",
    "FSA_CATEGORY_BY_MEASURE",
    "FSA_DIAGNOSTIC_TABLES",
    "FSA_MATERIALIZER",
    "FSA_MEASURE_KEY",
    "FSA_MEASURES",
    "FSA_PERIOD_ALIGNMENT",
    "FSA_SOURCE_NAME",
    "FSA_TARGET_FAMILY",
    "FsaTargetPlan",
    "apply_fsa_student_loan_plan",
    "fsa_student_loan_household_values",
    "fsa_student_loan_target_receipt",
    "plan_fsa_student_loan_targets",
]

FSA_SOURCE_NAME = "ed_fsa"
FSA_TARGET_FAMILY = "ed_fsa"
#: The release materializer that reads the stage's stored person columns.
FSA_MATERIALIZER = "fsa_stored_student_loan_balances"
#: Declared in :mod:`microcosm.build.us_runtime.target_aging`: the value is a
#: stock at its source reference date, consumed unaged at the build period.
FSA_PERIOD_ALIGNMENT = "stock_at_source_reference_date"
FSA_OWN_BALANCE_COLUMN = "federal_student_loan_balance"
FSA_PARENT_PLUS_BALANCE_COLUMN = "federal_parent_plus_loan_balance"
FSA_BALANCE_COLUMNS: tuple[str, str] = (
    FSA_OWN_BALANCE_COLUMN,
    FSA_PARENT_PLUS_BALANCE_COLUMN,
)

#: Spec metadata naming what the materializer sums.
FSA_MEASURE_KEY = "fsa_measure"
FSA_TOTAL_BALANCE = "total_balance"
FSA_BORROWERS = "borrowers"
FSA_PARENT_PLUS_BALANCE = "parent_plus_balance"
FSA_BALANCE_EXCLUDING_PARENT_PLUS = "balance_excluding_parent_plus"
#: measure -> (person columns summed, positive-indicator count).
FSA_MEASURES: Mapping[str, tuple[tuple[str, ...], bool]] = {
    FSA_TOTAL_BALANCE: (FSA_BALANCE_COLUMNS, False),
    FSA_BORROWERS: (FSA_BALANCE_COLUMNS, True),
    FSA_PARENT_PLUS_BALANCE: ((FSA_PARENT_PLUS_BALANCE_COLUMN,), False),
    FSA_BALANCE_EXCLUDING_PARENT_PLUS: ((FSA_OWN_BALANCE_COLUMN,), False),
}
#: Hierarchy category (under provider ``ed_fsa``) of each measure.
FSA_CATEGORY_BY_MEASURE: Mapping[str, str] = {
    FSA_TOTAL_BALANCE: "federal_student_loan_balance",
    FSA_BORROWERS: "federal_student_loan_borrowers",
    FSA_PARENT_PLUS_BALANCE: "parent_plus_loan_balance",
    FSA_BALANCE_EXCLUDING_PARENT_PLUS: "loan_balance_excluding_parent_plus",
}
#: Compiled edges of a debt-size band, on the person's federal balance.
FSA_BALANCE_LOWER_BOUND_KEY = "fsa_balance_lower_bound"
FSA_BALANCE_UPPER_BOUND_KEY = "fsa_balance_upper_bound"
#: Compiled edges of an age band; the names the release's restated-age guard
#: reads (``RESTATED_AGE_BAND_COMPILED_KEYS``), with the same attestation.
_AGE_LOWER_BOUND_KEY = "age_lower_bound"
_AGE_UPPER_BOUND_KEY = "age_upper_bound"
_AGE_BOUND_STAMP_SOURCE_KEY = "age_bound_stamp_source"
_AGE_BOUND_STAMP_FROM_CONSTRAINT_ROWS = "constraint_rows"

_LOCATION = "portfolio_by_location"
_AGE = "portfolio_by_age"
_DEBT_SIZE = "portfolio_by_debt_size"
_LOAN_TYPE = "portfolio_by_loan_type"
FSA_BOUND_TABLES: tuple[str, ...] = (_LOCATION, _AGE, _DEBT_SIZE, _LOAN_TYPE)
_DOLLARS = "dollars_outstanding"
_BORROWER_COUNT = "borrowers"
_MEASURE_BY_SOURCE_MEASURE = {
    _DOLLARS: FSA_TOTAL_BALANCE,
    _BORROWER_COUNT: FSA_BORROWERS,
}
_PARENT_PLUS_LOAN_TYPE = "parent_plus"
_NOT_REPORTED_LOCATION = "not_reported"
_OTHER_LOCATION = "other"
_PUERTO_RICO_FIPS = "72"
_AGE_VARIABLE = "age"
_BALANCE_VARIABLE = "federal_student_loan_balance"

#: Every other FSA record-set table, with why it stays diagnostic only.
FSA_DIAGNOSTIC_TABLES: Mapping[str, str] = {
    "portfolio_summary": (
        "Recipient-based totals by loan program (Direct Loan, FFEL, Perkins). "
        "The stage's balances do not record the program, and FSA counts the "
        "student, not the parent, as the recipient of a Parent PLUS loan; the "
        "borrower-based location table sets the level instead."
    ),
    "dl_by_repayment_plan": (
        "Repayment plan: diagnostic only. The SCF records only an "
        "income-driven yes/no, the build carries no plan column, and FSA's "
        "recipient counts are loan-level (microcosm#1074)."
    ),
    "dl_idr_by_partial_financial_hardship": (
        "Income-driven plans by partial financial hardship: no plan or "
        "hardship column in the build; diagnostic only."
    ),
    "ed_held_ffel_by_repayment_plan": (
        "ED-held FFEL by repayment plan: no plan column in the build; diagnostic only."
    ),
    "ed_held_ffel_idr_by_partial_financial_hardship": (
        "ED-held FFEL IBR by partial financial hardship: no plan or hardship "
        "column in the build; diagnostic only."
    ),
    "federally_managed_by_repayment_plan": (
        "Federally managed portfolio by repayment plan: no plan column in the "
        "build; diagnostic only."
    ),
    "federally_managed_idr_by_partial_financial_hardship": (
        "Federally managed IBR/PAYE by partial financial hardship: no plan or "
        "hardship column in the build; diagnostic only."
    ),
}

_REASON_OTHER_QUARTER = (
    "Not the reference quarter: every bound table binds at the latest quarter "
    "all four publish ({reference})."
)
_REASON_PUERTO_RICO = (
    "Puerto Rico is outside the host universe (the CPS samples the 50 states "
    "and DC). Its rows, with their pro-rata share of Not Reported, leave the "
    "in-scope control."
)
_REASON_OTHER_LOCATION = (
    "'Other' is a borrower address outside the states and territories the "
    "table lists, outside the host universe; it leaves the in-scope control "
    "with its pro-rata share of Not Reported."
)
_REASON_NOT_REPORTED_LOCATION = (
    "'Not Reported' (FFEL and Perkins loans whose commercial lenders need not "
    "report an address) has no state; it is allocated to every reported "
    "location in proportion to that location's reported value."
)
_REASON_NOT_REPORTED_AGE = (
    "Age not reported: in no partition, so the reported bands, which bind as "
    "shares of the in-scope control, absorb it pro rata."
)
_REASON_LOAN_TYPE_RECIPIENTS = (
    "Loan-type recipient counts are loan-level (a borrower with two loan types "
    "counts twice), so they are not counts of people."
)
_REASON_LOAN_TYPE_MEMBER = (
    "A member of the balance-excluding-Parent-PLUS target: the build records "
    "own and Parent PLUS balances, not loan types."
)
_REASON_STAFFORD_SUBTOTAL = (
    "Stafford (Combined) is a subtotal of rows the loan-type targets already count."
)
_REASON_UNREVIEWED_TABLE = (
    "An FSA table with no reviewed binding: inert until one is added."
)


@dataclass(frozen=True)
class FsaTargetPlan:
    """The FSA references a feed yields, their reconciled values and receipt.

    ``references`` compile through
    :func:`microcosm.build.ledger_targets.compile_ledger_target_references`;
    :func:`apply_fsa_student_loan_plan` then sets each compiled spec's value to
    ``values[name]`` and stamps ``metadata[name]``. ``dispositions`` names
    every FSA fact in the feed once, with what the plan did with it.
    """

    reference_period: str | None
    reference_label: str | None
    references: tuple[LedgerTargetReference, ...] = ()
    values: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    labels: Mapping[str, str] = field(default_factory=dict)
    dispositions: tuple[Mapping[str, str], ...] = ()
    controls: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    status: str = "no_fsa_facts"


def plan_fsa_student_loan_targets(
    facts: Iterable[object],
    *,
    target_period: int | str,
) -> FsaTargetPlan:
    """Plan the FSA target references and their reconciled values.

    Raises when the feed carries some bound tables but not all of them, when
    they share no quarter, or when a reference-quarter table is malformed (a
    band partition that is not contiguous, a duplicate row, a state twice), so
    a partial or changed FSA table cannot bind silently.
    """

    fsa_facts = tuple(fact for fact in facts if _source_name(fact) == FSA_SOURCE_NAME)
    if not fsa_facts:
        return FsaTargetPlan(reference_period=None, reference_label=None)
    by_table: dict[str, list[object]] = {}
    for fact in fsa_facts:
        by_table.setdefault(_table(fact), []).append(fact)
    missing = [table for table in FSA_BOUND_TABLES if table not in by_table]
    if missing:
        raise ValueError(
            "FSA student loan targets need every bound table; the feed carries "
            f"ed_fsa facts but not {missing!r} (microcosm#1074)."
        )
    periods = [
        {
            _period(fact)
            for fact in by_table[table]
            if _measure_id(fact) in _MEASURE_BY_SOURCE_MEASURE
        }
        for table in FSA_BOUND_TABLES
    ]
    common = set.intersection(*periods)
    if not common:
        raise ValueError(
            "FSA bound tables share no quarter: "
            + "; ".join(
                f"{table}={sorted(table_periods)}"
                for table, table_periods in zip(FSA_BOUND_TABLES, periods, strict=True)
            )
        )
    reference = max(common, key=_month_key)
    at_reference = {
        table: [fact for fact in by_table[table] if _period(fact) == reference]
        for table in FSA_BOUND_TABLES
    }
    reference_label = _location_reference_label(at_reference[_LOCATION])
    dispositions: dict[int, dict[str, str]] = {}

    def dispose(fact: object, disposition: str, reason: str, **extra: str) -> None:
        dispositions[id(fact)] = {
            "source_record_id": _source_record_id(fact),
            "table": _table(fact),
            "period": _period(fact),
            "measure": _measure_id(fact),
            "disposition": disposition,
            "reason": reason,
            **extra,
        }

    for table, table_facts in by_table.items():
        if table not in FSA_BOUND_TABLES:
            reason = FSA_DIAGNOSTIC_TABLES.get(table, _REASON_UNREVIEWED_TABLE)
            kind = "diagnostic" if table in FSA_DIAGNOSTIC_TABLES else "unreviewed"
            for fact in table_facts:
                dispose(fact, kind, reason)
            continue
        for fact in table_facts:
            if _period(fact) != reference:
                dispose(
                    fact,
                    "other_quarter",
                    _REASON_OTHER_QUARTER.format(reference=reference),
                )

    builder = _PlanBuilder(
        target_period=target_period,
        reference_period=reference,
        reference_label=reference_label,
    )
    controls = _plan_locations(at_reference[_LOCATION], builder, dispose)
    for table in (_AGE, _DEBT_SIZE):
        _plan_bands(table, at_reference[table], controls, builder, dispose)
    _plan_loan_types(at_reference[_LOAN_TYPE], controls, builder, dispose)
    for fact in fsa_facts:
        if id(fact) not in dispositions:
            raise AssertionError(
                f"FSA fact {_source_record_id(fact)!r} has no disposition."
            )
    return FsaTargetPlan(
        reference_period=reference,
        reference_label=reference_label,
        references=tuple(builder.references),
        values=dict(builder.values),
        metadata={name: dict(meta) for name, meta in builder.metadata.items()},
        labels=dict(builder.labels),
        dispositions=tuple(dispositions[id(fact)] for fact in fsa_facts),
        controls={
            measure: dict(control) for measure, control in sorted(controls.items())
        },
        status="bound",
    )


def apply_fsa_student_loan_plan(
    registry: TargetRegistry,
    plan: FsaTargetPlan,
    *,
    target_period: int | str,
) -> TargetRegistry:
    """Set each compiled FSA spec to its reconciled value at the build period.

    The compile resolved each reference at its fact's own period (exact); the
    spec then moves to the build period, keeps the published value in
    metadata, and takes the Microcosm target label, because its value is no
    longer the published cell.
    """

    if not plan.values:
        return registry
    specs: list[TargetSpec] = []
    seen: set[str] = set()
    for spec in registry.specs:
        if spec.name not in plan.values:
            specs.append(spec)
            continue
        seen.add(spec.name)
        hierarchy = spec.hierarchy
        if hierarchy is not None:
            hierarchy = replace(
                hierarchy,
                target=HierarchyNode(id=spec.name, label=plan.labels[spec.name]),
            )
        # The Chronicle cell's own label describes the published value at its
        # own period, which this spec no longer carries.
        metadata = {
            key: value
            for key, value in spec.metadata.items()
            if key != "diagnostic_target_label"
        }
        specs.append(
            replace(
                spec,
                value=float(plan.values[spec.name]),
                period=target_period,
                metadata={
                    **metadata,
                    **dict(plan.metadata[spec.name]),
                    "fsa_published_value": _format_float(spec.value),
                    "target_period": str(target_period),
                },
                hierarchy=hierarchy,
            )
        )
    missing = sorted(set(plan.values) - seen)
    if missing:
        raise ValueError(
            f"FSA plan references did not compile: {missing[:5]!r} "
            f"({len(missing)} in all)."
        )
    return TargetRegistry(specs, country=registry.country)


def fsa_student_loan_target_receipt(
    facts: Iterable[object],
    *,
    target_period: int | str,
    plan: FsaTargetPlan | None = None,
) -> dict[str, Any]:
    """Every FSA fact's disposition, the controls and each target's value."""

    plan = plan or plan_fsa_student_loan_targets(facts, target_period=target_period)
    counts: dict[str, int] = {}
    for disposition in plan.dispositions:
        counts[disposition["disposition"]] = (
            counts.get(disposition["disposition"], 0) + 1
        )
    return {
        "schema_version": 1,
        "issue": "PolicyEngine/microcosm#1074",
        "status": plan.status,
        "target_period": str(target_period),
        "reference_period": plan.reference_period,
        "reference_label": plan.reference_label,
        "period_alignment": FSA_PERIOD_ALIGNMENT,
        "controls": {
            measure: dict(control) for measure, control in plan.controls.items()
        },
        "targets": [
            {
                "name": name,
                "label": plan.labels[name],
                "value": plan.values[name],
                **dict(plan.metadata[name]),
            }
            for name in sorted(plan.values)
        ],
        "disposition_counts": dict(sorted(counts.items())),
        "dispositions": [dict(item) for item in plan.dispositions],
    }


def fsa_student_loan_household_values(
    frame: Any,
    metadata: Mapping[str, str],
) -> np.ndarray | None:
    """One FSA target's household column, from the stage's stored balances.

    Sums ``FSA_MEASURES[metadata["fsa_measure"]]`` over each household's
    people (a positive-balance indicator for borrower counts), within the
    person's age band and balance band when the spec compiles them, and
    zeroes households outside the spec's state. Returns ``None`` when the
    frame does not carry both balance columns, so the target is not
    materialized and the release reports it dropped.
    """

    person = frame.table("person")
    if any(column not in person.columns for column in FSA_BALANCE_COLUMNS):
        return None
    measure = metadata.get(FSA_MEASURE_KEY, "")
    if measure not in FSA_MEASURES:
        raise ValueError(f"Unknown FSA measure {measure!r}.")
    columns, indicator = FSA_MEASURES[measure]
    balances = {
        column: pd.to_numeric(person[column], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
        for column in FSA_BALANCE_COLUMNS
    }
    total = balances[FSA_OWN_BALANCE_COLUMN] + balances[FSA_PARENT_PLUS_BALANCE_COLUMN]
    values = np.zeros(len(person), dtype=np.float64)
    for column in columns:
        values = values + balances[column]
    if indicator:
        values = (values > 0.0).astype(np.float64)
    mask = np.ones(len(person), dtype=bool)
    if _AGE_LOWER_BOUND_KEY in metadata or _AGE_UPPER_BOUND_KEY in metadata:
        age = (
            pd.to_numeric(person[_AGE_VARIABLE], errors="coerce")
            .fillna(-1.0)
            .to_numpy(dtype=np.float64)
        )
        mask &= _band_mask(age, metadata, _AGE_LOWER_BOUND_KEY, _AGE_UPPER_BOUND_KEY)
    if (
        FSA_BALANCE_LOWER_BOUND_KEY in metadata
        or FSA_BALANCE_UPPER_BOUND_KEY in metadata
    ):
        mask &= _band_mask(
            total,
            metadata,
            FSA_BALANCE_LOWER_BOUND_KEY,
            FSA_BALANCE_UPPER_BOUND_KEY,
        )
    household = frame.table("household")
    household_ids = household["household_id"].to_numpy()
    positions = (
        pd.Series(np.arange(len(household_ids), dtype=np.int64), index=household_ids)
        .reindex(person["person_household_id"].to_numpy())
        .to_numpy()
    )
    if np.isnan(positions.astype(np.float64)).any():
        raise ValueError("A person's household is missing from the household table.")
    out = np.zeros(len(household_ids), dtype=np.float64)
    np.add.at(out, positions.astype(np.int64), values * mask)
    state_fips = metadata.get("state_fips")
    if state_fips:
        out = np.where(
            household["state_fips"].to_numpy().astype(np.int64) == int(state_fips),
            out,
            0.0,
        )
    return out


class _PlanBuilder:
    def __init__(
        self,
        *,
        target_period: int | str,
        reference_period: str,
        reference_label: str | None,
    ) -> None:
        self.target_period = target_period
        self.reference_period = reference_period
        self.reference_label = reference_label
        self.references: list[LedgerTargetReference] = []
        self.values: dict[str, float] = {}
        self.metadata: dict[str, dict[str, str]] = {}
        self.labels: dict[str, str] = {}

    def add(
        self,
        *,
        name: str,
        measure: str,
        target_role: str,
        source_measure_id: str,
        value: float,
        label: str,
        compiled: Mapping[str, str],
        reconciliation: Mapping[str, str],
        facts: tuple[object, ...],
    ) -> None:
        if name in self.values:
            raise ValueError(f"FSA target {name!r} planned twice.")
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"FSA target {name!r} reconciled to {value!r}.")
        category = FSA_CATEGORY_BY_MEASURE[measure]
        _, indicator = FSA_MEASURES[measure]
        metadata = {
            "materializer": FSA_MATERIALIZER,
            FSA_MEASURE_KEY: measure,
            "measure_mode": "indicator_sum" if indicator else "sum",
            "fsa_columns": ",".join(FSA_MEASURES[measure][0]),
            "target_role": target_role,
            "source_measure_id": source_measure_id,
            "source_period": self.reference_period,
            "target_period": str(self.target_period),
            "period_alignment": FSA_PERIOD_ALIGNMENT,
            "fsa_reference_period": self.reference_period,
            **compiled,
        }
        if self.reference_label:
            metadata["fsa_reference_label"] = self.reference_label
        hierarchy = _hierarchy_seed(category, target_label=label)
        if len(facts) == 1:
            reference = LedgerTargetReference(
                name=name,
                ledger_source_record_id=_source_record_id(facts[0]),
                entity="household",
                measure=name,
                period=self.reference_period,
                period_match_policy="exact",
                family=FSA_TARGET_FAMILY,
                metadata=metadata,
                hierarchy=hierarchy,
            )
        else:
            reference = LedgerTargetReference(
                name=name,
                ledger_selector={"source_name": FSA_SOURCE_NAME},
                value_operation="linear_combination",
                value_operands=tuple(
                    {"source_record_id": _source_record_id(fact), "weight": 1.0}
                    for fact in facts
                ),
                expected_member_count=len(facts),
                entity="household",
                measure=name,
                period=self.reference_period,
                period_match_policy="exact",
                family=FSA_TARGET_FAMILY,
                metadata=metadata,
                hierarchy=hierarchy,
            )
        self.references.append(reference)
        self.values[name] = float(value)
        self.metadata[name] = dict(reconciliation)
        self.labels[name] = label


def _plan_locations(
    facts: list[object],
    builder: _PlanBuilder,
    dispose: Any,
) -> dict[str, dict[str, float]]:
    """State targets with Not Reported allocated; returns the in-scope controls."""

    controls: dict[str, dict[str, float]] = {}
    for source_measure, measure in _MEASURE_BY_SOURCE_MEASURE.items():
        rows = [fact for fact in facts if _measure_id(fact) == source_measure]
        states: dict[str, object] = {}
        out_of_scope: list[tuple[object, str]] = []
        not_reported: list[object] = []
        for fact in rows:
            value_id = _str_at(fact, "layout", "groupby_value_id")
            fips = _state_fips(fact)
            if value_id == _NOT_REPORTED_LOCATION:
                not_reported.append(fact)
            elif fips is not None and fips != _PUERTO_RICO_FIPS:
                if fips in states:
                    raise ValueError(f"FSA location table repeats state {fips}.")
                states[fips] = fact
            elif fips == _PUERTO_RICO_FIPS or _geography_id(fact).endswith("US72"):
                out_of_scope.append((fact, _REASON_PUERTO_RICO))
            elif value_id == _OTHER_LOCATION:
                out_of_scope.append((fact, _REASON_OTHER_LOCATION))
            else:
                raise ValueError(
                    f"FSA location row {_source_record_id(fact)!r} is neither a "
                    "state nor a reviewed statistical-scope row."
                )
        if sorted(states) != sorted(US_STATE_FIPS_TO_POSTAL):
            missing = sorted(set(US_STATE_FIPS_TO_POSTAL) - set(states))
            raise ValueError(
                f"FSA location table ({source_measure}) is missing states {missing}."
            )
        if len(not_reported) > 1:
            raise ValueError("FSA location table has more than one Not Reported row.")
        state_sum = math.fsum(_value(fact) for fact in states.values())
        out_of_scope_sum = math.fsum(_value(fact) for fact, _ in out_of_scope)
        not_reported_value = math.fsum(_value(fact) for fact in not_reported)
        reported = state_sum + out_of_scope_sum
        if reported <= 0:
            raise ValueError("FSA location table reports no positive balance.")
        published_total = reported + not_reported_value
        allocation = published_total / reported
        in_scope = state_sum * allocation
        controls[measure] = {
            "published_states_and_dc": state_sum,
            "published_out_of_scope": out_of_scope_sum,
            "published_not_reported": not_reported_value,
            "published_total": published_total,
            "not_reported_allocation_factor": allocation,
            "in_scope_control": in_scope,
            "in_scope_share_of_total": in_scope / published_total,
        }
        for fact in not_reported:
            dispose(fact, "allocated", _REASON_NOT_REPORTED_LOCATION)
        for fact, reason in out_of_scope:
            dispose(fact, "out_of_scope", reason)
        for fips, fact in sorted(states.items()):
            name = _source_record_id(fact)
            state = _str_at(fact, "geography", "name") or US_STATE_FIPS_TO_POSTAL[fips]
            builder.add(
                name=name,
                measure=measure,
                target_role=f"fsa_state_{_role_suffix(measure)}",
                source_measure_id=source_measure,
                value=_value(fact) * allocation,
                label=f"{_measure_label(measure)}, {state}",
                compiled={"state_fips": fips},
                reconciliation={
                    "fsa_reconciliation": "not_reported_allocated_pro_rata",
                    "fsa_reconciliation_factor": _format_float(allocation),
                    "fsa_control_value": _format_float(in_scope),
                },
                facts=(fact,),
            )
            dispose(
                fact,
                "bound",
                "State level, Not Reported allocated pro rata.",
                target=name,
            )
    if set(controls) != set(_MEASURE_BY_SOURCE_MEASURE.values()):
        raise ValueError("FSA location table lacks a balance or a borrower count.")
    return controls


def _band_lower_bound(band: tuple[float, float, object]) -> float:
    """A (lower, upper, fact) band's lower edge, its sort key."""

    lower, _, _ = band
    return lower


def _plan_bands(
    table: str,
    facts: list[object],
    controls: Mapping[str, Mapping[str, float]],
    builder: _PlanBuilder,
    dispose: Any,
) -> None:
    variable = _AGE_VARIABLE if table == _AGE else _BALANCE_VARIABLE
    for source_measure, measure in _MEASURE_BY_SOURCE_MEASURE.items():
        rows = [fact for fact in facts if _measure_id(fact) == source_measure]
        bands: list[tuple[float, float, object]] = []
        for fact in rows:
            bounds = _constraint_bounds(fact, variable)
            if bounds is None:
                if table == _AGE and _has_constraint(fact, "age_reporting_status"):
                    dispose(fact, "allocated", _REASON_NOT_REPORTED_AGE)
                    continue
                raise ValueError(
                    f"FSA {table} row {_source_record_id(fact)!r} carries no "
                    f"{variable} band."
                )
            bands.append((*bounds, fact))
        bands.sort(key=_band_lower_bound)
        if not bands:
            raise ValueError(f"FSA {table} has no {source_measure} bands.")
        if bands[0][0] != -math.inf or bands[-1][1] != math.inf:
            raise ValueError(f"FSA {table} bands do not cover every {variable}.")
        for (_, upper, _), (lower, _, _) in zip(bands, bands[1:], strict=False):
            if upper != lower:
                raise ValueError(
                    f"FSA {table} {source_measure} bands are not contiguous at "
                    f"{upper!r} / {lower!r}."
                )
        partition = math.fsum(_value(fact) for _, _, fact in bands)
        if partition <= 0:
            raise ValueError(f"FSA {table} {source_measure} partition is empty.")
        control = controls[measure]["in_scope_control"]
        for lower, upper, fact in bands:
            name = _source_record_id(fact)
            share = _value(fact) / partition
            compiled = (
                _age_band_metadata(lower, upper)
                if table == _AGE
                else _balance_band_metadata(lower, upper)
            )
            builder.add(
                name=name,
                measure=measure,
                target_role=(
                    f"fsa_{'age' if table == _AGE else 'debt_size'}_band_"
                    f"{_role_suffix(measure)}"
                ),
                source_measure_id=source_measure,
                value=control * share,
                label=(
                    f"{_measure_label(measure)}, "
                    f"{_str_at(fact, 'layout', 'groupby_dimension_label').lower()} "
                    f"{_str_at(fact, 'layout', 'groupby_value_label')}"
                ),
                compiled=compiled,
                reconciliation={
                    "fsa_reconciliation": "share_of_in_scope_location_control",
                    "fsa_share": _format_float(share),
                    "fsa_partition_value": _format_float(partition),
                    "fsa_control_value": _format_float(control),
                },
                facts=(fact,),
            )
            dispose(fact, "bound", "Band share of the in-scope control.", target=name)


def _plan_loan_types(
    facts: list[object],
    controls: Mapping[str, Mapping[str, float]],
    builder: _PlanBuilder,
    dispose: Any,
) -> None:
    rows = []
    for fact in facts:
        measure = _measure_id(fact)
        kind = _str_at(fact, "layout", "table_record_kind")
        if measure == _DOLLARS and kind in {"detail", ""}:
            rows.append(fact)
        elif kind == "subtotal" or measure.startswith("stafford_combined"):
            dispose(fact, "excluded", _REASON_STAFFORD_SUBTOTAL)
        elif "recipients" in measure:
            dispose(fact, "excluded", _REASON_LOAN_TYPE_RECIPIENTS)
        else:
            raise ValueError(
                f"FSA loan-type row {_source_record_id(fact)!r} has no reviewed "
                "disposition."
            )
    parent_plus = [
        fact
        for fact in rows
        if _str_at(fact, "layout", "groupby_value_id") == _PARENT_PLUS_LOAN_TYPE
    ]
    if len(parent_plus) != 1:
        raise ValueError("FSA loan-type table needs exactly one Parent PLUS row.")
    others = tuple(
        sorted(
            (fact for fact in rows if fact is not parent_plus[0]),
            key=_source_record_id,
        )
    )
    if not others:
        raise ValueError("FSA loan-type table has no loan type besides Parent PLUS.")
    keys = [_str_at(fact, "layout", "groupby_value_id") for fact in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("FSA loan-type table repeats a loan type.")
    partition = math.fsum(_value(fact) for fact in rows)
    control = controls[FSA_TOTAL_BALANCE]["in_scope_control"]
    record_set = _str_at(parent_plus[0], "layout", "record_set_id")
    for measure, members, name in (
        (
            FSA_PARENT_PLUS_BALANCE,
            (parent_plus[0],),
            _source_record_id(parent_plus[0]),
        ),
        (
            FSA_BALANCE_EXCLUDING_PARENT_PLUS,
            others,
            f"{record_set}.excluding_parent_plus.{_DOLLARS}",
        ),
    ):
        published = math.fsum(_value(fact) for fact in members)
        share = published / partition
        builder.add(
            name=name,
            measure=measure,
            target_role=f"fsa_{measure}",
            source_measure_id=_DOLLARS,
            value=control * share,
            label=_measure_label(measure),
            compiled={},
            reconciliation={
                "fsa_reconciliation": "share_of_in_scope_location_control",
                "fsa_share": _format_float(share),
                "fsa_partition_value": _format_float(partition),
                "fsa_control_value": _format_float(control),
            },
            facts=members,
        )
        for fact in members:
            if measure == FSA_PARENT_PLUS_BALANCE:
                dispose(
                    fact,
                    "bound",
                    "Parent PLUS share of the in-scope control.",
                    target=name,
                )
            else:
                dispose(fact, "bound", _REASON_LOAN_TYPE_MEMBER, target=name)


def _hierarchy_seed(category_id: str, *, target_label: str) -> CalibrationHierarchySeed:
    provider_label = calibration_provider_label("us", FSA_SOURCE_NAME)
    category_label = calibration_variable_label("us", FSA_SOURCE_NAME, category_id)
    if provider_label is None or not category_label:
        raise ValueError(
            f"FSA calibration category {category_id!r} has no declared label."
        )
    return CalibrationHierarchySeed(
        provider=HierarchyNode(FSA_SOURCE_NAME, provider_label),
        category=HierarchyCategory(
            f"{FSA_SOURCE_NAME}.{category_id}", category_label, FSA_SOURCE_NAME
        ),
        target_label=target_label,
    )


def _measure_label(measure: str) -> str:
    return calibration_variable_label(
        "us", FSA_SOURCE_NAME, FSA_CATEGORY_BY_MEASURE[measure]
    )


def _role_suffix(measure: str) -> str:
    return "borrowers" if measure == FSA_BORROWERS else "balance"


def _age_band_metadata(lower: float, upper: float) -> dict[str, str]:
    metadata = {_AGE_BOUND_STAMP_SOURCE_KEY: _AGE_BOUND_STAMP_FROM_CONSTRAINT_ROWS}
    if lower != -math.inf:
        metadata[_AGE_LOWER_BOUND_KEY] = _format_bound(lower)
    if upper != math.inf:
        metadata[_AGE_UPPER_BOUND_KEY] = _format_bound(upper)
    return metadata


def _balance_band_metadata(lower: float, upper: float) -> dict[str, str]:
    metadata: dict[str, str] = {}
    if lower != -math.inf:
        metadata[FSA_BALANCE_LOWER_BOUND_KEY] = _format_bound(lower)
    if upper != math.inf:
        metadata[FSA_BALANCE_UPPER_BOUND_KEY] = _format_bound(upper)
    return metadata


def _band_mask(
    values: np.ndarray,
    metadata: Mapping[str, str],
    lower_key: str,
    upper_key: str,
) -> np.ndarray:
    lower = float(metadata[lower_key]) if lower_key in metadata else -math.inf
    upper = float(metadata[upper_key]) if upper_key in metadata else math.inf
    return (values >= lower) & (values < upper)


def _constraint_bounds(fact: object, variable: str) -> tuple[float, float] | None:
    """``[lower, upper)`` from a fact's ``>=`` / ``<`` constraints on variable."""

    lower, upper = -math.inf, math.inf
    found = False
    for row in _constraint_rows(fact):
        if _str_at(row, "variable") != variable:
            continue
        operator = _str_at(row, "operator")
        value = _at(row, "value")
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(
                f"FSA fact {_source_record_id(fact)!r} has a non-numeric "
                f"{variable} bound {value!r}."
            )
        if operator == ">=":
            lower = float(value)
        elif operator == "<":
            upper = float(value)
        else:
            raise ValueError(
                f"FSA fact {_source_record_id(fact)!r} bounds {variable} with "
                f"{operator!r}; only >= and < are reviewed."
            )
        found = True
    return (lower, upper) if found else None


def _has_constraint(fact: object, variable: str) -> bool:
    return any(_str_at(row, "variable") == variable for row in _constraint_rows(fact))


def _location_reference_label(facts: list[object]) -> str | None:
    labels = {
        _str_at(fact, "period_coverage", "source_period_label") for fact in facts
    } - {""}
    return next(iter(labels)) if len(labels) == 1 else None


def _table(fact: object) -> str:
    parts = _str_at(fact, "layout", "record_set_id").split(".")
    return parts[1] if len(parts) > 1 else ""


def _month_key(period: str) -> tuple[int, int]:
    year, _, month = period.partition("-")
    try:
        return (int(year), int(month or 0))
    except ValueError:
        return (0, 0)


def _state_fips(fact: object) -> str | None:
    geography = _geography_id(fact)
    if _str_at(fact, "geography", "level") != "state":
        return None
    if not geography.startswith("0400000US"):
        return None
    return geography.removeprefix("0400000US")


def _value(fact: object) -> float:
    value = _at(fact, "value")
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(
            f"FSA fact {_source_record_id(fact)!r} has a non-numeric value {value!r}."
        )
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"FSA fact {_source_record_id(fact)!r} value {value!r}.")
    return value


def _format_float(value: float) -> str:
    return repr(float(value))


def _format_bound(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else repr(float(value))


def _source_name(fact: object) -> str:
    return _str_at(fact, "observed_measure", "source_name") or _str_at(
        fact, "source", "source_name"
    )


def _measure_id(fact: object) -> str:
    return _str_at(fact, "observed_measure", "source_measure_id") or _str_at(
        fact, "layout", "measure_id"
    )


def _source_record_id(fact: object) -> str:
    return _str_at(fact, "lineage", "source_record_id")


def _period(fact: object) -> str:
    return _str_at(fact, "period", "value")


def _geography_id(fact: object) -> str:
    return _str_at(fact, "geography", "id")


def _constraint_rows(fact: object) -> tuple[object, ...]:
    constraints = _at(fact, "universe_constraints", "constraints")
    return tuple(constraints) if isinstance(constraints, list | tuple) else ()


def _at(obj: object, *path: str) -> Any:
    current = obj
    for key in path:
        if current is None:
            return None
        if isinstance(current, Mapping):
            current = current.get(key)
        else:
            current = getattr(current, key, None)
    return current


def _str_at(obj: object, *path: str) -> str:
    value = _at(obj, *path)
    return "" if value is None else str(value)
