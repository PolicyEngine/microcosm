"""UK Ledger target-reference compilation and materialization helpers."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from importlib import resources as importlib_resources
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.country_spec import load_country_spec
from microcosm.build.ledger_targets import (
    LedgerTargetReference,
    _fact_matches_selector,
    compile_ledger_target_references,
)
from microcosm.build.target_materialization import (
    TargetMaterializationResult,
    materialize_target_bindings,
)
from microcosm.build.uk_runtime.cgt_calibration import uk_cgt_annual_exempt_amount
from microcosm.calibrate import TargetRegistry
from microcosm.frame import Frame


@dataclass(frozen=True)
class UKLedgerTargetCompilation:
    """Compiled UK Ledger target registry and unsupported-reference report."""

    registry: TargetRegistry
    unsupported: tuple[dict[str, str], ...]


def compile_uk_target_registry(
    facts: Iterable[Mapping[str, Any]],
    *,
    target_period: int | str,
) -> UKLedgerTargetCompilation:
    """Compile packaged UK Ledger references against consumer fact rows."""

    fact_rows = tuple(facts)
    spec = load_country_spec("uk")
    compiled = []
    unsupported: list[dict[str, str]] = []
    for reference in spec.target_references:
        restamped = LedgerTargetReference(
            **{**reference.__dict__, "period": target_period}
        )
        candidate_facts = _candidate_facts_for_reference(fact_rows, restamped)
        try:
            registry = compile_ledger_target_references(
                candidate_facts,
                [restamped],
                country="uk",
            )
        except ValueError as error:
            unsupported.append(
                {
                    "name": reference.name,
                    "period": target_period,
                    "reason": str(error),
                }
            )
        else:
            compiled.extend(registry.specs)
    return UKLedgerTargetCompilation(
        TargetRegistry(compiled, country="uk"),
        tuple(unsupported),
    )


def _candidate_facts_for_reference(
    facts: tuple[Mapping[str, Any], ...],
    reference: LedgerTargetReference,
) -> tuple[Mapping[str, Any], ...]:
    if not reference.ledger_selector:
        return facts
    return tuple(
        fact for fact in facts if _fact_matches_selector(fact, reference.ledger_selector)
    )


def materialize_uk_ledger_targets(
    adapter: Any,
    registry: TargetRegistry,
    *,
    period: int | str,
) -> TargetMaterializationResult:
    """Materialize compiled UK Ledger target bindings on an adapter."""

    contract = _uk_contract_targets()
    return materialize_target_bindings(
        adapter,
        registry,
        contract,
        period=period,
        providers={
            "parameter_gated_threshold": _uk_parameter_gated_threshold,
            "baseline_flag_crosstab": _uk_baseline_flag_crosstab,
            "input_substitution_counterfactual": _uk_input_substitution,
        },
    )


class UKPolicyEngineAdapter:
    """Small adapter around policyengine-uk simulation-like objects."""

    def __init__(self, simulation: Any):
        self.simulation = simulation
        self.tables: dict[str, dict[str, np.ndarray]] = {}

    def column(self, entity: str, variable: str) -> np.ndarray:
        table = self.tables.get(entity, {})
        if variable in table:
            return np.asarray(table[variable], dtype=float)
        values = self.simulation.calculate(variable)
        return np.asarray(values, dtype=float)

    def set_column(self, entity: str, variable: str, values: object) -> None:
        self.tables.setdefault(entity, {})[variable] = np.asarray(values, dtype=float)

    def parameter(self, parameter: str, period: int | str) -> float:
        if parameter in {
            "cgt_calibration.uk_cgt_annual_exempt_amount",
            "gov.hmrc.cgt.annual_exempt_amount",
        }:
            return uk_cgt_annual_exempt_amount(period)
        raise KeyError(parameter)


#: Published-fact reductions rewritten to the internal reduction that carries
#: the same meaning on our frame. Facts keep the semantics of the source that
#: published them; translating those onto the model's own concepts is our job,
#: and a fact we cannot phrase internally gets translated and recorded, never
#: dropped.
#:
#: ``any_child_under`` (DWP Stat-Xplore, Scottish UC households with a child
#: under 1) names a dependent-child concept the model does not carry. There is
#: no ``is_child`` column because the model has no need of one: dependency is
#: derived from age where it is wanted. So "any child under N" is exactly "any
#: person aged under N" here, and the condition already supplies the age bound.
#: The rewrite is declared rather than aliased at the call site so it stays
#: greppable, testable, and visible in review.
UK_TRANSLATED_HOUSEHOLD_REDUCTIONS: Mapping[str, str] = {
    "any_child_under": "any",
}


class UKFrameTargetAdapter:
    """Frame-backed target materialization adapter for UK calibration stages."""

    def __init__(self, frame: Frame):
        self.frame = frame
        self.tables = {entity: frame.table(entity).copy() for entity in frame.entities}
        self.link_tables = {name: frame.link(name).copy() for name in frame.links}

    def has_column(self, entity: str, variable: str) -> bool:
        return variable in self.tables[entity]

    def column(self, entity: str, variable: str) -> np.ndarray:
        table = self.tables[entity]
        count_variables = {
            "person_count",
            "household_count",
            "benunit_count",
            f"{entity}_count",
        }
        if variable in count_variables:
            return np.ones(len(table), dtype=float)
        if variable not in table:
            raise KeyError(f"{entity}.{variable}")
        return np.asarray(table[variable])

    def set_column(self, entity: str, variable: str, values: object) -> None:
        self.tables[entity][variable] = np.asarray(values)

    def parameter(self, parameter: str, period: int | str) -> float:
        if parameter == "gov.hmrc.cgt.annual_exempt_amount":
            return uk_cgt_annual_exempt_amount(period)
        raise KeyError(parameter)

    def counterfactual_delta(
        self,
        binding: Mapping[str, Any],
        period: int | str,
    ) -> np.ndarray:
        del period
        entity = str(binding.get("from_entity") or "person")
        metric_name = str(binding.get("metric_name") or "")
        if metric_name and metric_name in self.tables[entity]:
            return np.asarray(self.tables[entity][metric_name], dtype=float)
        raise ValueError(
            "frame does not carry precomputed counterfactual delta "
            f"{metric_name!r}"
        )

    def _household_ids_for(self, entity: str) -> Any:
        """The household each row of ``entity`` belongs to, row-aligned."""

        source = self.tables[entity]
        if entity == "household":
            return source["household_id"]
        if entity == "person":
            # People sit directly in a household; only group entities need the
            # membership lookup below, whose column would be the nonexistent
            # "person_person_id" here.
            return source["person_household_id"]
        people = self.tables["person"]
        entity_membership = f"person_{entity}_id"
        if entity_membership not in people:
            raise KeyError(entity_membership)
        group_to_household = (
            people[[entity_membership, "person_household_id"]]
            .drop_duplicates()
            .set_index(entity_membership)["person_household_id"]
        )
        if group_to_household.index.has_duplicates:
            raise ValueError(f"UK {entity} groups span multiple households.")
        return source[f"{entity}_id"].map(group_to_household)

    def entity_reduction(self, reduction: Mapping[str, Any]) -> np.ndarray:
        """Reduce a member-level variable to a household-grain value.

        The numeric sibling of :meth:`household_condition`: where that answers
        "does this household satisfy the predicate", this answers "what does
        this household's members sum to". A count target declared over a
        boolean member variable — the number of children, not whether the
        household has any — is only honest through this path; collapsing the
        boolean with ``any`` would publish an indicator against a count.
        """

        entity = str(reduction.get("entity") or "person")
        source = self.tables[entity]
        variable = str(reduction["variable"])
        if variable not in source:
            raise KeyError(f"{entity}.{variable}")
        household_ids = self._household_ids_for(entity)
        reduce = str(reduction.get("reduce") or "sum")
        values = source[variable]
        if reduce == "sum":
            aggregate = values.astype(float).groupby(household_ids).sum()
        elif reduce == "count":
            aggregate = values.groupby(household_ids).count()
        else:
            raise ValueError(f"Unsupported UK entity reduction {reduce!r}.")
        households = self.tables["household"]
        return np.asarray(
            households["household_id"].map(aggregate).fillna(0.0),
            dtype=float,
        )

    def household_condition(self, condition: Mapping[str, Any]) -> np.ndarray:
        entity = str(condition.get("entity") or "household")
        source = self.tables[entity]
        household_ids = self._household_ids_for(entity)

        published = str(condition["reduce"])
        reduce = UK_TRANSLATED_HOUSEHOLD_REDUCTIONS.get(published, published)
        variable = str(condition["variable"])
        if reduce == "any":
            matched = _compare_series(source[variable], condition)
            aggregate = matched.groupby(household_ids).any().astype(float)
            expected = {"operator": "==", "value": True}
        elif reduce == "sum":
            aggregate = source[variable].groupby(household_ids).sum()
            expected = condition
        elif reduce == "count":
            aggregate = source[variable].groupby(household_ids).count()
            expected = condition
        else:
            raise ValueError(
                f"Unsupported UK household reduction {published!r}."
                if published == reduce
                else f"Unsupported UK household reduction {published!r} "
                f"(translated to {reduce!r})."
            )

        households = self.tables["household"]
        ids = households["household_id"]
        return np.asarray(
            _compare_series(ids.map(aggregate).fillna(0.0), expected),
            dtype=bool,
        )

    def to_frame(self) -> Frame:
        tables = {**self.tables, **self.link_tables}
        weights = {
            entity: self.frame.weights_for(entity)
            for entity in self.frame.weighted_entities
        }
        return Frame(
            tables,
            self.frame.schema,
            weights,
            self.frame.strata,
            mass_log=self.frame.mass_log,
            metadata=self.frame.metadata,
        )


def _uk_contract_targets() -> dict[str, Mapping[str, Any]]:
    payload = (
        importlib_resources.files("microcosm.build.uk")
        .joinpath("uk_national_targets.json")
        .read_text()
    )
    contract = json.loads(payload)
    return {target["target_id"]: target for target in contract["targets"]}


def _uk_parameter_gated_threshold(
    adapter: Any,
    binding: Mapping[str, Any],
    period: int | str,
) -> np.ndarray:
    from microcosm.build.target_materialization import parameter_gated_threshold

    return parameter_gated_threshold(adapter, binding, period)


def _uk_baseline_flag_crosstab(
    adapter: Any,
    binding: Mapping[str, Any],
    period: int | str,
) -> np.ndarray:
    from microcosm.build.target_materialization import baseline_flag_crosstab

    return baseline_flag_crosstab(adapter, binding, period)


def _uk_input_substitution(
    adapter: Any,
    binding: Mapping[str, Any],
    period: int | str,
) -> np.ndarray:
    from microcosm.build.target_materialization import input_substitution_counterfactual

    return input_substitution_counterfactual(adapter, binding, period)


def _compare_series(
    values: pd.Series,
    condition: Mapping[str, Any],
) -> pd.Series:
    operator = condition.get("operator")
    if operator is None:
        operator, expected = "==", condition["equals"]
    else:
        expected = condition["value"]
    if operator == "in":
        return values.isin(expected)
    operations = {
        "==": values.eq,
        "!=": values.ne,
        ">": values.gt,
        ">=": values.ge,
        "<": values.lt,
        "<=": values.le,
    }
    if operator not in operations:
        raise ValueError(f"Unsupported UK target condition operator {operator!r}.")
    return operations[operator](expected)
