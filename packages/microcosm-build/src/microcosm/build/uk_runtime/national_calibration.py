"""Ledger-backed calibration stage for the UK national build."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from importlib import resources
from typing import Any

import pandas as pd

from microcosm.build.ledger_targets import compile_ledger_target_references
from microcosm.build.plan import Stage
from microcosm.calibrate import calibrate, effective_sample_size
from microcosm.frame import Frame

__all__ = ["UKNationalCalibrationStage", "uk_national_calibration_stage"]


class UKNationalCalibrationStage:
    """Fail-closed national calibration transform and its manifest evidence."""

    def __init__(
        self,
        facts: Sequence[Mapping[str, Any]],
        *,
        references: Sequence[object] | None = None,
        epochs: int = 256,
        learning_rate: float = 0.02,
        max_weight_ratio: float = 10.0,
        seed: int = 0,
    ) -> None:
        from microcosm.build.country_spec import load_country_spec

        self.facts = tuple(facts)
        self.references = tuple(
            load_country_spec("uk").target_references
            if references is None
            else references
        )
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.max_weight_ratio = max_weight_ratio
        self.seed = seed
        self.manifest: dict[str, object] | None = None
        self.diagnostics: tuple[dict[str, object], ...] = ()

    def __call__(self, frame: Frame) -> Frame:
        registry = compile_ledger_target_references(
            self.facts, self.references, country="uk"
        )
        declared = len(self.references)
        resolved = len(registry.specs)
        if resolved != declared:
            raise RuntimeError(
                "UK national calibration resolved "
                f"{resolved} of {declared} activated target references."
            )
        prepared = _prepare_target_columns(frame, registry.specs)
        result = calibrate(
            prepared,
            registry.to_target_set(),
            weight_entity="household",
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            max_weight_ratio=self.max_weight_ratio,
            seed=self.seed,
        )
        if result.skipped or len(result.problem.names) != declared:
            skipped = [item.name for item in result.skipped]
            raise RuntimeError(
                "UK national calibration matrix did not contain every activated "
                f"reference: declared={declared}, rows={len(result.problem.names)}, "
                f"skipped={skipped}."
            )
        self.diagnostics = tuple(
            {
                "name": row.name,
                "estimate": row.final_estimate,
                "target": row.target,
                "relative_error": row.relative_error,
            }
            for row in result.diagnostics
        )
        ratios = result.weights / result.initial_weights
        self.manifest = {
            "activated_reference_count": declared,
            "resolved_reference_count": resolved,
            "matrix_target_count": len(result.problem.names),
            "loss": result.final_loss,
            "effective_sample_size": effective_sample_size(result.weights),
            "max_weight_ratio": float(ratios.max()),
            "max_weight_ratio_bound": self.max_weight_ratio,
        }
        return result.frame

    def checkpoint_metadata(self) -> Mapping[str, object]:
        if self.manifest is None:
            raise RuntimeError("UK national calibration has not run.")
        return {"calibration": self.manifest, "diagnostics": self.diagnostics}


def uk_national_calibration_stage(
    facts: Sequence[Mapping[str, Any]], **kwargs: Any
) -> Stage:
    """Return the named ordered-stage entry for national calibration."""

    transform = UKNationalCalibrationStage(facts, **kwargs)
    return Stage(name="national_calibration", transform=transform)


def _contract_targets() -> dict[str, Mapping[str, Any]]:
    payload = json.loads(
        resources.files("microcosm.build.uk")
        .joinpath("uk_national_targets.json")
        .read_text()
    )
    return {row["target_id"]: row for row in payload["targets"]}


def _prepare_target_columns(frame: Frame, specs: Sequence[object]) -> Frame:
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables.update({name: frame.link(name).copy() for name in frame.links})
    contracts = _contract_targets()
    for spec in specs:
        target_id = spec.metadata["contract_target_id"]
        binding = contracts[target_id]["bindings"]["policyengine"]
        entity = spec.entity
        table = tables[entity]
        if spec.measure in table:
            continue
        value_name = binding["value_variable"]
        if value_name in {"person_count", "household_count", "benunit_count"}:
            values = pd.Series(1.0, index=table.index)
        else:
            if value_name not in table:
                raise ValueError(
                    f"Activated UK target {target_id!r} requires missing "
                    f"{entity} column {value_name!r}."
                )
            values = table[value_name].astype(float)
        mask = pd.Series(True, index=table.index)
        for condition in binding.get("filters", ()):
            variable = condition["variable"]
            if variable not in table:
                raise ValueError(
                    f"Activated UK target {target_id!r} requires missing "
                    f"{entity} filter column {variable!r}."
                )
            mask &= _compare(table[variable], condition)
        if binding.get("household_conditions"):
            if entity != "household":
                raise ValueError(
                    f"Activated UK target {target_id!r} declares household "
                    f"conditions on entity {entity!r}."
                )
            for condition in binding["household_conditions"]:
                mask &= _household_condition(tables, condition, table)
        table[spec.measure] = values.where(mask, 0.0)
    weights = {entity: frame.weights_for(entity) for entity in frame.weighted_entities}
    return Frame(
        tables,
        frame.schema,
        weights,
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def _compare(values: pd.Series, condition: Mapping[str, Any]) -> pd.Series:
    operator = condition.get("operator")
    if operator is None:
        operator, expected = "==", condition["equals"]
    else:
        expected = condition["value"]
    operations = {
        "==": values.eq,
        ">": values.gt,
        ">=": values.ge,
        "<": values.lt,
        "<=": values.le,
    }
    if operator not in operations:
        raise ValueError(f"Unsupported UK target condition operator {operator!r}.")
    return operations[operator](expected)


def _household_condition(
    tables: Mapping[str, pd.DataFrame],
    condition: Mapping[str, Any],
    households: pd.DataFrame,
) -> pd.Series:
    entity = condition["entity"]
    source = tables[entity]
    if entity == "household":
        household_ids = source["household_id"]
    else:
        people = tables["person"]
        entity_membership = f"person_{entity}_id"
        if entity_membership not in people:
            raise ValueError(f"UK person table is missing {entity_membership!r}.")
        group_to_household = (
            people[[entity_membership, "person_household_id"]]
            .drop_duplicates()
            .set_index(entity_membership)["person_household_id"]
        )
        if group_to_household.index.has_duplicates:
            raise ValueError(f"UK {entity} groups span multiple households.")
        household_ids = source[f"{entity}_id"].map(group_to_household)
    reduce = condition["reduce"]
    if reduce == "any":
        matched = _compare(source[condition["variable"]], condition)
        aggregate = matched.groupby(household_ids).any().astype(float)
        expected_condition = {"operator": "==", "value": True}
    elif reduce == "sum":
        aggregate = source[condition["variable"]].groupby(household_ids).sum()
        expected_condition = condition
    elif reduce == "count":
        aggregate = source[condition["variable"]].groupby(household_ids).count()
        expected_condition = condition
    else:
        raise ValueError(f"Unsupported UK household reduction {reduce!r}.")
    ids = households["household_id"]
    return _compare(ids.map(aggregate).fillna(0.0), expected_condition).set_axis(
        households.index
    )
