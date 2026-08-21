"""Ledger-backed calibration stage for the UK national build."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.plan import Stage
from microcosm.build.uk_runtime.content_identity import uk_frame_content_identity
from microcosm.build.uk_runtime.ledger_targets import (
    UKLedgerTargetCompilation,
    materialize_uk_ledger_targets,
)
from microcosm.build.uk_runtime.national_doctrine import (
    UK_NATIONAL_SOLVE_DOCTRINE,
    UKNationalSolveDoctrine,
)
from microcosm.calibrate import (
    CalibrationResult,
    TargetRegistry,
    calibrate,
    effective_sample_size,
)
from microcosm.frame import Frame, WeightKind, Weights

__all__ = [
    "UKNationalCalibrationStage",
    "national_calibration_mass_reason",
    "uk_national_calibration_stage",
]


class UKNationalCalibrationStage:
    """Fail-closed national calibration transform and its manifest evidence."""

    def __init__(
        self,
        registry: UKLedgerTargetCompilation | TargetRegistry,
        *,
        period: int,
        doctrine: UKNationalSolveDoctrine = UK_NATIONAL_SOLVE_DOCTRINE,
    ) -> None:
        self.compilation = (
            registry
            if isinstance(registry, UKLedgerTargetCompilation)
            else UKLedgerTargetCompilation(registry=registry, unsupported=())
        )
        self.registry = self.compilation.registry
        # The materialization period is the declared calibration year the
        # registry was compiled at — never the input frame's base-year
        # time_period, which lags it (survey 2024, calibration 2025).
        if not isinstance(period, int) or isinstance(period, bool) or period <= 0:
            raise ValueError(
                f"period must be the declared calibration year, got {period!r}."
            )
        self.period = period
        self.doctrine = doctrine
        self.manifest: dict[str, object] | None = None
        self.diagnostics: tuple[dict[str, object], ...] = ()
        self.solve_result: CalibrationResult | None = None
        self.output_content_identity: str | None = None

    def __call__(self, frame: Frame) -> Frame:
        declared = len(self.registry.specs) + len(self.compilation.unsupported)
        resolved = len(self.registry.specs)
        if self.compilation.unsupported:
            raise RuntimeError(
                "UK national calibration resolved "
                f"{resolved} of {declared} activated target references; "
                f"unsupported={self.compilation.unsupported!r}."
            )
        adapter = _FrameTargetAdapter(frame)
        materialized = materialize_uk_ledger_targets(
            adapter,
            self.registry,
            period=self.period,
        )
        if materialized.skipped:
            skipped = [skip.__dict__ for skip in materialized.skipped]
            raise RuntimeError(
                "UK national calibration could not materialize every activated "
                f"target reference: skipped={skipped}."
            )
        prepared = adapter.frame()
        mass_reason = national_calibration_mass_reason(
            spec.family for spec in self.registry.specs
        )
        mass_log_records_before_calibration = len(frame.mass_log)
        result = calibrate(
            prepared,
            self.registry.to_target_set(),
            weight_entity="household",
            epochs=self.doctrine.epochs,
            learning_rate=self.doctrine.learning_rate,
            mass=self.doctrine.mass_rule,
            mass_reason=mass_reason,
            max_weight_ratio=self.doctrine.max_weight_ratio,
            seed=self.doctrine.seed,
            l0_lambda=self.doctrine.l0_lambda,
            target_loss_cap=self.doctrine.target_loss_cap,
        )
        if result.skipped or len(result.problem.names) != declared:
            skipped = [item.name for item in result.skipped]
            raise RuntimeError(
                "UK national calibration matrix did not contain every activated "
                f"reference: declared={declared}, rows={len(result.problem.names)}, "
                f"skipped={skipped}."
            )
        clean_frame = adapter.restore(result.frame)
        self.solve_result = result
        calibration_record = _post_solve_calibration_record(
            frame,
            clean_frame,
            before_count=mass_log_records_before_calibration,
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
        old_total = float(calibration_record.old_total)
        new_total = float(calibration_record.new_total)
        before_kind = frame.weights_for("household").kind
        after_kind = clean_frame.weights_for("household").kind
        self.manifest = {
            "activated_reference_count": declared,
            "resolved_reference_count": resolved,
            "matrix_target_count": len(result.problem.names),
            "loss": result.final_loss,
            "effective_sample_size": effective_sample_size(result.weights),
            "max_weight_ratio": float(ratios.max()),
            "max_weight_ratio_bound": self.doctrine.max_weight_ratio,
            "target_materialization": materialized.report(),
            "weights": {
                "household_weight_kind": after_kind.value,
                "household_weight_kind_chain": [
                    {"stage": "staging", "kind": before_kind.value},
                    {"stage": "national_calibration", "kind": after_kind.value},
                ],
                "mass_log_records_before_calibration": (
                    mass_log_records_before_calibration
                ),
                "mass_log_records": len(clean_frame.mass_log),
                "calibration_mass_change": {
                    "entity": str(calibration_record.entity),
                    "old_total": old_total,
                    "new_total": new_total,
                    "relative_shift": (new_total - old_total) / old_total,
                    "declared_factor": calibration_record.declared_factor,
                    "reason": str(calibration_record.reason),
                },
            },
            "solve": {
                "n_targets": len(result.problem.names),
                "n_households": len(clean_frame.table("household")),
                "initial_loss": float(result.initial_loss),
                "final_loss": float(result.final_loss),
                "n_nonzero": int(np.count_nonzero(result.weights)),
            },
            "parameters": {"doctrine": _doctrine_bounds(self.doctrine)},
        }
        self.output_content_identity = uk_frame_content_identity(clean_frame)
        return clean_frame

    def checkpoint_metadata(self) -> Mapping[str, object]:
        if self.manifest is None:
            raise RuntimeError("UK national calibration has not run.")
        return {
            "calibration": self.manifest,
            "diagnostics": self.diagnostics,
            "output_content_identity": self.output_content_identity,
        }

    def resume_from_checkpoint(
        self,
        metadata: Mapping[str, object],
        frame: Frame,
    ) -> None:
        """Rehydrate completed calibration evidence from its checkpoint record."""

        calibration = metadata.get("calibration")
        diagnostics = metadata.get("diagnostics")
        output_identity = metadata.get("output_content_identity")
        count_keys = (
            "activated_reference_count",
            "resolved_reference_count",
            "matrix_target_count",
        )
        if (
            not isinstance(calibration, Mapping)
            or not all(key in calibration for key in count_keys)
            or not isinstance(diagnostics, list)
            or not all(isinstance(row, Mapping) for row in diagnostics)
            or not isinstance(output_identity, str)
            or not output_identity
        ):
            raise RuntimeError(
                "UK national calibration resume requires the checkpoint record "
                "to carry calibration counts, diagnostics, and output content "
                "identity; a record without them cannot feed the calibration "
                "reference coverage gate or the drift check."
            )
        if uk_frame_content_identity(frame) != output_identity:
            raise RuntimeError(
                "UK national calibration checkpoint content does not match its "
                "recorded output identity; refusing to resume from a drifted "
                "record."
            )
        self.manifest = dict(calibration)
        self.diagnostics = tuple(dict(row) for row in diagnostics)
        self.output_content_identity = output_identity


def uk_national_calibration_stage(
    registry: UKLedgerTargetCompilation, **kwargs: Any
) -> Stage:
    """Return the named ordered-stage entry for national calibration."""

    transform = UKNationalCalibrationStage(registry, **kwargs)
    return Stage(name="national_calibration", transform=transform)


def national_calibration_mass_reason(bound_families: Sequence[str]) -> str:
    """The mass-record reason a national doctrine calibration declares."""

    families = sorted({str(name) for name in bound_families})
    if not families or any(not name.strip() for name in families):
        raise ValueError("bound_families must name at least one target family.")
    return (
        "National doctrine calibration to bound target family(ies) "
        f"{', '.join(families)}; total household mass moved with the targets."
    )


def _post_solve_calibration_record(
    before: Frame,
    after: Frame,
    *,
    before_count: int,
):
    if after.weights_for("household").kind is not WeightKind.CALIBRATED:
        raise RuntimeError(
            "UK national calibration returned household weights whose kind is "
            f"{after.weights_for('household').kind.value!r}, not 'calibrated'."
        )
    if len(after.mass_log) != before_count + 1:
        raise RuntimeError(
            "UK national calibration must append exactly one mass record; "
            f"before={before_count}, after={len(after.mass_log)}."
        )
    if before.mass_log != after.mass_log[:before_count]:
        raise RuntimeError(
            "UK national calibration changed pre-existing mass-log records."
        )
    record = after.mass_log[-1]
    if record.entity != "household" or "calibration" not in record.reason:
        raise RuntimeError(
            "UK national calibration latest mass record is not the calibration "
            f"record: entity={record.entity!r}, reason={record.reason!r}."
        )
    return record


def _doctrine_bounds(doctrine: UKNationalSolveDoctrine) -> dict[str, object]:
    return {
        "epochs": doctrine.epochs,
        "learning_rate": doctrine.learning_rate,
        "max_weight_ratio": doctrine.max_weight_ratio,
        "seed": doctrine.seed,
        "target_loss_cap": doctrine.target_loss_cap,
        "scale_rule": doctrine.scale_rule,
        "target_weight_rule": doctrine.target_weight_rule,
        "mass_rule": doctrine.mass_rule,
        "l0_lambda": doctrine.l0_lambda,
    }


class _FrameTargetAdapter:
    def __init__(self, frame: Frame) -> None:
        self._frame = frame
        self.tables = {entity: frame.table(entity).copy() for entity in frame.entities}
        self.tables.update({name: frame.link(name).copy() for name in frame.links})
        self._original_tables = {
            name: table.copy() for name, table in self.tables.items()
        }
        self._scratch_columns: dict[str, set[str]] = {
            name: set() for name in self.tables
        }

    def entity_length(self, entity: str) -> int:
        return len(self.tables[entity])

    def column(self, entity: str, variable: str) -> np.ndarray:
        if variable not in self.tables[entity]:
            raise KeyError(variable)
        return np.asarray(self.tables[entity][variable])

    def set_column(self, entity: str, variable: str, values: object) -> None:
        table = self.tables[entity]
        if variable not in self._original_tables[entity]:
            self._scratch_columns[entity].add(variable)
        table[variable] = np.asarray(values, dtype=float)

    def parameter(self, parameter: str, period: int | str) -> float:
        from microcosm.build.uk_runtime.ledger_targets import UKPolicyEngineAdapter

        return UKPolicyEngineAdapter(None).parameter(parameter, period)

    def household_condition_mask(
        self,
        target_entity: str,
        condition: Mapping[str, Any],
    ) -> np.ndarray:
        if target_entity != "household":
            raise ValueError(
                "household_conditions can only gate household-grain targets, "
                f"got {target_entity!r}."
            )
        entity = str(condition.get("entity") or "household")
        source = self.tables[entity]
        if entity == "household":
            household_ids = source["household_id"]
        else:
            household_ids = self._group_household_ids(entity)
        reduce = str(condition.get("reduce") or "any")
        if reduce in {"any", "any_child_under"}:
            matched = _compare_series(source[str(condition["variable"])], condition)
            aggregate = matched.groupby(household_ids).any().astype(float)
            comparison = {"operator": "==", "value": True}
        elif reduce == "sum":
            aggregate = source[str(condition["variable"])].groupby(household_ids).sum()
            comparison = condition
        elif reduce == "count":
            aggregate = (
                source[str(condition["variable"])].groupby(household_ids).count()
            )
            comparison = condition
        else:
            raise ValueError(f"Unsupported UK household reduction {reduce!r}.")
        values = self.tables["household"]["household_id"].map(aggregate).fillna(0.0)
        return np.asarray(_compare_series(values, comparison), dtype=bool)

    def frame(self) -> Frame:
        return Frame(
            self.tables,
            self._frame.schema,
            {
                entity: self._frame.weights_for(entity)
                for entity in self._frame.weighted_entities
            },
            self._frame.strata,
            mass_log=self._frame.mass_log,
            metadata=self._frame.metadata,
        )

    def restore(self, calibrated: Frame) -> Frame:
        tables = {name: table.copy() for name, table in self._original_tables.items()}
        return Frame(
            tables,
            calibrated.schema,
            {
                entity: Weights(
                    calibrated.weights_for(entity).values,
                    calibrated.weights_for(entity).kind,
                )
                for entity in calibrated.weighted_entities
            },
            calibrated.strata,
            mass_log=calibrated.mass_log,
            metadata=calibrated.metadata,
        )

    def _group_household_ids(self, entity: str) -> pd.Series:
        people = self.tables["person"]
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
        return self.tables[entity][f"{entity}_id"].map(group_to_household)


def _compare_series(values: pd.Series, condition: Mapping[str, Any]) -> pd.Series:
    operator = condition.get("operator")
    if operator is None:
        operator, expected = "==", condition["equals"]
    else:
        expected = condition["value"]
    if operator == "==":
        return values.eq(expected)
    if operator == "!=":
        return values.ne(expected)
    if operator == "in":
        return values.isin(expected)
    operations = {
        ">": values.gt,
        ">=": values.ge,
        "<": values.lt,
        "<=": values.le,
    }
    if operator not in operations:
        raise ValueError(f"Unsupported UK target condition operator {operator!r}.")
    return operations[operator](expected)
