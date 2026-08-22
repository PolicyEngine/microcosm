"""Ledger-backed calibration stage for the UK national build."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from microcosm.build.ledger_targets import compile_ledger_target_references
from microcosm.build.plan import Stage
from microcosm.build.uk_runtime.content_identity import uk_frame_content_identity
from microcosm.build.uk_runtime.ledger_targets import (
    UKFrameTargetAdapter,
    materialize_uk_ledger_targets,
)
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
        self.output_content_identity: str | None = None

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
        materialized = materialize_uk_ledger_targets(
            UKFrameTargetAdapter(frame),
            registry,
            period=_registry_materialization_period(registry.specs),
        )
        if materialized.skipped:
            skipped = [
                {
                    "name": item.name,
                    "measure": item.measure,
                    "reason": item.reason,
                }
                for item in materialized.skipped
            ]
            raise RuntimeError(
                "UK national calibration could not materialize every "
                f"activated reference measure: {skipped}."
            )
        prepared = materialized.adapter.to_frame()
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
        self.output_content_identity = uk_frame_content_identity(result.frame)
        return result.frame

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
    facts: Sequence[Mapping[str, Any]], **kwargs: Any
) -> Stage:
    """Return the named ordered-stage entry for national calibration."""

    transform = UKNationalCalibrationStage(facts, **kwargs)
    return Stage(name="national_calibration", transform=transform)


def _registry_materialization_period(specs: Sequence[object]) -> int | str:
    periods = {spec.period for spec in specs}
    if not periods:
        raise RuntimeError("UK national calibration has no target specs to materialize.")
    if len(periods) != 1:
        raise RuntimeError(
            "UK national calibration cannot materialize mixed-period target "
            f"specs: {sorted(str(period) for period in periods)}."
        )
    return next(iter(periods))
