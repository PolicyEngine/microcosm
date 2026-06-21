"""UK bus-spending imputation plan (LCFS fares, ETB subsidy).

Declares the source stages and donor graph that impute the two DfT-anchored
bus consumption variables onto the UK population, and assembles them into a
:class:`~populace.build.plan.StagePlan`. The executable stage transforms are
injected by the build caller — there are no stubs or fallbacks, exactly as the
US plan works.

The calibration anchors for the two outputs live in
:mod:`populace.build.uk.bus_calibration_targets`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from importlib.resources import files

from populace.build.plan import DonorSpec, Stage, StagePlan
from populace.build.source_manifest import (
    SourceManifest,
    SourceStageSpec,
    load_source_manifest,
)
from populace.frame import Frame


def _load_uk_bus_source_manifest() -> SourceManifest:
    return load_source_manifest(files(__package__).joinpath("bus_source_stages.json"))


UK_BUS_SOURCE_MANIFEST: SourceManifest = _load_uk_bus_source_manifest()
UK_BUS_SOURCE_STAGE_SPECS: tuple[SourceStageSpec, ...] = UK_BUS_SOURCE_MANIFEST.stages
UK_BUS_NONNEGATIVE_SOURCE_OUTPUTS: frozenset[str] = frozenset(
    output
    for stage in UK_BUS_SOURCE_STAGE_SPECS
    for output in stage.nonnegative_outputs
)

UK_BUS_STAGE_NAMES: tuple[str, ...] = tuple(
    stage.stage for stage in UK_BUS_SOURCE_STAGE_SPECS
)

UK_BUS_DONORS: Mapping[str, DonorSpec] = {
    stage.stage: DonorSpec(survey=stage.survey, source=stage.source, notes=stage.notes)
    for stage in UK_BUS_SOURCE_STAGE_SPECS
}


def uk_bus_plan(
    implementations: Mapping[str, Callable[[Frame], Frame]],
) -> StagePlan:
    """Assemble the UK bus-spending imputation plan.

    Mirrors ``us_plan``: every declared stage needs an injected transform;
    there are no stubs or fallbacks by design.

    Args:
        implementations: ``stage name -> transform(frame) -> frame`` for every
            stage in :data:`UK_BUS_STAGE_NAMES`.

    Raises:
        ValueError: If an implementation is missing for a declared stage, or an
            unknown stage name is supplied.
    """
    missing = [name for name in UK_BUS_STAGE_NAMES if name not in implementations]
    if missing:
        raise ValueError(
            f"uk_bus_plan needs an implementation for every declared stage; "
            f"missing {missing}. There are no stubs or fallbacks by design."
        )
    unknown = sorted(set(implementations) - set(UK_BUS_STAGE_NAMES))
    if unknown:
        raise ValueError(
            f"Unknown stage implementation(s) {unknown}; declared stages "
            f"are {list(UK_BUS_STAGE_NAMES)}."
        )
    stage_map = UK_BUS_SOURCE_MANIFEST.stage_map()
    return StagePlan(
        Stage(
            name=name,
            transform=implementations[name],
            produces=stage_map[name].outputs,
            donor=UK_BUS_DONORS[name],
        )
        for name in UK_BUS_STAGE_NAMES
    )


__all__ = [
    "UK_BUS_DONORS",
    "UK_BUS_NONNEGATIVE_SOURCE_OUTPUTS",
    "UK_BUS_SOURCE_MANIFEST",
    "UK_BUS_SOURCE_STAGE_SPECS",
    "UK_BUS_STAGE_NAMES",
    "uk_bus_plan",
]
