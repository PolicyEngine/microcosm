"""UK household-wealth imputation plan (WAS holdings, incl. ISA split).

Declares the source stage and donor graph that impute household wealth
holdings onto the UK population from the ONS Wealth and Assets Survey, and
assembles them into a :class:`~populace.build.plan.StagePlan`. The executable
stage transforms are injected by the build caller — there are no stubs or
fallbacks, exactly as the US and bus plans work.

The stage surfaces ``cash_isa`` and ``stocks_and_shares_isa`` as standalone
outputs (the legacy archived UK data imputation folded investment ISAs into
``corporate_wealth`` and never represented cash ISAs);
``stocks_and_shares_isa`` is also folded into ``corporate_wealth`` for
back-compatibility.
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


def _load_uk_wealth_source_manifest() -> SourceManifest:
    return load_source_manifest(
        files(__package__).joinpath("wealth_source_stages.json")
    )


UK_WEALTH_SOURCE_MANIFEST: SourceManifest = _load_uk_wealth_source_manifest()
UK_WEALTH_SOURCE_STAGE_SPECS: tuple[SourceStageSpec, ...] = (
    UK_WEALTH_SOURCE_MANIFEST.stages
)
UK_WEALTH_NONNEGATIVE_SOURCE_OUTPUTS: frozenset[str] = frozenset(
    output
    for stage in UK_WEALTH_SOURCE_STAGE_SPECS
    for output in stage.nonnegative_outputs
)

UK_WEALTH_STAGE_NAMES: tuple[str, ...] = tuple(
    stage.stage for stage in UK_WEALTH_SOURCE_STAGE_SPECS
)

UK_WEALTH_DONORS: Mapping[str, DonorSpec] = {
    stage.stage: DonorSpec(survey=stage.survey, source=stage.source, notes=stage.notes)
    for stage in UK_WEALTH_SOURCE_STAGE_SPECS
}


def uk_wealth_plan(
    implementations: Mapping[str, Callable[[Frame], Frame]],
) -> StagePlan:
    """Assemble the UK household-wealth imputation plan.

    Mirrors ``us_plan`` / ``uk_bus_plan``: every declared stage needs an
    injected transform; there are no stubs or fallbacks by design.

    Args:
        implementations: ``stage name -> transform(frame) -> frame`` for every
            stage in :data:`UK_WEALTH_STAGE_NAMES`.

    Raises:
        ValueError: If an implementation is missing for a declared stage, or an
            unknown stage name is supplied.
    """
    missing = [
        name for name in UK_WEALTH_STAGE_NAMES if name not in implementations
    ]
    if missing:
        raise ValueError(
            f"uk_wealth_plan needs an implementation for every declared stage; "
            f"missing {missing}. There are no stubs or fallbacks by design."
        )
    unknown = sorted(set(implementations) - set(UK_WEALTH_STAGE_NAMES))
    if unknown:
        raise ValueError(
            f"Unknown stage implementation(s) {unknown}; declared stages "
            f"are {list(UK_WEALTH_STAGE_NAMES)}."
        )
    stage_map = UK_WEALTH_SOURCE_MANIFEST.stage_map()
    return StagePlan(
        Stage(
            name=name,
            transform=implementations[name],
            produces=stage_map[name].outputs,
            donor=UK_WEALTH_DONORS[name],
        )
        for name in UK_WEALTH_STAGE_NAMES
    )


__all__ = [
    "UK_WEALTH_DONORS",
    "UK_WEALTH_NONNEGATIVE_SOURCE_OUTPUTS",
    "UK_WEALTH_SOURCE_MANIFEST",
    "UK_WEALTH_SOURCE_STAGE_SPECS",
    "UK_WEALTH_STAGE_NAMES",
    "uk_wealth_plan",
]
