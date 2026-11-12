"""Each geographic clone's geography key, from its source and clone path (microcosm#931).

G1b of the clone-first plan runs right after the geographic expansion
(:mod:`~microcosm.build.uk_runtime.geographic_expand`). It gives every
household the key the area assignment draws on, ``geography_household_key``:
``household_draw_key`` over the FRS source, the household's FRS vintage, its
true source and its copy path, the key the dense line's own identity node gave
its pool before the dense line took its clones from the spine (microcosm#931).

- **Path.** The shared reader (``source_lineage.copy_paths`` on
  ``household_lineage.UK_COPY_LINEAGE``) reads it from the lineage columns,
  never from ids, weights or values. Before any copy step it is ``()`` for
  clone 0 and ``(("geographic_support", k),)`` for clone ``k``.
- **Vintage.** The FRS vintage frs_release.json pins for the household's
  ``source_year`` (:func:`frs_vintages_by_source_year`). A year it pins no
  vintage for is refused.
- **Dtype.** The key is written in the graph's ``string`` dtype (pandas'
  ``StringDtype("python")``), the dtype the graph stores a ``string`` cell
  at, so it is not a legacy object-string column. The step names the dtype
  itself, as ``atomic_household_lineage`` does, rather than through the
  graph's token table: a stage's transform computes through the Frame's
  modules alone, never the executor's.

So clone 0's key is the key the same rule gives the base household, clone
``k``'s the key it gives that household's clone ``k``, and growing K never
changes an existing household's key.

Refusals give counts, columns, dtypes and declared constants only, never a
key, an id or a year of a row. The evidence gives the distinct source
households rounded through ``receipt_counts`` (a household's clones counted
once), that the keys are unique, the source and the vintages; no key is ever
printed.

The spine's ``geographic_identity`` stage runs this step, right after
``geographic_expand``. Its declared operation, ``key_households_by_clone_path``,
must be the reviewed one (:func:`geographic_identity_operation_parameters`),
and its one artifact the FRS release, with ``runtime_sha256_required``, whose
bytes enter the stage's contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np
import pandas as pd

from microcosm.build.source_lineage import copy_paths
from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.uk_runtime.atomic_household_identity import household_draw_key
from microcosm.build.uk_runtime.cgt_structure import _assert_closed_world_operations
from microcosm.build.uk_runtime.frs_release import UKFRSRelease, load_uk_frs_release
from microcosm.build.uk_runtime.geographic_clones import (
    GEOGRAPHY_HOUSEHOLD_KEY_COLUMN,
    SOURCE_YEAR_COLUMN,
    ladder_clone_index_column,
)
from microcosm.build.uk_runtime.household_lineage import (
    GEOGRAPHIC_SUPPORT_BRANCH,
    UK_COPY_LINEAGE,
)
from microcosm.build.uk_runtime.national_frame import validate_uk_national_frame
from microcosm.build.uk_runtime.receipt_counts import (
    distinct_carriers,
    rounded_count,
)
from microcosm.frame import Frame, Weights
from microcosm.frame.concepts import SOURCE_HOUSEHOLD_ID_COLUMN

__all__ = [
    "FRS_RELEASE_ARTIFACT_ROLE",
    "FRS_RELEASE_RESOURCE",
    "GEOGRAPHIC_IDENTITY_INPUTS",
    "GEOGRAPHIC_IDENTITY_OPERATION_KIND",
    "GEOGRAPHIC_IDENTITY_SOURCE",
    "GEOGRAPHIC_IDENTITY_STAGE_NAME",
    "UK_GEOGRAPHIC_IDENTITY_OUTPUT_COLUMNS",
    "UKGeographicIdentityResult",
    "UKGeographicIdentityStageTransform",
    "frs_vintages_by_source_year",
    "geographic_identity_operation_parameters",
    "key_uk_households_by_clone_path",
]

GEOGRAPHIC_IDENTITY_STAGE_NAME = "geographic_identity"
#: The source every key names: the FRS, as the dense line's own identity node
#: named it before the dense line took its keys from the spine (microcosm#931).
GEOGRAPHIC_IDENTITY_SOURCE = "frs"
#: The household columns the step reads.
GEOGRAPHIC_IDENTITY_INPUTS = (
    SOURCE_HOUSEHOLD_ID_COLUMN,
    SOURCE_YEAR_COLUMN,
    ladder_clone_index_column("household"),
)
UK_GEOGRAPHIC_IDENTITY_OUTPUT_COLUMNS = (GEOGRAPHY_HOUSEHOLD_KEY_COLUMN,)
#: The operation the stage declares (microcosm#931).
GEOGRAPHIC_IDENTITY_OPERATION_KIND = "key_households_by_clone_path"
#: The stage's one artifact: the packaged FRS release, which pins the vintage
#: each key names, so its bytes enter the stage's contract.
FRS_RELEASE_ARTIFACT_ROLE = "frs_release"
FRS_RELEASE_RESOURCE = "frs_release.json"


def geographic_identity_operation_parameters() -> tuple[
    tuple[str, dict[str, object]], ...
]:
    """The reviewed operation the stage must declare verbatim: the source the
    keys name and the resource their vintage comes from."""

    return (
        (
            GEOGRAPHIC_IDENTITY_OPERATION_KIND,
            {
                "source": GEOGRAPHIC_IDENTITY_SOURCE,
                "vintage_resource": FRS_RELEASE_RESOURCE,
            },
        ),
    )


def frs_vintages_by_source_year(
    release: UKFRSRelease | None = None,
) -> Mapping[int, str]:
    """The FRS vintage frs_release.json pins for each source year it covers.

    The release pins one vintage, which the loader checks starts with its
    survey year. The geographic expansion writes each household's
    ``source_year`` from the frame's time period, the release's base year, as
    the SPI support channel did before that stage took the column over (and
    still does on a household table without the clone index). So both years
    map to the vintage (one year when they are equal), and every household's
    key names it; any other year maps to none. ``release`` defaults to the
    packaged pin.
    """

    release = load_uk_frs_release() if release is None else release
    vintage = str(release.vintage)
    return MappingProxyType(
        {int(release.survey_year): vintage, int(release.base_year): vintage}
    )


@dataclass(frozen=True)
class UKGeographicIdentityResult:
    """The keyed frame and what the step's evidence reads of it.

    ``source_households`` is the exact count of distinct source households;
    the evidence gives it rounded, and neither field prints in the repr.
    """

    frame: Frame = field(repr=False)
    source_households: int = field(repr=False)
    source_vintages: tuple[str, ...]

    def evidence(self) -> dict[str, object]:
        # A result exists only for unique keys: repeated keys are refused.
        return {
            "stage": GEOGRAPHIC_IDENTITY_STAGE_NAME,
            "source_households": rounded_count(self.source_households),
            "unique": True,
            "source": GEOGRAPHIC_IDENTITY_SOURCE,
            "source_vintages": list(self.source_vintages),
            "inputs": list(GEOGRAPHIC_IDENTITY_INPUTS),
        }


def _assert_geographic_identity_stage(stage: SourceStageSpec) -> None:
    """The stage a transform runs as: the step's name, household grain,
    exactly the key as its output with no rewrites, the reviewed operation
    (:func:`geographic_identity_operation_parameters`), and the FRS release as
    its one artifact, with ``runtime_sha256_required``."""

    if not isinstance(stage, SourceStageSpec):
        raise TypeError("The geography identity runs as a SourceStageSpec stage.")
    if stage.stage != GEOGRAPHIC_IDENTITY_STAGE_NAME:
        raise ValueError(
            f"Expected stage {GEOGRAPHIC_IDENTITY_STAGE_NAME!r}, got {stage.stage!r}."
        )
    if stage.grain != "household":
        raise ValueError(
            "The geography identity keys households; grain must be 'household', "
            f"got {stage.grain!r}."
        )
    if tuple(stage.outputs) != UK_GEOGRAPHIC_IDENTITY_OUTPUT_COLUMNS or tuple(
        stage.rewrites
    ):
        raise ValueError(
            "The geography identity writes exactly its key; the stage must "
            f"declare outputs {UK_GEOGRAPHIC_IDENTITY_OUTPUT_COLUMNS} and no "
            f"rewrites, got {tuple(stage.outputs)} / {tuple(stage.rewrites)}."
        )
    _assert_closed_world_operations(stage, geographic_identity_operation_parameters())
    releases = [
        artifact
        for artifact in stage.artifacts
        if artifact.get("role") == FRS_RELEASE_ARTIFACT_ROLE
        and artifact.get("resource") == FRS_RELEASE_RESOURCE
        and artifact.get("runtime_sha256_required") is True
    ]
    if len(stage.artifacts) != 1 or len(releases) != 1:
        raise ValueError(
            "The geography identity reads its vintages from one artifact, role "
            f"{FRS_RELEASE_ARTIFACT_ROLE!r} and resource {FRS_RELEASE_RESOURCE!r} "
            "with runtime_sha256_required: true; the stage must declare it and no "
            f"other, got {len(stage.artifacts)} artifact(s), {len(releases)} of "
            "them that one."
        )


def _require_int64(household: pd.DataFrame, column: str) -> np.ndarray:
    if column not in household.columns:
        raise ValueError(
            f"The geography identity reads household {column!r}, which the "
            "geographic expansion writes; the frame lacks it."
        )
    dtype = household[column].dtype
    if dtype != np.dtype("int64"):
        raise ValueError(
            f"The geography identity reads an int64 {column!r}; it holds {dtype}."
        )
    return household[column].to_numpy()


def _vintages(years: np.ndarray, release: UKFRSRelease | None) -> list[str]:
    """Each household's FRS vintage, refusing years the release pins none for."""

    by_year = frs_vintages_by_source_year(release)
    unmapped = int((~np.isin(years, np.fromiter(by_year, np.int64))).sum())
    if unmapped:
        raise ValueError(
            f"{unmapped} household(s) carry a {SOURCE_YEAR_COLUMN} that "
            "frs_release.json pins no FRS vintage for; it pins the source "
            f"year(s) {sorted(by_year)} (microcosm#931)."
        )
    return [by_year[year] for year in years.tolist()]


def key_uk_households_by_clone_path(
    frame: Frame, *, release: UKFRSRelease | None = None
) -> UKGeographicIdentityResult:
    """``frame`` with every household's ``geography_household_key``.

    Each key is ``household_draw_key`` over :data:`GEOGRAPHIC_IDENTITY_SOURCE`,
    the vintage ``release`` (frs_release.json by default) pins for the
    household's ``source_year``, its ``source_household_id`` and its copy path
    read from the lineage columns, which must hold the geographic expansion's
    clone index. Rows, ids, weights, the mass log and the metadata are the
    frame's own.

    Refused, by count or by name: a frame that already carries the key, or
    lacks the clone index or an int64 ``source_household_id`` or
    ``source_year``; malformed lineage, through the shared reader; a source
    year the release pins no vintage for; and keys that are not unique.
    """

    if not isinstance(frame, Frame):
        raise TypeError("The geography identity keys a Frame.")
    validate_uk_national_frame(frame)
    household = frame.table("household")
    if GEOGRAPHY_HOUSEHOLD_KEY_COLUMN in household.columns:
        raise ValueError(
            f"The frame already carries {GEOGRAPHY_HOUSEHOLD_KEY_COLUMN!r}; the "
            "geography identity keys each household once."
        )
    _require_int64(household, SOURCE_HOUSEHOLD_ID_COLUMN)
    years = _require_int64(household, SOURCE_YEAR_COLUMN)
    sources, paths = copy_paths(
        household, UK_COPY_LINEAGE, require=(GEOGRAPHIC_SUPPORT_BRANCH,)
    )
    vintages = _vintages(years, release)
    keys = [
        household_draw_key(
            source=GEOGRAPHIC_IDENTITY_SOURCE,
            source_vintage=vintage,
            source_household_id=source,
            clone_path=path,
        )
        for source, vintage, path in zip(sources.tolist(), vintages, paths, strict=True)
    ]
    repeated = len(keys) - len(set(keys))
    if repeated:
        raise ValueError(
            f"{repeated} household(s) repeat an earlier household's geography key; "
            "every household needs its own source and clone path (microcosm#931)."
        )
    keyed = household.copy()
    keyed[GEOGRAPHY_HOUSEHOLD_KEY_COLUMN] = pd.Series(
        keys, index=household.index, dtype=pd.StringDtype(storage="python")
    )
    tables = {entity: frame.table(entity) for entity in frame.entities}
    tables["household"] = keyed
    # The household weights pass through unchanged. Annotated so the stage's
    # static walk meets microcosm.frame's Weights module, as
    # test_the_core_modules_are_on_every_nodes_path requires.
    weights: Weights = frame.weights_for("household")
    result = Frame(
        tables,
        frame.schema,
        {"household": weights},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )
    validate_uk_national_frame(result)
    return UKGeographicIdentityResult(
        frame=result,
        source_households=distinct_carriers(sources.tolist()),
        source_vintages=tuple(sorted(set(vintages))),
    )


@dataclass(frozen=True)
class UKGeographicIdentityStageTransform:
    """Whole-stage transform for the geography identity (G1b).

    Constructing it checks the stage it runs as; running it calls
    :func:`key_uk_households_by_clone_path` with the packaged
    frs_release.json.
    """

    stage: SourceStageSpec
    last_result: UKGeographicIdentityResult | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        _assert_geographic_identity_stage(self.stage)

    def __call__(self, frame: Frame) -> Frame:
        _assert_geographic_identity_stage(self.stage)
        result = key_uk_households_by_clone_path(frame)
        object.__setattr__(self, "last_result", result)
        return result.frame

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return UK_GEOGRAPHIC_IDENTITY_OUTPUT_COLUMNS

    def checkpoint_metadata(self) -> dict[str, object]:
        if self.last_result is None:
            raise RuntimeError("checkpoint metadata requires a completed stage run.")
        return {"evidence": self.last_result.evidence()}
