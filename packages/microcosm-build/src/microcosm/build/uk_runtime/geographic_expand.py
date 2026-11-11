"""The UK spine's geographic clones: K copies of every FRS household (microcosm#931).

The clone-first spine gives every FRS household ``K`` geographic clones before
any copy step runs. Each clone carries an equal share of its household's
design weight, so a later stage can give each clone an area of its own while
together they stand for the survey household once. This module is that step,
G1 of the clone-first plan: :class:`UKGeographicExpandStageTransform` and the
function it runs, :func:`expand_uk_geographic_clones`.

The clones come from the shared, country-neutral core
(:func:`microcosm.build.clone_expansion.expand_clones`; shared rather than
UK-local, María's ruling of 2026-11-10) under the UK's declaration,
:data:`UK_GEOGRAPHIC_CLONES`:

- **Ids.** Clone ``k`` of a row takes ``id + k x radix`` on all five id
  columns, in the copy-id table's ``geographic_support`` field
  (:func:`mint_geographic_clone_ids`). So a clone's id is a function of its
  source id and its clone alone, and clone 0 keeps the frame's own ids.
  Incoming ids the field has no room for are refused first
  (:func:`require_room_for_geographic_clones`).
- **Clone indices.** Each table's ``{entity}_clone_index``, 0 to ``K - 1``,
  with exactly ``K`` values on every table.
- **Weights.** Design weights only. Each clone takes ``w / K`` of its
  household's weight, the same float in every clone, and the mass log gains
  one record whose reason names no clone count.
- **Source id.** ``source_household_id``, the clone-0 id.

The UK wrapper adds what the core does not do:

- It refuses a frame that already carries a clone index, a source column or
  any of the UK copy lineage's columns, since the step runs once, before any
  copy. It also refuses weights on any entity but the household: the core
  splits every weighted entity, and the UK weights households only. And it
  refuses id and membership columns that are not int64, the dtype the
  copy-id table mints, at every clone count, so every clone count takes the
  same frames.
- Before the core runs, it writes each household's ``source_year`` (the
  frame's time period, as the SPI support channel reads it) and
  ``source_household_key`` (``f"{source_year}:{source_household_id}"``, the
  values that stage writes, as Python strings in an object column) on the
  frame's own rows, so every clone carries its household's.
- It validates the clones as a UK national frame.

Every refusal gives counts, entities, columns, dtypes and declared constants
only, never an id, a weight or another value of a row. The evidence gives
``K``, the copy-id field and its radix, the distinct source households
rounded through ``receipt_counts``, and the weight split and kind; no row
count, id or weight.

No stage runs this step yet, so it moves no output and no node key.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from microcosm.build.clone_expansion import (
    CloneDeclaration,
    CloneIdRule,
    expand_clones,
)
from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.uk_runtime import copy_id_table
from microcosm.build.uk_runtime.geographic_clones import ladder_clone_index_column
from microcosm.build.uk_runtime.household_lineage import UK_COPY_LINEAGE
from microcosm.build.uk_runtime.national_frame import (
    UK_NATIONAL_SCHEMA,
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.build.uk_runtime.receipt_counts import (
    distinct_carriers,
    record_source_households,
    rounded_count,
)
from microcosm.frame import Frame, WeightKind
from microcosm.frame.concepts import SOURCE_HOUSEHOLD_ID_COLUMN

__all__ = [
    "GEOGRAPHIC_CLONE_LIMIT",
    "GEOGRAPHIC_COPY_ID_FIELD",
    "GEOGRAPHIC_EXPAND_MASS_CHANGE_REASON",
    "GEOGRAPHIC_EXPAND_STAGE_NAME",
    "GEOGRAPHIC_EXPAND_WEIGHT_SPLIT",
    "SOURCE_HOUSEHOLD_KEY_COLUMN",
    "SOURCE_YEAR_COLUMN",
    "UK_GEOGRAPHIC_CLONES",
    "UK_GEOGRAPHIC_EXPAND_OUTPUT_COLUMNS",
    "UK_GEOGRAPHIC_SOURCE_COLUMNS",
    "UKGeographicExpandResult",
    "UKGeographicExpandStageTransform",
    "expand_uk_geographic_clones",
    "mint_geographic_clone_ids",
    "require_room_for_geographic_clones",
]

GEOGRAPHIC_EXPAND_STAGE_NAME = "geographic_expand"
#: The copy-id table's field the clones' ids are minted in.
GEOGRAPHIC_COPY_ID_FIELD = "geographic_support"
#: The most clones the field numbers: clone ordinals run below it. Exported
#: so the spine and its graph can read the limit from here.
GEOGRAPHIC_CLONE_LIMIT = copy_id_table.ordinal_bound(GEOGRAPHIC_COPY_ID_FIELD)
#: The household's survey year, which the step writes on every household.
SOURCE_YEAR_COLUMN = "source_year"
#: ``f"{source_year}:{source_household_id}"``, which the step writes on every
#: household.
SOURCE_HOUSEHOLD_KEY_COLUMN = "source_household_key"
#: The source columns the step writes, which a frame entering it must lack.
UK_GEOGRAPHIC_SOURCE_COLUMNS = (
    SOURCE_HOUSEHOLD_ID_COLUMN,
    SOURCE_YEAR_COLUMN,
    SOURCE_HOUSEHOLD_KEY_COLUMN,
)
#: Every column the step adds: each table's clone index, then the household's
#: source columns.
UK_GEOGRAPHIC_EXPAND_OUTPUT_COLUMNS = (
    *(ladder_clone_index_column(entity) for entity in UK_NATIONAL_SCHEMA.entities),
    *UK_GEOGRAPHIC_SOURCE_COLUMNS,
)
#: The reason of the step's mass record. It names no clone count, so it is the
#: same at every K.
GEOGRAPHIC_EXPAND_MASS_CHANGE_REASON = (
    "Geographic clones split each household's design weight equally among its "
    "clones, so household mass is conserved."
)
#: How the clones share a household's weight: each takes an equal share.
GEOGRAPHIC_EXPAND_WEIGHT_SPLIT = "equal"


def mint_geographic_clone_ids(ids: np.ndarray, k: int) -> np.ndarray:
    """Clone ``k``'s ids: each id plus ``k`` times the ``geographic_support``
    field's radix, through the copy-id table, which refuses an ordinal outside
    the field, an id already in it and an int64 overflow, by count."""

    return copy_id_table.mint(ids, GEOGRAPHIC_COPY_ID_FIELD, k)


def require_room_for_geographic_clones(ids: np.ndarray, what: str) -> None:
    """Refuse incoming ids at or above the ``geographic_support`` field's
    radix, printing only how many, through the copy-id table."""

    copy_id_table.require_below(ids, GEOGRAPHIC_COPY_ID_FIELD, what=what)


#: The UK's declaration for the shared clone expansion core: ids minted in the
#: ``geographic_support`` field, a clone index on every table, design weights,
#: the step's mass record reason and the household's clone-0 id.
UK_GEOGRAPHIC_CLONES = CloneDeclaration(
    ids=CloneIdRule(
        bound=GEOGRAPHIC_CLONE_LIMIT,
        mint=mint_geographic_clone_ids,
        require_below=require_room_for_geographic_clones,
    ),
    clone_index_columns={
        entity: ladder_clone_index_column(entity)
        for entity in UK_NATIONAL_SCHEMA.entities
    },
    weight_kinds=frozenset({WeightKind.DESIGN}),
    reason=GEOGRAPHIC_EXPAND_MASS_CHANGE_REASON,
    source_id_columns={"household": SOURCE_HOUSEHOLD_ID_COLUMN},
)
#: The columns a frame entering the step must lack: what the step writes, and
#: every copy step's columns, since the step runs before any copy.
_RESERVED_COLUMNS = tuple(
    dict.fromkeys((*UK_GEOGRAPHIC_EXPAND_OUTPUT_COLUMNS, *UK_COPY_LINEAGE.copy_columns))
)


@dataclass(frozen=True)
class UKGeographicExpandResult:
    """The clones and what the step's evidence reads of them.

    ``source_households`` is the exact count of distinct source households;
    the evidence gives it rounded, and neither field prints in the repr.
    """

    frame: Frame = field(repr=False)
    clone_count: int
    source_households: int = field(repr=False)

    def evidence(self) -> dict[str, object]:
        return {
            "stage": GEOGRAPHIC_EXPAND_STAGE_NAME,
            "clone_count": int(self.clone_count),
            "copy_id_field": GEOGRAPHIC_COPY_ID_FIELD,
            "copy_id_radix": copy_id_table.radix(GEOGRAPHIC_COPY_ID_FIELD),
            "source_households": rounded_count(self.source_households),
            "weight_split": GEOGRAPHIC_EXPAND_WEIGHT_SPLIT,
            "weight_kind": WeightKind.DESIGN.value,
        }


def _clone_count(clones: object) -> int:
    """``clones`` as an int, refusing anything outside [1, the limit]."""

    if (
        isinstance(clones, bool)
        or not isinstance(clones, int | np.integer)
        or not 1 <= clones <= GEOGRAPHIC_CLONE_LIMIT
    ):
        raise ValueError(
            "The geographic expansion takes a clone count from 1 to "
            f"{GEOGRAPHIC_CLONE_LIMIT}, the {GEOGRAPHIC_COPY_ID_FIELD!r} copy-id "
            "field's declared bound (microcosm#931)."
        )
    return int(clones)


def _assert_geographic_expand_stage(stage: SourceStageSpec) -> None:
    """The stage a transform runs as: the step's name, household grain, and
    exactly the step's columns as outputs, with no rewrites."""

    if not isinstance(stage, SourceStageSpec):
        raise TypeError("The geographic expansion runs as a SourceStageSpec stage.")
    if stage.stage != GEOGRAPHIC_EXPAND_STAGE_NAME:
        raise ValueError(
            f"Expected stage {GEOGRAPHIC_EXPAND_STAGE_NAME!r}, got {stage.stage!r}."
        )
    if stage.grain != "household":
        raise ValueError(
            "The geographic expansion clones households; grain must be "
            f"'household', got {stage.grain!r}."
        )
    if tuple(stage.outputs) != UK_GEOGRAPHIC_EXPAND_OUTPUT_COLUMNS or tuple(
        stage.rewrites
    ):
        raise ValueError(
            "The geographic expansion writes exactly the clone indices and the "
            "source columns; the stage must declare outputs "
            f"{UK_GEOGRAPHIC_EXPAND_OUTPUT_COLUMNS} and no rewrites, got "
            f"{tuple(stage.outputs)} / {tuple(stage.rewrites)}."
        )


def _refuse_other_weighted_entities(frame: Frame) -> None:
    weighted = tuple(frame.weighted_entities)
    if weighted != ("household",):
        raise ValueError(
            "The geographic expansion splits household weights alone, as the UK "
            f"weights households only; the frame weights {list(weighted)}."
        )


def _refuse_narrow_ids(frame: Frame) -> None:
    """Refuse id and membership columns that are not int64.

    The copy-id table mints int64 ids, so the core refuses narrower ids from
    two clones on; refusing them at one clone too keeps every clone count
    taking the same frames.
    """

    schema = frame.schema
    person = frame.table(schema.person_entity)
    columns = [(frame.table(e), schema.entity_id_column(e)) for e in frame.entities]
    columns += [(person, schema.membership_column(g)) for g in schema.group_entities]
    narrow = [
        f"{column!r} holds {table[column].dtype}"
        for table, column in columns
        if table[column].dtype != np.dtype("int64")
    ]
    if narrow:
        raise ValueError(
            "The geographic expansion takes int64 id and membership columns, "
            f"the dtype the copy-id table mints; {', '.join(narrow)}."
        )


def _refuse_carried_lineage(frame: Frame) -> None:
    carried = set().union(*(frame.table(entity).columns for entity in frame.entities))
    found = [column for column in _RESERVED_COLUMNS if column in carried]
    if found:
        raise ValueError(
            "The geographic expansion clones a frame once, before any copy "
            f"step; the frame already carries {found}."
        )


def _source_year(frame: Frame) -> int:
    """The frame's time period as a year, as the SPI support channel reads it."""

    try:
        return int(uk_time_period(frame))
    except ValueError as error:
        raise ValueError(
            "The geographic expansion writes each household's source_year from "
            "the frame's time period, which must name a year."
        ) from error


def _with_source_year_and_key(frame: Frame, source_year: int) -> Frame:
    """``frame`` with each household's ``source_year`` and
    ``source_household_key``, before any clone exists.

    The key is written as Python strings in an object column whatever pandas
    infers for text: the dtype the SPI support channel's own write gives with
    string inference off, as in the legacy plan, so that plan's object-string
    columns stay what they were. The graph casts the column to its declared
    string dtype.
    """

    household = frame.table("household").copy()
    ids = household[UK_NATIONAL_SCHEMA.id_column("household")].tolist()
    household[SOURCE_YEAR_COLUMN] = np.full(len(household), source_year, np.int64)
    household[SOURCE_HOUSEHOLD_KEY_COLUMN] = pd.Series(
        [f"{source_year}:{source}" for source in ids],
        index=household.index,
        dtype=object,
    )
    tables = {entity: frame.table(entity) for entity in frame.entities}
    tables["household"] = household
    return Frame(
        tables,
        frame.schema,
        {"household": frame.weights_for("household")},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def expand_uk_geographic_clones(
    frame: Frame, *, clones: int
) -> UKGeographicExpandResult:
    """``frame`` as ``clones`` geographic clones of every household, clone-major.

    Clone 0 is the frame's own rows and ids. Clone ``k`` repeats every
    person, benefit unit and household with each id minted in the
    ``geographic_support`` field at ordinal ``k``. Every row carries its
    clone in its table's ``{entity}_clone_index``, every household its
    ``source_household_id`` (the clone-0 id), ``source_year`` and
    ``source_household_key``, and each clone ``w / clones`` of its
    household's design weight, with one mass record stating
    :data:`GEOGRAPHIC_EXPAND_MASS_CHANGE_REASON`. At one clone the rows, ids
    and weights are the frame's own; only those columns and the record are
    added.

    Refused, by count or by name: a clone count outside [1,
    :data:`GEOGRAPHIC_CLONE_LIMIT`]; weights on any entity but the household,
    or of any kind but design; a frame already carrying a clone index, a
    source column or a copy step's column; id or membership columns that are
    not int64; a time period that is not a year; incoming ids at or above the
    field's radix; and every refusal of the shared core.
    """

    count = _clone_count(clones)
    if not isinstance(frame, Frame):
        raise TypeError("The geographic expansion clones a Frame.")
    _refuse_other_weighted_entities(frame)
    validate_uk_national_frame(frame)
    _refuse_carried_lineage(frame)
    _refuse_narrow_ids(frame)
    base = _with_source_year_and_key(frame, _source_year(frame))
    expanded = expand_clones(base, clones=count, declaration=UK_GEOGRAPHIC_CLONES)
    validate_uk_national_frame(expanded)
    sources = record_source_households(
        expanded, "household", stage=GEOGRAPHIC_EXPAND_STAGE_NAME
    )
    return UKGeographicExpandResult(
        frame=expanded,
        clone_count=count,
        source_households=distinct_carriers(sources.tolist()),
    )


@dataclass(frozen=True)
class UKGeographicExpandStageTransform:
    """Whole-stage transform for the geographic expansion (G1).

    ``clones`` is the clone count ``K``. Constructing the transform checks
    the stage it runs as and the count; running it calls
    :func:`expand_uk_geographic_clones`.
    """

    stage: SourceStageSpec
    clones: int
    last_result: UKGeographicExpandResult | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        _assert_geographic_expand_stage(self.stage)
        _clone_count(self.clones)

    def __call__(self, frame: Frame) -> Frame:
        _assert_geographic_expand_stage(self.stage)
        result = expand_uk_geographic_clones(frame, clones=self.clones)
        object.__setattr__(self, "last_result", result)
        return result.frame

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return UK_GEOGRAPHIC_EXPAND_OUTPUT_COLUMNS

    def checkpoint_metadata(self) -> dict[str, object]:
        if self.last_result is None:
            raise RuntimeError("checkpoint metadata requires a completed stage run.")
        return {"evidence": self.last_result.evidence()}
