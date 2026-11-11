"""The clone-first spine's area stages (microcosm#931).

Once the geography identity has keyed every geographic clone (G1b), these
stages give each clone a place of its own, so every later stage reads a
clone's area as its household's:

- **G2, ``atomic_area_assign``** (:class:`UKAtomicAreaAssignStageTransform`)
  draws each household's atomic area within its FRS region with the shared
  operator, conditioned on its IMD decile and observed council-tax band
  (:mod:`~microcosm.build.uk_runtime.atomic_area_conditioning`): the chooser
  picks each household's fallback level, then ``assign_atomic``,
  ``derive_geography`` and ``validate_geography`` run on one narrow,
  transient frame of draw inputs, which carries no area column of its own.
  It writes the three assignment columns and the ten ladder columns in the
  graph's string dtype (``StringDtype("python")``), and refuses when the
  validation fails.
- **G3, ``local_authority``** (:class:`UKLocalAuthorityStageTransform`)
  resolves the engine's ``local_authority`` from the drawn
  ``local_authority_code`` through the committed names roster, fail-closed,
  never defaulted (microcosm#953), and writes it in the graph's string dtype,
  as the dense line's node did, not as the resolver's object column.
- **G4, ``area_brma``** (:class:`UKAreaBRMAStageTransform`), in
  ``frs_brma``'s slot after ``frs_household_draws``, writes each
  household's ``brma``.
  Great Britain households read it off their drawn LSOA (England and Wales)
  or output area (Scotland) through the committed area-to-BRMA layer,
  fail-closed on an area the layer does not hold or places in another FRS
  region. Northern Ireland has no open BRMA geography: one engine call over
  the whole frame materialises ``LHA_category``, and its benefit units draw
  from the census count table with ``frs_brma``'s keys, seed and salts
  (:func:`~microcosm.build.uk_runtime.frs_brma.draw_household_brmas`), so
  each takes the BRMA ``frs_brma`` gives it. ``brma`` stays an object
  column, as ``frs_brma`` writes it.
- **G5, ``rfdr``** (:class:`UKRFDRStageTransform`) sets each household's
  ``in_rural_fuel_duty_relief_area`` from its drawn output area through the
  committed RFDR layer. It refuses a layer that flags an output area outside
  its system's support, so the layer and the supports must share one code
  vintage; a system with households none of which is flagged is reported, as
  a boolean, not refused (C-6).
- **The SPI-copy area stage, ``spi_copy_areas``**
  (:class:`UKSPICopyAreasStageTransform`), after the SPI support channel and
  its income band donors (``spi_support_channel`` and
  ``spi_income_band_donors``), gives every row those stages copied a key and
  a place of its own, where the copy had its parent's: a geography key over
  its own copy path, an area drawn at L3 (its region, by households) with
  G2's definition, then G3's local authority, G4's BRMA and G5's flag. Its
  Northern Ireland copies draw their BRMA from the census counts on their
  own benefit units, with G4's seed and salts, after one engine call over
  their households alone materialises their ``LHA_category`` (C-9, María's
  ruling of 2026-11-11). That call is one simulation over the copies of
  every geographic clone together, as it is at one clone, never one per
  clone: the ruling rests on ``LHA_category`` reading only a household's own
  bedroom entitlement and shared-rate status. Every other row is left as it
  was.

G2, G3 and G5 compute with keyed integer draws, exact integer sampling
masses and string lookups, so each is bitwise, the class of every spine stage
(C-2). G4 and the copy stage read ``LHA_category`` from the rules engine and
draw Northern Ireland's BRMAs as ``frs_brma`` does. No stage declares them
yet: each transform checks the stage it runs as against the reviewed
declaration this module states (name, household grain, outputs, rewrites,
the reviewed operations verbatim and its artifacts), which the declarations
will carry.

This module's refusals give columns, dtypes and declared constants only,
never a key, an id, a code of a household, a value or a count of households.
The evidence counts distinct true source households through
``receipt_counts`` (a household's clones and copies counted once), never per
area or per cell, and prints no key or code of a household (C-44): G2's
sources by system, by level and by fallback reason, G3's sources, and G4's
sources by grain and Northern Ireland's sources drawn, over every clone,
since each is the same at every clone count; G5's flagged sources and the
copy stage's copied, donor and flagged sources among clone-0 rows, where a
union could grow with K.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from importlib.resources import files
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.atomic_geography import (
    AtomicSupport,
    assign_atomic,
    decode_atomic_support,
    derive_geography,
    validate_assignment_spec,
    validate_geography,
)
from microcosm.build.source_lineage import copy_paths
from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.uk_runtime.atomic_area_conditioning import (
    BAND_INPUT_COLUMN,
    COUNCIL_TAX_BAND_COLUMN,
    COUNCIL_TAX_BAND_OBSERVED_COLUMN,
    DECILE_INPUT_COLUMN,
    DEFAULT_WEIGHT_COLUMN,
    IDENTITY_COLUMN,
    REGION_COLUMN,
    SUPPORT_CHANNEL_BRANCH,
    UK_ATOMIC_FALLBACK_LEVELS,
    UK_DRAW_CONSTRAINTS,
    UK_FALLBACK_STATUSES,
    UK_SPINE_AREA_COLUMNS,
    UK_SPINE_ATOMIC_AREA_SEED,
    UK_SPINE_ATOMIC_AREA_STREAM,
    UK_SPINE_LADDER_COLUMNS,
    UNOBSERVED_IMD_DECILE,
    uk_band_weight_map,
    uk_conditioned_assignment_definition,
    uk_conditioned_supports,
    uk_draw_frame,
    uk_draw_inputs,
    uk_draw_levels,
    uk_fallback_reasons,
)
from microcosm.build.uk_runtime.atomic_area_support import (
    ASSIGNMENT_COLUMNS,
    CONDITIONED_SOURCES,
    IMD_DECILE_COLUMN,
    SYSTEMS,
)
from microcosm.build.uk_runtime.atomic_household_identity import household_draw_key
from microcosm.build.uk_runtime.brma_area_layer import (
    GRAINS as BRMA_GRAINS,
)
from microcosm.build.uk_runtime.brma_area_layer import (
    METHOD_VERSION as BRMA_METHOD_VERSION,
)
from microcosm.build.uk_runtime.brma_area_layer import (
    NI_SYSTEM,
    BRMAAreaLayer,
    area_brmas,
    load_brma_area_layer,
)
from microcosm.build.uk_runtime.cgt_structure import _assert_closed_world_operations
from microcosm.build.uk_runtime.frs_brma import (
    FRS_BRMA_OUTPUT_COLUMNS,
    UK_BRMA_BENUNIT_SALT,
    UK_BRMA_COUNT_RESOURCE,
    UK_BRMA_DECLARED_SEEDS,
    UK_BRMA_HOUSEHOLD_PICK_SALT,
    UK_BRMA_PREDICTORS,
    draw_household_brmas,
)
from microcosm.build.uk_runtime.geographic_clones import (
    SOURCE_YEAR_COLUMN,
    geographic_clone_count,
    ladder_clone_index_column,
)
from microcosm.build.uk_runtime.geographic_identity import (
    FRS_RELEASE_ARTIFACT_ROLE,
    FRS_RELEASE_RESOURCE,
    GEOGRAPHIC_IDENTITY_SOURCE,
    frs_vintages_by_source_year,
)
from microcosm.build.uk_runtime.geography_sources import LAD23_NAMES_SHA256
from microcosm.build.uk_runtime.household_lineage import (
    GEOGRAPHIC_SUPPORT_BRANCH,
    UK_COPY_LINEAGE,
)
from microcosm.build.uk_runtime.local_authority_input import (
    UK_LOCAL_AUTHORITY_CODE_COLUMN,
    UK_LOCAL_AUTHORITY_INPUT_COLUMN,
    UK_LOCAL_AUTHORITY_NAMES_RESOURCE,
    UK_LOCAL_AUTHORITY_VINTAGE,
    local_authority_engine_key_by_code,
    resolve_local_authority_engine_keys,
)
from microcosm.build.uk_runtime.national_frame import (
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.build.uk_runtime.receipt_counts import (
    distinct_carriers,
    record_source_households,
    rounded_count,
)
from microcosm.build.uk_runtime.rfdr_area_layer import (
    METHOD_VERSION as RFDR_METHOD_VERSION,
)
from microcosm.build.uk_runtime.rfdr_area_layer import (
    RFDR_COLUMN,
    RFDRAreaLayer,
    area_flags,
    load_rfdr_area_layer,
)
from microcosm.build.uk_runtime.rfdr_area_layer import (
    SYSTEMS as RFDR_SYSTEMS,
)
from microcosm.frame import Frame, MassChangeRecord, Weights
from microcosm.frame.rules import assert_rules_engine_country
from microcosm.graph.canonical import canonical_json

__all__ = [
    "AREA_BRMA_OPERATION_KIND",
    "AREA_BRMA_STAGE_NAME",
    "ATOMIC_AREA_ASSIGN_OPERATION_KIND",
    "ATOMIC_AREA_ASSIGN_STAGE_NAME",
    "BRMA_COUNT_ARTIFACT_ROLE",
    "BRMA_LAYER_ARTIFACT_ROLE",
    "BRMA_LAYER_RESOURCE",
    "LOCAL_AUTHORITY_ARTIFACT_ROLE",
    "LOCAL_AUTHORITY_OPERATION_KIND",
    "LOCAL_AUTHORITY_STAGE_NAME",
    "RFDR_ARTIFACT_ROLE",
    "RFDR_LAYER_RESOURCE",
    "RFDR_OPERATION_KIND",
    "RFDR_STAGE_NAME",
    "SPI_COPY_AREAS_OPERATION_KIND",
    "SPI_COPY_AREAS_STAGE_NAME",
    "UK_AREA_BRMA_OUTPUT_COLUMNS",
    "UK_ATOMIC_AREA_ASSIGN_OUTPUT_COLUMNS",
    "UK_LOCAL_AUTHORITY_OUTPUT_COLUMNS",
    "UK_RFDR_OUTPUT_COLUMNS",
    "UK_SPI_COPY_AREAS_REWRITE_COLUMNS",
    "UKAreaBRMAResult",
    "UKAreaBRMAStageTransform",
    "UKAtomicAreaAssignResult",
    "UKAtomicAreaAssignStageTransform",
    "UKLocalAuthorityResult",
    "UKLocalAuthorityStageTransform",
    "UKRFDRResult",
    "UKRFDRStageTransform",
    "UKSPICopyAreasResult",
    "UKSPICopyAreasStageTransform",
    "UKSpineAreaDraw",
    "area_brma_operation_parameters",
    "assign_uk_spi_copy_areas",
    "assign_uk_spine_areas",
    "assign_uk_spine_brmas",
    "atomic_area_assign_operation_parameters",
    "draw_uk_spine_areas",
    "flag_uk_spine_rfdr",
    "load_committed_brma_counts",
    "load_committed_brma_layer",
    "load_committed_rfdr_layer",
    "local_authority_operation_parameters",
    "resolve_uk_spine_local_authorities",
    "rfdr_operation_parameters",
    "spi_copy_areas_operation_parameters",
    "uk_area_brmas",
    "uk_local_authorities",
    "uk_rfdr_flags",
    "uk_spi_copy_keys",
]

_STRING = pd.StringDtype(storage="python")
_CLONE_INDEX_COLUMN = ladder_clone_index_column("household")


# --- Shared helpers ---------------------------------------------------------------


def _assert_stage(
    stage: SourceStageSpec,
    *,
    name: str,
    what: str,
    outputs: tuple[str, ...],
    operations: tuple[tuple[str, dict[str, object]], ...],
    artifacts: tuple[tuple[str, str], ...] = (),
    rewrites: tuple[str, ...] = (),
) -> None:
    """The stage a transform runs as: ``name``, household grain, exactly
    ``outputs`` and ``rewrites`` (none unless given), the reviewed
    ``operations`` verbatim, and exactly ``artifacts`` (role, resource), each
    ``runtime_sha256_required``."""

    if not isinstance(stage, SourceStageSpec):
        raise TypeError(f"{what} runs as a SourceStageSpec stage.")
    if stage.stage != name:
        raise ValueError(f"Expected stage {name!r}, got {stage.stage!r}.")
    if stage.grain != "household":
        raise ValueError(
            f"{what} writes households; grain must be 'household', got {stage.grain!r}."
        )
    if tuple(stage.outputs) != outputs or tuple(stage.rewrites) != rewrites:
        raise ValueError(
            f"{what} writes exactly its own columns; the stage must declare "
            f"outputs {outputs} and rewrites {rewrites}, got "
            f"{tuple(stage.outputs)} / {tuple(stage.rewrites)}."
        )
    _assert_closed_world_operations(stage, operations)
    declared = sorted(
        (str(artifact.get("role")), str(artifact.get("resource")))
        for artifact in stage.artifacts
        if artifact.get("runtime_sha256_required") is True
    )
    if len(stage.artifacts) != len(artifacts) or declared != sorted(artifacts):
        raise ValueError(
            f"{what} reads the artifacts {list(artifacts)} (role, resource), each "
            "with runtime_sha256_required: true, and no other; the stage "
            f"declares {len(stage.artifacts)} artifact(s)."
        )


def _require_columns(
    household: pd.DataFrame, columns: Sequence[str], what: str
) -> None:
    missing = [column for column in columns if column not in household.columns]
    if missing:
        raise ValueError(
            f"{what} reads household {missing}, which earlier stages write; "
            "the frame lacks them (microcosm#931)."
        )


def _refuse_written(household: pd.DataFrame, columns: Sequence[str], what: str) -> None:
    present = [column for column in columns if column in household.columns]
    if present:
        raise ValueError(
            f"The frame already carries {present}; {what} writes each household's "
            "once (microcosm#931)."
        )


def _with_household(frame: Frame, household: pd.DataFrame) -> Frame:
    """``frame`` with ``household`` in place of its household table; ids,
    weights, strata, the mass log and the metadata are the frame's own."""

    tables = {entity: frame.table(entity) for entity in frame.entities}
    tables["household"] = household
    # The household weights pass through unchanged. Annotated so the stage's
    # static walk meets microcosm.frame's Weights module.
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
    return result


def _clone_zero(household: pd.DataFrame) -> np.ndarray:
    """Rows of geographic clone 0: every row of a table without the index."""

    if _CLONE_INDEX_COLUMN not in household.columns:
        return np.ones(len(household), dtype=bool)
    return household[_CLONE_INDEX_COLUMN].to_numpy(dtype=np.int64) == 0


def _sources_by(
    sources: np.ndarray, labels: np.ndarray, names: Sequence[str]
) -> dict[str, int]:
    """Distinct true sources among the rows carrying each label, exact."""

    return {name: distinct_carriers(sources[labels == name].tolist()) for name in names}


# --- G2, atomic_area_assign -------------------------------------------------------

ATOMIC_AREA_ASSIGN_STAGE_NAME = "atomic_area_assign"
ATOMIC_AREA_ASSIGN_OPERATION_KIND = "assign_atomic_area"
#: The columns G2 writes: the three assignment columns, then the ten ladder
#: columns, every one in the graph's string dtype.
UK_ATOMIC_AREA_ASSIGN_OUTPUT_COLUMNS = UK_SPINE_AREA_COLUMNS
_VALIDATION_SCOPE = "atomic_geography_mapping_integrity"


def atomic_area_assign_operation_parameters() -> tuple[
    tuple[str, dict[str, object]], ...
]:
    """The reviewed operation G2's stage must declare verbatim: the draw's
    key, sources, constraints, per-household weight, inputs, fallback ladder,
    layers, stream and seed, as
    :func:`~microcosm.build.uk_runtime.atomic_area_conditioning.uk_conditioned_assignment_definition`
    and :func:`~microcosm.build.uk_runtime.atomic_area_conditioning.uk_draw_inputs`
    use them."""

    return (
        (
            ATOMIC_AREA_ASSIGN_OPERATION_KIND,
            {
                "identity": IDENTITY_COLUMN,
                "sources": {system: CONDITIONED_SOURCES[system] for system in SYSTEMS},
                "constraints": [
                    {
                        "input": column,
                        "support": matched,
                        "required": column == REGION_COLUMN,
                    }
                    for column, matched in UK_DRAW_CONSTRAINTS.items()
                ],
                "weight_by": {
                    "input": BAND_INPUT_COLUMN,
                    "default": DEFAULT_WEIGHT_COLUMN,
                    "bands": {
                        system: sorted(uk_band_weight_map(system)) for system in SYSTEMS
                    },
                },
                "draw_inputs": {
                    DECILE_INPUT_COLUMN: {
                        "column": IMD_DECILE_COLUMN,
                        "unobserved": UNOBSERVED_IMD_DECILE,
                    },
                    BAND_INPUT_COLUMN: {
                        "column": COUNCIL_TAX_BAND_COLUMN,
                        "observed": COUNCIL_TAX_BAND_OBSERVED_COLUMN,
                    },
                    "missing_on_copies_of": SUPPORT_CHANNEL_BRANCH,
                },
                "fallback_levels": [
                    {
                        "name": level.name,
                        "constraints": list(level.constraints),
                        "weight_by": level.weight_by,
                    }
                    for level in UK_ATOMIC_FALLBACK_LEVELS
                ],
                "layers": list(UK_SPINE_LADDER_COLUMNS),
                "outputs": list(ASSIGNMENT_COLUMNS),
                "stream": UK_SPINE_ATOMIC_AREA_STREAM,
                "seed": UK_SPINE_ATOMIC_AREA_SEED,
            },
        ),
    )


def _assert_atomic_area_assign_stage(stage: SourceStageSpec) -> None:
    _assert_stage(
        stage,
        name=ATOMIC_AREA_ASSIGN_STAGE_NAME,
        what="The area draw",
        outputs=UK_ATOMIC_AREA_ASSIGN_OUTPUT_COLUMNS,
        operations=atomic_area_assign_operation_parameters(),
    )


@dataclass(frozen=True)
class UKSpineAreaDraw:
    """The draw on one household table: G2's columns, and each row's level
    and fallback reason. Nothing here prints in the repr but the digests."""

    columns: pd.DataFrame = field(repr=False)
    levels: np.ndarray = field(repr=False)
    reasons: pd.DataFrame = field(repr=False)
    definition_sha256: str
    support_sha256: Mapping[str, str]


def draw_uk_spine_areas(
    household: pd.DataFrame,
    *,
    definition: Mapping[str, object],
    supports: Mapping[str, AtomicSupport],
) -> UKSpineAreaDraw:
    """Every household row's area, ladder, level and fallback reason.

    Builds the narrow draw frame (:func:`uk_draw_inputs`), picks each row's
    level (:func:`uk_draw_levels`), sets the inputs its level drops to missing
    (:func:`uk_draw_frame`), then runs ``assign_atomic``, ``derive_geography``
    and ``validate_geography`` on that one frame, which carries no area column
    until the draw adds its own. Returns the three assignment columns and the
    ten ladder columns in the graph's string dtype, on ``household``'s index
    and in its row order. Each row's draw is keyed on its geography key alone,
    so a row's area does not depend on the table's other rows or their order.

    Refused: what the adapter and the chooser refuse; a null or repeated key
    and a failed validation, by the shared operator's count-free texts.
    """

    inputs = uk_draw_inputs(household)
    levels = uk_draw_levels(inputs, definition, supports)
    draw = uk_draw_frame(inputs, levels)
    assigned = assign_atomic(draw, definition, supports)
    placed = pd.concat([draw, assigned], axis=1)
    derived = derive_geography(placed, definition, supports)
    if set(derived.columns) != set(UK_SPINE_LADDER_COLUMNS):
        raise ValueError(
            "The area draw derives exactly the ten ladder columns "
            f"{list(UK_SPINE_LADDER_COLUMNS)}; the definition derives others."
        )
    receipt = validate_geography(
        pd.concat([placed, derived], axis=1), definition, supports
    )
    if receipt.get("outcome") != "pass" or receipt.get("scope") != _VALIDATION_SCOPE:
        raise ValueError("The area draw's mapping-integrity validation did not pass.")
    values = pd.concat([assigned, derived], axis=1)
    columns = pd.DataFrame(
        {
            column: pd.Series(
                values[column].to_numpy(dtype=object),
                index=household.index,
                dtype=_STRING,
            )
            for column in UK_SPINE_AREA_COLUMNS
        },
        index=household.index,
    )
    return UKSpineAreaDraw(
        columns=columns,
        levels=levels,
        reasons=uk_fallback_reasons(inputs, levels, definition),
        definition_sha256=str(receipt["definition_sha256"]),
        support_sha256=MappingProxyType(
            dict(sorted(receipt["support_sha256"].items()))
        ),
    )


@dataclass(frozen=True)
class UKAtomicAreaAssignResult:
    """G2's frame and what its evidence reads of it.

    The source counts are exact distinct true source households; the
    evidence gives them rounded, and none prints in the repr.
    """

    frame: Frame = field(repr=False)
    clone_count: int
    seed: int
    stream: tuple[object, ...]
    definition_sha256: str
    support_sha256: Mapping[str, str]
    source_households: int = field(repr=False)
    by_system: Mapping[str, int] = field(repr=False)
    by_level: Mapping[str, int] = field(repr=False)
    by_reason: Mapping[str, Mapping[str, int]] = field(repr=False)

    def evidence(self) -> dict[str, object]:
        # A result exists only for a draw whose validation passed.
        return {
            "stage": ATOMIC_AREA_ASSIGN_STAGE_NAME,
            "clone_count": int(self.clone_count),
            "seed": int(self.seed),
            "stream": list(self.stream),
            "definition_sha256": self.definition_sha256,
            "support_sha256": dict(self.support_sha256),
            "validation": {"outcome": "pass", "scope": _VALIDATION_SCOPE},
            "fallback_levels": [level.name for level in UK_ATOMIC_FALLBACK_LEVELS],
            "source_households": rounded_count(self.source_households),
            "source_households_by_system": {
                system: rounded_count(count) for system, count in self.by_system.items()
            },
            "source_households_by_level": {
                level: rounded_count(count) for level, count in self.by_level.items()
            },
            "source_households_by_fallback_reason": {
                column: {
                    status: rounded_count(count) for status, count in statuses.items()
                }
                for column, statuses in self.by_reason.items()
            },
            "outputs": list(UK_ATOMIC_AREA_ASSIGN_OUTPUT_COLUMNS),
        }


def assign_uk_spine_areas(
    frame: Frame,
    *,
    definition: Mapping[str, object],
    supports: Mapping[str, AtomicSupport],
) -> UKAtomicAreaAssignResult:
    """``frame`` with every household's area and ladder (G2).

    ``definition`` is
    :func:`~microcosm.build.uk_runtime.atomic_area_conditioning.uk_conditioned_assignment_definition`'s
    and ``supports`` the decoded supports it declares. Rows, ids, weights,
    the mass log and the metadata are the frame's own. Refused, by name: a
    frame already carrying any column G2 writes, and every refusal of
    :func:`draw_uk_spine_areas`.
    """

    if not isinstance(frame, Frame):
        raise TypeError("The area draw places the households of a Frame.")
    validate_uk_national_frame(frame)
    household = frame.table("household")
    _refuse_written(household, UK_ATOMIC_AREA_ASSIGN_OUTPUT_COLUMNS, "the area draw")
    draw = draw_uk_spine_areas(household, definition=definition, supports=supports)
    placed = household.copy()
    for column in UK_ATOMIC_AREA_ASSIGN_OUTPUT_COLUMNS:
        placed[column] = draw.columns[column]
    result = _with_household(frame, placed)
    sources = record_source_households(
        result, "household", stage=ATOMIC_AREA_ASSIGN_STAGE_NAME
    )
    systems = draw.columns[ASSIGNMENT_COLUMNS[1]].to_numpy(dtype=object)
    stream = tuple(definition["stream"])
    return UKAtomicAreaAssignResult(
        frame=result,
        clone_count=geographic_clone_count(household),
        seed=int(stream[3]),
        stream=stream,
        definition_sha256=draw.definition_sha256,
        support_sha256=draw.support_sha256,
        source_households=distinct_carriers(sources.tolist()),
        by_system=_sources_by(sources, systems, SYSTEMS),
        by_level=_sources_by(
            sources, draw.levels, [level.name for level in UK_ATOMIC_FALLBACK_LEVELS]
        ),
        by_reason={
            column: _sources_by(
                sources, draw.reasons[column].to_numpy(dtype=object), statuses
            )
            for column, statuses in UK_FALLBACK_STATUSES.items()
        },
    )


@dataclass(frozen=True)
class UKAtomicAreaAssignStageTransform:
    """Whole-stage transform for the area draw (G2).

    ``supports`` holds the three conditioned supports' bytes, keyed by
    system. Constructing the transform checks the stage it runs as and builds
    the definition, which holds each support to canonical bytes, with the
    declared seed; running it calls :func:`assign_uk_spine_areas`.
    """

    stage: SourceStageSpec
    supports: Mapping[str, bytes] = field(repr=False)
    definition: Mapping[str, object] = field(init=False, repr=False, compare=False)
    decoded_supports: Mapping[str, AtomicSupport] = field(
        init=False, repr=False, compare=False
    )
    last_result: UKAtomicAreaAssignResult | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        _assert_atomic_area_assign_stage(self.stage)
        # The definition holds every support to canonical bytes, so the
        # shared decoder alone reads them after it.
        definition = uk_conditioned_assignment_definition(
            self.supports, seed=UK_SPINE_ATOMIC_AREA_SEED
        )
        decoded = {
            system: decode_atomic_support(self.supports[system]) for system in SYSTEMS
        }
        object.__setattr__(self, "definition", definition)
        object.__setattr__(self, "decoded_supports", MappingProxyType(decoded))

    def __call__(self, frame: Frame) -> Frame:
        _assert_atomic_area_assign_stage(self.stage)
        result = assign_uk_spine_areas(
            frame, definition=self.definition, supports=self.decoded_supports
        )
        object.__setattr__(self, "last_result", result)
        return result.frame

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return UK_ATOMIC_AREA_ASSIGN_OUTPUT_COLUMNS

    def checkpoint_metadata(self) -> dict[str, object]:
        if self.last_result is None:
            raise RuntimeError("checkpoint metadata requires a completed stage run.")
        return {"evidence": self.last_result.evidence()}


# --- G3, local_authority ----------------------------------------------------------

LOCAL_AUTHORITY_STAGE_NAME = "local_authority"
LOCAL_AUTHORITY_OPERATION_KIND = "resolve_local_authority"
#: The stage's one artifact: the committed ONS names roster, whose bytes enter
#: the stage's contract.
LOCAL_AUTHORITY_ARTIFACT_ROLE = "local_authority_names"
UK_LOCAL_AUTHORITY_OUTPUT_COLUMNS = (UK_LOCAL_AUTHORITY_INPUT_COLUMN,)


def local_authority_operation_parameters() -> tuple[tuple[str, dict[str, object]], ...]:
    """The reviewed operation G3's stage must declare verbatim: the code it
    reads, the engine input it writes, the roster and the roster's vintage."""

    return (
        (
            LOCAL_AUTHORITY_OPERATION_KIND,
            {
                "code_column": UK_LOCAL_AUTHORITY_CODE_COLUMN,
                "output": UK_LOCAL_AUTHORITY_INPUT_COLUMN,
                "names_resource": UK_LOCAL_AUTHORITY_NAMES_RESOURCE,
                "vintage": UK_LOCAL_AUTHORITY_VINTAGE,
            },
        ),
    )


def _assert_local_authority_stage(stage: SourceStageSpec) -> None:
    _assert_stage(
        stage,
        name=LOCAL_AUTHORITY_STAGE_NAME,
        what="The local-authority stage",
        outputs=UK_LOCAL_AUTHORITY_OUTPUT_COLUMNS,
        operations=local_authority_operation_parameters(),
        artifacts=((LOCAL_AUTHORITY_ARTIFACT_ROLE, UK_LOCAL_AUTHORITY_NAMES_RESOURCE),),
    )


def uk_local_authorities(codes: pd.Series) -> pd.Series:
    """The engine's ``local_authority`` member for each local-authority code.

    Every code must be on the April 2023 roster the committed names resource
    carries; a missing or off-roster code is refused before the shared
    resolver runs, by a text that names no code. Returns the member names in
    the graph's string dtype on ``codes``' index, as the dense line wrote them.
    """

    roster = local_authority_engine_key_by_code()
    on_roster = codes.isin(list(roster)).to_numpy(dtype=bool) & codes.notna().to_numpy(
        dtype=bool
    )
    if not on_roster.all():
        raise ValueError(
            f"{UK_LOCAL_AUTHORITY_CODE_COLUMN} holds codes off the April 2023 local "
            f"authority roster {UK_LOCAL_AUTHORITY_NAMES_RESOURCE} carries; the engine "
            f"input {UK_LOCAL_AUTHORITY_INPUT_COLUMN} is never defaulted "
            "(microcosm#953)."
        )
    keys = resolve_local_authority_engine_keys(codes.to_numpy(dtype=object))
    return pd.Series(keys, index=codes.index, dtype=_STRING)


@dataclass(frozen=True)
class UKLocalAuthorityResult:
    """G3's frame and what its evidence reads of it; the source count is
    exact and prints only rounded."""

    frame: Frame = field(repr=False)
    clone_count: int
    source_households: int = field(repr=False)

    def evidence(self) -> dict[str, object]:
        return {
            "stage": LOCAL_AUTHORITY_STAGE_NAME,
            "clone_count": int(self.clone_count),
            "source_households": rounded_count(self.source_households),
            "local_authority_vintage": UK_LOCAL_AUTHORITY_VINTAGE,
            "names_source_sha256": LAD23_NAMES_SHA256,
            "output": UK_LOCAL_AUTHORITY_INPUT_COLUMN,
        }


def resolve_uk_spine_local_authorities(frame: Frame) -> UKLocalAuthorityResult:
    """``frame`` with every household's ``local_authority`` (G3).

    Refused, by name: a frame already carrying the column, a frame without
    the drawn code, and a code off the roster (:func:`uk_local_authorities`).
    """

    if not isinstance(frame, Frame):
        raise TypeError("The local-authority stage reads the households of a Frame.")
    validate_uk_national_frame(frame)
    household = frame.table("household")
    _refuse_written(
        household, UK_LOCAL_AUTHORITY_OUTPUT_COLUMNS, "the local-authority stage"
    )
    _require_columns(
        household, (UK_LOCAL_AUTHORITY_CODE_COLUMN,), "The local-authority stage"
    )
    resolved = household.copy()
    resolved[UK_LOCAL_AUTHORITY_INPUT_COLUMN] = uk_local_authorities(
        household[UK_LOCAL_AUTHORITY_CODE_COLUMN]
    )
    result = _with_household(frame, resolved)
    sources = record_source_households(
        result, "household", stage=LOCAL_AUTHORITY_STAGE_NAME
    )
    return UKLocalAuthorityResult(
        frame=result,
        clone_count=geographic_clone_count(household),
        source_households=distinct_carriers(sources.tolist()),
    )


@dataclass(frozen=True)
class UKLocalAuthorityStageTransform:
    """Whole-stage transform for the local authority (G3).

    Constructing it checks the stage it runs as; running it calls
    :func:`resolve_uk_spine_local_authorities`, which reads the committed
    names roster.
    """

    stage: SourceStageSpec
    last_result: UKLocalAuthorityResult | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        _assert_local_authority_stage(self.stage)

    def __call__(self, frame: Frame) -> Frame:
        _assert_local_authority_stage(self.stage)
        result = resolve_uk_spine_local_authorities(frame)
        object.__setattr__(self, "last_result", result)
        return result.frame

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return UK_LOCAL_AUTHORITY_OUTPUT_COLUMNS

    def checkpoint_metadata(self) -> dict[str, object]:
        if self.last_result is None:
            raise RuntimeError("checkpoint metadata requires a completed stage run.")
        return {"evidence": self.last_result.evidence()}


# --- G4, area_brma ----------------------------------------------------------------

AREA_BRMA_STAGE_NAME = "area_brma"
AREA_BRMA_OPERATION_KIND = "assign_area_brma"
#: The stage's two artifacts, whose bytes enter its contract: the committed
#: area-to-BRMA layer Great Britain reads, and the census counts Northern
#: Ireland draws from, under the role ``frs_brma`` declares them by.
BRMA_LAYER_ARTIFACT_ROLE = "brma_area_layer"
BRMA_LAYER_RESOURCE = "brma_area_layer.json"
BRMA_COUNT_ARTIFACT_ROLE = "count_resource"
UK_AREA_BRMA_OUTPUT_COLUMNS = FRS_BRMA_OUTPUT_COLUMNS
_BRMA_COLUMN = FRS_BRMA_OUTPUT_COLUMNS[0]
_PREDICTORS_OPERATION_KIND = "materialize_rules_engine_predictors"
#: The FRS region of Northern Ireland, the one region of its system
#: (``atomic_area_support``'s profile; a test holds the spelling equal).
_NORTHERN_IRELAND_REGION = "NORTHERN_IRELAND"
#: The household columns the BRMA lookup reads: the drawn system, the two
#: area codes the layer is keyed by and the FRS region.
_BRMA_INPUT_COLUMNS = (
    ASSIGNMENT_COLUMNS[1],
    *dict.fromkeys(spec.key for spec in BRMA_GRAINS.values()),
    REGION_COLUMN,
)


def _brma_draw_parameters() -> dict[str, object]:
    """``frs_brma``'s count-table draw, which Northern Ireland keeps."""

    return {
        "count_resource": UK_BRMA_COUNT_RESOURCE,
        "count_table_systems": [NI_SYSTEM],
        "cell_columns": [REGION_COLUMN, UK_BRMA_PREDICTORS[0]],
        "assignment_grain": "benunit",
        "collapse_to": "household",
        "collapse_method": "uniform_member_pick",
        "seed": UK_BRMA_DECLARED_SEEDS[_BRMA_COLUMN],
        "salts": {
            "benunit": UK_BRMA_BENUNIT_SALT,
            "household_pick": UK_BRMA_HOUSEHOLD_PICK_SALT,
        },
    }


def area_brma_operation_parameters() -> tuple[tuple[str, dict[str, object]], ...]:
    """The reviewed operations G4's stage must declare verbatim: the engine
    predictor it consumes, then the BRMA it writes, from the layer in Great
    Britain (keyed by each system's area code) and from ``frs_brma``'s
    count-table draw in Northern Ireland, with that draw's seed and salts."""

    return (
        (
            _PREDICTORS_OPERATION_KIND,
            {"predictors": list(UK_BRMA_PREDICTORS), "consumed_only": True},
        ),
        (
            AREA_BRMA_OPERATION_KIND,
            {
                "output": _BRMA_COLUMN,
                "area_system": ASSIGNMENT_COLUMNS[1],
                "layer_resource": BRMA_LAYER_RESOURCE,
                "method": BRMA_METHOD_VERSION,
                "area_codes": {spec.system: spec.key for spec in BRMA_GRAINS.values()},
                **_brma_draw_parameters(),
            },
        ),
    )


def _brma_artifacts() -> tuple[tuple[str, str], ...]:
    return (
        (BRMA_LAYER_ARTIFACT_ROLE, BRMA_LAYER_RESOURCE),
        (BRMA_COUNT_ARTIFACT_ROLE, UK_BRMA_COUNT_RESOURCE),
    )


def _assert_area_brma_stage(stage: SourceStageSpec) -> None:
    _assert_stage(
        stage,
        name=AREA_BRMA_STAGE_NAME,
        what="The BRMA stage",
        outputs=UK_AREA_BRMA_OUTPUT_COLUMNS,
        operations=area_brma_operation_parameters(),
        artifacts=_brma_artifacts(),
    )


def load_committed_brma_layer() -> tuple[BRMAAreaLayer, str]:
    """The committed area-to-BRMA layer and its bytes' SHA-256."""

    payload = files("microcosm.build.uk").joinpath(BRMA_LAYER_RESOURCE).read_bytes()
    return load_brma_area_layer(payload), hashlib.sha256(payload).hexdigest()


def load_committed_brma_counts() -> tuple[Mapping[str, Any], str]:
    """The committed census count table ``frs_brma`` draws from, and its
    bytes' SHA-256."""

    payload = files("microcosm.build.uk").joinpath(UK_BRMA_COUNT_RESOURCE).read_bytes()
    return json.loads(payload), hashlib.sha256(payload).hexdigest()


def _brma_systems(household: pd.DataFrame, what: str) -> np.ndarray:
    """Each row's drawn area system, every one of the three, Northern
    Ireland's rows in its region and no other row there."""

    _require_columns(household, _BRMA_INPUT_COLUMNS, what)
    systems = household[ASSIGNMENT_COLUMNS[1]].to_numpy(dtype=object)
    if not np.isin(systems, list(SYSTEMS)).all():
        raise ValueError(
            f"{what} reads a BRMA for households of the systems {list(SYSTEMS)}; "
            f"{ASSIGNMENT_COLUMNS[1]} holds another value."
        )
    northern_ireland = systems == NI_SYSTEM
    in_region = (
        household[REGION_COLUMN].to_numpy(dtype=object) == _NORTHERN_IRELAND_REGION
    )
    if not np.array_equal(northern_ireland, in_region):
        raise ValueError(
            f"{what} draws {NI_SYSTEM} households from the count table of their "
            f"region, {_NORTHERN_IRELAND_REGION!r}, alone; the drawn system and the "
            "FRS region disagree."
        )
    return systems


def _check_layer_areas(
    household: pd.DataFrame,
    systems: np.ndarray,
    *,
    layer: BRMAAreaLayer,
    what: str,
) -> None:
    """Every Great Britain row's area is on the layer, in the household's own
    FRS region; refused by texts that name no code, so ``area_brmas``, which
    reads the layer after this check, never reaches its own refusals."""

    regions = household[REGION_COLUMN].to_numpy(dtype=object)
    for grain, spec in BRMA_GRAINS.items():
        rows = systems == spec.system
        if not rows.any():
            continue
        codes = household.loc[rows, spec.key].to_numpy(dtype=object)
        lookup = layer.brma_by_area[grain]
        if not all(isinstance(code, str) and code in lookup for code in codes):
            raise ValueError(
                f"{what} reads each {spec.system} household's BRMA off its "
                f"{spec.key} through {BRMA_LAYER_RESOURCE}; some drawn areas are not "
                "on the layer, so the layer and the supports must carry one code "
                "vintage (microcosm#931)."
            )
        placed = np.asarray(
            [layer.region_by_area[grain][code] for code in codes], dtype=object
        )
        if not np.array_equal(placed, regions[rows]):
            raise ValueError(
                f"{what} reads each {spec.system} household's BRMA off its "
                f"{spec.key}; {BRMA_LAYER_RESOURCE} places some drawn areas in "
                "another FRS region than their households'."
            )


def _layer_brmas(
    household: pd.DataFrame,
    systems: np.ndarray,
    current: np.ndarray,
    *,
    layer: BRMAAreaLayer,
    what: str,
) -> np.ndarray:
    """``current`` with each Great Britain row's BRMA read off its area
    through ``area_brmas``, after :func:`_check_layer_areas`."""

    _check_layer_areas(household, systems, layer=layer, what=what)
    brmas, _ = area_brmas(
        layer=layer,
        area_system=systems,
        lsoa_code=household["lsoa_code"].to_numpy(dtype=object),
        oa_code=household["oa_code"].to_numpy(dtype=object),
        region=household[REGION_COLUMN].to_numpy(dtype=object),
        brma=current,
    )
    return np.asarray(brmas, dtype=object)


def uk_area_brmas(
    person: pd.DataFrame,
    benunit: pd.DataFrame,
    household: pd.DataFrame,
    *,
    lha_category: Sequence[object],
    layer: BRMAAreaLayer,
    count_resource: Mapping[str, Any],
    what: str = "The BRMA stage",
) -> np.ndarray:
    """Each of ``household``'s rows' BRMA, as an object array in row order.

    A Great Britain row reads its BRMA off its drawn LSOA (England and
    Wales) or output area (Scotland) through ``layer``, refusing an area the
    layer does not hold or places in another FRS region. A Northern Ireland
    row's BRMA is drawn on its own benefit units from ``count_resource`` with
    ``frs_brma``'s keys, seed and salts
    (:func:`~microcosm.build.uk_runtime.frs_brma.draw_household_brmas`);
    ``lha_category`` is aligned to ``benunit``'s rows, and only Northern
    Ireland's are read. Refused, too: a system outside the three, and a
    Northern Ireland row outside its region or another system's row inside it.
    """

    systems = _brma_systems(household, what)
    current = np.full(len(household), None, dtype=object)
    northern_ireland = systems == NI_SYSTEM
    if northern_ireland.any():
        current[northern_ireland] = draw_household_brmas(
            person,
            benunit,
            household.loc[northern_ireland],
            lha_category=lha_category,
            count_resource=count_resource,
            seed=UK_BRMA_DECLARED_SEEDS[_BRMA_COLUMN],
        )
    return _layer_brmas(household, systems, current, layer=layer, what=what)


@dataclass(frozen=True)
class UKAreaBRMAResult:
    """G4's frame and what its evidence reads of it; the source counts are
    exact and print only rounded."""

    frame: Frame = field(repr=False)
    clone_count: int
    layer_sha256: str
    method: str
    count_resource_sha256: str
    by_grain: Mapping[str, int] = field(repr=False)
    northern_ireland_drawn: int = field(repr=False)

    def evidence(self) -> dict[str, object]:
        draw = _brma_draw_parameters()
        return {
            "stage": AREA_BRMA_STAGE_NAME,
            "clone_count": int(self.clone_count),
            "source_households_by_grain": {
                grain: rounded_count(count) for grain, count in self.by_grain.items()
            },
            "northern_ireland_source_households_drawn": rounded_count(
                self.northern_ireland_drawn
            ),
            "layer_sha256": self.layer_sha256,
            "method": self.method,
            "count_resource_sha256": self.count_resource_sha256,
            "seed": draw["seed"],
            "salts": draw["salts"],
            "predictors": list(UK_BRMA_PREDICTORS),
            "output": _BRMA_COLUMN,
        }


def assign_uk_spine_brmas(
    frame: Frame,
    *,
    lha_category: Sequence[object],
    layer: BRMAAreaLayer,
    layer_sha256: str,
    count_resource: Mapping[str, Any],
    count_resource_sha256: str,
) -> UKAreaBRMAResult:
    """``frame`` with every household's ``brma`` (G4), as an object column.

    ``lha_category`` is the engine's, aligned to the frame's benefit units.
    The evidence counts, over every clone, the distinct source households
    whose BRMA each Great Britain grain reads and those Northern Ireland
    draws: sources, not benefit units. Refused, by name: a frame already
    carrying ``brma`` or without the drawn columns, a misaligned
    ``lha_category``, and every refusal of :func:`uk_area_brmas`.
    """

    if not isinstance(frame, Frame):
        raise TypeError("The BRMA stage places the households of a Frame.")
    validate_uk_national_frame(frame)
    household = frame.table("household")
    _refuse_written(household, UK_AREA_BRMA_OUTPUT_COLUMNS, "the BRMA stage")
    benunit = frame.table("benunit")
    if len(lha_category) != len(benunit):
        raise ValueError(
            f"The BRMA stage reads {UK_BRMA_PREDICTORS[0]} for every benefit unit; "
            "the engine's values are not aligned to the frame's benefit units."
        )
    brmas = uk_area_brmas(
        frame.table("person"),
        benunit,
        household,
        lha_category=lha_category,
        layer=layer,
        count_resource=count_resource,
    )
    placed = household.copy()
    placed[_BRMA_COLUMN] = pd.Series(brmas, index=household.index, dtype=object)
    result = _with_household(frame, placed)
    sources = record_source_households(result, "household", stage=AREA_BRMA_STAGE_NAME)
    systems = household[ASSIGNMENT_COLUMNS[1]].to_numpy(dtype=object)
    return UKAreaBRMAResult(
        frame=result,
        clone_count=geographic_clone_count(household),
        layer_sha256=layer_sha256,
        method=layer.method,
        count_resource_sha256=count_resource_sha256,
        by_grain=MappingProxyType(
            {
                grain: distinct_carriers(sources[systems == spec.system].tolist())
                for grain, spec in BRMA_GRAINS.items()
            }
        ),
        northern_ireland_drawn=distinct_carriers(
            sources[systems == NI_SYSTEM].tolist()
        ),
    )


@dataclass(frozen=True)
class UKAreaBRMAStageTransform:
    """Whole-stage transform for the BRMA (G4), in ``frs_brma``'s slot.

    ``engine`` is the UK rules engine. Constructing the transform checks the
    stage it runs as and loads the committed layer and count table; running
    it makes one engine call over the whole frame for ``LHA_category``, as
    ``frs_brma`` does, then calls :func:`assign_uk_spine_brmas`.
    """

    stage: SourceStageSpec
    engine: object = field(repr=False)
    layer: BRMAAreaLayer = field(init=False, repr=False, compare=False)
    layer_sha256: str = field(init=False, compare=False)
    count_resource: Mapping[str, Any] = field(init=False, repr=False, compare=False)
    count_resource_sha256: str = field(init=False, compare=False)
    last_result: UKAreaBRMAResult | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        _assert_area_brma_stage(self.stage)
        layer, layer_sha256 = load_committed_brma_layer()
        counts, counts_sha256 = load_committed_brma_counts()
        object.__setattr__(self, "layer", layer)
        object.__setattr__(self, "layer_sha256", layer_sha256)
        object.__setattr__(self, "count_resource", MappingProxyType(dict(counts)))
        object.__setattr__(self, "count_resource_sha256", counts_sha256)

    def __call__(self, frame: Frame) -> Frame:
        _assert_area_brma_stage(self.stage)
        assert_rules_engine_country(self.engine, "uk")
        if not isinstance(frame, Frame):
            raise TypeError("The BRMA stage places the households of a Frame.")
        # The frame's own refusals come before the engine call they would
        # otherwise follow.
        household = frame.table("household")
        _refuse_written(household, UK_AREA_BRMA_OUTPUT_COLUMNS, "the BRMA stage")
        _check_layer_areas(
            household,
            _brma_systems(household, "The BRMA stage"),
            layer=self.layer,
            what="The BRMA stage",
        )
        materialized = self.engine.materialize(
            frame, UK_BRMA_PREDICTORS, uk_time_period(frame)
        )
        result = assign_uk_spine_brmas(
            frame,
            lha_category=materialized[UK_BRMA_PREDICTORS[0]],
            layer=self.layer,
            layer_sha256=self.layer_sha256,
            count_resource=self.count_resource,
            count_resource_sha256=self.count_resource_sha256,
        )
        object.__setattr__(self, "last_result", result)
        return result.frame

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return UK_AREA_BRMA_OUTPUT_COLUMNS

    def checkpoint_metadata(self) -> dict[str, object]:
        if self.last_result is None:
            raise RuntimeError("checkpoint metadata requires a completed stage run.")
        return {"evidence": self.last_result.evidence()}


# --- G5, rfdr ---------------------------------------------------------------------

RFDR_STAGE_NAME = "rfdr"
RFDR_OPERATION_KIND = "assign_rfdr_flags"
#: The stage's one artifact: the committed RFDR layer, whose bytes enter the
#: stage's contract.
RFDR_ARTIFACT_ROLE = "rfdr_area_layer"
RFDR_LAYER_RESOURCE = "rfdr_area_layer.json"
UK_RFDR_OUTPUT_COLUMNS = (RFDR_COLUMN,)


def rfdr_operation_parameters() -> tuple[tuple[str, dict[str, object]], ...]:
    """The reviewed operation G5's stage must declare verbatim: the layer and
    its method, the drawn columns it reads, the supports whose areas are its
    roster, and that a system with no flagged household is reported, not
    refused (C-6)."""

    return (
        (
            RFDR_OPERATION_KIND,
            {
                "layer_resource": RFDR_LAYER_RESOURCE,
                "method": RFDR_METHOD_VERSION,
                "area_system": ASSIGNMENT_COLUMNS[1],
                "area_code": ASSIGNMENT_COLUMNS[0],
                "roster_sources": {
                    system: CONDITIONED_SOURCES[system] for system in RFDR_SYSTEMS
                },
                "require_flagged": False,
            },
        ),
    )


def _assert_rfdr_stage(stage: SourceStageSpec) -> None:
    _assert_stage(
        stage,
        name=RFDR_STAGE_NAME,
        what="The fuel-relief stage",
        outputs=UK_RFDR_OUTPUT_COLUMNS,
        operations=rfdr_operation_parameters(),
        artifacts=((RFDR_ARTIFACT_ROLE, RFDR_LAYER_RESOURCE),),
    )


def load_committed_rfdr_layer() -> tuple[RFDRAreaLayer, str]:
    """The committed RFDR layer and its bytes' SHA-256."""

    payload = files("microcosm.build.uk").joinpath(RFDR_LAYER_RESOURCE).read_bytes()
    return load_rfdr_area_layer(payload), hashlib.sha256(payload).hexdigest()


def uk_rfdr_flags(
    household: pd.DataFrame,
    *,
    layer: RFDRAreaLayer,
    supports: Mapping[str, AtomicSupport],
) -> np.ndarray:
    """Each household row's fuel-relief flag from its drawn output area.

    ``area_flags`` with the support areas of each Great Britain system as the
    layer's roster and no zero-flag refusal: a layer code outside its
    system's support is refused, a malformed drawn code is refused, and a
    system none of whose households is flagged is not.
    """

    _require_columns(household, ASSIGNMENT_COLUMNS[:2], "The fuel-relief stage")
    missing = [system for system in RFDR_SYSTEMS if system not in supports]
    if missing:
        raise ValueError(
            f"The fuel-relief stage reads the support areas of {list(RFDR_SYSTEMS)} "
            f"as the layer's roster; it lacks {missing}."
        )
    flags, _ = area_flags(
        layer=layer,
        area_system=household[ASSIGNMENT_COLUMNS[1]].to_numpy(dtype=object),
        area_code=household[ASSIGNMENT_COLUMNS[0]].to_numpy(dtype=object),
        require_flagged=False,
        roster={system: supports[system].arrays["area"] for system in RFDR_SYSTEMS},
    )
    return flags


@dataclass(frozen=True)
class UKRFDRResult:
    """G5's frame and what its evidence reads of it, counted among clone-0
    rows; the source counts are exact and print only rounded."""

    frame: Frame = field(repr=False)
    clone_count: int
    layer_sha256: str
    method: str
    support_sha256: Mapping[str, str]
    flagged_by_system: Mapping[str, int] = field(repr=False)
    no_flagged_household: Mapping[str, bool]

    def evidence(self) -> dict[str, object]:
        # A result exists only for a layer whose codes are all on the roster.
        return {
            "stage": RFDR_STAGE_NAME,
            "clone_count": int(self.clone_count),
            "counted_among": "clone_0",
            "flagged_source_households": {
                system: rounded_count(count)
                for system, count in self.flagged_by_system.items()
            },
            "no_flagged_household": dict(self.no_flagged_household),
            "roster": "pass",
            "require_flagged": False,
            "layer_sha256": self.layer_sha256,
            "method": self.method,
            "support_sha256": dict(self.support_sha256),
            "output": RFDR_COLUMN,
        }


def flag_uk_spine_rfdr(
    frame: Frame,
    *,
    layer: RFDRAreaLayer,
    layer_sha256: str,
    supports: Mapping[str, AtomicSupport],
) -> UKRFDRResult:
    """``frame`` with every household's fuel-relief flag (G5).

    The flag is written as a boolean column. The evidence counts, among the
    rows of geographic clone 0, the distinct source households each Great
    Britain system flags, and whether a system holds households none of which
    is flagged, the outcome ``area_flags``' ``require_flagged`` refuses (C-6).
    Refused, by name: a frame already carrying the flag or without the drawn
    area, and every refusal of :func:`uk_rfdr_flags`.
    """

    if not isinstance(frame, Frame):
        raise TypeError("The fuel-relief stage flags the households of a Frame.")
    validate_uk_national_frame(frame)
    household = frame.table("household")
    _refuse_written(household, UK_RFDR_OUTPUT_COLUMNS, "the fuel-relief stage")
    flags = uk_rfdr_flags(household, layer=layer, supports=supports)
    flagged = household.copy()
    flagged[RFDR_COLUMN] = pd.Series(flags, index=household.index, dtype=bool)
    result = _with_household(frame, flagged)
    sources = record_source_households(result, "household", stage=RFDR_STAGE_NAME)
    systems = household[ASSIGNMENT_COLUMNS[1]].to_numpy(dtype=object)
    clone_zero = _clone_zero(household)
    flagged_by_system, none_flagged = {}, {}
    for system in RFDR_SYSTEMS:
        rows = clone_zero & (systems == system)
        flagged_by_system[system] = distinct_carriers(sources[rows & flags].tolist())
        none_flagged[system] = bool(rows.any() and not (rows & flags).any())
    return UKRFDRResult(
        frame=result,
        clone_count=geographic_clone_count(household),
        layer_sha256=layer_sha256,
        method=layer.method,
        support_sha256=MappingProxyType(
            {system: supports[system].sha256 for system in sorted(supports)}
        ),
        flagged_by_system=MappingProxyType(flagged_by_system),
        no_flagged_household=MappingProxyType(none_flagged),
    )


@dataclass(frozen=True)
class UKRFDRStageTransform:
    """Whole-stage transform for the fuel-relief flag (G5).

    ``supports`` holds the three conditioned supports' bytes, keyed by
    system, each held to canonical bytes; their Great Britain areas are the
    layer's roster. Constructing the transform checks the stage it runs as
    and loads the committed layer; running it calls
    :func:`flag_uk_spine_rfdr`.
    """

    stage: SourceStageSpec
    supports: Mapping[str, bytes] = field(repr=False)
    decoded_supports: Mapping[str, AtomicSupport] = field(
        init=False, repr=False, compare=False
    )
    layer: RFDRAreaLayer = field(init=False, repr=False, compare=False)
    layer_sha256: str = field(init=False, compare=False)
    last_result: UKRFDRResult | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        _assert_rfdr_stage(self.stage)
        object.__setattr__(
            self,
            "decoded_supports",
            MappingProxyType(uk_conditioned_supports(self.supports)),
        )
        layer, digest = load_committed_rfdr_layer()
        object.__setattr__(self, "layer", layer)
        object.__setattr__(self, "layer_sha256", digest)

    def __call__(self, frame: Frame) -> Frame:
        _assert_rfdr_stage(self.stage)
        result = flag_uk_spine_rfdr(
            frame,
            layer=self.layer,
            layer_sha256=self.layer_sha256,
            supports=self.decoded_supports,
        )
        object.__setattr__(self, "last_result", result)
        return result.frame

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return UK_RFDR_OUTPUT_COLUMNS

    def checkpoint_metadata(self) -> dict[str, object]:
        if self.last_result is None:
            raise RuntimeError("checkpoint metadata requires a completed stage run.")
        return {"evidence": self.last_result.evidence()}


# --- The SPI-copy area stage, spi_copy_areas --------------------------------------

SPI_COPY_AREAS_STAGE_NAME = "spi_copy_areas"
SPI_COPY_AREAS_OPERATION_KIND = "assign_spi_copy_areas"
#: The columns the stage rewrites on the rows the SPI support channel copied,
#: each in the dtype its column holds: the geography key, G2's three
#: assignment and ten ladder columns, G3's local authority, G4's BRMA and G5's
#: flag.
UK_SPI_COPY_AREAS_REWRITE_COLUMNS = (
    IDENTITY_COLUMN,
    *UK_SPINE_AREA_COLUMNS,
    UK_LOCAL_AUTHORITY_INPUT_COLUMN,
    _BRMA_COLUMN,
    RFDR_COLUMN,
)
#: The level every copy is drawn at: its inputs are missing by rule (C-5,
#: C-10), so it draws in its region by households.
_COPY_LEVEL = UK_ATOMIC_FALLBACK_LEVELS[-1].name
#: The two copies the support channel makes, by their ordinal on its branch
#: (the copy-id table's ``spi_support_channel`` field): the SPI copy
#: (``spi_support_channel``) and the income band donor
#: (``spi_income_band_donors``). Spelled here, so that an edit to those
#: stages' modules does not re-key this one; a test holds them equal.
_COPY_KINDS = MappingProxyType({1: "spi_support_copy", 2: "income_band_donor"})
#: The branches a copy's path may take at this stage's place: the support
#: channel's and the geographic clones'.
_COPY_STAGE_BRANCHES = (SUPPORT_CHANNEL_BRANCH, GEOGRAPHIC_SUPPORT_BRANCH)
#: Which households the copy stage's engine call reads: Northern Ireland's
#: copies alone (C-9).
_ENGINE_HOUSEHOLDS = "northern_ireland_copies"
#: How many simulations that call runs: one, over every clone's Northern
#: Ireland copies together. The scratch frame carries no clone index, so the
#: UK adapter, which batches a call by clone and refuses a frame holding
#: fewer clones than the spine's, takes it as one simulation over the rows it
#: holds, as it does at one clone.
_ENGINE_SIMULATIONS = "one"
#: The outcome the copy stage's evidence gives its draw's validation and the
#: RFDR layer's roster check: passed where it drew copies, not run where the
#: frame holds none.
_CHECKS_PASSED = "pass"
_CHECKS_NOT_RUN = "not_run"
#: The mass record on the engine call's scratch frame, which holds the copies'
#: households alone and is discarded once ``LHA_category`` is read.
_SCRATCH_MASS_REASON = (
    "spi_copy_areas engine call: a scratch subset of the frame holding the "
    "Northern Ireland SPI-channel copies' households, read for LHA_category "
    "only and discarded"
)


def spi_copy_areas_operation_parameters() -> tuple[tuple[str, dict[str, object]], ...]:
    """The reviewed operations the copy stage must declare verbatim.

    The engine predictor it consumes, on Northern Ireland's copies alone, in
    one simulation over every clone's copies; then which rows it re-places
    (the copies the support channel's branch makes, read on the support and
    geographic branches), the key it gives them, the area draw (G2's
    definition at L3, with G2's stream and seed), the stages whose lookups it
    re-runs (G3, G4 with Northern Ireland's count-table draw, seed and salts,
    and G5) and the columns it rewrites.
    """

    return (
        (
            _PREDICTORS_OPERATION_KIND,
            {"predictors": list(UK_BRMA_PREDICTORS), "consumed_only": True},
        ),
        (
            SPI_COPY_AREAS_OPERATION_KIND,
            {
                "copies": {
                    "branch": SUPPORT_CHANNEL_BRANCH,
                    "paths_require": list(_COPY_STAGE_BRANCHES),
                    "kinds": {str(k): kind for k, kind in _COPY_KINDS.items()},
                },
                "key": {
                    "column": IDENTITY_COLUMN,
                    "source": GEOGRAPHIC_IDENTITY_SOURCE,
                    "vintage_resource": FRS_RELEASE_RESOURCE,
                },
                "area_draw": {
                    "definition_of": ATOMIC_AREA_ASSIGN_STAGE_NAME,
                    "level": _COPY_LEVEL,
                    "stream": UK_SPINE_ATOMIC_AREA_STREAM,
                    "seed": UK_SPINE_ATOMIC_AREA_SEED,
                },
                "local_authority": {"as_stage": LOCAL_AUTHORITY_STAGE_NAME},
                "brma": {
                    "as_stage": AREA_BRMA_STAGE_NAME,
                    "engine_households": _ENGINE_HOUSEHOLDS,
                    "engine_simulations": _ENGINE_SIMULATIONS,
                    **_brma_draw_parameters(),
                },
                "rfdr": {"as_stage": RFDR_STAGE_NAME, "require_flagged": False},
                "rewrites": list(UK_SPI_COPY_AREAS_REWRITE_COLUMNS),
            },
        ),
    )


def _spi_copy_areas_artifacts() -> tuple[tuple[str, str], ...]:
    """Every artifact the stages it re-runs read: the FRS release (G1b's
    vintage), the names roster (G3), the BRMA layer and counts (G4) and the
    RFDR layer (G5)."""

    return (
        (FRS_RELEASE_ARTIFACT_ROLE, FRS_RELEASE_RESOURCE),
        (LOCAL_AUTHORITY_ARTIFACT_ROLE, UK_LOCAL_AUTHORITY_NAMES_RESOURCE),
        *_brma_artifacts(),
        (RFDR_ARTIFACT_ROLE, RFDR_LAYER_RESOURCE),
    )


def _assert_spi_copy_areas_stage(stage: SourceStageSpec) -> None:
    _assert_stage(
        stage,
        name=SPI_COPY_AREAS_STAGE_NAME,
        what="The SPI-copy area stage",
        outputs=(),
        rewrites=UK_SPI_COPY_AREAS_REWRITE_COLUMNS,
        operations=spi_copy_areas_operation_parameters(),
        artifacts=_spi_copy_areas_artifacts(),
    )


@dataclass(frozen=True)
class _CopyRows:
    """Every row's true source and copy path, which rows the support channel
    copied and each row's ordinal on its branch (0 for a row it did not)."""

    sources: np.ndarray
    paths: list[tuple[tuple[str, int], ...]]
    copies: np.ndarray
    ordinals: np.ndarray


def _copy_rows(household: pd.DataFrame, what: str) -> _CopyRows:
    """The rows the support channel copied, through the shared reader with
    the support and geographic branches required, as G1b reads a path."""

    sources, paths = copy_paths(
        household, UK_COPY_LINEAGE, require=_COPY_STAGE_BRANCHES
    )
    if any(step not in _COPY_STAGE_BRANCHES for path in paths for step, _ in path):
        raise ValueError(
            f"{what} re-places the support channel's copies right after its "
            f"stages, before any later copy step; a household's path takes a "
            f"branch other than {list(_COPY_STAGE_BRANCHES)}."
        )
    ordinals = np.asarray(
        [dict(path).get(SUPPORT_CHANNEL_BRANCH, 0) for path in paths], dtype=np.int64
    )
    copies = ordinals != 0
    if not np.isin(ordinals[copies], list(_COPY_KINDS)).all():
        raise ValueError(
            f"{what} re-places the support channel's SPI copies and income band "
            f"donors, its ordinals {list(_COPY_KINDS)}; a household carries "
            "another ordinal."
        )
    return _CopyRows(sources=sources, paths=paths, copies=copies, ordinals=ordinals)


def _copy_keys(household: pd.DataFrame, rows: _CopyRows, what: str) -> list[str]:
    """Each copy's own geography key, in row order (:func:`uk_spi_copy_keys`)."""

    _require_columns(household, (SOURCE_YEAR_COLUMN,), what)
    copies = rows.copies
    years = household[SOURCE_YEAR_COLUMN]
    if not pd.api.types.is_integer_dtype(years.dtype) or years.isna().any():
        raise ValueError(
            f"{what} reads an integer {SOURCE_YEAR_COLUMN!r}; it holds {years.dtype}."
        )
    vintages = frs_vintages_by_source_year()
    copy_years = years.to_numpy(dtype=np.int64)[copies]
    if not np.isin(copy_years, np.fromiter(vintages, np.int64)).all():
        raise ValueError(
            f"{what} keys each copy with the FRS vintage frs_release.json pins for "
            f"its {SOURCE_YEAR_COLUMN}; it pins the source year(s) "
            f"{sorted(vintages)}, and some copies carry another."
        )
    return [
        household_draw_key(
            source=GEOGRAPHIC_IDENTITY_SOURCE,
            source_vintage=vintages[int(year)],
            source_household_id=int(source),
            clone_path=path,
        )
        for source, year, path in zip(
            rows.sources[copies].tolist(),
            copy_years.tolist(),
            [
                path
                for path, copy in zip(rows.paths, copies.tolist(), strict=True)
                if copy
            ],
            strict=True,
        )
    ]


def uk_spi_copy_keys(
    household: pd.DataFrame, *, what: str = "The SPI-copy area stage"
) -> tuple[np.ndarray, list[str]]:
    """Which rows the SPI support channel copied, and each one's own key.

    A copy's key is ``household_draw_key`` over the geography identity's
    source, the FRS vintage frs_release.json pins for its ``source_year``,
    its true source and its copy path read on the support and geographic
    branches, as the geography identity keys a household (G1b). So it is the
    key a reader of the copy's full path gives it once the later copy steps'
    columns exist at ordinal 0, and clone 0's copies keep their keys as K
    grows. Returns the copy mask and the copies' keys, in row order.

    Refused, by name: a frame without either branch's columns, malformed
    lineage, a path taking a later branch, an ordinal that is neither an SPI
    copy's nor a band donor's, a non-integer ``source_year`` and a year the
    release pins no vintage for.
    """

    rows = _copy_rows(household, what)
    return rows.copies, _copy_keys(household, rows, what)


def _rewrite(
    column: pd.Series, rows: np.ndarray, values: Sequence[object]
) -> pd.Series:
    """``column`` with ``values`` at ``rows``, in the column's own dtype."""

    data = column.to_numpy(dtype=object).copy()
    data[rows] = np.asarray(list(values), dtype=object)
    rewritten = pd.Series(data, index=column.index, dtype=column.dtype)
    if rewritten.dtype != column.dtype:
        raise ValueError(
            f"The SPI-copy area stage keeps {column.name!r} in its dtype "
            f"{column.dtype}; the rewrite gave {rewritten.dtype}."
        )
    return rewritten


def _scratch_frame(frame: Frame, household_ids: np.ndarray) -> Frame:
    """The frame's rows of ``household_ids``' households, for one engine call.

    Every column and the frame's metadata as the frame holds them, but the
    household clone index, with a mass record for the subset's weight, so
    the scratch frame is a valid UK frame of its own; it is read for the
    engine call and discarded. Without the clone index the UK adapter runs
    one simulation over the rows the frame holds, at every clone count: with
    it, a spine built with ``clones=K`` would refuse the subset whenever some
    clone holds no Northern Ireland copy, since it refuses a call over part
    of its pool (:data:`_ENGINE_SIMULATIONS`, C-9).
    """

    person = frame.table("person")
    selected = frame.select(
        person["person_household_id"].isin(household_ids).to_numpy(dtype=bool)
    )
    old_total = float(frame.weights_for("household").total)
    new_total = float(selected.weights_for("household").total)
    record = MassChangeRecord(
        entity="household",
        old_total=old_total,
        new_total=new_total,
        declared_factor=new_total / old_total if old_total else None,
        reason=_SCRATCH_MASS_REASON,
    )
    weights: Weights = selected.weights_for("household")
    tables = {entity: selected.table(entity) for entity in selected.entities}
    tables["household"] = tables["household"].drop(
        columns=[_CLONE_INDEX_COLUMN], errors="ignore"
    )
    scratch = Frame(
        tables,
        selected.schema,
        {"household": weights},
        selected.strata,
        mass_log=(*selected.mass_log, record),
        metadata=selected.metadata,
    )
    validate_uk_national_frame(scratch)
    return scratch


def _copy_brmas(
    frame: Frame,
    rows: np.ndarray,
    *,
    engine: object,
    layer: BRMAAreaLayer,
    count_resource: Mapping[str, Any],
    what: str,
) -> np.ndarray:
    """The BRMA of each copy at ``rows`` of ``frame``'s household table.

    ``frame`` holds the copies' new areas. A Great Britain copy reads its
    BRMA off its area through the layer; Northern Ireland's copies take one
    engine call over their households alone for ``LHA_category``, which reads
    only a household's own bedroom entitlement and shared-rate status, one
    simulation over every clone's copies (:func:`_scratch_frame`), and then
    draw on their own benefit units with ``frs_brma``'s seed and salts, each
    copy a draw of its own (C-9). No call runs without such a copy.
    """

    household = frame.table("household")
    copies = household.loc[rows]
    systems = _brma_systems(copies, what)
    # Great Britain's areas are checked before the engine call.
    _check_layer_areas(copies, systems, layer=layer, what=what)
    current = np.full(len(copies), None, dtype=object)
    northern_ireland = systems == NI_SYSTEM
    if northern_ireland.any():
        ids = copies.loc[northern_ireland, "household_id"].to_numpy()
        scratch = _scratch_frame(frame, ids)
        materialized = engine.materialize(
            scratch, UK_BRMA_PREDICTORS, uk_time_period(frame)
        )
        lha_category = materialized[UK_BRMA_PREDICTORS[0]]
        scratch_benunit = scratch.table("benunit")
        if len(lha_category) != len(scratch_benunit):
            raise ValueError(
                f"{what} reads {UK_BRMA_PREDICTORS[0]} for the Northern Ireland "
                "copies' benefit units; the engine's values are not aligned to them."
            )
        scratch_household = scratch.table("household")
        drawn = pd.Series(
            draw_household_brmas(
                scratch.table("person"),
                scratch_benunit,
                scratch_household,
                lha_category=lha_category,
                count_resource=count_resource,
                seed=UK_BRMA_DECLARED_SEEDS[_BRMA_COLUMN],
            ),
            index=scratch_household["household_id"].to_numpy(),
            dtype=object,
        )
        if not np.isin(ids, drawn.index.to_numpy()).all():
            raise ValueError(
                f"{what} draws each Northern Ireland copy's BRMA on the engine "
                "call's frame of their households; that frame lacks some of them."
            )
        current[northern_ireland] = drawn.reindex(ids).to_numpy(dtype=object)
    return _layer_brmas(copies, systems, current, layer=layer, what=what)


@dataclass(frozen=True)
class UKSPICopyAreasResult:
    """The copy stage's frame and what its evidence reads of it, counted
    among clone-0 rows; the source counts are exact and print only rounded."""

    frame: Frame = field(repr=False)
    clone_count: int
    seed: int
    stream: tuple[object, ...]
    definition_sha256: str
    support_sha256: Mapping[str, str]
    brma_layer_sha256: str
    brma_method: str
    count_resource_sha256: str
    rfdr_layer_sha256: str
    rfdr_method: str
    by_kind: Mapping[str, Mapping[str, int]] = field(repr=False)
    flagged_by_system: Mapping[str, int] = field(repr=False)
    no_flagged_copy: Mapping[str, bool]
    #: "pass" where the copies were drawn, "not_run" for a frame without a
    #: copy, where neither the draw's validation nor the RFDR layer's roster
    #: check runs.
    checks: str

    def evidence(self) -> dict[str, object]:
        # A result with copies exists only where every copy's draw validated
        # and the RFDR layer's codes are all on the roster; without a copy
        # neither check runs, and both read not_run.
        draw = _brma_draw_parameters()
        return {
            "stage": SPI_COPY_AREAS_STAGE_NAME,
            "clone_count": int(self.clone_count),
            "counted_among": "clone_0",
            "copy_source_households": {
                kind: {system: rounded_count(count) for system, count in counts.items()}
                for kind, counts in self.by_kind.items()
            },
            "flagged_copy_source_households": {
                system: rounded_count(count)
                for system, count in self.flagged_by_system.items()
            },
            "no_flagged_copy": dict(self.no_flagged_copy),
            "level": _COPY_LEVEL,
            "seed": int(self.seed),
            "stream": list(self.stream),
            "definition_sha256": self.definition_sha256,
            "support_sha256": dict(self.support_sha256),
            "validation": {"outcome": self.checks, "scope": _VALIDATION_SCOPE},
            "local_authority_vintage": UK_LOCAL_AUTHORITY_VINTAGE,
            "names_source_sha256": LAD23_NAMES_SHA256,
            "brma": {
                "layer_sha256": self.brma_layer_sha256,
                "method": self.brma_method,
                "count_resource_sha256": self.count_resource_sha256,
                "engine_households": _ENGINE_HOUSEHOLDS,
                "engine_simulations": _ENGINE_SIMULATIONS,
                "seed": draw["seed"],
                "salts": draw["salts"],
            },
            "rfdr": {
                "roster": self.checks,
                "require_flagged": False,
                "layer_sha256": self.rfdr_layer_sha256,
                "method": self.rfdr_method,
            },
            "rewrites": list(UK_SPI_COPY_AREAS_REWRITE_COLUMNS),
        }


def _definition_sha256(definition: Mapping[str, object]) -> str:
    """The definition's digest, as the shared operator computes it."""

    return hashlib.sha256(
        canonical_json(validate_assignment_spec(definition))
    ).hexdigest()


def assign_uk_spi_copy_areas(
    frame: Frame,
    *,
    definition: Mapping[str, object],
    supports: Mapping[str, AtomicSupport],
    engine: object,
    brma_layer: BRMAAreaLayer,
    brma_layer_sha256: str,
    count_resource: Mapping[str, Any],
    count_resource_sha256: str,
    rfdr_layer: RFDRAreaLayer,
    rfdr_layer_sha256: str,
) -> UKSPICopyAreasResult:
    """``frame`` with every SPI-channel copy re-keyed and re-placed.

    On the rows the SPI support channel copied (an SPI copy or an income band
    donor, :func:`uk_spi_copy_keys`), in turn: each copy's own geography key;
    its area, drawn by G2's ``definition`` on a narrow frame of the copies
    (their inputs are missing by rule, so every copy draws at L3), derived
    and validated; its local authority (G3); its BRMA (G4, Northern Ireland's
    copies drawn on their own benefit units after one engine call, a single
    simulation, over their households alone); and its fuel-relief flag (G5).
    Here G4 runs before G5, the reverse of their order on the spine; neither
    reads the other's column. Each rewrite keeps its column's dtype, and
    every other row and column is left as it was. The evidence counts, among
    clone-0 rows, the distinct source households copied, by kind and system,
    and those whose copies are flagged, by Great Britain system; on a frame
    without a copy it says the validation and the roster check did not run.

    Refused, naming columns and declared constants only: a frame without a
    column the stage rewrites or reads, every refusal of
    :func:`uk_spi_copy_keys`, repeated keys, a copy drawn at any level but
    L3, engine values not aligned to the Northern Ireland copies' benefit
    units or a scratch frame without one of their households, and every
    refusal of :func:`draw_uk_spine_areas`,
    :func:`uk_local_authorities`, :func:`uk_area_brmas` and
    :func:`uk_rfdr_flags`.
    """

    what = "The SPI-copy area stage"
    if not isinstance(frame, Frame):
        raise TypeError(f"{what} re-places the households of a Frame.")
    validate_uk_national_frame(frame)
    household = frame.table("household")
    _require_columns(household, UK_SPI_COPY_AREAS_REWRITE_COLUMNS, what)
    lineage = _copy_rows(household, what)
    copies = lineage.copies
    keys = _copy_keys(household, lineage, what)
    rewritten = household.copy()
    rewritten[IDENTITY_COLUMN] = _rewrite(household[IDENTITY_COLUMN], copies, keys)
    if not rewritten[IDENTITY_COLUMN].is_unique:
        raise ValueError(
            f"{what} gives every copy a key of its own; the keys repeat, so a "
            "copy's source and path repeat another household's (microcosm#931)."
        )
    by_kind: dict[str, dict[str, int]] = {}
    flagged_by_system: dict[str, int] = {}
    no_flagged: dict[str, bool] = {}
    if copies.any():
        draw = draw_uk_spine_areas(
            rewritten.loc[copies], definition=definition, supports=supports
        )
        if not (np.asarray(draw.levels, dtype=object) == _COPY_LEVEL).all():
            raise ValueError(
                f"{what} draws every copy at {_COPY_LEVEL}, its region by "
                "households; the draw placed some copies at another level."
            )
        for column in UK_SPINE_AREA_COLUMNS:
            rewritten[column] = _rewrite(
                household[column], copies, draw.columns[column].to_numpy(dtype=object)
            )
        rewritten[UK_LOCAL_AUTHORITY_INPUT_COLUMN] = _rewrite(
            household[UK_LOCAL_AUTHORITY_INPUT_COLUMN],
            copies,
            uk_local_authorities(draw.columns[UK_LOCAL_AUTHORITY_CODE_COLUMN]).to_numpy(
                dtype=object
            ),
        )
        placed = _with_household(frame, rewritten)
        rewritten[_BRMA_COLUMN] = _rewrite(
            household[_BRMA_COLUMN],
            copies,
            _copy_brmas(
                placed,
                copies,
                engine=engine,
                layer=brma_layer,
                count_resource=count_resource,
                what=what,
            ),
        )
        flags = uk_rfdr_flags(
            rewritten.loc[copies], layer=rfdr_layer, supports=supports
        )
        rewritten[RFDR_COLUMN] = _rewrite(household[RFDR_COLUMN], copies, flags)
    result = _with_household(frame, rewritten)
    sources = record_source_households(
        result, "household", stage=SPI_COPY_AREAS_STAGE_NAME
    )
    zero_copies = _clone_zero(household) & copies
    systems = rewritten[ASSIGNMENT_COLUMNS[1]].to_numpy(dtype=object)
    flagged = rewritten[RFDR_COLUMN].to_numpy(dtype=bool)
    for ordinal, kind in _COPY_KINDS.items():
        rows = zero_copies & (lineage.ordinals == ordinal)
        by_kind[kind] = _sources_by(sources[rows], systems[rows], SYSTEMS)
    for system in RFDR_SYSTEMS:
        rows = zero_copies & (systems == system)
        flagged_by_system[system] = distinct_carriers(sources[rows & flagged].tolist())
        no_flagged[system] = bool(rows.any() and not (rows & flagged).any())
    stream = tuple(validate_assignment_spec(definition)["stream"])
    return UKSPICopyAreasResult(
        frame=result,
        clone_count=geographic_clone_count(household),
        seed=int(stream[3]),
        stream=stream,
        definition_sha256=_definition_sha256(definition),
        support_sha256=MappingProxyType(
            {system: supports[system].sha256 for system in sorted(supports)}
        ),
        brma_layer_sha256=brma_layer_sha256,
        brma_method=brma_layer.method,
        count_resource_sha256=count_resource_sha256,
        rfdr_layer_sha256=rfdr_layer_sha256,
        rfdr_method=rfdr_layer.method,
        by_kind=MappingProxyType(
            {kind: MappingProxyType(counts) for kind, counts in by_kind.items()}
        ),
        flagged_by_system=MappingProxyType(flagged_by_system),
        no_flagged_copy=MappingProxyType(no_flagged),
        checks=_CHECKS_PASSED if copies.any() else _CHECKS_NOT_RUN,
    )


@dataclass(frozen=True)
class UKSPICopyAreasStageTransform:
    """Whole-stage transform for the SPI-copy area stage, after
    spi_income_band_donors.

    ``engine`` is the UK rules engine and ``supports`` the three conditioned
    supports' bytes, keyed by system, each held to canonical bytes, as G2's
    and G5's transforms take them. Constructing the transform checks the
    stage it runs as, builds G2's definition with G2's seed, and loads the
    committed BRMA layer and counts and the RFDR layer; running it calls
    :func:`assign_uk_spi_copy_areas`.
    """

    stage: SourceStageSpec
    engine: object = field(repr=False)
    supports: Mapping[str, bytes] = field(repr=False)
    definition: Mapping[str, object] = field(init=False, repr=False, compare=False)
    decoded_supports: Mapping[str, AtomicSupport] = field(
        init=False, repr=False, compare=False
    )
    brma_layer: BRMAAreaLayer = field(init=False, repr=False, compare=False)
    brma_layer_sha256: str = field(init=False, compare=False)
    count_resource: Mapping[str, Any] = field(init=False, repr=False, compare=False)
    count_resource_sha256: str = field(init=False, compare=False)
    rfdr_layer: RFDRAreaLayer = field(init=False, repr=False, compare=False)
    rfdr_layer_sha256: str = field(init=False, compare=False)
    last_result: UKSPICopyAreasResult | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        _assert_spi_copy_areas_stage(self.stage)
        # The definition holds every support to canonical bytes, so the
        # shared decoder alone reads them after it.
        definition = uk_conditioned_assignment_definition(
            self.supports, seed=UK_SPINE_ATOMIC_AREA_SEED
        )
        decoded = {
            system: decode_atomic_support(self.supports[system]) for system in SYSTEMS
        }
        brma_layer, brma_layer_sha256 = load_committed_brma_layer()
        counts, counts_sha256 = load_committed_brma_counts()
        rfdr_layer, rfdr_layer_sha256 = load_committed_rfdr_layer()
        for name, value in (
            ("definition", definition),
            ("decoded_supports", MappingProxyType(decoded)),
            ("brma_layer", brma_layer),
            ("brma_layer_sha256", brma_layer_sha256),
            ("count_resource", MappingProxyType(dict(counts))),
            ("count_resource_sha256", counts_sha256),
            ("rfdr_layer", rfdr_layer),
            ("rfdr_layer_sha256", rfdr_layer_sha256),
        ):
            object.__setattr__(self, name, value)

    def __call__(self, frame: Frame) -> Frame:
        _assert_spi_copy_areas_stage(self.stage)
        assert_rules_engine_country(self.engine, "uk")
        result = assign_uk_spi_copy_areas(
            frame,
            definition=self.definition,
            supports=self.decoded_supports,
            engine=self.engine,
            brma_layer=self.brma_layer,
            brma_layer_sha256=self.brma_layer_sha256,
            count_resource=self.count_resource,
            count_resource_sha256=self.count_resource_sha256,
            rfdr_layer=self.rfdr_layer,
            rfdr_layer_sha256=self.rfdr_layer_sha256,
        )
        object.__setattr__(self, "last_result", result)
        return result.frame

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return ()

    def checkpoint_metadata(self) -> dict[str, object]:
        if self.last_result is None:
            raise RuntimeError("checkpoint metadata requires a completed stage run.")
        return {"evidence": self.last_result.evidence()}
