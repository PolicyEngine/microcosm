"""The UK's conditioned area draw: its definition, inputs and fallback ladder (microcosm#931).

G2 of the clone-first plan draws each household's atomic area within its FRS
region, conditioned on what the FRS records about the real address (O14 iv
and v): the address's IMD decile within its nation, and the household's
council-tax band where the survey gives it a band its region uses. The shared
operator takes both (:mod:`microcosm.build.atomic_geography`: a code
constraint that may be missing, and ``weight_by``, a weight column chosen per
household), and the shared chooser
(:func:`~microcosm.build.atomic_support_conditioning.choose_fallback_levels`)
picks the level each household is drawn at. This module declares the UK's use
of both; it runs no stage:

- :func:`uk_conditioned_assignment_definition` is the operator's declaration
  over the three conditioned supports. It is today's region-only declaration
  (``atomic_area_support.uk_atomic_assignment_definition``) with the
  conditioned sources; the region (required) and the transient decile input
  (``required: false``) as constraints; England and Wales and Scotland
  sampling by the household's band weight, or by households where it has no
  band; Northern Ireland, which has domestic rates, by households; the ten
  ladder columns as its only layers; and the spine's own stream.
- :data:`UK_ATOMIC_FALLBACK_LEVELS` is the ladder, in the plan's order: L0
  the decile and the band's weight, L1 the decile and households, L2 the
  region and the band's weight, L3 the region and households. Northern
  Ireland, with no band weights, takes L1 and L3 only.
- :func:`uk_draw_inputs` is the narrow transient frame the draw reads: each
  household's geography key and region, its decile as "1" to "10" (missing
  for 0, no observed decile) and its band where ``council_tax_band_observed``
  (missing otherwise). Rows the SPI support channel copied get both inputs
  missing, so they draw at L3, the level the SPI-copy area stage draws them
  at, though a copy keeps its parent's decile and band flag (C-5, C-10).
- :func:`uk_draw_levels` gives each household its level: the first whose
  inputs are present and whose cell has mass in the weight it samples by
  there, the chooser's rule, with the chooser's arguments read from the
  definition itself, so the chooser and the draw agree.
- :func:`uk_draw_frame` sets every input a household's level drops to
  missing, so one definition serves every level.
- :func:`uk_fallback_reasons` says, for the evidence, why each household is
  drawn where it is.

The module is its own, not part of :mod:`atomic_area_support`, whose hash the
national problem, the dense measures, the inspection export and the geography
identity bind: it imports that module's constants and two of its helpers and
leaves it as it is (C-28).

Refusals give columns and declared constants only, never a key, an id, a code
of a household or a count.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral
from types import MappingProxyType

import numpy as np
import pandas as pd

from microcosm.build.atomic_geography import (
    AtomicSupport,
    decode_atomic_support,
    validate_assignment_spec,
)
from microcosm.build.atomic_support_conditioning import (
    FallbackLevel,
    choose_fallback_levels,
)
from microcosm.build.source_lineage import copy_paths
from microcosm.build.uk_runtime.atomic_area_support import (
    ASSIGNMENT_COLUMNS,
    BAND_WEIGHT_COLUMNS,
    CONDITIONED_SOURCES,
    IMD_DECILE_COLUMN,
    IMD_DECILES,
    SYSTEMS,
    _profile,
    uk_conditioned_support_base,
)
from microcosm.build.uk_runtime.council_tax_constants import UK_COUNCIL_TAX_BANDS
from microcosm.build.uk_runtime.geographic_clones import (
    GEOGRAPHY_HOUSEHOLD_KEY_COLUMN,
)
from microcosm.build.uk_runtime.household_lineage import UK_COPY_LINEAGE

__all__ = [
    "BAND_INPUT_COLUMN",
    "COUNCIL_TAX_BAND_COLUMN",
    "COUNCIL_TAX_BAND_OBSERVED_COLUMN",
    "DECILE_INPUT_COLUMN",
    "DEFAULT_WEIGHT_COLUMN",
    "DRAW_INPUT_COLUMNS",
    "IDENTITY_COLUMN",
    "REGION_COLUMN",
    "SUPPORT_CHANNEL_BRANCH",
    "UK_ATOMIC_FALLBACK_LEVELS",
    "UK_DRAW_CONSTRAINTS",
    "UK_FALLBACK_STATUSES",
    "UK_SPINE_AREA_COLUMNS",
    "UK_SPINE_ATOMIC_AREA_DRAW_STREAM",
    "UK_SPINE_ATOMIC_AREA_SEED",
    "UK_SPINE_ATOMIC_AREA_STREAM",
    "UK_SPINE_LADDER_COLUMNS",
    "UK_SPINE_LAYERS",
    "UNOBSERVED_IMD_DECILE",
    "uk_band_weight_map",
    "uk_conditioned_assignment_definition",
    "uk_conditioned_supports",
    "uk_draw_frame",
    "uk_draw_inputs",
    "uk_draw_levels",
    "uk_draw_systems",
    "uk_fallback_reasons",
    "uk_support_channel_copies",
    "uk_system_fallback_levels",
]

#: The key every draw is keyed on: each clone's geography key, which the
#: spine's geographic_identity stage writes and the SPI-copy area stage
#: rewrites on copies (named in the geographic_clones leaf;
#: ``atomic_area_support.IDENTITY_COLUMN`` is the same name).
IDENTITY_COLUMN = GEOGRAPHY_HOUSEHOLD_KEY_COLUMN
#: The spine's own stream for the area draw, and its seed: 0, declared by the
#: stage (C-3), one constant for both lines, so clone 0 nests at every K.
UK_SPINE_ATOMIC_AREA_STREAM = "uk-spine-atomic-area-v1"
UK_SPINE_ATOMIC_AREA_SEED = 0
_STREAM_ALGORITHM = "sha256-u53-v1"
#: The definition's stream at that seed, the draw's ``stream`` entry
#: (:func:`uk_conditioned_assignment_definition`), which a spine's geography
#: binding records with the seed at its index 3 (microcosm#931).
UK_SPINE_ATOMIC_AREA_DRAW_STREAM = (
    _STREAM_ALGORITHM,
    UK_SPINE_ATOMIC_AREA_STREAM,
    0,
    UK_SPINE_ATOMIC_AREA_SEED,
)

#: The household columns the inputs are read from. The band flag and the
#: unobserved decile are frs_spine's (``FRS_COUNCIL_TAX_BAND_OBSERVED_COLUMN``
#: and ``FRS_IMD_DECILE_UNOBSERVED``), spelled here so that an edit to that
#: module does not re-key the area stages; a test holds the spellings equal.
REGION_COLUMN = "region"
COUNCIL_TAX_BAND_COLUMN = "council_tax_band"
COUNCIL_TAX_BAND_OBSERVED_COLUMN = "council_tax_band_observed"
UNOBSERVED_IMD_DECILE = 0
#: The transient inputs the draw reads: never stage outputs, they live on the
#: narrow frame :func:`uk_draw_inputs` builds and go with it.
DECILE_INPUT_COLUMN = "atomic_draw_imd_decile"
BAND_INPUT_COLUMN = "atomic_draw_council_tax_band"
DRAW_INPUT_COLUMNS = (
    IDENTITY_COLUMN,
    REGION_COLUMN,
    DECILE_INPUT_COLUMN,
    BAND_INPUT_COLUMN,
)
#: Each constraint input and the support code column it must match.
UK_DRAW_CONSTRAINTS: Mapping[str, str] = MappingProxyType(
    {REGION_COLUMN: "frs_region", DECILE_INPUT_COLUMN: IMD_DECILE_COLUMN}
)
#: The weight a household samples by where it has no band, or its level
#: drops it.
DEFAULT_WEIGHT_COLUMN = "households"
#: The lineage branch whose copies draw without inputs: the SPI support
#: channel, which copies households and makes the income band donors.
SUPPORT_CHANNEL_BRANCH = "spi_support_channel"

#: The ten ladder code columns the draw writes: the geography ladder's columns
#: without the engine's ``local_authority``, which the local-authority stage
#: resolves from ``local_authority_code``. Spelled here, as
#: ``geography_ladder.UK_GEOGRAPHY_LADDER_COLUMNS`` orders them, so that an
#: edit to that module does not re-key the area stages; a test holds them
#: equal.
UK_SPINE_LADDER_COLUMNS = (
    "oa_code",
    "lsoa_code",
    "msoa_code",
    "local_authority_code",
    "ward_code",
    "constituency_code",
    "region_code",
    "itl3_code",
    "itl2_code",
    "itl1_code",
)
#: Each layer's support column and output: the atomic area itself for
#: ``oa_code``, the column of the same name otherwise. The nation-native
#: aliases (``atomic_area_support.UK_NATIVE_ALIAS_COLUMNS``) are left out:
#: they are missing outside their own nation and dropped at the export
#: boundary.
UK_SPINE_LAYERS = tuple(
    ("area" if column == "oa_code" else column, column)
    for column in UK_SPINE_LADDER_COLUMNS
)
#: Every column the draw writes: the three assignment columns, then the ladder.
UK_SPINE_AREA_COLUMNS = (*ASSIGNMENT_COLUMNS, *UK_SPINE_LADDER_COLUMNS)

#: The fallback ladder, in the plan's order. A household takes the first level
#: whose inputs are present and whose cell has mass in the weight it samples
#: by there; a level reads every input it does not keep as missing.
UK_ATOMIC_FALLBACK_LEVELS = (
    FallbackLevel("L0", (REGION_COLUMN, DECILE_INPUT_COLUMN), True),
    FallbackLevel("L1", (REGION_COLUMN, DECILE_INPUT_COLUMN), False),
    FallbackLevel("L2", (REGION_COLUMN,), True),
    FallbackLevel("L3", (REGION_COLUMN,), False),
)
_LEVELS = {level.name: level for level in UK_ATOMIC_FALLBACK_LEVELS}
_BAND_WEIGHT_PREFIX = "households_band_"
#: The statuses :func:`uk_fallback_reasons` gives each input, in order.
UK_FALLBACK_STATUSES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "decile": ("kept", "unobserved", "dropped"),
        "band": ("kept", "unobserved", "dropped", "none"),
    }
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("UK conditioned area draw: " + message)


def uk_band_weight_map(system: str) -> dict[str, str]:
    """Each council-tax band the system's support weighs, and its column.

    ``{"A": "households_band_a", ...}``: A to I in England and Wales, A to H
    in Scotland, none in Northern Ireland (``BAND_WEIGHT_COLUMNS``).
    """

    _require(system in SYSTEMS, "unknown area system")
    columns = BAND_WEIGHT_COLUMNS[system]
    mapping = {
        column.removeprefix(_BAND_WEIGHT_PREFIX).upper(): column for column in columns
    }
    _require(
        all(column.startswith(_BAND_WEIGHT_PREFIX) for column in columns)
        and set(mapping) <= set(UK_COUNCIL_TAX_BANDS),
        "band weight columns must name council-tax bands",
    )
    return mapping


def uk_system_fallback_levels(system: str) -> tuple[FallbackLevel, ...]:
    """The levels a system's households take: all four where the support
    weighs bands, L1 and L3 in Northern Ireland, which has none."""

    _require(system in SYSTEMS, "unknown area system")
    if BAND_WEIGHT_COLUMNS[system]:
        return UK_ATOMIC_FALLBACK_LEVELS
    return tuple(level for level in UK_ATOMIC_FALLBACK_LEVELS if not level.weight_by)


def _seed(seed: object) -> int:
    _require(
        isinstance(seed, Integral)
        and not isinstance(seed, (bool, np.bool_))
        and 0 <= int(seed) < 2**32,
        "seed must be a bounded integer",
    )
    return int(seed)


def uk_conditioned_supports(supports: Mapping[str, bytes]) -> dict[str, AtomicSupport]:
    """The three conditioned supports, decoded, each held to canonical bytes.

    Each payload must be exactly what
    ``atomic_area_support.assemble_uk_conditioned_area_support`` makes of its
    system's canonical support, as the definition requires.
    """

    _require(
        isinstance(supports, Mapping) and set(supports) == set(SYSTEMS),
        "the three systems' supports are required",
    )
    decoded = {}
    for system in SYSTEMS:
        payload = supports[system]
        _require(type(payload) is bytes, "immutable support bytes required")
        # Refuses bytes that are not the canonical conditioned output.
        uk_conditioned_support_base(payload)
        support = decode_atomic_support(payload)
        _require(support.metadata["system"] == system, "support system differs")
        decoded[system] = support
    return decoded


def uk_conditioned_assignment_definition(
    supports: Mapping[str, bytes],
    *,
    seed: int,
) -> dict:
    """Declare the spine's conditioned area draw over the three supports.

    As ``atomic_area_support.uk_atomic_assignment_definition`` declares the
    region-only draw, but over the conditioned supports
    (``CONDITIONED_SOURCES``, canonical bytes only), with the region
    (required) and :data:`DECILE_INPUT_COLUMN` (``required: false``) as
    constraints, one stage at the atomic area that samples by
    ``weight_by`` (input :data:`BAND_INPUT_COLUMN`, each band to its
    ``households_band_*`` column, default ``households``) where the system
    weighs bands and by ``households`` in Northern Ireland, the ten ladder
    columns as layers (:data:`UK_SPINE_LAYERS`), and the stream
    ``("sha256-u53-v1", "uk-spine-atomic-area-v1", 0, seed)``.

    A missing input is skipped (the decile) or samples by the default weight
    (the band), so one definition draws every level of the ladder: a level's
    draw sets the inputs it drops to missing (:func:`uk_draw_frame`).
    """

    seed = _seed(seed)
    decoded = uk_conditioned_supports(supports)
    systems = []
    for system in SYSTEMS:
        support = decoded[system]
        level, vintage, regions, _ = _profile(system)
        _require(
            all(
                support.metadata[key] == value
                for key, value in (
                    ("system", system),
                    ("level", level),
                    ("code_system", "uk_gss"),
                    ("vintage", vintage),
                )
            ),
            "support system identity differs",
        )
        bands = uk_band_weight_map(system)
        stage = (
            {
                "level": "area",
                "weight_by": {
                    "input": BAND_INPUT_COLUMN,
                    "map": bands,
                    "default": DEFAULT_WEIGHT_COLUMN,
                },
            }
            if bands
            else {"level": "area", "weight": DEFAULT_WEIGHT_COLUMN}
        )
        columns = support.metadata["columns"]
        systems.append(
            {
                "id": system,
                "level": level,
                "code_system": "uk_gss",
                "vintage": vintage,
                "source": CONDITIONED_SOURCES[system],
                "selector": {REGION_COLUMN: list(regions)},
                "constraints": [
                    {
                        "input": column,
                        "support": matched,
                        "required": column == REGION_COLUMN,
                    }
                    for column, matched in UK_DRAW_CONSTRAINTS.items()
                ],
                "observed_area": None,
                "stages": [stage],
                "layers": [
                    {
                        "input": column,
                        "output": output,
                        **{
                            key: columns[column][key]
                            for key in ("source", "vintage", "relation")
                        },
                    }
                    for column, output in UK_SPINE_LAYERS
                ],
            }
        )
    return validate_assignment_spec(
        {
            "version": 1,
            "identity": [IDENTITY_COLUMN],
            "stream": [_STREAM_ALGORITHM, UK_SPINE_ATOMIC_AREA_STREAM, 0, seed],
            "outputs": dict(
                zip(("area", "system", "basis"), ASSIGNMENT_COLUMNS, strict=True)
            ),
            "systems": systems,
        }
    )


def uk_support_channel_copies(household: pd.DataFrame) -> np.ndarray:
    """Whether each row is a copy the SPI support channel made.

    Read from the lineage columns through the shared reader on the UK's copy
    lineage, never from ids or values: a row whose copy path takes the
    support channel's branch, an SPI copy or an income band donor or any copy
    a later stage makes of one. A table without the channel's columns has no
    such row.
    """

    branch = next(
        b for b in UK_COPY_LINEAGE.branches if b.name == SUPPORT_CHANNEL_BRANCH
    )
    if not any(column in household.columns for column in branch.columns):
        return np.zeros(len(household), dtype=bool)
    _, paths = copy_paths(household, UK_COPY_LINEAGE, require=(SUPPORT_CHANNEL_BRANCH,))
    return np.fromiter(
        (any(step == SUPPORT_CHANNEL_BRANCH for step, _ in path) for path in paths),
        dtype=bool,
        count=len(paths),
    )


def _strings(series: pd.Series, column: str) -> list[str]:
    values = series.to_numpy(dtype=object)
    _require(
        not series.isna().any()
        and all(
            isinstance(value, str) and value and value.strip() == value
            for value in values
        ),
        f"{column} must hold a nonempty code on every row",
    )
    return [str(value) for value in values]


def uk_draw_inputs(household: pd.DataFrame) -> pd.DataFrame:
    """The narrow transient frame the area draw reads, one row per household.

    Columns :data:`DRAW_INPUT_COLUMNS`, in the household table's row order
    and index, each in the graph's string dtype (``StringDtype("python")``):

    - the geography key and the region, as the table holds them;
    - :data:`DECILE_INPUT_COLUMN`: the household's ``imd_decile`` as "1" to
      "10", missing where it is 0, no observed decile (O14 v);
    - :data:`BAND_INPUT_COLUMN`: its ``council_tax_band`` where
      ``council_tax_band_observed`` is true, missing otherwise (O14 iv).

    A row the SPI support channel copied (:func:`uk_support_channel_copies`)
    gets both inputs missing, whatever its decile and flag, which it keeps
    from its parent (C-5, C-10).

    Refused, by column: a missing column, a missing or blank key or region, a
    decile column that is not integer or holds codes outside 0 to 10, a flag
    column that is not a complete boolean, and an observed band that is not a
    band A to I.
    """

    _require(isinstance(household, pd.DataFrame), "the household table is a DataFrame")
    required = (
        IDENTITY_COLUMN,
        REGION_COLUMN,
        IMD_DECILE_COLUMN,
        COUNCIL_TAX_BAND_COLUMN,
        COUNCIL_TAX_BAND_OBSERVED_COLUMN,
    )
    missing = [column for column in required if column not in household.columns]
    _require(not missing, f"the household table lacks {missing}")
    keys = _strings(household[IDENTITY_COLUMN], IDENTITY_COLUMN)
    regions = _strings(household[REGION_COLUMN], REGION_COLUMN)
    decile = household[IMD_DECILE_COLUMN]
    _require(
        pd.api.types.is_integer_dtype(decile.dtype)
        and not pd.api.types.is_bool_dtype(decile.dtype)
        and not decile.isna().any(),
        f"{IMD_DECILE_COLUMN} must be a complete integer column",
    )
    deciles = decile.to_numpy(dtype=np.int64)
    _require(
        bool(
            np.all((deciles >= UNOBSERVED_IMD_DECILE) & (deciles <= len(IMD_DECILES)))
        ),
        f"{IMD_DECILE_COLUMN} holds codes outside 0 to 10",
    )
    observed = household[COUNCIL_TAX_BAND_OBSERVED_COLUMN]
    _require(
        pd.api.types.is_bool_dtype(observed.dtype) and not observed.isna().any(),
        f"{COUNCIL_TAX_BAND_OBSERVED_COLUMN} must be a complete boolean column",
    )
    flags = observed.to_numpy(dtype=bool)
    bands = household[COUNCIL_TAX_BAND_COLUMN].to_numpy(dtype=object)
    _require(
        all(band in UK_COUNCIL_TAX_BANDS for band in bands[flags]),
        f"an observed {COUNCIL_TAX_BAND_COLUMN} is not a band A to I",
    )
    copies = uk_support_channel_copies(household)
    decile_inputs = [
        pd.NA if copy or value == UNOBSERVED_IMD_DECILE else str(int(value))
        for value, copy in zip(deciles.tolist(), copies.tolist(), strict=True)
    ]
    band_inputs = [
        str(band) if flag and not copy else pd.NA
        for band, flag, copy in zip(
            bands.tolist(), flags.tolist(), copies.tolist(), strict=True
        )
    ]
    string = pd.StringDtype(storage="python")
    return pd.DataFrame(
        {
            IDENTITY_COLUMN: pd.Series(keys, index=household.index, dtype=string),
            REGION_COLUMN: pd.Series(regions, index=household.index, dtype=string),
            DECILE_INPUT_COLUMN: pd.Series(
                decile_inputs, index=household.index, dtype=string
            ),
            BAND_INPUT_COLUMN: pd.Series(
                band_inputs, index=household.index, dtype=string
            ),
        },
        index=household.index,
    )


def uk_draw_systems(inputs: pd.DataFrame, definition: Mapping) -> np.ndarray:
    """Each row's area system, by the definition's region selectors.

    Refuses a row that no system, or more than one, selects, as the shared
    operator does.
    """

    spec = validate_assignment_spec(definition)
    systems = np.full(len(inputs), "", dtype=object)
    claimed = np.zeros(len(inputs), dtype=np.int64)
    for system in spec["systems"]:
        mask = np.ones(len(inputs), dtype=bool)
        for column, accepted in system["selector"].items():
            _require(column in inputs.columns, f"the draw inputs lack {column!r}")
            mask &= inputs[column].isin(accepted).to_numpy(dtype=bool)
        systems[mask] = system["id"]
        claimed += mask
    _require(
        bool(np.all(claimed == 1)),
        "every household must fall in exactly one area system by its region",
    )
    return systems


def _system_rules(system: Mapping) -> tuple[dict | None, str]:
    """A system's ``weight_by`` rule without its default, and the default."""

    stages = system["stages"]
    _require(
        len(stages) == 1 and stages[0]["level"] == "area",
        "the conditioned draw samples one stage, at the atomic area",
    )
    (stage,) = stages
    rule = stage.get("weight_by")
    if rule is None:
        return None, stage["weight"]
    return {"input": rule["input"], "map": dict(rule["map"])}, rule["default"]


def uk_draw_levels(
    inputs: pd.DataFrame,
    definition: Mapping,
    supports: Mapping[str, AtomicSupport],
) -> np.ndarray:
    """Each household's fallback level, by the shared chooser, in row order.

    ``inputs`` is :func:`uk_draw_inputs`' frame and ``definition``
    :func:`uk_conditioned_assignment_definition`'s. Each system's households
    go to the chooser with its support and the definition's own arguments:
    its constraints as the input-to-support mapping, its stage's
    ``weight_by`` rule without the default, and the default as
    ``default_weight``; a system without ``weight_by`` (Northern Ireland)
    takes its unweighted levels only. So a household's level is the first of
    :data:`UK_ATOMIC_FALLBACK_LEVELS` whose inputs are present and whose cell
    has mass in the weight it samples by there, and a household with neither
    input, an SPI-channel row among them, takes L3. Returns level names.
    """

    spec = validate_assignment_spec(definition)
    _require(
        isinstance(supports, Mapping)
        and set(supports) == {system["id"] for system in spec["systems"]}
        and all(isinstance(support, AtomicSupport) for support in supports.values()),
        "a decoded support for every system of the definition is required",
    )
    systems = uk_draw_systems(inputs, spec)
    names = np.full(len(inputs), "", dtype=object)
    for system in spec["systems"]:
        rows = np.flatnonzero(systems == system["id"])
        if not len(rows):
            continue
        weight_by, default = _system_rules(system)
        levels = (
            UK_ATOMIC_FALLBACK_LEVELS
            if weight_by is not None
            else tuple(
                level for level in UK_ATOMIC_FALLBACK_LEVELS if not level.weight_by
            )
        )
        names[rows] = choose_fallback_levels(
            supports[system["id"]],
            inputs.iloc[rows],
            constraint_columns={
                c["input"]: c["support"] for c in system["constraints"]
            },
            weight_by=weight_by,
            default_weight=default,
            levels=levels,
        )
    return np.asarray(names.tolist(), dtype=str)


def _level_names(inputs: pd.DataFrame, levels: Sequence[str]) -> list[str]:
    names = [str(name) for name in np.asarray(levels, dtype=object).tolist()]
    _require(len(names) == len(inputs), "one level per household")
    _require(set(names) <= set(_LEVELS), "a level outside the UK fallback ladder")
    return names


def uk_draw_frame(inputs: pd.DataFrame, levels: Sequence[str]) -> pd.DataFrame:
    """The draw's inputs at each household's level.

    A copy of ``inputs`` with every input the household's level does not
    keep set to missing: the decile where the level keeps the region alone,
    the band where the level samples by households. The definition skips a
    missing decile and samples a missing band by households, so the draw of
    this frame draws each household at its level.
    """

    names = _level_names(inputs, levels)
    frame = inputs.copy()
    constraints = dict.fromkeys(
        column for level in UK_ATOMIC_FALLBACK_LEVELS for column in level.constraints
    )
    kept = {
        column: np.asarray(
            [column in _LEVELS[name].constraints for name in names], dtype=bool
        )
        for column in constraints
    }
    kept[BAND_INPUT_COLUMN] = np.asarray(
        [_LEVELS[name].weight_by for name in names], dtype=bool
    )
    for column, keep in kept.items():
        if not keep.all():
            frame[column] = frame[column].mask(~keep)
    return frame


def uk_fallback_reasons(
    inputs: pd.DataFrame, levels: Sequence[str], definition: Mapping
) -> pd.DataFrame:
    """Why each household is drawn at its level: one status per input.

    Column ``decile``: ``kept`` where the level keeps the decile,
    ``unobserved`` where the input is missing, ``dropped`` where it is
    present but no level that keeps it has mass. Column ``band`` likewise for
    the band's weight, or ``none`` where the household's system weighs no
    band (Northern Ireland). An L0 household reads ``kept`` twice. An
    SPI-channel row's inputs are missing by rule, so it reads ``unobserved``.
    On ``inputs``' index, in its row order.
    """

    names = _level_names(inputs, levels)
    spec = validate_assignment_spec(definition)
    systems = uk_draw_systems(inputs, spec)
    banded = {
        system["id"]: _system_rules(system)[0] is not None for system in spec["systems"]
    }
    decile_present = inputs[DECILE_INPUT_COLUMN].notna().to_numpy(dtype=bool)
    band_present = inputs[BAND_INPUT_COLUMN].notna().to_numpy(dtype=bool)
    deciles, bands = [], []
    for name, system, has_decile, has_band in zip(
        names,
        systems.tolist(),
        decile_present.tolist(),
        band_present.tolist(),
        strict=True,
    ):
        level = _LEVELS[name]
        if DECILE_INPUT_COLUMN in level.constraints:
            deciles.append("kept")
        else:
            deciles.append("dropped" if has_decile else "unobserved")
        if not banded[system]:
            bands.append("none")
        elif level.weight_by:
            bands.append("kept")
        else:
            bands.append("dropped" if has_band else "unobserved")
    return pd.DataFrame(
        {
            "decile": pd.Series(deciles, index=inputs.index, dtype=object),
            "band": pd.Series(bands, index=inputs.index, dtype=object),
        },
        index=inputs.index,
    )
