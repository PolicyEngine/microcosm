#!/usr/bin/env python3
"""Measure the clone-first spine's own area draw (microcosm#931).

G2, the draw of the spine's ``atomic_area_assign`` stage
(``spine_geography``), places every geographic clone in an atomic area within
its household's FRS region, conditioned on its IMD decile and its observed
council-tax band, falling back along a declared ladder where an input is
missing or a cell has no mass. This tool measures that draw on a bound spine
checkpoint, in process, with no engine and no graph:

* It reads the checkpoint's household table (``--input-h5``, authenticated
  against its build sidecar and spine gate report as the dense and national
  lines bind one) and keeps its base rows, the FRS households on no copy
  branch (clone 0): each one's true source, region, IMD decile, council-tax
  band and band flag. A household's weight is its mass in its areas: the
  spine's household weights as the checkpoint holds them (importance
  weights, once the SPI support channel has moved part of each stratum's
  mass onto its copies), summed over every row that keeps the household's
  areas, its geographic clones and the copies later stages make of them.
  SPI-channel rows, which draw areas of their own, are left out.
* For each requested clone count K (``--n-clones``) it clones the base rows
  with the spine's own geographic expansion (G1, ``expand_uk_geographic_clones``)
  and keys every clone with the spine's geography identity (G1b,
  ``key_uk_households_by_clone_path``), so each clone draws on the key the
  spine gives it; each clone carries 1/K of its household's weight.
* It draws two laws on those rows with the shared atomic operator:
  ``region_only``, today's v1 definition
  (``atomic_area_support.uk_atomic_assignment_definition``: by census
  households within the FRS region, at ``--region-only-seed``, the dense
  role's seed by default) over the registered supports the conditioned ones
  extend; and ``conditioned``, G2's own draw
  (``spine_geography.draw_uk_spine_areas`` on ``atomic_area_conditioning``'s
  definition and draw inputs, at the spine's seed) over the three
  conditioned supports (``--supports-dir``). Each validates its mapping and
  resolves the engine's local authority as the spine's local-authority stage
  does.

Per cell {law} x {K} it records, at the spine's weights and with no solve:

* the expected rows per constituency and local authority, and their
  variance, read off the definition the law's draw runs: each row draws an
  area of its system's support among the rows that match every constraint
  input it carries (its region, and its decile where its level keeps one),
  in proportion to the weight column its stage gives it (its band's where
  its level keeps the band, the declared default otherwise), each on its own
  key, so an area's rows are a sum of independent draws;
* realized support on the supports' roster of areas (rows, Kish ESS,
  distinct source households) through ``uk_ladder_area_support_summary``,
  and the 50-floor breach tables of ``size_evaluation.area_support_tables``
  over the published areas;
* z-scores of realized against expected rows, the ladder gate's verdict and
  London's weighted household share;
* Northern Ireland's local government districts: each one's placements
  against its census household share (decision 9);
* identity stability: the same draw on reversed rows and on the clone-0
  subset, unique keys, the spine's own areas where the checkpoint carries
  them, and each clone's area unchanged from every lower K to the highest;
* for the conditioned law, the source households at each fallback level.

It is a diagnostic recorder only: no build runs it and it never gates.

Disclosure (UKDS EUL CD137 clause 8, CD171 s5.2.1): the evidence and the
receipt carry no unit record, say ``"publishable": false`` and stay with the
run, since the tool refuses an output path inside a checkout
(``receipt_counts.outside_work_trees``). Neither is published or quoted: a
region's areas' rows, and the region-only law's expected rows, sum to its
exact base households times K, so the rounded source counts protect only the
receipt and the terminal. The evidence is written under disclosure rule 2
(:func:`resuppress_evidence`), and the receipt is read off it with its
minima from 3 to 9 banded. Rule 2 audits each cell's groups against the
counts the same law publishes at its other clone counts, since a clone keeps
its area as K grows; it does not treat the difference between an area's
counts at two clone counts, the clones between them that land there, as a
cell of its own. Every count the tool prints is rounded through
``receipt_counts``, and its refusals name checks, never counts.

Example:
    uv run python tools/measure_uk_atomic_assignment.py \
        --input-h5 spine.h5 --supports-dir <supports dir> \
        --n-clones 1,15 --evidence-out <run dir>/cells.json \
        --receipt-out <run dir>/receipt.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import resource
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pandas as pd

from microcosm.build import atomic_geography as geo
from microcosm.build.source_lineage import copy_paths
from microcosm.build.trace import sha256_file
from microcosm.build.uk_runtime.atomic_area_conditioning import (
    COUNCIL_TAX_BAND_COLUMN,
    COUNCIL_TAX_BAND_OBSERVED_COLUMN,
    IDENTITY_COLUMN,
    REGION_COLUMN,
    UK_ATOMIC_FALLBACK_LEVELS,
    UK_SPINE_AREA_COLUMNS,
    UK_SPINE_ATOMIC_AREA_SEED,
    uk_conditioned_assignment_definition,
    uk_draw_frame,
    uk_draw_inputs,
    uk_support_channel_copies,
)
from microcosm.build.uk_runtime.atomic_area_support import (
    ASSIGNMENT_COLUMNS,
    CONDITIONED_SOURCES,
    IMD_DECILE_COLUMN,
    SYSTEMS,
    uk_atomic_assignment_definition,
    uk_conditioned_support_base,
)
from microcosm.build.uk_runtime.calibration_run import load_bound_spine_checkpoint
from microcosm.build.uk_runtime.country_adapter import uk_atomic_support_register
from microcosm.build.uk_runtime.geographic_clones import (
    SOURCE_YEAR_COLUMN,
    geographic_clone_count,
    ladder_clone_index_column,
)
from microcosm.build.uk_runtime.geographic_expand import (
    GEOGRAPHIC_CLONE_LIMIT,
    expand_uk_geographic_clones,
)
from microcosm.build.uk_runtime.geographic_identity import (
    key_uk_households_by_clone_path,
)
from microcosm.build.uk_runtime.geography_ladder import (
    UK_GEOGRAPHY_LADDER_COLUMNS,
    UK_LONDON_REGION_CODE,
    uk_geography_ladder_gate,
)
from microcosm.build.uk_runtime.household_lineage import (
    GEOGRAPHIC_SUPPORT_BRANCH,
    UK_COPY_LINEAGE,
)
from microcosm.build.uk_runtime.local_authority_input import (
    UK_LOCAL_AUTHORITY_CODE_COLUMN,
    UK_LOCAL_AUTHORITY_INPUT_COLUMN,
)
from microcosm.build.uk_runtime.local_doctrine import UK_LOCAL_CLONE_COUNT
from microcosm.build.uk_runtime.local_rowwise import uk_ladder_area_support_summary
from microcosm.build.uk_runtime.national_frame import (
    load_uk_national_frame,
    uk_national_frame,
)
from microcosm.build.uk_runtime.receipt_counts import (
    UK_BELOW_TEN_BAND,
    UK_DISCLOSURE_FLOOR,
    distinct_carriers,
    outside_work_trees,
    rounded_count,
)
from microcosm.build.uk_runtime.rowwise_geography import FRS_REGION_TO_REGION_CODE
from microcosm.build.uk_runtime.rowwise_posture import UK_DENSE_SEED
from microcosm.build.uk_runtime.size_evaluation import area_support_tables
from microcosm.build.uk_runtime.spine_geography import (
    draw_uk_spine_areas,
    uk_local_authorities,
)
from microcosm.frame import Frame
from microcosm.frame.concepts import SOURCE_HOUSEHOLD_ID_COLUMN
from microcosm.graph.canonical import canonical_json

_REPOSITORY = Path(__file__).resolve().parents[1]
_UK = _REPOSITORY / "packages/microcosm-build/src/microcosm/build/uk"
#: The two laws: today's region-only declaration and the spine's own
#: conditioned draw.
LAWS = ("region_only", "conditioned")
LEVELS = (
    ("constituency", "constituency_code"),
    ("la", UK_LOCAL_AUTHORITY_CODE_COLUMN),
)
_CONSTITUENCY_LEVEL = LEVELS[0][0]
CLONE = ladder_clone_index_column("household")
SOURCE_COLUMN = SOURCE_HOUSEHOLD_ID_COLUMN
#: The measurement table's weight column: each clone's share of its
#: household's spine weight.
WEIGHT_COLUMN = "household_weight"
#: The base rows' draw inputs, as the spine's root writes them.
BASE_INPUT_COLUMNS = (
    REGION_COLUMN,
    IMD_DECILE_COLUMN,
    COUNCIL_TAX_BAND_COLUMN,
    COUNCIL_TAX_BAND_OBSERVED_COLUMN,
)
#: Northern Ireland's FRS region, the system of its data zones and the level
#: its local government districts sit at.
NI_REGION = "NORTHERN_IRELAND"
NI_SYSTEM = SYSTEMS[2]
_NI_LEVEL = "la"
#: The disclosure rule this tool writes its evidence under
#: (:func:`resuppress_evidence`).
DISCLOSURE_RULE = 2
#: Rule 2 withholds within each level and region.
_RULE_2_GROUP = ("level", "region_code")
#: A published count withheld only to protect suppressed ones beside it.
WITHHELD = "withheld"
_AREA_COUNT_FIELDS = ("rows", "ess", "sources")
#: How a cell gives its expectations: per area, or (the conditioned law, whose
#: per-area expected rows are linear in the base households of its draw
#: cells) in its summaries only.
PER_AREA = "per_area"
SUMMARIES_ONLY = "summaries_only"
#: The level summary fields that read expected rows or z-scores, which a
#: cell of summaries only keeps from when they were computed.
_EXPECTATION_SUMMARY_FIELDS = (
    "max_abs_z",
    "share_abs_z_gt_3",
    "areas_with_expected_ge_5",
    "max_relative_error_expected_ge_100",
    "areas_with_expected_ge_100",
)
#: The sidecar's sampling fields the evidence records: the rung, never its
#: household counts.
_SAMPLING_FIELDS = ("fraction", "seed", "rung_token")
#: The per-cell fields the receipt repeats from the evidence.
RECEIPT_FIELDS = (
    "law",
    "n_clones",
    "source_households",
    "mapping_integrity",
    "ladder_gate",
    "levels",
    "identity_stability",
    "source_households_by_fallback_level",
)
#: The level minima the receipt bands from the floor to nine.
_RECEIPT_MINIMA = ("min_rows", "min_sources")
_STRING = pd.StringDtype(storage="python")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--input-h5", type=Path, required=True, help="A bound spine checkpoint."
    )
    parser.add_argument(
        "--input-sidecar",
        type=Path,
        help="Its build sidecar; the H5's .build.json by default.",
    )
    parser.add_argument(
        "--input-spine-gates",
        type=Path,
        help="Its spine gate report; the H5's .spine_gates.json by default.",
    )
    parser.add_argument("--input-sha256", help="The SHA-256 the spine H5 must have.")
    parser.add_argument(
        "--supports-dir",
        type=Path,
        required=True,
        help=(
            "The three conditioned supports' directory, <source>.npz each, as "
            "tools/build_uk_conditioned_area_supports.py writes them."
        ),
    )
    parser.add_argument(
        "--no-pin-supports",
        action="store_true",
        help="Skip holding the supports to the registers (synthetic supports only).",
    )
    parser.add_argument(
        "--n-clones",
        default=f"1,{UK_LOCAL_CLONE_COUNT}",
        help=f"Comma-separated clone counts K; 1,{UK_LOCAL_CLONE_COUNT} by default.",
    )
    parser.add_argument(
        "--laws",
        default=",".join(LAWS),
        help="Comma-separated laws, of region_only and conditioned; both by default.",
    )
    parser.add_argument(
        "--region-only-seed",
        type=int,
        default=UK_DENSE_SEED,
        help="The region-only law's seed; the dense role's by default.",
    )
    parser.add_argument(
        "--minimum-count",
        type=int,
        default=UK_DISCLOSURE_FLOOR,
        help=(
            "Counts resting on fewer rows or source households are withheld; "
            f"at least {UK_DISCLOSURE_FLOOR}, the default."
        ),
    )
    parser.add_argument(
        "--evidence-out",
        type=Path,
        required=True,
        help="The evidence JSON; refused inside a checkout.",
    )
    parser.add_argument(
        "--receipt-out",
        type=Path,
        help="A receipt JSON read off the evidence; refused inside a checkout.",
    )
    args = parser.parse_args(argv)
    try:
        clones = tuple(int(value) for value in str(args.n_clones).split(","))
    except ValueError:
        parser.error("--n-clones must list clone counts, comma-separated.")
    if len(set(clones)) != len(clones) or not all(
        1 <= k <= GEOGRAPHIC_CLONE_LIMIT for k in clones
    ):
        parser.error(
            "--n-clones must list distinct clone counts from 1 to "
            f"{GEOGRAPHIC_CLONE_LIMIT}."
        )
    args.n_clones = clones
    laws = tuple(str(args.laws).split(","))
    if set(laws) - set(LAWS) or len(set(laws)) != len(laws):
        parser.error(f"--laws must name distinct laws of {LAWS}.")
    args.laws = laws
    if args.minimum_count < UK_DISCLOSURE_FLOOR:
        parser.error(
            f"--minimum-count must be at least {UK_DISCLOSURE_FLOOR}, CD171's floor."
        )
    return args


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _rounded(count: int, minimum: int) -> int | str:
    """A count of source households as the tool writes and prints it: "<n"
    below the minimum, ``receipt_counts.rounded_count`` from it up."""

    value = int(count)
    if 0 < value < minimum:
        return f"<{minimum}"
    return rounded_count(value)


# --- supports and laws ------------------------------------------------------


def base_supports(conditioned: Mapping[str, bytes]) -> dict[str, bytes]:
    """The registered support each conditioned support extends, by system.

    ``atomic_area_support.uk_conditioned_support_base`` gives it back and
    refuses a payload that is not the canonical conditioned output.
    """

    return {
        system: uk_conditioned_support_base(conditioned[system]) for system in SYSTEMS
    }


def _hold_to_register(
    payloads: Mapping[str, bytes], register: Mapping[str, Mapping], kind: str
) -> None:
    for system in SYSTEMS:
        pin = {
            "sha256": hashlib.sha256(payloads[system]).hexdigest(),
            "size_bytes": len(payloads[system]),
        }
        if pin != dict(register[system]):
            raise ValueError(
                f"The {kind} support for {system} is not the register's: sha256 "
                f"{pin['sha256']} ({pin['size_bytes']} bytes), where sources.yaml "
                f"pins {register[system]['sha256']} "
                f"({register[system]['size_bytes']} bytes). Only synthetic "
                "supports run with --no-pin-supports."
            )


def load_supports(
    directory: Path, *, pin: bool
) -> tuple[dict[str, bytes], dict[str, bytes]]:
    """The three conditioned supports and the registered ones they extend.

    Reads ``<directory>/<source>.npz`` for each system's conditioned source
    (``CONDITIONED_SOURCES``). With ``pin``, each must be the conditioned
    register's bytes and the support it extends the register's
    (``country_adapter.uk_atomic_support_register``), so the region-only law
    draws on the very supports today's v1 definition binds. Returns the
    conditioned and the registered payloads, by system.
    """

    conditioned = {}
    for system in SYSTEMS:
        path = Path(directory) / f"{CONDITIONED_SOURCES[system]}.npz"
        if not path.is_file():
            raise ValueError(
                f"--supports-dir holds no {path.name}, the {system} conditioned "
                "support."
            )
        conditioned[system] = path.read_bytes()
    if pin:
        _hold_to_register(
            conditioned, uk_atomic_support_register(conditioned=True), "conditioned"
        )
    base = base_supports(conditioned)
    if pin:
        _hold_to_register(base, uk_atomic_support_register(), "registered")
    return conditioned, base


@dataclass(frozen=True)
class LawDraw:
    """One law's draw of one table.

    ``columns`` holds the three assignment columns, the ten ladder columns and
    the engine's ``local_authority``, in the graph's string dtype; ``cells``
    each row's system and the inputs the law's draw read, at the row's level
    (an input its level drops is missing); ``levels`` each row's fallback
    level (the conditioned law only). All on the table's index.
    """

    columns: pd.DataFrame = field(repr=False)
    cells: pd.DataFrame = field(repr=False)
    levels: np.ndarray | None = field(default=None, repr=False)


def _string_columns(
    values: pd.DataFrame, columns: Sequence[str], index: pd.Index
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            column: pd.Series(
                values[column].to_numpy(dtype=object), index=index, dtype=_STRING
            )
            for column in columns
        },
        index=index,
    )


def _with_local_authority(columns: pd.DataFrame) -> pd.DataFrame:
    """The draw's columns with the engine's local authority, resolved as the
    spine's local-authority stage (G3) resolves it."""

    out = columns.copy()
    out[UK_LOCAL_AUTHORITY_INPUT_COLUMN] = uk_local_authorities(
        out[UK_LOCAL_AUTHORITY_CODE_COLUMN]
    )
    return out


def _drawn_cells(systems: pd.Series, inputs: pd.DataFrame) -> pd.DataFrame:
    cells = inputs.drop(columns=[IDENTITY_COLUMN])
    cells.insert(0, "system", systems.to_numpy(dtype=object))
    return cells


def draw_region_only(
    household: pd.DataFrame,
    definition: Mapping[str, object],
    supports: Mapping[str, geo.AtomicSupport],
) -> LawDraw:
    """Today's v1 draw: each row's area by census households within its region.

    ``assign_atomic``, ``derive_geography`` and ``validate_geography`` run on
    a narrow frame of the geography key and the region alone, as the dense
    line's retired geography node drew them (microcosm#931).
    """

    inputs = pd.DataFrame(
        {
            IDENTITY_COLUMN: household[IDENTITY_COLUMN],
            REGION_COLUMN: household[REGION_COLUMN],
        },
        index=household.index,
    )
    assigned = geo.assign_atomic(inputs, definition, supports)
    placed = pd.concat([inputs, assigned], axis=1)
    derived = geo.derive_geography(placed, definition, supports)
    receipt = geo.validate_geography(
        pd.concat([placed, derived], axis=1), definition, supports
    )
    if receipt.get("outcome") != "pass":
        raise ValueError("The region-only draw's mapping-integrity validation failed.")
    columns = _string_columns(
        pd.concat([assigned, derived], axis=1), UK_SPINE_AREA_COLUMNS, household.index
    )
    return LawDraw(
        columns=_with_local_authority(columns),
        cells=_drawn_cells(columns[ASSIGNMENT_COLUMNS[1]], inputs),
    )


def draw_conditioned(
    household: pd.DataFrame,
    definition: Mapping[str, object],
    supports: Mapping[str, geo.AtomicSupport],
) -> LawDraw:
    """G2's draw: ``spine_geography.draw_uk_spine_areas`` itself.

    The cells are the draw frame G2 samples (``uk_draw_frame`` of
    ``uk_draw_inputs`` at each row's level): the inputs a level drops are
    missing there.
    """

    draw = draw_uk_spine_areas(household, definition=definition, supports=supports)
    frame = uk_draw_frame(uk_draw_inputs(household), draw.levels)
    return LawDraw(
        columns=_with_local_authority(draw.columns),
        cells=_drawn_cells(draw.columns[ASSIGNMENT_COLUMNS[1]], frame),
        levels=np.asarray(draw.levels, dtype=object),
    )


@dataclass(frozen=True)
class Laws:
    """The two laws' definitions and decoded supports.

    ``region_only`` is today's v1 definition over the registered supports at
    the given seed; ``conditioned`` is G2's definition over the conditioned
    supports at the spine's seed (``UK_SPINE_ATOMIC_AREA_SEED``).
    """

    definitions: Mapping[str, Mapping[str, object]] = field(repr=False)
    supports: Mapping[str, Mapping[str, geo.AtomicSupport]] = field(repr=False)
    support_sha256: Mapping[str, Mapping[str, str]]

    @classmethod
    def build(
        cls,
        conditioned: Mapping[str, bytes],
        base: Mapping[str, bytes],
        *,
        region_only_seed: int,
    ) -> Laws:
        payloads = {
            "region_only": {system: base[system] for system in SYSTEMS},
            "conditioned": {system: conditioned[system] for system in SYSTEMS},
        }
        definitions = {
            "region_only": uk_atomic_assignment_definition(
                payloads["region_only"], seed=region_only_seed
            ),
            "conditioned": uk_conditioned_assignment_definition(
                payloads["conditioned"], seed=UK_SPINE_ATOMIC_AREA_SEED
            ),
        }
        return cls(
            definitions=MappingProxyType(definitions),
            supports=MappingProxyType(
                {
                    law: MappingProxyType(
                        {
                            system: geo.decode_atomic_support(payloads[law][system])
                            for system in SYSTEMS
                        }
                    )
                    for law in LAWS
                }
            ),
            support_sha256=MappingProxyType(
                {
                    law: MappingProxyType(
                        {
                            system: hashlib.sha256(payloads[law][system]).hexdigest()
                            for system in SYSTEMS
                        }
                    )
                    for law in LAWS
                }
            ),
        )

    def definition_sha256(self, law: str) -> str:
        return hashlib.sha256(canonical_json(self.definitions[law])).hexdigest()

    def draw(self, law: str, household: pd.DataFrame) -> LawDraw:
        if law == "region_only":
            return draw_region_only(
                household, self.definitions[law], self.supports[law]
            )
        if law == "conditioned":
            return draw_conditioned(
                household, self.definitions[law], self.supports[law]
            )
        raise ValueError(f"No law is named {law!r}; the laws are {LAWS}.")


# --- the spine's base rows --------------------------------------------------


def load_spine(args: argparse.Namespace) -> tuple[Frame, dict, str]:
    """The bound spine checkpoint: its frame, its sidecar and its SHA-256.

    The H5 is hashed first and held to ``--input-sha256`` where one is given,
    then loaded and authenticated by ``calibration_run.load_bound_spine_checkpoint``
    against its build sidecar and spine gate report, as the dense and national
    lines bind a spine. The loaders' refusals can count the spine's rows and
    weights, so the tool keeps only the name of the check that refused.
    """

    input_h5 = Path(args.input_h5)
    if not input_h5.is_file():
        raise ValueError(f"--input-h5 is not a file: {input_h5}")
    digest = sha256_file(input_h5)
    if args.input_sha256 is not None and args.input_sha256 != digest:
        raise ValueError(
            f"--input-sha256 is not the spine H5's SHA-256 ({digest}): {input_h5}"
        )
    try:
        frame, _ = load_uk_national_frame(input_h5)
    except ValueError:
        raise ValueError(
            "The spine H5 does not load as a UK national frame "
            "(national_frame.load_uk_national_frame refused it; its text, which "
            "can count rows, is withheld)."
        ) from None
    try:
        sidecar = load_bound_spine_checkpoint(
            args.input_sidecar or input_h5.with_suffix(".build.json"),
            frame,
            gate_report_path=args.input_spine_gates
            or input_h5.with_suffix(".spine_gates.json"),
            input_sha256=digest,
        )
    except ValueError as error:
        check = str(error).split(":", 1)[0].strip()
        raise ValueError(
            f"The spine checkpoint does not bind: {check} "
            "(calibration_run.load_bound_spine_checkpoint; the rest of its text, "
            "which can count rows, is withheld)."
        ) from None
    return frame, sidecar, digest


@dataclass(frozen=True)
class SpineBase:
    """A bound spine's base rows, one per true source household.

    ``household`` holds each source's id (ascending), its draw inputs and its
    weight; ``time_period`` the year the spine's expansion wrote as each
    household's source year; ``clone_count`` the spine's K; ``spine_areas``
    the area and ladder columns the spine itself drew on its geographic
    clones, by geography key, where the checkpoint carries them.
    """

    household: pd.DataFrame = field(repr=False)
    time_period: str
    clone_count: int
    spine_areas: pd.DataFrame | None = field(default=None, repr=False)


def spine_base(frame: Frame) -> SpineBase:
    """The base rows of a clone-first spine's household table.

    The base rows are the households on no copy branch (geographic clone 0),
    read through the shared copy-path reader on the UK's copy lineage, never
    from ids. Each one's inputs are its own. Its weight is its mass in its
    areas: the spine's household weights as the checkpoint holds them
    (importance weights once the SPI support channel has moved part of each
    stratum's mass onto its copies), summed over every row that keeps the
    household's areas, its geographic clones and the copies later stages make
    of them, which keep their parent's key and areas. SPI-channel rows draw
    areas of their own and are left out. At one clone and before any copy it
    is the household's own weight.

    Refused, naming columns only: a table without the base inputs, the source
    columns or the clone index; malformed lineage; a source with two base
    rows or a clone without one; geographic clones whose inputs differ from
    their household's; and more than one source year.
    """

    household = frame.table("household")
    required = (SOURCE_COLUMN, SOURCE_YEAR_COLUMN, CLONE, *BASE_INPUT_COLUMNS)
    missing = [column for column in required if column not in household.columns]
    if missing:
        raise ValueError(
            f"The spine's household table lacks {missing}: the tool reads a "
            "clone-first spine's base rows, with their draw inputs, source "
            "columns and clone index (microcosm#931)."
        )
    clone_count = geographic_clone_count(household, stage="The area-draw measurement")
    _, paths = copy_paths(
        household, UK_COPY_LINEAGE, require=(GEOGRAPHIC_SUPPORT_BRANCH,)
    )
    geographic = np.fromiter(
        (all(step == GEOGRAPHIC_SUPPORT_BRANCH for step, _ in path) for path in paths),
        dtype=bool,
        count=len(paths),
    )
    is_base = np.fromiter((not path for path in paths), dtype=bool, count=len(paths))
    base_rows = household.iloc[np.flatnonzero(is_base)]
    clone_rows = household.iloc[np.flatnonzero(geographic)]
    base_sources = base_rows[SOURCE_COLUMN].to_numpy(dtype=np.int64)
    if len(np.unique(base_sources)) != len(base_sources):
        raise ValueError(
            "The spine's base rows repeat a source household; each FRS household "
            "has one base row (microcosm#931)."
        )
    clone_sources = clone_rows[SOURCE_COLUMN].to_numpy(dtype=np.int64)
    if not np.isin(clone_sources, base_sources).all():
        raise ValueError(
            "A geographic clone's source household has no base row in the spine "
            "(microcosm#931)."
        )
    for column in BASE_INPUT_COLUMNS:
        varied = clone_rows.groupby(SOURCE_COLUMN, sort=False)[column].nunique(
            dropna=False
        )
        if bool((varied > 1).any()):
            raise ValueError(
                f"The spine's geographic clones of a household disagree on "
                f"{column!r}; the area draw reads each household's inputs once "
                "(microcosm#931)."
            )
    years = np.unique(base_rows[SOURCE_YEAR_COLUMN].to_numpy(dtype=np.int64))
    if len(years) != 1:
        raise ValueError(
            f"The spine's base rows carry more than one {SOURCE_YEAR_COLUMN!r}; its "
            "geographic expansion writes one."
        )
    weights = np.asarray(frame.weights_for("household").values, dtype=np.float64)
    # Every row that keeps a base household's areas carries its mass: its
    # geographic clones and the copies later stages make of them, which
    # inherit its key and areas; SPI-channel rows draw areas of their own.
    placed = ~uk_support_channel_copies(household)
    weight = (
        pd.Series(
            weights[placed],
            index=household[SOURCE_COLUMN].to_numpy(dtype=np.int64)[placed],
        )
        .groupby(level=0)
        .sum()
    )
    order = np.argsort(base_sources, kind="stable")
    sources = base_sources[order]
    table = pd.DataFrame(
        {
            SOURCE_COLUMN: sources,
            **{
                column: pd.Series(
                    base_rows[column].to_numpy()[order],
                    dtype=object
                    if column in (REGION_COLUMN, COUNCIL_TAX_BAND_COLUMN)
                    else None,
                )
                for column in BASE_INPUT_COLUMNS
            },
            WEIGHT_COLUMN: weight.reindex(sources).to_numpy(dtype=np.float64),
        }
    )
    areas = None
    if all(
        column in household.columns
        for column in (IDENTITY_COLUMN, *UK_SPINE_AREA_COLUMNS)
    ):
        keys = clone_rows[IDENTITY_COLUMN].astype(str).to_numpy()
        if len(np.unique(keys)) != len(keys):
            raise ValueError(
                "The spine's geographic clones repeat a geography key; its "
                "geography identity gives each its own (microcosm#931)."
            )
        areas = pd.DataFrame(
            {
                column: clone_rows[column].astype(str).to_numpy()
                for column in UK_SPINE_AREA_COLUMNS
            },
            index=pd.Index(keys, name=IDENTITY_COLUMN),
        )
    return SpineBase(
        household=table,
        time_period=str(int(years[0])),
        clone_count=clone_count,
        spine_areas=areas,
    )


def expand_base(base: SpineBase, k: int) -> pd.DataFrame:
    """The base households as ``k`` geographic clones, keyed as the spine keys them.

    A frame of the base households alone (one person and one benefit unit
    each, every id the source id, unit design weights, the spine's source year
    as its time period) goes through the spine's geographic expansion at
    ``k`` clones and its geography identity, which write each clone's index,
    source columns and geography key. Returns that household table with each
    clone's weight: its household's spine weight over ``k``, the share the
    expansion gives every clone. The two steps' refusals can count
    households, so the tool names the step that refused.
    """

    table = base.household
    ids = table[SOURCE_COLUMN].to_numpy(dtype=np.int64)
    household = pd.DataFrame(
        {
            "household_id": ids,
            **{column: table[column] for column in BASE_INPUT_COLUMNS},
        }
    )
    frame = uk_national_frame(
        person=pd.DataFrame(
            {"person_id": ids, "person_benunit_id": ids, "person_household_id": ids}
        ),
        benunit=pd.DataFrame({"benunit_id": ids}),
        household=household,
        household_weights=np.ones(len(ids), dtype=np.float64),
        time_period=base.time_period,
    )
    try:
        expanded = expand_uk_geographic_clones(frame, clones=k).frame
    except ValueError:
        raise ValueError(
            "The spine's geographic expansion refused the base households "
            "(geographic_expand.expand_uk_geographic_clones; its text, which can "
            "count households, is withheld)."
        ) from None
    try:
        keyed = key_uk_households_by_clone_path(expanded).frame
    except ValueError:
        raise ValueError(
            "The spine's geography identity refused the base households' clones "
            "(geographic_identity.key_uk_households_by_clone_path; its text, "
            "which can count households, is withheld): frs_release.json must pin "
            "the spine's source year."
        ) from None
    out = keyed.table("household").reset_index(drop=True)
    weight = pd.Series(table[WEIGHT_COLUMN].to_numpy(dtype=np.float64), index=ids)
    out[WEIGHT_COLUMN] = (
        weight.reindex(out[SOURCE_COLUMN].to_numpy(dtype=np.int64)).to_numpy() / k
    )
    return out


# --- expectations -----------------------------------------------------------


def support_roster(supports: Mapping[str, geo.AtomicSupport]) -> SimpleNamespace:
    """The supports' areas as one roster, as the support summaries read a
    ladder: each area's region, constituency and local-authority codes."""

    return SimpleNamespace(
        **{
            column: np.concatenate(
                [
                    np.asarray(supports[system].arrays[column]).astype(str)
                    for system in SYSTEMS
                ]
            )
            for column in (
                "region_code",
                "constituency_code",
                UK_LOCAL_AUTHORITY_CODE_COLUMN,
            )
        }
    )


def _area_regions(ladder) -> dict[str, dict[str, str]]:
    """The region every constituency and local authority lies in.

    Rule 2 withholds within each level and region and audits each group
    against its region's total, so every area must lie in one region; a
    roster that places one in two is refused, naming the level only.
    """

    frame = pd.DataFrame(
        {
            "region_code": ladder.region_code.astype(str),
            "constituency_code": ladder.constituency_code.astype(str),
            "local_authority_code": ladder.local_authority_code.astype(str),
        }
    )
    regions = {}
    for level, column in LEVELS:
        held = frame.groupby(column)["region_code"].unique()
        if any(len(values) != 1 for values in held):
            raise ValueError(
                f"The supports place a {level} area in more than one region; "
                "rule 2 withholds within each region, so each area must lie in "
                "one (microcosm#931)."
            )
        regions[column] = {str(code): str(values[0]) for code, values in held.items()}
    return regions


def _present(value: object) -> str | None:
    return None if pd.isna(value) else str(value)


def _declared_laws(definition: Mapping[str, object]) -> dict[str, tuple]:
    """Each system's sampling law, as the definition the draw runs declares it.

    Returns, by system id: its constraints as (input, support column,
    required) triples, the input its stage's ``weight_by`` reads (None where
    the stage samples one weight column), that rule's map, and the weight
    column a row samples by where the rule gives none. The expectation holds
    for a draw of the atomic area in one stage with no observed area, as both
    laws draw; any other is refused.
    """

    spec = geo.validate_assignment_spec(definition)
    laws = {}
    for system in spec["systems"]:
        stages = system["stages"]
        if (
            len(stages) != 1
            or stages[0]["level"] != "area"
            or system["observed_area"] is not None
        ):
            raise ValueError(
                "The expectation reads a law that draws the atomic area in one "
                "stage with no observed area; the definition declares another "
                "(microcosm#931)."
            )
        (stage,) = stages
        rule = stage.get("weight_by")
        laws[system["id"]] = (
            tuple(
                (str(c["input"]), str(c["support"]), bool(c["required"]))
                for c in system["constraints"]
            ),
            None if rule is None else str(rule["input"]),
            {} if rule is None else dict(rule["map"]),
            str(stage["weight"] if rule is None else rule["default"]),
        )
    return laws


def expected_area_support(
    cells: pd.DataFrame,
    definition: Mapping[str, object],
    supports: Mapping[str, geo.AtomicSupport],
) -> pd.DataFrame:
    """Expected rows and their variance per constituency and local authority.

    ``cells`` gives each drawn row its system and the inputs the law's draw
    read (missing where its level drops one); ``definition`` is the law the
    draw ran, read here as the shared operator reads it
    (:func:`_declared_laws`): a row draws an area of its system's support
    among the rows matching every constraint input it carries (a missing
    input that is not required is skipped), with probability proportional to
    the weight column its stage gives it (``weight_by``'s column for its code,
    the declared default where the code is missing, or the stage's one weight
    column): the shared operator's single-stage law. Every row draws on its
    own key, so an area's realized rows are a sum of independent draws: the
    expectation is the sum of the rows' probabilities and the variance the
    sum of p(1 - p). Under the region-only law (the region, households) it is
    the region's rows times the area's share of the region's census
    households. Every area of every support is returned, those with no
    expectation included, as ``area_type``, ``area_code``, ``expected_rows``
    and ``variance``.
    """

    laws = _declared_laws(definition)
    expected: dict[str, dict[str, float]] = {area_type: {} for area_type, _ in LEVELS}
    variance: dict[str, dict[str, float]] = {area_type: {} for area_type, _ in LEVELS}
    for system in SYSTEMS:
        for area_type, column in LEVELS:
            codes = np.asarray(supports[system].arrays[column]).astype(str)
            for code in np.unique(codes).tolist():
                expected[area_type].setdefault(code, 0.0)
                variance[area_type].setdefault(code, 0.0)
    names = sorted(
        {
            name
            for constraints, weight_input, _, _ in laws.values()
            for name in (
                *(constraint[0] for constraint in constraints),
                *(() if weight_input is None else (weight_input,)),
            )
        }
    )
    inputs = [
        [_present(value) for value in cells[name].to_numpy(dtype=object)]
        if name in cells.columns
        else [None] * len(cells)
        for name in names
    ]
    systems = [str(value) for value in cells["system"].to_numpy(dtype=object)]
    drawn: Counter = Counter()
    for combination, rows in Counter(zip(systems, *inputs, strict=True)).items():
        system, values = combination[0], dict(zip(names, combination[1:], strict=True))
        if system not in laws:
            raise ValueError(
                "A drawn row's system is not one the law declares: the cells do "
                "not describe the draw."
            )
        constraints, weight_input, weight_map, default = laws[system]
        matched = []
        for name, support_column, required in constraints:
            if values[name] is None:
                if required:
                    raise ValueError(
                        "A drawn row lacks an input its law requires, which the "
                        "draw refuses: the cells do not describe the draw."
                    )
                continue
            matched.append((support_column, values[name]))
        weight = default
        if weight_input is not None and values[weight_input] is not None:
            if values[weight_input] not in weight_map:
                raise ValueError(
                    "A drawn row reads a weight code its law maps to no weight "
                    "column, which the draw refuses: the cells do not describe "
                    "the draw."
                )
            weight = weight_map[values[weight_input]]
        drawn[(system, tuple(sorted(matched)), weight)] += rows
    for (system, matched, weight), rows in sorted(drawn.items(), key=str):
        support = supports[system].arrays
        mask = np.ones(len(np.asarray(support["area"])), dtype=bool)
        for column, value in matched:
            mask &= np.asarray(support[column]).astype(str) == value
        weights = np.asarray(support[weight], dtype=np.float64)[mask]
        total = float(weights.sum())
        if not total > 0:
            raise ValueError(
                "A drawn cell has no sampling mass in its support, which the draw "
                "refuses: the cells do not describe the supports drawn on."
            )
        probabilities = weights / total
        for area_type, level_column in LEVELS:
            codes, inverse = np.unique(
                np.asarray(support[level_column])[mask].astype(str),
                return_inverse=True,
            )
            shares = np.bincount(inverse, weights=probabilities, minlength=len(codes))
            for code, share in zip(codes.tolist(), shares.tolist(), strict=True):
                expected[area_type][code] += rows * share
                variance[area_type][code] += rows * share * (1.0 - share)
    return pd.DataFrame(
        [
            {
                "area_type": area_type,
                "area_code": code,
                "expected_rows": expected[area_type][code],
                "variance": variance[area_type][code],
            }
            for area_type, _ in LEVELS
            for code in sorted(expected[area_type])
        ],
        columns=("area_type", "area_code", "expected_rows", "variance"),
    )


def ni_census_shares(support: geo.AtomicSupport) -> dict[str, float]:
    """Each Northern Ireland local government district's share of the
    support's census households."""

    codes = np.asarray(support.arrays[UK_LOCAL_AUTHORITY_CODE_COLUMN]).astype(str)
    households = np.asarray(support.arrays["households"], dtype=np.float64)
    total = float(households.sum())
    return {
        code: float(households[codes == code].sum()) / total
        for code in sorted(set(codes.tolist()))
    }


# --- realized support -------------------------------------------------------


def realized_area_support(household: pd.DataFrame, ladder) -> dict[str, pd.DataFrame]:
    return uk_ladder_area_support_summary(
        household,
        ladder,
        weight_column=WEIGHT_COLUMN,
        source_column=SOURCE_COLUMN,
    )


def breach_tables(summaries: dict[str, pd.DataFrame], exclusions: dict) -> dict:
    support = pd.concat(
        [
            frame.assign(geography_level=level)[
                [
                    "geography_level",
                    "area_code",
                    "assigned_households",
                    "effective_sample_size",
                    "nonzero_source_households",
                ]
            ]
            for level, frame in summaries.items()
        ],
        ignore_index=True,
    )
    run = SimpleNamespace(
        area_support=support,
        gates={
            "gates": {
                "uk_local_area_support": {
                    "details": {"reviewed_exclusions": exclusions}
                }
            }
        },
    )
    return area_support_tables(run)


def _cell_rows(
    expected: pd.DataFrame,
    summaries: dict[str, pd.DataFrame],
    area_regions: dict[str, dict[str, str]],
) -> list[dict]:
    """Each area's expected and realized rows, ESS, sources and z-score.

    ``z`` is the realized rows' distance from their expectation in standard
    deviations of the law's own variance, missing where that is zero.
    """

    rows = []
    for area_type, column in LEVELS:
        level = expected[expected["area_type"] == area_type].set_index("area_code")
        summary = summaries[area_type].set_index("area_code")
        for code in level.index:
            expected_rows = float(level.at[code, "expected_rows"])
            spread = float(level.at[code, "variance"])
            observed = float(summary["assigned_households"].get(code, 0.0))
            rows.append(
                {
                    "level": area_type,
                    "area_code": str(code),
                    "region_code": area_regions[column].get(str(code)),
                    "expected_rows": expected_rows,
                    "rows": observed,
                    "ess": float(summary["effective_sample_size"].get(code, 0.0)),
                    "sources": float(
                        summary["nonzero_source_households"].get(code, 0.0)
                    ),
                    "z": (observed - expected_rows) / math.sqrt(spread)
                    if spread > 0
                    else None,
                }
            )
    return rows


# --- identity stability -----------------------------------------------------


def spine_areas_equal(
    household: pd.DataFrame, drawn: pd.DataFrame, spine_areas: pd.DataFrame
) -> bool | None:
    """Whether the draw gives every clone the spine also holds the areas the
    spine drew for it, matched on the geography key; None where none of the
    table's keys is the spine's."""

    keys = household[IDENTITY_COLUMN].astype(str).to_numpy()
    present = np.isin(keys, spine_areas.index.to_numpy(dtype=str))
    if not present.any():
        return None
    theirs = spine_areas.loc[keys[present]]
    mine = drawn.iloc[np.flatnonzero(present)]
    return all(
        np.array_equal(
            mine[column].astype(str).to_numpy(), theirs[column].astype(str).to_numpy()
        )
        for column in UK_SPINE_AREA_COLUMNS
    )


def identity_stability(
    household: pd.DataFrame,
    draw: Callable[[pd.DataFrame], pd.DataFrame],
    drawn: pd.DataFrame,
    *,
    spine_areas: pd.DataFrame | None = None,
) -> dict:
    """The law's draw again on the same rows reversed and on the clone-0
    subset, against the cell's own (``drawn``); unique keys; and, where given,
    the spine's own areas."""

    columns = list(UK_SPINE_AREA_COLUMNS)
    reversed_rows = draw(household.iloc[::-1]).loc[household.index]
    pruned = draw(household[household[CLONE] == 0])

    def equal(left: pd.DataFrame, right: pd.DataFrame) -> bool:
        return all(
            left[column]
            .astype("string")
            .equals(right.loc[left.index, column].astype("string"))
            for column in columns
        )

    return {
        "reversed_rows_equal": equal(drawn, reversed_rows),
        "clone_zero_subset_equal": equal(pruned, drawn),
        "keys_unique": bool(household[IDENTITY_COLUMN].is_unique),
        "spine_areas_equal": None
        if spine_areas is None
        else spine_areas_equal(household, drawn, spine_areas),
    }


def _nested(draws: Mapping[int, Mapping[str, str]]) -> dict:
    """Whether every clone drawn at each lower K keeps its area at the highest."""

    high = max(draws)
    lower = sorted(k for k in draws if k != high)
    nested = all(
        set(draws[k]) <= set(draws[high])
        and all(draws[high][key] == area for key, area in draws[k].items())
        for k in lower
    )
    return {"lower": [int(k) for k in lower], "high": int(high), "nested": nested}


# --- disclosure -------------------------------------------------------------


def _suppress(value: float, minimum: int) -> float | str:
    return f"<{minimum}" if 0 < value < minimum else value


def _suppressed_area(row: dict, minimum: int) -> dict:
    """One area row: counts resting on too few rows or sources withheld, z with them.

    An area with rows has its rows, ESS and distinct source households read
    "<n" where its rows, or the source households of positive weight they
    rest on (a household's clones once, the receipts' rule), are fewer than
    the minimum; zeros are published. ``z`` is a function of ``rows`` and
    ``expected_rows``, so a published z-score would give a withheld count
    back; it is blanked with them. Where both are at least the minimum, an
    ESS below it alone reads "<n".
    """

    if row["rows"] > 0 and (row["rows"] < minimum or row["sources"] < minimum):
        marker = f"<{minimum}"
        return {
            **row,
            "rows": marker,
            "ess": marker,
            "sources": marker,
            "z": None,
        }
    ess = row["ess"]
    return {**row, "ess": ess if isinstance(ess, str) else _suppress(ess, minimum)}


def _area_record(row: dict, minimum: int, region_sources: Mapping[str, int]) -> dict:
    """An area's row as a cell is written: :func:`_suppressed_area`, and every
    figure of it, its expected rows included, withheld where its region rests
    on fewer source households than the minimum."""

    record = _suppressed_area(row, minimum)
    if 0 < region_sources.get(row["region_code"], 0) < minimum:
        marker = f"<{minimum}"
        record = {
            **record,
            **dict.fromkeys(_AREA_COUNT_FIELDS, marker),
            "expected_rows": marker,
            "z": None,
        }
    return record


def _complement_order(level: str, code: str) -> str:
    """A value-independent order over areas (the complement choice must not
    depend on the counts it protects)."""

    return hashlib.sha256(f"{level}:{code}".encode()).hexdigest()


def _withhold(area: dict) -> None:
    for field_ in _AREA_COUNT_FIELDS:
        area[field_] = WITHHELD
    area["z"] = None


def _suppressed_marker(area: Mapping) -> bool:
    return isinstance(area["rows"], str) and area["rows"] != WITHHELD


def _group_total(totals: Mapping[str, object] | None, members: list[dict]) -> object:
    """A group's total from ``totals`` (the region's rows), or None to read it
    off the group's expected rows."""

    if totals is None:
        return None
    return totals.get(members[0]["region_code"], 0.0)


def _group_residual(members: list[dict], total: object) -> int | None:
    """The group's total less its published counts; None where no total is
    known (a region's withheld below the floor, or expected rows withheld)."""

    if total is None:
        if not all(_number(area["expected_rows"]) for area in members):
            return None
        total = sum(area["expected_rows"] for area in members)
    if not _number(total):
        return None
    return round(total) - sum(area["rows"] for area in members if _number(area["rows"]))


def _reader_ranges(
    members: list[dict],
    minimum: int,
    bounds: Mapping[tuple[str, str], tuple[float, float]] | None = None,
) -> list[tuple[bool, float, float]]:
    """The range a reader who knows rule 2 gives each unpublished count of a group.

    A suppressed count ("<n") from 1 to n - 1, its range at one clone; the one
    count withheld beside suppressed ones above n, as the complement is
    chosen; any other withheld count from 0. ``bounds`` narrows each by the
    counts the law publishes for the same area at its other clone counts
    (:func:`_clone_count_bounds`). Returns (suppressed, low, high) for each
    unpublished count, in the group's order.
    """

    marked = [_suppressed_marker(area) for area in members]
    complement = any(marked) and sum(area["rows"] == WITHHELD for area in members) == 1
    ranges = []
    for area, suppressed in zip(members, marked, strict=True):
        if _number(area["rows"]):
            continue
        if suppressed:
            low, high = 1.0, float(minimum - 1)
        elif complement:
            low, high = float(minimum + 1), math.inf
        else:
            low, high = 0.0, math.inf
        bound_low, bound_high = (bounds or {}).get(
            (area["level"], area["area_code"]), (0.0, math.inf)
        )
        ranges.append((suppressed, max(low, bound_low), min(high, bound_high)))
    return ranges


def _sum_high(highs: Sequence[float]) -> float:
    return math.inf if any(math.isinf(high) for high in highs) else float(sum(highs))


def _pinned(ranges: list[tuple[bool, float, float]], residual: int) -> bool:
    """Whether a residual leaves an unpublished count other than a zero, or the
    suppressed counts' sum, one feasible value (or none)."""

    lows = [low for _, low, _ in ranges]
    highs = [high for _, _, high in ranges]
    for index, (_, low, high) in enumerate(ranges):
        others_high = _sum_high(highs[:index] + highs[index + 1 :])
        lowest = max(low, residual - others_high)
        highest = min(high, residual - (sum(lows) - low))
        if highest - lowest < 1 and not lowest == highest == 0:
            return True
    marked = [index for index, (suppressed, _, _) in enumerate(ranges) if suppressed]
    if not marked:
        return False
    rest = [index for index in range(len(ranges)) if index not in marked]
    lowest = max(
        sum(lows[index] for index in marked),
        residual - _sum_high([highs[index] for index in rest]),
    )
    highest = min(
        sum(highs[index] for index in marked),
        residual - sum(lows[index] for index in rest),
    )
    return highest - lowest < 1


def _group_pinned(
    members: list[dict],
    minimum: int,
    total: object,
    bounds: Mapping[tuple[str, str], tuple[float, float]] | None = None,
) -> bool:
    """Whether a group's published figures pin an unpublished count or the
    suppressed counts' sum.

    The group's counts sum to its total (the region's rows, or its expected
    rows' sum), so a reader has the residual, the total less the published
    counts. Each unpublished count lies in the range :func:`_reader_ranges`
    gives it, and the group is pinned where the residual leaves one of them,
    other than a zero, or the suppressed counts' sum, one feasible value.
    Each suppressed count is read as from 1 to ``minimum - 1``, its range at
    one clone; that narrower range pins the more often, so the check holds at
    every clone count. A group with no known total is never pinned.
    """

    if not any(isinstance(area["rows"], str) for area in members):
        return False
    residual = _group_residual(members, total)
    if residual is None:
        return False
    return _pinned(_reader_ranges(members, minimum, bounds), residual)


def _complement_suppressed(
    areas: list[dict],
    minimum: int,
    *,
    within: Sequence[str] = ("level",),
    audit: bool = False,
    totals: Mapping[str, object] | None = None,
    bounds: Mapping[tuple[str, str], tuple[float, float]] | None = None,
) -> list[dict]:
    """Withhold one published count beside the suppressed ones of each group.

    Areas are grouped by ``within`` (each level by default; each level and
    region under rule 2). Every household row lies in exactly one area per
    level, and a group's counts sum to its total (its expected rows' sum, or
    ``totals``' entry for its region), so the total less the published
    counts is the sum of the unpublished ones. With zeros published, each
    suppressed count lies in a range a sum at either end of which pins every
    one of them (microcosm#1057 review round 2, item 8). In each group with
    suppressed counts and none withheld, the first area, in a
    value-independent order, whose count exceeds the minimum is withheld as
    well (its z-score, ESS and source count with it), so the residual carries
    an unknown of at least ``minimum + 1``; a group with no such area has
    every count withheld, zeros included. With ``audit``, a group whose
    residual would still pin an unpublished count or the suppressed counts'
    sum (:func:`_group_pinned`, narrowed by ``bounds`` where given) has every
    count withheld too. A group that already carries a withheld area gains no
    other complement, so the pass is idempotent.
    """

    out = [dict(area) for area in areas]
    groups: dict[tuple, list[dict]] = {}
    for area in out:
        groups.setdefault(tuple(area[key] for key in within), []).append(area)
    for key in sorted(groups, key=lambda value: tuple(map(str, value))):
        members = groups[key]
        if any(_suppressed_marker(area) for area in members) and not any(
            area["rows"] == WITHHELD for area in members
        ):
            eligible = sorted(
                (
                    area
                    for area in members
                    if not isinstance(area["rows"], str) and area["rows"] > minimum
                ),
                key=lambda area: _complement_order(area["level"], area["area_code"]),
            )
            for area in eligible[:1] or members:
                _withhold(area)
        if audit and _group_pinned(
            members, minimum, _group_total(totals, members), bounds
        ):
            for area in members:
                _withhold(area)
    return out


def _published_keys(cell: Mapping) -> set[tuple[str, str]]:
    return {
        (area["level"], area["area_code"])
        for area in cell["areas"]
        if _number(area["rows"])
    }


def _groups(cell: Mapping) -> dict[tuple[str, str], list[dict]]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for area in cell["areas"]:
        groups.setdefault((area["level"], area["region_code"]), []).append(area)
    return groups


def _clone_count_bounds(cells: Sequence[dict]) -> list[dict]:
    """Each cell's bounds on its areas' rows from the same law's other cells.

    The geography identity keys every clone by its own path, so a clone keeps
    its area as K grows and an area's rows at a lower K are at most its rows
    at a higher one: a count the law publishes at a lower K bounds the area's
    rows from below, and one at a higher K from above. Returns, per cell, the
    bounds by (level, area code); a cell without a law or a clone count takes
    none.
    """

    published = [
        {
            (area["level"], area["area_code"]): float(area["rows"])
            for area in cell["areas"]
            if _number(area["rows"])
        }
        for cell in cells
    ]
    out = []
    for index, cell in enumerate(cells):
        bounds: dict[tuple[str, str], tuple[float, float]] = {}
        law, k = cell.get("law"), cell.get("n_clones")
        for other_index, other in enumerate(cells):
            other_k = other.get("n_clones")
            if (
                other_index == index
                or law is None
                or other.get("law") != law
                or not isinstance(k, int)
                or not isinstance(other_k, int)
                or other_k == k
            ):
                continue
            for key, rows in published[other_index].items():
                low, high = bounds.get(key, (0.0, math.inf))
                bounds[key] = (
                    (max(low, rows), high) if other_k < k else (low, min(high, rows))
                )
        out.append(bounds)
    return out


def _audit_across_clone_counts(cells: list[dict], minimum: int) -> None:
    """Rule 2's residual audit, read with the bounds the law's other clone
    counts publish (:func:`_clone_count_bounds`).

    A group pinned under the bounds has every count withheld. One still
    pinned with every count withheld (other than a group of one area, whose
    count is its region's total) has the same group withheld in the law's
    other cells, which takes the bounds away. The passes repeat until nothing
    changes; each only withholds, so they end, at a fixed point.
    """

    while True:
        changed = False
        for cell, bounds in zip(cells, _clone_count_bounds(cells), strict=True):
            if not bounds:
                continue
            before = [area["rows"] for area in cell["areas"]]
            cell["areas"] = _complement_suppressed(
                cell["areas"],
                minimum,
                within=_RULE_2_GROUP,
                audit=True,
                totals=cell.get("region_rows"),
                bounds=bounds,
            )
            changed = changed or before != [area["rows"] for area in cell["areas"]]
        for cell, bounds in zip(cells, _clone_count_bounds(cells), strict=True):
            if not bounds:
                continue
            for key, members in sorted(_groups(cell).items()):
                if (
                    len(members) < 2
                    or not all(area["rows"] == WITHHELD for area in members)
                    or not _group_pinned(
                        members,
                        minimum,
                        _group_total(cell.get("region_rows"), members),
                        bounds,
                    )
                ):
                    continue
                for other in cells:
                    if other is cell or other.get("law") != cell.get("law"):
                        continue
                    for area in _groups(other).get(key, []):
                        if area["rows"] != WITHHELD:
                            _withhold(area)
                            changed = True
        if not changed:
            return


def _published_z_summary(
    areas: list[dict], level: str
) -> tuple[float | None, float | None]:
    """max |z| and the share above 3 over the level's published areas only
    (expected rows of at least 5), so no summary carries a suppressed z."""

    z = [
        abs(float(area["z"]))
        for area in areas
        if area["level"] == level
        and area["z"] is not None
        and float(area["expected_rows"]) >= 5.0
    ]
    if not z:
        return None, None
    return max(z), sum(value > 3 for value in z) / len(z)


def _published_level_summary(areas: list[dict], level: str) -> dict:
    """A level's summary over its published areas only (rule 2).

    The minima of rows, ESS and sources are withheld where any area of the
    level is; the z summary covers published z-scores with at least five
    expected rows, and the relative error published areas with published
    expected rows of at least 100.
    """

    members = [area for area in areas if area["level"] == level]
    published = [area for area in members if _number(area["rows"])]
    protected = any(
        isinstance(area[field_], str)
        for area in members
        for field_ in _AREA_COUNT_FIELDS
    )
    z = [
        abs(float(area["z"]))
        for area in published
        if area["z"] is not None
        and _number(area["expected_rows"])
        and area["expected_rows"] >= 5.0
    ]
    big = [
        area
        for area in published
        if _number(area["expected_rows"]) and area["expected_rows"] >= 100.0
    ]
    relative = [
        abs(area["rows"] - area["expected_rows"]) / area["expected_rows"]
        for area in big
    ]

    def minimum_of(field_: str) -> float | None:
        if protected or not members:
            return None
        return float(min(area[field_] for area in members))

    return {
        "areas": len(members),
        "min_rows": minimum_of("rows"),
        "min_ess": minimum_of("ess"),
        "min_sources": minimum_of("sources"),
        "max_abs_z": max(z) if z else None,
        "share_abs_z_gt_3": sum(value > 3 for value in z) / len(z) if z else None,
        "areas_with_expected_ge_5": len(z),
        "max_relative_error_expected_ge_100": max(relative) if relative else None,
        "areas_with_expected_ge_100": len(big),
    }


def _withhold_breach_statistics(
    breaches: dict, *, protected: bool, minimum: int
) -> None:
    """Rule 1's breach-table pass: the minima of a protected level and any
    statistic inside the suppressed range."""

    for field_ in _AREA_COUNT_FIELDS:
        stats = breaches.get(field_)
        if not isinstance(stats, dict):
            continue
        for name, value in list(stats.items()):
            if not isinstance(value, int | float):
                continue
            if (protected and name == "min") or 0 < value < minimum:
                stats[name] = None


def published_breach_tables(
    areas: list[dict], exclusions: Mapping | None = None
) -> dict:
    """Rule 2's 50-floor breach tables, over each level's published areas only.

    ``size_evaluation.area_support_tables`` runs on the areas whose rows, ESS
    and sources are all published (``exclusions`` reaches it as the run's
    reviewed exclusions would; its per-level tables do not read them), so
    every distribution, breach count and code is a function of published
    figures. ``n_areas`` counts every area of the level and
    ``unpublished_areas`` those left out; a level that leaves one out
    withholds its minima, which are then not the level's.
    """

    tables = {}
    for level in sorted({area["level"] for area in areas}):
        members = [area for area in areas if area["level"] == level]
        published = [
            area
            for area in members
            if all(_number(area[field_]) for field_ in _AREA_COUNT_FIELDS)
        ]
        summary = pd.DataFrame(
            {
                "area_code": [str(area["area_code"]) for area in published],
                "assigned_households": [float(area["rows"]) for area in published],
                "effective_sample_size": [float(area["ess"]) for area in published],
                "nonzero_source_households": [
                    float(area["sources"]) for area in published
                ],
            }
        )
        table = breach_tables({level: summary}, dict(exclusions or {}))[
            "by_geography_level"
        ].get(level)
        if table is None:
            empty = {"min": None, "p10": None, "median": None}
            table = {
                **{field_: dict(empty) for field_ in _AREA_COUNT_FIELDS},
                "breaches": {
                    "rows_lt_50": 0,
                    "ess_lt_50": 0,
                    "sources_lt_50": 0,
                    "any": 0,
                    "codes": [],
                },
            }
        unpublished = len(members) - len(published)
        table = {**table, "n_areas": len(members), "unpublished_areas": unpublished}
        if unpublished:
            for field_ in _AREA_COUNT_FIELDS:
                table[field_] = {**table[field_], "min": None}
        tables[level] = table
    return tables


def _ni_block(block: dict | None, areas: list[dict]) -> dict | None:
    """Northern Ireland's districts, their figures read off the level's areas.

    Each district's expected rows, rows, sources and z-score are its area's at
    the level, as suppressed and withheld there, and its weighted share is
    withheld with its rows, so the block withholds what the level withholds
    and no more. Its census share is public. The summaries cover published
    districts only.
    """

    if block is None:
        return None
    by_code = {
        area["area_code"]: area for area in areas if area["level"] == block["level"]
    }
    districts = []
    for district in block["districts"]:
        area = by_code[district["area_code"]]
        record = {
            "area_code": district["area_code"],
            "census_share": district["census_share"],
            **{
                field_: area[field_]
                for field_ in ("expected_rows", "rows", "sources", "z")
            },
            "weighted_share": area["rows"]
            if isinstance(area["rows"], str)
            else district["weighted_share"],
        }
        districts.append(record)
    published = [district for district in districts if _number(district["rows"])]
    z = [
        abs(float(district["z"]))
        for district in published
        if district["z"] is not None
        and _number(district["expected_rows"])
        and district["expected_rows"] >= 5.0
    ]
    gaps = [
        abs(float(district["weighted_share"]) - float(district["census_share"]))
        for district in published
        if _number(district["weighted_share"])
    ]
    return {
        **block,
        "districts": districts,
        "max_abs_z": max(z) if z else None,
        "max_abs_share_gap": max(gaps) if gaps else None,
    }


def _suppress_cell(cell: dict, minimum: int) -> None:
    """Rule 2's area passes on one cell, in place: primary suppression, the
    linked levels and the complement with its audit (:func:`resuppress_evidence`)."""

    areas = [
        dict(area)
        if any(isinstance(area[field_], str) for field_ in ("rows", "sources"))
        else _suppressed_area(area, minimum)
        for area in cell["areas"]
    ]
    linked = set(cell.get("constituencies_withheld_in") or ())
    for area in areas:
        if area["level"] == _CONSTITUENCY_LEVEL and area["region_code"] in linked:
            _withhold(area)
    cell["areas"] = _complement_suppressed(
        areas,
        minimum,
        within=_RULE_2_GROUP,
        audit=True,
        totals=cell.get("region_rows"),
    )


def _summarize_cell(
    cell: dict,
    minimum: int,
    *,
    exclusions: Mapping | None = None,
    changed: frozenset[str] = frozenset(),
) -> None:
    """Rule 2's summaries of one cell, in place, over its published areas.

    A cell of summaries only (its expectations withheld) keeps the summary
    fields that read expected rows or z-scores, which were computed over the
    same published areas before they were withheld, except at a level whose
    published areas ``changed`` since: those read None.
    """

    areas = cell["areas"]
    kept = cell.get("expectation") == SUMMARIES_ONLY
    previous = cell.get("levels") or {}
    levels = {}
    for level in sorted({area["level"] for area in areas}):
        summary = _published_level_summary(areas, level)
        if kept:
            for key in _EXPECTATION_SUMMARY_FIELDS:
                summary[key] = (
                    None if level in changed else (previous.get(level) or {}).get(key)
                )
        levels[level] = summary
    cell["levels"] = levels
    cell["breaches"] = published_breach_tables(areas, exclusions)
    block = cell.get("ni_lgd_placements")
    kept_z = None if block is None else block.get("max_abs_z")
    cell["ni_lgd_placements"] = _ni_block(block, areas)
    if kept and cell["ni_lgd_placements"] is not None:
        cell["ni_lgd_placements"]["max_abs_z"] = (
            None if _NI_LEVEL in changed else kept_z
        )


def _apply_rule_2(
    cells: list[dict], minimum: int, *, exclusions: Mapping | None = None
) -> None:
    """Rule 2 on a payload's cells, in place: see :func:`resuppress_evidence`."""

    before = [_published_keys(cell) for cell in cells]
    for cell in cells:
        _suppress_cell(cell, minimum)
    _audit_across_clone_counts(cells, minimum)
    for cell, was in zip(cells, before, strict=True):
        _summarize_cell(
            cell,
            minimum,
            exclusions=exclusions,
            changed=frozenset(level for level, _ in was - _published_keys(cell)),
        )


def _resuppress_cell(cell: dict, minimum: int) -> None:
    """Rule 2 on one cell standing alone, in place."""

    _apply_rule_2([cell], minimum)


def _withhold_expectations(cell: dict) -> None:
    """A cell's per-area and per-district expected rows and z-scores withheld.

    Under the conditioned law an area's expected rows are linear in the base
    households of each draw cell (region x decile x observed band x fallback
    level), with coefficients the public supports and definition give, and a
    z-score beside its rows gives the variance, another such equation; so
    they feed the cell's summaries only, which were computed over its
    published areas first (:func:`_summarize_cell`).
    """

    cell["expectation"] = SUMMARIES_ONLY
    for area in cell["areas"]:
        area["expected_rows"] = None
        area["z"] = None
    block = cell.get("ni_lgd_placements")
    if block is not None:
        for district in block["districts"]:
            district["expected_rows"] = None
            district["z"] = None


def level_summary_cells(cell: dict, level: str) -> tuple[str, str]:
    """A receipts table's two summary cells for one level of a committed cell.

    Returns the "min rows / ESS / sources" cell and the "max abs z / share abs
    z>3 (n published)" cell, rendered from the re-suppressed evidence, so a
    receipts table cannot republish a minimum or a z summary that the evidence
    withholds (microcosm#1059 review round 2, item 7). ``n`` counts the level's
    published areas with at least five expected rows, the denominator of the
    share.
    """

    summary = cell["levels"][level]
    if summary["min_rows"] is None:
        minima = "withheld"
    else:
        minima = (
            f"{summary['min_rows']:.0f} / {summary['min_ess']:.1f} / "
            f"{summary['min_sources']:.0f}"
        )
    published = sum(
        1
        for area in cell["areas"]
        if area["level"] == level
        and area["z"] is not None
        and _number(area["expected_rows"])
        and float(area["expected_rows"]) >= 5.0
    )
    if summary["max_abs_z"] is None:
        z = f"n/a ({published})"
    else:
        z = (
            f"{summary['max_abs_z']:.2f} / {summary['share_abs_z_gt_3']:.4f} "
            f"({published})"
        )
    return minima, z


def resuppress_evidence(evidence: dict) -> dict:
    """The disclosure rule an evidence payload declares, applied to it.

    Primary suppression (``_suppressed_area``) runs when a cell is written;
    this pass adds the complementary suppression and withholds the summaries
    that would reveal a suppressed value. Rule 2 (``disclosure_rule: 2``, the
    rule this tool writes, at a minimum of at least CD171's floor) on every
    cell:

    * re-applies the primary suppression;
    * withholds every constituency count of a region whose
      constituency-by-local-authority pieces include one resting on fewer
      rows or source households than the minimum
      (``constituencies_withheld_in``, recorded when the cell was written),
      so the two levels cannot be differenced;
    * withholds a complement within each level and region, or the whole
      group, and audits each group's residual against its region's total
      (``region_rows``, or its expected rows' sum);
    * audits again with the bounds the law's other clone counts publish
      (:func:`_audit_across_clone_counts`);
    * recomputes every level summary, the breach tables
      (:func:`published_breach_tables`) and Northern Ireland's districts over
      published areas only.

    A cell whose expectations feed its summaries only
    (``expectation: summaries_only``, the conditioned law) keeps those
    summaries, computed over the same published areas before its per-area
    expected rows and z-scores were withheld, and withholds them at a level
    whose published areas change. Rule 1 is the rule of evidence written
    before the tool measured the spine's draw (no ``disclosure_rule`` key),
    kept so that such a file stays the fixed point it was written as. Each is
    idempotent, so a written file is a fixed point.
    """

    minimum = int(evidence["minimum_count"])
    out = json.loads(json.dumps(evidence))
    rule = out.get("disclosure_rule", 1)
    if rule == DISCLOSURE_RULE:
        if minimum < UK_DISCLOSURE_FLOOR:
            raise ValueError(
                f"Rule {DISCLOSURE_RULE} evidence must declare a minimum_count of "
                f"at least {UK_DISCLOSURE_FLOOR}, CD171's floor."
            )
        _apply_rule_2(out["cells"], minimum)
        return out
    if rule != 1:
        raise ValueError(
            f"Unknown disclosure_rule {rule!r}: the tool applies rules 1 and "
            f"{DISCLOSURE_RULE}."
        )
    for cell in out["cells"]:
        cell["areas"] = _complement_suppressed(cell["areas"], minimum)
        for level, summary in cell["levels"].items():
            protected = any(
                isinstance(area[field_], str)
                for area in cell["areas"]
                if area["level"] == level
                for field_ in _AREA_COUNT_FIELDS
            )
            if protected:
                summary["min_rows"] = None
                summary["min_ess"] = None
                summary["min_sources"] = None
            summary["max_abs_z"], summary["share_abs_z_gt_3"] = _published_z_summary(
                cell["areas"], level
            )
            _withhold_breach_statistics(
                cell["breaches"].get(level, {}), protected=protected, minimum=minimum
            )
    return out


# --- the cells --------------------------------------------------------------


def _region_codes(household: pd.DataFrame) -> pd.Series:
    return household[REGION_COLUMN].astype(str).map(FRS_REGION_TO_REGION_CODE)


def _region_sources(household: pd.DataFrame) -> dict[str, int]:
    """The distinct true sources of positive weight in each region, by its
    region code."""

    codes = _region_codes(household)
    sources = household[SOURCE_COLUMN].to_numpy()
    weights = household[WEIGHT_COLUMN].to_numpy(dtype=np.float64)
    out = {}
    for code in codes.dropna().unique().tolist():
        mask = (codes == code).to_numpy()
        out[str(code)] = distinct_carriers(
            sources[mask].tolist(), weights[mask].tolist()
        )
    return out


def _region_rows(
    household: pd.DataFrame, minimum: int, region_sources: Mapping[str, int]
) -> dict[str, float | str]:
    """Each region's rows, the total its groups' counts sum to, "<n" where the
    region rests on fewer source households than the minimum."""

    counts = _region_codes(household).value_counts()
    return {
        str(code): f"<{minimum}"
        if 0 < region_sources.get(str(code), 0) < minimum
        else float(rows)
        for code, rows in sorted(counts.items())
    }


def _unsafe_piece_regions(measured: pd.DataFrame, minimum: int) -> list[str]:
    """The regions where the two levels could be differenced to a small count.

    Constituencies and local authorities are linked tables: an area at one
    level can be a union of whole areas at the other, and the difference
    between two nested areas is a constituency-by-local-authority piece.
    Where every piece of a region holds no rows, or at least the minimum of
    rows and of positive-weight source households, any figure the two levels
    give there is a sum of such pieces. Returns, sorted, the regions holding
    a piece below the minimum; rule 2 withholds every constituency count
    there (:func:`_suppress_cell`) and keeps the local-authority level, which
    Northern Ireland's districts and the local-authority z-bounds read.
    """

    columns = ["region_code", *(column for _, column in LEVELS)]
    unsafe = set()
    for (region, *_), piece in measured.groupby(
        [measured[column].astype(str) for column in columns], sort=True
    ):
        rows = len(piece)
        sources = distinct_carriers(
            piece[SOURCE_COLUMN].tolist(), piece[WEIGHT_COLUMN].tolist()
        )
        if rows and (rows < minimum or sources < minimum):
            unsafe.add(str(region))
    return sorted(unsafe)


def _ladder_gate(measured: pd.DataFrame, minimum: int) -> dict:
    """The ladder gate's verdict and London's weighted household share.

    Where fewer than ``minimum`` source households carry London's share, the
    share is withheld, and the verdict is the gate's without its London
    check, whose pass or fail would say on which side of its bounds the share
    lies (``london_share_checked`` false). The gate's failure texts, which
    count rows, are not kept.
    """

    table = measured[[REGION_COLUMN, *UK_GEOGRAPHY_LADDER_COLUMNS]]
    weights = measured[WEIGHT_COLUMN].to_numpy(dtype=np.float64)
    gate = uk_geography_ladder_gate(table, weights)
    share = gate.details.get("london_weighted_household_share")
    london = measured["region_code"].astype(str).to_numpy() == UK_LONDON_REGION_CODE
    carriers = distinct_carriers(
        measured.loc[london, SOURCE_COLUMN].tolist(),
        measured.loc[london, WEIGHT_COLUMN].tolist(),
    )
    if share is not None and 0 < carriers < minimum:
        gate = uk_geography_ladder_gate(table, weights, london_share_bounds=(0.0, 1.0))
        return {
            "outcome": "pass" if gate.passed else "fail",
            "london_weighted_household_share": f"<{minimum}",
            "london_share_checked": False,
        }
    return {
        "outcome": "pass" if gate.passed else "fail",
        "london_weighted_household_share": None if share is None else float(share),
        "london_share_checked": share is not None,
    }


def _fallback_sources(
    household: pd.DataFrame, levels: np.ndarray, minimum: int = UK_DISCLOSURE_FLOOR
) -> dict:
    sources = household[SOURCE_COLUMN].to_numpy()
    weights = household[WEIGHT_COLUMN].to_numpy(dtype=np.float64)
    names = np.asarray(levels, dtype=object)
    return {
        level.name: _rounded(
            distinct_carriers(
                sources[names == level.name].tolist(),
                weights[names == level.name].tolist(),
            ),
            minimum,
        )
        for level in UK_ATOMIC_FALLBACK_LEVELS
    }


def _ni_placements(
    measured: pd.DataFrame,
    census_shares: Mapping[str, float],
    minimum: int = UK_DISCLOSURE_FLOOR,
) -> dict:
    """Northern Ireland's households by local government district, before the
    level's suppression is read onto them (:func:`_ni_block`)."""

    ni = measured[REGION_COLUMN].astype(str).to_numpy() == NI_REGION
    codes = measured.loc[ni, UK_LOCAL_AUTHORITY_CODE_COLUMN].astype(str).to_numpy()
    weights = measured.loc[ni, WEIGHT_COLUMN].to_numpy(dtype=np.float64)
    total = float(weights.sum())
    return {
        "region": NI_REGION,
        "level": _NI_LEVEL,
        "source_households": _rounded(
            distinct_carriers(
                measured.loc[ni, SOURCE_COLUMN].tolist(), weights.tolist()
            ),
            minimum,
        ),
        "districts": [
            {
                "area_code": code,
                "census_share": float(share),
                "weighted_share": float(weights[codes == code].sum()) / total
                if total > 0
                else 0.0,
            }
            for code, share in census_shares.items()
        ],
    }


def run_cell(
    base: SpineBase,
    laws: Laws,
    *,
    law: str,
    k: int,
    roster: SimpleNamespace,
    area_regions: dict[str, dict[str, str]],
    census_shares: Mapping[str, float],
    minimum: int,
) -> tuple[dict, dict[str, str]]:
    """One cell {law} x {K}: the base rows at K clones, drawn and measured.

    Returns the cell as it is written, before rule 2's complement, audits and
    summaries (:func:`measure` applies them across its cells): its areas
    under primary suppression and the region floor (:func:`_area_record`),
    with the regions' rows and the linked-level regions rule 2 reads. Also
    returns each clone's area by its geography key.
    """

    started = time.perf_counter()
    table = expand_base(base, k)
    drawn = laws.draw(law, table)
    measured = pd.concat([table, drawn.columns], axis=1)
    expected = expected_area_support(
        drawn.cells, laws.definitions[law], laws.supports[law]
    )
    summaries = realized_area_support(measured, roster)
    region_sources = _region_sources(table)
    cell = {
        "law": law,
        "n_clones": int(k),
        "source_households": _rounded(
            distinct_carriers(
                table[SOURCE_COLUMN].tolist(), table[WEIGHT_COLUMN].tolist()
            ),
            minimum,
        ),
        "mapping_integrity": "pass",
        "ladder_gate": _ladder_gate(measured, minimum),
        "expectation": PER_AREA,
        "levels": {},
        "breaches": {},
        "identity_stability": identity_stability(
            table,
            lambda rows: laws.draw(law, rows).columns,
            drawn.columns,
            spine_areas=base.spine_areas if law == "conditioned" else None,
        ),
        "source_households_by_fallback_level": None
        if drawn.levels is None
        else _fallback_sources(table, drawn.levels, minimum),
        "region_rows": _region_rows(table, minimum, region_sources),
        "constituencies_withheld_in": _unsafe_piece_regions(measured, minimum),
        "areas": [
            _area_record(row, minimum, region_sources)
            for row in _cell_rows(expected, summaries, area_regions)
        ],
        "ni_lgd_placements": _ni_placements(measured, census_shares, minimum),
    }
    cell["wall_seconds"] = time.perf_counter() - started
    cell["peak_rss_bytes"] = _peak_rss_bytes()
    keyed = dict(
        zip(
            table[IDENTITY_COLUMN].astype(str),
            drawn.columns[ASSIGNMENT_COLUMNS[0]].astype(str),
            strict=True,
        )
    )
    return cell, keyed


def measure(
    base: SpineBase,
    laws: Laws,
    *,
    law_names: Sequence[str] = LAWS,
    n_clones: Sequence[int],
    exclusions: dict,
    minimum: int = UK_DISCLOSURE_FLOOR,
) -> dict:
    """Every cell {law} x {K} under rule 2 and, per law, whether every lower
    K's draws nest in the highest's.

    Rule 2 runs across the cells (:func:`_apply_rule_2`), then the
    conditioned law's per-area expectations are withheld
    (:func:`_withhold_expectations`).
    """

    roster = support_roster(laws.supports["conditioned"])
    area_regions = _area_regions(roster)
    census = ni_census_shares(laws.supports["conditioned"][NI_SYSTEM])
    cells = []
    keyed: dict[str, dict[int, dict[str, str]]] = {law: {} for law in law_names}
    for k in n_clones:
        for law in law_names:
            cell, draws = run_cell(
                base,
                laws,
                law=law,
                k=k,
                roster=roster,
                area_regions=area_regions,
                census_shares=census,
                minimum=minimum,
            )
            cells.append(cell)
            keyed[law][k] = draws
            _log(
                f"  cell law={law} K={k}: mapping integrity "
                f"{cell['mapping_integrity']}, ladder gate "
                f"{cell['ladder_gate']['outcome']}, {cell['wall_seconds']:.0f}s"
            )
    _apply_rule_2(cells, minimum, exclusions=exclusions)
    for cell in cells:
        if cell["law"] == "conditioned":
            _withhold_expectations(cell)
    return {
        "cells": cells,
        "k_growth_draws_nested": {
            law: _nested(draws) for law, draws in keyed.items() if len(draws) >= 2
        },
    }


def _sampling(sidecar: Mapping) -> dict | None:
    sampling = sidecar.get("sampling")
    if not isinstance(sampling, Mapping):
        return None
    return {key: sampling[key] for key in _SAMPLING_FIELDS if key in sampling}


def _banded_minimum(value: object) -> object:
    """A level minimum as the receipt gives it: in ``rounded_count``'s band
    from the floor to nine, as it is otherwise."""

    if _number(value) and value >= UK_DISCLOSURE_FLOOR:
        if rounded_count(int(value)) == UK_BELOW_TEN_BAND:
            return UK_BELOW_TEN_BAND
    return value


def receipt_of(evidence: dict, evidence_sha256: str) -> dict:
    """The receipt: the evidence's digest and its per-cell summaries, read off
    the re-suppressed evidence, so it carries nothing the evidence withholds;
    a level minimum from the floor to nine reads "3-9"."""

    return {
        "evidence_sha256": evidence_sha256,
        "disclosure_rule": evidence["disclosure_rule"],
        "publishable": False,
        "spine": evidence["spine"],
        "cells": [
            {
                **{key: cell[key] for key in RECEIPT_FIELDS},
                "levels": {
                    level: {
                        key: _banded_minimum(value) if key in _RECEIPT_MINIMA else value
                        for key, value in summary.items()
                    }
                    for level, summary in cell["levels"].items()
                },
                "ni_lgd_placements": {
                    key: cell["ni_lgd_placements"][key]
                    for key in ("source_households", "max_abs_z", "max_abs_share_gap")
                },
            }
            for cell in evidence["cells"]
        ],
        "k_growth_draws_nested": evidence["k_growth_draws_nested"],
    }


def _peak_rss_bytes() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(usage if sys.platform == "darwin" else usage * 1024)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    # Both outputs rest on licensed microdata: a path git could commit is
    # refused before any input is read.
    evidence_out = outside_work_trees(
        args.evidence_out, what="The area-draw measurement's evidence"
    )
    receipt_out = (
        None
        if args.receipt_out is None
        else outside_work_trees(
            args.receipt_out, what="The area-draw measurement's receipt"
        )
    )
    conditioned, base_payloads = load_supports(
        args.supports_dir, pin=not args.no_pin_supports
    )
    laws = Laws.build(
        conditioned, base_payloads, region_only_seed=args.region_only_seed
    )
    frame, sidecar, digest = load_spine(args)
    base = spine_base(frame)
    del frame
    _log(
        f"spine: K={base.clone_count}, "
        f"{_rounded(len(base.household), args.minimum_count)} source households"
    )
    exclusions = json.loads(
        (_UK / "local_area_support_exclusions.json").read_text(encoding="utf-8")
    )["exclusions"]
    measured = measure(
        base,
        laws,
        law_names=args.laws,
        n_clones=args.n_clones,
        exclusions=exclusions,
        minimum=args.minimum_count,
    )
    evidence = {
        "disclosure_rule": DISCLOSURE_RULE,
        "publishable": False,
        "purpose": (
            "Assignment-level evidence for the clone-first spine's own area draw "
            "(microcosm#931): the spine's base rows at each clone count, keyed as "
            "the spine keys them and drawn by today's region-only definition and "
            "by the spine's conditioned draw, at the spine's weights with no "
            "solve. It is private, never published or quoted: an area's rows sum "
            "to each region's exact base households times the clone count. Under "
            "disclosure rule 2, counts resting on fewer rows or source households "
            "than the minimum are withheld with their z-scores beside a "
            "complement of their level and region, audited against the region's "
            "total and the law's other clone counts; a region holding a "
            "constituency-by-local-authority piece below the minimum withholds "
            "its constituency counts; the conditioned law's per-area expected "
            "rows and z-scores feed its summaries only; the summaries and breach "
            "tables cover published areas only; and no unit record appears."
        ),
        "date": dt.date.today().isoformat(),
        "spine": {
            "file": Path(args.input_h5).name,
            "sha256": digest,
            "geographic_clone_count": int(base.clone_count),
            "source_households": _rounded(len(base.household), args.minimum_count),
            "sampling": _sampling(sidecar),
            "carries_areas": base.spine_areas is not None,
        },
        "supports": {
            "conditioned": dict(laws.support_sha256["conditioned"]),
            "region_only": dict(laws.support_sha256["region_only"]),
            "pinned": not args.no_pin_supports,
        },
        "definitions": {
            law: {
                "sha256": laws.definition_sha256(law),
                "stream": list(laws.definitions[law]["stream"]),
            }
            for law in LAWS
        },
        "laws": list(args.laws),
        "n_clones": list(args.n_clones),
        "minimum_count": args.minimum_count,
        **measured,
    }
    evidence = resuppress_evidence(evidence)
    evidence_out.parent.mkdir(parents=True, exist_ok=True)
    evidence_out.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if receipt_out is not None:
        receipt_out.parent.mkdir(parents=True, exist_ok=True)
        receipt_out.write_text(
            json.dumps(
                receipt_of(evidence, sha256_file(evidence_out)),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    _log(f"evidence: {evidence_out}")


if __name__ == "__main__":
    main()
