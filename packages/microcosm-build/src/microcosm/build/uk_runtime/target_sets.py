"""The declared target sets of the UK builds (microcosm#1759).

``uk/spec/target_sets.yaml`` declares the UK level rule, which places every
compiled UK target at one geography level; the full build's selection of its
solve surface by level (``uk.full.target_selection@1``, from the full build's
``--target-geographies``); and the two calibration lines, which drop the
compiled national references a line does not bind (``calibration_line``, one
mode for each line). The file is a typed resource of the UK spec bundle
(kind ``target_sets``): the spec engine reads it strictly, checks it against
``target_sets.schema.json`` and keeps its descriptions and rationales apart
from its content. The shared reader and selector are in
:mod:`microcosm.calibrate.target_sets`; this module names the UK file and
products. The US declares its target sets the same way, in
``us/spec/target_sets.yaml`` (microcosm#1728).
"""

from __future__ import annotations

from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any

from microcosm.calibrate.target_sets import (
    TargetSetDeclaration,
    TargetSetDeclarations,
    load_target_set_declarations,
)

__all__ = [
    "CALIBRATION_LINE_PRODUCT",
    "FULL_LEVELS_MODE",
    "FULL_PRODUCT",
    "OFF_CALIBRATION_LINE",
    "UK_TARGET_SETS_RESOURCE",
    "uk_calibration_line_selection",
    "uk_calibration_lines",
    "uk_full_level_selection",
    "uk_target_level",
    "uk_target_set_declarations",
    "uk_target_sets_path",
]

UK_TARGET_SETS_RESOURCE = "spec/target_sets.yaml"
#: The full build's product, and its one mode: selection by level.
FULL_PRODUCT = "full"
FULL_LEVELS_MODE = "levels"
#: The calibration lines' product, one mode for each line, and the predicate
#: its clause excludes by: the targets whose contract leaves the line out.
CALIBRATION_LINE_PRODUCT = "calibration_line"
OFF_CALIBRATION_LINE = "off_calibration_line"


def uk_target_sets_path() -> Path:
    return Path(str(files("microcosm.build.uk").joinpath(UK_TARGET_SETS_RESOURCE)))


@lru_cache(maxsize=1)
def uk_target_set_declarations() -> TargetSetDeclarations:
    """The packaged UK declarations, read and checked once.

    They come through :func:`microcosm.build.country_spec.load_country_target_sets`,
    which reads the one typed resource as the bundle loader does.
    """

    # Imported here: the country spec module is large, and this module is
    # imported by modules that the country spec's own imports reach.
    from microcosm.build.country_spec import load_country_target_sets

    document = load_country_target_sets("uk")
    if document is None:  # pragma: no cover - the package declares the resource
        raise RuntimeError("The UK package declares no target_sets resource.")
    return load_target_set_declarations(document)


def uk_full_level_selection() -> TargetSetDeclaration:
    """The full build's selection by level (``uk.full.target_selection@1``)."""

    return uk_target_set_declarations().declaration(FULL_PRODUCT, FULL_LEVELS_MODE)


def uk_calibration_lines() -> tuple[str, ...]:
    """The declared calibration lines, in declaration order."""

    return uk_target_set_declarations().product(CALIBRATION_LINE_PRODUCT).mode_names


def uk_calibration_line_selection(line: str) -> TargetSetDeclaration:
    """The selection of one calibration line's national references."""

    return uk_target_set_declarations().declaration(CALIBRATION_LINE_PRODUCT, line)


def uk_target_level(spec: Any) -> str:
    """A compiled UK target's geography level, by the declared level rule.

    Refuses a target whose two level keys disagree, or whose level is not one
    of the declared five.
    """

    return uk_target_set_declarations().level_rule.level_of(spec)
