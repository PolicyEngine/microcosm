"""US weighted per-column input-mass helper.

The computation was promoted to :mod:`microcosm.build.input_mass` when the UK
terminal battery adopted the same #278 input-mass parity gate (#609); it is
schema-driven and carries no country logic. This module remains so existing US
call sites keep their import path, and now applies the one US-specific
subtraction the shared helper cannot make: the person-referencing id columns.

``input_mass_totals`` treats every numeric non-structural column as a measured
layer and reports its weighted sum, but its structural set is only the entity
ids and memberships. ``parent_1_id`` is neither: it is an arbitrary integer
label whose weighted sum means nothing, and it moves whenever person ids are
renumbered — so comparing it against a reference release built on a different
id space would fail the parity gate for no reason (microcosm#884).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from microcosm.build.input_mass import input_mass_totals
from microcosm.build.us_runtime.puf_support import US_PERSON_REFERENCE_ID_COLUMNS
from microcosm.frame import Frame

__all__ = ["US_INPUT_MASS_EXCLUDED_COLUMNS", "us_input_mass_totals"]

#: Columns whose weighted sum is not a mass. See the module docstring.
US_INPUT_MASS_EXCLUDED_COLUMNS: frozenset[str] = frozenset(
    US_PERSON_REFERENCE_ID_COLUMNS
)


def us_input_mass_totals(
    frame: Frame,
    *,
    columns: Iterable[str] | None = None,
) -> Mapping[str, float]:
    """Weighted per-column totals, less the person-referencing id columns."""

    totals = dict(input_mass_totals(frame, columns=columns))
    for column in US_INPUT_MASS_EXCLUDED_COLUMNS:
        totals.pop(column, None)
    return totals
