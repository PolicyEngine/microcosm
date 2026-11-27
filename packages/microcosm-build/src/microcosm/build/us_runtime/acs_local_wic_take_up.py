"""WIC take-up for the ACS rows of the retained ACS local lane (microcosm#1154).

The national draws ``takes_up_wic_if_eligible`` once per person in the
``wic_claim_input`` stage (:mod:`~microcosm.build.us_runtime.wic_claim`): a
stable seeded Bernoulli at USDA FNS's CY2022 coverage rate for the person's
WIC category. The ACS transfer does not carry the flag and no later stage drew
it for ACS rows, so the engine pass filled the engine default, ``True``, and
every eligible ACS person took WIC up (microcosm#1155). This module carries
PolicyEngine/microcosm#1156's repair onto the current lane:

- :func:`with_acs_local_wic_take_up` runs the national generator, unchanged,
  on the ACS-only frame after the transfer has completed the inputs it reads
  (``is_pregnant`` and ``own_children_in_household``) and before the frame is
  pooled with the donor. The loader's source identity (``source_year``,
  ``source_household_id``, ``source_person_id``) still keys each person's draw
  there. Categories, rates and draws stay the generator's.
- :func:`require_complete_us_wic_take_up` refuses an absent, partly missing or
  non-boolean column. The pool builder holds the donor to it before it loads
  any ACS record, since donor decisions are kept and never redrawn, and holds
  the pooled frame to it after assembly. The release tool holds the staging
  frame to it, refuses a null cell before any reviewed fill, and grades both
  spines on the packaged bytes
  (:func:`~microcosm.build.us_runtime.acs_local_take_up.acs_local_take_up_signal_gate`).

The rates are shares of the eligible in each category, so the ACS rows draw at
the national's rates with no adjustment for the pooled record count.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from microcosm.build.us_runtime.wic_claim import (
    _SOURCE_IDENTITY_COLUMNS,
    US_WIC_CLAIM_OUTPUT_COLUMNS,
    WIC_CLAIM_FNS_SOURCE_URL,
    us_wic_claim_summary,
    with_us_wic_claim_input,
)
from microcosm.frame import Frame
from microcosm.frame.units import US_SCHEMA

__all__ = [
    "ACS_LOCAL_WIC_TAKE_UP_ISSUE",
    "US_WIC_TAKE_UP_COLUMN",
    "require_complete_us_wic_take_up",
    "with_acs_local_wic_take_up",
]

ACS_LOCAL_WIC_TAKE_UP_ISSUE = "microcosm#1154"
(US_WIC_TAKE_UP_COLUMN,) = US_WIC_CLAIM_OUTPUT_COLUMNS


def require_complete_us_wic_take_up(person: pd.DataFrame, *, what: str) -> None:
    """Refuse an absent, partly missing or non-boolean WIC take-up column.

    Never fills: a missing decision would reach the engine as its default,
    ``True``, universal WIC take-up.

    Raises:
        ValueError: naming ``what`` (the frame being checked).
    """

    column = US_WIC_TAKE_UP_COLUMN
    if column not in person:
        raise ValueError(
            f"{what}: WIC take-up requires person column {column!r} "
            f"({ACS_LOCAL_WIC_TAKE_UP_ISSUE}); without it the engine default "
            "makes every eligible person take WIC up."
        )
    missing = int(person[column].isna().sum())
    if missing:
        raise ValueError(
            f"{what}: WIC take-up {column!r} is missing on {missing} of "
            f"{len(person)} person rows ({ACS_LOCAL_WIC_TAKE_UP_ISSUE}); re-run "
            "staging with the current builder, which draws the ACS rows' "
            "decisions."
        )
    if not pd.api.types.is_bool_dtype(person[column].dtype):
        raise ValueError(
            f"{what}: WIC take-up {column!r} must be boolean, not "
            f"{person[column].dtype} ({ACS_LOCAL_WIC_TAKE_UP_ISSUE})."
        )


def with_acs_local_wic_take_up(
    frame: Frame, *, seed: int, time_period: int
) -> tuple[Frame, dict[str, object]]:
    """Draw WIC take-up for an ACS-only frame with the national's generator.

    Args:
        frame: The transferred ACS frame, before it is pooled with the donor.
            Every person must carry the loader's source identity, from
            ``time_period``, and no WIC decision yet.
        seed: The build seed, the one the transfer uses.
        time_period: The ACS vintage, which is also every person's
            ``source_year``.

    Returns:
        The frame with a complete boolean ``takes_up_wic_if_eligible`` and a
        JSON-ready receipt: the seed, the source, the weighted take-up share
        overall and per WIC category, and the category counts.

    Raises:
        ValueError: If the frame is not US-schema, already carries a decision,
            lacks a complete source identity, or holds a person whose
            ``source_year`` is not ``time_period`` (a donor row).
    """

    if frame.schema != US_SCHEMA:
        raise ValueError("ACS WIC take-up requires the US schema.")
    person = frame.table("person")
    column = US_WIC_TAKE_UP_COLUMN
    if column in person and person[column].notna().any():
        raise ValueError(
            f"ACS WIC take-up: the ACS frame already carries {column!r}; the "
            "transfer must not, and donor decisions are never redrawn."
        )
    missing = [name for name in _SOURCE_IDENTITY_COLUMNS if name not in person]
    if missing:
        raise ValueError(
            f"ACS WIC take-up requires the loader's source identity: {missing}."
        )
    if person.loc[:, list(_SOURCE_IDENTITY_COLUMNS)].isna().any(axis=None):
        raise ValueError(
            "ACS WIC take-up requires a complete source identity on every person."
        )
    years = pd.to_numeric(person["source_year"], errors="coerce").to_numpy()
    if not np.all(years == int(time_period)):
        raise ValueError(
            f"ACS WIC take-up draws ACS rows only: every source_year must be "
            f"{int(time_period)}."
        )
    # The generator writes the whole column when any cell is missing, so an
    # all-null column from the pooled schema is simply replaced.
    result = with_us_wic_claim_input(
        frame, seed=int(seed), time_period=int(time_period)
    )
    require_complete_us_wic_take_up(result.table("person"), what="ACS WIC take-up")
    summary = us_wic_claim_summary(result)
    receipt: dict[str, object] = {
        "issue": ACS_LOCAL_WIC_TAKE_UP_ISSUE,
        "design": "microcosm#1156",
        "seed": int(seed),
        "source_year": int(time_period),
        "person_rows": int(len(result.table("person"))),
        "output_columns": list(US_WIC_CLAIM_OUTPUT_COLUMNS),
        "category_rates_source": WIC_CLAIM_FNS_SOURCE_URL,
        "category_rates": summary["category_rates"],
        "weighted_take_up_share": summary["weighted_claim_share"],
        "category_counts": summary["category_counts"],
        "category_weighted_take_up_shares": summary["category_weighted_claim_shares"],
    }
    return result, receipt
