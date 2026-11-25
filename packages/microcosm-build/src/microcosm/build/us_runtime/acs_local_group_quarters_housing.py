"""Housing take-up of ACS group-quarters units (ACS local lane, microcosm#975).

``takes_up_housing_assistance_if_eligible`` is a target of the shared
declared transfer plan, which has no group-quarters carve-out, so ACS SPM
units in group quarters (household ``TYPEHUGQ`` 2, institutional, and 3,
noninstitutional) drew the flag from housing-unit donors although they have no
housing unit (microcosm#975). The release tool already writes
``receives_housing_assistance`` False there; the take-up flag was the one
housing-receipt signal those units still carried.

This lane-own step sets the flag False on every ACS group-quarters SPM unit,
on the ACS-only frame after the shared transfer and before pooling, the place
of the native receipt anchors. It differs from microcosm#975's proposed
design, a structural policy inside the shared transfer (``acs_transfer``):
that would move the shared execution contract, the attested
``imputation.yaml`` entry and the pool lane's pins. The shared plan is
unchanged here, so the flag is still drawn (and its draw recorded) before the
step overrides it, and housing-unit SPM units keep their draw.

The step records a receipt (group-quarters units by ``TYPEHUGQ`` and how many
it turned from True to False) and grades nothing. It refuses only a frame
without the columns it reads; an SPM unit whose household carries no
``TYPEHUGQ`` 1, 2 or 3 keeps its draw and is counted.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.us_runtime.acs_pums import ACS_2024_1YR_SPINE
from microcosm.frame import Frame
from microcosm.frame.units import US_SCHEMA

__all__ = [
    "ACS_GROUP_QUARTERS_KINDS",
    "ACS_LOCAL_GROUP_QUARTERS_HOUSING_COLUMN",
    "ACS_LOCAL_GROUP_QUARTERS_HOUSING_ISSUE",
    "ACS_LOCAL_GROUP_QUARTERS_HOUSING_METHOD",
    "with_acs_local_group_quarters_housing",
]

ACS_LOCAL_GROUP_QUARTERS_HOUSING_ISSUE = "microcosm#975"
ACS_LOCAL_GROUP_QUARTERS_HOUSING_COLUMN = "takes_up_housing_assistance_if_eligible"
ACS_LOCAL_GROUP_QUARTERS_HOUSING_METHOD = (
    "lane_own_step_after_shared_transfer_group_quarters_false"
)
#: ACS ``TYPEHUGQ``: 1 housing unit, 2 institutional and 3 noninstitutional
#: group quarters.
ACS_GROUP_QUARTERS_KINDS: tuple[int, ...] = (2, 3)
_HOUSEHOLD_KINDS = (1, *ACS_GROUP_QUARTERS_KINDS)
_KIND = "TYPEHUGQ"
_COLUMN = ACS_LOCAL_GROUP_QUARTERS_HOUSING_COLUMN


def _unit_kinds(frame: Frame) -> np.ndarray:
    """``TYPEHUGQ`` of each SPM unit's household, aligned to the spm_unit table."""

    household = frame.table("household")
    person = frame.table("person")
    spm_unit = frame.table("spm_unit")
    missing = [
        f"household.{column}"
        for column in ("household_id", _KIND)
        if column not in household
    ] + [
        f"person.{column}"
        for column in ("person_spm_unit_id", "person_household_id")
        if column not in person
    ]
    if "spm_unit_id" not in spm_unit:
        missing.append("spm_unit.spm_unit_id")
    if missing:
        raise ValueError(
            f"ACS group-quarters housing take-up requires {missing} "
            f"({ACS_LOCAL_GROUP_QUARTERS_HOUSING_ISSUE})."
        )
    unit_household = person.drop_duplicates("person_spm_unit_id").set_index(
        "person_spm_unit_id"
    )["person_household_id"]
    kinds = pd.to_numeric(household.set_index("household_id")[_KIND], errors="coerce")
    return pd.to_numeric(
        spm_unit["spm_unit_id"].map(unit_household).map(kinds), errors="coerce"
    ).to_numpy(dtype=np.float64)


def with_acs_local_group_quarters_housing(
    frame: Frame,
) -> tuple[Frame, dict[str, Any]]:
    """Set the housing take-up flag False on ACS group-quarters SPM units.

    Args:
        frame: The ACS-only frame after the shared transfer.

    Returns:
        The frame (with the flag False in group quarters) and a JSON-ready
        receipt.

    Raises:
        ValueError: If the frame is not US-schema or a column it reads is
            absent.
    """

    if frame.schema != US_SCHEMA:
        raise ValueError("ACS group-quarters housing take-up requires the US schema.")
    spm_unit = frame.table("spm_unit")
    if _COLUMN not in spm_unit:
        raise ValueError(
            f"ACS group-quarters housing take-up requires spm_unit.{_COLUMN}; the "
            "shared transfer draws it "
            f"({ACS_LOCAL_GROUP_QUARTERS_HOUSING_ISSUE})."
        )
    kinds = _unit_kinds(frame)
    unknown = ~np.isin(kinds, _HOUSEHOLD_KINDS)
    group_quarters = np.isin(kinds, ACS_GROUP_QUARTERS_KINDS)
    before = spm_unit[_COLUMN]
    was_true = (before.notna() & before.astype(object).eq(True)).to_numpy(dtype=bool)
    was_missing = before.isna().to_numpy(dtype=bool)
    updated = spm_unit.copy()
    values = before.astype(object).copy()
    values.iloc[np.flatnonzero(group_quarters)] = False
    updated[_COLUMN] = values.astype(bool) if values.notna().all() else values
    tables = {
        entity: updated if entity == "spm_unit" else frame.table(entity)
        for entity in frame.entities
    }
    tables.update({name: frame.link(name) for name in frame.links})
    result = Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )
    by_kind = {
        str(kind): {
            "spm_units": int((kinds == kind).sum()),
            "set_false_from_true": int(((kinds == kind) & was_true).sum()),
            "set_false_from_missing": int(((kinds == kind) & was_missing).sum()),
        }
        for kind in ACS_GROUP_QUARTERS_KINDS
    }
    housing_units = kinds == 1
    receipt: dict[str, Any] = {
        "issue": ACS_LOCAL_GROUP_QUARTERS_HOUSING_ISSUE,
        "column": _COLUMN,
        "method": ACS_LOCAL_GROUP_QUARTERS_HOUSING_METHOD,
        "spine": ACS_2024_1YR_SPINE,
        "rule": (
            f"{_COLUMN} = False on every SPM unit whose household has TYPEHUGQ 2 "
            "or 3; housing units (TYPEHUGQ 1) keep the shared transfer's draw"
        ),
        "acs_spm_units": int(len(spm_unit)),
        "group_quarters_spm_units": int(group_quarters.sum()),
        "by_typehugq": by_kind,
        "housing_unit_spm_units": int(housing_units.sum()),
        "housing_unit_take_up_true": int((housing_units & was_true).sum()),
        "unknown_typehugq_spm_units": int(unknown.sum()),
    }
    return result, receipt
