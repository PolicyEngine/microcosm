"""Invented release-shaped frames for the SPM measurement release gate tests.

Every fixture here is invented. Nothing in this module opens a build artifact,
a pinned source, or a release.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from microcosm.build.us_runtime.spm_role_source import NATIVE_SPM_ROLE
from microcosm.build.us_runtime.spm_universe_source import (
    ACS_HOUSEHOLD_KIND_COLUMN,
    ASEC_NATIVE_UNIT_COLUMN,
    HOUSEHOLD_SPINE_COLUMN,
    INCLUDED,
    OUTSIDE,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

__all__ = [
    "ACS",
    "ASEC",
    "Household",
    "expected_unclassified",
    "release_frame",
]

ACS = "acs_2024_1yr"
ASEC = "asec_puf"
#: Unit ids far from every count the gate reports, so a test can prove no id
#: leaks into its output.
_UNIT_ID_OFFSET = 987_000


@dataclass(frozen=True)
class Household:
    """One invented household, which is one SPM unit.

    ``kind`` is the ACS ``TYPEHUGQ`` (1 housing unit; 2 or 3 group quarters)
    and is ignored on the ASEC spine. ``members`` are ``(age, role)`` pairs.
    """

    members: Sequence[tuple[float, bool]]
    kind: int = 1
    spine: str = ACS
    weight: float = 100.0
    extra: dict = field(default_factory=dict)

    @property
    def universe(self) -> str:
        """The source declaration the gate's universe producer must reach."""

        return OUTSIDE if self.spine == ACS and self.kind in (2, 3) else INCLUDED


def release_frame(
    households: Sequence[Household],
    *,
    role: bool = True,
    declare: bool | str = False,
    person_order: Sequence[int] | None = None,
) -> Frame:
    """A release-shaped US frame: one SPM unit per household.

    ``role=False`` stores no role column. ``declare=True`` stores the source
    declaration on the SPM unit table; a string stores that value instead,
    for every unit. ``person_order`` permutes the person rows.
    """

    rows = []
    for index, household in enumerate(households, start=1):
        unit = _UNIT_ID_OFFSET + index
        for age, member_role in household.members:
            row = {
                "person_household_id": index,
                "person_tax_unit_id": index,
                "person_spm_unit_id": unit,
                "person_family_id": index,
                "age": float(age),
                "person_support_clone_index": 0,
                ASEC_NATIVE_UNIT_COLUMN: (
                    float(unit) if household.spine == ASEC else np.nan
                ),
            }
            if role:
                row[NATIVE_SPM_ROLE] = bool(member_role)
            rows.append(row)
    person = pd.DataFrame(rows)
    person.insert(0, "person_id", np.arange(1, len(person) + 1, dtype=np.int64))
    person["person_marital_unit_id"] = person["person_id"]
    if person_order is not None:
        person = person.iloc[list(person_order)].reset_index(drop=True)
    ids = np.arange(1, len(households) + 1, dtype=np.int64)
    household_table = pd.DataFrame(
        {
            "household_id": ids,
            HOUSEHOLD_SPINE_COLUMN: [household.spine for household in households],
            ACS_HOUSEHOLD_KIND_COLUMN: pd.array(
                [
                    household.kind if household.spine == ACS else None
                    for household in households
                ],
                dtype=object,
            ),
        }
    )
    unit_ids = _UNIT_ID_OFFSET + ids
    spm_unit = pd.DataFrame({"spm_unit_id": unit_ids})
    if declare is True:
        spm_unit["spm_unit_spm_universe_status"] = [
            household.universe for household in households
        ]
    elif isinstance(declare, str):
        spm_unit["spm_unit_spm_universe_status"] = declare
    return Frame(
        {
            "person": person,
            "household": household_table,
            "tax_unit": pd.DataFrame({"tax_unit_id": ids}),
            "spm_unit": spm_unit,
            "family": pd.DataFrame({"family_id": ids}),
            "marital_unit": pd.DataFrame(
                {"marital_unit_id": np.sort(person["person_marital_unit_id"])}
            ),
        },
        US_SCHEMA,
        {
            "household": Weights(
                np.asarray([household.weight for household in households]),
                WeightKind.CALIBRATED,
            )
        },
    )


def expected_unclassified(households: Sequence[Household]) -> list[bool]:
    """The adult rule, restated: ``age >= 18 or (age >= 15 and role)``."""

    return [
        not any(age >= 18 or (age >= 15 and role) for age, role in household.members)
        for household in households
    ]
