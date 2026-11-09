"""Invented ACS-shaped frames for the ACS SPM independence role tests.

Every fixture here is invented. Nothing in this module opens a build artifact,
a pinned source, or a release.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from microcosm.build.us_runtime.spm_role_source import NATIVE_SPM_ROLE
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

__all__ = ["OTHER_MEMBER_CODES", "ROLE_CODES", "acs_frame", "role_by_person"]

ROLE_CODES = frozenset({20, 21, 23, 37, 38})
OTHER_MEMBER_CODES = (22, 24, *range(25, 37))


def acs_frame(
    households: list[list[tuple[int, float]]],
    *,
    typehugq: list[int] | None = None,
    spm_units: list[int] | None = None,
    order: list[int] | None = None,
) -> Frame:
    """An ACS-shaped US frame: one ``(RELSHIPP, age)`` list per household."""

    rows = [
        {"household": index + 1, "RELSHIPP": code, "age": float(age)}
        for index, members in enumerate(households)
        for code, age in members
    ]
    person = pd.DataFrame(rows)
    person.insert(0, "person_id", np.arange(1, len(person) + 1, dtype=np.int64))
    if order is not None:
        person = person.iloc[order].reset_index(drop=True)
    household_id = person.pop("household").to_numpy(dtype=np.int64)
    person["person_household_id"] = household_id
    person["person_tax_unit_id"] = household_id
    person["person_spm_unit_id"] = (
        household_id if spm_units is None else np.asarray(spm_units, dtype=np.int64)
    )
    person["person_family_id"] = household_id
    person["person_marital_unit_id"] = person["person_id"]
    ids = np.arange(1, len(households) + 1, dtype=np.int64)
    household = pd.DataFrame({"household_id": ids})
    if typehugq is not None:
        household["TYPEHUGQ"] = typehugq
    spm_ids = np.unique(person["person_spm_unit_id"].to_numpy())
    return Frame(
        {
            "person": person,
            "household": household,
            "tax_unit": pd.DataFrame({"tax_unit_id": ids}),
            "spm_unit": pd.DataFrame({"spm_unit_id": spm_ids}),
            "family": pd.DataFrame({"family_id": ids}),
            "marital_unit": pd.DataFrame(
                {"marital_unit_id": person["person_id"].sort_values().to_numpy()}
            ),
        },
        US_SCHEMA,
        {
            "household": Weights(
                np.arange(10.0, 10.0 * (len(households) + 1), 10.0),
                WeightKind.DESIGN,
            )
        },
    )


def role_by_person(frame: Frame) -> pd.Series:
    person = frame.table("person")
    return pd.Series(
        person[NATIVE_SPM_ROLE].to_numpy(), index=person["person_id"].to_numpy()
    )
