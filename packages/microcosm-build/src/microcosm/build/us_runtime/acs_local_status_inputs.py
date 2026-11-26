"""Marital, military and occupation status of ACS rows (ACS local lane).

Seven person inputs reached the ACS local-area release missing on every ACS
row, and the reviewed-null fill gave them the engine default, ``False``
(microcosm#1720), although the ACS asks the items they come from:

- **is_separated** and **is_surviving_spouse** follow the donor's rule
  (:func:`~microcosm.build.us_runtime.relationship_inputs.us_marital_status_inputs`)
  on the ACS rows' ``A_MARITL``, which the loader maps from ACS ``MAR``
  (``_ACS_TO_CPS_MARITAL_STATUS``: MAR 4, separated, is code 6; MAR 2,
  widowed, is code 4). As on the donor, ``is_surviving_spouse`` is
  "widowed", wider than the tax status, which needs a spouse's death in the
  two prior years.
- **is_military** is ACS ``MIL`` == 1, "now on active duty" (loaded as
  ``ACS_MIL``, microcosm#1020), the item microcosm#1720 names. ``MIL`` is
  blank under 17, which reads as not on active duty. The donor's rule is
  ``POCCU2`` == 52 (a military occupation on the longest job last year), which
  on ACS rows would also mark a veteran whose last job, up to five years ago,
  was military. The receipt records how often the two rules, and ACS
  ``ESR`` 4 or 5 (Armed Forces), agree on the ACS rows.
- **has_never_worked**, **is_computer_scientist**, **is_farmer_fisher** and
  **is_executive_administrative_professional** follow the donor's rule
  (:func:`~microcosm.build.us_runtime.org_wages.us_flsa_occupation_flags`) on
  the ACS rows' native ``detailed_occupation_recode``, the ``POCCU2`` code of
  ``OCCP`` (``ACS_OCCP_TO_POCCU2``); a blank ``OCCP`` from age 16 is code 53,
  so ``has_never_worked`` there means no job in the past five years.

The stage runs on the ACS-only frame after the shared native mapping and
before any transfer; no transfer plan names these columns. It fills only
missing cells, so a stored ACS value stands, and it never sees a donor row.
It records a receipt and refuses nothing beyond malformed source codes.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.us_runtime.acs_inputs import _nullable_source_codes
from microcosm.build.us_runtime.acs_pums import (
    ACS_2024_1YR_SPINE,
    ACS_SOURCE_COLUMN_RENAMES,
)
from microcosm.build.us_runtime.org_wages import (
    FLSA_OVERTIME_OCCUPATION_CODES,
    us_flsa_occupation_flags,
)
from microcosm.build.us_runtime.relationship_inputs import (
    US_MARITAL_STATUS_INPUT_A_MARITL_CODES,
    us_marital_status_inputs,
)
from microcosm.frame import Frame
from microcosm.frame.units import US_SCHEMA

__all__ = [
    "ACS_LOCAL_MARITAL_STATUS_COLUMNS",
    "ACS_LOCAL_MILITARY_COLUMN",
    "ACS_LOCAL_OCCUPATION_FLAG_COLUMNS",
    "ACS_LOCAL_STATUS_INPUTS_COLUMNS",
    "ACS_LOCAL_STATUS_INPUTS_ISSUE",
    "ACS_LOCAL_STATUS_INPUTS_METHOD",
    "AcsLocalStatusInputsResult",
    "map_acs_local_status_inputs",
]

ACS_LOCAL_STATUS_INPUTS_ISSUE = "microcosm#1720"
ACS_LOCAL_STATUS_INPUTS_METHOD = "native_acs_items_donor_rules_missing_cells_only"
ACS_LOCAL_MARITAL_STATUS_COLUMNS: tuple[str, ...] = tuple(
    US_MARITAL_STATUS_INPUT_A_MARITL_CODES
)
ACS_LOCAL_MILITARY_COLUMN = "is_military"
#: The donor's POCCU2 flags except ``is_military``, which ACS MIL measures.
ACS_LOCAL_OCCUPATION_FLAG_COLUMNS: tuple[str, ...] = (
    *(
        column
        for column in FLSA_OVERTIME_OCCUPATION_CODES
        if column != ACS_LOCAL_MILITARY_COLUMN
    ),
    "is_executive_administrative_professional",
)
#: Every column this stage writes on ACS rows.
ACS_LOCAL_STATUS_INPUTS_COLUMNS: tuple[str, ...] = (
    *ACS_LOCAL_MARITAL_STATUS_COLUMNS,
    ACS_LOCAL_MILITARY_COLUMN,
    *ACS_LOCAL_OCCUPATION_FLAG_COLUMNS,
)

_NATIVE_PROVENANCE = "acs_2024_1yr_native"
_MARITAL_SOURCE = "A_MARITL"
_MILITARY_SOURCE = ACS_SOURCE_COLUMN_RENAMES["MIL"]
_OCCUPATION_SOURCE = "detailed_occupation_recode"
_EMPLOYMENT_STATUS_SOURCE = "ESR"
_AGE_SOURCE = "AGEP"
#: ACS MIL is asked from age 17; below it the Census code is blank.
_MILITARY_MINIMUM_AGE = 17
_ACTIVE_DUTY_CODE = 1
#: ACS ESR 4 (Armed Forces, at work) and 5 (with a job but not at work).
_ARMED_FORCES_EMPLOYMENT_CODES = (4, 5)
_DONOR_MILITARY_OCCUPATION_CODE = FLSA_OVERTIME_OCCUPATION_CODES[
    ACS_LOCAL_MILITARY_COLUMN
]
_RULES: Mapping[str, tuple[str, tuple[str, ...], str]] = {
    **{
        column: (
            f"A_MARITL == {code} (the donor's relationship_inputs rule; the "
            f"loader maps ACS MAR {4 if code == 6 else 2} to it)",
            (_MARITAL_SOURCE, "MAR"),
            "relationship_inputs: A_MARITL == " + str(code),
        )
        for column, code in US_MARITAL_STATUS_INPUT_A_MARITL_CODES.items()
    },
    ACS_LOCAL_MILITARY_COLUMN: (
        "ACS MIL == 1 (now on active duty); blank under 17 is False",
        (_MILITARY_SOURCE, _AGE_SOURCE),
        f"org_wages: POCCU2 == {_DONOR_MILITARY_OCCUPATION_CODE}",
    ),
    **{
        column: (
            "the donor's org_wages FLSA rule on detailed_occupation_recode "
            "(OCCP through ACS_OCCP_TO_POCCU2)",
            (_OCCUPATION_SOURCE, "OCCP"),
            "org_wages: the same rule on ASEC POCCU2",
        )
        for column in ACS_LOCAL_OCCUPATION_FLAG_COLUMNS
    },
}


@dataclass(frozen=True)
class AcsLocalStatusInputsResult:
    """The mapped ACS frame, its native-input receipts and the stage receipt."""

    frame: Frame
    native_inputs: Mapping[str, Mapping[str, Any]]
    receipt: Mapping[str, Any]


def _weighted_share(weights: np.ndarray, flags: np.ndarray) -> float:
    total = float(weights.sum())
    return float(weights[flags].sum()) / total if total > 0 else 0.0


def _fill_missing(
    person: pd.DataFrame, column: str, values: np.ndarray
) -> tuple[np.ndarray, int, int]:
    """Write ``values`` into the missing cells of ``column``; keep the rest.

    Returns the final flags, the filled count and the preserved count.

    Raises:
        ValueError: If a stored value is not boolean.
    """

    if column not in person:
        person[column] = values.astype(bool)
        return values.astype(bool), len(values), 0
    stored = person[column]
    present = stored.notna().to_numpy(dtype=bool)
    if not stored[present].isin([0, 1]).all():
        raise ValueError(f"Stored ACS {column} values must be boolean.")
    final = np.where(present, stored.fillna(False).astype(bool), values).astype(bool)
    person[column] = final
    return final, int((~present).sum()), int(present.sum())


def _military_agreement(
    military: np.ndarray,
    occupation_military: np.ndarray,
    armed_forces: np.ndarray,
    weights: np.ndarray,
) -> dict[str, Any]:
    def table(other: np.ndarray) -> dict[str, Any]:
        return {
            "both_rows": int((military & other).sum()),
            "mil_only_rows": int((military & ~other).sum()),
            "other_only_rows": int((~military & other).sum()),
            "other_weighted_true_share": _weighted_share(weights, other),
        }

    return {
        f"donor_rule_poccu2_{_DONOR_MILITARY_OCCUPATION_CODE}": table(
            occupation_military
        ),
        "esr_armed_forces_4_or_5": table(armed_forces),
    }


def map_acs_local_status_inputs(frame: Frame) -> AcsLocalStatusInputsResult:
    """Write the marital, military and occupation inputs on ACS rows.

    Runs on the ACS-only frame after
    :func:`~microcosm.build.us_runtime.acs_inputs.map_acs_native_inputs`
    (which writes ``detailed_occupation_recode``) and before any transfer.

    Returns:
        The mapped frame, native-input receipts in the ``acs_inputs`` format,
        and the stage receipt (rules, source columns, filled and true rows,
        weighted true shares and the ``is_military`` agreement tables).

    Raises:
        ValueError: If the frame is not US-schema, a source column is absent,
            an ``A_MARITL``, ``MIL`` or ``ESR`` code is malformed, or a
            stored output value is not boolean.
    """

    if frame.schema != US_SCHEMA:
        raise ValueError("ACS local status inputs require the US schema.")
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables.update({name: frame.link(name) for name in frame.links})
    person = tables["person"]
    required = (
        _MARITAL_SOURCE,
        _MILITARY_SOURCE,
        _OCCUPATION_SOURCE,
        _EMPLOYMENT_STATUS_SOURCE,
        _AGE_SOURCE,
    )
    missing = [column for column in required if column not in person]
    if missing:
        raise ValueError(
            f"ACS local status inputs require person column(s) {missing}; the "
            "pinned ACS PUMS source carries MAR (as A_MARITL), MIL (as "
            "ACS_MIL), OCCP, ESR and AGEP "
            f"({ACS_LOCAL_STATUS_INPUTS_ISSUE})."
        )
    age = pd.to_numeric(person[_AGE_SOURCE], errors="coerce")
    if age.isna().any() or age.lt(0).any():
        raise ValueError("ACS local status inputs require a nonnegative AGEP.")
    marital = pd.to_numeric(person[_MARITAL_SOURCE], errors="coerce")
    derived: dict[str, np.ndarray] = dict(
        us_marital_status_inputs(marital.to_numpy(dtype=np.float64))
    )
    military_codes = _nullable_source_codes(
        person[_MILITARY_SOURCE].rename("MIL"), minimum=1, maximum=4
    )
    derived[ACS_LOCAL_MILITARY_COLUMN] = military_codes.eq(_ACTIVE_DUTY_CODE).to_numpy(
        dtype=bool
    )
    occupation = pd.to_numeric(person[_OCCUPATION_SOURCE], errors="coerce").fillna(0)
    flags = us_flsa_occupation_flags(occupation)
    for column in ACS_LOCAL_OCCUPATION_FLAG_COLUMNS:
        derived[column] = flags[column].to_numpy(dtype=bool)
    employment = _nullable_source_codes(
        person[_EMPLOYMENT_STATUS_SOURCE], minimum=1, maximum=6
    )

    weights = np.asarray(frame.resolve_weights("person").values, dtype=np.float64)
    native: dict[str, Mapping[str, Any]] = {}
    columns: dict[str, Any] = {}
    final: dict[str, np.ndarray] = {}
    for column in ACS_LOCAL_STATUS_INPUTS_COLUMNS:
        values, filled, preserved = _fill_missing(person, column, derived[column])
        final[column] = values
        rule, source_columns, donor_rule = _RULES[column]
        native[column] = {
            "entity": "person",
            "source_columns": list(source_columns),
            "transformation": rule,
            "provenance": _NATIVE_PROVENANCE,
            "observed_rows": len(values),
            "missing_rows": 0,
        }
        columns[column] = {
            "rule": rule,
            "donor_rule": donor_rule,
            "source_columns": list(source_columns),
            "filled_rows": filled,
            "preserved_rows": preserved,
            "true_rows": int(values.sum()),
            "weighted_true_share": _weighted_share(weights, values),
        }

    military = final[ACS_LOCAL_MILITARY_COLUMN]
    blank_military = military_codes.isna().to_numpy(dtype=bool)
    mapped = Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )
    receipt: dict[str, Any] = {
        "issue": ACS_LOCAL_STATUS_INPUTS_ISSUE,
        "method": ACS_LOCAL_STATUS_INPUTS_METHOD,
        "spine": ACS_2024_1YR_SPINE,
        "acs_persons": len(person),
        "columns": columns,
        "a_maritl_rows": {
            str(int(code)): int(count)
            for code, count in marital.value_counts().sort_index().items()
        },
        "military": {
            "mil_code_rows": {
                **{
                    str(int(code)): int(count)
                    for code, count in military_codes.value_counts()
                    .sort_index()
                    .items()
                },
                "blank": int(blank_military.sum()),
            },
            "blank_rows_age_17_plus": int(
                (blank_military & age.ge(_MILITARY_MINIMUM_AGE).to_numpy()).sum()
            ),
            "coded_rows_under_17": int(
                (~blank_military & age.lt(_MILITARY_MINIMUM_AGE).to_numpy()).sum()
            ),
            "agreement": _military_agreement(
                military,
                occupation.eq(_DONOR_MILITARY_OCCUPATION_CODE).to_numpy(dtype=bool),
                employment.isin(_ARMED_FORCES_EMPLOYMENT_CODES).to_numpy(dtype=bool),
                weights,
            ),
        },
    }
    return AcsLocalStatusInputsResult(mapped, native, receipt)
