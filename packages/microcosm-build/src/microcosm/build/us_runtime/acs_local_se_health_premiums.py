"""Self-employed health insurance inputs of ACS rows (ACS local lane).

``is_self_employed`` and ``health_insurance_premiums`` reached the ACS
local-area release missing on every ACS row, and the reviewed-null fill gave
them the engine defaults, ``False`` and 0 (microcosm#1720). No ACS person
could then take the self-employed health insurance deduction, so AGI and MAGI
were too high for self-employed people who buy their own coverage.

This fresh-build stage runs the donor's own rule,
:func:`~microcosm.build.us_runtime.other_health_insurance.attribute_us_se_health_premiums`,
unchanged, on the ACS-only frame after the shared transfer:

- **The flag** marks every person with strictly positive combined Schedule C
  income (``self_employment_income_before_lsr`` +
  ``sstb_self_employment_income_before_lsr``), the donor's rule. It is not ACS
  ``COW`` 6 or 7; the receipt records how often the two agree.
- **The premium** copies the transferred
  ``health_insurance_premiums_without_medicare_part_b`` for a flagged person
  outside the Medicare proxy (age 65+ or any Social Security disability) and
  without employer-sponsored coverage (``has_esi``), and is 0 elsewhere.
- **Inputs.** The self-employment total is native ACS ``SEMP``, blank under
  15, which reads as 0 here, the reviewed fill's value. The SSTB component, the
  premium and Social Security disability come from the shared transfer, whose
  post-transfer structure (``_apply_post_transfer_structure``) has already
  zeroed the self-employment components where ``SEMP`` stands, so the stage
  must run after it. ``has_esi`` is ACS ``HINS1`` on the local lane: the
  coverage step (:mod:`~.acs_local_coverage_anchors`) writes it over the
  transferred draw first.
- **Null cells only.** A stored ACS value stands; donor rows are never seen.

With this stage the ACS rows' MAGI is final before the release tool assigns
Medicaid (and any later ACA) take-up. The national build applies the
attribution in another order (microcosm#1689). The stage records a receipt
and refuses only inputs the rule cannot read.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.source_runtime import SourceRuntimeError
from microcosm.build.us_runtime.acs_pums import ACS_2024_1YR_SPINE
from microcosm.build.us_runtime.other_health_insurance import (
    US_SE_HEALTH_ATTRIBUTION_OUTPUT_COLUMNS,
    US_SE_HEALTH_MEDICARE_AGE_THRESHOLD,
    US_SE_HEALTH_SELF_EMPLOYMENT_INCOME_SOURCES,
    attribute_us_se_health_premiums,
)
from microcosm.frame import Frame
from microcosm.frame.units import US_SCHEMA

__all__ = [
    "ACS_LOCAL_SE_HEALTH_PREMIUM_COLUMNS",
    "ACS_LOCAL_SE_HEALTH_PREMIUMS_ISSUE",
    "ACS_LOCAL_SE_HEALTH_PREMIUMS_METHOD",
    "ACS_LOCAL_SE_HEALTH_PREMIUMS_SOURCE_COLUMNS",
    "with_acs_local_se_health_premiums",
]

ACS_LOCAL_SE_HEALTH_PREMIUMS_ISSUE = "microcosm#1720"
ACS_LOCAL_SE_HEALTH_PREMIUMS_METHOD = (
    "donor_attribute_us_se_health_premiums_after_shared_transfer_missing_cells_only"
)
#: ``("health_insurance_premiums", "is_self_employed")``.
ACS_LOCAL_SE_HEALTH_PREMIUM_COLUMNS: tuple[str, ...] = (
    US_SE_HEALTH_ATTRIBUTION_OUTPUT_COLUMNS
)
_PREMIUMS, _FLAG = ACS_LOCAL_SE_HEALTH_PREMIUM_COLUMNS
_SELF_EMPLOYMENT, _SSTB = US_SE_HEALTH_SELF_EMPLOYMENT_INCOME_SOURCES
_REPORTED = "health_insurance_premiums_without_medicare_part_b"
_AGE = "age"
_SSDI = "social_security_disability"
_ESI = "has_esi"
#: The person columns the donor's rule reads, in its own names.
ACS_LOCAL_SE_HEALTH_PREMIUMS_SOURCE_COLUMNS: tuple[str, ...] = (
    _REPORTED,
    _SELF_EMPLOYMENT,
    _SSTB,
    _AGE,
    _SSDI,
    _ESI,
)
#: ACS asks SEMP from age 15; below it the Census blank is the universe.
_SELF_EMPLOYMENT_MINIMUM_AGE = 15
_CLASS_OF_WORKER = "COW"
#: ACS COW 6 (self-employed, not incorporated) and 7 (incorporated).
_SELF_EMPLOYED_COW_CODES = (6, 7)
_RULE = (
    "is_self_employed: self_employment_income_before_lsr + "
    "sstb_self_employment_income_before_lsr > 0; health_insurance_premiums: "
    "health_insurance_premiums_without_medicare_part_b where flagged, under "
    f"{US_SE_HEALTH_MEDICARE_AGE_THRESHOLD}, with no Social Security disability "
    "and no has_esi, else 0 (other_health_insurance."
    "attribute_us_se_health_premiums, unchanged)"
)


def _weighted_share(weights: np.ndarray, flags: np.ndarray) -> float:
    total = float(weights.sum())
    return float(weights[flags].sum()) / total if total > 0 else 0.0


def _boolean_view(values: pd.Series, column: str) -> np.ndarray:
    if values.isna().any() or not values.isin([0, 1]).all():
        raise ValueError(
            f"ACS {column} must be a complete boolean after the shared transfer "
            f"({ACS_LOCAL_SE_HEALTH_PREMIUMS_ISSUE})."
        )
    return values.astype(bool).to_numpy(dtype=bool)


def _rule_view(person: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """The donor rule's inputs on ACS rows; the SEMP blank under 15 reads 0.

    Returns the view and how many under-15 blanks read as 0.

    Raises:
        ValueError: If a column is absent, an age is missing, a blank sits in
            SEMP's universe (15+) or in a transferred amount, or ``has_esi``
            is not a complete boolean.
    """

    missing = [
        column
        for column in ACS_LOCAL_SE_HEALTH_PREMIUMS_SOURCE_COLUMNS
        if column not in person
    ]
    if missing:
        raise ValueError(
            f"ACS self-employed health premiums require person column(s) {missing}; "
            "they come from the native mapping and the shared transfer "
            f"({ACS_LOCAL_SE_HEALTH_PREMIUMS_ISSUE})."
        )
    age = pd.to_numeric(person[_AGE], errors="coerce")
    if age.isna().any():
        raise ValueError("ACS self-employed health premiums require every age.")
    view = pd.DataFrame(index=person.index)
    view[_AGE] = age.to_numpy(dtype=np.float64)
    self_employment = pd.to_numeric(person[_SELF_EMPLOYMENT], errors="coerce")
    universe_blank = self_employment.isna() & age.ge(_SELF_EMPLOYMENT_MINIMUM_AGE)
    if universe_blank.any():
        raise ValueError(
            f"{int(universe_blank.sum())} ACS person(s) aged "
            f"{_SELF_EMPLOYMENT_MINIMUM_AGE}+ have a blank {_SELF_EMPLOYMENT}."
        )
    under_age_blanks = int(self_employment.isna().sum())
    view[_SELF_EMPLOYMENT] = self_employment.fillna(0.0).to_numpy(dtype=np.float64)
    for column in (_SSTB, _REPORTED, _SSDI):
        values = pd.to_numeric(person[column], errors="coerce")
        if values.isna().any():
            raise ValueError(
                f"{int(values.isna().sum())} ACS person(s) have a blank "
                f"transferred {column}; the shared transfer must fill every ACS "
                "row before this stage runs."
            )
        view[column] = values.to_numpy(dtype=np.float64)
    view[_ESI] = _boolean_view(person[_ESI], _ESI)
    return view, under_age_blanks


def _fill_missing(
    person: pd.DataFrame, column: str, values: np.ndarray
) -> tuple[int, int]:
    """Write ``values`` into the missing cells of ``column``; keep the rest."""

    if column not in person:
        person[column] = values
        return len(values), 0
    present = person[column].notna().to_numpy(dtype=bool)
    merged = (
        person[column]
        .astype(object)
        .where(present, pd.Series(values, index=person.index))
    )
    person[column] = merged.astype(values.dtype)
    return int((~present).sum()), int(present.sum())


def _cow_agreement(person: pd.DataFrame, flag: np.ndarray) -> dict[str, Any]:
    if _CLASS_OF_WORKER not in person:
        return {"loaded": False}
    cow = pd.to_numeric(person[_CLASS_OF_WORKER], errors="coerce")
    self_employed_cow = cow.isin(_SELF_EMPLOYED_COW_CODES).to_numpy(dtype=bool)
    return {
        "loaded": True,
        "cow_codes": list(_SELF_EMPLOYED_COW_CODES),
        "both_rows": int((flag & self_employed_cow).sum()),
        "flag_only_rows": int((flag & ~self_employed_cow).sum()),
        "cow_only_rows": int((~flag & self_employed_cow).sum()),
        "cow_blank_rows": int(cow.isna().sum()),
    }


def with_acs_local_se_health_premiums(
    frame: Frame,
) -> tuple[Frame, dict[str, Any]]:
    """Fill missing ACS ``is_self_employed`` and ``health_insurance_premiums``.

    Args:
        frame: The ACS-only frame after the shared transfer (and its
            post-transfer structure).

    Returns:
        The frame with filled cells and a JSON-ready receipt: the rule, its
        source columns, filled and preserved rows, flagged rows, attributed
        premiums, the rows the Medicare proxy and employer coverage withheld,
        and the agreement of the flag with ACS COW 6 or 7.

    Raises:
        ValueError: If the frame is not US-schema or the rule cannot read an
            input (see :func:`_rule_view`).
    """

    if frame.schema != US_SCHEMA:
        raise ValueError("ACS self-employed health premiums require the US schema.")
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables.update({name: frame.link(name) for name in frame.links})
    person = tables["person"]
    view, under_age_blanks = _rule_view(person)
    try:
        attributed = attribute_us_se_health_premiums(view)
    except SourceRuntimeError as exc:
        raise ValueError(
            f"ACS self-employed health premiums: {exc} "
            f"({ACS_LOCAL_SE_HEALTH_PREMIUMS_ISSUE})"
        ) from exc
    derived = {
        _PREMIUMS: attributed[_PREMIUMS].to_numpy(dtype=np.float64),
        _FLAG: attributed[_FLAG].to_numpy(dtype=bool),
    }
    counts = {
        column: _fill_missing(person, column, derived[column])
        for column in ACS_LOCAL_SE_HEALTH_PREMIUM_COLUMNS
    }
    flag = person[_FLAG].astype(bool).to_numpy(dtype=bool)
    premiums = pd.to_numeric(person[_PREMIUMS], errors="coerce").to_numpy(
        dtype=np.float64
    )
    positive = premiums > 0
    reported = view[_REPORTED].to_numpy(dtype=np.float64) > 0
    medicare_proxy = (view[_AGE].to_numpy() >= US_SE_HEALTH_MEDICARE_AGE_THRESHOLD) | (
        view[_SSDI].to_numpy() > 0
    )
    esi = view[_ESI].to_numpy(dtype=bool)
    weights = np.asarray(frame.resolve_weights("person").values, dtype=np.float64)
    positive_weight = float(weights[positive].sum())
    receipt: dict[str, Any] = {
        "issue": ACS_LOCAL_SE_HEALTH_PREMIUMS_ISSUE,
        "method": ACS_LOCAL_SE_HEALTH_PREMIUMS_METHOD,
        "spine": ACS_2024_1YR_SPINE,
        "rule": _RULE,
        "source_columns": list(ACS_LOCAL_SE_HEALTH_PREMIUMS_SOURCE_COLUMNS),
        "acs_persons": len(person),
        "self_employment_blank_under_15_read_as_zero_rows": under_age_blanks,
        "columns": {
            column: {"filled_rows": filled, "preserved_rows": preserved}
            for column, (filled, preserved) in counts.items()
        },
        "self_employed_rows": int(flag.sum()),
        "self_employed_weighted_share": _weighted_share(weights, flag),
        "premium_positive_rows": int(positive.sum()),
        "premium_positive_weighted_share": _weighted_share(weights, positive),
        "premium_weighted_total": float((weights * np.nan_to_num(premiums)).sum()),
        "premium_weighted_mean_where_positive": (
            float((weights[positive] * premiums[positive]).sum()) / positive_weight
            if positive_weight > 0
            else 0.0
        ),
        "withheld": {
            "self_employed_with_reported_premium_rows": int((flag & reported).sum()),
            "medicare_proxy_rows": int((flag & reported & medicare_proxy).sum()),
            "employer_coverage_rows": int(
                (flag & reported & ~medicare_proxy & esi).sum()
            ),
        },
        "cow_agreement": _cow_agreement(person, flag),
    }
    result = Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )
    return result, _json_ready(receipt)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value
