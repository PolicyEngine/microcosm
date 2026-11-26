"""Weeks unemployed and retirement deferrals of ACS rows (ACS local lane).

Four person inputs reached the ACS local-area release missing on every ACS
row, and the reviewed-null fill gave them the engine default, 0
(microcosm#1720). The ACS asks none of them:

- ``weeks_unemployed``: a UC recipient with 0 weeks looks like a recipient
  for no month, so SNAP's work-registration exemption, and with it the ABAWD
  time limit, read wrong.
- ``traditional_401k_contributions_desired``,
  ``roth_401k_contributions_desired`` and ``roth_ira_contributions_desired``:
  ACS ``WAGP`` is gross of deferrals, so with them at 0, AGI and MAGI include
  deferred wages, and no Roth contribution earns the saver's credit.

This fresh-build stage is a separate, local-lane-only QRF pass on the donor's
ASEC observation role, the pattern of the local income pass
(:mod:`~microcosm.build.us_runtime.acs_local_income`). It never edits the
shared declared transfer plan, which the pool lane shares.

- **Donor channel.** The ASEC observation role carries the measured values:
  ``weeks_unemployed`` is ASEC ``LKWEEKS``
  (:mod:`~microcosm.build.us_runtime.weeks_unemployed`), and the deferrals
  are the measured ``RETCB_VAL`` split by the archived shares
  (:mod:`~microcosm.build.us_runtime.retirement_contributions`). The PUF
  clone role carries QRF predictions of both, so fitting on it would transfer
  a model of a model.
- **Place.** It runs after the shared transfer, on the ACS-only frame: the
  weeks family reads the transferred ``unemployment_compensation`` as a
  predictor, which the ACS rows carry only from then on. Every earlier fit
  (hours, income, ESI premium and the shared plan) has drawn before it, and
  each family's seed comes from its own name, so their draws are unchanged.
- **Predictors.** The existing transfer predictors (age, sex and state, plus
  wages, self-employment, Social Security, retirement, interest, household
  head and tenure where both sides observe them). The weeks family adds one
  local predictor extension of this call alone:
  ``unemployment_compensation``, on the donor the measured ASEC ``UC_VAL``
  and on ACS rows the shared transfer's draw.
- **The donor's post-processing.** Weeks are clipped to 0-52, rounded and set
  to 0 wherever ``unemployment_compensation`` is 0, the archived zero rule
  (:func:`~microcosm.build.us_runtime.weeks_unemployed.postprocess_us_weeks_unemployed_predictions`).
  Deferrals are clipped at 0 and the 401(k) deferrals set to 0 without wages
  (:func:`~microcosm.build.us_runtime.retirement_contributions.postprocess_us_retirement_contribution_predictions`).
  Both are the rules the donor applies to its own PUF clone half.
- **No double counting.** The shared plan transfers
  ``traditional_ira_contributions_desired`` and
  ``self_employed_pension_contributions_desired`` (from the PUF role's
  deductions), so this pass refuses to draw any column the shared plan or the
  local income pass draws.
- **Null cells only.** A stored ACS value stands, and the post-processing
  touches only the cells this pass filled.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.gates import FitWeightRecord
from microcosm.build.us_runtime.acs_local_income import (
    ACS_LOCAL_INCOME_TRANSFER_COLUMNS,
)
from microcosm.build.us_runtime.acs_pums import ACS_2024_1YR_SPINE
from microcosm.build.us_runtime.acs_transfer import (
    ASEC_PUF_DONOR_SPINE,
    DEFAULT_ACS_TRANSFER_MAX_TARGETS_PER_FIT,
    AcsImputedInput,
    AcsPersonPredictorExtension,
    TargetFamilies,
    required_acs_transfer_inputs,
    resolve_acs_donor_channel,
    transfer_acs_inputs,
)
from microcosm.build.us_runtime.retirement_contributions import (
    postprocess_us_retirement_contribution_predictions,
)
from microcosm.build.us_runtime.support_provenance import BASE_ASEC_SUPPORT_CHANNEL
from microcosm.build.us_runtime.weeks_unemployed import (
    postprocess_us_weeks_unemployed_predictions,
)
from microcosm.frame import US_SCHEMA, Frame

__all__ = [
    "ACS_LOCAL_RETIREMENT_DEFERRAL_COLUMNS",
    "ACS_LOCAL_RETIREMENT_DEFERRAL_FAMILY",
    "ACS_LOCAL_UNEMPLOYMENT_COMPENSATION_FEATURE",
    "ACS_LOCAL_WEEKS_DEFERRALS_COLUMNS",
    "ACS_LOCAL_WEEKS_DEFERRALS_DONOR_CHANNEL",
    "ACS_LOCAL_WEEKS_DEFERRALS_FAMILIES",
    "ACS_LOCAL_WEEKS_DEFERRALS_ISSUE",
    "ACS_LOCAL_WEEKS_DEFERRALS_METHOD",
    "ACS_LOCAL_WEEKS_DEFERRALS_PREDICTOR_EXTENSIONS",
    "ACS_LOCAL_WEEKS_UNEMPLOYED_COLUMN",
    "ACS_LOCAL_WEEKS_UNEMPLOYED_FAMILY",
    "AcsLocalWeeksDeferralsResult",
    "acs_local_weeks_deferrals_target_families",
    "require_acs_local_weeks_deferrals_donor",
    "with_acs_local_weeks_deferrals",
]

ACS_LOCAL_WEEKS_DEFERRALS_ISSUE = "microcosm#1720"
ACS_LOCAL_WEEKS_DEFERRALS_METHOD = (
    "separate_local_qrf_pass_after_shared_transfer_asec_role_missing_cells_only"
)
ACS_LOCAL_WEEKS_DEFERRALS_DONOR_CHANNEL = BASE_ASEC_SUPPORT_CHANNEL
ACS_LOCAL_WEEKS_UNEMPLOYED_FAMILY = "acs_local_weeks_unemployed"
ACS_LOCAL_RETIREMENT_DEFERRAL_FAMILY = "acs_local_retirement_deferrals"
ACS_LOCAL_WEEKS_UNEMPLOYED_COLUMN = "weeks_unemployed"
ACS_LOCAL_RETIREMENT_DEFERRAL_COLUMNS: tuple[str, ...] = (
    "traditional_401k_contributions_desired",
    "roth_401k_contributions_desired",
    "roth_ira_contributions_desired",
)
#: One chained QRF per family keeps the jointly reported deferrals together.
ACS_LOCAL_WEEKS_DEFERRALS_FAMILIES: Mapping[str, tuple[str, ...]] = {
    ACS_LOCAL_WEEKS_UNEMPLOYED_FAMILY: (ACS_LOCAL_WEEKS_UNEMPLOYED_COLUMN,),
    ACS_LOCAL_RETIREMENT_DEFERRAL_FAMILY: ACS_LOCAL_RETIREMENT_DEFERRAL_COLUMNS,
}
ACS_LOCAL_WEEKS_DEFERRALS_COLUMNS: tuple[str, ...] = tuple(
    column
    for columns in ACS_LOCAL_WEEKS_DEFERRALS_FAMILIES.values()
    for column in columns
)
_UNEMPLOYMENT_COMPENSATION = "unemployment_compensation"
ACS_LOCAL_UNEMPLOYMENT_COMPENSATION_FEATURE = "__acs_transfer_unemployment_compensation"
#: The weeks family's local predictor: the measured ASEC UC on the donor, the
#: shared transfer's UC on ACS rows.
ACS_LOCAL_WEEKS_DEFERRALS_PREDICTOR_EXTENSIONS: tuple[
    AcsPersonPredictorExtension, ...
] = (
    AcsPersonPredictorExtension(
        feature=ACS_LOCAL_UNEMPLOYMENT_COMPENSATION_FEATURE,
        donor_components=(_UNEMPLOYMENT_COMPENSATION,),
        recipient_source=_UNEMPLOYMENT_COMPENSATION,
        families=(ACS_LOCAL_WEEKS_UNEMPLOYED_FAMILY,),
    ),
)
_WAGES = "employment_income_before_lsr"
_AGE = "age"
#: ACS asks WAGP from age 15; below it the blank reads as no wages.
_WAGES_MINIMUM_AGE = 15


def acs_local_weeks_deferrals_target_families() -> TargetFamilies:
    """The separate ASEC-only pass, disjoint from every other transfer."""

    overlap = sorted(
        set(ACS_LOCAL_WEEKS_DEFERRALS_COLUMNS)
        & (required_acs_transfer_inputs() | set(ACS_LOCAL_INCOME_TRANSFER_COLUMNS))
    )
    if overlap:
        raise ValueError(
            "The ACS local weeks and deferrals pass would transfer column(s) "
            f"another transfer already draws {overlap}; double counting "
            f"({ACS_LOCAL_WEEKS_DEFERRALS_ISSUE})."
        )
    return {"person": dict(ACS_LOCAL_WEEKS_DEFERRALS_FAMILIES)}


def _amounts(rows: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(rows[column], errors="coerce").to_numpy(
        dtype=np.float64, na_value=np.nan
    )


def _summary(values: np.ndarray, weights: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(values)
    positive = finite & (values > 0)
    total = float(weights[finite].sum())
    recipients = float(weights[positive].sum())
    amount = float((weights[positive] * values[positive]).sum())
    return {
        "rows": int(len(values)),
        "missing_rows": int((~finite).sum()),
        "negative_rows": int((finite & (values < 0)).sum()),
        "positive_rows": int(positive.sum()),
        "weighted_positive_share": recipients / total if total > 0 else 0.0,
        "weighted_mean_where_positive": amount / recipients if recipients > 0 else 0.0,
    }


def require_acs_local_weeks_deferrals_donor(frame: Frame) -> dict[str, Any]:
    """The donor's ASEC role coverage of the four targets and the predictor.

    Refuses only a donor without the ASEC role or without a column, which the
    pass cannot fit; missing and negative cells are recorded, not refused
    (the transfer fits on complete donor rows).
    """

    try:
        donor, channel = resolve_acs_donor_channel(
            frame, ACS_LOCAL_WEEKS_DEFERRALS_DONOR_CHANNEL
        )
    except ValueError as exc:
        raise ValueError(
            "The ACS local weeks and deferrals donor needs the ASEC observation "
            f"role ({ACS_LOCAL_WEEKS_DEFERRALS_ISSUE}): {exc}"
        ) from exc
    person = donor.table("person")
    needed = (*ACS_LOCAL_WEEKS_DEFERRALS_COLUMNS, _UNEMPLOYMENT_COMPENSATION)
    missing = [column for column in needed if column not in person]
    if missing:
        raise ValueError(
            f"The ASEC donor lacks {missing} ({ACS_LOCAL_WEEKS_DEFERRALS_ISSUE})."
        )
    weights = np.asarray(donor.resolve_weights("person").values, dtype=np.float64)
    weeks = _amounts(person, ACS_LOCAL_WEEKS_UNEMPLOYED_COLUMN)
    uc = _amounts(person, _UNEMPLOYMENT_COMPENSATION)
    return {
        "channel": channel,
        "person_rows": int(len(person)),
        "columns": {
            column: _summary(_amounts(person, column), weights) for column in needed
        },
        # The zero rule is the donor's rule for its PUF half; on its measured
        # ASEC role, weeks can be positive without UC (looking for work
        # without a claim), which the receipt counts.
        "weeks_positive_without_uc_rows": int(
            (np.nan_to_num(weeks) > 0)[np.nan_to_num(uc) <= 0].sum()
        ),
        "weeks_positive_without_uc_weighted_share": float(
            weights[(np.nan_to_num(weeks) > 0) & (np.nan_to_num(uc) <= 0)].sum()
        )
        / float(weights.sum())
        if float(weights.sum()) > 0
        else 0.0,
    }


@dataclass(frozen=True)
class AcsLocalWeeksDeferralsResult:
    """The ACS frame with filled cells, the pass's fits and its receipt."""

    frame: Frame
    fit_records: tuple[FitWeightRecord, ...]
    imputed_inputs: tuple[AcsImputedInput, ...]
    receipt: Mapping[str, Any]


def _wages(person: pd.DataFrame) -> np.ndarray:
    """ACS wages for the 401(k) gate; the blank under 15 reads as none."""

    if _WAGES not in person or _AGE not in person:
        raise ValueError(
            f"The ACS deferral gate requires {_WAGES} and {_AGE} on ACS rows "
            f"({ACS_LOCAL_WEEKS_DEFERRALS_ISSUE})."
        )
    wages = pd.to_numeric(person[_WAGES], errors="coerce")
    age = pd.to_numeric(person[_AGE], errors="coerce")
    blank = wages.isna() & age.ge(_WAGES_MINIMUM_AGE)
    if blank.any():
        raise ValueError(
            f"{int(blank.sum())} ACS person(s) aged {_WAGES_MINIMUM_AGE}+ have a "
            f"blank {_WAGES}."
        )
    return wages.fillna(0.0).to_numpy(dtype=np.float64)


def _unemployment_compensation(person: pd.DataFrame) -> np.ndarray:
    if _UNEMPLOYMENT_COMPENSATION not in person:
        raise ValueError(
            f"The ACS weeks zero rule requires {_UNEMPLOYMENT_COMPENSATION} on "
            "ACS rows: run the pass after the shared transfer "
            f"({ACS_LOCAL_WEEKS_DEFERRALS_ISSUE})."
        )
    uc = pd.to_numeric(person[_UNEMPLOYMENT_COMPENSATION], errors="coerce")
    if uc.isna().any():
        raise ValueError(
            f"{int(uc.isna().sum())} ACS person(s) have a blank transferred "
            f"{_UNEMPLOYMENT_COMPENSATION}."
        )
    return uc.to_numpy(dtype=np.float64)


def _missing_masks(person: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        column: (
            person[column].isna().to_numpy(dtype=bool)
            if column in person
            else np.ones(len(person), dtype=bool)
        )
        for column in ACS_LOCAL_WEEKS_DEFERRALS_COLUMNS
    }


def _postprocessed(
    person: pd.DataFrame, filled: Mapping[str, np.ndarray]
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply the donor's post-processing to the cells this pass filled."""

    person = person.copy()
    changed: dict[str, int] = {}
    weeks_rows = filled[ACS_LOCAL_WEEKS_UNEMPLOYED_COLUMN]
    if weeks_rows.any():
        column = ACS_LOCAL_WEEKS_UNEMPLOYED_COLUMN
        before = _amounts(person, column)
        after = before.copy()
        after[weeks_rows] = postprocess_us_weeks_unemployed_predictions(
            before[weeks_rows],
            unemployment_compensation=_unemployment_compensation(person)[weeks_rows],
        )
        changed[column] = int((after != before).sum())
        person[column] = after
    deferral_rows = np.zeros(len(person), dtype=bool)
    for column in ACS_LOCAL_RETIREMENT_DEFERRAL_COLUMNS:
        deferral_rows |= filled[column]
    if deferral_rows.any():
        wages = _wages(person)
        before = person[list(ACS_LOCAL_RETIREMENT_DEFERRAL_COLUMNS)].apply(
            pd.to_numeric, errors="coerce"
        )
        processed = postprocess_us_retirement_contribution_predictions(
            before.loc[deferral_rows].reset_index(drop=True),
            employment_income=wages[deferral_rows],
        )
        for column in ACS_LOCAL_RETIREMENT_DEFERRAL_COLUMNS:
            values = before[column].to_numpy(dtype=np.float64).copy()
            rows = filled[column]
            new = processed[column].to_numpy(dtype=np.float64)
            # ``processed`` is aligned to ``deferral_rows``; keep only this
            # column's own filled cells.
            values[rows] = new[rows[deferral_rows]]
            changed[column] = int(
                (values != before[column].to_numpy(dtype=np.float64)).sum()
            )
            person[column] = values
    return person, changed


def _with_person(frame: Frame, person: pd.DataFrame) -> Frame:
    tables = {entity: frame.table(entity) for entity in frame.entities}
    tables["person"] = person
    tables.update({name: frame.link(name) for name in frame.links})
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def _column_receipts(
    imputed: Sequence[AcsImputedInput],
    person: pd.DataFrame,
    filled: Mapping[str, np.ndarray],
    changed: Mapping[str, int],
    weights: np.ndarray,
) -> dict[str, Any]:
    columns: dict[str, Any] = {}
    for family, targets in ACS_LOCAL_WEEKS_DEFERRALS_FAMILIES.items():
        for column in targets:
            entries = [item for item in imputed if item.column == column]
            columns[column] = {
                "family": family,
                "transfer_entries": len(entries),
                "donor_channel": sorted({str(item.donor_channel) for item in entries}),
                "imputed_rows": int(
                    sum(item.imputed_recipient_rows for item in entries)
                ),
                "unmodeled_rows": int(
                    sum(item.unmodeled_recipient_rows for item in entries)
                ),
                "predictors": sorted(
                    {str(p) for item in entries for p in item.predictors}
                ),
                "filled_rows": int(filled[column].sum()),
                "postprocess_changed_rows": int(changed.get(column, 0)),
                "acs": _summary(_amounts(person, column), weights),
            }
    return columns


def with_acs_local_weeks_deferrals(
    frame: Frame,
    donor: Frame,
    *,
    seed: int,
    n_estimators: int,
    max_targets_per_fit: int = DEFAULT_ACS_TRANSFER_MAX_TARGETS_PER_FIT,
    donor_spine: str = ASEC_PUF_DONOR_SPINE,
) -> AcsLocalWeeksDeferralsResult:
    """Fill the ACS rows' weeks unemployed and deferrals from the ASEC role.

    Args:
        frame: The ACS-only frame after the shared transfer.
        donor: The ASEC-by-PUF base, whose ASEC observation role is the donor.
        seed, n_estimators, max_targets_per_fit, donor_spine: As the other
            transfers of the lane.

    Returns:
        The frame with filled cells, the fit records, the imputed-input
        provenance and the receipt (method, donor coverage, predictors, fill
        and post-processing counts, ACS shares). The receipt is not graded.

    Raises:
        ValueError: If the plan overlaps another transfer, the donor lacks
            the ASEC role or a column, or an input the post-processing reads
            is blank on an ACS row.
    """

    if frame.schema != US_SCHEMA:
        raise ValueError("The ACS local weeks and deferrals pass needs the US schema.")
    plan = acs_local_weeks_deferrals_target_families()
    donor_summary = require_acs_local_weeks_deferrals_donor(donor)
    filled = _missing_masks(frame.table("person"))
    transfer = transfer_acs_inputs(
        frame,
        donor,
        target_families=plan,
        donor_spine=donor_spine,
        donor_channel=ACS_LOCAL_WEEKS_DEFERRALS_DONOR_CHANNEL,
        seed=seed,
        n_estimators=n_estimators,
        max_targets_per_fit=max_targets_per_fit,
        person_predictor_extensions=ACS_LOCAL_WEEKS_DEFERRALS_PREDICTOR_EXTENSIONS,
    )
    person, changed = _postprocessed(transfer.frame.table("person"), filled)
    result = _with_person(transfer.frame, person)
    weights = np.asarray(result.resolve_weights("person").values, dtype=np.float64)
    uc = _unemployment_compensation(person)
    weeks = _amounts(person, ACS_LOCAL_WEEKS_UNEMPLOYED_COLUMN)
    receipt: dict[str, Any] = {
        "issue": ACS_LOCAL_WEEKS_DEFERRALS_ISSUE,
        "method": ACS_LOCAL_WEEKS_DEFERRALS_METHOD,
        "spine": ACS_2024_1YR_SPINE,
        "donor_channel": ACS_LOCAL_WEEKS_DEFERRALS_DONOR_CHANNEL,
        "resolved_donor_channel": transfer.resolved_donor_channel,
        "donor": donor_summary,
        "families": {
            family: list(targets)
            for family, targets in ACS_LOCAL_WEEKS_DEFERRALS_FAMILIES.items()
        },
        "predictor_extensions": [
            extension.identity()
            for extension in ACS_LOCAL_WEEKS_DEFERRALS_PREDICTOR_EXTENSIONS
        ],
        "postprocessing": {
            ACS_LOCAL_WEEKS_UNEMPLOYED_COLUMN: (
                "weeks_unemployed.postprocess_us_weeks_unemployed_predictions: "
                "clip to 0-52, round, 0 where unemployment_compensation is 0"
            ),
            "deferrals": (
                "retirement_contributions."
                "postprocess_us_retirement_contribution_predictions: clip at "
                "0; 401(k) deferrals 0 where wages are 0 (the WAGP blank under "
                "15 reads as no wages)"
            ),
        },
        "not_transferred_shared_plan_columns": [
            "traditional_ira_contributions_desired",
            "self_employed_pension_contributions_desired",
        ],
        "acs_persons": int(len(person)),
        "acs_uc_recipient_rows": int((uc > 0).sum()),
        "acs_uc_recipients_with_positive_weeks_rows": int(
            ((uc > 0) & (np.nan_to_num(weeks) > 0)).sum()
        ),
        "columns": _column_receipts(
            transfer.imputed_inputs, person, filled, changed, weights
        ),
    }
    return AcsLocalWeeksDeferralsResult(
        frame=result,
        fit_records=tuple(transfer.fit_records),
        imputed_inputs=tuple(transfer.imputed_inputs),
        receipt=receipt,
    )
