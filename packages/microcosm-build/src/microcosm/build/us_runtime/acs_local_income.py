"""SNAP-relevant ASEC income inputs transferred onto ACS rows (ACS local lane).

Nine person income leaves reached the ACS local-area release missing on every
ACS row, and the reviewed-null fill gave them the engine default of zero
(microcosm#1022): child support received and paid, workers' compensation,
non-SSA disability benefits, and five retirement-account distributions. SNAP
counts each as unearned income (child support paid is a SNAP deduction), SSI
counts them as unearned income, and the retirement distributions are also
federal and state gross income. The ACS asks none of them separately: ``OIP``
and ``RETP`` are combined amounts.

This fresh-build stage is a separate, local-lane-only QRF pass, the pattern of
the qualified usual-hours pass (:mod:`~microcosm.build.us_runtime.acs_local_hours`).
It never edits the shared declared transfer plan, which the pool lane shares.
Its two local predictors enter through an opt-in extension of this pass's own
transfer call, so the shared execution contract, its SHA-256 and the shared
transfer's draws are unchanged for every other caller.

- **Donor channel.** The ASEC observation role
  (:data:`~microcosm.build.us_runtime.support_provenance.BASE_ASEC_SUPPORT_CHANNEL`)
  carries the measured survey values: ``CSP_VAL``/``CHSP_VAL``, ``WC_VAL``,
  ``DIS_VAL1``/``DIS_VAL2`` where ``DIS_SC`` is not 1, and the ``DST_SC*``/
  ``DST_VAL*`` account slots. The PUF clone role carries CPS-trained QRF
  predictions of the same leaves, so fitting on it would transfer a model of
  a model.
- **Predictors.** The existing transfer predictors (age, sex and state, plus
  the optional wage, self-employment, Social Security, retirement (ACS
  ``RETP``), interest/dividend/rental, household-head and tenure features
  where both sides observe them) and two local-only
  :class:`~microcosm.build.us_runtime.acs_transfer.AcsPersonPredictorExtension`
  predictors, chosen in a weighted 5-fold CPS ASEC 2024 holdout that ran this
  pass's own QRF, seeds and families (microcosm#1056 review):

  - **ACS ``OIP``** (all other income, ``OIP * ADJINC / 1e6``) for the child
    support family only. Its donor analog sums the ASEC-role leaves in
    :data:`ACS_LOCAL_OTHER_INCOME_DONOR_COMPONENTS`. ASEC ``FIN_VAL``
    (financial assistance from people outside the household, part of OIP) is
    not carried by the pinned donor; this buildable analog cut holdout CRPS
    for child support received by 29% (37% with ``FIN_VAL``). Child support
    paid, which shares the family, was neutral. Workers' compensation was
    mixed (hit rate up, CRPS 11-14% worse) and awaits review, so OIP stays out
    of that family, and out of the retirement family, where it was neutral.
  - **An ACS-aligned ``RETP`` analog** for the work/disability and retirement
    distribution families, standing in for the shared retirement predictor in
    those families only. ACS RETP counts pensions and annuities, every
    retirement-account distribution, survivor income and disability pensions;
    the shared donor analog sums pensions, annuities and regular-IRA
    distributions only. The local analog adds the five account-distribution
    leaves and ``disability_benefits``
    (:data:`ACS_LOCAL_ALIGNED_RETIREMENT_DONOR_COMPONENTS`); survivor income
    (``SUR_VAL``) and other-account distributions (``DST`` code 7) are not
    carried. It cut holdout CRPS by 18% for disability benefits and 19% for
    401(k) distributions. ACS rows keep ``RETP * ADJINC / 1e6``, and the shared
    transfer's own retirement predictor is untouched (aligning it is
    microcosm#1065).
- **No double counting.** The shared plan already transfers
  ``taxable_private_pension_income``, ``tax_exempt_private_pension_income``
  and ``taxable_ira_distributions`` onto ACS rows, so this pass transfers only
  the account types it does not: 401(k), 403(b), SEP, Keogh and Roth-IRA
  (``tax_exempt_ira_distributions``). :func:`acs_local_income_transfer_target_families`
  refuses any overlap with the shared plan. On the ASEC donor the pension
  leaves come from the pension/annuity items and the distributions from the
  ``DST`` account slots, so the components are disjoint; ``RETP`` is only a
  predictor on ACS rows, never an amount.
- **Null cells only.** The shared transfer merge fills only missing cells, so
  ASEC rows and any existing ACS value are untouched. The mapped
  ``acs_other_income`` column is this pass's predictor source only and is
  dropped before the shared transfer, so it never reaches the pool or the
  release.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.gates import GateResult
from microcosm.build.us_runtime.acs_inputs import _adjusted_dollars
from microcosm.build.us_runtime.acs_pums import ACS_2024_1YR_SPINE
from microcosm.build.us_runtime.acs_transfer import (
    ASEC_PUF_DONOR_SPINE,
    AcsPersonPredictorExtension,
    TargetFamilies,
    required_acs_transfer_inputs,
    resolve_acs_donor_channel,
)
from microcosm.build.us_runtime.base_pool import spine_column
from microcosm.build.us_runtime.support_provenance import BASE_ASEC_SUPPORT_CHANNEL
from microcosm.frame import US_SCHEMA, Frame

__all__ = [
    "ACS_LOCAL_ALIGNED_RETIREMENT_DONOR_COMPONENTS",
    "ACS_LOCAL_ALIGNED_RETIREMENT_FEATURE",
    "ACS_LOCAL_CHILD_SUPPORT_FAMILY",
    "ACS_LOCAL_INCOME_DONOR_CHANNEL",
    "ACS_LOCAL_INCOME_PREDICTOR_EXTENSIONS",
    "ACS_LOCAL_INCOME_REVIEW_BAND",
    "ACS_LOCAL_INCOME_SHARED_RETIREMENT_COMPONENTS",
    "ACS_LOCAL_INCOME_TRANSFER_COLUMNS",
    "ACS_LOCAL_INCOME_TRANSFER_FAMILIES",
    "ACS_LOCAL_INCOME_TRANSFER_GATE_NAME",
    "ACS_LOCAL_INCOME_TRANSFER_ISSUE",
    "ACS_LOCAL_INCOME_TRANSFER_METHOD",
    "ACS_LOCAL_OTHER_INCOME_COLUMN",
    "ACS_LOCAL_OTHER_INCOME_DONOR_COMPONENTS",
    "ACS_LOCAL_OTHER_INCOME_FEATURE",
    "ACS_LOCAL_RETIREMENT_FAMILY",
    "ACS_LOCAL_SHARED_RETIREMENT_FEATURE",
    "ACS_LOCAL_WORK_DISABILITY_FAMILY",
    "AcsLocalOtherIncomeResult",
    "acs_local_income_predictor_extension_receipt",
    "acs_local_income_transfer_signal_gate",
    "acs_local_income_transfer_target_families",
    "map_acs_local_other_income",
    "record_acs_local_income_transfer",
    "require_acs_local_income_donor",
    "without_acs_local_other_income",
]

ACS_LOCAL_INCOME_TRANSFER_ISSUE = "microcosm#1022"
ACS_LOCAL_INCOME_TRANSFER_GATE_NAME = "acs_local_income_transfer_signal"
#: The reviewed method. A receipt from a staging run before the OIP and
#: aligned-RETP predictors (microcosm#1056 review) records another id and is
#: refused by the gate and the release tool.
ACS_LOCAL_INCOME_TRANSFER_METHOD = (
    "separate_local_qrf_pass_oip_aligned_retp_missing_cells_only"
)
ACS_LOCAL_INCOME_DONOR_CHANNEL = BASE_ASEC_SUPPORT_CHANNEL
ACS_LOCAL_CHILD_SUPPORT_FAMILY = "acs_local_child_support"
ACS_LOCAL_WORK_DISABILITY_FAMILY = "acs_local_work_disability_income"
ACS_LOCAL_RETIREMENT_FAMILY = "acs_local_retirement_distributions"
#: One chained QRF per family keeps jointly reported amounts together.
ACS_LOCAL_INCOME_TRANSFER_FAMILIES: Mapping[str, tuple[str, ...]] = {
    ACS_LOCAL_CHILD_SUPPORT_FAMILY: ("child_support_received", "child_support_expense"),
    ACS_LOCAL_WORK_DISABILITY_FAMILY: ("workers_compensation", "disability_benefits"),
    ACS_LOCAL_RETIREMENT_FAMILY: (
        "taxable_401k_distributions",
        "taxable_403b_distributions",
        "taxable_sep_distributions",
        "keogh_distributions",
        "tax_exempt_ira_distributions",
    ),
}
ACS_LOCAL_INCOME_TRANSFER_COLUMNS: tuple[str, ...] = tuple(
    column
    for columns in ACS_LOCAL_INCOME_TRANSFER_FAMILIES.values()
    for column in columns
)
#: Retirement leaves the shared plan already transfers onto ACS rows; this
#: pass must never transfer them again.
ACS_LOCAL_INCOME_SHARED_RETIREMENT_COMPONENTS: tuple[str, ...] = (
    "taxable_private_pension_income",
    "tax_exempt_private_pension_income",
    "taxable_ira_distributions",
)
#: Informational ACS/donor ratio band for the weighted recipient share and
#: recipient mean. Outside it is reported for review, never a failure: the
#: ACS population differs from the ASEC donor (age, state mix, universe).
ACS_LOCAL_INCOME_REVIEW_BAND = (0.5, 2.0)

#: 2024 ACS PUMS data dictionary: ``OIP`` ("All other income past 12 months")
#: and ``RETP`` ("Retirement income past 12 months"), both blank under 15.
ACS_LOCAL_INCOME_PUMS_REFERENCE = (
    "https://www2.census.gov/programs-surveys/acs/tech_docs/pums/"
    "data_dict/PUMS_Data_Dictionary_2024.pdf#page=43"
)
#: 2024 ACS subject definitions: item 7 (retirement, survivor or disability
#: income) and item 8 (all other income).
ACS_LOCAL_INCOME_SUBJECT_DEFINITIONS_REFERENCE = (
    "https://www2.census.gov/programs-surveys/acs/tech_docs/"
    "subject_definitions/2024_ACSSubjectDefinitions.pdf#page=95"
)
ACS_LOCAL_OTHER_INCOME_SOURCE = "OIP"
#: ``OIP * ADJINC / 1e6`` on ACS rows: this pass's predictor source, dropped
#: again before the shared transfer and the pool.
ACS_LOCAL_OTHER_INCOME_COLUMN = "acs_other_income"
#: ACS asks the income items from age 15; ``OIP`` is blank exactly below it.
ACS_LOCAL_OTHER_INCOME_MIN_AGE = 15
ACS_LOCAL_OTHER_INCOME_FEATURE = "__acs_transfer_other_income"
#: The ASEC analog of ACS OIP, from leaves the donor's ASEC role carries:
#: ``UC_VAL``, ``WC_VAL``, ``VET_VAL``, ``CSP_VAL``, and ``OI_VAL`` split into
#: alimony (``OI_OFF`` 20), strike benefits (12) and every other code. ASEC
#: ``FIN_VAL`` (financial assistance from people outside the household, part
#: of ACS OIP) is not carried by the pinned donor; miscellaneous income keeps
#: the few ``OI_OFF`` codes ACS files under RETP or other items (about $1B).
ACS_LOCAL_OTHER_INCOME_DONOR_COMPONENTS: tuple[str, ...] = (
    "unemployment_compensation",
    "workers_compensation",
    "veterans_benefits",
    "child_support_received",
    "alimony_income",
    "strike_benefits",
    "miscellaneous_income",
)
#: The shared transfer's retirement predictor and its ACS source. The local
#: aligned analog stands in for the predictor in this pass's work/disability
#: and retirement families only; the shared transfer keeps it unchanged.
ACS_LOCAL_SHARED_RETIREMENT_FEATURE = "__acs_transfer_retirement_income"
ACS_LOCAL_RETIREMENT_SOURCE = "acs_retirement_income"
ACS_LOCAL_ALIGNED_RETIREMENT_FEATURE = "__acs_transfer_retirement_income_acs_aligned"
#: ACS RETP counts pensions and annuities, every retirement-account
#: distribution (IRA, Roth IRA, 401(k), 403(b), SEP, Keogh), survivor income
#: and disability pensions. The shared analog is the first three leaves
#: (pensions and annuities, regular-IRA distributions); this adds the five
#: account-distribution leaves and non-SSA, non-workers'-compensation
#: disability benefits. Survivor income (``SUR_VAL``) and other-account
#: distributions (``DST`` code 7) are not carried by the pinned donor.
ACS_LOCAL_ALIGNED_RETIREMENT_DONOR_COMPONENTS: tuple[str, ...] = (
    "taxable_private_pension_income",
    "tax_exempt_private_pension_income",
    "taxable_ira_distributions",
    "taxable_401k_distributions",
    "taxable_403b_distributions",
    "taxable_sep_distributions",
    "keogh_distributions",
    "tax_exempt_ira_distributions",
    "disability_benefits",
)
#: The local pass's predictor extensions: OIP for the child support family,
#: the aligned RETP analog for the work/disability and retirement families.
ACS_LOCAL_INCOME_PREDICTOR_EXTENSIONS: tuple[AcsPersonPredictorExtension, ...] = (
    AcsPersonPredictorExtension(
        feature=ACS_LOCAL_OTHER_INCOME_FEATURE,
        donor_components=ACS_LOCAL_OTHER_INCOME_DONOR_COMPONENTS,
        recipient_source=ACS_LOCAL_OTHER_INCOME_COLUMN,
        families=(ACS_LOCAL_CHILD_SUPPORT_FAMILY,),
    ),
    AcsPersonPredictorExtension(
        feature=ACS_LOCAL_ALIGNED_RETIREMENT_FEATURE,
        donor_components=ACS_LOCAL_ALIGNED_RETIREMENT_DONOR_COMPONENTS,
        recipient_source=ACS_LOCAL_RETIREMENT_SOURCE,
        families=(ACS_LOCAL_WORK_DISABILITY_FAMILY, ACS_LOCAL_RETIREMENT_FAMILY),
        replaces=ACS_LOCAL_SHARED_RETIREMENT_FEATURE,
    ),
)


def acs_local_income_predictor_extension_receipt() -> list[dict[str, Any]]:
    """The reviewed predictor extensions, as the receipt records them.

    The gate refuses a receipt whose extensions differ, so a staging run that
    predates them, or used other definitions, cannot be released.
    """

    other_income, aligned_retirement = ACS_LOCAL_INCOME_PREDICTOR_EXTENSIONS
    references = [
        ACS_LOCAL_INCOME_PUMS_REFERENCE,
        ACS_LOCAL_INCOME_SUBJECT_DEFINITIONS_REFERENCE,
    ]
    return [
        {
            **other_income.identity(),
            "acs_definition": (
                "OIP * ADJINC / 1_000_000: ACS all other income (unemployment "
                "and workers' compensation, VA payments, alimony and child "
                "support, contributions from people outside the household, "
                "military family allotments and other periodic income); blank "
                "under age 15 stays missing"
            ),
            "donor_definition": (
                "ASEC UC_VAL + WC_VAL + VET_VAL + CSP_VAL + OI_VAL (alimony, "
                "strike benefits and every other other-income code)"
            ),
            "not_carried_by_donor": {
                "FIN_VAL": "financial assistance from people outside the household"
            },
            "withheld_from": {
                ACS_LOCAL_WORK_DISABILITY_FAMILY: (
                    "workers' compensation: mixed holdout result (hit rate up, "
                    "CRPS 11-14% worse); awaits review"
                ),
                ACS_LOCAL_RETIREMENT_FAMILY: (
                    "no holdout gain; account distributions are RETP, not OIP"
                ),
            },
            "references": references,
        },
        {
            **aligned_retirement.identity(),
            "acs_definition": (
                "RETP * ADJINC / 1_000_000: ACS retirement, survivor or "
                "disability income (the shared recipient source, unchanged)"
            ),
            "donor_definition": (
                "the shared analog (private pensions and annuities, regular-IRA "
                "distributions) plus 401(k), 403(b), SEP, Keogh and Roth-IRA "
                "distributions and non-SSA, non-workers'-compensation "
                "disability benefits"
            ),
            "not_carried_by_donor": {
                "SUR_VAL": "survivor income",
                "DST code 7": "other retirement-account distributions",
            },
            "references": references,
        },
    ]


def acs_local_income_transfer_target_families() -> TargetFamilies:
    """The separate ASEC-only income pass, disjoint from the shared plan."""

    overlap = sorted(
        set(ACS_LOCAL_INCOME_TRANSFER_COLUMNS) & required_acs_transfer_inputs()
    )
    if overlap:
        raise ValueError(
            "The ACS local income pass would transfer column(s) the shared plan "
            f"already transfers {overlap}; double counting "
            f"({ACS_LOCAL_INCOME_TRANSFER_ISSUE})."
        )
    return {"person": dict(ACS_LOCAL_INCOME_TRANSFER_FAMILIES)}


@dataclass(frozen=True)
class AcsLocalOtherIncomeResult:
    """The ACS frame with this pass's OIP predictor source, and its coverage."""

    frame: Frame
    coverage: Mapping[str, Any]


def _with_person_table(frame: Frame, person: pd.DataFrame) -> Frame:
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


def map_acs_local_other_income(frame: Frame) -> AcsLocalOtherIncomeResult:
    """Map ACS ``OIP`` to dollars as this pass's predictor source.

    Writes ``acs_other_income = OIP * ADJINC / 1e6`` on the ACS-only frame.
    A Census blank (under 15) stays missing, so those rows are fit without
    the predictor; it is never a zero. The coverage records the blank count.

    Raises:
        ValueError: If the frame is not US-schema; ``OIP``, ``ADJINC`` or
            ``AGEP`` is absent; an ``OIP`` value is non-numeric or negative;
            a blank or a value contradicts the age-15 universe; or the output
            column already exists.
    """

    if frame.schema != US_SCHEMA:
        raise ValueError("The ACS local income predictors require the US schema.")
    person = frame.table("person")
    missing = [
        column
        for column in (ACS_LOCAL_OTHER_INCOME_SOURCE, "ADJINC", "AGEP")
        if column not in person
    ]
    if missing:
        raise ValueError(
            f"The ACS local income pass requires person column(s) {missing}; the "
            f"pinned ACS PUMS source carries OIP ({ACS_LOCAL_INCOME_TRANSFER_ISSUE})."
        )
    if ACS_LOCAL_OTHER_INCOME_COLUMN in person:
        raise ValueError(
            "ACS local mapping refuses to overwrite existing column "
            f"{ACS_LOCAL_OTHER_INCOME_COLUMN!r}."
        )
    raw = person[ACS_LOCAL_OTHER_INCOME_SOURCE]
    blank = (
        raw.isna() | raw.astype("string").str.strip().eq("").fillna(False)
    ).to_numpy(dtype=bool)
    amounts = pd.to_numeric(raw.where(~blank), errors="coerce").to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    if (~blank & ~np.isfinite(amounts)).any():
        raise ValueError("ACS OIP must be blank or a finite dollar amount.")
    if (amounts[~blank] < 0).any():
        raise ValueError("ACS OIP must be a non-negative dollar amount.")
    age = pd.to_numeric(person["AGEP"], errors="coerce").to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    if np.isnan(age).any():
        raise ValueError("ACS OIP's universe requires AGEP on every person.")
    in_universe = age >= ACS_LOCAL_OTHER_INCOME_MIN_AGE
    contradictions = (in_universe & blank) | (~in_universe & ~blank)
    if contradictions.any():
        raise ValueError(
            "ACS OIP contradicts its universe (blank exactly under age "
            f"{ACS_LOCAL_OTHER_INCOME_MIN_AGE}): {int(contradictions.sum())} row(s)."
        )
    dollars = _adjusted_dollars(
        pd.Series(amounts, index=person.index, name=ACS_LOCAL_OTHER_INCOME_SOURCE),
        person,
        factor="ADJINC",
    )
    mapped = _with_person_table(
        frame, person.assign(**{ACS_LOCAL_OTHER_INCOME_COLUMN: dollars})
    )
    weights = np.asarray(mapped.resolve_weights("person").values, dtype=np.float64)
    observed = ~blank
    positive = observed & (np.nan_to_num(dollars, nan=0.0) > 0)
    observed_weight = float(weights[observed].sum())
    coverage = {
        "column": ACS_LOCAL_OTHER_INCOME_COLUMN,
        "source_columns": [ACS_LOCAL_OTHER_INCOME_SOURCE, "ADJINC", "AGEP"],
        "transformation": (
            "OIP * ADJINC / 1_000_000; blank under age 15 stays missing"
        ),
        "reference": ACS_LOCAL_INCOME_PUMS_REFERENCE,
        "persons": int(len(person)),
        "observed_rows": int(observed.sum()),
        "blank_rows": int(blank.sum()),
        "positive_rows": int(positive.sum()),
        "weighted_positive_share": (
            float(weights[positive].sum()) / observed_weight
            if observed_weight > 0
            else 0.0
        ),
    }
    return AcsLocalOtherIncomeResult(frame=mapped, coverage=coverage)


def without_acs_local_other_income(frame: Frame) -> Frame:
    """Drop this pass's OIP predictor source before the shared transfer.

    The shared transfer does not read it, and the pool and the release never
    carry it.
    """

    person = frame.table("person")
    if ACS_LOCAL_OTHER_INCOME_COLUMN not in person:
        raise ValueError(
            f"The ACS frame carries no {ACS_LOCAL_OTHER_INCOME_COLUMN!r} to drop."
        )
    return _with_person_table(
        frame, person.drop(columns=[ACS_LOCAL_OTHER_INCOME_COLUMN])
    )


def _amount_summary(values: np.ndarray, weights: np.ndarray) -> dict[str, Any]:
    """Weighted recipient share and means of one non-negative amount column."""

    finite = np.isfinite(values)
    positive = finite & (values > 0)
    total = float(weights[finite].sum())
    recipients = float(weights[positive].sum())
    amount = float((weights[positive] * values[positive]).sum())
    return {
        "missing_rows": int((~finite).sum()),
        "negative_rows": int((finite & (values < 0)).sum()),
        "positive_rows": int(positive.sum()),
        "weighted_recipient_share": recipients / total if total > 0 else 0.0,
        "weighted_mean": amount / total if total > 0 else 0.0,
        "weighted_recipient_mean": amount / recipients if recipients > 0 else 0.0,
    }


def _amounts(rows: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(rows[column], errors="coerce").to_numpy(
        dtype=np.float64, na_value=np.nan
    )


def require_acs_local_income_donor(frame: Frame) -> dict[str, Any]:
    """Refuse a donor whose ASEC role cannot supply the income targets.

    Every target, and every component of the two predictor extensions, must
    be present, finite and non-negative on the ASEC observation role. A
    target with no positive donor value is allowed (the transfer then fills
    zeros) and recorded, never invented. Returns the donor summary the
    staging receipt records, including each extension's weighted share
    positive (all ages and 15+).
    """

    try:
        donor, channel = resolve_acs_donor_channel(
            frame, ACS_LOCAL_INCOME_DONOR_CHANNEL
        )
    except ValueError as exc:
        raise ValueError(
            "The ACS local income donor needs the ASEC observation role "
            f"({ACS_LOCAL_INCOME_TRANSFER_ISSUE}): {exc}"
        ) from exc
    person = donor.table("person")
    missing = [
        column for column in ACS_LOCAL_INCOME_TRANSFER_COLUMNS if column not in person
    ]
    if missing:
        raise ValueError(
            f"The ASEC income donor lacks {missing} ({ACS_LOCAL_INCOME_TRANSFER_ISSUE})."
        )
    weights = np.asarray(donor.resolve_weights("person").values, dtype=np.float64)
    columns: dict[str, Any] = {}
    for column in ACS_LOCAL_INCOME_TRANSFER_COLUMNS:
        summary = _amount_summary(_amounts(person, column), weights)
        if summary["missing_rows"] or summary["negative_rows"]:
            raise ValueError(
                f"The ASEC income donor's {column} has {summary['missing_rows']} "
                f"missing and {summary['negative_rows']} negative row(s); the "
                "measured amounts must be complete, non-negative dollars."
            )
        columns[column] = summary
    adults = None
    if "age" in person:
        age = _amounts(person, "age")
        adults = np.isfinite(age) & (age >= ACS_LOCAL_OTHER_INCOME_MIN_AGE)
    extensions: dict[str, Any] = {}
    for extension in ACS_LOCAL_INCOME_PREDICTOR_EXTENSIONS:
        absent = [
            component
            for component in extension.donor_components
            if component not in person
        ]
        if absent:
            raise ValueError(
                f"The ASEC income donor lacks {absent}, component(s) of the "
                f"{extension.feature} predictor ({ACS_LOCAL_INCOME_TRANSFER_ISSUE})."
            )
        parts = []
        for component in extension.donor_components:
            part = _amounts(person, component)
            finite = np.isfinite(part)
            if not finite.all() or (part[finite] < 0).any():
                raise ValueError(
                    f"The ASEC income donor's {extension.feature} predictor "
                    f"component {component} has {int((~finite).sum())} missing "
                    f"and {int((part[finite] < 0).sum())} negative row(s); the "
                    "measured amounts must be complete, non-negative dollars."
                )
            parts.append(part)
        values = np.sum(parts, axis=0)
        summary = _amount_summary(values, weights)
        entry: dict[str, Any] = {
            "donor_components": list(extension.donor_components),
            **summary,
        }
        if adults is not None:
            entry["weighted_recipient_share_age_15_plus"] = _amount_summary(
                values[adults], weights[adults]
            )["weighted_recipient_share"]
        extensions[extension.feature] = entry
    return {
        "channel": channel,
        "person_rows": int(len(person)),
        "columns": columns,
        "predictor_extensions": extensions,
    }


def record_acs_local_income_transfer(
    donor: Mapping[str, Any],
    imputed_inputs: Sequence[Mapping[str, Any]],
    *,
    acs_persons: int,
    other_income: Mapping[str, Any],
) -> dict[str, Any]:
    """The staging receipt: method, predictors, coverage and fill counts.

    ``imputed_inputs`` is the JSON-ready provenance of the income pass alone;
    ``other_income`` is the ACS ``OIP`` coverage from
    :func:`map_acs_local_other_income`. The receipt records the predictor
    extensions' definitions, the predictors each family was fit with, and
    the donor's extension coverage (in ``donor``).
    """

    columns: dict[str, Any] = {}
    for family, targets in ACS_LOCAL_INCOME_TRANSFER_FAMILIES.items():
        for column in targets:
            entries = [
                item
                for item in imputed_inputs
                if isinstance(item, Mapping) and item.get("column") == column
            ]
            columns[column] = {
                "family": family,
                "transfer_entries": len(entries),
                "donor_channel": sorted(
                    {str(item.get("donor_channel")) for item in entries}
                ),
                "imputed_rows": int(
                    sum(
                        int(item.get("imputed_recipient_rows") or 0) for item in entries
                    )
                ),
                "unmodeled_rows": int(
                    sum(
                        int(item.get("unmodeled_recipient_rows") or 0)
                        for item in entries
                    )
                ),
                "predictors": sorted(
                    {
                        str(predictor)
                        for item in entries
                        for predictor in item.get("predictors") or ()
                    }
                ),
            }
    return {
        "issue": ACS_LOCAL_INCOME_TRANSFER_ISSUE,
        "method": ACS_LOCAL_INCOME_TRANSFER_METHOD,
        "donor_channel": donor.get("channel"),
        "donor": dict(donor),
        "families": {
            family: list(targets)
            for family, targets in ACS_LOCAL_INCOME_TRANSFER_FAMILIES.items()
        },
        "not_transferred_shared_plan_components": list(
            ACS_LOCAL_INCOME_SHARED_RETIREMENT_COMPONENTS
        ),
        "predictor_extensions": acs_local_income_predictor_extension_receipt(),
        "family_predictors": {
            family: sorted(
                {
                    predictor
                    for column in targets
                    for predictor in columns[column]["predictors"]
                }
            )
            for family, targets in ACS_LOCAL_INCOME_TRANSFER_FAMILIES.items()
        },
        "acs_other_income": dict(other_income),
        "acs_persons": int(acs_persons),
        "columns": columns,
    }


def _predictor_failures(column: str, family: str, predictors: object) -> list[str]:
    """Each extension on exactly its reviewed families, standing in if asked."""

    used = set(predictors) if isinstance(predictors, (list, tuple)) else set()
    failures: list[str] = []
    for extension in ACS_LOCAL_INCOME_PREDICTOR_EXTENSIONS:
        reviewed = family in extension.families
        if reviewed and extension.feature not in used:
            failures.append(
                f"receipt: {column} ({family}) was fit without {extension.feature}."
            )
        if not reviewed and extension.feature in used:
            failures.append(
                f"receipt: {column} ({family}) was fit with {extension.feature}, "
                f"which is reviewed for {list(extension.families)} only."
            )
        if reviewed and extension.replaces in used:
            failures.append(
                f"receipt: {column} ({family}) still uses the shared "
                f"{extension.replaces}."
            )
    return failures


def _extension_receipt_failures(receipt: Mapping[str, Any]) -> list[str]:
    """The reviewed method and extensions, with ACS and donor coverage."""

    failures: list[str] = []
    if receipt.get("method") != ACS_LOCAL_INCOME_TRANSFER_METHOD:
        failures.append(
            f"receipt: method {receipt.get('method')!r}, not "
            f"{ACS_LOCAL_INCOME_TRANSFER_METHOD!r}; the staging run predates the "
            "reviewed OIP and aligned-RETP predictors. Re-run staging with the "
            "current builder."
        )
    if receipt.get("predictor_extensions") != (
        acs_local_income_predictor_extension_receipt()
    ):
        failures.append(
            "receipt: its predictor extensions are not the reviewed OIP and "
            "aligned-RETP definitions."
        )
    coverage = receipt.get("acs_other_income")
    if not isinstance(coverage, Mapping) or any(
        type(coverage.get(key)) is not int for key in ("observed_rows", "blank_rows")
    ):
        failures.append("receipt: no ACS OIP coverage (observed and blank rows).")
    donor = receipt.get("donor")
    donor_extensions = (
        donor.get("predictor_extensions") if isinstance(donor, Mapping) else None
    )
    for extension in ACS_LOCAL_INCOME_PREDICTOR_EXTENSIONS:
        entry = (
            donor_extensions.get(extension.feature)
            if isinstance(donor_extensions, Mapping)
            else None
        )
        share = (
            entry.get("weighted_recipient_share")
            if isinstance(entry, Mapping)
            else None
        )
        if isinstance(share, bool) or not isinstance(share, (int, float)):
            failures.append(f"receipt: no donor coverage for {extension.feature}.")
    return failures


def _receipt_failures(
    receipt: object, acs_rows: int, details: dict[str, object]
) -> list[str]:
    """The staging receipt must show the reviewed ASEC-channel pass, no gaps."""

    if not isinstance(receipt, Mapping) or (
        receipt.get("issue") != ACS_LOCAL_INCOME_TRANSFER_ISSUE
    ):
        details["receipt"] = {"present": False}
        return [
            "No acs_local_income_transfer staging receipt "
            f"({ACS_LOCAL_INCOME_TRANSFER_ISSUE}); the ACS income inputs cannot be "
            "shown to be transferred. Re-run staging with the current builder."
        ]
    failures = _extension_receipt_failures(receipt)
    if receipt.get("donor_channel") != ACS_LOCAL_INCOME_DONOR_CHANNEL:
        failures.append(
            f"receipt: donor channel {receipt.get('donor_channel')!r}, not "
            f"{ACS_LOCAL_INCOME_DONOR_CHANNEL!r}."
        )
    if receipt.get("acs_persons") != acs_rows:
        failures.append(
            f"receipt: records {receipt.get('acs_persons')!r} ACS person(s) but "
            f"the frame has {acs_rows}."
        )
    columns = receipt.get("columns")
    columns = columns if isinstance(columns, Mapping) else {}
    for family, targets in ACS_LOCAL_INCOME_TRANSFER_FAMILIES.items():
        for column in targets:
            entry = columns.get(column)
            if (
                not isinstance(entry, Mapping)
                or type(entry.get("imputed_rows")) is not int
            ):
                failures.append(f"receipt: no transfer count for {column}.")
                continue
            if entry.get("unmodeled_rows") != 0:
                failures.append(
                    f"receipt: {column} left {entry.get('unmodeled_rows')!r} ACS "
                    "row(s) unmodeled."
                )
            failures += _predictor_failures(column, family, entry.get("predictors"))
    details["receipt"] = {"present": True, "failures": len(failures)}
    return failures


def acs_local_income_transfer_signal_gate(
    frame: Frame, *, receipt: Mapping[str, Any] | None
) -> GateResult:
    """Require complete, non-negative transferred income on both spines.

    Fails when a column is absent or has a missing or negative cell on either
    spine (the reviewed-null fill would make a missing cell zero), when the
    ACS spine is all zero although the donor spine carries positive amounts,
    and unless ``receipt`` shows the reviewed ASEC-channel pass
    (:data:`ACS_LOCAL_INCOME_TRANSFER_METHOD`) with no unmodeled row, the
    reviewed OIP and aligned-RETP predictor extensions each fit on exactly
    its own families (the aligned analog in place of the shared retirement
    predictor), and the ACS ``OIP`` and donor extension coverage.
    Details report, per column and spine, the weighted recipient share, mean
    and recipient mean, and the ACS/donor ratios against
    :data:`ACS_LOCAL_INCOME_REVIEW_BAND` (informational only).
    """

    person = frame.table("person")
    tag = spine_column("person")
    failures: list[str] = []
    per_spine: dict[str, Any] = {}
    comparison: dict[str, Any] = {}
    details: dict[str, object] = {
        "per_spine": per_spine,
        "comparison": comparison,
        "review_band": list(ACS_LOCAL_INCOME_REVIEW_BAND),
    }
    if tag not in person or person[tag].isna().any():
        failures.append(f"Missing person origin tags: {tag}.")
        return GateResult(
            name=ACS_LOCAL_INCOME_TRANSFER_GATE_NAME,
            passed=False,
            failures=tuple(failures),
            details=details,
        )
    weights = np.asarray(frame.resolve_weights("person").values, dtype=np.float64)
    for spine in (ASEC_PUF_DONOR_SPINE, ACS_2024_1YR_SPINE):
        selected = person[tag].eq(spine).to_numpy(dtype=bool)
        columns: dict[str, Any] = {}
        per_spine[spine] = {"rows": int(selected.sum()), "columns": columns}
        if not selected.any():
            failures.append(f"{spine}: no person rows.")
            continue
        rows = person.loc[selected]
        for column in ACS_LOCAL_INCOME_TRANSFER_COLUMNS:
            if column not in rows:
                failures.append(
                    f"{spine}: missing {column}; the engine default 0 drops every "
                    "recipient's income."
                )
                continue
            summary = _amount_summary(_amounts(rows, column), weights[selected])
            columns[column] = summary
            if summary["missing_rows"]:
                failures.append(
                    f"{spine}: {column} has {summary['missing_rows']} missing "
                    "row(s); the reviewed-null fill would make them 0."
                )
            if summary["negative_rows"]:
                failures.append(
                    f"{spine}: {column} has {summary['negative_rows']} negative row(s)."
                )
    donor_columns = per_spine[ASEC_PUF_DONOR_SPINE]["columns"]
    acs_columns = per_spine[ACS_2024_1YR_SPINE]["columns"]
    low, high = ACS_LOCAL_INCOME_REVIEW_BAND
    for column in ACS_LOCAL_INCOME_TRANSFER_COLUMNS:
        donor, acs = donor_columns.get(column), acs_columns.get(column)
        if donor is None or acs is None:
            continue
        if donor["positive_rows"] and not acs["positive_rows"]:
            failures.append(
                f"{ACS_2024_1YR_SPINE}: {column} is all zero although the donor "
                "spine carries positive amounts; the transfer carried no signal."
            )
        ratios = {
            name: (acs[name] / donor[name] if donor[name] > 0 else None)
            for name in ("weighted_recipient_share", "weighted_recipient_mean")
        }
        comparison[column] = {
            **{f"{name}_ratio": value for name, value in ratios.items()},
            "within_review_band": all(
                value is not None and low <= value <= high for value in ratios.values()
            ),
        }
    acs_rows = int(per_spine[ACS_2024_1YR_SPINE]["rows"])
    if set(person[tag].unique()) - {ASEC_PUF_DONOR_SPINE, ACS_2024_1YR_SPINE}:
        failures.append("Income origin tags contain an unsupported spine.")
    failures += _receipt_failures(receipt, acs_rows, details)
    return GateResult(
        name=ACS_LOCAL_INCOME_TRANSFER_GATE_NAME,
        passed=not failures,
        failures=tuple(failures),
        details=details,
    )
