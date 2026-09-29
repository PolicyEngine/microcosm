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
It never edits the shared declared transfer plan, which the pool lane shares,
nor its execution contract.

- **Donor channel.** The ASEC observation role
  (:data:`~microcosm.build.us_runtime.support_provenance.BASE_ASEC_SUPPORT_CHANNEL`)
  carries the measured survey values: ``CSP_VAL``/``CHSP_VAL``, ``WC_VAL``,
  ``DIS_VAL1``/``DIS_VAL2`` where ``DIS_SC`` is not 1, and the ``DST_SC*``/
  ``DST_VAL*`` account slots. The PUF clone role carries CPS-trained QRF
  predictions of the same leaves, so fitting on it would transfer a model of
  a model.
- **Predictors.** The existing transfer predictors: age, sex and state, plus
  the optional wage, self-employment, Social Security, retirement (ACS
  ``RETP``), interest/dividend/rental, household-head and tenure features
  where both sides observe them. ACS ``OIP`` is not loaded: a new combined
  predictor would change the shared execution contract.
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
  ASEC rows and any existing ACS value are untouched.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.gates import GateResult
from microcosm.build.us_runtime.acs_pums import ACS_2024_1YR_SPINE
from microcosm.build.us_runtime.acs_transfer import (
    ASEC_PUF_DONOR_SPINE,
    TargetFamilies,
    required_acs_transfer_inputs,
    resolve_acs_donor_channel,
)
from microcosm.build.us_runtime.base_pool import spine_column
from microcosm.build.us_runtime.support_provenance import BASE_ASEC_SUPPORT_CHANNEL
from microcosm.frame import Frame

__all__ = [
    "ACS_LOCAL_INCOME_DONOR_CHANNEL",
    "ACS_LOCAL_INCOME_REVIEW_BAND",
    "ACS_LOCAL_INCOME_SHARED_RETIREMENT_COMPONENTS",
    "ACS_LOCAL_INCOME_TRANSFER_COLUMNS",
    "ACS_LOCAL_INCOME_TRANSFER_FAMILIES",
    "ACS_LOCAL_INCOME_TRANSFER_GATE_NAME",
    "ACS_LOCAL_INCOME_TRANSFER_ISSUE",
    "acs_local_income_transfer_signal_gate",
    "acs_local_income_transfer_target_families",
    "record_acs_local_income_transfer",
    "require_acs_local_income_donor",
]

ACS_LOCAL_INCOME_TRANSFER_ISSUE = "microcosm#1022"
ACS_LOCAL_INCOME_TRANSFER_GATE_NAME = "acs_local_income_transfer_signal"
ACS_LOCAL_INCOME_DONOR_CHANNEL = BASE_ASEC_SUPPORT_CHANNEL
#: One chained QRF per family keeps jointly reported amounts together.
ACS_LOCAL_INCOME_TRANSFER_FAMILIES: Mapping[str, tuple[str, ...]] = {
    "acs_local_child_support": ("child_support_received", "child_support_expense"),
    "acs_local_work_disability_income": ("workers_compensation", "disability_benefits"),
    "acs_local_retirement_distributions": (
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

    Every target must be present, finite and non-negative on the ASEC
    observation role. A target with no positive donor value is allowed (the
    transfer then fills zeros) and recorded, never invented. Returns the
    donor summary the staging receipt records.
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
    return {"channel": channel, "person_rows": int(len(person)), "columns": columns}


def record_acs_local_income_transfer(
    donor: Mapping[str, Any],
    imputed_inputs: Sequence[Mapping[str, Any]],
    *,
    acs_persons: int,
) -> dict[str, Any]:
    """The staging receipt: method, donor summary and per-column fill counts.

    ``imputed_inputs`` is the JSON-ready provenance of the income pass alone.
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
        "method": "separate_local_qrf_pass_missing_cells_only",
        "donor_channel": donor.get("channel"),
        "donor": dict(donor),
        "families": {
            family: list(targets)
            for family, targets in ACS_LOCAL_INCOME_TRANSFER_FAMILIES.items()
        },
        "not_transferred_shared_plan_components": list(
            ACS_LOCAL_INCOME_SHARED_RETIREMENT_COMPONENTS
        ),
        "acs_persons": int(acs_persons),
        "columns": columns,
    }


def _receipt_failures(
    receipt: object, acs_rows: int, details: dict[str, object]
) -> list[str]:
    """The staging receipt must show an ASEC-channel pass with no gaps."""

    if not isinstance(receipt, Mapping) or (
        receipt.get("issue") != ACS_LOCAL_INCOME_TRANSFER_ISSUE
    ):
        details["receipt"] = {"present": False}
        return [
            "No acs_local_income_transfer staging receipt "
            f"({ACS_LOCAL_INCOME_TRANSFER_ISSUE}); the ACS income inputs cannot be "
            "shown to be transferred. Re-run staging with the current builder."
        ]
    failures: list[str] = []
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
    for column in ACS_LOCAL_INCOME_TRANSFER_COLUMNS:
        entry = columns.get(column)
        if not isinstance(entry, Mapping) or type(entry.get("imputed_rows")) is not int:
            failures.append(f"receipt: no transfer count for {column}.")
            continue
        if entry.get("unmodeled_rows") != 0:
            failures.append(
                f"receipt: {column} left {entry.get('unmodeled_rows')!r} ACS "
                "row(s) unmodeled."
            )
    details["receipt"] = {"present": True, "failures": len(failures)}
    return failures


def acs_local_income_transfer_signal_gate(
    frame: Frame, *, receipt: Mapping[str, Any] | None
) -> GateResult:
    """Require complete, non-negative transferred income on both spines.

    Fails when a column is absent or has a missing or negative cell on either
    spine (the reviewed-null fill would make a missing cell zero), when the
    ACS spine is all zero although the donor spine carries positive amounts,
    and unless ``receipt`` shows the ASEC-channel pass with no unmodeled row.
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
