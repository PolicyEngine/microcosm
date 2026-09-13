"""Read the pinned 2024 NSECE V1 household and child-calendar public files.

Source: ICPSR 39466 V1, Household Data Files User's Guide, HH-57, HH-81,
HH-279--280, HH-334, HH-554 and HH-565--568. This adapter measures ECE
attendance (including unpaid care), excluding K-8 schooling. It does not
determine licensed-provider status or CCDF eligibility/receipt.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.us_runtime.childcare_attendance import (
    US_CHILDCARE_ATTENDANCE_COLUMNS,
    childcare_attendance_contract,
    childcare_income_band,
    impute_us_childcare_attendance,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

_SOURCE_ARTIFACTS = {
    artifact["dataset"]: artifact
    for artifact in childcare_attendance_contract()["artifacts"]
}
NSECE_2024_HOUSEHOLD_SHA256 = _SOURCE_ARTIFACTS["DS5"]["sha256"]
NSECE_2024_CALENDAR_SHA256 = _SOURCE_ARTIFACTS["DS4"]["sha256"]
NSECE_CHILD_INDICES = tuple(range(1, 10))
NSECE_PROVIDER_INDICES = tuple(range(1, 16))
NSECE_CALENDAR_BLOCKS = 672
NSECE_CHILDCARE_MATCH_COLUMNS = ("age", "region", "parent_work_status", "income_band")
NSECE_CHILDCARE_FALLBACK_COLUMNS = (
    ("age", "parent_work_status", "income_band"),
    ("age", "parent_work_status"),
    ("age",),
)
# HH-280: individual regular paid/unpaid, center, other organizational, irregular.
NSECE_ECE_TYPES = frozenset({1, 2, 3, 4, 5, 7})
# HH-566--568: parental care, self-care, or school only. Mixed/unclear gap-check
# codes are deliberately unresolved, even if part of the interval involved ECE.
NSECE_NON_ECE_CALENDAR_CODES = frozenset({0, 50, 53, 56, 57, 60, 65})
NSECE_UNPAID_GAP_CODES = frozenset({54, 61, 62, 69})


@dataclass(frozen=True)
class NSECEChildcareSource:
    """All source children, including explicit reasons for unresolved schedules."""

    children: pd.DataFrame
    weights: Weights
    source_receipt: dict[str, object]

    def donors(self) -> tuple[pd.DataFrame, Weights]:
        usable = (
            self.children["attendance_status"]
            .isin(["complete", "summary_bridge"])
            .to_numpy()
        )
        return (
            self.children.loc[usable].reset_index(drop=True).copy(),
            Weights(self.weights.values[usable], self.weights.kind),
        )


def nsece_childcare_household_columns() -> tuple[str, ...]:
    return (
        "HH4_METH_CASEID",
        "HH4_METH_WEIGHT",
        "HH4_METH_QUEXVERSION",
        "HH4_REGION",
        "HH4_PARWORK_STATUS",
        "HH4_RPARENT",
        "HH4_ECON_INCOME_ANNUAL",
        *(
            f"{prefix}_{child}"
            for child in NSECE_CHILD_INDICES
            for prefix in (
                "HHC4_AGE_AT_USAGE",
                "HHC4_METH_WEIGHT",
                "HH4_MISSING_STATUS_CC",
            )
        ),
        *(
            f"HH4_TYPEOFCARE_AGG_{child}_{provider}"
            for child in NSECE_CHILD_INDICES
            for provider in NSECE_PROVIDER_INDICES
        ),
        *(
            f"HHC4_NPC_HRSWEEK_TOC{kind}_{child}"
            for child in NSECE_CHILD_INDICES
            for kind in range(1, 10)
        ),
    )


def nsece_childcare_calendar_columns() -> tuple[str, ...]:
    return (
        "HH4_METH_CASEID",
        *(
            f"HH4_CHCAL_R_{child}_{block}"
            for child in NSECE_CHILD_INDICES
            for block in range(1, NSECE_CALENDAR_BLOCKS + 1)
        ),
    )


def derive_nsece_childcare(
    household: pd.DataFrame, calendar: pd.DataFrame
) -> NSECEChildcareSource:
    """Normalize source children; never treat missing calendars as no care.

    A day is attended if any 15-minute ECE block occurs on that day. Average
    daily hours are total ECE block-hours divided by attended days. Monthly
    days use the explicit representative-week approximation floor(d * 52/12
    + 0.5). This is an approximation, not observed monthly attendance. Blocks
    already encode a single final provider, so overlapping care is not added.
    """
    for table, columns in (
        (household, nsece_childcare_household_columns()),
        (calendar, nsece_childcare_calendar_columns()),
    ):
        missing = sorted(set(columns) - set(table.columns))
        if missing or not table.columns.is_unique:
            raise ValueError(f"NSECE source columns invalid; missing={missing[:10]}.")
        ids = table["HH4_METH_CASEID"]
        if ids.isna().any() or ids.duplicated().any():
            raise ValueError("NSECE household IDs must be complete and unique.")
    if set(household.HH4_METH_CASEID) != set(calendar.HH4_METH_CASEID):
        raise ValueError("NSECE household/calendar ID sets must match exactly.")
    hh = household.set_index("HH4_METH_CASEID").sort_index()
    cal = calendar.set_index("HH4_METH_CASEID").reindex(hh.index)
    parts = []
    for child in NSECE_CHILD_INDICES:
        age_months = pd.to_numeric(hh[f"HHC4_AGE_AT_USAGE_{child}"], errors="raise")
        if (
            not np.isfinite(age_months).all()
            or (age_months != np.floor(age_months)).any()
            or (~((age_months >= 0) | (age_months == -9))).any()
        ):
            raise ValueError("NSECE child age must be measured or the no-child code.")
        present = age_months >= 0
        if not present.any():
            continue
        rows = hh.loc[present]
        values = cal.loc[
            present,
            [f"HH4_CHCAL_R_{child}_{b}" for b in range(1, NSECE_CALENDAR_BLOCKS + 1)],
        ].to_numpy(dtype=float)
        types = rows[
            [f"HH4_TYPEOFCARE_AGG_{child}_{p}" for p in NSECE_PROVIDER_INDICES]
        ].to_numpy(dtype=float)
        status = rows[f"HH4_MISSING_STATUS_CC_{child}"].to_numpy(dtype=float)
        if not np.isin(status, [0, 1, 2]).all():
            raise ValueError("Unknown NSECE calendar completeness code.")
        ece = np.zeros(values.shape, dtype=bool)
        known = np.isin(values, tuple(NSECE_NON_ECE_CALENDAR_CODES))
        # HH-314/566: respondent/spouse care depends on parent status. School
        # without a provider type is not evidence of K-8 for preschool children.
        respondent_parent = rows.HH4_RPARENT.to_numpy()
        respondent_care = np.isin(values, [51, 52])
        ece |= np.isin(values, tuple(NSECE_UNPAID_GAP_CODES))
        ece |= respondent_care & (respondent_parent == 0)[:, None]
        known |= ece | (respondent_care & (respondent_parent == 1)[:, None])
        known |= (values == 68) & (age_months.loc[present].to_numpy() >= 72)[:, None]
        regular_ece = ece.copy()
        provider_count = np.zeros(len(rows), dtype=int)
        for p in NSECE_PROVIDER_INDICES:
            used = values == p
            ece_type = np.isin(types[:, p - 1], tuple(NSECE_ECE_TYPES))
            ece |= used & ece_type[:, None]
            regular_ece |= used & np.isin(types[:, p - 1], [1, 2, 3, 4, 5])[:, None]
            known |= used & (ece_type | (types[:, p - 1] == 6))[:, None]
            provider_count += used.any(axis=1) & ece_type
        age = np.floor(age_months.loc[present].to_numpy() / 12)
        reason = np.select(
            [age > 12, status == 0, status == 1, ~known.all(axis=1)],
            [
                "age_out_of_scope",
                "missing_calendar",
                "partial_calendar",
                "ambiguous_calendar",
            ],
            default="complete",
        )
        complete = reason == "complete"
        summary_hours = rows[
            [f"HHC4_NPC_HRSWEEK_TOC{kind}_{child}" for kind in range(1, 10)]
        ].to_numpy(dtype=float)
        regular_hours = summary_hours[:, :5].sum(axis=1)
        summary_known = (
            np.isfinite(summary_hours).all(axis=1)
            & (summary_hours >= 0).all(axis=1)
            & (summary_hours[:, 7] == 0)
            & (regular_hours <= 168)
            & (complete | rows.HH4_METH_QUEXVERSION.isin([2, 3]).to_numpy())
        )
        days = ece.reshape(-1, 7, 96).any(axis=2).sum(axis=1).astype(float)
        weekly_hours = ece.sum(axis=1) / 4
        regular_hours = np.where(complete, regular_ece.sum(axis=1) / 4, regular_hours)
        summary_known |= complete
        hours = np.divide(weekly_hours, days, out=np.zeros(len(rows)), where=days > 0)
        monthly = np.floor(days * 52 / 12 + 0.5)
        weight = pd.to_numeric(rows[f"HHC4_METH_WEIGHT_{child}"], errors="raise")
        if not np.isfinite(weight).all() or (weight <= 0).any():
            raise ValueError("Existing NSECE children require positive child weights.")
        part = pd.DataFrame(
            {
                "donor_id": [f"nsece2024:{case}:{child}" for case in rows.index],
                "source_household_id": rows.index.astype(str),
                "age": age.astype(int),
                "region": rows.HH4_REGION.to_numpy(),
                "parent_work_status": rows.HH4_PARWORK_STATUS.to_numpy(),
                "household_income": rows.HH4_ECON_INCOME_ANNUAL.to_numpy(),
                "income_band": childcare_income_band(rows.HH4_ECON_INCOME_ANNUAL),
                "questionnaire_version": rows.HH4_METH_QUEXVERSION.to_numpy(),
                "regular_hours_per_week": np.where(
                    summary_known, regular_hours, np.nan
                ),
                "irregular_hours_per_week": np.where(
                    complete, weekly_hours - regular_hours, np.nan
                ),
                "attendance_status": reason,
                "child_weight": weight.to_numpy(),
                "household_weight": rows.HH4_METH_WEIGHT.to_numpy(),
                "ece_provider_count": np.where(complete, provider_count, np.nan),
                "ece_hours_per_week": np.where(complete, weekly_hours, np.nan),
            }
        )
        for column, data in zip(
            US_CHILDCARE_ATTENDANCE_COLUMNS, (monthly, days, hours), strict=True
        ):
            part[column] = np.where(complete, data, np.nan)
        parts.append(part)
    if not parts:
        raise ValueError("NSECE source contains no children.")
    children = (
        pd.concat(parts, ignore_index=True)
        .sort_values("donor_id")
        .reset_index(drop=True)
    )
    return NSECEChildcareSource(
        children,
        Weights(children.child_weight.to_numpy(), WeightKind.DESIGN),
        {
            "study": "ICPSR39466.v1",
            "source_year": 2024,
            "measurement": "union_of_ECE_calendar_blocks_excluding_K8",
            "monthly_conversion": "floor(days_per_week * 52 / 12 + 0.5)",
            "national_representativeness_validated": False,
        },
    )


def load_nsece_childcare(
    household_path: str | Path, calendar_path: str | Path
) -> NSECEChildcareSource:
    """Load locally downloaded, hash-verified V1 TSV files; no automatic download."""
    tables = []
    receipts = []
    for path, expected_hash, columns in (
        (
            household_path,
            NSECE_2024_HOUSEHOLD_SHA256,
            nsece_childcare_household_columns(),
        ),
        (calendar_path, NSECE_2024_CALENDAR_SHA256, nsece_childcare_calendar_columns()),
    ):
        path = Path(path)
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected_hash:
            raise ValueError(f"NSECE source hash mismatch: {path.name}.")
        tables.append(
            pd.read_csv(path, sep="\t", usecols=list(columns), na_values=[" "])
        )
        receipts.append({"sha256": actual, "size_bytes": path.stat().st_size})
    result = derive_nsece_childcare(*tables)
    result.source_receipt["artifacts"] = receipts
    return result


def with_us_nsece_childcare_attendance(
    frame: Frame,
    source: NSECEChildcareSource,
    *,
    seed: int,
    match_columns: tuple[str, ...],
    fallback_match_columns: tuple[tuple[str, ...], ...] = (),
    sibling_dependence: float = 0.0,
) -> Frame:
    """Apply the source to a candidate Frame, preserving links, weights and receipts.

    Matching fields must already be harmonized on person rows. This function
    does not label all adults as parents or guess parental work from household
    earnings. Missing values outside ages 0--12 remain unresolved. The caller
    must pass ``assert_childcare_attendance_exportable`` before engine export.
    This is not registered in the default production build.
    """
    if frame.schema != US_SCHEMA:
        raise ValueError("NSECE childcare attendance requires the US schema.")
    donor, weights = source.donors()
    original_people = frame.table("person")
    recipients = original_people.copy()
    # Native BuildP IDs are exact int64, while the pure donor API uses strings.
    # Encode integers losslessly for hashing, then restore the native column.
    if "person_source_id" in recipients and pd.api.types.is_integer_dtype(
        recipients.person_source_id
    ):
        if recipients.person_source_id.isna().any():
            raise ValueError("Childcare source person IDs cannot be missing.")
        recipients["person_source_id"] = recipients.person_source_id.astype(str)
    people = impute_us_childcare_attendance(
        recipients,
        donor,
        donor_weights=weights,
        seed=seed,
        match_columns=match_columns,
        fallback_match_columns=fallback_match_columns,
        sibling_dependence=sibling_dependence,
    )
    # Float storage supports unresolved nulls in ordinary Frame checkpoints;
    # the monthly variable has already been validated to be integral when known.
    for column in US_CHILDCARE_ATTENDANCE_COLUMNS:
        people[column] = people[column].astype(float)
    people["person_source_id"] = original_people.person_source_id
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["person"] = people
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata={
            **frame.metadata,
            "nsece_childcare_attendance": {
                **source.source_receipt,
                "seed": int(seed),
                "match_columns": match_columns,
                "fallback_match_columns": fallback_match_columns,
                "sibling_dependence": sibling_dependence,
                "candidate_only": True,
            },
        },
    )


def assert_childcare_attendance_exportable(frame: Frame) -> None:
    """Reject unresolved inputs before an engine can coerce them to default zero.

    This is a completeness check, not statistical or publication approval.
    """
    people = frame.table("person")
    missing = [c for c in US_CHILDCARE_ATTENDANCE_COLUMNS if c not in people]
    if missing:
        raise ValueError(f"Childcare export is missing inputs: {missing}.")
    values = people[list(US_CHILDCARE_ATTENDANCE_COLUMNS)].to_numpy(
        dtype=float, na_value=np.nan
    )
    if not np.isfinite(values).all():
        raise ValueError(
            "Childcare export has unresolved attendance; do not fill with zero."
        )


def nsece_childcare_validation_report(
    source: NSECEChildcareSource,
    *,
    seed: int = 915,
    match_columns: tuple[str, ...] = NSECE_CHILDCARE_MATCH_COLUMNS,
) -> dict[str, object]:
    """Source attrition and a household-separated 20% holdout; never certification.

    Holdout households are excluded from the donor pool before any prediction.
    Source selection and this diagnostic's limited covariates remain explicit.
    Repeated use of this fixed holdout does not create fresh independent evidence.
    """
    children = source.children
    if children.attendance_status.eq("summary_bridge").any():
        raise ValueError(
            "Validate original calendars before bridge completion to avoid leakage."
        )
    donors, _ = source.donors()
    in_domain = children.age.between(0, 12)
    domain_mass = float(children.loc[in_domain, "child_weight"].sum())
    complete_mass = float(donors.child_weight.sum())
    split = donors.source_household_id.map(
        lambda value: (
            int.from_bytes(
                hashlib.sha256(f"{seed}:holdout:{value}".encode()).digest()[:8], "big"
            )
            % 5
            == 0
        )
    )
    train = donors.loc[~split].reset_index(drop=True)
    heldout = donors.loc[split].reset_index(drop=True)
    if train.empty or heldout.empty:
        raise ValueError("NSECE validation needs training and held-out households.")
    train_cells = set(train[list(match_columns)].itertuples(index=False, name=None))
    supported = np.array(
        [
            key in train_cells
            for key in heldout[list(match_columns)].itertuples(index=False, name=None)
        ]
    )
    observed = heldout.loc[supported].reset_index(drop=True)
    if observed.empty:
        raise ValueError("NSECE holdout has no supported matching cells.")
    recipients = observed.drop(columns=list(US_CHILDCARE_ATTENDANCE_COLUMNS)).assign(
        person_source_id=observed.donor_id
    )
    predicted = impute_us_childcare_attendance(
        recipients,
        train,
        donor_weights=Weights(train.child_weight.to_numpy(), WeightKind.DESIGN),
        match_columns=match_columns,
        seed=seed,
    )

    def metrics(table: pd.DataFrame, weights: np.ndarray) -> dict[str, float]:
        days = table[US_CHILDCARE_ATTENDANCE_COLUMNS[1]].to_numpy(dtype=float)
        hours = table[US_CHILDCARE_ATTENDANCE_COLUMNS[2]].to_numpy(dtype=float)
        return {
            "participation": float(np.average(days > 0, weights=weights)),
            "days_per_week": float(np.average(days, weights=weights)),
            "hours_per_week": float(np.average(days * hours, weights=weights)),
        }

    comparisons = []
    for grouping in (None, "age", "region", "parent_work_status"):
        groups = (
            [("all", observed.index)]
            if grouping is None
            else observed.groupby(grouping).groups.items()
        )
        for label, indices in groups:
            weights = observed.loc[indices, "child_weight"].to_numpy()
            comparisons.append(
                {
                    "grouping": grouping or "all",
                    "group": str(label),
                    "n": len(indices),
                    "observed": metrics(observed.loc[indices], weights),
                    "predicted": metrics(predicted.loc[indices], weights),
                }
            )
    attrition = [
        {
            "status": str(status),
            "n": len(group),
            "child_weight": float(group.child_weight.sum()),
        }
        for status, group in children.groupby("attendance_status", sort=True)
    ]
    return {
        "source": source.source_receipt,
        "source_child_count": len(children),
        "source_attrition": attrition,
        "under13_complete_weight_share": complete_mass / domain_mass,
        "seed": seed,
        "evaluation_design": "fixed household diagnostic split; not final population certification",
        "match_columns": list(match_columns),
        "training_children": len(train),
        "heldout_children": len(heldout),
        "unsupported_heldout_children": int((~supported).sum()),
        "unsupported_heldout_weight": float(
            heldout.loc[~supported, "child_weight"].sum()
        ),
        "household_overlap": len(
            set(train.source_household_id) & set(heldout.source_household_id)
        ),
        "comparisons": comparisons,
        "production_ready": False,
        "limitations": [
            "Only complete, unambiguous calendars enter this holdout; source selection remains unvalidated.",
            "The diagnostic matches age and caller-specified covariates; it is not a nationally validated fitted model.",
            "Sibling assignments, mixed providers, older-child care and scalar monthly conversion need validation.",
            "No full Microcosm population build or state CCDF distribution is certified by this source-only report.",
        ],
    }
