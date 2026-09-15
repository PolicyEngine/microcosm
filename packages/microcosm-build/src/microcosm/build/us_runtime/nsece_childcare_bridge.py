"""Complete noncalendar schedules conditional on measured regular-care hours.

NSECE HH-10/11 describes the summer/fall instruments: regular weekly hours are
observed, days and irregular arrangements are not collected. Preserve those
hours and impute days/irregular intensity jointly from a compatible calendar.
Zero regular hours do not establish zero irregular care. No source observation
is replaced; completed rows have separate status and donor provenance.
"""

from __future__ import annotations

import hashlib

import numpy as np

from microcosm.build.us_runtime.childcare_attendance import (
    US_CHILDCARE_ATTENDANCE_COLUMNS,
    childcare_attendance_contract,
)
from microcosm.build.us_runtime.nsece_childcare import (
    NSECE_CHILDCARE_FALLBACK_COLUMNS,
    NSECE_CHILDCARE_MATCH_COLUMNS,
    NSECEChildcareSource,
)


def bridge_nsece_noncalendar_attendance(
    source: NSECEChildcareSource, *, seed: int
) -> NSECEChildcareSource:
    """Use nearest regular-hour donors, retaining all ties at the cutoff.

    All matching levels require the same regular-care participation status.
    Hours cannot exceed 24 per attended day or 168 per week. The parent source
    remains unchanged. Synthetic validation must mask whole households before
    this function is called; it never learns from completed bridge records.
    """
    children = source.children.copy()
    nearest_donors = childcare_attendance_contract()["operations"][1]["nearest_donors"]
    pool = (
        children.loc[
            children.attendance_status.eq("complete")
            & children.regular_hours_per_week.notna()
        ]
        .sort_values("donor_id")
        .copy()
    )
    pool["has_regular_care"] = pool.regular_hours_per_week > 0
    target = children.loc[
        children.age.between(0, 12)
        & children.attendance_status.eq("missing_calendar")
        & children.questionnaire_version.isin([2, 3])
        & children.regular_hours_per_week.notna()
    ]
    unsupported = 0
    month, days_column, hours_column = US_CHILDCARE_ATTENDANCE_COLUMNS
    for index, child in target.iterrows():
        candidates = pool.iloc[:0]
        for level in (NSECE_CHILDCARE_MATCH_COLUMNS, *NSECE_CHILDCARE_FALLBACK_COLUMNS):
            mask = pool.has_regular_care.eq(child.regular_hours_per_week > 0)
            for column in level:
                mask &= pool[column].eq(child[column])
            candidates = pool.loc[
                mask
                & (
                    (pool.irregular_hours_per_week + child.regular_hours_per_week)
                    <= 168
                )
            ]
            if not candidates.empty:
                break
        if candidates.empty:
            unsupported += 1
            continue
        candidates = candidates.assign(
            distance=np.abs(
                np.log1p(candidates.regular_hours_per_week)
                - np.log1p(child.regular_hours_per_week)
            )
        )
        candidates = candidates.sort_values(["distance", "donor_id"])
        cutoff = candidates.distance.iloc[min(nearest_donors, len(candidates)) - 1]
        candidates = candidates.loc[candidates.distance <= cutoff]
        weights = candidates.child_weight.to_numpy()
        cumulative = np.cumsum(weights / weights.sum())
        digest = hashlib.sha256(
            f"{seed}:nsece_bridge:{child.donor_id}".encode()
        ).digest()
        draw = (int.from_bytes(digest[:8], "big") >> 11) / 2**53
        donor = candidates.iloc[
            min(
                int(np.searchsorted(cumulative, draw, side="right")),
                len(candidates) - 1,
            )
        ]
        total_hours = child.regular_hours_per_week + donor.irregular_hours_per_week
        days = max(float(donor[days_column]), float(np.ceil(total_hours / 24)))
        if total_hours == 0:
            days = 0.0
        children.loc[index, [month, days_column, hours_column]] = [
            np.floor(days * 52 / 12 + 0.5),
            days,
            total_hours / days if days else 0.0,
        ]
        children.loc[index, "ece_hours_per_week"] = total_hours
        children.loc[index, "irregular_hours_per_week"] = donor.irregular_hours_per_week
        children.loc[index, "attendance_status"] = "summary_bridge"
        children.loc[index, "schedule_bridge_donor"] = donor.donor_id
        children.loc[index, "schedule_bridge_match"] = ",".join(level)
    completed = children.attendance_status.eq("summary_bridge")
    return NSECEChildcareSource(
        children,
        source.weights,
        {
            **source.source_receipt,
            "noncalendar_bridge": {
                "seed": seed,
                "completed_children": int(completed.sum()),
                "unsupported_children": unsupported,
                "observed": "regular weekly hours from May/fall questionnaire",
                "imputed": "attended days and irregular care jointly from nearest regular-hour calendar donors",
                "assumption": "conditional calendar pattern and irregular care transfer across questionnaire instruments",
            },
        },
    )
