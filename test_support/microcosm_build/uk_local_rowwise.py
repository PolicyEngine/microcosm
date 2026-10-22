"""The three-household importance-weighted clone frame and its area
assignment, shared by the rowwise local solve tests, the full-graph
calibration tests and the terminal-node tests."""

from __future__ import annotations

from datetime import date

import pandas as pd

from microcosm.build.uk_runtime import uk_national_frame
from microcosm.build.uk_runtime.local_rowwise import (
    UK_LOCAL_BINDING_ADJUDICATION_REGISTER_RESOURCE,
)
from microcosm.build.uk_runtime.weighted_integrity import (
    load_uk_reviewed_exclusion_register,
)
from microcosm.frame import WeightKind


def _committed_binding_now() -> date:
    """The latest approval in the committed local binding adjudications.

    A rowwise solve in a test reviews the register on this date rather than
    today, so the calendar cannot fail a test on a later day (microcosm#1278).
    The review is a build's review on that day: a fence a fixture binds that
    has lapsed by then still fails, and a dormant one does not, so a partial
    renewal is judged as a build would judge it. Nothing is asserted here,
    so a register change cannot break the collection of every module that
    imports this. ``test_no_committed_local_adjudication_is_about_to_lapse``
    keeps the calendar's warning; the expiry tests pass their own dates.
    """

    records = load_uk_reviewed_exclusion_register(
        None, resource=UK_LOCAL_BINDING_ADJUDICATION_REGISTER_RESOURCE
    ).values()
    return date.fromisoformat(max(record.approved_on for record in records))


COMMITTED_BINDING_NOW = _committed_binding_now()


def _clone_frame(weights=(1.0, 1.0, 1.0)):
    return uk_national_frame(
        person=pd.DataFrame(
            {
                "person_id": [1, 2, 3],
                "person_household_id": [101, 102, 103],
                "person_benunit_id": [11, 12, 13],
            }
        ),
        benunit=pd.DataFrame({"benunit_id": [11, 12, 13]}),
        household=pd.DataFrame(
            {
                "household_id": [101, 102, 103],
                "household_weight": list(weights),
            }
        ),
        time_period="2023",
        weight_kind=WeightKind.IMPORTANCE,
    )


def _assigned() -> pd.Series:
    return pd.Series(["E001", "E001", "S001"], index=[101, 102, 103])


__all__ = [name for name in globals() if not name.startswith("__")]
