"""Pure all-age ASEC hours observations, without imputation or source authority.

HRSWK=0 is a source NIU code. Only coherent final nonwork for someone in the
work-experience universe supports a zero-hours completion. Children remain
unresolved here; an explicit modeled assumption is a separate operation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from . import current_survey_hours as common

PROTOCOL = "microcosm.us.asec-usual-hours-observation.v1"
COMPLETION_PROTOCOL = "microcosm.us.asec-usual-hours-completion.v1"
EARNINGS_FIELDS = ("WSAL_VAL", "SEMP_VAL", "FRSE_VAL")
SOURCE_FIELDS = (*common.ASEC_FIELDS, *EARNINGS_FIELDS)


def recode_asec_usual_hours(row: Mapping[str, str]) -> common.HoursProposal:
    """Preserve original annual work-history literals for March 2025 persons.

    Unlike empirical donor admission, observing an original source value does not
    require positive person weight. Earnings and last-week hours are not predictors
    or substitutes. Raw allocation/response status and the 99-plus code survive.
    """
    raw = common._literals(row, common.ASEC_FIELDS)
    key = common._asec_key(row)
    age = common._integer(row["A_AGE"], low=0, high=85)
    hours = common._integer(row["HRSWK"], low=0, high=99)
    weeks = common._integer(row["WKSWORK"], low=0, high=52)
    initial = common._integer(row["WORKYN"], low=0, high=2)
    temporary = common._integer(row["WTEMP"], low=0, high=2)
    final = common._integer(row["WRK_CK"], low=0, high=2)
    common._integer(row["MARSUPWT"], low=0, high=2**63 - 1)
    flags = common._asec_allocation_flags(row)

    all_niu = (hours, weeks, initial, temporary, final) == (0, 0, 0, 0, 0)
    if age < 15:
        common._require(all_niu, "ASEC_CHILD_WORK_UNIVERSE")
        output, provenance = None, "asec_source_universe_unavailable"
    elif all_niu:
        # Keep an unanswered/inapplicable record visible. This neither qualifies
        # a complete adult work history nor turns missing information into zero.
        output, provenance = None, "asec_work_history_unresolved"
    else:
        common._require(initial in (1, 2) and final in (1, 2), "ASEC_WORK_HISTORY")
        positive = hours > 0
        common._asec_work_consistency(hours, weeks, initial, temporary, final)
        output = float(hours)
        provenance = (
            "asec_positive_source_hours"
            if positive
            else "asec_source_nonwork_completion"
        )
    return common.HoursProposal(key, output, provenance, raw, flags)


def propose_asec_usual_hours(
    row: Mapping[str, str], *, under15_policy: str | None = None
) -> common.HoursProposal:
    """Complete a child only under an explicit zero policy and zero earnings.

    Earnings are conflict checks, never predictors or hours substitutes. Preserve
    their literal values alongside the source work history. Adult NIU remains a
    refusal in this complete-input proposal even though observation can describe
    it without imputing a value.
    """
    common._require(under15_policy in (None, common.UNDER15_POLICY), "UNDER15_POLICY")
    raw = common._literals(row, SOURCE_FIELDS)
    earnings = tuple(
        common._integer(row[name], low=-(2**63), high=2**63 - 1)
        for name in EARNINGS_FIELDS
    )
    proposal = replace(recode_asec_usual_hours(row), raw=raw)
    if proposal.hours is not None:
        return proposal
    age = common._integer(row["A_AGE"], low=0, high=85)
    common._require(age < 15, "UNRESOLVED_ADULT")
    common._require(under15_policy == common.UNDER15_POLICY, "UNDER15_POLICY_REQUIRED")
    common._require(all(value == 0 for value in earnings), "UNDER15_EARNINGS")
    return replace(
        proposal,
        hours=0.0,
        provenance="under15_explicit_modeled_zero",
        policy=under15_policy,
    )
