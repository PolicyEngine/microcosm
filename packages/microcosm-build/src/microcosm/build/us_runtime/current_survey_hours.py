"""Pure native-keyed usual-hours proposals, without source or release authority.

Callers supply original source literals, not engine-filled hours or clone rows.
This module performs no I/O, modifies no Frame and authenticates no donor cohort.
An enclosing source owner must bind actual bytes, complete coverage and roles.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np

PROTOCOL = "microcosm.us.native-usual-hours-proposal.v1"
AGE15_POLICY = "acs-age15-empirical-full-asec2025-v1"
UNDER15_POLICY = "us_hours_under15_zero_completion_v1"
SEED = 20260914
TARGET = "weekly_hours_worked_before_lsr"
ALLOCATION_FLAGS = ("I_HRSWK", "I_WKSWK", "I_WORKYN", "I_WTEMP", "FL_665")
ACS_FIELDS = ("SERIALNO", "SPORDER", "AGEP", "WKHP", "WKL", "FWKHP", "WAGP", "SEMP")
ASEC_FIELDS = (
    "PERIDNUM",
    "PH_SEQ",
    "A_LINENO",
    "A_AGE",
    "HRSWK",
    "WKSWORK",
    "WORKYN",
    "WTEMP",
    "WRK_CK",
    "MARSUPWT",
    *ALLOCATION_FLAGS,
)


def _require(condition, reason):
    if not condition:
        raise ValueError("NATIVE_HOURS_" + reason)


def _literals(row, fields):
    _require(isinstance(row, Mapping), "ROW_TYPE")
    _require(all(name in row for name in fields), "SOURCE_FIELDS")
    values = tuple((name, row[name]) for name in fields)
    _require(
        all(type(value) is str and len(value) <= 64 for _, value in values),
        "SOURCE_LITERAL",
    )
    return values


def _integer(token, *, low, high, nullable=False):
    _require(type(token) is str and len(token) <= 64, "SOURCE_LITERAL")
    stripped = token.strip()
    if nullable and not stripped:
        return None
    _require(re.fullmatch(r"-?[0-9]+", stripped, re.ASCII) is not None, "INTEGER_CODE")
    value = int(stripped)
    _require(low <= value <= high, "CODE_DOMAIN")
    return value


@dataclass(frozen=True, order=True)
class NativeHoursKey:
    survey: str
    survey_year: int
    income_year: int
    household: str
    person: str
    line: int

    def __post_init__(self):
        _require(
            type(self.survey) is str
            and type(self.household) is str
            and type(self.person) is str,
            "KEY_TYPE",
        )
        _require(
            type(self.survey_year) is int
            and type(self.income_year) is int
            and type(self.line) is int,
            "KEY_TYPE",
        )
        _require(
            1 <= self.line <= 20 and self.income_year == 2024, "KEY_PERIOD_OR_LINE"
        )
        if self.survey == "acs":
            _require(
                self.survey_year == 2024
                and self.person == ""
                and re.fullmatch(r"2024(?:HU|GQ)[0-9]{7}", self.household, re.ASCII)
                is not None,
                "ACS_KEY",
            )
        else:
            _require(
                self.survey == "asec"
                and self.survey_year == 2025
                and re.fullmatch(r"[1-9][0-9]{0,4}", self.household, re.ASCII)
                is not None
                and re.fullmatch(r"[0-9]{22}", self.person, re.ASCII) is not None,
                "ASEC_KEY",
            )


@dataclass(frozen=True)
class HoursProposal:
    key: NativeHoursKey
    hours: float | None
    provenance: str
    raw: tuple[tuple[str, str], ...]
    allocation_flags: tuple[tuple[str, int | None], ...]
    policy: str | None = None
    donor_key: NativeHoursKey | None = None
    donor_provenance: str | None = None
    donor_allocation_flags: tuple[tuple[str, int | None], ...] = ()
    unknown_earnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class HoursProposalBatch:
    proposals: tuple[HoursProposal, ...]
    age15_policy: str | None
    under15_policy: str | None
    supplied_donors: int
    # These are scope statements, not capabilities or source attestations.
    source_authenticated: bool = False
    complete_donor_cohort_authenticated: bool = False
    release_qualified: bool = False


def _acs_key(row):
    serial = row["SERIALNO"]
    _require(
        re.fullmatch(r"2024(?:HU|GQ)[0-9]{7}", serial, re.ASCII) is not None, "ACS_KEY"
    )
    line = _integer(row["SPORDER"], low=1, high=20)
    return NativeHoursKey("acs", 2024, 2024, serial, "", line)


def _asec_key(row):
    person = row["PERIDNUM"]
    _require(re.fullmatch(r"[0-9]{22}", person, re.ASCII) is not None, "ASEC_KEY")
    household = _integer(row["PH_SEQ"], low=1, high=99999)
    line = _integer(row["A_LINENO"], low=1, high=20)
    return NativeHoursKey("asec", 2025, 2024, str(household), person, line)


def recode_acs_usual_hours(row: Mapping[str, str]) -> HoursProposal:
    """Preserve raw usual hours, source nonwork, and survey-universe absence.

    All fields are explicit; a missing header is not a blank answer. Unknown
    AGEP cannot qualify an original native roster and therefore refuses here.
    """
    raw = _literals(row, ACS_FIELDS)
    key = _acs_key(row)
    age = _integer(row["AGEP"], low=0, high=99)
    hours = _integer(row["WKHP"], low=1, high=99, nullable=True)
    work = _integer(row["WKL"], low=1, high=3, nullable=True)
    allocation = _integer(row["FWKHP"], low=0, high=1, nullable=True)
    earnings = {
        name: _integer(row[name], low=-(2**63), high=2**63 - 1, nullable=True)
        for name in ("WAGP", "SEMP")
    }
    _require(not (age < 16 and (hours is not None or work is not None)), "ACS_UNIVERSE")
    _require(not (hours is not None and work in (2, 3)), "ACS_WORK_CONTRADICTION")
    if hours is not None:
        provenance = {
            0: "acs_native_hours_not_allocated",
            1: "acs_native_hours_allocated",
            None: "acs_native_hours_allocation_unknown",
        }[allocation]
    elif work in (2, 3):
        hours, provenance = 0, "acs_source_nonwork_completion"
    else:
        provenance = "acs_source_universe_unavailable" if age < 16 else "acs_unresolved"
    return HoursProposal(
        key,
        None if hours is None else float(hours),
        provenance,
        raw,
        (("FWKHP", allocation),),
        unknown_earnings=tuple(
            name for name, value in earnings.items() if value is None
        ),
    )


def recode_asec_age15_hours(row: Mapping[str, str]) -> HoursProposal:
    """Recode a supplied March2025 age15 donor, without qualifying its source.

    This narrow donor routine is not the ASEC all-age engine-input producer.
    Initial WORKYN2 alone is not final nonwork. Allocation flags stay distinct
    from the later modeled ACS result.
    """
    raw = _literals(row, ASEC_FIELDS)
    key = _asec_key(row)
    _integer(row["A_AGE"], low=15, high=15)
    hours = _integer(row["HRSWK"], low=0, high=99)
    weeks = _integer(row["WKSWORK"], low=0, high=52)
    initial = _integer(row["WORKYN"], low=1, high=2)
    temporary = _integer(row["WTEMP"], low=0, high=2)
    final = _integer(row["WRK_CK"], low=1, high=2)
    _integer(row["MARSUPWT"], low=1, high=2**63 - 1)
    flags = []
    for name in ALLOCATION_FLAGS:
        value = _integer(row[name], low=0, high=9)
        _require(
            value in ((1, 2, 3) if name == "FL_665" else (0, 1, 9)), "ASEC_ALLOCATION"
        )
        flags.append((name, value))
    positive = hours > 0
    _require(positive == (weeks > 0) == (final == 1), "ASEC_WORK_CONTRADICTION")
    _require(initial != 1 or positive, "ASEC_WORK_CONTRADICTION")
    _require(temporary != 1 or positive, "ASEC_WORK_CONTRADICTION")
    _require(
        not (initial == 2 and temporary == 2 and positive), "ASEC_WORK_CONTRADICTION"
    )
    return HoursProposal(
        key,
        float(hours),
        "asec_positive_source_hours" if positive else "asec_source_nonwork_completion",
        raw,
        tuple(flags),
    )


def native_hours_uniform(key: NativeHoursKey) -> float:
    """A 53-bit [0,1) draw keyed by original native identity, never row rank.

    This version intentionally need not reproduce the old dense-parent draws.
    Public proposal construction accepts only keys derived from source literals.
    """
    _require(type(key) is NativeHoursKey, "KEY_TYPE")
    payload = json.dumps(
        [
            PROTOCOL,
            AGE15_POLICY,
            SEED,
            "annual_hours",
            key.survey,
            key.survey_year,
            key.income_year,
            key.household,
            key.person,
            key.line,
        ],
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    return (
        int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") >> 11
    ) / 2**53


def propose_acs_usual_hours(
    rows: Sequence[Mapping[str, str]],
    *,
    donors: Sequence[Mapping[str, str]] = (),
    age15_policy: str | None = None,
    under15_policy: str | None = None,
) -> HoursProposalBatch:
    """Return proposals for original ACS persons; no I/O, authority or mutation.

    Donors are a supplied, unqualified cohort. The enclosing owner must prove
    the full pinned2174-person cohort; cardinality alone cannot prove that.
    The original-source key determines draws, so a later clone adapter must
    transport this proposal through ancestry rather than redraw for each clone.
    """
    _require(age15_policy in (None, AGE15_POLICY), "AGE15_POLICY")
    _require(under15_policy in (None, UNDER15_POLICY), "UNDER15_POLICY")
    _require(0 < len(rows) <= 4_000_000 and len(donors) <= 3000, "ROW_BOUND")
    proposals = [recode_acs_usual_hours(row) for row in rows]
    _require(
        len({p.key for p in proposals}) == len(proposals), "DUPLICATE_RECIPIENT_KEY"
    )
    donor_proposals = [recode_asec_age15_hours(row) for row in donors]
    _require(
        len({p.key for p in donor_proposals}) == len(donor_proposals),
        "DUPLICATE_DONOR_KEY",
    )
    _require(
        len({p.key.person for p in donor_proposals}) == len(donor_proposals),
        "DUPLICATE_ASEC_PERSON",
    )
    _require(
        len({(p.key.household, p.key.line) for p in donor_proposals})
        == len(donor_proposals),
        "DUPLICATE_ASEC_COORDINATE",
    )
    ordered = sorted(donor_proposals, key=lambda p: (p.hours, p.key))
    cumulative = None
    if ordered:
        # Same weighted empirical CDF arithmetic as the reviewed pure helper;
        # ties now use true native identity rather than historical source rank.
        weights = np.asarray(
            [int(dict(p.raw)["MARSUPWT"]) for p in ordered], dtype=float
        )
        weights /= 100.0  # Preserve the reviewed MARSUPWT unit conversion order.
        weights /= np.max(weights)
        cumulative = np.cumsum(weights / weights.sum())
        cumulative[-1] = 1.0
    output = []
    for proposal in proposals:
        if proposal.hours is not None:
            output.append(proposal)
            continue
        raw = dict(proposal.raw)
        age = int(raw["AGEP"])
        if age < 15:
            _require(under15_policy == UNDER15_POLICY, "UNDER15_POLICY_REQUIRED")
            wage = _integer(raw["WAGP"], low=-(2**63), high=2**63 - 1, nullable=True)
            self_employment = _integer(
                raw["SEMP"], low=-(2**63), high=2**63 - 1, nullable=True
            )
            _require(
                not (wage is not None and wage > 0) and self_employment in (None, 0),
                "UNDER15_EARNINGS",
            )
            output.append(
                replace(
                    proposal,
                    hours=0.0,
                    provenance="under15_explicit_modeled_zero",
                    policy=under15_policy,
                )
            )
        elif age == 15:
            _require(
                age15_policy == AGE15_POLICY and cumulative is not None,
                "AGE15_DONORS_REQUIRED",
            )
            index = int(
                np.searchsorted(
                    cumulative, native_hours_uniform(proposal.key), side="right"
                )
            )
            donor = ordered[index]
            output.append(
                replace(
                    proposal,
                    hours=donor.hours,
                    provenance="age15_weighted_empirical_draw",
                    policy=age15_policy,
                    donor_key=donor.key,
                    donor_provenance=donor.provenance,
                    donor_allocation_flags=donor.allocation_flags,
                )
            )
        else:
            raise ValueError("NATIVE_HOURS_UNRESOLVED_ADULT")
    return HoursProposalBatch(tuple(output), age15_policy, under15_policy, len(donors))
