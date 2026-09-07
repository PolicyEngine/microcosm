"""Invented annual-record evidence and frozen diagnostic behavior contracts.

Expected quantities follow the accepted 02e definition and invented fixtures.
No frozen source is dynamically imported or caller rows authenticated here.
"""

from __future__ import annotations

from collections import defaultdict
from decimal import Decimal

import pytest

from microcosm.build import sipp_annual_panel as p
from tools.prepare_sipp_annual_panel_fixture import person_rows

SOURCE_SHA256 = "a" * 64
SSUID = "00000000000001"


def panel(rows):
    return p.Panel(
        {p.key(row): row for row in rows},
        tuple(sorted({row["SSUID"] for row in rows})),
        {},
        {},
    )


def roster(rows):
    """Supply explicit complete published rosters for invented groups."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["SSUID"], row["ERESIDENCEID"], row["MONTHCODE"]].append(row)
    for members in groups.values():
        members.sort(key=lambda row: int(row["PNUM"]))
        total = sum(
            (
                Decimal(row["TPTOTINC"])
                for row in members
                if row["TPTOTINC"] and row["APTOTINC"] != "0"
            ),
            Decimal(0),
        )
        for row in members:
            row["THTOTINC"] = str(total)
            row.update(RHNUMPER=str(len(members)), RHNUMPERWT2=str(len(members)))
            for slot in range(1, 17):
                other = members[slot - 1] if slot <= len(members) else None
                row[f"RREL_PNUM{slot}"] = other["PNUM"] if other else ""
                row[f"RREL{slot}"] = (
                    ("99" if other["PNUM"] == row["PNUM"] else "19") if other else ""
                )
    return rows


def child_rows(*, receipt="2", amount="", weight="0"):
    rows = person_rows(pnum=103, amount=amount, weight=weight)
    for row in rows:
        row.update(
            TAGE_EHC="10",
            RSSI_MNYN=receipt,
            ASSI_MNYN="1" if receipt else "0",
            APTOTINC="1" if amount else "0",
            TLMPAMT="",
            ALMPAMT="0",
            TROLLAMT="",
            AROLLAMT="0",
        )
    return rows


@pytest.mark.parametrize(
    ("amount", "weight", "field", "replacement", "cause", "sign"),
    [
        ("10", "2", None, None, None, "positive"),
        ("-5", "2", None, None, None, "negative"),
        ("0", "2", None, None, None, "zero"),
        ("10", "0", None, None, None, "positive"),
        ("10", "", None, None, None, "positive"),
        ("10", "2", "TPTOTINC", "", "missing_value", None),
        ("10", "2", "APTOTINC", "0", "structural_out_of_universe", None),
        ("10", "2", "APTOTINC", "", "missing_status", None),
        ("10", "2", "RIN_UNIV", "2", "outside_or_unknown_frame", None),
        ("10", "2", "MONTHCODE", None, "missing_contributor_months", None),
        ("10", "2", "APTOTINC", "9.0", None, "positive"),
        ("10", "2", "APTOTINC", "2", None, "positive"),
    ],
)
def test_records_preserve_frozen_annual_diagnostic_values(
    amount, weight, field, replacement, cause, sign
):
    rows = person_rows(amount=amount, weight=weight)
    if field == "MONTHCODE":
        rows.pop(0)
    elif field:
        rows[0][field] = replacement
    evidence = panel(rows)
    (record,) = p.annual_december_households(evidence, SOURCE_SHA256)
    result = p.diagnostics(evidence, SOURCE_SHA256)
    counts = {"december_households": 1}
    if weight == "":
        counts["unknown_weight_households"] = 1
    elif weight == "0":
        counts["zero_weight_households"] = 1
    total = {"10": "120", "-5": "-60", "0": "0"}[amount]
    weighted = {"10": "240", "-5": "-120", "0": "0"}[amount]
    if cause:
        counts["incomplete_households"] = 1
        total = weighted = None
        assert record.amount is None
        assert record.reasons == (cause,)
    else:
        counts["complete_households"] = 1
        counts[sign] = 1
        if weight != "":
            counts["complete_households_with_known_weight"] = 1
        weighted = None if weight == "" else "0" if weight == "0" else weighted
        assert record.amount == Decimal(total)
        assert record.reasons == ()
    assert result["annual_december_adult_contributors"] == {
        "counts": counts,
        "contributors": 1,
        "under15_members_retained": 0,
        "incomplete_causes": {cause: 1} if cause else {},
        "complete_household_sum": total,
        "complete_household_weighted_sum": weighted,
        "full_selected_household_sum": total,
        "weight_rule": "accepted_production_December_mean_all_live_members_including_zero_weights",
        "scope": "December_age15plus_persons_over_all_their_months_and_residences; under15_amounts_not_added",
    }
    # Conditional person fields remain separate and once per person.
    lump = result["annual_person_fields"]["TLMPAMT"]
    assert lump["observed_sum"] == "120"
    assert lump["observed_person_count"] == 1
    assert result["annual_person_fields"]["TROLLAMT"]["observed_sum"] == "50"
    assert result["panel"]["person_month_rows"] == (11 if field == "MONTHCODE" else 12)
    assert result["schema"] == "sipp-annual-panel-diagnostics/1"
    assert result["source"] == {"sha256": SOURCE_SHA256, "passes": [{}, {}]}


def test_mover_months_and_exact_child_membership_across_three_residences():
    stay = person_rows()
    mover = person_rows(pnum=102, amount="20")
    for row in mover:
        row["THHLDSTATUS"] = "4"
        if int(row["MONTHCODE"]) >= 7:
            row["ERESIDENCEID"] = "100002"
    child = child_rows()
    other = person_rows(ssuid="00000000000002", amount="-5")
    other[0]["APTOTINC"] = ""
    evidence = panel(roster(list(reversed(stay + mover + child + other))))
    records = p.annual_december_households(evidence, SOURCE_SHA256)
    assert tuple(record.key for record in records) == (
        (SSUID, 100001),
        (SSUID, 100002),
        ("00000000000002", 100001),
    )
    stay_record, mover_record, unavailable_record = records
    assert stay_record.members == ((SSUID, 101), (SSUID, 103))
    assert stay_record.december_person_months == ((SSUID, 101, 12), (SSUID, 103, 12))
    assert stay_record.contributors == ((SSUID, 101),)
    assert stay_record.children_under15 == ((SSUID, 103),)
    assert stay_record.amount == Decimal("120")
    assert stay_record.weight.value == Decimal("1")
    assert mover_record.members == ((SSUID, 102),)
    assert mover_record.amount == Decimal("240")
    assert mover_record.contributor_months_present == {
        (SSUID, 102): tuple(range(1, 13))
    }
    months = mover_record.contributor_months
    assert tuple(month.month for month in months) == tuple(range(1, 13))
    assert tuple(month.residence for month in months) == ((100001,) * 6 + (100002,) * 6)
    assert unavailable_record.amount is None
    assert unavailable_record.reasons == ("missing_status",)
    assert len(unavailable_record.contributor_months) == 12
    assert unavailable_record.contributor_months[0].income_status_code is None
    assert unavailable_record.contributor_months[0].amount is None
    assert unavailable_record.contributor_months[1].amount == Decimal("-5")


@pytest.mark.parametrize("code", range(1, 10))
def test_records_retain_reported_allocated_and_code9_statuses(code):
    rows = person_rows(amount="0")
    rows[0]["APTOTINC"] = str(code)
    (record,) = p.annual_december_households(panel(rows), SOURCE_SHA256)
    month = record.contributor_months[0]
    assert month.income_status_code == code
    assert month.income_state == "available"
    assert month.amount == Decimal(0)
    assert record.amount == Decimal("0")
    assert record.reasons == ()


@pytest.mark.parametrize("weights", [("2", "0"), ("0", "0"), ("", ""), ("2", "")])
def test_weights_keep_all_members_including_unknown_and_zero(weights):
    evidence = panel(
        roster(person_rows(weight=weights[0]) + child_rows(weight=weights[1]))
    )
    (record,) = p.annual_december_households(evidence, SOURCE_SHA256)
    expected = {("2", "0"): Decimal(1), ("0", "0"): Decimal(0)}.get(weights)
    assert record.weight.value == expected
    assert record.weight_state == (
        "unknown" if expected is None else "zero" if expected == 0 else "positive"
    )
    assert record.members == ((SSUID, 101), (SSUID, 103))
    assert record.amount == Decimal("120")


def test_unknown_child_receipt_skips_monthly_identity_without_erasing_child():
    evidence = panel(roster(person_rows() + child_rows(receipt="")))
    (record,) = p.annual_december_households(evidence, SOURCE_SHA256)
    result = p.diagnostics(evidence, SOURCE_SHA256)
    assert record.children_under15 == ((SSUID, 103),)
    assert record.amount == Decimal("120")
    assert record.reasons == ()
    assert result["panel"]["masks"]["unknown_child_receipt"] == 12
    assert result["monthly"]["12"]["identity_skipped_income_universe"] == 1
    assert "identity_tested" not in result["monthly"]["12"]


@pytest.mark.parametrize("defect", ["missing_child", "missing_self", "saturated_tail"])
def test_bad_december_membership_remains_unavailable_evidence(defect):
    rows = roster(person_rows() + child_rows())
    if defect == "missing_child":
        rows = [
            row
            for row in rows
            if not (row["PNUM"] == "103" and row["MONTHCODE"] == "12")
        ]
        expected = "named_member_row_missing"
    elif defect == "missing_self":
        rows[11].update(RREL_PNUM1="", RREL1="")
        expected = "missing_or_duplicate_self"
    else:
        for row in rows:
            if row["MONTHCODE"] == "12":
                row.update(RREL_PNUM16="60", RREL16="19", RHNUMPERWT2="3")
        expected = "suppressed_tail_unknown"
    (record,) = p.annual_december_households(panel(rows), SOURCE_SHA256)
    assert expected in record.membership_reasons
    assert f"december_{expected}" in record.reasons
    assert record.amount is None


def test_diagnostics_reduce_shared_records_from_one_validated_state(monkeypatch):
    evidence = panel(person_rows())
    original_state = p.panel_state
    original_records = p.annual_december_households
    seen_states, seen_records = [], []

    def state_spy(value):
        state = original_state(value)
        seen_states.append(state)
        return state

    def records_spy(value, sha256, *, state=None):
        assert state is seen_states[0]
        records = original_records(value, sha256, state=state)
        seen_records.append(records)
        return records

    def forbid_source_access(*args, **kwargs):
        pytest.fail("pure supplied-row calculation attempted source access")

    monkeypatch.setattr(p, "panel_state", state_spy)
    monkeypatch.setattr(p, "annual_december_households", records_spy)
    monkeypatch.setattr(p.production, "_open_regular", forbid_source_access)
    result = p.diagnostics(evidence, SOURCE_SHA256)
    assert len(seen_states) == len(seen_records) == 1
    (record,) = seen_records[0]
    annual = result["annual_december_adult_contributors"]
    assert annual["complete_household_sum"] == str(record.amount)


def test_records_keep_unknown_december_age_explicit():
    rows = person_rows()
    rows[-1]["TAGE_EHC"] = ""
    (record,) = p.annual_december_households(panel(rows), SOURCE_SHA256)
    assert record.members == record.unknown_age_members == ((SSUID, 101),)
    assert record.contributors == record.children_under15 == ()
    assert record.reasons == ("no_adult_contributors", "unknown_december_age")
    assert record.amount is None
