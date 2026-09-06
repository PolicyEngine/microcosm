"""Invented-only behavioral tests for the two-pass annual panel boundary."""

from __future__ import annotations

import hashlib
import io
import json
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

import pytest

from microcosm.build import sipp_annual_panel as p
from tools.prepare_sipp_annual_panel_fixture import person_rows, write_rows


def regroup(rows):
    groups = defaultdict(list)
    for row in rows:
        if row["THHLDSTATUS"] in ("1", "2", "3", "4"):
            groups[row["SSUID"], row["ERESIDENCEID"], row["MONTHCODE"]].append(row)
    for members in groups.values():
        total = sum(
            (
                Decimal(r["TPTOTINC"])
                for r in members
                if r["TPTOTINC"] and r["APTOTINC"] != "0"
            ),
            Decimal(0),
        )
        for row in members:
            row.update(
                RHNUMPER=str(len(members)),
                RHNUMPERWT2=str(len(members)),
                THTOTINC=str(total),
            )
            for i in range(1, 17):
                other = members[i - 1] if i <= len(members) else None
                row[f"RREL_PNUM{i}"] = other["PNUM"] if other else ""
                row[f"RREL{i}"] = (
                    ("99" if other["PNUM"] == row["PNUM"] else "19") if other else ""
                )
    return rows


def run(tmp_path, rows):
    path = tmp_path / "invented.psv"
    write_rows(path, rows)
    raw = path.read_bytes()
    return p.extract(path, hashlib.sha256(raw).hexdigest(), len(raw))


def annual(result):
    return result["annual_december_adult_contributors"]


def child_rows(age="10", receipt="2", amount="", weight="0"):
    rows = person_rows(pnum=102, amount=amount, weight=weight)
    for row in rows:
        row.update(
            TAGE_EHC=age,
            RSSI_MNYN=receipt,
            TPTOTINC=amount,
            APTOTINC="1" if amount else "0",
            TLMPAMT="",
            ALMPAMT="0",
            TROLLAMT="",
            AROLLAMT="0",
        )
    return rows


def test_first32_is_lexical_amount_blind_and_complete(tmp_path):
    rows = [
        r
        for i in reversed(range(1, 41))
        for r in person_rows(
            ssuid=f"{i:014d}", amount="10" if i <= 32 else "DO_NOT_INTERPRET"
        )
    ]
    result = run(tmp_path, rows)
    assert result["selection"]["selected_ssuid_count"] == 32
    assert result["source"]["passes"][0]["eligible_ssuids"] == 40
    assert [x["rows"] for x in result["source"]["passes"]] == [480, 480]
    assert result["panel"]["person_month_rows"] == 384
    assert annual(result)["full_selected_household_sum"] == "3840"
    assert annual(result)["complete_household_weighted_sum"] == "7680"
    assert "00000000000001" not in json.dumps(result)


def test_exactly_two_streams_and_id_only_pass_a(tmp_path, monkeypatch):
    original, calls = p.stream_rows, []

    def spy(path, fields, audit):
        calls.append(fields)
        return original(path, fields, audit)

    monkeypatch.setattr(p, "stream_rows", spy)
    run(tmp_path, person_rows())
    assert calls == [p.INDEX_FIELDS, p.FIELDS]
    assert not set(p.INDEX_FIELDS) & {"THTOTINC", "TPTOTINC", "WPFINWGT", "TLMPAMT"}


@pytest.mark.parametrize("month", [1, 12])
def test_duplicate_rows_refuse(tmp_path, month):
    rows = person_rows()
    rows.append(dict(rows[month - 1]))
    with pytest.raises(p.PanelError, match="DUPLICATE"):
        run(tmp_path, rows)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [("TLMPAMT", "121"), ("AROLLAMT", "2"), ("THHLDSTATUS", "4"), ("TROLLAMT", "")],
)
def test_person_grain_conflicts_refuse(tmp_path, field, replacement):
    rows = person_rows()
    rows[5][field] = replacement
    with pytest.raises(p.PanelError, match="PERSON_GRAIN_VALUE_CONFLICT"):
        run(tmp_path, rows)


def test_missing_adult_month_is_not_zero(tmp_path):
    result = run(tmp_path, person_rows()[1:])
    assert annual(result)["full_selected_household_sum"] is None
    assert annual(result)["complete_household_sum"] is None
    assert annual(result)["incomplete_causes"]["missing_contributor_months"] == 1


def test_pre15_months_explicitly_block_full_annual_income(tmp_path):
    rows = person_rows()
    for row in rows[:6]:
        row.update(TAGE_EHC="14", TPTOTINC="", APTOTINC="0", THTOTINC="0")
    for row in rows[6:]:
        row["TAGE_EHC"] = "15"
    result = run(tmp_path, rows)
    assert result["panel"]["masks"]["pre15_without_ssi"] == 6
    assert annual(result)["incomplete_causes"]["pre15_without_ssi"] == 1
    assert annual(result)["full_selected_household_sum"] is None


def test_pre15_inconsistent_status_refuses(tmp_path):
    rows = person_rows()
    rows[0]["TAGE_EHC"] = "14"
    with pytest.raises(p.PanelError, match="PRE15_INCOME_UNIVERSE_CONFLICT"):
        run(tmp_path, rows)


def test_child_ssi_in_monthly_identity_not_adult_annual_sum(tmp_path):
    result = run(tmp_path, regroup(person_rows() + child_rows(receipt="1", amount="3")))
    assert result["monthly"]["1"]["identity_equal"] == 1
    assert annual(result)["full_selected_household_sum"] == "120"
    assert annual(result)["under15_members_retained"] == 1
    assert result["panel"]["person_month_rows"] == 24


def test_newborn_does_not_impose_twelve_month_requirement(tmp_path):
    newborn = child_rows(age="0")[9:]
    for row in newborn:
        row.update(RSSI_MNYN="", ASSI_MNYN="0")
    result = run(tmp_path, regroup(person_rows() + newborn))
    assert result["panel"]["person_month_rows"] == 15
    assert annual(result)["full_selected_household_sum"] == "120"
    assert result["monthly"]["12"]["identity_skipped_income_universe"] == 1
    assert annual(result)["complete_household_weighted_sum"] == "120"


def test_mover_keeps_all_months_and_both_residences(tmp_path):
    mover = person_rows(pnum=102, amount="20")
    for row in mover:
        row["THHLDSTATUS"] = "4"
        if int(row["MONTHCODE"]) >= 7:
            row["ERESIDENCEID"] = "100002"
    result = run(tmp_path, regroup(person_rows() + mover))
    assert result["monthly"]["1"]["identity_equal"] == 1
    assert result["monthly"]["12"]["identity_equal"] == 2
    assert annual(result)["counts"]["december_households"] == 2
    assert annual(result)["full_selected_household_sum"] == "360"


def test_person_without_december_still_read_for_monthly_identity(tmp_path):
    leaver = person_rows(pnum=102, amount="20")[:6]
    for row in leaver:
        row["THHLDSTATUS"] = "3"
    result = run(tmp_path, regroup(person_rows() + leaver))
    assert result["panel"]["person_month_rows"] == 18
    assert result["monthly"]["1"]["identity_equal"] == 1
    assert annual(result)["full_selected_household_sum"] == "120"


def test_type2_counts_references_without_unique_owner_or_amounts(tmp_path):
    rows = regroup(person_rows() + person_rows(pnum=102))
    for row in rows:
        row.update(ET2_LNO1="60", RREL_PNUM3="60", RREL3="19", RHNUMPERWT2="3")
    result = run(tmp_path, rows)
    assert result["type2"]["respondent_month_rows_with_references"] == 24
    assert result["type2"]["respondent_month_slot_occurrences"] == 24
    assert result["type2"]["unique_person_count"] is None
    assert result["type2"]["annual_income_sum"] is None
    assert not any(f.startswith("TT2INC") for f in p.FIELDS)
    assert result["monthly"]["1"]["identity_equal"] == 1


def test_zero_weight_members_remain_in_production_mean(tmp_path):
    result = run(tmp_path, regroup(person_rows(weight="2") + child_rows(weight="0")))
    assert annual(result)["complete_household_weighted_sum"] == "120"


def test_all_zero_weight_does_not_remove_ssuid(tmp_path):
    result = run(tmp_path, person_rows(weight="0"))
    assert result["selection"]["selected_ssuid_count"] == 1
    assert annual(result)["counts"]["zero_weight_households"] == 1
    assert annual(result)["complete_household_weighted_sum"] == "0"


def test_negative_income_is_preserved(tmp_path):
    result = run(tmp_path, person_rows(amount="-5"))
    assert annual(result)["full_selected_household_sum"] == "-60"
    assert annual(result)["counts"]["negative"] == 1
    assert result["monthly"]["1"]["identity_equal"] == 1


def test_rollover_greater_than_gross_diagnostic_once_per_person(tmp_path):
    rows = person_rows()
    for row in rows:
        row["TROLLAMT"] = "240"
    result = run(tmp_path, rows)
    assert result["rollover_greater_than_gross_persons"] == 1
    assert result["annual_person_fields"]["TROLLAMT"]["observed_sum"] == "240"
    assert result["annual_person_fields"]["TLMPAMT"]["observed_sum"] == "120"
    assert annual(result)["full_selected_household_sum"] == "120"


@pytest.mark.parametrize("status", list(range(1, 10)))
def test_exact_status_codes_retained_with_unknown_component_quality(tmp_path, status):
    rows = person_rows()
    for row in rows:
        row["APTOTINC"] = str(status)
    result = run(tmp_path, rows)
    assert result["status_masks"]["APTOTINC"]["counts"][str(status)] == 12
    assert annual(result)["full_selected_household_sum"] == "120"
    assert "quality_unknown" in result["status_code_meanings"]["9"]


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("APTOTINC", "", "missing_status"),
        ("TPTOTINC", "", "missing_value"),
        ("APTOTINC", "0", "structural_out_of_universe"),
        ("RIN_UNIV", "2", "outside_or_unknown_frame"),
    ],
)
def test_incomplete_universes_never_fill_zero(tmp_path, field, value, reason):
    rows = person_rows()
    rows[0][field] = value
    result = run(tmp_path, rows)
    assert annual(result)["full_selected_household_sum"] is None
    assert annual(result)["incomplete_causes"][reason] == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("APTOTINC", "-99"),
        ("TPTOTINC", ".N"),
        ("TPTOTINC", "NaN"),
        ("WPFINWGT", "-1"),
        ("RREL1", "20"),
    ],
)
def test_unrecognized_tokens_and_domains_refuse(tmp_path, field, value):
    rows = person_rows()
    rows[0][field] = value
    with pytest.raises(p.PanelError):
        run(tmp_path, rows)


def test_saturated_last_live_slot_is_unknown_not_proof(tmp_path):
    rows = regroup([r for i in range(101, 117) for r in person_rows(pnum=i)])
    result = run(tmp_path, rows)
    assert result["monthly"]["1"]["suppressed_tail_unknown"] == 1
    assert "identity_tested" not in result["monthly"]["1"]
    assert annual(result)["full_selected_household_sum"] is None


def test_singleton_includes_self(tmp_path):
    result = run(tmp_path, person_rows())
    assert result["monthly"]["1"]["identity_equal"] == 1


def test_missing_self_is_not_manufactured(tmp_path):
    rows = person_rows()
    rows[0].update(RREL_PNUM1="", RREL1="")
    result = run(tmp_path, rows)
    assert result["monthly"]["1"]["missing_or_duplicate_self"] == 1
    assert "identity_tested" not in result["monthly"]["1"]


def test_missing_named_member_skips_identity(tmp_path):
    rows = regroup(person_rows() + person_rows(pnum=102))
    del rows[12]
    result = run(tmp_path, rows)
    assert result["monthly"]["1"]["named_member_row_missing"] == 1
    assert "identity_tested" not in result["monthly"]["1"]


def test_conflicting_household_totals_skip_identity(tmp_path):
    rows = regroup(person_rows() + person_rows(pnum=102))
    rows[12]["THTOTINC"] = "999"
    result = run(tmp_path, rows)
    assert (
        result["monthly"]["1"]["identity_skipped_household_total_conflict_or_missing"]
        == 1
    )


@pytest.mark.parametrize("status", ["5", "6"])
def test_inactive_person_rows_are_retained_outside_monthly_groups(tmp_path, status):
    inactive = person_rows(pnum=102)
    for row in inactive:
        for field in p.FIELDS:
            if field not in (
                "SSUID",
                "PNUM",
                "MONTHCODE",
                "THHLDSTATUS",
                "WPFINWGT",
                "TLMPAMT",
                "ALMPAMT",
                "TROLLAMT",
                "AROLLAMT",
            ):
                row[field] = (
                    "0"
                    if field in p.CONTRACT["fields"]
                    and "flag_for" in p.CONTRACT["fields"][field]
                    else ""
                )
        row["THHLDSTATUS"] = status
    result = run(tmp_path, person_rows() + inactive)
    assert result["panel"]["person_month_rows"] == 24
    assert result["panel"]["native_codes"]["THHLDSTATUS"][status] == 12
    assert annual(result)["full_selected_household_sum"] == "120"


def test_wrong_source_digest_refuses_before_panel(tmp_path):
    source = tmp_path / "invented.psv"
    write_rows(source, person_rows())
    with pytest.raises(p.PanelError, match="SOURCE_PIN_MISMATCH"):
        p.read_panel(source, "0" * 64, len(source.read_bytes()))


def test_unbound_source_refuses_before_open_or_stat(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("unbound source was accessed")

    monkeypatch.setattr(p.production, "_open_regular", forbidden)
    monkeypatch.setattr(Path, "stat", forbidden)
    with pytest.raises(p.PanelError, match="UNBOUND"):
        p.read_panel(Path("/UNBOUND/source.psv"), None, None)


def test_mutation_between_passes_refuses(tmp_path, monkeypatch):
    source = tmp_path / "invented.psv"
    write_rows(source, person_rows())
    raw = source.read_bytes()
    original = p.stream_rows
    calls = 0

    def mutate(path, fields, audit):
        nonlocal calls
        calls += 1
        if calls == 2:
            path.write_bytes(raw.replace(b"|120|", b"|121|"))
        yield from original(path, fields, audit)

    monkeypatch.setattr(p, "stream_rows", mutate)
    with pytest.raises(p.PanelError, match="SOURCE_PIN_MISMATCH"):
        p.read_panel(source, hashlib.sha256(raw).hexdigest(), len(raw))


def test_production_reader_rejects_bad_physical_records():
    for raw in (b"\xef\xbb\xbfabc\n", b"abc\x00\n", b"\n"):
        with pytest.raises(p.production.SourceError):
            p.production._line(io.BytesIO(raw))


def test_public_contract_records_all_five_corrections():
    decisions = p.CONTRACT["decisions"]
    assert "99" in decisions["self_membership"]
    assert "Unknown scope match" in decisions["acs_child_ssi"]
    assert "conditional" in decisions["T2"]
    assert "Six groups" in decisions["status_partition"]
    assert "No TT2INC" in decisions["type2"]


def test_annual_completeness_not_gated_by_old_household_members(tmp_path):
    leaver = person_rows(pnum=102, amount="20")[:6]
    for row in leaver:
        row["THHLDSTATUS"] = "3"
    rows = regroup(person_rows() + leaver)
    del rows[12]
    result = run(tmp_path, rows)
    assert result["monthly"]["1"]["named_member_row_missing"] == 1
    assert annual(result)["full_selected_household_sum"] == "120"


def test_unanchored_december_member_discloses_incomplete_total(tmp_path):
    rows = person_rows() + person_rows(pnum=102)
    for row in rows[12:]:
        row.update(ERESIDENCEID="", ARESIDENCEID="0")
    result = run(tmp_path, rows)
    assert result["panel"]["masks"]["unanchored_december_members"] == 1
    assert annual(result)["full_selected_household_sum"] is None


def test_decimal_status_lexemes_retain_native_integer_codes(tmp_path):
    rows = person_rows()
    rows[0]["APTOTINC"] = "9.0"
    result = run(tmp_path, rows)
    assert result["status_masks"]["APTOTINC"]["counts"]["9"] == 1


def test_runner_unbound_declaration_refuses_without_source_access(
    tmp_path, monkeypatch
):
    import sys

    from tools import run_sipp_annual_panel as runner

    declaration = tmp_path / "binding.json"
    declaration.write_text(
        '{"schema":"sipp-annual-panel-binding/1","source_path":null}'
    )
    output = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        ["runner", "--binding", str(declaration), "--output-dir", str(output)],
    )
    monkeypatch.setattr(
        p.production, "_open_regular", lambda *a: pytest.fail("source open")
    )
    with pytest.raises(ValueError, match="UNBOUND"):
        runner.main()
    assert not output.exists()


def test_runner_requires_explicit_flags_not_environment_only(monkeypatch):
    import sys

    from tools.run_sipp_annual_panel import bytecode_flags

    monkeypatch.setattr(sys, "orig_argv", ["python", "-I", "script.py"])
    with pytest.raises(ValueError, match="EXPLICIT_B"):
        bytecode_flags()


def test_runner_refuses_existing_cache_prefix(tmp_path, monkeypatch):
    import sys

    from tools.run_sipp_annual_panel import bytecode_flags

    monkeypatch.setattr(sys, "pycache_prefix", str(tmp_path))
    with pytest.raises(ValueError, match="FRESH_ABSOLUTE"):
        bytecode_flags()


def runner_declaration(tmp_path):
    declaration = tmp_path / "binding.json"
    declaration.write_text(
        json.dumps(
            {
                "schema": "sipp-annual-panel-binding/1",
                "source_kind": "invented",
                "survey_year": 2023,
                "income_year": 2022,
                "source_path": "/UNBOUND/never-opened",
                "source_sha256": "a" * 64,
                "source_size": 1,
            }
        )
    )
    return declaration


def test_runner_no_overwrite_before_child_launch(tmp_path, monkeypatch):
    import sys

    from tools import run_sipp_annual_panel as runner

    declaration = runner_declaration(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    (output / "keep").write_text("unchanged")
    monkeypatch.setattr(
        sys,
        "argv",
        ["runner", "--binding", str(declaration), "--output-dir", str(output)],
    )
    monkeypatch.setattr(
        runner.subprocess, "Popen", lambda *a, **k: pytest.fail("child launched")
    )
    with pytest.raises(FileExistsError):
        runner.main()
    assert (output / "keep").read_text() == "unchanged"


@pytest.mark.parametrize("rss", [0, 2 * 1024**3])
def test_runner_self_rss_refuses_missing_or_excessive_reading(monkeypatch, rss):
    import sys
    from types import SimpleNamespace

    from tools import run_sipp_annual_panel as runner

    monkeypatch.setattr(
        runner.resource,
        "getrusage",
        lambda *a: SimpleNamespace(
            ru_maxrss=rss // (1 if sys.platform == "darwin" else 1024)
        ),
    )
    with pytest.raises(
        ValueError, match="RSS_MONITOR_UNAVAILABLE" if rss == 0 else "RSS_CEILING"
    ):
        runner.check_rss()


def test_runner_kills_child_on_wall_ceiling(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    from tools import run_sipp_annual_panel as runner

    declaration = runner_declaration(tmp_path)
    output = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        ["runner", "--binding", str(declaration), "--output-dir", str(output)],
    )

    def wait(timeout=None):
        if timeout is not None:
            raise runner.subprocess.TimeoutExpired("invented", timeout)
        return -9

    process = SimpleNamespace(pid=123, wait=wait)
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: process)
    killed = []
    monkeypatch.setattr(runner.os, "killpg", lambda *a: killed.append(a))
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2 and killed
    receipt = json.loads((output / "execution-receipt.json").read_text())
    assert receipt["refusal"] == "WALL_CEILING" and receipt["status"] == "refused"
    assert "-B" in receipt["child_command"] and "-X" in receipt["child_command"]
