"""Invented-source checks of the authenticating December boundary."""

import fcntl
import hashlib
import json
import os
from dataclasses import replace
from decimal import ROUND_UP, Decimal, Inexact, localcontext

import pytest

from microcosm.build import sipp_december_source as producer
from microcosm.graph.canonical import canonical_json
from microcosm.graph.store import ContentStore


def row(ssuid="10", person="101", residence="100001", **changes):
    values = dict.fromkeys(producer.FIELDS, "1")
    values.update(
        SSUID=ssuid,
        PNUM=person,
        MONTHCODE="12",
        ERESIDENCEID=residence,
        WPFINWGT="1.25",
        TVAL_BANK="0",
        TAGE_EHC="15",
    )
    values.update(changes)
    return values


def source_file(tmp_path, monkeypatch, rows=None, *, raw=None, extra_header=()):
    fields = (*producer.FIELDS, *extra_header)
    if raw is None:
        records = ["|".join(fields)]
        for values in rows or [row()]:
            records.append("|".join(values.get(f, "") for f in fields))
        raw = ("\n".join(records) + "\n").encode()
    path = tmp_path / "invented-source.psv"
    path.write_bytes(raw)
    pins = replace(
        producer._PINS,
        sha256=hashlib.sha256(raw).hexdigest(),
        size=len(raw),
        expected_counts=None,
    )
    monkeypatch.setattr(producer, "_PINS", pins)
    monkeypatch.setattr(producer, "_MIN_RAM", 0)
    store = ContentStore(tmp_path / "store")
    return path, store


def test_whole_ssuid_residences_unknown_and_zero_weights_roundtrip(
    tmp_path, monkeypatch
):
    path, store = source_file(
        tmp_path,
        monkeypatch,
        [
            row("2", "101", "100001"),
            row(
                "10",
                "102.0",
                "100002",
                WPFINWGT="0",
                TVAL_BANK="",
                AVAL_BANK="0",
                TAGE_EHC="14",
            ),
            row("10", "101", "100001", TVEH_NUM="", AVEH_NUM="0"),
        ],
    )
    made = producer.produce_sipp_december_source(
        path, store, selection="first_32_ssuids"
    )
    loaded = producer.load_sipp_december_source(
        path, store, made.receipt_key, expected_selection="first_32_ssuids"
    )
    assert loaded == made
    assert [h.key for h in made.households] == [
        ("10", 100001),
        ("10", 100002),
        ("2", 100001),
    ]
    assert len(made.bank) == 3 and made.bank[1].value is None
    assert made.households[1].weight.value == 0
    receipt = json.loads(made.receipt_json)
    assert receipt["full_counts"]["december_people"] == 3
    assert receipt["full_counts"]["multi_residence_ssuids"] == 1
    assert receipt["full_counts"]["zero_weight_december_people"] == 1
    assert not list((store.root / producer._NAMESPACE).glob("snapshot-*/parent.bin"))


@pytest.mark.parametrize("bad", ["\n", "a|b\n", "|extra\n"])
def test_full_header_width_and_blank_records_refuse(tmp_path, monkeypatch, bad):
    header = "|".join((*producer.FIELDS, "NONPROJECTED")) + "\n"
    valid = "|".join(row().get(f, "") for f in (*producer.FIELDS, "NONPROJECTED"))
    body = valid + bad if bad == "|extra\n" else bad
    path, store = source_file(tmp_path, monkeypatch, raw=(header + body).encode())
    with pytest.raises(producer.SourceError):
        producer.produce_sipp_december_source(path, store, selection="all_ssuids")
    assert list((store.root / producer._NAMESPACE).glob("snapshot-*/parent.bin"))


@pytest.mark.parametrize(
    "mutation", ["bom", "utf8", "duplicate_header", "bad_month", "duplicate_key"]
)
def test_lexical_and_identity_refusals(tmp_path, monkeypatch, mutation):
    header = "|".join(producer.FIELDS)
    values = row()
    if mutation == "bad_month":
        values["MONTHCODE"] = ""
    body = "|".join(values[f] for f in producer.FIELDS)
    raw = (header + "\n" + body + "\n").encode()
    if mutation == "bom":
        raw = b"\xef\xbb\xbf" + raw
    if mutation == "utf8":
        raw += b"\xff\n"
    if mutation == "duplicate_header":
        raw = raw.replace(b"SSUID|PNUM", b"SSUID|SSUID", 1)
    if mutation == "duplicate_key":
        raw += (body.replace("|101|", "|101.0|") + "\n").encode()
    path, store = source_file(tmp_path, monkeypatch, raw=raw)
    with pytest.raises(producer.SourceError):
        producer.produce_sipp_december_source(path, store, selection="first_32_ssuids")


def test_selection_string_order_complete_ssuids_and_invalid_outside_selection(
    tmp_path, monkeypatch
):
    rows = [row(str(n), "101", "100001") for n in range(1, 35)]
    rows += [row("1", "102", "100002")]
    path, store = source_file(tmp_path, monkeypatch, rows)
    result = producer.produce_sipp_december_source(
        path, store, selection="first_32_ssuids"
    )
    expected = sorted({r["SSUID"] for r in rows})[:32]
    assert sorted({h.key[0] for h in result.households}) == expected
    assert sum(h.key[0] == "1" for h in result.households) == 2
    with pytest.raises(producer.SourceError):
        producer.load_sipp_december_source(
            path, store, result.receipt_key, expected_selection="all_ssuids"
        )
    rows[-1] = row("zzzz", "invalid", "100001")
    path, store = source_file(tmp_path, monkeypatch, rows)
    with pytest.raises(producer.SourceError):
        producer.produce_sipp_december_source(path, store, selection="first_32_ssuids")


def test_quotes_are_literal_not_csv_quoting(tmp_path, monkeypatch):
    path, store = source_file(tmp_path, monkeypatch, [row('"10"', TVAL_BANK='"0"')])
    result = producer.produce_sipp_december_source(path, store, selection="all_ssuids")
    assert result.households[0].key[0] == '"10"'
    assert result.bank[0].value is None


def test_actual_store_collision_and_forged_receipt_refuse(tmp_path, monkeypatch):
    path, store = source_file(tmp_path, monkeypatch)
    result = producer.produce_sipp_december_source(path, store, selection="all_ssuids")
    object_path = store.object_path(result.receipt_key)
    receipt_path = object_path / "payload.json"
    changed = json.loads(receipt_path.read_bytes())
    changed["selected_counts"]["december_people"] += 1
    data = canonical_json(changed)
    receipt_path.write_bytes(data)
    meta_path = object_path / "meta.json"
    meta = json.loads(meta_path.read_bytes())
    meta["payloads"]["payload.json"].update(
        size=len(data), sha256=hashlib.sha256(data).hexdigest()
    )
    meta_path.write_bytes(canonical_json(meta))
    with pytest.raises(producer.SourceError):
        producer.load_sipp_december_source(
            path, store, result.receipt_key, expected_selection="all_ssuids"
        )
    with pytest.raises(producer.SourceError):
        producer.produce_sipp_december_source(path, store, selection="all_ssuids")


def test_changed_parent_and_nonregular_original_refuse(tmp_path, monkeypatch):
    path, store = source_file(tmp_path, monkeypatch)
    result = producer.produce_sipp_december_source(path, store, selection="all_ssuids")
    path.write_bytes(path.read_bytes().replace(b"1.25", b"1.26"))
    with pytest.raises(producer.SourceError):
        producer.load_sipp_december_source(
            path, store, result.receipt_key, expected_selection="all_ssuids"
        )
    link = tmp_path / "link"
    link.symlink_to(path)
    with pytest.raises(producer.SourceError):
        producer.produce_sipp_december_source(link, store, selection="all_ssuids")
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(producer.SourceError):
        producer.produce_sipp_december_source(fifo, store, selection="all_ssuids")


def test_hostile_decimal_context_and_source_order_do_not_change_digest(
    tmp_path, monkeypatch
):
    rows = [
        row(person=str(n), WPFINWGT="1" if n == 101 else "0") for n in (101, 102, 103)
    ]
    path, store = source_file(tmp_path, monkeypatch, rows)
    first = producer.produce_sipp_december_source(path, store, selection="all_ssuids")
    with localcontext() as context:
        context.prec = 2
        context.rounding = ROUND_UP
        context.traps[Inexact] = True
        context.capitals = 0
        second = producer.produce_sipp_december_source(
            path, store, selection="all_ssuids"
        )
    assert first == second
    assert first.households[0].weight.value == Decimal("0." + "3" * 80)


def test_bounds_and_extra_store_payload_refuse_before_generic_read(
    tmp_path, monkeypatch
):
    path, store = source_file(tmp_path, monkeypatch)
    result = producer.produce_sipp_december_source(path, store, selection="all_ssuids")
    obj = store.object_path(result.receipt_key)
    (obj / "extra").write_bytes(b"bad")
    with pytest.raises(producer.SourceError):
        producer.load_sipp_december_source(
            path, store, result.receipt_key, expected_selection="all_ssuids"
        )
    (obj / "extra").unlink()
    monkeypatch.setattr(producer, "_META_MAX", 8)
    with pytest.raises(producer.SourceError):
        producer.load_sipp_december_source(
            path, store, result.receipt_key, expected_selection="all_ssuids"
        )


def test_short_row_missing_only_unprojected_field_refuses(tmp_path, monkeypatch):
    raw = (
        "|".join((*producer.FIELDS, "OUTSIDE_PROJECTION"))
        + "\n"
        + "|".join(row()[f] for f in producer.FIELDS)
        + "\n"
    ).encode()
    path, store = source_file(tmp_path, monkeypatch, raw=raw)
    with pytest.raises(producer.SourceError, match="ROW_WIDTH"):
        producer.produce_sipp_december_source(path, store, selection="all_ssuids")


def test_unknown_member_weight_keeps_complete_household_and_known_denominator(
    tmp_path, monkeypatch
):
    path, store = source_file(
        tmp_path, monkeypatch, [row(), row(person="102", WPFINWGT="")]
    )
    result = producer.produce_sipp_december_source(path, store, selection="all_ssuids")
    assert len(result.households[0].members) == 2
    assert result.households[0].weight.value is None
    profile = json.loads(result.receipt_json)["aggregates"]
    assert profile["person_weights"] == {
        "count": 2,
        "known_denominator": 1,
        "unknown_count": 1,
        "zero_count": 0,
        "known_weight_sum": "1.25",
    }
    assert profile["household_weights"]["known_denominator"] == 0


def test_same_projection_under_row_permutation_binds_different_parent(
    tmp_path, monkeypatch
):
    rows = [row("2"), row("10", "101"), row("10", "102")]
    path, store = source_file(tmp_path, monkeypatch, rows)
    first = producer.produce_sipp_december_source(path, store, selection="all_ssuids")
    first_projection = store.load_bytes(first.projection_key)
    path, store = source_file(tmp_path, monkeypatch, list(reversed(rows)))
    second = producer.produce_sipp_december_source(path, store, selection="all_ssuids")
    assert first_projection == store.load_bytes(second.projection_key)
    assert first.producer_binding != second.producer_binding


def test_named_single_writer_lock_refuses_and_is_reusable(tmp_path, monkeypatch):
    path, store = source_file(tmp_path, monkeypatch)
    namespace = store.root / producer._NAMESPACE
    namespace.mkdir()
    with (namespace / "producer.lock").open("wb") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(producer.SourceError, match="PRODUCER_BUSY"):
            producer.produce_sipp_december_source(path, store, selection="all_ssuids")
    producer.produce_sipp_december_source(path, store, selection="all_ssuids")


def test_receipt_last_interruption_and_retained_failure_accounting(
    tmp_path, monkeypatch
):
    path, store = source_file(tmp_path, monkeypatch)
    actual = store.put_json

    def interrupt(key, payload, **kwargs):
        if kwargs.get("kind") == "sipp_december_source_v1":
            raise OSError("invented interruption")
        return actual(key, payload, **kwargs)

    monkeypatch.setattr(store, "put_json", interrupt)
    with pytest.raises(producer.SourceError):
        producer.produce_sipp_december_source(path, store, selection="all_ssuids")
    metas = [
        json.loads(p.read_bytes()) for p in store.root.glob("objects/*/*/meta.json")
    ]
    assert len(metas) == 1 and metas[0]["kind"] == "bytes"
    assert metas[0]["node_key"] != metas[0]["key"]
    monkeypatch.setattr(store, "put_json", actual)
    made = producer.produce_sipp_december_source(path, store, selection="all_ssuids")
    info = store.load_json(made.run_information_key, kind="sipp_december_run_info_v1")
    preflight = info["resource_preflight"]
    assert preflight["retained_snapshot_count"] == 1
    assert preflight["retained_snapshot_bytes"] == path.stat().st_size
    assert preflight["occupied_snapshot_bytes_subtracted_again"] is False
    assert info["receipt_key"] == made.receipt_key


def test_omitted_member_and_forged_projection_metadata_refuse(tmp_path, monkeypatch):
    path, store = source_file(tmp_path, monkeypatch, [row(), row(person="102")])
    made = producer.produce_sipp_december_source(path, store, selection="all_ssuids")
    obj = store.object_path(made.projection_key)
    body = json.loads((obj / "payload.bin").read_bytes())
    body["rows"].pop()
    data = canonical_json(body)
    (obj / "payload.bin").write_bytes(data)
    meta = json.loads((obj / "meta.json").read_bytes())
    meta["payloads"]["payload.bin"].update(
        size=len(data), sha256=hashlib.sha256(data).hexdigest()
    )
    (obj / "meta.json").write_bytes(canonical_json(meta))
    with pytest.raises(producer.SourceError, match="PROJECTION_CHANGED"):
        producer.load_sipp_december_source(
            path, store, made.receipt_key, expected_selection="all_ssuids"
        )


def test_actual_bounded_copy_refuses_growth_and_fifo_without_streaming(
    tmp_path, monkeypatch
):
    source = tmp_path / "invented-binary"
    source.write_bytes(b"abc")
    real_open = producer._open_regular

    def altered(path, flags=os.O_RDONLY):
        descriptor = real_open(path, flags)
        if path == source:
            source.write_bytes(b"abcdef")
        return descriptor

    monkeypatch.setattr(producer, "_open_regular", altered)
    with pytest.raises(producer.SourceError):
        producer._copy_regular(source, tmp_path / "owned", 3)
    fifo = tmp_path / "invented-fifo"
    os.mkfifo(fifo)
    with pytest.raises(producer.SourceError, match="NONREGULAR_FILE"):
        producer._copy_regular(fifo, tmp_path / "owned-fifo", 10)


def test_source_selection_function_receives_only_identity_keys(tmp_path, monkeypatch):
    path, store = source_file(
        tmp_path, monkeypatch, [row(TVAL_BANK="99999", WPFINWGT="0")]
    )
    actual = producer._selected_ssuids
    seen = []

    def checked(keys, selection):
        assert keys == (("10", 100001),)
        seen.append(keys)
        return actual(keys, selection)

    monkeypatch.setattr(producer, "_selected_ssuids", checked)
    producer.produce_sipp_december_source(path, store, selection="first_32_ssuids")
    assert len(seen) == 2


def test_duplicate_person_month_across_parser_chunks_refuses(tmp_path, monkeypatch):
    path, _ = source_file(tmp_path, monkeypatch, [row(), row(person="101.0")])
    rows, parser = producer._parse(path, chunksize=1)
    assert parser["physical_records"] == 2
    source = producer.m.SippSource(
        producer._PINS.sha256, producer._PINS.metadata_sha256, 2023, 2022
    )
    with pytest.raises(producer.m.MeasurementError):
        producer.m.december_households(rows, source=source, complete_membership=True)


def test_valid_nondecember_month_does_not_claim_person_or_residence_audit(
    tmp_path, monkeypatch
):
    path, store = source_file(
        tmp_path,
        monkeypatch,
        [row(), row("other", "invalid", "invalid", MONTHCODE="11")],
    )
    made = producer.produce_sipp_december_source(path, store, selection="all_ssuids")
    assert len(made.households) == 1
    assert json.loads(made.receipt_json)["parser"]["month_records"] == {
        "11": 1,
        "12": 1,
    }
