"""Invented-only source authority and explicitly ungranted native consistency."""

import csv
import hashlib
import importlib.util
import io
import json
import os
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from microcosm.build.us_runtime import acs_housing_universe_source as custody
from microcosm.build.us_runtime import acs_person_coverage_authentication as owner
from microcosm.build.us_runtime import acs_person_coverage_columns as literal
from microcosm.build.us_runtime.acs_inputs import map_acs_native_inputs
from microcosm.build.us_runtime.acs_pums import AcsPumsSource, build_acs_pums_unit_frame


def _helper(filename):
    spec = importlib.util.spec_from_file_location(
        "coverage_helpers_" + filename, Path(__file__).with_name(filename + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_authority = _helper("test_us_acs_housing_source")
Member = _authority.Member
build_fixture = _authority.build_fixture
flip_archive_byte = _authority.flip_archive_byte
write_archive = _authority.write_archive
_native = _helper("test_us_acs_pums")
_household, _person = _native._household, _native._person


def _csv(rows, *, bom=False, ending="\n"):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=list(rows[0]), quoting=csv.QUOTE_ALL, lineterminator=ending
    )
    writer.writeheader()
    writer.writerows(rows)
    return (("\ufeff" if bom else "") + stream.getvalue()).encode()


@pytest.fixture
def invented(tmp_path, monkeypatch):
    monkeypatch.setattr(
        custody.shutil, "disk_usage", lambda _p: SimpleNamespace(free=64 * 1024**3)
    )
    h = [_household("2024HU0000001", NP=2), _household("2024HU0000002", NP=1)]
    p = [
        _person("2024HU0000001", 1, 20, AGEP=30, MIL="1", ESR="4"),
        _person("2024HU0000001", 2, 25, AGEP=15, MIL="", ESR=""),
        _person("2024HU0000002", 1, 20, AGEP=80, MIL="4", ESR="6"),
    ]
    fixture = build_fixture(
        tmp_path,
        monkeypatch,
        household_members=(Member("psam_husa.csv", _csv(h)),),
        person_members=(
            Member("psam_pusa.csv", _csv(p[:1])),
            Member("psam_pusb.csv", _csv(p[1:])),
        ),
    )
    frame = map_acs_native_inputs(
        build_acs_pums_unit_frame(
            AcsPumsSource(fixture.household_zip, fixture.person_zip)
        )[0]
    ).frame
    return fixture, frame, h, p


def _load(invented, **kwargs):
    fixture, frame, _h, _p = invented
    return owner.load_authenticated_acs_person_coverage(
        fixture.source_dir, snapshot_root=fixture.snapshot_root, frame=frame, **kwargs
    )


def _rewrite(
    invented, *, persons=None, households=None, members=None, bom=False, ending="\n"
):
    fixture, _frame, h, p = invented
    if households is not None:
        write_archive(
            fixture.household_zip, (Member("psam_husa.csv", _csv(households)),)
        )
    if persons is not None or members is not None:
        p = persons if persons is not None else p
        write_archive(
            fixture.person_zip,
            members
            if members is not None
            else (
                Member("psam_pusa.csv", _csv(p[:1], bom=bom, ending=ending)),
                Member("psam_pusb.csv", _csv(p[1:], bom=bom, ending=ending)),
            ),
        )
    fixture.pin()


def _refuses(code):
    return pytest.raises(owner.ACSCoverageAuthenticationError, match="^" + code + "$")


def test_source_authentication_preserves_low_receipt_and_parent_without_population_grant(
    invented,
):
    fixture, frame, _h, _p = invented
    before = custody.frame_content_sha256(frame)
    original_tables = {e: frame.table(e).copy(deep=True) for e in frame.entities}
    result = _load(invented)
    table, receipt = literal.read_acs_person_coverage_columns(
        AcsPumsSource(fixture.household_zip, fixture.person_zip),
        person_keys=owner._native_roster(frame)[0],
        chunksize=3,
    )
    pd.testing.assert_frame_equal(result.table, table)
    assert result.receipt["original_literal_receipt"] == receipt
    assert receipt["source_authenticated"] is False
    assert result.receipt["source_authenticated"] is True
    assert result.native_binding.receipt["population_binding_authenticated"] is False
    assert (
        result.native_binding.receipt["missing_authority"]
        == "closed_prepared_acs_native_population"
    )
    assert (
        result.native_binding.receipt["original_age"]["relation"] == "numeric_identity"
    )
    for flag in (
        "domain_assignment_authenticated",
        "period_harmonized",
        "release_eligible",
        "cross_survey_coverage_equivalence_established",
        "population_binding_authenticated",
    ):
        assert result.receipt[flag] is False
    assert custody.frame_content_sha256(frame) == before
    for entity, original in original_tables.items():
        pd.testing.assert_frame_equal(frame.table(entity), original)
    assert (
        owner.verify_acs_coverage_native_consistency(result, frame)
        == result.native_binding
    )
    assert [m["rows"] for m in result.receipt["members"]["person"]] == [1, 2]
    assert all(len(m["sha256"]) == 64 for m in result.receipt["members"]["person"])


def test_first_frame_hash_precedes_the_roster_snapshot(invented, monkeypatch):
    """The returned binding must describe one state, including at hash entry."""
    _fixture, frame, _h, _p = invented
    original_hash = custody.frame_content_sha256
    first = True

    def at_hash_entry(value):
        nonlocal first
        if first:
            first = False
            value.person.loc[:, "source_row_id"] += 100
        return original_hash(value)

    monkeypatch.setattr(custody, "frame_content_sha256", at_hash_entry)
    result = _load(invented)
    assert (
        owner.verify_acs_coverage_native_consistency(result, frame)
        == result.native_binding
    )


def test_mutation_during_roster_snapshot_refuses_issuance(invented, monkeypatch):
    original_roster = owner._native_roster
    first = True

    def after_roster_read(value):
        nonlocal first
        result = original_roster(value)
        if first:
            first = False
            value.person.loc[:, "source_row_id"] += 100
        return result

    monkeypatch.setattr(owner, "_native_roster", after_roster_read)
    with _refuses("NATIVE_FRAME_CHANGED"):
        _load(invented)


def test_closed_default_manifest_has_no_caller_pin_or_prepared_authority(
    invented, monkeypatch
):
    fixture, frame, _h, _p = invented
    pins = custody._ARCHIVE_PINS
    monkeypatch.setattr(custody, "_ARCHIVE_PINS", None)
    calls = []

    def default_manifest(*args, **kwargs):
        assert not args and not kwargs
        calls.append(True)
        return SimpleNamespace(
            artifacts=tuple(
                SimpleNamespace(role=r, filename=n, sha256=d, size_bytes=s)
                for r, n, d, s in pins
            )
        )

    monkeypatch.setattr(custody, "load_acs_source_manifest", default_manifest)
    result = _load(invented)
    assert len(calls) == 2
    with pytest.raises(TypeError):
        _load(invented, manifest={"sha256": "0" * 64})
    for constructor, payload in (
        (owner.AuthenticatedACSPersonCoverage, result.payload),
        (owner.UngrantedACSNativeBinding, result.native_binding.receipt_json),
    ):
        with _refuses("SOURCE_CONSTRUCTOR|BINDING_CONSTRUCTOR"):
            constructor(payload)
    for forged in (
        frame.person,
        {"frame": frame, "frame_sha256": custody.frame_content_sha256(frame)},
        "0" * 64,
    ):
        with _refuses("NATIVE_FRAME_REQUIRED"):
            owner.load_authenticated_acs_person_coverage(
                fixture.source_dir, snapshot_root=fixture.snapshot_root, frame=forged
            )


@pytest.mark.parametrize(
    "field,value,code",
    [
        ("SPORDER", 3, "NATIVE_ALIASES"),
        ("A_LINENO", 3, "NATIVE_ALIASES"),
        ("source_person_id", "2", "NATIVE_ALIASES"),
        ("source_household_id", 2, "NATIVE_ALIASES"),
        ("source_year", 2023, "NATIVE_VINTAGE"),
        ("person_household_id", 2, "NATIVE_ALIASES"),
        ("source_row_id", 1, "NATIVE_IDS"),
    ],
)
def test_native_alias_and_membership_changes_with_same_person_ids_refuse(
    invented, field, value, code
):
    invented[1].person.loc[0, field] = value
    with _refuses(code):
        _load(invented)


@pytest.mark.parametrize(
    "change", ["serial", "age", "membership", "weight", "source_row"]
)
def test_live_binding_refuses_same_ids_changed_native_parent(invented, change):
    _fixture, frame, _h, _p = invented
    result = _load(invented)
    if change == "serial":
        h = frame.table("household")
        h["SERIALNO"] = h.SERIALNO.iloc[::-1].to_numpy()
    elif change == "age":
        frame.person.loc[2, "age"] = 82
    elif change == "membership":
        for c in ("person_household_id", "source_household_id"):
            frame.person.loc[[1, 2], c] = frame.person.loc[[2, 1], c].to_numpy()
    elif change == "weight":
        frame.table("household")["extra_cell"] = (
            123  # all parent cells bind, including additions
        )
    else:
        frame.person.loc[0, "source_row_id"] = 100
    with _refuses(
        "NATIVE_FRAME_CHANGED|NATIVE_HOUSEHOLD_COUNT|NATIVE_CONSISTENCY_REFUSED"
    ):
        owner.verify_acs_coverage_native_consistency(result, frame)


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "extra",
        "duplicate",
        "household_np",
        "household_missing",
        "wrong_key",
        "wrong_vintage",
    ],
)
def test_source_complete_household_mismatch_refuses(invented, change):
    _fixture, _frame, h, p = invented
    if change == "missing":
        p = p[:2]
    elif change == "extra":
        p += [{**p[1], "SPORDER": 3}]
    elif change == "duplicate":
        p[1] = p[0]
    elif change == "household_np":
        h[0]["NP"] = 1
    elif change == "household_missing":
        h = h[:1]
    elif change == "wrong_vintage":
        p[0]["SERIALNO"] = "2023HU0000001"
    else:
        p[0]["SPORDER"] = 3
    _rewrite(invented, persons=p, households=h)
    with _refuses("SOURCE_PERSON_ROSTER|SOURCE_HOUSEHOLD_ROSTER|SOURCE_DUPLICATE_KEY"):
        _load(invented)


@pytest.mark.parametrize(
    "age,mil,esr",
    [
        ("", "", ""),
        ("NA", "NA", "1.0"),
        ("30 ", " 1", "4 "),
        ("30", '1,"quoted"\r\n\t', "4\r\n\t"),
        ("30\t", "1", "4"),
        ("100", "0", "7"),
    ],
)
@pytest.mark.parametrize("ending", ["\n", "\r", "\r\n"])
def test_bom_multiline_tabs_and_malformed_literals_survive_envelope(
    invented, age, mil, esr, ending
):
    _fixture, frame, _h, p = invented
    p[0].update(AGEP=age, MIL=mil, ESR=esr)
    _rewrite(invented, persons=p, bom=True, ending=ending)
    result = _load(invented)
    assert result.table.loc[0, ["AGEP", "MIL", "ESR"]].tolist() == [age, mil, esr]
    assert not result.table.isna().any().any()
    assert frame.person.loc[0, "age"] == 30
    if age != "30":
        assert result.table.loc[0, "MIL_state"] == "age_unresolved"
        assert result.native_binding.receipt["original_age"]["relation"] == "unproven"


@pytest.mark.parametrize(
    "column,value", [("AGEP", 82), ("A_AGE", 82), ("age", 82), ("age", float("nan"))]
)
def test_model_age_never_fills_or_authenticates_original_relation(
    invented, column, value
):
    frame = invented[1]
    if isinstance(value, float):
        frame.person[column] = frame.person[column].astype("float64")
    frame.person.loc[2, column] = value
    result = _load(invented)
    assert result.table.loc[2, "AGEP"] == "80"
    assert result.native_binding.receipt["original_age"]["mismatching_rows"] == 1
    assert result.native_binding.receipt["population_binding_authenticated"] is False


def test_candidate_reconstructed_before_comparison_and_self_rehash_cannot_authorize(
    invented, monkeypatch
):
    result = _load(invented)
    candidate = invented[0].tmp_path / "candidate.coverage"
    candidate.write_bytes(result.payload)
    assert _load(invented, candidate_path=candidate).payload == result.payload
    header_raw, body = result._parts()
    header = json.loads(header_raw)
    tampered = body.replace(b'"30","1","4"', b'"30","2","4"', 1)
    assert tampered != body
    header["body_sha256"] = hashlib.sha256(tampered).hexdigest()
    raw = owner._json(header)
    candidate.write_bytes(owner._MAGIC + len(raw).to_bytes(4, "big") + raw + tampered)
    with _refuses("CANDIDATE_MISMATCH"):
        _load(invented, candidate_path=candidate)
    flip_archive_byte(invented[0].person_zip)
    original_copy = custody._copy

    def guarded_copy(source, destination, *args, **kwargs):
        assert source != candidate, (
            "candidate was accessed before source reconstruction"
        )
        return original_copy(source, destination, *args, **kwargs)

    monkeypatch.setattr(custody, "_copy", guarded_copy)
    with _refuses("SOURCE_RECONSTRUCTION_REFUSED"):
        _load(invented, candidate_path=candidate)


@pytest.mark.parametrize("replacement", [None, "", "different"])
def test_candidate_null_empty_and_field_contract_tampering_refuse(
    invented, replacement
):
    result = _load(invented)
    header, body = result._parts()
    rows = [json.loads(row) for row in body.splitlines()]
    rows[1][3] = replacement
    if replacement == "":
        metadata = json.loads(header)
        metadata["field_contract"]["fields"]["MIL"]["minimum_age"] = 16
    else:
        metadata = json.loads(header)
    changed_body = b"".join(owner._json(row) + b"\n" for row in rows)
    metadata.update(
        body_sha256=hashlib.sha256(changed_body).hexdigest(),
        body_bytes=len(changed_body),
    )
    changed_header = owner._json(metadata)
    candidate = invented[0].tmp_path / "candidate.coverage"
    candidate.write_bytes(
        owner._MAGIC
        + len(changed_header).to_bytes(4, "big")
        + changed_header
        + changed_body
    )
    with _refuses("CANDIDATE_MISMATCH|SOURCE_RECONSTRUCTION_REFUSED"):
        _load(invented, candidate_path=candidate)


def test_views_and_closed_objects_cannot_mutate_owned_evidence(invented):
    result = _load(invented)
    before = result.payload
    view = result.table
    view.loc[0, "MIL"] = "2"
    result.receipt["field_contract"].clear()
    result.native_binding.receipt["population_binding_authenticated"] = True
    assert result.payload == before
    assert result.table.loc[0, "MIL"] == "1"
    assert result.native_binding.receipt["population_binding_authenticated"] is False
    with pytest.raises(FrozenInstanceError):
        result.payload = b"forged"
    with _refuses("SOURCE_CONSTRUCTOR"):
        replace(result, payload=b"forged")


@pytest.mark.parametrize(
    "kind", ["duplicate", "case_duplicate", "unsafe", "symlink", "prefix_non_csv"]
)
def test_entire_zip_directory_is_checked_before_literal_reader(
    invented, monkeypatch, kind
):
    p = invented[3]
    extra = {
        "duplicate": Member("psam_pusa.csv", _csv(p)),
        "case_duplicate": Member("PSAM_PUSA.CSV", _csv(p)),
        "unsafe": Member("../unrelated.txt", b"ignored"),
        "symlink": Member("unrelated.txt", b"target", external_attr=0o120777 << 16),
        "prefix_non_csv": Member("psam_pus.txt", b"ignored"),
    }[kind]
    if kind == "duplicate":
        with pytest.warns(UserWarning, match="Duplicate name"):
            _rewrite(invented, members=(Member("psam_pusa.csv", _csv(p)), extra))
    else:
        _rewrite(invented, members=(Member("psam_pusa.csv", _csv(p)), extra))
    monkeypatch.setattr(
        literal,
        "read_acs_person_coverage_columns",
        lambda *a, **k: pytest.fail("reader entered"),
    )
    with _refuses(
        "ZIP_DUPLICATE_MEMBER|ZIP_MEMBER_PATH|ZIP_MEMBER_TYPE|ZIP_PREFIX_NONCSV"
    ):
        _load(invented)


@pytest.mark.parametrize(
    "target,constant,value,code",
    [
        (custody, "_MEMBER_COUNT_MAX", 1, "ZIP_MEMBER_COUNT"),
        (custody, "_MEMBER_MAX", 100, "ZIP_MEMBER_SIZE"),
        (custody, "_EXPANDED_MAX", 100, "ZIP_EXPANDED_SIZE"),
        (owner, "MAX_CSV_HEADER_BYTES", 30, "CSV_RECORD_BYTES"),
        (owner, "MAX_RECORD_BYTES", 64, "CSV_RECORD_BYTES"),
        (owner, "MAX_TOKEN_BYTES", 10, "CSV_TOKEN_BYTES"),
        (owner, "MAX_BODY_BYTES", 100, "SELECTED_BODY_BUDGET"),
        (owner, "MAX_HEADER_BYTES", 100, "CANONICAL_SIZE"),
        (literal, "MAX_ROWS", 1, "SOURCE_ROWS"),
        (literal, "MAX_SELECTED_ROWS", 2, "NATIVE_ROWS"),
    ],
)
def test_small_limits_refuse_before_relevant_allocation(
    invented, monkeypatch, target, constant, value, code
):
    monkeypatch.setattr(target, constant, value)
    with _refuses(code):
        _load(invented)


@pytest.mark.parametrize("kind", ["corrupt", "growth", "symlink", "fifo"])
def test_exact_capture_refusals_are_static_and_hide_cause_chains(invented, kind):
    path = invented[0].person_zip
    if kind == "corrupt":
        flip_archive_byte(path)
    elif kind == "growth":
        with path.open("ab") as stream:
            stream.write(b"extra")
    else:
        saved = path.with_suffix(".saved")
        path.rename(saved)
        if kind == "symlink":
            path.symlink_to(saved)
        else:
            os.mkfifo(path)
        # Keep directory roster exact; saved bytes live outside the source dir.
        saved.rename(invented[0].tmp_path / "saved.zip")
    with _refuses("SOURCE_RECONSTRUCTION_REFUSED") as exc:
        _load(invented)
    assert exc.value.__cause__ is None
    assert exc.value.__suppress_context__


@pytest.mark.parametrize(
    "race", ["private_bytes", "original_replace", "pins", "producer", "frame"]
)
def test_races_refuse_before_issuance(invented, monkeypatch, race):
    original = literal.read_acs_person_coverage_columns

    def raced(source, **kwargs):
        result = original(source, **kwargs)
        if race == "private_bytes":
            source.person_zip.chmod(0o600)
            flip_archive_byte(source.person_zip)
        elif race == "original_replace":
            # Replace during _copy instead, below.
            pass
        elif race == "pins":
            monkeypatch.setattr(custody, "_ARCHIVE_PINS", ())
        elif race == "producer":
            monkeypatch.setattr(owner, "MAX_TOKEN_BYTES", owner.MAX_TOKEN_BYTES - 1)
        else:
            invented[1].person.loc[0, "age"] = 31
        return result

    if race == "original_replace":
        old_identity = custody._identity
        changed = False

        def replaced_identity(value):
            nonlocal changed
            if not changed:
                changed = True
                path = invented[0].household_zip
                raw = path.read_bytes()
                path.unlink()
                path.write_bytes(raw)
            return old_identity(value)

        monkeypatch.setattr(custody, "_identity", replaced_identity)
    monkeypatch.setattr(literal, "read_acs_person_coverage_columns", raced)
    with _refuses(
        "SNAPSHOT_CHANGED|SOURCE_RECONSTRUCTION_REFUSED|PRODUCER_CHANGED|NATIVE_FRAME_CHANGED"
    ):
        _load(invented)


@pytest.mark.parametrize("ending", [b"\n", b"\r", b"\r\n", b""])
def test_raw_record_byte_boundary_precedes_decoding_and_preserves_endings(
    monkeypatch, ending
):
    header = b"A,B\n"
    body = b'"x\r\n\ty",z' + ending
    monkeypatch.setattr(owner, "MAX_RECORD_BYTES", len(body))
    assert list(owner._records(io.BytesIO(header + body))) == [header, body]
    monkeypatch.setattr(owner, "MAX_RECORD_BYTES", len(body) - 1)
    with _refuses("CSV_RECORD_BYTES"):
        list(owner._records(io.BytesIO(header + body)))


@pytest.mark.parametrize(
    "value",
    [
        "",
        None,
        "\r\n\t",
        '😀"\\\b\f',
        ["", None, "é"],
        {"b": True, "a": [1, -2, False]},
    ],
)
def test_canonical_size_is_checked_before_string_encoding(monkeypatch, value):
    expected = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    assert owner._json(value, len(expected)) == expected

    def forbidden_encoder(*a, **k):
        pytest.fail("JSON encoder allocated before preflight refusal")

    monkeypatch.setattr(owner.json, "JSONEncoder", forbidden_encoder)
    with _refuses("CANONICAL_SIZE"):
        owner._json(value, len(expected) - 1)
