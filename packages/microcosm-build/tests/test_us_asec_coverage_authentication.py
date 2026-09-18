"""Invented closed source-parent fixtures only; no genuine preparation."""

import _csv
import csv
import hashlib
import importlib.util
import json
import os
import struct
import subprocess
import sys
import textwrap
import traceback
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.frame_checkpoint import (
    load_frame_checkpoint,
    write_frame_checkpoint,
)
from microcosm.build.outer_stage_runtime import frame_identity
from microcosm.build.us_runtime import asec_coverage_authentication as coverage
from microcosm.build.us_runtime import asec_current_money_source as money
from microcosm.build.us_runtime import asec_person_coverage_source as literal


def _helper(name):
    spec = importlib.util.spec_from_file_location(
        "coverage_invented_" + name, Path(__file__).with_name(name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pin(paths, monkeypatch):
    pins = []
    for year, member, archive, _, _, _ in coverage._MEMBER_PINS:
        value = paths[year].read_bytes()
        with paths[year].open(newline="") as handle:
            rows = list(csv.reader(handle))
        pins.append(
            (year, member, archive, coverage._sha(value), len(rows) - 1, len(value))
        )
    monkeypatch.setattr(coverage, "_MEMBER_PINS", tuple(pins))


def _rewrite(path, change):
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    change(rows)
    with path.open("w", newline="") as handle:
        csv.writer(handle).writerows(rows)


def _changed_parent(parent, attachment, monkeypatch, changes):
    for path in (parent, attachment):
        loaded = load_frame_checkpoint(path)
        for column, values in changes.items():
            loaded.frame.person[column] = values
        identity = frame_identity(loaded.frame)
        for key in ("identity", "source_construction_identity"):
            if key in loaded.metadata:
                loaded.metadata[key] = identity.to_payload()
        if path == parent:
            parent_identity = identity
        else:
            digest = coverage._sha(parent.read_bytes())
            loaded.metadata["parent_checkpoint_sha256"] = digest
            loaded.metadata["household_observations"]["input_checkpoint_sha256"] = (
                digest
            )
            loaded.metadata["household_observations"][
                "input_structural_identity_sha256"
            ] = parent_identity.sha256
        write_frame_checkpoint(path, loaded.frame, metadata=loaded.metadata)
    monkeypatch.setattr(
        money,
        "_SOURCE_PINS",
        (
            coverage._sha(parent.read_bytes()),
            coverage._sha(attachment.read_bytes()),
            money._SOURCE_PINS[2],
        ),
    )


def _fixtures(
    tmp_path,
    monkeypatch,
    tokens=("1", "2", "3", "", "-4", "NA"),
    *,
    native_ids=None,
    person_changes=None,
    raw_ages=None,
):
    parent, attachment, paths = _helper(
        "test_us_asec_person_income_source"
    ).invented_sources(tmp_path, monkeypatch, missing=False)
    if native_ids is not None:
        _helper("test_us_asec_demographic_source")._repoint_published_households(
            parent, attachment, monkeypatch, native_ids
        )
    if person_changes:
        _changed_parent(parent, attachment, monkeypatch, person_changes)
    source = money.load_authenticated_current_money_source(parent, attachment)
    for path in paths.values():

        def change(rows):
            rows[0].append("PRPERTYP")
            for row in rows[1:]:
                position = int(row[0]) - 1
                row.append(tokens[position % len(tokens)])
                if native_ids is not None:
                    row[1] = str(native_ids[position])
                if raw_ages is not None:
                    row[3] = str(raw_ages[position])
                if person_changes and "PERIDNUM" in person_changes:
                    row[0] = person_changes["PERIDNUM"][position]

        _rewrite(path, change)
    _pin(paths, monkeypatch)
    return source, paths


def _read(source, paths, **kwargs):
    return coverage.authenticate_asec_coverage(source, member_paths=paths, **kwargs)


def test_closed_complete_source_exact_parent_and_defensive_views(tmp_path, monkeypatch):
    source, paths = _fixtures(tmp_path, monkeypatch)
    before = money._frame_signature(source.frame)
    tables = {e: source.frame.table(e).copy(deep=True) for e in source.frame.entities}
    result = _read(source, paths)
    coverage.verify_asec_coverage_parent(result, source)
    view = result.table()
    assert view.PRPERTYP.tolist() == ["1", "2", "3", "", "-4", "NA"]
    assert view.PRPERTYP_state.tolist() == ["observed_code"] * 3 + [
        "blank_unresolved",
        "unlabelled_in_range",
        "malformed_token",
    ]
    assert view.PERIDNUM.tolist() == list(source.scope.person_native_keys)
    assert view.person_household_id.tolist() == [1, 1, 2, 2, 3, 3]
    raw, original_receipt = literal.read_asec_person_coverage_source(
        paths, person_roster=source.frame.person
    )
    receipt = result.receipt
    assert receipt["literal_reader_receipt"] == original_receipt
    # Native key cells retain their exact strings; the artifact view declares
    # one non-nullable vocabulary independently of the parent's string storage.
    pd.testing.assert_frame_equal(view[list(raw)], raw.astype({"PERIDNUM": "string"}))
    assert receipt["source_authenticated"] is True
    assert receipt["named_parent_binding_authenticated"] is True
    for name in (
        "release_eligible",
        "domain_authority",
        "period_harmonized",
        "cross_survey_coverage_equivalence_established",
        "original_design_weight_semantics_established",
        "archive_bytes_read",
    ):
        assert receipt[name] is False
    assert receipt["literal_reader_receipt"]["source_authenticated"] is False
    assert (
        receipt["literal_reader_receipt"]["population_binding_authenticated"] is False
    )
    assert receipt["age_relation"]["compared_rows"] == 6
    assert (
        receipt["age_relation"]["relation"] == "original_csv_A_AGE_equals_parent_A_AGE"
    )
    assert receipt["household_partitions"]["households"] == 3
    assert [p["rows"] for p in receipt["sources"]] == [2, 2, 2]
    assert str(tmp_path) not in json.dumps(receipt)
    assert "0000000000000000000001" not in json.dumps(receipt)
    pin = result.content_sha256
    view.loc[0, "PRPERTYP"] = "changed"
    view.index = pd.Index([99, 1, 2, 3, 4, 5])
    receipt["parent"]["frame_sha256"] = "changed"
    assert result.content_sha256 == pin and result.table().PRPERTYP.iloc[0] == "1"
    assert money._frame_signature(source.frame) == before
    for entity, expected in tables.items():
        pd.testing.assert_frame_equal(
            source.frame.table(entity), expected, check_exact=True
        )
    with pytest.raises(FrozenInstanceError):
        result._body = b"changed"
    with pytest.raises(coverage.AsecCoverageAuthenticationError, match="CONSTRUCTOR"):
        coverage.AuthenticatedAsecCoverage(result._header, result._body)
    with pytest.raises(coverage.AsecCoverageAuthenticationError, match="CONSTRUCTOR"):
        replace(result)


@pytest.mark.parametrize(
    "token,state",
    [
        ("", "blank_unresolved"),
        ("\0", "malformed_token"),
        ("1\0x", "malformed_token"),
        (" 1", "malformed_token"),
        ("1.0", "malformed_token"),
        ("NA", "malformed_token"),
        ('a"b', "malformed_token"),
        ("a,b\r\n\t😀", "malformed_token"),
        ("-4", "unlabelled_in_range"),
        ("4", "out_of_range"),
        ("3", "observed_code"),
    ],
)
def test_literal_utf8_roundtrip_without_coercion(tmp_path, monkeypatch, token, state):
    source, paths = _fixtures(tmp_path, monkeypatch, (token,))
    result = _read(source, paths)
    assert result.table().PRPERTYP.tolist() == [token] * 6
    assert result.table().PRPERTYP_state.tolist() == [state] * 6
    assert not result.table().PRPERTYP.isna().any()


@pytest.mark.parametrize("kind", ["frame", "receipt", "hash"])
def test_no_plain_parent_can_issue_authority(tmp_path, monkeypatch, kind):
    source, paths = _fixtures(tmp_path, monkeypatch)
    candidate = {
        "frame": source.frame,
        "receipt": json.loads(source.source.identity),
        "hash": money._frame_signature(source.frame),
    }[kind]
    with pytest.raises(
        coverage.AsecCoverageAuthenticationError, match="AUTHENTICATED_PARENT"
    ):
        _read(candidate, paths)
    with pytest.raises(TypeError):
        coverage.authenticate_asec_coverage(source, member_paths=paths, member_pins=())


@pytest.mark.parametrize(
    "column,value",
    [
        ("A_AGE", "82"),
        ("A_AGE", ""),
        ("A_AGE", "55.0"),
        ("A_AGE", "\0"),
        ("PH_SEQ", "8"),
        ("A_LINENO", "8"),
        ("PERIDNUM", "9" * 22),
    ],
)
def test_authenticated_raw_coordinate_conflicts_refuse(
    tmp_path, monkeypatch, column, value
):
    source, paths = _fixtures(tmp_path, monkeypatch)
    _rewrite(
        paths[2022], lambda rows: rows[1].__setitem__(rows[0].index(column), value)
    )
    _pin(paths, monkeypatch)
    with pytest.raises(
        coverage.AsecCoverageAuthenticationError, match="SOURCE_REFUSAL"
    ):
        _read(source, paths)


@pytest.mark.parametrize(
    "column",
    [
        "A_AGE",
        "PERIDNUM",
        "person_household_id",
        "unrelated_raw_observation",
        "WSAL_VAL",
    ],
)
def test_same_ids_mutated_parent_cannot_reuse_binding(tmp_path, monkeypatch, column):
    source, paths = _fixtures(tmp_path, monkeypatch)
    result = _read(source, paths)
    source.frame.person.loc[:, column] = "9" * 22 if column == "PERIDNUM" else 1
    with pytest.raises(
        coverage.AsecCoverageAuthenticationError, match="PARENT_REFUSAL"
    ):
        coverage.verify_asec_coverage_parent(result, source)


def test_distinct_authenticated_parent_with_same_ids_cannot_reuse_artifact(
    tmp_path, monkeypatch
):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    source, paths = _fixtures(first, monkeypatch)
    result = _read(source, paths)
    other, other_paths = _fixtures(
        second,
        monkeypatch,
        person_changes={"unrelated_raw_observation": np.arange(6, dtype=np.int64) + 1},
    )
    assert other.scope.person_ids == source.scope.person_ids
    # Restore the producer's fixture pins so this specifically checks parent
    # identity rather than the independent source-pin implementation guard.
    _pin(paths, monkeypatch)
    with pytest.raises(
        coverage.AsecCoverageAuthenticationError, match="PARENT_CHANGED"
    ):
        coverage.verify_asec_coverage_parent(result, other)
    assert set(other_paths) == {2022, 2023, 2024}


def test_merged_native_households_refuse_even_with_matching_per_row_coordinates(
    tmp_path, monkeypatch
):
    # The current-money parent already refuses this shape, before coverage
    # issuance. Keep that layering and also test the wrapper's own fence.
    with pytest.raises(ValueError, match="SOURCE_CONTRACT_REFUSAL"):
        _fixtures(tmp_path, monkeypatch, native_ids=[7, 8, 7, 7, 7, 7])
    person = pd.DataFrame(
        {
            "source_year": [2022, 2022],
            "source_household_id": [7, 8],
            "person_household_id": [1, 1],
        },
        dtype="int64",
    )
    with pytest.raises(
        coverage.AsecCoverageAuthenticationError, match="HOUSEHOLD_MERGE"
    ):
        coverage._partitions(person)


@pytest.mark.parametrize("parent_age,raw_age", [(80, 80), (82, 80), (0, "")])
def test_raw_age_identity_never_substitutes_model_age(
    tmp_path, monkeypatch, parent_age, raw_age
):
    source, paths = _fixtures(
        tmp_path,
        monkeypatch,
        person_changes={
            "A_AGE": np.array([parent_age, 14] * 3, dtype=np.int64),
        },
        raw_ages=[raw_age, 14] * 3,
    )
    before = money._frame_signature(source.frame)
    if parent_age == raw_age:
        result = _read(source, paths)
        assert result.table().A_AGE.tolist() == [80, 14] * 3
        assert result.receipt["age_relation"]["model_age_used"] is False
    else:
        with pytest.raises(
            coverage.AsecCoverageAuthenticationError, match="SOURCE_REFUSAL"
        ):
            _read(source, paths)
    assert money._frame_signature(source.frame) == before


def test_parent_authority_refuses_model_age_before_source_issuance(
    tmp_path, monkeypatch
):
    # Canonical model age is already forbidden by the raw parent boundary.
    # Keep that refusal; do not loosen the parent merely to exercise coverage.
    with pytest.raises(ValueError, match="SOURCE_CONTRACT_REFUSAL"):
        _fixtures(
            tmp_path,
            monkeypatch,
            person_changes={
                "A_AGE": np.array([80, 14] * 3, dtype=np.int64),
                "age": [82.0, 14.0] * 3,
            },
            raw_ages=[80, 14] * 3,
        )


def test_same_native_person_keys_across_cohorts_remain_distinct(tmp_path, monkeypatch):
    keys = [str(i).zfill(22) for i in (1, 2)] * 3
    source, paths = _fixtures(tmp_path, monkeypatch, person_changes={"PERIDNUM": keys})
    result = _read(source, paths)
    assert result.table().PERIDNUM.tolist() == keys
    assert result.table().PRPERTYP.tolist() == ["1", "2", "3", "", "-4", "NA"]


def test_partition_bijection_refuses_splits_and_cross_cohort_merges():
    person = pd.DataFrame(
        {
            "source_year": [2022, 2022],
            "source_household_id": [7, 7],
            "person_household_id": [1, 2],
        },
        dtype="int64",
    )
    with pytest.raises(
        coverage.AsecCoverageAuthenticationError, match="HOUSEHOLD_SPLIT"
    ):
        coverage._partitions(person)
    person.source_year = [2022, 2023]
    person.person_household_id = [1, 1]
    with pytest.raises(
        coverage.AsecCoverageAuthenticationError, match="HOUSEHOLD_MERGE"
    ):
        coverage._partitions(person)


@pytest.mark.parametrize(
    "kind",
    [
        "missing_cohort",
        "duplicate_cohort",
        "duplicate_row",
        "missing_row",
        "swapped_cohort",
    ],
)
def test_complete_cohort_authority_refuses_incomplete_or_duplicate_inputs(
    tmp_path, monkeypatch, kind
):
    source, paths = _fixtures(tmp_path, monkeypatch)
    if kind == "missing_cohort":
        paths.pop(2023)
    elif kind == "duplicate_cohort":
        pins = coverage._MEMBER_PINS
        monkeypatch.setattr(coverage, "_MEMBER_PINS", (pins[0], pins[0], pins[2]))
    elif kind == "swapped_cohort":
        paths[2022], paths[2023] = paths[2023], paths[2022]
    else:
        _rewrite(
            paths[2022],
            lambda rows: (
                rows.__setitem__(1, rows[2].copy())
                if kind == "duplicate_row"
                else rows.pop()
            ),
        )
        _pin(paths, monkeypatch)
    with pytest.raises(coverage.AsecCoverageAuthenticationError):
        _read(source, paths)


def test_candidate_self_rehash_is_not_authority_and_reconstruction_runs_first(
    tmp_path, monkeypatch
):
    source, paths = _fixtures(tmp_path, monkeypatch)
    result = _read(source, paths)
    candidate = tmp_path / "candidate.bin"
    candidate.write_bytes(result.to_bytes())
    assert (
        _read(source, paths, candidate_path=candidate).to_bytes() == result.to_bytes()
    )
    body = result._body.replace(b"observed_code", b"invented_code", 1)
    header = json.loads(result._header)
    header["body_sha256"] = coverage._sha(body)
    header = coverage._json(header)
    payload = coverage.MAGIC + struct.pack("<I", len(header)) + header + body
    candidate.write_bytes(payload + hashlib.sha256(payload).digest())
    with pytest.raises(
        coverage.AsecCoverageAuthenticationError, match="CANONICAL_BYTES"
    ):
        _read(source, paths, candidate_path=candidate)
    paths[2022].write_bytes(b"corrupt original")
    with pytest.raises(coverage.AsecCoverageAuthenticationError, match="SOURCE_SIZE"):
        _read(source, paths, candidate_path=tmp_path / "absent-candidate")


@pytest.mark.parametrize(
    "mutation",
    [
        "body",
        "header",
        "self_rehash",
        "field",
        "status_function",
        "roster_contract",
        "integer_contract",
        "string_contract",
        "encoding",
        "dependency",
    ],
)
def test_runtime_mutation_cannot_inherit_issued_authority(
    tmp_path, monkeypatch, mutation
):
    source, paths = _fixtures(tmp_path, monkeypatch)
    result = _read(source, paths)
    if mutation in ("body", "self_rehash"):
        object.__setattr__(
            result, "_body", result._body.replace(b"observed_code", b"invented_code", 1)
        )
        if mutation == "self_rehash":
            header = json.loads(result._header)
            header["body_sha256"] = coverage._sha(result._body)
            object.__setattr__(result, "_header", coverage._json(header))
    elif mutation == "header":
        header = json.loads(result._header)
        header["parent"]["frame_sha256"] = "0" * 64
        object.__setattr__(result, "_header", coverage._json(header))
    elif mutation == "field":
        contract = literal.coverage_field_contract()
        contract["codes"]["1"] = "changed"
        monkeypatch.setattr(literal, "coverage_field_contract", lambda: contract)
    elif mutation == "status_function":
        monkeypatch.setattr(literal, "_state", lambda token: "observed_code")
    elif mutation == "roster_contract":
        monkeypatch.setattr(coverage, "COLUMNS", tuple(reversed(coverage.COLUMNS)))
    elif mutation == "integer_contract":
        monkeypatch.setattr(coverage, "_INT_COLUMNS", ())
    elif mutation == "string_contract":
        monkeypatch.setattr(coverage, "_STRING_COLUMNS", ())
    elif mutation == "encoding":
        monkeypatch.setattr(coverage, "MAGIC", b"changed")
    else:
        monkeypatch.setattr(coverage.metadata, "version", lambda name: "changed")
    with pytest.raises(coverage.AsecCoverageAuthenticationError, match="CHANGED"):
        result.table()


@pytest.mark.parametrize(
    "limit,reason",
    [
        ("_HEADER_MAX", "HEADER_BYTES"),
        ("_BODY_MAX", "BODY_BYTES"),
        ("_MAX_PERSONS", "ROWS"),
        ("_ROW_MAX", "CSV_RECORD_BYTES"),
        ("_TOKEN_MAX", "CSV_TOKEN_BYTES"),
        ("_CSV_HEADER_MAX", "CSV_RECORD_BYTES"),
    ],
)
def test_tiny_resource_ceilings_refuse_without_large_allocation(
    tmp_path, monkeypatch, limit, reason
):
    source, paths = _fixtures(tmp_path, monkeypatch)
    monkeypatch.setattr(coverage, limit, 1)
    with pytest.raises(coverage.AsecCoverageAuthenticationError, match=reason):
        _read(source, paths)


def test_header_budget_runs_before_json_allocation(monkeypatch):
    monkeypatch.setattr(coverage, "_HEADER_MAX", 8)
    monkeypatch.setattr(
        coverage, "_json", lambda value: pytest.fail("encoded oversized header")
    )
    with pytest.raises(coverage.AsecCoverageAuthenticationError, match="HEADER_BYTES"):
        coverage._bounded_header({"a": "\0\0"})


@pytest.mark.parametrize("kind", ["header", "body", "rows", "row", "token"])
def test_decoder_refuses_lengths_before_allocating_views(monkeypatch, kind):
    body = (
        struct.pack("<I", 61) + struct.pack("<6q", *([1] * 6)) + struct.pack("<I", 1000)
    )
    if kind == "header":
        fake = object.__new__(coverage.AuthenticatedAsecCoverage)
        object.__setattr__(fake, "_header", b"{}")
        object.__setattr__(fake, "_body", b"")
        monkeypatch.setattr(coverage, "_HEADER_MAX", 1)
        with pytest.raises(
            coverage.AsecCoverageAuthenticationError, match="HEADER_BYTES"
        ):
            _ = fake.receipt
        return
    if kind == "body":
        monkeypatch.setattr(coverage, "_BODY_MAX", 1)
    elif kind == "rows":
        monkeypatch.setattr(coverage, "_MAX_PERSONS", 1)
    elif kind == "row":
        monkeypatch.setattr(coverage, "_ROW_MAX", 1)
    else:
        body += b"\0" * (65 - len(body))
        monkeypatch.setattr(coverage, "_TOKEN_MAX", 1)
    with pytest.raises(coverage.AsecCoverageAuthenticationError):
        list(coverage._decode_rows(body, 2 if kind == "rows" else 1))


@pytest.mark.parametrize(
    "kind",
    [
        "fifo",
        "directory",
        "symlink",
        "growth",
        "corrupt",
        "private_race",
        "path_replacement",
    ],
)
def test_capture_races_nonregular_cleanup_and_static_refusals(
    tmp_path, monkeypatch, kind
):
    source, paths = _fixtures(tmp_path, monkeypatch)
    path = paths[2022]
    original = path.read_bytes()
    if kind in ("fifo", "directory", "symlink"):
        path.unlink()
        if kind == "fifo":
            os.mkfifo(path)
        elif kind == "directory":
            path.mkdir()
        else:
            target = tmp_path / "literal-secret-token"
            target.write_bytes(original)
            path.symlink_to(target)
    elif kind == "corrupt":
        path.write_bytes(original.replace(b",55,", b",56,", 1))
    elif kind in ("growth", "path_replacement"):
        original_fdopen = coverage.os.fdopen

        def fdopen(descriptor, *args, **kwargs):
            if os.fstat(descriptor).st_ino == path.stat().st_ino:
                if kind == "growth":
                    with path.open("ab") as handle:
                        handle.write(b"excess")
                else:
                    path.unlink()
                    path.write_bytes(b"replacement")
            return original_fdopen(descriptor, *args, **kwargs)

        monkeypatch.setattr(coverage.os, "fdopen", fdopen)
    else:
        original_capture = coverage._capture

        def capture(path, destination, **kwargs):
            identity = original_capture(path, destination, **kwargs)
            destination.write_bytes(
                destination.read_bytes().replace(b",55,", b",55,", 1)
            )
            return identity

        monkeypatch.setattr(coverage, "_capture", capture)
    owned = tmp_path / "private"
    owned.mkdir()
    monkeypatch.setattr(coverage.tempfile, "tempdir", str(owned))
    with pytest.raises(coverage.AsecCoverageAuthenticationError) as caught:
        _read(source, paths)
    rendered = "".join(traceback.format_exception(caught.value))
    assert str(path) not in rendered and "literal-secret-token" not in rendered
    assert caught.value.__suppress_context__ or caught.value.__context__ is None
    assert list(owned.iterdir()) == []


def test_new_file_ci_inventory_uses_authored_country_group():
    path = Path(__file__).resolve().parents[3] / "tools/ci_test_groups.py"
    spec = importlib.util.spec_from_file_location("coverage_ci_inventory", path)
    groups = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(groups)
    test = "packages/microcosm-build/tests/test_us_asec_coverage_authentication.py"
    assert groups.fast_group(test) == "rest"
    assert groups.engine_group(test) == "us-am"
    assert groups.process_for("us-am", test) == "build"


def test_literal_code_mutation_refuses_before_original_read(tmp_path, monkeypatch):
    source, paths = _fixtures(tmp_path, monkeypatch)
    monkeypatch.setattr(literal, "_state", lambda token: "observed_code")
    monkeypatch.setattr(
        coverage, "_capture", lambda *a, **k: pytest.fail("source read")
    )
    with pytest.raises(
        coverage.AsecCoverageAuthenticationError, match="IMPLEMENTATION_CHANGED"
    ):
        _read(source, paths)


def test_canonical_byte_limits_admit_exact_boundary_and_refuse_one_more(monkeypatch):
    header = {"a": "\0😀"}
    encoded = coverage._json(header)
    monkeypatch.setattr(coverage, "_HEADER_MAX", len(encoded))
    assert coverage._bounded_header(header) == encoded
    monkeypatch.setattr(coverage, "_HEADER_MAX", len(encoded) - 1)
    with pytest.raises(coverage.AsecCoverageAuthenticationError, match="HEADER_BYTES"):
        coverage._bounded_header(header)
    row = [1] * 6 + ["key", "😀", "state"]
    table = pd.DataFrame([row], columns=coverage.COLUMNS)
    body = coverage._body(table)
    monkeypatch.setattr(coverage, "_BODY_MAX", len(body))
    monkeypatch.setattr(coverage, "_ROW_MAX", len(body) - 4)
    monkeypatch.setattr(coverage, "_TOKEN_MAX", 5)
    assert coverage._body(table) == body
    assert list(coverage._decode_rows(body, 1)) == [row]
    monkeypatch.setattr(coverage, "_BODY_MAX", len(body) - 1)
    with pytest.raises(coverage.AsecCoverageAuthenticationError, match="BODY_BYTES"):
        coverage._body(table)
    monkeypatch.setattr(coverage, "_BODY_MAX", len(body))
    monkeypatch.setattr(coverage, "_TOKEN_MAX", 3)
    with pytest.raises(coverage.AsecCoverageAuthenticationError, match="TOKEN_BYTES"):
        coverage._body(table)


@pytest.mark.parametrize("codepoint", [*range(128), 0xFFFF, 0x10000, 0x1F600])
def test_header_string_budget_matches_exact_ascii_json_bytes(monkeypatch, codepoint):
    header = {"a": chr(codepoint)}
    encoded = coverage._json(header)
    monkeypatch.setattr(coverage, "_HEADER_MAX", len(encoded))
    assert coverage._bounded_header(header) == encoded
    monkeypatch.setattr(coverage, "_HEADER_MAX", len(encoded) - 1)
    with pytest.raises(coverage.AsecCoverageAuthenticationError, match="HEADER_BYTES"):
        coverage._bounded_header(header)


def test_runtime_identity_is_stable_in_fresh_interpreters(tmp_path):
    # Exercise the actual owner fingerprint functions and generated frozen
    # dataclass methods without importing country/source-resource facades in
    # child interpreters. This is a runtime-identity regression, not a complete
    # candidate reconstruction test.
    script = textwrap.dedent(
        """
        import ast
        import hashlib
        import json
        import sys
        from dataclasses import InitVar, dataclass
        from pathlib import Path
        from types import CodeType, FunctionType, ModuleType

        names = {
            '_json', '_sha', '_constant_document', '_code_document',
            '_runtime_code', 'AuthenticatedAsecCoverage',
            'AsecCoverageAuthenticationError',
        }
        path = Path(sys.argv[1])
        tree = ast.parse(path.read_text())
        tree.body = [node for node in tree.body if getattr(node, 'name', '') in names]
        module = ModuleType('coverage_fingerprint_leaf')
        sys.modules[module.__name__] = module
        module.__dict__.update({
            'hashlib': hashlib, 'json': json, 'InitVar': InitVar,
            'dataclass': dataclass, 'CodeType': CodeType, 'FunctionType': FunctionType,
        })
        exec(compile(tree, str(path), 'exec'), module.__dict__)
        identity = module._runtime_code(module)
        assert 'AuthenticatedAsecCoverage.__setattr__' in identity
        print(module._sha(module._json(identity)))
        """
    )
    values = [
        subprocess.run(
            [sys.executable, "-I", "-B", "-S", "-c", script, str(coverage.__file__)],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        ).stdout.strip()
        for _ in range(8)
    ]
    assert len(set(values)) == 1
    assert len(values[0]) == 64


def test_code_constants_have_stable_typed_identity():
    first = frozenset(("_body", "_header", (b"a", 1, True, 1.5, 2j, None, ...)))
    second = frozenset(reversed(tuple(first)))
    assert coverage._constant_document(first) == coverage._constant_document(second)
    assert coverage._constant_document(True) != coverage._constant_document(1)
    assert coverage._constant_document(1) != coverage._constant_document(1.0)
    assert coverage._constant_document(0.0) != coverage._constant_document(-0.0)
    assert coverage._constant_document(slice(None, 2)) != coverage._constant_document(
        (None, 2, None)
    )
    assert coverage._constant_document(slice(None, 2)) != coverage._constant_document(
        slice(None, 3)
    )
    with pytest.raises(coverage.AsecCoverageAuthenticationError, match="CODE_CONSTANT"):
        coverage._constant_document(object())


def test_complete_owner_and_literal_module_constants_are_supported():
    # Python 3.14 folds literal slices into code constants. Traverse actual
    # complete modules, not only the generated dataclass methods.
    for module in (coverage, literal):
        path = Path(module.__file__)
        code = compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
        document = coverage._json(coverage._code_document(code))
        assert document
        assert document == coverage._json(coverage._code_document(code))


@pytest.mark.parametrize("kind", ["growth", "directory", "fifo", "too_large"])
def test_candidate_resource_checks_are_independent_of_candidate_header(
    tmp_path, monkeypatch, kind
):
    path = tmp_path / "candidate"
    payload = b"invented-canonical"
    if kind == "directory":
        path.mkdir()
    elif kind == "fifo":
        os.mkfifo(path)
    else:
        path.write_bytes(payload + (b"x" if kind == "too_large" else b""))
    if kind == "growth":
        original = coverage.os.fdopen

        def fdopen(*args, **kwargs):
            with path.open("ab") as handle:
                handle.write(b"x")
            return original(*args, **kwargs)

        monkeypatch.setattr(coverage.os, "fdopen", fdopen)
    with pytest.raises(coverage.AsecCoverageAuthenticationError):
        coverage._compare_candidate(path, payload)


def test_total_projected_budget_refuses_before_literal_projection(
    tmp_path, monkeypatch
):
    source, paths = _fixtures(tmp_path, monkeypatch)
    # Each cohort fits separately. The shared three-cohort budget must refuse
    # before the reader can retain the projected DataFrame/string buffers.
    monkeypatch.setattr(coverage, "_BODY_MAX", 500)
    monkeypatch.setattr(
        literal,
        "read_asec_person_coverage_source",
        lambda *a, **k: pytest.fail("projection allocated"),
    )
    monkeypatch.setattr(coverage, "_LITERAL_AUTHORITY", coverage._runtime_code(literal))
    with pytest.raises(coverage.AsecCoverageAuthenticationError, match="BODY_BYTES"):
        _read(source, paths)


def test_capture_budget_tracks_multiline_literal_and_ignores_unused_fields(monkeypatch):
    monkeypatch.setattr(coverage, "_TOKEN_MAX", 100)
    raw = b'PRPERTYP,unused\r\n"a\r\n,b",' + b"x" * 60 + b"\r\n"
    budget = [128]
    guard = coverage._CsvBounds(budget)
    for byte in raw:
        guard.feed(bytes([byte]))  # Split CRLF/quotes across capture chunks.
    assert budget[0] == 128 - (118 + len(b'"a\r\n,b",'))
    with pytest.raises(coverage.AsecCoverageAuthenticationError, match="BODY_BYTES"):
        guard.feed(b'"another row",x\r\n')


@pytest.mark.parametrize(
    "aliases", ["csv", "native", "both", "owner_module", "literal_module"]
)
def test_live_csv_reader_drift_refuses_before_capture(tmp_path, monkeypatch, aliases):
    parent, paths = _fixtures(tmp_path, monkeypatch)
    issued = _read(parent, paths)
    assert issued.table().PRPERTYP.iloc[0] == "1"
    original_reader = csv.reader

    def changed_reader(*args, **kwargs):
        reader = original_reader(*args, **kwargs)
        header = next(reader)
        yield header
        for original in reader:
            row = list(original)
            if "PRPERTYP" in header:
                row[header.index("PRPERTYP")] = "2"
            yield row

    def forbidden_capture(*args, **kwargs):
        raise AssertionError("Reader drift must refuse before first source read")

    with monkeypatch.context() as patch:
        patch.setattr(coverage, "_capture", forbidden_capture)
        if aliases in ("csv", "both"):
            patch.setattr(csv, "reader", changed_reader)
        if aliases in ("native", "both"):
            patch.setattr(_csv, "reader", changed_reader)
        if aliases in ("owner_module", "literal_module"):
            module = coverage if aliases == "owner_module" else literal
            patch.setattr(
                module,
                "csv",
                SimpleNamespace(**{**vars(csv), "reader": changed_reader}),
            )
        with pytest.raises(
            coverage.AsecCoverageAuthenticationError,
            match="^SOURCE_CSV_READER_CHANGED$",
        ):
            _read(parent, paths)
        with pytest.raises(
            coverage.AsecCoverageAuthenticationError,
            match="^SOURCE_CSV_READER_CHANGED$",
        ):
            issued.validate()
    replay = _read(parent, paths)
    assert replay.to_bytes() == issued.to_bytes()
    coverage.verify_asec_coverage_parent(replay, parent)


def test_csv_binding_helper_bytes_bind_person_capsule(tmp_path, monkeypatch):
    parent, paths = _fixtures(tmp_path, monkeypatch)
    issued = _read(parent, paths)
    real_files = coverage.resources.files

    def changed_files(package):
        package_path = real_files(package)

        def child(name):
            path = package_path.joinpath(name)
            if name == "source_csv_builtin.py":
                return SimpleNamespace(
                    read_bytes=lambda: path.read_bytes() + b"\n# drift\n"
                )
            return path

        return SimpleNamespace(joinpath=child)

    monkeypatch.setattr(coverage.resources, "files", changed_files)
    with pytest.raises(
        coverage.AsecCoverageAuthenticationError,
        match="^COVERAGE_IMPLEMENTATION_CHANGED$",
    ):
        issued.validate()


@pytest.mark.parametrize("aliases", ["csv", "both"])
@pytest.mark.parametrize("moment", ["entry", "before_call"])
@pytest.mark.parametrize("stage", ["literal", "header"])
def test_transient_csv_rebinding_never_supplies_person_cells(
    tmp_path, monkeypatch, aliases, moment, stage
):
    from test_us_asec_household_coverage_fields import transient_csv_rebinding

    source, paths = _fixtures(tmp_path, monkeypatch)
    control = _read(source, paths)
    coverage.verify_asec_coverage_parent(control, source)
    candidate = tmp_path / "candidate.bin"
    candidate.write_bytes(control.to_bytes())
    assert (
        _read(source, paths, candidate_path=candidate).to_bytes() == control.to_bytes()
    )
    before = coverage._implementation()
    target = (
        literal.read_asec_person_coverage_source
        if stage == "literal"
        else coverage._CsvBounds._end_header
    )
    with transient_csv_rebinding(target, aliases, moment) as (fired, calls):
        try:
            result = _read(source, paths)
        except coverage.AsecCoverageAuthenticationError as exc:
            # Literal refusals retain the public owner's existing static wrapper.
            expected = (
                "COVERAGE_SOURCE_REFUSAL"
                if stage == "literal"
                else "SOURCE_CSV_READER_CHANGED"
            )
            assert str(exc) == expected
        else:
            coverage.verify_asec_coverage_parent(result, source)
            assert result.to_bytes() == control.to_bytes()
    assert fired and not calls
    assert coverage._implementation() == before
    assert (
        _read(source, paths, candidate_path=candidate).to_bytes() == control.to_bytes()
    )
