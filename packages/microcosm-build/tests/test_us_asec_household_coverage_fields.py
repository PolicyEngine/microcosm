"""Invented hhpub members for source-only household coverage observations."""

import _csv as native_csv
import copy
import csv
import gc
import hashlib
import importlib
import inspect
import io
import os
import pickle
import sys
import weakref
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import pytest

from microcosm.build.us_runtime import asec_household_coverage_fields as coverage
from microcosm.build.us_runtime import asec_original_household_weights as shared
from microcosm.build.us_runtime import native_household_origin as origin


def _csv(
    rows, header=("H_SEQ", "H_HHTYPE", "HRHTYPE", "H_LIVQRT", "H_NUMPER", "EXTRA")
):
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return output.getvalue().encode()


@pytest.fixture
def members(tmp_path, monkeypatch):
    def make(rows=None, *, payload=None, years=(2024,), expected_rows=None):
        rows = rows if rows is not None else [("00001", "1", "09", "08", "03", "")]
        payload = payload if payload is not None else _csv(rows)
        paths, pins = {}, []
        for year in years:
            path = tmp_path / f"invented-{year}.csv"
            path.write_bytes(payload)
            paths[year] = path
            pins.append(
                origin.AsecNativeMemberPin(
                    income_year=year,
                    survey_year=year + 1,
                    canonical_member_id=f"invented/{year + 1}/household",
                    member_name=f"hhpub{str(year + 1)[-2:]}.csv",
                    archive_sha256=origin.ASEC_EDUCATION_ASSISTANCE_ARCHIVES[
                        year
                    ].zip_sha256,
                    member_sha256=hashlib.sha256(payload).hexdigest(),
                    size_bytes=len(payload),
                    rows=len(rows) if expected_rows is None else expected_rows,
                )
            )
        monkeypatch.setattr(origin, "_ASEC_MEMBER_PINS", tuple(pins))
        return paths

    return make


def test_whole_cohort_lookup_preserves_literal_codes_and_reported_count(members):
    source = coverage.load_authenticated_asec_household_coverage_fields(
        members(years=(2022, 2023, 2024))
    )
    doc = source.document
    assert [row["income_year"] for row in doc["records"]] == [2022, 2023, 2024]
    assert len({row["origin_key"] for row in doc["records"]}) == 3
    row = source.households_for([(2024, 1)])[0]
    assert [row[name] for name in coverage.COLUMNS] == ["00001", "1", "09", "08", "03"]
    assert (
        row["field_states"]["HRHTYPE"]["label"]
        == "Group quarters with actual families (This is new in 1994)"
    )
    assert row["reported_person_count"] == 3
    assert row["coverage_interpretation"] == "unresolved"
    assert doc["source_authenticated"] is True
    assert doc["population_binding_authenticated"] is False
    assert doc["release_eligible"] is False
    assert "weights" not in doc


@pytest.mark.parametrize(
    "name,token,status",
    [
        ("H_HHTYPE", "1", "valid"),
        ("H_HHTYPE", "01", "malformed"),
        ("HRHTYPE", "9", "valid"),
        ("HRHTYPE", "09", "valid"),
        ("HRHTYPE", "11", "unlabelled"),
        ("H_LIVQRT", "00", "unlabelled"),
        ("H_NUMPER", "17", "unlabelled"),
        ("H_NUMPER", "", "missing"),
        ("H_NUMPER", "NA", "malformed"),
        ("H_NUMPER", "3.0", "malformed"),
        ("H_NUMPER", " 3", "malformed"),
        ("H_NUMPER", "3\x00", "malformed"),
        ("H_NUMPER", "-1", "malformed"),
        ("HRHTYPE", "９", "malformed"),
    ],
)
def test_literal_valid_missing_and_unresolved_states(members, name, token, status):
    row = dict(zip(coverage.COLUMNS, ["1", "1", "09", "08", "3"], strict=True))
    row[name] = token
    source = coverage.load_authenticated_asec_household_coverage_fields(
        members([(*row.values(), "")])
    )
    actual = source.document["records"][0]
    assert actual[name] == token
    assert actual["field_states"][name]["code_status"] == status
    if name == "H_NUMPER" and status != "valid":
        assert actual["reported_person_count"] is None


@pytest.mark.parametrize(
    "hh,hr,count,diagnostics",
    [
        (
            "1",
            "00",
            "0",
            [
                "interview_with_noninterview_household_type",
                "interview_with_noninterview_person_count",
            ],
        ),
        (
            "2",
            "09",
            "03",
            [
                "noninterview_with_interview_household_type",
                "noninterview_with_positive_person_count",
            ],
        ),
        ("3", "00", "00", []),
        ("", "09", "3", []),
    ],
)
def test_interview_and_noninterview_contradictions_remain_diagnostics(
    members, hh, hr, count, diagnostics
):
    source = coverage.load_authenticated_asec_household_coverage_fields(
        members([("1", hh, hr, "08", count, "")])
    )
    row = source.document["records"][0]
    assert row["diagnostics"] == diagnostics
    assert row["field_states"]["HRHTYPE"]["universe_status"] == (
        "in" if hh == "1" else "outside" if hh in ("2", "3") else "unresolved"
    )
    if hh != "1" or count in ("0", "00"):
        assert row["reported_person_count"] is None


def test_student_quarters_retains_dictionary_code_without_population_inference(members):
    source = coverage.load_authenticated_asec_household_coverage_fields(
        members([("1", "1", "10", "11", "1", "")])
    )
    row = source.document["records"][0]
    assert (
        row["field_states"]["H_LIVQRT"]["label"]
        == "Student quarters in college dormitory"
    )
    assert row["diagnostics"] == ["student_quarters_dictionary_methodology_unresolved"]
    assert row["coverage_interpretation"] == "unresolved"
    assert (
        source.document["interpretation_notes"]["student_quarters"][
            "observed_presence_asserted"
        ]
        is False
    )
    assert (
        source.document["interpretation_notes"]["student_quarters"]["methodology_url"]
        == "https://www.census.gov/topics/income-poverty/guidance/group-quarters.html"
    )


@pytest.mark.parametrize("hr,quarters", [("01", "12"), ("09", "01"), ("10", "07")])
def test_household_and_living_quarter_codes_do_not_assign_shared_or_exclusive_frame(
    members, hr, quarters
):
    row = coverage.load_authenticated_asec_household_coverage_fields(
        members([("1", "1", hr, quarters, "2", "")])
    ).document["records"][0]
    assert row["diagnostics"] == []
    assert row["coverage_interpretation"] == "unresolved"
    assert "domain" not in row


def test_definition_pages_and_vintage_positions_are_exact_and_defensive(members):
    source = coverage.load_authenticated_asec_household_coverage_fields(members())
    field = source.document["fields"]["H_NUMPER"]
    assert field["position_by_survey_year"] == {"2023": 82, "2024": 82, "2025": 84}
    assert field["pdf_page_1based_by_survey_year"] == {"2023": 3, "2024": 3, "2025": 10}
    field["labels"]["0"] = "invented mutation"
    assert (
        source.document["fields"]["H_NUMPER"]["labels"]["0"] == "Noninterview household"
    )


def test_exact_candidate_reconstruction_and_rehashed_mutation_refusal(members):
    paths = members()
    source = coverage.load_authenticated_asec_household_coverage_fields(paths)
    assert (
        coverage.load_authenticated_asec_household_coverage_fields(
            paths, candidate=source.payload
        ).payload
        == source.payload
    )
    altered = source.document
    altered["records"][0]["HRHTYPE"] = "01"
    altered["projection_sha256"] = shared._sha(shared._encode(altered["records"]))
    with pytest.raises(
        coverage.HouseholdCoverageSourceError, match="CANDIDATE_MISMATCH"
    ):
        coverage.load_authenticated_asec_household_coverage_fields(
            paths, candidate=shared._encode(altered)
        )
    paths[2024].write_bytes(paths[2024].read_bytes().replace(b",09,", b",01,"))
    with pytest.raises(coverage.HouseholdCoverageSourceError, match="MEMBER_SHA256"):
        coverage.load_authenticated_asec_household_coverage_fields(
            paths, candidate=source.payload
        )


def test_constructor_and_defensive_views_cannot_mint_authority(members):
    source = coverage.load_authenticated_asec_household_coverage_fields(members())
    with pytest.raises(
        coverage.HouseholdCoverageSourceError, match="SOURCE_CONSTRUCTOR"
    ):
        coverage.AuthenticatedAsecHouseholdCoverageFields(source.payload)
    with pytest.raises(
        coverage.HouseholdCoverageSourceError, match="SOURCE_CONSTRUCTOR"
    ):
        replace(source, payload=source.payload)
    with pytest.raises(FrozenInstanceError):
        source.payload = b"{}"
    source.households_for([(2024, 1)])[0]["H_NUMPER"] = "16"
    assert source.households_for([(2024, 1)])[0]["H_NUMPER"] == "03"
    object.__setattr__(source, "payload", source.payload + b" ")
    with pytest.raises(coverage.HouseholdCoverageSourceError, match="SOURCE_MUTATION"):
        _ = source.document


@pytest.mark.parametrize(
    "roster",
    [
        [],
        [(2024, 2)],
        [(2024, 1), (2024, 1)],
        [(2023, 1)],
        [(2024, True)],
        [(2024, "1")],
    ],
)
def test_native_lookup_refuses_unknown_duplicate_or_coerced_keys(members, roster):
    source = coverage.load_authenticated_asec_household_coverage_fields(members())
    with pytest.raises(coverage.HouseholdCoverageSourceError):
        source.households_for(roster)


@pytest.mark.parametrize(
    "payload,expected",
    [
        (b"H_SEQ,H_HHTYPE\n1,1\n", 1),
        (b"H_SEQ,H_HHTYPE,HRHTYPE,H_LIVQRT,H_NUMPER,H_SEQ\n1,1,9,8,3,1\n", 1),
        (_csv([("1", "1", "9", "8", "3")]), 1),
        (_csv([("1", "1", "9", "8", "3", "", "extra")]), 1),
        (_csv([("1", "1", "9", "8", "3", ""), ("01", "1", "9", "8", "3", "")]), 2),
        (_csv([("", "1", "9", "8", "3", "")]), 1),
        (_csv([("1", "1", "9", "8", "3", "")]), 2),
        (b"H_SEQ,H_HHTYPE,HRHTYPE,H_LIVQRT,H_NUMPER\n1,1,9,8,\xff\n", 1),
    ],
)
def test_complete_cohort_structure_and_encoding_refuse(members, payload, expected):
    with pytest.raises(coverage.HouseholdCoverageSourceError):
        coverage.load_authenticated_asec_household_coverage_fields(
            members(payload=payload, expected_rows=expected)
        )


@pytest.mark.parametrize("mutation", ["field", "helper", "registry"])
def test_live_contract_dependency_or_registry_mutation_refuses(
    members, monkeypatch, mutation
):
    source = coverage.load_authenticated_asec_household_coverage_fields(members())
    if mutation == "field":
        monkeypatch.setattr(coverage, "_FIELDS", {"changed": True})
    elif mutation == "helper":
        original = shared._implementation
        monkeypatch.setattr(
            shared, "_implementation", lambda: {**original(), "changed": True}
        )
    else:
        monkeypatch.setattr(
            origin, "_ASEC_MEMBER_PINS", (replace(origin._ASEC_MEMBER_PINS[0], rows=2),)
        )
    with pytest.raises(coverage.HouseholdCoverageSourceError, match="PRODUCER_CHANGED"):
        _ = source.document


def test_record_and_projection_token_limits_are_checked(members, monkeypatch):
    paths = members([("1", "1", "9", "8", "3", "x" * 150)])
    monkeypatch.setattr(shared, "_MAX_RECORD_CHARS", 90)
    with pytest.raises(
        coverage.HouseholdCoverageSourceError, match="MEMBER_RECORD_SIZE"
    ):
        coverage.load_authenticated_asec_household_coverage_fields(paths)
    monkeypatch.setattr(shared, "_MAX_RECORD_CHARS", 65536)
    paths = members([("1", "1", "9", "8", "3" * 129, "")])
    with pytest.raises(
        coverage.HouseholdCoverageSourceError, match="MEMBER_FIELD_SIZE"
    ):
        coverage.load_authenticated_asec_household_coverage_fields(paths)


@pytest.mark.parametrize("kind", ["directory", "fifo", "grown"])
def test_regular_exact_source_capture_is_required(members, kind):
    paths = members()
    path = paths[2024]
    if kind == "grown":
        path.write_bytes(path.read_bytes() + b"extra")
    else:
        path.unlink()
        path.mkdir() if kind == "directory" else os.mkfifo(path)
    with pytest.raises(coverage.HouseholdCoverageSourceError, match="MEMBER_CAPTURE"):
        coverage.load_authenticated_asec_household_coverage_fields(paths)


@pytest.mark.parametrize("mutation", ["clear", "replace"])
def test_caller_path_mapping_is_snapshotted_before_fingerprinting(
    members, monkeypatch, mutation
):
    paths = members()
    original = coverage._producer

    def changing_producer():
        producer = original()
        if mutation == "clear":
            paths.clear()
        else:
            paths[2024] = paths[2024].with_name("must-not-be-opened.csv")
        return producer

    monkeypatch.setattr(coverage, "_producer", changing_producer)
    source = coverage.load_authenticated_asec_household_coverage_fields(paths)
    assert len(source.document["records"]) == 1
    assert len(source.document["members"]) == 1


def test_selected_pins_must_match_fingerprinted_registry(members, monkeypatch):
    paths = members()
    original = shared._registry
    first = True

    def changing_registry():
        nonlocal first
        pins = original()
        if first:
            first = False
            monkeypatch.setattr(
                origin, "_ASEC_MEMBER_PINS", (replace(pins[0], rows=2),)
            )
        return pins

    monkeypatch.setattr(shared, "_registry", changing_registry)
    with pytest.raises(
        coverage.HouseholdCoverageSourceError, match="REGISTRY_TRANSITION"
    ):
        coverage.load_authenticated_asec_household_coverage_fields(paths)


@pytest.mark.parametrize(
    "filename",
    [
        "asec_household_coverage_fields.py",
        "asec_original_household_weights.py",
        "asec_student_controls.py",
    ],
)
def test_actual_owner_and_helper_code_bytes_bind_issued_capsule(
    members, monkeypatch, filename
):
    source = coverage.load_authenticated_asec_household_coverage_fields(members())
    actual_files = coverage.resources.files

    def changed_files(package):
        package_path = actual_files(package)

        def child(name):
            path = package_path.joinpath(name)
            if name == filename:
                return SimpleNamespace(
                    read_bytes=lambda: path.read_bytes() + b"\n# changed\n"
                )
            return path

        return SimpleNamespace(joinpath=child)

    monkeypatch.setattr(coverage.resources, "files", changed_files)
    with pytest.raises(coverage.HouseholdCoverageSourceError, match="PRODUCER_CHANGED"):
        _ = source.document


def test_payload_bound_prevents_encoding_and_oversize_candidate(members, monkeypatch):
    paths = members()
    monkeypatch.setattr(shared, "_MAX_PAYLOAD_BYTES", 100)
    with pytest.raises(coverage.HouseholdCoverageSourceError, match="CANDIDATE_SIZE"):
        coverage.load_authenticated_asec_household_coverage_fields(
            paths, candidate=b"x" * 101
        )
    with pytest.raises(coverage.HouseholdCoverageSourceError, match="ENCODING_SIZE"):
        coverage.load_authenticated_asec_household_coverage_fields(paths)


def test_private_capture_rewrite_refuses_even_with_original_digest(
    members, monkeypatch
):
    paths = members()
    original = shared._capture_owner._snapshot

    def changed(path, destination, *, size):
        digest = original(path, destination, size=size)
        destination.write_bytes(destination.read_bytes().replace(b",09,", b",01,"))
        return digest

    monkeypatch.setattr(shared._capture_owner, "_snapshot", changed)
    with pytest.raises(coverage.HouseholdCoverageSourceError, match="CAPTURE_CHANGED"):
        coverage.load_authenticated_asec_household_coverage_fields(paths)


def test_capture_growth_refuses_below_text_decoder(members, monkeypatch):
    paths = members()
    original = shared._DigestReader

    class GrowingReader(original):
        def __init__(self, raw, size):
            super().__init__(raw, size)
            with open(raw.name, "ab") as output:
                output.write(b"\nextra")

    monkeypatch.setattr(shared, "_DigestReader", GrowingReader)
    with pytest.raises(coverage.HouseholdCoverageSourceError, match="CAPTURE_SIZE"):
        coverage.load_authenticated_asec_household_coverage_fields(paths)


@pytest.mark.parametrize(
    "invalid", [{}, {True: "invented.csv"}, {2025: "invented.csv"}, {2024: object()}]
)
def test_source_requests_require_a_nonempty_closed_cohort_subset(members, invalid):
    members()
    with pytest.raises(coverage.HouseholdCoverageSourceError, match="MEMBER_PATHS"):
        coverage.load_authenticated_asec_household_coverage_fields(invalid)


def test_combined_request_rows_are_bounded_before_any_source_capture(
    members, monkeypatch
):
    paths = members(years=(2023, 2024))
    monkeypatch.setattr(shared, "_MAX_ROWS", 1)

    def forbidden_capture(*args, **kwargs):
        raise AssertionError("Must refuse aggregate allocation before source access")

    monkeypatch.setattr(shared._capture_owner, "_snapshot", forbidden_capture)
    with pytest.raises(coverage.HouseholdCoverageSourceError, match="REQUEST_ROWS"):
        coverage.load_authenticated_asec_household_coverage_fields(paths)


@pytest.fixture(params=["coverage", "weights"])
def capsule_owner(request, members):
    """Both real owners load the same single invented household member."""
    paths = members(
        payload=_csv(
            [("00001", "1", "09", "08", "03", "255212")],
            header=(*coverage.COLUMNS, "HSUP_WGT"),
        )
    )
    if request.param == "coverage":
        return SimpleNamespace(
            module=coverage,
            load=coverage.load_authenticated_asec_household_coverage_fields,
            cls=coverage.AuthenticatedAsecHouseholdCoverageFields,
            error=coverage.HouseholdCoverageSourceError,
            paths=paths,
        )
    return SimpleNamespace(
        module=shared,
        load=shared.load_authenticated_asec_household_weights,
        cls=shared.AuthenticatedAsecHouseholdWeights,
        error=shared.HouseholdWeightSourceError,
        paths=paths,
    )


def _rewritten_payload(source):
    document = source.document
    row = document["records"][0]
    if "H_NUMPER" in row:
        assert row["reported_person_count"] == 3
        row.update(H_NUMPER="16", reported_person_count=16)
    else:
        assert row["weight_integer_units"] == 255212
        row.update(HSUP_WGT="1", weight_integer_units=1)
    document["projection_sha256"] = shared._sha(shared._encode(document["records"]))
    return shared._encode(document)


def test_recomputed_self_checksum_cannot_authorize_rewritten_source(capsule_owner):
    owner = capsule_owner
    source = owner.load(owner.paths)
    payload = _rewritten_payload(source)
    object.__setattr__(source, "payload", payload)
    object.__setattr__(source, "_issued_sha256", shared._sha(payload))
    with pytest.raises(owner.error, match="^SOURCE_MUTATION$"):
        _ = source.document


@pytest.mark.parametrize(
    "kind", ["new", "forged", "empty", "copy", "deepcopy", "pickle"]
)
def test_new_or_copied_value_does_not_inherit_issuance(capsule_owner, kind):
    owner = capsule_owner
    source = owner.load(owner.paths)
    if kind in ("new", "forged", "empty"):
        unissued = object.__new__(owner.cls)
        if kind != "empty":
            payload = _rewritten_payload(source) if kind == "forged" else source.payload
            object.__setattr__(unissued, "payload", payload)
            object.__setattr__(unissued, "_issued_sha256", shared._sha(payload))
    elif kind == "pickle":
        unissued = pickle.loads(pickle.dumps(source))
    else:
        unissued = getattr(copy, kind)(source)
    assert unissued is not source
    with pytest.raises(owner.error, match="^SOURCE_MUTATION$"):
        _ = unissued.document
    assert source.document["source_authenticated"] is True


def test_independent_equal_reissue_retains_its_own_weak_issuance(capsule_owner):
    owner = capsule_owner
    source = owner.load(owner.paths)
    replay = owner.load(owner.paths, candidate=source.payload)
    assert replay == source and replay is not source
    assert replay.payload == source.payload == shared._encode(source.document)
    source_id, replay_id = id(source), id(replay)
    source_ref = weakref.ref(source)
    del source
    gc.collect()
    assert source_ref() is None
    assert source_id not in shared._CAPSULE_ISSUANCE
    assert replay_id in shared._CAPSULE_ISSUANCE
    assert replay.document["source_authenticated"] is True
    del replay
    gc.collect()
    assert replay_id not in shared._CAPSULE_ISSUANCE


def test_verification_decodes_the_checked_payload_snapshot(capsule_owner, monkeypatch):
    owner = capsule_owner
    source = owner.load(owner.paths)
    original_payload = source.payload
    changed_payload = _rewritten_payload(source)
    original_sha = shared._sha

    def mutate_after_hash(payload):
        result = original_sha(payload)
        if payload is original_payload:
            object.__setattr__(source, "payload", changed_payload)
            object.__setattr__(source, "_issued_sha256", original_sha(changed_payload))
        return result

    # Deterministic interleaving of a public object write after checksum work.
    monkeypatch.setattr(shared, "_sha", mutate_after_hash)
    assert shared._encode(source.document) == original_payload
    with pytest.raises(owner.error, match="^SOURCE_MUTATION$"):
        _ = source.document


def test_stale_weak_callback_cannot_remove_reused_identity(capsule_owner, monkeypatch):
    owner = capsule_owner
    first = owner.load(owner.paths)
    second = owner.load(owner.paths)
    first_reference = shared._CAPSULE_ISSUANCE[id(first)][0]
    replacement_entry = shared._CAPSULE_ISSUANCE[id(second)]
    # Model reuse deterministically instead of relying on allocator-specific ids.
    with monkeypatch.context() as patch:
        patch.setitem(shared._CAPSULE_ISSUANCE, id(first), replacement_entry)
        first_reference.__callback__(first_reference)
        assert shared._CAPSULE_ISSUANCE[id(first)] is replacement_entry
    assert first.document == second.document


@pytest.mark.parametrize("payload", [None, bytearray(b"{}"), b"12345"])
def test_private_constructor_bounds_precede_issuance(
    capsule_owner, monkeypatch, payload
):
    owner = capsule_owner
    monkeypatch.setattr(shared, "_MAX_PAYLOAD_BYTES", 4)
    with pytest.raises(owner.error, match="^SOURCE_BYTES$"):
        owner.cls(payload, _token=owner.module._TOKEN)


@pytest.mark.parametrize(
    "keys,reported_size,lookups",
    [
        ([2022, 2023, 2024, 2025], 4, []),
        ([2025], 1, []),
        ([True], 1, []),
        (["2024"], 1, []),
        ([2024, 2024], 2, [2024]),
        ([2022, 2023, 2024, 2025], 3, [2022, 2023, 2024]),
        ([2024, 2023], 1, [2024]),
        ([2024], 2, [2024]),
    ],
)
def test_mapping_bounds_and_keys_precede_value_lookup(
    capsule_owner, members, keys, reported_size, lookups
):
    owner = capsule_owner
    paths = members(years=(2022, 2023, 2024))

    class HostileMapping(Mapping):
        fetched = []
        yielded = 0

        def __len__(self):
            return reported_size

        def __iter__(self):
            for key in keys:
                self.yielded += 1
                yield key

        def __getitem__(self, key):
            self.fetched.append(key)
            return paths.get(key, "must-not-be-opened.csv")

    mapping = HostileMapping()
    with pytest.raises(owner.error, match="^MEMBER_PATHS$"):
        owner.load(mapping)
    assert mapping.fetched == lookups
    assert mapping.yielded <= 4


def test_mapping_size_transition_refuses_before_capture(capsule_owner, monkeypatch):
    owner = capsule_owner

    class ChangingMapping(Mapping):
        size = 1

        def __len__(self):
            return self.size

        def __iter__(self):
            yield 2024

        def __getitem__(self, key):
            self.size = 2
            return owner.paths[key]

    def forbidden_capture(*args, **kwargs):
        raise AssertionError("Mapping transition must refuse before capture")

    monkeypatch.setattr(shared._capture_owner, "_snapshot", forbidden_capture)
    with pytest.raises(owner.error, match="^MEMBER_PATHS$"):
        owner.load(ChangingMapping())


@pytest.mark.parametrize("aliases", ["csv", "native", "both", "owner_module"])
def test_live_csv_reader_drift_refuses_before_capture(
    capsule_owner, monkeypatch, aliases
):
    owner = capsule_owner
    issued = owner.load(owner.paths)
    assert owner.load(owner.paths, candidate=issued.payload) == issued
    original_reader = csv.reader

    def changed_reader(*args, **kwargs):
        reader = original_reader(*args, **kwargs)
        header = next(reader)
        yield header
        for original in reader:
            row = list(original)
            for name, value in (("HSUP_WGT", "1"), ("H_NUMPER", "16")):
                if name in header:
                    row[header.index(name)] = value
            yield row

    def forbidden_capture(*args, **kwargs):
        raise AssertionError("Reader drift must refuse before source capture")

    with monkeypatch.context() as patch:
        patch.setattr(shared._capture_owner, "_snapshot", forbidden_capture)
        if aliases in ("csv", "both"):
            patch.setattr(csv, "reader", changed_reader)
        if aliases in ("native", "both"):
            patch.setattr(native_csv, "reader", changed_reader)
        if aliases == "owner_module":
            patch.setattr(
                owner.module,
                "csv",
                SimpleNamespace(**{**vars(csv), "reader": changed_reader}),
            )
        with pytest.raises(owner.error, match="^SOURCE_CSV_READER_CHANGED$"):
            owner.load(owner.paths)
        with pytest.raises(owner.error, match="^SOURCE_CSV_READER_CHANGED$"):
            _ = issued.document
    assert owner.load(owner.paths, candidate=issued.payload) == issued


@pytest.mark.parametrize("replacement", ["python_wrapper", "wrong_builtin"])
def test_csv_binding_refuses_replaced_aliases_before_helper_import(
    monkeypatch, replacement
):
    name = "microcosm.build.us_runtime.source_csv_builtin"

    def wrapper(*args, **kwargs):
        raise AssertionError("Spoofed reader must never execute")

    wrapper.__module__ = "_csv"
    wrapper.__name__ = "reader"
    wrapper.__self__ = native_csv
    fake = wrapper if replacement == "python_wrapper" else native_csv.writer
    with monkeypatch.context() as patch:
        patch.delitem(sys.modules, name, raising=False)
        patch.setattr(csv, "reader", fake)
        patch.setattr(native_csv, "reader", fake)
        with pytest.raises(ValueError, match="^SOURCE_CSV_READER_CHANGED$"):
            importlib.import_module(name)


def test_csv_binding_helper_bytes_bind_each_capsule(capsule_owner, monkeypatch):
    owner = capsule_owner
    issued = owner.load(owner.paths)
    real_files = shared.resources.files

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

    monkeypatch.setattr(shared.resources, "files", changed_files)
    with pytest.raises(owner.error, match="^PRODUCER_CHANGED$"):
        _ = issued.document


@contextmanager
def transient_csv_rebinding(target, aliases, moment):
    """Time only public alias mutation; never replace a production function."""
    source, first = inspect.getsourcelines(target)
    call_line = next(
        first + i for i, line in enumerate(source) if "strict=True" in line
    )
    real_csv, real_native = csv.reader, native_csv.reader
    fired, calls = [], []

    def restore():
        csv.reader, native_csv.reader = real_csv, real_native

    def replacement(*args, **kwargs):
        calls.append(True)
        restore()
        reader = real_csv(*args, **kwargs)
        header = next(reader)
        yield header
        for original in reader:
            row = list(original)
            for name, value in (
                ("HSUP_WGT", "1"),
                ("H_NUMPER", "16"),
                ("PRPERTYP", "2"),
            ):
                if name in header:
                    row[header.index(name)] = value
            yield row

    def timing(frame, event, arg):
        if frame.f_code is target.__code__:
            trigger = (moment == "entry" and event == "call") or (
                moment == "before_call"
                and event == "line"
                and frame.f_lineno == call_line
            )
            if trigger and not fired:
                fired.append(True)
                csv.reader = replacement
                if aliases == "both":
                    native_csv.reader = replacement
            if event == "return":
                restore()
        return timing

    previous = sys.gettrace()
    sys.settrace(timing)
    try:
        yield fired, calls
    finally:
        sys.settrace(previous)
        restore()


@pytest.mark.parametrize("aliases", ["csv", "both"])
@pytest.mark.parametrize("moment", ["entry", "before_call"])
def test_transient_csv_rebinding_never_supplies_source_cells(
    capsule_owner, aliases, moment
):
    owner = capsule_owner
    control = owner.load(owner.paths)
    assert owner.load(owner.paths, candidate=control.payload) == control
    before = (
        owner.module._producer()
        if owner.module is coverage
        else shared._implementation()
    )
    with transient_csv_rebinding(owner.module._read_capture, aliases, moment) as (
        fired,
        calls,
    ):
        try:
            result = owner.load(owner.paths)
        except owner.error as exc:
            assert str(exc) == "SOURCE_CSV_READER_CHANGED"
        else:
            assert result == control and result.document == control.document
    assert fired and not calls
    assert before == (
        owner.module._producer()
        if owner.module is coverage
        else shared._implementation()
    )
    assert owner.load(owner.paths, candidate=control.payload) == control
