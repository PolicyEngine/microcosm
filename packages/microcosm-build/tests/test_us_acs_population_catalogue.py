"""Actual pinned invented archives; never a native Frame or fake source issuer."""

import _csv
import ast
import copy
import csv
import gc
import hashlib
import inspect
import io
import json
import pickle
import sys
import weakref
import zipfile
from pathlib import Path
from types import CodeType, FunctionType, SimpleNamespace

import pytest

from microcosm.build.us_runtime import acs_population_catalogue as owner
from microcosm.frame import Frame


def csv_bytes(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


@pytest.fixture
def invented(tmp_path, monkeypatch):
    source, snapshots = tmp_path / "source", tmp_path / "snapshots"
    source.mkdir()
    snapshots.mkdir()
    households = [
        dict(
            SERIALNO="2024HU0000001",
            TYPEHUGQ="1",
            NP="2",
            WGTP="0010",
            TEN="1",
            ST="01",
            PUMA="00100",
        ),
        dict(
            SERIALNO="2024GQ0000001",
            TYPEHUGQ="2",
            NP="1",
            WGTP="0",
            TEN="",
            ST="01",
            PUMA="00100",
        ),
        dict(
            SERIALNO="2024HU0000002",
            TYPEHUGQ="1",
            NP="0",
            WGTP="20",
            TEN="",
            ST="01",
            PUMA="00100",
        ),
        dict(
            SERIALNO="2024GQ0000002",
            TYPEHUGQ="3",
            NP="1",
            WGTP="0",
            TEN="",
            ST="01",
            PUMA="00100",
        ),
        dict(
            SERIALNO="2024HU0000003",
            TYPEHUGQ="1",
            NP="1",
            WGTP="30",
            TEN="3",
            ST="01",
            PUMA="00100",
        ),
    ]
    people = [
        dict(
            SERIALNO="2024HU0000001",
            SPORDER="01",
            PWGTP="0011",
            AGEP="30",
            MIL="1",
            ESR="4",
        ),
        dict(
            SERIALNO="2024GQ0000001",
            SPORDER="01",
            PWGTP="0077",
            AGEP="40",
            MIL="4",
            ESR="6",
        ),
        dict(
            SERIALNO="2024HU0000001",
            SPORDER="02",
            PWGTP="12",
            AGEP="15",
            MIL="",
            ESR="",
        ),
        dict(
            SERIALNO="2024GQ0000002",
            SPORDER="1",
            PWGTP="78",
            AGEP="50",
            MIL="4",
            ESR="6",
        ),
        dict(
            SERIALNO="2024HU0000003", SPORDER="1", PWGTP="31", AGEP="99", MIL="", ESR=""
        ),
    ]

    def write():
        for _role, filename, prefix, rows in (
            ("household", "csv_hus.zip", "psam_hus", households),
            ("person", "csv_pus.zip", "psam_pus", people),
        ):
            with zipfile.ZipFile(source / filename, "w") as archive:
                empty = (
                    b"SERIALNO,SPORDER,PWGTP,AGEP,MIL,ESR\n"
                    if _role == "person"
                    else b"SERIALNO,TYPEHUGQ,NP,WGTP,TEN,ST,PUMA\n"
                )
                archive.writestr(
                    prefix + "a.csv", csv_bytes(rows[:2]) if rows[:2] else empty
                )
                archive.writestr(
                    prefix + "b.csv", csv_bytes(rows[2:]) if rows[2:] else empty
                )
        pins = tuple(
            (
                role,
                name,
                hashlib.sha256((source / name).read_bytes()).hexdigest(),
                (source / name).stat().st_size,
            )
            for role, name in (("household", "csv_hus.zip"), ("person", "csv_pus.zip"))
        )
        monkeypatch.setattr(owner.housing, "_ARCHIVE_PINS", pins)

    monkeypatch.setattr(
        owner.housing.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=64 * 1024**3),
    )
    monkeypatch.setattr(owner, "_BATCH_PEOPLE", 2)
    write()
    return SimpleNamespace(
        source=source,
        snapshots=snapshots,
        households=households,
        people=people,
        write=write,
        root=tmp_path,
    )


def issue(fixture, **kwargs):
    return owner.issue_acs_source_catalogue(
        fixture.source, snapshot_root=fixture.snapshots, **kwargs
    )


def test_real_complete_catalogue_without_population_or_classification(invented):
    forbidden = []

    def trace(frame, event, arg):
        module = frame.f_globals.get("__name__", "")
        if event == "call" and (
            frame.f_code is Frame.__init__.__code__
            or module.startswith("microunit")
            or frame.f_code.co_name
            in {
                "build_acs_pums_unit_frame",
                "prepare_acs_housing_population",
                "classify_household",
                "classify_households",
            }
        ):
            forbidden.append(frame.f_code.co_name)
            raise AssertionError("Population construction/classification occurred")

    previous = sys.getprofile()
    sys.setprofile(trace)
    try:
        result = issue(invented)
        rows = result.households
        receipt = result.receipt
    finally:
        sys.setprofile(previous)
    assert not forbidden
    assert result.validate() is result
    assert owner.verify_acs_source_catalogue(result) is result
    assert len(rows) == 4 and len(result.exclusion_ledger) == 1
    assert {r.key.native_id for r in rows} == {
        r["SERIALNO"] for r in invented.households if r["NP"] != "0"
    }
    row = rows[0]
    assert row.wgtp == "0010"
    assert [(p.sporder, p.age, p.esr, p.mil, p.pwgtp) for p in row.persons] == [
        ("01", "30", "4", "1", "0011"),
        ("02", "15", "", "", "12"),
    ]
    assert row.persons[1].esr_state == "outside_age_universe"
    assert rows[-1].persons[0].esr_state == "missing_in_universe"
    assert rows[1].persons[0].pwgtp == "0077"
    assert result.lineage[row.key.native_id]["persons"] == (
        ("01", "psam_pusa.csv", 1),
        ("02", "psam_pusb.csv", 1),
    )
    assert receipt["counts"]["people"] == 5
    assert receipt["counts"]["literal_batches"] == 3
    assert (
        receipt["source_authenticated"]
        and not receipt["population_binding_authenticated"]
    )
    assert (
        not receipt["domain_assignment_authenticated"]
        and not receipt["selection_performed"]
    )
    assert not receipt["execution"]["full_person_scan_per_literal_batch"]
    assert receipt["execution"]["literal_full_scans"] == 1
    assert issue(invented, candidate=result.to_bytes()).to_bytes() == result.to_bytes()
    (invented.root / "catalogue.receipt.json").write_bytes(result.to_bytes())
    held = owner._lookup(result)
    (invented.root / "invented-records.json").write_bytes(
        owner.native.coverage._json(
            {"households": held.records, "vacancy_ledger": held.vacancies}
        )
    )


def test_independent_views_cannot_rewrite_issued_rows(invented):
    result = issue(invented)
    original = result.to_bytes()
    row = result.households[0]
    object.__setattr__(row, "wgtp", "9999")
    object.__setattr__(row.persons[0], "esr", "6")
    result.receipt["source_authenticated"] = False
    result.lineage.clear()
    assert result.households[0].wgtp == "0010"
    assert result.households[0].persons[0].esr == "4"
    assert result.to_bytes() == original


def expected_records():
    """Explicit invented source-order records, independent of collector logic."""
    return (
        (
            "2024HU0000001",
            "1",
            "2",
            "0010",
            "psam_husa.csv",
            1,
            (
                (
                    "01",
                    "30",
                    "4",
                    "observed_code",
                    "1",
                    "observed_code",
                    "0011",
                    "psam_pusa.csv",
                    1,
                ),
                (
                    "02",
                    "15",
                    "",
                    "outside_age_universe",
                    "",
                    "outside_age_universe",
                    "12",
                    "psam_pusb.csv",
                    1,
                ),
            ),
        ),
        (
            "2024GQ0000001",
            "2",
            "1",
            "0",
            "psam_husa.csv",
            2,
            (
                (
                    "01",
                    "40",
                    "6",
                    "observed_code",
                    "4",
                    "observed_code",
                    "0077",
                    "psam_pusa.csv",
                    2,
                ),
            ),
        ),
        ("2024HU0000002", "1", "0", "20", "psam_husb.csv", 1, ()),
        (
            "2024GQ0000002",
            "3",
            "1",
            "0",
            "psam_husb.csv",
            2,
            (
                (
                    "1",
                    "50",
                    "6",
                    "observed_code",
                    "4",
                    "observed_code",
                    "78",
                    "psam_pusb.csv",
                    2,
                ),
            ),
        ),
        (
            "2024HU0000003",
            "1",
            "1",
            "30",
            "psam_husb.csv",
            3,
            (
                (
                    "1",
                    "99",
                    "",
                    "missing_in_universe",
                    "",
                    "missing_in_universe",
                    "31",
                    "psam_pusb.csv",
                    3,
                ),
            ),
        ),
    )


@pytest.mark.parametrize("batch_limit", [2, 3, 5])
def test_explicit_canonical_records_and_bytes_ignore_batch_boundaries(
    invented, monkeypatch, batch_limit
):
    monkeypatch.setattr(owner, "_BATCH_PEOPLE", batch_limit)
    result = issue(invented)
    held, expected = owner._lookup(result), expected_records()
    assert held.records == (expected[0], expected[1], expected[3], expected[4])
    assert held.vacancies == (expected[2],)
    raw = b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        + b"\n"
        for row in expected
    )
    assert result.receipt["counts"]["canonical_record_bytes"] == len(raw)
    assert (
        result.receipt["counts"]["canonical_record_sha256"]
        == hashlib.sha256(raw).hexdigest()
    )


@pytest.mark.parametrize("batch_limit", [2, 5])
def test_catalogue_literal_member_scan_count_is_independent_of_batches(
    invented, monkeypatch, batch_limit
):
    monkeypatch.setattr(owner, "_BATCH_PEOPLE", batch_limit)
    scanner = owner.literal._scan_acs_person_coverage.__code__
    active, scans, literal_opens, all_person_opens = set(), [], [], []

    def trace(frame, event, arg):
        if frame.f_code is scanner:
            if event == "call":
                scans.append(True)
                active.add(id(frame))
            elif event == "return":
                active.remove(id(frame))
        if event == "call" and frame.f_code is zipfile.ZipFile.open.__code__:
            name = frame.f_locals["name"]
            name = name.filename if isinstance(name, zipfile.ZipInfo) else name
            if name.startswith("psam_pus"):
                all_person_opens.append(name)
                if active:
                    literal_opens.append(name)

    previous = sys.getprofile()
    sys.setprofile(trace)
    try:
        result = issue(invented)
    finally:
        sys.setprofile(previous)
    assert scans == [True] and not active
    assert literal_opens == ["psam_pusa.csv", "psam_pusb.csv"]
    assert all_person_opens == literal_opens * 3
    assert result.receipt["counts"]["literal_batches"] == (3 if batch_limit == 2 else 1)


@pytest.mark.parametrize(
    "mutation,code",
    [
        ("duplicate", "DUPLICATE_LITERAL_PERSON"),
        ("extra", "UNEXPECTED_LITERAL_PERSON"),
        ("orphan", "UNEXPECTED_LITERAL_PERSON"),
        ("missing", "GLOBAL_PERSON_KEYS"),
        ("key_suffix", "LITERAL_PERSON_KEY"),
    ],
)
def test_collector_reconciles_every_literal_slot_after_housing_validation(
    invented, mutation, code
):
    # Exercise this private reconciliation separately from the unchanged
    # housing/preflight checks. A mismatched helper input issues no authority.
    original = issue(invented)
    projection = json.loads(owner._lookup(original).projection_path.read_bytes())
    if mutation == "duplicate":
        invented.people.append(dict(invented.people[0], SPORDER="1"))
    elif mutation == "extra":
        invented.people.append(dict(invented.people[0], SPORDER="3"))
    elif mutation == "orphan":
        invented.people[-1]["SERIALNO"] = "2024HU0000099"
    elif mutation == "missing":
        invented.people.pop()
    else:
        invented.people[-1]["SPORDER"] = "1 "
    invented.write()
    paths = {
        "household": invented.source / "csv_hus.zip",
        "person": invented.source / "csv_pus.zip",
    }
    with pytest.raises(owner.ACSSourceCatalogueError, match=f"^{code}$"):
        owner._collect(projection, paths)


def test_prospective_byte_charge_refuses_before_retaining_the_next_person(
    invented, monkeypatch
):
    original = issue(invented)
    held = owner._lookup(original)
    projection = json.loads(held.projection_path.read_bytes())
    expected = expected_records()

    def encoded(value):
        return json.dumps(value, separators=(",", ":"), allow_nan=False).encode()

    overhead = sum(len(encoded((*row[:6], ()))) + 1 for row in expected)
    monkeypatch.setattr(
        owner, "_CATALOGUE_BYTES", overhead + len(encoded(expected[0][6][0]))
    )
    retained = []

    def trace(frame, event, arg):
        if (
            event == "return"
            and frame.f_globals.get("__name__") == owner.__name__
            and frame.f_code.co_qualname == "_collect.<locals>.consume_row"
        ):
            retained.append(
                sum(person is not None for person in frame.f_locals["people"])
            )

    previous = sys.getprofile()
    sys.setprofile(trace)
    try:
        with pytest.raises(
            owner.native.coverage.ACSCoverageAuthenticationError,
            match="^CANONICAL_SIZE$",
        ):
            owner._collect(projection, dict(held.paths))
    finally:
        sys.setprofile(previous)
    assert retained == [1, 1]


def test_source_order_and_original_person_order_survive_reverse_zip_insertion(invented):
    # Keep serials unsorted and make the split household's raw person order
    # descending; canonical order remains source-member then source-row order.
    invented.people[0]["SPORDER"], invented.people[2]["SPORDER"] = "02", "01"
    invented.write()
    first = issue(invented)
    before = owner._lookup(first)
    for name in ("csv_hus.zip", "csv_pus.zip"):
        path = invented.source / name
        with zipfile.ZipFile(path) as archive:
            members = [
                (member.filename, archive.read(member)) for member in archive.infolist()
            ]
        with zipfile.ZipFile(path, "w") as archive:
            for member, raw in reversed(members):
                archive.writestr(member, raw)
    pins = tuple(
        (
            role,
            name,
            hashlib.sha256((invented.source / name).read_bytes()).hexdigest(),
            (invented.source / name).stat().st_size,
        )
        for role, name in (("household", "csv_hus.zip"), ("person", "csv_pus.zip"))
    )
    # New physical ZIPs have their own invented pins; compare source records,
    # not archive/producer receipts from different physical artifacts.
    owner.housing._ARCHIVE_PINS = pins
    second = issue(invented)
    after = owner._lookup(second)
    assert after.records == before.records and after.vacancies == before.vacancies
    assert [p[0] for p in after.records[0][6]] == ["02", "01"]


@pytest.mark.parametrize("kind", ["new", "copy", "deepcopy", "pickle", "payload"])
def test_foreign_changed_or_copied_authority_refuses(invented, kind):
    result = issue(invented)
    if kind == "new":
        other = object.__new__(owner.AuthenticatedACSSourceCatalogue)
        object.__setattr__(other, "payload", result.payload)
    elif kind == "payload":
        other = result
        raw = json.loads(result.payload)
        raw["counts"]["people"] = 16
        object.__setattr__(other, "payload", owner.native.coverage._json(raw))
    else:
        try:
            other = {
                "copy": copy.copy,
                "deepcopy": copy.deepcopy,
                "pickle": lambda x: pickle.loads(pickle.dumps(x)),
            }[kind](result)
        except (TypeError, owner.ACSSourceCatalogueError):
            return
    with pytest.raises(
        owner.ACSSourceCatalogueError, match="^ISSUANCE_(NOT_OWNED|CHANGED)$"
    ):
        other.validate()


def test_independent_reissue_and_weak_cleanup(invented):
    one, two = issue(invented), issue(invented)
    assert one.to_bytes() == two.to_bytes() and one is not two
    identity, reference = id(one), weakref.ref(one)
    del one
    gc.collect()
    assert reference() is None and identity not in owner._ISSUED
    assert two.validate() is two


@pytest.mark.parametrize("kind", ["duplicate", "orphan", "missing", "late_extra", "np"])
def test_global_complete_roster_refuses_across_members_and_batches(invented, kind):
    if kind == "duplicate":
        invented.people[-1] = dict(invented.people[0])
    elif kind == "orphan":
        invented.people[-1]["SERIALNO"] = "2024HU0000099"
    elif kind == "missing":
        invented.people.pop()
    elif kind == "late_extra":
        extra = dict(invented.people[0], SPORDER="3")
        invented.people.append(extra)
    else:
        invented.households[0]["NP"] = "3"
    invented.write()
    with pytest.raises(owner.ACSSourceCatalogueError):
        issue(invented)


@pytest.mark.parametrize(
    "cap,value",
    [
        ("_ARCHIVE_BYTES", 1),
        ("_EXPANDED_BYTES", 1),
        ("_MEMBERS", 1),
        ("_SOURCE_ROWS", 1),
        ("_BATCH_PEOPLE", 1),
        ("_CATALOGUE_BYTES", 1),
        ("_RECEIPT_BYTES", 1),
    ],
)
def test_prospective_small_caps_refuse(invented, monkeypatch, cap, value):
    monkeypatch.setattr(owner, cap, value)
    with pytest.raises(owner.ACSSourceCatalogueError):
        issue(invented)


@pytest.mark.parametrize("escaped_literals", [False, True])
def test_byte_ceiling_exact_boundary_is_record_based(
    invented, monkeypatch, escaped_literals
):
    if escaped_literals:
        invented.people[0]["MIL"] = 'x,"\\\t'
        invented.people[0]["ESR"] = "é"
        invented.write()
    size = issue(invented).receipt["counts"]["canonical_record_bytes"]
    monkeypatch.setattr(owner, "_CATALOGUE_BYTES", size)
    assert issue(invented).receipt["counts"]["canonical_record_bytes"] == size
    monkeypatch.setattr(owner, "_CATALOGUE_BYTES", size - 1)
    with pytest.raises(owner.ACSSourceCatalogueError):
        issue(invented)


def test_candidate_is_reconstructed_and_cannot_authorize_changed_values(invented):
    result = issue(invented)
    candidate = result.to_bytes()
    invented.people[0]["ESR"] = "6"
    invented.write()
    with pytest.raises(owner.ACSSourceCatalogueError, match="^CANDIDATE_MISMATCH$"):
        issue(invented, candidate=candidate)
    assert issue(invented).households[0].persons[0].esr == "6"
    with pytest.raises(
        owner.ACSSourceCatalogueError, match="^SOURCE_AUTHORITY_CHANGED$"
    ):
        result.validate()


@pytest.mark.parametrize("target", ["original", "projection", "producer"])
def test_live_source_projection_and_producer_changes_refuse(
    invented, monkeypatch, target
):
    result = issue(invented)
    if target == "original":
        path = invented.source / "csv_pus.zip"
        raw = path.read_bytes()
        path.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
    elif target == "projection":
        owned = owner._lookup(result)
        raw = owned.projection_path.read_bytes()
        owned.projection_path.chmod(0o600)
        owned.projection_path.write_bytes(b" " + raw[1:])
    else:
        monkeypatch.setattr(owner, "_BATCH_PEOPLE", 3)
    with pytest.raises(owner.ACSSourceCatalogueError):
        result.validate()


def test_new_owner_static_surface_excludes_population_selection_and_classification():
    tree = ast.parse(Path(owner.__file__).read_text())
    calls = {
        node.func.id if isinstance(node.func, ast.Name) else node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    assert not calls & {
        "Frame",
        "build_acs_pums_unit_frame",
        "prepare_acs_housing_population",
        "assign_us_unit_structure",
        "classify_household",
        "classify_households",
        "issue_acs_native_coverage",
        "sample",
        "choice",
    }
    assert {"_reconstruct", "_scan_acs_person_coverage", "_live_code"} <= calls
    assert "read_acs_person_coverage_columns" not in calls


@pytest.mark.parametrize("mutation", ["source", "limit"])
def test_mutations_at_literal_return_refuse(invented, mutation):
    target = owner.literal._scan_acs_person_coverage
    fired = []

    def timing(frame, event, arg):
        if frame.f_code is target.__code__ and event == "return" and not fired:
            fired.append(True)
            if mutation == "source":
                path = invented.source / "csv_pus.zip"
                raw = path.read_bytes()
                path.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
            else:
                owner._BATCH_PEOPLE = 3

    before, previous = owner._BATCH_PEOPLE, sys.getprofile()
    sys.setprofile(timing)
    try:
        with pytest.raises(owner.ACSSourceCatalogueError):
            issue(invented)
    finally:
        sys.setprofile(previous)
        owner._BATCH_PEOPLE = before
    assert fired


def test_returned_housing_capsule_cannot_replace_actual_projected_anchors(invented):
    target, fired = owner.housing._reconstruct, []

    def timing(frame, event, arg):
        if frame.f_code is target.__code__ and event == "return" and not fired:
            fired.append(True)
            changed = json.loads(arg.projection_json)
            changed["households"][0][changed["household_columns"].index("WGTP")] = (
                "9999"
            )
            object.__setattr__(arg, "projection_json", owner.housing._json(changed))

    previous = sys.getprofile()
    sys.setprofile(timing)
    try:
        result = issue(invented)
    finally:
        sys.setprofile(previous)
    assert fired and result.households[0].wgtp == "0010"


@pytest.mark.parametrize("aliases", ["csv", "native", "both"])
@pytest.mark.parametrize("moment", ["entry", "before_call"])
def test_actual_csv_alias_timing_cannot_supply_catalogue_values(
    invented, aliases, moment
):
    target = owner.literal._literal_csv_records
    source, first = inspect.getsourcelines(target)
    call_line = next(
        first + i for i, line in enumerate(source) if "reader = csv_reader(" in line
    )
    original, native, fired, calls = csv.reader, _csv.reader, [], []

    def restore():
        csv.reader, _csv.reader = original, native

    def replacement(*args, **kwargs):
        calls.append(True)
        restore()
        return original(*args, **kwargs)

    def timing(frame, event, arg):
        if frame.f_code is target.__code__:
            trigger = (moment == "entry" and event == "call") or (
                moment == "before_call"
                and event == "line"
                and frame.f_lineno == call_line
            )
            if trigger and not fired:
                fired.append(True)
                if aliases in ("csv", "both"):
                    csv.reader = replacement
                if aliases in ("native", "both"):
                    _csv.reader = replacement
            if event == "return":
                restore()
        return timing

    previous = sys.gettrace()
    sys.settrace(timing)
    try:
        if moment == "entry":
            with pytest.raises(owner.ACSSourceCatalogueError):
                issue(invented)
        else:
            result = issue(invented)
    finally:
        sys.settrace(previous)
        restore()
    assert fired and not calls
    if moment == "before_call":
        assert result.households[0].persons[0].esr == "4"


def test_public_constructor_has_no_authority(invented):
    raw = issue(invented).to_bytes()
    with pytest.raises(owner.ACSSourceCatalogueError, match="^ISSUANCE_CONSTRUCTOR$"):
        owner.AuthenticatedACSSourceCatalogue(raw)


def test_full_source_can_exceed_unchanged_selected_reader_cap(invented, monkeypatch):
    monkeypatch.setattr(owner.literal, "MAX_SELECTED_ROWS", 2)
    result = issue(invented)
    assert result.receipt["counts"]["people"] == 5
    assert result.receipt["counts"]["literal_batches"] == 3


def test_vacancy_only_catalogue_has_no_literal_batch(invented):
    invented.households[:] = [dict(invented.households[2])]
    invented.people.clear()
    invented.write()
    result = issue(invented)
    assert result.households == ()
    assert len(result.exclusion_ledger) == 1
    assert result.exclusion_ledger[0].persons == ()
    assert result.receipt["counts"]["literal_batches"] == 0
    assert result.receipt["execution"]["literal_full_scans"] == 0


def test_reordered_literal_column_declaration_refuses_before_capture(
    invented, monkeypatch
):
    monkeypatch.setattr(
        owner.literal, "READ_COLUMNS", ("SERIALNO", "SPORDER", "ESR", "MIL", "AGEP")
    )
    with pytest.raises(
        owner.ACSSourceCatalogueError, match="^LITERAL_CONTRACT_CHANGED$"
    ):
        issue(invented)
    assert list(invented.snapshots.iterdir()) == []


@pytest.mark.parametrize(
    "interface",
    ["validate", "to_bytes", "receipt", "households", "exclusion_ledger", "lineage"],
)
@pytest.mark.parametrize("moment", ["source_checks", "final_producer"])
def test_final_source_registry_seal_all_borrows(
    invented, monkeypatch, interface, moment
):
    result = issue(invented)
    pins = owner.housing._ARCHIVE_PINS
    changed = tuple((r, n, "0" * 64, s) for r, n, d, s in pins)
    fired, producers = [], []

    def timing(frame, event, arg):
        if event != "return":
            return
        if frame.f_code is owner._producer.__code__:
            producers.append(True)
        if not fired and (
            moment == "source_checks"
            and frame.f_code is owner._source_checks.__code__
            or moment == "final_producer"
            and frame.f_code is owner._producer.__code__
            and len(producers) == 2
        ):
            fired.append(True)
            monkeypatch.setattr(owner.housing, "_ARCHIVE_PINS", changed)

    previous = sys.getprofile()
    try:
        sys.setprofile(timing)
        with pytest.raises(
            owner.ACSSourceCatalogueError, match="SOURCE_AUTHORITY_CHANGED"
        ):
            value = getattr(result, interface)
            if callable(value):
                value()
    finally:
        sys.setprofile(previous)
    assert fired


@pytest.mark.parametrize("semantic", [False, True])
def test_import_time_bytes_and_matching_live_code_remain_authority(
    invented, monkeypatch, semantic
):
    source = Path(owner.__file__).read_text()
    if semantic:
        before = "        weight,\n        tuple("
        assert source.count(before) == 1
        source = source.replace(before, '        "9999",\n        tuple(')
    else:
        source += "\n# invented drift after the real module import\n"
    replacement = invented.root / "changed-catalogue.py"
    replacement.write_text(source)
    monkeypatch.setattr(owner, "__file__", str(replacement))
    if semantic:
        code = next(
            c
            for c in compile(
                source, str(replacement), "exec", dont_inherit=True
            ).co_consts
            if isinstance(c, CodeType) and c.co_name == "_household"
        )
        monkeypatch.setattr(owner._household, "__code__", code)
    with pytest.raises(owner.ACSSourceCatalogueError, match="PRODUCER_CODE_CHANGED"):
        issue(invented)


@pytest.mark.parametrize("change", ["bytes", "generated_constructor"])
def test_raw_domain_import_authority_precedes_first_issuance(
    invented, monkeypatch, change
):
    if change == "bytes":
        replacement = invented.root / "changed-raw-types.py"
        replacement.write_bytes(
            Path(owner.domains.__file__).read_bytes() + b"\n# drift\n"
        )
        monkeypatch.setattr(owner.domains, "__file__", str(replacement))
    else:
        # Native live-code traversal explicitly skips generated <string> code;
        # the local import-time seal must still bind those actual constructors.
        code = next(
            c
            for c in compile(
                "def __init__(self, *args, **kwargs): pass", "<string>", "exec"
            ).co_consts
            if isinstance(c, CodeType)
        )
        monkeypatch.setattr(
            owner.domains.AcsHousehold,
            "__init__",
            FunctionType(code, vars(owner.domains)),
        )
    with pytest.raises(owner.ACSSourceCatalogueError, match="PRODUCER_CODE_CHANGED"):
        issue(invented)


@pytest.mark.parametrize("issuance", [False, True])
@pytest.mark.parametrize("change", ["pins", "live_code"])
def test_final_producer_return_seal_issuance_and_borrow(
    invented, monkeypatch, issuance, change
):
    result = None if issuance else issue(invented)
    fired, producers = [], []
    original = owner._household

    def changed(record):
        return original(record)

    def timing(frame, event, arg):
        if event == "return" and frame.f_code is owner._producer.__code__:
            producers.append(True)
            if len(producers) == 2:
                fired.append(True)
                if change == "pins":
                    monkeypatch.setattr(
                        owner.housing,
                        "_ARCHIVE_PINS",
                        tuple(
                            (r, n, "0" * 64, s)
                            for r, n, d, s in owner.housing._ARCHIVE_PINS
                        ),
                    )
                else:
                    monkeypatch.setattr(owner, "_household", changed)

    previous = sys.getprofile()
    try:
        sys.setprofile(timing)
        with pytest.raises(owner.ACSSourceCatalogueError):
            issue(invented) if issuance else result.households
    finally:
        sys.setprofile(previous)
    assert fired


@pytest.mark.parametrize(
    "shape", ["outer_list", "inner_list", "string_subclass", "bool_size"]
)
def test_source_authority_requires_deeply_immutable_primitive_pins(
    invented, monkeypatch, shape
):
    original = owner.housing._ARCHIVE_PINS
    if shape == "outer_list":
        changed = list(original)
    elif shape == "inner_list":
        changed = tuple(list(pin) for pin in original)
    elif shape == "string_subclass":

        class MutableString(str):
            pass

        changed = ((MutableString(original[0][0]), *original[0][1:]), original[1])
    else:
        changed = ((*original[0][:3], True), original[1])
    monkeypatch.setattr(owner.housing, "_ARCHIVE_PINS", changed)
    with pytest.raises(owner.ACSSourceCatalogueError, match="SOURCE_AUTHORITY_TYPE"):
        issue(invented)
    assert list(invented.snapshots.iterdir()) == []
