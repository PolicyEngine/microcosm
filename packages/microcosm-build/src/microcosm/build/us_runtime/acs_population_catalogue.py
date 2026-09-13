"""Closed raw ACS catalogue before native population or unit construction.

The full housing lexical projection and raw catalogue remain in memory. Literal
persons are scanned once and records are assembled in whole-household batches;
this does not make national preparation memory-bounded. No domain is classified
here.
"""

from __future__ import annotations

import hashlib
import json
import sys
import weakref
from dataclasses import InitVar, dataclass
from pathlib import Path
from types import FunctionType

from . import acs_housing_universe_source as housing
from . import acs_native_coverage_binding as native
from . import acs_person_coverage_columns as literal
from . import survey_population_domains as domains

PROTOCOL = "microcosm.acs-source-catalogue.v1"
_ARCHIVE_BYTES = 8 * 1024**3
_EXPANDED_BYTES = 16 * 1024**3
_MEMBERS = 64
_SOURCE_ROWS = 6_000_000
_BATCH_PEOPLE = 100_000
_CATALOGUE_BYTES = 2 * 1024**3
_RECEIPT_BYTES = 1024**2
_TOKEN = object()
_ISSUED = {}


class ACSSourceCatalogueError(ValueError):
    """Static refusal, without source identities, paths or cause chains."""


def _require(condition, code):
    if not condition:
        raise ACSSourceCatalogueError(code)


def _local_bytes():
    return {
        module.__name__: housing._sha(Path(module.__file__).read_bytes())
        for module in (sys.modules[__name__], domains)
    }


def _local_runtime():
    """Capture exact loaded code, including generated raw dataclass methods."""
    result = {}
    for module in (sys.modules[__name__], domains):
        for name, value in vars(module).items():
            if isinstance(value, FunctionType):
                result[(module.__name__, name)] = value.__code__
            elif isinstance(value, type) and value.__module__ == module.__name__:
                result[(module.__name__, name)] = value
                for method, function in vars(value).items():
                    if isinstance(function, (staticmethod, classmethod)):
                        function = function.__func__
                    if isinstance(function, property):
                        function = function.fget
                    if isinstance(function, FunctionType):
                        result[(module.__name__, name, method)] = function.__code__
    return result


def _immutable_pins(pins):
    _require(
        type(pins) is tuple
        and len(pins) == 2
        and all(
            type(pin) is tuple
            and len(pin) == 4
            and all(type(value) is str for value in pin[:3])
            and type(pin[3]) is int
            for pin in pins
        ),
        "SOURCE_AUTHORITY_TYPE",
    )
    return pins


def _final_seals(pins, pin_authority):
    # Pure comparisons after the last producer/source I/O, including capture
    # exit. Another read would reopen the same interval this seal closes.
    # _pins() loads the code manifest when the private override is None. Bind
    # that exact authority mode and any override here without reopening I/O.
    current = housing._ARCHIVE_PINS
    if current is not None:
        _immutable_pins(current)
    _require(
        current == pin_authority and (current is None or current == pins),
        "SOURCE_AUTHORITY_CHANGED",
    )
    _require(_local_runtime() == _RUNTIME_AUTHORITY, "PRODUCER_CODE_CHANGED")


def _producer():
    _require(PROTOCOL == "microcosm.acs-source-catalogue.v1", "PROTOCOL_CHANGED")
    _require(
        literal.KEYS == ("SERIALNO", "SPORDER")
        and literal.READ_COLUMNS == ("SERIALNO", "SPORDER", "AGEP", "MIL", "ESR"),
        "LITERAL_CONTRACT_CHANGED",
    )
    limits = (
        _ARCHIVE_BYTES,
        _EXPANDED_BYTES,
        _MEMBERS,
        _SOURCE_ROWS,
        _BATCH_PEOPLE,
        _CATALOGUE_BYTES,
        _RECEIPT_BYTES,
    )
    ceilings = (8 * 1024**3, 16 * 1024**3, 64, 6_000_000, 100_000, 2 * 1024**3, 1024**2)
    _require(
        all(
            type(v) is int and 0 < v <= cap
            for v, cap in zip(limits, ceilings, strict=True)
        ),
        "LIMITS",
    )
    checked = native._producer()
    compiled = {}
    for module in (domains, sys.modules[__name__]):
        native._live_code(module, compiled)
    code = _local_bytes()
    _require(code == _BYTE_AUTHORITY, "PRODUCER_CODE_CHANGED")
    _require(_local_runtime() == _RUNTIME_AUTHORITY, "PRODUCER_CODE_CHANGED")
    return {
        "native_source_closure": checked,
        "catalogue_sha256": code[__name__],
        "raw_domain_types_sha256": code[domains.__name__],
        "limits": list(limits),
    }


def _preflight(paths):
    """Keep full-source gates separate from native's 1M selected-row gate."""
    expanded = 0
    for role, path in paths.items():
        with native.coverage.zipfile.ZipFile(path) as archive:
            members, _prefix = native.coverage._members(archive, role)
            _require(len(members) <= _MEMBERS, "MEMBER_LIMIT")
            expanded += sum(member.file_size for member in members)
            _require(expanded <= _EXPANDED_BYTES, "EXPANDED_LIMIT")
    inventories = {}
    for role, path in paths.items():
        inventory, empty = native.coverage._inventory(path, role, frozenset())
        _require(not empty, "UNEXPECTED_SELECTED_ROWS")
        _require(
            sum(member["rows"] for member in inventory) <= _SOURCE_ROWS,
            "SOURCE_ROW_LIMIT",
        )
        inventories[role] = inventory
    return inventories


def _source_checks(source_dir, paths, pins):
    _require(housing._pins() == pins, "SOURCE_AUTHORITY_CHANGED")
    for role, name, digest, size in pins:
        for path in (source_dir / name, paths[role]):
            housing._path_components(path)
            _require(housing._persisted_sha(path, size) == digest, "SOURCE_CHANGED")


def _charge(record, digest, size):
    raw = native.coverage._json(record, max(0, _CATALOGUE_BYTES - size - 1)) + b"\n"
    size += len(raw)
    _require(size <= _CATALOGUE_BYTES, "CATALOGUE_BYTE_LIMIT")
    digest.update(raw)
    return size


def _collect(projection, paths):
    # Validate the actual persisted full projection, independently of the
    # publicly constructible returned housing capsule's mutable attributes.
    housing._tables(projection)
    hcols = {name: i for i, name in enumerate(projection["household_columns"])}
    pcols = {name: i for i, name in enumerate(projection["person_columns"])}
    by_household = {}
    positions = {}
    people = [None] * len(projection["persons"])
    for index, row in enumerate(projection["persons"]):
        serial, raw_order = row[pcols["SERIALNO"]], row[pcols["SPORDER"]]
        key = (serial, int(raw_order))
        _require(key not in positions, "DUPLICATE_PERSON")
        positions[key] = index
        by_household.setdefault(serial, []).append(index)
    household_keys, started_households = set(), set()
    prospective_size = 0

    def reserve(value, punctuation):
        # An empty household's [] gains exactly the encoded person lengths and
        # one comma between people. Charge before retaining literal fields.
        nonlocal prospective_size
        raw = native.coverage._json(
            value, max(0, _CATALOGUE_BYTES - prospective_size - punctuation)
        )
        prospective_size += len(raw) + punctuation
        _require(prospective_size <= _CATALOGUE_BYTES, "CATALOGUE_BYTE_LIMIT")

    def household_record(row, members):
        return (
            row[hcols["SERIALNO"]],
            row[hcols["TYPEHUGQ"]],
            row[hcols["NP"]],
            row[hcols["WGTP"]],
            row[hcols["source_member"]],
            row[hcols["source_row_ordinal"]],
            members,
        )

    for row in projection["households"]:
        serial, count = row[hcols["SERIALNO"]], int(row[hcols["NP"]])
        _require(serial not in household_keys, "DUPLICATE_HOUSEHOLD")
        household_keys.add(serial)
        _require(count <= _BATCH_PEOPLE, "HOUSEHOLD_EXCEEDS_BATCH")
        reserve(household_record(row, ()), 1)
    _require(set(by_household) <= household_keys, "GLOBAL_PERSON_KEYS")
    contract = literal.coverage_field_contract()

    def consume_row(cells):
        serial, order, age, mil, esr = cells
        _require(
            all(type(v) is str for v in cells)
            and literal.re.fullmatch(r"2024(?:HU|GQ)[0-9]{7}", serial) is not None
            and literal.re.fullmatch(r"[0-9]{1,2}", order) is not None
            and 1 <= int(order) <= 20,
            "LITERAL_PERSON_KEY",
        )
        key = (serial, int(order))
        _require(key in positions, "UNEXPECTED_LITERAL_PERSON")
        index = positions[key]
        _require(people[index] is None, "DUPLICATE_LITERAL_PERSON")
        states = {
            field: literal._field_state(
                age,
                raw,
                minimum_age=contract["fields"][field]["minimum_age"],
                codes=contract["fields"][field]["codes"],
            )
            for field, raw in (("MIL", mil), ("ESR", esr))
        }
        _require(all(type(v) is str for v in states.values()), "LITERAL_TYPES")
        original = projection["persons"][index]
        person = (
            original[pcols["SPORDER"]],
            age,
            esr,
            states["ESR"],
            mil,
            states["MIL"],
            original[pcols["PWGTP"]],
            original[pcols["source_member"]],
            original[pcols["source_row_ordinal"]],
        )
        reserve(person, int(serial in started_households))
        people[index] = person
        started_households.add(serial)

    if positions:
        _members, rows = literal._scan_acs_person_coverage(
            housing.AcsPumsSource(paths["household"], paths["person"], vintage=2024),
            consume_row,
        )
        _require(rows == len(positions), "GLOBAL_PERSON_KEYS")
    _require(all(person is not None for person in people), "GLOBAL_PERSON_KEYS")
    records, vacancies, batch = [], [], []
    digest, size, batches, batch_people = hashlib.sha256(), 0, 0, 0

    def consume():
        nonlocal size, batches
        count = sum(len(by_household.get(row[hcols["SERIALNO"]], ())) for row in batch)
        if count:
            # Preserve the prior selected-reader ceiling on each assembly
            # batch, even though no selected DataFrame is allocated here.
            _require(
                count <= min(_BATCH_PEOPLE, literal.MAX_SELECTED_ROWS),
                "BATCH_PERSON_LIMIT",
            )
            batches += 1
        for row in batch:
            serial = row[hcols["SERIALNO"]]
            members = tuple(people[index] for index in by_household.get(serial, ()))
            record = household_record(row, members)
            _require(len(members) == int(record[2]), "HOUSEHOLD_COMPLETENESS")
            size = _charge(record, digest, size)
            (
                vacancies if int(record[1]) == 1 and int(record[2]) == 0 else records
            ).append(record)

    for row in projection["households"]:
        count = int(row[hcols["NP"]])
        if batch and batch_people + count > _BATCH_PEOPLE:
            consume()
            batch, batch_people = [], 0
        batch.append(row)
        batch_people += count
    if batch:
        consume()
    _require(size == prospective_size, "CATALOGUE_BYTE_ACCOUNTING")
    return (
        tuple(records),
        tuple(vacancies),
        {
            "households": len(household_keys),
            "people": len(positions),
            "occupied_hu": sum(int(r[1]) == 1 for r in records),
            "institutional_gq": sum(int(r[1]) == 2 for r in records),
            "noninstitutional_gq": sum(int(r[1]) == 3 for r in records),
            "vacancies": len(vacancies),
            "literal_batches": batches,
            "canonical_record_bytes": size,
            "canonical_record_sha256": digest.hexdigest(),
        },
    )


def _household(record):
    serial, kind, count, weight, _member, _ordinal, people = record
    key = domains.HouseholdKey(domains.Source.ACS, 2024, 2024, serial)
    return domains.AcsHousehold(
        key,
        kind,
        count,
        weight,
        tuple(
            domains.AcsPerson(order, age, esr, esr_state, mil, mil_state, pwgtp, key)
            for order, age, esr, esr_state, mil, mil_state, pwgtp, _pmember, _pordinal in people
        ),
    )


@dataclass(frozen=True, slots=True)
class _Owned:
    receipt: bytes
    records: tuple
    vacancies: tuple
    source_dir: Path
    paths: tuple
    pins: tuple
    pin_authority: tuple | None
    producer: bytes
    projection_path: Path
    projection_bytes: int
    projection_sha256: str


@dataclass(frozen=True, slots=True, weakref_slot=True, eq=False)
class AuthenticatedACSSourceCatalogue:
    """Process-issued compact receipt; raw views are independent frozen values."""

    payload: bytes
    _token: InitVar[object] = None

    def __post_init__(self, _token):
        _require(_token is _TOKEN, "ISSUANCE_CONSTRUCTOR")

    def validate(self):
        return verify_acs_source_catalogue(self)

    def to_bytes(self):
        return _checked(self).receipt

    @property
    def receipt(self):
        return json.loads(self.to_bytes())

    @property
    def households(self):
        return tuple(_household(record) for record in _checked(self).records)

    @property
    def exclusion_ledger(self):
        return tuple(_household(record) for record in _checked(self).vacancies)

    @property
    def lineage(self):
        owned = _checked(self)
        return {
            r[0]: {
                "household": (r[4], r[5]),
                "persons": tuple((p[0], p[7], p[8]) for p in r[6]),
            }
            for group in (owned.records, owned.vacancies)
            for r in group
        }


def _lookup(value):
    _require(type(value) is AuthenticatedACSSourceCatalogue, "ISSUANCE_TYPE")
    entry = _ISSUED.get(id(value))
    _require(entry is not None and entry[0]() is value, "ISSUANCE_NOT_OWNED")
    owned = entry[1]
    _require(
        type(value.payload) is bytes and value.payload == owned.receipt,
        "ISSUANCE_CHANGED",
    )
    return owned


def _checked(value):
    try:
        owned = _lookup(value)
        before = native.coverage._json(_producer(), _RECEIPT_BYTES)
        _require(before == owned.producer, "PRODUCER_CHANGED")
        _source_checks(owned.source_dir, dict(owned.paths), owned.pins)
        _require(
            housing._persisted_sha(owned.projection_path, owned.projection_bytes)
            == owned.projection_sha256,
            "PROJECTION_CHANGED",
        )
        _require(
            native.coverage._json(_producer(), _RECEIPT_BYTES) == before,
            "PRODUCER_CHANGED",
        )
        _final_seals(owned.pins, owned.pin_authority)
        _require(_lookup(value) is owned, "ISSUANCE_CHANGED")
        return owned
    except ACSSourceCatalogueError:
        raise
    except Exception:
        raise ACSSourceCatalogueError("CATALOGUE_VERIFICATION_REFUSED") from None


def verify_acs_source_catalogue(value):
    _checked(value)
    return value


def issue_acs_source_catalogue(source_dir, *, snapshot_root, candidate=None):
    """Execute closed real owners; candidate bytes never supply source authority."""
    try:
        source_dir = Path(source_dir).absolute()
        snapshot_root = Path(snapshot_root).absolute()
        _require(
            candidate is None
            or (type(candidate) is bytes and len(candidate) <= _RECEIPT_BYTES),
            "CANDIDATE_TYPE_SIZE",
        )
        pin_authority = housing._ARCHIVE_PINS
        if pin_authority is not None:
            _immutable_pins(pin_authority)
        pins = _immutable_pins(housing._pins())
        archives = native._archives(pins)
        _require(sum(p[3] for p in pins) <= _ARCHIVE_BYTES, "ARCHIVE_LIMIT")
        producer = _producer()
        producer_bytes = native.coverage._json(producer, _RECEIPT_BYTES)
        with housing._capture(source_dir, snapshot_root) as (private, paths, captured):
            _require(captured == pins, "SOURCE_AUTHORITY_CHANGED")
            inventories = _preflight(paths)
            source = housing._reconstruct(
                private, paths, pins, None, housing._implementation()
            )
            source_receipt = json.loads(source.receipt_json)
            native._members_equal(source_receipt["members"], inventories)
            projection_path = private / "full-projection.json"
            with projection_path.open("rb") as stream:
                projection_raw = stream.read(housing.ACS_HU_SOURCE_MAX_BYTES + 1)
            _require(
                len(projection_raw) <= housing.ACS_HU_SOURCE_MAX_BYTES,
                "PROJECTION_SIZE",
            )
            projection_sha256 = housing._sha(projection_raw)
            _require(
                projection_sha256 == source_receipt["full_projection_sha256"],
                "PROJECTION_CHANGED",
            )
            projection_bytes = len(projection_raw)
            projection = json.loads(projection_raw)
            del projection_raw
            records, vacancies, counts = _collect(projection, paths)
            _require(
                counts["households"] == sum(m["rows"] for m in inventories["household"])
                and counts["people"] == sum(m["rows"] for m in inventories["person"]),
                "GLOBAL_SOURCE_COMPLETENESS",
            )
            receipt = native.coverage._json(
                {
                    "protocol": PROTOCOL,
                    "producer": producer,
                    "archives": archives,
                    "members": inventories,
                    "source_authenticated": True,
                    "population_binding_authenticated": False,
                    "release_eligible": False,
                    "domain_assignment_authenticated": False,
                    "selection_performed": False,
                    "source_year": 2024,
                    "survey_year": 2024,
                    "scope": "all_occupied_hu_and_both_gq_with_separate_vacancies",
                    "field_contract": literal.coverage_field_contract(),
                    "complete_original_membership": True,
                    "original_anchors_preserved": True,
                    "housing_projection_sha256": projection_sha256,
                    "counts": counts,
                    "canonical_record_schema": [
                        "SERIALNO",
                        "TYPEHUGQ",
                        "NP",
                        "WGTP",
                        "household_member",
                        "household_row",
                        [
                            "SPORDER",
                            "AGEP",
                            "ESR",
                            "ESR_state",
                            "MIL",
                            "MIL_state",
                            "PWGTP",
                            "person_member",
                            "person_row",
                        ],
                    ],
                    "canonical_record_order": "housing_source_member_then_row_ordinal",
                    "execution": {
                        "native_frame_constructed": False,
                        "unit_assignment_executed": False,
                        "full_housing_projection_retained": True,
                        "full_person_scan_per_literal_batch": False,
                        "literal_full_scans": int(bool(counts["people"])),
                        "national_efficiency_claimed": False,
                    },
                },
                _RECEIPT_BYTES,
            )
            if candidate is not None:
                _require(candidate == receipt, "CANDIDATE_MISMATCH")
            _source_checks(source_dir, paths, pins)
            _require(
                housing._persisted_sha(projection_path, projection_bytes)
                == projection_sha256,
                "PROJECTION_CHANGED",
            )
            owned = _Owned(
                receipt,
                records,
                vacancies,
                source_dir,
                tuple(paths.items()),
                pins,
                pin_authority,
                producer_bytes,
                projection_path,
                projection_bytes,
                projection_sha256,
            )
        # Capture exit can read the default source manifest. Recheck producer
        # bytes/live code after that I/O, then finish with pure final seals.
        _require(
            native.coverage._json(_producer(), _RECEIPT_BYTES) == producer_bytes,
            "PRODUCER_CHANGED",
        )
        _final_seals(pins, pin_authority)
        result = AuthenticatedACSSourceCatalogue(receipt, _token=_TOKEN)
        identity = id(result)

        def cleanup(reference):
            entry = _ISSUED.get(identity)
            if entry is not None and entry[0] is reference:
                del _ISSUED[identity]

        _ISSUED[identity] = (weakref.ref(result, cleanup), owned)
        return result
    except ACSSourceCatalogueError:
        raise
    except Exception:
        raise ACSSourceCatalogueError("CATALOGUE_ISSUANCE_REFUSED") from None


# Authority begins at this import, rather than accepting replacement source and
# matching live code as a newly approved implementation on first issuance.
_BYTE_AUTHORITY = _local_bytes()
_RUNTIME_AUTHORITY = _local_runtime()
