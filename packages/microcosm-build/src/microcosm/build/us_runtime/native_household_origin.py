"""Private native household origins, authenticated from original source members.

ACS uses the existing archive codec. ASEC requires a separately reviewed original
HOUSEHOLD CSV member registry: prepared household HDF and PERSON-member pins are
not substitutes. The three registered members have an independently reviewed
archive/member audit and a complete accepted-population identifier bridge.

This module owns source lineage, not population treatments or benchmark scoring.
Source capsules can be issued only by a reader. Graph transport is distinct from
source authentication; its consumer must retain the actual producer artifact edge.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import InitVar, asdict, dataclass, field
from pathlib import Path

import numpy as np

from microcosm.build.cd_benchmark.origin import (
    AcsHouseholdOrigin,
    AsecHouseholdOrigin,
    SourceMember,
    origin_key,
)
from microcosm.frame import US_SCHEMA, Frame
from microcosm.graph.canonical import canonical_json

from . import acs_housing_universe_source as acs
from .asec_student_controls import _snapshot
from .education_assistance_source import ASEC_EDUCATION_ASSISTANCE_ARCHIVES
from .support_provenance import (
    support_channel_column,
    support_clone_index_column,
    support_source_id_column,
)

_TOKEN = object()
_BOUND_TOKEN = object()
SOURCE_SCHEMA = "microcosm.us.native-household-origin-source.v1"
BINDING_SCHEMA = "microcosm.us.population-household-origins.v1"
MAX_SOURCE_BYTES = 512 * 1024**2
_SHA = re.compile(r"[0-9a-f]{64}")


class NativeOriginError(ValueError):
    """Bounded refusal code, without a private identifier or source token."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise NativeOriginError(reason)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _integer(value: object) -> int:
    _require(
        not isinstance(value, (bool, np.bool_))
        and isinstance(value, (int, np.integer)),
        "NATIVE_INTEGER",
    )
    return int(value)


def _carried_native_integer(value: object) -> int:
    # Union assembly may store a source-only integer column as canonical text
    # beside the other arm's missing values. Accept only its lossless canonical
    # decimal representation; source membership is still established against
    # the independently authenticated integer H_SEQ roster, never this cast.
    if isinstance(value, str):
        _require(
            re.fullmatch(r"[1-9][0-9]{0,17}", value) is not None,
            "CARRIED_NATIVE_INTEGER",
        )
        return int(value)
    return _integer(value)


@dataclass(frozen=True)
class AsecNativeMemberPin:
    """A reviewed original household member, not authority merely by construction.

    A root source audit must establish membership in the already-pinned Census
    archive before adding this declaration to the closed registry below.
    """

    income_year: int
    survey_year: int
    canonical_member_id: str
    member_name: str
    archive_sha256: str
    member_sha256: str
    size_bytes: int
    rows: int

    def __post_init__(self) -> None:
        _require(
            type(self.income_year) is int
            and self.income_year in ASEC_EDUCATION_ASSISTANCE_ARCHIVES
            and self.survey_year == self.income_year + 1,
            "ASEC_MEMBER_COHORT",
        )
        _require(
            isinstance(self.canonical_member_id, str)
            and bool(self.canonical_member_id)
            and self.member_name == f"hhpub{str(self.survey_year)[-2:]}.csv",
            "ASEC_MEMBER_IDENTITY",
        )
        _require(
            isinstance(self.member_sha256, str)
            and _SHA.fullmatch(self.member_sha256) is not None
            and self.archive_sha256
            == ASEC_EDUCATION_ASSISTANCE_ARCHIVES[self.income_year].zip_sha256,
            "ASEC_MEMBER_ARCHIVE_BINDING",
        )
        _require(
            type(self.size_bytes) is int
            and 0 < self.size_bytes <= MAX_SOURCE_BYTES
            and type(self.rows) is int
            and 0 < self.rows <= 1_000_000,
            "ASEC_MEMBER_BOUNDS",
        )


# Root acceptance: us-asec-native-household-root-acceptance.json, SHA-256
# f997bbdc3573ed346d0e84d29f90c7b7ded9a12720c036d2789a7d561a04d602.
# The original archive/member bytes and all accepted attachment/selected native
# household identifiers were independently verified. Tests use invented pins.
_ASEC_MEMBER_PINS: tuple[AsecNativeMemberPin, ...] = (
    AsecNativeMemberPin(
        income_year=2022,
        survey_year=2023,
        canonical_member_id="census/cps/asec/2023/household/hhpub23.csv",
        member_name="hhpub23.csv",
        archive_sha256="d2e000250782adfbdd7f29c82b66d866591a30f0d330496698ec19f9c784ce11",
        member_sha256="c88192cb3c963a90ea98022606112a01ade0eb8e2c17df4149052970a5d5246f",
        size_bytes=30_259_450,
        rows=88_978,
    ),
    AsecNativeMemberPin(
        income_year=2023,
        survey_year=2024,
        canonical_member_id="census/cps/asec/2024/household/hhpub24.csv",
        member_name="hhpub24.csv",
        archive_sha256="cdb39cdac34bef99dd0940ab28e306f692404c2eea44d85dfd634214872a0a09",
        member_sha256="b12c1078e02766a4033ba1a031abdd6b9ef9df546aee5886dc8439a35abaa2ac",
        size_bytes=30_448_089,
        rows=89_473,
    ),
    AsecNativeMemberPin(
        income_year=2024,
        survey_year=2025,
        canonical_member_id="census/cps/asec/2025/household/hhpub25.csv",
        member_name="hhpub25.csv",
        archive_sha256="318845a2b5e0034eb2973898de1738f4df0025727de38499e7669cb9c0deef0b",
        member_sha256="b5b7351d5d4e5d79ff189f1d90096b16b2d4749671c14328ec6474cfe83ce116",
        size_bytes=33_298_479,
        rows=88_932,
    ),
)


def asec_member_registry() -> list[dict[str, object]]:
    """Return the exact live registry for the separately scoped graph identity."""
    return [asdict(pin) for pin in _ASEC_MEMBER_PINS]


@dataclass(frozen=True)
class AuthenticatedNativeOriginSource:
    """Owned source projection. Its constructor is not a public admission path."""

    payload: bytes
    _token: InitVar[object] = None
    _issued_sha256: str = field(init=False, repr=False)

    def __post_init__(self, _token: object) -> None:
        _require(_token is _TOKEN, "SOURCE_CONSTRUCTOR")
        object.__setattr__(self, "_issued_sha256", _sha(self.payload))

    @property
    def document(self) -> dict:
        _require(_sha(self.payload) == self._issued_sha256, "SOURCE_MUTATION")
        return _source_document(self.payload)


def _source_document(payload: bytes) -> dict:
    _require(type(payload) is bytes and len(payload) <= MAX_SOURCE_BYTES, "SOURCE_SIZE")
    try:
        value = json.loads(payload)
        _require(canonical_json(value) == payload, "SOURCE_CANONICAL")
        _require(
            set(value)
            == {
                "schema",
                "arm",
                "members",
                "records",
                "source_receipt_sha256",
                "release_eligible",
            }
            and value["schema"] == SOURCE_SCHEMA
            and value["arm"] in ("acs", "asec")
            and value["release_eligible"] is False
            and _SHA.fullmatch(value["source_receipt_sha256"]) is not None,
            "SOURCE_SHAPE",
        )
        members = [SourceMember(**member) for member in value["members"]]
        _require(bool(members) and len(set(members)) == len(members), "SOURCE_MEMBERS")
        seen = set()
        for period, native, member_index, key in value["records"]:
            _require(
                type(member_index) is int and 0 <= member_index < len(members),
                "SOURCE_MEMBER_INDEX",
            )
            origin = (
                AcsHouseholdOrigin(members[member_index], period, native)
                if value["arm"] == "acs"
                else AsecHouseholdOrigin(members[member_index], period, native)
            )
            _require(origin_key(origin) == key, "SOURCE_ORIGIN_KEY")
            identity = (period, native)
            _require(identity not in seen, "SOURCE_NATIVE_KEY_AMBIGUOUS")
            seen.add(identity)
        _require(bool(seen), "SOURCE_EMPTY")
        return value
    except NativeOriginError:
        raise
    except (ValueError, TypeError, KeyError, IndexError, OverflowError):
        raise NativeOriginError("SOURCE_SHAPE") from None


def _issue(arm, members, records, receipt_sha):
    payload = canonical_json(
        {
            "schema": SOURCE_SCHEMA,
            "arm": arm,
            "members": members,
            "records": records,
            "source_receipt_sha256": receipt_sha,
            "release_eligible": False,
        }
    )
    _source_document(payload)
    return AuthenticatedNativeOriginSource(payload, _token=_TOKEN)


def produce_acs_native_origins(source_dir, *, snapshot_root, output_dir):
    """Reuse the real ACS source reader; never accept caller-provided source rows."""
    source = acs.produce_acs_housing_source(
        source_dir,
        snapshot_root=snapshot_root,
        output_dir=output_dir,
    )
    receipt = json.loads(source.receipt_json)
    table = source.households
    members = [
        SourceMember(
            # Original member identity is stable across staging aliases and
            # recompression, and its original member bytes hash is explicit.
            f"census/acs/pums/2024/household/{member['name']}",
            member["sha256"],
        )
        for member in receipt["members"]["household"]
        if member["is_data"]
    ]
    names = [
        member["name"]
        for member in receipt["members"]["household"]
        if member["is_data"]
    ]
    indices = {name: i for i, name in enumerate(names)}
    records = []
    for native, name in zip(table.SERIALNO, table.source_member, strict=True):
        index = indices[name]
        key = origin_key(AcsHouseholdOrigin(members[index], 2024, native))
        records.append([2024, native, index, key])
    return _issue(
        "acs",
        [member.as_payload() for member in members],
        records,
        _sha(source.receipt_json),
    )


def produce_asec_native_origins(member_paths: Mapping[int, Path]):
    """Hash and parse registered original HOUSEHOLD CSV members only.

    No member name or digest is inferred from the person-side restoration.
    Unregistered genuine sources refuse before any path is opened.
    """
    pins = tuple(_ASEC_MEMBER_PINS)
    _require(bool(pins), "ASEC_MEMBER_REGISTRY_UNAVAILABLE")
    _require(
        len({pin.income_year for pin in pins}) == len(pins)
        and set(member_paths) == {pin.income_year for pin in pins},
        "ASEC_MEMBER_ROSTER",
    )
    records, members = [], []
    with tempfile.TemporaryDirectory(prefix="native-asec-origin-") as tmp:
        for index, pin in enumerate(pins):
            # Revalidate the entire declared registry, including archive/cohort.
            AsecNativeMemberPin(**asdict(pin))
            capture = Path(tmp) / f"{index}.csv"
            _require(
                _snapshot(member_paths[pin.income_year], capture, size=pin.size_bytes)
                == pin.member_sha256,
                "ASEC_MEMBER_SHA256",
            )
            member = SourceMember(pin.canonical_member_id, pin.member_sha256)
            members.append(member.as_payload())
            with capture.open("r", encoding="utf-8", newline="") as handle:
                rows = csv.reader(handle, strict=True)
                header = next(rows, [])
                _require(
                    bool(header)
                    and len(header) == len(set(header))
                    and "H_SEQ" in header,
                    "ASEC_MEMBER_HEADER",
                )
                position = header.index("H_SEQ")
                count = 0
                for row in rows:
                    _require(len(row) == len(header), "ASEC_MEMBER_ROW_WIDTH")
                    token = row[position]
                    _require(
                        re.fullmatch(r"[0-9]{1,18}", token) is not None
                        and int(token) > 0,
                        "ASEC_MEMBER_NATIVE_KEY",
                    )
                    native = int(token)
                    records.append(
                        [
                            pin.income_year,
                            native,
                            index,
                            origin_key(
                                AsecHouseholdOrigin(member, pin.income_year, native)
                            ),
                        ]
                    )
                    count += 1
                    _require(count <= pin.rows, "ASEC_MEMBER_ROWS")
                _require(count == pin.rows, "ASEC_MEMBER_ROWS")
            _require(
                _sha(capture.read_bytes()) == pin.member_sha256, "ASEC_CAPTURE_CHANGED"
            )
    return _issue(
        "asec", members, records, _sha(canonical_json(asec_member_registry()))
    )


def _population_content(frame: Frame) -> str:
    # Defined over all cells/axes, effective weights and strata. Graph context
    # deliberately has no Frame.metadata, graph ledger or original design anchors.
    try:
        normalized = Frame(
            {
                entity: frame.table(entity).loc[:, sorted(frame.table(entity))]
                for entity in frame.entities
            },
            frame.schema,
            {entity: frame.resolve_weights(entity) for entity in frame.entities},
            frame.strata,
        )
        return acs.frame_content_sha256(normalized)
    except (ValueError, TypeError, KeyError, OverflowError):
        raise NativeOriginError("POPULATION_CONTENT") from None


@dataclass(frozen=True)
class PopulationOriginBinding:
    """Private population-grain evidence; its summary carries no native keys."""

    payload: bytes
    _token: InitVar[object] = None
    _issued_sha256: str = field(init=False, repr=False)

    def __post_init__(self, _token):
        _require(_token is _BOUND_TOKEN, "POPULATION_BINDING_CONSTRUCTOR")
        object.__setattr__(self, "_issued_sha256", _sha(self.payload))

    @property
    def document(self):
        _require(_sha(self.payload) == self._issued_sha256, "BINDING_MUTATION")
        return json.loads(self.payload)

    @property
    def summary(self):
        return self.document["summary"]


def verify_population_origin_binding(frame: Frame, binding: PopulationOriginBinding):
    """Reject evidence from any other full population content/ordered axes."""
    _require(type(binding) is PopulationOriginBinding, "POPULATION_BINDING_TYPE")
    _require(
        binding.document["population_content_sha256"] == _population_content(frame),
        "POPULATION_CONTENT",
    )


def _entity_rows(frame, household_origins):
    person = frame.person
    household_ids = person.person_household_id.to_numpy()
    rows = {}
    for entity in frame.entities:
        table = frame.table(entity)
        ids = table[US_SCHEMA.entity_id_column(entity)].to_numpy()
        if entity == "person":
            membership = dict(zip(ids.tolist(), household_ids.tolist(), strict=True))
        elif entity == "household":
            membership = dict(zip(ids.tolist(), ids.tolist(), strict=True))
        else:
            pairs = person[
                [US_SCHEMA.membership_column(entity), "person_household_id"]
            ].drop_duplicates()
            _require(
                not pairs.iloc[:, 0].duplicated().any(), "GROUP_CROSSES_HOUSEHOLDS"
            )
            membership = dict(pairs.itertuples(index=False, name=None))
        _require(set(ids) == set(membership), "ENTITY_HOUSEHOLD_COVERAGE")
        records = []
        effective_weights = frame.resolve_weights(entity).values
        for i, identity in enumerate(ids):
            hh = household_origins[_integer(membership[identity])]
            channel = table[support_channel_column(entity)].iloc[i]
            source_id = _integer(table[support_source_id_column(entity)].iloc[i])
            role = _integer(table[support_clone_index_column(entity)].iloc[i])
            _require(
                channel == hh["channel"] and role == hh["clone_index"],
                "ENTITY_HOUSEHOLD_LINEAGE",
            )
            records.append(
                {
                    "entity_id": _integer(identity),
                    "entity_source_id": source_id,
                    "clone_index": role,
                    "channel": channel,
                    "household_source_id": hh["entity_source_id"],
                    "origin_key": hh["origin_key"],
                    "weight_hex": float(effective_weights[i]).hex(),
                }
            )
        _require(
            len({(r["entity_source_id"], r["clone_index"]) for r in records})
            == len(records),
            "ENTITY_LINEAGE_DUPLICATE",
        )
        rows[entity] = records
    groups = {
        entity: {row["entity_id"]: row for row in rows[entity]}
        for entity in US_SCHEMA.group_entities
    }
    for position, row in enumerate(rows["person"]):
        memberships = {}
        for entity, lookup in groups.items():
            member_id = _integer(
                person[US_SCHEMA.membership_column(entity)].iloc[position]
            )
            member = lookup[member_id]
            _require(
                all(
                    member[k] == row[k]
                    for k in (
                        "clone_index",
                        "channel",
                        "household_source_id",
                        "origin_key",
                    )
                ),
                "PERSON_GROUP_LINEAGE",
            )
            memberships[entity] = member["entity_source_id"]
        row["membership_source_ids"] = memberships
    return rows


def _bind_population_origins(
    frame: Frame,
    *,
    sources: Sequence[AuthenticatedNativeOriginSource],
    parent: PopulationOriginBinding | None = None,
):
    """Resolve every household and entity row, including all zero-weight records.

    Native columns identify original records; structural source IDs bind the
    selection/clone ancestry and cannot substitute for that native identity.
    """
    _require(frame.schema == US_SCHEMA and len(sources) == 2, "SOURCE_ARMS")
    documents = []
    for source in sources:
        _require(type(source) is AuthenticatedNativeOriginSource, "SOURCE_TYPE")
        documents.append(source.document)
    _require({d["arm"] for d in documents} == {"acs", "asec"}, "SOURCE_ARMS")
    lookups = {
        d["arm"]: {(p, n): key for p, n, _member, key in d["records"]}
        for d in documents
    }
    household = frame.table("household")
    person = frame.person
    cohort_rows = person[["person_household_id", "source_year"]].drop_duplicates()
    _require(
        not cohort_rows.person_household_id.duplicated().any(),
        "HOUSEHOLD_COHORT_AMBIGUOUS",
    )
    cohorts = dict(cohort_rows.itertuples(index=False, name=None))
    weights = frame.resolve_weights("household").values
    rows = []
    for i, hh in household.iterrows():
        identity = _integer(hh.household_id)
        channel = hh[support_channel_column("household")]
        _require(channel in lookups and identity in cohorts, "NATIVE_KEY_UNRESOLVED")
        raw_period = cohorts[identity]
        _require(
            isinstance(raw_period, (str, int, np.integer))
            and re.fullmatch(r"[0-9]{4}", str(raw_period)) is not None,
            "HOUSEHOLD_COHORT",
        )
        period = int(raw_period)
        native = (
            hh.SERIALNO if channel == "acs" else _carried_native_integer(hh.asec_H_SEQ)
        )
        key = lookups[channel].get((period, native))
        _require(key is not None, "NATIVE_KEY_UNRESOLVED")
        role = _integer(hh[support_clone_index_column("household")])
        _require(role in (0, 1), "CLONE_ROLE")
        position = household.index.get_loc(i)
        rows.append(
            {
                "household_id": identity,
                "entity_source_id": _integer(hh[support_source_id_column("household")]),
                "clone_index": role,
                "channel": channel,
                "origin_key": key,
                "weight_hex": float(weights[position]).hex(),
            }
        )
    _require(len({r["household_id"] for r in rows}) == len(rows), "HOUSEHOLD_IDS")
    entity_rows = _entity_rows(frame, {r["household_id"]: r for r in rows})
    cloned = any(r["clone_index"] for r in rows)
    _require(not cloned or parent is not None, "CLONE_PARENT_REQUIRED")
    pair_conserved = None
    if parent is not None:
        _require(type(parent) is PopulationOriginBinding, "PARENT_TYPE")
        prior = parent.document
        _require(
            prior["source_payload_sha256"] == sorted(_sha(s.payload) for s in sources),
            "PARENT_SOURCE_MISMATCH",
        )
        _require(
            not any(r["clone_index"] for r in prior["households"]),
            "PARENT_MUST_BE_NATIVE",
        )
        for entity, current in entity_rows.items():
            old = {r["entity_source_id"]: r for r in prior["entities"][entity]}
            for row in current:
                before = old.get(row["entity_source_id"])
                _require(
                    before is not None
                    and all(
                        row[k] == before[k]
                        for k in ("channel", "household_source_id", "origin_key")
                    ),
                    "PARENT_MEMBERSHIP_LINEAGE",
                )
                if entity == "person":
                    _require(
                        row["membership_source_ids"] == before["membership_source_ids"],
                        "PARENT_MEMBERSHIP_LINEAGE",
                    )
                before_weight = float.fromhex(before["weight_hex"])
                if cloned:
                    half = before_weight * 0.5
                    _require(
                        row["weight_hex"] == half.hex() and half * 2.0 == before_weight,
                        "CLONE_ENTITY_PAIR_WEIGHT",
                    )
                else:
                    _require(
                        row["weight_hex"] == before["weight_hex"],
                        "PARENT_ENTITY_WEIGHT",
                    )
            if cloned:
                _require(
                    {(r["entity_source_id"], r["clone_index"]) for r in current}
                    == {(source_id, role) for source_id in old for role in (0, 1)},
                    "CLONE_PAIR_COVERAGE",
                )
            else:
                retained_households = {r["entity_source_id"] for r in rows}
                _require(
                    {r["entity_source_id"] for r in current}
                    == {
                        r["entity_source_id"]
                        for r in prior["entities"][entity]
                        if r["household_source_id"] in retained_households
                    },
                    "PARENT_HOUSEHOLD_SELECTION",
                )
        if cloned:
            old = {
                r["entity_source_id"]: float.fromhex(r["weight_hex"])
                for r in prior["households"]
            }
            _require(
                all(
                    row["weight_hex"] == (old[row["entity_source_id"]] * 0.5).hex()
                    and (old[row["entity_source_id"]] * 0.5) * 2.0
                    == old[row["entity_source_id"]]
                    for row in rows
                ),
                "CLONE_PAIR_WEIGHT",
            )
            pair_conserved = True
    content = _population_content(frame)
    summary = {
        "households": len(rows),
        "original_household_origins": len({r["origin_key"] for r in rows}),
        "zero_weight_households_with_lineage": sum(
            float.fromhex(r["weight_hex"]) == 0 for r in rows
        ),
        "entity_rows": {entity: len(values) for entity, values in entity_rows.items()},
        "population_content_sha256": content,
        "parent_pair_mass_conserved": pair_conserved,
        "design_anchors": "not_in_frame_context_not_evaluated",
        "release_eligible": False,
    }
    payload = canonical_json(
        {
            "schema": BINDING_SCHEMA,
            "population_content_sha256": content,
            "ordered_entity_ids_sha256": {
                e: _sha(canonical_json([r["entity_id"] for r in records]))
                for e, records in entity_rows.items()
            },
            "source_payload_sha256": sorted(_sha(s.payload) for s in sources),
            "parent_binding_sha256": None if parent is None else _sha(parent.payload),
            "households": rows,
            "entities": entity_rows,
            "summary": summary,
        }
    )
    return PopulationOriginBinding(payload, _token=_BOUND_TOKEN)


def bind_population_origins(
    frame: Frame,
    *,
    sources: Sequence[AuthenticatedNativeOriginSource],
    parent: PopulationOriginBinding | None = None,
):
    """Bind every current row; malformed inputs refuse without leaking native IDs."""
    try:
        return _bind_population_origins(frame, sources=sources, parent=parent)
    except NativeOriginError:
        raise
    except (ValueError, TypeError, KeyError, IndexError, OverflowError):
        raise NativeOriginError("POPULATION_BINDING_CONTRACT") from None
