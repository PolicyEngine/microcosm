"""Pinned Census source bytes to normalized atomic-block support, without I/O.

The caller establishes publisher identity and obtains the bytes. This adapter
checks those bytes against explicit pins, checks full ZIP member rosters, and
decompresses only the selected geography members. Unselected PL segment contents
are neither opened nor CRC-checked. No source or release admission is issued.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from io import BytesIO
from pathlib import PurePosixPath
from zipfile import ZIP_DEFLATED, ZIP_STORED, BadZipFile, ZipFile
from zlib import error as zlib_error

from microcosm.build.atomic_geography import decode_atomic_support
from microcosm.graph.canonical import canonical_json
from microcosm.graph.codecs import RAW_BYTES_MAX_BYTES

from . import atomic_block_support as normalized
from .block_ladder_sources import (
    US_STATES,
    parse_national_cd_bef,
    parse_pl_geo_blocks,
)
from .puma_ladder_sources import parse_tract_to_puma_relationship

PROTOCOL = "microcosm.us.atomic-block-sources.v1"
CD_MEMBER = "NationalCD119.txt"
_GEO_MEMBERS = {state: f"{usps.lower()}geo2020.pl" for state, usps, _ in US_STATES}
_CHUNK_BYTES = 64 * 1024


@dataclass(frozen=True, slots=True)
class AtomicBlockSourceBytes:
    """Literal bytes and expected digest/roster; construction conveys no authority."""

    payload: bytes
    sha256: str
    zip_members: tuple[str, ...] = ()


def _require(condition, reason):
    if not condition:
        raise ValueError("ATOMIC_BLOCK_SOURCES_" + reason)


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _checked_source(record, *, archive, max_source_bytes):
    _require(type(record) is AtomicBlockSourceBytes, "SOURCE_TYPE")
    payload, digest, members = record.payload, record.sha256, record.zip_members
    _require(
        type(payload) is bytes and 0 < len(payload) <= max_source_bytes,
        "SOURCE_BYTES",
    )
    _require(
        type(digest) is str
        and re.fullmatch(r"[0-9a-f]{64}", digest) is not None
        and _sha(payload) == digest,
        "SOURCE_DIGEST",
    )
    _require(type(members) is tuple, "MEMBER_ROSTER")
    if archive:
        _require(
            bool(members)
            and all(
                type(name) is str
                and 0 < len(name) <= 255
                and "\0" not in name
                and "\\" not in name
                and not name.endswith("/")
                and not PurePosixPath(name).is_absolute()
                and all(part not in {"", ".", ".."} for part in name.split("/"))
                for name in members
            )
            and len(set(members)) == len(members),
            "MEMBER_ROSTER",
        )
    else:
        _require(members == (), "RAW_MEMBER_ROSTER")
    return payload, digest, members


def _selected_zip_member(record, member, *, bounds, expanded):
    source_payload, source_digest, expected_members = _checked_source(
        record, archive=True, max_source_bytes=bounds["max_source_bytes"]
    )
    provenance = {"sha256": source_digest, "size_bytes": len(source_payload)}
    _require(
        member in expected_members
        and len(expected_members) <= bounds["max_zip_members"],
        "SELECTED_MEMBER",
    )
    try:
        with ZipFile(BytesIO(source_payload)) as archive:
            members = archive.infolist()
            names = tuple(info.filename for info in members)
            _require(
                len(names) == len(expected_members)
                and len(set(names)) == len(names)
                and all(info.orig_filename == info.filename for info in members)
                and set(names) == set(expected_members),
                "ZIP_MEMBER_ROSTER",
            )
            info = archive.getinfo(member)
            _require(
                not info.is_dir()
                and not info.flag_bits & 1
                and info.compress_type in {ZIP_STORED, ZIP_DEFLATED},
                "ZIP_MEMBER_FORMAT",
            )
            _require(
                0 < info.file_size <= bounds["max_member_bytes"], "ZIP_MEMBER_SIZE"
            )
            _require(
                expanded[0] + info.file_size <= bounds["max_total_geography_bytes"],
                "TOTAL_GEOGRAPHY_BYTES",
            )
            # ZipExtFile validates the selected member's CRC when exhausted.
            # No other member is opened, even to check its CRC.
            output = BytesIO()
            with archive.open(info) as stream:
                while chunk := stream.read(_CHUNK_BYTES):
                    _require(
                        output.tell() + len(chunk) <= info.file_size,
                        "ZIP_MEMBER_SIZE",
                    )
                    output.write(chunk)
            payload = output.getvalue()
            _require(len(payload) == info.file_size, "ZIP_MEMBER_SIZE")
            expanded[0] += len(payload)
            provenance.update(
                {
                    "zip_members": sorted(names),
                    "selected_member": member,
                    "selected_member_sha256": _sha(payload),
                    "selected_member_size_bytes": len(payload),
                    "crc_checked_members": [member],
                    "unselected_member_contents_read": False,
                }
            )
    except (BadZipFile, EOFError, RuntimeError, NotImplementedError, zlib_error):
        raise ValueError("ATOMIC_BLOCK_SOURCES_ZIP_INVALID") from None
    return payload, provenance


def _lines(payload, *, encoding, max_line_bytes):
    stream = BytesIO(payload)
    while line := stream.readline(max_line_bytes + 1):
        _require(len(line) <= max_line_bytes and b"\0" not in line, "TEXT_LINE")
        yield line.decode(encoding)


def _pl_lines(payload, *, state, max_line_bytes, statistics):
    """Guard parser skips: count every block, including zero-population rows."""
    seen = set()
    for line in _lines(payload, encoding="latin-1", max_line_bytes=max_line_bytes):
        if not line.strip():
            continue
        fields = line.rstrip("\r\n").split("|")
        _require(len(fields) == 97, "PL_ROW_WIDTH")
        statistics["geography_rows"] += 1
        level = fields[2]
        if level in {"040", "750"}:
            raw = fields[90].strip()
            _require(re.fullmatch(r"[0-9]+", raw) is not None, "PL_POPULATION")
            population = int(raw)
            if level == "040":
                statistics["state_rows"] += 1
                _require(statistics["state_rows"] == 1, "PL_STATE_ROWS")
                statistics["state_population"] = population
            else:
                geoid = fields[9].strip()
                _require(
                    re.fullmatch(r"[0-9]{15}", geoid) is not None
                    and geoid.startswith(state),
                    "PL_BLOCK_STATE",
                )
                _require(geoid not in seen, "PL_DUPLICATE_BLOCK")
                seen.add(geoid)
                statistics["block_rows"] += 1
                statistics["zero_population_blocks"] += int(population == 0)
        yield line


def _cd_lines(payload, *, max_line_bytes, statistics, unassigned):
    first = True
    for line in _lines(payload, encoding="latin-1", max_line_bytes=max_line_bytes):
        if not line.strip():
            continue
        if first:
            first = False
        else:
            parts = line.strip().split(",")
            _require(len(parts) == 2, "CD_ROW_WIDTH")
            statistics["source_records"] += 1
            geoid, district = (part.strip() for part in parts)
            if district == "ZZ":
                _require(re.fullmatch(r"[0-9]{15}", geoid) is not None, "CD_BLOCK")
                block = int(geoid)
                _require(block not in unassigned, "CD_DUPLICATE_BLOCK")
                unassigned.add(block)
            statistics["delegate_records"] += int(district == "98")
        yield line


def assemble_atomic_block_sources(
    *,
    pl_archives: Iterable[tuple[str, AtomicBlockSourceBytes]],
    cd_archive: AtomicBlockSourceBytes,
    tract_to_puma: AtomicBlockSourceBytes,
    state_fips: tuple[str, ...],
    source_ids: Mapping[str, str],
    max_source_bytes: int = 512 * 1024**2,
    max_member_bytes: int = 2 * 1024**3,
    max_total_geography_bytes: int = 32 * 1024**3,
    max_zip_members: int = 16,
    max_line_bytes: int = 64 * 1024,
) -> tuple[bytes, bytes]:
    """Return deterministic normalized support and a canonical source receipt.

    ``state_fips`` is a nonempty sorted tuple drawn from the 50 states plus DC.
    ``pl_archives`` yields exactly that ordered roster, one archive at a time;
    only the current PL archive/member is retained while accumulating block maps.
    Required members are ``{usps_lower}geo2020.pl`` and ``NationalCD119.txt``.
    The tract relationship is raw UTF-8 text. ZIP target text uses the maintained
    source readers' Latin-1 convention. All positive-POP100 blocks are retained;
    zero-population exclusions are counted in each state's receipt.

    The source byte cap and hash check precede ZIP metadata parsing. Member
    roster/count and expanded-byte limits precede selected-member inflation and
    text parsing, with a cumulative bound on selected geography plus the raw
    relationship. The member-count limit does not bound ZipFile's initial
    metadata allocation. No unselected segment is decompressed or parsed.
    Normalized output must fit the receiving raw-bytes-v1 codec's 64 MiB cap.

    Matching caller-supplied hashes proves byte integrity, not that the caller's
    pins identify an official publisher, a current source, or an admitted dataset.
    """
    _require(
        type(state_fips) is tuple
        and bool(state_fips)
        and all(type(state) is str and state in _GEO_MEMBERS for state in state_fips)
        and state_fips == tuple(sorted(set(state_fips))),
        "STATE_ROSTER",
    )
    identities = normalized._sources(source_ids)
    bounds = {
        "max_source_bytes": max_source_bytes,
        "max_member_bytes": max_member_bytes,
        "max_total_geography_bytes": max_total_geography_bytes,
        "max_zip_members": max_zip_members,
        "max_line_bytes": max_line_bytes,
    }
    _require(
        all(type(value) is int and 0 < value <= 2**63 - 1 for value in bounds.values()),
        "BOUNDS",
    )
    puma_payload, puma_digest, _members = _checked_source(
        tract_to_puma, archive=False, max_source_bytes=max_source_bytes
    )
    puma_provenance = {"sha256": puma_digest, "size_bytes": len(puma_payload)}
    _require(
        len(puma_payload) <= min(max_member_bytes, max_total_geography_bytes),
        "TOTAL_GEOGRAPHY_BYTES",
    )
    expanded = [len(puma_payload)]
    cd_payload, cd_provenance = _selected_zip_member(
        cd_archive, CD_MEMBER, bounds=bounds, expanded=expanded
    )
    cd_statistics = {"source_records": 0, "delegate_records": 0}
    unassigned = set()
    districts = parse_national_cd_bef(
        _cd_lines(
            cd_payload,
            max_line_bytes=max_line_bytes,
            statistics=cd_statistics,
            unassigned=unassigned,
        )
    )
    _require(not unassigned.intersection(districts), "CD_CONFLICTING_UNASSIGNED")
    _require(
        cd_statistics["source_records"] == len(districts) + len(unassigned),
        "CD_RECONCILIATION",
    )
    cd_provenance.update(
        source_id=identities["district"],
        **cd_statistics,
        assigned_blocks=len(districts),
        unassigned_blocks=len(unassigned),
        delegate_normalization="98_to_00",
        district_relation="official_tabulation",
    )
    del cd_payload, unassigned
    pumas = parse_tract_to_puma_relationship(
        _lines(puma_payload, encoding="utf-8", max_line_bytes=max_line_bytes),
        allowed_state_fips=frozenset(state_fips),
    )
    puma_provenance.update(
        source_id=identities["puma"], selected_state_tract_mappings=len(pumas)
    )

    iterator, exhausted = iter(pl_archives), object()
    population, population_provenance, state_totals = {}, [], {}
    for state in state_fips:
        item = next(iterator, exhausted)
        _require(
            type(item) is tuple and len(item) == 2 and item[0] == state,
            "PL_SOURCE_ROSTER",
        )
        record = item[1]
        payload, provenance = _selected_zip_member(
            record, _GEO_MEMBERS[state], bounds=bounds, expanded=expanded
        )
        statistics = {
            "geography_rows": 0,
            "state_rows": 0,
            "state_population": 0,
            "block_rows": 0,
            "zero_population_blocks": 0,
        }
        state_population = parse_pl_geo_blocks(
            _pl_lines(
                payload,
                state=state,
                max_line_bytes=max_line_bytes,
                statistics=statistics,
            ),
            state_fips=state,
        )
        _require(
            statistics["state_rows"] == 1
            and statistics["state_population"] == sum(state_population.values())
            and statistics["block_rows"]
            == len(state_population) + statistics["zero_population_blocks"],
            "PL_RECONCILIATION",
        )
        _require(
            not population.keys() & state_population.keys(), "PL_OVERLAPPING_STATES"
        )
        population.update(state_population)
        state_totals[state] = statistics["state_population"]
        provenance.update(
            source_id=identities["population"],
            state_fips=state,
            populated_blocks=len(state_population),
            **statistics,
        )
        population_provenance.append(provenance)
        del item, record, payload, state_population
    _require(next(iterator, exhausted) is exhausted, "PL_SOURCE_ROSTER")
    payload = normalized.assemble_atomic_block_support(
        block_population=population,
        cd_by_block=districts,
        puma_by_tract=pumas,
        source_ids=identities,
    )
    _require(len(payload) <= RAW_BYTES_MAX_BYTES, "SUPPORT_BYTES")
    support = decode_atomic_support(payload)
    output_totals = dict.fromkeys(state_fips, 0)
    for state, value in zip(
        support.arrays["state"], support.arrays["population"], strict=True
    ):
        output_totals[str(state)] += int(value)
    ordered_blocks = sorted(population)
    _require(
        support.arrays["area"].tolist() == [f"{block:015d}" for block in ordered_blocks]
        and support.arrays["population"].tolist()
        == [population[block] for block in ordered_blocks]
        and support.arrays["puma"].tolist()
        == [f"{pumas[block // 10000]:07d}" for block in ordered_blocks]
        and support.arrays["district"].tolist()
        == [f"{districts[block]:04d}" for block in ordered_blocks]
        and output_totals == state_totals
        and len(support.arrays["area"]) == len(population),
        "OUTPUT_RECONCILIATION",
    )
    receipt = canonical_json(
        {
            "protocol": PROTOCOL,
            "state_fips": list(state_fips),
            "source_ids": identities,
            "limits": bounds,
            "sources": {
                "population": population_provenance,
                "district": cd_provenance,
                "puma": puma_provenance,
            },
            "reconciliation": {
                "state_population": state_totals,
                "output_state_population": output_totals,
                "population_total": sum(state_totals.values()),
                "populated_blocks": len(population),
                "selected_geography_bytes": expanded[0],
            },
            "support_sha256": _sha(payload),
            "exact_input_hashes_verified": True,
            "unselected_zip_member_contents_read": False,
            "publisher_provenance_established": False,
            "source_admission_issued": False,
            "release_eligible": False,
        }
    )
    return payload, receipt
