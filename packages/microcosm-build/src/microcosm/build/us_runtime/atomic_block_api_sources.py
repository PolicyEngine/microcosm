"""Pinned population-only Census API responses to atomic support, without I/O.

The caller obtains original ``2020/dec/pl`` responses requesting only
``P1_001N``. No legacy PL archive or housing table is accepted. Byte pins and
request descriptions confer no publisher, source, or release authority.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping

from microcosm.build.atomic_geography import decode_atomic_support
from microcosm.graph.canonical import canonical_json
from microcosm.graph.codecs import RAW_BYTES_MAX_BYTES

from . import atomic_block_sources as sources
from . import atomic_block_support as normalized
from .block_ladder_sources import parse_national_cd_bef
from .puma_ladder_sources import parse_tract_to_puma_relationship

PROTOCOL = "microcosm.us.atomic-block-api-sources.v1"
ENDPOINT = "https://api.census.gov/data/2020/dec/pl"
POPULATION_VARIABLE = "P1_001N"
_BLOCK_HEADER = (POPULATION_VARIABLE, "state", "county", "tract", "block")
_STATE_HEADER = (POPULATION_VARIABLE, "state")


def _require(condition, reason):
    if not condition:
        raise ValueError("ATOMIC_BLOCK_API_SOURCES_" + reason)


def _request(state, *, blocks):
    parameters = [["get", POPULATION_VARIABLE]]
    if blocks:
        parameters.extend(
            [
                ["for", "block:*"],
                ["in", "state:" + state],
                ["in", "county:*"],
                ["in", "tract:*"],
            ]
        )
    else:
        parameters.append(["for", "state:" + state])
    return {"endpoint": ENDPOINT, "parameters": parameters}


def _rows(payload, *, header, max_rows):
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        raise ValueError("ATOMIC_BLOCK_API_SOURCES_RESPONSE_JSON") from None
    _require(
        type(document) is list
        and 1 < len(document) <= max_rows + 1
        and document[0] == list(header),
        "RESPONSE_HEADER_OR_COUNT",
    )
    for row in document[1:]:
        _require(
            type(row) is list
            and len(row) == len(header)
            and all(type(value) is str for value in row),
            "RESPONSE_ROW",
        )
        yield row


def _population(value):
    _require(
        len(value) <= 16
        and re.fullmatch(r"[0-9]+", value) is not None
        and int(value) <= 2**53,
        "POPULATION",
    )
    return int(value)


def _state_population(block_payload, total_payload, *, state, max_rows):
    total_rows = list(_rows(total_payload, header=_STATE_HEADER, max_rows=1))
    total_raw, total_state = total_rows[0]
    _require(total_state == state, "STATE_TOTAL_IDENTITY")
    total = _population(total_raw)
    blocks, seen, zero_count = {}, set(), 0
    for row in _rows(block_payload, header=_BLOCK_HEADER, max_rows=max_rows):
        raw_population, raw_state, county, tract, block = row
        _require(
            raw_state == state
            and all(
                re.fullmatch(r"[0-9]{" + str(width) + r"}", value) is not None
                for value, width in ((county, 3), (tract, 6), (block, 4))
            ),
            "BLOCK_IDENTITY",
        )
        geoid = int(raw_state + county + tract + block)
        _require(geoid not in seen, "DUPLICATE_BLOCK")
        seen.add(geoid)
        population = _population(raw_population)
        if population:
            blocks[geoid] = population
        else:
            zero_count += 1
    _require(
        bool(blocks)
        and sum(blocks.values()) == total
        and len(seen) == len(blocks) + zero_count,
        "STATE_RECONCILIATION",
    )
    return blocks, {
        "state_population": total,
        "block_rows": len(seen),
        "zero_population_blocks": zero_count,
        "populated_blocks": len(blocks),
    }


def _raw_source(record, *, bounds, expanded):
    payload, digest, _members = sources._checked_source(
        record, archive=False, max_source_bytes=bounds["max_source_bytes"]
    )
    _require(
        len(payload) <= bounds["max_member_bytes"]
        and expanded[0] + len(payload) <= bounds["max_total_geography_bytes"],
        "TOTAL_GEOGRAPHY_BYTES",
    )
    expanded[0] += len(payload)
    return payload, {"sha256": digest, "size_bytes": len(payload)}


def assemble_atomic_block_api_sources(
    *,
    population_responses: Iterable[
        tuple[str, sources.AtomicBlockSourceBytes, sources.AtomicBlockSourceBytes]
    ],
    cd_archive: sources.AtomicBlockSourceBytes,
    tract_to_puma: sources.AtomicBlockSourceBytes,
    state_fips: tuple[str, ...],
    source_ids: Mapping[str, str],
    max_source_bytes: int = 512 * 1024**2,
    max_member_bytes: int = 2 * 1024**3,
    max_total_geography_bytes: int = 32 * 1024**3,
    max_zip_members: int = 16,
    max_line_bytes: int = 64 * 1024,
    max_response_rows: int = 2_000_000,
) -> tuple[bytes, bytes]:
    """Normalize exact original API responses and mapping bytes, with a receipt.

    Each ordered item is ``(state, block_response, single_state_total_response)``.
    There is exactly one item per selected state, drawn from the 50 states plus
    DC. Expected API requests appear in the descriptive receipt with repeated
    ``in`` parameters preserved and no credential parameter. The caller owns
    acquisition and publisher qualification; the response cannot prove its URL.

    Response byte limits precede JSON parsing; the row limit is enforced after
    JSON allocation. CD ZIP metadata and inflation follow the existing bounded
    source reader. All rows, including zero-population blocks, are counted;
    every positive block is retained exactly once and reconciled to its separate
    state response. No county/tract aggregation or synthetic PL file is used.
    """
    _require(
        type(state_fips) is tuple
        and bool(state_fips)
        and all(
            type(state) is str and state in sources._GEO_MEMBERS for state in state_fips
        )
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
        "max_response_rows": max_response_rows,
    }
    _require(
        all(type(value) is int and 0 < value <= 2**63 - 1 for value in bounds.values()),
        "BOUNDS",
    )
    expanded = [0]
    puma_payload, puma_provenance = _raw_source(
        tract_to_puma, bounds=bounds, expanded=expanded
    )
    cd_payload, cd_provenance = sources._selected_zip_member(
        cd_archive, sources.CD_MEMBER, bounds=bounds, expanded=expanded
    )
    cd_statistics, unassigned = {"source_records": 0, "delegate_records": 0}, set()
    districts = parse_national_cd_bef(
        sources._cd_lines(
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
        sources._lines(puma_payload, encoding="utf-8", max_line_bytes=max_line_bytes),
        allowed_state_fips=frozenset(state_fips),
    )
    puma_provenance.update(
        source_id=identities["puma"], selected_state_tract_mappings=len(pumas)
    )
    del puma_payload

    iterator, exhausted = iter(population_responses), object()
    population, population_provenance, state_totals = {}, [], {}
    for state in state_fips:
        item = next(iterator, exhausted)
        _require(
            type(item) is tuple and len(item) == 3 and item[0] == state,
            "POPULATION_SOURCE_ROSTER",
        )
        block_payload, block_provenance = _raw_source(
            item[1], bounds=bounds, expanded=expanded
        )
        total_payload, total_provenance = _raw_source(
            item[2], bounds=bounds, expanded=expanded
        )
        blocks, statistics = _state_population(
            block_payload, total_payload, state=state, max_rows=max_response_rows
        )
        _require(not population.keys() & blocks.keys(), "OVERLAPPING_STATES")
        population.update(blocks)
        state_totals[state] = statistics["state_population"]
        population_provenance.append(
            {
                "source_id": identities["population"],
                "state_fips": state,
                "blocks": {**block_provenance, "request": _request(state, blocks=True)},
                "state_total": {
                    **total_provenance,
                    "request": _request(state, blocks=False),
                },
                **statistics,
            }
        )
        del item, block_payload, total_payload, blocks
    _require(next(iterator, exhausted) is exhausted, "POPULATION_SOURCE_ROSTER")
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
            "support_sha256": sources._sha(payload),
            "exact_input_hashes_verified": True,
            "publisher_provenance_established": False,
            "request_origin_verified": False,
            "source_admission_issued": False,
            "release_eligible": False,
        }
    )
    return payload, receipt
