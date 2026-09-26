"""Opt-in ACS SPM construction before source amounts, assembly or cloning.

This pure seam changes only the SPM partition of an ID-only source Frame. Its
registry binds one complete selected native roster, not a national ID namespace.
No country input role, annual universe declaration or source-file authority is
issued here. Source owners must authenticate their own files and implementation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from copy import copy
from dataclasses import asdict, dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build import acs_spm_partition as partition
from microcosm.frame import US_SCHEMA, Frame, WeightKind

RECEIPT_KEY = "acs_spm_source_assembly"
PROTOCOL = "microcosm.acs-spm-source-assembly.v1"


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError("ACS_SPM_SOURCE_" + reason)


def _integer(value: Any, *, maximum: int = np.iinfo(np.int64).max) -> int:
    _require(
        isinstance(value, Integral)
        and not isinstance(value, (bool, np.bool_))
        and 0 <= value <= maximum,
        "EXACT_INTEGER",
    )
    return int(value)


def _json_value(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Mapping):
        _require(all(type(key) is str for key in value), "METADATA_KEYS")
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if type(value) is float and np.isnan(value):
        return None
    _require(type(value) in (str, int, float, bool), "PRIMITIVE_VALUE")
    _require(type(value) is not float or np.isfinite(value), "FINITE_VALUE")
    return value


def _bytes(value: Any) -> bytes:
    return json.dumps(
        _json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_bytes(value)).hexdigest()


def _table_digest(table: pd.DataFrame) -> str:
    digest = hashlib.sha256(
        _bytes(
            {
                "columns": list(table.columns),
                "dtypes": [str(dtype) for dtype in table.dtypes],
                "index_type": type(table.index).__qualname__,
                "index_dtype": str(table.index.dtype),
                "index_name": table.index.name,
                "column_name": table.columns.name,
            }
        )
    )
    for index, row in zip(
        table.index, table.itertuples(index=False, name=None), strict=True
    ):
        digest.update(_bytes([index, row]))
        digest.update(b"\n")
    return digest.hexdigest()


@dataclass(frozen=True)
class AcsSpmSourceAssemblyOptions:
    """An explicit development construction, never a default source rule."""

    policy: str
    minor_partner_role: bool

    def __post_init__(self) -> None:
        _require(self.policy == partition.ACS_SPM_DEVELOPMENT_POLICY, "POLICY")
        _require(type(self.minor_partner_role) is bool, "PARTNER_ROLE_BOOL")


@dataclass(frozen=True)
class AcsSpmSourceRegistryEntry:
    """One native person in a whole-selected-roster identity map."""

    serialno: str
    sporder: int
    person_id: int
    household_id: int
    old_spm_unit_id: int
    new_spm_unit_id: int
    component_sha256: str
    source_record_sha256: str


@dataclass(frozen=True)
class AcsSpmSourceRegistry:
    """Immutable supplied-table identity evidence, not authenticated file reads.

    Allocate this once over the whole selected roster, then pass the same value
    to complete-household chunks. IDs have no stability claim across different
    selection domains. The native constructor supplies its reviewed clone-safe
    ceiling; pure callers must declare their own exact int64 ceiling.
    """

    options: AcsSpmSourceAssemblyOptions
    id_ceiling: int
    assembler_sha256: str
    entries: tuple[AcsSpmSourceRegistryEntry, ...]

    @property
    def sha256(self) -> str:
        return _digest(asdict(self))

    @property
    def selected_roster_sha256(self) -> str:
        return _digest(
            [
                (row.serialno, row.sporder, row.source_record_sha256)
                for row in self.entries
            ]
        )


@dataclass(frozen=True)
class AcsSpmSourceAssemblyResult:
    frame: Frame
    partition: partition.AcsSpmPartitionResult
    registry: AcsSpmSourceRegistry
    crosswalk: pd.DataFrame
    unit_evidence: pd.DataFrame
    receipt: Mapping[str, Any]


def require_acs_spm_source_capability(options: AcsSpmSourceAssemblyOptions) -> None:
    """Refuse an explicit constructor request before opening source archives."""
    _require(type(options) is AcsSpmSourceAssemblyOptions, "OPTIONS_TYPE")
    options.__post_init__()
    probe = partition.probe_acs_spm_assembler()
    if not probe.supported:
        raise partition.UnsupportedAssembler(probe)


def _source(frame: Frame) -> tuple[pd.DataFrame, dict[int, int], dict[int, str]]:
    _require(type(frame) is Frame and frame.schema == US_SCHEMA, "FRAME_SCHEMA")
    _require(not frame.links, "SOURCE_LINK_TABLES")
    copy(frame).revalidate()
    _require(list(frame.table("spm_unit")) == ["spm_unit_id"], "ID_ONLY_SPM_TABLE")
    _require(
        frame.weighted_entities == ("household",)
        and frame.weights_for("household").kind is WeightKind.DESIGN,
        "HOUSEHOLD_DESIGN_WEIGHTS",
    )
    _require(RECEIPT_KEY not in frame.metadata, "ALREADY_CONSTRUCTED")
    people, households = frame.person, frame.table("household")
    _require(
        {"SERIALNO", "NP", "TYPEHUGQ"}.issubset(households)
        and {"SPORDER", "RELSHIPP", "AGEP", "MAR"}.issubset(people),
        "RAW_ROSTER_COLUMNS",
    )
    _require("is_spm_independent_minor_role" not in people, "PREEXISTING_ENGINE_ROLE")
    _require(
        households.SERIALNO.map(lambda value: type(value) is str and bool(value)).all()
        and households.SERIALNO.is_unique,
        "HOUSEHOLD_NATIVE_KEYS",
    )
    for entity in frame.entities:
        _require(frame.table(entity).columns.is_unique, "COLUMN_ROSTER")
        for value in frame.table(entity)[frame.schema.entity_id_column(entity)]:
            _integer(value)
    for group in frame.schema.group_entities:
        for value in people[frame.schema.membership_column(group)]:
            _integer(value)
    kinds = {
        _integer(hid): _integer(kind, maximum=3)
        for hid, kind in zip(households.household_id, households.TYPEHUGQ, strict=True)
    }
    _require(all(kind in (1, 2, 3) for kind in kinds.values()), "HOUSEHOLD_UNIVERSE")
    counts = {
        _integer(hid): _integer(count)
        for hid, count in zip(households.household_id, households.NP, strict=True)
    }
    _require(all(count > 0 for count in counts.values()), "HOUSEHOLD_COUNT")
    _require(all(counts[hid] == 1 for hid in kinds if kinds[hid] != 1), "GQ_ROSTER")
    work = people.copy(deep=True)
    # PUMS deliberately preserves SPORDER as a source token. Normalize only the
    # temporary helper view; never rewrite that token in the returned Frame.
    orders = []
    for value in work.SPORDER:
        if type(value) is str:
            _require(value.isascii() and value.isdecimal(), "SPORDER_TOKEN")
            value = int(value)
        order = _integer(value)
        _require(order > 0, "SPORDER_POSITIVE")
        orders.append(order)
    work["SPORDER"] = np.asarray(orders, dtype=np.int64)
    mapped = work.person_household_id.map(kinds)
    if "TYPEHUGQ" in work:
        _require(work.TYPEHUGQ.eq(mapped).fillna(False).all(), "UNIVERSE_DISAGREEMENT")
    work["TYPEHUGQ"] = mapped
    if "PERIDNUM" in work:
        _require(
            not work.PERIDNUM.dropna().astype(str).str.strip().ne("").any(),
            "NON_ACS_PERSON",
        )
    serials = dict(zip(households.household_id, households.SERIALNO, strict=True))
    return work, counts, serials


def _proposal(frame: Frame, options: AcsSpmSourceAssemblyOptions):
    _require(type(options) is AcsSpmSourceAssemblyOptions, "OPTIONS_TYPE")
    options.__post_init__()
    work, counts, serials = _source(frame)
    result = partition.reconstruct_acs_spm_partition(
        work,
        frame.table("spm_unit"),
        household_person_counts=counts,
        policy=options.policy,
        minor_partner_role=options.minor_partner_role,
    )
    result.require_resolved()
    # Even an all-GQ request must prove the explicitly requested capability.
    probe = partition.probe_acs_spm_assembler()
    if not probe.supported:
        raise partition.UnsupportedAssembler(probe)
    _require(probe.module_sha256 is not None, "ASSEMBLER_IDENTITY")
    native_keys = {
        int(row.person_id): (serials[row.person_household_id], int(row.SPORDER))
        for row in work.itertuples()
    }
    _require(len(set(native_keys.values())) == len(work), "DUPLICATE_NATIVE_PERSON")
    records = {}
    for row in work.itertuples():
        records[int(row.person_id)] = _digest(
            {
                "native_person": native_keys[row.person_id],
                "person_id": row.person_id,
                "household_id": row.person_household_id,
                "old_spm_unit_id": row.person_spm_unit_id,
                "RELSHIPP": row.RELSHIPP,
                "AGEP": row.AGEP,
                "MAR": row.MAR,
                "TYPEHUGQ": row.TYPEHUGQ,
                "NP": counts[row.person_household_id],
            }
        )
    entries = []
    for _, unit in result.membership.groupby("proposed_spm_unit_id", sort=True):
        _require(unit.old_spm_unit_id.nunique() == 1, "OLD_UNIT_MERGE")
        component = _digest(sorted(native_keys[int(pid)] for pid in unit.person_id))
        for row in unit.itertuples():
            serial, order = native_keys[row.person_id]
            entries.append(
                AcsSpmSourceRegistryEntry(
                    serial,
                    order,
                    int(row.person_id),
                    int(row.person_household_id),
                    int(row.old_spm_unit_id),
                    0,
                    component,
                    records[row.person_id],
                )
            )
    return (
        result,
        tuple(sorted(entries, key=lambda row: (row.serialno, row.sporder))),
        probe,
    )


def _allocated(entries: tuple[AcsSpmSourceRegistryEntry, ...], ceiling: int):
    """Preserve unchanged units; assign all split components once in native order."""
    old, components = {}, {}
    for row in entries:
        _integer(row.old_spm_unit_id, maximum=ceiling)
        key = (row.serialno, row.sporder)
        old.setdefault(row.old_spm_unit_id, set()).add(key)
        components.setdefault(row.component_sha256, []).append(row)
    next_id = max(old) + 1
    result = {}
    for rows in sorted(
        components.values(), key=lambda rows: [(r.serialno, r.sporder) for r in rows]
    ):
        old_ids = {row.old_spm_unit_id for row in rows}
        _require(len(old_ids) == 1, "OLD_UNIT_MERGE")
        old_id = next(iter(old_ids))
        if {(row.serialno, row.sporder) for row in rows} == old[old_id]:
            new_id = old_id
        else:
            _require(next_id <= ceiling, "REGISTRY_ID_OVERFLOW")
            new_id, next_id = next_id, next_id + 1
        result[rows[0].component_sha256] = new_id
    _require(len(set(result.values())) == len(result), "REGISTRY_ID_COLLISION")
    return result


def _validate_registry(registry: AcsSpmSourceRegistry) -> None:
    _require(type(registry) is AcsSpmSourceRegistry, "REGISTRY_TYPE")
    _integer(registry.id_ceiling)
    _require(type(registry.options) is AcsSpmSourceAssemblyOptions, "OPTIONS_TYPE")
    registry.options.__post_init__()
    _require(
        type(registry.entries) is tuple and bool(registry.entries), "REGISTRY_ROWS"
    )
    _require(
        all(type(row) is AcsSpmSourceRegistryEntry for row in registry.entries),
        "REGISTRY_ROW_TYPE",
    )
    for digest in (
        registry.assembler_sha256,
        *(row.source_record_sha256 for row in registry.entries),
    ):
        _require(
            type(digest) is str
            and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest),
            "REGISTRY_DIGEST",
        )
    keys = [(row.serialno, row.sporder) for row in registry.entries]
    _require(keys == sorted(set(keys)), "REGISTRY_NATIVE_KEYS")
    _require(
        len({row.person_id for row in registry.entries}) == len(keys),
        "REGISTRY_PERSON_IDS",
    )
    for row in registry.entries:
        _require(type(row.serialno) is str and bool(row.serialno), "REGISTRY_SERIAL")
        for value in (row.person_id, row.household_id, row.sporder):
            _integer(value)
        _require(row.sporder > 0, "REGISTRY_SPORDER")
        _integer(row.new_spm_unit_id, maximum=registry.id_ceiling)
    components = {}
    for row in registry.entries:
        components.setdefault(row.component_sha256, []).append(row)
    for component, members in components.items():
        _require(
            component == _digest(sorted((row.serialno, row.sporder) for row in members))
            and len({row.household_id for row in members}) == 1
            and len({row.serialno for row in members}) == 1,
            "REGISTRY_COMPONENT_IDENTITY",
        )
    expected = _allocated(registry.entries, registry.id_ceiling)
    _require(
        all(
            row.new_spm_unit_id == expected[row.component_sha256]
            for row in registry.entries
        ),
        "REGISTRY_ASSIGNMENT",
    )


def _unit_evidence(result: partition.AcsSpmPartitionResult) -> pd.DataFrame:
    uncertain = set(
        result.membership.loc[
            result.membership.secondary_link_status.isin(("unassessed", "ambiguous")),
            "person_household_id",
        ]
    )
    records = []
    for label, unit in result.membership.groupby("proposed_spm_unit_id", sort=True):
        records.append(
            {
                "proposed_spm_unit_id": label,
                "role_sources": tuple(sorted(set(unit.role_source))),
                "partition_assumptions": tuple(sorted(set(unit.partition_assumption))),
                "secondary_link_statuses": tuple(
                    sorted(set(unit.secondary_link_status))
                ),
                "household_source_uncertain": bool(
                    unit.person_household_id.isin(uncertain).any()
                ),
                "outside_acs_household_universe": bool(unit.TYPEHUGQ.ne(1).all()),
            }
        )
    return pd.DataFrame(records)


def assemble_acs_spm_source(
    frame: Frame,
    *,
    options: AcsSpmSourceAssemblyOptions,
    id_ceiling: int,
    registry: AcsSpmSourceRegistry | None = None,
) -> AcsSpmSourceAssemblyResult:
    """Construct only SPM structure; optionally apply a whole-roster registry.

    No native files are read. A supplied registry must bind this exact source
    roster (or complete-household subset), construction policy and assembler.
    Roles and unit evidence are returned separately, never as engine inputs.
    """
    ceiling = _integer(id_ceiling)
    result, rows, probe = _proposal(frame, options)
    if registry is None:
        assigned = _allocated(rows, ceiling)
        registry = AcsSpmSourceRegistry(
            options,
            ceiling,
            probe.module_sha256,
            tuple(
                AcsSpmSourceRegistryEntry(
                    **{**asdict(row), "new_spm_unit_id": assigned[row.component_sha256]}
                )
                for row in rows
            ),
        )
    _validate_registry(registry)
    _require(
        registry.options == options
        and registry.id_ceiling == ceiling
        and registry.assembler_sha256 == probe.module_sha256,
        "REGISTRY_CONTRACT",
    )
    lookup = {(row.serialno, row.sporder): row for row in registry.entries}
    applied = []
    for row in rows:
        registered = lookup.get((row.serialno, row.sporder))
        _require(registered is not None, "REGISTRY_ROSTER_DOMAIN")
        _require(
            {
                key: value
                for key, value in asdict(registered).items()
                if key != "new_spm_unit_id"
            }
            == {
                key: value
                for key, value in asdict(row).items()
                if key != "new_spm_unit_id"
            },
            "REGISTRY_SOURCE_OR_PARTITION_CHANGED",
        )
        applied.append(registered)
    # Whole-household chunks must not accept an incomplete native household from
    # a forged smaller NP declaration that happens to retain one component.
    serials = {row.serialno for row in applied}
    _require(
        {(row.serialno, row.sporder) for row in applied}
        == {key for key in lookup if key[0] in serials},
        "REGISTRY_COMPLETE_HOUSEHOLDS",
    )
    crosswalk = pd.DataFrame([asdict(row) for row in applied])
    by_person = crosswalk.set_index("person_id").new_spm_unit_id
    tables = {entity: frame.table(entity).copy(deep=True) for entity in frame.entities}
    tables["person"]["person_spm_unit_id"] = frame.person.person_id.map(
        by_person
    ).astype("int64")
    tables["spm_unit"] = pd.DataFrame(
        {"spm_unit_id": sorted(set(by_person))}, dtype="int64"
    )
    evidence = _unit_evidence(result)
    receipt = {
        "protocol": PROTOCOL,
        "scope": "development_source_structure_only",
        "options": asdict(options),
        "assembler": probe.as_provenance(),
        "implementation_file_sha256": {
            "adapter": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "partition_helper": hashlib.sha256(
                Path(partition.__file__).read_bytes()
            ).hexdigest(),
        },
        "registry_sha256": registry.sha256,
        "registry_id_ceiling": ceiling,
        "registry_selected_roster_sha256": registry.selected_roster_sha256,
        "registry_domain": "whole_selected_native_roster_not_national_ids",
        "registry_persons": len(registry.entries),
        "applied_persons": len(applied),
        "input_table_sha256": {
            entity: _table_digest(frame.table(entity)) for entity in frame.entities
        },
        "output_table_sha256": {
            entity: _table_digest(table) for entity, table in tables.items()
        },
        "partition_sha256": _table_digest(result.membership),
        "crosswalk_sha256": _table_digest(crosswalk),
        "unit_evidence_sha256": _table_digest(evidence),
        "partition_evidence": _json_value(result.provenance),
        "role_source_counts": result.membership.role_source.value_counts().to_dict(),
        "units_with_household_source_uncertainty": int(
            evidence.household_source_uncertain.sum()
        ),
        "unit_authority": "construction_and_link_uncertainty_retained_no_consumer_qualification",
        "relationship_allocation_provenance": "unresolved",
        "secondary_links_assessments_role_decisions": "none_supplied",
        "source_file_authentication": False,
        "engine_role_delivered": False,
        "annual_universe_declared": False,
        "unit_amounts_copied": False,
        "tax_units_modified": False,
        "household_weight_sha256": hashlib.sha256(
            frame.weights_for("household").values.tobytes()
        ).hexdigest(),
        "household_weight_kind": frame.weights_for("household").kind.value,
        "strata_sha256": _table_digest(frame.strata.to_frame()),
    }
    updated = Frame(
        tables,
        frame.schema,
        dict(frame._weights),
        frame.strata,
        mass_log=frame.mass_log,
        metadata={**frame.metadata, RECEIPT_KEY: receipt},
    )
    for entity in frame.entities:
        if entity == "spm_unit":
            continue
        columns = [name for name in frame.table(entity) if name != "person_spm_unit_id"]
        _require(
            frame.table(entity)[columns].equals(updated.table(entity)[columns]),
            "NON_SPM_CHANGED",
        )
    _require(
        frame.strata.equals(updated.strata)
        and frame.mass_log == updated.mass_log
        and np.array_equal(
            frame.weights_for("household").values,
            updated.weights_for("household").values,
        ),
        "MASS_OR_STRATA_CHANGED",
    )
    return AcsSpmSourceAssemblyResult(
        updated, result, registry, crosswalk, evidence, receipt
    )
