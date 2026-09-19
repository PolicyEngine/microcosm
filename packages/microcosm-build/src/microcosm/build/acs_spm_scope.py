"""Pure ACS SPM evidence classification, not a source-authority issuer.

The caller must authenticate the live native preparation owner, implementation
and retained evidence before and after using this result. Accepting a mapping
here grants no source authority, engine permission or release qualification.
No files, assembler, country model or population runtime are accessed.

The reduction is the per-unit form of the reviewed source-integration helper
``acs_spm_source_receipt.py:108-147`` at 4d567e2e04bb6494ae895638fe5f675bda4bc0f7.
The annual decisions preserve the separately reviewed 512B household-only ACS
2024 profile. Construction success itself is never an inclusion decision.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from microcosm.build.acs_spm_partition import ACS_SPM_DEVELOPMENT_POLICY
from microcosm.build.spm_input_contract import INCLUDED, OUTSIDE, UNRESOLVED

# One canonical domain definition for the per-unit and aggregate views. This is
# the reviewed receipt's evidence order, not an official or release scope scale.
ACS_SPM_AUTHORITY_ORDER = (
    "observed_relationship_rule",
    "approved_inference",
    "modeled_assumption",
    "unresolved",
)
ACS_SPM_ASSUMPTION_AUTHORITY = MappingProxyType(
    {
        "reference_relationship_rule": 0,
        "accepted_sharing_links": 1,
        "parent_unknown_reference_pooling": 2,
        "parent_unknown_residual_separation": 2,
        "accepted_sharing_and_parent_unknown_reference_pooling": 2,
        "pending_relationship_resolution": 3,
    }
)
ACS_SPM_SECONDARY_AUTHORITY = MappingProxyType(
    {
        "not_required": 0,
        "complete": 1,
        "modeled_residual_child_attachment": 2,
        "modeled_residual_separation": 2,
        "unassessed": 3,
        "ambiguous": 3,
    }
)
ACS_SPM_LINK_AUTHORITY = MappingProxyType(
    {"source_observed": 0, "approved_inference": 1}
)
ACS_SPM_OUTSIDE_AUTHORITY = "outside_acs_household_universe"
ACS_SPM_PROFILE_NAME = "acs_spm_household_only_analysis_2024"
ACS_SPM_METHODOLOGY = "https://www.census.gov/content/dam/Census/library/working-papers/2020/demo/SEHSD-WP2020-09.pdf#page=6"
ACS_SPM_LINEAGE = (
    "https://www.census.gov/library/working-papers/2026/demo/sehsd-wp2026-12.html"
)


class ACSSPMScopeError(ValueError):
    """The supplied evidence is incomplete or inconsistent with this profile."""


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise ACSSPMScopeError("ACS_SPM_SCOPE_" + code)


@dataclass(frozen=True)
class ACSAnalysisProfile:
    """Explicit development analysis scope; admission flags have no defaults."""

    year: int
    source_vintage: str
    admit_modeled: bool
    admit_approved_inference: bool
    name: str = ACS_SPM_PROFILE_NAME
    methodology: str = ACS_SPM_METHODOLOGY
    lineage: str = ACS_SPM_LINEAGE

    def __post_init__(self) -> None:
        _require(type(self.year) is int and self.year == 2024, "ANNUAL_YEAR")
        _require(self.source_vintage == "acs_2024_1yr", "SOURCE_VINTAGE")
        _require(self.name == ACS_SPM_PROFILE_NAME, "PROFILE_NAME")
        _require(type(self.admit_modeled) is bool, "MODELED_ADMISSION")
        _require(type(self.admit_approved_inference) is bool, "INFERENCE_ADMISSION")
        _require(
            self.methodology == ACS_SPM_METHODOLOGY and self.lineage == ACS_SPM_LINEAGE,
            "PROFILE_REFERENCES",
        )


@dataclass(frozen=True)
class ACSUnitAuthority:
    """Classification of supplied evidence; not an authenticated source claim."""

    canonical_id: str
    household_id: int
    person_ids: tuple[int, ...]
    authority: str
    household_kind: int
    missing_role: bool
    uncertain_household: bool


@dataclass(frozen=True)
class ACSUnitScope:
    """Annual decision in native IDs, before assembly/clone identity transport."""

    spm_unit_id: int
    canonical_id: str
    household_id: int
    person_ids: tuple[int, ...]
    authority: str
    status: str
    reason: str


@dataclass(frozen=True)
class ACSSPMRole:
    """Unmodified nullable role with the source's exact membership and labels."""

    person_id: int
    household_id: int
    spm_unit_id: int
    canonical_id: str
    value: bool | None
    role_source: str
    role_rule: str


@dataclass(frozen=True)
class ACSSPMScopeResult:
    """Detached pure result; the calling owner must bind its provenance."""

    units: tuple[ACSUnitScope, ...]
    roles: tuple[ACSSPMRole, ...]
    profile: ACSAnalysisProfile
    minor_partner_role: bool


def _mapping(value: Any, required: Sequence[str], code: str) -> Mapping:
    _require(isinstance(value, Mapping) and set(required).issubset(value), code)
    return value


def _sequence(value: Any, code: str) -> Sequence:
    _require(isinstance(value, (list, tuple)), code)
    return value


def _integer(value: Any, code: str, *, maximum: int = 2**63 - 1) -> int:
    _require(type(value) is int and 0 <= value <= maximum, code)
    return value


def _text(value: Any, code: str) -> str:
    _require(type(value) is str and bool(value), code)
    return value


def _digest(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and not set(value) - set("0123456789abcdef")
    )


def _table(value: Any, required: Sequence[str]) -> tuple[dict, ...]:
    """Validate full serialized shape without dtype coercion or missing defaults."""
    value = _mapping(value, ("columns", "dtypes", "index", "rows"), "TABLE_SHAPE")
    columns = _sequence(value["columns"], "TABLE_COLUMNS")
    _require(all(type(item) is str and item for item in columns), "TABLE_COLUMNS")
    _require(
        len(set(columns)) == len(columns) and set(required).issubset(columns),
        "TABLE_COLUMNS",
    )
    dtypes = _sequence(value["dtypes"], "TABLE_DTYPES")
    _require(
        len(dtypes) == len(columns)
        and all(type(item) is str and item for item in dtypes),
        "TABLE_DTYPES",
    )
    rows = _sequence(value["rows"], "TABLE_ROWS")
    index = _sequence(value["index"], "TABLE_INDEX")
    _require(len(index) == len(rows), "TABLE_INDEX")
    _require(
        all(type(item) in (int, str) for item in index)
        and len(set(index)) == len(index),
        "TABLE_INDEX",
    )
    result = []
    for row in rows:
        _require(
            isinstance(row, (list, tuple)) and len(row) == len(columns),
            "TABLE_ROW_WIDTH",
        )
        result.append(dict(zip(columns, row, strict=True)))
    return tuple(result)


def derive_acs_spm_unit_authorities(
    membership: Sequence[Mapping[str, Any]],
    links: Sequence[Mapping[str, Any]],
) -> tuple[ACSUnitAuthority, ...]:
    """Reduce exact supplied person/partition/link evidence using reviewed rules.

    No membership is constructed or inferred. Household uncertainty is computed
    before grouping units, so an observed reference person cannot conceal an
    unresolved secondary relationship elsewhere in the same household.
    """
    required = (
        "person_id",
        "person_household_id",
        "proposed_spm_unit_id",
        "TYPEHUGQ",
        "independent_minor_role",
        "role_source",
        "partition_assumption",
        "secondary_link_status",
    )
    people, units, kinds, uncertain = {}, defaultdict(list), {}, set()
    for row in _sequence(membership, "MEMBERSHIP"):
        row = _mapping(row, required, "MEMBERSHIP_COLUMNS")
        pid = _integer(row["person_id"], "PERSON_ID")
        _require(pid not in people, "DUPLICATE_PERSON")
        household = _integer(row["person_household_id"], "HOUSEHOLD_ID")
        canonical = _text(row["proposed_spm_unit_id"], "CANONICAL_ID")
        kind = _integer(row["TYPEHUGQ"], "HOUSEHOLD_KIND")
        _require(
            kind in (1, 2, 3) and kinds.get(household, kind) == kind, "HOUSEHOLD_KIND"
        )
        kinds[household] = kind
        _require(
            row["independent_minor_role"] is None
            or type(row["independent_minor_role"]) is bool,
            "ROLE_TYPE",
        )
        role_source = _text(row["role_source"], "ROLE_SOURCE")
        _require(
            role_source
            in (
                *ACS_SPM_AUTHORITY_ORDER,
                "age_not_role_sensitive",
                ACS_SPM_OUTSIDE_AUTHORITY,
            ),
            "ROLE_SOURCE",
        )
        assumption = _text(row["partition_assumption"], "PARTITION_ASSUMPTION")
        secondary = _text(row["secondary_link_status"], "SECONDARY_STATUS")
        _require(secondary in ACS_SPM_SECONDARY_AUTHORITY, "SECONDARY_STATUS")
        if kind == 1:
            _require(
                role_source != ACS_SPM_OUTSIDE_AUTHORITY
                and assumption in ACS_SPM_ASSUMPTION_AUTHORITY,
                "HOUSEHOLD_AUTHORITY",
            )
        else:
            _require(
                row["independent_minor_role"] is None
                and role_source == ACS_SPM_OUTSIDE_AUTHORITY
                and assumption == "preserved_gq_membership",
                "OUTSIDE_ROLE",
            )
        if secondary in ("unassessed", "ambiguous"):
            uncertain.add(household)
        people[pid] = row
        units[canonical].append(row)
    _require(bool(people), "EMPTY_MEMBERSHIP")
    unit_links = defaultdict(list)
    for link in _sequence(links, "LINKS"):
        link = _mapping(link, ("person_id", "relative_id", "source"), "LINK_COLUMNS")
        pid, relative = (
            _integer(link[name], "LINK_PERSON") for name in ("person_id", "relative_id")
        )
        _require(
            pid in people and relative in people and pid != relative, "LINK_PERSON"
        )
        _require(
            people[pid]["person_household_id"]
            == people[relative]["person_household_id"],
            "LINK_HOUSEHOLD",
        )
        source = _text(link["source"], "LINK_SOURCE")
        _require(source in ACS_SPM_LINK_AUTHORITY, "LINK_SOURCE")
        if (
            people[pid]["proposed_spm_unit_id"]
            == people[relative]["proposed_spm_unit_id"]
        ):
            unit_links[people[pid]["proposed_spm_unit_id"]].append(
                ACS_SPM_LINK_AUTHORITY[source]
            )
    result = []
    for canonical, rows in sorted(units.items()):
        households = {row["person_household_id"] for row in rows}
        _require(len(households) == 1, "UNIT_CROSSES_HOUSEHOLDS")
        household = next(iter(households))
        kind = kinds[household]
        if kind != 1:
            authority = ACS_SPM_OUTSIDE_AUTHORITY
        else:
            strengths = [3 if household in uncertain else 0, *unit_links[canonical]]
            for row in rows:
                if row["role_source"] != "age_not_role_sensitive":
                    strengths.append(ACS_SPM_AUTHORITY_ORDER.index(row["role_source"]))
                strengths.extend(
                    (
                        ACS_SPM_ASSUMPTION_AUTHORITY[row["partition_assumption"]],
                        ACS_SPM_SECONDARY_AUTHORITY[row["secondary_link_status"]],
                    )
                )
            authority = ACS_SPM_AUTHORITY_ORDER[max(strengths)]
        result.append(
            ACSUnitAuthority(
                canonical,
                household,
                tuple(sorted(row["person_id"] for row in rows)),
                authority,
                kind,
                any(row["independent_minor_role"] is None for row in rows),
                household in uncertain,
            )
        )
    return tuple(result)


def count_acs_spm_authorities(
    units: Sequence[ACSUnitAuthority | ACSUnitScope],
) -> dict[str, int]:
    """Aggregate the canonical per-unit reduction; never classify a second way."""
    return dict(sorted(Counter(row.authority for row in units).items()))


def _status(
    authority: ACSUnitAuthority, profile: ACSAnalysisProfile
) -> tuple[str, str]:
    if authority.uncertain_household or authority.authority == "unresolved":
        return UNRESOLVED, "source_membership_or_authority_unresolved"
    if authority.authority == ACS_SPM_OUTSIDE_AUTHORITY:
        _require(authority.household_kind in (2, 3), "OUTSIDE_UNIVERSE")
        return OUTSIDE, "outside_declared_acs_household_only_analysis"
    if authority.missing_role:
        return UNRESOLVED, "source_role_missing"
    if authority.authority == "modeled_assumption":
        return (
            (INCLUDED, "explicit_modeled_admission")
            if profile.admit_modeled
            else (UNRESOLVED, "modeled_not_admitted")
        )
    if authority.authority == "approved_inference":
        return (
            (INCLUDED, "explicit_approved_inference_admission")
            if profile.admit_approved_inference
            else (UNRESOLVED, "inference_not_admitted")
        )
    _require(authority.authority == "observed_relationship_rule", "AUTHORITY")
    return INCLUDED, "observed_relationship_rule_with_resolved_membership"


def _options(evidence: Mapping) -> bool:
    required = (
        "implementation",
        "registry",
        "construction_receipt",
        "partition_provenance",
    )
    choices = []
    for name in required:
        item = _mapping(
            evidence[name],
            ("policy", "minor_partner_role")
            if name == "partition_provenance"
            else ("options",),
            "CONSTRUCTION_OPTIONS",
        )
        options = (
            item
            if name == "partition_provenance"
            else _mapping(
                item["options"],
                ("policy", "minor_partner_role"),
                "CONSTRUCTION_OPTIONS",
            )
        )
        _require(
            options["policy"] == ACS_SPM_DEVELOPMENT_POLICY
            and type(options["minor_partner_role"]) is bool,
            "CONSTRUCTION_OPTIONS",
        )
        choices.append(options["minor_partner_role"])
    _require(
        all(value == choices[0] for value in choices), "CONSTRUCTION_OPTIONS_DISAGREE"
    )
    return choices[0]


def _registry(evidence: Mapping, members: Sequence[Mapping]) -> dict[int, Mapping]:
    fields = (
        "serialno",
        "sporder",
        "person_id",
        "household_id",
        "old_spm_unit_id",
        "new_spm_unit_id",
        "component_sha256",
        "source_record_sha256",
    )
    registry = _mapping(
        evidence["registry"], ("entries", "id_ceiling", "assembler_sha256"), "REGISTRY"
    )
    ceiling = _integer(registry["id_ceiling"], "REGISTRY_CEILING")
    _require(_digest(registry["assembler_sha256"]), "ASSEMBLER_DIGEST")
    lookup, keys, household_serials = {}, set(), {}
    for row in _sequence(registry["entries"], "REGISTRY_ROWS"):
        row = _mapping(row, fields, "REGISTRY_COLUMNS")
        _require(set(row) == set(fields), "REGISTRY_COLUMNS")
        pid = _integer(row["person_id"], "REGISTRY_PERSON")
        _require(pid not in lookup, "REGISTRY_DUPLICATE_PERSON")
        household = _integer(row["household_id"], "REGISTRY_HOUSEHOLD")
        serial = _text(row["serialno"], "REGISTRY_SERIAL")
        order = _integer(row["sporder"], "REGISTRY_SPORDER")
        _require(order > 0 and (serial, order) not in keys, "REGISTRY_NATIVE_KEYS")
        keys.add((serial, order))
        _require(
            household_serials.get(household, serial) == serial,
            "REGISTRY_HOUSEHOLD_SERIAL",
        )
        household_serials[household] = serial
        _integer(row["old_spm_unit_id"], "REGISTRY_OLD_UNIT", maximum=ceiling)
        _integer(row["new_spm_unit_id"], "REGISTRY_NEW_UNIT", maximum=ceiling)
        _require(
            _digest(row["component_sha256"]) and _digest(row["source_record_sha256"]),
            "REGISTRY_DIGEST",
        )
        lookup[pid] = dict(row)
    _require(
        len(set(household_serials.values())) == len(household_serials),
        "REGISTRY_SERIAL_COLLISION",
    )
    _require(set(lookup) == {row["person_id"] for row in members}, "REGISTRY_ROSTER")
    native = _table(evidence["native_crosswalk"], fields)
    native_ids = [_integer(row["person_id"], "CROSSWALK_PERSON") for row in native]
    for row in native:
        for name in ("sporder", "household_id", "old_spm_unit_id", "new_spm_unit_id"):
            _integer(row[name], "CROSSWALK_EXACT_INTEGER")
    _require(
        len(set(native_ids)) == len(native_ids)
        and {row["person_id"]: row for row in native} == lookup,
        "NATIVE_CROSSWALK",
    )
    canonical_native, native_canonical, components = {}, {}, defaultdict(list)
    old_households = {}
    for row in members:
        registered = lookup[row["person_id"]]
        for source, target in (
            ("person_household_id", "household_id"),
            ("old_spm_unit_id", "old_spm_unit_id"),
            ("SPORDER", "sporder"),
        ):
            _require(
                _integer(row[source], "MEMBERSHIP_ID") == registered[target],
                "REGISTRY_MEMBERSHIP",
            )
        _text(row["role_rule"], "ROLE_RULE")
        old, household = row["old_spm_unit_id"], row["person_household_id"]
        _require(old_households.get(old, household) == household, "OLD_UNIT_HOUSEHOLD")
        old_households[old] = household
        canonical, native_id = (
            row["proposed_spm_unit_id"],
            registered["new_spm_unit_id"],
        )
        _require(
            canonical_native.get(canonical, native_id) == native_id
            and native_canonical.get(native_id, canonical) == canonical,
            "CANONICAL_NATIVE_BIJECTION",
        )
        canonical_native[canonical], native_canonical[native_id] = native_id, canonical
        components[canonical].append(registered)
    for rows in components.values():
        _require(len({row["old_spm_unit_id"] for row in rows}) == 1, "OLD_UNIT_MERGE")
        expected = hashlib.sha256(
            json.dumps(
                sorted((row["serialno"], row["sporder"]) for row in rows),
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        _require(
            all(row["component_sha256"] == expected for row in rows),
            "COMPONENT_MEMBERSHIP",
        )
    return lookup


def _summaries(
    evidence: Mapping,
    members: Sequence[Mapping],
    authorities: Sequence[ACSUnitAuthority],
) -> None:
    cross_fields = (
        "person_household_id",
        "old_spm_unit_id",
        "proposed_spm_unit_id",
        "person_count",
    )
    actual = _table(evidence["partition"]["crosswalk"], cross_fields)
    expected = Counter(
        (
            row["person_household_id"],
            row["old_spm_unit_id"],
            row["proposed_spm_unit_id"],
        )
        for row in members
    )
    observed = {}
    for row in actual:
        key = (
            _integer(row["person_household_id"], "PARTITION_CROSSWALK_ID"),
            _integer(row["old_spm_unit_id"], "PARTITION_CROSSWALK_ID"),
            _text(row["proposed_spm_unit_id"], "PARTITION_CROSSWALK_ID"),
        )
        _require(key not in observed, "PARTITION_CROSSWALK_DUPLICATE")
        observed[key] = _integer(row["person_count"], "PARTITION_CROSSWALK_COUNT")
    _require(observed == expected, "PARTITION_CROSSWALK")
    _table(evidence["partition"]["regrouping"], ("field", "action"))
    fields = (
        "proposed_spm_unit_id",
        "role_sources",
        "partition_assumptions",
        "secondary_link_statuses",
        "household_source_uncertain",
        "outside_acs_household_universe",
    )
    observed_units = _table(evidence["unit_evidence"], fields)
    _require(len(observed_units) == len(authorities), "UNIT_EVIDENCE_ROSTER")
    lookup = {row.canonical_id: row for row in authorities}
    groups = defaultdict(list)
    for row in members:
        groups[row["proposed_spm_unit_id"]].append(row)
    seen = set()
    for row in observed_units:
        canonical = _text(row["proposed_spm_unit_id"], "UNIT_EVIDENCE_ID")
        _require(canonical in lookup and canonical not in seen, "UNIT_EVIDENCE_ID")
        seen.add(canonical)
        authority = lookup[canonical]
        for column, source in (
            ("role_sources", "role_source"),
            ("partition_assumptions", "partition_assumption"),
            ("secondary_link_statuses", "secondary_link_status"),
        ):
            _require(
                list(_sequence(row[column], "UNIT_EVIDENCE_LABELS"))
                == sorted({member[source] for member in groups[canonical]}),
                "UNIT_EVIDENCE_LABELS",
            )
        for column, expected_flag in (
            ("household_source_uncertain", authority.uncertain_household),
            ("outside_acs_household_universe", authority.household_kind != 1),
        ):
            _require(
                type(row[column]) is bool and row[column] == expected_flag,
                "UNIT_EVIDENCE_FLAG",
            )


def classify_acs_spm_scope(
    evidence: Mapping[str, Any], profile: ACSAnalysisProfile
) -> ACSSPMScopeResult:
    """Classify owner-supplied construction rows without issuing source authority.

    Requires the complete encoded membership, links, registry, crosswalks and
    unit evidence emitted by the native constructor. Returned IDs remain native;
    the owner must bind them through prepared origins and final clone membership.
    Nullable roles are never filled or cast, including for OUTSIDE units.
    """
    _require(type(profile) is ACSAnalysisProfile, "PROFILE_TYPE")
    profile.__post_init__()
    fields = (
        "protocol",
        "source_binding",
        "implementation",
        "construction_receipt",
        "partition_provenance",
        "registry",
        "partition",
        "native_crosswalk",
        "unit_evidence",
        "engine_role_delivered",
        "annual_universe_declared",
        "release_eligible",
    )
    evidence = _mapping(evidence, fields, "CONSTRUCTION_EVIDENCE")
    _require(
        evidence["protocol"] == "microcosm.acs-spm-native-construction-evidence.v1",
        "EVIDENCE_PROTOCOL",
    )
    _require(
        evidence["source_binding"]
        == "actual_construction_from_owner_captured_archives",
        "EVIDENCE_BINDING",
    )
    _require(
        all(
            evidence[name] is False
            for name in (
                "engine_role_delivered",
                "annual_universe_declared",
                "release_eligible",
            )
        ),
        "CONSTRUCTION_SCOPE",
    )
    partner = _options(evidence)
    partition = _mapping(
        evidence["partition"],
        ("membership", "links", "crosswalk", "regrouping"),
        "PARTITION",
    )
    members = _table(
        partition["membership"],
        (
            "person_id",
            "person_household_id",
            "old_spm_unit_id",
            "SPORDER",
            "proposed_spm_unit_id",
            "TYPEHUGQ",
            "independent_minor_role",
            "role_source",
            "role_rule",
            "partition_assumption",
            "secondary_link_status",
            "measurement_status",
        ),
    )
    links = _table(
        partition["links"], ("person_id", "relative_id", "kind", "source", "rule_id")
    )
    authorities = derive_acs_spm_unit_authorities(members, links)
    for row in members:
        expected = (
            "included_acs_household"
            if row["TYPEHUGQ"] == 1
            else ACS_SPM_OUTSIDE_AUTHORITY
        )
        _require(row["measurement_status"] == expected, "PARTITION_MEASUREMENT_LABEL")
    registry = _registry(evidence, members)
    _summaries(evidence, members, authorities)
    units = tuple(
        sorted(
            (
                ACSUnitScope(
                    registry[row.person_ids[0]]["new_spm_unit_id"],
                    row.canonical_id,
                    row.household_id,
                    row.person_ids,
                    row.authority,
                    *_status(row, profile),
                )
                for row in authorities
            ),
            key=lambda row: row.spm_unit_id,
        )
    )
    roles = tuple(
        ACSSPMRole(
            row["person_id"],
            row["person_household_id"],
            registry[row["person_id"]]["new_spm_unit_id"],
            row["proposed_spm_unit_id"],
            row["independent_minor_role"],
            row["role_source"],
            row["role_rule"],
        )
        for row in sorted(members, key=lambda row: row["person_id"])
    )
    return ACSSPMScopeResult(units, roles, profile, partner)
