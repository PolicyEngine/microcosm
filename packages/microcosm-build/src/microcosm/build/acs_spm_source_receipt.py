"""Aggregate development evidence for ACS partition and regroup proposals.

This receipt authenticates consistency with supplied tables and reviewed helper
rules, not source files, official measurement scope, or a country consumer. It
replays only pure table transformations; no country runtime is imported.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.acs_spm_partition import (
    ACS_SPM_DEVELOPMENT_POLICY,
    AcsSpmLink,
    AcsSpmLinkAssessment,
    AcsSpmPartitionResult,
    AcsSpmRoleDecision,
    reconstruct_acs_spm_partition,
)
from microcosm.build.acs_spm_regroup import (
    AcsSpmLegacyDefaults,
    AcsSpmRegroupResult,
    regroup_acs_spm_units,
)

# Evidence order for this receipt only, not a release-eligibility classification.
_AUTHORITY = (
    "observed_relationship_rule",
    "approved_inference",
    "modeled_assumption",
    "unresolved",
)
_ASSUMPTION_AUTHORITY = {
    "reference_relationship_rule": 0,
    "accepted_sharing_links": 1,
    "parent_unknown_reference_pooling": 2,
    "parent_unknown_residual_separation": 2,
    "accepted_sharing_and_parent_unknown_reference_pooling": 2,
    "pending_relationship_resolution": 3,
}
_SECONDARY_AUTHORITY = {
    "not_required": 0,
    "complete": 1,
    "modeled_residual_child_attachment": 2,
    "modeled_residual_separation": 2,
    "unassessed": 3,
    "ambiguous": 3,
}


def _json_value(value: Any, *, table_cell: bool = False) -> Any:
    """Normalize frozen mappings/scalars; allow missing table cells, not infinity."""
    if value is pd.NA or value is None:
        return None
    if isinstance(value, np.generic):
        return _json_value(value.item(), table_cell=table_cell)
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("Receipt metadata keys must be strings.")
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        if table_cell and math.isnan(value):
            return None  # A declared unavailable cell, not a measured zero.
        raise ValueError("Receipt cannot bind nonfinite metadata or infinite cells.")
    if type(value) in (str, int, float, bool):
        return value
    raise ValueError("Receipt payload contains an unsupported value type.")


def _encoded(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_encoded(_json_value(value)).encode()).hexdigest()


def _table_digest(table: pd.DataFrame) -> str:
    if not table.columns.is_unique or not all(isinstance(c, str) for c in table):
        raise ValueError("Receipt tables require unique named columns.")
    columns = sorted(table.columns)
    rows = [
        [_json_value(cell, table_cell=True) for cell in row]
        for row in table.loc[:, columns].itertuples(index=False, name=None)
    ]
    return _digest({"columns": columns, "rows": sorted(rows, key=_encoded)})


def _same_table(actual: pd.DataFrame, expected: pd.DataFrame, name: str) -> str:
    digest = _table_digest(actual)
    if digest != _table_digest(expected):
        raise ValueError(f"Stale {name}: supplied evidence differs from source rules.")
    return digest


def _unit_authorities(result: AcsSpmPartitionResult) -> dict[str, int]:
    counts: Counter[str] = Counter()
    uncertain_households = set(
        result.membership.loc[
            result.membership.secondary_link_status.isin({"unassessed", "ambiguous"}),
            "person_household_id",
        ]
    )
    for _, unit in result.membership.groupby("proposed_spm_unit_id", sort=True):
        if unit.TYPEHUGQ.ne(1).all():
            if unit.independent_minor_role.notna().any():
                raise ValueError("Group-quarters roles must remain unavailable.")
            counts["outside_acs_household_universe"] += 1
            continue
        # Unresolved links can change component boundaries within a household;
        # an observed head alone cannot authenticate that household's partition.
        strengths = [
            3 if unit.person_household_id.isin(uncertain_households).any() else 0
        ]
        for row in unit.itertuples():
            # Age-insensitive roles have no authority of their own; construction
            # and links still decide the weakest evidence for these members.
            if row.role_source != "age_not_role_sensitive":
                strengths.append(_AUTHORITY.index(row.role_source))
            strengths.extend(
                [
                    _ASSUMPTION_AUTHORITY[row.partition_assumption],
                    _SECONDARY_AUTHORITY[row.secondary_link_status],
                ]
            )
        people = set(unit.person_id)
        internal = result.links.loc[
            result.links.person_id.isin(people) & result.links.relative_id.isin(people)
        ]
        strengths.extend(
            {"source_observed": 0, "approved_inference": 1}[source]
            for source in internal.source
        )
        counts[_AUTHORITY[max(strengths)]] += 1
    return dict(counts)


def build_acs_spm_source_receipt(
    persons: pd.DataFrame,
    spm_units: pd.DataFrame,
    households: pd.DataFrame,
    *,
    partitions: Mapping[bool, AcsSpmPartitionResult],
    membership: pd.DataFrame,
    regroup: AcsSpmRegroupResult,
    legacy_defaults: AcsSpmLegacyDefaults,
    tenure_policy: str,
    source_references: Mapping[str, str],
    links: Sequence[AcsSpmLink] = (),
    assessments: Sequence[AcsSpmLinkAssessment] = (),
    role_decisions: Sequence[AcsSpmRoleDecision] = (),
    older_care_evidence: pd.Series | None = None,
) -> dict[str, Any]:
    """Check proposals against supplied ACS evidence and return aggregate hashes.

    ``partitions`` supplies both Boolean partner sensitivities under the explicit
    development policy. ``membership`` is the already allocated pilot registry,
    including canonical labels; this function never allocates IDs. The pure
    helpers are replayed to reject stale roles, policies, crosswalks and ledgers.
    Reference hashes are caller-supplied identities, never authenticated files.
    Unresolved care and relationship evidence remains visible in the receipt.
    """
    legacy_defaults.validate()
    if not {"household_id", "TYPEHUGQ"}.issubset(households) or not {
        "person_household_id",
        "TYPEHUGQ",
    }.issubset(persons):
        raise ValueError("Receipt requires person and household universe evidence.")
    if not households.household_id.is_unique:
        raise ValueError("Receipt household universe identities must be unique.")
    household_kind = persons.person_household_id.map(
        households.set_index("household_id").TYPEHUGQ
    )
    if not persons.TYPEHUGQ.eq(household_kind).fillna(False).all():
        raise ValueError("Person and household TYPEHUGQ universe evidence disagrees.")
    if (
        "PERIDNUM" in persons
        and persons.PERIDNUM.dropna().astype(str).str.strip().ne("").any()
    ):
        raise ValueError("ACS receipt refuses nonblank ASEC PERIDNUM evidence.")
    for table, column in ((persons, "person_spine"), (spm_units, "spm_unit_spine")):
        if (
            column in table
            and not table[column].eq(legacy_defaults.acs_spine).fillna(False).all()
        ):
            raise ValueError(
                "Receipt spine evidence must match the declared ACS spine."
            )
    if (
        len(partitions) != 2
        or any(type(key) is not bool for key in partitions)
        or set(partitions) != {False, True}
    ):
        raise ValueError("Both explicit Boolean partner sensitivities are required.")
    if not source_references or not all(
        isinstance(name, str)
        and name.strip()
        and isinstance(value, str)
        and re.fullmatch(r"[0-9a-f]{64}", value)
        for name, value in source_references.items()
    ):
        raise ValueError("Supplied source references require named SHA256 identities.")
    bindings = {
        "persons": _table_digest(persons),
        "old_spm_units": _table_digest(spm_units),
        "households": _table_digest(households),
        "identity_registry": _table_digest(membership),
        "approved_links": _digest([asdict(link) for link in links]),
        "assessments": _digest([asdict(item) for item in assessments]),
        "role_decisions": _digest([asdict(item) for item in role_decisions]),
        "legacy_defaults": _digest(
            {
                "acs_spine": legacy_defaults.acs_spine,
                "parent_sha256": legacy_defaults.parent_sha256,
                "null_register_sha256": legacy_defaults.null_register_sha256,
                "values": legacy_defaults.values,
            }
        ),
        "older_care_evidence": _table_digest(
            older_care_evidence.rename_axis("person_id").rename("care").reset_index()
        )
        if older_care_evidence is not None
        else None,
    }
    partition_bindings = {}
    authorities = {}
    projections = []
    counts = persons.groupby("person_household_id").size().to_dict()
    for sensitivity in (False, True):
        supplied = partitions[sensitivity]
        # Normalize first so malformed/nonfinite metadata cannot get hidden by a
        # more generic stale-evidence comparison.
        provenance_digest = _digest(supplied.provenance)
        expected = reconstruct_acs_spm_partition(
            persons,
            spm_units,
            household_person_counts=counts,
            policy=ACS_SPM_DEVELOPMENT_POLICY,
            minor_partner_role=sensitivity,
            links=links,
            assessments=assessments,
            role_decisions=role_decisions,
        )
        expected.require_resolved()
        evidence = {
            name: _same_table(getattr(supplied, name), getattr(expected, name), name)
            for name in ("membership", "links", "crosswalk", "regrouping")
        }
        if provenance_digest != _digest(expected.provenance):
            raise ValueError(
                "Stale partition policy/provenance differs from source rules."
            )
        evidence["provenance"] = provenance_digest
        partition_bindings[str(sensitivity).lower()] = evidence
        authorities[str(sensitivity).lower()] = _unit_authorities(supplied)
        projection = supplied.membership.loc[
            :,
            [
                "person_id",
                "person_household_id",
                "old_spm_unit_id",
                "proposed_spm_unit_id",
            ],
        ]
        projections.append(_table_digest(projection))
    if projections[0] != projections[1]:
        raise ValueError("Partner sensitivities disagree on membership.")
    required_registry = {
        "person_id",
        "canonical_spm_unit_id",
        "new_spm_unit_id",
        "new_spm_unit_source_id",
    }
    if not required_registry.issubset(membership) or not membership.person_id.is_unique:
        raise ValueError(
            "Identity registry requires unique people and complete unit labels."
        )
    canonical = (
        partitions[False]
        .membership[["person_id", "proposed_spm_unit_id"]]
        .rename(columns={"proposed_spm_unit_id": "canonical_spm_unit_id"})
    )
    _same_table(
        membership[["person_id", "canonical_spm_unit_id"]],
        canonical,
        "identity registry",
    )
    expected_regroup = regroup_acs_spm_units(
        persons,
        spm_units,
        households,
        membership,
        legacy_defaults=legacy_defaults,
        tenure_policy=tenure_policy,
        older_care_evidence=older_care_evidence,
    )
    regroup_bindings = {
        name: _same_table(getattr(regroup, name), getattr(expected_regroup, name), name)
        for name in (
            "spm_units",
            "crosswalk",
            "childcare_ledger",
            "field_provenance",
            "exceptions",
        )
    }
    if _digest(regroup.metadata) != _digest(expected_regroup.metadata):
        raise ValueError("Stale regroup policy/provenance differs from source rules.")
    regroup_bindings["metadata"] = _digest(regroup.metadata)
    receipt = {
        "scope": "development_source_only",
        "acs_spine": legacy_defaults.acs_spine,
        "partition_policy": ACS_SPM_DEVELOPMENT_POLICY,
        "tenure_policy": tenure_policy,
        "assembler": _json_value(partitions[False].provenance["assembler"]),
        "bindings": bindings,
        "partitions": partition_bindings,
        "membership_projection_sha256": projections[0],
        "regroup": regroup_bindings,
        "unit_authority_counts": authorities,
        "counts": {
            "persons": len(persons),
            "old_units": len(spm_units),
            "new_units": len(regroup.spm_units),
            "group_quarters_units": int(
                regroup.metadata["group_quarters_units_preserved"]
            ),
            "source_uncertain_households": int(
                partitions[False].provenance[
                    "source_unresolved_relationship_households"
                ]
            ),
            "childcare_ledger_rows": len(regroup.childcare_ledger),
            "unresolved_childcare_units": len(regroup.exceptions),
        },
        "source_references": {
            "verification": "supplied_not_authenticated",
            "sha256_by_name": dict(source_references),
        },
    }
    return {**receipt, "receipt_sha256": _digest(receipt)}
