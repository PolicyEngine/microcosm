"""Final-value ownership for late US targets touched by multiple producers.

The stacked pipeline intentionally exposes three PUF-recipient targets to a
second post-clone callback before the late ACS transfer pass.  This module
turns that historical ordering into a closed, content-addressed owner matrix:
one final owner for every target, origin, and clone role, plus an explicit
disposition for each non-owner.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from types import MappingProxyType

__all__ = [
    "US_LATE_EDUCATION_NOOP_TARGETS",
    "US_LATE_OVERLAP_OWNERSHIP_TARGETS",
    "US_LATE_RETIREMENT_SOURCE_MIRROR_TARGETS",
    "US_LATE_SOURCE_CALLBACK_PASSTHROUGH_OUTPUTS",
    "us_late_overlap_ownership_receipt",
    "validate_us_late_overlap_ownership_receipt",
]

_PRIMARY_PUF_PRODUCER = "primary_puf_qrf"
_EDUCATION_SOURCE_PRODUCER = "source:with_us_education_inputs"
_RETIREMENT_SOURCE_PRODUCER = "source:with_us_retirement_contribution_inputs"
_TUITION_TRANSFER_PRODUCER = "transfer:person/puf_tax_itemization__batch_2"
_TRADITIONAL_IRA_TRANSFER_PRODUCER = "transfer:person/puf_tax_itemization__batch_2"
_SELF_EMPLOYED_PENSION_TRANSFER_PRODUCER = (
    "transfer:person/puf_tax_itemization__batch_3"
)

_QUALIFIED_TUITION = "qualified_tuition_expenses"
_TRADITIONAL_IRA = "traditional_ira_contributions_desired"
_SELF_EMPLOYED_PENSION = "self_employed_pension_contributions_desired"

US_LATE_EDUCATION_NOOP_TARGETS = (_QUALIFIED_TUITION,)
US_LATE_RETIREMENT_SOURCE_MIRROR_TARGETS = (
    _TRADITIONAL_IRA,
    _SELF_EMPLOYED_PENSION,
)
US_LATE_OVERLAP_OWNERSHIP_TARGETS = tuple(
    ("person", target)
    for target in (
        _QUALIFIED_TUITION,
        _TRADITIONAL_IRA,
        _SELF_EMPLOYED_PENSION,
    )
)
US_LATE_SOURCE_CALLBACK_PASSTHROUGH_OUTPUTS: Mapping[
    str, tuple[tuple[str, str], ...]
] = MappingProxyType(
    {
        "with_us_education_inputs": (("person", _QUALIFIED_TUITION),),
    }
)

_TARGET_SPECS = (
    {
        "entity": "person",
        "target": _QUALIFIED_TUITION,
        "source_producer": _EDUCATION_SOURCE_PRODUCER,
        "source_touch": "consume_only_byte_exact_noop",
        "transfer_producer": _TUITION_TRANSFER_PRODUCER,
    },
    {
        "entity": "person",
        "target": _TRADITIONAL_IRA,
        "source_producer": _RETIREMENT_SOURCE_PRODUCER,
        "source_touch": "persisted_owner_last_write",
        "transfer_producer": _TRADITIONAL_IRA_TRANSFER_PRODUCER,
    },
    {
        "entity": "person",
        "target": _SELF_EMPLOYED_PENSION,
        "source_producer": _RETIREMENT_SOURCE_PRODUCER,
        "source_touch": "persisted_owner_last_write",
        "transfer_producer": _SELF_EMPLOYED_PENSION_TRANSFER_PRODUCER,
    },
)


def _producer_action(
    producer: str,
    *,
    final_owner: str,
    action: str,
) -> dict[str, object]:
    return {
        "producer": producer,
        "owns_final": producer == final_owner,
        "action": action,
    }


def _ownership_row(
    spec: Mapping[str, str],
    *,
    origin: str,
    clone_index: int,
) -> dict[str, object]:
    source = spec["source_producer"]
    transfer = spec["transfer_producer"]
    tuition = spec["target"] == _QUALIFIED_TUITION

    if tuition:
        if clone_index == 0:
            owner = transfer
            finalization = "late_transfer_owner_last"
            primary_action = "scope_masked_noop"
            source_action = "consume_only_byte_exact_noop"
            transfer_action = "final_write"
        else:
            owner = _PRIMARY_PUF_PRODUCER
            finalization = (
                "primary_write"
                if clone_index == 1
                else "byte_exact_clone_1_inheritance"
            )
            primary_action = finalization
            source_action = "consume_only_byte_exact_noop"
            transfer_action = "producer_masked_byte_exact_noop"
    elif origin == "asec":
        owner = source
        transfer_action = "producer_masked_byte_exact_noop"
        if clone_index == 0:
            finalization = "source_direct_split"
            primary_action = "scope_masked_noop"
            source_action = "final_write"
        elif clone_index == 1:
            finalization = "source_owner_last_overwrite"
            primary_action = "interim_write_overwritten_by_owner_last"
            source_action = "final_write"
        else:
            finalization = "byte_exact_clone_1_mirror"
            primary_action = "interim_clone_1_inheritance_overwritten_by_owner_last"
            source_action = "byte_exact_clone_1_mirror"
    elif clone_index == 0:
        owner = transfer
        finalization = "late_transfer_owner_last"
        primary_action = "scope_masked_noop"
        source_action = "origin_projection_masked_noop"
        transfer_action = "final_write"
    else:
        owner = _PRIMARY_PUF_PRODUCER
        finalization = (
            "primary_write" if clone_index == 1 else "byte_exact_clone_1_inheritance"
        )
        primary_action = finalization
        source_action = "origin_projection_masked_noop"
        transfer_action = "producer_masked_byte_exact_noop"

    actions = [
        _producer_action(
            _PRIMARY_PUF_PRODUCER,
            final_owner=owner,
            action=primary_action,
        ),
        _producer_action(source, final_owner=owner, action=source_action),
        _producer_action(transfer, final_owner=owner, action=transfer_action),
    ]
    return {
        "entity": spec["entity"],
        "target": spec["target"],
        "origin": origin,
        "clone_index": clone_index,
        "final_owner": owner,
        "finalization": finalization,
        "producer_actions": actions,
    }


def _ownership_payload() -> dict[str, object]:
    ownership = [
        _ownership_row(spec, origin=origin, clone_index=clone_index)
        for spec in _TARGET_SPECS
        for origin in ("asec", "acs")
        for clone_index in range(3)
    ]
    return {
        "artifact_kind": "microcosm_us_late_overlap_ownership",
        "schema_version": 1,
        "doctrine": {
            "owner_cardinality": "exactly_one_per_target_origin_clone_role",
            "non_owner_write_policy": "masked_or_verified_byte_exact_noop",
            "retirement_legacy_order": (
                "primary_then_postclone_source_owner_then_late_transfer"
            ),
            "clone_2_policy": "inherit_or_mirror_clone_1_final_owner_bytes",
            "tail_preservation_guard": "unchanged_fail_closed",
        },
        "targets": [dict(spec) for spec in _TARGET_SPECS],
        "ownership": ownership,
    }


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def us_late_overlap_ownership_receipt() -> Mapping[str, object]:
    """Return the canonical content-addressed 3 x 2 x 3 owner matrix."""

    payload = _ownership_payload()
    return MappingProxyType({**payload, "sha256": _canonical_sha256(payload)})


def validate_us_late_overlap_ownership_receipt(
    receipt: Mapping[str, object],
) -> str:
    """Validate both the digest and exact reviewed overlap-ownership content."""

    if not isinstance(receipt, Mapping):
        raise TypeError("US late overlap ownership receipt must be a mapping.")
    observed = dict(receipt)
    claimed_sha256 = observed.pop("sha256", None)
    if (
        not isinstance(claimed_sha256, str)
        or len(claimed_sha256) != 64
        or any(character not in "0123456789abcdef" for character in claimed_sha256)
    ):
        raise ValueError("US late overlap ownership receipt has an invalid sha256.")
    actual_sha256 = _canonical_sha256(observed)
    if claimed_sha256 != actual_sha256:
        raise ValueError("US late overlap ownership receipt sha256 does not match.")
    expected = _ownership_payload()
    if observed != expected:
        raise ValueError(
            "US late overlap ownership receipt differs from the canonical owner matrix."
        )

    ownership = observed["ownership"]
    if not isinstance(ownership, list) or len(ownership) != 18:
        raise ValueError("US late overlap ownership must contain exactly 18 rows.")
    for row in ownership:
        if not isinstance(row, Mapping):
            raise ValueError("US late overlap ownership rows must be mappings.")
        actions = row.get("producer_actions")
        if (
            not isinstance(actions, list)
            or sum(
                action.get("owns_final") is True
                for action in actions
                if isinstance(action, Mapping)
            )
            != 1
        ):
            raise ValueError(
                "US late overlap ownership requires exactly one final producer."
            )
    return claimed_sha256
