"""Bundle-authoritative materialization of stacked checkpoint identities.

The compiler seals a generation-0-compatible static projection, while sampling,
input pins, and clone attachment are run request values.  This module combines
those surfaces without reopening the bundle or consulting constants.  In
particular, clone attachment also updates the nested producer-resource receipt;
overlaying only the top-level request would create a false cache identity.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy

from microcosm.build.spec_engine.canonical import sha256_json
from microcosm.build.spec_engine.model import thaw_json
from microcosm.build.us_runtime.pool_runtime_plan import USPoolRuntimePlan


class CheckpointAuthorityError(ValueError):
    """A compiler checkpoint authority or run request is not closed."""


def _mapping(value: object, *, location: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise CheckpointAuthorityError(f"{location}: object required")
    return value


def _array(value: object, *, location: str) -> Sequence[object]:
    if not isinstance(value, (list, tuple)):
        raise CheckpointAuthorityError(f"{location}: array required")
    return value


def _finite_float(value: object, *, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CheckpointAuthorityError(f"{location}: finite number required")
    result = float(value)
    if not math.isfinite(result):
        raise CheckpointAuthorityError(f"{location}: finite number required")
    return result


def _seed(value: object, *, location: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > 2**63 - 1
    ):
        raise CheckpointAuthorityError(
            f"{location}: non-negative signed 64-bit integer required"
        )
    return value


def _unit_fraction(value: object, *, location: str) -> float:
    result = _finite_float(value, location=location)
    if not 0.0 < result <= 1.0:
        raise CheckpointAuthorityError(f"{location}: value in (0, 1] required")
    return result


def _legacy_sha256(value: object) -> str:
    """Use the constants-era ASCII-escaped canonical identity codec."""

    try:
        payload = json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise CheckpointAuthorityError(
            "checkpoint identity must contain canonical JSON values"
        ) from error
    return hashlib.sha256(payload).hexdigest()


def _input_pins(
    value: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for role in sorted(value):
        if not isinstance(role, str) or not role:
            raise CheckpointAuthorityError("input pin roles must be non-empty strings")
        row = _mapping(value[role], location=f"input_pins/{role}")
        digest = row.get("sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise CheckpointAuthorityError(
                f"input_pins/{role}/sha256: lowercase SHA-256 required"
            )
        size = row.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise CheckpointAuthorityError(
                f"input_pins/{role}/size_bytes: non-negative integer required"
            )
        result[role] = {"sha256": digest, "size_bytes": size}
    return result


def _rung_token(publication: Mapping[str, object], fraction: float) -> str:
    matches = [
        row.get("token")
        for index, value in enumerate(
            _array(
                publication.get("rung_fractions"),
                location="publication/rung_fractions",
            )
        )
        for row in [_mapping(value, location=f"publication/rung_fractions/{index}")]
        if _finite_float(
            row.get("fraction"),
            location=f"publication/rung_fractions/{index}/fraction",
        )
        == fraction
    ]
    if len(matches) != 1 or not isinstance(matches[0], str) or not matches[0]:
        raise CheckpointAuthorityError(
            "sample_fraction must name exactly one compiler-declared rung"
        )
    return matches[0]


def _resolve_clone_attachment(
    projected: Mapping[str, object],
    *,
    fraction: float,
    seed: int,
) -> dict[str, object]:
    result = deepcopy(dict(projected))
    matches: list[dict[str, object]] = []
    for producer_value in _array(
        result.get("producers"), location="late resource semantics/producers"
    ):
        producer = _mapping(producer_value, location="late resource semantics/producer")
        resources = _mapping(
            producer.get("resources"),
            location="late resource semantics/producer/resources",
        )
        for resource_value in resources.values():
            resource = _mapping(
                resource_value,
                location="late resource semantics/producer/resource",
            )
            binding = resource.get("binding")
            if not isinstance(binding, dict):
                continue
            attachment = binding.get("clone_attachment")
            if isinstance(attachment, dict):
                matches.append(attachment)
    if len(matches) != 1:
        raise CheckpointAuthorityError(
            "late resource semantics must expose exactly one clone attachment"
        )
    matches[0]["fraction"] = fraction
    matches[0]["seed"] = seed
    result.pop("sha256", None)
    result["sha256"] = _legacy_sha256(result)
    return result


def materialize_stacked_checkpoint_base_identity(
    plan: USPoolRuntimePlan,
    *,
    input_pins: Mapping[str, Mapping[str, object]],
    stack_receipt: Mapping[str, object],
    sample_fraction: float,
    sample_seed: int,
    clone_attachment_fraction: float,
    clone_attachment_seed: int,
) -> dict[str, object]:
    """Return the exact generation-0 checkpoint identity from bundle authority."""

    if not isinstance(plan, USPoolRuntimePlan):
        raise TypeError("checkpoint materialization requires USPoolRuntimePlan")
    fraction = _finite_float(sample_fraction, location="sample_fraction")
    sample_seed_value = _seed(sample_seed, location="sample_seed")
    attachment_fraction = _unit_fraction(
        clone_attachment_fraction, location="clone_attachment_fraction"
    )
    attachment_seed = _seed(clone_attachment_seed, location="clone_attachment_seed")
    stack = deepcopy(dict(_mapping(stack_receipt, location="stack_receipt")))
    receipt_fraction = _finite_float(
        stack.get("sample_fraction"),
        location="stack_receipt/sample_fraction",
    )
    if receipt_fraction != fraction:
        raise CheckpointAuthorityError(
            "stack receipt sample_fraction differs from the run request"
        )
    receipt_seed = _seed(
        stack.get("sample_seed"),
        location="stack_receipt/sample_seed",
    )
    if receipt_seed != sample_seed_value:
        raise CheckpointAuthorityError(
            "stack receipt sample_seed differs from the run request"
        )

    static = thaw_json(plan.execution.checkpoint_static_components)
    if not isinstance(static, dict):  # pragma: no cover - frozen root invariant
        raise CheckpointAuthorityError("static checkpoint authority is absent")
    pool_code = deepcopy(
        dict(_mapping(static.get("pool_code"), location="checkpoint/pool_code"))
    )
    resource_semantics = _mapping(
        pool_code.get("late_producer_resource_semantics"),
        location="checkpoint/pool_code/late_producer_resource_semantics",
    )
    pool_code["late_producer_resource_semantics"] = _resolve_clone_attachment(
        resource_semantics,
        fraction=attachment_fraction,
        seed=attachment_seed,
    )
    publication = thaw_json(plan.publication.runtime)
    if not isinstance(publication, dict):  # pragma: no cover - frozen root invariant
        raise CheckpointAuthorityError("publication authority is absent")

    result = {
        key: deepcopy(value) for key, value in static.items() if key != "pool_code"
    }
    try:
        stack_manifest_sha256 = sha256_json(stack)
    except (TypeError, ValueError) as error:
        raise CheckpointAuthorityError(
            "stack receipt must contain canonical JSON values"
        ) from error
    result.update(
        {
            "inputs": _input_pins(input_pins),
            "sampling": {
                "sample_fraction": fraction,
                "fraction_token": _rung_token(publication, fraction),
                "sample_seed": sample_seed_value,
                "stack_manifest_sha256": stack_manifest_sha256,
                "stack_manifest": stack,
            },
            "clone_attachment": {
                "fraction": attachment_fraction,
                "seed": attachment_seed,
            },
            "pool_code": pool_code,
        }
    )
    return result


__all__ = [
    "CheckpointAuthorityError",
    "materialize_stacked_checkpoint_base_identity",
]
