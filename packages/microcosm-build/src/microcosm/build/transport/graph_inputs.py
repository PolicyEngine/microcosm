"""Decoding a transport kernel's declared inputs: params, tables, artifacts.

Graph parameters are pure data (``bool``, ``int``, ``float``, ``str``,
``None`` or tuples of those; ``microcosm.graph.decl``), so a structured spec
value such as a target-reference document or a gate manifest travels as one
canonical-JSON string. Requiring the canonical spelling means two equal
documents always give the same node key, and a value can never re-key a node
through whitespace or key order alone.

Spec data enters a node only as the resolved values it reads plus the SHA-256
of the resource they came from. No kernel here binds a whole country spec's
fingerprint: that would re-key every spec-reading node on any unrelated edit.

:func:`context_frame` rebuilds a :class:`~microcosm.frame.Frame` from the
tables the executor projected for a node. It needs the person table, because
the executor only hands a node the entities it slices or owns, and the person
table is what carries every group membership column.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

from microcosm.frame import EntitySchema, Frame
from microcosm.graph import ROWS_ALL, KernelContext
from microcosm.graph.canonical import canonical_json

from .artifact_types import KERNEL_OUTPUTS

__all__ = [
    "bool_param",
    "canonical_document_param",
    "context_frame",
    "entities_param",
    "optional_bool_param",
    "require_outputs",
    "require_params",
    "sha256_param",
    "sha256_text",
    "single_source",
    "string_param",
]

_HEX = frozenset("0123456789abcdef")


def sha256_text(payload: bytes) -> str:
    """Hex SHA-256 of ``payload``."""

    return hashlib.sha256(payload).hexdigest()


def require_params(
    context: KernelContext,
    ref: str,
    *,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> None:
    """Refuse unknown or missing parameters, naming the kernel."""

    params = set(context.params)
    unknown = sorted(params - required - optional)
    if unknown:
        raise ValueError(f"{ref} received unknown parameter(s): {unknown}.")
    missing = sorted(required - params)
    if missing:
        raise ValueError(f"{ref} is missing required parameter(s): {missing}.")


def require_outputs(context: KernelContext, ref: str) -> None:
    """Refuse a node whose typed outputs differ from :data:`KERNEL_OUTPUTS`."""

    expected = dict(KERNEL_OUTPUTS[ref])
    declared = {output.name: output.type for output in context.node.artifact_outputs}
    if declared != expected:
        raise ValueError(
            f"{ref} nodes declare exactly the typed outputs {expected!r}; this "
            f"node declares {declared!r}."
        )


def string_param(context: KernelContext, ref: str, name: str) -> str:
    """A required non-empty string parameter."""

    value = context.params.get(name)
    if not isinstance(value, str) or not value:
        raise TypeError(f"{ref} parameter {name!r} must be a non-empty string.")
    return value


def bool_param(context: KernelContext, ref: str, name: str) -> bool:
    """A required boolean parameter."""

    value = context.params.get(name)
    if not isinstance(value, bool):
        raise TypeError(f"{ref} parameter {name!r} must be a boolean.")
    return value


def optional_bool_param(context: KernelContext, ref: str, name: str) -> bool:
    """A boolean parameter that is ``False`` when the node omits it."""

    if name not in context.params:
        return False
    return bool_param(context, ref, name)


def sha256_param(context: KernelContext, ref: str, name: str) -> str:
    """A lowercase hex SHA-256 parameter (a spec resource's identity)."""

    value = context.params.get(name)
    if not isinstance(value, str) or len(value) != 64 or not set(value) <= _HEX:
        raise ValueError(
            f"{ref} parameter {name!r} must be a lowercase hex SHA-256 digest."
        )
    return value


def canonical_document_param(
    context: KernelContext, ref: str, name: str
) -> Mapping[str, object]:
    """Decode a JSON-object parameter that must be spelled canonically."""

    raw = context.params.get(name)
    if not isinstance(raw, str):
        raise TypeError(f"{ref} parameter {name!r} must be a canonical JSON string.")
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"{ref} parameter {name!r} is not JSON: {error}.") from error
    if not isinstance(document, dict):
        raise ValueError(f"{ref} parameter {name!r} must encode a JSON object.")
    if canonical_json(document).decode("utf-8") != raw:
        raise ValueError(
            f"{ref} parameter {name!r} must be canonical JSON (sorted keys, no "
            "whitespace), so equal documents give equal node keys."
        )
    return document


def entities_param(context: KernelContext, ref: str, name: str) -> tuple[str, ...]:
    """The person entity followed by the group entities a node rebuilds."""

    value = context.params.get(name)
    if (
        not isinstance(value, tuple)
        or len(value) < 2
        or any(not isinstance(item, str) or not item for item in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError(
            f"{ref} parameter {name!r} must be a tuple of distinct entity names, "
            "the person entity first and at least one group entity after it."
        )
    return value


def single_source(context: KernelContext, ref: str) -> str:
    """The one source a node declares; the kernel reads no other."""

    if len(context.node.sources) != 1:
        raise ValueError(
            f"{ref} reads exactly one declared source, got {context.node.sources!r}."
        )
    name = context.node.sources[0]
    if name not in context.sources:
        raise ValueError(f"{ref} declared source {name!r} has no verified path.")
    return name


def context_frame(
    context: KernelContext,
    ref: str,
    *,
    entities: tuple[str, ...],
    weight_entity: str,
) -> Frame:
    """Rebuild the declared entities as a ``Frame`` with the context's weights.

    Columns keep the population version's own order
    (``KernelContext.frame_column_order``); the weights are the node's
    effective typed weights on ``weight_entity``. Membership columns of groups
    outside ``entities`` stay on the person table as ordinary columns. Every
    slice must cover all rows: a row-masked slice is refused.
    """

    if weight_entity not in entities:
        raise ValueError(
            f"{ref} weight entity {weight_entity!r} is not one of {entities!r}."
        )
    masked = sorted(
        f"{item.entity}[{item.rows}]"
        for item in context.node.inputs
        if item.rows != ROWS_ALL
    )
    if masked:
        # A masked slice projects a subset of rows; a frame rebuilt from it
        # would silently aggregate, calibrate or export that subset only.
        raise ValueError(
            f"{ref} rebuilds whole entity tables and refuses row-masked "
            f"slices {masked}."
        )
    missing = [entity for entity in entities if entity not in context.tables]
    if missing:
        raise ValueError(
            f"{ref} needs the {missing!r} table(s): declare one data-column "
            "slice on each so the executor projects its rows."
        )
    if weight_entity not in context.weights:
        raise ValueError(
            f"{ref} has no effective weights for {weight_entity!r} in its context."
        )
    person, *groups = entities
    tables = {
        entity: context.tables[entity]
        .loc[
            :,
            list(
                context.frame_column_order.get(
                    entity, tuple(context.tables[entity].columns)
                )
            ),
        ]
        .copy(deep=True)
        for entity in entities
    }
    return Frame(
        tables,
        EntitySchema(person_entity=person, group_entities=tuple(groups)),
        {weight_entity: context.weights[weight_entity]},
        context.strata.copy(deep=True),
        mass_log=context.frame_mass_log,
        metadata=dict(context.frame_metadata),
    )
