"""Ordered entity-axis evidence for a declared same-kind weight update.

A :class:`~microcosm.graph.decl.WeightUpdate` kernel returns positional
weight values. Positions alone are not evidence: the same vector is correct
against one row order and silently wrong against another, and nothing in a
replayed receipt would show the difference. A kernel therefore binds the
ordered entity axis its values were computed against, and the executor
recomputes that binding from the incumbent axis it is about to apply them
to — on cold execution and on every replay of the cached receipt
(amendment 25).

The binding is a digest, not the ids: an axis of millions of rows does not
belong in a manifest, and the executor only ever needs to answer whether
the axis it holds is the axis the kernel used.
"""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral

from .canonical import canonical_json, sha256_domain

__all__ = ["WEIGHT_UPDATE_AXIS_SCHEMA", "weight_update_receipt"]

#: The tagged schema of the axis binding. The executor authors the same
#: token when it recomputes, so a receipt written under another schema is a
#: mismatch rather than an unchecked pass.
WEIGHT_UPDATE_AXIS_SCHEMA = "microcosm.graph.weight-update-axis.v1"


def weight_update_receipt(entity_ids: Sequence[int | str]) -> dict[str, object]:
    """Bind positional replacement weights to a unique, ordered entity axis.

    Place the result under ``KernelResult.receipt['weight_update']``.

    Args:
        entity_ids: The entity ids, in the order the returned weight values
            are positional against. Integers or non-empty strings, unique.
            ``bool`` is refused: it is an ``Integral`` in Python, and an
            axis of ``True``/``False`` is a mistake, not an id space.

    Returns:
        The tagged binding: its schema, the row count, and a domain-
        separated SHA-256 over the canonical JSON of the ordered ids.

    Raises:
        ValueError: An id is neither an integer nor a non-empty string, or
            the ids repeat.
    """

    ids: list[int | str] = []
    for value in entity_ids:
        if isinstance(value, Integral) and not isinstance(value, bool):
            ids.append(int(value))
        elif isinstance(value, str) and value:
            ids.append(value)
        else:
            raise ValueError(
                "Weight update ids must be integers or non-empty strings; got "
                f"{value!r}."
            )
    if len(set(ids)) != len(ids):
        raise ValueError("Weight update ids must be unique.")
    return {
        "schema": WEIGHT_UPDATE_AXIS_SCHEMA,
        "count": len(ids),
        "entity_ids_sha256": sha256_domain("weight-update-axis", canonical_json(ids)),
    }
