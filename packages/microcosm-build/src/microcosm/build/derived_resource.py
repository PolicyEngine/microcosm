"""Render, digest and check committed resources that a builder derives.

A derived resource is a JSON file a tool writes from pinned inputs and
commits. Three things make it reviewable and reproducible:

- **One rendering** (:func:`render_json`): sorted keys, ASCII, one indent,
  and a table's rows one per line, so a diff shows the rows that moved and a
  table of tens of thousands of rows stays a few megabytes.
- **Digests** (:func:`payload_sha256`, :func:`rows_sha256`): a resource
  records the digest of what it was built from, so files built together
  chain to each other.
- **A staleness check** (:func:`stale_reasons`): a ``--check`` run rebuilds
  the payload and lists what differs from the committed file.

:func:`rows_sha256` moved here from the UK high-value home stock library
(microcosm#1524) and keeps its bytes.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Collection, Iterable, Mapping
from pathlib import Path
from typing import Any

__all__ = [
    "payload_sha256",
    "render_json",
    "rows_sha256",
    "stale_reasons",
]


def _scalar(value: Any) -> str:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("A derived resource cannot hold NaN or an infinity.")
    return json.dumps(value, ensure_ascii=True, allow_nan=False)


def _is_row_table(value: list) -> bool:
    """A list of rows: every element a list of scalars, at least one row."""
    return bool(value) and all(
        isinstance(row, (list, tuple))
        and all(not isinstance(cell, (Mapping, list, tuple)) for cell in row)
        for row in value
    )


def _render(value: Any, level: int, indent: int) -> str:
    pad = " " * (indent * (level + 1))
    end = " " * (indent * level)
    if isinstance(value, Mapping):
        if not value:
            return "{}"
        if any(not isinstance(key, str) for key in value):
            raise ValueError("A derived resource's keys must be text.")
        items = (
            f"{pad}{_scalar(key)}: {_render(value[key], level + 1, indent)}"
            for key in sorted(value)
        )
        return "{\n" + ",\n".join(items) + "\n" + end + "}"
    if isinstance(value, (list, tuple)):
        value = list(value)
        if not value:
            return "[]"
        if _is_row_table(value):
            rows = (
                pad + "[" + ",".join(_scalar(cell) for cell in row) + "]"
                for row in value
            )
            return "[\n" + ",\n".join(rows) + "\n" + end + "]"
        items = (f"{pad}{_render(item, level + 1, indent)}" for item in value)
        return "[\n" + ",\n".join(items) + "\n" + end + "]"
    return _scalar(value)


def render_json(payload: Mapping[str, Any], *, indent: int = 1) -> str:
    """The committed text of a derived resource, ending in a newline.

    It equals ``json.dumps(payload, indent=indent, sort_keys=True,
    ensure_ascii=True)`` except that a list of rows (lists of scalars) puts
    each row on one line. NaN and infinities are refused, so a gap in a
    table must be written as ``null``.
    """
    if not isinstance(payload, Mapping):
        raise ValueError("A derived resource must be a JSON object.")
    return _render(payload, 0, indent) + "\n"


def payload_sha256(payload: Any) -> str:
    """The sha256 of a payload's canonical JSON: sorted keys, no spaces."""
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def rows_sha256(rows: Iterable[Mapping[str, object]]) -> str:
    """Digest of consumer-fact rows, independent of their order."""
    digest = hashlib.sha256()
    for line in sorted(
        json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        for row in rows
    ):
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def stale_reasons(
    path: Path,
    payload: Mapping[str, Any],
    *,
    ignore: Collection[str] = ("built_on",),
    indent: int = 1,
) -> list[str]:
    """Why the committed file at ``path`` is not what the payload renders.

    Top-level keys in ``ignore`` (the build date, say) are left out of the
    comparison. Each differing key gets a reason. When every key agrees,
    the file's bytes must also equal the rendering with the committed
    values of the ignored keys; otherwise the reason is the rendering. An
    empty list means the file is current.
    """
    name = path.name
    if not path.is_file():
        return [f"{name}: missing"]
    text = path.read_text(encoding="utf-8")
    current = json.loads(text)
    rebuilt = json.loads(render_json(payload, indent=indent))
    reasons = []
    for key in sorted((set(current) | set(rebuilt)) - set(ignore)):
        if key not in current:
            reasons.append(f"{name}: {key} is missing from the committed file")
        elif key not in rebuilt:
            reasons.append(f"{name}: {key} is no longer built")
        elif current[key] != rebuilt[key]:
            reasons.append(f"{name}: {key} differs")
    if reasons:
        return reasons
    restored = {
        **{key: value for key, value in rebuilt.items() if key not in ignore},
        **{key: current[key] for key in ignore if key in current},
    }
    if render_json(restored, indent=indent) != text:
        return [f"{name}: the committed text is not the rendering of its content"]
    return []
