"""Reviewed registers declared as data, for any country (microcosm#1728).

A country keeps reviewed lists in its package: the facts its compile drops and
why, the source columns a package reads under the wrong measure, the source
packages whose period stamps are wrong, the variable each published measure
maps to. A JSON file (``version``, ``country``, ``description``,
``registers``) holds one section for each register:

- ``description``: why the register exists and how the code reads it;
- ``key``: the entry fields that make the key. With one field the key is that
  field's value; with several it is the tuple of their values, in order;
- ``value``: the entry field that is the value, a list of fields (the value is
  the tuple of their values, in order), or null (the register is a set of
  keys);
- ``entries``: the entries, in order. An entry may also carry a ``note``.

A description or a note is a string or a list of lines. :func:`load_register`
reads one section. JSON arrays read back as tuples and JSON objects as dicts,
so a register reads back as the Python literal it replaces, in entry order.
Nothing here holds country data.

**Signed entries** (microcosm#1759). Many reviewed files sign each entry: who
approved it, on what date, and until when it is in force. Every such file
needs the same few things, which the end of this module holds for any
country:

- :func:`read_strict_json` reads JSON and refuses duplicate keys and
  non-finite numbers;
- :func:`canonical_date` reads a date only in its one canonical spelling
  (``YYYY-MM-DD``);
- :class:`ReviewWindow` is an entry's window, ``approved_on`` to
  ``expires_on`` with both days included, and its state on a date;
- :func:`stale_entries` gives the declared entries that name nothing found.

They return values. The country's code writes the error texts and decides
what a lapse does: refuse a run, fail a gate, or report it.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from functools import cache
from importlib.resources import files
from typing import Any

__all__ = [
    "NOTE_FIELD",
    "WINDOW_STATES",
    "DeclaredRegisterError",
    "ReviewWindow",
    "canonical_date",
    "load_register",
    "package_register",
    "package_registers",
    "read_strict_json",
    "stale_entries",
]

#: The optional entry field that explains one entry.
NOTE_FIELD = "note"

_FILE_KEYS = frozenset({"version", "country", "description", "registers"})
_SECTION_KEYS = frozenset({"description", "key", "value", "entries"})


class DeclaredRegisterError(ValueError):
    """A declared register is not valid."""


def _text_or_lines(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value)
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(line, str) for line in value)
    )


def _frozen(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_frozen(item) for item in value)
    if isinstance(value, dict):
        return {key: _frozen(item) for key, item in value.items()}
    return value


def _fields(value: Any, *, context: str, allow_none: bool) -> tuple[str, ...]:
    if value is None and allow_none:
        return ()
    if isinstance(value, str) and value:
        return (value,)
    if (
        isinstance(value, list)
        and value
        and all(isinstance(field, str) and field for field in value)
        and len(set(value)) == len(value)
    ):
        return tuple(value)
    raise DeclaredRegisterError(
        f"{context} must be a field name or a list of distinct field names."
    )


def load_register(section: Mapping[str, Any], *, name: str = "register") -> Any:
    """Read one declared register: a dict in entry order, or a frozenset of keys."""

    if not isinstance(section, Mapping) or set(section) != _SECTION_KEYS:
        raise DeclaredRegisterError(
            f"Register {name!r} must hold exactly the keys {sorted(_SECTION_KEYS)}."
        )
    if not _text_or_lines(section["description"]):
        raise DeclaredRegisterError(
            f"Register {name!r} description must be a string or a list of lines."
        )
    key_fields = _fields(section["key"], context=f"{name} key", allow_none=False)
    raw_value = section["value"]
    value_fields = _fields(raw_value, context=f"{name} value", allow_none=True)
    if set(key_fields) & set(value_fields) or NOTE_FIELD in {
        *key_fields,
        *value_fields,
    }:
        raise DeclaredRegisterError(
            f"Register {name!r} key and value fields overlap or use {NOTE_FIELD!r}."
        )
    entries = section["entries"]
    if not isinstance(entries, list):
        raise DeclaredRegisterError(f"Register {name!r} entries must be a list.")
    fields = {*key_fields, *value_fields}
    result: dict[Any, Any] = {}
    for position, entry in enumerate(entries):
        context = f"register {name!r} entry {position}"
        if not isinstance(entry, Mapping):
            raise DeclaredRegisterError(f"{context} must be an object.")
        unknown = sorted(set(entry) - fields - {NOTE_FIELD})
        missing = sorted(fields - set(entry))
        if unknown or missing:
            raise DeclaredRegisterError(
                f"{context}: unknown fields {unknown}, missing fields {missing}."
            )
        if NOTE_FIELD in entry and not _text_or_lines(entry[NOTE_FIELD]):
            raise DeclaredRegisterError(
                f"{context} note must be a string or a list of lines."
            )
        key = (
            _frozen(entry[key_fields[0]])
            if len(key_fields) == 1
            else tuple(_frozen(entry[field]) for field in key_fields)
        )
        if key in result:
            raise DeclaredRegisterError(f"Register {name!r} repeats the key {key!r}.")
        if raw_value is None:
            result[key] = None
        elif isinstance(raw_value, str):
            result[key] = _frozen(entry[raw_value])
        else:
            result[key] = tuple(_frozen(entry[field]) for field in value_fields)
    if raw_value is None:
        return frozenset(result)
    return result


@cache
def package_registers(package: str, resource: str) -> Mapping[str, Any]:
    """The register sections of a package's JSON resource, read and checked once."""

    # Read strictly: plain json.loads keeps the last of two equal keys, so a
    # second register of one name would replace the first without a word.
    payload = read_strict_json(
        files(package).joinpath(resource).read_text(encoding="utf-8"),
        label=f"{package}/{resource}",
    )
    if not isinstance(payload, dict) or set(payload) != _FILE_KEYS:
        raise DeclaredRegisterError(
            f"{package}/{resource} must hold exactly the keys {sorted(_FILE_KEYS)}."
        )
    if type(payload["version"]) is not int or payload["version"] != 1:
        raise DeclaredRegisterError(f"{package}/{resource} must be version 1.")
    if not isinstance(payload["country"], str) or not payload["country"]:
        raise DeclaredRegisterError(f"{package}/{resource} must name its country.")
    if not _text_or_lines(payload["description"]):
        raise DeclaredRegisterError(f"{package}/{resource} needs a description.")
    registers = payload["registers"]
    if not isinstance(registers, dict) or not registers:
        raise DeclaredRegisterError(f"{package}/{resource} declares no register.")
    return registers


def package_register(package: str, resource: str, name: str) -> Any:
    """Read one register of a package's JSON resource (see :func:`load_register`)."""

    registers = package_registers(package, resource)
    if name not in registers:
        raise DeclaredRegisterError(
            f"{package}/{resource} declares no register {name!r}; it has "
            f"{sorted(registers)}."
        )
    return load_register(registers[name], name=name)


# ---------------------------------------------------------------------------
# Signed entries: strict reads and review windows
# ---------------------------------------------------------------------------

#: The states of a review window on a date, in time order.
WINDOW_STATES = ("premature", "in_force", "expired")


def read_strict_json(text: str, *, label: str) -> Any:
    """Read JSON text; refuse duplicate object keys and non-finite numbers.

    Raises:
        DeclaredRegisterError: ``"{label}: duplicate JSON key(s): [...]."``,
            ``"{label}: non-finite JSON value 'NaN' is invalid."`` or
            ``"{label}: malformed JSON: <reason>."``.
    """

    def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        duplicates: set[str] = set()
        for name, value in pairs:
            if name in result:
                duplicates.add(name)
            result[name] = value
        if duplicates:
            raise DeclaredRegisterError(
                f"{label}: duplicate JSON key(s): {sorted(duplicates)}."
            )
        return result

    def reject_nonfinite(value: str) -> Any:
        raise DeclaredRegisterError(
            f"{label}: non-finite JSON value {value!r} is invalid."
        )

    try:
        return json.loads(
            text, object_pairs_hook=strict_object, parse_constant=reject_nonfinite
        )
    except json.JSONDecodeError as exc:
        raise DeclaredRegisterError(f"{label}: malformed JSON: {exc.msg}.") from exc


def canonical_date(value: Any) -> date | None:
    """``value`` as a date if it is a canonical ISO date string, else None.

    ``date.fromisoformat`` also reads the compact (``20270210``) and week-date
    forms. A signed date has one spelling, because the raw text is often
    sealed into a digest, so only ``YYYY-MM-DD`` counts.
    """

    if not isinstance(value, str):
        return None
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.isoformat() == value else None


@dataclass(frozen=True)
class ReviewWindow:
    """The days an approval is in force: ``approved_on`` to ``expires_on``.

    Both days are in force. A date before ``approved_on`` is premature; a
    date after ``expires_on`` is expired.
    """

    approved_on: date
    expires_on: date

    def premature(self, on: date) -> bool:
        return on < self.approved_on

    def expired(self, on: date) -> bool:
        return on > self.expires_on

    def in_force(self, on: date) -> bool:
        return self.approved_on <= on <= self.expires_on

    def state(self, on: date) -> str:
        """One of :data:`WINDOW_STATES`."""

        if self.premature(on):
            return "premature"
        return "expired" if self.expired(on) else "in_force"

    def days_left(self, on: date) -> int:
        """Days from ``on`` to ``expires_on``; negative once expired."""

        return (self.expires_on - on).days


def stale_entries(declared: Iterable[str], matched: Iterable[str]) -> list[str]:
    """The declared entries that name nothing found, sorted."""

    return sorted(set(declared) - set(matched))
