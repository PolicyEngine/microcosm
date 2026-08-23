"""Parse the spec engine's deterministic YAML 1.2 subset.

PyYAML intentionally defaults to YAML 1.1 scalar resolution.  This module
gives the compiler a deliberately smaller surface: YAML 1.2 core scalars,
JSON-compatible values, no explicit tags or merge keys, and exactly one
document.  It composes before constructing so duplicate keys and recursive
aliases cannot be hidden by Python ``dict`` construction.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator
from os import PathLike
from pathlib import Path

import yaml
from yaml.composer import ComposerError
from yaml.error import Mark, YAMLError
from yaml.loader import SafeLoader
from yaml.nodes import MappingNode, Node, ScalarNode, SequenceNode
from yaml.tokens import DirectiveToken, TagToken

from .errors import SpecParseError

type JSONScalar = None | bool | int | float | str
type JSONValue = JSONScalar | list[JSONValue] | dict[str, JSONValue]

_BOOL_TAG = "tag:yaml.org,2002:bool"
_FLOAT_TAG = "tag:yaml.org,2002:float"
_INT_TAG = "tag:yaml.org,2002:int"
_MERGE_TAG = "tag:yaml.org,2002:merge"
_NULL_TAG = "tag:yaml.org,2002:null"
_STR_TAG = "tag:yaml.org,2002:str"
_TIMESTAMP_TAG = "tag:yaml.org,2002:timestamp"
_JSON_SCALAR_TAGS = frozenset({_BOOL_TAG, _FLOAT_TAG, _INT_TAG, _NULL_TAG, _STR_TAG})


class _StrictYAML12Loader(SafeLoader):
    """SafeLoader with YAML 1.2 core boolean/number resolution."""


# Resolver tables are mutable class state.  Copy both levels before removing
# YAML 1.1's bool/int/float patterns so importing this module cannot alter
# ``yaml.safe_load`` elsewhere in Microcosm.
_StrictYAML12Loader.yaml_implicit_resolvers = {
    first: list(resolvers)
    for first, resolvers in SafeLoader.yaml_implicit_resolvers.items()
}
for _first, _resolvers in tuple(_StrictYAML12Loader.yaml_implicit_resolvers.items()):
    _StrictYAML12Loader.yaml_implicit_resolvers[_first] = [
        (tag, regexp)
        for tag, regexp in _resolvers
        if tag not in {_BOOL_TAG, _FLOAT_TAG, _INT_TAG}
    ]

_StrictYAML12Loader.add_implicit_resolver(
    _BOOL_TAG,
    re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"),
    list("tTfF"),
)
_StrictYAML12Loader.add_implicit_resolver(
    _INT_TAG,
    re.compile(
        r"^(?:[-+]?0b[0-1_]+|[-+]?0o[0-7_]+|"
        r"[-+]?0x[0-9a-fA-F_]+|[-+]?[0-9][0-9_]*)$"
    ),
    list("-+0123456789"),
)
_StrictYAML12Loader.add_implicit_resolver(
    _FLOAT_TAG,
    re.compile(
        r"^(?:"
        r"[-+]?(?:[0-9][0-9_]*\.[0-9_]*|\.[0-9_]+)"
        r"(?:[eE][-+]?[0-9]+)?|"
        r"[-+]?[0-9][0-9_]*(?:[eE][-+]?[0-9]+)|"
        r"[-+]?\.(?:inf|Inf|INF)|[-+]?\.(?:nan|NaN|NAN)"
        r")$"
    ),
    list("-+0123456789."),
)


def _construct_yaml12_int(loader: SafeLoader, node: ScalarNode) -> int:
    value = loader.construct_scalar(node).replace("_", "")
    sign = -1 if value.startswith("-") else 1
    unsigned = value[1:] if value[:1] in {"+", "-"} else value
    if unsigned.startswith("0b"):
        return sign * int(unsigned[2:], 2)
    if unsigned.startswith("0o"):
        return sign * int(unsigned[2:], 8)
    if unsigned.startswith("0x"):
        return sign * int(unsigned[2:], 16)
    return sign * int(unsigned, 10)


_StrictYAML12Loader.add_constructor(_INT_TAG, _construct_yaml12_int)


def _error(
    message: str,
    *,
    source: str,
    mark: Mark | None = None,
) -> SpecParseError:
    return SpecParseError(
        message,
        source=source,
        line=None if mark is None else mark.line + 1,
        column=None if mark is None else mark.column + 1,
    )


def _marked_yaml_error(exc: YAMLError, *, source: str) -> SpecParseError:
    mark = getattr(exc, "problem_mark", None) or getattr(exc, "context_mark", None)
    if isinstance(exc, ComposerError) and "single document" in str(exc):
        message = "multiple YAML documents are not allowed"
    else:
        problem = getattr(exc, "problem", None)
        message = f"invalid YAML: {problem}" if problem else "invalid YAML"
    return _error(message, source=source, mark=mark)


def _tokens(text: str, *, source: str) -> Iterator[object]:
    try:
        yield from yaml.scan(text, Loader=_StrictYAML12Loader)
    except YAMLError as exc:
        raise _marked_yaml_error(exc, source=source) from exc


def _reject_syntax_extensions(text: str, *, source: str) -> None:
    for token in _tokens(text, source=source):
        if isinstance(token, TagToken):
            raise _error(
                "explicit YAML tags are not allowed",
                source=source,
                mark=token.start_mark,
            )
        if isinstance(token, DirectiveToken):
            if token.name == "TAG":
                raise _error(
                    "YAML tag directives are not allowed",
                    source=source,
                    mark=token.start_mark,
                )
            if token.name == "YAML" and token.value != (1, 2):
                raise _error(
                    "only the YAML 1.2 directive is allowed",
                    source=source,
                    mark=token.start_mark,
                )
            if token.name not in {"TAG", "YAML"}:
                raise _error(
                    "YAML directives other than %YAML 1.2 are not allowed",
                    source=source,
                    mark=token.start_mark,
                )


def _validate_node(
    node: Node,
    *,
    source: str,
    active: set[int],
    validated: set[int],
) -> None:
    identity = id(node)
    if identity in active:
        raise _error(
            "cyclic YAML aliases are not allowed",
            source=source,
            mark=node.start_mark,
        )
    if identity in validated:
        return

    active.add(identity)
    try:
        if isinstance(node, ScalarNode):
            if node.tag == _TIMESTAMP_TAG:
                raise _error(
                    "timestamps and dates are not allowed",
                    source=source,
                    mark=node.start_mark,
                )
            if node.tag not in _JSON_SCALAR_TAGS:
                raise _error(
                    "only JSON-compatible scalar values are allowed",
                    source=source,
                    mark=node.start_mark,
                )
            if node.tag == _FLOAT_TAG:
                normalized = node.value.replace("_", "").lower()
                if normalized.lstrip("+-") in {".inf", ".nan"}:
                    finite = False
                else:
                    try:
                        finite = math.isfinite(float(normalized))
                    except ValueError:
                        raise _error(
                            "invalid YAML 1.2 number",
                            source=source,
                            mark=node.start_mark,
                        ) from None
                if not finite:
                    raise _error(
                        "non-finite numbers are not allowed",
                        source=source,
                        mark=node.start_mark,
                    )
            if node.tag == _INT_TAG:
                normalized = node.value.replace("_", "").lstrip("+-")
                if normalized.startswith(("0b", "0o", "0x")):
                    normalized = normalized[2:]
                if not normalized:
                    raise _error(
                        "invalid YAML 1.2 number",
                        source=source,
                        mark=node.start_mark,
                    )
            return

        if isinstance(node, SequenceNode):
            for item in node.value:
                _validate_node(
                    item,
                    source=source,
                    active=active,
                    validated=validated,
                )
            return

        if isinstance(node, MappingNode):
            keys: dict[str, ScalarNode] = {}
            for key_node, value_node in node.value:
                if key_node.tag == _MERGE_TAG:
                    raise _error(
                        "YAML merge keys are not allowed",
                        source=source,
                        mark=key_node.start_mark,
                    )
                if not isinstance(key_node, ScalarNode) or key_node.tag != _STR_TAG:
                    raise _error(
                        "mapping keys must be strings",
                        source=source,
                        mark=key_node.start_mark,
                    )
                key = key_node.value
                if key in keys:
                    raise _error(
                        f"duplicate mapping key {key!r}",
                        source=source,
                        mark=key_node.start_mark,
                    )
                keys[key] = key_node
                _validate_node(
                    value_node,
                    source=source,
                    active=active,
                    validated=validated,
                )
            return

        raise _error(
            "only JSON-compatible YAML nodes are allowed",
            source=source,
            mark=node.start_mark,
        )
    finally:
        active.remove(identity)
        validated.add(identity)


def _ensure_json_value(value: object, *, source: str) -> JSONValue:
    """Defend the parser boundary if a future PyYAML constructor changes."""

    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _error("non-finite numbers are not allowed", source=source)
        return value
    if isinstance(value, list):
        return [_ensure_json_value(item, source=source) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise _error("mapping keys must be strings", source=source)
        return {
            key: _ensure_json_value(item, source=source) for key, item in value.items()
        }
    raise _error("only JSON-compatible values are allowed", source=source)


def load_yaml12(text: str, *, source: str = "<string>") -> JSONValue:
    """Load one document from the compiler's strict YAML 1.2 subset.

    All authored-input failures are normalized to :class:`SpecParseError`.
    The returned graph contains only JSON-compatible Python values.
    """

    if not isinstance(text, str):
        raise TypeError("YAML input must be text")

    _reject_syntax_extensions(text, source=source)
    loader = _StrictYAML12Loader(text)
    try:
        node = loader.get_single_node()
        if node is None:
            return None
        _validate_node(
            node,
            source=source,
            active=set(),
            validated=set(),
        )
        value = loader.construct_document(node)
    except SpecParseError:
        raise
    except YAMLError as exc:
        raise _marked_yaml_error(exc, source=source) from exc
    finally:
        loader.dispose()

    return _ensure_json_value(value, source=source)


def load_yaml12_file(path: str | PathLike[str]) -> JSONValue:
    """Read and parse a UTF-8 YAML resource, reporting its path in errors."""

    resource = Path(path)
    return load_yaml12(resource.read_text(encoding="utf-8"), source=str(resource))


__all__ = ["JSONScalar", "JSONValue", "load_yaml12", "load_yaml12_file"]
