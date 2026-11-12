"""Rendering, digests and staleness of derived resources (microcosm#1524)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from microcosm.build.derived_resource import (
    payload_sha256,
    render_json,
    rows_sha256,
    stale_reasons,
)

_SETTINGS = settings(max_examples=80, deadline=None)

_scalars = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(-(10**12), 10**12),
    st.floats(allow_nan=False, allow_infinity=False),
    st.text(max_size=8),
)
_rows = st.lists(st.lists(_scalars, max_size=5), min_size=1, max_size=6)
_documents = st.recursive(
    _scalars,
    lambda children: st.one_of(
        st.lists(children, max_size=4),
        st.dictionaries(st.text(max_size=6), children, max_size=4),
        _rows,
    ),
    max_leaves=25,
)


@_SETTINGS
@given(st.dictionaries(st.text(max_size=6), _documents, max_size=5))
def test_render_json_round_trips(payload) -> None:
    text = render_json(payload)
    assert text.endswith("\n")
    assert json.loads(text) == payload


_plain = st.recursive(
    _scalars,
    lambda children: st.one_of(
        st.lists(children.filter(lambda c: not isinstance(c, list)), max_size=4),
        st.dictionaries(st.text(max_size=6), children, max_size=4),
    ),
    max_leaves=25,
)


@_SETTINGS
@given(st.dictionaries(st.text(max_size=6), _plain, max_size=5))
def test_render_json_is_json_dumps_without_row_tables(payload) -> None:
    expected = json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=True) + "\n"
    assert render_json(payload) == expected


def test_render_json_puts_each_row_on_one_line() -> None:
    text = render_json({"columns": ["a", "b"], "rows": [[1, "x"], [2.5, None]]})
    assert text == (
        '{\n "columns": [\n  "a",\n  "b"\n ],\n'
        ' "rows": [\n  [1,"x"],\n  [2.5,null]\n ]\n}\n'
    )


def test_render_json_refuses_nan() -> None:
    with pytest.raises(ValueError, match="NaN"):
        render_json({"rows": [[float("nan")]]})


def test_rows_sha256_keeps_its_bytes() -> None:
    """Golden digests from the UK high-value home stock library before the move."""
    assert rows_sha256([{"k": 1, "v": "x"}, {"k": 2, "v": "y"}]) == (
        "a440fa0acfad4febb280ec472136c48e5f46a4a7c34bf16ae99ad59537ec8ec6"
    )
    assert (
        rows_sha256(
            [
                {"value": 1.25, "key": "b", "nested": {"z": [1, 2], "a": None}},
                {"value": 3.0, "key": "a"},
            ]
        )
        == "2d4699154964849f4dbafbe648f0300d3a7d213a3804640ab2dab1e972fae396"
    )
    a = [{"k": 1, "v": "x"}, {"k": 2, "v": "y"}]
    assert rows_sha256(a) == rows_sha256(list(reversed(a)))


def test_payload_sha256_ignores_key_order() -> None:
    assert payload_sha256({"a": 1, "b": [1, 2]}) == payload_sha256(
        {"b": [1, 2], "a": 1}
    )
    assert payload_sha256({"a": 1}) != payload_sha256({"a": 2})


def test_stale_reasons_name_what_differs(tmp_path: Path) -> None:
    path = tmp_path / "resource.json"
    payload = {"built_on": "2026-11-12", "kind": "example", "rows": [[1, 2]]}
    assert stale_reasons(path, payload) == ["resource.json: missing"]
    path.write_text(render_json(payload), encoding="utf-8")
    assert stale_reasons(path, payload) == []
    # A new build date alone is not staleness.
    assert stale_reasons(path, {**payload, "built_on": "2027-01-01"}) == []
    assert stale_reasons(path, {**payload, "rows": [[1, 3]]}) == [
        "resource.json: rows differs"
    ]
    assert stale_reasons(path, {**payload, "extra": 1}) == [
        "resource.json: extra is missing from the committed file"
    ]
    assert stale_reasons(path, {"built_on": "x", "kind": "example"}) == [
        "resource.json: rows is no longer built"
    ]
    # Same content, other formatting.
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert stale_reasons(path, payload) == [
        "resource.json: the committed text is not the rendering of its content"
    ]
