"""The US county facts pin and its loader (microcosm#922).

The county facts sit in their own pin beside the main US feed pin, so the main
feed, its scope and the parity resources do not move when county facts do.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from microcosm.build.us_runtime.chronicle_feed import (
    US_CHRONICLE_COUNTY_FACTS_SCOPE_RESOURCE,
    US_CHRONICLE_FEED_SCOPE_RESOURCE,
    load_us_chronicle_county_facts_pin,
    load_us_chronicle_feed,
    us_chronicle_county_facts_scope_sha256,
)
from microcosm.build.us_runtime.county_facts import (
    COUNTY_FACT_FIELDS,
    load_us_county_facts,
    us_county_facts_scope_pairs,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
_US_PACKAGE = _REPOSITORY_ROOT / "packages/microcosm-build/src/microcosm/build/us"


def _builder():
    path = _REPOSITORY_ROOT / "tools" / "build_us_chronicle_feed.py"
    spec = importlib.util.spec_from_file_location("build_us_chronicle_feed", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# --- the packaged pin and scope ---------------------------------------------


def test_the_county_pin_loads_and_names_its_scope() -> None:
    pin = load_us_chronicle_county_facts_pin()
    assert pin.scope == "us_county_facts"
    assert pin.scope_sha256 == us_chronicle_county_facts_scope_sha256()
    assert pin.is_bare_feed


def test_the_county_pin_is_not_the_main_pin() -> None:
    county = load_us_chronicle_county_facts_pin()
    main = load_us_chronicle_feed()
    assert county.facts_sha256 != main.facts_sha256
    assert county.scope_sha256 != main.scope_sha256


def test_the_feed_builder_accepts_the_county_scope() -> None:
    scope = _builder().load_scope(
        _US_PACKAGE / US_CHRONICLE_COUNTY_FACTS_SCOPE_RESOURCE
    )
    pin = load_us_chronicle_county_facts_pin()
    assert scope["commit"] == pin.source_commit
    assert len(scope["pairs"]) == scope["raw"]["pair_count"]


def test_county_and_main_scopes_share_no_pair() -> None:
    builder = _builder()
    county = builder.load_scope(_US_PACKAGE / US_CHRONICLE_COUNTY_FACTS_SCOPE_RESOURCE)
    main = builder.load_scope(_US_PACKAGE / US_CHRONICLE_FEED_SCOPE_RESOURCE)
    assert set(county["pairs"]).isdisjoint(main["pairs"])


def test_the_loader_reads_the_county_scope_pairs() -> None:
    builder = _builder()
    county = builder.load_scope(_US_PACKAGE / US_CHRONICLE_COUNTY_FACTS_SCOPE_RESOURCE)
    assert us_county_facts_scope_pairs() == frozenset(county["pairs"])


# --- the loader --------------------------------------------------------------

_PAIRS = sorted(us_county_facts_scope_pairs())


def _row(index: int, *, state: str = "01", pair=None, level="county", contract=None):
    record_set_id, period_type, period_value = pair or _PAIRS[index % len(_PAIRS)]
    return {
        "aggregate_fact_key": f"key-{index:06d}",
        "schema_version": contract or "chronicle.consumer_fact.v5",
        "geography": {
            "id": f"0500000US{state}{index % 1000:03d}",
            "level": level,
            "name": f"County {index}",
            "vintage": "2024",
        },
        "observed_measure": {
            "source_name": "census_acs",
            "source_measure_id": "X_001E",
            "unit": "count",
        },
        "dimensions": {"statistic": "estimate"},
        "period": {"type": period_type, "value": int(period_value)},
        "value": index,
        "layout": {"record_set_id": record_set_id, "record_set_spec_id": "spec"},
        "lineage": {"heavy": "x" * 50},
        "source": {"heavy": "y" * 50},
    }


def _write(path: Path, rows, *, separators=(",", ":")) -> bytes:
    payload = b"".join(
        json.dumps(row, separators=separators, sort_keys=True).encode() + b"\n"
        for row in rows
    )
    path.write_bytes(payload)
    return payload


def _pin_for(payload: bytes, rows: int):
    pin = load_us_chronicle_county_facts_pin()
    return dataclasses.replace(
        pin,
        facts_sha256=hashlib.sha256(payload).hexdigest(),
        fact_row_count=rows,
        consumer_fact_schema_versions=("chronicle.consumer_fact.v5",),
    )


_STATES = st.sampled_from(["01", "06", "09", "15", "48", "72"])


@settings(max_examples=40, deadline=None)
@given(
    states=st.lists(_STATES, min_size=1, max_size=30),
    spaced=st.booleans(),
)
def test_the_loader_keeps_projected_mainland_rows_and_counts_puerto_rico(
    tmp_path_factory, states, spaced
) -> None:
    path = tmp_path_factory.mktemp("cf") / "facts.jsonl"
    rows = [_row(i, state=s) for i, s in enumerate(states)]
    payload = _write(path, rows, separators=(", ", ": ") if spaced else (",", ":"))
    facts = load_us_county_facts(path, pin=_pin_for(payload, len(rows)))
    expected = [r for r in rows if r["geography"]["id"][9:11] != "72"]
    assert [r["aggregate_fact_key"] for r in facts.rows] == [
        r["aggregate_fact_key"] for r in expected
    ]
    assert all(set(r) == set(COUNTY_FACT_FIELDS) & set(rows[0]) for r in facts.rows)
    assert all("lineage" not in r and "source" not in r for r in facts.rows)
    assert facts.dropped == (
        {"state_72": len(rows) - len(expected)} if len(rows) != len(expected) else {}
    )
    assert facts.facts_sha256 == hashlib.sha256(payload).hexdigest()
    assert facts.row_count == len(rows)


@settings(max_examples=40, deadline=None)
@given(
    count=st.integers(min_value=1, max_value=20),
    data=st.data(),
)
def test_any_one_byte_change_is_refused(tmp_path_factory, count, data) -> None:
    path = tmp_path_factory.mktemp("cf") / "facts.jsonl"
    rows = [_row(i) for i in range(count)]
    payload = _write(path, rows)
    pin = _pin_for(payload, count)
    position = data.draw(st.integers(min_value=0, max_value=len(payload) - 1))
    mutated = bytearray(payload)
    # A digit for a digit keeps the line valid JSON, so the digest must refuse it.
    mutated[position] = ord("7") if mutated[position] != ord("7") else ord("3")
    path.write_bytes(bytes(mutated))
    with pytest.raises(ValueError):
        load_us_county_facts(path, pin=pin)


@pytest.mark.parametrize(
    ("bad", "match"),
    [
        (lambda: _row(9, level="state"), "county rows"),
        (
            lambda: _row(9, pair=("census_acs.not_in_scope", "calendar_year", "2024")),
            "outside the county scope",
        ),
        (
            lambda: _row(9, contract="chronicle.consumer_fact.v3"),
            "unpinned row contract",
        ),
    ],
    ids=["non-county", "out-of-scope", "unpinned-contract"],
)
def test_a_bad_row_between_good_ones_is_refused(tmp_path, bad, match) -> None:
    rows = [_row(0), bad(), _row(1)]
    path = tmp_path / "facts.jsonl"
    payload = _write(path, rows)
    with pytest.raises(ValueError, match=match):
        load_us_county_facts(path, pin=_pin_for(payload, len(rows)))


def test_a_row_count_other_than_the_pins_is_refused(tmp_path) -> None:
    rows = [_row(0), _row(1)]
    path = tmp_path / "facts.jsonl"
    payload = _write(path, rows)
    with pytest.raises(ValueError, match="rows; the county pin names 3"):
        load_us_county_facts(path, pin=_pin_for(payload, 3))
