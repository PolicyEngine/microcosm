"""The US nesting inventory covers every family that two levels measure.

``us/cross_grain_declarations.json`` lists, under ``inventory``, each US
family that two geography levels measure, with the route that ties the levels
today or the reason none does (microcosm#1760). It records what the code
does; it decides no rule. These tests read it:

- each entry is well formed, the eight rows of the issue each have one entry,
  and each mechanism on the shared operator has one;
- ``us_cross_grain_coverage_violations`` finds a family at a lower level with
  no entry, an entry whose rows are gone, and a compile rule whose marker is
  on no row;
- on the pinned feed, the compiled registry has no violation. That test runs
  only when ``MICROCOSM_US_CHRONICLE_FACTS`` names the pinned feed, because
  ``main`` has no committed list of the US families and levels.

The check is a test of the declaration. No build calls it and it refuses
nothing.
"""

from __future__ import annotations

import copy
import importlib.util
import os
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from microcosm.build.us_runtime import cross_grain_declarations as declarations
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")


def _entries():
    return declarations.us_cross_grain_inventory()


def test_each_row_of_the_issue_has_one_entry() -> None:
    rows = [entry["issue_row"] for entry in _entries() if "issue_row" in entry]
    assert sorted(rows) == [1, 2, 3, 4, 5, 6, 7, 8]
    routes = {
        entry["issue_row"]: entry["route"]
        for entry in _entries()
        if "issue_row" in entry
    }
    # Rows 1 to 4 stay in the compile; rows 5 to 8 run on the shared operator.
    assert routes == {
        **dict.fromkeys((1, 2, 3, 4), "compile_rule"),
        **dict.fromkeys((5, 6, 7, 8), "mechanism"),
    }


def test_each_mechanism_has_one_entry_that_matches_it() -> None:
    mechanisms = declarations.load_us_cross_grain_declarations()["mechanisms"]
    named = [
        entry["mechanism"] for entry in _entries() if entry["route"] == "mechanism"
    ]
    assert sorted(named) == sorted(mechanisms)
    for entry in _entries():
        if entry["route"] != "mechanism":
            continue
        mechanism = mechanisms[entry["mechanism"]]
        assert entry["issue_row"] == mechanism["issue_row"]
        assert entry["family"] == mechanism["family"]
        assert entry["levels"] == [mechanism["lower_grain"]]
        assert entry["higher_level"] == mechanism["higher_grain"]


def test_each_entry_gives_a_route_and_what_the_route_needs() -> None:
    declared = declarations.load_us_cross_grain_declarations()
    routes = declared["inventory"]["routes"]
    ids = [entry["id"] for entry in _entries()]
    assert len(ids) == len(set(ids))
    for entry in _entries():
        assert entry["route"] in routes
        assert entry["issues"] and all(
            re.fullmatch(r"microcosm#\d+", issue) for issue in entry["issues"]
        )
        if entry["route"] != "mechanism":
            # A route with no mechanism says why in words.
            assert len(entry["reason"]) > 40
        if entry["route"] == "compile_rule":
            module, _, function = entry["code"].split(",")[0].strip().rpartition(".")
            source = (
                _TEST_PATHS.package / "src" / Path(*module.split("."))
            ).with_suffix(".py")
            assert f"def {function}(" in source.read_text()
            assert entry["marker"]["metadata_key"]
            assert entry["compiled"] is True
    # Every route the file lists is in use, so none is a dead name.
    assert {entry["route"] for entry in _entries()} == set(routes)


def _tampered(change) -> dict:
    payload = copy.deepcopy(dict(declarations.load_us_cross_grain_declarations()))
    change(payload["inventory"]["entries"])
    return payload


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda entries: entries[0].update(route="merged"), "unknown route"),
        (lambda entries: entries[0].update(reason=" "), "no reason"),
        (lambda entries: entries[0].pop("marker"), "the marker it writes"),
        (lambda entries: entries[4].update(family="usda_snap"), "does not match"),
        (lambda entries: entries[1].update(id=entries[0]["id"]), "blank or repeated"),
        (lambda entries: entries[0].update(levels=["tract"]), "must name a family"),
        (
            lambda entries: entries[0].update(higher_level="county"),
            "must name a family",
        ),
        (lambda entries: entries[0].update(issues=[]), "name an issue"),
    ],
)
def test_a_malformed_entry_is_refused_when_the_file_loads(change, message) -> None:
    with pytest.raises(ValueError, match=message):
        declarations._validate_inventory(_tampered(change))


def _spec(family: str, level: str, **metadata):
    return SimpleNamespace(
        family=family, metadata={"ledger_geography_level": level, **metadata}
    )


def _covered_specs() -> list:
    """One spec for each compiled entry and level, with the rule's marker."""

    specs = [_spec("irs_soi", "country"), _spec("bea", "country")]
    for entry in _entries():
        if not entry["compiled"]:
            continue
        marker = entry.get("marker") or {}
        metadata = (
            {
                marker["metadata_key"]: marker.get(
                    "value", marker.get("value_prefix", "") + "x"
                )
            }
            if marker
            else {}
        )
        specs += [
            _spec(entry["family"], level, **metadata) for level in entry["levels"]
        ]
    return specs


def test_the_coverage_check_finds_each_kind_of_gap() -> None:
    covered = _covered_specs()
    assert declarations.us_cross_grain_coverage_violations(covered) == []

    # A family at a lower level with no entry.
    unrouted = [*covered, _spec("usda_wic", "state")]
    assert declarations.us_cross_grain_coverage_violations(unrouted) == [
        {
            "kind": "family_level_without_route",
            "family": "usda_wic",
            "level": "state",
            "where": "compiled specs",
        }
    ]
    # An entry whose compiled rows are gone.
    without_tanf = [spec for spec in covered if spec.family != "hhs_acf_tanf"]
    assert declarations.us_cross_grain_coverage_violations(without_tanf) == [
        {
            "kind": "stale_inventory_entry",
            "family": "hhs_acf_tanf",
            "level": "state",
            "where": "hhs_acf_tanf_state_rows_and_national_row",
        }
    ]
    # A compile rule that marks no row of its declared level.
    unmarked = [spec for spec in covered if "state_agi_band_share" not in spec.metadata]
    assert declarations.us_cross_grain_coverage_violations(unmarked) == [
        {
            "kind": "compile_rule_marker_absent",
            "family": "irs_soi",
            "level": "state",
            "where": "soi_state_agi_bands_share_of_state_total",
        }
    ]
    # Rows at the top level, and rows with no level, nest in nothing.
    top_only = [
        *covered,
        _spec("jct", "country"),
        SimpleNamespace(family="x", metadata={}),
    ]
    assert declarations.us_cross_grain_coverage_violations(top_only) == []


def test_the_coverage_check_is_a_receipt_and_refuses_nothing() -> None:
    receipt = declarations.us_cross_grain_coverage_receipt(
        [*_covered_specs(), _spec("usda_wic", "state")]
    )
    assert receipt["covered"] is False
    assert [violation["kind"] for violation in receipt["violations"]] == [
        "family_level_without_route"
    ]
    assert receipt["inventory_entries"] == len(_entries())
    assert declarations.us_cross_grain_coverage_receipt(_covered_specs())["covered"]


def test_no_build_code_calls_the_coverage_check() -> None:
    """It is a test of the declaration, not a gate: nothing in a build runs it."""

    repository = _TEST_PATHS.repository
    callers = [
        path.relative_to(repository).as_posix()
        for root in (repository / "tools", repository / "packages")
        for path in root.rglob("*.py")
        if "/tests/" not in path.as_posix()
        and "us_cross_grain_coverage" in path.read_text(encoding="utf-8")
    ]
    assert callers == [
        "packages/microcosm-build/src/microcosm/build/us_runtime/"
        "cross_grain_declarations.py"
    ]


def test_the_pinned_feeds_compiled_registry_is_covered() -> None:
    feed = os.environ.get("MICROCOSM_US_CHRONICLE_FACTS", "")
    if not feed or not Path(feed).exists():
        pytest.skip("set MICROCOSM_US_CHRONICLE_FACTS to the pinned 41b2fb2 feed")
    path = _TEST_PATHS.repository / "tools" / "build_us_acs_local_release.py"
    spec = importlib.util.spec_from_file_location(
        "build_us_acs_local_release_nesting_coverage", path
    )
    tool = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(tool)
    registry, _substitutions = tool._compiled_fiscal_registry(feed)

    receipt = declarations.us_cross_grain_coverage_receipt(registry.specs)
    assert receipt["violations"] == []
    assert receipt["checked_specs"] == len(registry.specs)
    # The families the registry holds below the national level are the ones
    # the inventory's compiled entries name: no more and no fewer.
    compiled = {
        (spec.family, spec.metadata["ledger_geography_level"])
        for spec in registry.specs
        if spec.metadata.get("ledger_geography_level") not in (None, "country")
    }
    declared = {
        (entry["family"], level)
        for entry in _entries()
        if entry["compiled"]
        for level in entry["levels"]
    }
    assert compiled == declared
