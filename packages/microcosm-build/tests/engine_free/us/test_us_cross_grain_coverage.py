"""The US nesting declarations cover every family that two levels measure.

``us/cross_grain_declarations.json`` says, for each US family that two
geography levels measure, what ties the levels (microcosm#1760): a bridge
selector, a derived control, a same-measurement entry, a relation that is not
reconciled, a local route with no higher control, a compile rule that stays
until the feed re-pin, or a known gap. These tests read it:

- the eight rows of the issue each have one entry, and each compile rule
  names code that exists;
- ``us_cross_grain_coverage_violations`` finds a family at a lower level that
  nothing covers, a role that two levels measure with no relation of its own,
  a declaration whose rows are gone, a compile rule whose marker is on no
  row, a subset that is larger than its whole, and a whole that is not one
  positive row;
- on the pinned feed, the compiled registry has no violation. That test runs
  only when ``MICROCOSM_US_CHRONICLE_FACTS`` names the pinned feed, because
  ``main`` has no committed list of the US families and levels.

The check is a test of the declaration. No build calls it and it refuses
nothing.
"""

from __future__ import annotations

import importlib.util
import os
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from microcosm.build.us_runtime import cross_grain_declarations as declarations
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")


def _declared():
    return declarations.load_us_cross_grain_declarations()


def _transitional_entries() -> list:
    transitional = _declared()["until_the_feed_repin"]
    return [*transitional["compile_rules"], *transitional["known_gaps"]]


def test_each_row_of_the_issue_has_one_entry() -> None:
    declared = _declared()
    passes = {
        entry["issue_row"]: section
        for section in ("bridge_selectors", "derived_controls", "same_measurement")
        for entry in declared[section]
    }
    compile_rules = {
        rule["issue_row"]: "compile_rules"
        for rule in declared["until_the_feed_repin"]["compile_rules"]
    }
    assert not set(passes) & set(compile_rules)
    # Rows 1 to 4 stay in the compile until the feed re-pin. Rows 5 to 8 run
    # on the shared operator.
    assert {**passes, **compile_rules} == {
        **dict.fromkeys((1, 2, 3, 4), "compile_rules"),
        5: "bridge_selectors",
        6: "bridge_selectors",
        7: "derived_controls",
        8: "same_measurement",
    }


def test_each_entry_gives_a_reason_and_a_compile_rule_names_its_code() -> None:
    declared = _declared()
    ids = [entry["id"] for entry in _transitional_entries()]
    assert len(ids) == len(set(ids))
    for entry in _transitional_entries():
        assert len(entry["reason"]) > 40
        assert all(re.fullmatch(r"microcosm#\d+", issue) for issue in entry["issues"])
    for rule in declared["until_the_feed_repin"]["compile_rules"]:
        module, _, function = rule["code"].split(",")[0].strip().rpartition(".")
        source = (_TEST_PATHS.package / "src" / Path(*module.split("."))).with_suffix(
            ".py"
        )
        assert f"def {function}(" in source.read_text()
        assert rule["marker"]["metadata_key"]
    for section in ("bridge_selectors", "derived_controls", "same_measurement"):
        for entry in declared[section]:
            assert len(entry["reason"]) > 40
    for entry in (*declared["relations"], *declared["local_routes"].values()):
        assert len(entry["reason"]) > 40


def _spec(family: str, level: str | None, value: float = 1.0, **metadata):
    level_metadata = {} if level is None else {"ledger_geography_level": level}
    return SimpleNamespace(
        family=family, value=value, metadata={**level_metadata, **metadata}
    )


def _covered_specs() -> list:
    """Rows for each declared relation, local route, compile rule and gap."""

    declared = _declared()
    grains = tuple(declared["grain_precedence"])
    specs = [_spec("irs_soi", "country"), _spec("bea", "country")]
    for relation in declared["relations"]:
        for target in relation["targets"]:
            family, role, level = declarations._relation_target(target, grains)
            if target == relation["part"]:
                # Two parts that sum to nine tenths of their whole.
                specs.append(_spec(family, level, 40.0, target_role=role))
                specs.append(_spec(family, level, 50.0, target_role=role))
            else:
                specs.append(_spec(family, level, 100.0, target_role=role))
    for target in declared["local_routes"]:
        specs.append(_spec(*declarations._route_target(target, grains)))
    for entry in _transitional_entries():
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
        # An entry that lists roles is about rows that carry them.
        specs += [
            _spec(entry["family"], level, **metadata, **role)
            for level in entry["levels"]
            for role in (
                [{"target_role": name} for name in entry.get("roles", ())] or [{}]
            )
        ]
    return specs


def test_the_coverage_check_finds_each_kind_of_gap() -> None:
    covered = _covered_specs()
    assert declarations.us_cross_grain_coverage_violations(covered) == []

    # A family at a lower level that nothing covers.
    unrouted = [*covered, _spec("usda_wic", "state")]
    assert declarations.us_cross_grain_coverage_violations(unrouted) == [
        {
            "kind": "family_level_without_route",
            "family": "usda_wic",
            "level": "state",
            "where": "compiled specs",
        }
    ]
    # A relation whose rows are gone, at both of its levels.
    without_tanf = [spec for spec in covered if spec.family != "hhs_acf_tanf"]
    assert declarations.us_cross_grain_coverage_violations(without_tanf) == [
        {
            "kind": "stale_declaration",
            "family": "hhs_acf_tanf",
            "level": level,
            "where": f"hhs_acf_tanf.tanf_total@{level}",
        }
        for level in ("state", "country")
    ]
    # A local route whose rows are gone.
    without_ssa = [spec for spec in covered if spec.family != "ssa"]
    assert declarations.us_cross_grain_coverage_violations(without_ssa) == [
        {
            "kind": "stale_declaration",
            "family": "ssa",
            "level": "state",
            "where": "ssa@state",
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
    top_only = [*covered, _spec("jct", "country"), _spec("x", None)]
    assert declarations.us_cross_grain_coverage_violations(top_only) == []


def test_a_subset_larger_than_its_whole_is_a_violation() -> None:
    covered = _covered_specs()

    def with_part(family: str, role: str, value: float) -> list:
        return [*covered, _spec(family, "state", value, target_role=role)]

    # SNAP declares no excess: 110 of 100 is beyond it.
    assert declarations.us_cross_grain_coverage_violations(
        with_part("usda_snap", "snap_total", 20.0)
    ) == [
        {
            "kind": "subset_exceeded",
            "family": "usda_snap",
            "level": "state",
            "where": "usda_snap.snap_total@state is 1.1 of its whole",
        }
    ]
    # Medicaid declares an excess of one percent.
    within = with_part("cms_medicaid", "medicaid_enrollment", 10.5)
    assert declarations.us_cross_grain_coverage_violations(within) == []
    beyond = with_part("cms_medicaid", "medicaid_enrollment", 11.5)
    (violation,) = declarations.us_cross_grain_coverage_violations(beyond)
    assert violation["kind"] == "subset_exceeded"
    assert violation["where"].startswith("cms_medicaid.medicaid_enrollment@state is ")


def test_a_role_at_two_levels_needs_a_relation_of_its_own() -> None:
    """A family is covered as a whole, so a new role in it must not pass unseen."""

    covered = _covered_specs()
    # A new role of a covered family, at state level and at the top level.
    two_levels = [
        *covered,
        _spec("usda_snap", "state", 5.0, target_role="snap_new_role"),
        _spec("usda_snap", "country", 9.0, target_role="snap_new_role"),
    ]
    assert declarations.us_cross_grain_coverage_violations(two_levels) == [
        {
            "kind": "role_at_two_levels_without_relation",
            "family": "usda_snap",
            "level": "state",
            "where": "usda_snap.snap_new_role",
        }
    ]
    # A known gap can list the role: then the gap is declared, with its issue.
    listed = [
        *covered,
        _spec("irs_soi", "country", 9.0, target_role="soi_fiscal_distribution"),
    ]
    assert declarations.us_cross_grain_coverage_violations(listed) == []
    # The rows of the gap are there, and none carries the role it lists.
    without_listed_role = [
        _spec(spec.family, spec.metadata["ledger_geography_level"])
        if spec.metadata.get("target_role") == "population_age"
        else spec
        for spec in covered
    ]
    assert declarations.us_cross_grain_coverage_violations(without_listed_role) == [
        {
            "kind": "stale_declaration",
            "family": "census_population",
            "level": "state",
            "where": "population_age",
        }
    ]
    # A role at one level only measures nothing twice.
    one_level = [*covered, _spec("usda_snap", "state", 5.0, target_role="snap_other")]
    assert declarations.us_cross_grain_coverage_violations(one_level) == []


def test_a_subset_whole_must_be_one_positive_row() -> None:
    covered = _covered_specs()
    second_whole = [
        *covered,
        _spec("usda_snap", "country", 100.0, target_role="snap_total"),
    ]
    assert declarations.us_cross_grain_coverage_violations(second_whole) == [
        {
            "kind": "subset_whole_not_one_positive_row",
            "family": "usda_snap",
            "level": "country",
            "where": "usda_snap.snap_total@country holds 2 row(s)",
        }
    ]
    negative = [
        _spec(spec.family, "country", -100.0, **spec.metadata)
        if spec.family == "usda_snap"
        and spec.metadata.get("target_role") == "snap_total"
        and spec.metadata["ledger_geography_level"] == "country"
        else spec
        for spec in covered
    ]
    (violation,) = declarations.us_cross_grain_coverage_violations(negative)
    assert violation["kind"] == "subset_whole_not_one_positive_row"
    assert violation["where"] == "usda_snap.snap_total@country holds 1 row(s)"


def test_the_coverage_check_is_a_receipt_and_refuses_nothing() -> None:
    specs = [*_covered_specs(), _spec("usda_wic", "state")]
    receipt = declarations.us_cross_grain_coverage_receipt(specs)
    assert receipt["covered"] is False
    assert [violation["kind"] for violation in receipt["violations"]] == [
        "family_level_without_route"
    ]
    assert receipt["checked_specs"] == len(specs)
    assert receipt["declarations"] == "cross_grain_declarations.json"
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

    # No violation: each family that the registry holds below the national
    # level is covered, no declaration names rows that are gone, each compile
    # rule's marker is on its rows, and no subset is larger than its whole.
    receipt = declarations.us_cross_grain_coverage_receipt(registry.specs)
    assert receipt["violations"] == []
    assert receipt["checked_specs"] == len(registry.specs)
    # The relations read real rows: each side holds at least one.
    declared = _declared()
    grains = tuple(declared["grain_precedence"])
    for relation in declared["relations"]:
        for target in relation["targets"]:
            family, role, level = declarations._relation_target(target, grains)
            assert [
                row
                for row in registry.specs
                if row.family == family
                and row.metadata.get("ledger_geography_level") == level
                and row.metadata.get("target_role") == role
            ], target
