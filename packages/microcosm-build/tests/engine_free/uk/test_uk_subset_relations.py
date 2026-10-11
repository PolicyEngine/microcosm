"""A declared part is tested against its whole on the compiled values (#1123)."""

from __future__ import annotations

import copy

import pandas as pd
import pytest

from microcosm.build.uk_runtime.cross_grain_declarations import (
    load_uk_cross_grain_declarations,
)
from microcosm.build.uk_runtime.national_reconciliation import (
    assert_uk_subset_relations,
)
from microcosm.calibrate import TargetRegistry, TargetSpec


def _spec(target_id: str, cell: str, value: float) -> TargetSpec:
    return TargetSpec(
        name=f"{target_id}@{cell}@2025",
        entity="person",
        measure=target_id,
        value=value,
        period=2025,
        family="test",
        source="test",
        metadata={"contract_target_id": target_id},
    )


def _declarations(*relations: dict) -> dict:
    return {"relations": list(relations)}


def _subset(part: str, whole: str, **fields) -> dict:
    return {
        "targets": sorted([part, whole]),
        "relation": "subset",
        "part": part,
        "whole": whole,
        "reason": "test",
        **fields,
    }


def _surface(*rows: tuple[str, str, float]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["contract_target_id", "area_type", "value"])


def test_a_whole_sums_every_cell_of_its_target():
    registry = TargetRegistry(
        [
            _spec("whole", "a", 60.0),
            _spec("whole", "b", 40.0),
            _spec("part", "x", 99.0),
        ],
        country="uk",
    )
    (receipt,) = assert_uk_subset_relations(
        registry, declarations=_declarations(_subset("part", "whole"))
    )
    assert receipt["status"] == "holds"
    assert (receipt["part_total"], receipt["whole_total"]) == (99.0, 100.0)


def test_a_part_above_its_whole_is_refused():
    registry = TargetRegistry(
        [_spec("whole", "a", 100.0), _spec("part", "x", 101.0)], country="uk"
    )
    with pytest.raises(ValueError, match="part \\(national\\) sums to 101.0"):
        assert_uk_subset_relations(
            registry, declarations=_declarations(_subset("part", "whole"))
        )


def test_a_declared_excess_allows_two_tables_of_one_total_to_round_apart():
    registry = TargetRegistry(
        [_spec("whole", "a", 100.0), _spec("part", "x", 100.5)], country="uk"
    )
    declarations = _declarations(_subset("part", "whole", max_excess=0.01))
    (receipt,) = assert_uk_subset_relations(registry, declarations=declarations)
    assert receipt["status"] == "holds"
    declarations = _declarations(_subset("part", "whole", max_excess=0.001))
    with pytest.raises(ValueError, match="breached"):
        assert_uk_subset_relations(registry, declarations=declarations)


def test_a_local_part_is_tested_at_each_grain_on_its_own():
    registry = TargetRegistry([_spec("whole", "a", 100.0)], country="uk")
    declarations = _declarations(_subset("area", "whole"))
    fits = _surface(("area", "constituency", 60.0), ("area", "la", 59.0))
    rows = assert_uk_subset_relations(
        registry, local_surface=fits, declarations=declarations
    )
    assert [(row["grain"], row["status"]) for row in rows] == [
        ("constituency", "holds"),
        ("la", "holds"),
    ]
    # Each grain tiles the country once: neither may exceed the whole, and
    # the two are never added together.
    over = _surface(("area", "constituency", 60.0), ("area", "la", 101.0))
    with pytest.raises(ValueError, match="area \\(la\\)"):
        assert_uk_subset_relations(
            registry, local_surface=over, declarations=declarations
        )


def test_an_unbound_side_has_nothing_to_test():
    registry = TargetRegistry([_spec("whole", "a", 100.0)], country="uk")
    (receipt,) = assert_uk_subset_relations(
        registry, declarations=_declarations(_subset("part", "whole"))
    )
    assert receipt == {"part": "part", "whole": "whole", "status": "skipped"}


def test_every_committed_subset_names_its_part_and_whole():
    relations = [
        relation
        for relation in load_uk_cross_grain_declarations()["relations"]
        if relation["relation"] == "subset"
    ]
    assert len(relations) == 17
    for relation in relations:
        assert {relation["part"], relation["whole"]} == set(relation["targets"])
        assert 0.0 <= relation.get("max_excess", 0.0) <= 0.01


def test_a_subset_without_its_direction_is_a_malformed_relation():
    from microcosm.build.uk_runtime.cross_grain_declarations import (
        uk_cross_grain_coverage_violations,
    )
    from microcosm.build.uk_runtime.ledger_targets import _uk_contract_targets

    declarations = copy.deepcopy(dict(load_uk_cross_grain_declarations()))
    relation = next(
        entry for entry in declarations["relations"] if entry["relation"] == "subset"
    )
    del relation["whole"]
    violations = uk_cross_grain_coverage_violations(
        _uk_contract_targets(national_only=False), declarations
    )
    assert "malformed_relation" in {violation["kind"] for violation in violations}
