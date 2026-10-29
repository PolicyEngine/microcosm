"""A declared part is tested against its whole on the compiled values (#1123)."""

from __future__ import annotations

import copy

import pandas as pd
import pytest

from microcosm.build.uk_runtime.cross_grain_declarations import (
    load_uk_cross_grain_declarations,
)
from microcosm.build.uk_runtime.national_reconciliation import (
    UK_SUBSET_MAX_EXCESS,
    assert_uk_subset_relations,
)
from microcosm.calibrate import TargetRegistry, TargetSpec


def _spec(
    target_id: str, cell: str, value: float, geography_id: str = "K02000001"
) -> TargetSpec:
    return TargetSpec(
        name=f"{target_id}@{cell}@2025",
        entity="person",
        measure=target_id,
        value=value,
        period=2025,
        family="test",
        source="test",
        metadata={
            "contract_target_id": target_id,
            "geography_level": "country",
            "geography_id": geography_id,
        },
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


def test_disjoint_parts_are_tested_on_their_sum():
    # No band alone can breach the total; the bands together can.
    registry = TargetRegistry(
        [
            _spec("whole", "a", 100.0),
            _spec("band_0", "x", 40.0),
            _spec("band_1", "x", 35.0),
            _spec("band_2", "x", 30.0),
        ],
        country="uk",
    )
    one_band = _declarations(_subset("band_0", "whole"))
    (receipt,) = assert_uk_subset_relations(registry, declarations=one_band)
    assert receipt["status"] == "holds"
    all_bands = _declarations(
        _subset("band_0", "whole", other_parts=["band_1", "band_2"])
    )
    with pytest.raises(ValueError, match="band_0 \\(national\\) sums to 105.0"):
        assert_uk_subset_relations(registry, declarations=all_bands)
    two_bands = _declarations(_subset("band_0", "whole", other_parts=["band_1"]))
    (receipt,) = assert_uk_subset_relations(registry, declarations=two_bands)
    assert receipt["part_total"] == 75.0
    assert receipt["other_parts"] == ["band_1"]


def test_a_part_is_tested_at_each_geography_it_shares_with_its_whole():
    # The totals fit (90 against 100), but the part is above the whole in N.
    registry = TargetRegistry(
        [
            _spec("whole", "s", 70.0, "S92000003"),
            _spec("whole", "n", 30.0, "N92000002"),
            _spec("part", "s", 50.0, "S92000003"),
            _spec("part", "n", 40.0, "N92000002"),
        ],
        country="uk",
    )
    with pytest.raises(ValueError, match="part at N92000002 is 40.0, above whole"):
        assert_uk_subset_relations(
            registry, declarations=_declarations(_subset("part", "whole"))
        )


def test_an_allowance_wider_than_rounding_is_refused():
    registry = TargetRegistry(
        [_spec("whole", "a", 100.0), _spec("part", "x", 50.0)], country="uk"
    )
    declarations = _declarations(
        _subset("part", "whole", max_excess=UK_SUBSET_MAX_EXCESS + 0.01)
    )
    with pytest.raises(ValueError, match="rounding allowance"):
        assert_uk_subset_relations(registry, declarations=declarations)


def test_a_register_that_binds_neither_side_has_nothing_to_test():
    registry = TargetRegistry([_spec("other", "a", 100.0)], country="uk")
    (receipt,) = assert_uk_subset_relations(
        registry, declarations=_declarations(_subset("part", "whole"))
    )
    assert receipt == {
        "part": "part",
        "whole": "whole",
        "status": "skipped",
        "reason": "neither side is bound",
    }


@pytest.mark.parametrize("bound", ["part", "whole"])
def test_a_relation_with_one_side_unbound_is_refused(bound):
    registry = TargetRegistry([_spec(bound, "a", 100.0)], country="uk")
    with pytest.raises(ValueError, match="cannot be tested on this surface"):
        assert_uk_subset_relations(
            registry, declarations=_declarations(_subset("part", "whole"))
        )


def test_a_missing_other_part_is_refused():
    registry = TargetRegistry(
        [_spec("whole", "a", 100.0), _spec("part", "x", 50.0)], country="uk"
    )
    declarations = _declarations(_subset("part", "whole", other_parts=["absent"]))
    with pytest.raises(ValueError, match="unbound here: \\['absent'\\]"):
        assert_uk_subset_relations(registry, declarations=declarations)


def test_a_local_part_waits_for_a_loader_with_a_local_surface():
    registry = TargetRegistry([_spec("whole", "a", 100.0)], country="uk")
    declarations = _declarations(_subset("area", "whole"))
    (receipt,) = assert_uk_subset_relations(
        registry, declarations=declarations, local_target_ids={"area"}
    )
    assert receipt["status"] == "skipped"
    assert receipt["reason"].startswith("the part is local")
    # With a local surface that does not bind it, the relation is untested.
    with pytest.raises(ValueError, match="cannot be tested on this surface"):
        assert_uk_subset_relations(
            registry,
            local_surface=_surface(("elsewhere", "la", 1.0)),
            declarations=declarations,
            local_target_ids={"area"},
        )


def test_every_committed_subset_names_its_part_and_whole():
    relations = [
        relation
        for relation in load_uk_cross_grain_declarations()["relations"]
        if relation["relation"] == "subset"
    ]
    # England's single adult discount within its households (microcosm#953),
    # its billing-authority cells within theirs (microcosm#1095), and ITL
    # Table 2.2's taxpayers of State Pension age and aged 65 and over within
    # all taxpayers (microcosm#1069).
    assert len(relations) == 21
    for relation in relations:
        assert {relation["part"], relation["whole"]} == set(relation["targets"])
        assert 0.0 <= relation.get("max_excess", 0.0) <= 0.01


def test_the_age_bands_and_the_oldest_rows_are_tested_together():
    (relation,) = [
        relation
        for relation in load_uk_cross_grain_declarations()["relations"]
        if relation.get("part") == "ons.population.age_0_9_by_region"
    ]
    assert relation["whole"] == "ons.population.uk_total"
    assert len(relation["other_parts"]) == 10
    assert {"ons.population.female_90_plus", "ons.population.male_90_plus"} < set(
        relation["other_parts"]
    )


def test_an_other_part_outside_the_contract_is_a_coverage_violation():
    from microcosm.build.uk_runtime.cross_grain_declarations import (
        uk_cross_grain_coverage_violations,
    )
    from microcosm.build.uk_runtime.ledger_targets import _uk_contract_targets

    declarations = copy.deepcopy(dict(load_uk_cross_grain_declarations()))
    relation = next(
        entry for entry in declarations["relations"] if "other_parts" in entry
    )
    relation["other_parts"].append("ons.population.not_a_target")
    violations = uk_cross_grain_coverage_violations(
        _uk_contract_targets(national_only=False), declarations
    )
    assert {
        "kind": "unknown_target",
        "target_id": "ons.population.not_a_target",
        "where": "relations",
    } in violations


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
