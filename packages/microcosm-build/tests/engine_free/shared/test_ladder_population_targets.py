"""Shared ladder population targets (microcosm.build.ladder_population_targets).

The tests use a made-up country of regions (code ``R``) and districts (code
``RDD``). They check the reader, the code fields, the rows (dropped cells, a
household of size zero), and the specs with their labels. No country code is
read (microcosm#1728).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.ladder_population_targets import (
    CodeField,
    LadderPopulationError,
    load_ladder_population,
    population_rows,
    population_target_specs,
)


def _section() -> dict:
    return {
        "name_prefix": "pop_",
        "record": {"source": "test_ladder", "family": "test_population_ladder"},
        "spec": {
            "entity": "household",
            "family": "test_population",
            "source": "Test ladder",
            "provider": {"id": "census", "label": "Census"},
            "category": {"id": "census.people", "label": "People"},
            "target_label": "{label} people",
        },
        "levels": [
            {
                "geography": "region",
                "level": "region",
                "household_column": "region_code",
                "code_digits": 1,
                "geography_id_prefix": "R:",
                "label": "region_name",
                "record_codes": [
                    {"field": "region", "divisor": 1, "digits": 1},
                    {"field": "district", "divisor": None, "digits": None},
                ],
            },
            {
                "geography": "district",
                "level": "district",
                "household_column": "district_code",
                "code_digits": 3,
                "geography_id_prefix": "D:",
                "label": "district_name",
                "record_codes": [
                    {"field": "region", "divisor": 100, "digits": 1},
                    {"field": "district", "divisor": 1, "digits": 3},
                ],
            },
        ],
        "rationale": "Region and district totals.",
    }


LABELS = {
    "region_name": lambda code, geography_id: {"1": "North", "2": "South"}.get(code),
    "district_name": lambda code, geography_id: f"District {geography_id}",
}


def test_code_fields() -> None:
    assert CodeField("region", 100, 1).value(205) == "2"
    assert CodeField("district", 1, 3).value(7) == "007"
    assert CodeField("district", None, None).value(205) is None


def test_names_and_levels() -> None:
    declaration = load_ladder_population(_section())
    assert declaration.target_name("district", 105) == "pop_district_105"
    assert declaration.target_name("region", 2) == "pop_region_2"
    assert declaration.level_of_name("pop_district_105").geography == "district"
    assert declaration.level_of_name("pop_ward_1") is None
    with pytest.raises(LadderPopulationError, match="No ladder population level"):
        declaration.level("ward")


def test_rows() -> None:
    declaration = load_ladder_population(_section())
    households = pd.DataFrame(
        {"region_code": [1, 1, 2, 1], "district_code": [101, 102, 201, 101]}
    )
    sizes = np.asarray([2.0, 0.0, 3.0, 1.0], dtype=np.float32)
    populations = {
        "region": {1: 100.0, 2: 50.0, 3: 10.0},
        "district": {101: 60.0, 102: 40.0, 201: 50.0},
    }
    records, matrix, dropped = population_rows(
        households, sizes, populations, ["region", "district"], declaration
    )
    assert dropped == ["pop_region_3"]
    assert [record["name"] for record in records] == [
        "pop_region_1",
        "pop_region_2",
        "pop_district_101",
        "pop_district_102",
        "pop_district_201",
    ]
    assert records[0] == {
        "name": "pop_region_1",
        "value": 100.0,
        "source": "test_ladder",
        "family": "test_population_ladder",
        "geography_level": "region",
        "region": "1",
        "district": None,
    }
    assert records[2]["region"] == "1" and records[2]["district"] == "101"
    dense = matrix.toarray()
    # The size-zero household supports district 102 but adds nothing to it.
    assert dense[3].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert dense[0].tolist() == [2.0, 0.0, 0.0, 1.0]
    assert matrix.data.dtype == np.float32


def test_specs() -> None:
    declaration = load_ladder_population(_section())
    specs = population_target_specs(
        ["pop_region_1", "pop_district_101"],
        [100.0, 60.0],
        declaration,
        period=2030,
        labels=LABELS,
    )
    region, district = specs
    assert region.metadata == {"geography_level": "region"}
    assert region.hierarchy.geography.id == "R:1"
    assert region.hierarchy.target.label == "North people"
    assert district.hierarchy.geography.label == "District D:101"
    assert region.hierarchy.provider is district.hierarchy.provider
    assert (region.period, region.family, region.source) == (
        2030,
        "test_population",
        "Test ladder",
    )
    with pytest.raises(LadderPopulationError, match="names no region"):
        population_target_specs(
            ["pop_region_9"], [1.0], declaration, period=2030, labels=LABELS
        )
    with pytest.raises(LadderPopulationError, match="Unknown population target"):
        population_target_specs(
            ["pop_ward_1"], [1.0], declaration, period=2030, labels=LABELS
        )


def _mutations():
    def set_(path, key, value):
        def mutate(section):
            target = section
            for step in path:
                target = target[step]
            target[key] = value

        return mutate

    def drop(path, key):
        def mutate(section):
            target = section
            for step in path:
                target = target[step]
            del target[key]

        return mutate

    first = ("levels", 0)
    return {
        "unknown key": set_((), "extra", 1),
        "missing record": drop((), "record"),
        "empty levels": set_((), "levels", []),
        "repeated level": set_(("levels", 1), "geography", "region"),
        "zero digits": set_(first, "code_digits", 0),
        "label without placeholder": set_(("spec",), "target_label", "people"),
        "digits without divisor": set_(
            first, "record_codes", [{"field": "x", "divisor": None, "digits": 2}]
        ),
        "repeated record field": set_(
            first,
            "record_codes",
            [
                {"field": "x", "divisor": 1, "digits": 1},
                {"field": "x", "divisor": 1, "digits": 1},
            ],
        ),
        "boolean divisor": set_(
            first, "record_codes", [{"field": "x", "divisor": True, "digits": 1}]
        ),
    }


@pytest.mark.parametrize("case", sorted(_mutations()))
def test_the_reader_refuses_a_malformed_declaration(case: str) -> None:
    section = _section()
    _mutations()[case](section)
    with pytest.raises(LadderPopulationError):
        load_ladder_population(section)
