from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from importlib.resources import files
from pathlib import Path

import pandas as pd
import pytest

from microcosm.build.uk_runtime.frs_brma import assign_brma_by_cell
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")


ROOT = _TEST_PATHS.repository
# The census households CSV is a staged input, not repo content: the
# regeneration check runs wherever it is staged (point UK_BRMA_HOUSEHOLDS_CSV
# at it) and skips elsewhere, including PR CI. In CI the committed cells are
# pinned by digest and by each nation's census total below.
STAGED_HOUSEHOLDS = Path(
    os.environ.get(
        "UK_BRMA_HOUSEHOLDS_CSV",
        ROOT / ".codex-work/incumbent/storage/brma_private_rented_households.csv",
    )
)
GENERATOR = ROOT / "tools/build_uk_brma_household_table.py"
_SPEC = importlib.util.spec_from_file_location(
    "build_uk_brma_household_table", GENERATOR
)
assert _SPEC is not None and _SPEC.loader is not None
_generator = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_generator)
EXPECTED_ROWS = _generator.EXPECTED_ROWS
EXPECTED_SHA256 = _generator.EXPECTED_SHA256
build_resource = _generator.build_resource

REGIONS = {
    "NORTH_EAST",
    "NORTH_WEST",
    "YORKSHIRE",
    "EAST_MIDLANDS",
    "WEST_MIDLANDS",
    "EAST_OF_ENGLAND",
    "LONDON",
    "SOUTH_EAST",
    "SOUTH_WEST",
    "WALES",
    "SCOTLAND",
    "NORTHERN_IRELAND",
}


def _resource() -> dict:
    return json.loads(
        files("microcosm.build.uk")
        .joinpath("brma_private_rented_households.json")
        .read_text()
    )


@pytest.mark.skipif(
    not STAGED_HOUSEHOLDS.exists(),
    reason=f"staged census households CSV not present at {STAGED_HOUSEHOLDS}",
)
def test_committed_brma_resource_matches_regenerator() -> None:
    assert _resource() == build_resource(STAGED_HOUSEHOLDS)


def test_brma_resource_shape_and_source_facts() -> None:
    resource = _resource()

    assert resource["country"] == "uk"
    assert resource["source"]["sha256"] == EXPECTED_SHA256
    assert resource["source"]["rows"] == EXPECTED_ROWS
    assert resource["source"]["census_years"] == {
        "england_wales": 2021,
        "northern_ireland": 2021,
        "scotland": 2022,
    }
    assert resource["source"]["cell_count"] == 60
    assert resource["source"]["unique_brmas"] == 200
    assert resource["chronicle"]["status"] == "registration pending"
    assert set(resource["cells"]) == REGIONS
    assert all(set(c) == set("ABCDE") for c in resource["cells"].values())
    cell_counts = [
        sum(brmas.values())
        for categories in resource["cells"].values()
        for brmas in categories.values()
    ]
    assert len(cell_counts) == 60
    assert min(cell_counts) >= 18_000
    assert all(
        count > 0
        for categories in resource["cells"].values()
        for brmas in categories.values()
        for count in brmas.values()
    )


def test_shared_and_one_bedroom_categories_share_one_bedroom_counts() -> None:
    for categories in _resource()["cells"].values():
        assert categories["A"] == categories["B"]


def test_northern_ireland_weights_do_not_depend_on_category() -> None:
    # Northern Ireland's 2021 census has no bedrooms question.
    categories = _resource()["cells"]["NORTHERN_IRELAND"]
    assert all(categories[c] == categories["A"] for c in "BCDE")
    assert len(categories["A"]) == 8


def test_glasgow_and_edinburgh_lead_scotland() -> None:
    # Census 2022 private-rented households: Lothian 19.3%, Greater Glasgow
    # 15.9%. The 2019-20 list-of-rents counts this replaced ranked Greater
    # Glasgow last of 18.
    scotland = _resource()["cells"]["SCOTLAND"]
    totals = {
        brma: sum(scotland[c].get(brma, 0) for c in "BCDE") for brma in scotland["B"]
    }
    top_two = sorted(totals, key=totals.get, reverse=True)[:2]
    assert set(top_two) == {"LOTHIAN", "GREATER_GLASGOW"}


# sha256 of the cells as compact sorted JSON; regenerate only with the CSV.
CELLS_SHA256 = "74a69e169516cbfc439b0d094a609dc71bc0288e6c61c7e71a4adfd53d426f0e"
# Private-rented households per nation in the source CSV (Census 2021/2022).
NATION_TOTALS = {
    "ENGLAND": 4_795_158,
    "WALES": 228_601,
    "SCOTLAND": 323_001,
    "NORTHERN_IRELAND": 132_449,
}


def test_cells_are_pinned_in_ci() -> None:
    cells = _resource()["cells"]
    canonical = json.dumps(cells, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(canonical).hexdigest() == CELLS_SHA256
    totals: dict[str, int] = {}
    for region, categories in cells.items():
        nation = region if region in NATION_TOTALS else "ENGLAND"
        # A repeats B (both one-bedroom); Northern Ireland repeats one cell.
        used = "A" if region == "NORTHERN_IRELAND" else "BCDE"
        totals[nation] = totals.get(nation, 0) + sum(
            sum(categories[c].values()) for c in used
        )
    assert totals == NATION_TOTALS


def test_missing_brma_cell_fails_closed() -> None:
    benunits = pd.DataFrame(
        {
            "benunit_id": [1, 2],
            "region": ["LONDON", "LONDON"],
            "LHA_category": ["B", "Z"],
        }
    )
    with pytest.raises(KeyError, match="missing BRMA count-table cell"):
        assign_brma_by_cell(benunits, count_resource=_resource(), seed=0)
