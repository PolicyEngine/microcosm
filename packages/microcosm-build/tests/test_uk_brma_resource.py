from __future__ import annotations

import importlib.util
import json
import os
from importlib.resources import files
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
# The VOA rents CSV is a staged input, not repo content: the regeneration
# check runs wherever it is staged (point UK_LHA_RENTS_CSV at it) and skips
# elsewhere, including PR CI. The committed resource's integrity is still
# pinned in CI by the source-facts test below, which holds its sha256, row
# count, and cell/BRMA counts against the generator's constants.
STAGED_LHA_RENTS = Path(
    os.environ.get(
        "UK_LHA_RENTS_CSV",
        ROOT / ".codex-work/incumbent/storage/lha_list_of_rents.csv.gz",
    )
)
GENERATOR = ROOT / "tools/build_uk_brma_count_table.py"
_SPEC = importlib.util.spec_from_file_location("build_uk_brma_count_table", GENERATOR)
assert _SPEC is not None and _SPEC.loader is not None
_generator = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_generator)
EXPECTED_ROWS = _generator.EXPECTED_ROWS
EXPECTED_SHA256 = _generator.EXPECTED_SHA256
build_resource = _generator.build_resource


def _resource() -> dict:
    return json.loads(
        files("microcosm.build.uk").joinpath("brma_rent_counts.json").read_text()
    )


@pytest.mark.skipif(
    not STAGED_LHA_RENTS.exists(),
    reason=f"staged VOA rents CSV not present at {STAGED_LHA_RENTS}",
)
def test_committed_brma_resource_matches_regenerator() -> None:
    assert _resource() == build_resource(STAGED_LHA_RENTS)


def test_brma_resource_shape_and_source_facts() -> None:
    resource = _resource()

    assert resource["country"] == "uk"
    assert resource["source"]["sha256"] == EXPECTED_SHA256
    assert resource["source"]["rows"] == EXPECTED_ROWS
    assert resource["source"]["years"] == [2019, 2020]
    assert resource["source"]["cell_count"] == 60
    assert resource["source"]["unique_brmas"] == 200
    assert resource["chronicle"]["status"] == "registration pending"
    cell_counts = [
        sum(brmas.values())
        for categories in resource["cells"].values()
        for brmas in categories.values()
    ]
    assert len(cell_counts) == 60
    assert min(cell_counts) >= 1_144


def test_missing_brma_cell_fails_closed() -> None:
    resource = _resource()

    with pytest.raises(KeyError):
        resource["cells"]["LONDON"]["Z"]
