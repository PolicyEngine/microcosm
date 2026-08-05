"""Disclosure control and classification in the #612 Frame-linkage preflight.

The preflight's output is written to be posted on the tracking issue, which
makes it a publication under UK Data Service EUL CD137 clause 8 / CD171
§5.2.1: no unit-record values, nothing reported from one or two cases. Frame's
own exception messages embed real ids, so the tool must classify without ever
echoing them — these tests plant a sentinel id in violating tables and assert
it appears nowhere in any output.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]

SENTINEL_ID = 987654321


def _preflight_module():
    path = ROOT / "tools" / "preflight_uk_frame_linkage.py"
    spec = importlib.util.spec_from_file_location("preflight_uk_frame_linkage", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PREFLIGHT = _preflight_module()


def _clean_tables() -> dict[str, pd.DataFrame]:
    return {
        "person": pd.DataFrame(
            {
                "person_id": [1, 2, 3, 4],
                "person_benunit_id": [10, 10, 20, 20],
                "person_household_id": [100, 100, 200, 200],
                "age": [34.0, 3.0, 61.0, 59.0],
            }
        ),
        "benunit": pd.DataFrame({"benunit_id": [10, 20]}),
        "household": pd.DataFrame(
            {
                "household_id": [100, 200],
                "household_weight": [1000.0, 2000.0],
            }
        ),
    }


def _violating_tables() -> dict[str, pd.DataFrame]:
    """Every violation class at once, each below the SDC minimum, with a
    sentinel id on the orphaned benunit row (the value a leak would expose)."""

    return {
        "person": pd.DataFrame(
            {
                "person_id": [1, 2, 3, 4],
                "person_benunit_id": [10, 10, 20, 20],
                # Benunit 20 spans households 200 and 300 (split benunit);
                # household 300 does not exist (dangling membership).
                "person_household_id": [100, 100, 200, 300],
            }
        ),
        # Unsorted, duplicated, and one orphan row nobody references.
        "benunit": pd.DataFrame({"benunit_id": [20, 10, 10, SENTINEL_ID]}),
        # Unsorted ids and a negative weight.
        "household": pd.DataFrame(
            {
                "household_id": [200, 100],
                "household_weight": [1000.0, -5.0],
            }
        ),
    }


def _write_artifact(path: Path, tables: dict[str, pd.DataFrame]) -> None:
    with pd.HDFStore(path, mode="w") as store:
        for name, table in tables.items():
            store.put(name, table)
        store.put("time_period", pd.Series(["2023"]))


def test_sdc_count_masks_small_nonzero_counts_only() -> None:
    assert PREFLIGHT.sdc_count(0, minimum=10) == 0
    assert PREFLIGHT.sdc_count(3, minimum=10) == "< 10"
    assert PREFLIGHT.sdc_count(9, minimum=10) == "< 10"
    assert PREFLIGHT.sdc_count(10, minimum=10) == 10
    assert PREFLIGHT.sdc_count(25, minimum=10) == 25


def test_clean_artifact_constructs_a_frame(tmp_path: Path) -> None:
    path = tmp_path / "clean.h5"
    _write_artifact(path, _clean_tables())

    report = PREFLIGHT.preflight_artifact(path, minimum=10)

    assert report["frame_constructed"] is True
    assert report["time_period"] == "2023"
    assert report["household_weight_kind"] == "design"
    assert report["column_collisions"] == []
    for group in ("benunit", "household"):
        linkage = report["linkage"][group]
        assert linkage["ids_sorted_ascending"] is True
        assert linkage["orphaned_group_rows"] == 0
        assert linkage["dangling_memberships"] == 0
        assert linkage["id_duplicated"] == 0
    assert report["linkage"]["split_benunits"] == 0
    weights = report["household_weights"]
    assert weights["negative"] == 0
    assert weights["total_mass"] == 3000.0
    assert report["tables"]["person"]["rows"] == 4


def test_violations_are_classified_masked_and_never_leak_ids(
    tmp_path: Path,
) -> None:
    path = tmp_path / "violating.h5"
    _write_artifact(path, _violating_tables())

    report = PREFLIGHT.preflight_artifact(path, minimum=10)

    assert report["frame_constructed"] is False
    benunit = report["linkage"]["benunit"]
    assert benunit["ids_sorted_ascending"] is False
    assert benunit["id_duplicated"] == "< 10"
    assert benunit["orphaned_group_rows"] == "< 10"
    household = report["linkage"]["household"]
    assert household["ids_sorted_ascending"] is False
    assert household["dangling_memberships"] == "< 10"
    assert report["linkage"]["split_benunits"] == "< 10"
    assert report["household_weights"]["negative"] == "< 10"

    # The canary: the planted unit-record id appears nowhere in the report.
    rendered = json.dumps(report)
    assert str(SENTINEL_ID) not in rendered


def test_main_reports_both_artifacts_and_exits_nonzero_on_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    clean = tmp_path / "clean.h5"
    violating = tmp_path / "violating.h5"
    json_out = tmp_path / "report.json"
    _write_artifact(clean, _clean_tables())
    _write_artifact(violating, _violating_tables())

    exit_code = PREFLIGHT.main(
        [str(clean), str(violating), "--json-out", str(json_out)]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    report = json.loads(captured.out)
    assert report["sdc_minimum_count"] == 10
    constructed = {
        Path(artifact["path"]).name: artifact["frame_constructed"]
        for artifact in report["artifacts"]
    }
    assert constructed == {"clean.h5": True, "violating.h5": False}
    # The canary holds across every output channel: stdout, stderr, and the
    # JSON file all omit the planted unit-record id.
    assert str(SENTINEL_ID) not in captured.out
    assert str(SENTINEL_ID) not in captured.err
    assert str(SENTINEL_ID) not in json_out.read_text()


def test_main_exits_zero_when_every_artifact_constructs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    clean = tmp_path / "clean.h5"
    _write_artifact(clean, _clean_tables())

    assert PREFLIGHT.main([str(clean)]) == 0
    assert json.loads(capsys.readouterr().out)["artifacts"][0]["frame_constructed"]
