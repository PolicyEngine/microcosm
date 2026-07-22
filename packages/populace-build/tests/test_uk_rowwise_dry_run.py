"""Dry-run plan and manifest weight-chain tests for the rowwise driver.

populace#495 increment 2. The dry-run computes the clone plan — rows and
byte estimates at ``n_clones``, the sampler's exact expected per-area support,
and the input's weight-kind chain — without writing a dataset, so clone-count
adjudication happens before a multi-gigabyte build, not after.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from populace.build.uk_runtime import expected_uk_rowwise_area_support
from populace.build.uk_runtime.national_build import UKNationalDataset
from populace.frame import MassChangeRecord, WeightKind


def _load_builder_module():
    root = Path(__file__).resolve().parents[3]
    path = root / "tools" / "build_uk_rowwise_dataset.py"
    spec = importlib.util.spec_from_file_location("build_uk_rowwise_dataset", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _household_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "household_id": [1, 2],
            "household_weight": [10.0, 20.0],
            "region": ["LONDON", "WALES"],
        }
    )


def _person_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "person_id": [1001, 2001, 2002],
            "person_household_id": [1, 2, 2],
            "person_benunit_id": [101, 201, 201],
        }
    )


def _benunit_frame() -> pd.DataFrame:
    return pd.DataFrame({"benunit_id": [101, 201]})


def _crosswalk_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "oa_code": "E0001",
                "lsoa_code": "E0101",
                "msoa_code": "E0201",
                "la_code": "E06000063",
                "constituency_code": "E14000001",
                "region_code": "E12000007",
                "country": "England",
                "population": 75,
            },
            {
                "oa_code": "E0002",
                "lsoa_code": "E0102",
                "msoa_code": "E0202",
                "la_code": "E06000064",
                "constituency_code": "E14000002",
                "region_code": "E12000007",
                "country": "England",
                "population": 25,
            },
            {
                "oa_code": "W0001",
                "lsoa_code": "W0101",
                "msoa_code": "W0201",
                "la_code": "W06000001",
                "constituency_code": "W07000041",
                "region_code": "W99999999",
                "country": "Wales",
                "population": 80,
            },
        ]
    )


def _write_toy_h5(path: Path) -> None:
    with pd.HDFStore(path) as store:
        store.put("household", _household_frame(), format="table", data_columns=True)
        store.put("person", _person_frame(), format="table", data_columns=True)
        store.put("benunit", _benunit_frame(), format="table", data_columns=True)
        store.put(
            "time_period",
            pd.Series(["2023"]),
            format="table",
            data_columns=True,
        )


def test_expected_area_support_matches_sampler_expectation() -> None:
    support = expected_uk_rowwise_area_support(
        _household_frame(),
        _crosswalk_frame(),
        n_clones=2,
    )
    rows = {
        (row.area_type, row.area_code): row.expected_rows
        for row in support.itertuples(index=False)
    }
    # One London household at K=2 splits 75/25 across the two England
    # constituencies; the Wales household lands on the single Welsh one.
    assert rows[("constituency", "E14000001")] == pytest.approx(2 * 0.75)
    assert rows[("constituency", "E14000002")] == pytest.approx(2 * 0.25)
    assert rows[("constituency", "W07000041")] == pytest.approx(2.0)
    assert rows[("la", "E06000063")] == pytest.approx(1.5)
    assert rows[("la", "W06000001")] == pytest.approx(2.0)
    total_constituency = sum(
        value for (area_type, _), value in rows.items() if area_type == "constituency"
    )
    assert total_constituency == pytest.approx(2 * len(_household_frame()))


def test_expected_area_support_requires_covered_countries() -> None:
    crosswalk = _crosswalk_frame()
    england_only = crosswalk[crosswalk["country"] == "England"]
    with pytest.raises(ValueError, match="Wales"):
        expected_uk_rowwise_area_support(
            _household_frame(),
            england_only,
            n_clones=1,
        )
    support = expected_uk_rowwise_area_support(
        _household_frame(),
        england_only,
        n_clones=1,
        require_all_countries=False,
    )
    assert set(support["area_code"]) == {
        "E14000001",
        "E14000002",
        "E06000063",
        "E06000064",
    }


def test_driver_dry_run_writes_plan_only(monkeypatch, tmp_path) -> None:
    pytest.importorskip("tables")
    builder = _load_builder_module()
    input_h5 = tmp_path / "populace_uk_2023.h5"
    crosswalk_path = tmp_path / "crosswalk.csv.gz"
    output_dir = tmp_path / "out"
    _write_toy_h5(input_h5)
    _crosswalk_frame().to_csv(crosswalk_path, index=False)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_uk_rowwise_dataset.py",
            "--input-h5",
            str(input_h5),
            "--out",
            str(output_dir),
            "--crosswalk",
            str(crosswalk_path),
            "--n-clones",
            "2",
            "--allow-missing-country",
            "--dry-run",
        ],
    )
    assert builder.main() == 0

    plan_path = output_dir / builder.DRY_RUN_PLAN_FILENAME
    assert plan_path.exists()
    assert not (output_dir / "populace_uk_2023_rowwise.h5").exists()
    assert not (output_dir / builder.MANIFEST_FILENAME).exists()

    plan = json.loads(plan_path.read_text())
    assert plan["build_kind"] == "uk_rowwise_local_geography_dry_run"
    assert plan["parameters"]["n_clones"] == 2
    assert plan["input"]["dataset"]["sha256"]
    assert plan["input"]["household_weight_kind"] == WeightKind.DESIGN.value
    assert plan["input"]["mass_log_records"] == 0
    assert plan["plan"]["rows"] == {
        "person": 6,
        "benunit": 4,
        "household": 4,
    }
    assert plan["plan"]["output_bytes_estimate"] >= input_h5.stat().st_size
    assert plan["plan"]["output_bytes_estimate_basis"]

    constituency = plan["expected_support"]["constituency"]
    assert constituency["n_areas"] == 3
    assert constituency["min_expected_rows"] == pytest.approx(0.5)
    by_code = {row["area_code"]: row["expected_rows"] for row in constituency["bottom"]}
    assert by_code["E14000002"] == pytest.approx(0.5)
    la = plan["expected_support"]["la"]
    assert la["n_areas"] == 3


def test_driver_dry_run_reports_weight_chain_from_staging_attrs(
    monkeypatch, tmp_path
) -> None:
    pytest.importorskip("tables")
    pytest.importorskip("h5py")
    from populace.build.uk_runtime import write_uk_national_dataset

    builder = _load_builder_module()
    staging = tmp_path / "staging.h5"
    dataset = UKNationalDataset(
        person=_person_frame(),
        benunit=_benunit_frame(),
        household=_household_frame(),
        time_period="2023",
        household_weight_kind=WeightKind.IMPORTANCE,
        mass_log=(
            MassChangeRecord(
                entity="household",
                old_total=30.0,
                new_total=30.0,
                declared_factor=1.0,
                reason="Toy reviewed SPI-channel allocation record.",
            ),
        ),
    )
    write_uk_national_dataset(dataset, staging)
    crosswalk_path = tmp_path / "crosswalk.csv.gz"
    _crosswalk_frame().to_csv(crosswalk_path, index=False)
    output_dir = tmp_path / "out"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_uk_rowwise_dataset.py",
            "--input-h5",
            str(staging),
            "--out",
            str(output_dir),
            "--crosswalk",
            str(crosswalk_path),
            "--n-clones",
            "2",
            "--allow-missing-country",
            "--dry-run",
        ],
    )
    assert builder.main() == 0
    plan = json.loads((output_dir / builder.DRY_RUN_PLAN_FILENAME).read_text())
    assert plan["input"]["household_weight_kind"] == WeightKind.IMPORTANCE.value
    assert plan["input"]["mass_log_records"] == 1


def test_driver_full_build_records_weight_chain_and_lineage(
    monkeypatch, tmp_path
) -> None:
    pytest.importorskip("tables")
    pytest.importorskip("h5py")
    builder = _load_builder_module()
    input_h5 = tmp_path / "pool.h5"
    with pd.HDFStore(input_h5) as store:
        store.put(
            "household",
            pd.DataFrame(
                {
                    "household_id": [101, 102, 100000101, 100000102],
                    "household_weight": [10.0, 20.0, 10.0, 20.0],
                    "region": ["LONDON", "WALES", "LONDON", "WALES"],
                }
            ),
            format="table",
            data_columns=True,
        )
        store.put(
            "person",
            pd.DataFrame(
                {
                    "person_id": [1, 2, 3, 4],
                    "person_household_id": [101, 102, 100000101, 100000102],
                    "person_benunit_id": [11, 12, 13, 14],
                }
            ),
            format="table",
            data_columns=True,
        )
        store.put(
            "benunit",
            pd.DataFrame({"benunit_id": [11, 12, 13, 14]}),
            format="table",
            data_columns=True,
        )
        store.put(
            "time_period",
            pd.Series(["2023"]),
            format="table",
            data_columns=True,
        )
    crosswalk_path = tmp_path / "crosswalk.csv.gz"
    _crosswalk_frame().to_csv(crosswalk_path, index=False)
    output_dir = tmp_path / "out"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_uk_rowwise_dataset.py",
            "--input-h5",
            str(input_h5),
            "--out",
            str(output_dir),
            "--crosswalk",
            str(crosswalk_path),
            "--n-clones",
            "2",
            "--allow-missing-country",
            "--allow-constituency-collisions",
            "--source-lineage-modulus",
            "100000000",
        ],
    )
    assert builder.main() == 0
    manifest = json.loads((output_dir / builder.MANIFEST_FILENAME).read_text())

    weights = manifest["rowwise_dataset"]["weights"]
    assert weights["household_weight_kind"] == WeightKind.DESIGN.value
    assert weights["mass_log_records"] == 1
    conservation = weights["mass_conservation"]
    assert conservation["passed"] is True
    assert conservation["input_total"] == pytest.approx(60.0)
    assert conservation["output_total"] == pytest.approx(60.0)
    assert conservation["relative_tolerance"] > 0

    lineage = manifest["rowwise_dataset"]["source_lineage"]
    assert lineage["modulus"] == 100000000
    assert lineage["distinct_source_households"] == 2
    assert lineage["pool_copies_per_source"]["min"] == 2
    assert lineage["pool_copies_per_source"]["max"] == 2

    output_h5 = output_dir / "populace_uk_2023_rowwise.h5"
    import h5py

    from populace.build.uk_runtime.national_build import (
        UK_HOUSEHOLD_WEIGHT_KIND_ATTR,
    )

    with h5py.File(output_h5, mode="r") as file:
        stored = file.attrs[UK_HOUSEHOLD_WEIGHT_KIND_ATTR]
        if isinstance(stored, bytes):
            stored = stored.decode("utf-8")
        assert stored == WeightKind.DESIGN.value
