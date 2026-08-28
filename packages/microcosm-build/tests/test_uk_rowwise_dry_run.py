"""Dry-run plan and manifest weight-chain tests for the rowwise driver.

microcosm#495 increment 2. The dry-run computes the clone plan — rows and
byte estimates at ``n_clones``, the input's weight-kind chain, both lineage
layers, and per-area support — without writing a dataset, so clone-count
adjudication happens before a multi-gigabyte build. Support ships twice,
honestly labelled: the *realized* assignment (the real sampler at the build
seed — identical draws to the real build, collision avoidance included) and
the analytic *collision-free* expectation, which can differ substantially
when ``n_clones`` is comparable to a group's sampleable constituency count.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from microcosm.build.uk_runtime import expected_uk_rowwise_area_support
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.frame import MassChangeRecord, WeightKind


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


def _write_toy_h5(
    path: Path,
    *,
    household: pd.DataFrame | None = None,
    fmt: str = "table",
) -> None:
    with pd.HDFStore(path) as store:
        store.put(
            "household",
            _household_frame() if household is None else household,
            format=fmt,
            data_columns=fmt == "table",
        )
        store.put("person", _person_frame(), format=fmt, data_columns=fmt == "table")
        store.put("benunit", _benunit_frame(), format=fmt, data_columns=fmt == "table")
        store.put(
            "time_period",
            pd.Series(["2023"]),
            format=fmt,
            data_columns=fmt == "table",
        )


def _dry_run_argv(
    input_h5: Path,
    output_dir: Path,
    crosswalk_path: Path,
    *extra: str,
) -> list[str]:
    return [
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
        *extra,
    ]


def test_collision_free_expectation_matches_distribution_math() -> None:
    support = expected_uk_rowwise_area_support(
        _household_frame(),
        _crosswalk_frame(),
        n_clones=2,
    )
    rows = {
        (row.area_type, row.area_code): row.expected_rows
        for row in support.itertuples(index=False)
    }
    # Collision-free: one London household at K=2 splits 75/25 across the two
    # England constituencies; the Wales household lands on the single Welsh
    # one. (The real sampler's collision avoidance forces the two London
    # clones apart — the dry-run's realized support covers that.)
    assert rows[("constituency", "E14000001")] == pytest.approx(2 * 0.75)
    assert rows[("constituency", "E14000002")] == pytest.approx(2 * 0.25)
    assert rows[("constituency", "W07000041")] == pytest.approx(2.0)
    assert rows[("la", "E06000063")] == pytest.approx(1.5)
    assert rows[("la", "W06000001")] == pytest.approx(2.0)
    total_constituency = sum(
        value for (area_type, _), value in rows.items() if area_type == "constituency"
    )
    assert total_constituency == pytest.approx(2 * len(_household_frame()))


def test_collision_free_expectation_requires_covered_countries() -> None:
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
    pytest.importorskip("h5py")
    builder = _load_builder_module()
    input_h5 = tmp_path / "populace_uk_2023.h5"
    crosswalk_path = tmp_path / "crosswalk.csv.gz"
    output_dir = tmp_path / "out"
    _write_toy_h5(input_h5)
    _crosswalk_frame().to_csv(crosswalk_path, index=False)

    monkeypatch.setattr(
        sys, "argv", _dry_run_argv(input_h5, output_dir, crosswalk_path)
    )
    assert builder.main() == 0

    plan_path = output_dir / builder.DRY_RUN_PLAN_FILENAME
    assert plan_path.exists()
    assert not (output_dir / "populace_uk_2023_rowwise.h5").exists()
    assert not (output_dir / builder.MANIFEST_FILENAME).exists()
    assert not (output_dir / "logbook-spool").exists()

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
    assert "lower-bound" in plan["plan"]["output_bytes_estimate_basis"]

    # Realized support: collision avoidance forces the London household's two
    # clones into distinct England constituencies, and the single Welsh
    # constituency absorbs both Welsh clones.
    realized = plan["realized_support"]
    assert "realized assignment at seed 42" in realized["basis"]
    constituency = realized["constituency"]
    assert constituency["n_areas"] == 3
    assert constituency["min_rows"] == pytest.approx(1.0)
    by_code = {row["area_code"]: row["rows"] for row in constituency["bottom"]}
    assert by_code["E14000001"] == pytest.approx(1.0)
    assert by_code["E14000002"] == pytest.approx(1.0)
    assert by_code["W07000041"] == pytest.approx(2.0)
    assert realized["la"]["n_areas"] == 3
    assert realized["la"]["min_rows"] == pytest.approx(1.0)

    collision_free = plan["collision_free_expected_support"]
    assert collision_free["constituency"]["min_rows"] == pytest.approx(0.5)

    assert plan["source_lineage"]["pool_modulus"] is None
    assert plan["source_lineage"]["pool"] is None
    assert plan["source_lineage"]["explicit"] is None


def test_driver_dry_run_reports_weight_chain_and_pool_lineage(
    monkeypatch, tmp_path
) -> None:
    pytest.importorskip("tables")
    pytest.importorskip("h5py")
    from microcosm.build.uk_runtime import write_uk_national_frame

    builder = _load_builder_module()
    staging = tmp_path / "staging.h5"
    household = pd.DataFrame(
        {
            "household_id": [101, 102, 100000101, 100000102],
            "household_weight": [10.0, 20.0, 10.0, 20.0],
            "region": ["LONDON", "WALES", "LONDON", "WALES"],
            "source_household_id": [901, 902, 903, 904],
        }
    )
    person = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4],
            "person_household_id": [101, 102, 100000101, 100000102],
            "person_benunit_id": [11, 12, 13, 14],
        }
    )
    benunit = pd.DataFrame({"benunit_id": [11, 12, 13, 14]})
    dataset = uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period="2023",
        weight_kind=WeightKind.IMPORTANCE,
        mass_log=(
            MassChangeRecord(
                entity="household",
                old_total=60.0,
                new_total=60.0,
                declared_factor=1.0,
                reason="Toy reviewed SPI-channel allocation record.",
            ),
        ),
    )
    write_uk_national_frame(dataset, staging)
    crosswalk_path = tmp_path / "crosswalk.csv.gz"
    _crosswalk_frame().to_csv(crosswalk_path, index=False)
    output_dir = tmp_path / "out"

    monkeypatch.setattr(
        sys,
        "argv",
        _dry_run_argv(
            staging,
            output_dir,
            crosswalk_path,
            "--allow-constituency-collisions",
            "--source-lineage-modulus",
            "100000000",
        ),
    )
    assert builder.main() == 0
    plan = json.loads((output_dir / builder.DRY_RUN_PLAN_FILENAME).read_text())
    assert plan["input"]["household_weight_kind"] == WeightKind.IMPORTANCE.value
    assert plan["input"]["mass_log_records"] == 1

    lineage = plan["source_lineage"]
    assert lineage["pool_modulus"] == 100000000
    assert lineage["pool"]["distinct_pool_source_households"] == 2
    assert lineage["pool"]["pool_copies_per_source"]["min"] == 2
    assert lineage["pool"]["pool_copies_per_source"]["max"] == 2
    # The staging input's immediate layer is reported untouched alongside.
    assert lineage["immediate"]["distinct_source_households"] == 4
    assert lineage["explicit"] is None


def test_driver_dry_run_supports_fixed_format_stores(monkeypatch, tmp_path) -> None:
    pytest.importorskip("tables")
    pytest.importorskip("h5py")
    builder = _load_builder_module()
    input_h5 = tmp_path / "fixed.h5"
    _write_toy_h5(input_h5, fmt="fixed")
    crosswalk_path = tmp_path / "crosswalk.csv.gz"
    _crosswalk_frame().to_csv(crosswalk_path, index=False)
    output_dir = tmp_path / "out"

    monkeypatch.setattr(
        sys, "argv", _dry_run_argv(input_h5, output_dir, crosswalk_path)
    )
    assert builder.main() == 0
    plan = json.loads((output_dir / builder.DRY_RUN_PLAN_FILENAME).read_text())
    assert plan["plan"]["rows"]["person"] == 6


def test_driver_dry_run_preserves_generated_crosswalk_cache(
    monkeypatch, tmp_path
) -> None:
    pytest.importorskip("tables")
    pytest.importorskip("h5py")
    builder = _load_builder_module()
    input_h5 = tmp_path / "populace_uk_2023.h5"
    _write_toy_h5(input_h5)
    crosswalk_path = tmp_path / "crosswalk.csv.gz"
    _crosswalk_frame().to_csv(crosswalk_path, index=False)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    cache = output_dir / builder.CROSSWALK_FILENAME
    cache.write_text("previously generated cache")

    monkeypatch.setattr(
        sys, "argv", _dry_run_argv(input_h5, output_dir, crosswalk_path)
    )
    assert builder.main() == 0
    assert cache.read_text() == "previously generated cache"


def test_driver_dry_run_rejects_broken_links(monkeypatch, tmp_path) -> None:
    pytest.importorskip("tables")
    pytest.importorskip("h5py")
    builder = _load_builder_module()
    input_h5 = tmp_path / "broken.h5"
    person = _person_frame()
    person.loc[0, "person_household_id"] = 999
    with pd.HDFStore(input_h5) as store:
        store.put("household", _household_frame(), format="table", data_columns=True)
        store.put("person", person, format="table", data_columns=True)
        store.put("benunit", _benunit_frame(), format="table", data_columns=True)
        store.put("time_period", pd.Series(["2023"]), format="table", data_columns=True)
    crosswalk_path = tmp_path / "crosswalk.csv.gz"
    _crosswalk_frame().to_csv(crosswalk_path, index=False)
    output_dir = tmp_path / "out"

    monkeypatch.setattr(
        sys, "argv", _dry_run_argv(input_h5, output_dir, crosswalk_path)
    )
    with pytest.raises(ValueError, match="person_household_id"):
        builder.main()


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
    assert lineage["pool_modulus"] == 100000000
    assert lineage["pool"]["distinct_pool_source_households"] == 2
    assert lineage["pool"]["pool_copies_per_source"]["min"] == 2
    assert lineage["pool"]["pool_copies_per_source"]["max"] == 2
    assert lineage["immediate"] is None
    assert lineage["explicit"] is None
    assert manifest["base_dataset"]["distinct_source_households"] is None

    output_h5 = output_dir / "pool_rowwise.h5"
    import h5py

    from microcosm.build.uk_runtime.national_frame import (
        UK_HOUSEHOLD_WEIGHT_KIND_ATTR,
    )

    with h5py.File(output_h5, mode="r") as file:
        stored = file.attrs[UK_HOUSEHOLD_WEIGHT_KIND_ATTR]
        if isinstance(stored, bytes):
            stored = stored.decode("utf-8")
        assert stored == WeightKind.DESIGN.value
