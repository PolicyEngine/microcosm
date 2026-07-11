from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


def _load_builder_module():
    root = Path(__file__).resolve().parents[3]
    path = root / "tools" / "build_uk_rowwise_dataset.py"
    spec = importlib.util.spec_from_file_location("build_uk_rowwise_dataset", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_toy_h5(
    path: Path,
    *,
    regions: tuple[str, str] = ("LONDON", "WALES"),
    time_period: str = "2023",
) -> None:
    with pd.HDFStore(path) as store:
        store.put(
            "household",
            pd.DataFrame(
                {
                    "household_id": [1, 2],
                    "household_weight": [10.0, 20.0],
                    "region": list(regions),
                }
            ),
            format="table",
            data_columns=True,
        )
        store.put(
            "person",
            pd.DataFrame(
                {
                    "person_id": [1001, 2001, 2002],
                    "person_household_id": [1, 2, 2],
                    "person_benunit_id": [101, 201, 201],
                }
            ),
            format="table",
            data_columns=True,
        )
        store.put(
            "benunit",
            pd.DataFrame({"benunit_id": [101, 201]}),
            format="table",
            data_columns=True,
        )
        store.put(
            "time_period",
            pd.Series([time_period]),
            format="table",
            data_columns=True,
        )


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
                "population": 100,
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
            {
                "oa_code": "S0001",
                "lsoa_code": "S0101",
                "msoa_code": "S0201",
                "la_code": "S12000033",
                "constituency_code": "S14000001",
                "region_code": "S99999999",
                "country": "Scotland",
                "population": 90,
            },
            {
                "oa_code": "N20000001",
                "lsoa_code": "N20000001",
                "msoa_code": "N21000001",
                "la_code": "N09000001",
                "constituency_code": "N05000001",
                "region_code": "N99999999",
                "country": "Northern Ireland",
                "population": 70,
            },
        ]
    )


def _allow_input_coverage(builder, monkeypatch) -> None:
    """Keep unrelated driver tests focused while the wiring gets its own tests."""
    monkeypatch.setattr(
        builder,
        "uk_release_input_coverage_gate",
        lambda result, engine: SimpleNamespace(
            passed=True,
            failures=(),
            details={"required_columns": 132, "missing": []},
        ),
    )


def test_build_uk_rowwise_dataset_writes_manifest_and_outputs(monkeypatch, tmp_path):
    pytest.importorskip("tables")
    builder = _load_builder_module()
    _allow_input_coverage(builder, monkeypatch)
    input_h5 = tmp_path / "populace_uk_2023.h5"
    crosswalk_path = tmp_path / "crosswalk.csv.gz"
    constituency_codes = tmp_path / "constituencies.csv"
    la_codes = tmp_path / "local_authorities.csv"
    output_dir = tmp_path / "out"
    _write_toy_h5(input_h5)
    _crosswalk_frame().to_csv(crosswalk_path, index=False)
    pd.DataFrame({"code": ["E14000001", "W07000041", "S14000001", "N05000001"]}).to_csv(
        constituency_codes, index=False
    )
    pd.DataFrame({"code": ["E06000063", "W06000001", "S12000033", "N09000001"]}).to_csv(
        la_codes, index=False
    )
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
            "--constituency-codes",
            str(constituency_codes),
            "--la-codes",
            str(la_codes),
            "--n-clones",
            "2",
            "--allow-missing-country",
            "--allow-constituency-collisions",
        ],
    )

    assert builder.main() == 0

    output_h5 = output_dir / "populace_uk_2023_rowwise.h5"
    manifest_path = output_dir / builder.MANIFEST_FILENAME
    coverage_path = output_dir / builder.COVERAGE_FILENAME
    input_coverage_path = output_dir / builder.INPUT_COVERAGE_FILENAME
    assert output_h5.exists()
    assert manifest_path.exists()
    assert coverage_path.exists()
    assert input_coverage_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["build_kind"] == "uk_rowwise_local_geography_dataset"
    assert manifest["parameters"]["n_clones"] == 2
    assert manifest["parameters"]["source_year"] == 2023
    assert manifest["parameters"]["require_all_countries"] is False
    assert manifest["base_dataset"]["household_weight_sum"] == pytest.approx(30.0)
    assert manifest["rowwise_dataset"]["household_weight_sum"] == pytest.approx(30.0)
    assert manifest["rowwise_dataset"]["household_weight_delta"] == pytest.approx(0.0)
    assert manifest["rowwise_dataset"]["missing_geography_rows"] == 0
    assert manifest["rowwise_dataset"]["assigned_constituencies"] == 2
    assert manifest["rowwise_dataset"]["assigned_local_authorities"] == 2
    assert manifest["coverage"][0]["covered_areas"] == 4
    assert manifest["input_coverage"]["passed"] is True
    assert manifest["outputs"]["input_coverage"]["sha256"]
    assert manifest["outputs"]["crosswalk"] is None
    with pd.HDFStore(output_h5, mode="r") as store:
        assert store["household"].shape[0] == 4
        assert store["person"].shape[0] == 6
        assert store["benunit"].shape[0] == 4

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
            "1",
            "--allow-missing-country",
        ],
    )

    assert builder.main() == 0
    assert not coverage_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["coverage"] == []
    assert manifest["outputs"]["coverage_summary"] is None


def test_build_uk_rowwise_dataset_coverage_gate_blocks_h5_write(monkeypatch, tmp_path):
    pytest.importorskip("tables")
    pytest.importorskip("policyengine_uk")
    builder = _load_builder_module()
    input_h5 = tmp_path / "populace_uk_2023.h5"
    crosswalk_path = tmp_path / "crosswalk.csv.gz"
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    stale_manifest = output_dir / builder.MANIFEST_FILENAME
    stale_coverage = output_dir / builder.COVERAGE_FILENAME
    stale_manifest.write_text('{"stale_success": true}\n')
    stale_coverage.write_text("stale,success\n")
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
            "1",
            "--allow-missing-country",
        ],
    )

    with pytest.raises(RuntimeError, match="Input coverage failed"):
        builder.main()

    output_h5 = output_dir / "populace_uk_2023_rowwise.h5"
    assert not output_h5.exists()
    assert not stale_manifest.exists()
    assert not stale_coverage.exists()
    diagnostic = json.loads((output_dir / builder.INPUT_COVERAGE_FILENAME).read_text())
    assert diagnostic["enforced"] is True
    assert diagnostic["input_coverage"]["passed"] is False
    assert "employment_income" in diagnostic["input_coverage"]["details"]["missing"]


def test_build_uk_rowwise_dataset_rejects_target_csv_without_code(tmp_path):
    builder = _load_builder_module()
    bad_codes = tmp_path / "bad.csv"
    bad_codes.write_text("name\nAldershot\n")

    with pytest.raises(ValueError, match="code"):
        builder._read_code_csv(bad_codes)


def test_build_uk_rowwise_dataset_counts_blank_geography(monkeypatch, tmp_path):
    pytest.importorskip("tables")
    builder = _load_builder_module()
    _allow_input_coverage(builder, monkeypatch)
    input_h5 = tmp_path / "populace_uk_2023.h5"
    crosswalk_path = tmp_path / "england_only_crosswalk.csv.gz"
    output_dir = tmp_path / "out"
    _write_toy_h5(input_h5, regions=("LONDON", "SCOTLAND"))
    _crosswalk_frame().iloc[:1].to_csv(crosswalk_path, index=False)
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
            "1",
            "--allow-missing-country",
        ],
    )

    assert builder.main() == 0

    manifest = json.loads((output_dir / builder.MANIFEST_FILENAME).read_text())
    assert manifest["rowwise_dataset"]["missing_geography_rows"] == 1
    assert manifest["rowwise_dataset"]["assigned_constituencies"] == 1
    assert manifest["rowwise_dataset"]["assigned_local_authorities"] == 1
    assert (
        manifest["rowwise_dataset"]["duplicate_source_household_constituency_pairs"]
        == 0
    )


def test_build_uk_rowwise_dataset_infers_source_year_from_h5(monkeypatch, tmp_path):
    pytest.importorskip("tables")
    builder = _load_builder_module()
    _allow_input_coverage(builder, monkeypatch)
    input_h5 = tmp_path / "populace_uk_2024.h5"
    crosswalk_path = tmp_path / "crosswalk.csv.gz"
    output_dir = tmp_path / "out"
    _write_toy_h5(input_h5, time_period="2024")
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
            "1",
            "--allow-missing-country",
            "--allow-constituency-collisions",
        ],
    )

    assert builder.main() == 0

    manifest = json.loads((output_dir / builder.MANIFEST_FILENAME).read_text())
    assert manifest["parameters"]["source_year"] == 2024
    assert manifest["rowwise_dataset"]["time_period"] == "2024"
    assert (output_dir / "populace_uk_2024_rowwise.h5").exists()
    with pd.HDFStore(output_dir / "populace_uk_2024_rowwise.h5", mode="r") as store:
        household = store["household"]
    assert household["source_year"].unique().tolist() == [2024]
    assert household["source_household_key"].tolist() == [
        "2024:1",
        "2024:2",
    ]


@pytest.mark.parametrize(
    "dataset_filename",
    [
        "../escaped.h5",
        "/tmp/escaped.h5",
        "rowwise_build_manifest.json",
        "geography_coverage_summary.csv",
        "input_coverage.json",
        "uk_official_geography_crosswalk.csv.gz",
    ],
)
def test_dataset_output_path_rejects_paths_and_reserved_names(
    dataset_filename, tmp_path
):
    builder = _load_builder_module()

    with pytest.raises(ValueError, match="dataset-filename"):
        builder._dataset_output_path(
            tmp_path,
            dataset_filename=dataset_filename,
            source_year=2023,
        )


def test_validate_output_paths_rejects_crosswalk_collision(tmp_path):
    builder = _load_builder_module()
    crosswalk = tmp_path / "rowwise.h5"
    args = type(
        "Args",
        (),
        {
            "out": tmp_path,
            "crosswalk": crosswalk,
        },
    )

    with pytest.raises(ValueError, match="differ"):
        builder._validate_output_paths(
            input_h5=tmp_path / "source.h5",
            output_h5=crosswalk,
            args=args,
        )


@pytest.mark.parametrize(
    "sidecar_name",
    [
        "rowwise_build_manifest.json",
        "geography_coverage_summary.csv",
        "input_coverage.json",
    ],
)
def test_validate_output_paths_rejects_supplied_crosswalk_sidecar_collision(
    sidecar_name,
    tmp_path,
):
    builder = _load_builder_module()
    sidecar_path = tmp_path / sidecar_name
    args = type(
        "Args",
        (),
        {
            "out": tmp_path,
            "crosswalk": sidecar_path,
        },
    )

    with pytest.raises(ValueError, match="crosswalk.*sidecars"):
        builder._validate_output_paths(
            input_h5=tmp_path / "source.h5",
            output_h5=tmp_path / "rowwise.h5",
            args=args,
        )


def test_load_or_build_crosswalk_unlinks_stale_generated_sidecar(tmp_path):
    builder = _load_builder_module()
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    supplied_crosswalk = tmp_path / "supplied_crosswalk.csv.gz"
    stale_generated_crosswalk = output_dir / builder.CROSSWALK_FILENAME
    _crosswalk_frame().to_csv(supplied_crosswalk, index=False)
    stale_generated_crosswalk.write_text("stale")
    args = type(
        "Args",
        (),
        {
            "out": output_dir,
            "crosswalk": supplied_crosswalk,
        },
    )

    source = builder._load_or_build_crosswalk(args)

    assert source.generated is False
    assert source.path == supplied_crosswalk.resolve()
    assert not stale_generated_crosswalk.exists()


def test_load_or_build_crosswalk_keeps_supplied_generated_path(tmp_path):
    builder = _load_builder_module()
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    supplied_crosswalk = output_dir / builder.CROSSWALK_FILENAME
    _crosswalk_frame().to_csv(supplied_crosswalk, index=False)
    args = type(
        "Args",
        (),
        {
            "out": output_dir,
            "crosswalk": supplied_crosswalk,
        },
    )

    source = builder._load_or_build_crosswalk(args)

    assert source.generated is False
    assert source.path == supplied_crosswalk.resolve()
    assert supplied_crosswalk.exists()


def test_build_uk_rowwise_dataset_rejects_overwriting_input(monkeypatch, tmp_path):
    pytest.importorskip("tables")
    builder = _load_builder_module()
    input_h5 = tmp_path / "populace_uk_2023_rowwise.h5"
    _write_toy_h5(input_h5)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_uk_rowwise_dataset.py",
            "--input-h5",
            str(input_h5),
            "--out",
            str(tmp_path),
        ],
    )

    with pytest.raises(ValueError, match="must differ"):
        builder.main()
