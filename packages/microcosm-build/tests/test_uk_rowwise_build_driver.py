from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from microcosm.build.logbook import LOGBOOK_ROW_FIELDS, load_spool_rows


@pytest.fixture(autouse=True)
def _spool_only_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POPULACE_LEDGER_URL", raising=False)
    monkeypatch.delenv("POPULACE_LEDGER_KEY", raising=False)
    monkeypatch.delenv("POPULACE_LEDGER_API_KEY", raising=False)
    monkeypatch.delenv("POPULACE_LOGBOOK_PREV_ROW_DIGEST", raising=False)


def _spool_rows(output_dir: Path):
    rows = load_spool_rows(output_dir / "logbook-spool")
    for row in rows:
        assert frozenset(row.to_mapping()) == LOGBOOK_ROW_FIELDS
    return rows


def _local_ref(path: Path) -> str:
    return f"local://{path.resolve().as_posix().lstrip('/')}"


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


@pytest.mark.parametrize("route_option", ["--crosswalk", "--ladder"])
def test_input_pin_mismatch_refuses_before_side_effects(
    monkeypatch,
    tmp_path,
    route_option,
) -> None:
    pytest.importorskip("tables")
    builder = _load_builder_module()
    input_h5 = tmp_path / "input.h5"
    route_artifact = tmp_path / "route-artifact"
    output_dir = tmp_path / "out"
    _write_toy_h5(input_h5)
    route_artifact.write_bytes(b"must not be read")

    def unexpected_h5_read(_path):
        raise AssertionError("input pin mismatch must refuse before H5 parsing")

    monkeypatch.setattr(builder, "_h5_summary", unexpected_h5_read)

    with pytest.raises(SystemExit, match=r"--input-h5 sha mismatch: measured"):
        builder.main(
            [
                "--input-h5",
                str(input_h5),
                "--input-sha256",
                "0" * 64,
                "--out",
                str(output_dir),
                route_option,
                str(route_artifact),
            ]
        )

    assert not output_dir.exists()


def test_build_uk_rowwise_dataset_writes_manifest_and_outputs(
    monkeypatch, tmp_path, capsys
):
    pytest.importorskip("tables")
    builder = _load_builder_module()
    input_h5 = tmp_path / "populace_uk_2023.h5"
    crosswalk_path = tmp_path / "crosswalk.csv.gz"
    constituency_codes = tmp_path / "constituencies.csv"
    la_codes = tmp_path / "local_authorities.csv"
    output_dir = tmp_path / "out"
    _write_toy_h5(input_h5)
    input_sha256 = builder._sha256(input_h5)
    _crosswalk_frame().to_csv(crosswalk_path, index=False)
    output_dir.mkdir()
    stale_area_support = output_dir / builder.AREA_SUPPORT_FILENAME
    stale_area_support.write_text("stale")
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
            "--input-sha256",
            input_sha256,
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
    captured = capsys.readouterr()
    assert "Wrote Logbook row:" in captured.err

    output_h5 = output_dir / "populace_uk_2023_rowwise.h5"
    manifest_path = output_dir / builder.MANIFEST_FILENAME
    coverage_path = output_dir / builder.COVERAGE_FILENAME
    assert output_h5.exists()
    assert manifest_path.exists()
    assert coverage_path.exists()
    assert not stale_area_support.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["build_kind"] == "uk_rowwise_local_geography_dataset"
    assert manifest["parameters"]["n_clones"] == 2
    assert manifest["parameters"]["source_year"] == 2023
    assert manifest["parameters"]["require_all_countries"] is False
    assert manifest["inputs"]["dataset"]["pin_verified"] is True
    assert manifest["base_dataset"]["household_weight_sum"] == pytest.approx(30.0)
    assert manifest["rowwise_dataset"]["household_weight_sum"] == pytest.approx(30.0)
    assert manifest["rowwise_dataset"]["household_weight_delta"] == pytest.approx(0.0)
    assert manifest["rowwise_dataset"]["missing_geography_rows"] == 0
    assert manifest["rowwise_dataset"]["assigned_constituencies"] == 2
    assert manifest["rowwise_dataset"]["assigned_local_authorities"] == 2
    assert manifest["coverage"][0]["covered_areas"] == 4
    assert manifest["outputs"]["crosswalk"] is None
    assert manifest["outputs"]["area_support_summary"] is None
    with pd.HDFStore(output_h5, mode="r") as store:
        assert store["household"].shape[0] == 4
        assert store["person"].shape[0] == 6
        assert store["benunit"].shape[0] == 4
    rows = _spool_rows(output_dir)
    assert len(rows) == 1
    first_row = rows[0]
    assert first_row.pipeline == "uk-local-rowwise"
    assert first_row.rung == "f100"
    assert first_row.seed == 42
    assert first_row.disposition == "iterating"
    assert first_row.artifact_location == _local_ref(output_h5)
    assert first_row.gate_verdicts == {
        "uk_mass_conservation": {
            "verdict": "passed",
            "receipt": f"{_local_ref(manifest_path)}#/rowwise_dataset/weights/mass_conservation",
        },
        "uk_coverage": {
            "verdict": "passed",
            "receipt": f"{_local_ref(manifest_path)}#/rowwise_dataset/coverage",
        },
    }

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
            "--logbook-prev-row-digest",
            first_row.row_digest,
        ],
    )

    assert builder.main() == 0
    assert not coverage_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["coverage"] == []
    assert manifest["outputs"]["coverage_summary"] is None
    rows = _spool_rows(output_dir)
    assert [row.prev_row_digest for row in rows] == [None, first_row.row_digest]
    assert rows[1].gate_verdicts == {
        "uk_mass_conservation": {
            "verdict": "passed",
            "receipt": f"{_local_ref(manifest_path)}#/rowwise_dataset/weights/mass_conservation",
        }
    }


def test_build_uk_rowwise_dataset_rejects_target_csv_without_code(tmp_path):
    builder = _load_builder_module()
    bad_codes = tmp_path / "bad.csv"
    bad_codes.write_text("name\nAldershot\n")

    with pytest.raises(ValueError, match="code"):
        builder._read_code_csv(bad_codes)


def test_build_uk_rowwise_dataset_counts_blank_geography(monkeypatch, tmp_path):
    pytest.importorskip("tables")
    builder = _load_builder_module()
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
    input_h5 = tmp_path / "microcosm_uk_2024.h5"
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
    assert (output_dir / "microcosm_uk_2024_rowwise.h5").exists()
    with pd.HDFStore(output_dir / "microcosm_uk_2024_rowwise.h5", mode="r") as store:
        household = store["household"]
    assert household["source_year"].unique().tolist() == [2024]
    assert household["source_household_key"].tolist() == [
        "2024:1",
        "2024:2",
    ]


def test_build_uk_rowwise_dataset_ladder_route_records_gate_verdict(
    monkeypatch,
    tmp_path,
) -> None:
    pytest.importorskip("tables")
    builder = _load_builder_module()
    input_h5 = tmp_path / "populace_uk_2023.h5"
    ladder_path = tmp_path / "ladder.npz"
    output_dir = tmp_path / "out"
    _write_toy_h5(input_h5)
    ladder_path.write_bytes(b"ladder")

    def fake_clone(*_args, output_path: Path, **_kwargs):
        output_path.write_bytes(b"rowwise")
        household = pd.DataFrame(
            {
                "household_id": [1, 2],
                "household_weight": [10.0, 20.0],
                "oa_code": ["E0001", "W0001"],
                "lsoa_code": ["E0101", "W0101"],
                "msoa_code": ["E0201", "W0201"],
                "local_authority_code": ["E06000063", "W06000001"],
                "ward_code": ["E05000001", "W05000001"],
                "constituency_code": ["E14000001", "W07000041"],
                "region_code": ["E12000007", "W99999999"],
                "region": ["LONDON", "WALES"],
                "itl3_code": ["TLI", "TLL"],
                "itl2_code": ["TL", "TL"],
                "itl1_code": ["T", "T"],
                "country": ["England", "Wales"],
                "rowwise_household_clone_index": [0, 0],
            }
        )
        return type(
            "Result",
            (),
            {
                "person": pd.DataFrame({"person_id": [1, 2]}),
                "benunit": pd.DataFrame({"benunit_id": [1, 2]}),
                "household": household,
                "household_weight_kind": type("WeightKind", (), {"value": "design"})(),
                "mass_log": (),
                "time_period": "2023",
                "n_clones": 1,
                "id_multiplier": 10,
                "gate": type(
                    "Gate",
                    (),
                    {"passed": True, "details": {"areas": 2}},
                )(),
            },
        )()

    ladder = type(
        "Ladder",
        (),
        {
            "constituency_code": pd.Series(["E14000001", "W07000041"]).to_numpy(),
            "local_authority_code": pd.Series(
                ["E06000063", "W06000001"]
            ).to_numpy(),
        },
    )()
    monkeypatch.setattr(builder, "load_uk_oa_ladder", lambda _path: ladder)
    monkeypatch.setattr(
        builder,
        "clone_uk_dataset_with_ladder_geography",
        fake_clone,
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
            "--ladder",
            str(ladder_path),
            "--n-clones",
            "1",
        ],
    )

    assert builder.main() == 0

    manifest_path = output_dir / builder.MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    assert manifest["rowwise_dataset"]["area_support"]["source_basis"] == (
        "household_id"
    )
    rows = _spool_rows(output_dir)
    assert len(rows) == 1
    assert rows[0].gate_verdicts == {
        "uk_geography_ladder": {
            "verdict": "passed",
            "receipt": f"{_local_ref(manifest_path)}#/rowwise_dataset/gate",
        }
    }


def test_build_uk_rowwise_dataset_failure_records_pipeline_error(
    monkeypatch,
    tmp_path,
) -> None:
    pytest.importorskip("tables")
    builder = _load_builder_module()
    input_h5 = tmp_path / "populace_uk_2023.h5"
    crosswalk_path = tmp_path / "crosswalk.csv.gz"
    output_dir = tmp_path / "out"
    _write_toy_h5(input_h5)
    _crosswalk_frame().to_csv(crosswalk_path, index=False)

    def fail_clone(*_args, **_kwargs):
        raise RuntimeError("rowwise clone failed")

    monkeypatch.setattr(builder, "clone_uk_dataset_with_rowwise_geography", fail_clone)
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

    with pytest.raises(RuntimeError, match="rowwise clone failed"):
        builder.main()

    rows = _spool_rows(output_dir)
    assert len(rows) == 1
    row = rows[0]
    assert row.disposition == "failed"
    assert row.gate_verdicts["pipeline_error"]["verdict"] == "error"
    assert row.gate_verdicts["pipeline_error"]["receipt"].endswith("#/error_type")


@pytest.mark.parametrize(
    "dataset_filename",
    [
        "../escaped.h5",
        "/tmp/escaped.h5",
        "rowwise_build_manifest.json",
        "geography_coverage_summary.csv",
        "area_support_summary.csv",
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
            input_stem="microcosm_uk_2024",
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
        "area_support_summary.csv",
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
                "--dataset-filename",
                input_h5.name,
            ],
        )

    with pytest.raises(ValueError, match="must differ"):
        builder.main()
