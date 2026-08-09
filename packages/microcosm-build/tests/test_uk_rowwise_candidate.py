"""Synthetic end-to-end contract for the first UK rowwise candidate."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime import (
    assemble_uk_oa_ladder,
    ladder_target_provenance,
    load_uk_oa_ladder,
    read_uk_single_year_weight_metadata,
    write_uk_national_frame,
)
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.frame import MassChangeRecord, WeightKind


def _load_builder_module():
    root = Path(__file__).resolve().parents[3]
    path = root / "tools" / "build_uk_rowwise_candidate.py"
    spec = importlib.util.spec_from_file_location(
        "build_uk_rowwise_candidate",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _ladder_metadata() -> dict[str, object]:
    def layer(vintage: str) -> dict[str, object]:
        return {"vintage": vintage, "source": "synthetic test source"}

    return {
        "schema_version": 1,
        "kind": "uk_oa_ladder",
        "coverage": "uk",
        "oa_vintage": "synthetic",
        "constituency_sampling_basis": "synthetic household counts",
        "oa_sampling_basis": "synthetic population",
        "layers": {
            "constituency": layer("2024_pcon"),
            "lsoa": layer("synthetic"),
            "msoa": layer("synthetic"),
            "local_authority": layer("synthetic"),
            "ward": layer("synthetic"),
            "itl": layer("2021_itl"),
            "region": layer("synthetic"),
        },
    }


def _ladder_frame(
    household_counts: tuple[float, float, float, float] = (
        3.0,
        10.0,
        10.0,
        10.0,
    ),
) -> pd.DataFrame:
    rows = [
        (
            "E00000001",
            "E12000007",
            "E14000001",
            "E05014284",
            "E09000001",
            "TLI31",
        ),
        (
            "W00000001",
            "W99999999",
            "W07000041",
            "W05001517",
            "W06000001",
            "TLL11",
        ),
        (
            "S00000001",
            "S99999999",
            "S14000001",
            "S13002835",
            "S12000033",
            "TLM50",
        ),
        (
            "N20000001",
            "N99999999",
            "N05000001",
            "N10000104",
            "N09000001",
            "TLN0A",
        ),
    ]
    return pd.DataFrame(
        [
            {
                "oa_code": oa,
                "population": 100.0,
                "households": households,
                "constituency_code": constituency,
                "region_code": region,
                "lsoa_code": oa,
                "msoa_code": oa,
                "local_authority_code": local_authority,
                "ward_code": ward,
                "itl3_code": itl3,
            }
            for (
                oa,
                region,
                constituency,
                ward,
                local_authority,
                itl3,
            ), households in zip(rows, household_counts, strict=True)
        ]
    )


def _write_ladder(
    path: Path,
    *,
    household_counts: tuple[float, float, float, float] = (
        3.0,
        10.0,
        10.0,
        10.0,
    ),
):
    payload = assemble_uk_oa_ladder(
        _ladder_frame(household_counts),
        _ladder_metadata(),
    )
    np.savez_compressed(path, **payload)
    return load_uk_oa_ladder(path)


def _write_staging_h5(path: Path) -> None:
    household = pd.DataFrame(
        {
            "household_id": [1, 2, 3, 4],
            "household_weight": [3.0, 10.0, 10.0, 10.0],
            "region": [
                "LONDON",
                "WALES",
                "SCOTLAND",
                "NORTHERN_IRELAND",
            ],
        }
    )
    person = pd.DataFrame(
        {
            "person_id": [11, 21, 31, 41],
            "person_household_id": [1, 2, 3, 4],
            "person_benunit_id": [101, 201, 301, 401],
        }
    )
    benunit = pd.DataFrame({"benunit_id": [101, 201, 301, 401]})
    dataset = uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period="2023",
        weight_kind=WeightKind.IMPORTANCE,
        mass_log=(
            MassChangeRecord(
                entity="household",
                old_total=33.0,
                new_total=33.0,
                declared_factor=1.0,
                reason="Synthetic staging mass record.",
            ),
        ),
    )
    write_uk_national_frame(dataset, path)


def test_candidate_build_writes_calibrated_h5_and_evidence(tmp_path) -> None:
    pytest.importorskip("tables")
    pytest.importorskip("h5py")
    builder = _load_builder_module()
    input_h5 = tmp_path / "staging.h5"
    ladder_path = tmp_path / "ladder.npz"
    output_dir = tmp_path / "candidate"
    _write_staging_h5(input_h5)
    ladder = _write_ladder(ladder_path)

    assert (
        builder.main(
            [
                "--input-h5",
                str(input_h5),
                "--ladder",
                str(ladder_path),
                "--out",
                str(output_dir),
                "--n-clones",
                "2",
                "--seed",
                "7",
                "--epochs",
                "2",
            ]
        )
        == 0
    )

    candidate_h5 = output_dir / builder.CANDIDATE_FILENAME_TEMPLATE.format(
        source_year=2023
    )
    expected_sidecars = {
        builder.MANIFEST_FILENAME,
        builder.SOLVE_DIAGNOSTICS_FILENAME,
        builder.AREA_SUPPORT_FILENAME,
        builder.PAST_CAP_FILENAME,
    }
    assert candidate_h5.exists()
    assert expected_sidecars <= {path.name for path in output_dir.iterdir()}

    candidate_kind, candidate_mass_log = read_uk_single_year_weight_metadata(
        candidate_h5
    )
    with pd.HDFStore(candidate_h5, mode="r") as store:
        candidate_household = store["household"]
    assert candidate_kind is WeightKind.CALIBRATED
    assert candidate_household["source_year"].unique().tolist() == [2023]
    assert set(candidate_household["source_household_key"]) == {
        "2023:1",
        "2023:2",
        "2023:3",
        "2023:4",
    }
    assert len(candidate_mass_log) == 3
    calibration_records = [
        record
        for record in candidate_mass_log
        if "census_households/constituency" in record.reason
    ]
    assert calibration_records == [candidate_mass_log[-1]]
    assert candidate_mass_log[-1].declared_factor is None

    manifest = json.loads((output_dir / builder.MANIFEST_FILENAME).read_text())
    assert manifest["candidate_scope"] == "adjudicated_partial"
    assert manifest["bound_target_families"] == ["census_households/constituency"]
    assert manifest["ladder_target_provenance"] == ladder_target_provenance(ladder)
    assert manifest["gate"]["passed"] is True
    assert manifest["gate"]["phase"] == "post_calibration"
    assert manifest["gate"]["details"]
    assert (
        manifest["inputs"]["dataset"]["sha256"]
        == hashlib.sha256(input_h5.read_bytes()).hexdigest()
    )
    assert manifest["inputs"]["dataset"]["bytes"] == input_h5.stat().st_size
    assert (
        manifest["inputs"]["ladder"]["sha256"]
        == hashlib.sha256(ladder_path.read_bytes()).hexdigest()
    )
    assert manifest["inputs"]["ladder"]["bytes"] == ladder_path.stat().st_size
    assert manifest["parameters"]["n_clones"] == 2
    assert manifest["parameters"]["seed"] == 7
    assert manifest["parameters"]["source_year"] == 2023
    assert manifest["parameters"]["source_lineage_modulus"] is None
    assert manifest["parameters"]["epochs"] == 2
    assert manifest["parameters"]["learning_rate"] == pytest.approx(0.15)
    assert manifest["parameters"]["expected_constituency_vintage"] == "2024_pcon"
    assert [
        row["kind"] for row in manifest["weights"]["household_weight_kind_chain"]
    ] == ["importance", "importance", "calibrated"]
    assert manifest["weights"]["mass_log_records_before_calibration"] == 2
    assert manifest["weights"]["mass_log_records"] == 3
    mass_change = manifest["weights"]["calibration_mass_change"]
    assert mass_change["old_total"] == pytest.approx(33.0)
    assert mass_change["new_total"] == pytest.approx(
        candidate_household["household_weight"].sum()
    )
    assert mass_change["relative_shift"] == pytest.approx(
        (mass_change["new_total"] - 33.0) / 33.0
    )
    assert manifest["parameters"]["doctrine"] == {
        "target_loss_cap": 10.0,
        "max_weight_ratio": 100.0,
        "scale_rule": "default_target_loss_scales",
        "target_weight_rule": "uniform",
    }
    assert manifest["solve"]["n_targets"] == 4
    assert manifest["solve"]["n_households"] == 8
    assert np.isfinite(manifest["solve"]["initial_loss"])
    assert np.isfinite(manifest["solve"]["final_loss"])
    assert np.isfinite(manifest["solve"]["max_abs_relative_error"])
    assert np.isfinite(manifest["solve"]["median_abs_relative_error"])
    assert manifest["solve"]["past_cap"]["n_targets"] == 4
    assert manifest["support"]["min_assigned_households"] == 2
    assert manifest["support"]["min_nonzero_households"] == 2
    assert manifest["support"]["min_effective_sample_size"] == pytest.approx(2.0)

    diagnostics = pd.read_csv(output_dir / builder.SOLVE_DIAGNOSTICS_FILENAME)
    support = pd.read_csv(output_dir / builder.AREA_SUPPORT_FILENAME)
    past_cap = json.loads((output_dir / builder.PAST_CAP_FILENAME).read_text())
    assert len(diagnostics) == 4
    assert diagnostics["metric"].unique().tolist() == ["households"]
    assert len(support) == 4
    assert past_cap["n_targets"] == 4


def test_candidate_dry_run_plans_without_solve_or_write(
    monkeypatch,
    capsys,
    tmp_path,
) -> None:
    pytest.importorskip("tables")
    pytest.importorskip("h5py")
    builder = _load_builder_module()
    input_h5 = tmp_path / "staging.h5"
    ladder_path = tmp_path / "ladder.npz"
    output_dir = tmp_path / "dry-run-output"
    _write_staging_h5(input_h5)
    ladder = _write_ladder(ladder_path)

    def forbidden(*_args, **_kwargs):
        pytest.fail("dry run called a solve or dataset writer")

    monkeypatch.setattr(
        builder,
        "solve_uk_rowwise_weights_under_doctrine",
        forbidden,
    )
    monkeypatch.setattr(builder, "write_uk_rowwise_dataset", forbidden)

    assert (
        builder.main(
            [
                "--input-h5",
                str(input_h5),
                "--ladder",
                str(ladder_path),
                "--out",
                str(output_dir),
                "--n-clones",
                "2",
                "--seed",
                "7",
                "--dry-run",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    plan = json.loads(captured.out)
    assert plan["dry_run"] is True
    assert plan["bound_target_families"] == ["census_households/constituency"]
    assert plan["ladder_target_provenance"] == ladder_target_provenance(ladder)
    assert plan["shapes"]["person"][0] == 8
    assert plan["shapes"]["benunit"][0] == 8
    assert plan["shapes"]["household"][0] == 8
    assert plan["shapes"]["local_matrix"] == [4, 8]
    assert plan["target_count"] == 4
    assert not output_dir.exists()


def test_candidate_refuses_separate_assignment_and_target_ladders(
    tmp_path,
) -> None:
    pytest.importorskip("tables")
    pytest.importorskip("h5py")
    builder = _load_builder_module()
    input_h5 = tmp_path / "staging.h5"
    first_path = tmp_path / "assignment_ladder.npz"
    second_path = tmp_path / "target_ladder.npz"
    _write_staging_h5(input_h5)
    assignment_ladder = _write_ladder(first_path)
    target_ladder = _write_ladder(
        second_path,
        household_counts=(4.0, 9.0, 10.0, 10.0),
    )
    assignment = builder._clone_with_ladder_binding(
        input_h5,
        assignment_ladder,
        n_clones=2,
        seed=7,
        source_year=2023,
        expected_constituency_vintage="2024_pcon",
        source_lineage_modulus=None,
    )

    with pytest.raises(ValueError, match="same loaded"):
        builder._build_bound_problem(
            assignment,
            target_ladder=target_ladder,
        )


def test_candidate_dry_run_refuses_ladder_sidecar_collision(
    tmp_path,
) -> None:
    pytest.importorskip("tables")
    pytest.importorskip("h5py")
    builder = _load_builder_module()
    input_h5 = tmp_path / "staging.h5"
    output_dir = tmp_path / "candidate"
    temporary_ladder = tmp_path / "ladder.npz"
    ladder_path = output_dir / builder.MANIFEST_FILENAME
    _write_staging_h5(input_h5)
    _write_ladder(temporary_ladder)
    output_dir.mkdir()
    temporary_ladder.replace(ladder_path)
    ladder_bytes = ladder_path.read_bytes()

    with pytest.raises(ValueError, match="differ"):
        builder.main(
            [
                "--input-h5",
                str(input_h5),
                "--ladder",
                str(ladder_path),
                "--out",
                str(output_dir),
                "--dry-run",
            ]
        )

    assert ladder_path.read_bytes() == ladder_bytes
    assert list(output_dir.iterdir()) == [ladder_path]


def test_candidate_publication_rolls_back_on_interrupt(
    monkeypatch,
    tmp_path,
) -> None:
    builder = _load_builder_module()
    staging_dir = tmp_path / "staging"
    output_dir = tmp_path / "candidate"
    staging_dir.mkdir()
    output_paths = builder._output_paths(output_dir, source_year=2023)
    staged = {key: staging_dir / path.name for key, path in output_paths.items()}
    for path in staged.values():
        path.write_text("complete staged artifact\n")

    original_replace = Path.replace

    def interrupt_support(self, target):
        if Path(target) == output_paths["support"]:
            raise KeyboardInterrupt
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", interrupt_support)
    with pytest.raises(KeyboardInterrupt):
        builder._publish_staged_files(staged, output_paths)

    assert not output_dir.exists()
