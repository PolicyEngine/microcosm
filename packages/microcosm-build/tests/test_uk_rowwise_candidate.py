"""Synthetic end-to-end contract for the first UK rowwise candidate."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.logbook import LOGBOOK_ROW_FIELDS, load_spool_rows
from microcosm.build.uk_runtime import (
    assemble_uk_oa_ladder,
    ladder_target_provenance,
    load_uk_oa_ladder,
    read_uk_single_year_weight_metadata,
    write_uk_national_frame,
)
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.frame import MassChangeRecord, WeightKind


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
    # The kernel-minted record declares the realized factor (the hand-minted
    # predecessor left it None) — declared-vs-realized is validated by the
    # kernel at with_weights time.
    record = candidate_mass_log[-1]
    assert record.declared_factor == pytest.approx(record.new_total / record.old_total)

    manifest = json.loads((output_dir / builder.MANIFEST_FILENAME).read_text())
    assert manifest["candidate_scope"] == "adjudicated_partial"
    assert manifest["bound_target_families"] == ["census_households/constituency"]
    adjudications = manifest["binding_adjudications"]
    assert adjudications["register_resource"] == "local_binding_adjudications.json"
    assert adjudications["bound_families"] == [
        "census_households/constituency"
    ]
    assert adjudications["evaluated_on"]
    seed = adjudications["stood_on"]["census_households/constituency"][
        "census_disclosure_control_noise"
    ]
    assert seed["approved_by"] == "juaristi22"
    assert seed["approved_on"] == "2026-08-27"
    assert seed["expires_on"] == "2026-11-27"
    assert adjudications["dormant"] == []
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
    rows = _spool_rows(output_dir)
    assert len(rows) == 1
    row = rows[0]
    assert row.pipeline == "uk-locals-candidate"
    assert row.rung == "f100"
    assert row.seed == 7
    assert row.disposition == "iterating"
    assert row.artifact_location == _local_ref(candidate_h5)
    assert row.gate_verdicts == {
        "uk_geography_ladder_post_calibration": {
            "verdict": "passed",
            "receipt": f"{_local_ref(output_dir / builder.MANIFEST_FILENAME)}#/gate",
        },
        "uk_target_fit": {
            "verdict": "passed",
            "receipt": (
                f"{_local_ref(output_dir / builder.MANIFEST_FILENAME)}"
                "#/solve/max_abs_relative_error"
            ),
        },
        "uk_area_support": {
            "verdict": "passed",
            "receipt": f"{_local_ref(output_dir / builder.MANIFEST_FILENAME)}#/support",
        },
    }


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
    adjudications = plan["binding_adjudications"]
    assert adjudications["register_resource"] == "local_binding_adjudications.json"
    assert adjudications["bound_families"] == [
        "census_households/constituency"
    ]
    assert adjudications["evaluated_on"]
    assert (
        "census_disclosure_control_noise"
        in adjudications["stood_on"]["census_households/constituency"]
    )
    assert adjudications["dormant"] == []
    assert plan["ladder_target_provenance"] == ladder_target_provenance(ladder)
    assert plan["shapes"]["person"][0] == 8
    assert plan["shapes"]["benunit"][0] == 8
    assert plan["shapes"]["household"][0] == 8
    assert plan["shapes"]["local_matrix"] == [4, 8]
    assert plan["target_count"] == 4
    assert not output_dir.exists()
    assert not (output_dir / "logbook-spool").exists()


def test_candidate_refusal_records_receipt_and_reraises(
    monkeypatch,
    tmp_path,
) -> None:
    pytest.importorskip("tables")
    pytest.importorskip("h5py")
    builder = _load_builder_module()
    input_h5 = tmp_path / "staging.h5"
    ladder_path = tmp_path / "ladder.npz"
    output_dir = tmp_path / "candidate"
    _write_staging_h5(input_h5)
    _write_ladder(ladder_path)

    def failing_gate(*_args, **_kwargs):
        return builder.GateResult(
            name="uk_geography_ladder",
            passed=False,
            failures=("post-calibration coverage failed",),
            details={"minimum": 0},
        )

    monkeypatch.setattr(builder, "uk_geography_ladder_gate", failing_gate)

    with pytest.raises(ValueError, match="post-calibration coverage failed"):
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

    rows = _spool_rows(output_dir)
    assert len(rows) == 1
    row = rows[0]
    assert row.disposition == "failed"
    refusal_path = (
        output_dir / "logbook-receipts" / row.build_id / "candidate-refusal.json"
    )
    assert refusal_path.exists()
    refusal = json.loads(refusal_path.read_text())
    assert refusal["gate"]["phase"] == "post_calibration"
    assert refusal["gate"]["passed"] is False
    assert row.gate_verdicts["uk_geography_ladder_post_calibration"] == {
        "verdict": "failed",
        "receipt": f"{_local_ref(refusal_path)}#/gate",
    }
    assert row.gate_verdicts["pipeline_error"]["verdict"] == "error"
    assert row.gate_verdicts["pipeline_error"]["receipt"].endswith("#/error_type")


def test_candidate_binding_adjudication_failure_records_failed_row(
    monkeypatch,
    tmp_path,
) -> None:
    pytest.importorskip("tables")
    pytest.importorskip("h5py")
    builder = _load_builder_module()
    input_h5 = tmp_path / "staging.h5"
    ladder_path = tmp_path / "ladder.npz"
    output_dir = tmp_path / "candidate"
    _write_staging_h5(input_h5)
    _write_ladder(ladder_path)

    import microcosm.build.uk_runtime.local_rowwise as local_rowwise

    monkeypatch.setattr(
        local_rowwise,
        "load_uk_reviewed_exclusion_register",
        lambda *_args, **_kwargs: {},
    )

    def unexpected_solve(*_args, **_kwargs):
        raise AssertionError("solve should not run without adjudication")

    monkeypatch.setattr(
        builder,
        "solve_uk_rowwise_weights_under_doctrine",
        unexpected_solve,
    )

    with pytest.raises(ValueError, match="census_disclosure_control_noise"):
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

    rows = _spool_rows(output_dir)
    assert len(rows) == 1
    row = rows[0]
    assert row.disposition == "failed"
    assert "cloned" in row.phases_reached
    assert "targets_bound" not in row.phases_reached
    assert row.gate_verdicts["pipeline_error"]["verdict"] == "error"
    assert row.gate_verdicts["pipeline_error"]["receipt"].endswith("#/error_type")


def test_candidate_setup_failure_records_failed_row(monkeypatch, tmp_path) -> None:
    """A pre-solve setup failure (ladder load) still spools a failed row.

    Adversarial-review finding on #666: input verification, frame/ladder
    loading, cloning, and target binding used to run before the recording
    envelope opened, so their failures escaped with no Logbook row.
    """

    pytest.importorskip("tables")
    pytest.importorskip("h5py")
    builder = _load_builder_module()
    input_h5 = tmp_path / "staging.h5"
    ladder_path = tmp_path / "ladder.npz"
    output_dir = tmp_path / "candidate"
    _write_staging_h5(input_h5)
    _write_ladder(ladder_path)

    def failing_ladder_load(_path):
        raise RuntimeError("ladder artifact refused to parse")

    monkeypatch.setattr(builder, "load_uk_oa_ladder", failing_ladder_load)

    with pytest.raises(RuntimeError, match="ladder artifact refused to parse"):
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

    rows = _spool_rows(output_dir)
    assert len(rows) == 1
    row = rows[0]
    assert row.disposition == "failed"
    assert row.gate_verdicts["pipeline_error"]["verdict"] == "error"
    assert row.gate_verdicts["pipeline_error"]["receipt"].endswith("#/error_type")
    assert "inputs_pinned" in row.phases_reached
    assert "cloned" not in row.phases_reached
    # Real input pins were promoted before the failure; the preflight
    # placeholder digest must not survive into the row.
    assert row.input_pins_digest != builder.preflight_digest(
        builder._UK_CANDIDATE_PIPELINE
    )


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
