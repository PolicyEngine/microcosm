"""Synthetic end-to-end contract for the first UK rowwise candidate."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

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
from microcosm.calibrate import TargetRegistry, TargetSpec
from microcosm.frame import MassChangeRecord, WeightKind


@pytest.fixture(autouse=True)
def _spool_only_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POPULACE_LEDGER_URL", raising=False)
    monkeypatch.delenv("POPULACE_LEDGER_KEY", raising=False)
    monkeypatch.delenv("POPULACE_LEDGER_API_KEY", raising=False)
    monkeypatch.delenv("POPULACE_LOGBOOK_PREV_ROW_DIGEST", raising=False)
    monkeypatch.setenv(
        "MICROCOSM_UK_TERMINAL_GATE_SIGNING_KEY",
        base64.b64encode(b"\x07" * 32).decode("ascii"),
    )


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


def _write_staging_h5(
    path: Path,
    *,
    households_per_region: int = 3,
    region_masses: tuple[float, float, float, float] = (3.0, 10.0, 10.0, 10.0),
) -> None:
    if households_per_region < 3:
        raise ValueError("spine fixture needs one raw row and two derivatives")
    region_names = (
        "LONDON",
        "WALES",
        "SCOTLAND",
        "NORTHERN_IRELAND",
    )
    household_ids = list(range(1, 4 * households_per_region + 1))
    source_household_ids: list[int] = []
    support_clone_indices: list[int] = []
    spi_flags: list[bool] = []
    for region_index in range(4):
        first = region_index * households_per_region + 1
        raw_count = households_per_region - 2
        source_household_ids.extend(range(first, first + raw_count))
        source_household_ids.extend([first, first])
        support_clone_indices.extend([0] * raw_count + [0, 1])
        spi_flags.extend([False] * raw_count + [True, False])
    household = pd.DataFrame(
        {
            "household_id": household_ids,
            "household_weight": [
                mass / households_per_region
                for mass in region_masses
                for _ in range(households_per_region)
            ],
            "region": [
                region for region in region_names for _ in range(households_per_region)
            ],
            "source_household_id": source_household_ids,
            "source_household_key": [
                f"2023:{source_id}" for source_id in source_household_ids
            ],
            "household_source_id": source_household_ids,
            "household_support_clone_index": support_clone_indices,
            "household_is_spi_synthetic": spi_flags,
            "household_is_capital_gains_clone": [False] * len(household_ids),
            "household_is_cgt_band_donor": [False] * len(household_ids),
        }
    )
    person_ids = [10_000 + household_id for household_id in household_ids]
    benunit_ids = [20_000 + household_id for household_id in household_ids]
    person = pd.DataFrame(
        {
            "person_id": person_ids,
            "person_household_id": household_ids,
            "person_benunit_id": benunit_ids,
        }
    )
    benunit = pd.DataFrame({"benunit_id": benunit_ids})
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


def test_candidate_build_writes_calibrated_h5_and_evidence(
    monkeypatch, tmp_path
) -> None:
    pytest.importorskip("tables")
    pytest.importorskip("h5py")
    builder = _load_builder_module()
    input_h5 = tmp_path / "staging.h5"
    ladder_path = tmp_path / "ladder.npz"
    output_dir = tmp_path / "candidate"
    _write_staging_h5(input_h5, households_per_region=52)
    ladder = _write_ladder(ladder_path)
    import microcosm.build.uk_runtime.battery_bindings as battery_bindings

    monkeypatch.setattr(
        battery_bindings,
        "_local_area_roster",
        lambda _resource, levels: {
            "constituency": tuple(sorted(set(ladder.constituency_code))),
            "local_authority": tuple(sorted(set(ladder.local_authority_code))),
        },
    )
    holdout = {
        "report_only": True,
        "method": "rotated_folds",
        "n_folds": 5,
        "seed": 20260529,
        "solve_seed": 7,
        "mean_holdout_loss": 0.1,
        "worst_holdout_loss": 0.2,
        "fold_losses": [0.1, 0.1, 0.2, 0.05, 0.05],
        "folds": [],
    }
    monkeypatch.setattr(
        builder,
        "rotated_uk_local_holdout",
        lambda *_args, **_kwargs: holdout,
    )

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
        calibration_year=2025
    )
    expected_sidecars = {
        builder.MANIFEST_FILENAME,
        builder.SOLVE_DIAGNOSTICS_FILENAME,
        builder.AREA_SUPPORT_FILENAME,
        builder.PAST_CAP_FILENAME,
        builder.CALIBRATION_DIAGNOSTICS_FILENAME,
        builder.LOCAL_REGISTRY_FILENAME,
        builder.LOCAL_GATE_REPORT_FILENAME_TEMPLATE.format(source_year=2023),
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
    assert len(set(candidate_household["source_household_key"])) == 200
    assert {"2023:1", "2023:206"} <= set(candidate_household["source_household_key"])
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
    assert adjudications["bound_families"] == ["census_households/constituency"]
    assert adjudications["evaluated_on"]
    seed = adjudications["stood_on"]["census_households/constituency"][
        "census_disclosure_control_noise"
    ]
    assert seed["approved_by"] == "juaristi22"
    assert seed["adjudication"] == "microcosm#802"
    assert seed["approved_on"] == "2026-08-31"
    assert seed["expires_on"] == "2026-11-30"
    assert adjudications["dormant"] == []
    cross_grain = manifest["cross_grain"]
    assert cross_grain["bound_national_targets"] == []
    assert cross_grain["bound_higher_targets"] == []
    assert cross_grain["inconsistencies_in_force"] == []
    assert cross_grain["groups"] == []
    assert cross_grain["absence"]
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
    assert manifest["solve"]["n_households"] == 416
    assert np.isfinite(manifest["solve"]["initial_loss"])
    assert np.isfinite(manifest["solve"]["final_loss"])
    assert np.isfinite(manifest["solve"]["max_abs_relative_error"])
    assert np.isfinite(manifest["solve"]["median_abs_relative_error"])
    assert manifest["solve"]["past_cap"]["n_targets"] == 4
    assert manifest["support"]["min_assigned_households"] == 104
    assert manifest["support"]["min_nonzero_households"] == 104
    assert manifest["support"]["min_effective_sample_size"] == pytest.approx(104.0)

    diagnostics = pd.read_csv(output_dir / builder.SOLVE_DIAGNOSTICS_FILENAME)
    support = pd.read_csv(output_dir / builder.AREA_SUPPORT_FILENAME)
    past_cap = json.loads((output_dir / builder.PAST_CAP_FILENAME).read_text())
    calibration_diagnostics = json.loads(
        (output_dir / builder.CALIBRATION_DIAGNOSTICS_FILENAME).read_text()
    )
    assert len(diagnostics) == 4
    assert diagnostics["metric"].unique().tolist() == ["households"]
    assert len(support) == 8
    assert past_cap["n_targets"] == 4
    assert calibration_diagnostics["schema_version"] == 6
    uk_diagnostics = calibration_diagnostics["uk_diagnostics"]
    assert len(uk_diagnostics["weakest_families"]) == 1
    assert len(uk_diagnostics["weakest_areas_by_fit"]["bottom_by_fit"]) == 4
    assert uk_diagnostics["weakest_areas_by_fit"]["n_areas_scored"] == 4
    assert {
        row["country"] for row in uk_diagnostics["weakest_areas_by_fit"]["countries"]
    } == {
        "England",
        "Northern Ireland",
        "Scotland",
        "Wales",
    }
    assert (
        manifest["diagnostics"]["weakest_families"]
        == uk_diagnostics["weakest_families"]
    )
    assert uk_diagnostics["rotated_holdout"] == holdout
    assert manifest["diagnostics"]["rotated_holdout"] == holdout
    assert "calibration_diagnostics" in manifest["outputs"]
    rows = _spool_rows(output_dir)
    assert len(rows) == 1
    row = rows[0]
    assert row.pipeline == "uk-local-candidate"
    assert row.rung == "f100"
    assert row.seed == 7
    assert row.disposition == "iterating"
    assert row.artifact_location == _local_ref(candidate_h5)
    assert set(row.gate_verdicts) == set(builder.UK_LOCAL_GATE_SCOPE)
    assert {item["verdict"] for item in row.gate_verdicts.values()} == {"passed"}
    assert all(
        ".local_gates.json#/gates/" in item["receipt"]
        for item in row.gate_verdicts.values()
    )


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
    assert plan["sampling"] == {
        "fraction": 1.0,
        "seed": 578,
        "rung_token": "f100",
        "sampled": False,
        "pre_household_count": 12,
        "post_household_count": 12,
    }
    assert plan["bound_target_families"] == ["census_households/constituency"]
    adjudications = plan["binding_adjudications"]
    assert adjudications["register_resource"] == "local_binding_adjudications.json"
    assert adjudications["bound_families"] == ["census_households/constituency"]
    assert adjudications["evaluated_on"]
    assert (
        "census_disclosure_control_noise"
        in adjudications["stood_on"]["census_households/constituency"]
    )
    assert adjudications["dormant"] == []
    cross_grain = plan["cross_grain"]
    assert cross_grain["bound_national_targets"] == []
    assert cross_grain["bound_higher_targets"] == []
    assert cross_grain["inconsistencies_in_force"] == []
    assert cross_grain["groups"] == []
    assert cross_grain["absence"]
    assert plan["ladder_target_provenance"] == ladder_target_provenance(ladder)
    assert plan["shapes"]["person"][0] == 24
    assert plan["shapes"]["benunit"][0] == 24
    assert plan["shapes"]["household"][0] == 24
    assert plan["shapes"]["local_matrix"] == [4, 24]
    assert plan["target_count"] == 4
    assert not output_dir.exists()
    assert not (output_dir / "logbook-spool").exists()


def test_candidate_sampling_rung_receipt_and_engine_block_validation(
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
    _write_ladder(ladder_path)

    def compact_sampler_forbidden(*_args, **_kwargs):
        pytest.fail("rowwise spine path called the certified-compact sampler")

    monkeypatch.setattr(
        builder,
        "sample_uk_national_frame",
        compact_sampler_forbidden,
        raising=False,
    )
    monkeypatch.setattr(
        builder,
        "sample_uk_spine_frame",
        lambda frame, **_kwargs: (
            frame,
            {
                "fraction": 0.01,
                "seed": 578,
                "rung_token": "f001",
                "pre_household_count": 12,
                "post_household_count": 12,
                "pre_family_count": 4,
                "post_family_count": 4,
                "normalization_factor": 1.0,
                "strata_count": 4,
                "receipt": {"synthetic_fixture": True},
            },
        ),
        raising=False,
    )

    assert (
        builder._parse_args(
            [
                "--input-h5",
                str(input_h5),
                "--ladder",
                str(ladder_path),
                "--out",
                str(output_dir),
            ]
        ).n_clones
        == 4
    )
    with pytest.raises(ValueError, match="must equal --n-clones"):
        builder.main(
            [
                "--input-h5",
                str(input_h5),
                "--ladder",
                str(ladder_path),
                "--out",
                str(output_dir),
                "--n-clones",
                "4",
                "--engine-blocks",
                "2",
                "--dry-run",
            ]
        )
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
                "--sample-fraction",
                "0.01",
                "--sample-seed",
                "578",
                "--dry-run",
            ]
        )
        == 0
    )
    plan = json.loads(capsys.readouterr().out)
    assert plan["sampling"]["fraction"] == 0.01
    assert plan["sampling"]["rung_token"] == "f001"
    assert plan["sampling"]["pre_household_count"] == 12
    assert plan["sampling"]["post_household_count"] >= 1
    assert plan["sampling"]["normalization_factor"] > 0


def test_candidate_f100_does_not_call_any_sampler(monkeypatch, tmp_path) -> None:
    pytest.importorskip("tables")
    builder = _load_builder_module()
    input_h5 = tmp_path / "staging.h5"
    _write_staging_h5(input_h5)
    frame, _ = builder.load_uk_national_frame(input_h5)

    def forbidden(*_args, **_kwargs):
        pytest.fail("f100 called a sampler")

    monkeypatch.setattr(builder, "sample_uk_national_frame", forbidden, raising=False)
    monkeypatch.setattr(builder, "sample_uk_spine_frame", forbidden, raising=False)

    sampled, receipt = builder._sample_candidate_frame(
        frame,
        fraction=1.0,
        seed=578,
    )

    assert sampled is frame
    assert receipt == {
        "fraction": 1.0,
        "seed": 578,
        "rung_token": "f100",
        "sampled": False,
        "pre_household_count": 12,
        "post_household_count": 12,
    }


def test_candidate_clone_count_planning_is_dry_run_only(tmp_path) -> None:
    builder = _load_builder_module()
    with pytest.raises(ValueError, match="only with --dry-run"):
        builder.main(
            [
                "--input-h5",
                str(tmp_path / "missing.h5"),
                "--ladder",
                str(tmp_path / "missing.npz"),
                "--out",
                str(tmp_path / "out"),
                "--candidate-clone-counts",
                "1,2,4",
            ]
        )


def test_candidate_engine_surface_reuses_one_resolver(
    monkeypatch,
    tmp_path,
) -> None:
    builder = _load_builder_module()
    input_h5 = tmp_path / "staging.h5"
    _write_staging_h5(input_h5)
    frame, _ = builder.load_uk_national_frame(input_h5)
    constructions = []

    class StubResolver:
        def __init__(self, **kwargs):
            constructions.append(kwargs)
            self.simulation = object()
            self.contract_targets = {}

        def receipt(self):
            return {
                "mode": "stub",
                "policyengine_uk_version": "test",
            }

    monkeypatch.setattr(
        builder,
        "compute_household_metrics",
        lambda _simulation, area_type, *, household_ids, **_kwargs: pd.DataFrame(
            {f"{area_type}_metric": np.ones(len(household_ids))},
            index=household_ids,
        ),
    )
    registry = TargetRegistry([], country="uk")
    prepared, restore, national, local_metrics, receipt = (
        builder._resolve_candidate_engine_surface(
            frame,
            registry,
            period=2025,
            scratch_dir=tmp_path / "scratch",
            resolver_factory=StubResolver,
        )
    )

    assert len(constructions) == 1
    assert receipt == {
        "mode": "stub",
        "engine_version": "test",
        "households": 12,
        "persons": 12,
        "benunits": 12,
        "national_inputs": 0,
        "local_metrics": {"constituency": 1, "la": 1},
        "blocks": 1,
    }
    assert set(local_metrics) == {"constituency", "la"}
    assert len(national.targets) == 0
    assert restore(prepared).table("household").equals(frame.table("household"))


def test_candidate_engine_surface_resolves_real_per_clone_blocks(
    monkeypatch,
    tmp_path,
) -> None:
    builder = _load_builder_module()
    input_h5 = tmp_path / "staging.h5"
    ladder_path = tmp_path / "ladder.npz"
    _write_staging_h5(input_h5)
    frame, _ = builder.load_uk_national_frame(input_h5)
    ladder = _write_ladder(ladder_path)
    clone = builder._clone_with_ladder_binding(
        frame,
        ladder,
        n_clones=2,
        seed=7,
        source_year=2023,
        expected_constituency_vintage="2024_pcon",
        source_lineage_modulus=None,
    ).result
    constructions = []

    class StubResolver:
        def __init__(self, **kwargs):
            constructions.append(kwargs)
            self.simulation = object()
            self.contract_targets = {}

        def receipt(self):
            return {"mode": "stub", "policyengine_uk_version": "test"}

    monkeypatch.setattr(
        builder,
        "compute_household_metrics",
        lambda _simulation, area_type, *, household_ids, **_kwargs: pd.DataFrame(
            {f"{area_type}_metric": np.arange(len(household_ids), dtype=float)},
            index=household_ids,
        ),
    )
    prepared, restore, _, metrics, receipt = builder._resolve_candidate_engine_surface(
        clone.frame,
        TargetRegistry([], country="uk"),
        period=2025,
        scratch_dir=tmp_path / "block-scratch",
        resolver_factory=StubResolver,
        blocks=2,
    )

    assert len(constructions) == 2
    assert [len(call["frame"].table("household")) for call in constructions] == [
        12,
        12,
    ]
    assert receipt["blocks"] == 2
    assert receipt["deviation"] == "per_clone_block_engine_resolution"
    assert (
        metrics["constituency"].index.tolist()
        == clone.frame.table("household")["household_id"].tolist()
    )
    assert restore(prepared).table("household").equals(clone.frame.table("household"))


def test_joint_candidate_f100_and_f001_end_to_end(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    """The driver solves one local/ladder/national matrix at both rung postures."""

    pytest.importorskip("tables")
    pytest.importorskip("h5py")
    builder = _load_builder_module()
    input_h5 = tmp_path / "staging.h5"
    ladder_path = tmp_path / "ladder.npz"
    _write_staging_h5(
        input_h5,
        households_per_region=200,
        region_masses=(4.0, 10.0, 10.0, 9.0),
    )
    ladder_frame = _ladder_frame()
    ladder_frame.loc[0, "households"] = 1.0
    english = ladder_frame.iloc[0].copy()
    extra_english = []
    for suffix in (2, 3):
        row = english.copy()
        row["oa_code"] = f"E0000000{suffix}"
        row["lsoa_code"] = row["oa_code"]
        row["msoa_code"] = row["oa_code"]
        row["constituency_code"] = f"E1400000{suffix}"
        row["local_authority_code"] = f"E0900000{suffix}"
        row["households"] = 1.0
        extra_english.append(row)
    ladder_frame = pd.concat(
        [ladder_frame, pd.DataFrame(extra_english)], ignore_index=True
    )
    payload = assemble_uk_oa_ladder(ladder_frame, _ladder_metadata())
    np.savez_compressed(ladder_path, **payload)
    ladder = load_uk_oa_ladder(ladder_path)

    from microcosm.build.uk_runtime.ledger_targets import UK_CROSS_GRAIN_BRIDGES

    household_bridge = UK_CROSS_GRAIN_BRIDGES[0]
    reviewed_missing = {
        "ons.household_composition.unrelated_adult_households",
        "ons.household_composition.lone_parent_non_dependent_children_households",
        "ons.household_composition.multi_family_households",
    }
    selected_composition = tuple(
        target_id
        for target_id in household_bridge.higher_target_ids
        if target_id not in reviewed_missing
    )
    fanout_target_id = "dwp.uc.payment_distribution_single"
    fanout_names = tuple(f"payment-band-{index}" for index in range(3))
    national_registry = TargetRegistry(
        [
            *[
                TargetSpec(
                    name=target_id,
                    entity="household",
                    measure=f"national/composition_{index}",
                    value=33.0,
                    period=2025,
                    source="synthetic national fixture",
                    family="ons",
                    metadata={
                        "contract_target_id": target_id,
                        "ledger_geography_level": "country",
                        "ledger_geography_id": "K02000001",
                    },
                )
                for index, target_id in enumerate(selected_composition)
            ],
            *[
                TargetSpec(
                    name=name,
                    entity="household",
                    measure=f"national/payment_band_{index}",
                    value=33.0,
                    period=2025,
                    source="synthetic fan-out fixture",
                    family="dwp_uc",
                    metadata={
                        "contract_target_id": fanout_target_id,
                        "ledger_geography_level": "country",
                        "ledger_geography_id": "K03000001",
                    },
                )
                for index, name in enumerate(fanout_names)
            ],
        ],
        country="uk",
    )
    local_registry = TargetRegistry(
        [
            TargetSpec(
                name="ons.tenure.owned_outright@E09000001",
                entity="household",
                measure="tenure/owned_outright",
                value=1.0,
                period=2025,
                source="synthetic local fact fixture",
                family="ons",
                metadata={
                    "contract_target_id": "ons.tenure.owned_outright",
                    "geography_level": "local_authority",
                    "geography_id": "E09000001",
                    "ledger_fact_period": "2023",
                },
            )
        ],
        country="uk",
    )
    artifact = SimpleNamespace(
        provenance=lambda: {
            "facts_sha256": "1" * 64,
            "manifest_sha256": "2" * 64,
            "artifact_id": "synthetic-joint-fixture",
        }
    )
    joint_inputs = {
        "artifact": artifact,
        "calibration_year": 2025,
        "national_registry": national_registry,
        "band_edge_registry": national_registry,
        "local_registry": local_registry,
        "measure_exclusions": {
            f"compiled::{target_id}": {
                "tracking": "microcosm#791",
                "reason": "relationship-to-head is unavailable",
            }
            for target_id in reviewed_missing
        },
        "reviewed_unbound_higher_targets": {
            target_id: {
                "tracking": "microcosm#791",
                "reason": "relationship-to-head is unavailable",
            }
            for target_id in reviewed_missing
        },
    }
    monkeypatch.setattr(
        builder, "_load_joint_target_inputs", lambda _args: joint_inputs
    )

    constructions = []

    class StubResolver:
        def __init__(self, **kwargs):
            constructions.append(kwargs)
            self.frame = kwargs["frame"]
            self.simulation = object()
            self.contract_targets = {}

        def receipt(self):
            return {"mode": "stub", "policyengine_uk_version": "test"}

    monkeypatch.setattr(
        builder,
        "resolve_target_measures",
        lambda _factory, _registry, provider, **_kwargs: SimpleNamespace(
            measure_inputs={
                (spec.entity, spec.measure): np.ones(
                    len(provider.frame.table("household")), dtype=float
                )
                for spec in _registry.specs
            }
        ),
    )
    monkeypatch.setattr(
        builder,
        "materialize_uk_ledger_targets",
        lambda *_args, **_kwargs: SimpleNamespace(skipped=()),
    )
    monkeypatch.setattr(
        builder,
        "compute_household_metrics",
        lambda _simulation, area_type, *, household_ids, **_kwargs: pd.DataFrame(
            {
                "households": np.ones(len(household_ids), dtype=float),
                **(
                    {"tenure/owned_outright": np.ones(len(household_ids), dtype=float)}
                    if area_type == "la"
                    else {}
                ),
            },
            index=household_ids,
        ),
    )
    real_resolve = builder._resolve_candidate_engine_surface
    monkeypatch.setattr(
        builder,
        "_resolve_candidate_engine_surface",
        lambda *args, **kwargs: real_resolve(
            *args, resolver_factory=StubResolver, **kwargs
        ),
    )
    monkeypatch.setattr(
        builder,
        "rotated_uk_local_holdout",
        lambda *_args, **_kwargs: {"report_only": True, "folds": []},
    )
    import microcosm.build.uk_runtime.battery_bindings as battery_bindings

    monkeypatch.setattr(
        battery_bindings,
        "_local_area_roster",
        lambda _resource, levels: {
            "constituency": tuple(sorted(set(ladder.constituency_code))),
            "local_authority": tuple(sorted(set(ladder.local_authority_code))),
        },
    )
    real_support_summary = builder.uk_ladder_area_support_summary

    def support_summary(household, ladder_arg):
        if "household_weight" in household:
            return real_support_summary(household, ladder_arg)
        support = pd.DataFrame(
            {
                "nonzero_households": [len(household)],
                "effective_sample_size": [float(len(household))],
                "nonzero_source_households": [
                    household["source_household_id"].nunique()
                ],
            }
        )
        return {"constituency": support, "la": support}

    monkeypatch.setattr(builder, "uk_ladder_area_support_summary", support_summary)

    dry_out = tmp_path / "joint-dry"
    assert (
        builder.main(
            [
                "--input-h5",
                str(input_h5),
                "--ladder",
                str(ladder_path),
                "--out",
                str(dry_out),
                "--n-clones",
                "2",
                "--dry-run",
            ]
        )
        == 0
    )
    dry_plan = json.loads(capsys.readouterr().out)
    dry_unbound = dry_plan["cross_grain"]["unbound_bridges"]
    assert [entry["bridge_id"] for entry in dry_unbound] == [household_bridge.bridge_id]
    assert dry_unbound[0]["missing"] == sorted(reviewed_missing)
    dry_fanout = dry_plan["cross_grain"]["fanout_controls_summed"]
    assert dry_fanout == [
        {
            "target_id": fanout_target_id,
            "geography_id": "K03000001",
            "cells": 3,
            "cell_names": list(fanout_names),
            "total": 99.0,
        }
    ]
    assert not dry_out.exists()

    f100_out = tmp_path / "joint-f100"
    assert (
        builder.main(
            [
                "--input-h5",
                str(input_h5),
                "--ladder",
                str(ladder_path),
                "--out",
                str(f100_out),
                "--n-clones",
                "2",
                "--epochs",
                "2",
                "--skip-holdout",
            ]
        )
        == 0
    )
    f100 = json.loads((f100_out / builder.MANIFEST_FILENAME).read_text())
    assert f100["schema_version"] == 2
    assert f100["solve"]["n_targets_by_kind"] == {
        "local": 1,
        "ladder": 12,
        "national": len(national_registry.specs),
    }
    assert f100["solve"]["n_targets"] == 13 + len(national_registry.specs)
    assert f100["solve"]["measure_resolution"]["mode"] == "stub"
    assert f100["cross_grain"]["unbound_bridges"] == dry_unbound
    assert f100["solve"]["cross_grain"]["unbound_bridges"] == dry_unbound
    assert f100["cross_grain"]["fanout_controls_summed"] == dry_fanout
    assert f100["solve"]["cross_grain"]["fanout_controls_summed"] == dry_fanout
    assert f100["releasable"] is True
    assert _spool_rows(f100_out)[0].rung == "f100"

    f001_out = tmp_path / "joint-f001"
    assert (
        builder.main(
            [
                "--input-h5",
                str(input_h5),
                "--ladder",
                str(ladder_path),
                "--out",
                str(f001_out),
                "--n-clones",
                "1",
                "--sample-fraction",
                "0.01",
                "--epochs",
                "2",
                "--skip-holdout",
            ]
        )
        == 0
    )
    f001 = json.loads((f001_out / builder.MANIFEST_FILENAME).read_text())
    assert f001["rung_surface"]["dropped_cells"] > 0
    assert f001["rung_surface"]["dropped_by_grain"]["constituency"] >= 1
    assert f001["rung_surface"]["dropped_by_grain"]["la"] >= 1
    assert "uk_local_area_support" in f001["failing_gate_ids"]
    assert f001["releasable"] is False
    assert _spool_rows(f001_out)[0].rung == "f001"
    assert len(constructions) == 2


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
            name="spine_agreement",
            passed=False,
            failures=("post-calibration coverage failed",),
            details={"minimum": 0},
        )

    original = builder.UK_GATE_REGISTRY["spine_agreement"]
    monkeypatch.setattr(
        builder,
        "UK_GATE_REGISTRY",
        {
            **builder.UK_GATE_REGISTRY,
            "spine_agreement": replace(original, evaluator=failing_gate),
        },
    )

    with pytest.raises(
        builder.GateBatteryBlockedError, match="post-calibration coverage failed"
    ):
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
    gate_report_path = output_dir / builder.LOCAL_GATE_REPORT_FILENAME_TEMPLATE.format(
        source_year=2023
    )
    assert gate_report_path.exists()
    assert row.gate_verdicts["uk_local_geography_ladder_post_calibration"] == {
        "verdict": "failed",
        "receipt": (
            f"{_local_ref(gate_report_path)}"
            "#/gates/uk_local_geography_ladder_post_calibration"
        ),
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
    assert "targets_bound" in row.phases_reached
    assert "solved" not in row.phases_reached
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
    output_paths = builder._output_paths(
        output_dir,
        source_year=2023,
        calibration_year=2025,
    )
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
