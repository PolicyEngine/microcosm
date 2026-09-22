"""Synthetic end-to-end contract for the first UK rowwise candidate."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import shutil
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
from microcosm.build.uk_runtime.national_frame import (
    uk_national_frame,
    validate_uk_national_frame,
)
from microcosm.calibrate import (
    CalibrationHierarchy,
    HierarchyCategory,
    HierarchyGeography,
    HierarchyNode,
    TargetRegistry,
    TargetSpec,
)
from microcosm.frame import MassChangeRecord, WeightKind


@pytest.fixture(autouse=True)
def _empty_support_exclusions_for_synthetic_rosters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Synthetic candidates never carry the committed micro-LA exclusions.

    The committed ``local_area_support_exclusions.json`` names real local
    authorities measured on the licensed spine; a synthetic roster either
    lacks them (unknown) or meets the floor (stale), and the gate rightly
    fails either way. These tests exercise the machinery, so the support
    register is pinned empty here; the committed entries are covered by
    ``test_uk_battery_bindings.py``.
    """

    import microcosm.build.uk_runtime.battery_bindings as battery_bindings

    real_loader = battery_bindings.load_uk_local_area_support_exclusion_register

    def _loader(path, *, resource, **kwargs):
        if resource == "local_area_support_exclusions.json":
            return {"exclusions": {}, "bound_despite_support_floor": {}}
        return real_loader(path, resource=resource, **kwargs)

    monkeypatch.setattr(
        battery_bindings, "load_uk_local_area_support_exclusion_register", _loader
    )


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


def _fixture_hierarchy(
    name: str,
    *,
    provider_id: str,
    provider_label: str,
    category_id: str,
    category_label: str,
    geography_id: str,
    geography_label: str,
    geography_level: str,
    target_label: str,
) -> CalibrationHierarchy:
    return CalibrationHierarchy(
        provider=HierarchyNode(provider_id, provider_label),
        category=HierarchyCategory(category_id, category_label, provider_id),
        geography=HierarchyGeography(
            geography_id,
            geography_label,
            geography_level,
        ),
        dimensions=(),
        target=HierarchyNode(name, target_label),
    )


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


def _household_specs_for_ladder(ladder) -> list[TargetSpec]:
    specs = []
    for level, codes in (
        ("constituency", ladder.constituency_code),
        ("local_authority", ladder.local_authority_code),
    ):
        grouped = (
            pd.DataFrame({"code": codes, "value": ladder.households})
            .groupby("code", sort=True)["value"]
            .sum()
        )
        for area_code, value in grouped.items():
            census_year = 2022 if str(area_code).startswith("S") else 2021
            specs.append(
                TargetSpec(
                    name=f"ons.census.households@{area_code}",
                    entity="household",
                    measure="households",
                    value=float(value),
                    period=2025,
                    source="synthetic Chronicle fixture",
                    family="census_households",
                    metadata={
                        "contract_target_id": "ons.census.households",
                        "geography_level": level,
                        "geography_id": str(area_code),
                        "uprating_from_period": census_year,
                        "uprating_to_period": 2025,
                    },
                    hierarchy=_fixture_hierarchy(
                        f"ons.census.households@{area_code}",
                        provider_id="ons",
                        provider_label="Office for National Statistics",
                        category_id="ons.household_composition",
                        category_label="Household composition",
                        geography_id=str(area_code),
                        geography_label=f"Area {area_code}",
                        geography_level=level,
                        target_label="Occupied households",
                    ),
                )
            )
    return specs


def _mandatory_input_flags(input_h5: Path, ladder_path: Path) -> list[str]:
    return [
        "--input-sha256",
        hashlib.sha256(input_h5.read_bytes()).hexdigest(),
        "--ladder-sha256",
        hashlib.sha256(ladder_path.read_bytes()).hexdigest(),
        "--ledger-facts",
        str(ladder_path.parent / "synthetic-ledger"),
        "--ledger-facts-sha256",
        "1" * 64,
        "--ledger-manifest-sha256",
        "2" * 64,
        # Tests never reach the Hub: telemetry and the staged bundle stay local.
        "--staging-local-only",
    ]


def _configure_households_only_inputs(
    builder,
    monkeypatch: pytest.MonkeyPatch,
    *,
    input_h5: Path,
    ladder_path: Path,
) -> list[str]:
    ladder = load_uk_oa_ladder(ladder_path)
    artifact = SimpleNamespace(
        facts=None,
        provenance=lambda: {
            "facts_sha256": "1" * 64,
            "manifest_sha256": "2" * 64,
            "artifact_id": "synthetic-households-only-fixture",
        },
    )
    joint_inputs = {
        "artifact": artifact,
        "calibration_year": 2025,
        "national_registry": TargetRegistry([], country="uk"),
        "band_edge_registry": TargetRegistry([], country="uk"),
        "local_registry": TargetRegistry(
            _household_specs_for_ladder(ladder), country="uk"
        ),
        "measure_exclusions": {},
        "reviewed_unbound_higher_targets": {},
    }
    monkeypatch.setattr(
        builder, "_load_joint_target_inputs", lambda _args: joint_inputs
    )
    return [
        *_mandatory_input_flags(input_h5, ladder_path),
        "--households-only",
    ]


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
    household_flags = _configure_households_only_inputs(
        builder, monkeypatch, input_h5=input_h5, ladder_path=ladder_path
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
                "--release-role",
                "dense",
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
                *household_flags,
            ]
        )
        == 0
    )

    candidate_h5 = output_dir / "microcosm_uk_2024_25_local.h5"
    expected_sidecars = {
        builder.MANIFEST_FILENAME,
        builder.SOLVE_DIAGNOSTICS_FILENAME,
        builder.AREA_SUPPORT_FILENAME,
        builder.PAST_CAP_FILENAME,
        builder.CALIBRATION_DIAGNOSTICS_FILENAME,
        builder.LOCAL_REGISTRY_FILENAME,
        "microcosm_uk_2024_25_local.local_gates.json",
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
    # The manifest declares the role it was built under (microcosm#823): the
    # dense pre-flight and assembler refuse any other, and the parameters
    # carry the role's doctrine block verbatim.
    assert manifest["schema_version"] == 4
    assert manifest["release_role"] == "dense"
    assert manifest["release_id"] == "microcosm-uk-2024-25-dense"
    assert manifest["parameters"]["release_role"] == "dense"
    assert manifest["parameters"]["n_clones"] == 2
    assert manifest["parameters"]["doctrine"] == (
        builder.UK_ROWWISE_DENSE_POSTURE.doctrine_bounds()
    )
    assert manifest["outputs"]["dataset"]["path"].endswith(
        "microcosm_uk_2024_25_local.h5"
    )
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
    assert adjudications["dormant"] == [
        "full_frs_tei_band_unavailable",
        "hmrc_spi_frame_model_proxy",
        "population_universe_private_households",
        "uc_unit_vs_household_grain",
        "voa_dwellings_vs_household_frame",
    ]
    cross_grain = manifest["cross_grain"]
    assert cross_grain["bound_national_targets"] == []
    assert cross_grain["bound_higher_targets"] == []
    assert cross_grain["inconsistencies_in_force"] == []
    assert cross_grain["groups"] == []
    assert cross_grain["empty_legs_licensed"] == []
    assert cross_grain["controls_without_lower_rows"] == []
    assert cross_grain["absence"]
    assert manifest["ladder_assignment_provenance"] == ladder_target_provenance(ladder)
    assert manifest["identity"]["targets"]["paired_ladder_sha256"] == (
        hashlib.sha256(ladder_path.read_bytes()).hexdigest()
    )
    assert manifest["identity"]["targets"]["chronicle"]["artifact_id"] == (
        "synthetic-households-only-fixture"
    )
    assert manifest["household_dispersion"]["countries"]
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
        "max_weight_ratio": 10.0,
        "scale_rule": "default_target_loss_scales",
        "target_weight_rule": "grain_equal",
        "solve_epochs": 1500,
        "clone_count": 15,
    }
    assert manifest["census_household_uprating"]["applied"] is False
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
    assert calibration_diagnostics["schema_version"] == 8
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
    household_flags = _configure_households_only_inputs(
        builder, monkeypatch, input_h5=input_h5, ladder_path=ladder_path
    )

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
                "--release-role",
                "dense",
                "--ladder",
                str(ladder_path),
                "--out",
                str(output_dir),
                "--n-clones",
                "2",
                "--seed",
                "7",
                "--dry-run",
                *household_flags,
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
    assert adjudications["dormant"] == [
        "full_frs_tei_band_unavailable",
        "hmrc_spi_frame_model_proxy",
        "population_universe_private_households",
        "uc_unit_vs_household_grain",
        "voa_dwellings_vs_household_frame",
    ]
    cross_grain = plan["cross_grain"]
    assert cross_grain["bound_national_targets"] == []
    assert cross_grain["bound_higher_targets"] == []
    assert cross_grain["inconsistencies_in_force"] == []
    assert cross_grain["groups"] == []
    assert cross_grain["empty_legs_licensed"] == []
    assert cross_grain["controls_without_lower_rows"] == []
    assert cross_grain["absence"]
    assert plan["ladder_assignment_provenance"] == ladder_target_provenance(ladder)
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
    household_flags = _configure_households_only_inputs(
        builder, monkeypatch, input_h5=input_h5, ladder_path=ladder_path
    )

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
                "--release-role",
                "dense",
                "--ladder",
                str(ladder_path),
                "--out",
                str(output_dir),
            ]
        ).n_clones
        == builder.UK_ROWWISE_DENSE_POSTURE.clone_count
    )
    with pytest.raises(ValueError, match="must equal --n-clones"):
        builder.main(
            [
                "--input-h5",
                str(input_h5),
                "--release-role",
                "dense",
                "--ladder",
                str(ladder_path),
                "--out",
                str(output_dir),
                "--n-clones",
                "4",
                "--engine-blocks",
                "2",
                "--dry-run",
                *household_flags,
            ]
        )
    assert (
        builder.main(
            [
                "--input-h5",
                str(input_h5),
                "--release-role",
                "dense",
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
                *household_flags,
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
                "--release-role",
                "dense",
                "--ladder",
                str(tmp_path / "missing.npz"),
                "--out",
                str(tmp_path / "out"),
                "--candidate-clone-counts",
                "1,2,4",
                "--input-sha256",
                "0" * 64,
                "--ladder-sha256",
                "0" * 64,
                "--ledger-facts",
                str(tmp_path / "ledger"),
                "--ledger-facts-sha256",
                "0" * 64,
                "--ledger-manifest-sha256",
                "0" * 64,
            ]
        )


def test_candidate_engine_surface_reuses_one_resolver(
    monkeypatch,
    tmp_path,
) -> None:
    pytest.importorskip("tables")
    builder = _load_builder_module()
    input_h5 = tmp_path / "staging.h5"
    _write_staging_h5(input_h5)
    frame, _ = builder.load_uk_national_frame(input_h5)
    constructions = []
    cgt_period_contract = {
        "version": "uk-cgt-measurement-v2",
        "input_period": "2024",
        "calibration_period": 2025,
        "bound_measurements": {
            "cgt_2024_gains": {
                "model_variable": "capital_gains",
                "measurement_period": 2024,
            }
        },
    }

    class StubResolver:
        def __init__(self, **kwargs):
            constructions.append(kwargs)
            self.simulation = object()
            self.contract_targets = {}

        def receipt(self):
            return {
                "mode": "stub",
                "policyengine_uk_version": "test",
                "cgt_period_contract": cgt_period_contract,
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
        "cgt_period_contract": cgt_period_contract,
    }
    assert set(local_metrics) == {"constituency", "la"}
    assert len(national.targets) == 0
    assert restore(prepared).table("household").equals(frame.table("household"))


@pytest.mark.parametrize("second_cgt_period", [2024, 2025, None])
def test_candidate_engine_surface_resolves_real_per_clone_blocks(
    monkeypatch,
    tmp_path,
    second_cgt_period,
) -> None:
    pytest.importorskip("tables")
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
            self.cgt_period = 2024 if len(constructions) == 1 else second_cgt_period

        def receipt(self):
            receipt = {"mode": "stub", "policyengine_uk_version": "test"}
            if self.cgt_period is not None:
                receipt["cgt_period_contract"] = {
                    "version": "uk-cgt-measurement-v2",
                    "input_period": "2024",
                    "calibration_period": 2025,
                    "bound_measurements": {
                        "cgt_2024_gains": {
                            "model_variable": "capital_gains",
                            "measurement_period": self.cgt_period,
                        }
                    },
                }
            return receipt

    monkeypatch.setattr(
        builder,
        "compute_household_metrics",
        lambda _simulation, area_type, *, household_ids, **_kwargs: pd.DataFrame(
            {f"{area_type}_metric": np.arange(len(household_ids), dtype=float)},
            index=household_ids,
        ),
    )

    def resolve():
        return builder._resolve_candidate_engine_surface(
            clone.frame,
            TargetRegistry([], country="uk"),
            period=2025,
            scratch_dir=tmp_path / "block-scratch",
            resolver_factory=StubResolver,
            blocks=2,
        )

    if second_cgt_period != 2024:
        with pytest.raises(RuntimeError, match="CGT period contract is inconsistent"):
            resolve()
        return

    prepared, restore, _, metrics, receipt = resolve()

    assert len(constructions) == 2
    assert [len(call["frame"].table("household")) for call in constructions] == [
        12,
        12,
    ]
    assert receipt["blocks"] == 2
    assert receipt["cgt_period_contract"]["bound_measurements"] == {
        "cgt_2024_gains": {
            "model_variable": "capital_gains",
            "measurement_period": 2024,
        }
    }
    assert receipt["deviation"] == "per_clone_block_engine_resolution"
    sensitivity = receipt["block_sensitivity"]
    assert (
        "ons/corporate_land_value"
        in sensitivity["known_population_normalised_measures"]
    )
    assert set(sensitivity["present_in_this_run"]) <= set(
        sensitivity["known_population_normalised_measures"]
    )
    assert "not evidence for adjudication" in sensitivity["caveat"]
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
                    hierarchy=_fixture_hierarchy(
                        target_id,
                        provider_id="ons",
                        provider_label="Office for National Statistics",
                        category_id="ons.household_composition",
                        category_label="Household composition",
                        geography_id="K02000001",
                        geography_label="United Kingdom",
                        geography_level="country",
                        target_label="Household composition",
                    ),
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
                    hierarchy=_fixture_hierarchy(
                        name,
                        provider_id="dwp",
                        provider_label="Department for Work and Pensions",
                        category_id="dwp.universal_credit",
                        category_label="Universal Credit",
                        geography_id="K03000001",
                        geography_label="Great Britain",
                        geography_level="country",
                        target_label="Universal Credit payment distribution",
                    ),
                )
                for index, name in enumerate(fanout_names)
            ],
        ],
        country="uk",
    )
    local_registry = TargetRegistry(
        [
            *_household_specs_for_ladder(ladder),
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
                hierarchy=_fixture_hierarchy(
                    "ons.tenure.owned_outright@E09000001",
                    provider_id="ons",
                    provider_label="Office for National Statistics",
                    category_id="ons.housing",
                    category_label="Housing",
                    geography_id="E09000001",
                    geography_label="City of London",
                    geography_level="local_authority",
                    target_label="Owned outright",
                ),
            ),
        ],
        country="uk",
    )
    artifact = SimpleNamespace(
        facts=None,
        provenance=lambda: {
            "facts_sha256": "1" * 64,
            "manifest_sha256": "2" * 64,
            "artifact_id": "synthetic-joint-fixture",
        },
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
    monkeypatch.setattr(builder, "load_bound_spine_sidecar", lambda *_args: {})
    monkeypatch.setattr(
        builder, "spine_provenance_from_sidecar", lambda *_args: {"synthetic": True}
    )
    joint_flags = _mandatory_input_flags(input_h5, ladder_path)

    constructions = []

    class StubResolver:
        def __init__(self, **kwargs):
            constructions.append(kwargs)
            self.frame = kwargs["frame"]
            # A live resolver writes the frame to a scratch H5 through the
            # national-frame writer, which validates the mass chain; a block
            # must therefore carry a record whose total equals its weights.
            validate_uk_national_frame(self.frame)
            self.simulation = object()
            self.contract_targets = {}

        def receipt(self):
            return {"mode": "stub", "policyengine_uk_version": "test"}

    monkeypatch.setattr(
        builder,
        "resolve_target_measures",
        # A live resolver injects ENGINE INPUTS (scratch columns the
        # materialization reads), some of which also exist on another entity
        # (region, esa_* on the spine); the driver must drop them before the
        # prepared frame or the flattening rule refuses the duplicate column.
        lambda _factory, _registry, provider, **_kwargs: SimpleNamespace(
            measure_inputs={
                ("household", "stub_engine_input"): np.ones(
                    len(provider.frame.table("household")), dtype=float
                ),
                ("person", "region"): np.zeros(
                    len(provider.frame.table("person")), dtype=float
                ),
            }
        ),
    )

    def _stub_materialize(adapter, registry, *, period, band_edge_registry=None):
        # Materialization is what mints the prepared measure columns the
        # national rows compile against; the stub writes them from the
        # injected input so the lifecycle matches the real stage.
        for spec in registry.specs:
            table = adapter.tables[spec.entity]
            table[spec.measure] = np.ones(len(table), dtype=float)
        return SimpleNamespace(skipped=())

    monkeypatch.setattr(builder, "materialize_uk_ledger_targets", _stub_materialize)
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
                "--release-role",
                "dense",
                "--ladder",
                str(ladder_path),
                "--out",
                str(dry_out),
                "--n-clones",
                "2",
                "--dry-run",
                *joint_flags,
            ]
        )
        == 0
    )
    dry_plan = json.loads(capsys.readouterr().out)
    dry_unbound = dry_plan["cross_grain"]["unbound_bridges"]
    assert dry_plan["cross_grain"]["empty_legs_licensed"] == []
    assert dry_plan["cross_grain"]["controls_without_lower_rows"] == []
    assert [entry["bridge_id"] for entry in dry_unbound] == [household_bridge.bridge_id]
    assert dry_unbound[0]["missing"] == sorted(reviewed_missing)
    dry_fanout = dry_plan["cross_grain"]["fanout_targets_not_controls"]
    assert dry_fanout == [
        {
            "target_id": fanout_target_id,
            "geography_id": "K03000001",
            "cells": 3,
            "cell_names": list(fanout_names),
            "activated_sum": 99.0,
            "reason": (
                "The activated cells are a band subset, so this distribution "
                "is not a cross-grain control."
            ),
        }
    ]
    assert "fanout_controls_summed" not in dry_plan["cross_grain"]
    assert not dry_out.exists()

    f100_out = tmp_path / "joint-f100"
    assert (
        builder.main(
            [
                "--input-h5",
                str(input_h5),
                "--release-role",
                "dense",
                "--ladder",
                str(ladder_path),
                "--out",
                str(f100_out),
                "--n-clones",
                "2",
                "--epochs",
                "2",
                "--skip-holdout",
                *joint_flags,
            ]
        )
        == 0
    )
    f100 = json.loads((f100_out / builder.MANIFEST_FILENAME).read_text())
    assert f100["schema_version"] == 4
    # The written rowwise artifact carries the shared ``clone_index`` name on
    # every table: the compact national loader must refuse it (flattening
    # rule) and the rowwise reader must undo the export rename.
    from microcosm.build.uk_runtime.rowwise_dataset import (
        ladder_clone_index_column,
        load_uk_rowwise_dataset,
    )

    dataset_path = Path(f100["outputs"]["dataset"]["path"])
    with pytest.raises(ValueError, match="globally unique"):
        builder.load_uk_national_frame(dataset_path)
    reloaded, provenance = load_uk_rowwise_dataset(dataset_path)
    assert provenance.source_h5 == dataset_path.resolve() or str(
        provenance.source_h5
    ).endswith(dataset_path.name)
    for entity in ("person", "benunit", "household"):
        assert ladder_clone_index_column(entity) in reloaded.table(entity).columns
        assert "clone_index" not in reloaded.table(entity).columns
    assert len(reloaded.table("household")) == f100["solve"]["n_households"]
    assert reloaded.weights_for("household").total == pytest.approx(
        f100["weights"]["calibration_mass_change"]["new_total"]
    )
    # Exact: the reader undoes the export rename and nothing else, so every
    # table equals the written one with clone_index renamed back.
    for entity in ("person", "benunit", "household"):
        written = pd.read_hdf(dataset_path, entity)
        expected = written.rename(
            columns={"clone_index": ladder_clone_index_column(entity)}
        )
        if entity == "household":
            # The frame carries the weight as its typed vector, not a column.
            np.testing.assert_array_equal(
                reloaded.weights_for("household").values,
                expected["household_weight"].to_numpy(dtype="float64"),
            )
            expected = expected.drop(columns=["household_weight"])
        got = reloaded.table(entity)
        assert sorted(got.columns) == sorted(expected.columns)
        pd.testing.assert_frame_equal(
            got[sorted(got.columns)].reset_index(drop=True),
            expected[sorted(expected.columns)].reset_index(drop=True),
            check_dtype=True,
        )
    assert f100["solve"]["n_targets_by_kind"] == {
        "local": 1,
        "ladder": 12,
        "national": len(national_registry.specs),
    }
    assert f100["solve"]["n_targets"] == 13 + len(national_registry.specs)
    assert f100["solve"]["measure_resolution"]["mode"] == "stub"
    assert f100["cross_grain"]["unbound_bridges"] == dry_unbound
    assert f100["solve"]["cross_grain"]["unbound_bridges"] == dry_unbound
    assert f100["cross_grain"]["empty_legs_licensed"] == []
    assert f100["solve"]["cross_grain"]["empty_legs_licensed"] == []
    assert f100["cross_grain"]["controls_without_lower_rows"] == []
    assert f100["solve"]["cross_grain"]["controls_without_lower_rows"] == []
    assert f100["cross_grain"]["fanout_targets_not_controls"] == dry_fanout
    assert f100["solve"]["cross_grain"]["fanout_targets_not_controls"] == dry_fanout
    assert "fanout_controls_summed" not in f100["cross_grain"]
    assert "fanout_controls_summed" not in f100["solve"]["cross_grain"]
    assert f100["releasable"] is True
    assert f100["measure_exclusions"] == joint_inputs["measure_exclusions"]
    assert f100["census_household_uprating"]["applied"] is False
    assert _spool_rows(f100_out)[0].rung == "f100"

    f001_out = tmp_path / "joint-f001"
    assert (
        builder.main(
            [
                "--input-h5",
                str(input_h5),
                "--release-role",
                "dense",
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
                *joint_flags,
            ]
        )
        == 0
    )
    f001 = json.loads((f001_out / builder.MANIFEST_FILENAME).read_text())
    assert f001["rung_surface"]["dropped_cells"] > 0
    assert f001["rung_surface"]["dropped_unreachable_cells"] >= 0
    assert isinstance(f001["rung_surface"]["dropped_unreachable_by_grain"], dict)
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
    household_flags = _configure_households_only_inputs(
        builder, monkeypatch, input_h5=input_h5, ladder_path=ladder_path
    )

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
                "--release-role",
                "dense",
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
                *household_flags,
            ]
        )

    rows = _spool_rows(output_dir)
    assert len(rows) == 1
    row = rows[0]
    assert row.disposition == "failed"
    gate_report_path = output_dir / "microcosm_uk_2024_25_local.local_gates.json"
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
    household_flags = _configure_households_only_inputs(
        builder, monkeypatch, input_h5=input_h5, ladder_path=ladder_path
    )

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
                "--release-role",
                "dense",
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
                *household_flags,
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
    household_flags = _configure_households_only_inputs(
        builder, monkeypatch, input_h5=input_h5, ladder_path=ladder_path
    )

    def failing_ladder_load(_path):
        raise RuntimeError("ladder artifact refused to parse")

    monkeypatch.setattr(builder, "load_uk_oa_ladder", failing_ladder_load)

    with pytest.raises(RuntimeError, match="ladder artifact refused to parse"):
        builder.main(
            [
                "--input-h5",
                str(input_h5),
                "--release-role",
                "dense",
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
                *household_flags,
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


def test_households_only_targets_come_from_compiled_chronicle_registry(
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

    registry = TargetRegistry(_household_specs_for_ladder(target_ladder), country="uk")
    _, problem, cross_grain = builder._build_bound_problem(
        assignment,
        local_registry=registry,
        period=2025,
    )

    expected = sorted(
        float(spec.value)
        for spec in registry.specs
        if spec.metadata["geography_level"] == "constituency"
    )
    assert sorted(problem.targets.tolist()) == expected
    assert cross_grain["census_household_uprating"]["applied"] is False
    assert cross_grain["bound_national_targets"] == []

    # The households-only scope applies the same per-grain A15 factor as the
    # joint scope and carries its receipt into the manifest (Max's review).
    uprating = {
        "applied": True,
        "period": 2025,
        "grains": {
            "constituency": {
                "cells": len(expected),
                "census_households_total": sum(expected),
                "census_years": [2021, 2022],
                "factor": 1.1,
            }
        },
    }
    _, uprated_problem, uprated_cross_grain = builder._build_bound_problem(
        assignment,
        local_registry=registry,
        period=2025,
        census_household_uprating=uprating,
    )
    assert sorted(uprated_problem.targets.tolist()) == pytest.approx(
        [value * 1.1 for value in expected]
    )
    receipt = uprated_cross_grain["census_household_uprating"]
    assert receipt["applied"] is True
    assert receipt["household_cells"]["cells"] == len(expected)
    assert receipt["household_cells"]["skipped_cells"] == 0
    assert (
        problem.target_frame["target_name"]
        .str.startswith("ons.census.households@")
        .all()
    )
    assert dict(
        zip(
            problem.target_frame["area_code"],
            problem.target_frame["target_name"],
            strict=True,
        )
    ) == {
        str(spec.metadata["geography_id"]): spec.name
        for spec in registry.specs
        if spec.metadata["geography_level"] == "constituency"
    }


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
    household_flags = [
        *_mandatory_input_flags(input_h5, ladder_path),
        "--households-only",
    ]

    with pytest.raises(ValueError, match="differ"):
        builder.main(
            [
                "--input-h5",
                str(input_h5),
                "--release-role",
                "dense",
                "--ladder",
                str(ladder_path),
                "--out",
                str(output_dir),
                "--dry-run",
                *household_flags,
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
        posture=builder.UK_ROWWISE_DENSE_POSTURE,
        vintage="2024_25",
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


def _failing_gate_evaluator(builder, name: str, message: str):
    def evaluator(*_args, **_kwargs):
        return builder.GateResult(
            name=name, passed=False, failures=(message,), details={"minimum": 0}
        )

    return evaluator


def _joint_f100_args(input_h5: Path, ladder_path: Path, output_dir: Path) -> list[str]:
    return [
        "--input-h5",
        str(input_h5),
        "--release-role",
        "dense",
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
        "--skip-holdout",
        *_mandatory_input_flags(input_h5, ladder_path),
        "--households-only",
    ]


def test_candidate_weight_ratio_failure_is_reported_and_blocks(
    monkeypatch, tmp_path, capsys
) -> None:
    pytest.importorskip("tables")
    pytest.importorskip("h5py")
    builder = _load_builder_module()
    input_h5 = tmp_path / "staging.h5"
    ladder_path = tmp_path / "ladder.npz"
    output_dir = tmp_path / "candidate"
    _write_staging_h5(
        input_h5, households_per_region=200, region_masses=(4.0, 10.0, 10.0, 9.0)
    )
    _write_ladder(ladder_path)
    ladder = load_uk_oa_ladder(ladder_path)
    _configure_households_only_inputs(
        builder, monkeypatch, input_h5=input_h5, ladder_path=ladder_path
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
    ratio = builder.UK_GATE_REGISTRY["weight_ratio"]
    monkeypatch.setattr(
        builder,
        "UK_GATE_REGISTRY",
        {
            **builder.UK_GATE_REGISTRY,
            "weight_ratio": replace(
                ratio,
                evaluator=_failing_gate_evaluator(
                    builder, "weight_ratio", "ratio 104.6 > 100"
                ),
            ),
        },
    )

    assert builder.main(_joint_f100_args(input_h5, ladder_path, output_dir)) == 1

    capsys.readouterr()
    manifest = json.loads((output_dir / builder.MANIFEST_FILENAME).read_text())
    assert manifest["failing_gate_ids"] == ["uk_local_weight_ratio"]
    assert manifest["blocked_at_f100"] is True
    assert manifest["diagnostic_failures"] == []
    assert manifest["blocking_failures"] == [
        "[uk_local_weight_ratio] ratio 104.6 > 100"
    ]
    assert manifest["releasable"] is False
    report = json.loads(
        Path(manifest["outputs"]["local_gate_report"]["path"]).read_text()
    )
    assert report["gates"]["uk_local_weight_ratio"]["criticality"] == "release_blocking"
    assert report["gates"]["uk_local_weight_ratio"]["status"] == "failed"
    assert _spool_rows(output_dir)[0].disposition == "failed"


def test_candidate_block_partitions_failures_by_criticality(
    monkeypatch, tmp_path, capsys
) -> None:
    pytest.importorskip("tables")
    pytest.importorskip("h5py")
    builder = _load_builder_module()
    input_h5 = tmp_path / "staging.h5"
    ladder_path = tmp_path / "ladder.npz"
    output_dir = tmp_path / "candidate"
    _write_staging_h5(
        input_h5, households_per_region=200, region_masses=(4.0, 10.0, 10.0, 9.0)
    )
    _write_ladder(ladder_path)
    _configure_households_only_inputs(
        builder, monkeypatch, input_h5=input_h5, ladder_path=ladder_path
    )
    registry = builder.UK_GATE_REGISTRY
    monkeypatch.setattr(
        builder,
        "UK_GATE_REGISTRY",
        {
            **registry,
            "area_support": replace(
                registry["area_support"],
                evaluator=_failing_gate_evaluator(
                    builder, "area_support", "ESS 42.3 < 50"
                ),
            ),
            "weight_ratio": replace(
                registry["weight_ratio"],
                evaluator=_failing_gate_evaluator(
                    builder, "weight_ratio", "ratio 578 > 100"
                ),
            ),
        },
    )

    assert builder.main(_joint_f100_args(input_h5, ladder_path, output_dir)) == 1

    captured = capsys.readouterr()
    assert "artifact unreleasable" in captured.err
    manifest = json.loads((output_dir / builder.MANIFEST_FILENAME).read_text())
    assert manifest["failing_gate_ids"] == [
        "uk_local_area_support",
        "uk_local_weight_ratio",
    ]
    assert manifest["blocked_at_f100"] is True
    assert manifest["blocking_failures"] == [
        "[uk_local_area_support] ESS 42.3 < 50",
        "[uk_local_weight_ratio] ratio 578 > 100",
    ]
    assert manifest["diagnostic_failures"] == []
    assert manifest["releasable"] is False
    assert _spool_rows(output_dir)[0].disposition == "failed"


def test_release_verdict_requires_single_block_engine() -> None:
    builder = _load_builder_module()
    releasable, posture = builder._release_verdict(
        sample_fraction=1.0, engine_blocks=1, release_blocking_gates_passed=True
    )
    assert releasable is True and all(posture.values())
    # A per-block engine resolution never writes a releasable artifact, even
    # with every release-blocking gate passed on the full rung (#736 erratum).
    releasable, posture = builder._release_verdict(
        sample_fraction=1.0, engine_blocks=15, release_blocking_gates_passed=True
    )
    assert releasable is False
    assert posture == {
        "full_rung": True,
        "single_block_engine": False,
        "release_blocking_gates_passed": True,
    }
    assert (
        builder._release_verdict(
            sample_fraction=0.1, engine_blocks=1, release_blocking_gates_passed=True
        )[0]
        is False
    )


def test_gate_criticality_reads_fail_closed() -> None:
    builder = _load_builder_module()
    assert builder._is_release_blocking({"criticality": "release_blocking"}) is True
    assert builder._is_release_blocking({"criticality": "diagnostic"}) is False
    # Missing or unknown criticality vetoes: schema drift on one entry cannot
    # drop a failed gate out of both the blocking list and all_gates_passed.
    assert builder._is_release_blocking({}) is True
    assert builder._is_release_blocking({"criticality": "advisory"}) is True
    blocking, diagnostic = builder._gate_failures_by_criticality(
        {
            "gates": {
                "uk_local_area_support": {
                    "status": "failed",
                    "failures": ["ESS 42.3 < 50"],
                },
                "uk_local_weight_ratio": {
                    "status": "failed",
                    "criticality": "diagnostic",
                    "failures": ["ratio 578 > 100"],
                },
                "uk_local_target_fit": {
                    "status": "passed",
                    "criticality": "diagnostic",
                },
            }
        }
    )
    assert blocking == ["[uk_local_area_support] ESS 42.3 < 50"]
    assert diagnostic == ["[uk_local_weight_ratio] ratio 578 > 100"]


def test_release_candidate_refuses_non_doctrine_solve_settings(tmp_path) -> None:
    builder = _load_builder_module()
    pin = "0" * 64
    base = [
        "--input-h5",
        str(tmp_path / "spine.h5"),
        "--release-role",
        "dense",
        "--input-sha256",
        pin,
        "--ladder",
        str(tmp_path / "ladder.npz"),
        "--ladder-sha256",
        pin,
        "--ledger-facts",
        str(tmp_path / "ledger"),
        "--ledger-facts-sha256",
        pin,
        "--ledger-manifest-sha256",
        pin,
        "--out",
        str(tmp_path / "out"),
        "--release-candidate",
    ]
    # The doctrine defaults are the release posture: nothing to refuse.
    args = builder._parse_args(base)
    builder._validate_cli_args(args)
    assert args.n_clones == builder.UK_ROWWISE_DENSE_POSTURE.clone_count == 15
    assert args.epochs == builder.UK_ROWWISE_DENSE_POSTURE.epochs == 1500
    assert args.target_weight_rule == "grain_equal"

    with pytest.raises(ValueError, match=r"--epochs != doctrine 1500"):
        builder._validate_cli_args(builder._parse_args([*base, "--epochs", "512"]))
    with pytest.raises(ValueError, match=r"--n-clones != doctrine 15"):
        builder._validate_cli_args(builder._parse_args([*base, "--n-clones", "10"]))
    with pytest.raises(ValueError, match=r"--target-weight-rule"):
        builder._validate_cli_args(
            builder._parse_args([*base, "--target-weight-rule", "uniform"])
        )


def test_candidate_requires_pinned_ledger_inputs(tmp_path) -> None:
    builder = _load_builder_module()
    args = builder._parse_args(
        [
            "--input-h5",
            str(tmp_path / "spine.h5"),
            "--release-role",
            "dense",
            "--input-sha256",
            "0" * 64,
            "--ladder",
            str(tmp_path / "ladder.npz"),
            "--ladder-sha256",
            "0" * 64,
            "--out",
            str(tmp_path / "out"),
        ]
    )
    with pytest.raises(ValueError, match="mandatory"):
        builder._validate_cli_args(args)


def test_candidate_multi_block_engine_run_is_never_releasable(
    monkeypatch, tmp_path, capsys
) -> None:
    """End to end: ``--engine-blocks K`` on f100 writes ``releasable: false``.

    Every release-blocking gate passes here; the posture alone withholds the
    verdict, and the manifest names the leg (``single_block_engine``). This is
    the assertion that catches a future caller bypassing ``_release_verdict``.
    """

    pytest.importorskip("tables")
    pytest.importorskip("h5py")
    builder = _load_builder_module()
    input_h5 = tmp_path / "staging.h5"
    ladder_path = tmp_path / "ladder.npz"
    output_dir = tmp_path / "candidate"
    _write_staging_h5(
        input_h5, households_per_region=200, region_masses=(4.0, 10.0, 10.0, 9.0)
    )
    _write_ladder(ladder_path)
    ladder = load_uk_oa_ladder(ladder_path)
    _configure_households_only_inputs(
        builder, monkeypatch, input_h5=input_h5, ladder_path=ladder_path
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

    args = [
        *_joint_f100_args(input_h5, ladder_path, output_dir),
        "--engine-blocks",
        "2",
    ]
    assert builder.main(args) == 0

    capsys.readouterr()
    manifest = json.loads((output_dir / builder.MANIFEST_FILENAME).read_text())
    assert manifest["parameters"]["engine_blocks"] == 2
    assert manifest["blocking_failures"] == []
    assert manifest["releasable"] is False
    assert manifest["release_posture"] == {
        "full_rung": True,
        "single_block_engine": False,
        "release_blocking_gates_passed": True,
    }


def test_size_candidate_exports_compact_links_and_cannot_claim_dense_release(
    monkeypatch, tmp_path
):
    pytest.importorskip("tables")
    pytest.importorskip("h5py")
    builder = _load_builder_module()
    input_h5 = tmp_path / "spine.h5"
    ladder_path = tmp_path / "ladder.npz"
    out = tmp_path / "k300"
    _write_staging_h5(input_h5, households_per_region=52)
    ladder = _write_ladder(ladder_path)
    import microcosm.build.uk_runtime.battery_bindings as bindings

    monkeypatch.setattr(
        bindings,
        "_local_area_roster",
        lambda _resource, levels: {
            "constituency": tuple(sorted(set(ladder.constituency_code))),
            "local_authority": tuple(sorted(set(ladder.local_authority_code))),
        },
    )
    flags = _configure_households_only_inputs(
        builder, monkeypatch, input_h5=input_h5, ladder_path=ladder_path
    )
    status = builder.main(
        [
            "--input-h5",
            str(input_h5),
            "--release-role",
            "dense",
            "--ladder",
            str(ladder_path),
            *flags,
            "--out",
            str(out),
            "--n-clones",
            "2",
            "--dataset-households",
            "300",
            "--selection-seed",
            "11",
            "--epochs",
            "2",
            "--skip-holdout",
            "--seed",
            "7",
        ]
    )
    assert status in (0, 1)  # Gate failures remain reportable candidates.
    manifest = json.loads((out / builder.MANIFEST_FILENAME).read_text())
    assert manifest["releasable"] is False
    assert manifest["release_posture"]["size_certification_present"] is False
    size = manifest["solve"]["dataset_size"]
    assert size["requested_households"] == size["realized_households"] == 300
    assert size["pool_households"] == 416
    assert manifest["parameters"]["n_clones"] == 2
    assert (
        manifest["weights"]["stretch_reference"]
        == "normalized_horvitz_thompson_w_over_q"
    )
    path = out / "microcosm_uk_2024_25_local.h5"
    with pd.HDFStore(path, "r") as store:
        households = store["household"]
        persons = store["person"]
        benunits = store["benunit"]
    assert len(households) == 300
    assert set(persons.person_household_id) == set(households.household_id)
    assert set(persons.person_benunit_id) == set(benunits.benunit_id)
    assert len(_spool_rows(out)) == 1

    # The selection seed moves the draw only; the manifest records both seeds.
    assert manifest["parameters"]["seed"] == 7
    assert manifest["parameters"]["selection_seed"] == 11
    assert size["seed"] == 11
    assert manifest["parameters"]["selection_pi_hi"] == 1.0
    assert size["selection_pi_hi"] == 1.0
    assert manifest["parameters"]["baseline_pi_floor"] == 0.0
    assert size["baseline_pi_floor"] == 0.0
    assert size["baseline_floored_rows"] == 0
    assert size["refit_baseline"] == "normalized_horvitz_thompson_w_over_q"
    assert size["selection_receipt"]["pi_hi"] == 1.0
    assert size["selection_feasibility"]["requested_pi_hi"] == 1.0
    assert size["selection_feasibility"]["feasible_at_requested_pi_hi"] is True

    # The dense solve the selection was cut from ships as evidence.
    dense = size["dense_reference"]
    assert dense["final_loss"] == size["dense_loss"]
    assert dense["n_households"] == 416
    assert dense["weights"]["n_records"] == 416
    assert {"effective_sample_size", "max_to_median_positive_weight"} <= set(
        dense["weights"]
    )
    assert dense["diagnostics_file"] == builder.DENSE_REFERENCE_DIAGNOSTICS_FILENAME
    outputs = manifest["outputs"]
    dense_csv = out / builder.DENSE_REFERENCE_DIAGNOSTICS_FILENAME
    selection_csv = out / builder.DATASET_SIZE_SELECTION_FILENAME
    assert Path(outputs["dense_reference_diagnostics"]["path"]) == dense_csv.resolve()
    assert Path(outputs["dataset_size_selection"]["path"]) == selection_csv.resolve()
    assert (
        outputs["dataset_size_selection"]["sha256"]
        == hashlib.sha256(selection_csv.read_bytes()).hexdigest()
    )
    dense_rows = pd.read_csv(dense_csv)
    assert len(dense_rows) == manifest["solve"]["n_targets"]
    assert dense_rows.columns[0] == "grain"
    assert {"target", "final_estimate", "abs_relative_error"} <= set(dense_rows.columns)
    selection = pd.read_csv(selection_csv)
    assert list(selection.columns) == [
        "pool_row_index",
        "household_id",
        "clone_index",
        "design_weight",
        "inclusion_probability",
        "certainty",
        "ht_baseline_weight",
        "refit_weight",
    ]
    assert len(selection) == 300
    assert selection["pool_row_index"].is_unique
    assert selection["pool_row_index"].max() < 416
    assert set(selection["household_id"]) == set(households.household_id)
    assert (selection["refit_weight"] > 0).all()
    assert (selection["design_weight"] > 0).all()
    assert (
        int(selection["certainty"].sum())
        == size["selection_receipt"]["certainty_count"]
    )
    assert int(selection["certainty"].sum()) == size["protected_carriers"]


def test_selection_seed_requires_a_dataset_size(tmp_path):
    builder = _load_builder_module()
    args = builder._parse_args(
        [
            "--input-h5",
            str(tmp_path / "spine.h5"),
            "--release-role",
            "dense",
            "--ladder",
            str(tmp_path / "ladder.npz"),
            "--out",
            str(tmp_path / "out"),
            "--selection-seed",
            "11",
        ]
    )
    with pytest.raises(ValueError, match="requires --dataset-households"):
        builder._validate_cli_args(args)


@pytest.mark.parametrize(
    ("argv_tail", "message"),
    [
        (["--selection-pi-hi", "0.95"], "requires --dataset-households"),
        (["--dataset-households", "10", "--selection-pi-hi", "0"], r"in \(0, 1\]"),
        (["--dataset-households", "10", "--selection-pi-hi", "1.5"], r"in \(0, 1\]"),
        (["--baseline-pi-floor", "0.01"], "requires --dataset-households"),
        (["--dataset-households", "10", "--baseline-pi-floor", "-0.1"], r"in \[0, 1\]"),
        (["--dataset-households", "10", "--baseline-pi-floor", "1.5"], r"in \[0, 1\]"),
    ],
)
def test_selection_pi_hi_is_candidate_only_and_bounded(tmp_path, argv_tail, message):
    builder = _load_builder_module()
    args = builder._parse_args(
        [
            "--input-h5",
            str(tmp_path / "spine.h5"),
            "--release-role",
            "dense",
            "--ladder",
            str(tmp_path / "ladder.npz"),
            "--out",
            str(tmp_path / "out"),
            *argv_tail,
        ]
    )
    with pytest.raises(ValueError, match=message):
        builder._validate_cli_args(args)


def test_dense_candidate_manifest_has_no_size_sidecars(tmp_path):
    builder = _load_builder_module()
    paths = builder._output_paths(
        tmp_path, posture=builder.UK_ROWWISE_DENSE_POSTURE, vintage="2024_25"
    )
    assert paths["dataset"].name == "microcosm_uk_2024_25_local.h5"
    assert paths["dense_reference"].name == builder.DENSE_REFERENCE_DIAGNOSTICS_FILENAME
    assert paths["selection"].name == builder.DATASET_SIZE_SELECTION_FILENAME
    assert builder._SIZE_RUN_ONLY_OUTPUTS == {"dense_reference", "selection"}


def test_size_cli_refuses_promotion_without_separate_certification(tmp_path):
    builder = _load_builder_module()
    args = builder._parse_args(
        [
            "--input-h5",
            str(tmp_path / "spine.h5"),
            "--release-role",
            "dense",
            "--ladder",
            str(tmp_path / "ladder.npz"),
            "--out",
            str(tmp_path / "out"),
            "--dataset-households",
            "50000",
            "--release-candidate",
        ]
    )
    with pytest.raises(ValueError, match="candidate-only"):
        builder._validate_cli_args(args)


def test_size_candidate_checkpoints_before_the_draw_and_resumes_from_it(
    monkeypatch, tmp_path, capsys
):
    pytest.importorskip("tables")
    pytest.importorskip("h5py")
    builder = _load_builder_module()
    from microcosm.build.uk_runtime.size_checkpoint import (
        SIZE_CHECKPOINT_ARRAYS_FILENAME,
        SIZE_CHECKPOINT_MANIFEST_FILENAME,
    )

    input_h5 = tmp_path / "spine.h5"
    ladder_path = tmp_path / "ladder.npz"
    _write_staging_h5(input_h5, households_per_region=52)
    ladder = _write_ladder(ladder_path)
    import microcosm.build.uk_runtime.battery_bindings as bindings

    monkeypatch.setattr(
        bindings,
        "_local_area_roster",
        lambda _resource, levels: {
            "constituency": tuple(sorted(set(ladder.constituency_code))),
            "local_authority": tuple(sorted(set(ladder.local_authority_code))),
        },
    )
    common = [
        "--input-h5",
        str(input_h5),
        "--release-role",
        "dense",
        "--ladder",
        str(ladder_path),
        *_configure_households_only_inputs(
            builder, monkeypatch, input_h5=input_h5, ladder_path=ladder_path
        ),
        "--n-clones",
        "2",
        "--dataset-households",
        "300",
        "--epochs",
        "2",
        "--skip-holdout",
        "--seed",
        "7",
    ]
    first = tmp_path / "first"
    status = builder.main([*common, "--out", str(first), "--selection-pi-hi", "0.5"])
    assert status in (0, 1)
    # The solve is no longer silent: probe verdicts and the search stop reach
    # stderr as they happen, beside the phase lines.
    err = capsys.readouterr().err
    assert "probe 1/10 done:" in err and "search stopped:" in err
    assert "dense solve: epoch 2/2" in err and "refit: epoch 2/2" in err
    assert "size selection checkpoint written to" in err
    assert (first / SIZE_CHECKPOINT_ARRAYS_FILENAME).is_file()
    checkpoint = json.loads((first / SIZE_CHECKPOINT_MANIFEST_FILENAME).read_text())
    assert checkpoint["selection"]["households"] == 300
    assert checkpoint["selection"]["search_pi_hi"] == 0.5
    assert checkpoint["identity"]["dataset_households"] == 300
    assert checkpoint["identity"]["epochs"] == 2
    # The identity carries the solve doctrine; the provenance names the
    # writing run (reported on resume, not compared).
    assert checkpoint["identity"]["doctrine"] == builder._doctrine_bounds(
        builder.UK_ROWWISE_DENSE_POSTURE
    )
    assert checkpoint["identity"]["release_role"] == "dense"
    assert set(checkpoint["provenance"]) == {"code_pin", "build_id"}
    manifest = json.loads((first / builder.MANIFEST_FILENAME).read_text())
    written = manifest["solve"]["dataset_size"]["checkpoint"]["written"]
    assert "written_at" not in written and "directory" not in written
    size_first = manifest["solve"]["dataset_size"]
    assert 0 < size_first["certainty_share"] <= 1
    assert (
        size_first["boundary_draws"]
        == 300 - (size_first["selection_receipt"]["certainty_count"])
    )
    assert size_first["zero_target_rows"] >= 0
    weights_block = manifest["weights"]
    assert weights_block["stretch_reference"] == "normalized_horvitz_thompson_w_over_q"
    assert weights_block["realized_max_weight_ratio_vs_stretch_reference"] > 0
    assert weights_block["realized_max_weight_ratio_vs_design"] > 0
    assert manifest["parameters"]["size_checkpoint"] is True
    assert manifest["parameters"]["resume_size_checkpoint"] is None
    written = manifest["solve"]["dataset_size"]["checkpoint"]["written"]
    assert written["arrays_sha256"] == checkpoint["arrays_sha256"]
    assert manifest["solve"]["dataset_size"]["selection_reused"] is False
    rows = _spool_rows(first)
    assert len(rows) == 1
    assert "size_selection_checkpointed" in rows[0].phases_reached

    # Resume on the same inputs: no dense solve, no search, same draw and refit.
    second = tmp_path / "second"
    status = builder.main(
        [
            *common,
            "--out",
            str(second),
            "--selection-pi-hi",
            "0.5",
            "--resume-size-checkpoint",
            str(first),
        ]
    )
    assert status in (0, 1)
    assert not (second / SIZE_CHECKPOINT_ARRAYS_FILENAME).exists()
    resumed = json.loads((second / builder.MANIFEST_FILENAME).read_text())
    assert resumed["parameters"]["size_checkpoint"] is False
    assert resumed["parameters"]["resume_size_checkpoint"] == str(first.resolve())
    size = resumed["solve"]["dataset_size"]
    assert size["selection_reused"] is True
    assert size["selection_search_pi_hi"] == 0.5
    assert (
        size["checkpoint"]["resumed_from"]["arrays_sha256"]
        == (checkpoint["arrays_sha256"])
    )
    assert size["dense_loss"] == manifest["solve"]["dataset_size"]["dense_loss"]
    assert (
        size["selection_l0_lambda"]
        == (manifest["solve"]["dataset_size"]["selection_l0_lambda"])
    )
    first_selection = pd.read_csv(first / builder.DATASET_SIZE_SELECTION_FILENAME)
    second_selection = pd.read_csv(second / builder.DATASET_SIZE_SELECTION_FILENAME)
    pd.testing.assert_frame_equal(first_selection, second_selection)
    assert "size_selection_resumed" in _spool_rows(second)[0].phases_reached

    # Another threshold re-draws from the same checkpoint and records both.
    third = tmp_path / "third"
    status = builder.main(
        [
            *common,
            "--out",
            str(third),
            "--selection-pi-hi",
            "1.0",
            "--resume-size-checkpoint",
            str(first),
        ]
    )
    assert status in (0, 1)
    redrawn = json.loads((third / builder.MANIFEST_FILENAME).read_text())
    assert redrawn["solve"]["dataset_size"]["selection_pi_hi"] == 1.0
    assert redrawn["solve"]["dataset_size"]["selection_search_pi_hi"] == 0.5

    # A resume whose inputs differ refuses by name, before any solve.
    different_epochs = list(common)
    different_epochs[different_epochs.index("--epochs") + 1] = "3"
    with pytest.raises(ValueError, match="epochs: checkpoint 2 != run 3"):
        builder.main(
            [
                *different_epochs,
                "--out",
                str(tmp_path / "fourth"),
                "--resume-size-checkpoint",
                str(first),
            ]
        )
    with pytest.raises(ValueError, match="requires --dataset-households"):
        builder.main(
            [
                "--input-h5",
                str(input_h5),
                "--release-role",
                "dense",
                "--ladder",
                str(ladder_path),
                "--out",
                str(tmp_path / "fifth"),
                "--resume-size-checkpoint",
                str(first),
            ]
        )
    # An --out that already holds a checkpoint refuses before any solve
    # (Vahid's should-fix 2): the checkpoint writer's own refusal came hours
    # too late.
    stale = tmp_path / "stale"
    stale.mkdir()
    (stale / SIZE_CHECKPOINT_ARRAYS_FILENAME).write_bytes(b"stale")
    (stale / SIZE_CHECKPOINT_MANIFEST_FILENAME).write_text("{}")
    with pytest.raises(FileExistsError, match="already holds a size checkpoint"):
        builder.main([*common, "--out", str(stale)])
    assert not (stale / builder.MANIFEST_FILENAME).exists()
    assert manifest["solve"]["dataset_size"]["checkpoint"]["written"]["stage"] == (
        "before_exact_count_draw"
    )
    resumed_receipt = resumed["solve"]["dataset_size"]["checkpoint"]["resumed_from"]
    assert (
        resumed_receipt["provenance"]["build_id"]
        == checkpoint["provenance"]["build_id"]
    )
    assert "written_at" not in resumed_receipt

    # ---------------------------------------------------------------------------
    # Staging: telemetry to runs/<run_id>/ and the staged dataset bundle.

    # A checkpoint written before the release role existed (no release_role
    # in its identity) refuses to resume: the identity is the run's
    # contract, and a pre-role checkpoint is rebuilt, never grandfathered.
    legacy = tmp_path / "legacy"
    shutil.copytree(first, legacy)
    legacy_manifest = json.loads(
        (legacy / SIZE_CHECKPOINT_MANIFEST_FILENAME).read_text()
    )
    del legacy_manifest["identity"]["release_role"]
    (legacy / SIZE_CHECKPOINT_MANIFEST_FILENAME).write_text(json.dumps(legacy_manifest))
    with pytest.raises(ValueError, match="release_role: absent in checkpoint"):
        builder.main(
            [
                *common,
                "--out",
                str(tmp_path / "from-legacy"),
                "--resume-size-checkpoint",
                str(legacy),
            ]
        )
    assert not (tmp_path / "from-legacy" / builder.MANIFEST_FILENAME).exists()


def _load_tool(name: str):
    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(name, root / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _FakeHub:
    """One fake Hub serving the telemetry repo and the private dataset repo."""

    def __init__(
        self,
        *,
        fail_commit: bool = False,
        role: str = "write",
        scopes: list[dict] | None = None,
    ) -> None:
        self.role = role
        self.scopes = scopes
        self.commit_of: dict[tuple[str, str], str] = {}
        self.files: dict[tuple[str, str], bytes] = {}
        self.uploads: list[tuple[str, str]] = []
        self.commits: list[dict[str, object]] = []
        self.fail_commit = fail_commit
        self.sha = "a" * 40

    def paths(self, repo_id: str) -> list[str]:
        return sorted(path for repo, path in self.files if repo == repo_id)

    def upload_file(self, *, path_or_fileobj, path_in_repo, repo_id, repo_type):
        assert repo_type == "dataset"
        self.files[(repo_id, path_in_repo)] = Path(path_or_fileobj).read_bytes()
        self.uploads.append((repo_id, path_in_repo))

    def hf_hub_download(self, *, repo_id, filename, repo_type, **kwargs):
        assert repo_type == "dataset"
        if (repo_id, filename) not in self.files:
            raise FileNotFoundError(filename)
        destination = (
            Path(kwargs.get("local_dir") or _FAKE_HUB_CACHE) / repo_id / filename
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.files[(repo_id, filename)])
        return str(destination)

    def file_exists(self, *, repo_id, filename, repo_type):
        assert repo_type == "dataset"
        return (repo_id, filename) in self.files

    def repo_info(self, *, repo_id, repo_type):
        assert repo_type == "dataset"
        return SimpleNamespace(sha=self.sha)

    def whoami(self):
        token = {"role": self.role}
        if self.role == "fineGrained":
            token["fineGrained"] = {"global": [], "scoped": self.scopes or []}
        return {"name": "tester", "auth": {"accessToken": token}}

    def get_paths_info(self, *, repo_id, paths, expand, repo_type):
        assert expand and repo_type == "dataset"
        return [
            SimpleNamespace(
                path=path,
                last_commit=SimpleNamespace(oid=self.commit_of[(repo_id, path)]),
            )
            for path in paths
            if (repo_id, path) in self.commit_of
        ]

    def create_commit(
        self, *, repo_id, operations, commit_message, repo_type, parent_commit
    ):
        assert repo_type == "dataset"
        if self.fail_commit:
            raise RuntimeError("403 Forbidden token=do-not-record")
        assert parent_commit == self.sha
        for operation in operations:
            self.files[(repo_id, operation.path_in_repo)] = Path(
                operation.path_or_fileobj
            ).read_bytes()
        self.commits.append(
            {
                "repo_id": repo_id,
                "message": commit_message,
                "paths": sorted(op.path_in_repo for op in operations),
            }
        )
        self.sha = hashlib.sha256(commit_message.encode()).hexdigest()[:40]
        for operation in operations:
            self.commit_of[(repo_id, operation.path_in_repo)] = self.sha
        return SimpleNamespace(oid=self.sha)


_FAKE_HUB_CACHE = "/tmp/microcosm-fake-hub-cache"


def _staging_run_setup(builder, monkeypatch, tmp_path, *, remote: bool = False):
    """Fixture inputs plus the flag list; ``remote`` drops the local-only switch."""

    pytest.importorskip("tables")
    pytest.importorskip("h5py")
    input_h5 = tmp_path / "staging.h5"
    ladder_path = tmp_path / "ladder.npz"
    _write_staging_h5(input_h5, households_per_region=52)
    ladder = _write_ladder(ladder_path)
    flags = _configure_households_only_inputs(
        builder, monkeypatch, input_h5=input_h5, ladder_path=ladder_path
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
    if remote:
        flags = [flag for flag in flags if flag != "--staging-local-only"]
    return input_h5, ladder_path, flags


def _build_args(input_h5, ladder_path, flags, out, *extra):
    return [
        "--input-h5",
        str(input_h5),
        "--release-role",
        "dense",
        "--ladder",
        str(ladder_path),
        *flags,
        "--out",
        str(out),
        "--n-clones",
        "2",
        "--seed",
        "7",
        "--epochs",
        "2",
        "--skip-holdout",
        *extra,
    ]


def _single_run_id(out: Path) -> str:
    runs = sorted(path.name for path in (out / "staging" / "runs").iterdir())
    assert len(runs) == 1, runs
    return runs[0]


def test_candidate_build_stages_telemetry_locally_and_inventories_the_bundle(
    monkeypatch, tmp_path, capsys
):
    from microcosm.build.staging_v2 import validate_v2_bundle

    builder = _load_builder_module()
    input_h5, ladder_path, flags = _staging_run_setup(builder, monkeypatch, tmp_path)
    out = tmp_path / "candidate"
    status = builder.main(_build_args(input_h5, ladder_path, flags, out))
    assert status == 0
    captured = capsys.readouterr()
    manifest = json.loads((out / builder.MANIFEST_FILENAME).read_text())
    # The stdout manifest is the on-disk manifest, evidence blocks included.
    assert json.loads(captured.out)["staged_dataset"] == manifest["staged_dataset"]
    assert "staged dataset: skipped (local_only)" in captured.err

    run_id = _single_run_id(out)
    rows = load_spool_rows(out / "logbook-spool")
    assert rows[0].build_id == run_id
    bundle = validate_v2_bundle(out / "staging", run_id)
    run_manifest = bundle["run_manifest"]
    assert run_manifest["status"] == "completed"
    assert run_manifest["run_kind"] == "calibration"
    assert run_manifest["operation_id"] == "uk_rowwise_candidate"
    assert run_manifest["pipeline"]["id"] == "uk-local-candidate"
    assert run_manifest["non_release"] is True
    assert run_manifest["sample"] == {"mode": "full"}
    assert run_manifest["delivery"]["mode"] == "local_only"
    assert run_manifest["delivery"]["upload_attempts"] == 0
    assert {a["logical_name"] for a in run_manifest["artifacts"]} == {
        "fit_summary",
        "staged_dataset",
    }
    run_dir = out / "staging" / "runs" / run_id
    fit_summary = json.loads((run_dir / "artifacts" / "fit_summary.json").read_text())
    assert fit_summary["run_id"] == run_id
    assert set(fit_summary["gates"]) == set(builder.UK_LOCAL_GATE_SCOPE)
    assert fit_summary["loss"]["final"] == manifest["solve"]["final_loss"]
    assert set(fit_summary["fit_by_family"]["local"]) == {"census_households"}
    staged_artifact = json.loads(
        (run_dir / "artifacts" / "staged_dataset.json").read_text()
    )
    assert staged_artifact == manifest["staged_dataset"]

    # Every phase reports started then completed, in build order; the
    # calibration progress keeps the last epoch only (thinning at 2 epochs).
    events = bundle["events"]
    completed = [e["stage_id"] for e in events if e["status"] == "completed"]
    assert completed == [
        "input_pinning",
        "target_compilation",
        "cloning",
        "surface_resolution",
        "calibration",
        "gate_battery",
        "holdout",
        "output_bundle",
        "dataset_staging",
        "complete",
    ]
    for stage in ("target_compilation", "cloning", "calibration", "gate_battery"):
        transitions = [e["status"] for e in events if e["stage_id"] == stage]
        assert transitions == ["started", "completed"], stage
    calibration_rows = json.loads((run_dir / "calibration_progress.json").read_text())[
        "events"
    ]
    assert [
        (row["epoch"], row["epochs"], row["phase"]) for row in calibration_rows
    ] == [(2, 2, None)]
    calibration_done = next(
        e
        for e in events
        if e["stage_id"] == "calibration" and e["status"] == "completed"
    )
    assert calibration_done["details"]["final_loss"] == manifest["solve"]["final_loss"]
    assert calibration_done["details"]["size_checkpoint"] is None

    # The manifest carries both receipts; the bundle inventory is the outputs.
    assert manifest["staging_delivery"]["mode"] == "local_only"
    assert manifest["staging_delivery"]["run_id"] == run_id
    staged = manifest["staged_dataset"]
    assert staged["mode"] == "local_only" and staged["status"] == "skipped"
    assert staged["prefix"] == f"staged/{run_id}"
    assert set(staged["files"]) == {
        Path(entry["path"]).name for entry in manifest["outputs"].values()
    }
    for entry in manifest["outputs"].values():
        assert staged["files"][Path(entry["path"]).name]["sha256"] == entry["sha256"]
    # The local sums verify the directory as it is, evidence blocks included.
    for line in (out / "sha256sums.txt").read_text().splitlines():
        digest, name = line.split("  ")
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest, name
    inventory = json.loads((out / "staged_manifest.json").read_text())
    assert inventory["run_id"] == run_id and inventory["files"] == staged["files"]
    assert inventory["summary"]["releasable"] is True
    assert inventory["telemetry"] == {
        "repository": None,
        "prefix": f"runs/{run_id}",
        "mode": "local_only",
    }
    # The sidecars are evidence about the outputs, never outputs themselves.
    assert "sha256sums" not in manifest["outputs"]
    assert "staged_manifest" not in manifest["outputs"]


def test_size_candidate_stages_the_search_and_refit_phases(monkeypatch, tmp_path):
    from microcosm.build.staging_v2 import validate_v2_bundle

    builder = _load_builder_module()
    input_h5, ladder_path, flags = _staging_run_setup(builder, monkeypatch, tmp_path)
    out = tmp_path / "k300"
    status = builder.main(
        _build_args(
            input_h5,
            ladder_path,
            flags,
            out,
            "--dataset-households",
            "300",
            "--selection-seed",
            "11",
        )
    )
    assert status in (0, 1)
    run_id = _single_run_id(out)
    bundle = validate_v2_bundle(out / "staging", run_id)
    assert bundle["run_manifest"]["status"] == "completed"
    rows = json.loads(
        (out / "staging" / "runs" / run_id / "calibration_progress.json").read_text()
    )["events"]
    phases = [row["phase"] for row in rows]
    assert phases[0] is None and "size_search" in phases and phases[-1] == "size_refit"
    probe_rows = [row for row in rows if row["budget_search"] == 1]
    assert probe_rows and all(row["l0_lambda"] is not None for row in probe_rows)
    assert all(row["epoch"] == 2 for row in rows)
    calibration_done = next(
        e
        for e in bundle["events"]
        if e["stage_id"] == "calibration" and e["status"] == "completed"
    )
    assert calibration_done["details"]["size_checkpoint"] == "written"
    assert calibration_done["details"]["realized_households"] == 300
    manifest = json.loads((out / builder.MANIFEST_FILENAME).read_text())
    assert manifest["releasable"] is False
    assert manifest["staged_dataset"]["status"] == "skipped"
    assert "dataset_size_selection.csv" in manifest["staged_dataset"]["files"]
    fit_summary = json.loads(
        (
            out / "staging" / "runs" / run_id / "artifacts" / "fit_summary.json"
        ).read_text()
    )
    assert fit_summary["dataset_size"]["requested_households"] == 300
    assert "pool_row_indices" not in fit_summary["dataset_size"]
    assert fit_summary["releasable"] is False


def test_staging_epoch_stride_keeps_the_forwarded_rows_bounded():
    builder = _load_builder_module()

    def stride(epochs, households):
        return builder._staging_epoch_every(
            SimpleNamespace(epochs=epochs, dataset_households=households)
        )

    assert stride(2, None) == 10 and stride(2, 300) == 10
    assert stride(2000, None) == 10
    # dense + ten probes + refit at 2,000 epochs: 24,000 epochs -> 2,400 rows
    assert stride(2000, 55000) == 10
    assert stride(10000, 55000) == 50
    assert stride(100000, None) == 42
    for epochs, households in ((2000, 55000), (10000, 55000), (100000, None)):
        solves = 1 if households is None else 2 + builder._BUDGET_ITERS
        assert (
            epochs * solves / stride(epochs, households)
            <= builder._STAGING_MAX_EPOCH_ROWS
        )


def test_telemetry_content_refusal_never_aborts_the_solve(
    monkeypatch, tmp_path, capsys
):
    from microcosm.build.staging_v2 import StagingContentError, validate_v2_bundle

    builder = _load_builder_module()
    input_h5, ladder_path, flags = _staging_run_setup(builder, monkeypatch, tmp_path)

    class Refusing(builder.StagingTelemetryV2):
        def calibration_progress(self, event):
            raise StagingContentError("Staging file exceeds the 5242880-byte limit.")

    monkeypatch.setattr(builder, "StagingTelemetryV2", Refusing)
    out = tmp_path / "refused-rows"
    status = builder.main(_build_args(input_h5, ladder_path, flags, out))
    assert status == 0
    err = capsys.readouterr().err
    assert err.count("no longer forwarded") == 1
    run_id = _single_run_id(out)
    bundle = validate_v2_bundle(out / "staging", run_id)
    assert bundle["run_manifest"]["status"] == "completed"
    assert not (
        out / "staging" / "runs" / run_id / "calibration_progress.json"
    ).exists()
    manifest = json.loads((out / builder.MANIFEST_FILENAME).read_text())
    assert manifest["staging_delivery"]["mode"] == "local_only"
    assert load_spool_rows(out / "logbook-spool")[0].disposition == "iterating"


def test_invalid_local_telemetry_bundle_is_a_warning_not_the_runs_failure(
    monkeypatch, tmp_path, capsys
):
    from microcosm.build.staging_v2 import StagingContractError

    builder = _load_builder_module()
    input_h5, ladder_path, flags = _staging_run_setup(builder, monkeypatch, tmp_path)

    class Invalid(builder.StagingTelemetryV2):
        def validate_local_bundle(self):
            raise StagingContractError("synthetic bundle defect")

    monkeypatch.setattr(builder, "StagingTelemetryV2", Invalid)
    out = tmp_path / "invalid-bundle"
    status = builder.main(_build_args(input_h5, ladder_path, flags, out))
    assert status == 0
    err = capsys.readouterr().err
    assert "does not validate" in err and "synthetic bundle defect" in err
    manifest = json.loads((out / builder.MANIFEST_FILENAME).read_text())
    assert manifest["staging_delivery"]["mode"] == "local_only"
    assert manifest["staged_dataset"]["status"] == "skipped"
    rows = load_spool_rows(out / "logbook-spool")
    assert rows and rows[0].disposition == "iterating"


def test_no_staging_records_both_opt_outs(monkeypatch, tmp_path):
    builder = _load_builder_module()
    input_h5, ladder_path, flags = _staging_run_setup(
        builder, monkeypatch, tmp_path, remote=True
    )
    out = tmp_path / "quiet"
    status = builder.main(
        _build_args(input_h5, ladder_path, flags, out, "--no-staging")
    )
    assert status == 0
    assert not (out / "staging").exists()
    assert not (out / "sha256sums.txt").exists()
    manifest = json.loads((out / builder.MANIFEST_FILENAME).read_text())
    assert manifest["staging_delivery"]["mode"] == "disabled"
    assert manifest["staging_delivery"]["opt_out_reason"] == "--no-staging"
    assert manifest["staged_dataset"] == {
        "contract_version": 1,
        "mode": "disabled",
        "repository": None,
        "prefix": None,
        "run_id": None,
        "revision": None,
        "status": "skipped",
        "error_code": None,
        "opt_out_reason": "--no-staging",
        "files": {},
    }
    rows = load_spool_rows(out / "logbook-spool")
    assert "dataset_stage_skipped" in rows[0].phases_reached


def test_remote_staging_uploads_telemetry_and_the_bundle_in_one_commit(
    monkeypatch, tmp_path, capsys
):
    builder = _load_builder_module()
    input_h5, ladder_path, flags = _staging_run_setup(
        builder, monkeypatch, tmp_path, remote=True
    )
    hub = _FakeHub()
    monkeypatch.setattr(builder, "_hub_api", lambda: hub)
    monkeypatch.setattr(builder, "_hub_token", lambda: "hf_test_token")
    out = tmp_path / "remote"
    status = builder.main(
        _build_args(
            input_h5, ladder_path, flags, out, "--staging-upload-interval-seconds", "0"
        )
    )
    assert status == 0
    err = capsys.readouterr().err
    run_id = _single_run_id(out)
    manifest = json.loads((out / builder.MANIFEST_FILENAME).read_text())

    # Telemetry went to runs/<run_id>/ of the staging repository, artifacts
    # included, under the fixed prefix and nothing else.
    telemetry_paths = hub.paths("policyengine/populace-uk-staging")
    assert telemetry_paths == sorted(
        f"runs/{run_id}/{name}"
        for name in (
            "run_manifest.json",
            "progress.json",
            "events.ndjson",
            "calibration_progress.json",
            "artifacts/fit_summary.json",
            "artifacts/staged_dataset.json",
        )
    )
    delivery = manifest["staging_delivery"]
    assert delivery["mode"] == "local_and_remote"
    assert delivery["configured_repository"] == "policyengine/populace-uk-staging"
    assert delivery["upload_successes"] == delivery["upload_attempts"] > 0
    remote_progress = json.loads(
        hub.files[("policyengine/populace-uk-staging", f"runs/{run_id}/progress.json")]
    )
    assert remote_progress["status"] == "completed"

    # The bundle went to staged/<run_id>/ of the private repository in one
    # commit: every output, the manifest as built, and the two sidecars.
    assert len(hub.commits) == 1
    commit = hub.commits[0]
    assert commit["repo_id"] == "policyengine/populace-uk-private"
    expected = {Path(e["path"]).name for e in manifest["outputs"].values()} | {
        builder.MANIFEST_FILENAME,
        "staged_manifest.json",
        "sha256sums.txt",
    }
    assert commit["paths"] == sorted(f"staged/{run_id}/{name}" for name in expected)
    assert hub.paths("policyengine/populace-uk-private") == commit["paths"]
    staged = manifest["staged_dataset"]
    assert staged["status"] == "uploaded"
    assert staged["repository"] == "policyengine/populace-uk-private"
    assert staged["prefix"] == f"staged/{run_id}"
    assert staged["revision"] == hub.sha
    assert (
        f"staged dataset: uploaded at policyengine/populace-uk-private/staged/{run_id}"
        in err
    )
    remote_h5 = hub.files[
        (
            "policyengine/populace-uk-private",
            f"staged/{run_id}/{Path(manifest['outputs']['dataset']['path']).name}",
        )
    ]
    assert (
        remote_h5
        == (out / Path(manifest["outputs"]["dataset"]["path"]).name).read_bytes()
    )
    # The uploaded manifest is the one the bundle was built from; the local
    # copy gained the two evidence blocks afterwards.
    remote_manifest = json.loads(
        hub.files[
            (
                "policyengine/populace-uk-private",
                f"staged/{run_id}/{builder.MANIFEST_FILENAME}",
            )
        ]
    )
    assert (
        "staged_dataset" not in remote_manifest
        and "staging_delivery" not in remote_manifest
    )
    assert remote_manifest["outputs"] == manifest["outputs"]
    rows = load_spool_rows(out / "logbook-spool")
    assert "dataset_staged" in rows[0].phases_reached
    assert rows[0].disposition == "iterating"

    # Re-staging a directory whose record already says these outputs are
    # uploaded touches nothing: the driver's record and revision stand, the
    # sidecars keep their bytes, and no commit is made.
    stager = _load_tool("stage_uk_rowwise_candidate")
    monkeypatch.setattr(stager, "_hub_api", lambda: hub)
    sidecar_bytes = (out / "staged_manifest.json").read_bytes()
    capsys.readouterr()
    assert stager.main(["--run-dir", str(out)]) == 0
    assert "nothing to do" in capsys.readouterr().err
    restaged = json.loads((out / builder.MANIFEST_FILENAME).read_text())[
        "staged_dataset"
    ]
    assert restaged == staged
    assert (out / "staged_manifest.json").read_bytes() == sidecar_bytes
    for line in (out / "sha256sums.txt").read_text().splitlines():
        digest, name = line.split("  ")
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest, name
    assert len(hub.commits) == 1

    # A record that says the upload failed while the Hub already holds these
    # outputs: the re-stage finds the bundle and records its own commit, not
    # the repository head, which has moved on since.
    manifest_path = out / builder.MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    manifest["staged_dataset"] = {
        **staged,
        "status": "failed",
        "revision": None,
        "error_code": "UPLOAD_FAILED",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    hub.sha = "e" * 40
    assert stager.main(["--run-dir", str(out)]) == 0
    recovered = json.loads(manifest_path.read_text())["staged_dataset"]
    assert recovered["status"] == "already_staged"
    assert recovered["revision"] == staged["revision"] != hub.sha
    assert len(hub.commits) == 1
    for line in (out / "sha256sums.txt").read_text().splitlines():
        digest, name = line.split("  ")
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest, name

    # Consumers fetch by run id and get digest-verified local files.
    fetcher = _load_tool("fetch_uk_staged_dataset")
    monkeypatch.setattr(fetcher, "_hub_api", lambda: hub)
    dest = tmp_path / "fetched"
    capsys.readouterr()
    assert fetcher.main(["--run-id", run_id, "--dest", str(dest)]) == 0
    listed = capsys.readouterr().out.splitlines()
    assert str(dest / Path(manifest["outputs"]["dataset"]["path"]).name) in listed
    assert (dest / "sha256sums.txt").is_file()
    assert (
        dest / Path(manifest["outputs"]["dataset"]["path"]).name
    ).read_bytes() == remote_h5


def test_remote_staging_failure_is_recorded_and_the_build_still_succeeds(
    monkeypatch, tmp_path, capsys
):
    builder = _load_builder_module()
    input_h5, ladder_path, flags = _staging_run_setup(
        builder, monkeypatch, tmp_path, remote=True
    )
    hub = _FakeHub(fail_commit=True)
    monkeypatch.setattr(builder, "_hub_api", lambda: hub)
    monkeypatch.setattr(builder, "_hub_token", lambda: "hf_test_token")
    out = tmp_path / "failed-upload"
    status = builder.main(_build_args(input_h5, ladder_path, flags, out))
    assert status == 0
    err = capsys.readouterr().err
    assert "staged dataset upload failed" in err and "do-not-record" not in err
    manifest = json.loads((out / builder.MANIFEST_FILENAME).read_text())
    staged = manifest["staged_dataset"]
    assert staged["status"] == "failed" and staged["error_code"] == "UPLOAD_FAILED"
    assert staged["revision"] is None and staged["files"]
    assert "do-not-record" not in json.dumps(manifest)
    assert hub.paths("policyengine/populace-uk-private") == []
    # Telemetry still completed and recorded the outcome.
    run_id = _single_run_id(out)
    progress = json.loads(
        hub.files[("policyengine/populace-uk-staging", f"runs/{run_id}/progress.json")]
    )
    assert progress["status"] == "completed"
    events = [
        json.loads(line)
        for line in hub.files[
            ("policyengine/populace-uk-staging", f"runs/{run_id}/events.ndjson")
        ]
        .decode()
        .splitlines()
        if line
    ]
    done = next(
        e
        for e in events
        if e["stage_id"] == "dataset_staging" and e["status"] == "completed"
    )
    assert done["details"]["status"] == "failed"
    assert done["details"]["error_code"] == "UPLOAD_FAILED"
    rows = load_spool_rows(out / "logbook-spool")
    assert rows[0].disposition == "iterating"
    assert "dataset_stage_failed" in rows[0].phases_reached
    # The sidecars are in place for a later re-stage.
    assert (out / "sha256sums.txt").is_file() and (
        out / "staged_manifest.json"
    ).is_file()


def test_no_staged_dataset_keeps_telemetry_remote_and_the_bundle_local(
    monkeypatch, tmp_path
):
    builder = _load_builder_module()
    input_h5, ladder_path, flags = _staging_run_setup(
        builder, monkeypatch, tmp_path, remote=True
    )
    hub = _FakeHub()
    monkeypatch.setattr(builder, "_hub_api", lambda: hub)
    monkeypatch.setattr(builder, "_hub_token", lambda: "hf_test_token")
    out = tmp_path / "telemetry-only"
    status = builder.main(
        _build_args(input_h5, ladder_path, flags, out, "--no-staged-dataset")
    )
    assert status == 0
    assert hub.paths("policyengine/populace-uk-private") == []
    assert hub.commits == []
    assert hub.paths("policyengine/populace-uk-staging")
    manifest = json.loads((out / builder.MANIFEST_FILENAME).read_text())
    assert manifest["staging_delivery"]["mode"] == "local_and_remote"
    assert manifest["staged_dataset"]["mode"] == "disabled"
    assert manifest["staged_dataset"]["opt_out_reason"] == "--no-staged-dataset"
    assert not (out / "sha256sums.txt").exists()


def test_remote_dataset_staging_is_refused_up_front_without_credential_or_repo(
    monkeypatch, tmp_path, capsys
):
    builder = _load_builder_module()
    input_h5, ladder_path, flags = _staging_run_setup(
        builder, monkeypatch, tmp_path, remote=True
    )
    out = tmp_path / "refused"
    monkeypatch.setattr(builder, "_hub_token", lambda: None)
    with pytest.raises(ValueError, match="write credential"):
        builder.main(_build_args(input_h5, ladder_path, flags, out))
    assert not out.exists()

    class Unreachable:
        def repo_info(self, **kwargs):
            raise RuntimeError("503 token=do-not-record")

    monkeypatch.setattr(builder, "_hub_token", lambda: "hf_test_token")
    monkeypatch.setattr(builder, "_hub_api", lambda: Unreachable())
    with pytest.raises(ValueError, match="cannot reach") as info:
        builder.main(_build_args(input_h5, ladder_path, flags, out))
    assert "do-not-record" not in str(info.value)
    assert not out.exists()

    # A read token sees the private repository but cannot upload: refused
    # before the spine is read, not after the solve (the Hub answers 403).
    monkeypatch.setattr(builder, "_hub_api", lambda: _FakeHub(role="read"))
    with pytest.raises(ValueError, match="read-only"):
        builder.main(_build_args(input_h5, ladder_path, flags, out))
    assert not out.exists()

    # A fine-grained token scoped to another owner is refused the same way;
    # one scoped to the repository's organisation passes the pre-flight.
    user_scoped = [
        {"entity": {"type": "user", "name": "someone"}, "permissions": ["repo.write"]}
    ]
    monkeypatch.setattr(
        builder, "_hub_api", lambda: _FakeHub(role="fineGrained", scopes=user_scoped)
    )
    with pytest.raises(ValueError, match="repo.write"):
        builder.main(_build_args(input_h5, ladder_path, flags, out))
    assert not out.exists()
    org_scoped = [
        {
            "entity": {"type": "org", "name": "policyengine"},
            "permissions": ["repo.write"],
        }
    ]
    org_hub = _FakeHub(role="fineGrained", scopes=org_scoped)
    monkeypatch.setattr(builder, "_hub_api", lambda: org_hub)
    assert builder.main(
        _build_args(input_h5, ladder_path, flags, tmp_path / "org-scoped")
    ) in (0, 1)
    assert org_hub.commits and org_hub.commits[0]["repo_id"] == (
        "policyengine/populace-uk-private"
    )

    # Argument refusals cost nothing and come first: a missing credential is
    # never the reported reason when the arguments are wrong.
    monkeypatch.setattr(builder, "_hub_token", lambda: None)
    with pytest.raises(ValueError, match="only with --dry-run"):
        builder.main(
            _build_args(
                input_h5, ladder_path, flags, out, "--candidate-clone-counts", "2,3"
            )
        )
    assert not out.exists()

    # The re-stage tool refuses the same credential the same way.
    stager = _load_tool("stage_uk_rowwise_candidate")
    monkeypatch.setattr(stager, "_hub_api", lambda: _FakeHub(role="read"))
    local_out = tmp_path / "local"
    monkeypatch.setattr(builder, "_hub_token", lambda: None)
    assert builder.main(
        _build_args(input_h5, ladder_path, flags, local_out, "--staging-local-only")
    ) in (0, 1)
    with pytest.raises(SystemExit, match="read-only"):
        stager.main(["--run-dir", str(local_out)])

    # A dry run plans without staging, so it needs neither credential nor repo.
    capsys.readouterr()
    monkeypatch.setattr(builder, "_hub_token", lambda: None)
    assert (
        builder.main(_build_args(input_h5, ladder_path, flags, out, "--dry-run")) == 0
    )
    plan = json.loads(capsys.readouterr().out)
    assert plan["parameters"]["dataset_households"] is None
    assert not out.exists()


def _role_argv(tmp_path: Path, role: str, *extra: str) -> list[str]:
    return [
        "--input-h5",
        str(tmp_path / "spine.h5"),
        "--release-role",
        role,
        "--input-sha256",
        "2" * 64,
        "--ledger-facts",
        str(tmp_path / "ledger"),
        "--ledger-facts-sha256",
        "0" * 64,
        "--ledger-manifest-sha256",
        "1" * 64,
        "--out",
        str(tmp_path / "out"),
        *extra,
    ]


def _dense_argv(tmp_path: Path, *extra: str) -> list[str]:
    return _role_argv(
        tmp_path,
        "dense",
        "--ladder",
        str(tmp_path / "ladder.npz"),
        "--ladder-sha256",
        "3" * 64,
        *extra,
    )


def test_release_role_is_required(tmp_path) -> None:
    builder = _load_builder_module()
    argv = _dense_argv(tmp_path)
    argv.remove("--release-role")
    argv.remove("dense")
    with pytest.raises(SystemExit):
        builder._parse_args(argv)
    with pytest.raises(SystemExit):
        builder._parse_args([*argv, "--release-role", "local"])


def test_release_role_supplies_the_solve_defaults(tmp_path) -> None:
    builder = _load_builder_module()
    dense = builder._parse_args(_dense_argv(tmp_path))
    posture = builder.UK_ROWWISE_DENSE_POSTURE
    assert (dense.n_clones, dense.seed, dense.epochs, dense.learning_rate) == (
        posture.clone_count,
        posture.seed,
        posture.epochs,
        posture.learning_rate,
    )
    assert dense.target_weight_rule == "grain_equal"
    assert dense.expected_constituency_vintage == "2024_pcon"
    assert dense.staging_upload_interval_seconds == 300.0
    assert dense._explicit_arguments == frozenset()
    builder._validate_cli_args(dense)

    national = builder._parse_args(_role_argv(tmp_path, "national"))
    posture = builder.uk_rowwise_posture("national")
    assert national._posture is posture
    assert national.n_clones is None
    assert (national.seed, national.epochs, national.learning_rate) == (0, 1500, 0.02)
    assert national.target_weight_rule == "family_equal"
    assert national.expected_constituency_vintage is None
    builder._validate_cli_args(national)
    explicit = builder._parse_args(_role_argv(tmp_path, "national", "--epochs", "5"))
    assert explicit.epochs == 5
    assert explicit._explicit_arguments == frozenset({"epochs"})
    # The doctrine's own seed may be spelled out; only another seed is refused.
    builder._validate_cli_args(
        builder._parse_args(_role_argv(tmp_path, "national", "--seed", "0"))
    )


@pytest.mark.parametrize(
    ("extra", "needle"),
    [
        (["--target-loss-cap", "5"], "--target-loss-cap"),
        (["--allow-unpinned-feed"], "--allow-unpinned-feed"),
        (["--target-weight-rule", "family_equal"], "--target-weight-rule family_equal"),
    ],
)
def test_dense_role_refusal_table(tmp_path, extra, needle) -> None:
    builder = _load_builder_module()
    args = builder._parse_args(_dense_argv(tmp_path, *extra))
    with pytest.raises(ValueError, match="--release-role dense refuses") as excinfo:
        builder._validate_cli_args(args)
    assert needle in str(excinfo.value)


def test_dense_role_requires_the_ladder(tmp_path) -> None:
    builder = _load_builder_module()
    argv = _role_argv(tmp_path, "dense", "--ladder-sha256", "3" * 64)
    with pytest.raises(ValueError, match="requires --ladder"):
        builder._validate_cli_args(builder._parse_args(argv))


@pytest.mark.parametrize(
    ("extra", "needle"),
    [
        (["--ladder", "ladder.npz"], "--ladder"),
        (["--ladder-sha256", "3" * 64], "--ladder-sha256"),
        (["--expected-constituency-vintage", "2024_pcon"], "--expected-constituency"),
        (["--source-year", "2023"], "--source-year"),
        (["--source-lineage-modulus", "7"], "--source-lineage-modulus"),
        (["--n-clones", "15"], "--n-clones"),
        (["--candidate-clone-counts", "2,4", "--dry-run"], "--candidate-clone-counts"),
        (["--engine-blocks", "2"], "--engine-blocks"),
        (["--households-only"], "--households-only"),
        (["--skip-holdout"], "--skip-holdout"),
        (["--dataset-households", "10"], "--dataset-households"),
        (["--selection-seed", "3"], "--selection-seed"),
        (["--selection-pi-hi", "0.5"], "--selection-pi-hi"),
        (["--baseline-pi-floor", "0.1"], "--baseline-pi-floor"),
        (["--no-size-checkpoint"], "--no-size-checkpoint"),
        (["--resume-size-checkpoint", "dir"], "--resume-size-checkpoint"),
        (["--sample-fraction", "0.1"], "--sample-fraction"),
        (["--sample-seed", "9"], "--sample-seed"),
        (["--seed", "7"], "--seed != doctrine 0"),
        (["--target-weight-rule", "grain_equal"], "--target-weight-rule grain_equal"),
    ],
)
def test_national_role_refusal_table(tmp_path, extra, needle) -> None:
    builder = _load_builder_module()
    args = builder._parse_args(_role_argv(tmp_path, "national", *extra))
    with pytest.raises(ValueError, match="--release-role national refuses") as excinfo:
        builder._validate_cli_args(args)
    assert needle in str(excinfo.value)


def test_national_role_refuses_release_candidate_with_the_seam_reason(tmp_path):
    builder = _load_builder_module()
    args = builder._parse_args(_role_argv(tmp_path, "national", "--release-candidate"))
    with pytest.raises(ValueError, match="cannot sign shippability"):
        builder._validate_cli_args(args)
