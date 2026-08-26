"""Unit tests for the pure parts of tools/build_us_acs_local_release.py."""

from __future__ import annotations

import importlib.util
import json
import shlex
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.gates import GateResult
from microcosm.build.us_runtime import puma_ladder
from microcosm.build.us_runtime.hours_worked import (
    US_HOURS_WORKED_POOL_OUTPUT_COLUMNS,
    us_hours_worked_signal_gate,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights


def _load_tool_module():
    root = Path(__file__).resolve().parents[3]
    path = root / "tools" / "build_us_acs_local_release.py"
    spec = importlib.util.spec_from_file_location(
        "build_us_acs_local_release",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_staging_builder_module():
    root = Path(__file__).resolve().parents[3]
    path = root / "tools" / "build_us_acs_multispine_base.py"
    spec = importlib.util.spec_from_file_location(
        "build_us_acs_multispine_base",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _defaulted_local_hours_frame() -> Frame:
    ids = np.arange(1, 9)
    spines = np.asarray(["asec_puf"] * 4 + ["acs_2024_1yr"] * 4)
    person = pd.DataFrame(
        {
            "person_id": ids,
            "person_household_id": ids,
            "person_tax_unit_id": ids,
            "person_spm_unit_id": ids,
            "person_family_id": ids,
            "person_marital_unit_id": ids,
            "person_spine": spines,
            # Reproduce #765's published local artifact, which materialized
            # the then-current 40-hour engine default on every ACS person.
            "weekly_hours_worked_before_lsr": [0, 0, 35, 40, 40, 40, 40, 40],
            "hours_worked_last_week": [0, 0, 30, 42, 0, 20, 40, 0],
        }
    )
    return Frame(
        {
            "person": person,
            "household": pd.DataFrame(
                {
                    "household_id": ids,
                    "household_spine": spines,
                }
            ),
            "tax_unit": pd.DataFrame({"tax_unit_id": ids}),
            "spm_unit": pd.DataFrame({"spm_unit_id": ids}),
            "family": pd.DataFrame({"family_id": ids}),
            "marital_unit": pd.DataFrame({"marital_unit_id": ids}),
        },
        US_SCHEMA,
        {
            "household": Weights(
                np.asarray([10, 10, 10, 10, 1, 1, 1, 1], dtype=np.float64),
                WeightKind.CALIBRATED,
            )
        },
    )


def test_spine_composition_reports_per_spine_weight_and_size() -> None:
    module = _load_tool_module()
    households = pd.DataFrame(
        {
            "household_id": [1, 2, 3],
            "household_spine": ["asec_puf", "asec_puf", "acs_2024_1yr"],
        }
    )
    persons = pd.DataFrame({"person_household_id": [1, 2, 2, 3, 3, 3]})
    weights = np.asarray([10.0, 30.0, 60.0])

    composition = module.spine_composition(households, persons, weights)

    asec = composition["asec_puf"]
    assert asec["households"] == 2
    assert asec["household_weight"] == pytest.approx(40.0)
    assert asec["household_weight_share"] == pytest.approx(0.4)
    # person weight: 10*1 + 30*2 = 70 -> 70/40 persons per household.
    assert asec["person_weight"] == pytest.approx(70.0)
    assert asec["implied_persons_per_household"] == pytest.approx(1.75)
    acs = composition["acs_2024_1yr"]
    assert acs["implied_persons_per_household"] == pytest.approx(3.0)
    overall = composition["_all"]
    assert overall["households"] == 3
    assert overall["effective_sample_size"] == pytest.approx(
        (10 + 30 + 60) ** 2 / (10**2 + 30**2 + 60**2)
    )


def test_finalize_reviewed_limitations_carries_staging_and_dedupes() -> None:
    module = _load_tool_module()
    staging_summary = {
        "base": {
            "donor_release": {
                "release_id": "populace-us-2024-buildo-sparse-rmloss100-x",
            }
        },
        "reviewed_limitations": [
            {"id": "acs_group_quarters_housing_universe", "status": "reviewed"},
            {
                "id": "cd_population_marginal_vintage_2020",
                "status": "stale_staging_copy",
            },
        ],
    }
    diagnostics = {
        "effective_sample_size": 4853.9,
        "ess_fraction": 0.003,
        "households": 1_600_000,
    }
    spine_qa = {
        "per_spine": {
            "acs_2024_1yr": {"ssi_incidence": 0.0259},
            "asec_puf": {"ssi_incidence": 0.0185},
        }
    }

    limitations = module.finalize_reviewed_limitations(
        staging_summary, diagnostics, spine_qa
    )

    by_id = {item["id"]: item for item in limitations}
    # Staging entries carried; finalize's own entry wins the id collision.
    assert "acs_group_quarters_housing_universe" in by_id
    assert by_id["cd_population_marginal_vintage_2020"]["status"] == (
        "reviewed_vintage"
    )
    # Lineage entries present, none blocking.
    for required in (
        "ssi_aged_band_collapse_inherited",
        "miscellaneous_income_loss_side_donor_defect",
        "tips_return_count_carrier_deficit_inherited",
        "low_effective_sample_size_lambda_zero",
        "donor_sparse_selection_training_set",
        "mixed_sub_puma_column_coverage",
    ):
        assert required in by_id, required
        assert by_id[required]["calibration_blocker"] is False
    ssi = by_id["ssi_aged_band_collapse_inherited"]
    assert ssi["measured_spine_ssi"] == {
        "acs_2024_1yr": 0.0259,
        "asec_puf": 0.0185,
    }
    donor = by_id["donor_sparse_selection_training_set"]
    assert "populace-us-2024-buildo-sparse-rmloss100-x" in donor["reason"]
    # Ids are unique after dedupe.
    assert len(by_id) == len(limitations)


def test_parse_args_enforces_stage_requirements(tmp_path: Path) -> None:
    module = _load_tool_module()
    base = [
        "--staging-h5",
        str(tmp_path / "staging.h5"),
        "--checkpoint-dir",
        str(tmp_path / "ckpt"),
    ]

    with pytest.raises(SystemExit):
        module._parse_args(["--stage", "materialize", *base])
    with pytest.raises(SystemExit):
        module._parse_args(["--stage", "calibrate", *base])
    with pytest.raises(SystemExit):
        module._parse_args(
            ["--stage", "package", *base, "--out-h5", str(tmp_path / "o.h5")]
        )

    args = module._parse_args(
        [
            "--stage",
            "all",
            *base,
            "--feed",
            str(tmp_path / "facts.jsonl"),
            "--out-h5",
            str(tmp_path / "out.h5"),
            "--out",
            str(tmp_path / "release"),
        ]
    )
    assert args.stages == ["materialize", "calibrate", "qa", "finalize", "package"]
    assert args.out_summary == tmp_path / "out.summary.json"
    assert args.gate_report == tmp_path / "ckpt" / "gate_summary.json"


def test_release_id_prefix_and_manifest_constants() -> None:
    module = _load_tool_module()
    assert module.RELEASE_ID_PREFIX == "populace-us-2024-buildo-acs-local"
    assert module.RELEASE_NAMESPACE == "buildo_acs_local"
    assert module.ARTIFACT_FILENAME == "populace_us_2024_acs_local.h5"
    assert module.HF_REPO_ID == "policyengine/populace-us"


def test_documented_staging_recipe_matches_legacy_and_release_parsers(
    tmp_path: Path,
) -> None:
    release = _load_tool_module()
    staging_builder = _load_staging_builder_module()
    recipe = shlex.split(release.LEGACY_STAGING_REFRESH_RECIPE)

    assert recipe[:3] == [
        "uv",
        "run",
        "tools/build_us_acs_multispine_base.py",
    ]
    staging_args = staging_builder._legacy._parse_args(recipe[3:])
    staging_summary = staging_args.summary or staging_args.out_h5.with_suffix(
        ".summary.json"
    )
    release_args = release._parse_args(
        [
            "--stage",
            "materialize",
            "--staging-h5",
            str(staging_args.out_h5),
            "--checkpoint-dir",
            str(tmp_path / "ckpt"),
            "--feed",
            str(tmp_path / "facts.jsonl"),
        ]
    )

    assert release._staging_summary_path(release_args) == staging_summary


def test_do_finalize_requires_calibration_diagnostics(tmp_path: Path) -> None:
    module = _load_tool_module()
    staging = tmp_path / "staging.h5"
    staging.touch()
    (tmp_path / "staging.summary.json").write_text(json.dumps({}))
    args = module._parse_args(
        [
            "--stage",
            "finalize",
            "--staging-h5",
            str(staging),
            "--checkpoint-dir",
            str(tmp_path / "ckpt"),
            "--out-h5",
            str(tmp_path / "out.h5"),
        ]
    )

    with pytest.raises(SystemExit, match="No calibration diagnostics"):
        module.do_finalize(args)


def test_acs_local_hours_gate_accepts_modeled_acs_signal() -> None:
    module = _load_tool_module()
    frame = _defaulted_local_hours_frame()
    person = frame.table("person")
    acs = person["person_spine"].eq("acs_2024_1yr")
    person.loc[acs, "weekly_hours_worked_before_lsr"] = [0, 0, 35, 40]

    gate = module.acs_local_hours_worked_signal_gate(frame)

    assert gate.passed
    assert gate.details["spine"] == "acs_2024_1yr"
    assert gate.details["checked_person_rows"] == 4
    assert gate.details["checked_columns"] == list(US_HOURS_WORKED_POOL_OUTPUT_COLUMNS)


def test_do_finalize_rejects_defaulted_acs_weekly_hours(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_tool_module()
    frame = _defaulted_local_hours_frame()
    whole_gate = us_hours_worked_signal_gate(
        frame,
        required_columns=US_HOURS_WORKED_POOL_OUTPUT_COLUMNS,
    )
    assert whole_gate.passed

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    staging = tmp_path / "staging.h5"
    staging.write_bytes(b"staging")
    (tmp_path / "staging.summary.json").write_text("{}")
    out_h5 = tmp_path / "out.h5"
    out_h5.write_bytes(b"artifact")
    ladder = tmp_path / "ladder.npz"
    ladder.write_bytes(b"ladder")
    (ckpt / "calibration_diagnostics.json").write_text(
        json.dumps(
            {
                "initial_loss": 1.0,
                "final_loss": 0.5,
                "mass_conserved_ratio": 1.0,
                "effective_sample_size": 8.0,
                "ess_fraction": 1.0,
                "households": 8,
                "n_targets": 0,
            }
        )
    )
    (ckpt / "targets.json").write_text("[]")
    (ckpt / "materialize_rss.json").write_text("{}")
    (ckpt / "spine_qa.json").write_text(
        json.dumps(
            {
                "plain_consumption": True,
                "artifact_sha256": "fixture-sha256",
                "per_spine": {},
            }
        )
    )
    (ckpt / "consumer_export.json").write_text(json.dumps({"filled_columns": 1}))
    args = module._parse_args(
        [
            "--stage",
            "finalize",
            "--staging-h5",
            str(staging),
            "--checkpoint-dir",
            str(ckpt),
            "--out-h5",
            str(out_h5),
            "--ladder",
            str(ladder),
        ]
    )
    monkeypatch.setattr(module, "_load_staging_frame", lambda path: frame)
    monkeypatch.setattr(
        module,
        "_verify_run_identity",
        lambda actual_args: {"ladder_sha256": module._sha256(ladder)},
    )
    monkeypatch.setattr(puma_ladder, "load_us_puma_ladder", lambda path: object())
    monkeypatch.setattr(
        puma_ladder,
        "us_puma_ladder_gate",
        lambda households, weights: GateResult(
            name="us_puma_ladder",
            passed=True,
            details={},
        ),
    )

    with pytest.raises(SystemExit, match="hours_worked_signal"):
        module.do_finalize(args)

    report = json.loads(args.gate_report.read_text())
    hours_gate = report["gates"]["hours_worked_signal"]
    assert not hours_gate["passed"]
    assert hours_gate["detail"]["spine"] == "acs_2024_1yr"
    assert hours_gate["detail"]["checked_person_rows"] == 4
    assert any(
        "weekly_hours_worked_before_lsr: constant" in failure
        for failure in hours_gate["failures"]
    )
    summary = json.loads(args.out_summary.read_text())
    assert not summary["simulation_ready"]
    assert summary["simulation_readiness_blockers"] == ["hours_worked_signal"]


def test_do_package_requires_qa_and_consumer_evidence(tmp_path: Path) -> None:
    """Absent evidence must refuse packaging, never read as vacuously green."""

    module = _load_tool_module()
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    staging = tmp_path / "staging.h5"
    staging.write_bytes(b"staging")
    (tmp_path / "staging.summary.json").write_text("{}")
    out_h5 = tmp_path / "out.h5"
    out_h5.write_bytes(b"artifact")
    (ckpt / "calibration_diagnostics.json").write_text(json.dumps({"households": 1}))
    (ckpt / "gate_summary.json").write_text(json.dumps({"gates": {}}))
    (ckpt / "run_identity.json").write_text(
        json.dumps(
            {
                "staging_sha256": module._sha256(staging),
                "population_cells_dropped": [],
            }
        )
    )
    args = module._parse_args(
        [
            "--stage",
            "package",
            "--staging-h5",
            str(staging),
            "--checkpoint-dir",
            str(ckpt),
            "--out-h5",
            str(out_h5),
            "--out",
            str(tmp_path / "release"),
            "--allow-dirty",
        ]
    )
    (tmp_path / "out.summary.json").write_text(json.dumps({"simulation_ready": True}))

    with pytest.raises(SystemExit, match="spine_qa.json is missing"):
        module.do_package(args)

    (ckpt / "spine_qa.json").write_text(
        json.dumps(
            {
                "plain_consumption": True,
                "artifact_sha256": module._sha256(out_h5),
                "per_spine": {},
            }
        )
    )
    with pytest.raises(SystemExit, match="consumer_export.json is missing"):
        module.do_package(args)
