"""Unit tests for the pure parts of tools/build_us_acs_local_release.py."""

from __future__ import annotations

import importlib.util
import json
import shlex
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Tests that write real H5 bytes go through pandas' HDFStore, which needs
# pytables; the base wheel gate installs the shards without it.
requires_pytables = pytest.mark.skipif(
    importlib.util.find_spec("tables") is None,
    reason="requires pytables (the build environment)",
)


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


def test_local_hours_gate_refuses_missing_source_audit() -> None:
    module = _load_tool_module()
    with pytest.raises(SystemExit, match="staging input-null audit"):
        module._require_local_hours(None, {})


def test_local_hours_failure_propagates_to_release_boundary(monkeypatch) -> None:
    from microcosm.build.gates import GateResult

    module = _load_tool_module()
    seen = []

    def failed_gate(frame, *, source_null_audit):
        seen.append((frame, source_null_audit))
        return GateResult(
            name="acs_local_hours_signal",
            passed=False,
            failures=("acs_2024_1yr: unresolved hours",),
        )

    monkeypatch.setattr(module, "acs_local_hours_signal_gate", failed_gate)
    marker = object()
    audit = [{"entity": "person", "column": "weekly_hours_worked_before_lsr"}]
    with pytest.raises(SystemExit, match="acs_2024_1yr: unresolved hours"):
        module._require_local_hours(marker, {"reviewed_engine_input_nulls": audit})
    assert seen == [(marker, audit)]


def _package_evidence_args(module, tmp_path: Path, monkeypatch, *, hours_report):
    """Every package-stage input, with the finalize report's hours entry given.

    ``hours_report`` is what ``gate_summary.json`` records under
    ``acs_local_hours_signal`` (``None`` omits the key, as a report finalized
    before the gate existed would). The staging frame and the hours gate are
    stubbed: the test is about the binding, not the classification.
    """
    from microcosm.build.gates import GateResult

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    staging = tmp_path / "staging.h5"
    staging.write_bytes(b"staging")
    (tmp_path / "staging.summary.json").write_text(
        json.dumps(
            {
                "reviewed_engine_input_nulls": [],
                "reviewed_limitations": [],
                # An uncapped staging run, so the package stage's cap check
                # (which runs first) lets these inputs reach the hours gates.
                "orchestration": {"max_households": None},
            }
        )
    )
    out_h5 = tmp_path / "out.h5"
    out_h5.write_bytes(b"artifact")
    artifact_sha = module._sha256(out_h5)
    gates = {
        "us_puma_ladder_gate": {"passed": True, "failures": []},
        # The general hours gate, bound to these bytes: the package stage
        # requires it before it reaches the ACS local-hours re-check.
        "hours_worked_signal": {
            "passed": True,
            "failures": [],
            "detail": {},
            "artifact_sha256": artifact_sha,
        },
    }
    if hours_report is not None:
        gates["acs_local_hours_signal"] = hours_report
    evidence = {
        "calibration_diagnostics.json": {"households": 1},
        "gate_summary.json": {"gates": gates, "reviewed_limitations": []},
        "run_identity.json": {
            "staging_sha256": module._sha256(staging),
            "population_cells_dropped": [],
        },
        "spine_qa.json": {
            "plain_consumption": True,
            "artifact_sha256": artifact_sha,
            "per_spine": {},
        },
        "consumer_export.json": {"staging_sha256": module._sha256(staging)},
        "held_back_columns.json": {"total": 0},
        "reviewed_null_fills.json": {"columns_filled": []},
        "materialize_rss.json": {"materialize_peak_rss_gb": 1.0, "hh_chunk": 1},
        "consumer_reviewed_null_fills.json": {"columns_filled": []},
    }
    for name, payload in evidence.items():
        (ckpt / name).write_text(json.dumps(payload))
    (tmp_path / "out.summary.json").write_text(json.dumps({"simulation_ready": True}))
    monkeypatch.setattr(module, "_load_staging_frame", lambda *_a, **_k: object())
    monkeypatch.setattr(
        module,
        "acs_local_hours_signal_gate",
        lambda frame, *, source_null_audit: GateResult(
            name="acs_local_hours_signal",
            passed=True,
            failures=(),
            details={"per_spine": {"acs_2024_1yr": {"rows": 1}}},
        ),
    )
    return module._parse_args(
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


@pytest.mark.parametrize(
    "hours_report",
    [None, {"passed": False, "failures": ["invented"]}, {"passed": "true"}],
    ids=["finalized-before-the-gate", "finalize-failed", "truthy-not-true"],
)
def test_package_requires_a_passing_hours_gate_in_the_finalize_report(
    tmp_path: Path, monkeypatch, hours_report
) -> None:
    module = _load_tool_module()
    args = _package_evidence_args(
        module, tmp_path, monkeypatch, hours_report=hours_report
    )
    with pytest.raises(SystemExit, match="Re-run --stage finalize"):
        module.do_package(args)
    assert not (args.out / "package_result.json").exists()


def test_package_binds_the_hours_gate_to_the_packaged_bytes(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_tool_module()
    args = _package_evidence_args(
        module,
        tmp_path,
        monkeypatch,
        hours_report={"passed": True, "failures": [], "detail": {}},
    )
    result = module.do_package(args)
    release_dir = Path(result["release_dir"])
    expected_sha = module._sha256(args.out_h5)
    for name in ("build_manifest.json", "gate_summary.json"):
        gate = json.loads((release_dir / name).read_text())["gates"][
            "acs_local_hours_signal"
        ]
        assert gate["passed"] is True
        assert gate["artifact_sha256"] == expected_sha
        assert gate["checked_at_stage"] == "package"
        assert gate["detail"] == {"per_spine": {"acs_2024_1yr": {"rows": 1}}}
    # The cap check that runs before the hours gates records what it passed.
    build_manifest = json.loads((release_dir / "build_manifest.json").read_text())
    assert build_manifest["staging_orchestration"]["max_households"] is None


def test_package_refuses_when_the_packaged_bytes_fail_the_hours_gate(
    tmp_path: Path, monkeypatch
) -> None:
    """A passing finalize entry does not stand in for the package-time re-check.

    The re-check runs on the calibrated H5 being packaged; if it fails, nothing
    ships: no manifest, no package result, no artifact at the release root.
    """
    from microcosm.build.gates import GateResult

    module = _load_tool_module()
    args = _package_evidence_args(
        module,
        tmp_path,
        monkeypatch,
        hours_report={"passed": True, "failures": [], "detail": {}},
    )
    loaded = []

    def load_frame(path, *_a, **_k):
        loaded.append(Path(path))
        return object()

    monkeypatch.setattr(module, "_load_staging_frame", load_frame)
    monkeypatch.setattr(
        module,
        "acs_local_hours_signal_gate",
        lambda frame, *, source_null_audit: GateResult(
            name="acs_local_hours_signal",
            passed=False,
            failures=("acs_2024_1yr: invented unresolved hours",),
        ),
    )
    with pytest.raises(
        SystemExit, match="Local hours coverage failed: acs_2024_1yr: invented"
    ):
        module.do_package(args)
    assert loaded == [Path(args.out_h5)]
    assert not (args.out / "package_result.json").exists()
    assert not list((args.out / "releases").rglob("*.json"))
    assert not (args.out / module.ARTIFACT_FILENAME).exists()


_UNSET = object()


def _package_args_before_evidence(module, tmp_path: Path, *, max_households=_UNSET):
    """The package stage's inputs up to (not including) the qa/consumer evidence.

    ``max_households`` is what the staging summary records under
    ``orchestration``; the sentinel omits the block entirely.
    """
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    staging = tmp_path / "staging.h5"
    staging.write_bytes(b"staging")
    summary: dict = {}
    if max_households is not _UNSET:
        summary["orchestration"] = {
            "max_households": max_households,
            "n_estimators": 32,
            "max_targets_per_fit": 8,
        }
    (tmp_path / "staging.summary.json").write_text(json.dumps(summary))
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
    (tmp_path / "out.summary.json").write_text(json.dumps({"simulation_ready": True}))
    return module._parse_args(
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


@pytest.mark.parametrize(
    ("max_households", "message"),
    [
        (5000, r"capped at 5000 ACS household"),
        (0, r"capped at 0 ACS household"),
        (_UNSET, r"does not record orchestration\.max_households"),
    ],
    ids=["capped", "capped-at-zero", "cap-not-recorded"],
)
def test_package_refuses_a_capped_or_unattested_staging_run(
    tmp_path: Path, max_households, message
) -> None:
    """A capped smoke passes every finalize gate (the donor spine keeps every
    ladder cell populated), so the cap itself must refuse packaging, before
    any release directory exists."""

    module = _load_tool_module()
    args = _package_args_before_evidence(
        module, tmp_path, max_households=max_households
    )
    with pytest.raises(SystemExit, match=message):
        module.do_package(args)
    assert not (args.out / "package_result.json").exists()
    assert not (args.out / "releases").exists(), "a refused smoke leaves no release"


def test_package_checks_the_cap_before_reading_the_evidence(tmp_path: Path) -> None:
    """An uncapped summary reaches the evidence checks; the cap check is first."""

    module = _load_tool_module()
    args = _package_args_before_evidence(module, tmp_path, max_households=None)
    with pytest.raises(SystemExit, match="spine_qa.json is missing"):
        module.do_package(args)


def test_do_package_requires_qa_and_consumer_evidence(tmp_path: Path) -> None:
    """Absent evidence must refuse packaging, never read as vacuously green."""

    module = _load_tool_module()
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    staging = tmp_path / "staging.h5"
    staging.write_bytes(b"staging")
    (tmp_path / "staging.summary.json").write_text(
        json.dumps({"orchestration": {"max_households": None}})
    )
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


def _staging_frame_with_hours(weekly: list[float], last_week: list[float]):
    """A minimal US-schema staging frame carrying the two pool hours columns."""

    from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

    n = len(weekly)
    ids = np.arange(1, n + 1)
    person = pd.DataFrame(
        {
            "person_id": ids,
            "person_household_id": ids,
            "person_tax_unit_id": ids,
            "person_spm_unit_id": ids,
            "person_family_id": ids,
            "person_marital_unit_id": ids,
            "weekly_hours_worked_before_lsr": np.asarray(weekly, dtype=float),
            "hours_worked_last_week": np.asarray(last_week, dtype=float),
        }
    )
    tables = {
        "person": person,
        "household": pd.DataFrame({"household_id": ids}),
        "tax_unit": pd.DataFrame({"tax_unit_id": ids}),
        "spm_unit": pd.DataFrame({"spm_unit_id": ids}),
        "family": pd.DataFrame({"family_id": ids}),
        "marital_unit": pd.DataFrame({"marital_unit_id": ids}),
    }
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.ones(n, dtype=np.float64), WeightKind.DESIGN)},
    )


def _finalize_args(module, tmp_path: Path):
    staging = tmp_path / "staging.h5"
    staging.touch()
    (tmp_path / "staging.summary.json").write_text(
        json.dumps(
            {
                "reviewed_limitations": [],
                "reviewed_engine_input_nulls": [],
                # The current staging builder records its cap; an uncapped run
                # is what the package-stage tests built on this fixture need.
                "orchestration": {"max_households": None},
            }
        )
    )
    # finalize hashes the calibrated H5 before loading it; tests that do not
    # load real bytes still need bytes to hash.
    (tmp_path / "out.h5").write_bytes(b"invented-finalize-artifact")
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir(exist_ok=True)
    (ckpt / "calibration_diagnostics.json").write_text(
        json.dumps(
            {"final_loss": 0.1, "initial_loss": 0.5, "mass_conserved_ratio": 1.0}
        )
    )
    ladder = tmp_path / "ladder.npz"
    ladder.write_bytes(b"ladder-bytes")
    return module._parse_args(
        [
            "--stage",
            "finalize",
            "--staging-h5",
            str(staging),
            "--checkpoint-dir",
            str(ckpt),
            "--out-h5",
            str(tmp_path / "out.h5"),
            "--ladder",
            str(ladder),
        ]
    )


def _stub_local_hours_gate(module, monkeypatch) -> None:
    """Make the ACS local-hours classification pass; its own tests cover it."""

    from microcosm.build.gates import GateResult

    monkeypatch.setattr(
        module,
        "acs_local_hours_signal_gate",
        lambda frame, *, source_null_audit: GateResult(
            name="acs_local_hours_signal", passed=True, failures=(), details={}
        ),
    )


def _patch_finalize_collaborators(module, monkeypatch, frame=None, *, identity=True):
    """Stub the ladder gate and composition; ``frame=None`` loads real bytes."""

    import microcosm.build.us_runtime.puma_ladder as puma
    from microcosm.build.gates import GateResult

    monkeypatch.setattr(puma, "load_us_puma_ladder", lambda *a, **k: None)
    monkeypatch.setattr(
        puma,
        "us_puma_ladder_gate",
        lambda *a, **k: GateResult(
            name="us_puma_ladder", passed=True, failures=(), details={}
        ),
    )
    monkeypatch.setattr(module, "spine_composition", lambda *a, **k: {})
    _stub_local_hours_gate(module, monkeypatch)
    if frame is not None:
        monkeypatch.setattr(module, "_load_staging_frame", lambda *a, **k: frame)
    if identity:
        monkeypatch.setattr(
            module,
            "_verify_run_identity",
            lambda a: {"ladder_sha256": module._sha256(a.ladder)},
        )


def _run_finalize(module, monkeypatch, args, frame=None):
    _patch_finalize_collaborators(module, monkeypatch, frame)
    with pytest.raises(SystemExit) as exc:
        module.do_finalize(args)
    report = (
        json.loads(args.gate_report.read_text()) if args.gate_report.exists() else None
    )
    return str(exc.value), report


def _write_frame_h5(path: Path, frame) -> None:
    """Write a staging frame as the tool's loader reads it (fixed format)."""

    from microcosm.frame import put_frame_table

    with pd.HDFStore(path, mode="w") as store:
        for entity in frame.entities:
            table = frame.table(entity).copy()
            if entity == "household":
                table["household_weight"] = frame.weights_for(entity).values
            put_frame_table(store, entity, table, preferred_format="fixed")


def _plausible_hours_frame():
    return _staging_frame_with_hours(
        [40.0, 38.0, 20.0, 45.0, 0.0, 0.0, 0.0, 0.0],
        [40.0, 35.0, 22.0, 40.0, 0.0, 0.0, 0.0, 5.0],
    )


def test_do_finalize_hard_fails_on_constant_forty_hours(tmp_path, monkeypatch) -> None:
    # microcosm#765: an artifact whose usual weekly hours are the engine's
    # constant-40 default must block packaging via the finalize hard gate.
    module = _load_tool_module()
    args = _finalize_args(module, tmp_path)
    frame = _staging_frame_with_hours([40.0] * 8, [40.0] * 8)
    message, report = _run_finalize(module, monkeypatch, args, frame)
    assert "hours_worked_signal" in message
    assert report["gates"]["hours_worked_signal"]["passed"] is False


def test_do_finalize_hours_gate_passes_on_plausible_surface(
    tmp_path, monkeypatch
) -> None:
    module = _load_tool_module()
    args = _finalize_args(module, tmp_path)
    weekly = [40.0, 38.0, 20.0, 45.0, 0.0, 0.0, 0.0, 0.0]
    last_week = [40.0, 35.0, 22.0, 40.0, 0.0, 0.0, 0.0, 5.0]
    frame = _staging_frame_with_hours(weekly, last_week)
    _message, report = _run_finalize(module, monkeypatch, args, frame)
    assert report["gates"]["hours_worked_signal"]["passed"] is True


@requires_pytables
def test_finalize_binds_the_hours_gate_to_the_calibrated_artifact_bytes(
    tmp_path, monkeypatch
) -> None:
    """The gate entry must carry the digest of the bytes it evaluated."""

    module = _load_tool_module()
    args = _finalize_args(module, tmp_path)
    _write_frame_h5(args.out_h5, _plausible_hours_frame())
    hashed = []
    original_sha = module._sha256

    def recording_sha(path):
        result = original_sha(path)
        if Path(path) == args.out_h5:
            hashed.append(result)
        return result

    monkeypatch.setattr(module, "_sha256", recording_sha)
    loads = []
    real_load = module._load_staging_frame
    monkeypatch.setattr(
        module,
        "_load_staging_frame",
        lambda p: (loads.append(len(hashed)), real_load(p))[1],
    )
    _message, report = _run_finalize(module, monkeypatch, args)
    gate = report["gates"]["hours_worked_signal"]
    assert gate["passed"] is True
    assert gate["artifact_sha256"] == original_sha(args.out_h5)
    # Hashed once before the frame was loaded, then re-checked after the gate.
    assert loads == [1]
    assert hashed == [gate["artifact_sha256"]] * 2


@requires_pytables
def test_finalize_refuses_artifact_changed_during_hours_validation(
    tmp_path, monkeypatch
) -> None:
    import microcosm.build.us_runtime.hours_worked as hours_worked

    module = _load_tool_module()
    args = _finalize_args(module, tmp_path)
    _write_frame_h5(args.out_h5, _plausible_hours_frame())
    original_gate = hours_worked.us_hours_worked_signal_gate

    def changed_artifact(frame, **kwargs):
        result = original_gate(frame, **kwargs)
        args.out_h5.write_bytes(b"different-invented-artifact")
        return result

    monkeypatch.setattr(hours_worked, "us_hours_worked_signal_gate", changed_artifact)
    message, report = _run_finalize(module, monkeypatch, args)
    assert "changed during hours_worked_signal validation" in message
    assert report is None
    assert not args.out_summary.exists()


@requires_pytables
def test_finalize_report_round_trips_into_package(tmp_path, monkeypatch) -> None:
    """A finalize-written report must satisfy the package stage's binding."""

    module = _load_tool_module()
    args = _finalize_args(module, tmp_path)
    _write_frame_h5(args.out_h5, _plausible_hours_frame())
    artifact_sha = module._sha256(args.out_h5)
    evidence = {
        "run_identity.json": {
            "staging_sha256": module._sha256(args.staging_h5),
            "ladder_sha256": module._sha256(args.ladder),
            "population_cells_dropped": [],
        },
        "spine_qa.json": {
            "plain_consumption": True,
            "artifact_sha256": artifact_sha,
            "per_spine": {},
        },
        "consumer_export.json": {"staging_sha256": artifact_sha},
        "consumer_reviewed_null_fills.json": {"columns_filled": []},
    }
    for name, value in evidence.items():
        (args.checkpoint_dir / name).write_text(json.dumps(value))
    _patch_finalize_collaborators(module, monkeypatch, identity=False)
    module.do_finalize(args)
    report = json.loads(args.gate_report.read_text())
    assert report["gates"]["hours_worked_signal"]["artifact_sha256"] == artifact_sha
    assert json.loads(args.out_summary.read_text())["simulation_ready"] is True

    args.out = tmp_path / "release"
    args.allow_dirty = True
    result = module.do_package(args)
    shipped = json.loads(
        (Path(result["release_dir"]) / "gate_summary.json").read_text()
    )["gates"]["hours_worked_signal"]
    assert shipped["passed"] is True
    assert (
        shipped["artifact_sha256"]
        == result["root_artifact"]["sha256"]
        == module._sha256(Path(result["root_artifact"]["local_path"]))
        == artifact_sha
    )


def _package_args_with_hours(module, tmp_path, monkeypatch, *, gate_state):
    """Real tiny H5 bytes; gate edits model stale separately resumed finalize."""

    _stub_local_hours_gate(module, monkeypatch)
    args = _finalize_args(module, tmp_path)
    args.out = tmp_path / "release"
    args.allow_dirty = True
    _write_frame_h5(args.out_h5, _plausible_hours_frame())
    artifact_sha = module._sha256(args.out_h5)
    gate = {"passed": True, "failures": [], "artifact_sha256": artifact_sha}
    if gate_state == "failed":
        gate.update(passed=False, failures=["invented hours failure"])
    elif gate_state == "truthy":
        gate["passed"] = "true"
    elif gate_state == "unbound":
        del gate["artifact_sha256"]
    elif gate_state == "stale":
        gate["artifact_sha256"] = "0" * 64
    local_gate = {
        "passed": True,
        "failures": [],
        "detail": {},
        "artifact_sha256": artifact_sha,
    }
    gates = (
        {}
        if gate_state == "missing"
        else {"hours_worked_signal": gate, "acs_local_hours_signal": local_gate}
    )
    args.gate_report.write_text(json.dumps({"gates": gates}))
    args.out_summary.write_text(json.dumps({"simulation_ready": True}))
    evidence = {
        "run_identity.json": {
            "staging_sha256": module._sha256(args.staging_h5),
            "population_cells_dropped": [],
        },
        "spine_qa.json": {
            "plain_consumption": True,
            "artifact_sha256": artifact_sha,
            "per_spine": {},
        },
        "consumer_export.json": {"staging_sha256": artifact_sha},
        "consumer_reviewed_null_fills.json": {"columns_filled": []},
    }
    for name, value in evidence.items():
        (args.checkpoint_dir / name).write_text(json.dumps(value))
    return args


@requires_pytables
@pytest.mark.parametrize(
    "gate_state", ["missing", "failed", "truthy", "unbound", "stale"]
)
def test_package_requires_current_hours_gate_even_with_green_old_summary(
    tmp_path, monkeypatch, gate_state
):
    module = _load_tool_module()
    args = _package_args_with_hours(
        module, tmp_path, monkeypatch, gate_state=gate_state
    )
    with pytest.raises(SystemExit, match="hours_worked_signal"):
        module.do_package(args)
    assert not (args.out / "package_result.json").exists()


@requires_pytables
def test_package_accepts_passing_hours_gate_bound_to_the_packaged_bytes(
    tmp_path, monkeypatch
):
    module = _load_tool_module()
    args = _package_args_with_hours(module, tmp_path, monkeypatch, gate_state="passed")
    result = module.do_package(args)
    release_dir = Path(result["release_dir"])
    gate = json.loads((release_dir / "gate_summary.json").read_text())["gates"][
        "hours_worked_signal"
    ]
    copied_sha = module._sha256(Path(result["root_artifact"]["local_path"]))
    assert gate["passed"] is True
    assert gate["artifact_sha256"] == result["root_artifact"]["sha256"] == copied_sha


@requires_pytables
@pytest.mark.parametrize("copy_state", ["new", "already_present", "no_copy"])
def test_package_rechecks_final_bytes_after_copy_or_reuse(
    tmp_path, monkeypatch, copy_state
):
    import shutil as real_shutil

    module = _load_tool_module()
    args = _package_args_with_hours(module, tmp_path, monkeypatch, gate_state="passed")
    root_copy = args.out / module.ARTIFACT_FILENAME
    original_sha = module._sha256
    root_hashes = []
    if copy_state == "new":

        def changed_copy(source, destination):
            result = real_shutil.copy2(source, destination)
            Path(destination).write_bytes(b"changed-during-copy")
            return result

        class _ToolShutil:
            """Rebind only the tool's own ``shutil`` name, not the global module."""

            copy2 = staticmethod(changed_copy)

            def __getattr__(self, name):
                return getattr(real_shutil, name)

        monkeypatch.setattr(module, "shutil", _ToolShutil())
    elif copy_state == "already_present":
        root_copy.parent.mkdir(parents=True)
        root_copy.write_bytes(args.out_h5.read_bytes())

        def changed_after_reuse_check(path):
            result = original_sha(path)
            if Path(path) == root_copy:
                root_hashes.append(result)
                if len(root_hashes) == 1:
                    root_copy.write_bytes(b"changed-after-reuse-check")
            return result

        monkeypatch.setattr(module, "_sha256", changed_after_reuse_check)
    else:
        # The artifact root IS the calibrated H5, so nothing is copied and only
        # the final re-hash can notice a source that changed after the gates.
        root_copy.parent.mkdir(parents=True)
        args.out_h5.rename(root_copy)
        args.out_h5 = root_copy
        releases = args.out / "releases"
        changed = []

        def changed_while_writing_sums(path):
            if not changed and releases in Path(path).parents:
                changed.append(path)
                with root_copy.open("ab") as stream:
                    stream.write(b"changed-after-the-gates")
            return original_sha(path)

        monkeypatch.setattr(module, "_sha256", changed_while_writing_sums)
    with pytest.raises(SystemExit, match="packaged H5.*hours_worked_signal"):
        module.do_package(args)
    assert not (args.out / "package_result.json").exists()
    if copy_state == "already_present":
        assert len(root_hashes) == 2, "the reuse check and the final re-hash"
    if copy_state == "no_copy":
        assert root_copy.exists(), "the calibrated H5 itself is never removed"
    else:
        assert not root_copy.exists(), "a refused copy is not left at the root"
