"""The post-export probe end to end on a fake engine (``tools/probe_us_post_export.py``).

The sampler writes a stratified subsample of a nested synthetic export; the
probe runs every stage on it through the release tool's own functions, with
the post-export scoring tests' fake engine returning float32, uprated weights
as policyengine-core does. Invariants:

- every stage completes, the scorer cuts at least three batches, and each
  stage's artifact is written under the release tool's file name;
- every smoke probe decomposes: the per-household effects, summed with the
  engine's own row weights, reproduce the gate's effect, and the mapping
  recovers the uprating factor (1 for 2024, the population ratio for 2026);
- a census probe (no receipt) of the source export scores exactly what the
  gate scores on one whole-frame engine (differential), and all its smoke
  verdicts are authoritative;
- against that census as the reference release, the subsample's baseline
  plans and reform passes equal the reference's, its effects carry z-scores,
  and a stored column the engine does not define fails the stored-input gate
  authoritatively;
- the target surface resolves from the reference release manifest.
"""

# ruff: noqa: F403, F405
from __future__ import annotations

import json
import math

import pytest

from test_support.microcosm_build.us_post_export_probe import *


@pytest.fixture
def chain(sampler, tmp_path):
    """A 48-household export and its p = 0.5 subsample."""
    frame = synthetic_export_frame(48, seed=5, rare_households=(3, 30))
    path, receipt = sample_synthetic(
        sampler,
        tmp_path,
        frame,
        fraction=0.5,
        seed=1,
        probes=fixture_engine_probes(),
    )
    return frame, path, receipt, tmp_path / "export" / "populace_us_2024.h5"


def test_probe_runs_every_stage_on_a_subsample(
    probe_tool, builder, fake_reforms, chain, tmp_path, monkeypatch
) -> None:
    frame, path, receipt, source = chain
    monkeypatch.setattr(builder, "installed_us_engine", lambda: fixture_engine(frame))
    report, log = run_fixture_probe(
        probe_tool, builder, path, tmp_path / "probe", receipt=receipt
    )
    errors = {
        name: record.get("traceback")
        for name, record in report["stages"].items()
        if record["status"] != "completed"
    }
    assert not errors, errors
    assert report["stages"]["load"]["n_batches"] >= probe_tool.MINIMUM_BATCHES
    assert report["stages"]["load"]["design"]["verified_against_receipt"]
    assert report["stages"]["open_scorer"]["n_batches"] >= probe_tool.MINIMUM_BATCHES
    assert report["source_export"] == str(source.resolve())
    for filename in (
        "probe_report.json",
        "passes.jsonl",
        "reform_coverage_smoke.json",
        "reform_validation.json",
        "demographics.json",
        "us_source_coverage.json",
        "us_take_up_participation.json",
        "us_take_up_participation.source.json",
        "qrf_tail_concentration.json",
    ):
        assert (tmp_path / "probe" / filename).exists(), filename
    passes = [
        json.loads(line)
        for line in (tmp_path / "probe" / "passes.jsonl").read_text().splitlines()
    ]
    assert {row["stage"] for row in passes} == {
        "reform_coverage_smoke",
        "reform_validation",
        "demographics",
    }
    assert all(row["batches"] >= probe_tool.MINIMUM_BATCHES for row in passes)

    smoke = {
        row["probe"]: row for row in report["stages"]["reform_coverage_smoke"]["probes"]
    }
    assert set(smoke) == {"rare_keogh", "common_wages", "snap_take_up_2026"}
    for row in smoke.values():
        assert row["decomposition"] == "decomposed", row
        assert row["standard_error"] is not None and math.isfinite(
            row["standard_error"]
        )
        assert row["drawn_effect_households"] > 0
    assert smoke["common_wages"]["engine_weight_scale"] == pytest.approx(1.0, rel=1e-6)
    assert smoke["snap_take_up_2026"]["engine_weight_scale"] == pytest.approx(
        FAKE_UPRATING[2026], rel=1e-6
    )
    assert smoke["snap_take_up_2026"]["measure_entity"] == "spm_unit"
    assert smoke["rare_keogh"]["take_all"] is True
    assert smoke["common_wages"]["take_all"] is False

    checks = verdicts_by_check(report)
    stored = checks[
        (
            "stored_inputs",
            "stored-input gate (#1031): stored columns the installed engine "
            "does not define",
        )
    ]
    assert (stored["verdict"], stored["authority"]) == ("pass", "authoritative")
    premise = checks[("stored_inputs", "written-H5 stored-input premise (#1031)")]
    assert (premise["verdict"], premise["authority"]) == ("not_run", "informational")
    qrf = checks[("qrf_tail_concentration", "QRF tail gate and register (subsample)")]
    assert qrf["authority"] == "informational"
    assert (
        checks[
            ("qrf_tail_concentration", "QRF tail gate and register (source export)")
        ]["authority"]
        == "authoritative"
    )
    assert report["stages"]["reform_validation"]["in_sample_rows"].startswith(
        "simulated on the subsample"
    )
    assert [step["step"] for step in report["not_reproduced"]][0].startswith(
        "written-H5 stored-input premise"
    )
    assert report["target_surface"]["source"].startswith("default")
    assert report["summary"]["stage_status"]["reform_coverage_smoke"] == "completed"


def test_a_census_probe_matches_one_whole_frame_engine(
    probe_tool, builder, fake_reforms, chain, tmp_path, monkeypatch
) -> None:
    """Differential: scored without a receipt (a census of the source export),
    the probe's batched smoke equals the gate on one whole-frame engine, and
    every smoke verdict is authoritative."""
    frame, _, _, source = chain
    monkeypatch.setattr(builder, "installed_us_engine", lambda: fixture_engine(frame))
    report, _ = run_fixture_probe(
        probe_tool,
        builder,
        source,
        tmp_path / "census",
        stages=("reform_coverage_smoke",),
    )
    assert report["stages"]["reform_coverage_smoke"]["status"] == "completed"
    engine_cls = float32_engine(_EngineLog())
    whole = builder.us_reform_coverage_smoke_gate(
        simulate=lambda reform: engine_cls(dataset=frame, reform=reform),
        probes=fixture_engine_probes(),
        period=builder.PERIOD,
    )
    for row in report["stages"]["reform_coverage_smoke"]["probes"]:
        expected = whole.details["results"][row["probe"]]
        assert row["effect"] == pytest.approx(expected["effect"], rel=1e-12)
        assert row["standard_error"] == 0.0
        assert row["authority"] == "authoritative"


def test_a_reference_release_checks_plans_and_compares_effects(
    probe_tool, builder, fake_reforms, chain, tmp_path, monkeypatch
) -> None:
    """With the census run as the reference release: the subsample's plans
    and reform passes equal the reference's, every probe gets a z-score
    against the full-scale effect, and the release manifest's target surface
    is used. A column the engine does not define fails the stored-input gate
    authoritatively."""
    frame, path, receipt, source = chain
    monkeypatch.setattr(
        builder,
        "installed_us_engine",
        lambda: fixture_engine(frame, missing=("keogh_distributions",)),
    )
    diagnostics = {"targets": []}
    census, _ = run_fixture_probe(
        probe_tool,
        builder,
        source,
        tmp_path / "census",
        calibration_diagnostics=diagnostics,
        stages=("reform_coverage_smoke", "reform_validation", "demographics"),
    )
    reference_dir = tmp_path / "release"
    reference_dir.mkdir()
    (reference_dir / "reform_coverage_smoke.json").write_text(
        (tmp_path / "census" / "reform_coverage_smoke.json").read_text()
    )
    (reference_dir / "build_manifest.json").write_text(
        json.dumps({"post_export_scoring": census["post_export_scoring"]})
    )
    (reference_dir / "release_manifest.json").write_text(
        json.dumps(
            {
                "build": {
                    "target_surface_selection": {
                        "mode": "national_state",
                        "dropped_congressional_district_targets": 27_148,
                    }
                }
            }
        )
    )
    (reference_dir / "calibration_diagnostics.json").write_text(json.dumps(diagnostics))
    paths = probe_tool.reference_paths(reference_dir)
    surface = probe_tool.resolve_target_surface(
        None, None, json.loads(paths["release_manifest"].read_text())
    )
    assert surface["mode"] == "national_state"
    assert surface["dropped_congressional_district_targets"] == 27_148
    assert paths["validation"] is None and paths["qrf_tail"] is None
    report, _ = run_fixture_probe(
        probe_tool,
        builder,
        path,
        tmp_path / "probe",
        receipt=receipt,
        reference_smoke=json.loads(paths["smoke"].read_text()),
        reference_build_manifest=json.loads(paths["build_manifest"].read_text()),
        calibration_diagnostics=json.loads(
            paths["calibration_diagnostics"].read_text()
        ),
        target_surface=surface["mode"],
        target_surface_source=surface["source"],
        dropped_congressional_district_targets=surface[
            "dropped_congressional_district_targets"
        ],
    )
    checks = verdicts_by_check(report)
    plans = checks[
        (
            "post_export_scoring",
            "baseline plans and reform passes match the reference release's "
            "build manifest",
        )
    ]
    assert (plans["verdict"], plans["authority"]) == ("pass", "authoritative"), plans
    assert report["reference_plan_comparison"]["consumers"] == [
        "demographics",
        "reform_coverage_smoke",
        "reform_validation",
    ]
    for row in report["stages"]["reform_coverage_smoke"]["probes"]:
        reference = row["reference"]
        assert reference["same_definition"] is True
        assert reference["z"] is not None and math.isfinite(reference["z"])
    stored = checks[
        (
            "stored_inputs",
            "stored-input gate (#1031): stored columns the installed engine "
            "does not define",
        )
    ]
    assert (stored["verdict"], stored["authority"]) == ("fail", "authoritative")
    assert "keogh_distributions" in stored["reason"]
    assert stored in report["summary"]["authoritative_failures"]
    assert report["target_surface"]["mode"] == "national_state"
    coverage = json.loads((tmp_path / "probe" / "us_source_coverage.json").read_text())
    assert coverage["gate"]["passed"]


def test_probe_report_rss_falls_back_without_psutil(probe_tool, monkeypatch) -> None:
    """psutil comes only with policyengine-core; without it the stage clock
    reports the process's lifetime peak instead of failing."""
    import builtins

    real_import = builtins.__import__

    def refuse_psutil(name, *args, **kwargs):
        if name == "psutil":
            raise ImportError("no psutil")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse_psutil)
    sampler = probe_tool.RssSampler()
    assert sampler.source.startswith("ru_maxrss")
    assert sampler.peak() >= sampler.reset() > 0


def test_an_engine_failure_is_an_authoritative_error_and_later_stages_run(
    probe_tool, builder, fake_reforms, chain, tmp_path, monkeypatch
) -> None:
    """A stage that raises is a finding, not the end of the probe: the
    failure does not depend on the sample, so it is authoritative, and the
    stages after it still run."""
    frame, path, receipt, _ = chain
    monkeypatch.setattr(builder, "installed_us_engine", lambda: fixture_engine(frame))
    report, _ = run_fixture_probe(
        probe_tool,
        builder,
        path,
        tmp_path / "probe",
        receipt=receipt,
        fail_on="snap",
        stages=("reform_coverage_smoke", "demographics", "source_coverage"),
    )
    assert report["stages"]["reform_coverage_smoke"]["status"] == "error"
    assert (
        "fixture engine refuses snap"
        in report["stages"]["reform_coverage_smoke"]["error"]
    )
    assert report["stages"]["demographics"]["status"] == "completed"
    assert report["stages"]["source_coverage"]["status"] == "completed"
    errors = [
        row
        for row in report["summary"]["authoritative_failures"]
        if row["stage"] == "reform_coverage_smoke"
    ]
    assert len(errors) == 1 and errors[0]["verdict"] == "error"


def test_a_receipt_that_does_not_match_the_subsample_fails_visibly(
    probe_tool, builder, fake_reforms, chain, tmp_path, monkeypatch
) -> None:
    """A receipt whose strata disagree with the subsample is an authoritative
    failure, and no smoke probe reports a standard error from that design."""
    frame, path, receipt, _ = chain
    monkeypatch.setattr(builder, "installed_us_engine", lambda: fixture_engine(frame))
    tampered = json.loads(json.dumps(receipt))
    label = sorted(tampered["strata"])[0]
    tampered["strata"][label]["drawn_noncertainty_households"] += 1
    report, _ = run_fixture_probe(
        probe_tool,
        builder,
        path,
        tmp_path / "probe",
        receipt=tampered,
        stages=("reform_coverage_smoke",),
    )
    checks = verdicts_by_check(report)
    design = checks[
        (
            "load",
            "the sample design rebuilt from the subsample verifies against its receipt",
        )
    ]
    assert (design["verdict"], design["authority"]) == ("fail", "authoritative")
    assert label in design["reason"]
    for row in report["stages"]["reform_coverage_smoke"]["probes"]:
        assert row["standard_error"] is None
        assert row["authority"] == "informational"


def test_a_source_export_with_other_bytes_settles_nothing(
    probe_tool, builder, fake_reforms, chain, tmp_path, monkeypatch
) -> None:
    frame, path, receipt, source = chain
    monkeypatch.setattr(builder, "installed_us_engine", lambda: fixture_engine(frame))
    other = tmp_path / "other.h5"
    write_table_h5(synthetic_export_frame(10, seed=99), other)
    report, _ = run_fixture_probe(
        probe_tool,
        builder,
        path,
        tmp_path / "probe",
        receipt=receipt,
        source_export=other,
        stages=("take_up_participation",),
    )
    check = verdicts_by_check(report)[
        ("load", "the source export is the receipt's source")
    ]
    assert (check["verdict"], check["authority"]) == ("fail", "authoritative")
    assert "source" not in report["stages"]["take_up_participation"]
