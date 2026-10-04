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
- the target surface resolves from the reference release manifest;
- the report names the commit the probe was loaded from, not the worktree's
  HEAD when the report is written.
"""

# ruff: noqa: F403, F405
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
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
    assert report["stages"]["design"]["n_batches"] >= probe_tool.MINIMUM_BATCHES
    assert report["stages"]["design"]["verified_against_receipt"]
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
    # The fake engine's SNAP effect is the same for every household (one SPM
    # unit each), so the effect has no variance at all: exactly 0, on no
    # effective household, and no authority. Dividing the engine-weighted
    # effect by the design weight would leave float32 rounding of the
    # uprated weights behind as a spurious variance spread over every
    # household (the review's finding on the first version).
    snap = smoke["snap_take_up_2026"]
    assert snap["standard_error"] == 0.0
    assert snap["effective_variance_households"] == 0.0
    assert snap["authority"] == "informational"
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
        census=True,
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
        census=True,
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
        if row["standard_error"]:
            assert reference["z"] is not None and math.isfinite(reference["z"])
        else:
            # The fake SNAP effect is the same for every household, so the
            # ratio estimator reproduces the census total exactly (each
            # stratum's weight total is conserved) and there is no variance
            # to scale a difference by.
            assert row["probe"] == "snap_take_up_2026"
            assert reference["z"] is None
            assert abs(reference["relative_difference"]) < 1e-6
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
            "design",
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
        ("design", "the source export is the receipt's source")
    ]
    assert (check["verdict"], check["authority"]) == ("fail", "authoritative")
    assert "source" not in report["stages"]["take_up_participation"]


def test_a_subsample_without_its_receipt_is_refused(
    probe_tool, builder, fake_reforms, chain, tmp_path
) -> None:
    """Scored without a receipt, a subsample would pass for the full export
    and every scale-dependent verdict would be labelled authoritative."""
    _, path, _, _ = chain
    with pytest.raises(ValueError, match="No sample receipt"):
        run_fixture_probe(probe_tool, builder, path, tmp_path / "probe")


def test_a_full_fraction_subsample_is_a_census(
    sampler, probe_tool, builder, fake_reforms, tmp_path, monkeypatch
) -> None:
    """Every stratum taken whole (fraction 1) is a census by design: its
    smoke verdicts are authoritative without a variance."""
    frame = synthetic_export_frame(30, seed=8)
    monkeypatch.setattr(builder, "installed_us_engine", lambda: fixture_engine(frame))
    path, receipt = sample_synthetic(
        sampler, tmp_path, frame, fraction=1.0, seed=0, probes=fixture_engine_probes()
    )
    report, _ = run_fixture_probe(
        probe_tool,
        builder,
        path,
        tmp_path / "probe",
        receipt=receipt,
        stages=("reform_coverage_smoke",),
    )
    assert report["stages"]["design"]["census"] is True
    rows = report["stages"]["reform_coverage_smoke"]["probes"]
    assert rows and all(row["authority"] == "authoritative" for row in rows)


def test_an_unreceipted_stratum_is_reported_not_fatal(
    probe_tool, builder, fake_reforms, chain, tmp_path, monkeypatch
) -> None:
    """A household whose rebuilt stratum label the receipt does not know
    fails the design check visibly; the probe still runs its stages."""
    frame, path, receipt, _ = chain
    monkeypatch.setattr(builder, "installed_us_engine", lambda: fixture_engine(frame))
    tampered = json.loads(json.dumps(receipt))
    label = sorted(tampered["strata"])[0]
    tampered["strata"][label + "|renamed"] = tampered["strata"].pop(label)
    report, _ = run_fixture_probe(
        probe_tool,
        builder,
        path,
        tmp_path / "probe",
        receipt=tampered,
        stages=("reform_coverage_smoke", "demographics"),
    )
    assert report["stages"]["load"]["status"] == "completed"
    assert report["stages"]["design"]["status"] == "completed"
    assert report["stages"]["demographics"]["status"] == "completed"
    design = verdicts_by_check(report)[
        (
            "design",
            "the sample design rebuilt from the subsample verifies against its receipt",
        )
    ]
    assert design["verdict"] == "fail"
    assert "unreceipted strata" in design["reason"]
    assert design in report["summary"]["probe_failures"]
    assert design not in report["summary"]["authoritative_failures"]
    for row in report["stages"]["reform_coverage_smoke"]["probes"]:
        assert row["standard_error"] is None


def test_an_analysis_error_keeps_every_gate_verdict(
    probe_tool, builder, fake_reforms, chain, tmp_path, monkeypatch
) -> None:
    """A failure in the probe's own decomposition is not a release failure:
    every probe keeps the gate's pass/fail, without a standard error."""
    frame, path, receipt, _ = chain
    monkeypatch.setattr(builder, "installed_us_engine", lambda: fixture_engine(frame))

    def broken(*args, **kwargs):
        raise RuntimeError("decomposition exploded")

    monkeypatch.setattr(probe_tool, "household_effects", broken)
    report, _ = run_fixture_probe(
        probe_tool,
        builder,
        path,
        tmp_path / "probe",
        receipt=receipt,
        stages=("reform_coverage_smoke",),
    )
    assert report["stages"]["reform_coverage_smoke"]["status"] == "completed"
    rows = report["stages"]["reform_coverage_smoke"]["probes"]
    assert len(rows) == len(fixture_engine_probes())
    for row in rows:
        assert "decomposition exploded" in row["decomposition"]
        assert row["standard_error"] is None
        assert row["authority"] == "informational"
    smoke_checks = [
        v for v in report["verdicts"] if v["stage"] == "reform_coverage_smoke"
    ]
    assert len(smoke_checks) == len(rows)
    assert not report["summary"]["authoritative_failures"]


def test_an_older_reference_manifest_is_compared_not_fatal(probe_tool) -> None:
    record = {
        "baseline_plan": {"keys": [["income_tax", 2024, None]], "period_order": [2024]},
        "baseline_passes": 1,
        "reform_passes": 3,
        "reform_systems": 3,
    }
    problems = probe_tool.compare_post_export_plans(
        {"reform_coverage_smoke": record, "demographics": record},
        {
            "post_export_scoring": {
                "consumers": {
                    "reform_coverage_smoke": {
                        "baseline_plan": record["baseline_plan"],
                        "reform_passes": 3,
                    }
                }
            }
        },
    )
    assert problems == [
        "reform_coverage_smoke: the reference recorded no reform_systems",
        "demographics: the reference release recorded no plan",
    ]


def test_census_with_a_receipt_is_refused(
    probe_tool, builder, fake_reforms, chain, tmp_path
) -> None:
    _, path, receipt, _ = chain
    with pytest.raises(ValueError, match="contradicts the sample receipt"):
        run_fixture_probe(
            probe_tool, builder, path, tmp_path / "probe", receipt=receipt, census=True
        )


def test_a_release_loader_failure_is_the_releases_failure(
    probe_tool, builder, fake_reforms, chain, tmp_path, monkeypatch
) -> None:
    """The release's own loader failing on the export is a failure the
    release would hit too; the receipt and design checks are the probe's."""
    frame, path, receipt, _ = chain
    monkeypatch.setattr(builder, "installed_us_engine", lambda: fixture_engine(frame))

    def unreadable(path, *, expected_sha256=None):
        raise OSError("the loader cannot read this export")

    report = probe_tool.probe_export(
        path,
        tmp_path / "probe",
        sample_receipt=receipt,
        probes=fixture_engine_probes(),
        builder=builder,
        load_frame=unreadable,
        stages=("stored_inputs",),
    )
    assert report["stages"]["identify"]["status"] == "completed"
    assert report["stages"]["load"]["status"] == "error"
    assert "design" not in report["stages"]
    (error,) = report["summary"]["authoritative_failures"]
    assert error["stage"] == "load" and error["release_consequence"] == "raises"
    assert not report["summary"]["probe_failures"]


def test_a_reference_row_missing_fields_costs_only_its_comparison(
    probe_tool, builder, fake_reforms, chain, tmp_path, monkeypatch
) -> None:
    frame, path, receipt, _ = chain
    monkeypatch.setattr(builder, "installed_us_engine", lambda: fixture_engine(frame))
    reference = {
        "reform_coverage_smoke": {
            "details": {"results": {probe.id: {} for probe in fixture_engine_probes()}}
        }
    }
    report, _ = run_fixture_probe(
        probe_tool,
        builder,
        path,
        tmp_path / "probe",
        receipt=receipt,
        reference_smoke=reference,
        stages=("reform_coverage_smoke",),
    )
    rows = report["stages"]["reform_coverage_smoke"]["probes"]
    assert len(rows) == len(fixture_engine_probes())
    for row in rows:
        assert "error" in row["reference"]
        assert row["decomposition"] == "decomposed"


@pytest.mark.parametrize("fault", ["row_map", "reform_count", "missing_result"])
def test_a_fault_after_the_gate_keeps_one_verdict_per_probe(
    probe_tool, builder, fake_reforms, chain, tmp_path, monkeypatch, fault
) -> None:
    """Whatever breaks in the probe's own analysis after the smoke gate has
    run, each probe gets exactly one verdict and none is a release failure:
    a failing row map or a reform count that does not match the probes
    costs every probe its standard error; a probe the gate recorded no
    result for is one probe-integrity error."""
    frame, path, receipt, _ = chain
    monkeypatch.setattr(builder, "installed_us_engine", lambda: fixture_engine(frame))
    probes = fixture_engine_probes()
    if fault == "row_map":

        def broken(*args, **kwargs):
            raise RuntimeError("row map exploded")

        monkeypatch.setattr(probe_tool, "HouseholdRowMap", broken)
    elif fault == "reform_count":
        original = probe_tool.SimulateRecorder.__call__

        def drop_last_reform(self, reform):
            simulation = original(self, reform)
            if reform is not None and len(self.reforms) == len(probes):
                self.reforms.pop()
            return simulation

        monkeypatch.setattr(probe_tool.SimulateRecorder, "__call__", drop_last_reform)
    else:
        real_gate = builder.us_reform_coverage_smoke_gate

        def gate_without_last(**kwargs):
            result = real_gate(**kwargs)
            result.details["results"].pop(probes[-1].id, None)
            return result

        monkeypatch.setattr(builder, "us_reform_coverage_smoke_gate", gate_without_last)
    report, _ = run_fixture_probe(
        probe_tool,
        builder,
        path,
        tmp_path / "probe",
        receipt=receipt,
        probes=probes,
        stages=("reform_coverage_smoke",),
    )
    assert report["stages"]["reform_coverage_smoke"]["status"] == "completed"
    verdicts = [v for v in report["verdicts"] if v["stage"] == "reform_coverage_smoke"]
    assert sorted(v["check"] for v in verdicts) == sorted(
        f"probe {probe.id}" for probe in probes
    )
    assert not report["summary"]["authoritative_failures"]
    rows = report["stages"]["reform_coverage_smoke"]["probes"]
    if fault == "missing_result":
        assert len(rows) == len(probes) - 1
        (missing,) = [v for v in verdicts if v["verdict"] == "error"]
        assert missing["check"] == f"probe {probes[-1].id}"
        assert missing in report["summary"]["probe_failures"]
    else:
        assert len(rows) == len(probes)
        for row in rows:
            assert row["standard_error"] is None
            assert row["decomposition"].startswith("analysis not run")
            assert row["authority"] == "informational"


def test_a_receipt_without_probe_records_costs_only_take_all(
    probe_tool, builder, fake_reforms, chain, tmp_path, monkeypatch
) -> None:
    frame, path, receipt, _ = chain
    monkeypatch.setattr(builder, "installed_us_engine", lambda: fixture_engine(frame))
    older = json.loads(json.dumps(receipt))
    del older["probes"]
    report, _ = run_fixture_probe(
        probe_tool,
        builder,
        path,
        tmp_path / "probe",
        receipt=older,
        stages=("reform_coverage_smoke",),
    )
    rows = report["stages"]["reform_coverage_smoke"]["probes"]
    assert len(rows) == len(fixture_engine_probes())
    assert all(row["take_all"] is False for row in rows)
    assert all(row["decomposition"] == "decomposed" for row in rows)


def test_cancelling_non_carrier_effects_deny_take_all_exactness(
    sampler, probe_tool, builder, fake_reforms, chain, tmp_path, monkeypatch
) -> None:
    """A take-all probe none of whose drawn households carries an effect is
    exact only if no other household carries one either. Here two certainty
    households outside its carriers carry effects that cancel in sum: a
    summed test would call the probe exact (authoritative, rule 3); counting
    the households sends it to the variance rules, which find nothing to
    estimate from (informational)."""
    frame, _, _, _ = chain
    monkeypatch.setattr(builder, "installed_us_engine", lambda: fixture_engine(frame))
    # Size certainty keeps the households that dominate the wage probe's
    # mass: certainty households that do not carry the rare input.
    path, receipt = sample_synthetic(
        sampler,
        tmp_path / "sized",
        frame,
        fraction=0.5,
        seed=1,
        probes=fixture_engine_probes(),
        size_certainty_multiplier=1.0,
    )
    probe, _ = fixture_export_probe(
        probe_tool,
        builder,
        path,
        tmp_path / "base",
        receipt=receipt,
        stages=("stored_inputs",),
    )
    probe.run()
    design = probe.design
    rare = next(p for p in fixture_engine_probes() if p.id == "rare_keogh")
    carriers = probe.carriers_by_probe["rare_keogh"]
    certain_others = [
        h for h in design.household_ids[design.certainty] if h not in set(carriers)
    ]
    assert len(certain_others) >= 2, (
        "the fixture needs two non-carrier certainty households"
    )
    others = certain_others[:2]
    ids = np.asarray([*carriers, *others], dtype=np.int64)
    unweighted = pd.Series([0.0] * len(carriers) + [5.0, -5.0], index=ids)
    weight_of = pd.Series(design.adjusted_weights, index=design.household_ids)
    weighted = unweighted * weight_of.reindex(ids).to_numpy()
    weighted.iloc[-1] = -weighted.iloc[-2]  # the two cancel exactly in sum
    effects = probe_tool.HouseholdEffects(
        weighted,
        unweighted=unweighted,
        entity="tax_unit",
        weight_scale=1.0,
        magnitude=1.0,
    )
    monkeypatch.setattr(
        probe_tool,
        "household_effects",
        lambda *a, **k: (effects, "tax_unit", "decomposed"),
    )
    probe.design_problems = []
    analysis = probe._smoke_analysis(
        rare,
        {"period": 2024, "effect": 0.0},
        {("income_tax", 2024, None): None},
        {("income_tax", 2024, None): None},
        None,
        carriers,
    )
    assert analysis["drawn_effect_households"] == 0  # rule 3's first condition holds
    assert analysis["noncarrier_effect"] == 0.0  # a summed test would pass
    assert analysis["noncarrier_effect_households"] == 2
    kwargs = dict(
        signed_magnitude=0.0,
        floor=1.0,
        standard_error=analysis["standard_error"],
        take_all=True,
        drawn_effect_households=analysis["drawn_effect_households"],
        effective_households=analysis["effective_variance_households"],
        census=False,
    )
    authority, _ = probe_tool.classify_probe(
        **kwargs, noncarrier_effect_households=analysis["noncarrier_effect_households"]
    )
    assert authority == "informational"
    # The summed rule's verdict, for contrast: exact, hence authoritative.
    authority, _ = probe_tool.classify_probe(**kwargs, noncarrier_effect_households=0)
    assert authority == "authoritative"


def test_a_malformed_reference_smoke_file_costs_only_the_comparison(
    probe_tool, builder, fake_reforms, chain, tmp_path, monkeypatch
) -> None:
    frame, path, receipt, _ = chain
    monkeypatch.setattr(builder, "installed_us_engine", lambda: fixture_engine(frame))
    report, _ = run_fixture_probe(
        probe_tool,
        builder,
        path,
        tmp_path / "probe",
        receipt=receipt,
        reference_smoke={
            "reform_coverage_smoke": {"details": {"results": ["not", "a", "map"]}}
        },
        stages=("reform_coverage_smoke",),
    )
    smoke = report["stages"]["reform_coverage_smoke"]
    assert smoke["status"] == "completed"
    assert "malformed" in smoke["reference_error"]
    assert len(smoke["probes"]) == len(fixture_engine_probes())
    assert all("reference" not in row for row in smoke["probes"])


def test_the_report_names_the_commit_the_probe_loaded_from(
    probe_tool, builder, chain, tmp_path, monkeypatch
) -> None:
    """The Route A probe launched 2026-10-02 named a commit made ten minutes
    after it started: it read HEAD when it wrote its report header. The
    report now keeps the state read when the probe loaded, the sha256 of the
    probe and of the sibling tools it runs, and the state at each write,
    flagging a move (modules imported in between would then come from either
    state). The report is written here before ``run``, as no stage runs."""
    _, path, receipt, _ = chain
    with monkeypatch.context() as unmoved:
        pin_git_state_at_load(unmoved, probe_tool)
        probe, _ = fixture_export_probe(
            probe_tool, builder, path, tmp_path / "probe", receipt=receipt
        )
    assert probe.report["tool_source"]["moved_since_load"] is False

    move_head_after_load(monkeypatch)
    probe._write_report()
    written = tmp_path / "probe" / probe_tool.REPORT_FILENAME
    source = json.loads(written.read_text())["tool_source"]
    assert {key: source[key] for key in LOAD_STATE_FIELDS} == probe_tool._TOOL_SOURCE
    assert source["commit"] != LATER_HEAD
    assert source["commit_at_write"] == LATER_HEAD
    # Outside a git checkout there is no load state to compare with.
    expected = True if source["commit"] is not None else None
    assert source["moved_since_load"] is expected

    def sha256(path) -> str:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    assert source["sha256"] == sha256(probe_tool.__file__)
    # The fixtures load both siblings themselves, so the probe hashes their
    # files when it writes.
    assert source["tools"] == {
        "tools/build_us_fiscal_refresh_release.py": {
            "sha256": sha256(builder.__file__),
            "file": builder.__file__,
            "hashed": "at write",
        },
        "tools/sample_us_export_households.py": {
            "sha256": sha256(probe.sampler.__file__),
            "file": probe.sampler.__file__,
            "hashed": "at write",
        },
    }
