"""The ACS local calibrate stage weights its targets like the national release.

A small checkpoint shaped like the real surface (state SOI amounts and counts,
SNAP, Medicaid enrollment, two ``state_cd`` district blocks, one held out, and
the Census ladder population rows) goes through the real ``do_calibrate``.
The tests check that the shared weights reach ``calibrate`` row-aligned with
the training targets and are not uniform, that every output records them, and
that a resume refuses weights calibrated under another weighting or none.
"""

from __future__ import annotations

import importlib.util
import json
import shlex
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

import microcosm.calibrate as calibrate_package
from microcosm.build.us_runtime import target_loss_weights as lw
from microcosm.calibrate import (
    CalibrationHierarchy,
    HierarchyCategory,
    HierarchyGeography,
    HierarchyNode,
    TargetRegistry,
    TargetSpec,
)
from test_support.microcosm_build.us_batched_target_materialization import (
    _nested_frame,
)
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")
#: The fixture frame's households: (id, state, district), design weights 1-5.
_HOUSEHOLDS = ((1, "06", "0601"), (2, "36", "3601"), (3, "06", "0602"))
_HOUSEHOLDS += ((4, "24", "2401"), (5, "36", "3602"))


def _load_tool_module():
    path = _TEST_PATHS.repository / "tools" / "build_us_acs_local_release.py"
    spec = importlib.util.spec_from_file_location(
        "build_us_acs_local_release_loss_weights", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _hierarchy(name: str, provider: str, level: str, geography_id: str):
    return CalibrationHierarchy(
        provider=HierarchyNode(id=provider, label=provider),
        category=HierarchyCategory(
            id=f"{provider}.fixture", label="Fixture", provider_id=provider
        ),
        geography=HierarchyGeography(id=geography_id, label=geography_id, level=level),
        dimensions=(),
        target=HierarchyNode(id=name, label=name),
    )


def _ledger(name, family, value, unit, state, **metadata) -> TargetSpec:
    """A ledger-compiled state row with the metadata the mapping reads."""

    return TargetSpec(
        name=name,
        entity="household",
        measure=name,
        value=value,
        period=2024,
        family=family,
        source="fixture ledger feed",
        metadata={
            "measure_mode": "sum" if unit == "usd" else "indicator_sum",
            "source_measure_id": name.split(".")[-1],
            "ledger_measure_unit": unit,
            "ledger_geography_level": "state",
            "state_fips": state,
            "ledger_fact_label": f"{name} label",
            **metadata,
        },
        hierarchy=_hierarchy(name, family, "state", f"0400000US{state}"),
    )


def _district(name, value, unit, district, parent) -> TargetSpec:
    """A ``state_cd`` district row: compiler labels and rebase stamps."""

    spec = _ledger(
        name,
        "irs_soi",
        value,
        unit,
        district[:2],
        ledger_geography_level="congressional_district",
        congressional_district_geoid=district,
        ledger_geography_id=f"5001900US{district}",
        ledger_layout_groupby_value_label=f"congressional district {district}",
        ledger_fact_label=f"Congressional District {district} {name}",
        state_cd_parent_target_name=parent,
        state_cd_parent_basis="historic_table_2",
        state_cd_cd_file_value=repr(value * 0.97),
        state_cd_rebase_factor="1.03",
    )
    return TargetSpec(
        **{
            **spec.__dict__,
            "hierarchy": _hierarchy(
                name, "irs_soi", "congressional_district", f"5001900US{district}"
            ),
        }
    )


def _surface(module):
    """Specs, roles and a CSR matrix shaped like the release's checkpoint.

    Each row's measure is the household's design-weighted share of a
    geography, so every value sits within a factor of two of its design
    estimate and the solve is well posed.
    """

    states = np.asarray([state for _id, state, _cd in _HOUSEHOLDS])
    districts = np.asarray([cd for _id, _state, cd in _HOUSEHOLDS])
    design = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])

    def in_state(state):
        return (states == state).astype(float)

    def in_district(district):
        return (districts == district).astype(float)

    ca_agi = "irs_soi.ca.adjusted_gross_income_amount"
    ny_returns = "irs_soi.ny.return_count"
    rows = [
        (_ledger(ca_agi, "irs_soi", 6.0e5, "usd", "06"), in_state("06") * 1.5e5),
        (
            _ledger("irs_soi.ca.return_count", "irs_soi", 8.0, "count", "06"),
            in_state("06") * 2.0,
        ),
        (
            _ledger(
                "irs_soi.ny.adjusted_gross_income_amount", "irs_soi", 9.0e5, "usd", "36"
            ),
            in_state("36") * 1.0e5,
        ),
        (_ledger(ny_returns, "irs_soi", 7.0, "count", "36"), in_state("36")),
        (
            _ledger("usda_snap.ca.total_benefits", "usda_snap", 3.0e3, "usd", "06"),
            in_state("06") * 700.0,
        ),
        (
            _ledger(
                "usda_snap.ny.average_monthly_households",
                "usda_snap",
                2.0,
                "count",
                "36",
            ),
            in_state("36") * 0.3,
        ),
        (
            _ledger(
                "cms_medicaid.ca.total_medicaid_enrollment",
                "cms_medicaid",
                3.0,
                "count",
                "06",
            ),
            in_state("06") * 0.8,
        ),
        (
            _district(
                "irs_soi.cd0601.adjusted_gross_income_amount",
                1.4e5,
                "usd",
                "0601",
                ca_agi,
            ),
            in_district("0601") * 1.5e5,
        ),
        (
            _district(
                "irs_soi.cd0602.adjusted_gross_income_amount",
                4.6e5,
                "usd",
                "0602",
                ca_agi,
            ),
            in_district("0602") * 1.5e5,
        ),
        (
            _district("irs_soi.cd3601.return_count", 2.5, "count", "3601", ny_returns),
            in_district("3601"),
        ),
        (
            _district("irs_soi.cd3602.return_count", 4.5, "count", "3602", ny_returns),
            in_district("3602"),
        ),
    ]
    admin_specs = [spec for spec, _row in rows]
    population_names = [
        "pop_state_06", "pop_state_24", "pop_state_36",
        "pop_cd_0601", "pop_cd_0602", "pop_cd_2401", "pop_cd_3601", "pop_cd_3602",
    ]  # fmt: skip
    population_rows = [
        in_state(name[-2:]) * 2.0
        if name.startswith("pop_state_")
        else in_district(name[-4:]) * 2.0
        for name in population_names
    ]
    population_values = [float(row @ design) * 1.1 for row in population_rows]
    population_specs = module.population_target_specs(
        population_names, population_values
    )
    specs = (*admin_specs, *population_specs)
    matrix = sparse.csr_array(
        np.vstack([row for _spec, row in rows] + population_rows).astype(np.float32)
    )
    records = [module.cd_surface.target_record(spec) for spec in specs]
    for record in records:
        record["role"] = "train"
        record["holdout_unit"] = None
        if record["name"].startswith("irs_soi.cd36"):
            # The NY return-count block is held out, as a hash draw would.
            record["role"] = "holdout"
            record["holdout_unit"] = "36|return"
            record["cd_population"] = 2.0
            record["state_population"] = 4.0
    return TargetRegistry(specs, country="us"), records, matrix, design


def _args(tmp_path, **overrides):
    values = dict(
        checkpoint_dir=tmp_path,
        out_h5=None,
        resume=False,
        epochs=6,
        epoch_batch=3,
        max_weight_ratio=5.0,
        target_loss_cap=1.0,
        l2_lambda=0.0,
        l2_basis="record",
        mass_parametrization="projection",
        seed=0,
        families="snap,medicaid,soi",
        geographies="state,cd",
        target_family_loss_multipliers={},
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.fixture
def calibrate_run(monkeypatch, tmp_path):
    """Run ``do_calibrate`` on the fixture; record every ``calibrate`` call."""

    module = _load_tool_module()
    registry, records, matrix, design = _surface(module)
    frame = _nested_frame()
    identity = {"staging_sha256": "s", "target_roles_sha256": "r"}
    monkeypatch.setattr(module, "_verify_run_identity", lambda args: identity)
    monkeypatch.setattr(
        module,
        "load_checkpoint_surface",
        lambda *_a, **_k: (frame, design, registry, records, matrix),
    )
    calls: list[dict] = []
    real_calibrate = calibrate_package.calibrate

    def spy(frame, targets, **kwargs):
        calls.append(
            {
                "names": [target.name for target in targets],
                "target_loss_weights": kwargs.get("target_loss_weights"),
            }
        )
        return real_calibrate(frame, targets, **kwargs)

    monkeypatch.setattr(calibrate_package, "calibrate", spy)

    def run(**overrides):
        calls.clear()
        module.do_calibrate(_args(tmp_path, **overrides))
        return calls

    return SimpleNamespace(
        module=module, registry=registry, records=records, run=run, path=tmp_path
    )


def _train_specs(run):
    held = {r["name"] for r in run.records if r["role"] == "holdout"}
    return [spec for spec in run.registry.specs if spec.name not in held]


def test_the_shared_weights_reach_calibrate_on_every_batch(calibrate_run) -> None:
    calls = calibrate_run.run()
    train = _train_specs(calibrate_run)
    expected = lw.us_acs_local_target_loss_weights(train)
    assert len(calls) == 2  # 6 epochs in batches of 3
    for call in calls:
        assert call["names"] == [spec.name for spec in train]
        assert not any(name.startswith("irs_soi.cd36") for name in call["names"])
        assert np.array_equal(call["target_loss_weights"], expected)
    # Mixed rows get mixed weights: not the equal weights of before.
    assert len(np.unique(expected)) > 5
    assert expected.max() / expected.min() > 2.0
    # The surface's bases split the weight evenly; the CA district AGI block
    # shares one budget.
    bases = [lw.acs_local_target_value_basis(spec) for spec in train]
    count = sum(w for w, b in zip(expected, bases, strict=True) if b == "count")
    assert count == pytest.approx(len(train) / 2, rel=1e-12)


def test_every_calibrate_output_records_the_weights(calibrate_run) -> None:
    calibrate_run.run()
    train = _train_specs(calibrate_run)
    weights = lw.us_acs_local_target_loss_weights(train)
    digest = lw.target_loss_weights_sha256(
        [lw.target_row_name(spec) for spec in train], weights
    )
    summary = json.loads((calibrate_run.path / "calibration_summary.json").read_text())
    stamp = {
        "weighting": lw.US_FISCAL_TARGET_LOSS_WEIGHTING,
        "row_mapping": "us_acs_local.v1",
        "family_multipliers": {},
        "n_targets": len(train),
        "weights_sha256": digest,
    }
    assert summary["solver_settings"]["target_loss"] == stamp
    recorded = summary["target_loss"]
    assert {key: recorded[key] for key in stamp} == stamp
    distribution = recorded["distribution"]
    assert distribution["row_mapping"] == "us_acs_local.v1"
    assert distribution["by_basis"]["count"]["loss_share"] == pytest.approx(0.5)
    # Trained district groups: the CA AGI block (the NY block is held out)
    # and one pop_cd group per state, MD's a single at-large district.
    assert distribution["concept_groups"] == {
        "n_groups": 14,
        "n_district_rows": 7,
        "n_district_groups": 4,
        "n_singleton_district_groups": 1,
        "max_district_group_size": 2,
    }
    families = {cell["family"] for cell in distribution["by_family_level_basis"]}
    assert families == {"irs_soi", "usda_snap", "cms_medicaid", "census_population"}

    saved = np.load(calibrate_run.path / "weights_latest.npz")
    assert json.loads(str(saved["solver_settings"]))["target_loss"] == stamp

    assert summary["calibration_diagnostics"]["status"] == "available"
    diagnostics = json.loads(
        (calibrate_run.path / "calibration_diagnostics.json").read_text()
    )
    build = diagnostics["build"]
    assert build["target_loss_weighting"] == lw.US_FISCAL_TARGET_LOSS_WEIGHTING
    assert build["target_loss_family_multipliers"] is None
    assert build["target_loss_cap"] == 1.0
    assert build["target_loss_row_mapping"] == "us_acs_local.v1"
    assert build["target_loss_weights_sha256"] == digest
    assert diagnostics["options"]["target_loss_weights"]["kind"] == "provided"
    by_name = {
        target["target_name"]: target["target_loss_weight"]
        for target in diagnostics["targets"]
    }
    assert [by_name[spec.name] for spec in train] == pytest.approx(
        weights.tolist(), rel=1e-15
    )


def test_a_family_multiplier_reaches_calibrate_and_the_stamp(calibrate_run) -> None:
    calls = calibrate_run.run(target_family_loss_multipliers={"usda_snap": 4.0})
    train = _train_specs(calibrate_run)
    base = lw.us_acs_local_target_loss_weights(train)
    boosted = calls[0]["target_loss_weights"]
    snap = np.asarray([spec.family == "usda_snap" for spec in train])
    ratio = boosted / base
    assert ratio[snap] == pytest.approx(4.0 * ratio[~snap][0], rel=1e-12)
    summary = json.loads((calibrate_run.path / "calibration_summary.json").read_text())
    assert summary["target_loss"]["family_multipliers"] == {"usda_snap": 4.0}
    with pytest.raises(SystemExit, match="'ssa' matches no compiled target"):
        calibrate_run.run(target_family_loss_multipliers={"ssa": 2.0})


def test_resume_refuses_weights_from_another_loss_weighting(calibrate_run) -> None:
    calibrate_run.run()
    npz = calibrate_run.path / "weights_latest.npz"
    saved = dict(np.load(npz))
    settings = json.loads(str(saved["solver_settings"]))
    # The same run resumes (here: nothing left to do).
    calibrate_run.run(resume=True)
    # A run from before the weights: its settings carry no target_loss.
    unweighted = {key: value for key, value in settings.items() if key != "target_loss"}
    for stale in (
        unweighted,
        {**settings, "target_loss": {**settings["target_loss"], "weights_sha256": "0"}},
    ):
        np.savez(
            npz,
            **{**saved, "solver_settings": np.str_(json.dumps(stale, sort_keys=True))},
        )
        with pytest.raises(SystemExit, match="different solver settings"):
            calibrate_run.run(resume=True)
    # Other multipliers are other weights.
    np.savez(npz, **saved)
    with pytest.raises(SystemExit, match="different solver settings"):
        calibrate_run.run(
            resume=True, target_family_loss_multipliers={"cms_medicaid": 2.0}
        )


def test_the_cli_parses_family_multipliers_like_the_national_release(
    tmp_path, capsys
) -> None:
    module = _load_tool_module()
    base = [
        "--stage", "calibrate",
        "--staging-h5", str(tmp_path / "s.h5"),
        "--checkpoint-dir", str(tmp_path),
        "--out-h5", str(tmp_path / "out.h5"),
    ]  # fmt: skip
    assert module._parse_args(base).target_family_loss_multipliers == {}
    args = module._parse_args(
        [
            *base,
            "--target-family-loss-multiplier",
            "usda_snap=8",
            "--target-family-loss-multiplier",
            "cms_medicaid=0.5",
        ]
    )
    assert args.target_family_loss_multipliers == {
        "usda_snap": 8.0,
        "cms_medicaid": 0.5,
    }
    for bad in (["usda_snap"], ["usda_snap=0"], ["a=2", "a=3"]):
        argv = [*base]
        for entry in bad:
            argv += ["--target-family-loss-multiplier", entry]
        with pytest.raises(SystemExit) as raised:
            module._parse_args(argv)
        assert raised.value.code == 2
    assert "--target-family-loss-multiplier" in capsys.readouterr().err


def test_a_misaligned_target_set_is_refused(calibrate_run, monkeypatch) -> None:
    module = calibrate_run.module
    real = module.cd_surface.calibration_target_set

    def reversed_set(*args, **kwargs):
        from microcosm.calibrate.target import TargetSet

        return TargetSet(list(real(*args, **kwargs))[::-1])

    monkeypatch.setattr(module.cd_surface, "calibration_target_set", reversed_set)
    with pytest.raises(SystemExit, match="not row-aligned with the training specs"):
        calibrate_run.run()


def test_the_refresh_recipe_rebuilds_the_recorded_multipliers() -> None:
    module = _load_tool_module()
    plain = shlex.split(module.release_refresh_recipe("state"))
    assert "--target-family-loss-multiplier" not in plain
    assert module._parse_args(plain[3:]).target_family_loss_multipliers == {}
    recipe = shlex.split(
        module.release_refresh_recipe(
            "state",
            target_family_loss_multipliers={"usda_snap": 4.0, "cms_medicaid": 0.5},
        )
    )
    assert module._parse_args(recipe[3:]).target_family_loss_multipliers == {
        "cms_medicaid": 0.5,
        "usda_snap": 4.0,
    }


def _release_tool_tests():
    """The package-stage helpers of ``test_us_acs_local_release_tool.py``."""

    spec = importlib.util.spec_from_file_location(
        "acs_local_release_tool_tests",
        _TEST_PATHS.tests / "engine_free" / "us" / "test_us_acs_local_release_tool.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_the_package_manifest_records_the_loss_weights(tmp_path, monkeypatch) -> None:
    from microcosm.data import stored_inputs

    helpers = _release_tool_tests()
    # The stored-input contract stand-in that file's autouse fixture installs.
    monkeypatch.setattr(
        stored_inputs, "installed_us_engine", lambda: helpers._PACKAGE_ENGINE
    )
    monkeypatch.setattr(
        stored_inputs,
        "require_h5_stored_inputs",
        lambda path, *, engine: {
            "register_sha256": stored_inputs.register_sha256(),
            "registered_non_variables": [],
        },
    )
    module = helpers._load_tool_module()
    args = helpers._package_evidence_args(
        module,
        tmp_path,
        monkeypatch,
        hours_report={"passed": True, "failures": [], "detail": {}},
    )
    summary_path = args.checkpoint_dir / "calibration_summary.json"
    summary = json.loads(summary_path.read_text())
    target_loss = {
        "weighting": lw.US_FISCAL_TARGET_LOSS_WEIGHTING,
        "row_mapping": "us_acs_local.v1",
        "family_multipliers": {"usda_snap": 4.0},
        "n_targets": 3,
        "weights_sha256": "0" * 64,
    }
    summary["target_loss"] = target_loss
    summary_path.write_text(json.dumps(summary))

    result = module.do_package(args)

    manifest = json.loads(
        (Path(result["release_dir"]) / "build_manifest.json").read_text()
    )
    assert manifest["calibration"]["target_loss"] == target_loss
    recipe = shlex.split(manifest["refresh_recipe"]["release"])
    assert recipe[recipe.index("--target-family-loss-multiplier") + 1] == (
        "usda_snap=4.0"
    )
