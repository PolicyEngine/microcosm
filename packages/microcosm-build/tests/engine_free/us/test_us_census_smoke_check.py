"""``tools/check_us_census_smoke.py``: accepting a separate census smoke.

The check accepts a separate full-population reform-coverage smoke for a
release whose build skipped its own, and only then writes the attachments a
tag-only publish carries. These tests build one faithful set of inputs (a
throwaway repository holding the release tool's source, launch records taken
by the tool itself, a release directory and the probe's output), confirm the
check accepts it, and confirm it refuses each departure without writing
anything. Differential tests hold each mirror in the check (the probe
manifest parse, the gate's verdict, the baseline plan, the pass log, batching,
the runtime and inventory scripts) to the code it mirrors.
"""

# The shared fixtures (``builder``, ``fake_reforms``, ``probe_tool``) are
# imported by name, so pytest finds them; the tests take them as parameters.
# ruff: noqa: F811

from __future__ import annotations

import ast
import copy
import dataclasses
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from microcosm.build.us_runtime import reform_coverage_smoke as smoke_module
from microcosm.build.us_runtime.release_input_coverage import (
    ReformCoverageProbe,
    load_release_input_coverage_manifest,
)
from test_support.git_repositories import NO_BACKGROUND_MAINTENANCE
from test_support.microcosm_build.us_export_subsample import probe_tool  # noqa: F401
from test_support.microcosm_build.us_post_export_scoring import (  # noqa: F401
    _EngineLog,
    _nested_frame,
    _scorer,
    builder,
    fake_reforms,
)
from test_support.paths import paths_for

_ROOT = paths_for("microcosm-build").repository
_TOOL = _ROOT / "tools" / "check_us_census_smoke.py"

#: Launch boundaries in the past, so ``main`` (which reads the clock) accepts.
_T0 = datetime(2026, 10, 1, 10, 30, tzinfo=UTC)
_TIMES = {
    "build-start": _T0,
    "manifests": _T0 + timedelta(hours=1),
    "build-end": _T0 + timedelta(hours=2),
    "smoke-start": _T0 + timedelta(hours=3, microseconds=400_000),
    "report": _T0 + timedelta(hours=3),
    "smoke-end": _T0 + timedelta(hours=5),
    "check": _T0 + timedelta(hours=6),
}
_DATASET_KEY = "microcosm_us_2024"
_DATASET_FILE = "microcosm_us_2024.h5"
_DATASET_BYTES = b"a census-scored release H5, in miniature\n" * 64
_REPO_ID = "policyengine/microcosm-us-test"
_N_HOUSEHOLDS = 7_001
_BATCH = 2_000
_THREADS = "3"
#: The release tool's thread-pool variables, for parametrizing (a test holds
#: them to the check's ``THREAD_VARIABLES`` and the release tool's loop).
_THREAD_NAMES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "LOKY_MAX_CPU_COUNT",
)
_NAMESPACE_PACKAGES = (
    ("microcosm-build", "build"),
    ("microcosm-calibrate", "calibrate"),
    ("microcosm-data", "data"),
    ("microcosm-frame", "frame"),
)


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location("check_us_census_smoke", _TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return module


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *NO_BACKGROUND_MAINTENANCE, "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _current_pins(tool) -> tuple[dict, dict]:
    """The pins as this checkout's source would set them."""
    definitions = {
        path: tool.definition_digests((_ROOT / path).read_text(), names)
        for path, names in tool.MIRRORED_DEFINITIONS.items()
    }
    files = {
        path: hashlib.sha256((_ROOT / path).read_bytes()).hexdigest()
        for path in tool.PINNED_FILES
    }
    return definitions, files


def _source_repository(tool, root: Path) -> tuple[Path, str]:
    """A committed repository holding this checkout's release-tool source."""
    repo = root / "repo"
    for path in (*tool.SOURCE_PATHS, "uv.lock", "pyproject.toml"):
        target = repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((_ROOT / path).read_bytes())
    # Namespace portions, put first on PYTHONPATH for the launch records, so
    # the records see microcosm import from this repository.
    for package, module in _NAMESPACE_PACKAGES:
        init = repo / "packages" / package / "src" / "microcosm" / module
        init.mkdir(parents=True, exist_ok=True)
        (init / "__init__.py").write_text("")
    subprocess.run(
        ["git", *NO_BACKGROUND_MAINTENANCE, "init", "-q", str(repo)], check=True
    )
    _git(repo, "add", "-A")
    _git(
        repo,
        "-c",
        "user.name=test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-q",
        "-m",
        "release tool source",
    )
    return repo.resolve(), _git(repo, "rev-parse", "HEAD")


def _totals(expectation: dict) -> dict[str, tuple[float, float, float]]:
    """Totals that pass each probe: an effect of twice its floor."""
    totals = {}
    for index, probe in enumerate(expectation["probes"]):
        baseline = 1.0e13 + index * 1.0e9
        wanted = 2.0 * probe["min_abs_effect"]
        if probe["expected_sign"] == "negative":
            wanted = -wanted
        if probe["effect_direction"] == "baseline_minus_reform":
            reform = baseline - wanted
            effect = baseline - reform
        else:
            reform = baseline + wanted
            effect = reform - baseline
        totals[probe["id"]] = (baseline, reform, effect)
    return totals


def _signed(effect: float, sign: str) -> float:
    if sign == "either":
        return abs(effect)
    return effect if sign == "positive" else -effect


def _probe_outputs(tool, expectation, *, repo, commit, runtime, dataset, scoring):
    """The probe's report, gate file and pass log for a passing census smoke."""
    probes = expectation["probes"]
    n_probes = len(probes)
    dataset_sha = scoring["dataset_sha256"]
    totals = _totals(expectation)
    timing = {
        "wall_seconds": 1.5,
        "cpu_seconds": 1.25,
        "rss_before_gib": 1.0,
        "peak_rss_gib": 2.0,
    }
    rows = []
    results = {}
    for probe in probes:
        baseline, reform, effect = totals[probe["id"]]
        magnitude = _signed(effect, probe["expected_sign"])
        results[probe["id"]] = {
            "name": probe["name"],
            "period": probe["period"],
            "budget_measure": probe["budget_measure"],
            "baseline_total": baseline,
            "reform_total": reform,
            "effect": effect,
            "expected_sign": probe["expected_sign"],
            "min_abs_effect": probe["min_abs_effect"],
            "binding_inputs": probe["binding_inputs"],
            "issue": probe["issue"],
            "passed": True,
        }
        rows.append(
            {
                "probe": probe["id"],
                "period": probe["period"],
                "budget_measure": probe["budget_measure"],
                "baseline_total": baseline,
                "reform_total": reform,
                "effect": effect,
                "min_abs_effect": probe["min_abs_effect"],
                "expected_sign": probe["expected_sign"],
                "signed_magnitude": magnitude,
                "margin_to_floor": magnitude - probe["min_abs_effect"],
                "passed": True,
                "measure_entity": "household",
                "standard_error": 0.0,
                "decomposition": "decomposed",
                "engine_weight_scale": 1.0,
                "drawn_effect_households": 0,
                "effective_variance_households": 0.0,
                "certainty_effect_households": 12,
                "noncarrier_effect": 0.0,
                "noncarrier_effect_households": 0,
                "pool_carrier_households": None,
                "expected_sampled_carriers": None,
                "take_all": False,
                "sampled_carrier_households": 12,
                "authority": "authoritative",
                "authority_reason": tool.CENSUS_REASON,
            }
        )
    consumer = {
        "baseline_plan": expectation["baseline_plan"],
        "baseline_passes": 1,
        "reform_passes": n_probes,
        "reform_systems": n_probes,
    }
    block = {key: value for key, value in scoring.items() if key != "consumers"}
    gate = {
        "schema_version": 1,
        "enforced": True,
        "reform_coverage_smoke": {
            "passed": True,
            "failures": [],
            "details": {
                "default_period": tool.PERIOD,
                "probes": n_probes,
                "results": results,
            },
        },
        "post_export_scoring": {**block, **consumer},
    }
    stages = {
        "identify": {"status": "completed", "sha256": dataset_sha, "timing": timing},
        "load": {
            "status": "completed",
            "sha256": dataset_sha,
            "rows": {"household": _N_HOUSEHOLDS, "person": 2 * _N_HOUSEHOLDS},
            "timing": timing,
        },
        "design": {
            "status": "completed",
            "household_weight_total": 1.3e8,
            "batch_size": _BATCH,
            "n_batches": scoring["n_batches"],
            "census": True,
            "strata": 0,
            "verified_against_receipt": True,
            "problems": [],
            "timing": timing,
        },
        "open_scorer": {
            "status": "completed",
            "n_households": _N_HOUSEHOLDS,
            "n_batches": scoring["n_batches"],
            "max_batch_households": scoring["max_batch_households"],
            "maximum_batch_size": _BATCH,
            "timing": timing,
        },
        "reform_coverage_smoke": {
            "status": "completed",
            "passed": True,
            "failures": [],
            "analysis_setup_error": None,
            "reference_error": None,
            "baseline_keys": len(expectation["baseline_plan"]["keys"]),
            "reform_passes": n_probes,
            "reform_systems": n_probes,
            "probes": rows,
            "timing": timing,
        },
        "post_export_scoring_plans": {
            "status": "completed",
            "compared": False,
            "timing": timing,
        },
    }
    report = {
        "schema_version": tool.PROBE_SCHEMA_VERSION,
        "tool": tool.PROBE_PATH,
        "tool_source": {
            "commit": commit,
            "dirty": False,
            "changes_sha256": None,
            "sha256": expectation["sources"][tool.PROBE_PATH],
            "installed_distributions": runtime["installed_distributions"],
            "tools": {
                path: {
                    "sha256": expectation["sources"][path],
                    "file": str(repo / path),
                    "hashed": "at load",
                }
                for path in (tool.BUILDER_PATH, tool.SAMPLER_PATH)
            },
            "commit_at_write": commit,
            "dirty_at_write": False,
            "changes_sha256_at_write": None,
            "moved_since_load": False,
        },
        "packages": {
            name: runtime["runtime_versions"][name] for name in tool.PROBE_PACKAGES
        },
        "created_at": _TIMES["report"].isoformat(timespec="seconds"),
        "host": {"platform": "test-platform", "python": runtime["version"]},
        "classification": tool.PROBE_CLASSIFICATION,
        "export": str(dataset),
        "stages_requested": ["reform_coverage_smoke"],
        "rules": {"se_multiplier": 4.0, "min_effective_households": 30.0},
        "sample": None,
        "source_export": None,
        "target_surface": {
            "mode": "full",
            "dropped_congressional_district_targets": None,
            "source": "default (no flag and no reference release manifest)",
        },
        "not_reproduced": [],
        "references": {
            "paths": dict.fromkeys(tool.PROBE_REFERENCE_KEYS),
            "smoke": False,
            "validation": False,
            "build_manifest": False,
            "calibration_diagnostics": False,
            "qrf_tail": False,
        },
        "stages": stages,
        "verdicts": [
            {
                "stage": "reform_coverage_smoke",
                "check": f"probe {probe['id']}",
                "verdict": "pass",
                "authority": "authoritative",
                "reason": tool.CENSUS_REASON,
                "release_consequence": "raises",
            }
            for probe in probes
        ],
        "post_export_scoring": {
            **block,
            "consumers": {"reform_coverage_smoke": consumer},
        },
        "scored_dataset_sha256": dataset_sha,
        "summary": {
            "probe_failures": [],
            "authoritative_failures": [],
            "informational_failures": [],
            "stage_status": dict.fromkeys(tool.PROBE_STAGES, "completed"),
        },
        "process_peak_rss_gib": 3.0,
        "process_peak_rss_source": "test",
        "rss_source": "test",
    }
    passes = [
        {
            "stage": "reform_coverage_smoke",
            "label": row["label"],
            "keys": row["keys"],
            "reform": row["reform"],
            "batches": scoring["n_batches"],
            "wall_seconds": 2.0,
            "cpu_seconds": 1.5,
            "rss_before_gib": 1.0,
            "peak_rss_gib": 1.5,
        }
        for row in expectation["passes"]
    ]
    return report, gate, passes


@dataclass
class Layout:
    repo: Path
    commit: str
    pins: tuple[dict, dict]
    expectation: dict
    release_id: str
    out: Path
    release_dir: Path
    artifact_root: Path
    dataset: Path
    probe_dir: Path
    launch_dir: Path

    def inputs(self, tool, **overrides):
        values = {
            "release_dir": self.release_dir,
            "artifact_root": self.artifact_root,
            "probe_dir": self.probe_dir,
            "launch_dir": self.launch_dir,
            "repo": self.repo,
            "authorization": "the test ruling",
            "dedicated_worktree": True,
            "probe_count": len(self.expectation["probes"]),
        }
        values.update(overrides)
        return tool.Inputs(**values)

    def file(self, name: str) -> Path:
        return {
            "build": self.release_dir / "build_manifest.json",
            "release": self.release_dir / "release_manifest.json",
            "report": self.probe_dir / "probe_report.json",
            "gate": self.probe_dir / "reform_coverage_smoke.json",
            "passes": self.probe_dir / "passes.jsonl",
            "build-start": self.launch_dir / "build-start.json",
            "build-end": self.launch_dir / "build-end.json",
            "smoke-start": self.launch_dir / "smoke-start.json",
            "smoke-end": self.launch_dir / "smoke-end.json",
        }[name]


@pytest.fixture(scope="module")
def layout(tool, tmp_path_factory) -> Layout:
    """A faithful census smoke of a skipped-smoke build, built once."""
    root = tmp_path_factory.mktemp("census-smoke").resolve()
    repo, commit = _source_repository(tool, root)
    pins = _current_pins(tool)
    out = root / "build-out"
    release_id = f"microcosm-us-2024-{commit[:12]}-20261001T103000Z"
    release_dir = out / "releases" / release_id
    artifact_root = out / "artifacts"
    dataset = artifact_root / _DATASET_FILE
    probe_dir = root / "smoke-out"
    launch_dir = root / "launch"
    launch_dir.mkdir()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(tool, "MIRRORED_DEFINITIONS", pins[0])
        patch.setattr(tool, "PINNED_FILES", pins[1])
        expectation = tool.build_expectation(repo, commit)
        patch.chdir(repo)
        patch.setenv(
            "PYTHONPATH",
            os.pathsep.join(
                str(repo / "packages" / package / "src")
                for package, _ in _NAMESPACE_PACKAGES
            ),
        )
        for name in tool.THREAD_VARIABLES:
            patch.setenv(name, _THREADS)

        def record(stage, phase, *, out_dir=None, exit_code=None):
            snapshot = tool.snapshot(
                stage=stage,
                phase=phase,
                run_id="test-run",
                repo=repo,
                python=sys.executable,
                dataset=dataset,
                out_dir=out_dir,
                exit_code=exit_code,
                now=lambda: _TIMES[f"{stage}-{phase}"],
            )
            path = launch_dir / tool.LAUNCH_FILES[(stage, phase)]
            path.write_text(tool.strict_dumps(snapshot))
            return snapshot

        build_start = record("build", "start", out_dir=out)
        runtime = build_start["runtime"]
        artifact_root.mkdir(parents=True)
        release_dir.mkdir(parents=True)
        dataset.write_bytes(_DATASET_BYTES)
        dataset_sha = hashlib.sha256(_DATASET_BYTES).hexdigest()
        n_batches = -(-_N_HOUSEHOLDS // _BATCH)
        scoring = {
            "method": tool.SCORING_METHOD,
            "dataset_sha256": dataset_sha,
            "maximum_batch_size": _BATCH,
            "n_households": _N_HOUSEHOLDS,
            "n_batches": n_batches,
            "max_batch_households": _BATCH,
            "spm": {"geography_kind": "county"},
            "consumers": {
                "demographics": {
                    "baseline_plan": {
                        "keys": [{"variable": "age", "period": 2024, "map_to": None}],
                        "period_order": [2024],
                    },
                    "baseline_passes": 1,
                    "reform_passes": 0,
                    "reform_systems": 0,
                }
            },
        }
        evidence = {
            "esi_premiums": "bound",
            "input_coverage": "bound",
            "input_mass_parity": "bound",
            "qrf_tail_concentration": "bound",
            "reform_coverage_smoke": "skipped",
            "stored_inputs": "bound",
        }
        created_at = _TIMES["manifests"].isoformat()
        build_manifest = {
            "build_id": release_id,
            "build_sha": commit[:7],
            "created_at": created_at,
            "staging": None,
            "code": {
                "repository": "PolicyEngine/microcosm",
                "git_commit": commit,
                "git_dirty": False,
            },
            "runtime": runtime["runtime_versions"],
            "gate_evidence": evidence,
            "post_export_scoring": scoring,
            "dataset": {
                "filename": _DATASET_FILE,
                "sha256": dataset_sha,
                "default": None,
            },
        }
        release_manifest = {
            "schema_version": 1,
            "default_datasets": {"national": _DATASET_KEY},
            "artifacts": {
                _DATASET_KEY: {
                    "kind": "microdata",
                    "path": _DATASET_FILE,
                    "repo_id": _REPO_ID,
                    "revision": release_id,
                    "sha256": dataset_sha,
                }
            },
            "build": {
                "build_id": release_id,
                "built_at": created_at,
                "gate_evidence": evidence,
                "post_export_scoring": scoring,
            },
        }
        (release_dir / "build_manifest.json").write_text(
            json.dumps(build_manifest, indent=1, allow_nan=False)
        )
        (release_dir / "release_manifest.json").write_text(
            json.dumps(release_manifest, indent=2, allow_nan=False)
        )
        record("build", "end", exit_code=0)
        record("smoke", "start", out_dir=probe_dir)
        probe_dir.mkdir()
        report, gate, passes = _probe_outputs(
            tool,
            expectation,
            repo=repo,
            commit=commit,
            runtime=runtime,
            dataset=dataset,
            scoring=scoring,
        )
        (probe_dir / "probe_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        (probe_dir / "reform_coverage_smoke.json").write_text(
            json.dumps(gate, indent=2, sort_keys=True) + "\n"
        )
        (probe_dir / "passes.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in passes)
        )
        record("smoke", "end", exit_code=0)
    return Layout(
        repo=repo,
        commit=commit,
        pins=pins,
        expectation=expectation,
        release_id=release_id,
        out=out,
        release_dir=release_dir,
        artifact_root=artifact_root,
        dataset=dataset,
        probe_dir=probe_dir,
        launch_dir=launch_dir,
    )


@pytest.fixture
def quick(tool, layout, monkeypatch):
    """The check with its expectation served from the fixture (the git and
    parse work is exercised once, in the acceptance test)."""
    monkeypatch.setattr(
        tool,
        "build_expectation",
        lambda repo, commit: copy.deepcopy(layout.expectation),
    )
    return tool


def _attachments(tool, layout) -> list[str]:
    return sorted(
        name
        for name in tool.ATTACHMENTS.values()
        if (layout.release_dir / name).exists()
    )


@contextmanager
def _edited(path: Path, edit: Callable[[Any], None]) -> Iterator[None]:
    """``edit`` the JSON (or JSON lines) at ``path``; restore it afterwards."""
    original = path.read_bytes()
    try:
        if path.suffix == ".jsonl":
            rows = [json.loads(line) for line in original.decode().splitlines()]
            edit(rows)
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        else:
            payload = json.loads(original)
            edit(payload)
            path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        yield
    finally:
        path.write_bytes(original)


def _refusal(tool, layout, **overrides) -> str:
    with pytest.raises(tool.RefusalError) as refused:
        tool.check(layout.inputs(tool, **overrides), now=lambda: _TIMES["check"])
    assert _attachments(tool, layout) == []
    return str(refused.value)


# ---------------------------------------------------------------------------
# Acceptance
# ---------------------------------------------------------------------------


def test_a_faithful_census_smoke_is_accepted(tool, layout, monkeypatch) -> None:
    monkeypatch.setattr(tool, "MIRRORED_DEFINITIONS", layout.pins[0])
    monkeypatch.setattr(tool, "PINNED_FILES", layout.pins[1])
    inputs = layout.inputs(tool)
    accepted = tool.check(inputs, now=lambda: _TIMES["check"])
    try:
        written = tool.write_attachments(inputs=inputs, accepted=accepted)
        assert list(written) == list(tool.ATTACHMENTS.values())
        for name, source in (
            ("report", "probe_report.json"),
            ("gate", "reform_coverage_smoke.json"),
            ("passes", "passes.jsonl"),
        ):
            assert (layout.release_dir / tool.ATTACHMENTS[name]).read_bytes() == (
                layout.probe_dir / source
            ).read_bytes()
        receipt = tool.strict_loads(
            (layout.release_dir / tool.ATTACHMENTS["receipt"]).read_bytes()
        )
        assert receipt["scope"] == tool.SCOPE
        assert receipt["accepted"] is True
        assert receipt["release_id"] == receipt["tag"] == layout.release_id
        assert receipt["build_commit"] == layout.commit
        assert receipt["run_id"] == "test-run"
        assert receipt["authorization"] == "the test ruling"
        assert receipt["required_probes"] == len(layout.expectation["probes"])
        assert receipt["status"] == {
            "build_smoke": "skipped",
            "external_smoke": "passed-before-publish",
            "reform_validation": "not-run-not-certified",
            "default_promotion_allowed": False,
        }
        assert receipt["publication"]["tag_only"] is True
        assert receipt["publication"]["updates_latest"] is False
        assert receipt["expectation"] == layout.expectation
        assert (
            receipt["dataset"]["sha256"] == hashlib.sha256(_DATASET_BYTES).hexdigest()
        )
        assert receipt["smoke"]["probes"] == len(layout.expectation["probes"])
        assert receipt["smoke"]["analysis"]["decomposed"] == len(
            layout.expectation["probes"]
        )
        assert set(receipt["launches"]) == {
            "build-start",
            "build-end",
            "smoke-start",
            "smoke-end",
        }
        for name in ("report", "gate", "passes", "note"):
            filename = tool.ATTACHMENTS[name]
            data = (layout.release_dir / filename).read_bytes()
            assert receipt["files"][filename] == {
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
            }
        note = (layout.release_dir / tool.ATTACHMENTS["note"]).read_text()
        assert layout.release_id in note
        assert "not for promotion to the default release" in note
        preflight, publish = tool.publish_commands(accepted, inputs)
        assert preflight.endswith("--preflight-only")
        assert "--no-latest --tag-only" in publish
        assert f"--repo-id {_REPO_ID}" in publish
        assert "--repo-id" not in preflight
        for filename in tool.ATTACHMENTS.values():
            assert f"--extra-file {filename}" in preflight
    finally:
        for filename in tool.ATTACHMENTS.values():
            (layout.release_dir / filename).unlink(missing_ok=True)


def test_main_accepts_and_prints_the_commands(quick, layout, capsys) -> None:
    argv = [
        "check",
        "--release-dir",
        str(layout.release_dir),
        "--artifact-root",
        str(layout.artifact_root),
        "--probe-dir",
        str(layout.probe_dir),
        "--launch-dir",
        str(layout.launch_dir),
        "--repo",
        str(layout.repo),
        "--authorization",
        "the test ruling",
        "--probe-count",
        str(len(layout.expectation["probes"])),
        "--attest-dedicated-worktree",
    ]
    try:
        assert quick.main(argv) == 0
        out = capsys.readouterr().out
        assert f"accepted {layout.release_id}" in out
        assert "--preflight-only" in out
        assert _attachments(quick, layout) == sorted(quick.ATTACHMENTS.values())
        # The attachments are never overwritten: a second run refuses.
        assert quick.main(argv) == 1
        assert "already holds" in capsys.readouterr().err
    finally:
        for filename in quick.ATTACHMENTS.values():
            (layout.release_dir / filename).unlink(missing_ok=True)


def test_a_failed_write_leaves_no_attachment(quick, layout, monkeypatch) -> None:
    inputs = layout.inputs(quick)
    accepted = quick.check(inputs, now=lambda: _TIMES["check"])

    def broken(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(quick, "receipt", broken)
    with pytest.raises(OSError):
        quick.write_attachments(accepted, inputs)
    assert _attachments(quick, layout) == []


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def _set(*path: str | int, value: Any) -> Callable[[Any], None]:
    def edit(payload):
        target = payload
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    return edit


def _first_probe(layout) -> str:
    return layout.expectation["probes"][0]["id"]


_REFUSALS = [
    # (case, file, edit(layout) -> edit(payload), expected text)
    (
        "gate passed",
        "gate",
        lambda lo: _set("reform_coverage_smoke", "passed", value=False),
        "gate passed",
    ),
    (
        "gate enforced",
        "gate",
        lambda lo: _set("enforced", value=False),
        "gate enforced",
    ),
    (
        "gate count",
        "gate",
        lambda lo: _set("reform_coverage_smoke", "details", "probes", value=1),
        "gate probe count",
    ),
    (
        "result passed",
        "gate",
        lambda lo: _set(
            "reform_coverage_smoke",
            "details",
            "results",
            _first_probe(lo),
            "passed",
            value=False,
        ),
        "passed",
    ),
    (
        "result floor",
        "gate",
        lambda lo: _set(
            "reform_coverage_smoke",
            "details",
            "results",
            _first_probe(lo),
            "min_abs_effect",
            value=1.0,
        ),
        "min_abs_effect",
    ),
    (
        "result inputs",
        "gate",
        lambda lo: _set(
            "reform_coverage_smoke",
            "details",
            "results",
            _first_probe(lo),
            "binding_inputs",
            value=[],
        ),
        "binding_inputs",
    ),
    (
        "result extra field",
        "gate",
        lambda lo: _set(
            "reform_coverage_smoke",
            "details",
            "results",
            _first_probe(lo),
            "note",
            value="x",
        ),
        "fields",
    ),
    (
        "result missing",
        "gate",
        lambda lo: (
            lambda payload: payload["reform_coverage_smoke"]["details"]["results"].pop(
                _first_probe(lo)
            )
        ),
        "gate result ids",
    ),
    (
        "gate plan",
        "gate",
        lambda lo: (
            lambda payload: payload["post_export_scoring"]["baseline_plan"][
                "keys"
            ].pop()
        ),
        "post_export_scoring",
    ),
    (
        "report sample",
        "report",
        lambda lo: _set("sample", value={"fraction": 0.05}),
        "report sample",
    ),
    (
        "report census",
        "report",
        lambda lo: _set("stages", "design", "census", value=False),
        "design census",
    ),
    (
        "report stages",
        "report",
        lambda lo: _set(
            "stages_requested", value=["reform_coverage_smoke", "demographics"]
        ),
        "stages_requested",
    ),
    (
        "report verdict",
        "report",
        lambda lo: _set("verdicts", 0, "authority", value="informational"),
        "report verdicts",
    ),
    (
        "report summary",
        "report",
        lambda lo: _set(
            "summary", "informational_failures", value=[{"verdict": "fail"}]
        ),
        "report summary",
    ),
    (
        "report dirty",
        "report",
        lambda lo: _set("tool_source", "dirty", value=True),
        "tool_source.dirty",
    ),
    (
        "report moved",
        "report",
        lambda lo: _set("tool_source", "moved_since_load", value=True),
        "moved_since_load",
    ),
    (
        "report probe bytes",
        "report",
        lambda lo: _set("tool_source", "sha256", value="0" * 64),
        "tool_source.sha256",
    ),
    (
        "report packages",
        "report",
        lambda lo: _set("packages", "numpy", value="0.0.1"),
        "report packages",
    ),
    (
        "report digest",
        "report",
        lambda lo: _set("scored_dataset_sha256", value="0" * 64),
        "scored digest",
    ),
    (
        "report references",
        "report",
        lambda lo: _set("references", "smoke", value=True),
        "report references",
    ),
    (
        "report row authority",
        "report",
        lambda lo: _set(
            "stages",
            "reform_coverage_smoke",
            "probes",
            0,
            "authority",
            value="informational",
        ),
        "authority",
    ),
    (
        "report reform passes",
        "report",
        lambda lo: _set(
            "post_export_scoring",
            "consumers",
            "reform_coverage_smoke",
            "reform_passes",
            value=1,
        ),
        "report post_export_scoring",
    ),
    (
        "report export",
        "report",
        lambda lo: _set("export", value="/elsewhere.h5"),
        "export path",
    ),
    (
        "passes short",
        "passes",
        lambda lo: lambda rows: rows.pop(),
        "scoring passes in the log",
    ),
    (
        "passes label",
        "passes",
        lambda lo: _set(1, "label", value="reform_coverage_smoke reform x@2024"),
        "label",
    ),
    ("passes reform", "passes", lambda lo: _set(0, "reform", value=True), "reform"),
    (
        "passes stage",
        "passes",
        lambda lo: _set(0, "stage", value="demographics"),
        "stage",
    ),
    (
        "build smoke bound",
        "build",
        lambda lo: _set("gate_evidence", "reform_coverage_smoke", value="bound"),
        "gate_evidence.reform_coverage_smoke",
    ),
    (
        "build dirty",
        "build",
        lambda lo: _set("code", "git_dirty", value=True),
        "git_dirty",
    ),
    (
        "build scored the smoke",
        "build",
        lambda lo: _set(
            "post_export_scoring", "consumers", "reform_coverage_smoke", value={}
        ),
        "the build scored reform_coverage_smoke",
    ),
    (
        "build batches",
        "build",
        lambda lo: _set("post_export_scoring", "n_batches", value=3),
        "n_batches",
    ),
    (
        "release sha",
        "release",
        lambda lo: _set("artifacts", _DATASET_KEY, "sha256", value="0" * 64),
        "national artifact's sha256",
    ),
    (
        "release scoring",
        "release",
        lambda lo: _set(
            "build", "post_export_scoring", "spm", value={"geography_kind": "state"}
        ),
        "release build.post_export_scoring",
    ),
    (
        "release revision",
        "release",
        lambda lo: _set("artifacts", _DATASET_KEY, "revision", value="other"),
        "revision",
    ),
    (
        "smoke threads",
        "smoke-start",
        lambda lo: _set("environment", "OPENBLAS_NUM_THREADS", value="18"),
        "thread-pool",
    ),
    (
        "smoke hash seed",
        "smoke-start",
        lambda lo: _set("environment", "PYTHONHASHSEED", value="424242"),
        "PYTHONHASHSEED",
    ),
    (
        "build exit",
        "build-end",
        lambda lo: _set("exit_code", value=1),
        "build's exit code",
    ),
    (
        "smoke exit",
        "smoke-end",
        lambda lo: _set("exit_code", value=-9),
        "smoke's exit code",
    ),
    (
        "smoke inventory",
        "smoke-end",
        lambda lo: _set("runtime", "installed_distributions", "sha256", value="0" * 64),
        "smoke-end runtime",
    ),
    (
        "smoke head",
        "smoke-start",
        lambda lo: _set("repo", "head", value="0" * 40),
        "smoke-start HEAD",
    ),
    (
        "smoke dirty",
        "smoke-end",
        lambda lo: _set("repo", "clean", value=False),
        "smoke-end clean tree",
    ),
    (
        "smoke shadow",
        "smoke-start",
        lambda lo: _set("cwd_shadows", value=["release_input_coverage_manifest.json"]),
        "shadowing",
    ),
    (
        "smoke dataset",
        "smoke-end",
        lambda lo: _set("dataset", "sha256", value="0" * 64),
        "smoke-end dataset",
    ),
    (
        "smoke out dir",
        "smoke-start",
        lambda lo: _set("out_dir", "state", value="nonempty"),
        "new, empty output",
    ),
    (
        "build lock",
        "build-end",
        lambda lo: _set("files", "uv.lock", value="0" * 64),
        "build-end files",
    ),
    (
        "order",
        "smoke-start",
        lambda lo: _set(
            "recorded_at",
            value=(_TIMES["build-end"] - timedelta(minutes=1)).isoformat(),
        ),
        "build-end does not precede smoke-start",
    ),
    (
        "report time",
        "report",
        lambda lo: _set(
            "created_at",
            value=(_TIMES["smoke-start"] - timedelta(minutes=1)).isoformat(
                timespec="seconds"
            ),
        ),
        "smoke-start does not precede report",
    ),
    (
        "microcosm origin",
        "build-start",
        lambda lo: _set(
            "runtime",
            "import_origins",
            "microcosm.build",
            value={
                "origin": "/elsewhere/microcosm/build/__init__.py",
                "locations": None,
            },
        ),
        "imports from outside",
    ),
    # The release's own identity.
    (
        "release build id",
        "release",
        lambda lo: _set("build", "build_id", value="another-release"),
        "release build_id",
    ),
    (
        "release built at",
        "release",
        lambda lo: _set("build", "built_at", value=_TIMES["build-end"].isoformat()),
        "release build.built_at",
    ),
    (
        "build repository",
        "build",
        lambda lo: _set("code", "repository", value="someone/else"),
        "build code.repository",
    ),
    (
        "artifact kind",
        "release",
        lambda lo: _set("artifacts", _DATASET_KEY, "kind", value="calibration"),
        "the national artifact's kind",
    ),
    (
        "artifact outside the root",
        "release",
        lambda lo: _set("artifacts", _DATASET_KEY, "path", value=f"../{_DATASET_FILE}"),
        "is not a file under",
    ),
    (
        "artifact repo id",
        "release",
        lambda lo: _set("artifacts", _DATASET_KEY, "repo_id", value=""),
        "names no repo_id",
    ),
    (
        "scoring method",
        "build",
        lambda lo: _set("post_export_scoring", "method", value="whole_pool"),
        "build scoring method",
    ),
    (
        "scoring households",
        "build",
        lambda lo: _set("post_export_scoring", "n_households", value=7001.0),
        "not a positive integer",
    ),
    (
        "too few batches",
        "build",
        lambda lo: (
            lambda payload: payload["post_export_scoring"].update(
                maximum_batch_size=4_000, n_batches=2, max_batch_households=4_000
            )
        ),
        "the probe needs 3 or more",
    ),
    (
        "build validated",
        "build",
        lambda lo: _set(
            "post_export_scoring", "consumers", "reform_validation", value={}
        ),
        "the build scored reform_validation",
    ),
    # Where the stages wrote.
    (
        "build out dir in use",
        "build-start",
        lambda lo: _set("out_dir", "state", value="nonempty"),
        "the build did not start in a new, empty output directory",
    ),
    (
        "release outside the build output",
        "build-start",
        lambda lo: _set("out_dir", "path", value="/elsewhere/build-out"),
        "is not inside the build's output",
    ),
    (
        "smoke wrote elsewhere",
        "smoke-start",
        lambda lo: _set("out_dir", "path", value="/elsewhere/smoke-out"),
        "smoke output",
    ),
    (
        "smoke read another path",
        "smoke-end",
        lambda lo: _set("dataset", "path", value="/elsewhere/microcosm_us_2024.h5"),
        "smoke-end dataset path",
    ),
    (
        "another run",
        "smoke-end",
        lambda lo: _set("run_id", value="another-run"),
        "smoke-end run_id",
    ),
    (
        "another interpreter",
        "smoke-start",
        lambda lo: _set("python", "resolved", value="/usr/bin/python3"),
        "smoke-start python",
    ),
    (
        "naive time",
        "build-end",
        lambda lo: _set("recorded_at", value="2026-10-01T12:30:00"),
        "no timezone-aware time",
    ),
    (
        "a future smoke end",
        "smoke-end",
        lambda lo: _set(
            "recorded_at", value=(_TIMES["check"] + timedelta(hours=1)).isoformat()
        ),
        "smoke-end does not precede now",
    ),
]


@pytest.mark.parametrize(
    ("case", "name", "edit_for", "expected"),
    _REFUSALS,
    ids=[case for case, *_ in _REFUSALS],
)
def test_each_departure_is_refused(
    quick, layout, case, name, edit_for, expected
) -> None:
    with _edited(layout.file(name), edit_for(layout)):
        message = _refusal(quick, layout)
    assert expected in message, message


def test_the_faithful_set_passes_the_quick_check(quick, layout) -> None:
    # The refusals above each change one thing; this is the set they change.
    accepted = quick.check(layout.inputs(quick), now=lambda: _TIMES["check"])
    assert accepted.release_id == layout.release_id


def test_non_finite_and_duplicate_json_is_refused(quick, layout) -> None:
    path = layout.file("report")
    original = path.read_bytes()
    try:
        path.write_bytes(
            original.replace(
                b'"process_peak_rss_gib": 3.0', b'"process_peak_rss_gib": NaN'
            )
        )
        assert "not strict JSON" in _refusal(quick, layout)
        path.write_bytes(
            original.replace(
                b'"rss_source": "test"', b'"rss_source": "a", "rss_source": "b"'
            )
        )
        assert "duplicate key" in _refusal(quick, layout)
    finally:
        path.write_bytes(original)


@pytest.mark.parametrize(
    "filename", ["reform_validation.json", "reform_coverage_smoke.json"]
)
def test_canonical_evidence_in_the_release_directory_is_refused(
    quick, layout, filename
) -> None:
    path = layout.release_dir / filename
    path.write_text("{}")
    try:
        assert f"holds {filename}" in _refusal(quick, layout)
    finally:
        path.unlink()


def test_a_changed_dataset_is_refused(quick, layout) -> None:
    original = layout.dataset.read_bytes()
    try:
        layout.dataset.write_bytes(original + b"!")
        message = _refusal(quick, layout)
        assert "build dataset.sha256" in message
    finally:
        layout.dataset.write_bytes(original)


def test_the_operator_must_name_the_ruling_and_attest(quick, layout) -> None:
    assert "--authorization" in _refusal(quick, layout, authorization="  ")
    assert "--attest-dedicated-worktree" in _refusal(
        quick, layout, dedicated_worktree=False
    )


def test_the_probe_count_must_be_the_rulings(quick, layout) -> None:
    # The ruling names how many probes must pass; a build commit whose manifest
    # lists more or fewer is not the build it allowed.
    n = len(layout.expectation["probes"])
    for wrong in (n - 1, n + 1):
        message = _refusal(quick, layout, probe_count=wrong)
        assert "the build commit's probe count (--probe-count)" in message
    for invalid in (0, -1, True):
        message = _refusal(quick, layout, probe_count=invalid)
        assert "--probe-count must be the ruling's positive probe count" in message


def test_a_source_departure_refuses_the_expectation(tool, layout, monkeypatch) -> None:
    definitions, files = copy.deepcopy(layout.pins)
    definitions[tool.BUILDER_PATH]["_ascending_period_plan"] = "0" * 64
    monkeypatch.setattr(tool, "MIRRORED_DEFINITIONS", definitions)
    monkeypatch.setattr(tool, "PINNED_FILES", files)
    message = _refusal(tool, layout)
    assert "_ascending_period_plan differs" in message


# ---------------------------------------------------------------------------
# Property: one changed field anywhere the check reads is refused
# ---------------------------------------------------------------------------


def _leaves(value: Any, path: tuple = ()) -> list[tuple]:
    if isinstance(value, dict):
        return [leaf for key in value for leaf in _leaves(value[key], (*path, key))]
    if isinstance(value, list) and value:
        return [
            leaf
            for index, item in enumerate(value)
            for leaf in _leaves(item, (*path, index))
        ]
    return [path]


def _read_leaf(payload: Any, path: tuple) -> Any:
    for key in path:
        payload = payload[key]
    return payload


_REPORT_CHECKED = (
    ("schema_version",),
    ("tool",),
    ("classification",),
    ("stages_requested",),
    ("sample",),
    ("source_export",),
    ("references",),
    ("summary",),
    ("verdicts",),
    ("post_export_scoring",),
    ("scored_dataset_sha256",),
    ("tool_source",),
    ("packages",),
    ("host", "python"),
    ("export",),
    ("stages", "identify", "status"),
    ("stages", "identify", "sha256"),
    ("stages", "load", "sha256"),
    ("stages", "load", "rows", "household"),
    ("stages", "design", "census"),
    ("stages", "design", "problems"),
    ("stages", "design", "verified_against_receipt"),
    ("stages", "design", "strata"),
    ("stages", "design", "batch_size"),
    ("stages", "design", "n_batches"),
    ("stages", "open_scorer", "n_households"),
    ("stages", "open_scorer", "n_batches"),
    ("stages", "open_scorer", "max_batch_households"),
    ("stages", "open_scorer", "maximum_batch_size"),
    ("stages", "reform_coverage_smoke", "status"),
    ("stages", "reform_coverage_smoke", "passed"),
    ("stages", "reform_coverage_smoke", "failures"),
    ("stages", "reform_coverage_smoke", "analysis_setup_error"),
    ("stages", "reform_coverage_smoke", "reference_error"),
    ("stages", "reform_coverage_smoke", "baseline_keys"),
    ("stages", "reform_coverage_smoke", "reform_passes"),
    ("stages", "reform_coverage_smoke", "reform_systems"),
    ("stages", "post_export_scoring_plans", "compared"),
)
_ROW_CHECKED = (
    "probe",
    "period",
    "budget_measure",
    "min_abs_effect",
    "expected_sign",
    "passed",
    "authority",
    "authority_reason",
    "baseline_total",
    "reform_total",
    "effect",
    "signed_magnitude",
    "margin_to_floor",
)
_PASS_CHECKED = ("stage", "label", "keys", "reform", "batches")


def _checked_leaves(layout) -> list[tuple[str, tuple]]:
    report = json.loads(layout.file("report").read_text())
    gate = json.loads(layout.file("gate").read_text())
    passes = [
        json.loads(line) for line in layout.file("passes").read_text().splitlines()
    ]
    leaves = [("gate", path) for path in _leaves(gate)]
    for prefix in _REPORT_CHECKED:
        leaves += [
            ("report", path) for path in _leaves(_read_leaf(report, prefix), prefix)
        ]
    rows = report["stages"]["reform_coverage_smoke"]["probes"]
    leaves += [
        ("report", ("stages", "reform_coverage_smoke", "probes", index, field))
        for index in range(len(rows))
        for field in _ROW_CHECKED
    ]
    leaves += [
        ("passes", (index, field))
        for index in range(len(passes))
        for field in _PASS_CHECKED
    ]
    return leaves


_JSON_SCALARS = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-3, max_value=3),
    st.floats(allow_nan=False, allow_infinity=False, width=64),
    st.text(max_size=4),
    st.just([]),
    st.just({}),
)


@settings(
    max_examples=150,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(data=st.data())
def test_any_changed_checked_field_is_refused(quick, layout, data) -> None:
    leaves = _checked_leaves(layout)
    name, path = data.draw(st.sampled_from(leaves))
    file = layout.file(name)
    payload = json.loads(file.read_text()) if name != "passes" else None
    original = (
        _read_leaf(payload, path)
        if payload is not None
        else _read_leaf([json.loads(x) for x in file.read_text().splitlines()], path)
    )
    replacement = data.draw(
        _JSON_SCALARS.filter(lambda value: not quick.same(value, original))
    )
    with _edited(file, _set(*path, value=replacement)):
        _refusal(quick, layout)


# ---------------------------------------------------------------------------
# launch-record
# ---------------------------------------------------------------------------


def _bare_repo(root: Path) -> Path:
    repo = root / "repo"
    (repo / "tools").mkdir(parents=True)
    (repo / "tools" / "keep").write_text("")
    (repo / "uv.lock").write_text("lock\n")
    subprocess.run(
        ["git", *NO_BACKGROUND_MAINTENANCE, "init", "-q", str(repo)], check=True
    )
    _git(repo, "add", "-A")
    _git(
        repo,
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@example.invalid",
        "commit",
        "-q",
        "-m",
        "x",
    )
    return repo.resolve()


def _launch_args(tool, repo, launch_dir, stage, phase, **extra) -> list[str]:
    argv = [
        "launch-record",
        "--stage",
        stage,
        "--phase",
        phase,
        "--run-id",
        "r1",
        "--launch-dir",
        str(launch_dir),
        "--python",
        sys.executable,
        "--repo",
        str(repo),
    ]
    for key, value in extra.items():
        argv += [f"--{key.replace('_', '-')}", str(value)]
    return argv


@pytest.fixture
def stub_runtime(tool, monkeypatch):
    monkeypatch.setattr(tool, "_runtime", lambda python, tools: {"stub": True})
    return tool


def test_launch_records_refuse_a_wasted_start(
    stub_runtime, tmp_path, monkeypatch, capsys
) -> None:
    tool = stub_runtime
    repo = _bare_repo(tmp_path)
    launch_dir = tmp_path / "launch"
    monkeypatch.chdir(repo)
    (repo / "untracked.txt").write_text("x")
    argv = _launch_args(
        tool, repo, launch_dir, "build", "start", out_dir=tmp_path / "out"
    )
    assert tool.main(argv) == 1
    assert "uncommitted changes" in capsys.readouterr().err
    (repo / "untracked.txt").unlink()
    (repo / "release_input_coverage_manifest.json").write_text("{}")
    _git(repo, "add", "-A")
    _git(
        repo,
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@example.invalid",
        "commit",
        "-q",
        "-m",
        "y",
    )
    assert tool.main(argv) == 1
    assert "read before the package's own" in capsys.readouterr().err
    _git(repo, "rm", "-q", "release_input_coverage_manifest.json")
    _git(
        repo,
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@example.invalid",
        "commit",
        "-q",
        "-m",
        "z",
    )
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "old").write_text("x")
    assert tool.main(argv) == 1
    assert "already holds files" in capsys.readouterr().err
    monkeypatch.chdir(repo / "tools")
    (tmp_path / "out" / "old").unlink()
    assert tool.main(argv) == 1
    assert "not the repository root" in capsys.readouterr().err
    assert not launch_dir.exists() or not any(launch_dir.iterdir())


def test_launch_records_follow_the_run(
    stub_runtime, tmp_path, monkeypatch, capsys
) -> None:
    tool = stub_runtime
    repo = _bare_repo(tmp_path)
    launch_dir = tmp_path / "launch"
    monkeypatch.chdir(repo)
    assert (
        tool.main(_launch_args(tool, repo, launch_dir, "build", "end", exit_code=0))
        == 1
    )
    assert "no start record" in capsys.readouterr().err
    assert (
        tool.main(
            _launch_args(
                tool, repo, launch_dir, "smoke", "start", out_dir=tmp_path / "s"
            )
        )
        == 1
    )
    assert "after the build ends" in capsys.readouterr().err
    start = _launch_args(
        tool, repo, launch_dir, "build", "start", out_dir=tmp_path / "out"
    )
    assert tool.main(start) == 0
    assert tool.main(start) == 1
    assert "already exists" in capsys.readouterr().err
    assert tool.main(_launch_args(tool, repo, launch_dir, "build", "end")) == 1
    assert "needs --exit-code" in capsys.readouterr().err
    # An end record is evidence: it is written even after the tree changed.
    (repo / "scratch.txt").write_text("x")
    assert (
        tool.main(_launch_args(tool, repo, launch_dir, "build", "end", exit_code=3))
        == 0
    )
    record = tool.strict_loads((launch_dir / "build-end.json").read_bytes())
    assert record["exit_code"] == 3
    assert record["repo"]["clean"] is False
    (repo / "scratch.txt").unlink()
    assert (
        tool.main(
            _launch_args(
                tool, repo, launch_dir, "smoke", "start", out_dir=tmp_path / "s"
            )
        )
        == 1
    )
    assert "did not exit 0" in capsys.readouterr().err
    # An end record must close the start record's run.
    start = _launch_args(
        tool, repo, launch_dir, "smoke", "start", out_dir=tmp_path / "s2"
    )
    (launch_dir / "build-end.json").unlink()
    assert (
        tool.main(_launch_args(tool, repo, launch_dir, "build", "end", exit_code=0))
        == 0
    )
    assert tool.main(start) == 0
    other = _launch_args(tool, repo, launch_dir, "smoke", "end", exit_code=0)
    other[other.index("--run-id") + 1] = "r2"
    assert tool.main(other) == 1
    assert "names another run" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("environment", "cpus", "expected"),
    [
        (dict.fromkeys(_THREAD_NAMES), 18, "16"),
        (dict.fromkeys(_THREAD_NAMES), 4, "4"),
        (dict.fromkeys(_THREAD_NAMES), None, None),
        (dict.fromkeys(_THREAD_NAMES, "2"), 18, "2"),
    ],
)
def test_effective_threads_are_the_release_tools(
    tool, environment, cpus, expected
) -> None:
    """The release tool sets each pool with ``setdefault`` to
    ``min(cpu_count, 16)``; an operator's value wins."""
    assert _THREAD_NAMES == tool.THREAD_VARIABLES
    record = {"environment": environment, "cpu_count": cpus}
    assert tool._effective_threads(record) == dict.fromkeys(_THREAD_NAMES, expected)
    mixed = {**environment, "OMP_NUM_THREADS": "7"}
    assert (
        tool._effective_threads({"environment": mixed, "cpu_count": cpus})[
            "OMP_NUM_THREADS"
        ]
        == "7"
    )


def test_a_launch_record_captures_the_stage(tool, layout) -> None:
    record = tool.strict_loads(layout.file("build-start").read_bytes())
    runtime = record["runtime"]
    assert record["repo"] == {
        "toplevel": str(layout.repo),
        "head": layout.commit,
        "clean": True,
        "status_sha256": hashlib.sha256(b"").hexdigest(),
        "status_entries": 0,
    }
    assert record["cwd"] == str(layout.repo)
    assert record["environment"]["OMP_NUM_THREADS"] == _THREADS
    assert (
        record["files"]["uv.lock"]
        == hashlib.sha256((_ROOT / "uv.lock").read_bytes()).hexdigest()
    )
    assert runtime["sys_path"][0] == str(layout.repo / "tools")
    assert set(runtime["runtime_versions"]) == {"python", *tool.RUNTIME_PACKAGES}
    origin = runtime["import_origins"]["microcosm.build"]["origin"]
    assert Path(origin).is_relative_to(layout.repo)
    assert record["dataset"] == {
        "path": str(layout.dataset),
        "exists": False,
        "bytes": None,
        "sha256": None,
    }
    assert record["out_dir"] == {"path": str(layout.out), "state": "absent"}


# ---------------------------------------------------------------------------
# Pins and the mirrors, against the code they mirror
# ---------------------------------------------------------------------------


def test_the_pins_are_the_reviewed_commits(tool) -> None:
    """Each pin is the reviewed commit's own source (skipped in a shallow clone)."""
    present = subprocess.run(
        [
            "git",
            "-C",
            str(_ROOT),
            "cat-file",
            "-e",
            f"{tool.REVIEWED_COMMIT}^{{commit}}",
        ],
        capture_output=True,
        check=False,
    )
    if present.returncode != 0:
        pytest.skip("the reviewed commit is not in this clone")
    blobs = {
        path: tool.git_blob(_ROOT, tool.REVIEWED_COMMIT, path)
        for path in tool.SOURCE_PATHS
    }
    assert tool.pin_problems(blobs) == []


def test_definitions_include_decorators_and_refuse_rebinding(tool) -> None:
    source = (
        "import dataclasses\n\n"
        "@dataclasses.dataclass\n"
        "class A:\n    x: int = 1\n\n"
        "B = 2\n"
        "for name in ('a',):\n    pass\n"
        "def f():\n    return 1\n"
        "def f():\n    return 2\n"
    )
    found = tool.top_level_definitions(source)
    assert found["A"] == ["@dataclasses.dataclass\nclass A:\n    x: int = 1\n"]
    assert found["B"] == ["B = 2\n"]
    assert found["for name"] == ["for name in ('a',):\n    pass\n"]
    assert len(found["f"]) == 2
    assert tool.definition_digests(source, ["f", "missing"]) == {
        "f": None,
        "missing": None,
    }


@pytest.mark.parametrize(
    "text",
    [
        '{"a": 1, "a": 2}',
        '{"a": NaN}',
        '{"a": Infinity}',
        '{"a": -Infinity}',
        '{"a": 1e999}',
    ],
)
def test_strict_json_refuses_what_it_cannot_compare(tool, text) -> None:
    with pytest.raises(ValueError):
        tool.strict_loads(text)


def test_same_compares_json_types(tool) -> None:
    assert tool.same({"a": [1, 2.0, None]}, {"a": [1, 2.0, None]})
    assert not tool.same(1, 1.0)
    assert not tool.same(True, 1)
    assert not tool.same({"a": 1}, {"a": 1, "b": 2})
    assert not tool.same([1], [1, 1])


def _probe_fields(probe) -> dict:
    return {
        field: getattr(probe, field)
        for field in (
            "id",
            "name",
            "parameter_changes",
            "budget_measure",
            "binding_inputs",
            "min_abs_effect",
            "reason",
            "issue",
            "effect_direction",
            "period",
            "expected_sign",
            "neutralized_variable",
        )
    }


def test_the_manifest_parse_matches_the_loader(tool) -> None:
    mirrored = tool.parse_probe_manifest(
        tool.strict_loads((_ROOT / tool.MANIFEST_PATH).read_bytes())
    )
    loaded = load_release_input_coverage_manifest().probes
    assert [_probe_fields(probe) for probe in mirrored] == [
        _probe_fields(probe) for probe in loaded
    ]


_VALUES = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-5, max_value=2100),
    st.floats(allow_nan=False, allow_infinity=False, width=64),
    st.sampled_from(["", "x", "2026", "1e9", "positive", "negative", "either"]),
    st.sampled_from(["reform_minus_baseline", "baseline_minus_reform"]),
    st.lists(st.sampled_from(["a", "b"]), max_size=2),
    st.dictionaries(st.sampled_from(["gov.x"]), st.integers(0, 3), max_size=1),
)
_PROBE_FIELDS = (
    "id",
    "name",
    "parameter_changes",
    "budget_measure",
    "binding_inputs",
    "min_abs_effect",
    "reason",
    "issue",
    "effect_direction",
    "period",
    "expected_sign",
    "neutralized_variable",
)
#: A probe the loader accepts, more or less: most draws parse, and the
#: optional fields reach the defaults and the coercions.
_PLAUSIBLE_PROBE = st.fixed_dictionaries(
    {
        "id": st.sampled_from(["p1", "p2"]),
        "name": st.sampled_from(["probe", ""]),
        "parameter_changes": st.sampled_from([{"gov.x": {"2024-01-01": 1}}, {}]),
        "budget_measure": st.sampled_from(["income_tax", "household_net_income"]),
        "binding_inputs": st.sampled_from([["leaf"], ["leaf", "other"], []]),
        "min_abs_effect": st.one_of(
            st.floats(min_value=1e-3, max_value=1e12), st.sampled_from([1, "5e9"])
        ),
        "reason": st.just("why"),
        "issue": st.just("#1"),
    },
    optional={
        "effect_direction": st.sampled_from(
            ["reform_minus_baseline", "baseline_minus_reform"]
        ),
        "period": st.sampled_from([None, 2024, 2026, "2025", 1989]),
        "expected_sign": st.sampled_from(["positive", "negative", "either"]),
        "neutralized_variable": st.sampled_from([None, "leaf", "other"]),
    },
)


@st.composite
def _probes(draw):
    probe = draw(_PLAUSIBLE_PROBE)
    action = draw(st.sampled_from(["keep", "drop", "replace"]))
    if action != "keep":
        field = draw(st.sampled_from(_PROBE_FIELDS))
        probe.pop(field, None)
        if action == "replace":
            probe[field] = draw(_VALUES)
    return probe


_COLUMN = st.dictionaries(
    st.sampled_from(["status", "reason", "issue", "note"]),
    st.one_of(st.sampled_from(["required", "reviewed_exclusion", "", "x"]), _VALUES),
)
_MANIFESTS = st.fixed_dictionaries(
    {
        "reference": st.one_of(
            st.just({"a": "x"}),
            st.dictionaries(st.sampled_from(["a"]), st.text(max_size=2)),
            _VALUES,
        ),
        "columns": st.one_of(
            st.just(
                {
                    "c1": {"status": "required"},
                    "c2": {
                        "status": "reviewed_exclusion",
                        "reason": "r",
                        "issue": "#2",
                    },
                }
            ),
            st.dictionaries(st.sampled_from(["c1", "c2"]), _COLUMN, max_size=2),
            _VALUES,
        ),
    },
    optional={
        "reform_coverage_probes": st.one_of(st.lists(_probes(), max_size=3), _VALUES),
        "schema_version": st.one_of(st.just(1), _VALUES),
    },
)


@pytest.fixture(scope="module")
def manifest_path(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("manifests") / "manifest.json"


@settings(max_examples=400, deadline=None)
@given(payload=_MANIFESTS)
def test_the_manifest_parse_matches_the_loader_on_any_payload(
    tool, manifest_path, payload
) -> None:
    path = manifest_path
    path.write_text(json.dumps(payload))

    def outcome(parse):
        try:
            return "ok", [_probe_fields(probe) for probe in parse()]
        except Exception as error:  # the loader's own exceptions
            return "raises", type(error).__name__

    mirrored = outcome(
        lambda: tool.parse_probe_manifest(tool.strict_loads(path.read_bytes()))
    )
    loaded = outcome(lambda: load_release_input_coverage_manifest(str(path)).probes)
    assert mirrored == loaded


class _Totals:
    def __init__(self, total: float) -> None:
        self._total = total

    def calculate(self, measure, period):
        return self

    def sum(self) -> float:
        return self._total


_FINITE = st.floats(min_value=-1e15, max_value=1e15, allow_nan=False)


@settings(max_examples=300, deadline=None)
@given(
    baseline=_FINITE,
    reform=_FINITE,
    direction=st.sampled_from(["reform_minus_baseline", "baseline_minus_reform"]),
    sign=st.sampled_from(["positive", "negative", "either"]),
    floor=st.floats(min_value=1e-6, max_value=1e14),
)
def test_the_independent_verdict_matches_the_gate(
    tool, baseline, reform, direction, sign, floor
) -> None:
    probe = ReformCoverageProbe(
        id="p",
        name="probe",
        parameter_changes={"gov.x": {"2024-01-01": 1}},
        budget_measure="measure",
        binding_inputs=("leaf",),
        min_abs_effect=floor,
        reason="why",
        issue="#1",
        effect_direction=direction,
        expected_sign=sign,
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(smoke_module, "_build_reform", lambda probe: "reform")
        result = smoke_module.us_reform_coverage_smoke_gate(
            simulate=lambda r: _Totals(baseline if r is None else reform),
            probes=(probe,),
            period=2024,
        )
    row = result.details["results"]["p"]
    effect = tool.gate_effect(baseline, reform, direction)
    assert row["effect"] == effect
    assert row["passed"] is (tool.gate_signed_magnitude(effect, sign) >= floor)


def test_the_plan_and_pass_log_match_the_release_scorer(
    tool, builder, fake_reforms, layout, tmp_path
) -> None:
    plan = builder._record_post_export_baseline_plan(
        builder._reform_coverage_smoke_consumer
    )
    assert (
        builder._post_export_baseline_plan_record(plan)
        == layout.expectation["baseline_plan"]
    )
    scorer = _scorer(builder, _nested_frame(), _EngineLog(), 3, tmp_path)
    passes = []
    score = scorer._score

    def recording(keys, *, label, reform_system=None):
        passes.append(
            {"label": label, "keys": len(keys), "reform": reform_system is not None}
        )
        return score(keys, label=label, reform_system=reform_system)

    scorer._score = recording
    consumer = scorer.open_consumer("reform_coverage_smoke", plan)
    gate = builder._reform_coverage_smoke_consumer(consumer.simulate)
    record = scorer.finish_consumer(consumer)
    assert passes == layout.expectation["passes"]
    assert (
        record["reform_passes"]
        == record["reform_systems"]
        == len(layout.expectation["probes"])
    )
    assert record["baseline_passes"] == layout.expectation["baseline_passes"]
    assert sorted(gate.details["results"]) == sorted(
        p["id"] for p in layout.expectation["probes"]
    )
    for probe in layout.expectation["probes"]:
        result = gate.details["results"][probe["id"]]
        for field in (
            "name",
            "period",
            "budget_measure",
            "expected_sign",
            "min_abs_effect",
            "binding_inputs",
            "issue",
        ):
            assert result[field] == probe[field], (probe["id"], field)
        assert set(result) == tool.GATE_RESULT_FIELDS


@settings(max_examples=200, deadline=None)
@given(
    keys=st.lists(
        st.tuples(
            st.sampled_from(["a", "b", "c"]),
            st.sampled_from([2024, 2025, 2026]),
            st.none(),
        ),
        max_size=12,
    )
)
def test_the_plan_order_matches_the_release_tool(tool, builder, keys) -> None:
    assert tool.ascending_period_plan(keys) == list(
        builder._ascending_period_plan(keys)
    )


@settings(max_examples=300, deadline=None)
@given(
    n=st.integers(min_value=0, max_value=5_000),
    size=st.one_of(st.none(), st.integers(min_value=-2, max_value=6_000)),
)
def test_batching_matches_the_release_tool(tool, builder, n, size) -> None:
    batches = list(builder._household_position_batches(n, size))
    assert tool.batching(n, size) == (
        len(batches),
        max((len(b) for b in batches), default=0),
    )


def _function(source: str, name: str) -> ast.FunctionDef:
    node = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    body = node.body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
    ):
        body = body[1:]
    return ast.dump(ast.Module(body=body, type_ignores=[]))


def test_the_runtime_script_copies_its_sources(tool) -> None:
    probe = (_ROOT / tool.PROBE_PATH).read_text()
    release = (_ROOT / tool.BUILDER_PATH).read_text()
    script = tool.RUNTIME_SCRIPT
    assert _function(script, "installed_distributions") == _function(
        probe, "_installed_distributions"
    )
    assert _function(script, "_local_workspace_package_version") == _function(
        release, "_local_workspace_package_version"
    )
    assert _function(script, "version") == _function(
        release, "_package_or_workspace_version"
    )


def test_the_runtime_script_agrees_with_the_release_tool_and_probe(
    tool, builder, probe_tool, monkeypatch
) -> None:
    monkeypatch.chdir(_ROOT)
    runtime = tool._runtime(sys.executable, _ROOT / "tools")
    assert runtime["runtime_versions"] == builder._runtime_versions()
    assert runtime["installed_distributions"] == probe_tool._installed_distributions()
    assert runtime["version"] == sys.version.split()[0]


def test_the_mirrored_constants_match_the_release_tool(
    tool, builder, probe_tool
) -> None:
    assert tool.PERIOD == builder.PERIOD
    assert tool.SCORING_METHOD == builder.POST_EXPORT_SCORING_METHOD
    assert (
        builder.US_RELEASE_GATE_EVIDENCE_FILES["reform_coverage_smoke"]
        == "reform_coverage_smoke.json"
    )
    assert tuple(builder._runtime_versions()) == ("python", *tool.RUNTIME_PACKAGES)
    loop = next(
        node
        for node in ast.parse((_ROOT / tool.BUILDER_PATH).read_text()).body
        if isinstance(node, ast.For)
        and getattr(node.target, "id", None) == "_thread_pool_variable"
    )
    assert tuple(ast.literal_eval(loop.iter)) == tool.THREAD_VARIABLES
    assert builder._THREAD_POOL_DEFAULT == str(
        min(os.cpu_count() or 1, tool.THREAD_POOL_CEILING)
    )
    assert probe_tool.PROBE_REPORT_SCHEMA_VERSION == tool.PROBE_SCHEMA_VERSION
    assert probe_tool.MINIMUM_BATCHES == tool.PROBE_MINIMUM_BATCHES
    assert probe_tool.REPORT_FILENAME == tool.PROBE_REPORT
    assert probe_tool.PASSES_FILENAME == tool.PROBE_PASSES
    assert tuple(probe_tool.REFERENCE_FILES) == tool.PROBE_REFERENCE_KEYS
    # The probe writes its smoke gate under the release's evidence name.
    assert probe_tool.REFERENCE_FILES["smoke"] == tool.PROBE_GATE
    probe_source = (_ROOT / tool.PROBE_PATH).read_text()
    assert f'self.out_dir / "{tool.PROBE_GATE}"' in probe_source
    assert probe_tool.classify_probe(
        signed_magnitude=0.0,
        floor=1.0,
        standard_error=None,
        take_all=False,
        drawn_effect_households=None,
        effective_households=None,
        census=True,
    ) == (probe_tool.AUTHORITATIVE, tool.CENSUS_REASON)
    assert probe_tool.AUTHORITATIVE == "authoritative"
    assert tool.SMOKE_CONSUMER in probe_tool.STAGES


def test_the_probe_mirror_has_the_loaders_fields(tool) -> None:
    def shape(cls):
        return [(field.name, field.default) for field in dataclasses.fields(cls)]

    assert shape(tool.Probe) == shape(ReformCoverageProbe)


def test_expect_writes_the_expectation_and_refuses_a_short_id(
    tool, layout, monkeypatch, tmp_path, capsys
) -> None:
    monkeypatch.setattr(tool, "MIRRORED_DEFINITIONS", layout.pins[0])
    monkeypatch.setattr(tool, "PINNED_FILES", layout.pins[1])
    # The documented first step writes into a launch directory that does not
    # exist yet; a second write refuses rather than replace the first.
    out = tmp_path / "launch" / "expectation.json"
    argv = ["expect", "--commit", layout.commit, "--repo", str(layout.repo)]
    assert tool.main([*argv, "--out", str(out)]) == 0
    assert tool.strict_loads(out.read_bytes()) == layout.expectation
    assert out.read_text() == tool.strict_dumps(layout.expectation)
    capsys.readouterr()
    assert tool.main([*argv, "--out", str(out)]) == 1
    assert "already exists" in capsys.readouterr().err
    assert out.read_text() == tool.strict_dumps(layout.expectation)
    short = ["expect", "--commit", layout.commit[:12], "--repo", str(layout.repo)]
    assert tool.main(short) == 1
    assert "is not a full commit id" in capsys.readouterr().err
