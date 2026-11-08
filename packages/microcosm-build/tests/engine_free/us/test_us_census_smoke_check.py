"""``tools/check_us_census_smoke.py``: accepting a separate census smoke.

The check accepts a separate full-population reform-coverage smoke for a
release whose build skipped its own, and only then writes the attachments a
tag-only publish carries. These tests build one faithful set of inputs (a
throwaway repository holding the release tool's source, launch records taken
by the tool itself, a release directory and the probe's output), confirm the
check accepts it, and confirm it refuses each departure without writing
anything. One test runs the probe itself, through its command line on an
engine-free stand-in for policyengine-us, and confirms the check accepts what
it writes. Differential tests hold each mirror in the check (the probe
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
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from microcosm.build.us_runtime import reform_coverage_smoke as smoke_module
from microcosm.build.us_runtime import release_input_coverage as coverage_module
from microcosm.build.us_runtime.release_input_coverage import (
    ReformCoverageProbe,
    load_release_input_coverage_manifest,
)
from test_support.git_repositories import NO_BACKGROUND_MAINTENANCE
from test_support.microcosm_build.us_export_subsample import (  # noqa: F401
    load_table_h5,
    probe_tool,
    synthetic_export_frame,
    write_table_h5,
)
from test_support.microcosm_build.us_post_export_probe import (
    fixture_engine,
    fixture_engine_probes,
    float32_engine,
)
from test_support.microcosm_build.us_post_export_scoring import (  # noqa: F401
    _EngineLog,
    _native_entity,
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


def _source_repository(
    tool, root: Path, *, manifest: bytes | None = None
) -> tuple[Path, str]:
    """A committed repository holding this checkout's release-tool source and
    ``.gitignore`` (so the caches a run writes leave the tree clean), with
    ``manifest`` in place of the probe manifest when given."""
    repo = root / "repo"
    for path in (*tool.SOURCE_PATHS, "uv.lock", "pyproject.toml", ".gitignore"):
        target = repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((_ROOT / path).read_bytes())
    if manifest is not None:
        (repo / tool.MANIFEST_PATH).write_bytes(manifest)
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


def _record(
    tool,
    *,
    repo: Path,
    launch_dir: Path,
    dataset: Path,
    stage: str,
    phase: str,
    at: datetime,
    out_dir: Path | None = None,
    exit_code: int | None = None,
    command: list[str] | None = None,
    runtime_environment: dict | None = None,
) -> dict:
    """A launch record taken by the tool itself, stamped ``at``: a start
    record names the command it would become, and an end record measures the
    runtime with its start's environment."""
    snapshot = tool.snapshot(
        stage=stage,
        phase=phase,
        run_id="test-run",
        repo=repo,
        python=sys.executable,
        dataset=dataset,
        out_dir=out_dir,
        exit_code=exit_code,
        command=command,
        runtime_environment=runtime_environment,
        now=lambda: at,
    )
    path = launch_dir / tool.LAUNCH_FILES[(stage, phase)]
    path.write_text(tool.strict_dumps(snapshot))
    return snapshot


def _build_command(out: Path) -> list[str]:
    """The build's launch, as a start record execs it."""
    return [
        sys.executable,
        "-B",
        "tools/build_us_fiscal_refresh_release.py",
        "--out",
        str(out),
    ]


def _smoke_command(dataset: Path, probe_dir: Path, batch_size: int) -> list[str]:
    """The documented census smoke, as a start record execs it."""
    return [
        sys.executable,
        "-B",
        "tools/probe_us_post_export.py",
        "--census",
        "--stages",
        "reform_coverage_smoke",
        "--batch-size",
        str(batch_size),
        "--export",
        str(dataset),
        "--out",
        str(probe_dir),
    ]


def _clear_git_environment(patch: pytest.MonkeyPatch) -> None:
    """Run without the caller's ``GIT_*`` variables (a git hook sets some),
    which a start record would refuse."""
    for name in [name for name in os.environ if name.startswith("GIT_")]:
        patch.delenv(name)


#: What a skipped-smoke build still scores after export.
_DEMOGRAPHICS_CONSUMER = {
    "baseline_plan": {
        "keys": [{"variable": "age", "period": 2024, "map_to": None}],
        "period_order": [2024],
    },
    "baseline_passes": 1,
    "reform_passes": 0,
    "reform_systems": 0,
}
_EVIDENCE = {
    "esi_premiums": "bound",
    "input_coverage": "bound",
    "input_mass_parity": "bound",
    "qrf_tail_concentration": "bound",
    "reform_coverage_smoke": "skipped",
    "stored_inputs": "bound",
}


def _write_manifests(
    release_dir: Path,
    *,
    release_id: str,
    commit: str,
    runtime: dict,
    scoring: dict,
    created_at: str,
    calibration_runtime: dict | None = None,
) -> None:
    """The build and release manifests of a skipped-smoke build."""
    dataset_sha = scoring["dataset_sha256"]
    if calibration_runtime is None:
        calibration_runtime = {
            "torch": "2.12.0",
            "torch_num_threads": int(_THREADS),
            "omp_num_threads": _THREADS,
        }
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
        "runtime": runtime,
        "calibration_runtime": calibration_runtime,
        "gate_evidence": _EVIDENCE,
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
            "calibration_runtime": calibration_runtime,
            "gate_evidence": _EVIDENCE,
            "post_export_scoring": scoring,
        },
    }
    (release_dir / "build_manifest.json").write_text(
        json.dumps(build_manifest, indent=1, allow_nan=False)
    )
    (release_dir / "release_manifest.json").write_text(
        json.dumps(release_manifest, indent=2, allow_nan=False)
    )


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
        _clear_git_environment(patch)

        def record(stage, phase, **extra):
            return _record(
                tool,
                repo=repo,
                launch_dir=launch_dir,
                dataset=dataset,
                stage=stage,
                phase=phase,
                at=_TIMES[f"{stage}-{phase}"],
                **extra,
            )

        build_start = record("build", "start", out_dir=out, command=_build_command(out))
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
            "consumers": {"demographics": _DEMOGRAPHICS_CONSUMER},
        }
        _write_manifests(
            release_dir,
            release_id=release_id,
            commit=commit,
            runtime=runtime["runtime_versions"],
            scoring=scoring,
            created_at=_TIMES["manifests"].isoformat(),
        )
        record(
            "build",
            "end",
            exit_code=0,
            runtime_environment=build_start["environment"],
        )
        smoke_start = record(
            "smoke",
            "start",
            out_dir=probe_dir,
            command=_smoke_command(dataset, probe_dir, _BATCH),
        )
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
        record(
            "smoke",
            "end",
            exit_code=0,
            runtime_environment=smoke_start["environment"],
        )
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
        }
        # The receipt names the flags it printed; nothing reads it to enforce
        # them.
        assert receipt["publication"]["printed_flags"] == ["--no-latest", "--tag-only"]
        assert receipt["publication"]["publisher_checks_these_files"] is False
        assert receipt["expectation"] == layout.expectation
        assert (
            receipt["dataset"]["sha256"] == hashlib.sha256(_DATASET_BYTES).hexdigest()
        )
        assert receipt["smoke"]["probes"] == len(layout.expectation["probes"])
        assert receipt["smoke"]["diagnostic_analysis"]["decomposed"] == len(
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
        assert "uploads this note and the receipt but checks nothing in them" in note
        assert "with the build's runtime" not in note
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


def _fixture_manifest(tool) -> bytes:
    """The release's probe manifest with the probe tests' three engine probes
    (:func:`fixture_engine_probes`) in place of its own."""
    payload = json.loads((_ROOT / tool.MANIFEST_PATH).read_text())
    payload["reform_coverage_probes"] = [
        {
            **_probe_fields(probe),
            "parameter_changes": dict(probe.parameter_changes),
            "binding_inputs": list(probe.binding_inputs),
        }
        for probe in fixture_engine_probes()
    ]
    return (json.dumps(payload, indent=2) + "\n").encode()


def test_the_probes_own_census_run_is_accepted(
    tool, fake_reforms, tmp_path, monkeypatch
) -> None:
    """The probe's own command line, run with the documented smoke arguments
    on a written H5, writes files the check accepts.

    A throwaway repository holds this checkout's source, with the probe tests'
    three engine probes as its shipped probe set. The probe loads from it and
    loads the release tool and sampler from it with its own loader, as its
    command line does, so its report hashes them "at load". It runs the
    release's own smoke consumer on the shipped set. Only the engine boundary
    differs from a real run. The post-export scoring tests' float32 fake
    engine, a pandas reader of the H5 and their variable-entity map stand in
    for policyengine-us, and the in-process package reads the throwaway's
    manifest. The build's manifests carry the release tool's own runtime
    record, batching and SPM selection. The tool itself takes the four launch
    records, the build's at past times and the smoke's at the real times
    around the run.
    """
    root = tmp_path.resolve()
    manifest = _fixture_manifest(tool)
    repo, commit = _source_repository(tool, root, manifest=manifest)
    pins = _current_pins(tool)
    monkeypatch.setattr(tool, "MIRRORED_DEFINITIONS", pins[0])
    monkeypatch.setattr(tool, "PINNED_FILES", pins[1])
    out = root / "build-out"
    release_id = f"microcosm-us-2024-{commit[:12]}-20261001T103000Z"
    release_dir = out / "releases" / release_id
    artifact_root = out / "artifacts"
    dataset = artifact_root / _DATASET_FILE
    probe_dir = root / "smoke-out"
    launch_dir = root / "launch"
    launch_dir.mkdir()
    monkeypatch.chdir(repo)
    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join(
            str(repo / "packages" / package / "src")
            for package, _ in _NAMESPACE_PACKAGES
        ),
    )
    for name in tool.THREAD_VARIABLES:
        monkeypatch.setenv(name, _THREADS)
    _clear_git_environment(monkeypatch)

    def record(stage: str, phase: str, at: datetime, **extra) -> dict:
        return _record(
            tool,
            repo=repo,
            launch_dir=launch_dir,
            dataset=dataset,
            stage=stage,
            phase=phase,
            at=at,
            **extra,
        )

    # Load the tools as the probe's command line loads them. Other tests may
    # have registered the release tool or sampler, and the probe's loader
    # would reuse those, so they leave sys.modules for this test.
    probe_name = "probe_us_post_export_census_check"
    spec = importlib.util.spec_from_file_location(probe_name, repo / tool.PROBE_PATH)
    assert spec is not None and spec.loader is not None
    probe = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, probe_name, probe)
    for name in ("build_us_fiscal_refresh_release", "sample_us_export_households"):
        monkeypatch.setitem(sys.modules, name, None)
        monkeypatch.delitem(sys.modules, name)
    spec.loader.exec_module(probe)
    release_tool = probe.load_builder()

    # The engine boundary.
    frame = synthetic_export_frame(48, seed=5, rare_households=(3, 30))
    monkeypatch.setattr(release_tool, "_collect_family_garbage", lambda: None)
    monkeypatch.setattr(
        release_tool, "installed_us_engine", lambda: fixture_engine(frame)
    )
    shipped_text = coverage_module._resource_text
    monkeypatch.setattr(
        coverage_module,
        "_resource_text",
        lambda resource: (
            manifest.decode()
            if resource == coverage_module.US_RELEASE_INPUT_COVERAGE_RESOURCE
            else shipped_text(resource)
        ),
    )
    # This environment lacks the engine, and the probe records an absent
    # package as None where the release tool records "not-installed". A real
    # run has both installed.
    package_versions = probe._package_versions
    monkeypatch.setattr(
        probe,
        "_package_versions",
        lambda: {
            name: "not-installed" if version is None else version
            for name, version in package_versions().items()
        },
    )
    export_probe = probe.ExportProbe

    def engine_free_probe(export_path, out_dir, **options):
        return export_probe(
            export_path,
            out_dir,
            scorer_options={
                "microsimulation_cls": float32_engine(_EngineLog()),
                "dataset_from_frame": lambda batch_frame: batch_frame,
            },
            load_frame=load_table_h5,
            measure_entity=_native_entity,
            **options,
        )

    monkeypatch.setattr(probe, "ExportProbe", engine_free_probe)

    # The build, stamped in the past.
    started = datetime.now(UTC)
    build_start = record(
        "build",
        "start",
        started - timedelta(hours=3),
        out_dir=out,
        command=_build_command(out),
    )
    artifact_root.mkdir(parents=True)
    release_dir.mkdir(parents=True)
    write_table_h5(frame, dataset)
    n_households = int(frame.n("household"))
    batches = list(release_tool._household_position_batches(n_households, 16))
    _write_manifests(
        release_dir,
        release_id=release_id,
        commit=commit,
        runtime=release_tool._runtime_versions(),
        scoring={
            "method": release_tool.POST_EXPORT_SCORING_METHOD,
            "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
            "maximum_batch_size": 16,
            "n_households": n_households,
            "n_batches": len(batches),
            "max_batch_households": max(len(batch) for batch in batches),
            "spm": dict(release_tool.US_RELEASE_SPM_SELECTION),
            "consumers": {"demographics": _DEMOGRAPHICS_CONSUMER},
        },
        created_at=(started - timedelta(hours=2)).isoformat(),
    )
    record(
        "build",
        "end",
        started - timedelta(hours=1),
        exit_code=0,
        runtime_environment=build_start["environment"],
    )

    # The smoke, with the documented arguments: the start record names the
    # command it would exec, and the probe's own main runs those arguments.
    command = _smoke_command(dataset, probe_dir, 16)
    smoke_start = record(
        "smoke", "start", datetime.now(UTC), out_dir=probe_dir, command=command
    )
    assert probe.main(command[3:]) == 0
    record(
        "smoke",
        "end",
        datetime.now(UTC),
        exit_code=0,
        runtime_environment=smoke_start["environment"],
    )

    accepted = tool.check(
        tool.Inputs(
            release_dir=release_dir,
            artifact_root=artifact_root,
            probe_dir=probe_dir,
            launch_dir=launch_dir,
            repo=repo,
            authorization="the test ruling",
            dedicated_worktree=True,
            probe_count=len(fixture_engine_probes()),
        )
    )
    report = json.loads((probe_dir / tool.PROBE_REPORT).read_text())
    rows = report["stages"]["reform_coverage_smoke"]["probes"]
    assert [row["probe"] for row in rows] == [
        shipped.id for shipped in fixture_engine_probes()
    ]
    assert report["tool_source"]["tools"][tool.BUILDER_PATH]["hashed"] == "at load"
    assert len(batches) == 3
    assert accepted.smoke_summary["probes"] == 3
    assert accepted.smoke_summary["diagnostic_analysis"] == {
        "decomposed": 3,
        "other": {},
    }


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
        "a reference plan comparison",
        "report",
        lambda lo: _set("reference_plan_comparison", value={}),
        "the report compared plans against a reference release",
    ),
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
        "the smoke's PYTHONHASHSEED",
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
        "smoke-start interpreter",
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
    # Each start record's command is the stage's documented launch.
    (
        "build command script",
        "build-start",
        lambda lo: _set("command", 2, value="tools/probe_us_post_export.py"),
        "build-start: the command must run tools/build_us_fiscal_refresh_release.py",
    ),
    (
        "build command interpreter",
        "build-start",
        lambda lo: _set("command", 0, value="/usr/bin/python3"),
        "build-start: the command runs '/usr/bin/python3', not the --python",
    ),
    (
        "build interpreter option",
        "build-start",
        lambda lo: _set("command", 1, value="-X"),
        "build-start: the command must run tools/build_us_fiscal_refresh_release.py",
    ),
    (
        "smoke sample flag",
        "smoke-start",
        lambda lo: (
            lambda record: record["command"].extend(
                ["--sample-receipt", "sample_receipt.json"]
            )
        ),
        "a census smoke passes only",
    ),
    (
        "smoke stages",
        "smoke-start",
        lambda lo: _set("command", 5, value="reform_coverage_smoke,demographics"),
        "the smoke runs --stages 'reform_coverage_smoke,demographics'",
    ),
    (
        "smoke without census",
        "smoke-start",
        lambda lo: lambda record: record["command"].remove("--census"),
        "the smoke must pass each of ['--census'] once",
    ),
    (
        "smoke repeated flag",
        "smoke-start",
        lambda lo: (
            lambda record: record["command"].extend(
                ["--stages", "reform_coverage_smoke"]
            )
        ),
        "the smoke must pass each of ['--stages'] once",
    ),
    (
        "smoke batch size",
        "smoke-start",
        lambda lo: _set("command", 7, value="1000"),
        "smoke-start --batch-size (the build's maximum_batch_size)",
    ),
    (
        "smoke batch spelling",
        "smoke-start",
        lambda lo: _set("command", 7, value="2e3"),
        "the smoke's --batch-size '2e3' is not a positive integer",
    ),
    (
        "smoke export",
        "smoke-start",
        lambda lo: _set("command", 9, value="/elsewhere/microcosm_us_2024.h5"),
        "not the recorded dataset",
    ),
    (
        "smoke out",
        "smoke-start",
        lambda lo: _set("command", 11, value="/elsewhere/smoke-out"),
        "not the recorded output directory",
    ),
    (
        "start without a command",
        "build-start",
        lambda lo: _set("command", value=None),
        "build-start records no command, interpreter or output directory",
    ),
    # A start record describes the process it became; an end record repeats
    # the start's snapshot.
    (
        "start exit code",
        "build-start",
        lambda lo: _set("exit_code", value=0),
        "build-start exit_code",
    ),
    (
        "start measured elsewhere",
        "build-start",
        lambda lo: _set("runtime_environment", "PYTHONHASHSEED", value="1"),
        "build-start runtime_environment (its own environment)",
    ),
    (
        "end measured elsewhere",
        "smoke-end",
        lambda lo: _set("runtime_environment", "OMP_NUM_THREADS", value="9"),
        "smoke-end runtime_environment (smoke-start's environment)",
    ),
    (
        "end command",
        "build-end",
        lambda lo: _set("command", value=["x"]),
        "build-end command",
    ),
    (
        "end output directory",
        "smoke-end",
        lambda lo: _set("out_dir", value={"path": "/x", "state": "absent"}),
        "smoke-end out_dir",
    ),
    (
        "end interpreter",
        "build-end",
        lambda lo: _set("python", "resolved", value="/usr/bin/python3"),
        "build-end interpreter (build-start's)",
    ),
    (
        "git redirect",
        "smoke-start",
        lambda lo: _set("git_environment", value=["GIT_DIR"]),
        "smoke-start git-redirecting variables",
    ),
    (
        "git environment missing",
        "build-start",
        lambda lo: lambda record: record.pop("git_environment"),
        "build-start lists no git environment",
    ),
    (
        "status digest",
        "build-end",
        lambda lo: _set("repo", "status_sha256", value="0" * 64),
        "build-end git status digest",
    ),
    (
        "status entries",
        "smoke-start",
        lambda lo: _set("repo", "status_entries", value=1),
        "smoke-start git status entries",
    ),
    (
        "record tool",
        "smoke-end",
        lambda lo: _set("tool", value="tools/other.py"),
        "smoke-end tool",
    ),
    (
        "another checkout's record",
        "build-start",
        lambda lo: _set("repo", "toplevel", value="/elsewhere/repo"),
        "build-start repository",
    ),
    (
        "another working directory",
        "smoke-end",
        lambda lo: _set("cwd", value="/elsewhere/repo"),
        "smoke-end working directory",
    ),
    (
        "no uv.lock",
        "build-start",
        lambda lo: _set("files", "uv.lock", value=None),
        "build-start found no uv.lock",
    ),
    (
        "cpu count",
        "build-start",
        lambda lo: _set("cpu_count", value="18"),
        "the smoke's thread-pool settings",
    ),
    (
        "smoke runtime",
        "smoke-start",
        lambda lo: _set("runtime", "version", value="3.0.0"),
        "smoke-start runtime is",
    ),
    (
        "smoke dataset elsewhere",
        "smoke-start",
        lambda lo: _set("dataset", "path", value="/elsewhere/microcosm_us_2024.h5"),
        "smoke-start dataset path (build-end's)",
    ),
    (
        "build dataset existed",
        "build-start",
        lambda lo: _set("dataset", "exists", value=True),
        "build-start dataset exists",
    ),
    (
        "build dataset elsewhere",
        "build-end",
        lambda lo: _set("dataset", "path", value="/elsewhere/microcosm_us_2024.h5"),
        "build-end dataset path (the release's national artifact)",
    ),
    (
        "build dataset digest",
        "build-end",
        lambda lo: _set("dataset", "sha256", value=None),
        "build-end recorded no dataset digest",
    ),
    (
        "microcosm from the worktree's .venv",
        "build-start",
        lambda lo: _set(
            "runtime",
            "import_origins",
            "microcosm.build",
            value={
                "origin": str(
                    lo.repo
                    / ".venv/lib/python3.14/site-packages/microcosm/build/__init__.py"
                ),
                "locations": [
                    str(lo.repo / ".venv/lib/python3.14/site-packages/microcosm/build")
                ],
            },
        ),
        "microcosm.build imports from outside",
    ),
    # The calibration's thread record.
    (
        "calibration threads",
        "build",
        lambda lo: (
            lambda payload: payload["calibration_runtime"].update(omp_num_threads="16")
        ),
        "the build's calibration_runtime.omp_num_threads",
    ),
    (
        "release calibration runtime",
        "release",
        lambda lo: _set("build", "calibration_runtime", "torch_num_threads", value=16),
        "release build.calibration_runtime",
    ),
    # Each message below fails only beside others when one field changes, so
    # a case of its own notices the check's removal.
    (
        "smoke run id",
        "smoke-start",
        lambda lo: _set("run_id", value="r0"),
        "smoke-start run_id",
    ),
    (
        "smoke tool digest",
        "smoke-start",
        lambda lo: _set("tool_sha256", value="0" * 64),
        "smoke-start tool_sha256",
    ),
    (
        "smoke lockfile",
        "smoke-start",
        lambda lo: _set("files", "uv.lock", value="0" * 64),
        "smoke-start files",
    ),
    (
        "smoke environment",
        "smoke-start",
        lambda lo: _set("environment", value=None),
        "a start record lists no environment",
    ),
    (
        "build-end digest, against the smoke",
        "build-end",
        lambda lo: _set("dataset", "sha256", value="0" * 64),
        "smoke-start dataset (build-end's)",
    ),
    (
        "build-end digest, against the H5",
        "build-end",
        lambda lo: _set("dataset", "sha256", value="0" * 64),
        "build-end dataset is",
    ),
    (
        "build id",
        "build",
        lambda lo: _set("build_id", value="another-build"),
        "the release directory's name",
    ),
    (
        "absolute artifact path",
        "release",
        lambda lo: _set("artifacts", _DATASET_KEY, "path", value="/data/national.h5"),
        "the national artifact's path is not a relative path",
    ),
    (
        "build gate evidence",
        "build",
        lambda lo: _set("gate_evidence", "reform_coverage_smoke", value="passed"),
        "build gate_evidence.reform_coverage_smoke",
    ),
    (
        "no post-export scoring",
        "build",
        lambda lo: _set("post_export_scoring", value=None),
        "the build recorded no post_export_scoring",
    ),
    (
        "scoring digest",
        "build",
        lambda lo: _set("post_export_scoring", "dataset_sha256", value="0" * 64),
        "build scoring dataset_sha256",
    ),
    (
        "no scoring consumers",
        "build",
        lambda lo: _set("post_export_scoring", "consumers", value=None),
        "build scoring lists no consumers",
    ),
    (
        "scored the smoke",
        "build",
        lambda lo: _set(
            "post_export_scoring", "consumers", "reform_coverage_smoke", value={}
        ),
        "the build scored reform_coverage_smoke, which a skipped-smoke build does not",
    ),
    (
        "scored validation",
        "build",
        lambda lo: _set(
            "post_export_scoring", "consumers", "reform_validation", value={}
        ),
        "the build scored reform_validation, which a skipped-smoke build does not",
    ),
    (
        "no households",
        "build",
        lambda lo: _set("post_export_scoring", "n_households", value=0),
        "build scoring n_households is not a positive integer",
    ),
    (
        "scoring batch count",
        "build",
        lambda lo: _set("post_export_scoring", "n_batches", value=999),
        "build scoring n_batches",
    ),
    (
        "largest batch",
        "build",
        lambda lo: _set("post_export_scoring", "max_batch_households", value=1),
        "build scoring max_batch_households",
    ),
    (
        "one batch",
        "build",
        lambda lo: _set("post_export_scoring", "maximum_batch_size", value=10**9),
        "the build scored 1 batch(es); the probe needs 3 or more",
    ),
    (
        "no gate results",
        "gate",
        lambda lo: _set("reform_coverage_smoke", "details", "results", value=None),
        "the gate recorded no results",
    ),
    (
        "integer totals",
        "gate",
        lambda lo: _set(
            "reform_coverage_smoke",
            "details",
            "results",
            _first_probe(lo),
            "baseline_total",
            value=1,
        ),
        "totals are not finite floats",
    ),
    (
        "no report stages",
        "report",
        lambda lo: _set("stages", value=None),
        "the report has no stages",
    ),
    (
        "unreadable distributions",
        "report",
        lambda lo: _set(
            "tool_source", "installed_distributions", "unreadable", value=1
        ),
        "report installed_distributions.unreadable",
    ),
    (
        "no build runtime",
        "build",
        lambda lo: _set("runtime", value=None),
        "the build manifest records no runtime",
    ),
    (
        "report python",
        "report",
        lambda lo: _set("host", "python", value="3.0.0"),
        "report host.python",
    ),
    (
        "report python, against the launch",
        "report",
        lambda lo: _set("host", "python", value="3.0.0"),
        "launch python",
    ),
    (
        "no run id",
        "build-start",
        lambda lo: _set("run_id", value=""),
        "build-start names no run_id",
    ),
    # Each of these guards stops the check when it fails. Disabled, a guard
    # lets the check run on and refuse on a later line, so a test that only
    # expects a refusal still passes: each case asserts the guard's message.
    (
        "no build id",
        "build",
        lambda lo: _set("build_id", value=None),
        "build_manifest.json names no build_id",
    ),
    (
        "no build commit",
        "build",
        lambda lo: _set("code", "git_commit", value=None),
        "build code.git_commit is missing",
    ),
    (
        "no smoke rows",
        "report",
        lambda lo: _set("stages", "reform_coverage_smoke", "probes", value=None),
        "the smoke stage lists no probes",
    ),
    (
        "a missing gate result",
        "gate",
        lambda lo: _deleted(
            "reform_coverage_smoke", "details", "results", _first_probe(lo)
        ),
        "is missing",
    ),
    (
        "a pass that is not an object",
        "passes",
        lambda lo: _set(0, value="x"),
        "pass 1 is not an object",
    ),
    (
        "a smoke start with no dataset",
        "smoke-start",
        lambda lo: _set("dataset", "path", value=None),
        "smoke-start: the smoke's start record names no dataset for --export",
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


#: The nine inputs, by the names :meth:`Layout.file` takes.
_INPUT_NAMES = (
    "build",
    "release",
    "report",
    "gate",
    "passes",
    "build-start",
    "build-end",
    "smoke-start",
    "smoke-end",
)


@contextmanager
def _replaced(path: Path, content: bytes | None) -> Iterator[None]:
    """Replace the file at ``path`` (``None`` removes it); restore it afterwards."""
    original = path.read_bytes()
    try:
        if content is None:
            path.unlink()
        else:
            path.write_bytes(content)
        yield
    finally:
        path.write_bytes(original)


@pytest.mark.parametrize("name", _INPUT_NAMES)
def test_a_missing_input_is_refused(quick, layout, name) -> None:
    file = layout.file(name)
    with _replaced(file, None):
        message = _refusal(quick, layout)
    assert message == f"{file} is unreadable (FileNotFoundError)"


@pytest.mark.parametrize("content", [b'"x"', b"5", b"null", b"[]"])
@pytest.mark.parametrize("name", _INPUT_NAMES)
def test_an_input_that_is_not_an_object_is_refused(
    quick, layout, name, content
) -> None:
    """Valid JSON of the wrong shape refuses, and never crashes the check."""
    with _replaced(layout.file(name), content + b"\n"):
        _refusal(quick, layout)


def test_a_pass_log_cut_mid_line_is_refused(quick, layout) -> None:
    file = layout.file("passes")
    original = file.read_bytes()
    with _replaced(file, original.removesuffix(b"\n")):
        message = _refusal(quick, layout)
    assert message == f"{file} ends mid-line"


def test_a_pass_log_line_that_is_not_strict_json_is_refused(quick, layout) -> None:
    file = layout.file("passes")
    original = file.read_bytes()
    with _replaced(file, original + b'{"wall_seconds": NaN}\n'):
        message = _refusal(quick, layout)
    number = original.count(b"\n") + 1
    assert message.startswith(f"{file} line {number} is not strict JSON"), message


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
# Every field: one changed value anywhere in the nine inputs is refused
# ---------------------------------------------------------------------------


def _leaves(value: Any, path: tuple = ()) -> list[tuple]:
    if isinstance(value, dict) and value:
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


def _changed(value: Any) -> Any:
    """Another value: of the same type for a scalar, never equal to it."""
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + 0.5 * (abs(value) + 1.0)
    if isinstance(value, str):
        return value + "x"
    if value is None:
        return "x"
    if isinstance(value, list):
        return ["x"]
    if isinstance(value, dict):
        return {"x": 1}
    raise TypeError(f"not a JSON value: {value!r}")


#: The probe row fields that make up the per-household analysis.
_ANALYSIS_FIELDS = (
    "measure_entity",
    "standard_error",
    "decomposition",
    "engine_weight_scale",
    "drawn_effect_households",
    "effective_variance_households",
    "certainty_effect_households",
    "noncarrier_effect",
    "noncarrier_effect_households",
    "pool_carrier_households",
    "expected_sampled_carriers",
    "take_all",
    "sampled_carrier_households",
)
_END_RECORD_UNJUDGED = (
    (
        ("environment", "**"),
        "the environment of the process that recorded the end, after the stage "
        "exited; the end measures the runtime with the start's",
    ),
    (
        ("python", "requested"),
        "the end's own spelling of the interpreter; the check compares the "
        "resolved path",
    ),
    (
        ("git_environment",),
        "an end record's git environment moves nothing: the stage has exited",
    ),
    (
        ("cpu_count",),
        "only the build's start picks the thread defaults",
    ),
    (("dataset", "exists"), "the dataset's digest decides"),
    (("dataset", "bytes"), "the dataset's digest decides"),
)
#: What the check does not judge, by input, with the reason: in a pattern,
#: ``*`` stands for any one key or index, and a final ``**`` for any rest.
_UNJUDGED: dict[str, tuple[tuple[tuple, str], ...]] = {
    "gate": (),
    "report": (
        (("host", "platform"), "the probe host's platform string"),
        (
            ("rules", "**"),
            "the sample rules: a census row takes the census authority before "
            "they apply",
        ),
        (
            ("target_surface", "**"),
            "read only by the source-coverage stage, which a smoke-only run "
            "skips; the command passes no --target-surface",
        ),
        (
            ("not_reproduced", "**"),
            "the probe's constant list of release steps it does not run",
        ),
        (("process_peak_rss_gib",), "memory diagnostics"),
        (("process_peak_rss_source",), "memory diagnostics"),
        (("rss_source",), "memory diagnostics"),
        (("stages", "*", "timing", "**"), "per-stage timing"),
        (
            ("stages", "load", "rows", "person"),
            "another entity's row count; the household count fixes the batching",
        ),
        (
            ("stages", "design", "household_weight_total"),
            "a weighted total the census design reports and nothing reads",
        ),
        *(
            (
                ("stages", "reform_coverage_smoke", "probes", "*", field),
                "the per-household analysis, a diagnostic",
            )
            for field in _ANALYSIS_FIELDS
        ),
    ),
    "passes": tuple(
        (("*", field), "pass timing and memory")
        for field in ("wall_seconds", "cpu_seconds", "rss_before_gib", "peak_rss_gib")
    ),
    "build": (
        (("build_sha",), "a display abbreviation; code.git_commit names the commit"),
        (
            ("staging",),
            "where the build staged files; the check reads the release directory",
        ),
        (("dataset", "default"), "the calibration summary"),
    ),
    "release": (
        (
            ("schema_version",),
            "the publisher's layout version; the check reads the fields it names",
        ),
        (
            ("artifacts", "*", "repo_id"),
            "the destination: the check only requires one and prints it in the "
            "publish command",
        ),
    ),
    "build-start": (
        (
            ("command", 3),
            "the build's own arguments; its manifests record what they built",
        ),
        (
            ("command", 4),
            "the build's own arguments; its manifests record what they built",
        ),
        (
            ("cpu_count",),
            "this environment sets every thread pool, so no default applies",
        ),
        (("git_environment",), "a GIT_* name that points git nowhere"),
        (("dataset", "bytes"), "the H5 did not exist yet"),
        (("dataset", "sha256"), "the H5 did not exist yet"),
    ),
    "build-end": _END_RECORD_UNJUDGED,
    "smoke-start": (
        (("cpu_count",), "only the build's start picks the thread defaults"),
        (("git_environment",), "a GIT_* name that points git nowhere"),
        (("dataset", "exists"), "the dataset's digest decides"),
        (("dataset", "bytes"), "the dataset's digest decides"),
    ),
    "smoke-end": _END_RECORD_UNJUDGED,
}


def _matches(pattern: tuple, path: tuple) -> bool:
    if pattern and pattern[-1] == "**":
        head = pattern[:-1]
        return len(path) >= len(head) and _matches(head, path[: len(head)])
    return len(pattern) == len(path) and all(
        part == "*" or part == key for part, key in zip(pattern, path, strict=True)
    )


def test_the_allowlists_name_every_input() -> None:
    assert list(_UNJUDGED) == list(_LOOSE)
    assert sorted(_UNJUDGED) == sorted(_INPUT_NAMES)


@pytest.mark.parametrize("name", list(_UNJUDGED))
def test_every_field_the_check_judges_is_refused_when_changed(
    quick, layout, name
) -> None:
    """Change each leaf of one input, one at a time, to another value: the
    check refuses every change but those :data:`_UNJUDGED` lists, and each
    listed pattern covers a leaf it accepted (so the list cannot rot)."""
    file = layout.file(name)
    text = file.read_text()
    payload = (
        [json.loads(line) for line in text.splitlines()]
        if file.suffix == ".jsonl"
        else json.loads(text)
    )
    accepted = []
    for path in _leaves(payload):
        replacement = _changed(_read_leaf(payload, path))
        with _edited(file, _set(*path, value=replacement)):
            try:
                quick.check(layout.inputs(quick), now=lambda: _TIMES["check"])
            except quick.RefusalError:
                continue
            accepted.append(path)
    patterns = [pattern for pattern, _ in _UNJUDGED[name]]
    assert [
        path
        for path in accepted
        if not any(_matches(pattern, path) for pattern in patterns)
    ] == []
    assert [
        pattern
        for pattern in patterns
        if not any(_matches(pattern, path) for path in accepted)
    ] == []
    assert _attachments(quick, layout) == []


# ---------------------------------------------------------------------------
# Every structure: a missing member, a wrong container or a stray member
# ---------------------------------------------------------------------------


def _members(value: Any, path: tuple = ()) -> list[tuple]:
    """The path of every member below the top, containers and leaves alike."""
    if isinstance(value, dict):
        items = list(value.items())
    elif isinstance(value, list):
        items = list(enumerate(value))
    else:
        return []
    return [
        found
        for key, item in items
        for found in ((*path, key), *_members(item, (*path, key)))
    ]


def _deleted(*path: str | int) -> Callable[[Any], None]:
    def edit(payload):
        target = payload
        for key in path[:-1]:
            target = target[key]
        del target[path[-1]]

    return edit


def _with_stray(*path: str | int) -> Callable[[Any], None]:
    """Add a member: a new key to an object, a copy of the last item to a list."""

    def edit(payload):
        target = payload
        for key in path:
            target = target[key]
        if isinstance(target, dict):
            target["stray"] = 1
        else:
            target.append(copy.deepcopy(target[-1]))

    return edit


def _breaks(payload: Any) -> list[tuple[str, tuple, Callable[[Any], None]]]:
    breaks = []
    for path in _members(payload):
        breaks.append(("deleted", path, _deleted(*path)))
        if isinstance(_read_leaf(payload, path), dict | list):
            breaks.append(("not a container", path, _set(*path, value="x")))
    for path in ((), *_members(payload)):
        value = _read_leaf(payload, path)
        if isinstance(value, dict) or (isinstance(value, list) and value):
            breaks.append(("stray member", path, _with_stray(*path)))
    return breaks


_STRAY = "the check reads the members it names, and a stray one changes none"
_OPTIONAL_OPTION = (
    "-B, an interpreter option the command may leave out: it only stops "
    "Python writing bytecode"
)
_RECORD_LOOSE = tuple(
    ("stray member", path, _STRAY)
    for path in ((), ("dataset",), ("python",), ("repo",))
)
#: The breaks the check accepts outside what :data:`_UNJUDGED` lists, by
#: input, with the reason.
_LOOSE: dict[str, tuple[tuple[str, tuple, str], ...]] = {
    "gate": tuple(
        ("stray member", path, _STRAY)
        for path in (
            (),
            ("reform_coverage_smoke",),
            ("reform_coverage_smoke", "details"),
        )
    ),
    "report": (
        *(
            ("stray member", path, _STRAY)
            for path in (
                (),
                ("host",),
                ("tool_source",),
                ("stages", "*"),
                ("stages", "reform_coverage_smoke", "probes", "*"),
            )
        ),
        (
            "stray member",
            ("stages", "load", "rows"),
            "another entity's row count; the household count fixes the batching",
        ),
    ),
    "passes": (),
    "build": tuple(
        ("stray member", path, _STRAY) for path in ((), ("code",), ("dataset",))
    ),
    "release": (
        *(
            ("stray member", path, _STRAY)
            for path in ((), ("artifacts", "*"), ("build",))
        ),
        (
            "stray member",
            ("default_datasets",),
            "another default dataset; the smoke scores the national one",
        ),
        (
            "stray member",
            ("artifacts",),
            "another artifact (a real US release lists 14); the smoke scores only "
            "the national H5",
        ),
    ),
    "build-start": (
        *_RECORD_LOOSE,
        ("stray member", ("out_dir",), _STRAY),
        ("deleted", ("command", 1), _OPTIONAL_OPTION),
        (
            "stray member",
            ("command",),
            "another argument to the build, whose manifests record what it built",
        ),
    ),
    "build-end": _RECORD_LOOSE,
    "smoke-start": (
        *_RECORD_LOOSE,
        ("stray member", ("out_dir",), _STRAY),
        ("deleted", ("command", 1), _OPTIONAL_OPTION),
    ),
    "smoke-end": _RECORD_LOOSE,
}


@pytest.mark.parametrize("name", list(_UNJUDGED))
def test_every_structure_the_check_reads_is_refused_when_broken(
    quick, layout, name
) -> None:
    """Delete each member of one input, replace each container with a
    string, and add a stray member to each container, one at a time: the
    check refuses every break but those inside a member :data:`_UNJUDGED`
    lists and those :data:`_LOOSE` lists, and it accepts each of those."""
    file = layout.file(name)
    text = file.read_text()
    payload = (
        [json.loads(line) for line in text.splitlines()]
        if file.suffix == ".jsonl"
        else json.loads(text)
    )
    unjudged = [pattern for pattern, _ in _UNJUDGED[name]]
    accepted = []
    for kind, path, edit in _breaks(payload):
        with _edited(file, edit):
            try:
                quick.check(layout.inputs(quick), now=lambda: _TIMES["check"])
            except quick.RefusalError:
                continue
        if any(_matches(pattern, path) for pattern in unjudged):
            continue
        accepted.append((kind, path))
    loose = [(kind, pattern) for kind, pattern, _ in _LOOSE[name]]
    unexplained = [
        (kind, path)
        for kind, path in accepted
        if not any(k == kind and _matches(p, path) for k, p in loose)
    ]
    assert unexplained == [], "accepted:\n" + "\n".join(map(repr, unexplained))
    unused = [
        (kind, pattern)
        for kind, pattern in loose
        if not any(k == kind and _matches(pattern, p) for k, p in accepted)
    ]
    assert unused == [], "never accepted:\n" + "\n".join(map(repr, unused))
    assert _attachments(quick, layout) == []


#: What the check accepts when a stage's start and end records change alike,
#: beyond what :data:`_UNJUDGED` leaves unjudged in both, with the reason.
_PAIR_LOOSE: dict[str, tuple[tuple[tuple, str], ...]] = {"build": (), "smoke": ()}


@pytest.mark.parametrize("stage", ["build", "smoke"])
def test_a_departure_both_records_of_a_stage_share_is_refused(
    quick, layout, stage
) -> None:
    """Change each leaf a stage's start and end records share, the same way in
    both, as a forger who keeps the pair consistent would: the check refuses
    every change but those both records leave unjudged and those
    :data:`_PAIR_LOOSE` lists, and it accepts each of those."""
    start_file = layout.file(f"{stage}-start")
    end_file = layout.file(f"{stage}-end")
    start = json.loads(start_file.read_text())
    end_leaves = set(_leaves(json.loads(end_file.read_text())))
    unjudged = {
        phase: [pattern for pattern, _ in _UNJUDGED[f"{stage}-{phase}"]]
        for phase in ("start", "end")
    }
    accepted = []
    for path in _leaves(start):
        if path not in end_leaves:
            continue
        edit = _set(*path, value=_changed(_read_leaf(start, path)))
        with _edited(start_file, edit), _edited(end_file, edit):
            try:
                quick.check(layout.inputs(quick), now=lambda: _TIMES["check"])
            except quick.RefusalError:
                continue
        if all(
            any(_matches(pattern, path) for pattern in patterns)
            for patterns in unjudged.values()
        ):
            continue
        accepted.append(path)
    loose = [pattern for pattern, _ in _PAIR_LOOSE[stage]]
    unexplained = [
        path for path in accepted if not any(_matches(p, path) for p in loose)
    ]
    assert unexplained == [], "accepted:\n" + "\n".join(map(repr, unexplained))
    unused = [p for p in loose if not any(_matches(p, path) for path in accepted)]
    assert unused == [], "never accepted:\n" + "\n".join(map(repr, unused))
    assert _attachments(quick, layout) == []


def test_totals_under_the_floor_are_refused_whatever_the_flags_say(
    quick, layout
) -> None:
    """A gate file and report that agree on totals under a probe's floor, and
    still say it passed, fail on the floor alone."""
    probe = layout.expectation["probes"][0]
    baseline = 1.0e13
    wanted = 0.5 * probe["min_abs_effect"]
    if probe["expected_sign"] == "negative":
        wanted = -wanted
    if probe["effect_direction"] == "baseline_minus_reform":
        reform = baseline - wanted
    else:
        reform = baseline + wanted
    effect = quick.gate_effect(baseline, reform, probe["effect_direction"])
    magnitude = quick.gate_signed_magnitude(effect, probe["expected_sign"])
    assert magnitude < probe["min_abs_effect"]

    def gate(payload):
        result = payload["reform_coverage_smoke"]["details"]["results"][probe["id"]]
        result.update(baseline_total=baseline, reform_total=reform, effect=effect)

    def report(payload):
        row = payload["stages"]["reform_coverage_smoke"]["probes"][0]
        row.update(
            baseline_total=baseline,
            reform_total=reform,
            effect=effect,
            signed_magnitude=magnitude,
            margin_to_floor=magnitude - probe["min_abs_effect"],
        )

    with _edited(layout.file("gate"), gate), _edited(layout.file("report"), report):
        message = _refusal(quick, layout)
    assert message.splitlines() == [
        f"gate result {probe['id']}: the recorded totals give {magnitude!r}, under "
        f"the floor {probe['min_abs_effect']!r}"
    ]


def test_a_smoke_with_another_pythonpath_is_refused(quick, layout) -> None:
    """Both smoke records agree with each other on another PYTHONPATH; only
    the comparison with the build refuses it."""

    def other_path(record):
        for block in ("environment", "runtime_environment"):
            record[block]["PYTHONPATH"] = "/elsewhere"

    with (
        _edited(layout.file("smoke-start"), other_path),
        _edited(
            layout.file("smoke-end"),
            _set("runtime_environment", "PYTHONPATH", value="/elsewhere"),
        ),
    ):
        message = _refusal(quick, layout)
    assert len(message.splitlines()) == 1
    assert message.startswith('the smoke\'s PYTHONPATH is "/elsewhere"')


def test_records_from_another_checkout_are_refused(quick, layout) -> None:
    def elsewhere(record):
        record["repo"]["toplevel"] = "/elsewhere/repo"
        record["cwd"] = "/elsewhere/repo"

    names = ("build-start", "build-end", "smoke-start", "smoke-end")
    with ExitStack() as stack:
        for name in names:
            stack.enter_context(_edited(layout.file(name), elsewhere))
        message = _refusal(quick, layout)
    for name in names:
        assert f"{name} repository" in message
        assert f"{name} working directory" in message


def test_the_real_runs_thread_settings_are_accepted(quick, layout) -> None:
    """Route A's build sets no thread pool, so the release tool's defaults
    apply (``min(cpu_count, 16)``); the smoke sets each pool to that value."""

    def unset(record):
        record["cpu_count"] = 18
        for block in ("environment", "runtime_environment"):
            record[block].update(dict.fromkeys(quick.THREAD_VARIABLES))

    def build_end(record):
        record["runtime_environment"].update(dict.fromkeys(quick.THREAD_VARIABLES))

    def sixteen(record):
        for block in ("environment", "runtime_environment"):
            if block in record:
                record[block].update(dict.fromkeys(quick.THREAD_VARIABLES, "16"))

    def smoke_end(record):
        record["runtime_environment"].update(
            dict.fromkeys(quick.THREAD_VARIABLES, "16")
        )

    def calibration(threads):
        def edit(payload):
            block = payload["build"] if "build" in payload else payload
            block["calibration_runtime"]["omp_num_threads"] = threads

        return edit

    def edits(threads):
        return (
            ("build-start", unset),
            ("build-end", build_end),
            ("smoke-start", sixteen),
            ("smoke-end", smoke_end),
            ("build", calibration(threads)),
            ("release", calibration(threads)),
        )

    with ExitStack() as stack:
        for name, edit in edits("16"):
            stack.enter_context(_edited(layout.file(name), edit))
        quick.check(layout.inputs(quick), now=lambda: _TIMES["check"])
        # A host with fewer CPUs gives smaller defaults than the smoke set.
        with _edited(layout.file("build-start"), _set("cpu_count", value=8)):
            message = _refusal(quick, layout)
    assert "the smoke's thread-pool settings" in message
    assert "calibration_runtime.omp_num_threads" in message


def test_a_changed_probe_file_refuses_the_expectation(
    tool, layout, monkeypatch
) -> None:
    definitions, files = copy.deepcopy(layout.pins)
    files[tool.PROBE_PATH] = "0" * 64
    monkeypatch.setattr(tool, "MIRRORED_DEFINITIONS", definitions)
    monkeypatch.setattr(tool, "PINNED_FILES", files)
    with pytest.raises(tool.RefusalError, match="this check reads the layout of 0000"):
        tool.build_expectation(layout.repo, layout.commit)


def test_a_manifest_with_no_probes_refuses_the_expectation(
    tool, tmp_path, monkeypatch
) -> None:
    payload = json.loads((_ROOT / tool.MANIFEST_PATH).read_text())
    payload["reform_coverage_probes"] = []
    repo, commit = _source_repository(
        tool, tmp_path, manifest=(json.dumps(payload) + "\n").encode()
    )
    pins = _current_pins(tool)
    monkeypatch.setattr(tool, "MIRRORED_DEFINITIONS", pins[0])
    monkeypatch.setattr(tool, "PINNED_FILES", pins[1])
    with pytest.raises(tool.RefusalError, match="declares no probes"):
        tool.build_expectation(repo, commit)


# ---------------------------------------------------------------------------
# launch-record
# ---------------------------------------------------------------------------


def _commit_all(repo: Path, message: str) -> None:
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
        message,
    )


def _bare_repo(root: Path) -> Path:
    repo = root / "repo"
    (repo / "tools").mkdir(parents=True)
    (repo / "tools" / "keep").write_text("")
    (repo / "uv.lock").write_text("lock\n")
    subprocess.run(
        ["git", *NO_BACKGROUND_MAINTENANCE, "init", "-q", str(repo)], check=True
    )
    _commit_all(repo, "x")
    return repo.resolve()


@dataclass
class _Launches:
    """``launch-record`` in a bare repository, with a stub runtime that notes
    the environment it measures with, and an exec that notes the command (and
    the records written by then) instead of replacing the test process."""

    tool: Any
    repo: Path
    root: Path
    launch_dir: Path
    dataset: Path
    execs: list
    measured: list

    def args(self, stage, phase, *, command=None, **options) -> list[str]:
        argv = [
            "launch-record",
            "--stage",
            stage,
            "--phase",
            phase,
            "--run-id",
            "r1",
            "--launch-dir",
            str(self.launch_dir),
            "--python",
            sys.executable,
            "--repo",
            str(self.repo),
        ]
        for key, value in options.items():
            argv += [f"--{key.replace('_', '-')}", str(value)]
        return argv if command is None else [*argv, "--", *command]

    def build_start(self) -> list[str]:
        out = self.root / "build-out"
        return self.args("build", "start", out_dir=out, command=_build_command(out))

    def smoke_start(self, *, batch_size=_BATCH, dataset=None) -> list[str]:
        out = self.root / "smoke-out"
        dataset = self.dataset if dataset is None else dataset
        return self.args(
            "smoke",
            "start",
            out_dir=out,
            dataset=dataset,
            command=_smoke_command(dataset, out, batch_size),
        )

    def end(self, stage, exit_code=0) -> list[str]:
        return self.args(stage, "end", dataset=self.dataset, exit_code=exit_code)

    def record(self, stage, phase) -> dict:
        path = self.launch_dir / self.tool.LAUNCH_FILES[(stage, phase)]
        return self.tool.strict_loads(path.read_bytes())

    def written(self) -> list[str]:
        if not self.launch_dir.exists():
            return []
        return sorted(path.name for path in self.launch_dir.iterdir())

    def build(self) -> None:
        """A finished build: its records, its H5 and its manifest."""
        assert self.tool.main(self.build_start()) == 0
        self.dataset.parent.mkdir(parents=True)
        self.dataset.write_bytes(_DATASET_BYTES)
        manifest = self.root / "build-out" / "releases" / "r" / "build_manifest.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(
            json.dumps({"post_export_scoring": {"maximum_batch_size": _BATCH}})
        )
        assert self.tool.main(self.end("build")) == 0


@pytest.fixture
def launches(tool, tmp_path, monkeypatch) -> _Launches:
    repo = _bare_repo(tmp_path)
    monkeypatch.chdir(repo)
    # A known environment: the pools set, the other recorded variables unset.
    for name in tool.THREAD_VARIABLES:
        monkeypatch.setenv(name, "2")
    for name in tool.ENVIRONMENT_VARIABLES[len(tool.THREAD_VARIABLES) :]:
        monkeypatch.delenv(name, raising=False)
    _clear_git_environment(monkeypatch)
    launch_dir = tmp_path / "launch"
    execs: list = []
    measured: list = []

    def runtime(python, tools, environment=None):
        measured.append(dict(environment or {}))
        return {"stub": True}

    def become(path, argv):
        written = sorted(p.name for p in launch_dir.iterdir())
        execs.append((path, list(argv), written))

    monkeypatch.setattr(tool, "_runtime", runtime)
    monkeypatch.setattr(tool, "_exec", become)
    return _Launches(
        tool=tool,
        repo=repo,
        root=tmp_path,
        launch_dir=launch_dir,
        dataset=tmp_path / "build-out" / "artifacts" / _DATASET_FILE,
        execs=execs,
        measured=measured,
    )


def test_a_start_record_becomes_its_command(launches, capsys) -> None:
    tool = launches.tool
    argv = launches.build_start()
    command = argv[argv.index("--") + 1 :]
    assert tool.main(argv) == 0
    # The record exists before the process becomes the build.
    assert launches.execs == [(sys.executable, command, ["build-start.json"])]
    assert "recorded" in capsys.readouterr().out
    record = launches.record("build", "start")
    assert record["command"] == command
    assert record["runtime_environment"] == record["environment"]
    assert record["git_environment"] == []
    # The build's start may name its H5 before it exists, or not at all.
    assert record["dataset"] is None
    assert launches.measured == [record["environment"]]


def test_an_end_record_measures_with_its_starts_environment(
    launches, monkeypatch
) -> None:
    tool = launches.tool
    assert tool.main(launches.build_start()) == 0
    start = launches.record("build", "start")
    monkeypatch.setenv("OMP_NUM_THREADS", "5")
    monkeypatch.setenv("PYTHONPATH", "/elsewhere")
    assert tool.main(launches.end("build")) == 0
    end = launches.record("build", "end")
    assert end["environment"]["OMP_NUM_THREADS"] == "5"
    assert end["environment"]["PYTHONPATH"] == "/elsewhere"
    assert end["runtime_environment"] == start["environment"]
    assert launches.measured == [start["environment"], start["environment"]]
    assert end["command"] is None and end["out_dir"] is None
    # An end launches nothing.
    assert len(launches.execs) == 1


def test_the_runtime_is_measured_with_the_given_environment(tool, tmp_path) -> None:
    """``_runtime`` sets the listed variables in the process it measures."""
    tools = _ROOT / "tools"
    marker = str(tmp_path / "on-the-path")
    with_marker = tool._runtime(sys.executable, tools, {"PYTHONPATH": marker})
    without = tool._runtime(sys.executable, tools, {"PYTHONPATH": None})
    assert marker in with_marker["sys_path"]
    assert marker not in without["sys_path"]


_BAD_COMMANDS = [
    # (case, stage, edit(command) -> command, expected)
    ("no command", "build", lambda c: [], "give its command after --"),
    (
        "another spelling of the interpreter",
        "build",
        lambda c: [
            os.path.join(os.path.dirname(c[0]), ".", os.path.basename(c[0])),
            *c[1:],
        ],
        "not the --python interpreter",
    ),
    (
        "an interpreter option",
        "build",
        lambda c: [c[0], "-X", "dev", *c[1:]],
        "with no interpreter options but ['-B', '-u'] before it",
    ),
    (
        "another script",
        "build",
        lambda c: [*c[:2], "tools/probe_us_post_export.py", *c[3:]],
        "the command must run tools/build_us_fiscal_refresh_release.py",
    ),
    ("no script", "build", lambda c: c[:2], "the command must run"),
    (
        "a sample",
        "smoke",
        lambda c: [*c, "--sample-receipt", "sample_receipt.json"],
        "a census smoke passes only",
    ),
    (
        "a reference",
        "smoke",
        lambda c: [*c, "--reference-smoke", "reform_coverage_smoke.json"],
        "a census smoke passes only",
    ),
    (
        "no census",
        "smoke",
        lambda c: [token for token in c if token != "--census"],
        "the smoke must pass each of ['--census'] once",
    ),
    (
        "another stage",
        "smoke",
        lambda c: [*c[:5], "reform_coverage_smoke,demographics", *c[6:]],
        "the smoke runs --stages",
    ),
    (
        "a zero batch",
        "smoke",
        lambda c: [*c[:7], "0", *c[8:]],
        "the smoke's --batch-size '0' is not a positive integer",
    ),
    (
        "another export",
        "smoke",
        lambda c: [*c[:9], "/elsewhere/microcosm_us_2024.h5", *c[10:]],
        "not the recorded dataset",
    ),
    (
        "another output",
        "smoke",
        lambda c: [*c[:11], "/elsewhere/smoke-out"],
        "not the recorded output directory",
    ),
]


@pytest.mark.parametrize(
    ("case", "stage", "edit", "expected"),
    _BAD_COMMANDS,
    ids=[case for case, *_ in _BAD_COMMANDS],
)
def test_a_start_refuses_a_command_that_is_not_the_stages(
    launches, capsys, case, stage, edit, expected
) -> None:
    argv = launches.build_start() if stage == "build" else launches.smoke_start()
    split = argv.index("--")
    argv = [*argv[: split + 1], *edit(argv[split + 1 :])]
    assert launches.tool.main(argv) == 1
    assert expected in capsys.readouterr().err
    assert launches.execs == []
    assert launches.written() == []


def test_the_smoke_arguments_take_either_spelling(tool) -> None:
    values, unknown = tool._smoke_arguments(
        ["--census", "--stages=reform_coverage_smoke", "--batch-size", "16", "-v"]
    )
    assert values == {
        "--census": [""],
        "--stages": ["reform_coverage_smoke"],
        "--batch-size": ["16"],
    }
    assert unknown == ["-v"]
    # A switch never takes a value, and a flag at the end has none to take.
    assert tool._smoke_arguments(["--census=yes", "--out"]) == (
        {},
        ["--census=yes", "--out"],
    )


def test_launch_record_refuses_options_in_the_wrong_place(launches, capsys) -> None:
    tool = launches.tool
    out = launches.root / "build-out"
    build = _build_command(out)
    python_only = launches.build_start()
    python_only[python_only.index("--python") + 1] = "python3"
    cases = [
        (
            launches.args("build", "start", out_dir=out, exit_code=0, command=build),
            "--exit-code belongs to an end record",
        ),
        (
            launches.args("build", "start", command=build),
            "a start record needs --out-dir",
        ),
        (
            launches.args(
                "build", "end", dataset=launches.dataset, exit_code=0, out_dir=out
            ),
            "--out-dir belongs to a start record",
        ),
        (
            launches.args(
                "build", "end", dataset=launches.dataset, exit_code=0, command=["x"]
            ),
            "an end record launches nothing; give no command",
        ),
        (
            launches.args("build", "end", exit_code=0),
            "this launch record needs --dataset",
        ),
        (
            launches.args(
                "smoke",
                "start",
                out_dir=launches.root / "smoke-out",
                command=_smoke_command(
                    launches.dataset, launches.root / "smoke-out", 1
                ),
            ),
            "this launch record needs --dataset",
        ),
        (python_only, "must name the interpreter by its path"),
        (
            launches.args("build", "end", dataset=launches.dataset),
            "an end record needs --exit-code",
        ),
    ]
    for argv, expected in cases:
        assert tool.main(argv) == 1, argv
        assert expected in capsys.readouterr().err, argv
    assert launches.written() == []
    assert launches.execs == []
    with pytest.raises(SystemExit):
        tool.main(["expect", "--commit", "0" * 40, "--", "true"])
    assert "only launch-record takes a command after --" in capsys.readouterr().err


def test_launch_records_refuse_a_wasted_start(launches, monkeypatch, capsys) -> None:
    tool = launches.tool
    repo = launches.repo
    argv = launches.build_start()

    def refused(expected: str) -> None:
        assert tool.main(argv) == 1
        err = capsys.readouterr().err
        assert expected in err, err

    (repo / "untracked.txt").write_text("x")
    refused("uncommitted changes, counting files that only a global excludes")
    refused("?? untracked.txt")
    (repo / "untracked.txt").unlink()
    (repo / "release_input_coverage_manifest.json").write_text("{}")
    _commit_all(repo, "y")
    refused("read before the package's own")
    _git(repo, "rm", "-q", "release_input_coverage_manifest.json")
    _commit_all(repo, "z")
    (launches.root / "build-out").mkdir()
    (launches.root / "build-out" / "old").write_text("x")
    refused("already holds files")
    (launches.root / "build-out" / "old").unlink()
    monkeypatch.setenv("GIT_DIR", str(repo / ".git"))
    refused("which point git at another repository or configuration")
    monkeypatch.delenv("GIT_DIR")
    monkeypatch.chdir(repo / "tools")
    refused("not the repository root")
    monkeypatch.chdir(repo)
    missing = launches.root / "no-such-python"
    absent = launches.args(
        "build",
        "start",
        out_dir=launches.root / "build-out",
        command=[str(missing), "-B", "tools/build_us_fiscal_refresh_release.py"],
    )
    absent[absent.index("--python") + 1] = str(missing)
    assert tool.main(absent) == 1
    assert "is not an executable file" in capsys.readouterr().err
    assert launches.written() == []
    assert launches.execs == []


def test_launch_records_follow_the_run(launches, capsys) -> None:
    tool = launches.tool
    assert tool.main(launches.end("build")) == 1
    assert "no start record" in capsys.readouterr().err
    assert tool.main(launches.smoke_start()) == 1
    assert "after the build ends" in capsys.readouterr().err
    start = launches.build_start()
    assert tool.main(start) == 0
    assert tool.main(start) == 1
    assert "already exists" in capsys.readouterr().err
    # An end record is evidence: it is written even after the tree changed.
    launches.dataset.parent.mkdir(parents=True)
    launches.dataset.write_bytes(_DATASET_BYTES)
    (launches.repo / "scratch.txt").write_text("x")
    assert tool.main(launches.end("build", exit_code=3)) == 0
    record = launches.record("build", "end")
    assert record["exit_code"] == 3
    assert record["repo"]["clean"] is False
    (launches.repo / "scratch.txt").unlink()
    assert tool.main(launches.smoke_start()) == 1
    assert "the build's exit code is 3; required 0" in capsys.readouterr().err
    # An end record must close the start record's run.
    (launches.launch_dir / "build-end.json").unlink()
    assert tool.main(launches.end("build")) == 0
    assert tool.main(launches.smoke_start()) == 0
    other = launches.end("smoke")
    other[other.index("--run-id") + 1] = "r2"
    assert tool.main(other) == 1
    assert "names another run" in capsys.readouterr().err
    assert [command[2] for _, command, _ in launches.execs] == [
        "tools/build_us_fiscal_refresh_release.py",
        "tools/probe_us_post_export.py",
    ]


def test_an_end_record_needs_a_start_that_lists_its_environment(
    launches, capsys
) -> None:
    """An end measures the runtime with its start's environment, so a start
    record without one leaves it nothing to measure with."""
    tool = launches.tool
    assert tool.main(launches.build_start()) == 0
    launches.dataset.parent.mkdir(parents=True)
    launches.dataset.write_bytes(_DATASET_BYTES)
    start = launches.launch_dir / "build-start.json"
    payload = json.loads(start.read_text())
    payload["environment"] = None
    start.write_text(json.dumps(payload))
    assert tool.main(launches.end("build")) == 1
    assert "the start record lists no environment" in capsys.readouterr().err
    assert launches.written() == ["build-start.json"]


def test_the_smoke_start_compares_its_launch_with_the_builds(
    launches, monkeypatch, capsys
) -> None:
    """The smoke's start refuses, before the smoke runs, what ``check`` would
    refuse hours later."""
    tool = launches.tool
    launches.build()

    def refused(argv, expected):
        assert tool.main(argv) == 1
        err = capsys.readouterr().err
        assert expected in err, err
        assert launches.written() == ["build-end.json", "build-start.json"]

    monkeypatch.setenv("PYTHONHASHSEED", "7")
    refused(launches.smoke_start(), "the smoke's PYTHONHASHSEED")
    monkeypatch.delenv("PYTHONHASHSEED")
    monkeypatch.setenv("OMP_NUM_THREADS", "5")
    refused(launches.smoke_start(), "the smoke's thread-pool settings")
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    refused(
        launches.smoke_start(batch_size=1000),
        "the smoke's --batch-size 1000 is not the build manifest's "
        "maximum_batch_size 2000",
    )
    receipt = launches.dataset.parent / "sample_receipt.json"
    receipt.write_text("{}")
    refused(launches.smoke_start(), "the probe would score the sample it describes")
    receipt.unlink()
    other = launches.root / "other.h5"
    other.write_bytes(_DATASET_BYTES)
    refused(launches.smoke_start(dataset=other), "smoke-start dataset path")
    assert len(launches.execs) == 1
    assert tool.main(launches.smoke_start()) == 0
    smoke_out = launches.root / "smoke-out"
    assert launches.execs[-1][1] == _smoke_command(launches.dataset, smoke_out, _BATCH)


_SMOKE_START_DEPARTURES = [
    # (record, path, value, expected text)
    ("build-start", ("run_id",), "r0", "smoke-start run_id"),
    ("build-start", ("tool_sha256",), "0" * 64, "smoke-start tool_sha256"),
    ("build-start", ("files", "uv.lock"), "0" * 64, "smoke-start files"),
    ("build-start", ("repo", "head"), "0" * 40, "smoke-start HEAD"),
    (
        "build-start",
        ("python", "resolved"),
        "/elsewhere/python",
        "smoke-start interpreter",
    ),
    ("build-start", ("runtime", "version"), "3.0.0", "smoke-start runtime"),
    ("build-start", ("environment",), None, "a start record lists no environment"),
    ("build-end", ("exit_code",), 1, "the build's exit code"),
    ("build-end", ("dataset", "sha256"), None, "build-end recorded no dataset digest"),
    ("build-end", ("dataset", "sha256"), "0" * 64, "smoke-start dataset (build-end's)"),
]


@pytest.mark.parametrize(
    ("record", "path", "value", "expected"),
    _SMOKE_START_DEPARTURES,
    ids=[
        f"{record} {'.'.join(path)} {value!r}"
        for record, path, value, _ in _SMOKE_START_DEPARTURES
    ],
)
def test_the_smoke_start_refuses_build_records_it_departs_from(
    launches, capsys, record, path, value, expected
) -> None:
    tool = launches.tool
    launches.build()
    file = launches.launch_dir / tool.LAUNCH_FILES[tuple(record.split("-"))]
    payload = json.loads(file.read_text())
    _set(*path, value=value)(payload)
    file.write_text(json.dumps(payload))
    assert tool.main(launches.smoke_start()) == 1
    err = capsys.readouterr().err
    assert expected in err, err
    assert launches.written() == ["build-end.json", "build-start.json"]
    assert len(launches.execs) == 1


def test_git_reads_only_the_repository_it_is_pointed_at(
    tool, tmp_path, monkeypatch
) -> None:
    """Inherited ``GIT_*`` variables and the caller's configuration cannot
    make a snapshot describe another repository or hide a file."""
    here = _bare_repo(tmp_path / "here")
    there = _bare_repo(tmp_path / "there")
    (there / "other.txt").write_text("x")
    _commit_all(there, "another commit")
    head = _git(here, "rev-parse", "HEAD")
    assert head != _git(there, "rev-parse", "HEAD")
    ignore_all = tmp_path / "ignore-all"
    ignore_all.write_text("*\n")
    config = tmp_path / "gitconfig"
    config.write_text(f"[core]\n\texcludesFile = {ignore_all}\n")
    (here / "untracked.txt").write_text("x")
    for name, value in {
        "GIT_DIR": str(there / ".git"),
        "GIT_WORK_TREE": str(there),
        "GIT_INDEX_FILE": str(there / ".git" / "index"),
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "core.excludesFile",
        "GIT_CONFIG_VALUE_0": str(ignore_all),
        "GIT_CONFIG_GLOBAL": str(config),
        "XDG_CONFIG_HOME": str(tmp_path),
    }.items():
        monkeypatch.setenv(name, value)
    (tmp_path / "git").mkdir()
    (tmp_path / "git" / "ignore").write_text("*\n")
    state = tool._repo_state(here)
    assert state["toplevel"] == str(here)
    assert state["head"] == head
    assert state["clean"] is False
    assert state["status_entries"] == 1
    env = tool._git_env()
    assert not [
        name
        for name in env
        if name.startswith("GIT_CONFIG_")
        and name
        not in (
            "GIT_CONFIG_NOSYSTEM",
            "GIT_CONFIG_GLOBAL",
        )
    ]
    assert "GIT_DIR" not in env and "GIT_INDEX_FILE" not in env
    assert env["GIT_CONFIG_NOSYSTEM"] == "1"
    assert env["GIT_CONFIG_GLOBAL"] == os.devnull


@pytest.mark.parametrize(
    ("environment", "cpus", "expected"),
    [
        (dict.fromkeys(_THREAD_NAMES), 18, "16"),
        (dict.fromkeys(_THREAD_NAMES), 16, "16"),
        (dict.fromkeys(_THREAD_NAMES), 4, "4"),
        # os.cpu_count() can return None; the release tool then uses 1.
        (dict.fromkeys(_THREAD_NAMES), None, "1"),
        (dict.fromkeys(_THREAD_NAMES), 0, "1"),
        (dict.fromkeys(_THREAD_NAMES, "2"), 18, "2"),
        # setdefault keeps a variable the environment sets, even an empty one.
        (dict.fromkeys(_THREAD_NAMES, ""), 18, ""),
    ],
)
def test_effective_threads_are_the_release_tools(
    tool, environment, cpus, expected
) -> None:
    """The release tool sets each pool with ``setdefault`` to
    ``str(min(os.cpu_count() or 1, 16))``; an operator's value wins."""
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


@pytest.mark.parametrize(
    "record",
    [
        None,
        [],
        {"cpu_count": 18},
        {"environment": None, "cpu_count": 18},
        {"environment": {}},
        {"environment": {}, "cpu_count": "18"},
        {"environment": {}, "cpu_count": True},
        {"environment": {}, "cpu_count": -1},
        {"environment": {}, "cpu_count": 18.0},
    ],
)
def test_effective_threads_need_a_record_the_tool_writes(tool, record) -> None:
    assert tool._effective_threads(record) is tool.MISSING


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
    assert record["runtime_environment"] == record["environment"]
    assert record["git_environment"] == []
    assert record["command"] == _build_command(layout.out)
    assert record["python"] == {
        "requested": sys.executable,
        "resolved": str(Path(sys.executable).resolve()),
    }
    assert (
        record["files"]["uv.lock"]
        == hashlib.sha256((_ROOT / "uv.lock").read_bytes()).hexdigest()
    )
    assert runtime["sys_path"][0] == str(layout.repo / "tools")
    assert set(runtime["runtime_versions"]) == {"python", *tool.RUNTIME_PACKAGES}
    origin = runtime["import_origins"]["microcosm.build"]["origin"]
    assert Path(origin).is_relative_to(
        layout.repo / "packages" / "microcosm-build" / "src"
    )
    assert record["dataset"] == {
        "path": str(layout.dataset),
        "exists": False,
        "bytes": None,
        "sha256": None,
    }
    assert record["out_dir"] == {"path": str(layout.out), "state": "absent"}
    end = tool.strict_loads(layout.file("build-end").read_bytes())
    assert end["runtime_environment"] == record["environment"]
    assert end["command"] is None


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
# Exact multiples, which random sizes rarely draw.
@example(n=4_000, size=2_000)
@example(n=6_000, size=2_000)
@example(n=2_000, size=2_000)
@example(n=2_000, size=1)
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
    # The check's model of the pools, on this host's CPU count and an
    # environment that sets none of them, gives the release tool's default.
    assert tool._effective_threads(
        {
            "environment": dict.fromkeys(tool.THREAD_VARIABLES),
            "cpu_count": os.cpu_count(),
        }
    ) == dict.fromkeys(tool.THREAD_VARIABLES, builder._THREAD_POOL_DEFAULT)
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
