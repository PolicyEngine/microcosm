"""Hermetic command-line integration coverage for UK staging smoke builds."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import h5py

from microcosm.build.country_spec import load_country_spec
from microcosm.build.staging_v2 import validate_v2_bundle
from microcosm.build.uk_runtime.country_adapter import (
    UK_SYNTHETIC_SUPPORT_REGISTER,
    uk_spine_geography_binding,
)
from microcosm.build.uk_runtime.geography_first_roster import SIDECAR_GEOGRAPHY_KEYS
from microcosm.build.uk_runtime.graph_kernels import FIXTURE_SUPPORT_INPUTS
from microcosm.graph import ContentStore, collect_execution_evidence, load_run_evidence
from microcosm.graph.orrery import orrery_document_from_schema
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")

ROOT = _TEST_PATHS.repository
FIXTURE = ROOT / "packages/microcosm-graph/tests/fixtures/parity/uk_spine/sources"
DRIVER = ROOT / "tools/build_uk_frs_spine.py"
SEED = 42
RUN_ID = "ci-uk-smoke-full-s42"
#: The spine gates microcosm#931 registers or moves: the area draw's and the
#: SPI-copy area stage's mapping integrity, each later copy's agreement with
#: its parent on the areas, the local-authority and BRMA engine domains, the
#: geography ladder, the area home values ranking and the WAS support clip
#: with its exemptions.
_SPINE_GEOGRAPHY_GATES = (
    "uk_atomic_area_mapping_integrity",
    "uk_spi_copy_area_mapping_integrity",
    "uk_copy_area_agreement",
    "uk_local_authority_enum_domain",
    "uk_brma_enum_domain",
    "uk_geography_ladder",
    "uk_stage_area_home_values_ranking",
    "uk_stage_was_wealth_support",
)


def _tracked_state() -> str:
    return subprocess.run(
        ["git", "status", "--short", "--untracked-files=no"],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout


def _command(tmp_path: Path, *extra: str) -> list[str]:
    return [
        sys.executable,
        str(DRIVER),
        "--synthetic-fixture-dir",
        str(FIXTURE),
        "--spine-h5",
        str(tmp_path / "uk-smoke.h5"),
        "--sample-fraction",
        "1.0",
        "--sample-seed",
        str(SEED),
        "--smoke",
        "--staging-local-only",
        "--staging-dir",
        str(tmp_path / "staging"),
        "--staging-run-id",
        RUN_ID,
        *extra,
    ]


def test_uk_staging_smoke_command_runs_every_spine_stage(tmp_path: Path) -> None:
    """Run all current spine transforms without licensed or remote inputs."""

    before = _tracked_state()
    started = time.perf_counter()
    result = subprocess.run(
        _command(tmp_path),
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=12 * 60,
    )
    elapsed = time.perf_counter() - started
    assert result.returncode == 0, result.stderr
    assert _tracked_state() == before

    output = tmp_path / "uk-smoke.h5"
    sidecar = json.loads(output.with_suffix(".build.json").read_text())
    # The sidecar records how the spine drew its areas: the block the leaf
    # names, composed from the fixture's toy conditioned supports under the
    # synthetic register (microcosm#931). The file is written with its keys
    # sorted, so the block's keys are compared as a set here; the writer's
    # order is test_uk_spine_posture's.
    descriptor = json.loads((FIXTURE / "fixture.json").read_text())
    supports = {
        system: (FIXTURE / descriptor["inputs"][source_id]).read_bytes()
        for system, source_id in FIXTURE_SUPPORT_INPUTS.items()
    }
    assert set(sidecar["geography"]) == set(SIDECAR_GEOGRAPHY_KEYS)
    assert sidecar["geography"]["register"] == UK_SYNTHETIC_SUPPORT_REGISTER
    assert sidecar["geography"] == uk_spine_geography_binding(
        supports, register=UK_SYNTHETIC_SUPPORT_REGISTER
    )
    # The smoke spine's battery runs those gates, and none blocks it: each
    # passes, or, where it checks a fact of the population the fixture is
    # not (the geography ladder's London share; the area home values
    # ranking's declared regions, every English region and Wales, of which
    # the fixture holds three), fails recorded and not blocking
    # (microcosm#931).
    gate_report = json.loads(output.with_suffix(".spine_gates.json").read_text())
    entries = {entry.id: entry for entry in load_country_spec("uk").gates.gates}
    for gate_id in _SPINE_GEOGRAPHY_GATES:
        status = gate_report["gates"][gate_id]["status"]
        assert status == "passed" or (
            status == "failed" and entries[gate_id].population_fact_check
        ), (gate_id, status)
    # A failure of those two may only be the population fact itself, so a
    # code-level failure of either (the ladder's codes, nesting or local
    # authorities; the ranking's knots level, basis year, rank basis, clone-0
    # count or cap) still fails the smoke run. The fixture's root cycles four
    # regions, London among them, and holds three of England and Wales's ten.
    ladder = gate_report["gates"]["uk_geography_ladder"]
    assert ladder["status"] == "passed" or all(
        line.startswith("London weighted household share outside")
        for line in ladder["failures"]
    ), ladder["failures"]
    ranking = gate_report["gates"]["uk_stage_area_home_values_ranking"]
    assert ranking["status"] == "passed" or ranking["failures"] == [
        "was_wealth: area_home_values: the regions rewritten are not the declared "
        "regions."
    ], ranking["failures"]
    with h5py.File(output) as data:
        assert bool(data.attrs["populace_non_release"]) is True
        assert data.attrs["populace_release_posture"] == "smoke"
    assert sidecar["non_release"] is True
    assert sidecar["release_posture"] == "non_release_smoke"
    assert sidecar["sampling"] is None
    assert sidecar["synthetic_fixture"]["schema_version"] == (
        "uk-spine-parity-fixture.v1"
    )
    assert len(sidecar["synthetic_fixture"]["digest"]) == 64

    schema = json.loads((output.parent / sidecar["graph_schema"]["path"]).read_bytes())
    store = ContentStore(
        tmp_path / ".uk-smoke.checkpoints" / "node-graph", create=False
    )
    runs = load_run_evidence(
        output.parent / sidecar["graph_execution_evidence"]["path"], store=store
    )
    document = orrery_document_from_schema(
        schema, execution=collect_execution_evidence(schema, runs=runs, store=store)
    )
    operations = [node for node in document["nodes"] if node["kind"] == "operation"]
    assert len(operations) == len(schema["graph"]["nodes"])
    # These are the full executable gate contracts, not shortened display labels.
    assert any(
        len(node["data"]["declaration"]["params"].get("gate_manifest", "")) > 16_384
        for node in operations
    )

    bundle = validate_v2_bundle(tmp_path / "staging", RUN_ID)
    manifest = bundle["run_manifest"]
    assert manifest["status"] == "completed"
    assert manifest["non_release"] is True
    assert manifest["sample"] == {"mode": "full"}
    assert manifest["delivery"]["mode"] == "local_only"
    assert manifest["delivery"]["upload_attempts"] == 0

    spec = load_country_spec("uk")
    expected = [stage.stage for stage in spec.sources.stages]
    events = bundle["events"]
    for stage in expected:
        transitions = [
            event["status"] for event in events if event["stage_id"] == stage
        ]
        assert transitions == ["started", "completed"], stage
    stage_seconds = {
        event["stage_id"]: event["details"]["elapsed_seconds"]
        for event in events
        if event["status"] == "completed" and "elapsed_seconds" in event["details"]
    }
    assert set(stage_seconds) == set(expected)
    print(
        json.dumps(
            {
                "total_seconds": elapsed,
                "stage_seconds": stage_seconds,
                "household_rows": sidecar["entity_row_counts"]["household"],
            },
            sort_keys=True,
        )
    )


def test_synthetic_smoke_command_refuses_release_options(tmp_path: Path) -> None:
    result = subprocess.run(
        _command(tmp_path, "--release-candidate"),
        cwd=ROOT,
        text=True,
        capture_output=True,
        # The refusal comes from argparse, but reaching it imports the build
        # package and the engine; the 3.14 CI runner has taken over 30 s for
        # that alone (a flake seen on #966), so allow what a cold import needs.
        timeout=120,
    )

    assert result.returncode == 2
    assert "non-release smoke builds refuse --release-candidate" in result.stderr
    assert not (tmp_path / "uk-smoke.h5").exists()
