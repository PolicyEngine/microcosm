"""Synthetic dense full-build fixtures for the canonical UK CLI: the
kernel stand-ins, payload builders, the prepared build and the
``graph_dense_bundle`` runner the release pre-flight and assembler tests
feed."""

# ruff: noqa: F401

import hashlib
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from microcosm.build.gate_battery import (
    GateOutcome,
    GatePhaseReport,
    GateStatus,
    gate_phase_report_payload,
)
from microcosm.build.gates import GateResult
from microcosm.build.logbook import LOGBOOK_ROW_FIELDS, load_spool_rows
from microcosm.build.logbook_adoption import local_artifact_reference
from microcosm.build.uk_runtime import full_build_cli as cli
from microcosm.build.uk_runtime.full_certification import FULL_CERTIFICATION_TYPE
from microcosm.build.uk_runtime.full_gates import (
    classify_full_gate_outcomes,
    uk_full_gate_manifest,
)
from microcosm.build.uk_runtime.graph_build import UKFullBuildConfig, UKFullGraph
from microcosm.build.uk_runtime.graph_calibration import UKCalibrationNodes
from microcosm.build.uk_runtime.graph_population import (
    GEOGRAPHY_GATE_TYPE,
    POPULATION_RECEIPT_TYPE,
)
from microcosm.build.uk_runtime.graph_targets import (
    TARGET_SELECTION_TYPE,
    TARGET_SURFACE_TYPE,
    registry_payload,
)
from microcosm.build.uk_runtime.graph_terminal import (
    FULL_DIAGNOSTICS_CSV_TYPE,
    FULL_DIAGNOSTICS_TYPE,
    FULL_GATE_REPORT_TYPE,
    FULL_HOLDOUT_TYPE,
    FULL_SUPPORT_CSV_TYPE,
    add_uk_export_preparation,
    register_uk_terminal_kernels,
)
from microcosm.build.uk_runtime.release_identity import UK_DENSE_RELEASE_ID
from microcosm.calibrate import (
    Target,
    TargetRegistry,
    TargetSet,
    build_constraint_matrix,
)
from microcosm.calibrate.artifacts import PROBLEM_TYPE, encode_problem
from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    Capabilities,
    Determinism,
    Graph,
    KernelBase,
    KernelRegistry,
    KernelResult,
    Node,
    Owned,
    SourceRef,
    StructuralDelta,
)
from microcosm.graph.canonical import canonical_json
from test_support.microcosm_build.uk_graph_terminal import _frame
from test_support.paths import paths_for

PIN = "0" * 64

#: Synthetic atomic-area support pins (microcosm#932): the digests the
#: synthetic prepared build binds as its geography assignment and the ones
#: :func:`patch_support_register` registers, so a synthetic release candidate
#: passes the register check the way a real one passes on ``sources.yaml``.
#: They are the ``RELEASE_PINS`` digests.
SYNTHETIC_SUPPORT_PINS = {
    "uk_ew_output_area_2021": {"sha256": "c" * 64, "size_bytes": 1828698},
    "uk_scotland_output_area_2022": {"sha256": "d" * 64, "size_bytes": 440439},
    "uk_ni_data_zone_2021": {"sha256": "e" * 64, "size_bytes": 44638},
}


def synthetic_geography_binding() -> dict:
    """The request's geography binding as ``_prepare_geography`` shapes it."""
    return {
        "assignment": "atomic",
        "definition_sha256": "f" * 64,
        "support_pins": {k: dict(v) for k, v in SYNTHETIC_SUPPORT_PINS.items()},
        "identity": "geography_household_key",
        "stream": ["sha256-u53-v1", "uk-post-clone-atomic-area-v1"],
    }


def patch_support_register(monkeypatch, pins=None) -> None:
    """Register ``pins`` (default the synthetic ones) in place of sources.yaml."""
    from microcosm.build.uk_runtime import country_adapter

    register = {
        k: dict(v)
        for k, v in (SYNTHETIC_SUPPORT_PINS if pins is None else pins).items()
    }
    monkeypatch.setattr(
        country_adapter, "uk_atomic_support_register", lambda spec=None: register
    )


STEM = "microcosm_uk_2024_25_local"
SUPPORT_ARGUMENTS = (
    "--atomic-support-ew",
    "supports/ew.npz",
    "--atomic-support-scotland",
    "supports/scotland.npz",
    "--atomic-support-ni",
    "supports/ni.npz",
)
#: Synthetic support digests: a release candidate must pin all three.
RELEASE_PINS = (
    "--atomic-support-sha256-ew",
    "c" * 64,
    "--atomic-support-sha256-scotland",
    "d" * 64,
    "--atomic-support-sha256-ni",
    "e" * 64,
)

_TEST_PATHS = paths_for("microcosm-build")

# The real parser, bound before any test patches ``cli.parse_args`` to return
# one prepared namespace: a test that runs ``main`` twice must still parse
# its second request rather than receive the first one's.
_PARSE_ARGS = cli.parse_args


def _placeholder(path: Path, payload: bytes) -> str:
    if not path.exists():
        path.write_bytes(payload)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arguments(
    tmp_path,
    *extra,
    role="dense",
    staging="--no-staging",
    supports=SUPPORT_ARGUMENTS,
    release_pins=None,
):
    """A dense request over stand-in input files, with staging disabled.

    The pins are the stand-ins' real digests so the validator and a real
    preparation would both accept them; the Ledger pins are synthetic
    because these tests never compile targets. ``staging=None`` passes no
    staging switch at all, which is the remote (``local_and_remote``) mode.
    The atomic-area supports are unread stand-in paths (these tests stub the
    preparation) and a ``--release-candidate`` request gets the synthetic
    support pins unless ``release_pins`` says otherwise.
    """
    if release_pins is None:
        release_pins = RELEASE_PINS if "--release-candidate" in extra else ()
    spine = tmp_path / "spine.h5"
    ladder = tmp_path / "ladder.npz"
    return _PARSE_ARGS(
        [
            "--release-role",
            role,
            "--input-h5",
            str(spine),
            "--input-sha256",
            _placeholder(spine, b"spine stand-in"),
            "--ladder",
            str(ladder),
            "--ladder-sha256",
            _placeholder(ladder, b"ladder stand-in"),
            *supports,
            *release_pins,
            "--ledger-facts",
            str(tmp_path / "ledger"),
            "--ledger-facts-sha256",
            PIN,
            "--ledger-manifest-sha256",
            PIN,
            "--out",
            str(tmp_path / "out"),
            *(() if staging is None else (staging,)),
            *extra,
        ]
    )


def local_ref(path: Path) -> str:
    """The Logbook receipt reference of a file, as the driver writes it."""
    return local_artifact_reference(path, repository_hint=cli.REPOSITORY)


def spool_rows(out: Path):
    rows = load_spool_rows(out / "logbook-spool")
    for row in rows:
        assert frozenset(row.to_mapping()) == LOGBOOK_ROW_FIELDS
    return rows


def single_run_id(out: Path) -> str:
    runs = sorted(path.name for path in (out / "staging" / "runs").iterdir())
    assert len(runs) == 1, runs
    return runs[0]


def load_tool(name: str):
    """Execute ``tools/<name>.py`` as a private module copy (its seams patchable)."""
    root = _TEST_PATHS.repository
    spec = importlib.util.spec_from_file_location(name, root / "tools" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _national_argv(tmp_path, *extra):
    return [
        "--release-role",
        "national",
        "--input-h5",
        str(tmp_path / "spine.h5"),
        "--input-sha256",
        PIN,
        "--out",
        str(tmp_path / "out"),
        "--ledger-facts",
        str(tmp_path / "ledger"),
        "--ledger-facts-sha256",
        PIN,
        "--ledger-manifest-sha256",
        PIN,
        "--no-staging",
        *extra,
    ]


SELECTION = {
    "schema": "microcosm.calibrate.target-selection.v1",
    "selector": {"geography_levels": None, "explicit": False},
    "included": [
        {"name": "count", "period": 2025, "geography_level": "country"},
        {"name": "local", "period": 2025, "geography_level": "constituency"},
    ],
    "excluded": [],
}


def failure_lines(failed) -> dict[str, str]:
    """Gate ids to fail with their failure line.

    ``failed`` is one gate id (line ``"synthetic failure"``), a tuple of
    ``(gate_id, line)`` pairs, or ``None``; the pair form rides on a node
    parameter (tuples of strings), so a synthetic evidence node can fail
    several gates with distinct lines.
    """
    if failed is None:
        return {}
    if isinstance(failed, str):
        return {failed: "synthetic failure"}
    return {str(gate_id): str(line) for gate_id, line in failed}


def gate_payload(phase, failed=None):
    gates = uk_full_gate_manifest(SELECTION)
    lines = failure_lines(failed)
    report = GatePhaseReport(
        phase,
        tuple(
            GateOutcome(
                entry,
                GateStatus.FAILED if entry.id in lines else GateStatus.PASSED,
                GateResult(
                    name=entry.id,
                    passed=entry.id not in lines,
                    details={},
                    failures=(lines[entry.id],) if entry.id in lines else (),
                ),
            )
            for entry in gates.gates
            if entry.phase == phase
        ),
    )
    return canonical_json(
        {
            "schema_version": 1,
            "kind": "uk_full_gate_report",
            "selection_receipt": SELECTION,
            "sample_fraction": 1.0,
            "release_candidate": False,
            "report": gate_phase_report_payload(report, gates=gates),
            "enforcement": classify_full_gate_outcomes(
                report, sample_fraction=1.0, release_candidate=False
            ),
        }
    )


LADDER_TARGET = "ons.census.households@E14000001"

#: The synthetic measures node's engine resolution, as the problem bindings
#: carry it: ``run_dense_main`` sets the block count from ``--engine-blocks``;
#: tests flip ``ENGINE_POPULATION_EXACT`` to stage an inexact per-block run.
ENGINE_BLOCKS = 1
ENGINE_POPULATION_EXACT = True


def measure_resolution_evidence() -> dict:
    evidence: dict = {"blocks": ENGINE_BLOCKS}
    if ENGINE_BLOCKS > 1:
        evidence["engine_population_representation"] = {
            "mode": "block_weights_scaled_to_pool",
            "exact": bool(ENGINE_POPULATION_EXACT),
            "blocks": ENGINE_BLOCKS,
            "factor_by_block": {
                str(index): float(ENGINE_BLOCKS) for index in range(ENGINE_BLOCKS)
            },
        }
    return evidence


def problem_payload() -> bytes:
    """One ordered local problem over the two fixture households."""
    frame = _frame()
    targets = TargetSet(
        [
            Target(
                LADDER_TARGET,
                "household",
                lambda f: np.ones(f.n("household")),
                100.0,
                period=2025,
            )
        ]
    )
    return encode_problem(
        build_constraint_matrix(frame, targets, weight_entity="household"),
        entity_ids=frame.table("household")["household_id"].tolist(),
        target_metadata=[
            {
                "materialization": "uk_local_surface",
                "geography_level": "constituency",
                "geography_id": "E14000001",
                "family": "census_households",
                "source": "fixture",
            }
        ],
        bindings={
            "mass_reason": "fixture selected constraints",
            "max_weight_ratio": 10.0,
            "target_loss_weights": [1.0],
            "target_loss_cap": 10.0,
            "bound_families": ["census_households/constituency"],
            "binding_adjudications": {
                "stood_on": {"census_households/constituency": ["fixture"]}
            },
            "rung_surface": {"dropped_cells": 0},
            "measure_resolution": measure_resolution_evidence(),
            "cross_geography": {
                "unbound_bridges": [],
                "empty_legs_licensed": [],
                "census_household_uprating": uprating_receipt(),
            },
            "measure_exclusions": measure_exclusions(),
            "calibration_year": 2025,
        },
    )


def uprating_receipt() -> dict:
    def hold(name: str) -> dict:
        return {
            "target_name": name,
            "geography_level": "constituency",
            "from_period": 2021,
            "to_period": 2025,
            "attempted": True,
            "eligible": True,
            "applied": True,
            "skipped": False,
            "reason": "census_vintage_hold_uprated",
        }

    cells = {
        "applied": True,
        "cells": 1,
        "total_cells": 1,
        "attempted_cells": 1,
        "eligible_cells": 1,
        "skipped_cells": 0,
    }
    return {
        "applied": True,
        "grains": {
            "constituency": {"factor": 1.0335597414671631},
            "local_authority": {"factor": 1.0335595204737118},
        },
        "household_cells": {**cells, "holds": [hold(LADDER_TARGET)]},
        "tenure_cells": {
            **cells,
            "holds": [hold("ons.tenure.owned_outright@E14000001")],
        },
    }


#: The synthetic approval window. Release validation compares it with the
#: real date, so a real-looking month-long window made every test that
#: assembles this candidate fail once it lapsed (2026-10-04). The window is
#: synthetic, so it does not lapse; expiry itself is tested with explicit
#: dates (test_assembler_rejects_expired_or_missing_measure_approval).
SYNTHETIC_APPROVAL = {"approved_on": "2026-09-03", "expires_on": "2099-12-31"}


def measure_exclusions() -> dict:
    return {
        "obr.housing_benefit": {
            "reason": "synthetic gap",
            "tracking": "microcosm#869",
            "approved_by": "synthetic_reviewer",
            "adjudication": "synthetic decision",
            **SYNTHETIC_APPROVAL,
        }
    }


def surface_payload() -> bytes:
    return canonical_json(
        {
            "local_registry": registry_payload(TargetRegistry([], country="uk")),
            "census_household_uprating": uprating_receipt(),
            "household_dispersion": {"constituency": {"max_ratio": 1.0}},
            "ladder_provenance": {"constituency": "2024_pcon"},
            "measure_exclusions": measure_exclusions(),
            "source_validation": {
                "targets": {
                    "chronicle": {
                        "path_name": "chronicle-uk-artifact-fixture",
                        "facts_sha256": PIN,
                        "manifest_sha256": PIN,
                        "fact_row_count": 1,
                        "schema_version": "v1",
                    },
                    "paired_ladder_sha256": PIN,
                }
            },
            "calibration_year": 2025,
        }
    )


def diagnostics_payload() -> bytes:
    # A legacy-schema stand-in, like the rowwise-tool candidate fixture in the
    # assembler tests: the assembler validates only the current schema
    # (microcosm#1007), which the real gate kernel emits; this fake drives the
    # driver's plumbing, not the diagnostics contract.
    return canonical_json(
        {
            "schema_version": 6,
            "n_records": 2,
            "n_nonzero": 2,
            "initial_loss": 0.5,
            "final_loss": 0.01,
            "realized_max_weight_ratio": 1.0,
            "fraction_within_10pct": 1.0,
            "past_cap_census": {
                "n_targets": 1,
                "initial_past_cap": 0,
                "final_past_cap": 0,
                "escaped": 0,
                "frozen": 0,
                "pushed_out": 0,
            },
            "targets": [
                {
                    "name": f"{LADDER_TARGET}@2025",
                    "target": 100.0,
                    "compiled_target": 100.0,
                    "initial_estimate": 90.0,
                    "final_estimate": 100.0,
                }
            ],
            "target_registry": {"country": "uk", "specs": []},
            "uk_diagnostics": {
                "weights": {"n_nonzero": 2},
                "weakest_families": [],
                "weakest_areas_by_fit": {},
                "rotated_holdout": holdout_payload_dict(),
            },
        }
    )


def holdout_payload_dict() -> dict:
    return {
        "report_only": True,
        "method": "rotated_folds",
        "n_folds": 5,
        "mean_holdout_loss": 0.2,
        "worst_holdout_loss": 0.3,
        "folds": [],
    }


TARGET_ROWS_CSV = (
    "name,target_name,target,estimate,relative_error,abs_relative_error,family,"
    "geography_level,area_type,area_code,metric\n"
    f"{LADDER_TARGET}@2025,{LADDER_TARGET}@2025,100.0,100.0,0.0,0.0,"
    "census_households,constituency,constituency,E14000001,households\n"
).encode()

SUPPORT_CSV = (
    b"area_code,assigned_households,nonzero_households,nonzero_source_households,"
    b"weight_sum,max_weight,effective_sample_size,geography_level\n"
    b"E14000001,1,1,1,13.0,13.0,1.0,constituency\n"
    b"E14000002,1,1,1,87.0,87.0,1.0,constituency\n"
    b"E09000001,1,1,1,13.0,13.0,1.0,local_authority\n"
    b"E07000008,1,1,1,87.0,87.0,1.0,local_authority\n"
)


class Fixture(KernelBase):
    ref = "uk.test.cli-frame@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC, structural=StructuralDelta.CREATE
    )

    def implementation_hash(self):
        return hashlib.sha256(self.ref.encode()).hexdigest()

    def run(self, context):
        return KernelResult(frame=_frame())


class Evidence(KernelBase):
    ref = "uk.test.cli-evidence@1"
    capabilities = Capabilities(Determinism.DETERMINISTIC)

    def implementation_hash(self):
        return hashlib.sha256(self.ref.encode()).hexdigest()

    def run(self, context):
        phase = context.params["phase"]
        artifacts = {"gate_report": gate_payload(phase, context.params.get("failed"))}
        if phase == "terminal":
            artifacts.update(
                calibration_diagnostics=diagnostics_payload(),
                target_diagnostics_csv=TARGET_ROWS_CSV,
                area_support_csv=SUPPORT_CSV,
            )
        return KernelResult(artifacts=artifacts)


class Holdout(Evidence):
    ref = "uk.test.cli-holdout@1"

    def run(self, context):
        return KernelResult(
            artifacts={
                "holdout": canonical_json(holdout_payload_dict()),
                "selection": canonical_json(
                    {"registry": {"country": "uk", "specs": []}, "receipt": SELECTION}
                ),
            }
        )


class Targets(Evidence):
    """The compiled surface and the ordered problem the projection reads."""

    ref = "uk.test.cli-targets@1"

    def run(self, context):
        return KernelResult(
            artifacts={"surface": surface_payload(), "problem": problem_payload()}
        )


class Population(Evidence):
    """The sampling receipt and the geography gate of the pool."""

    ref = "uk.test.cli-population@1"

    def run(self, context):
        return KernelResult(
            artifacts={
                "sampling": canonical_json(
                    {
                        "receipt": {
                            "fraction": 1.0,
                            "seed": 578,
                            "sampled": False,
                            "pre_household_count": 2,
                            "post_household_count": 2,
                            "rung_token": "f100",
                        }
                    }
                ),
                "gate": canonical_json(
                    {
                        "name": "uk_local_geography_ladder",
                        "passed": True,
                        "failures": [],
                        "details": {},
                    }
                ),
            }
        )


class Certification(Evidence):
    ref = "uk.test.cli-certification@1"

    def run(self, context):
        return KernelResult(
            artifacts={
                "certification_readiness": b'{"fixture":true,"release_authorized":false}'
            }
        )


def patch_certification(monkeypatch) -> None:
    """Replace the scientific certification node with the synthetic one."""

    def append(graph, *, population, **kwargs):
        return replace(
            graph,
            nodes=(
                *graph.nodes,
                Node(
                    "uk.full.certification",
                    Certification.ref,
                    population=population,
                    artifact_outputs=(
                        ArtifactOutput(
                            "certification_readiness", FULL_CERTIFICATION_TYPE
                        ),
                    ),
                ),
            ),
        )

    monkeypatch.setattr(cli, "append_uk_full_certification_node", append)


@pytest.fixture(autouse=True)
def certification_service_fixture(monkeypatch):
    # Scientific certification validation has its own graph-artifact tests.
    # This suite tests the filesystem/execution service with synthetic evidence.
    patch_certification(monkeypatch)


def run_dense_main(
    tmp_path,
    monkeypatch,
    *extra,
    staging="--no-staging",
    failed=None,
    on_prepare=None,
    build=None,
) -> tuple[int, Path]:
    """Run the synthetic dense build through ``main``; return its status and bundle.

    ``failed`` fails those gates on the synthetic evidence nodes (see
    :func:`failure_lines`); ``on_prepare(args, telemetry)`` runs inside the
    patched preparation, where the driver's own solve observer can be driven
    with synthetic epochs (the synthetic graph has no calibration kernel);
    ``build`` replaces the synthetic prepared build (a :func:`prepared` with
    an altered configuration, say).
    """
    monkeypatch.delenv("POPULACE_LOGBOOK_PREV_ROW_DIGEST", raising=False)
    patch_certification(monkeypatch)
    args = arguments(tmp_path, *extra, staging=staging)
    build = prepared(tmp_path, failed) if build is None else build

    def prepare(args, *, telemetry=None, attempt=None):
        if on_prepare is not None:
            on_prepare(args, telemetry)
        return build

    monkeypatch.setattr(cli, "parse_args", lambda argv: args)
    monkeypatch.setattr(cli, "prepare_full_build", prepare)
    monkeypatch.setattr(sys.modules[__name__], "ENGINE_BLOCKS", int(args.engine_blocks))
    return cli.main([]), args.out


def graph_dense_bundle(tmp_path, monkeypatch, *extra, staging="--no-staging") -> Path:
    """Run the synthetic dense build through ``main`` and return its bundle.

    The other UK test modules feed the resulting ``rowwise_candidate_manifest.json``
    to the release pre-flight and the dense assembler.
    """
    status, out = run_dense_main(tmp_path, monkeypatch, *extra, staging=staging)
    assert status == 0
    return out


def prepared(tmp_path, failed=None):
    frame = _frame()
    fixture = tmp_path / "fixture.txt"
    fixture.write_text("constant source")
    identifiers = {
        "person_id",
        "person_household_id",
        "person_benunit_id",
        "household_id",
        "benunit_id",
    }
    root = Node(
        "uk.full.calibrated",
        Fixture.ref,
        structural=StructuralDelta.CREATE,
        sources=("fixture",),
        outputs=tuple(
            Owned(
                e,
                str(c),
                "string"
                if frame.table(e)[c].dtype.kind in "OUS"
                else str(frame.table(e)[c].dtype),
            )
            for e in frame.entities
            for c in frame.table(e).columns
            if c not in identifiers
        ),
    )
    nodes = [
        root,
        Node(
            "uk.full.gates.preflight",
            Evidence.ref,
            population=root.id,
            params={"phase": "preflight", "failed": failed},
            artifact_outputs=(ArtifactOutput("gate_report", FULL_GATE_REPORT_TYPE),),
        ),
        Node(
            "uk.full.gates.calibrated",
            Evidence.ref,
            population=root.id,
            params={"phase": "terminal", "failed": failed},
            artifact_outputs=(
                ArtifactOutput("gate_report", FULL_GATE_REPORT_TYPE),
                ArtifactOutput("calibration_diagnostics", FULL_DIAGNOSTICS_TYPE),
                ArtifactOutput("target_diagnostics_csv", FULL_DIAGNOSTICS_CSV_TYPE),
                ArtifactOutput("area_support_csv", FULL_SUPPORT_CSV_TYPE),
            ),
        ),
        Node(
            "uk.full.holdout",
            Holdout.ref,
            population=root.id,
            artifact_outputs=(
                ArtifactOutput("holdout", FULL_HOLDOUT_TYPE),
                ArtifactOutput("selection", TARGET_SELECTION_TYPE),
            ),
        ),
        # CLI materialization consumes the public target-selection endpoint.
        Node(
            "uk.full.target_selection",
            Holdout.ref,
            population=root.id,
            artifact_outputs=(
                ArtifactOutput("holdout", FULL_HOLDOUT_TYPE),
                ArtifactOutput("selection", TARGET_SELECTION_TYPE),
            ),
        ),
        # The manifest projection reads the compiled surface, the ordered
        # problem, the sampling receipt and the geography gate.
        Node(
            "uk.full.target_compilation",
            Targets.ref,
            population=root.id,
            artifact_outputs=(
                ArtifactOutput("surface", TARGET_SURFACE_TYPE),
                ArtifactOutput("problem", PROBLEM_TYPE),
            ),
        ),
        Node(
            "uk.full.problem",
            Targets.ref,
            population=root.id,
            artifact_outputs=(
                ArtifactOutput("surface", TARGET_SURFACE_TYPE),
                ArtifactOutput("problem", PROBLEM_TYPE),
            ),
        ),
        Node(
            "uk.full.sample",
            Population.ref,
            population=root.id,
            artifact_outputs=(
                ArtifactOutput("sampling", POPULATION_RECEIPT_TYPE),
                ArtifactOutput("gate", GEOGRAPHY_GATE_TYPE),
            ),
        ),
        Node(
            "uk.full.geography_gate",
            Population.ref,
            population=root.id,
            artifact_outputs=(
                ArtifactOutput("sampling", POPULATION_RECEIPT_TYPE),
                ArtifactOutput("gate", GEOGRAPHY_GATE_TYPE),
            ),
        ),
    ]
    nodes = [
        replace(
            node,
            artifact_inputs=(
                ArtifactInput(
                    "preflight",
                    "uk.full.gates.preflight",
                    "gate_report",
                    FULL_GATE_REPORT_TYPE,
                ),
                ArtifactInput(
                    "holdout", "uk.full.holdout", "holdout", FULL_HOLDOUT_TYPE
                ),
                ArtifactInput(
                    "selection",
                    "uk.full.target_selection",
                    "selection",
                    TARGET_SELECTION_TYPE,
                ),
                ArtifactInput(
                    "surface",
                    "uk.full.target_compilation",
                    "surface",
                    TARGET_SURFACE_TYPE,
                ),
                ArtifactInput("problem", "uk.full.problem", "problem", PROBLEM_TYPE),
                ArtifactInput(
                    "sampling", "uk.full.sample", "sampling", POPULATION_RECEIPT_TYPE
                ),
                ArtifactInput(
                    "geography_gate",
                    "uk.full.geography_gate",
                    "gate",
                    GEOGRAPHY_GATE_TYPE,
                ),
            ),
        )
        if node.id == "uk.full.gates.calibrated"
        else node
        for node in nodes
    ]
    graph = Graph("uk", (SourceRef("fixture", "raw-bytes-v1"),), tuple(nodes))
    graph = add_uk_export_preparation(
        graph,
        population=root.id,
        bindings={"target_scope": "all", "geography": synthetic_geography_binding()},
        artifact_inputs=(
            ArtifactInput(
                "gates",
                "uk.full.gates.calibrated",
                "gate_report",
                FULL_GATE_REPORT_TYPE,
            ),
        ),
    )
    calibration = UKCalibrationNodes(
        (), root.id, "unused", "unused", "unused", None, "unused"
    )
    full = UKFullGraph(graph, calibration, UKFullBuildConfig(calibration_year=2025))
    kernels = KernelRegistry()
    for kernel in (
        Fixture(),
        Evidence(),
        Holdout(),
        Targets(),
        Population(),
        Certification(),
    ):
        kernels.register(kernel)
    register_uk_terminal_kernels(kernels)
    spine = tmp_path / "spine.h5"
    ladder = tmp_path / "ladder.npz"
    pins = {
        "dataset": {
            "sha256": _placeholder(spine, b"spine stand-in"),
            "size_bytes": spine.stat().st_size,
        },
        "ladder": {
            "sha256": _placeholder(ladder, b"ladder stand-in"),
            "size_bytes": ladder.stat().st_size,
        },
    }
    inputs = {
        name: {
            "path": str(path.resolve()),
            "sha256": pins[name]["sha256"],
            "bytes": pins[name]["size_bytes"],
            "pin_verified": True,
        }
        for name, path in (("dataset", spine), ("ladder", ladder))
    }
    return cli.PreparedUKFullBuild(
        full,
        kernels,
        {"fixture": fixture},
        {"target_scope": "all"},
        pins=pins,
        inputs=inputs,
    )


__all__ = [name for name in globals() if not name.startswith("__")]
