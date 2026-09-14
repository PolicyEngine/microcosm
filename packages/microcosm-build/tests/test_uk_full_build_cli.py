"""The canonical CLI restores declared files and preserves failure/scope semantics."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from test_uk_graph_terminal import _frame

from microcosm.build.gate_battery import (
    GateOutcome,
    GatePhaseReport,
    GateStatus,
    gate_phase_report_payload,
)
from microcosm.build.gates import GateResult
from microcosm.build.uk_runtime import full_build_cli as cli
from microcosm.build.uk_runtime.full_certification import FULL_CERTIFICATION_TYPE
from microcosm.build.uk_runtime.full_gates import (
    classify_full_gate_outcomes,
    uk_full_gate_manifest,
)
from microcosm.build.uk_runtime.graph_build import UKFullBuildConfig, UKFullGraph
from microcosm.build.uk_runtime.graph_calibration import UKCalibrationNodes
from microcosm.build.uk_runtime.graph_targets import TARGET_SELECTION_TYPE
from microcosm.build.uk_runtime.graph_terminal import (
    FULL_DIAGNOSTICS_CSV_TYPE,
    FULL_DIAGNOSTICS_TYPE,
    FULL_GATE_REPORT_TYPE,
    FULL_HOLDOUT_TYPE,
    FULL_SUPPORT_CSV_TYPE,
    add_uk_export_preparation,
    register_uk_terminal_kernels,
)
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


def arguments(tmp_path, *extra):
    return cli.parse_args(
        [
            "--input-h5",
            str(tmp_path / "spine.h5"),
            "--ladder",
            str(tmp_path / "ladder.npz"),
            "--ledger-facts",
            str(tmp_path / "ledger"),
            "--out",
            str(tmp_path / "out"),
            *extra,
        ]
    )


def test_every_scope_and_size_control_keeps_default_all(tmp_path):
    for extra in (
        (),
        ("--target-geographies", "all"),
        ("--n-clones", "1"),
        ("--dataset-households", "10"),
        ("--n-clones", "1", "--dataset-households", "10"),
    ):
        assert arguments(tmp_path, *extra).target_geographies is None
    assert arguments(
        tmp_path, "--target-geographies", "country"
    ).target_geographies == ("country",)
    with pytest.raises(SystemExit):
        arguments(tmp_path, "--target-geographies", "national")


def test_source_sampling_cannot_be_reapplied_as_pool_sampling():
    config = UKFullBuildConfig(calibration_year=2025, source_sample_fraction=0.1)
    assert config.sample_fraction == 1.0
    assert config.effective_sample_fraction == 0.1
    with pytest.raises(ValueError, match="second time"):
        replace(config, sample_fraction=0.1)


def gate_payload(phase, failed=None):
    selection = {
        "schema": "microcosm.calibrate.target-selection.v1",
        "selector": {"geography_levels": None, "explicit": False},
        "included": [
            {"name": "count", "period": 2025, "geography_level": "country"},
            {"name": "local", "period": 2025, "geography_level": "constituency"},
        ],
        "excluded": [],
    }
    gates = uk_full_gate_manifest(selection)
    report = GatePhaseReport(
        phase,
        tuple(
            GateOutcome(
                entry,
                GateStatus.FAILED if entry.id == failed else GateStatus.PASSED,
                GateResult(
                    name=entry.id,
                    passed=entry.id != failed,
                    details={},
                    failures=("synthetic failure",) if entry.id == failed else (),
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
            "selection_receipt": selection,
            "sample_fraction": 1.0,
            "release_candidate": False,
            "report": gate_phase_report_payload(report, gates=gates),
            "enforcement": classify_full_gate_outcomes(
                report, sample_fraction=1.0, release_candidate=False
            ),
        }
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
                calibration_diagnostics=b'{"fixture":true}',
                target_diagnostics_csv=b"name,actual\ncount,100\n",
                area_support_csv=b"area,households\nfixture,2\n",
            )
        return KernelResult(artifacts=artifacts)


class Holdout(Evidence):
    ref = "uk.test.cli-holdout@1"

    def run(self, context):
        return KernelResult(
            artifacts={"holdout": b'{"fixture":true}', "selection": b'{"fixture":true}'}
        )


class Certification(Evidence):
    ref = "uk.test.cli-certification@1"

    def run(self, context):
        return KernelResult(
            artifacts={
                "certification_readiness": b'{"fixture":true,"release_authorized":false}'
            }
        )


@pytest.fixture(autouse=True)
def certification_service_fixture(monkeypatch):
    # Scientific certification validation has its own graph-artifact tests.
    # This suite tests the filesystem/execution service with synthetic evidence.
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
    ]
    # CLI materialization consumes the public target-selection endpoint.
    nodes.append(
        Node(
            "uk.full.target_selection",
            Holdout.ref,
            population=root.id,
            artifact_outputs=(
                ArtifactOutput("holdout", FULL_HOLDOUT_TYPE),
                ArtifactOutput("selection", TARGET_SELECTION_TYPE),
            ),
        )
    )
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
        bindings={"target_scope": "all"},
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
    for kernel in (Fixture(), Evidence(), Holdout(), Certification()):
        kernels.register(kernel)
    register_uk_terminal_kernels(kernels)
    return cli.PreparedUKFullBuild(
        full, kernels, {"fixture": fixture}, {"target_scope": "all"}
    )


def test_cli_cold_and_required_replay_recreate_dataset_and_sidecars(
    tmp_path, monkeypatch
):
    pytest.importorskip("tables")
    args = arguments(tmp_path)
    first = prepared(tmp_path)
    assert cli.execute_full_build(first, args) == 0
    out = args.out
    expected = json.loads((out / "build.json").read_text())
    assert expected["readback_passed"] is True
    assert expected["release_authorized"] is False
    assert (out / "microcosm_uk_2025.targets.csv").read_text().startswith("name,actual")
    for path in out.iterdir():
        if path.is_file():
            path.unlink()

    def forbidden(*args):
        raise AssertionError(
            "Required replay repeated a completed numerical/evidence node"
        )

    for kernel in (Fixture, Evidence, Holdout):
        monkeypatch.setattr(kernel, "run", forbidden)
    args.resume = "require"
    assert cli.execute_full_build(prepared(tmp_path), args) == 0
    actual = json.loads((out / "build.json").read_text())
    assert actual["content_sha256"] == expected["content_sha256"]
    assert (out / "microcosm_uk_2025.h5").is_file()
    assert (out / "microcosm_uk_2025.holdout.json").is_file()


@pytest.mark.parametrize(
    "failure,exported",
    [
        ("uk_local_geography_ladder_post_calibration", False),
        ("uk_local_target_fit", True),
    ],
)
def test_cli_retains_failed_evidence_and_correct_status(tmp_path, failure, exported):
    if exported:
        pytest.importorskip("tables")
    args = arguments(tmp_path)
    build = prepared(tmp_path, failure)
    assert cli.execute_full_build(build, args) == 1
    assert (args.out / "uk.full.gates.calibrated.gate_report.json").is_file()
    assert (args.out / "microcosm_uk_2025.h5").exists() == exported


def test_dry_run_has_no_files_or_kernel_execution(tmp_path, monkeypatch, capsys):
    args = arguments(tmp_path, "--dry-run")
    build = prepared(tmp_path)
    monkeypatch.setattr(
        cli, "run_graph", lambda *a, **k: pytest.fail("dry run executed graph")
    )
    assert cli.execute_full_build(build, args) == 0
    assert json.loads(capsys.readouterr().out)["default_scope"] == "all_geographies"
    assert not args.out.exists()


def test_rejected_output_inside_source_never_writes_failure_sidecar(
    tmp_path, monkeypatch
):
    args = arguments(tmp_path)
    build = prepared(tmp_path)
    build = replace(build, sources={"fixture": tmp_path})
    monkeypatch.setattr(cli, "parse_args", lambda argv: args)
    monkeypatch.setattr(cli, "prepare_full_build", lambda args: build)
    assert cli.main([]) == 1
    assert not args.out.exists()


@pytest.mark.parametrize("resume", ["auto", "require"])
def test_source_phase_evidence_survives_later_exception(tmp_path, monkeypatch, resume):
    args = arguments(tmp_path)
    build = prepared(tmp_path)
    assembled = Node(
        "spine.gates.assembled",
        Evidence.ref,
        population="uk.full.calibrated",
        params={"phase": "preflight", "failed": None},
        artifact_outputs=(ArtifactOutput("gate_report", FULL_GATE_REPORT_TYPE),),
    )
    transferred = replace(assembled, id="spine.gates.transferred")
    graph = replace(
        build.full.graph, nodes=(*build.full.graph.nodes, assembled, transferred)
    )
    build = replace(build, full=replace(build.full, graph=graph))
    args.out.mkdir()
    previous = b'{"kind":"previous-complete-build"}'
    (args.out / "build.json").write_bytes(previous)
    args.resume = resume
    original_run = cli.run_graph
    if resume == "require":
        original_run(
            cli.compile_graph(cli._through(graph, assembled.id)),
            sources=build.sources,
            store=cli.ContentStore(args.out / ".graph-store"),
            kernels=build.kernels,
        )
        monkeypatch.setattr(
            Fixture, "run", lambda *a: pytest.fail("Checkpoint replay reran the source")
        )
        monkeypatch.setattr(
            Evidence, "run", lambda *a: pytest.fail("Checkpoint replay reran the gate")
        )

    def refuse_transferred(compiled, **kwargs):
        if transferred.id in {node.id for node in compiled.graph.nodes}:
            raise RuntimeError("downstream source donor bin is empty")
        return original_run(compiled, **kwargs)

    monkeypatch.setattr(cli, "run_graph", refuse_transferred)
    monkeypatch.setattr(cli, "parse_args", lambda argv: args)
    monkeypatch.setattr(cli, "prepare_full_build", lambda args: build)
    assert cli.main([]) == 1
    assert (args.out / "build.json").read_bytes() == previous
    failure = json.loads((args.out / "failure.json").read_bytes())
    durable = Path(failure["evidence_directory"])
    assert (durable / "spine.gates.assembled.graph.json").is_file()
    assert not (durable / "spine.gates.transferred.graph.json").exists()
    gate_bytes = (durable / "spine.gates.assembled.gate_report.json").read_bytes()
    assert gate_bytes == gate_payload("preflight")
    index = json.loads((durable / "evidence-index.json").read_bytes())
    assert (
        index["artifacts"]["spine.gates.assembled/gate_report"]["sha256"]
        == hashlib.sha256(gate_bytes).hexdigest()
    )
