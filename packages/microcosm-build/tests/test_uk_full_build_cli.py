"""The canonical CLI restores declared files and preserves failure/scope semantics."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
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

PIN = "0" * 64
STEM = "microcosm_uk_2024_25_local"


def _placeholder(path: Path, payload: bytes) -> str:
    if not path.exists():
        path.write_bytes(payload)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arguments(tmp_path, *extra, role="dense", staging="--no-staging"):
    """A dense request over stand-in input files, with staging disabled.

    The pins are the stand-ins' real digests so the validator and a real
    preparation would both accept them; the Ledger pins are synthetic
    because these tests never compile targets.
    """
    spine = tmp_path / "spine.h5"
    ladder = tmp_path / "ladder.npz"
    return cli.parse_args(
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
            "--ledger-facts",
            str(tmp_path / "ledger"),
            "--ledger-facts-sha256",
            PIN,
            "--ledger-manifest-sha256",
            PIN,
            "--out",
            str(tmp_path / "out"),
            staging,
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


def test_release_role_is_required_and_supplies_the_dense_posture(tmp_path):
    with pytest.raises(SystemExit):
        cli.parse_args(["--input-h5", "x.h5", "--out", str(tmp_path)])
    args = arguments(tmp_path)
    posture = cli.UK_ROWWISE_DENSE_POSTURE
    assert args._posture is posture
    assert (args.n_clones, args.seed, args.epochs, args.learning_rate) == (
        posture.clone_count,
        posture.seed,
        posture.epochs,
        posture.learning_rate,
    )
    assert (args.n_clones, args.epochs, args.learning_rate) == (15, 1500, 0.15)
    assert args.target_weight_rule == "grain_equal"
    assert args.expected_constituency_vintage == "2024_pcon"
    assert args.sample_seed == 578
    assert args.staging_upload_interval_seconds == 300.0
    assert args._explicit_arguments == frozenset()
    cli.validate_cli_args(args)
    explicit = arguments(tmp_path, "--epochs", "8", "--n-clones", "2")
    assert (explicit.epochs, explicit.n_clones) == (8, 2)
    assert explicit._explicit_arguments == frozenset({"epochs", "n_clones"})


@pytest.mark.parametrize(
    ("extra", "needle"),
    [
        (["--target-loss-cap", "5"], "--target-loss-cap"),
        (["--allow-unpinned-feed"], "--allow-unpinned-feed"),
        (["--incumbent-h5", "incumbent.h5"], "--incumbent-h5"),
        (["--target-weight-rule", "family_equal"], "--target-weight-rule family_equal"),
    ],
)
def test_dense_role_refuses_the_national_knobs(tmp_path, extra, needle):
    args = arguments(tmp_path, *extra)
    with pytest.raises(ValueError, match="--release-role dense refuses") as excinfo:
        cli.validate_cli_args(args)
    assert needle in str(excinfo.value)


def test_dense_role_requires_the_ladder_and_the_pins(tmp_path):
    argv = [
        "--release-role",
        "dense",
        "--input-h5",
        str(tmp_path / "spine.h5"),
        "--out",
        str(tmp_path / "out"),
        "--ledger-facts",
        str(tmp_path / "ledger"),
        "--ledger-facts-sha256",
        PIN,
        "--ledger-manifest-sha256",
        PIN,
    ]
    with pytest.raises(ValueError, match="requires --ladder"):
        cli.validate_cli_args(cli.parse_args(argv))
    with pytest.raises(ValueError, match="--input-sha256 and --ladder-sha256"):
        cli.validate_cli_args(
            cli.parse_args([*argv, "--ladder", str(tmp_path / "ladder.npz")])
        )
    # A spine-request build has no input H5 to pin: only the ladder pin is due.
    request = [
        "--release-role",
        "dense",
        "--spine-request",
        str(tmp_path / "request.json"),
        "--ladder",
        str(tmp_path / "ladder.npz"),
        "--ladder-sha256",
        PIN,
        "--out",
        str(tmp_path / "out"),
        "--ledger-facts",
        str(tmp_path / "ledger"),
        "--ledger-facts-sha256",
        PIN,
        "--ledger-manifest-sha256",
        PIN,
    ]
    cli.validate_cli_args(cli.parse_args(request))


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


def test_national_role_is_validated_then_dispatched_to_the_seam(tmp_path, monkeypatch):
    """The national line never prepares a graph (microcosm#901 phase 4).

    The validated request goes to ``national_role.run_national_role`` after
    the role tables have run; the seam's own pre-flight, Logbook, staging
    and manifest live behind that call.
    """
    argv = _national_argv(tmp_path)
    national = cli.parse_args(argv)
    assert national._posture is cli.uk_rowwise_posture("national")
    assert national.n_clones is None
    assert (national.seed, national.epochs, national.learning_rate) == (0, 1500, 0.02)
    cli.validate_cli_args(national)
    served = []
    monkeypatch.setattr(
        cli.national_role, "run_national_role", lambda args: served.append(args) or 7
    )
    monkeypatch.setattr(
        cli,
        "prepare_full_build",
        lambda *a, **k: pytest.fail("the national role prepared a graph"),
    )
    monkeypatch.setattr(
        cli,
        "preflight_staged_dataset",
        lambda args: pytest.fail("the driver pre-flighted before dispatching"),
    )
    assert cli.main(argv) == 7
    (args,) = served
    assert args.release_role == "national"
    assert args._posture is national._posture
    assert not (tmp_path / "out").exists()
    # The dense refusal table still applies before the dispatch.
    with pytest.raises(ValueError, match="--release-role national refuses"):
        cli.main([*argv, "--ladder", str(tmp_path / "ladder.npz")])
    assert served == [args]


def test_national_dry_run_dispatches_to_the_seam_plan(tmp_path, monkeypatch, capsys):
    """``--dry-run`` on the national role plans through ``national_dry_run``:
    no graph, no staged-dataset pre-flight, nothing written."""
    monkeypatch.setattr(
        cli.national_role,
        "national_dry_run",
        lambda args: (
            print(json.dumps({"dry_run": args.dry_run, "role": args.release_role})) or 0
        ),
    )
    monkeypatch.setattr(
        cli,
        "prepare_full_build",
        lambda *a, **k: pytest.fail("the national dry run prepared a graph"),
    )
    monkeypatch.setattr(
        cli.national_role,
        "preflight_staged_dataset",
        lambda args: pytest.fail("a dry run reached the Hub pre-flight"),
    )
    assert cli.main(_national_argv(tmp_path, "--dry-run")) == 0
    assert json.loads(capsys.readouterr().out) == {"dry_run": True, "role": "national"}
    assert not (tmp_path / "out").exists()


def test_national_role_refuses_a_spine_request(tmp_path, monkeypatch):
    """The seam engine reads a bound spine; ``--spine-request`` is dense-only."""
    monkeypatch.setattr(
        cli.national_role,
        "preflight_staged_dataset",
        lambda args: pytest.fail("the refusal must precede the Hub pre-flight"),
    )
    argv = [
        "--release-role",
        "national",
        "--spine-request",
        str(tmp_path / "request.json"),
        "--out",
        str(tmp_path / "out"),
        "--ledger-facts",
        str(tmp_path / "ledger"),
        "--ledger-facts-sha256",
        PIN,
        "--ledger-manifest-sha256",
        PIN,
        "--no-staging",
    ]
    with pytest.raises(ValueError, match="bound spine checkpoint.*--input-h5"):
        cli.main(argv)
    assert not (tmp_path / "out").exists()


def test_candidate_clone_counts_are_dry_run_only(tmp_path):
    with pytest.raises(ValueError, match="only with --dry-run"):
        cli.main(
            [
                "--release-role",
                "dense",
                "--input-h5",
                str(tmp_path / "missing.h5"),
                "--input-sha256",
                PIN,
                "--ladder",
                str(tmp_path / "missing.npz"),
                "--ladder-sha256",
                PIN,
                "--ledger-facts",
                str(tmp_path / "ledger"),
                "--ledger-facts-sha256",
                PIN,
                "--ledger-manifest-sha256",
                PIN,
                "--out",
                str(tmp_path / "out"),
                "--candidate-clone-counts",
                "1,2,4",
                "--no-staging",
            ]
        )


def test_source_sampling_cannot_be_reapplied_as_pool_sampling():
    config = UKFullBuildConfig(calibration_year=2025, source_sample_fraction=0.1)
    assert config.sample_fraction == 1.0
    assert config.effective_sample_fraction == 0.1
    with pytest.raises(ValueError, match="second time"):
        replace(config, sample_fraction=0.1)


def test_households_only_binds_the_census_family_on_the_selection_node():
    config = UKFullBuildConfig(
        calibration_year=2025, target_families=("census_households/constituency",)
    )
    assert config.target_families == ("census_households/constituency",)
    with pytest.raises(ValueError, match="target families"):
        UKFullBuildConfig(calibration_year=2025, target_families=())


SELECTION = {
    "schema": "microcosm.calibrate.target-selection.v1",
    "selector": {"geography_levels": None, "explicit": False},
    "included": [
        {"name": "count", "period": 2025, "geography_level": "country"},
        {"name": "local", "period": 2025, "geography_level": "constituency"},
    ],
    "excluded": [],
}


def gate_payload(phase, failed=None):
    gates = uk_full_gate_manifest(SELECTION)
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
            "measure_resolution": {"blocks": 1},
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


def measure_exclusions() -> dict:
    return {
        "obr.housing_benefit": {
            "reason": "synthetic gap",
            "tracking": "microcosm#869",
            "approved_by": "synthetic_reviewer",
            "adjudication": "synthetic decision",
            "approved_on": "2026-09-03",
            "expires_on": "2026-10-03",
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
    return canonical_json(
        {
            "schema_version": 8,
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


def graph_dense_bundle(tmp_path, monkeypatch, *extra, staging="--no-staging") -> Path:
    """Run the synthetic dense build through ``main`` and return its bundle.

    The other UK test modules feed the resulting ``rowwise_candidate_manifest.json``
    to the release pre-flight and the dense assembler.
    """
    monkeypatch.delenv("POPULACE_LOGBOOK_PREV_ROW_DIGEST", raising=False)
    patch_certification(monkeypatch)
    args = arguments(tmp_path, *extra, staging=staging)
    build = prepared(tmp_path)
    monkeypatch.setattr(cli, "parse_args", lambda argv: args)
    monkeypatch.setattr(cli, "prepare_full_build", lambda args, **kwargs: build)
    assert cli.main([]) == 0
    return args.out


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
    assert expected["release_role"] == "dense"
    assert (out / f"{STEM}.targets.csv").read_text().startswith("name,target_name")
    for path in out.iterdir():
        if path.is_file():
            path.unlink()

    def forbidden(*args):
        raise AssertionError(
            "Required replay repeated a completed numerical/evidence node"
        )

    for kernel in (Fixture, Evidence, Holdout, Targets, Population):
        monkeypatch.setattr(kernel, "run", forbidden)
    args.resume = "require"
    assert cli.execute_full_build(prepared(tmp_path), args) == 0
    actual = json.loads((out / "build.json").read_text())
    assert actual["content_sha256"] == expected["content_sha256"]
    # The output names come from the dense posture and the FRS vintage.
    assert (out / f"{STEM}.h5").is_file()
    assert (out / f"{STEM}.holdout.json").is_file()
    assert (out / f"{STEM}.local_gates.json").is_file()
    assert not (out / "microcosm_uk_2025.h5").exists()


def test_dense_run_projects_the_rowwise_candidate_manifest(tmp_path):
    pytest.importorskip("tables")
    args = arguments(tmp_path, "--release-candidate")
    assert cli.execute_full_build(prepared(tmp_path), args) == 0
    out = args.out
    manifest = json.loads((out / "rowwise_candidate_manifest.json").read_text())
    assert manifest["schema_version"] == 4
    assert manifest["build_kind"] == "uk_rowwise_calibrated_candidate"
    assert manifest["release_role"] == "dense"
    assert manifest["release_id"] == UK_DENSE_RELEASE_ID
    assert manifest["parameters"]["release_role"] == "dense"
    assert manifest["parameters"]["release_candidate"] is True
    assert (
        manifest["parameters"]["doctrine"]
        == cli.UK_ROWWISE_DENSE_POSTURE.doctrine_bounds()
    )
    assert (manifest["parameters"]["n_clones"], manifest["parameters"]["epochs"]) == (
        15,
        1500,
    )
    for key in (
        "solve",
        "fit",
        "weights",
        "cross_grain",
        "census_household_uprating",
        "outputs",
        "releasable",
        "identity",
        "bound_target_families",
        "binding_adjudications",
        "measure_exclusions",
        "blocked_at_f100",
        "blocking_failures",
        "diagnostic_failures",
        "release_gate_failures_not_enforced",
        "failing_gate_ids",
        "sampling",
        "rung_surface",
        "support",
        "geography",
        "vintages",
        "graph",
    ):
        assert key in manifest, key
    assert set(manifest["identity"]) >= {
        "code",
        "ladder",
        "runtime",
        "spine",
        "targets",
    }
    assert manifest["identity"]["spine"]["pin_verified"] is True
    assert manifest["identity"]["ladder"]["pin_verified"] is True
    assert manifest["identity"]["targets"]["paired_ladder_sha256"] == PIN
    assert manifest["solve"]["n_targets_by_kind"] == {
        "local": 0,
        "ladder": 1,
        "national": 0,
    }
    assert manifest["solve"]["measure_resolution"] == {"blocks": 1}
    assert manifest["solve"]["target_weight_rule_override"] == {}
    assert manifest["solve"]["n_households"] == 2
    assert manifest["solve"]["pool_households"] == 2
    assert manifest["fit"]["rotated_holdout"]["n_folds"] == 5
    assert manifest["fit"]["local_by_family"][0]["family"] == "census_households"
    assert manifest["weights"]["realized_max_weight_ratio_vs_design"] == 1.0
    assert manifest["weights"]["household_weight_kind"] == "calibrated"
    assert manifest["census_household_uprating"]["applied"] is True
    assert manifest["releasable"] is True and manifest["blocked_at_f100"] is False
    assert manifest["release_posture"]["release_blocking_gates_passed"] is True
    dataset = manifest["outputs"]["dataset"]
    assert Path(dataset["path"]) == out / f"{STEM}.h5"
    assert (
        dataset["sha256"]
        == hashlib.sha256((out / f"{STEM}.h5").read_bytes()).hexdigest()
    )
    assert Path(manifest["outputs"]["local_gate_report"]["path"]).name == (
        f"{STEM}.local_gates.json"
    )
    assert manifest["graph"]["epoch_rows"] == "dense_solve_only"
    assert "uk.full.problem" in manifest["graph"]["artifacts"]
    # The staged bundle is every registered output plus the manifest; each
    # registered file is on disk beside it with its recorded digest.
    for entry in manifest["outputs"].values():
        path = Path(entry["path"])
        assert path.parent == out and path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]


def test_blocked_gate_projects_an_unreleasable_manifest(tmp_path):
    pytest.importorskip("tables")
    args = arguments(tmp_path)
    build = prepared(tmp_path, "uk_local_target_fit")
    assert cli.execute_full_build(build, args) == 1
    manifest = json.loads((args.out / "rowwise_candidate_manifest.json").read_text())
    assert manifest["releasable"] is False
    assert manifest["blocked_at_f100"] is True
    assert manifest["blocking_failures"] == ["[uk_local_target_fit] synthetic failure"]
    assert manifest["failing_gate_ids"] == ["uk_local_target_fit"]


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
    assert (args.out / f"{STEM}.h5").exists() == exported


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
    monkeypatch.setattr(cli, "prepare_full_build", lambda args, **kwargs: build)
    assert cli.main([]) == 1
    assert not args.out.exists()


def test_main_runs_the_logbook_envelope_around_a_dense_build(tmp_path, monkeypatch):
    pytest.importorskip("tables")
    from microcosm.build.logbook import load_spool_rows

    monkeypatch.delenv("POPULACE_LOGBOOK_PREV_ROW_DIGEST", raising=False)
    args = arguments(tmp_path, "--seed", "7")
    build = prepared(tmp_path)
    monkeypatch.setattr(cli, "parse_args", lambda argv: args)
    monkeypatch.setattr(cli, "prepare_full_build", lambda args, **kwargs: build)
    assert cli.main([]) == 0
    rows = load_spool_rows(args.out / "logbook-spool")
    assert len(rows) == 1
    row = rows[0]
    assert row.pipeline == "uk-local-candidate"
    assert row.build_id.startswith("uk-local-candidate-f100-s7-")
    assert row.rung == "f100" and row.seed == 7
    assert row.disposition == "iterating"
    assert row.artifact_location.endswith(f"{STEM}.h5")
    assert "published" in row.phases_reached
    assert row.gate_verdicts and all(
        item["verdict"] == "passed" and ".local_gates.json#/gates/" in item["receipt"]
        for item in row.gate_verdicts.values()
    )
    manifest = json.loads((args.out / "rowwise_candidate_manifest.json").read_text())
    assert manifest["staging_delivery"]["enabled"] is False
    assert manifest["staged_dataset"]["status"] == "skipped"


def test_main_records_a_failed_attempt_when_the_build_raises(tmp_path, monkeypatch):
    from microcosm.build.logbook import load_spool_rows

    monkeypatch.delenv("POPULACE_LOGBOOK_PREV_ROW_DIGEST", raising=False)
    args = arguments(tmp_path)
    build = prepared(tmp_path)
    monkeypatch.setattr(cli, "parse_args", lambda argv: args)
    monkeypatch.setattr(cli, "prepare_full_build", lambda args, **kwargs: build)
    monkeypatch.setattr(
        cli,
        "run_graph",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("synthetic refusal")),
    )
    assert cli.main([]) == 1
    failure = json.loads((args.out / "failure.json").read_text())
    assert failure["message"] == "synthetic refusal"
    rows = load_spool_rows(args.out / "logbook-spool")
    assert len(rows) == 1 and rows[0].disposition == "failed"
    receipts = list((args.out / "logbook-receipts").rglob("error.json"))
    assert len(receipts) == 1
    assert json.loads(receipts[0].read_text())["error_type"].endswith("RuntimeError")


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
    monkeypatch.setattr(cli, "prepare_full_build", lambda args, **kwargs: build)
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


def test_main_stages_the_bundle_locally_with_staging_local_only(tmp_path, monkeypatch):
    """``--staging-local-only`` writes and validates the v2 bundle, uploads nothing."""

    pytest.importorskip("tables")
    from microcosm.build.staging_dataset import (
        SHA256SUMS_FILENAME,
        STAGED_MANIFEST_FILENAME,
    )
    from microcosm.build.staging_v2 import validate_v2_bundle

    staging_dir = tmp_path / "staging-bundle"
    out = graph_dense_bundle(
        tmp_path,
        monkeypatch,
        "--staging-dir",
        str(staging_dir),
        staging="--staging-local-only",
    )
    manifest = json.loads((out / "rowwise_candidate_manifest.json").read_text())
    assert manifest["staging_delivery"]["mode"] == "local_only"
    assert manifest["staging_delivery"]["upload_attempts"] == 0
    assert manifest["staged_dataset"]["mode"] == "local_only"
    assert manifest["staged_dataset"]["status"] == "skipped"
    assert manifest["staged_dataset"]["repository"] is None
    # The published bundle carries its own inventory beside the manifest.
    assert (out / STAGED_MANIFEST_FILENAME).is_file()
    assert (out / SHA256SUMS_FILENAME).is_file()

    runs = sorted(path.name for path in (staging_dir / "runs").iterdir())
    assert len(runs) == 1
    bundle = validate_v2_bundle(staging_dir, runs[0])
    assert bundle["progress"]["status"] == "completed"
    assert bundle["run_manifest"]["run_kind"] == "calibration"
    transitions = [
        event["status"]
        for event in bundle["events"]
        if event["stage_id"] == "dataset_staging"
    ]
    assert transitions == ["started", "completed"]
    artifacts = staging_dir / "runs" / runs[0] / "artifacts"
    staged = json.loads((artifacts / "staged_dataset.json").read_text())
    assert staged["mode"] == "local_only" and staged["status"] == "skipped"
    assert (artifacts / "fit_summary.json").is_file()
