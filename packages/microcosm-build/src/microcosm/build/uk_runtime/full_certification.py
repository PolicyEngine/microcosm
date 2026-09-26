"""Unsigned certification readiness from one UK graph's identified artifacts.

This consumer does not rerun a battery, solve weights or authorize publication.
Historical split-lane signatures remain verifiable in release_certification;
current builds bind one complete gate roster and the selected target scope.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    ArtifactType,
    Capabilities,
    ContentStore,
    Determinism,
    Graph,
    KernelBase,
    KernelContext,
    KernelRegistry,
    KernelResult,
    Node,
    Numeric,
    RunManifest,
    SeedSource,
    SourceRef,
    source_hash,
)
from microcosm.graph.canonical import canonical_json

from ..artifact_files import file_artifact, materialize_bytes
from ..country_spec import load_country_spec
from ..gate_battery import GateStatus, gate_phase_report_from_payload
from .full_gates import uk_full_gate_manifest, uk_full_gate_scope_receipt
from .graph_evidence import SPINE_GATE_REPORT_TYPE, uk_spine_gate_manifest
from .graph_targets import TARGET_SELECTION_TYPE, TARGET_SURFACE_TYPE
from .graph_terminal import (
    EXPORT_DESCRIPTOR_TYPE,
    EXPORT_READBACK_TYPE,
    FULL_DIAGNOSTICS_TYPE,
    FULL_GATE_REPORT_TYPE,
    FULL_HOLDOUT_TYPE,
    PACKAGE_INVENTORY_TYPE,
    decode_full_gate_report,
)

FULL_CERTIFICATION_TYPE = ArtifactType("microcosm.uk.full-certification-readiness", 1)
_REQUIRED = frozenset(
    {
        "package",
        "export_descriptor",
        "export_readback",
        "preflight",
        "full_gates",
        "diagnostics",
        "holdout",
        "selection",
        "surface",
    }
)
_COMPARISON_SOURCES = {
    "native_scorecard": "uk_native_scorecard",
    "matched_size_scorecard": "uk_matched_size_scorecard",
}


def _object(payload: bytes) -> dict:
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("UK certification input must be a JSON object.")
    return value


def _equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise ValueError(f"UK certification {label} differs from its graph artifact.")


def _scorecard_status(payload, *, expected_identity, matched_households=None):
    """Use the existing release-quality assessment, preserving incomplete evidence."""
    from microcosm.data.contract import _check_uk_incumbent_surface_evaluation

    if payload is None:
        return {
            "status": "evidence_absent",
            "failures": ["No declared scorecard source."],
        }
    failures = []
    _check_uk_incumbent_surface_evaluation(
        payload, failures, expected_identity=expected_identity
    )
    if matched_households is not None:
        comparison = payload.get("comparison", {})
        if (
            not isinstance(comparison, Mapping)
            or comparison.get("kind") != "matched_size"
        ):
            failures.append(
                "Matched-size scorecard needs an explicit matched_size comparison identity."
            )
        elif any(
            type(comparison.get(field)) is not int
            or comparison[field] != matched_households
            for field in ("candidate_households", "incumbent_households")
        ):
            failures.append(
                "Matched-size scorecard population counts must both equal the exported k."
            )
    return {"status": "passed" if not failures else "failed", "failures": failures}


def compose_uk_full_certification_readiness(
    artifacts: Mapping[str, tuple[str, bytes]],
    *,
    comparison_sources: Mapping[str, tuple[Mapping, bytes]] | None = None,
) -> dict:
    """Verify one graph's byte joins and report readiness for external review."""
    missing = _REQUIRED - set(artifacts)
    if missing:
        raise ValueError(
            f"UK certification is missing graph artifacts {sorted(missing)}."
        )
    documents = {name: _object(payload) for name, (_, payload) in artifacts.items()}
    keys = {name: key for name, (key, _) in artifacts.items()}
    package, readback = documents["package"], documents["export_readback"]
    descriptor = documents["export_descriptor"]
    if (
        package.get("kind") != "uk_full_build_package"
        or package.get("schema_version") != 1
        or package.get("readback_passed") is not True
        or package.get("release_authorized") is not False
    ):
        raise ValueError("UK certification requires the full graph package inventory.")
    if (
        descriptor.get("kind") != "uk_full_build_export"
        or descriptor.get("schema_version") != 1
    ):
        raise ValueError("UK certification requires a typed export descriptor.")
    if (
        readback.get("kind") != "uk_full_build_export_readback"
        or readback.get("passed") is not True
    ):
        raise ValueError("UK certification requires passing exported-byte readback.")
    _equal(package["dataset"], readback["dataset"], "candidate bytes")
    _equal(package["content_sha256"], readback["content_sha256"], "candidate contents")
    _equal(
        readback["content_sha256"], descriptor["content_sha256"], "export descriptor"
    )
    _equal(readback["bindings"], descriptor["bindings"], "export bindings")
    _equal(package["build_bindings"], readback["bindings"], "package bindings")
    for name in ("full_gates", "diagnostics", "holdout"):
        _equal(package["artifacts"].get(name), keys[name], f"packaged {name}")
    _equal(
        package["artifacts"].get("export_readback"),
        keys["export_readback"],
        "packaged readback",
    )
    selection = documents["selection"]["receipt"]
    scope = uk_full_gate_scope_receipt(selection)
    manifest = uk_full_gate_manifest(selection)
    outcomes = {}
    phase_names = []
    for name in ("preflight", "full_gates"):
        document = documents[name]
        report, _ = decode_full_gate_report(document)
        _equal(document["selection_receipt"], selection, f"{name} target selection")
        _equal(document["scope"], scope, f"{name} declared scope")
        for dependency in ("selection", "surface"):
            _equal(
                document["artifacts"].get(dependency),
                keys[dependency],
                f"{name} {dependency}",
            )
        phase_names.append(report.phase)
        for outcome in report.outcomes:
            if outcome.entry.id in outcomes:
                raise ValueError("UK certification has duplicate gate outcomes.")
            outcomes[outcome.entry.id] = outcome.to_payload()
    final = documents["full_gates"]
    _equal(
        final["artifacts"].get("preflight"), keys["preflight"], "calibrated preflight"
    )
    _equal(final["artifacts"].get("holdout"), keys["holdout"], "calibrated holdout")
    _equal(
        final["sample_fraction"],
        documents["preflight"]["sample_fraction"],
        "sample fraction",
    )
    _equal(
        final["release_candidate"],
        documents["preflight"]["release_candidate"],
        "release posture",
    )
    if "spine_provenance" in documents:
        if {"spine_assembled", "spine_transferred"} & set(documents):
            raise ValueError(
                "Use raw spine reports or bound checkpoint provenance, not both."
            )
        provenance = documents["spine_provenance"]
        _equal(
            final["artifacts"].get("spine_provenance"),
            keys["spine_provenance"],
            "spine checkpoint provenance",
        )
        report = provenance["spine_gate_report"]["payload"]
        if not provenance.get("uk_frame_content_identity") or not provenance[
            "spine_gate_report"
        ].get("sha256"):
            raise ValueError("UK certification requires strict bound spine provenance.")
        spine_gates = uk_spine_gate_manifest(load_country_spec("uk"))
        expected = {entry.id: entry for entry in spine_gates.gates}
        _equal(set(report["gates"]), set(expected), "checkpoint spine scope")
        _equal(report.get("blocked_at_phase"), None, "checkpoint spine gate completion")
        for gate_id, entry in expected.items():
            outcome = report["gates"][gate_id]
            _equal(
                outcome.get("criticality"), entry.criticality, f"{gate_id} criticality"
            )
            _equal(outcome.get("phase"), entry.phase, f"{gate_id} phase")
            outcomes[gate_id] = outcome
        phase_names.extend(spine_gates.phases)
    else:
        spine_gates = uk_spine_gate_manifest(load_country_spec("uk"))
        for name, phase in (
            ("spine_assembled", "assembled"),
            ("spine_transferred", "transferred"),
        ):
            if name not in documents:
                raise ValueError(
                    f"UK certification needs {name} or strict bound spine provenance."
                )
            report = gate_phase_report_from_payload(documents[name], gates=spine_gates)
            _equal(report.phase, phase, "spine phase")
            phase_names.append(report.phase)
            for outcome in report.outcomes:
                if outcome.entry.id in outcomes:
                    raise ValueError(
                        "UK certification has duplicate spine gate outcomes."
                    )
                outcomes[outcome.entry.id] = outcome.to_payload()
    _equal(
        set(outcomes), {entry.id for entry in manifest.gates}, "complete gate roster"
    )
    _equal(set(phase_names), set(manifest.phases), "complete gate phases")
    reviewed_inapplicable = {
        entry.id for entry in manifest.gates if entry.not_applicable is not None
    }
    failed_gates = sorted(
        gate_id
        for gate_id, outcome in outcomes.items()
        if outcome["criticality"] == "release_blocking"
        and outcome["status"] != GateStatus.PASSED.value
        and not (
            gate_id in reviewed_inapplicable
            and outcome["status"] == GateStatus.NOT_APPLICABLE.value
        )
    )
    validation = documents["surface"]["source_validation"]
    ledger = validation["ledger_provenance"]
    expected_identity = {
        "candidate_dataset_sha256": package["dataset"]["sha256"],
        "candidate_manifest_sha256": hashlib.sha256(
            artifacts["package"][1]
        ).hexdigest(),
        "candidate_diagnostics_sha256": hashlib.sha256(
            artifacts["diagnostics"][1]
        ).hexdigest(),
        "ledger_facts_sha256": ledger["facts_sha256"],
        "ledger_manifest_sha256": ledger["manifest_sha256"],
    }
    config = package["build_bindings"].get("configuration", {})
    requested_k = config.get("calibration", {}).get("dataset_households")
    actual_k = descriptor["tables"]["household"]["rows"]
    if requested_k is not None:
        _equal(actual_k, requested_k, "exported exact household count")
    comparisons = {}
    comparison_sources = {} if comparison_sources is None else comparison_sources
    if set(comparison_sources) - set(_COMPARISON_SOURCES):
        raise ValueError("Unknown UK certification scorecard source.")
    for role in _COMPARISON_SOURCES:
        source = comparison_sources.get(role)
        if role == "matched_size_scorecard" and requested_k is None:
            comparisons[role] = {
                "status": "not_required",
                "reason": "No exported exact-count k was requested.",
            }
            if source is not None:
                raise ValueError(
                    "Matched-size scorecard supplied to a build without requested k."
                )
            continue
        payload = None if source is None else _object(source[1])
        comparisons[role] = _scorecard_status(
            payload,
            expected_identity=expected_identity,
            matched_households=actual_k if role == "matched_size_scorecard" else None,
        )
        if source is not None:
            file = dict(source[0])
            _equal(
                file["sha256"],
                hashlib.sha256(source[1]).hexdigest(),
                f"{role} source bytes",
            )
            _equal(file["size_bytes"], len(source[1]), f"{role} source length")
            comparisons[role]["source"] = file
    reasons = []
    if failed_gates:
        reasons.append("Non-passing release-blocking gates: " + ", ".join(failed_gates))
    if final["sample_fraction"] != 1.0:
        reasons.append(
            "Development sample fraction does not establish native population readiness."
        )
    holdout = documents["holdout"]
    if scope["local_fit_claim"] and (
        holdout.get("skipped")
        or holdout.get("method") != "rotated_folds"
        or holdout.get("n_folds") != 5
        or len(holdout.get("folds", ())) != 5
        or any(
            isinstance(holdout.get(field), bool)
            or not isinstance(holdout.get(field), int | float)
            or not math.isfinite(holdout[field])
            for field in ("mean_holdout_loss", "worst_holdout_loss")
        )
    ):
        reasons.append("The declared five-fold local holdout is skipped or incomplete.")
    reasons.extend(
        f"{role}: {record['status']}"
        for role, record in comparisons.items()
        if record["status"] not in {"passed", "not_required"}
    )
    return {
        "schema_version": 1,
        "kind": "uk_full_build_certification_readiness",
        "candidate": package["dataset"],
        "content_sha256": package["content_sha256"],
        "target_scope": scope,
        "gate_coverage": {
            "declared": sorted(outcomes),
            "phases": sorted(phase_names),
            "scope_exclusions": scope["scope_exclusions"],
        },
        "gate_outcomes": outcomes,
        "source_validation": validation,
        "comparisons": comparisons,
        "artifacts": {
            name: {
                "graph_artifact_key": key,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
            for name, (key, payload) in sorted(artifacts.items())
        },
        "ready_for_external_review": not reasons,
        "readiness_failures": reasons,
        "subnational_fit_certified": False,
        "release_authorized": False,
        "signing": "external",
    }


class UKFullCertificationKernel(KernelBase):
    ref = "uk.full.certification-readiness@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC, numeric=Numeric.BITWISE, seed_source=SeedSource.NONE
    )

    def implementation_hash(self):
        from microcosm.data import contract

        from . import full_gates, graph_terminal

        return source_hash(sys.modules[__name__], contract, full_gates, graph_terminal)

    def run(self, context: KernelContext) -> KernelResult:
        sources = {}
        for role, source_name in _COMPARISON_SOURCES.items():
            if source_name in context.sources:
                path = Path(context.sources[source_name])
                payload = path.read_bytes()
                sources[role] = (file_artifact(path), payload)
        report = compose_uk_full_certification_readiness(
            {
                name: (value.key, value.payload)
                for name, value in context.artifacts.items()
            },
            comparison_sources=sources,
        )
        return KernelResult(
            artifacts={"certification_readiness": canonical_json(report)}
        )


def append_uk_full_certification_node(
    graph: Graph,
    *,
    population: str,
    spine_provenance: ArtifactInput | None = None,
    comparison_sources: tuple[str, ...] = (),
) -> Graph:
    """Append readiness after packaging; comparison sources are explicit inputs."""
    if set(comparison_sources) - set(_COMPARISON_SOURCES.values()):
        raise ValueError("Unknown full-build comparison source.")
    inputs = [
        ArtifactInput(
            "package", "uk.full.package", "package_inventory", PACKAGE_INVENTORY_TYPE
        ),
        ArtifactInput(
            "export_descriptor",
            "uk.full.export.prepare",
            "export_descriptor",
            EXPORT_DESCRIPTOR_TYPE,
        ),
        ArtifactInput(
            "export_readback",
            "uk.full.export.readback",
            "export_readback",
            EXPORT_READBACK_TYPE,
        ),
        ArtifactInput(
            "preflight", "uk.full.gates.preflight", "gate_report", FULL_GATE_REPORT_TYPE
        ),
        ArtifactInput(
            "full_gates",
            "uk.full.gates.calibrated",
            "gate_report",
            FULL_GATE_REPORT_TYPE,
        ),
        ArtifactInput(
            "diagnostics",
            "uk.full.gates.calibrated",
            "calibration_diagnostics",
            FULL_DIAGNOSTICS_TYPE,
        ),
        ArtifactInput("holdout", "uk.full.holdout", "holdout", FULL_HOLDOUT_TYPE),
        ArtifactInput(
            "selection", "uk.full.target_selection", "selection", TARGET_SELECTION_TYPE
        ),
        ArtifactInput(
            "surface", "uk.full.target_compilation", "surface", TARGET_SURFACE_TYPE
        ),
    ]
    if spine_provenance is None:
        inputs.extend(
            ArtifactInput(
                f"spine_{phase}",
                f"spine.gates.{phase}",
                "gate_report",
                SPINE_GATE_REPORT_TYPE,
            )
            for phase in ("assembled", "transferred")
        )
    else:
        inputs.append(replace(spine_provenance, name="spine_provenance"))
    existing = {source.name for source in graph.sources}
    return replace(
        graph,
        sources=(
            *graph.sources,
            *(
                SourceRef(name, "raw-bytes-v1")
                for name in comparison_sources
                if name not in existing
            ),
        ),
        nodes=(
            *graph.nodes,
            Node(
                "uk.full.certification",
                UKFullCertificationKernel.ref,
                population=population,
                sources=comparison_sources,
                artifact_inputs=tuple(inputs),
                artifact_outputs=(
                    ArtifactOutput("certification_readiness", FULL_CERTIFICATION_TYPE),
                ),
                description="Bind complete selected-scope gates, exact exported bytes and native/size comparison readiness; publication and signing remain external.",
            ),
        ),
    )


def register_uk_full_certification_kernel(registry: KernelRegistry) -> None:
    registry.register(UKFullCertificationKernel())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Materialize one full UK graph's unsigned certification readiness; historical split-lane inputs are retired."
    )
    parser.add_argument("--graph-manifest", required=True, type=Path)
    parser.add_argument("--graph-store", required=True, type=Path)
    parser.add_argument("--candidate-h5", required=True, type=Path)
    parser.add_argument("--certification-json", required=True, type=Path)
    args = parser.parse_args(argv)
    store = ContentStore(args.graph_store)
    manifest = RunManifest.load(args.graph_manifest, store)
    key = manifest.nodes["uk.full.certification"].opaque_artifacts[
        "certification_readiness"
    ]
    payload = store.load_bytes(key)
    report = _object(payload)
    if (
        report.get("kind") != "uk_full_build_certification_readiness"
        or report.get("release_authorized") is not False
    ):
        raise ValueError(
            "Expected an unsigned full-graph certification readiness artifact."
        )
    _equal(
        file_artifact(args.candidate_h5), report["candidate"], "current candidate bytes"
    )
    materialize_bytes(payload, args.certification_json)
    return 0 if report["ready_for_external_review"] else 1
