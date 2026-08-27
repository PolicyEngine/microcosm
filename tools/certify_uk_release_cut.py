"""Certify a calibrated UK national candidate for release.

The release-cut certification producer (microcosm#757 item B5): runs the 18
declared national preflight/terminal gates over the calibrated candidate —
the executable home the June driver's retirement left empty — then composes
the multi-part certification over the spine build's battery report, the
calibration seam's battery report, and the fresh release-cut report. The
parts must union to the full declared gate-entry set with no gap and no
overlap beyond the declared shared ids, each signed by its producer, over
one closed identity join (spine report -> sidecar -> build record ->
diagnostics -> candidate bytes). A candidate's shippability verdict comes
only from the certification this driver writes.

The battery always runs at release-candidate strictness: evidence_absent
gaps block. The rule-1 score receipt is cross-pinned into the certification
(the audit's third carried defect), so the score is signed run evidence
rather than a null slot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path

from microcosm.build.ledger_artifact import load_ledger_consumer_artifact
from microcosm.build.logbook_adoption import (
    AttemptState,
    append_phase,
    apply_error_verdict,
    error_receipt_path,
    git_code_pin,
    local_artifact_reference,
    record_terminal_attempt,
    resolve_predecessor,
    role_pins_digest,
    write_error_receipt,
)
from microcosm.build.uk_runtime.frs_release import load_uk_frs_release
from microcosm.build.uk_runtime.ledger_targets import (
    compile_uk_local_target_registry,
    compile_uk_target_registry,
    load_uk_local_area_crosswalk,
)
from microcosm.build.uk_runtime.measure_simulation import (
    apply_uk_calibration_measure_exclusions,
    load_uk_calibration_measure_exclusions,
)
from microcosm.build.uk_runtime.national_frame import load_uk_national_frame
from microcosm.build.uk_runtime.parity_reference import load_efrs_parity_reference
from microcosm.build.uk_runtime.release_certification import (
    compose_uk_release_certification,
    rehydrate_uk_fit_weight_records,
    run_uk_release_cut_battery,
    uk_release_parity_evidence,
)
from microcosm.build.uk_runtime.release_input_coverage import (
    PolicyEngineUKCoverageEngine,
)
from microcosm.build.uk_runtime.weighted_integrity import (
    exclusion_evaluation_date,
    load_uk_input_mass_reference,
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_REPOSITORY = Path(__file__).resolve().parents[1]
_PIPELINE = "uk-frs-release-certification"
_LEDGER_COMPILE_PARITY_PERIODS = (2023, 2025)
_LOCAL_COMPILE_PARITY_PERIOD = 2025


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    started_at = time.perf_counter()
    started_ts = datetime.now(UTC)
    code_pin = git_code_pin(_REPOSITORY)
    predecessor = resolve_predecessor(args.logbook_prev_row_digest)
    source_pins = {
        "candidate_h5": {"sha256": args.candidate_sha256},
        "ledger_facts": {"sha256": args.ledger_facts_sha256},
    }
    state = AttemptState(
        build_id=f"{_PIPELINE}-attempt-{started_ts.strftime('%Y%m%dT%H%M%SZ')}",
        identity_digest=hashlib.sha256(
            json.dumps(
                {
                    "pipeline": _PIPELINE,
                    "release_id": args.release_id,
                    "candidate_sha256": args.candidate_sha256,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest(),
        input_pins_digest=role_pins_digest(source_pins),
        phases_reached=["attempt_started"],
        gate_verdicts={},
    )
    spool_dir = args.certification_json.parent / "logbook-spool"
    try:
        summary = _run(args, state)
    except BaseException as error:
        error_path = write_error_receipt(
            error_receipt_path(
                args.certification_json.parent / "logbook-receipts",
                build_id=state.build_id,
            ),
            state=state,
            pipeline=_PIPELINE,
            error=error,
        )
        apply_error_verdict(
            state,
            f"{local_artifact_reference(error_path, repository_hint=_REPOSITORY)}"
            "#/error_type",
        )
        record_terminal_attempt(
            state=state,
            started_at=started_at,
            started_ts=started_ts,
            pipeline=_PIPELINE,
            rung="f100",
            seed=None,
            code_pin=code_pin,
            disposition=(
                "discarded" if isinstance(error, KeyboardInterrupt) else "failed"
            ),
            predecessor=predecessor,
            spool_dir=spool_dir,
        )
        raise
    state.artifact_location = local_artifact_reference(
        args.certification_json, repository_hint=_REPOSITORY
    )
    record_terminal_attempt(
        state=state,
        started_at=started_at,
        started_ts=started_ts,
        pipeline=_PIPELINE,
        rung="f100",
        seed=None,
        code_pin=code_pin,
        disposition="certified",
        predecessor=predecessor,
        spool_dir=spool_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _run(args: argparse.Namespace, state: AttemptState) -> dict[str, object]:
    measured = hashlib.sha256(args.candidate_h5.read_bytes()).hexdigest()
    if measured != args.candidate_sha256:
        raise SystemExit(
            "error: --candidate-h5 sha mismatch: "
            f"measured {measured}, pinned {args.candidate_sha256}"
        )
    sidecar_path = args.spine_h5.with_suffix(".build.json")
    if not sidecar_path.is_file():
        raise SystemExit(f"error: spine build sidecar absent: {sidecar_path}")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    spine_report_path = args.spine_h5.with_suffix(".spine_gates.json")

    diagnostics_bytes = args.diagnostics_json.read_bytes()
    diagnostics = json.loads(diagnostics_bytes)
    diagnostics_sha = hashlib.sha256(diagnostics_bytes).hexdigest()
    build_record = json.loads(args.build_record_json.read_text(encoding="utf-8"))
    recorded_diagnostics = (
        build_record.get("artifacts", {}).get("diagnostics_json", {}).get("sha256")
    )
    if recorded_diagnostics != diagnostics_sha:
        raise SystemExit(
            "error: --diagnostics-json bytes do not match the build record's "
            f"binding ({diagnostics_sha} != {recorded_diagnostics})"
        )
    append_phase(state, "inputs_bound")

    artifact = load_ledger_consumer_artifact(
        args.ledger_facts,
        expected_facts_sha256=args.ledger_facts_sha256,
        expected_manifest_sha256=args.ledger_manifest_sha256,
    )
    ledger_registries = {}
    for period in _LEDGER_COMPILE_PARITY_PERIODS:
        compilation = compile_uk_target_registry(artifact.facts, target_period=period)
        ledger_registries[period] = compilation.registry
    local_compilation = compile_uk_local_target_registry(
        artifact.facts,
        target_period=_LOCAL_COMPILE_PARITY_PERIOD,
        crosswalk=load_uk_local_area_crosswalk(),
    )
    local_ledger_registries = {_LOCAL_COMPILE_PARITY_PERIOD: local_compilation.registry}
    calibration_year = load_uk_frs_release().calibration_year
    if calibration_year in ledger_registries:
        reference_compiled = ledger_registries[calibration_year]
    else:
        reference_compiled = compile_uk_target_registry(
            artifact.facts, target_period=calibration_year
        ).registry
    evaluated_on = exclusion_evaluation_date(None)
    exclusions = load_uk_calibration_measure_exclusions()
    reference_registry, _receipt = apply_uk_calibration_measure_exclusions(
        reference_compiled, exclusions, now=evaluated_on
    )
    append_phase(state, "registries_compiled")

    frame, _provenance = load_uk_national_frame(args.candidate_h5)
    engine = PolicyEngineUKCoverageEngine()
    parity_evidence = uk_release_parity_evidence(
        frame,
        diagnostics_targets=diagnostics["targets"],
        reference_registry=reference_registry,
        parity_reference=load_efrs_parity_reference(),
    )
    report = run_uk_release_cut_battery(
        frame,
        report_path=args.release_cut_gate_json,
        release_id=args.release_id,
        diagnostics_sha256=diagnostics_sha,
        coverage_engine=engine,
        build_stage_names=sidecar["stages"],
        ledger_registries=ledger_registries,
        local_ledger_registries=local_ledger_registries,
        parity_evidence=parity_evidence,
        fit_weight_records=rehydrate_uk_fit_weight_records(sidecar),
        input_mass_reference=load_uk_input_mass_reference(args.input_mass_reference),
        exclusions_evaluated_on=evaluated_on,
    )
    append_phase(state, "release_cut_gates_evaluated")
    for gate_id, payload in report["gates"].items():
        state.gate_verdicts[gate_id] = {
            "verdict": payload["status"],
            "receipt": (f"local://{args.release_cut_gate_json.name}#/gates/{gate_id}"),
        }

    certification = compose_uk_release_certification(
        release_id=args.release_id,
        candidate_name=args.candidate_name,
        candidate_path=args.candidate_h5,
        candidate_sha256=args.candidate_sha256,
        spine_report_path=spine_report_path,
        seam_report_path=args.seam_gate_report,
        release_cut_report_path=args.release_cut_gate_json,
        spine_sidecar=sidecar,
        build_record=build_record,
        score_receipt_path=args.score_receipt,
        exclusions_evaluated_on=evaluated_on,
        certification_path=args.certification_json,
    )
    append_phase(state, "certification_written")
    return {
        "certification_json": str(args.certification_json),
        "certification_sha256": hashlib.sha256(
            args.certification_json.read_bytes()
        ).hexdigest(),
        "release_cut_gate_json": str(args.release_cut_gate_json),
        "shippable": certification["shippable"],
        "parts": {
            name: part["statuses"] for name, part in certification["parts"].items()
        },
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-h5", required=True, type=Path)
    parser.add_argument("--candidate-sha256", required=True, type=_sha256)
    parser.add_argument(
        "--candidate-name",
        required=True,
        help="The dataset name the certification certifies, e.g. microcosm_uk_2024.",
    )
    parser.add_argument("--spine-h5", required=True, type=Path)
    parser.add_argument("--diagnostics-json", required=True, type=Path)
    parser.add_argument("--build-record-json", required=True, type=Path)
    parser.add_argument("--seam-gate-report", required=True, type=Path)
    parser.add_argument("--ledger-facts", required=True, type=Path)
    parser.add_argument("--ledger-facts-sha256", required=True, type=_sha256)
    parser.add_argument("--ledger-manifest-sha256", required=True, type=_sha256)
    parser.add_argument("--input-mass-reference", required=True, type=Path)
    parser.add_argument("--score-receipt", required=True, type=Path)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--release-cut-gate-json", type=Path)
    parser.add_argument("--certification-json", type=Path)
    parser.add_argument("--logbook-prev-row-digest", type=_sha256)
    args = parser.parse_args(argv)
    args.release_cut_gate_json = (
        args.release_cut_gate_json
        or args.candidate_h5.with_suffix(".release_cut_gates.json")
    )
    args.certification_json = args.certification_json or args.candidate_h5.with_suffix(
        ".release_certification.json"
    )
    distinct = {
        "--candidate-h5": args.candidate_h5,
        "--spine-h5": args.spine_h5,
        "--diagnostics-json": args.diagnostics_json,
        "--build-record-json": args.build_record_json,
        "--seam-gate-report": args.seam_gate_report,
        "--release-cut-gate-json": args.release_cut_gate_json,
        "--certification-json": args.certification_json,
        "--score-receipt": args.score_receipt,
    }
    resolved: dict[Path, str] = {}
    for flag, path in distinct.items():
        canonical = path.resolve()
        if canonical in resolved:
            parser.error(f"{flag} aliases {resolved[canonical]}: {path}")
        resolved[canonical] = flag
    return args


def _sha256(value: str) -> str:
    if not _SHA256.fullmatch(value):
        raise argparse.ArgumentTypeError("expected a 64-character lowercase sha256")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
