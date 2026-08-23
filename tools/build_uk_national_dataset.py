"""Build the national UK staging file with the guarded HMRC/SPI family."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import uuid
from datetime import UTC, datetime
from importlib import resources as importlib_resources
from itertools import combinations
from pathlib import Path

from microcosm.build.country_spec import country_stage_plan, load_country_spec
from microcosm.build.gate_battery import GateBatteryBlockedError
from microcosm.build.ledger_artifact import (
    add_ledger_artifact_args,
    resolve_ledger_artifact,
)
from microcosm.build.logbook import canonical_json_bytes
from microcosm.build.logbook_adoption import (
    AttemptState,
    append_phase,
    apply_error_verdict,
    error_receipt_path,
    git_code_pin,
    local_artifact_reference,
    preflight_digest,
    record_terminal_attempt,
    resolve_predecessor,
    role_pins_digest,
    sha256_argument,
    write_error_receipt,
)
from microcosm.build.plan import Stage as PlanStage
from microcosm.build.uk_runtime.cgt_imputation import (
    uk_capital_gains_imputation_stage,
)
from microcosm.build.uk_runtime.diagnostics import write_uk_calibration_diagnostics
from microcosm.build.uk_runtime.frs_hmrc_leaves import (
    UKFRSHMRCRetainedLeavesStageTransform,
)
from microcosm.build.uk_runtime.frs_release import load_uk_frs_release
from microcosm.build.uk_runtime.hmrc_replay import write_hmrc_replay_report
from microcosm.build.uk_runtime.hmrc_restoration import (
    UKHMRCIncomeStageTransform,
    verify_certified_uk_candidate,
    verify_staging_candidate_uk_input,
)
from microcosm.build.uk_runtime.ledger_targets import compile_uk_target_registry
from microcosm.build.uk_runtime.national_build import build_uk_national_dataset
from microcosm.build.uk_runtime.national_calibration import (
    UKNationalCalibrationStage,
)
from microcosm.build.uk_runtime.national_frame import (
    uk_household_weight_kind,
    uk_time_period,
)
from microcosm.build.uk_runtime.national_sampling import (
    UK_SAMPLE_RUNG_TOKENS,
    UK_SAMPLE_SEED_DEFAULT,
)
from microcosm.build.uk_runtime.parity_reference import (
    load_efrs_parity_reference,
)
from microcosm.build.uk_runtime.release_identity import UK_RELEASE_TIERS
from microcosm.build.uk_runtime.source_runtime import uk_stage_implementations
from microcosm.build.uk_runtime.terminal_gates import (
    uk_default_degenerate_reviewed_exclusions,
)
from microcosm.build.uk_runtime.weighted_integrity import (
    UK_DEGENERATE_EXCLUSION_REGISTER_RESOURCE,
    UK_INPUT_MASS_EXCLUSION_REGISTER_RESOURCE,
    UK_QRF_TAIL_EXCLUSION_REGISTER_RESOURCE,
    load_uk_input_mass_reference,
    load_uk_reference_scoped_exclusion_register,
    load_uk_reviewed_exclusion_register,
    uk_default_input_mass_reviewed_exclusions,
    uk_default_qrf_tail_reviewed_exclusions,
)

#: Canonical UK release ids (and the grandfathered June id) name shippable
#: artifacts; a sampled rung build must never carry one. Mirrors the
#: microcosm-data contract's release-identity check without importing the
#: data shard into the build tool. The durable coupling is the gate
#: battery's ``release_candidate`` flag (wired below: ``--release-candidate``
#: is refused on a rung); this fence stays as defense in depth over the id
#: namespace itself.
# Year and count widths mirror the microcosm-data contract's release-identity
# regex ([1-9][0-9]*), and the tier alternation is built from the build
# shard's ratified UK_RELEASE_TIERS so a newly ratified tier is fenced
# automatically (adversarial-review finding).
_CANONICAL_UK_RELEASE_ID = re.compile(
    r"populace-uk-[1-9][0-9]*-(?:"
    + "|".join(sorted(re.escape(tier) for tier in UK_RELEASE_TIERS))
    + r")-k[1-9][0-9]*"
)
_UK_JUNE_RELEASE_ID = "populace-uk-2023-dd68c73-4aa4b14-20260619T023711Z"

#: The one named dev-scale statistical edge receipted on a rung (#657,
#: closed without code): sklearn's stratified split inside the SPI imputation
#: refuses a singleton class on an unlucky small-sample composition. The
#: computation is never altered — the rung build aborts, but with a receipt
#: naming the edge instead of a bare traceback, and the remedy is re-rolling
#: ``--seed``. Only this named edge is receipted; unknown exceptions crash
#: loudly, so the receipt path can never absorb a real defect.
_RUNG_NAMED_EDGE_SIGNATURE = "The least populated classes in y have only 1 member"
_RUNG_ABORT_EXIT_CODE = 3
_UK_NATIONAL_PIPELINE = "uk-frs-staging"
_REPOSITORY = Path(__file__).resolve().parents[1]


def _rung_sample_fraction(value: str) -> float:
    """CLI rung policy (#624) over the permissive library validator."""

    try:
        fraction = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"sample fraction must be a number; got {value!r}."
        ) from error
    if fraction not in UK_SAMPLE_RUNG_TOKENS:
        raise argparse.ArgumentTypeError(
            "sample fraction must be one of 0.01, 0.10, or 1.0 (the #624 rungs)."
        )
    return fraction


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-h5",
        type=Path,
        required=True,
        help="Compact UK single-year H5 supplying the national base tables.",
    )
    parser.add_argument(
        "--staging-candidate-input-sha256",
        type=sha256_argument,
        help=(
            "Declare --input-h5 as a non-certified staging-candidate spine and "
            "bind it to this SHA-256. Refused with --release-candidate."
        ),
    )
    parser.add_argument(
        "--staging-h5",
        type=Path,
        required=True,
        help="Caller-owned path for the gated national staging H5.",
    )
    parser.add_argument(
        "--release-id",
        required=True,
        help="Canonical release id to bind into the signed terminal report.",
    )
    parser.add_argument(
        "--calibration-diagnostics-sha256",
        required=True,
        help=(
            "Lowercase SHA-256 of the exact calibration_diagnostics.json bytes "
            "that will ship with this release."
        ),
    )
    parser.add_argument(
        "--national-calibration-diagnostics-json",
        type=Path,
        help="Per-target diagnostics emitted by the national calibration stage.",
    )
    parser.add_argument(
        "--frs-raw-dir",
        type=Path,
        required=True,
        help=(
            "Raw FRS 2024-25 directory containing adult.tab and benefits.tab "
            "for source-faithful retained HMRC leaves."
        ),
    )
    parser.add_argument(
        "--spi-tab",
        type=Path,
        required=True,
        help="Licensed UKDS SPI 2022-23 donor named put2223uk.tab.",
    )
    parser.add_argument(
        "--hmrc-ods",
        type=Path,
        required=True,
        help="Official HMRC Personal Incomes 2023-24 collated ODS.",
    )
    parser.add_argument(
        "--cgt-ods",
        type=Path,
        required=True,
        help=(
            "Official HMRC Capital Gains Tax statistics table 3 ODS (size of "
            "gain by taxable income); fingerprint-verified before it is read."
        ),
    )
    gate_output = parser.add_mutually_exclusive_group()
    gate_output.add_argument(
        "--terminal-gates-json",
        type=Path,
        help=(
            "Consolidated terminal-gate report path. Defaults beside "
            "--staging-h5 with suffix '.terminal_gates.json'."
        ),
    )
    gate_output.add_argument(
        "--input-coverage-json",
        type=Path,
        help=(
            "Legacy schema-1 input-coverage diagnostic path. Cannot be "
            "supplied with the preferred terminal-gate option."
        ),
    )
    parser.add_argument(
        "--hmrc-evidence-json",
        type=Path,
        help=(
            "HMRC stage evidence path. Defaults beside --staging-h5 with "
            "suffix '.hmrc_income.json'."
        ),
    )
    parser.add_argument(
        "--hmrc-replay-json",
        type=Path,
        help=(
            "Aggregate-only 208-fact replay report path. Defaults beside "
            "--staging-h5 with suffix '.hmrc_replay.json'."
        ),
    )
    parser.add_argument(
        "--build-record-json",
        type=Path,
        help=(
            "Aggregate, path-free staging build record. Defaults beside "
            "--staging-h5 with suffix '.build.json'."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--qrf-estimators", type=int, default=100)
    parser.add_argument(
        "--sample-fraction",
        type=_rung_sample_fraction,
        default=1.0,
        help=(
            "Scale-ladder rung (#627): 0.01 smoke, 0.10 dev, or 1.0 full. "
            "Below 1.0 the loaded compact is sampled at clone-family grain, "
            "renormalized to full household mass, and refused a canonical "
            "release id — rung artifacts are receipts, never releases."
        ),
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=UK_SAMPLE_SEED_DEFAULT,
        help=(
            "Whole-clone-family survey sampling seed (default: "
            f"{UK_SAMPLE_SEED_DEFAULT}). Separate from --seed so dev-scale "
            "sweeps vary one draw at a time."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help=(
            "Persist a lossless Frame checkpoint at every stage boundary and "
            "resume completed stages from it (#612 increment 3). The run is "
            "pinned by a content-addressed run config (input digest, seed, "
            "QRF estimators, raw-source digests); rerunning the same command "
            "against the same directory resumes, a changed configuration is "
            "refused. Omit for the destructive single-process build."
        ),
    )
    parser.add_argument(
        "--input-mass-reference-json",
        type=Path,
        help=(
            "Frozen weighted per-column reference totals (schema_version 1: "
            "identity + totals) emitted by the #609 measurement tooling. "
            "Supplying it provides the licensed evidence for the spec-armed "
            "input_mass_parity gate; absent, the gate records evidence_absent "
            "and blocks release candidates."
        ),
    )
    parser.add_argument(
        "--input-mass-exclusions",
        type=Path,
        help=(
            "Reviewed input-mass exclusion register overriding the committed "
            f"{UK_INPUT_MASS_EXCLUSION_REGISTER_RESOURCE}. The override is "
            "schema-3 and scoped per named reference. Stale entries fail the "
            "gate; dormant entries are reported."
        ),
    )
    parser.add_argument(
        "--qrf-tail-exclusions",
        type=Path,
        help=(
            "Reviewed QRF tail-concentration exclusion register overriding "
            f"the committed {UK_QRF_TAIL_EXCLUSION_REGISTER_RESOURCE}. Stale "
            "entries fail the gate; dormant entries are reported."
        ),
    )
    parser.add_argument(
        "--degenerate-exclusions",
        type=Path,
        help=(
            "Reviewed degenerate-release-surface exclusion register "
            f"overriding the committed {UK_DEGENERATE_EXCLUSION_REGISTER_RESOURCE} "
            "(#630). Stale entries fail the gate; dormant entries are "
            "reported. The gate is always armed; the override is digested "
            "into the report's evidence_sha256, so an overridden run "
            "self-describes against the committed register."
        ),
    )
    parser.add_argument(
        "--release-candidate",
        action="store_true",
        help=(
            "Arm the battery's release-candidate posture: every "
            "evidence_absent gap blocks instead of being recorded. Refused "
            "on a sampled rung — a rung is structurally non-releasable "
            "(#627). Default off: the staging build records its gaps "
            "honestly and continues."
        ),
    )
    parser.add_argument(
        "--logbook-prev-row-digest",
        type=sha256_argument,
        help=(
            "Optional current Logbook chain head. If omitted, "
            "POPULACE_LOGBOOK_PREV_ROW_DIGEST is used, then genesis null."
        ),
    )
    add_ledger_artifact_args(parser)
    args = parser.parse_args(argv)
    if args.release_candidate and args.sample_fraction != 1.0:
        parser.error(
            "--release-candidate is refused on a sampled rung; a rung build "
            "is structurally non-releasable (#627)."
        )
    if args.release_candidate and args.input_coverage_json is not None:
        parser.error(
            "--release-candidate is refused with --input-coverage-json; the "
            "schema-1 alias is last-written over the report path and a "
            "candidate must keep its signed schema-4 report."
        )
    if args.release_candidate and args.staging_candidate_input_sha256 is not None:
        parser.error(
            "--release-candidate requires the certified input posture; "
            "--staging-candidate-input-sha256 declares a non-certified spine."
        )
    if args.sample_seed < 0:
        parser.error("sample seed must be a non-negative integer.")
    if args.sample_fraction != 1.0 and (
        _CANONICAL_UK_RELEASE_ID.fullmatch(args.release_id)
        or args.release_id == _UK_JUNE_RELEASE_ID
    ):
        parser.error(
            "a sampled build (--sample-fraction below 1.0) must not carry a "
            "canonical release id; rung artifacts are structurally "
            "non-releasable (#627)."
        )
    return args


def _weighted_integrity_arguments(args: argparse.Namespace) -> dict[str, object]:
    """Assemble weighted-integrity evidence and optional review overrides."""

    def parser_error(message: str) -> None:
        raise SystemExit(f"error: {message}")

    arguments: dict[str, object] = {}
    if args.input_mass_reference_json is not None:
        arguments["input_mass_reference"] = load_uk_input_mass_reference(
            args.input_mass_reference_json
        )
        if args.input_mass_exclusions is not None:
            arguments["reviewed_input_mass_exclusions"] = (
                load_uk_reference_scoped_exclusion_register(
                    args.input_mass_exclusions,
                    resource=UK_INPUT_MASS_EXCLUSION_REGISTER_RESOURCE,
                )
            )
        else:
            uk_default_input_mass_reviewed_exclusions()
    elif args.input_mass_exclusions is not None:
        parser_error("--input-mass-exclusions requires --input-mass-reference-json.")
    else:
        uk_default_input_mass_reviewed_exclusions()
    if args.qrf_tail_exclusions is not None:
        arguments["reviewed_qrf_tail_exclusions"] = load_uk_reviewed_exclusion_register(
            args.qrf_tail_exclusions,
            resource=UK_QRF_TAIL_EXCLUSION_REGISTER_RESOURCE,
        )
    else:
        uk_default_qrf_tail_reviewed_exclusions()
    return arguments


def _new_national_attempt_id(*, timestamp: datetime) -> str:
    instant = timestamp.astimezone(UTC)
    return (
        "uk-national-attempt-"
        f"{instant.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    )


def _new_national_build_id(
    *,
    rung: str,
    sample_seed: int,
    seed: int,
    timestamp: datetime,
) -> str:
    instant = timestamp.astimezone(UTC)
    return (
        f"uk-national-{rung}-ss{sample_seed}-s{seed}-"
        f"{instant.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    )


def _artifact_pin(path: str | Path) -> dict[str, object]:
    info = _artifact_info(path)
    return {"sha256": info["sha256"], "size_bytes": info["size_bytes"]}


def _source_pins(
    *,
    candidate: object,
    retained_leaves_transform: UKFRSHMRCRetainedLeavesStageTransform,
    hmrc_transform: UKHMRCIncomeStageTransform,
    cgt_ods_path: Path,
    ledger_artifact: object | None = None,
) -> dict[str, dict[str, object]]:
    pins = {
        "certified_candidate": {
            "sha256": str(candidate.sha256),
            "size_bytes": int(candidate.size_bytes),
        },
        "adult_tab": _artifact_pin(retained_leaves_transform.adult_tab_path),
        "benefits_tab": _artifact_pin(retained_leaves_transform.benefits_tab_path),
        "spi_tab": _artifact_pin(hmrc_transform.spi_tab_path),
        "hmrc_ods": _artifact_pin(hmrc_transform.hmrc_ods_path),
        "cgt_ods": _artifact_pin(cgt_ods_path),
    }
    if ledger_artifact is not None:
        pins["ledger_facts"] = _ledger_facts_pin(ledger_artifact)
    return pins


def _ledger_facts_pin(ledger_artifact: object) -> dict[str, object]:
    """Pin the consumer feed by content and size.

    Logbook role pins are exactly ``sha256`` and ``size_bytes``; the richer
    Ledger identity block travels separately in ``safe_artifacts`` and
    ``source_vintages``. The feed digest is already verified against the
    manifest and the CLI pin at load, so it is reused rather than recomputed
    over a multi-hundred-megabyte file.
    """

    path = Path(getattr(ledger_artifact, "path"))
    facts_path = path / "consumer_facts.jsonl" if path.is_dir() else path
    return {
        "sha256": str(getattr(ledger_artifact, "facts_sha256")),
        "size_bytes": int(facts_path.stat().st_size),
    }


def _input_posture(candidate: object) -> dict[str, object]:
    tier = str(getattr(candidate, "tier", "frs"))
    return {
        "posture": "staging_candidate" if tier == "staging_candidate" else "certified",
        "filename": str(getattr(candidate, "filename", "")),
        "tier": tier,
        "revision": str(getattr(candidate, "revision", "")),
        "sha256": str(candidate.sha256),
        "size_bytes": int(candidate.size_bytes),
    }


def _gate_verdicts_from_report(
    report: dict[str, object],
    *,
    gate_output_path: Path,
) -> dict[str, dict[str, object]]:
    gates = report.get("gates")
    if not isinstance(gates, dict):
        raise ValueError("terminal gate report must contain a gates object.")
    reference = local_artifact_reference(gate_output_path, repository_hint=_REPOSITORY)
    return {
        str(entry_id): {
            "verdict": str(entry["status"]),
            "receipt": f"{reference}#/gates/{entry_id}",
        }
        for entry_id, entry in gates.items()
        if isinstance(entry, dict) and "status" in entry
    }


def _record_national_attempt(
    *,
    state: AttemptState,
    started_at: float,
    started_ts: datetime,
    rung: str,
    seed: int | None,
    code_pin: str,
    disposition: str,
    predecessor: str | None,
    spool_dir: Path,
) -> Path:
    return record_terminal_attempt(
        state=state,
        started_at=started_at,
        started_ts=started_ts,
        pipeline=_UK_NATIONAL_PIPELINE,
        rung=rung,
        seed=seed,
        code_pin=code_pin,
        disposition=disposition,
        predecessor=predecessor,
        spool_dir=spool_dir,
    )


def _record_failed_exception(
    *,
    error: BaseException,
    state: AttemptState,
    started_at: float,
    started_ts: datetime,
    rung: str,
    seed: int | None,
    code_pin: str,
    predecessor: str | None,
    receipt_base_dir: Path,
    spool_dir: Path,
) -> None:
    error_path = write_error_receipt(
        error_receipt_path(receipt_base_dir, build_id=state.build_id),
        state=state,
        pipeline=_UK_NATIONAL_PIPELINE,
        error=error,
    )
    apply_error_verdict(
        state,
        f"{local_artifact_reference(error_path, repository_hint=_REPOSITORY)}#/error_type",
    )
    _record_national_attempt(
        state=state,
        started_at=started_at,
        started_ts=started_ts,
        rung=rung,
        seed=seed,
        code_pin=code_pin,
        disposition="failed",
        predecessor=predecessor,
        spool_dir=spool_dir,
    )


def _rung_abort_receipt(
    args: argparse.Namespace,
    *,
    rung: str,
    error: BaseException,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_kind": "uk_rung_abort_receipt",
        "build_kind": "uk_national_staging_dataset",
        "release_id": str(args.release_id),
        "sampling": {
            "sample_fraction": float(args.sample_fraction),
            "sample_seed": int(args.sample_seed),
            "rung_token": rung,
        },
        "seed": int(args.seed),
        "named_edge": "spi_split_singleton_class",
        "stage": "hmrc_spi_income",
        "error": str(error),
        "disposition": "aborted_with_receipt",
        "remedy": (
            "Re-roll --seed; accepted dev-scale statistical edge "
            "(microcosm#657, closed). The computation is never altered to avoid it."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    started_at = time.perf_counter()
    started_ts = datetime.now(UTC)
    rung = UK_SAMPLE_RUNG_TOKENS[args.sample_fraction]
    code_pin = "unresolved-local-git-code-pin"
    # Logbook chain configuration is validated before any side effect: a
    # malformed or conflicting predecessor refuses the run here, before the
    # build can unlink the prior attempt's sidecars (#666 adversarial-review
    # finding). Config refusals record no row, like argparse refusals.
    predecessor = resolve_predecessor(args.logbook_prev_row_digest)
    attempt_context: dict[str, object] = {
        "code_pin": code_pin,
        "predecessor": predecessor,
    }
    stage_context: dict[str, object] = {}
    logbook_seed: int | None = args.sample_seed
    receipt_base_dir = args.staging_h5.parent
    spool_dir = args.staging_h5.parent / "logbook-spool"
    digest = preflight_digest(_UK_NATIONAL_PIPELINE)
    state = AttemptState(
        build_id=_new_national_attempt_id(timestamp=started_ts),
        identity_digest=digest,
        input_pins_digest=digest,
        phases_reached=["attempt_started"],
        gate_verdicts={
            "pipeline": {
                "verdict": "running",
                "receipt": "pending-build-scoped-terminal-receipt",
            }
        },
    )
    try:
        return _main_recording(
            args=args,
            state=state,
            started_at=started_at,
            started_ts=started_ts,
            rung=rung,
            logbook_seed=logbook_seed,
            attempt_context=attempt_context,
            stage_context=stage_context,
            spool_dir=spool_dir,
        )
    except ValueError as error:
        if args.sample_fraction != 1.0 and _RUNG_NAMED_EDGE_SIGNATURE in str(error):
            rung_abort_path = args.staging_h5.with_suffix(".rung_abort.json")
            receipt = _rung_abort_receipt(
                args,
                rung=rung,
                error=error,
            )
            _write_json(rung_abort_path, receipt)
            state.gate_verdicts = {
                "uk_rung_abort": {
                    "verdict": "aborted",
                    "receipt": (
                        f"{local_artifact_reference(rung_abort_path, repository_hint=_REPOSITORY)}"
                        "#/named_edge"
                    ),
                }
            }
            append_phase(state, "rung_aborted")
            _record_national_attempt(
                state=state,
                started_at=started_at,
                started_ts=started_ts,
                rung=rung,
                seed=logbook_seed,
                code_pin=str(attempt_context["code_pin"]),
                disposition="discarded",
                predecessor=attempt_context["predecessor"],
                spool_dir=spool_dir,
            )
            print(json.dumps(receipt, indent=2, sort_keys=True))
            return _RUNG_ABORT_EXIT_CODE
        _record_failed_exception(
            error=error,
            state=state,
            started_at=started_at,
            started_ts=started_ts,
            rung=rung,
            seed=logbook_seed,
            code_pin=str(attempt_context["code_pin"]),
            predecessor=attempt_context["predecessor"],
            receipt_base_dir=receipt_base_dir,
            spool_dir=spool_dir,
        )
        raise
    except GateBatteryBlockedError as error:
        retained_leaves_transform = stage_context.get("retained_leaves_transform")
        hmrc_transform = stage_context.get("hmrc_transform")
        candidate = stage_context.get("candidate")
        evidence_path = stage_context.get("evidence_path")
        replay_path = stage_context.get("replay_path")
        if (
            error.phase == "terminal"
            and retained_leaves_transform is not None
            and hmrc_transform is not None
            and candidate is not None
            and evidence_path is not None
            and replay_path is not None
            and retained_leaves_transform.last_result is not None
            and hmrc_transform.last_result is not None
        ):
            _write_stage_reports(
                evidence_path=evidence_path,
                replay_path=replay_path,
                candidate=candidate,
                retained_leaves_transform=retained_leaves_transform,
                hmrc_transform=hmrc_transform,
            )
        gate_report = json.loads(error.report_path.read_text(encoding="utf-8"))
        state.gate_verdicts = _gate_verdicts_from_report(
            gate_report,
            gate_output_path=error.report_path,
        )
        append_phase(state, "gate_battery_blocked")
        _record_national_attempt(
            state=state,
            started_at=started_at,
            started_ts=started_ts,
            rung=rung,
            seed=logbook_seed,
            code_pin=str(attempt_context["code_pin"]),
            disposition="failed",
            predecessor=attempt_context["predecessor"],
            spool_dir=spool_dir,
        )
        raise
    except Exception as error:
        _record_failed_exception(
            error=error,
            state=state,
            started_at=started_at,
            started_ts=started_ts,
            rung=rung,
            seed=logbook_seed,
            code_pin=str(attempt_context["code_pin"]),
            predecessor=attempt_context["predecessor"],
            receipt_base_dir=receipt_base_dir,
            spool_dir=spool_dir,
        )
        raise


def _main_recording(
    *,
    args: argparse.Namespace,
    state: AttemptState,
    started_at: float,
    started_ts: datetime,
    rung: str,
    logbook_seed: int | None,
    attempt_context: dict[str, object],
    stage_context: dict[str, object],
    spool_dir: Path,
) -> int:
    legacy_input_coverage_path = args.input_coverage_json
    terminal_gate_path = (
        None
        if legacy_input_coverage_path is not None
        else (
            args.terminal_gates_json
            or args.staging_h5.with_suffix(".terminal_gates.json")
        )
    )
    gate_output_path = legacy_input_coverage_path or terminal_gate_path
    assert gate_output_path is not None
    evidence_path = args.hmrc_evidence_json or args.staging_h5.with_suffix(
        ".hmrc_income.json"
    )
    replay_path = args.hmrc_replay_json or args.staging_h5.with_suffix(
        ".hmrc_replay.json"
    )
    build_record_path = args.build_record_json or args.staging_h5.with_suffix(
        ".build.json"
    )
    stage_context.update(
        {
            "evidence_path": evidence_path,
            "replay_path": replay_path,
        }
    )
    rung_abort_path = args.staging_h5.with_suffix(".rung_abort.json")
    retained_leaves_transform = (
        UKFRSHMRCRetainedLeavesStageTransform.from_raw_frs_directory(
            args.frs_raw_dir,
            # A rung build's base deliberately carries a sampled subset of
            # source families; the stage receipts the dropped raw surface
            # instead of failing its completeness fence (#627).
            sampled_rung=args.sample_fraction != 1.0,
        )
    )
    _validate_distinct_paths(
        evidence_path=evidence_path,
        replay_path=replay_path,
        terminal_gate_path=gate_output_path,
        input_h5=args.input_h5,
        staging_h5=args.staging_h5,
        spi_tab=args.spi_tab,
        hmrc_ods=args.hmrc_ods,
        cgt_ods=args.cgt_ods,
        adult_tab=retained_leaves_transform.adult_tab_path,
        benefits_tab=retained_leaves_transform.benefits_tab_path,
        build_record_path=build_record_path,
        input_mass_reference_path=args.input_mass_reference_json,
        input_mass_exclusions_path=args.input_mass_exclusions,
        qrf_tail_exclusions_path=args.qrf_tail_exclusions,
        degenerate_exclusions_path=args.degenerate_exclusions,
        rung_abort_path=rung_abort_path,
    )
    # Read-only gate inputs are materialized before any sidecar unlink so a
    # path collision cannot consume a just-deleted file — and so a typo'd
    # register path dies here, before it can destroy a prior build's
    # sidecars. The default register is preflighted for the same reason: a
    # corrupted committed register must not surface hours later at
    # terminal-gate time.
    weighted_integrity_arguments = _weighted_integrity_arguments(args)
    if args.degenerate_exclusions is None:
        # Preflight the committed register without passing it: a corrupted
        # register dies here, while the absent artifact leaves the binding
        # resolving the same policy of record itself — the artifact stays
        # the review-time override channel, so a default run never
        # self-describes as an override.
        uk_default_degenerate_reviewed_exclusions()
        reviewed_degenerate_exclusions = None
    else:
        reviewed_degenerate_exclusions = load_uk_reviewed_exclusion_register(
            args.degenerate_exclusions,
            resource=UK_DEGENERATE_EXCLUSION_REGISTER_RESOURCE,
        )
    ledger_artifact = resolve_ledger_artifact(args)
    ledger_compilations = (
        None
        if ledger_artifact is None
        else {
            period: compile_uk_target_registry(
                ledger_artifact.facts, target_period=period
            )
            for period in (2023, 2025)
        }
    )
    if args.staging_candidate_input_sha256 is None:
        candidate = verify_certified_uk_candidate(args.input_h5)
    else:
        candidate = verify_staging_candidate_uk_input(
            args.input_h5,
            expected_sha256=args.staging_candidate_input_sha256,
        )
    evidence_path.unlink(missing_ok=True)
    replay_path.unlink(missing_ok=True)
    build_record_path.unlink(missing_ok=True)
    # A prior rung abort must never sit beside a fresh build's artifacts
    # (adversarial-review finding: a stale receipt contradicted a later
    # successful run at the same staging path).
    rung_abort_path.unlink(missing_ok=True)
    hmrc_transform = UKHMRCIncomeStageTransform(
        spi_tab_path=args.spi_tab,
        hmrc_ods_path=args.hmrc_ods,
        certified_candidate=candidate,
        retained_leaves_transform=retained_leaves_transform,
        seed=args.seed,
        qrf_estimators=args.qrf_estimators,
        sampled_rung=args.sample_fraction != 1.0,
    )
    stage_context.update(
        {
            "retained_leaves_transform": retained_leaves_transform,
            "hmrc_transform": hmrc_transform,
            "candidate": candidate,
        }
    )
    source_pins = _source_pins(
        candidate=candidate,
        retained_leaves_transform=retained_leaves_transform,
        hmrc_transform=hmrc_transform,
        cgt_ods_path=args.cgt_ods,
        ledger_artifact=ledger_artifact,
    )
    run_config = _staging_run_config(
        args,
        candidate=candidate,
        retained_leaves_transform=retained_leaves_transform,
        hmrc_transform=hmrc_transform,
        source_pins=source_pins,
    )
    attempt_context["code_pin"] = git_code_pin(_REPOSITORY)
    state.build_id = _new_national_build_id(
        rung=rung,
        sample_seed=args.sample_seed,
        seed=args.seed,
        timestamp=started_ts,
    )
    state.input_pins_digest = role_pins_digest(source_pins)
    state.identity_digest = hashlib.sha256(canonical_json_bytes(run_config)).hexdigest()
    append_phase(state, "configured")
    append_phase(state, "candidate_verified")
    append_phase(state, "inputs_pinned")
    # Without Ledger facts, this staging path has no real target-surface or
    # target-fit evidence; the schema-4 battery records the missing evidence
    # explicitly. Armed calibration builds add target evidence from the solve.
    # Input-mass evidence joins only when the caller supplies the licensed
    # frozen reference sidecar; QRF-tail is spec-armed and runs whenever the
    # frame evidence is present.
    gate_path_argument = (
        {"input_coverage_path": legacy_input_coverage_path}
        if legacy_input_coverage_path is not None
        else {"terminal_gate_path": terminal_gate_path}
    )
    checkpoint_arguments: dict[str, object] = {}
    if args.checkpoint_dir is not None:
        checkpoint_arguments = {
            "checkpoint_dir": args.checkpoint_dir,
            "run_config": run_config,
        }
    if (args.ledger_facts is None) != (
        args.national_calibration_diagnostics_json is None
    ):
        raise ValueError(
            "--ledger-facts and --national-calibration-diagnostics-json must "
            "be supplied together."
        )
    if args.release_candidate and args.ledger_facts is None:
        raise ValueError("a release candidate requires the national calibration stage.")
    calibration_transform = None
    calibration_stages: tuple[PlanStage, ...] = ()
    if args.ledger_facts is not None:
        assert ledger_artifact is not None  # resolved and pin-checked above
        assert ledger_compilations is not None
        calibration_year = load_uk_frs_release().calibration_year
        calibration_transform = UKNationalCalibrationStage(
            ledger_compilations[calibration_year],
            period=calibration_year,
        )
        calibration_stages = (
            PlanStage(
                name="national_calibration",
                transform=calibration_transform,
            ),
        )
    result = build_uk_national_dataset(
        input_h5=args.input_h5,
        staging_h5=args.staging_h5,
        release_id=args.release_id,
        calibration_diagnostics_sha256=args.calibration_diagnostics_sha256,
        reviewed_degenerate_exclusions=reviewed_degenerate_exclusions,
        stages=(
            *country_stage_plan(
                load_country_spec("uk"),
                uk_stage_implementations(
                    retained_leaves_transform=retained_leaves_transform,
                    hmrc_income_transform=hmrc_transform,
                ),
                # The manifest also declares the frs_spine pipeline root;
                # the national staging pipeline selects its own stages.
                stage_names=("frs_hmrc_retained_leaves", "hmrc_spi_income"),
            ).stages,
            # Runs after the SPI restoration so the taxable-income proxy
            # sees the restored income surface. Declared today in the
            # bespoke uk/cgt_source_stages.json; absorbing it into the
            # canonical source_stages.json is WS-E follow-up work.
            uk_capital_gains_imputation_stage(args.cgt_ods),
            *calibration_stages,
        ),
        **gate_path_argument,
        **weighted_integrity_arguments,
        **checkpoint_arguments,
        sample_fraction=args.sample_fraction,
        sample_seed=args.sample_seed,
        release_candidate=args.release_candidate,
        ledger_target_registry=(
            None
            if ledger_compilations is None
            else {
                period: compilation.registry
                for period, compilation in ledger_compilations.items()
            }
        ),
        # The frozen instrument supplies the parity trio's reference side;
        # the candidate side comes from the staged frame and the solve.
        parity_reference=(
            load_efrs_parity_reference() if calibration_transform is not None else None
        ),
    )
    if calibration_transform is not None:
        if calibration_transform.solve_result is None:
            raise RuntimeError(
                "national calibration diagnostics require the intact solve "
                "result; checkpoint-resumed runs must rerun calibration before "
                "writing calibration_diagnostics.json."
            )
        assert ledger_artifact is not None
        write_uk_calibration_diagnostics(
            calibration_transform.solve_result,
            args.national_calibration_diagnostics_json,
            result.frame,
            target_geography_levels=_uk_target_geography_levels(
                calibration_transform.registry
            ),
            target_registry=calibration_transform.registry,
            build={
                "build_id": state.build_id,
                "ledger_facts": ledger_artifact.provenance(),
                "code_pin": attempt_context["code_pin"],
                "source_pins": source_pins,
                "input_posture": _input_posture(candidate),
                "score_vs_enhanced_frs": None,
            },
        )
    append_phase(state, "build_completed")
    _write_stage_reports(
        evidence_path=evidence_path,
        replay_path=replay_path,
        candidate=candidate,
        retained_leaves_transform=retained_leaves_transform,
        hmrc_transform=hmrc_transform,
    )
    append_phase(state, "stage_reports_written")
    assert hmrc_transform.last_result is not None  # guarded by report writer
    hmrc_evidence = {
        "passed": True,
        "summary": _replay_summary(hmrc_transform.last_result),
    }
    artifact_paths = {
        "input_h5": result.input_h5,
        "staging_h5": result.staging_h5,
        "terminal_gates": gate_output_path,
        "hmrc_evidence": evidence_path,
        "hmrc_replay": replay_path,
        "spi_donor": args.spi_tab,
        "hmrc_surface": args.hmrc_ods,
        "frs_adult": retained_leaves_transform.adult_tab_path,
        "frs_benefits": retained_leaves_transform.benefits_tab_path,
    }
    artifacts = {role: _artifact_info(path) for role, path in artifact_paths.items()}
    build_record = _aggregate_build_record(
        result=result,
        artifacts=artifacts,
        retained_evidence=retained_leaves_transform.last_result.evidence(),
        family_evidence=hmrc_transform.last_result.evidence(),
        seed=args.seed,
        qrf_estimators=args.qrf_estimators,
        sample_fraction=args.sample_fraction,
        sample_seed=args.sample_seed,
        degenerate_exclusions_override=args.degenerate_exclusions is not None,
        ledger_artifact_provenance=(
            None if ledger_artifact is None else ledger_artifact.provenance()
        ),
        input_posture=_input_posture(candidate),
    )
    _write_json(build_record_path, build_record)
    append_phase(state, "build_record_written")
    payload = {
        "schema_version": 5,
        "build_kind": "uk_national_staging_dataset",
        "sampling": {
            "sample_fraction": float(args.sample_fraction),
            "sample_seed": int(args.sample_seed),
            "rung_token": UK_SAMPLE_RUNG_TOKENS[args.sample_fraction],
        },
        "stages": list(result.stage_names),
        "terminal_gates": dict(result.gate_report),
        "input_coverage": {
            "passed": result.input_coverage.passed,
            "failures": list(result.input_coverage.failures),
            "details": dict(result.input_coverage.details),
        },
        "artifacts": {
            **artifacts,
            "build_record": _artifact_info(build_record_path),
        },
        "hmrc_replay": hmrc_evidence,
    }
    state.gate_verdicts = _gate_verdicts_from_report(
        dict(result.gate_report),
        gate_output_path=gate_output_path,
    )
    state.artifact_location = local_artifact_reference(
        result.staging_h5,
        repository_hint=_REPOSITORY,
    )
    spool_path = _record_national_attempt(
        state=state,
        started_at=started_at,
        started_ts=started_ts,
        rung=rung,
        seed=logbook_seed,
        code_pin=str(attempt_context["code_pin"]),
        disposition="iterating",
        predecessor=attempt_context["predecessor"],
        spool_dir=spool_dir,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Wrote Logbook row: {spool_path}", file=sys.stderr)
    return 0


def _staging_run_config(
    args: argparse.Namespace,
    *,
    candidate: object,
    retained_leaves_transform: UKFRSHMRCRetainedLeavesStageTransform,
    hmrc_transform: UKHMRCIncomeStageTransform,
    source_pins: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    """The content-addressed identity of a checkpointed staging run.

    Everything that determines the stage outputs is pinned by content, not
    by path or stat: the verified certified-candidate digest, the raw-source
    digests (read from the transforms' own resolved paths, so the pinned
    files are exactly the files the stages consume), the seeds, the release
    coordinates, and the builder code identity (packaged sources plus the
    numeric-dependency versions). The stage runtime refuses to resume a
    checkpoint directory under a different config, so a drifted input,
    parameter, code change, or environment upgrade can never blend into an
    old run's prefix.
    """

    from microcosm.build.code_identity import builder_code_identity

    pins = source_pins or _source_pins(
        candidate=candidate,
        retained_leaves_transform=retained_leaves_transform,
        hmrc_transform=hmrc_transform,
        cgt_ods_path=args.cgt_ods,
    )

    return {
        "build_kind": "uk_national_staging_dataset",
        "release_id": str(args.release_id),
        "calibration_diagnostics_sha256": str(args.calibration_diagnostics_sha256),
        "seed": int(args.seed),
        "qrf_estimators": int(args.qrf_estimators),
        "sampling": {
            # Pinned as a string: run-config equality is exact over canonical
            # JSON, and float normalization across serializers is exactly the
            # ambiguity a run identity must not carry. Two rungs pointed at
            # one checkpoint directory refuse instead of cross-resuming.
            "sample_fraction": str(float(args.sample_fraction)),
            "sample_seed": int(args.sample_seed),
            "rung_token": UK_SAMPLE_RUNG_TOKENS[args.sample_fraction],
        },
        "certified_candidate": dict(pins["certified_candidate"]),
        "input_posture": _input_posture(candidate),
        "sources": {
            "adult_tab": dict(pins["adult_tab"]),
            "benefits_tab": dict(pins["benefits_tab"]),
            "spi_tab": dict(pins["spi_tab"]),
            "hmrc_ods": dict(pins["hmrc_ods"]),
            "cgt_ods": dict(pins["cgt_ods"]),
        },
        "code_identity": builder_code_identity(
            Path(__file__).resolve().parents[1],
            tool_path=Path(__file__).resolve(),
            distributions=(
                "h5py",
                "numpy",
                "pandas",
                "quantile-forest",
                "scikit-learn",
                "tables",
            ),
        ),
    }


def _artifact_info(path: str | Path) -> dict[str, str | int]:
    artifact = Path(path).resolve()
    digest = hashlib.sha256()
    with artifact.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(artifact),
        "sha256": digest.hexdigest(),
        "size_bytes": artifact.stat().st_size,
    }


def _aggregate_build_record(
    *,
    result: object,
    artifacts: dict[str, dict[str, str | int]],
    retained_evidence: dict[str, object],
    family_evidence: dict[str, object],
    seed: int,
    qrf_estimators: int,
    sample_fraction: float = 1.0,
    sample_seed: int = UK_SAMPLE_SEED_DEFAULT,
    degenerate_exclusions_override: bool = False,
    ledger_artifact_provenance: dict[str, object] | None = None,
    input_posture: dict[str, object] | None = None,
) -> dict[str, object]:
    """Return commit-safe aggregate evidence for one successful staging build."""

    details = dict(result.input_coverage.details)
    safe_artifacts = {
        role: {
            "sha256": str(info["sha256"]),
            "size_bytes": int(info["size_bytes"]),
        }
        for role, info in artifacts.items()
    }
    safe_artifacts["staging_h5"]["retention"] = "local_untracked"
    if ledger_artifact_provenance is not None:
        safe_artifacts["ledger_facts"] = ledger_artifact_provenance
    retained_sources = dict(retained_evidence.get("sources", {}))
    family_sources = dict(family_evidence.get("sources", {}))
    mass_changes = [
        {
            "entity": record.entity,
            "old_total": float(record.old_total),
            "new_total": float(record.new_total),
            "declared_factor": (
                None
                if record.declared_factor is None
                else float(record.declared_factor)
            ),
            "reason": record.reason,
        }
        for record in result.frame.mass_log
    ]
    release_evidence = dict(result.gate_report["release_evidence"])
    source_vintages = dict(family_evidence.get("source_vintages", {}))
    source_vintages["frs"] = load_uk_frs_release().vintage
    if ledger_artifact_provenance is not None:
        source_vintages["ledger_facts"] = ledger_artifact_provenance
    source_vintages["frs"] = load_uk_frs_release().vintage
    return {
        "schema_version": 3,
        "build_kind": "uk_national_staging_dataset",
        "status": "passed",
        "calibration_diagnostics_sha256": release_evidence[
            "calibration_diagnostics_sha256"
        ],
        "stages": list(result.stage_names),
        "parameters": {
            "seed": int(seed),
            "qrf_estimators": int(qrf_estimators),
            "sample_fraction": float(sample_fraction),
            "sample_seed": int(sample_seed),
            "rung_token": UK_SAMPLE_RUNG_TOKENS[sample_fraction],
            # Answers "did the operator invoke the override path" — the
            # operator-action record, kept path-free by contract. The signed
            # report's evidence answers the different question "which
            # register content governed" (``exclusions_policy``); a review
            # file byte-identical to the committed register makes the two
            # honestly disagree, which is why they carry distinct names.
            "degenerate_exclusions_override_supplied": (degenerate_exclusions_override),
        },
        "input_posture": dict(input_posture or {}),
        "sampling": (
            None if result.sampling_receipt is None else dict(result.sampling_receipt)
        ),
        "dataset": {
            "time_period": uk_time_period(result.frame),
            "entity_rows": {
                "person": len(result.frame.table("person")),
                "benunit": len(result.frame.table("benunit")),
                "household": len(result.frame.table("household")),
            },
            "household_weight_kind": uk_household_weight_kind(result.frame).value,
            "household_weight_total": float(
                result.frame.weights_for("household").total
            ),
            "mass_changes": mass_changes,
        },
        "source_rows": {
            "frs_adult": int(dict(retained_sources.get("adult", {})).get("rows", 0)),
            "frs_benefits": int(
                dict(retained_sources.get("benefits", {})).get("rows", 0)
            ),
            "spi_donor_used": int(
                dict(family_sources.get("spi_donor", {})).get("rows_used", 0)
            ),
        },
        "source_vintages": source_vintages,
        "terminal_gates": dict(result.gate_report),
        "input_coverage": {
            "passed": bool(result.input_coverage.passed),
            "failures": list(result.input_coverage.failures),
            "required_columns": int(details.get("required_columns", 0)),
            "reviewed_exclusion_columns": len(
                dict(details.get("reviewed_exclusions", {}))
            ),
            "missing": list(details.get("missing", ())),
            "degenerate_required": list(details.get("degenerate_required", ())),
            "insufficient_effective_mass": list(
                details.get("insufficient_effective_mass", ())
            ),
            "stale_exclusions": list(details.get("stale_exclusions", ())),
            "effective_mass_policy": dict(details.get("effective_mass_policy", {})),
            "family_effective_mass": dict(details.get("family_effective_mass", {})),
            "family_build_state": dict(details.get("family_build_state", {})),
        },
        "hmrc_replay": {
            "summary": dict(
                dict(family_evidence.get("targets", {})).get("classification", {})
            ),
            "post_draw_identity": dict(family_evidence.get("post_draw_identity", {})),
        },
        "artifacts": safe_artifacts,
    }


def _write_json(path: str | Path, payload: dict[str, object]) -> Path:
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _write_stage_reports(
    *,
    evidence_path: Path,
    replay_path: Path,
    candidate: object,
    retained_leaves_transform: UKFRSHMRCRetainedLeavesStageTransform,
    hmrc_transform: UKHMRCIncomeStageTransform,
) -> None:
    retained_result = retained_leaves_transform.last_result
    hmrc_result = hmrc_transform.last_result
    if retained_result is None or hmrc_result is None:
        raise RuntimeError(
            "UK national HMRC stages did not both complete; refusing to write "
            "partial or stale aggregate evidence."
        )
    payload = {
        "schema_version": 2,
        "base_candidate": {
            "path": str(candidate.path),
            "filename": candidate.filename,
            "tier": candidate.tier,
            "revision": candidate.revision,
            "sha256": candidate.sha256,
            "size_bytes": candidate.size_bytes,
        },
        "retained_leaves": retained_result.evidence(),
        "family": hmrc_result.evidence(),
    }
    _write_json(evidence_path, payload)
    # A checkpoint-resumed SPI stage carries no report object, only the
    # payload its real report produced at completion time; _write_json and
    # write_hmrc_replay_report share the exact serialization (indent=2,
    # sort_keys, trailing newline), so the resumed sidecar is byte-identical.
    replay_report = getattr(hmrc_result, "replay_report", None)
    if replay_report is not None:
        write_hmrc_replay_report(replay_report, replay_path)
    else:
        _write_json(replay_path, dict(_resumed_replay_payload(hmrc_result)))


def _resumed_replay_payload(hmrc_result: object) -> dict[str, object]:
    payload = getattr(hmrc_result, "replay_payload", None)
    if not isinstance(payload, dict):
        raise RuntimeError(
            "resumed SPI restoration carries no replay payload; the "
            "checkpoint record cannot feed the driver's stage reports."
        )
    return payload


def _replay_summary(hmrc_result: object) -> dict[str, object]:
    report = getattr(hmrc_result, "replay_report", None)
    if report is not None:
        return dict(report.summary)
    summary = _resumed_replay_payload(hmrc_result).get("summary")
    if not isinstance(summary, dict):
        raise RuntimeError("resumed SPI replay payload carries no summary block.")
    return dict(summary)


def _validate_distinct_paths(
    *,
    evidence_path: Path,
    replay_path: Path,
    terminal_gate_path: Path,
    input_h5: Path,
    staging_h5: Path,
    spi_tab: Path,
    hmrc_ods: Path,
    cgt_ods: Path,
    adult_tab: Path,
    benefits_tab: Path,
    build_record_path: Path,
    input_mass_reference_path: Path | None,
    input_mass_exclusions_path: Path | None,
    qrf_tail_exclusions_path: Path | None,
    degenerate_exclusions_path: Path | None,
    rung_abort_path: Path,
) -> None:
    paths = {
        "--input-h5": input_h5.resolve(),
        "--staging-h5": staging_h5.resolve(),
        "--spi-tab": spi_tab.resolve(),
        "--hmrc-ods": hmrc_ods.resolve(),
        "--cgt-ods": cgt_ods.resolve(),
        "--frs-raw-dir/adult.tab": adult_tab.resolve(),
        "--frs-raw-dir/benefits.tab": benefits_tab.resolve(),
        "--build-record-json": build_record_path.resolve(),
        "--terminal-gates-json/--input-coverage-json": terminal_gate_path.resolve(),
        "--hmrc-evidence-json": evidence_path.resolve(),
        "--hmrc-replay-json": replay_path.resolve(),
        "rung-abort receipt (derived from --staging-h5)": rung_abort_path.resolve(),
    }
    paths.update(
        (label, path.resolve())
        for label, path in {
            "--input-mass-reference-json": input_mass_reference_path,
            "--input-mass-exclusions": input_mass_exclusions_path,
            "--qrf-tail-exclusions": qrf_tail_exclusions_path,
            "--degenerate-exclusions": degenerate_exclusions_path,
        }.items()
        if path is not None
    )
    collisions = [
        (left_label, right_label, left_path, right_path)
        for (left_label, left_path), (right_label, right_path) in combinations(
            paths.items(),
            2,
        )
        if _paths_alias(left_path, right_path)
    ]
    if collisions:
        details = "; ".join(
            f"{left_label}, {right_label} -> {left_path} == {right_path}"
            for left_label, right_label, left_path, right_path in collisions
        )
        raise ValueError(
            "UK national build input, staging, and sidecar paths must be "
            f"pairwise distinct: {details}."
        )


def _uk_target_geography_levels(registry) -> dict[str, str]:
    payload = json.loads(
        importlib_resources.files("microcosm.build.uk")
        .joinpath("uk_national_targets.json")
        .read_text(encoding="utf-8")
    )
    targets = {row["target_id"]: row for row in payload["targets"]}
    levels: dict[str, str] = {}
    for spec in registry.specs:
        target_id = spec.metadata.get("contract_target_id")
        target = targets.get(str(target_id))
        if target is None:
            raise ValueError(
                f"UK calibration target {spec.name!r} references unknown "
                f"contract target {target_id!r}."
            )
        geography_levels = tuple(target.get("geography_levels") or ())
        if not geography_levels:
            raise ValueError(
                f"UK calibration target {spec.name!r} has no geography level."
            )
        levels[spec.to_target().row_name] = str(geography_levels[0])
    return levels


def _paths_alias(left: Path, right: Path) -> bool:
    """Conservatively identify aliases before any build path is unlinked."""

    if left == right:
        return True
    try:
        if left.exists() and right.exists() and left.samefile(right):
            return True
    except OSError:
        # The case-folded resolved identity below remains a safe fallback for
        # a path that changes between the existence and samefile checks.
        pass
    # macOS volumes are commonly case-insensitive even though Path.resolve()
    # preserves caller casing. Reject case-only distinctions everywhere: a
    # destructive build tool has no legitimate need for them, and doing so
    # also protects outputs that do not exist yet.
    return str(left).casefold() == str(right).casefold()


if __name__ == "__main__":
    raise SystemExit(main())
