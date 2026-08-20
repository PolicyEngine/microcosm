#!/usr/bin/env python3
"""Run one cold US F1 build or compare the four typed build receipts."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

from microcosm.build.spec_engine.artifact_comparison import ArtifactComparisonError
from microcosm.build.spec_engine.compiler_ir import CompiledSpecIR, compile_spec
from microcosm.build.spec_engine.f1_certification import (
    CERTIFICATION_JSON_FILENAME,
    CERTIFICATION_MARKDOWN_FILENAME,
    COLD_BUILD_RECEIPT_FILENAME,
    PRODUCTION_EVIDENCE_FILENAME,
    F1CertificationError,
    F1ColdBuildReceipt,
    F1RunRequest,
    assert_f1_selector_coverage_contract_current,
    assert_request_matches_evidence,
    atomic_write_bytes,
    atomic_write_json,
    certification_markdown,
    compare_f1_cold_build_receipts,
    emit_f1_cold_build_receipt,
    load_f1_cold_build_receipt,
    load_f1_production_evidence,
    resume_audit_from_evidence,
)
from microcosm.build.spec_engine.loader import load_bundle
from microcosm.build.spec_engine.plan_lock import (
    PlanLockError,
    assert_plan_lock_payload_current,
    plan_lock_payload,
)
from microcosm.build.spec_engine.runtime_authorities import compile_runtime_authorities
from microcosm.build.us_runtime.pool_artifact_coverage import (
    compile_pool_artifact_coverage,
)
from microcosm.build.us_runtime.pool_kernel_authority import USPoolKernelAuthorities
from microcosm.build.us_runtime.pool_runtime_plan import USPoolRuntimePlan
from microcosm.build.us_runtime.spec_authority import compile_us_spec_authority

_SOURCE_ARGUMENTS = (
    ("asec_raw_stage_h5", "--asec-raw-stage-h5"),
    ("asec_raw_stage_h5_sha256", "--asec-raw-stage-h5-sha256"),
    ("acs_household_zip", "--acs-household-zip"),
    ("acs_household_zip_sha256", "--acs-household-zip-sha256"),
    ("acs_person_zip", "--acs-person-zip"),
    ("acs_person_zip_sha256", "--acs-person-zip-sha256"),
    ("acs_rent_h5", "--acs-rent-h5"),
    ("acs_rent_h5_sha256", "--acs-rent-h5-sha256"),
    ("puf_h5", "--puf-h5"),
    ("puf_h5_sha256", "--puf-h5-sha256"),
    ("puf_source_year_csv", "--puf-source-year-csv"),
    ("puf_source_year_csv_sha256", "--puf-source-year-csv-sha256"),
)
_PATH_SOURCE_DESTINATIONS = frozenset(
    {
        "asec_raw_stage_h5",
        "acs_household_zip",
        "acs_person_zip",
        "acs_rent_h5",
        "puf_h5",
        "puf_source_year_csv",
    }
)
_STANDARD_SAMPLE_FRACTIONS = (0.01, 0.04, 0.10, 0.25, 1.0)
_SANITIZED_ENVIRONMENT_KEYS = frozenset(
    {
        "POPULACE_LOGBOOK_PREV_ROW_DIGEST",
        "POPULACE_LEDGER_URL",
        "POPULACE_LEDGER_KEY",
        "POPULACE_LEDGER_API_KEY",
    }
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser(
        "run",
        aliases=("build",),
        help="Run exactly one cold build and emit its typed receipt.",
    )
    run.add_argument("--mode", choices=("constants", "bundle"), required=True)
    run.add_argument(
        "--sample-fraction",
        choices=_STANDARD_SAMPLE_FRACTIONS,
        type=float,
        required=True,
    )
    run.add_argument("--seed", type=_nonnegative_seed, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    for destination, option in _SOURCE_ARGUMENTS:
        run.add_argument(
            option,
            dest=destination,
            type=Path if destination in _PATH_SOURCE_DESTINATIONS else _sha256,
            required=True,
        )
    run.set_defaults(handler=_run_one_cold_build)

    compare = subparsers.add_parser(
        "compare",
        help="Apply the D4 verdict to constants A/B and bundle C/D receipts.",
    )
    compare.add_argument("--constants-a", type=Path, required=True)
    compare.add_argument("--constants-b", type=Path, required=True)
    compare.add_argument("--bundle-a", type=Path, required=True)
    compare.add_argument("--bundle-b", type=Path, required=True)
    compare.add_argument("--output-root", type=Path, required=True)
    compare.set_defaults(handler=_compare_four_receipts)

    resume = subparsers.add_parser(
        "resume-gate",
        help="Document the host kill/resume predicate; execute nothing.",
    )
    resume.add_argument("--mode", choices=("constants", "bundle"), required=True)
    resume.add_argument(
        "--sample-fraction",
        choices=_STANDARD_SAMPLE_FRACTIONS,
        type=float,
        required=True,
    )
    resume.add_argument("--seed", type=_nonnegative_seed, required=True)
    resume.add_argument("--output", type=Path, required=True)
    resume.set_defaults(handler=_write_resume_gate)
    return parser


def _run_one_cold_build(args: argparse.Namespace) -> int:
    output_root = _claim_absent_output_root(args.output_root)
    pool_output = output_root / "pool.h5"
    checkpoint_root = output_root / "checkpoints"
    evidence_path = output_root / PRODUCTION_EVIDENCE_FILENAME
    plan_path = output_root / "plan.lock.json"
    receipt_path = output_root / COLD_BUILD_RECEIPT_FILENAME
    command = [
        sys.executable,
        str(Path(__file__).resolve().with_name("build_us_multispine_pool.py")),
    ]
    for destination, option in _SOURCE_ARGUMENTS:
        command.extend((option, str(getattr(args, destination))))
    command.extend(
        (
            "--out",
            str(pool_output),
            "--checkpoint-root",
            str(checkpoint_root),
            "--sample-fraction",
            str(args.sample_fraction),
            "--sample-seed",
            str(args.seed),
            "--clone-attachment-fraction",
            "1.0",
            "--clone-attachment-seed",
            str(args.seed),
            "--config-authority",
            args.mode,
            "--resume-policy",
            "forbid",
            "--f1-evidence-out",
            str(evidence_path),
        )
    )
    environment = dict(os.environ)
    for key in _SANITIZED_ENVIRONMENT_KEYS:
        environment.pop(key, None)
    return_code = _launch_pool_child(command, environment)
    if return_code != 0:
        return return_code

    _refuse_preexisting_runner_outputs((plan_path, receipt_path))

    evidence = load_f1_production_evidence(evidence_path)
    if evidence.mode != args.mode:
        raise F1CertificationError(
            f"production evidence mode {evidence.mode!r} differs from {args.mode!r}"
        )
    _assert_current_plan(evidence.plan_lock)
    request = F1RunRequest(
        sample_fraction=args.sample_fraction,
        seed=args.seed,
        clone_attachment_seed=args.seed,
    )
    assert_request_matches_evidence(request, evidence)
    resume_audit_from_evidence(evidence)
    atomic_write_bytes(
        plan_path,
        _canonical_plan_lock_bytes(evidence.plan_lock),
    )
    receipt = emit_f1_cold_build_receipt(
        receipt_path,
        request=request,
        production_evidence=evidence,
    )
    print(f"Wrote cold build receipt: {receipt_path}")
    print(f"Cold build receipt SHA-256: {receipt.receipt_sha256}")
    return 0


def _compare_four_receipts(args: argparse.Namespace) -> int:
    paths = (
        args.constants_a,
        args.constants_b,
        args.bundle_a,
        args.bundle_b,
    )
    resolved_paths = tuple(Path(path).resolve() for path in paths)
    if len(set(resolved_paths)) != 4:
        raise F1CertificationError("compare requires four distinct receipt paths")
    constants_a, constants_b, bundle_a, bundle_b = (
        load_f1_cold_build_receipt(path) for path in paths
    )
    _assert_current_plan(constants_a.production_evidence.plan_lock)
    _assert_current_selector_coverage_contracts(
        (constants_a, constants_b, bundle_a, bundle_b)
    )
    verdict = compare_f1_cold_build_receipts(
        constants_a=constants_a,
        constants_b=constants_b,
        bundle_a=bundle_a,
        bundle_b=bundle_b,
    )
    output_root = Path(args.output_root).resolve()
    json_path = output_root / CERTIFICATION_JSON_FILENAME
    markdown_path = output_root / CERTIFICATION_MARKDOWN_FILENAME
    existing = [
        str(path) for path in (json_path, markdown_path) if os.path.lexists(path)
    ]
    if existing:
        raise F1CertificationError(
            f"refusing to overwrite certification output: {sorted(existing)}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(json_path, verdict.to_wire())
    atomic_write_bytes(markdown_path, certification_markdown(verdict).encode("utf-8"))
    print(f"Wrote certification JSON: {json_path}")
    print(f"Wrote certification Markdown: {markdown_path}")
    print(f"D4 verdict: {'PASS' if verdict.body['passed'] else 'FAIL'}")
    return 0 if verdict.body["passed"] else 1


def _write_resume_gate(args: argparse.Namespace) -> int:
    plan = _compile_current_plan_lock()
    execution_abi = _mapping(
        plan.get("execution_abi"), location="plan_lock/execution_abi"
    )
    predicate = _mapping(
        execution_abi.get("resume_predicate"),
        location="execution_abi/resume_predicate",
    )
    output = Path(args.output).resolve()
    if os.path.lexists(output):
        raise F1CertificationError(f"refusing to overwrite resume gate: {output}")
    candidate_order = _string_list(
        predicate.get("candidate_order"), location="resume_predicate/candidate_order"
    )
    identity_fields = _string_list(
        predicate.get("identity_fields"), location="resume_predicate/identity_fields"
    )
    validators = _string_list(
        predicate.get("integrity_validators"),
        location="resume_predicate/integrity_validators",
    )
    required = _mapping(
        predicate.get("required_artifact_roles_by_stage"),
        location="resume_predicate/required_artifact_roles_by_stage",
    )
    lines = [
        "# US F1 kill/resume predicate",
        "",
        "> **DOCUMENTATION ONLY — this command executes no build, process kill, or resume.**",
        "",
        f"- Authority mode: `{args.mode}`",
        f"- Sample fraction: `{args.sample_fraction}`",
        f"- Seed: `{args.seed}`",
        f"- Execution ABI: `{execution_abi.get('sha256')}`",
        f"- Candidate order: `{', '.join(candidate_order)}`",
        f"- Last durable stage: `{predicate.get('last_durable_stage')}`",
        "",
        "## Sealed identity fields",
        "",
        *[f"- `{value}`" for value in identity_fields],
        "",
        "## Sealed integrity validators",
        "",
        *[f"- `{value}`" for value in validators],
        "",
        "## Required artifacts by candidate stage",
        "",
    ]
    for stage in candidate_order:
        roles = _string_list(required.get(stage), location=f"required/{stage}")
        lines.append(f"- `{stage}`: {', '.join(f'`{role}`' for role in roles)}")
    lines.extend(
        [
            "",
            "## Cold-build zero-resume predicate",
            "",
            "Every four-build receipt must record `deepest_resumed_stage: null`, primary QRF `resume_status: initialized`, zero resumed durable stages, and zero per-target ACS `load_status: resumed` / `source: checkpoint` counts. The raw build must use `--resume-policy forbid`.",
            "",
            "## Host procedure",
            "",
            "1. Allocate a fresh output and checkpoint namespace; do not use any of the four cold-build roots.",
            "2. Launch the raw pool command with `--resume-policy allow`, the mode/rung/seed above, and the same six source pins as the cold build.",
            "3. After the assembled checkpoint is durable and primary-QRF work is active, terminate that process from the host.",
            "4. Relaunch the exact same command with the exact same output and checkpoint paths.",
            "5. Require the compiler predicate to accept the deepest valid candidate and require positive typed resume evidence (`deepest_resumed_stage` non-null or a real QRF/target-bank resume).",
            "6. Collect the complete plan-derived artifact vector and require it to equal the uninterrupted same-mode cold build.",
            "",
            "The four certification builds use `--resume-policy forbid`; this separate exercise is not marked PASS by the four-receipt comparator.",
            "",
        ]
    )
    atomic_write_bytes(output, "\n".join(lines).encode("utf-8"))
    print(f"Wrote documentation-only resume gate: {output}")
    return 0


def _launch_pool_child(command: list[str], environment: Mapping[str, str]) -> int:
    """The sole subprocess launch point, kept narrow for proof-oriented tests."""

    completed = subprocess.run(command, env=dict(environment), check=False)
    return completed.returncode


def _assert_current_plan(observed: Mapping[str, object]) -> None:
    assert_plan_lock_payload_current(
        _compile_current_spec(),
        observed,
    )


def _assert_current_selector_coverage_contracts(
    receipts: tuple[F1ColdBuildReceipt, ...],
) -> None:
    """Bind receipt member coverage to freshly compiled US authority."""

    compiled = _compile_current_spec()
    runtime_plan = USPoolRuntimePlan.from_spec_authority(
        compile_us_spec_authority(compile_runtime_authorities(compiled))
    )
    expected = compile_pool_artifact_coverage(
        runtime_plan,
        USPoolKernelAuthorities.from_runtime_plan(runtime_plan),
    )
    for receipt in receipts:
        assert_f1_selector_coverage_contract_current(
            receipt.production_evidence.coverage,
            expected,
        )


def _compile_current_plan_lock() -> dict[str, object]:
    return plan_lock_payload(_compile_current_spec())


@lru_cache(maxsize=1)
def _compile_current_spec() -> CompiledSpecIR:
    return compile_spec(load_bundle("us"))


def _claim_absent_output_root(path: Path) -> Path:
    requested = Path(path)
    if os.path.lexists(requested):
        raise F1CertificationError(
            f"cold build output root must not pre-exist: {requested}"
        )
    output = requested.resolve()
    if output == Path(output.anchor):
        raise F1CertificationError("output root cannot be a filesystem root")
    if os.path.lexists(output):
        raise F1CertificationError(
            f"cold build output root must not pre-exist: {requested}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        output.mkdir()
    except FileExistsError as error:
        raise F1CertificationError(
            f"cold build output root was claimed concurrently: {output}"
        ) from error
    return output


def _refuse_preexisting_runner_outputs(paths: tuple[Path, ...]) -> None:
    """Keep the pool child from pre-empting runner-owned receipt surfaces."""

    existing = [str(path) for path in paths if os.path.lexists(path)]
    if existing:
        raise F1CertificationError(
            f"refusing pre-existing runner output: {sorted(existing)}"
        )


def _canonical_plan_lock_bytes(value: Mapping[str, object]) -> bytes:
    from microcosm.build.spec_engine.canonical import canonical_json_bytes

    return canonical_json_bytes(value)


def _mapping(value: object, *, location: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise F1CertificationError(f"{location}: object required")
    result: dict[str, object] = {}
    for key, child in value.items():
        if not isinstance(key, str):
            raise F1CertificationError(f"{location}: string keys required")
        result[key] = child
    return result


def _string_list(value: object, *, location: str) -> list[str]:
    if not isinstance(value, list):
        raise F1CertificationError(f"{location}: array required")
    if any(not isinstance(child, str) or not child for child in value):
        raise F1CertificationError(f"{location}: non-empty strings required")
    return list(value)


def _nonnegative_seed(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("seed must be an integer") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("seed must be non-negative")
    return parsed


def _sha256(value: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise argparse.ArgumentTypeError("lowercase SHA-256 required")
    return value


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (
        ArtifactComparisonError,
        F1CertificationError,
        PlanLockError,
        OSError,
    ) as error:
        print(f"F1 certification error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
