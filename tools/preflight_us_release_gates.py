"""Preflight the US release gates that are statically knowable without a solve.

Each Build M release attempt (9/10/11) spent ~2h of calibration to surface one
gate-group failure already determined by the base pool, the frozen selection,
and the registry. This tool recovers those signals in minutes and reports, per
check, PASS / FAIL / AT-RISK with the measured numbers. Run it:

- when the base build exits (before launching a release),
- before any release launch, and
- after any change to the selection-source manifest or the target/coverage
  registry.

Example (read-only against the artifacts)::

    uv run python tools/preflight_us_release_gates.py \
        --base-h5 out/base-m/base_populace_us_2024_puf_support.h5 \
        --selection-source-manifest inputs/buildm_keogh_swap_selection_source.json \
        --export-input-mass-reference-h5 forensics/populace_us_2024.h5

A release built on a fresh base with no selection source (a new lineage,
``docs/us-release-build-rule.md`` section 3) has no frozen selection to carry
over. Say so explicitly with ``--new-lineage`` in place of
``--selection-source-manifest``: the selection-carryover check is recorded as
SKIPPED with that reason, the base-level half of it (the materialized PUF
capital-gains own-tail) still runs, and every other check runs unchanged on the
whole base. The two options are mutually exclusive; with neither, the manifest
is required as before. With ``--release-manifest``, ``--new-lineage`` also
requires that release to record no selection source.

An SPM unit with no classified adult is a FAIL here: in spm-calculator 1.0.0
one such unit raises ``SPM_COMPOSITION_REQUIRED`` for the whole population's SPM
measurement. The release tool refuses the same composition by name in its
batched pre-export gate report — but only after a full calibration, because it
grades the calibrated export frame. This is the same verdict on the pool, in
seconds, before the solve.

Exit code: 1 on any static-check FAIL, 2 on static AT-RISK only, 0 clean. A
carried red base-pool battery is human-review evidence and does not by itself
change that exit code. When ``--release-manifest`` is supplied, its base-pool
receipt must exactly match the pool authenticated by this preflight.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Reuse the release tool's export-mass register and engine input-variable
# surface — never re-declared here (this file sits beside it in tools/).
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "packages" / "microcosm-build" / "src")
)

from microcosm.build.us_runtime.h5_io import (  # noqa: E402
    US_MULTISPINE_POOL_H5_ARTIFACT_KIND,
)
from microcosm.build.us_runtime.release_gate_preflight import (  # noqa: E402
    MAX_REPORTED_SPM_UNITS_HARD_CAP,
    run_preflight,
)

_ALLOW_GATE_FAILED_BASE_POOL_FLAG = "--allow-gate-failed-base-pool"
_CARRIED_BATTERY_PAYLOAD_KEY = "carried_base_pool_agreement_battery"
_NEW_LINEAGE_FLAG = "--new-lineage"
_SELECTION_SOURCE_MANIFEST_FLAG = "--selection-source-manifest"


def _json_object(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return value


def _load_release_base_pool_receipt(path: Path) -> dict[str, object] | None:
    """Read one optional base-pool receipt from a built release manifest."""

    release_manifest = _json_object(
        json.loads(path.read_text()),
        label=f"release manifest {path}",
    )
    build = release_manifest.get("build")
    if not isinstance(build, dict):
        raise ValueError(f"Release manifest {path} has no build object.")
    if "base_pool" not in build:
        return None
    base_pool = build.get("base_pool")
    if not isinstance(base_pool, dict):
        raise ValueError(
            f"Release manifest {path} build.base_pool must be a JSON object."
        )
    return base_pool


def _carried_base_pool_battery(
    base_pool: dict[str, object] | None,
    *,
    path: Path,
) -> dict[str, object] | None:
    """Validate the non-blocking red verdict carried by one pool receipt."""

    if base_pool is None:
        return None
    agreement_gate_reference = base_pool.get("agreement_gate_reference")
    carries_gate_failed_override = (
        base_pool.get("allow_gate_failed_base_pool") is True
        or base_pool.get("status") == "gate_failed"
        or (
            isinstance(agreement_gate_reference, dict)
            and agreement_gate_reference.get("battery_status") == "red"
        )
    )
    if not carries_gate_failed_override:
        return None
    if (
        base_pool.get("artifact_kind") != US_MULTISPINE_POOL_H5_ARTIFACT_KIND
        or base_pool.get("status") != "gate_failed"
        or base_pool.get("simulation_ready") is not False
        or base_pool.get("allow_gate_failed_base_pool") is not True
    ):
        raise ValueError(
            f"Release manifest {path} has an incoherent gate-failed base-pool "
            "carriage receipt."
        )
    gate_reference = _json_object(
        agreement_gate_reference,
        label=(f"release manifest {path} build.base_pool.agreement_gate_reference"),
    )
    failures = gate_reference.get("failures")
    failure_count = gate_reference.get("failure_count")
    gates_json_sha256 = gate_reference.get("gates_json_sha256")
    verdict = gate_reference.get("verdict")
    if (
        gate_reference.get("passed") is not False
        or gate_reference.get("battery_status") != "red"
        or not isinstance(failures, list)
        or not all(
            isinstance(failure, dict)
            and isinstance(failure.get("gate"), str)
            and bool(failure.get("gate"))
            and isinstance(failure.get("message"), str)
            for failure in failures
        )
        or type(failure_count) is not int
        or failure_count != len(failures)
        or not isinstance(gates_json_sha256, str)
        or len(gates_json_sha256) != 64
        or any(character not in "0123456789abcdef" for character in gates_json_sha256)
        or not isinstance(verdict, dict)
        or verdict.get("passed") is not False
    ):
        raise ValueError(
            f"Release manifest {path} has an incomplete or inconsistent "
            "carried red agreement-battery verdict."
        )
    verdict_gates = verdict.get("gates")
    if not isinstance(verdict_gates, dict):
        raise ValueError(f"Release manifest {path} has no full carried gate verdict.")
    verdict_failures: list[dict[str, str]] = []
    for gate_name, gate_payload in verdict_gates.items():
        if not isinstance(gate_name, str) or not isinstance(gate_payload, dict):
            raise ValueError(
                f"Release manifest {path} has a malformed carried gate verdict."
            )
        gate_failures = gate_payload.get("failures")
        gate_passed = gate_payload.get("passed")
        if (
            type(gate_passed) is not bool
            or not isinstance(gate_failures, list)
            or not all(isinstance(failure, str) for failure in gate_failures)
        ):
            raise ValueError(
                f"Release manifest {path} has a malformed carried failure list."
            )
        if gate_passed is bool(gate_failures):
            raise ValueError(
                f"Release manifest {path} has an incoherent nested gate verdict."
            )
        verdict_failures.extend(
            {"gate": gate_name, "message": failure} for failure in gate_failures
        )
    if not verdict_failures or verdict_failures != failures:
        raise ValueError(
            f"Release manifest {path} failure summary does not match its full "
            "carried gate verdict."
        )
    return {
        "battery_status": "red",
        "pool_status": "gate_failed",
        "simulation_ready": False,
        "allow_gate_failed_base_pool": True,
        "flag": _ALLOW_GATE_FAILED_BASE_POOL_FLAG,
        "gates_json_sha256": gates_json_sha256,
        "failure_count": failure_count,
        "failures": [dict(failure) for failure in failures],
        "agreement_gate_reference": dict(gate_reference),
        "publication_decision": "human_review_required",
        "affects_exit_code": False,
    }


def _require_release_without_selection_source(path: Path) -> None:
    """Refuse ``--new-lineage`` for a release that carried a frozen selection.

    The release tool records ``build.selection_source`` as ``{"enabled":
    false}`` when it ran without one, and as the selection report otherwise.
    Skipping the carryover check is sound only for the former, so a built
    release named beside ``--new-lineage`` must be the former.
    """

    release_manifest = _json_object(
        json.loads(path.read_text()),
        label=f"release manifest {path}",
    )
    build = release_manifest.get("build")
    if not isinstance(build, dict):
        raise ValueError(f"Release manifest {path} has no build object.")
    selection_source = build.get("selection_source")
    if not (
        isinstance(selection_source, dict) and selection_source.get("enabled") is False
    ):
        raise ValueError(
            f"{_NEW_LINEAGE_FLAG} was set, but release manifest {path} "
            f"build.selection_source is {selection_source!r}, not a record of "
            "a build without a selection source. A release built with a frozen "
            "selection is not a new lineage: preflight it with "
            f"{_SELECTION_SOURCE_MANIFEST_FLAG}."
        )


def _require_matching_release_base_pool(
    authenticated: dict[str, object] | None,
    carried: dict[str, object] | None,
    *,
    path: Path,
) -> None:
    """Bind a release manifest to the exact pool preflight authenticated."""

    if authenticated != carried:
        raise ValueError(
            f"Release manifest {path} build.base_pool does not exactly match "
            "the base-pool receipt authenticated by this preflight."
        )


def _carried_battery_banner(carried: dict[str, object]) -> str:
    count = int(carried["failure_count"])
    noun = "FAILURE" if count == 1 else "FAILURES"
    lines = [
        "=" * 72,
        f"CARRIED BASE-POOL AGREEMENT BATTERY: RED — {count} {noun}",
        "Pool status: gate_failed; simulation_ready: false",
        f"Build opt-in used: {carried['flag']}",
        f"Pool gates JSON SHA-256: {carried['gates_json_sha256']}",
        "Publication decision: HUMAN REVIEW REQUIRED",
        "This carried verdict does not alter the preflight exit code.",
    ]
    lines.extend(
        f"  FAILURE [{failure['gate']}]: {failure['message']}"
        for failure in carried["failures"]
    )
    lines.append("=" * 72)
    return "\n".join(lines)


def _non_negative_int(value: str) -> int:
    """An argparse ``int`` that refuses a negative cap.

    A negative value would reach the report's ``[:max_reported]`` slice and
    silently mean "every offending unit except the last |N|" — the opposite of
    a cap. The check clamps defensively too; refusing here is what tells the
    operator their flag was wrong instead of quietly repairing it.
    """
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected an integer, got {value!r}"
        ) from None
    if parsed < 0:
        raise argparse.ArgumentTypeError(
            f"must be zero or more, got {parsed}: a negative cap would report "
            "every offending unit except the last few, not cap the report"
        )
    return parsed


def _parser(*, selection_source_required: bool = True) -> argparse.ArgumentParser:
    """The CLI parser.

    ``selection_source_required`` is lifted only when the command line carries
    ``--new-lineage`` (see :func:`main`), so a default invocation refuses a
    missing manifest with argparse's usual required-arguments error, unchanged.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Statically preview the US release gates (selection carryover, "
            "zero-support, export-mass parity risk, reform-coverage smoke "
            "support, SPM measurement composition) without running a "
            "calibration solve."
        )
    )
    parser.add_argument(
        "--base-h5",
        required=True,
        type=Path,
        help="Base pool H5 (read-only).",
    )
    parser.add_argument(
        "--allow-gate-failed-base-pool",
        action="store_true",
        help=(
            "Allow an authenticated current stacked --base-h5 pool whose "
            "terminal battery is red. The verdict is displayed prominently "
            "but does not itself determine the preflight exit code."
        ),
    )
    parser.add_argument(
        _SELECTION_SOURCE_MANIFEST_FLAG,
        required=selection_source_required,
        type=Path,
        default=None,
        help=(
            "Frozen selection-source manifest JSON. Required unless "
            f"{_NEW_LINEAGE_FLAG} is set; mutually exclusive with it."
        ),
    )
    parser.add_argument(
        _NEW_LINEAGE_FLAG,
        action="store_true",
        help=(
            "The release is built on a fresh base with no selection source "
            "(a new lineage): there is no prior selection to carry over, so "
            "the selection-carryover check is recorded as SKIPPED with that "
            "reason. The base-level PUF capital-gains own-tail refusal inside "
            "it still runs, and every other check runs unchanged on the whole "
            "base. Mutually exclusive with "
            f"{_SELECTION_SOURCE_MANIFEST_FLAG}."
        ),
    )
    parser.add_argument(
        "--release-manifest",
        type=Path,
        help=(
            "Optional built release_manifest.json. A carried gate-failed "
            "base-pool battery is displayed as prominent, non-blocking "
            "evidence for the human publication decision after its receipt "
            "is matched to the authenticated --base-h5 pool."
        ),
    )
    parser.add_argument(
        "--export-input-mass-reference-h5",
        type=Path,
        default=None,
        help=(
            "Reference H5 for the export-mass parity band (read-only). Omit to "
            "skip the parity-risk check."
        ),
    )
    parser.add_argument(
        "--ledger-facts",
        type=Path,
        default=None,
        help=(
            "Optional Ledger consumer facts feed (directory or "
            "consumer_facts.jsonl). Required to preview zero-support — the "
            "compiled fiscal-target surface comes from it. Omitted -> the "
            "zero-support check is SKIPPED."
        ),
    )
    parser.add_argument(
        "--ledger-facts-sha256",
        default=None,
        help="Optional pin: expected SHA-256 of consumer_facts.jsonl.",
    )
    parser.add_argument(
        "--congressional-district-vintage-crosswalk",
        type=Path,
        default=None,
        help=(
            "CD vintage crosswalk the feed's congressional-district facts are "
            "translated through before the target surface is compiled. "
            "Defaults to the canonical packaged crosswalk, as the release "
            "tool's option of the same name does; pass the release run's "
            "replacement if it used one."
        ),
    )
    parser.add_argument(
        "--target-period",
        default=2024,
        help="Build period the fiscal targets are compiled for (default 2024).",
    )
    parser.add_argument(
        "--relative-tolerance",
        type=float,
        default=0.5,
        help=(
            "Export-mass parity band half-width (matches the release tool's "
            "--input-mass-relative-tolerance; default 0.5)."
        ),
    )
    parser.add_argument(
        "--minimum-reference-total",
        type=float,
        default=1e9,
        help=(
            "Reference-mass floor below which parity is not checked (matches "
            "the release tool's --input-mass-minimum-reference-total; default "
            "1e9)."
        ),
    )
    parser.add_argument(
        "--max-reported-spm-units",
        type=_non_negative_int,
        default=None,
        help=(
            "How many SPM units with no classified adult the composition check "
            "names individually, with their members' age bands — under_15 / "
            "15_to_17 / 18_plus / unknown, never an exact age (default 20, "
            f"clamped to at most {MAX_REPORTED_SPM_UNITS_HARD_CAP}). The "
            "failure line reports the full count either way."
        ),
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Write the machine-readable report JSON to this path.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    # Only the explicit flag lifts the manifest requirement. An abbreviation
    # that argparse would expand to it leaves the manifest required, which
    # fails closed rather than skipping a check nobody named.
    parser = _parser(selection_source_required=_NEW_LINEAGE_FLAG not in arguments)
    args = parser.parse_args(arguments)
    if args.new_lineage and args.selection_source_manifest is not None:
        parser.error(
            f"argument {_NEW_LINEAGE_FLAG}: not allowed with argument "
            f"{_SELECTION_SOURCE_MANIFEST_FLAG} (a new lineage has no prior "
            "selection to carry over; a release built with a frozen selection "
            "must have that selection preflighted)"
        )
    if args.new_lineage and args.release_manifest is not None:
        _require_release_without_selection_source(args.release_manifest)
    release_base_pool = (
        _load_release_base_pool_receipt(args.release_manifest)
        if args.release_manifest is not None
        else None
    )
    carried = (
        _carried_base_pool_battery(
            release_base_pool,
            path=args.release_manifest,
        )
        if args.release_manifest is not None
        else None
    )

    try:
        target_period: int | str = int(args.target_period)
    except (TypeError, ValueError):
        target_period = args.target_period

    report = run_preflight(
        base_h5=args.base_h5,
        selection_source_manifest=args.selection_source_manifest,
        new_lineage=args.new_lineage,
        export_input_mass_reference_h5=args.export_input_mass_reference_h5,
        ledger_facts=args.ledger_facts,
        ledger_facts_sha256=args.ledger_facts_sha256,
        congressional_district_vintage_crosswalk=(
            args.congressional_district_vintage_crosswalk
        ),
        target_period=target_period,
        relative_tolerance=args.relative_tolerance,
        minimum_reference_total=args.minimum_reference_total,
        allow_gate_failed_base_pool=args.allow_gate_failed_base_pool,
        **(
            {}
            if args.max_reported_spm_units is None
            else {"max_reported_spm_units": args.max_reported_spm_units}
        ),
    )
    if args.release_manifest is not None:
        _require_matching_release_base_pool(
            report.base_pool,
            release_base_pool,
            path=args.release_manifest,
        )
    if carried is not None:
        print(_carried_battery_banner(carried))
    print(report.human_table())
    payload = report.to_dict()
    exit_code = report.exit_code
    if carried is not None:
        payload[_CARRIED_BATTERY_PAYLOAD_KEY] = carried
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=False))
        print(f"\nWrote machine-readable report to {args.json_out}")
    else:
        print("\n--- machine-readable report ---")
        print(json.dumps(payload, indent=2, sort_keys=False))
    if carried is not None:
        print(
            "\nCARRIED RED BATTERY: HUMAN REVIEW REQUIRED; "
            f"automated preflight exit remains {exit_code}."
        )

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
