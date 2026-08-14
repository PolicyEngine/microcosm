"""Build the raw UK FRS spine Frame from pinned local tabs."""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from microcosm.build.country_spec import country_stage_plan, load_country_spec
from microcosm.build.logbook import canonical_json_bytes
from microcosm.build.logbook_adoption import (
    AttemptState,
    append_phase,
    apply_error_verdict,
    atomic_write_json,
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
from microcosm.build.uk_runtime.frs_spine import (
    UKFRSSpineStageTransform,
    uk_frs_spine_seed_frame,
)
from microcosm.build.uk_runtime.national_build import write_uk_national_frame
from microcosm.build.uk_runtime.national_frame import uk_household_weight_kind

_PIPELINE = "uk-frs-spine"
_RUNG = "f100"
_REPOSITORY = Path(__file__).resolve().parents[1]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the deterministic UK FRS spine from pinned raw tabs. The "
            "spine stage has no seed: determinism is structural because E2 "
            "does not run stochastic assignment or imputation."
        )
    )
    parser.add_argument(
        "--frs-raw-dir",
        type=Path,
        required=True,
        help="Directory containing the 14 licensed FRS 2023-24 tab files.",
    )
    parser.add_argument(
        "--spine-h5",
        type=Path,
        required=True,
        help="Output H5 path for the raw FRS spine Frame.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Optional directory for a copy of the completed spine checkpoint.",
    )
    parser.add_argument(
        "--emit-nonzero-shares",
        type=Path,
        help="Optional JSON path for unweighted per-produced-column nonzero shares.",
    )
    parser.add_argument(
        "--logbook-prev-row-digest",
        type=sha256_argument,
        help="Optional current Logbook chain head.",
    )
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if not args.frs_raw_dir.is_dir():
        raise ValueError(
            f"--frs-raw-dir must be an existing directory: {args.frs_raw_dir}"
        )
    if args.spine_h5.suffix != ".h5":
        raise ValueError("--spine-h5 must end with '.h5'.")
    paths = {
        "spine_h5": args.spine_h5,
        "build_sidecar": args.spine_h5.with_suffix(".build.json"),
    }
    if args.emit_nonzero_shares is not None:
        paths["emit_nonzero_shares"] = args.emit_nonzero_shares
    resolved: dict[Path, str] = {}
    for label, path in paths.items():
        target = Path(path).expanduser().resolve()
        other = resolved.get(target)
        if other is not None:
            raise ValueError(f"{label} path collides with {other}: {target}.")
        resolved[target] = label


def _artifact_pins(stage) -> dict[str, dict[str, object]]:
    pins = {}
    for artifact in stage.artifacts:
        table = str(artifact["table"])
        pins[table] = {
            "locator": str(artifact["locator"]),
            "sha256": str(artifact["sha256"]),
            "size_bytes": int(artifact["size_bytes"]),
        }
    return dict(sorted(pins.items()))


def _role_pins(pins: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    return {
        table: {
            "sha256": str(pin["sha256"]),
            "size_bytes": int(pin["size_bytes"]),
        }
        for table, pin in pins.items()
    }


def _entity_row_counts(frame) -> dict[str, int]:
    return {entity: int(len(frame.table(entity))) for entity in frame.entities}


def _build_sidecar(*, frame, stage, records, artifact_pins) -> dict[str, object]:
    household_weight = frame.weights_for("household")
    return {
        "schema_version": 1,
        "pipeline": _PIPELINE,
        "stage": "frs_spine",
        "time_period": str(frame.metadata["time_period"]),
        "household_weight_kind": uk_household_weight_kind(frame).value,
        "household_weight_total": float(household_weight.values.sum()),
        "entity_row_counts": _entity_row_counts(frame),
        "artifact_pins": artifact_pins,
        "stage_records": [
            {
                "stage": record.stage,
                "produced": list(record.produced),
                "nonzero_share": dict(record.nonzero_share),
                "seconds": record.seconds,
            }
            for record in records
        ],
        "operations": [operation.kind for operation in stage.operations],
    }


def _new_build_id(timestamp: datetime) -> str:
    return f"uk-frs-spine-{timestamp.strftime('%Y%m%dT%H%M%SZ')}"


def _record_attempt(
    *,
    state: AttemptState,
    started_at: float,
    started_ts: datetime,
    code_pin: str,
    disposition: str,
    predecessor: str | None,
    spool_dir: Path,
) -> Path:
    return record_terminal_attempt(
        state=state,
        started_at=started_at,
        started_ts=started_ts,
        pipeline=_PIPELINE,
        rung=_RUNG,
        seed=None,
        code_pin=code_pin,
        disposition=disposition,
        predecessor=predecessor,
        spool_dir=spool_dir,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    started_at = time.perf_counter()
    started_ts = datetime.now(UTC)
    predecessor = resolve_predecessor(args.logbook_prev_row_digest)
    digest = preflight_digest(_PIPELINE)
    state = AttemptState(
        build_id=_new_build_id(started_ts),
        identity_digest=digest,
        input_pins_digest=digest,
        phases_reached=["attempt_started"],
        gate_verdicts={
            "pipeline": {
                "verdict": "running",
                "receipt": "pending-build-scoped-spine-receipt",
            }
        },
    )
    code_pin = "unresolved-local-git-code-pin"
    spool_dir = args.spine_h5.parent / "logbook-spool"
    try:
        _validate_args(args)
        code_pin = git_code_pin(_REPOSITORY)
        append_phase(state, "configured")
        spec = load_country_spec("uk")
        if spec.sources is None:
            raise ValueError("UK country spec has no source stages.")
        stage = spec.sources.stage_map()["frs_spine"]
        artifact_pins = _artifact_pins(stage)
        state.input_pins_digest = role_pins_digest(_role_pins(artifact_pins))
        run_config = {
            "pipeline": _PIPELINE,
            "stage": "frs_spine",
            "artifact_pins_digest": state.input_pins_digest,
            "spine_h5": str(args.spine_h5),
        }
        state.identity_digest = hashlib.sha256(
            canonical_json_bytes(run_config)
        ).hexdigest()
        append_phase(state, "inputs_pinned")
        plan = country_stage_plan(
            spec,
            {
                "frs_spine": UKFRSSpineStageTransform(
                    args.frs_raw_dir,
                    stage=stage,
                )
            },
            stage_names=("frs_spine",),
        )
        frame, records = plan.run(uk_frs_spine_seed_frame())
        append_phase(state, "spine_built")
        output = write_uk_national_frame(frame, args.spine_h5)
        output = write_uk_national_frame(frame, output)
        append_phase(state, "spine_written")
        if args.checkpoint_dir is not None:
            args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            write_uk_national_frame(frame, args.checkpoint_dir / "frs_spine.h5")
            append_phase(state, "checkpoint_written")
        sidecar_path = output.with_suffix(".build.json")
        sidecar = _build_sidecar(
            frame=frame,
            stage=stage,
            records=records,
            artifact_pins=artifact_pins,
        )
        atomic_write_json(sidecar_path, sidecar)
        append_phase(state, "build_sidecar_written")
        if args.emit_nonzero_shares is not None:
            atomic_write_json(args.emit_nonzero_shares, dict(records[0].nonzero_share))
            append_phase(state, "nonzero_shares_written")
        state.artifact_location = local_artifact_reference(
            output,
            repository_hint=_REPOSITORY,
        )
        state.gate_verdicts = {
            "pipeline": {
                "verdict": "passed",
                "receipt": local_artifact_reference(
                    sidecar_path, repository_hint=_REPOSITORY
                ),
            }
        }
        spool_path = _record_attempt(
            state=state,
            started_at=started_at,
            started_ts=started_ts,
            code_pin=code_pin,
            disposition="iterating",
            predecessor=predecessor,
            spool_dir=spool_dir,
        )
        print(f"Wrote FRS spine H5: {output}", file=sys.stderr)
        print(f"Wrote Logbook row: {spool_path}", file=sys.stderr)
        return 0
    except Exception as error:
        try:
            receipt_path = write_error_receipt(
                error_receipt_path(args.spine_h5.parent, build_id=state.build_id),
                state=state,
                pipeline=_PIPELINE,
                error=error,
            )
            apply_error_verdict(
                state,
                local_artifact_reference(receipt_path, repository_hint=_REPOSITORY),
            )
            _record_attempt(
                state=state,
                started_at=started_at,
                started_ts=started_ts,
                code_pin=code_pin,
                disposition="failed",
                predecessor=predecessor,
                spool_dir=spool_dir,
            )
        except Exception:
            pass
        print(f"UK FRS spine build failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
