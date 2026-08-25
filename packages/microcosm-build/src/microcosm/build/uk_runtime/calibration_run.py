"""Production UK national calibration seam orchestration."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import os
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from microcosm.build.country_spec import GatesManifest, load_country_spec
from microcosm.build.gate_battery import (
    BlockingMode,
    EvidenceContext,
    GateBatteryRun,
    gate_signing_key_env,
)
from microcosm.build.logbook import canonical_json_bytes
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
from microcosm.build.target_materialization import assert_calibration_input_finite
from microcosm.build.uk_runtime.battery_bindings import UK_GATE_REGISTRY
from microcosm.build.uk_runtime.diagnostics import (
    uk_target_geography_levels,
    write_uk_calibration_diagnostics,
)
from microcosm.build.uk_runtime.national_calibration import UKNationalCalibrationStage
from microcosm.build.uk_runtime.national_frame import (
    load_uk_national_frame,
    write_uk_national_frame,
)
from microcosm.calibrate import TargetRegistry
from microcosm.frame import Frame

_REPOSITORY = Path(__file__).resolve().parents[6]
# The FRS line's spine, staging, imputation and calibration stages share one
# hash chain (logbook/README.md): the dataset token names the base data, not
# the build mechanism, so calibration derives the ratified `uk/frs` scope.
_PIPELINE = "uk-frs-calibration"


@dataclass(frozen=True)
class UKCalibrationRunPaths:
    input_h5: Path
    staging_h5: Path
    diagnostics_json: Path
    build_record_json: Path
    terminal_gate_json: Path


@dataclass(frozen=True)
class UKCalibrationRunResult:
    frame: Frame
    diagnostics_sha256: str
    staging_sha256: str
    build_record_sha256: str
    terminal_gate_sha256: str
    logbook_spool: Path
    gate_report: Mapping[str, object]
    build_record: Mapping[str, object]


UK_CALIBRATION_GATE_SCOPE = (
    "uk_target_fit",
    "uk_weight_ratio",
    "uk_weight_ess",
    "uk_zero_weight_strata",
    "uk_aggregate_admin",
    "uk_calibration_reference_coverage",
)


def _scope_exclusions() -> dict[str, str]:
    full = {entry.id for entry in load_country_spec("uk").gates.gates}
    excluded = full - set(UK_CALIBRATION_GATE_SCOPE)
    rationales: dict[str, str] = {}
    for gate_id in sorted(excluded):
        if "parity" in gate_id or gate_id in {"uk_export_surface", "uk_target_surface"}:
            reason = "swap-acceptance evidence; produced by the swap lane, not the calibration seam."
        elif gate_id in {
            "uk_release_input_coverage_manifest_current",
            "uk_release_family_build_stages",
            "uk_release_input_coverage",
            "uk_nonnegative_columns",
            "uk_support",
            "uk_take_up_signal",
            "uk_brma_enum_domain",
            "uk_degenerate_release_surface",
            "uk_input_mass_parity",
            "uk_qrf_tail_concentration",
            "uk_weights_audit",
        }:
            reason = "spine-construction gate; owned by the spine build's own battery."
        else:
            reason = "outside the calibration seam's reviewed gate scope."
        rationales[gate_id] = reason
    if set(UK_CALIBRATION_GATE_SCOPE) | set(rationales) != full:
        raise RuntimeError("UK calibration gate scope does not classify every gate id.")
    return rationales


UK_CALIBRATION_GATE_SCOPE_EXCLUSIONS = _scope_exclusions()


def run_uk_calibration(
    *,
    paths: UKCalibrationRunPaths,
    input_sha256: str,
    ledger_artifact: Any,
    register_registry: TargetRegistry,
    calibration_year: int,
    exclusion_receipt: Mapping[str, Mapping[str, str]],
    doctrine: Any,
    doctrine_overrides: Mapping[str, Mapping[str, object]],
    measure_resolver: object | None,
    source_pins: Mapping[str, Mapping[str, object]],
    run_config_extra: Mapping[str, object],
    release_candidate: bool,
    release_id: str,
    logbook_prev_row_digest: str | None = None,
) -> UKCalibrationRunResult:
    """Run the UK national calibration seam and write its sidecars."""

    started_at = time.perf_counter()
    started_ts = datetime.now(UTC)
    code_pin = git_code_pin(_REPOSITORY)
    # Predecessor configuration is validated before anything is written: a
    # disagreeing chain must refuse with no artifact on disk, not after a
    # staged H5, diagnostics and a signed gate report already exist.
    predecessor = resolve_predecessor(logbook_prev_row_digest)
    run_config = {
        "pipeline": _PIPELINE,
        "release_id": release_id,
        "register_sha256": register_registry.version,
        "calibration_year": int(calibration_year),
        "doctrine": _doctrine_payload(doctrine),
        "doctrine_overrides": dict(doctrine_overrides),
        # The caller verifies the feed's facts and manifest digests; sealing
        # the verified identity into run_config carries it through the
        # identity digest, the build record and the Logbook row, so the run
        # says which Ledger artifact it was measured against.
        "ledger": _ledger_provenance(ledger_artifact),
        **dict(run_config_extra),
    }
    state = AttemptState(
        # Attempts are distinct rows even when they re-run one release: both
        # the local chain and the store refuse a repeated build id.
        build_id=_new_calibration_attempt_id(timestamp=started_ts),
        identity_digest=hashlib.sha256(canonical_json_bytes(run_config)).hexdigest(),
        input_pins_digest=role_pins_digest(source_pins),
        phases_reached=["attempt_started"],
        gate_verdicts={},
    )
    spool_dir = paths.staging_h5.parent / "logbook-spool"
    try:
        return _run_uk_calibration_attempt(
            paths=paths,
            input_sha256=input_sha256,
            ledger_artifact=ledger_artifact,
            register_registry=register_registry,
            calibration_year=calibration_year,
            exclusion_receipt=exclusion_receipt,
            doctrine=doctrine,
            doctrine_overrides=doctrine_overrides,
            measure_resolver=measure_resolver,
            source_pins=source_pins,
            release_candidate=release_candidate,
            release_id=release_id,
            state=state,
            run_config=run_config,
            code_pin=code_pin,
            started_at=started_at,
            started_ts=started_ts,
            predecessor=predecessor,
            spool_dir=spool_dir,
        )
    except BaseException as error:
        # Every terminal disposition records a row — successful, failed, or
        # refused (logbook/README.md). A refusal that left no row would be a
        # silent gap in the chain the run is supposed to evidence.
        _record_failed_attempt(
            error=error,
            state=state,
            started_at=started_at,
            started_ts=started_ts,
            seed=getattr(doctrine, "seed", None),
            code_pin=code_pin,
            predecessor=predecessor,
            receipt_base_dir=paths.staging_h5.parent,
            spool_dir=spool_dir,
        )
        raise


def _new_calibration_attempt_id(*, timestamp: datetime) -> str:
    instant = timestamp.astimezone(UTC)
    return (
        "uk-frs-calibration-attempt-"
        f"{instant.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    )


def _record_failed_attempt(
    *,
    error: BaseException,
    state: AttemptState,
    started_at: float,
    started_ts: datetime,
    seed: int | None,
    code_pin: str,
    predecessor: str | None,
    receipt_base_dir: Path,
    spool_dir: Path,
) -> None:
    if state.spool_path is not None:
        return
    error_path = write_error_receipt(
        error_receipt_path(receipt_base_dir, build_id=state.build_id),
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
        seed=seed,
        code_pin=code_pin,
        disposition="failed",
        predecessor=predecessor,
        spool_dir=spool_dir,
    )


def _run_uk_calibration_attempt(
    *,
    paths: UKCalibrationRunPaths,
    input_sha256: str,
    ledger_artifact: Any,
    register_registry: TargetRegistry,
    calibration_year: int,
    exclusion_receipt: Mapping[str, Mapping[str, str]],
    doctrine: Any,
    doctrine_overrides: Mapping[str, Mapping[str, object]],
    measure_resolver: object | None,
    source_pins: Mapping[str, Mapping[str, object]],
    release_candidate: bool,
    release_id: str,
    state: AttemptState,
    run_config: Mapping[str, object],
    code_pin: str,
    started_at: float,
    started_ts: datetime,
    predecessor: str | None,
    spool_dir: Path,
) -> UKCalibrationRunResult:
    measured_input_sha = _sha256_file(paths.input_h5)
    if measured_input_sha != input_sha256:
        raise ValueError(
            "input H5 sha mismatch: "
            f"measured {measured_input_sha}, pinned {input_sha256}"
        )
    append_phase(state, "input_sha_verified")
    frame, _provenance = load_uk_national_frame(paths.input_h5)
    append_phase(state, "input_loaded")
    assert_calibration_input_finite(frame)
    append_phase(state, "input_finite")

    stage = UKNationalCalibrationStage(
        register_registry,
        # The declared calibration year the register was compiled at — the
        # stage validates it; there is deliberately no fallback to the input
        # frame's base-year time_period or any ambient default.
        period=calibration_year,
        doctrine=doctrine,
        measure_resolver=measure_resolver,
    )
    calibrated = stage(frame)
    append_phase(state, "national_calibration_solved")

    build_block = {
        "build_id": state.build_id,
        "code_pin": code_pin,
        "source_pins": dict(source_pins),
        "ledger": run_config["ledger"],
        "input_posture": {
            "tier": "staging_candidate",
            "sha256": measured_input_sha,
            "size_bytes": paths.input_h5.stat().st_size,
        },
        "doctrine": _doctrine_payload(doctrine),
        "doctrine_overrides": dict(doctrine_overrides),
        "measure_exclusions": dict(exclusion_receipt),
        "measure_resolution": (
            stage.manifest.get("measure_resolution")
            if isinstance(stage.manifest, Mapping)
            else None
        ),
        "register": _register_census(register_registry, exclusion_receipt),
        "score_vs_enhanced_frs": None,
    }
    write_uk_calibration_diagnostics(
        stage.solve_result,
        paths.diagnostics_json,
        calibrated,
        target_geography_levels=uk_target_geography_levels(stage.registry),
        target_registry=stage.registry,
        build=build_block,
    )
    diagnostics_sha = _sha256_file(paths.diagnostics_json)
    append_phase(state, "diagnostics_written")

    gate_report = _run_calibration_gate_battery(
        calibrated,
        stage,
        paths.terminal_gate_json,
        release_candidate=release_candidate,
        release_id=release_id,
        diagnostics_sha256=diagnostics_sha,
    )
    append_phase(state, "calibration_gates_evaluated")
    for gate_id, payload in gate_report["gates"].items():
        state.gate_verdicts[gate_id] = {
            "verdict": payload["status"],
            "receipt": f"local://{paths.terminal_gate_json.name}#/gates/{gate_id}",
        }

    write_uk_national_frame(calibrated, paths.staging_h5)
    staging_sha = _sha256_file(paths.staging_h5)
    append_phase(state, "staging_h5_written")

    record = {
        "schema_version": 1,
        "pipeline": _PIPELINE,
        "build_id": state.build_id,
        "run_config": run_config,
        "source_pins": dict(source_pins),
        "role_pins_digest": role_pins_digest(source_pins),
        "input_posture": build_block["input_posture"],
        "register": build_block["register"],
        "calibration": stage.manifest,
        "gate_summary": _gate_summary(gate_report),
        "shippable": False,
        "shippable_reason": (
            "calibration-scoped battery; release certification is the "
            "release-cut producer's job"
        ),
        "artifacts": {
            "staging_h5": {"path": str(paths.staging_h5), "sha256": staging_sha},
            "diagnostics_json": {
                "path": str(paths.diagnostics_json),
                "sha256": diagnostics_sha,
            },
            "terminal_gate_json": {
                "path": str(paths.terminal_gate_json),
                "sha256": _sha256_file(paths.terminal_gate_json),
            },
        },
    }
    _write_json(paths.build_record_json, record)
    build_record_sha = _sha256_file(paths.build_record_json)
    append_phase(state, "build_record_written")
    state.artifact_location = local_artifact_reference(
        paths.staging_h5, repository_hint=_REPOSITORY
    )
    spool = record_terminal_attempt(
        state=state,
        started_at=started_at,
        started_ts=started_ts,
        pipeline=_PIPELINE,
        rung="f100",
        seed=getattr(doctrine, "seed", None),
        code_pin=code_pin,
        disposition="iterating",
        predecessor=predecessor,
        spool_dir=spool_dir,
    )
    return UKCalibrationRunResult(
        frame=calibrated,
        diagnostics_sha256=diagnostics_sha,
        staging_sha256=staging_sha,
        build_record_sha256=build_record_sha,
        terminal_gate_sha256=_sha256_file(paths.terminal_gate_json),
        logbook_spool=spool,
        gate_report=gate_report,
        build_record=record,
    )


def _run_calibration_gate_battery(
    frame: Frame,
    stage: UKNationalCalibrationStage,
    path: Path,
    *,
    release_candidate: bool,
    release_id: str,
    diagnostics_sha256: str,
) -> dict[str, object]:
    manifest = _calibration_gate_manifest()
    admin_totals, admin_receipt = _aggregate_admin_totals(frame, manifest)
    artifacts = {
        "national_calibration": stage.manifest,
        "parity_evidence": SimpleNamespace(
            target_relative_errors={
                str(row["name"]): float(row["relative_error"])
                for row in stage.diagnostics
            }
        ),
        "aggregate_admin": admin_totals,
    }
    battery = GateBatteryRun(
        manifest,
        release_id=release_id,
        report_path=path,
        release_candidate=release_candidate,
        registry=UK_GATE_REGISTRY,
        release_evidence={"calibration_diagnostics_sha256": diagnostics_sha256},
    )
    battery.run_phase("terminal", EvidenceContext(frame=frame, artifacts=artifacts))
    battery.enforce("terminal", mode=BlockingMode.BLOCKS_ARTIFACT)
    payload = battery.report_payload()
    payload["posture"] = "calibration_seam"
    payload["scope_exclusions"] = dict(UK_CALIBRATION_GATE_SCOPE_EXCLUSIONS)
    payload["aggregate_admin_measurement"] = admin_receipt
    _resign_gate_report(payload)
    _write_json(path, payload)
    return payload


def _calibration_gate_manifest() -> GatesManifest:
    source = load_country_spec("uk").gates
    entries = tuple(
        entry for entry in source.gates if entry.id in UK_CALIBRATION_GATE_SCOPE
    )
    return GatesManifest(
        country=source.country,
        version=source.version,
        policy=f"{source.policy}; calibration_seam_scope",
        phases=("terminal",),
        gates=entries,
    )


def _aggregate_admin_totals(
    frame: Frame, manifest: GatesManifest
) -> tuple[dict[str, float], list[dict[str, object]]]:
    """Measure every declared admin anchor, fail-loud on absent evidence.

    The anchor value's magnitude tells its statistic — NEED per-household
    means are hundreds of pounds, the NHS anchor is a national total — the
    same reviewed convention the first armed-run receipts used. Non-household
    entities carry their household's weight through the person linkage.
    """

    anchors = []
    for entry in manifest.gates:
        if entry.id == "uk_aggregate_admin":
            anchors = list(entry.parameters.get("anchors", ()))
            break
    household_weights = np.asarray(
        frame.weights_for("household").values, dtype=float
    )
    totals: dict[str, float] = {}
    receipt: list[dict[str, object]] = []
    for anchor in anchors:
        entity = str(anchor.get("entity", "household"))
        name = str(anchor.get("name", anchor.get("measure")))
        measure = str(anchor.get("measure", anchor.get("name")))
        table = frame.table(entity)
        if measure not in table:
            raise ValueError(
                f"aggregate_admin anchor {name!r} needs {entity}.{measure}, "
                "which the calibrated frame does not carry; refusing to "
                "fabricate a measured value."
            )
        if entity == "household":
            weights = household_weights
        elif entity == "person":
            person = frame.table("person")
            lookup = dict(
                zip(
                    frame.table("household")["household_id"].to_numpy(),
                    household_weights,
                    strict=True,
                )
            )
            weights = np.asarray(
                [lookup[key] for key in person["person_household_id"].to_numpy()],
                dtype=float,
            )
        else:
            raise ValueError(
                f"aggregate_admin anchor {name!r} declares entity {entity!r}; "
                "the calibration seam measures household and person anchors "
                "only."
            )
        values = table[measure].to_numpy(dtype=float)
        total = float(np.dot(values, weights))
        carriers = values != 0
        carrier_weight = float(weights[carriers].sum())
        mean_carriers = (
            float(np.dot(values[carriers], weights[carriers]) / carrier_weight)
            if carrier_weight
            else float("nan")
        )
        declared = float(anchor["value"])
        measured = mean_carriers if abs(declared) < 1e6 else total
        totals[name] = measured
        receipt.append(
            {
                "anchor": name,
                "entity": entity,
                "measure": measure,
                "measured": measured,
                "weighted_total": total,
                "weighted_mean_carriers": mean_carriers,
                "statistic_convention": "assessed_by_anchor_magnitude",
            }
        )
    return totals, receipt


def _ledger_provenance(artifact: Any) -> dict[str, object]:
    """The verified identity of the Ledger consumer feed this run compiled.

    A bare ``consumer_facts.jsonl`` feed carries no manifest, so its
    Ledger-side provenance is recorded as absent rather than invented.
    """

    provenance: dict[str, object] = {
        "facts_sha256": getattr(artifact, "facts_sha256", None),
        "fact_row_count": getattr(artifact, "fact_row_count", None),
        "manifest_sha256": getattr(artifact, "manifest_sha256", None),
    }
    manifest = getattr(artifact, "manifest", None)
    if isinstance(manifest, Mapping):
        provenance["manifest"] = {
            key: manifest.get(key)
            for key in ("artifact_id", "profile", "schema_version", "generated_at")
            if manifest.get(key) is not None
        }
    return provenance


def _register_census(
    registry: TargetRegistry, exclusions: Mapping[str, Mapping[str, str]]
) -> dict[str, object]:
    return {
        "country": registry.country,
        "version": registry.version,
        "compiled_count": len(registry.specs) + len(exclusions),
        "excluded_count": len(exclusions),
        "calibrated_count": len(registry.specs),
    }


def _doctrine_payload(doctrine: Any) -> dict[str, object]:
    return {
        key: getattr(doctrine, key)
        for key in (
            "epochs",
            "learning_rate",
            "max_weight_ratio",
            "seed",
            "target_loss_cap",
            "scale_rule",
            "target_weight_rule",
            "mass_rule",
            "l0_lambda",
        )
        if hasattr(doctrine, key)
    }


def _gate_summary(report: Mapping[str, object]) -> dict[str, object]:
    gates = report.get("gates", {})
    if not isinstance(gates, Mapping):
        return {}
    return {
        gate_id: payload.get("status")
        for gate_id, payload in gates.items()
        if isinstance(payload, Mapping)
    }


def _resign_gate_report(payload: dict[str, object]) -> None:
    attestation = payload.get("attestation")
    if not isinstance(attestation, dict):
        raise RuntimeError("gate report has no attestation block.")
    env = gate_signing_key_env("uk")
    encoded = os.environ.get(env)
    if not encoded:
        raise RuntimeError(
            f"{env} must be set; unsigned full-scale calibration runs refuse to stage."
        )
    try:
        key = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise RuntimeError(f"{env} must be valid base64.") from exc
    if len(key) != 32:
        raise RuntimeError(f"{env} must decode to exactly 32 bytes.")
    attestation.pop("signing_error", None)
    attestation["signing_key_sha256"] = hashlib.sha256(key).hexdigest()
    attestation["signature"] = None
    attestation["signature"] = hmac.new(
        key, canonical_json_bytes(payload), hashlib.sha256
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
