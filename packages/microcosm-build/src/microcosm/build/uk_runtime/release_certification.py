"""UK release-cut certification: the national battery's runner and composer.

The June national driver retired with microcosm#757 and took the only
executor of the 16 declared national preflight/terminal gates with it.
This module is their executable home (issue #757 item B5): a scoped
``GateBatteryRun`` over ``UK_NATIONAL_GATE_SCOPE`` evaluated against the
calibrated candidate, plus the **multi-part release certification** the
2026-08-25 audit (issue comment 5413502559) specified — the spine build's
battery report, the calibration seam's battery report, and the release-cut
battery report must union to the full declared gate-entry set with no gap
and no overlap beyond ``UK_SHARED_GATE_IDS``, each part signed by its
producer, with the phase and digest checks moving from per-report to
per-certification. A candidate's shippability verdict comes only from the
certification, never from a single scoped report.

Evidence adaptation only, never verdict re-implementation: every gate in
the release-cut battery runs the same ``UK_GATE_REGISTRY`` binding the June
runner used; this module reconstructs the evidence the retired runner drew
from live stage objects out of the artifacts the split pipeline persists
(the spine build sidecar, the seam's diagnostics and build record, the
per-run licensed input-mass reference).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
from collections.abc import Mapping, Sequence
from datetime import date
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from microcosm.build.country_spec import GatesManifest, load_country_spec
from microcosm.build.gate_battery import (
    BlockingMode,
    EvidenceContext,
    GateBatteryRun,
    gate_signing_key_env,
)
from microcosm.build.gates import FitWeightRecord
from microcosm.build.logbook import canonical_json_bytes
from microcosm.build.uk_runtime.battery_bindings import UK_GATE_REGISTRY
from microcosm.build.uk_runtime.calibration_run import (
    UK_CALIBRATION_GATE_SCOPE,
    UK_NATIONAL_GATE_SCOPE,
    UK_SHARED_GATE_IDS,
    UK_SPINE_GATE_SCOPE,
    finalize_uk_scoped_gate_report,
    uk_aggregate_admin_totals,
    uk_scoped_gate_manifest,
)

__all__ = [
    "UK_RELEASE_CERTIFICATION_KIND",
    "UK_RELEASE_CERTIFICATION_SCHEMA_VERSION",
    "UK_RELEASE_CUT_POSTURE",
    "UKReleaseCertificationError",
    "compose_uk_release_certification",
    "rehydrate_uk_fit_weight_records",
    "run_uk_release_cut_battery",
    "uk_national_gate_manifest",
    "uk_release_cut_scope_exclusions",
    "uk_release_parity_evidence",
]

UK_RELEASE_CERTIFICATION_SCHEMA_VERSION = 1
UK_RELEASE_CERTIFICATION_KIND = "uk_release_certification"
UK_RELEASE_CUT_POSTURE = "release_cut"

#: Each certification part's declared scope, phases, and manifest policy
#: suffix. The suffixes are load-bearing: they are what the producers bake
#: into their scoped manifests, so the re-derived digests only match a part
#: that really ran the committed spec under its declared scope.
_PART_SCOPES: Mapping[str, Mapping[str, object]] = {
    "spine": {
        "scope": UK_SPINE_GATE_SCOPE,
        "phases": ("assembled", "transferred"),
        "policy_suffix": "spine_build_scope",
        "posture": None,
    },
    "calibration_seam": {
        "scope": UK_CALIBRATION_GATE_SCOPE,
        "phases": ("terminal",),
        "policy_suffix": "calibration_seam_scope",
        "posture": "calibration_seam",
    },
    "release_cut": {
        "scope": UK_NATIONAL_GATE_SCOPE,
        "phases": ("preflight", "terminal"),
        "policy_suffix": "release_cut_scope",
        "posture": UK_RELEASE_CUT_POSTURE,
    },
}


class UKReleaseCertificationError(ValueError):
    """A certification refusal: the parts do not certify the candidate."""


@lru_cache(maxsize=1)
def _uk_gates_spec() -> GatesManifest:
    # The committed spec is immutable within a process; the composer derives
    # digests for three scopes per certification, so one validated load
    # serves them all.
    return load_country_spec("uk").gates


@lru_cache(maxsize=1)
def uk_national_gate_manifest() -> GatesManifest:
    """The release-cut battery's scoped manifest (preflight + terminal)."""

    return uk_scoped_gate_manifest(
        UK_NATIONAL_GATE_SCOPE,
        phases=("preflight", "terminal"),
        policy_suffix="release_cut_scope",
    )


def uk_release_cut_scope_exclusions() -> dict[str, str]:
    """Why each declared gate outside the national scope is not run here."""

    spec = _uk_gates_spec()
    exclusions: dict[str, str] = {}
    for entry in spec.gates:
        if entry.id in UK_NATIONAL_GATE_SCOPE:
            continue
        if entry.id in UK_SPINE_GATE_SCOPE:
            exclusions[entry.id] = (
                "spine-construction gate; owned by the spine build's scoped battery."
            )
        elif entry.id in UK_CALIBRATION_GATE_SCOPE:
            exclusions[entry.id] = (
                "calibration-seam gate; owned by the seam's scoped battery."
            )
        else:  # pragma: no cover - the three-way partition is import-enforced
            raise RuntimeError(
                f"UK gate {entry.id!r} belongs to no declared battery scope."
            )
    return exclusions


def rehydrate_uk_fit_weight_records(
    sidecar: Mapping[str, Any],
) -> tuple[FitWeightRecord, ...] | None:
    """The weights-audit evidence, rehydrated from the spine build sidecar.

    ``None`` (sidecar carries no ``fit_weight_records`` block — a pre-#757
    spine) leaves the artifact unsupplied so the audit records a named
    ``evidence_absent`` gap, which blocks at release-candidate strictness. A
    present block with any empty per-stage list coerces to ``()``: a fitting
    stage that emitted nothing is a failed audit, never a vacuous pass.
    """

    block = sidecar.get("fit_weight_records")
    if block is None:
        return None
    if not isinstance(block, Mapping):
        raise UKReleaseCertificationError(
            "spine sidecar fit_weight_records must map stage names to record lists."
        )
    collected: list[FitWeightRecord] = []
    empty_stages: list[str] = []
    for stage_name, records in block.items():
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            raise UKReleaseCertificationError(
                f"spine sidecar fit_weight_records[{stage_name!r}] must be a "
                "list of records; an unreadable block is corruption, not an "
                "empty audit."
            )
        if not records:
            empty_stages.append(str(stage_name))
            continue
        for record in records:
            if not isinstance(record, Mapping) or not (
                {"fit_name", "weight_kind"} <= set(record)
            ):
                raise UKReleaseCertificationError(
                    f"spine sidecar fit_weight_records[{stage_name!r}] carries "
                    "a malformed record; an unreadable record is corruption, "
                    "not an empty audit."
                )
            collected.append(
                FitWeightRecord(
                    fit_name=str(record["fit_name"]),
                    weight_kind=str(record["weight_kind"]),
                )
            )
    if empty_stages:
        # A fitting stage that recorded nothing is a failed audit: hand the
        # binding the empty tuple its refusal path exists for.
        return ()
    return tuple(collected)


def uk_release_parity_evidence(
    frame: Any,
    *,
    diagnostics_targets: Sequence[Mapping[str, Any]],
    reference_registry: Any,
    parity_reference: Any,
) -> SimpleNamespace:
    """The parity-trio evidence over persisted inputs, sides never aliased.

    Candidate columns come from the staged frame; reference columns from the
    frozen parity instrument's declared input entities. Candidate targets
    come from the solve's realized diagnostics rows; reference targets from
    the independently recompiled (and exclusion-pruned) register at
    ``name@period`` grain — the same two-source rule the retired June
    runner's ``_stage_parity_evidence`` enforced.
    """

    target_relative_errors: dict[str, float] = {}
    for row in diagnostics_targets:
        name = str(row["name"])
        if "@" not in name:
            raise UKReleaseCertificationError(
                f"diagnostics target {name!r} is not labeled at name@period "
                "grain; the reference side is keyed name@period, so a "
                "bare-name row would silently fall out of the comparison."
            )
        target_relative_errors[name] = float(row["relative_error"])
    return SimpleNamespace(
        candidate_columns={
            f"{entity}.{column}"
            for entity in frame.entities
            for column in frame.table(entity).columns
        },
        reference_columns={
            f"{entity}.{name}"
            for name, entity in parity_reference.input_entities.items()
        },
        candidate_targets=set(target_relative_errors),
        reference_targets={
            f"{spec.name}@{spec.period}" for spec in reference_registry.specs
        },
        target_relative_errors=target_relative_errors,
    )


def run_uk_release_cut_battery(
    frame: Any,
    *,
    report_path: Path,
    release_id: str,
    diagnostics_sha256: str,
    coverage_engine: Any,
    build_stage_names: Sequence[str],
    ledger_registries: Mapping[object, Any],
    local_ledger_registries: Mapping[object, Any],
    parity_evidence: Any,
    fit_weight_records: tuple[FitWeightRecord, ...] | None,
    input_mass_reference: Mapping[str, Any],
    exclusions_evaluated_on: date,
    gate_registry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the 18 national gates over the calibrated candidate, signed.

    Always release-candidate strict: this battery exists to certify a cut,
    so an ``evidence_absent`` gap blocks rather than being tolerated, and a
    blocked phase persists its report and raises before any composition.
    The local-surface compile gates run here too: the certification is the
    declared owner of the whole national scope, so the caller supplies both
    the national and the local compiled registries.
    """

    battery = GateBatteryRun(
        uk_national_gate_manifest(),
        release_id=release_id,
        report_path=report_path,
        release_candidate=True,
        registry=UK_GATE_REGISTRY if gate_registry is None else gate_registry,
        release_evidence={"calibration_diagnostics_sha256": diagnostics_sha256},
    )
    preflight_artifacts: dict[str, Any] = {
        "coverage_engine": coverage_engine,
        "build_stage_names": tuple(str(name) for name in build_stage_names),
        "uk_ledger_compiled_registries": dict(ledger_registries),
        "uk_ledger_compiled_local_registries": dict(local_ledger_registries),
    }
    battery.run_phase("preflight", EvidenceContext(artifacts=preflight_artifacts))
    battery.enforce("preflight", mode=BlockingMode.BLOCKS_ARTIFACT)

    admin_totals, admin_receipt = uk_aggregate_admin_totals(
        frame, uk_national_gate_manifest()
    )
    terminal_artifacts: dict[str, Any] = {
        "coverage_engine": coverage_engine,
        "rules_engine": coverage_engine,
        "build_stage_names": tuple(str(name) for name in build_stage_names),
        "exclusions_evaluated_on": exclusions_evaluated_on,
        "parity_evidence": parity_evidence,
        "aggregate_admin": admin_totals,
        "input_mass_reference": input_mass_reference,
    }
    if fit_weight_records is not None:
        terminal_artifacts["fit_weight_records"] = fit_weight_records
    battery.run_phase(
        "terminal", EvidenceContext(frame=frame, artifacts=terminal_artifacts)
    )
    battery.enforce("terminal", mode=BlockingMode.BLOCKS_ARTIFACT)
    payload = battery.report_payload()
    finalize_uk_scoped_gate_report(
        payload,
        posture=UK_RELEASE_CUT_POSTURE,
        scope_exclusions=uk_release_cut_scope_exclusions(),
        aggregate_admin_measurement=admin_receipt,
    )
    _write_json(report_path, payload)
    return payload


# ---------------------------------------------------------------------------
# The multi-part certification composer
# ---------------------------------------------------------------------------


def compose_uk_release_certification(
    *,
    release_id: str,
    candidate_name: str,
    candidate_path: Path,
    candidate_sha256: str,
    spine_report_path: Path,
    seam_report_path: Path,
    release_cut_report_path: Path,
    spine_sidecar: Mapping[str, Any],
    build_record: Mapping[str, Any],
    score_receipt_path: Path,
    exclusions_evaluated_on: date,
    certification_path: Path,
) -> dict[str, Any]:
    """Verify the three scoped parts and compose the signed certification.

    Every check is a refusal: a certification only exists when the parts
    union to the full declared entry set with no gap, no overlap beyond
    ``UK_SHARED_GATE_IDS``, full phase coverage, verified signatures, the
    committed spec's scoped digests, green release-blocking verdicts, and a
    closed identity join from the spine report through the sidecar, the
    build record, the diagnostics digest, and the candidate bytes.
    """

    parts_raw = {
        "spine": _load_part(spine_report_path),
        "calibration_seam": _load_part(seam_report_path),
        "release_cut": _load_part(release_cut_report_path),
    }
    signing_key = _require_signing_key()
    declared = _uk_gates_spec()
    declared_ids = {entry.id for entry in declared.gates}
    declared_phases = tuple(declared.phases)

    part_summaries: dict[str, dict[str, Any]] = {}
    for part_name, (payload, raw_bytes) in parts_raw.items():
        spec = _PART_SCOPES[part_name]
        _verify_part(
            part_name,
            payload,
            scope=frozenset(spec["scope"]),
            phases=tuple(spec["phases"]),
            policy_suffix=str(spec["policy_suffix"]),
            posture=spec["posture"],
            signing_key=signing_key,
        )
        part_summaries[part_name] = {
            "path": str(
                _part_path(
                    part_name,
                    spine_report_path,
                    seam_report_path,
                    release_cut_report_path,
                )
            ),
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "release_id": str(payload["release_id"]),
            "phases": list(payload["phases"]),
            "entry_ids": sorted(payload["gates"]),
            "gates_manifest_sha256": str(payload["gates_manifest_sha256"]),
            "policy_sha256": str(payload["policy_sha256"]),
            "statuses": _status_census(payload),
        }

    _verify_union(
        {name: (payload, raw) for name, (payload, raw) in parts_raw.items()},
        declared_ids=declared_ids,
        declared_phases=declared_phases,
    )
    _verify_identity_join(
        spine_report_bytes=parts_raw["spine"][1],
        seam_report_bytes=parts_raw["calibration_seam"][1],
        seam_payload=parts_raw["calibration_seam"][0],
        release_cut_payload=parts_raw["release_cut"][0],
        spine_sidecar=spine_sidecar,
        build_record=build_record,
        candidate_path=candidate_path,
        candidate_sha256=candidate_sha256,
        release_id=release_id,
    )
    score_receipt_bytes = score_receipt_path.read_bytes()
    score_receipt = json.loads(score_receipt_bytes)
    _verify_score_receipt(score_receipt, candidate_sha256=candidate_sha256)

    full_digests = _full_manifest_digests()
    run_config = build_record.get("run_config", {})
    certification: dict[str, Any] = {
        "schema_version": UK_RELEASE_CERTIFICATION_SCHEMA_VERSION,
        "kind": UK_RELEASE_CERTIFICATION_KIND,
        "country": "uk",
        "release_id": release_id,
        "candidate": {
            "name": candidate_name,
            "filename": candidate_path.name,
            "sha256": candidate_sha256,
            "size_bytes": candidate_path.stat().st_size,
        },
        "parts": part_summaries,
        "spec": {
            "gates_manifest_sha256": full_digests["gates_manifest_sha256"],
            "policy_sha256": full_digests["policy_sha256"],
            "spec_fingerprint": full_digests["spec_fingerprint"],
            "declared_entry_count": len(declared_ids),
            "declared_phases": list(declared_phases),
            "shared_gate_ids": sorted(UK_SHARED_GATE_IDS),
        },
        "doctrine": {
            "payload": dict(run_config.get("doctrine", {})),
            "overrides": dict(run_config.get("doctrine_overrides", {})),
        },
        "diagnostics_sha256": str(
            parts_raw["calibration_seam"][0]["release_evidence"][
                "calibration_diagnostics_sha256"
            ]
        ),
        "score_receipt": {
            "filename": score_receipt_path.name,
            "sha256": hashlib.sha256(score_receipt_bytes).hexdigest(),
        },
        "exclusions_evaluated_on": exclusions_evaluated_on.isoformat(),
        "shippable": True,
    }
    _sign_certification(certification, signing_key)
    _write_json(certification_path, certification)
    return certification


# ---------------------------------------------------------------------------
# Refusal helpers
# ---------------------------------------------------------------------------


def _part_path(
    part_name: str,
    spine_report_path: Path,
    seam_report_path: Path,
    release_cut_report_path: Path,
) -> Path:
    return {
        "spine": spine_report_path,
        "calibration_seam": seam_report_path,
        "release_cut": release_cut_report_path,
    }[part_name]


def _load_part(path: Path) -> tuple[dict[str, Any], bytes]:
    if not path.is_file():
        raise UKReleaseCertificationError(f"certification part absent: {path}")
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, Mapping):
        raise UKReleaseCertificationError(
            f"certification part {path} is not a JSON object."
        )
    return dict(payload), raw


def _require_signing_key() -> bytes:
    env_name = gate_signing_key_env("uk")
    encoded = os.environ.get(env_name)
    if not encoded:
        raise UKReleaseCertificationError(
            f"{env_name} must be set: a certification and every part it "
            "verifies are signed artifacts."
        )
    key = base64.b64decode(encoded)
    if len(key) != 32:
        raise UKReleaseCertificationError(
            f"{env_name} must decode to exactly 32 bytes."
        )
    return key


def _verify_part(
    part_name: str,
    payload: Mapping[str, Any],
    *,
    scope: frozenset[str],
    phases: tuple[str, ...],
    policy_suffix: str,
    posture: str | None,
    signing_key: bytes,
) -> None:
    if payload.get("schema_version") != 4:
        raise UKReleaseCertificationError(
            f"{part_name}: schema_version must be 4, got "
            f"{payload.get('schema_version')!r}."
        )
    if payload.get("country") != "uk":
        raise UKReleaseCertificationError(f"{part_name}: country must be 'uk'.")
    if payload.get("blocked_at_phase") is not None:
        raise UKReleaseCertificationError(
            f"{part_name}: blocked at phase {payload['blocked_at_phase']!r}; "
            "a blocked part cannot certify."
        )
    if list(payload.get("phases", ())) != list(phases):
        raise UKReleaseCertificationError(
            f"{part_name}: phases must be {list(phases)}, got "
            f"{payload.get('phases')!r}."
        )
    if posture is not None and payload.get("posture") != posture:
        raise UKReleaseCertificationError(
            f"{part_name}: posture must be {posture!r}, got {payload.get('posture')!r}."
        )
    gates = payload.get("gates")
    if not isinstance(gates, Mapping) or set(gates) != set(scope):
        missing = sorted(set(scope) - set(gates or ()))
        extra = sorted(set(gates or ()) - set(scope))
        raise UKReleaseCertificationError(
            f"{part_name}: entry ids must equal the declared scope; "
            f"missing {missing}, extra {extra}."
        )
    failing = sorted(
        gate_id
        for gate_id, entry in gates.items()
        if entry.get("criticality") == "release_blocking"
        and entry.get("status") != "passed"
    )
    if failing:
        raise UKReleaseCertificationError(
            f"{part_name}: release-blocking entries not passed: {failing}."
        )
    expected = _scoped_digests(scope, phases=phases, policy_suffix=policy_suffix)
    for field in ("gates_manifest_sha256", "policy_sha256"):
        if payload.get(field) != expected[field]:
            raise UKReleaseCertificationError(
                f"{part_name}: {field} does not match the committed spec's "
                f"scoped manifest ({payload.get(field)!r} != "
                f"{expected[field]!r}); the part did not run the declared "
                "gate spec."
            )
    _verify_part_signature(part_name, payload, signing_key)


def _verify_part_signature(
    part_name: str, payload: Mapping[str, Any], signing_key: bytes
) -> None:
    attestation = payload.get("attestation")
    if not isinstance(attestation, Mapping):
        raise UKReleaseCertificationError(f"{part_name}: attestation absent.")
    if attestation.get("signing_error") is not None:
        raise UKReleaseCertificationError(
            f"{part_name}: unsigned report ({attestation['signing_error']}); "
            "every certification part must be signed by its producer."
        )
    signature = attestation.get("signature")
    if not isinstance(signature, str) or not signature:
        raise UKReleaseCertificationError(f"{part_name}: signature absent.")
    unsigned = json.loads(json.dumps(payload))
    unsigned["attestation"]["signature"] = None
    recomputed = hmac.new(
        signing_key, canonical_json_bytes(unsigned), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(recomputed, signature):
        raise UKReleaseCertificationError(
            f"{part_name}: signature does not authenticate under the "
            "release signing key."
        )


def _verify_union(
    parts: Mapping[str, tuple[Mapping[str, Any], bytes]],
    *,
    declared_ids: set[str],
    declared_phases: tuple[str, ...],
) -> None:
    seen: dict[str, list[str]] = {}
    phases_covered: set[str] = set()
    for part_name, (payload, _raw) in parts.items():
        for gate_id in payload["gates"]:
            seen.setdefault(gate_id, []).append(part_name)
        phases_covered.update(str(phase) for phase in payload["phases"])
    union = set(seen)
    gap = sorted(declared_ids - union)
    if gap:
        raise UKReleaseCertificationError(
            f"certification gap: declared gate ids evaluated by no part: {gap}."
        )
    undeclared = sorted(union - declared_ids)
    if undeclared:
        raise UKReleaseCertificationError(
            f"certification parts evaluate undeclared gate ids: {undeclared}."
        )
    overlap = sorted(
        gate_id
        for gate_id, owners in seen.items()
        if len(owners) > 1 and gate_id not in UK_SHARED_GATE_IDS
    )
    if overlap:
        raise UKReleaseCertificationError(
            f"certification overlap beyond the declared shared ids: {overlap}."
        )
    for shared in sorted(UK_SHARED_GATE_IDS):
        if len(seen.get(shared, [])) < 2:
            raise UKReleaseCertificationError(
                f"declared shared gate id {shared!r} was evaluated by "
                f"{seen.get(shared, [])}; a shared id must be measured on "
                "both of its frames."
            )
    if phases_covered != set(declared_phases):
        raise UKReleaseCertificationError(
            f"certification phase coverage {sorted(phases_covered)} does not "
            f"equal the declared phase order {list(declared_phases)}."
        )


def _verify_identity_join(
    *,
    spine_report_bytes: bytes,
    seam_report_bytes: bytes,
    seam_payload: Mapping[str, Any],
    release_cut_payload: Mapping[str, Any],
    spine_sidecar: Mapping[str, Any],
    build_record: Mapping[str, Any],
    candidate_path: Path,
    candidate_sha256: str,
    release_id: str,
) -> None:
    sidecar_binding = spine_sidecar.get("spine_gate_report")
    if not isinstance(sidecar_binding, Mapping):
        raise UKReleaseCertificationError(
            "spine sidecar carries no spine_gate_report binding."
        )
    spine_report_sha = hashlib.sha256(spine_report_bytes).hexdigest()
    if sidecar_binding.get("sha256") != spine_report_sha:
        raise UKReleaseCertificationError(
            "spine battery report bytes do not match the sidecar's binding; "
            "the report does not describe this spine build."
        )
    provenance = build_record.get("spine_provenance", {})
    recorded = provenance.get("spine_gate_report", {})
    if recorded.get("sha256") != spine_report_sha:
        raise UKReleaseCertificationError(
            "the seam's build record binds a different spine battery report "
            "than the one supplied; the calibration did not consume this "
            "spine build."
        )
    artifacts = build_record.get("artifacts", {})
    staged = artifacts.get("staging_h5", {})
    if staged.get("sha256") != candidate_sha256:
        raise UKReleaseCertificationError(
            "the seam's build record staged a different candidate than the "
            "one under certification."
        )
    measured_candidate = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    if measured_candidate != candidate_sha256:
        raise UKReleaseCertificationError(
            f"candidate bytes measure {measured_candidate}, not the pinned "
            f"{candidate_sha256}."
        )
    seam_report_sha = hashlib.sha256(seam_report_bytes).hexdigest()
    recorded_seam = artifacts.get("terminal_gate_json", {})
    if recorded_seam.get("sha256") != seam_report_sha:
        raise UKReleaseCertificationError(
            "the seam battery report bytes do not match the build record's binding."
        )
    diagnostics_sha = artifacts.get("diagnostics_json", {}).get("sha256")
    for part_name, payload in (
        ("calibration_seam", seam_payload),
        ("release_cut", release_cut_payload),
    ):
        evidence = payload.get("release_evidence", {})
        if evidence.get("calibration_diagnostics_sha256") != diagnostics_sha:
            raise UKReleaseCertificationError(
                f"{part_name}: release evidence pins a different diagnostics "
                "digest than the build record; the parts were not measured "
                "on one calibration."
            )
    if release_cut_payload.get("release_id") != release_id:
        raise UKReleaseCertificationError(
            "the release-cut battery ran under release id "
            f"{release_cut_payload.get('release_id')!r}, not the "
            f"certification's {release_id!r}."
        )
    if release_cut_payload.get("release_candidate") is not True:
        raise UKReleaseCertificationError(
            "the release-cut battery must run at release-candidate strictness."
        )
    if release_cut_payload.get("shippable") is not True:
        raise UKReleaseCertificationError(
            "the release-cut battery's own report is not shippable."
        )


def _verify_score_receipt(receipt: Mapping[str, Any], *, candidate_sha256: str) -> None:
    artifacts = receipt.get("artifacts")
    scored = (
        artifacts.get("candidate", {}).get("sha256")
        if isinstance(artifacts, Mapping)
        else None
    )
    if scored != candidate_sha256:
        raise UKReleaseCertificationError(
            "the score receipt's artifacts.candidate.sha256 is "
            f"{scored!r}, not the candidate under certification "
            f"({candidate_sha256!r}); the rule-1 score must be measured on "
            "this candidate's bytes."
        )


def _status_census(payload: Mapping[str, Any]) -> dict[str, int]:
    census: dict[str, int] = {}
    for entry in payload["gates"].values():
        status = str(entry.get("status"))
        census[status] = census.get(status, 0) + 1
    return census


@lru_cache(maxsize=8)
def _scoped_digests(
    scope: frozenset[str],
    *,
    phases: tuple[str, ...],
    policy_suffix: str,
) -> dict[str, str]:
    manifest = uk_scoped_gate_manifest(
        scope, phases=phases, policy_suffix=policy_suffix
    )
    return _manifest_digests(manifest)


@lru_cache(maxsize=1)
def _full_manifest_digests() -> dict[str, str]:
    return _manifest_digests(_uk_gates_spec())


def _manifest_digests(manifest: GatesManifest) -> dict[str, str]:
    run = GateBatteryRun(
        manifest,
        release_id="uk-certification-digest-derivation",
        report_path=Path(os.devnull),
        release_candidate=False,
        registry=UK_GATE_REGISTRY,
    )
    return {
        "gates_manifest_sha256": run.gates_manifest_sha256,
        "policy_sha256": run.policy_sha256,
        "spec_fingerprint": run.spec_fingerprint,
    }


def _sign_certification(payload: dict[str, Any], signing_key: bytes) -> None:
    attestation = {
        "producer": "microcosm.build.uk_runtime.release_certification",
        "signature_algorithm": "hmac-sha256",
        "signing_key_sha256": hashlib.sha256(signing_key).hexdigest(),
        "signature": None,
    }
    payload["attestation"] = attestation
    attestation["signature"] = hmac.new(
        signing_key, canonical_json_bytes(payload), hashlib.sha256
    ).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
