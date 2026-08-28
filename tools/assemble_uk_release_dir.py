"""Assemble a certified UK national candidate into a publishable release.

This derivative packaging step closes the certification/build-record identity
join, mints the calibration weight bundle, writes the two release manifests,
copies the signed evidence byte-for-byte, and validates the finished release
directory before printing the inspect-lane publication command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import shutil
from collections.abc import Mapping
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from microcosm.build.uk_runtime.national_frame import load_uk_national_frame
from microcosm.build.uk_runtime.release_identity import UK_NATIONAL_RELEASE_ID
from microcosm.data.contract import validate_release_dir

_ATTEMPT_PREFIX = "uk-frs-calibration-attempt-"
_ATTEMPT_SUFFIX = re.compile(r"(?P<timestamp>\d{8}T\d{6}Z)-(?P<uuid>[0-9a-f]{8})")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")
_REPO_ID = "policyengine/populace-uk-private"
_RUNTIME_PACKAGES = (
    "python",
    "policyengine-core",
    "policyengine-uk",
    "microcosm-data",
    "microcosm-build",
    "microcosm-calibrate",
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = _assemble(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _assemble(args: argparse.Namespace) -> dict[str, object]:
    certification_bytes = args.certification_json.read_bytes()
    certification = _load_json_bytes(
        certification_bytes, label="--certification-json"
    )
    build_record = _load_json(args.build_record_json, label="--build-record-json")
    diagnostics_bytes = args.diagnostics_json.read_bytes()
    diagnostics = _load_json_bytes(diagnostics_bytes, label="--diagnostics-json")

    spine_report_path = args.spine_h5.with_suffix(".spine_gates.json")
    inputs = {
        "candidate": args.candidate_h5,
        "certification": args.certification_json,
        "diagnostics": args.diagnostics_json,
        "calibration seam report": args.seam_gate_report,
        "release-cut report": args.release_cut_gate_json,
        "spine report": spine_report_path,
        "score receipt": args.score_receipt,
        "spine H5": args.spine_h5,
    }
    measured = {name: _sha256(path) for name, path in inputs.items()}

    candidate = _mapping(certification.get("candidate"), "certification.candidate")
    artifacts = _mapping(build_record.get("artifacts"), "build_record.artifacts")
    staging_h5 = _mapping(
        artifacts.get("staging_h5"), "build_record.artifacts.staging_h5"
    )
    diagnostics_artifact = _mapping(
        artifacts.get("diagnostics_json"),
        "build_record.artifacts.diagnostics_json",
    )
    terminal_artifact = _mapping(
        artifacts.get("terminal_gate_json"),
        "build_record.artifacts.terminal_gate_json",
    )
    parts = _mapping(certification.get("parts"), "certification.parts")
    spine_part = _mapping(parts.get("spine"), "certification.parts.spine")
    seam_part = _mapping(
        parts.get("calibration_seam"), "certification.parts.calibration_seam"
    )
    release_cut_part = _mapping(
        parts.get("release_cut"), "certification.parts.release_cut"
    )
    score_binding = _mapping(
        certification.get("score_receipt"), "certification.score_receipt"
    )
    input_posture = _mapping(
        build_record.get("input_posture"), "build_record.input_posture"
    )

    _require_equal(
        "candidate bytes vs certification.candidate.sha256",
        measured["candidate"],
        candidate.get("sha256"),
    )
    _require_equal(
        "candidate bytes vs build_record.artifacts.staging_h5.sha256",
        measured["candidate"],
        staging_h5.get("sha256"),
    )
    _require_equal(
        "diagnostics bytes vs certification.diagnostics_sha256",
        measured["diagnostics"],
        certification.get("diagnostics_sha256"),
    )
    _require_equal(
        "diagnostics bytes vs build_record.artifacts.diagnostics_json.sha256",
        measured["diagnostics"],
        diagnostics_artifact.get("sha256"),
    )
    _require_equal(
        "calibration seam report vs build_record.artifacts.terminal_gate_json.sha256",
        measured["calibration seam report"],
        terminal_artifact.get("sha256"),
    )
    _require_equal(
        "calibration seam report vs certification.parts.calibration_seam.sha256",
        measured["calibration seam report"],
        seam_part.get("sha256"),
    )
    _require_equal(
        "release-cut report vs certification.parts.release_cut.sha256",
        measured["release-cut report"],
        release_cut_part.get("sha256"),
    )
    _require_equal(
        "spine report vs certification.parts.spine.sha256",
        measured["spine report"],
        spine_part.get("sha256"),
    )
    _require_equal(
        "score receipt vs certification.score_receipt.sha256",
        measured["score receipt"],
        score_binding.get("sha256"),
    )
    _require_equal(
        "spine H5 vs build_record.input_posture.sha256",
        measured["spine H5"],
        input_posture.get("sha256"),
    )
    _require_equal(
        "certification.release_id",
        certification.get("release_id"),
        UK_NATIONAL_RELEASE_ID,
    )
    if certification.get("shippable") is not True:
        raise SystemExit(
            "error: certification.shippable must be true before release assembly"
        )

    attempt_id = build_record.get("build_id")
    if not isinstance(attempt_id, str):
        raise SystemExit("error: build_record.build_id must be a string")
    cut_tag = _cut_tag(attempt_id, args.cut_tag)
    runtime = _runtime_versions(build_record, dict(args.runtime_version))
    code_pin = _diagnostics_code_pin(diagnostics)
    candidate_filename = candidate.get("filename")
    if not isinstance(candidate_filename, str) or not candidate_filename:
        raise SystemExit("error: certification.candidate.filename must be non-empty")
    dataset_key = Path(candidate_filename).stem
    calibration_filename = f"{dataset_key}_calibration.npz"
    calibration_path = args.candidate_h5.parent / calibration_filename
    target_surface = _mapping(
        diagnostics.get("target_surface"), "diagnostics.target_surface"
    )
    target_registry = _mapping(
        diagnostics.get("target_registry"), "diagnostics.target_registry"
    )
    candidate_frame, _candidate_provenance = load_uk_national_frame(
        args.candidate_h5
    )
    spine_frame, _spine_provenance = load_uk_national_frame(args.spine_h5)
    household_weight = np.asarray(
        candidate_frame.weights_for("household").values, dtype=np.float64
    )
    initial_household_weight = np.asarray(
        spine_frame.weights_for("household").values, dtype=np.float64
    )

    np.savez(
        calibration_path,
        household_weight=household_weight,
        initial_household_weight=initial_household_weight,
    )
    calibration_sha = _sha256(calibration_path)

    created_at = datetime.now(UTC).isoformat()
    release_dir = args.out_dir / UK_NATIONAL_RELEASE_ID
    release_dir.mkdir(parents=True, exist_ok=True)
    build_manifest = {
        "build_id": UK_NATIONAL_RELEASE_ID,
        "code": {
            "repository": "PolicyEngine/microcosm",
            "git_commit": code_pin,
            "git_dirty": False,
        },
        "build_sha": code_pin[:7],
        "runtime": runtime,
        "dataset": {
            "filename": candidate_filename,
            "sha256": measured["candidate"],
        },
        "calibration": {
            "filename": calibration_filename,
            "sha256": calibration_sha,
            "target_surface": {
                "sha256": target_surface.get("sha256"),
                "n_targets": target_surface.get("n_targets"),
            },
            "target_registry": {
                "version": target_registry.get("version"),
                "n_specs": target_registry.get("n_specs"),
            },
        },
        "gates": {
            "spine": spine_part.get("statuses"),
            "calibration_seam": seam_part.get("statuses"),
            "release_cut": release_cut_part.get("statuses"),
            "gate_summary": build_record.get("gate_summary"),
        },
        "attempt_id": attempt_id,
        "cut_tag": cut_tag,
        "created_at": created_at,
    }
    build_manifest_path = release_dir / "build_manifest.json"
    _write_json(build_manifest_path, build_manifest)

    evidence = {
        "calibration_diagnostics": (
            args.diagnostics_json,
            "calibration_diagnostics.json",
        ),
        "release_certification": (
            args.certification_json,
            "release_certification.json",
        ),
        "terminal_gates": (args.seam_gate_report, "terminal_gates.json"),
        "release_cut_gates": (
            args.release_cut_gate_json,
            "release_cut_gates.json",
        ),
        "spine_gates": (spine_report_path, "spine_gates.json"),
        "score_vs_enhanced_frs": (
            args.score_receipt,
            "score_vs_enhanced_frs.json",
        ),
    }

    def artifact(kind: str, path: str, sha256: str) -> dict[str, str]:
        return {
            "kind": kind,
            "path": path,
            "repo_id": _REPO_ID,
            "revision": cut_tag,
            "sha256": sha256,
        }

    release_manifest = {
        "schema_version": 1,
        "data_package": {
            "name": "microcosm-data",
            "version": runtime["microcosm-data"],
        },
        "default_datasets": {"national": dataset_key},
        "build": {
            "build_id": UK_NATIONAL_RELEASE_ID,
            "built_at": created_at,
            "built_with_core_package": {
                "name": "policyengine-core",
                "version": runtime["policyengine-core"],
            },
            "built_with_model_package": {
                "name": "policyengine-uk",
                "version": runtime["policyengine-uk"],
            },
            "attempt_id": attempt_id,
            "cut_tag": cut_tag,
        },
        "compatible_core_packages": [
            {
                "name": "policyengine-core",
                "specifier": f"=={runtime['policyengine-core']}",
            }
        ],
        "compatible_model_packages": [
            {
                "name": "policyengine-uk",
                "specifier": f"=={runtime['policyengine-uk']}",
            }
        ],
        "artifacts": {
            dataset_key: artifact(
                "microdata", candidate_filename, measured["candidate"]
            ),
            f"{dataset_key}_calibration": artifact(
                "calibration", calibration_filename, calibration_sha
            ),
            **{
                key: artifact("diagnostics", filename, measured[name])
                for key, (_source, filename), name in (
                    (
                        "calibration_diagnostics",
                        evidence["calibration_diagnostics"],
                        "diagnostics",
                    ),
                    (
                        "release_certification",
                        evidence["release_certification"],
                        "certification",
                    ),
                    (
                        "terminal_gates",
                        evidence["terminal_gates"],
                        "calibration seam report",
                    ),
                    (
                        "release_cut_gates",
                        evidence["release_cut_gates"],
                        "release-cut report",
                    ),
                    (
                        "spine_gates",
                        evidence["spine_gates"],
                        "spine report",
                    ),
                    (
                        "score_vs_enhanced_frs",
                        evidence["score_vs_enhanced_frs"],
                        "score receipt",
                    ),
                )
            },
        },
        "description": (
            "UK national calibration pipeline release assembled from attempt "
            f"{attempt_id} at immutable cut {cut_tag}."
        ),
        "publisher_labels": {
            "obr": "Office for Budget Responsibility",
            "hmrc": "HM Revenue and Customs",
            "ons": "Office for National Statistics",
            "dwp": "Department for Work and Pensions",
        },
    }
    release_manifest_path = release_dir / "release_manifest.json"
    _write_json(release_manifest_path, release_manifest)

    evidence_summary: dict[str, dict[str, str]] = {}
    for key, (source, filename) in evidence.items():
        destination = release_dir / filename
        shutil.copyfile(source, destination)
        evidence_summary[key] = {
            "path": str(destination),
            "sha256": _sha256(destination),
        }

    validate_release_dir(release_dir)
    publish_command = (
        f"uv run python -m microcosm.data.publish_cli {release_dir} "
        f"--repo-id {_REPO_ID} --artifact-root {args.candidate_h5.parent} "
        f"--no-latest --tag-name {cut_tag}"
    )
    return {
        "release_id": UK_NATIONAL_RELEASE_ID,
        "release_dir": str(release_dir),
        "cut_tag": cut_tag,
        "dataset": {
            "path": str(args.candidate_h5),
            "sha256": measured["candidate"],
        },
        "calibration": {
            "path": str(calibration_path),
            "sha256": calibration_sha,
        },
        "build_manifest": {
            "path": str(build_manifest_path),
            "sha256": _sha256(build_manifest_path),
        },
        "release_manifest": {
            "path": str(release_manifest_path),
            "sha256": _sha256(release_manifest_path),
        },
        "evidence": evidence_summary,
        "publish_command": publish_command,
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-h5", required=True, type=Path)
    parser.add_argument("--spine-h5", required=True, type=Path)
    parser.add_argument("--certification-json", type=Path)
    parser.add_argument("--build-record-json", required=True, type=Path)
    parser.add_argument("--diagnostics-json", required=True, type=Path)
    parser.add_argument("--seam-gate-report", required=True, type=Path)
    parser.add_argument("--release-cut-gate-json", type=Path)
    parser.add_argument("--score-receipt", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("releases"))
    parser.add_argument("--cut-tag")
    parser.add_argument(
        "--runtime-version",
        action="append",
        default=[],
        type=_runtime_version,
        metavar="PACKAGE=VERSION",
    )
    args = parser.parse_args(argv)
    args.certification_json = (
        args.certification_json
        or args.candidate_h5.with_suffix(".release_certification.json")
    )
    args.release_cut_gate_json = (
        args.release_cut_gate_json
        or args.candidate_h5.with_suffix(".release_cut_gates.json")
    )
    distinct = {
        "--candidate-h5": args.candidate_h5,
        "--spine-h5": args.spine_h5,
        "--certification-json": args.certification_json,
        "--build-record-json": args.build_record_json,
        "--diagnostics-json": args.diagnostics_json,
        "--seam-gate-report": args.seam_gate_report,
        "--release-cut-gate-json": args.release_cut_gate_json,
        "--score-receipt": args.score_receipt,
    }
    resolved: dict[Path, str] = {}
    for flag, path in distinct.items():
        canonical = path.resolve()
        if canonical in resolved:
            parser.error(f"{flag} aliases {resolved[canonical]}: {path}")
        resolved[canonical] = flag
    return args


def _runtime_version(value: str) -> tuple[str, str]:
    package, separator, package_version = value.partition("=")
    if not separator or package not in _RUNTIME_PACKAGES or not package_version:
        raise argparse.ArgumentTypeError(
            "expected PACKAGE=VERSION for one of " + ", ".join(_RUNTIME_PACKAGES)
        )
    return package, package_version


def _runtime_versions(
    build_record: Mapping[str, object], overrides: dict[str, str]
) -> dict[str, str]:
    spine_provenance = _mapping(
        build_record.get("spine_provenance"), "build_record.spine_provenance"
    )
    rules_engine = _mapping(
        spine_provenance.get("rules_engine"),
        "build_record.spine_provenance.rules_engine",
    )
    policyengine_uk = overrides.get("policyengine-uk") or rules_engine.get("version")
    if not isinstance(policyengine_uk, str) or not policyengine_uk:
        raise SystemExit(
            "error: build_record.spine_provenance.rules_engine.version is required"
        )
    runtime = {
        "python": overrides.get("python", platform.python_version()),
        "policyengine-core": _distribution_version("policyengine-core", overrides),
        "policyengine-uk": policyengine_uk,
        "microcosm-data": _distribution_version("microcosm-data", overrides),
        "microcosm-build": _distribution_version("microcosm-build", overrides),
        "microcosm-calibrate": _distribution_version(
            "microcosm-calibrate", overrides
        ),
    }
    return runtime


def _distribution_version(package: str, overrides: Mapping[str, str]) -> str:
    if package in overrides:
        return overrides[package]
    try:
        return version(package)
    except PackageNotFoundError as error:
        raise SystemExit(
            f"error: runtime version for {package} is unavailable; pass "
            f"--runtime-version {package}=VERSION"
        ) from error


def _diagnostics_code_pin(diagnostics: Mapping[str, object]) -> str:
    build = _mapping(diagnostics.get("build"), "diagnostics.build")
    code_pin = build.get("code_pin")
    if not isinstance(code_pin, str) or not _GIT_COMMIT.fullmatch(code_pin):
        raise SystemExit(
            "error: diagnostics.build.code_pin must be a 40-character lowercase "
            "git commit"
        )
    return code_pin


def _cut_tag(attempt_id: str, override: str | None) -> str:
    prefix = UK_NATIONAL_RELEASE_ID + "-"
    if override is not None:
        if not override.startswith(prefix) or len(override) == len(prefix):
            raise SystemExit(
                f"error: --cut-tag must start with {prefix!r} and include a suffix"
            )
        return override
    if not attempt_id.startswith(_ATTEMPT_PREFIX):
        raise SystemExit(
            f"error: build_record.build_id must start with {_ATTEMPT_PREFIX!r}"
        )
    suffix = attempt_id.removeprefix(_ATTEMPT_PREFIX)
    if not _ATTEMPT_SUFFIX.fullmatch(suffix):
        raise SystemExit(
            "error: build_record.build_id must end with "
            "<YYYYMMDDTHHMMSSZ>-<uuid8>"
        )
    return f"{UK_NATIONAL_RELEASE_ID}-{suffix}"


def _load_json(path: Path, *, label: str) -> Mapping[str, object]:
    return _load_json_bytes(path.read_bytes(), label=label)


def _load_json_bytes(raw: bytes, *, label: str) -> Mapping[str, object]:
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"error: {label} is not valid JSON: {error}") from error
    return _mapping(payload, label)


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise SystemExit(f"error: {label} must be a JSON object")
    return value


def _require_equal(label: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise SystemExit(f"error: {label} mismatch: {actual!r} != {expected!r}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=1, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
