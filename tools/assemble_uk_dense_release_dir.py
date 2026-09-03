"""Assemble the dense joint UK line's release directory (microcosm#762 A18).

Mirrors ``assemble_uk_release_dir.py`` for the ``microcosm-uk-2024-25-dense``
line: verify the complete hash join over a finished ``--release-candidate``
run of ``build_uk_rowwise_candidate.py`` (its manifest, signed gate report,
diagnostics and incumbent score, plus the spine, ladder, Ledger artifact and
incumbent extraction it stood on), run the release-candidate pre-flight,
mint the per-cut tag from the run's attempt id, stage the bundle under the
non-default local-area role with UK evidence, validate it with the release
contract, and only then rename it into place. The candidate H5 is cloned
beside itself under the published filename; nothing is uploaded — the
summary prints the exact ``publish_cli`` command for the human step
(``--no-latest``, per-cut tag, private repo).
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import json
import re
import shlex
import shutil
import subprocess
import sys
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

from microcosm.build.uk_runtime.release_identity import UK_DENSE_RELEASE_ID
from microcosm.data.contract import validate_release_dir

_REPO_ID = "policyengine/populace-uk-private"
_NAMESPACE = "uk_dense"
_DATASET_KEY = "microcosm_uk_2025_dense"
_DATASET_FILENAME = f"{_DATASET_KEY}.h5"
_ATTEMPT_SUFFIX = re.compile(r"[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}")
_RUNTIME_PACKAGES = ("policyengine-core", "policyengine-uk", "microcosm-data")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, *, label: str) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"error: {label} is not readable JSON: {error}") from error
    if not isinstance(payload, Mapping):
        raise SystemExit(f"error: {label} must be a JSON object")
    return payload


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise SystemExit(f"error: {label} must be a JSON object")
    return value


def _require_equal(label: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise SystemExit(f"error: {label}: {actual!r} != {expected!r}")


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    temporary.replace(path)


def _clone_file(source: Path, destination: Path) -> None:
    """Copy with an APFS clone when the platform offers one, else bytes."""

    if sys.platform == "darwin":
        result = subprocess.run(
            ["cp", "-c", str(source), str(destination)],
            capture_output=True,
            check=False,
        )
        if result.returncode == 0:
            return
    shutil.copyfile(source, destination)


def _preflight_module():
    spec = importlib.util.spec_from_file_location(
        "preflight_uk_local_release_candidate",
        Path(__file__).resolve().with_name("preflight_uk_local_release_candidate.py"),
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _cut_tag(attempt_id: str, override: str | None) -> str:
    prefix = UK_DENSE_RELEASE_ID + "-"
    if override is not None:
        if not override.startswith(prefix) or not _ATTEMPT_SUFFIX.fullmatch(
            override[len(prefix) :]
        ):
            raise SystemExit(
                f"error: --cut-tag must be {prefix}<YYYYMMDDTHHMMSSZ>-<uuid8>"
            )
        return override
    match = re.search(r"([0-9]{8}T[0-9]{6}Z-[0-9a-f]{8})$", attempt_id)
    if match is None:
        raise SystemExit(
            "error: the run's build id must end with <YYYYMMDDTHHMMSSZ>-<uuid8>"
        )
    return f"{UK_DENSE_RELEASE_ID}-{match.group(1)}"


def _reviewed_limitations(
    manifest: Mapping[str, object], report: Mapping[str, object]
) -> list[dict[str, object]]:
    limitations: list[dict[str, object]] = []
    for gate_id, entry in (report.get("gates") or {}).items():
        if (
            isinstance(entry, Mapping)
            and entry.get("criticality") == "diagnostic"
            and entry.get("status") != "passed"
        ):
            limitations.append(
                {
                    "id": f"diagnostic_gate_{gate_id}",
                    "summary": f"diagnostic gate {gate_id} did not pass; it never blocks release.",
                    "failures": list(entry.get("failures") or [])[:5],
                }
            )
    for name, record in sorted((manifest.get("measure_exclusions") or {}).items()):
        if isinstance(record, Mapping):
            limitations.append(
                {
                    "id": f"measure_exclusion:{name}",
                    "summary": str(record.get("reason", ""))[:300],
                    "tracking": record.get("tracking"),
                    "expires_on": record.get("expires_on"),
                }
            )
    solve = _mapping(manifest.get("solve"), "manifest.solve")
    exclusions = solve.get("area_support_exclusions")
    if isinstance(exclusions, Mapping):
        for area in exclusions.get("entries_stood_on") or []:
            limitations.append(
                {
                    "id": f"area_support_exclusion:{area}",
                    "summary": "reviewed exclusion of the local-authority support floor.",
                }
            )
    cross_grain = _mapping(manifest.get("cross_grain"), "manifest.cross_grain")
    for bridge in cross_grain.get("unbound_bridges") or []:
        if isinstance(bridge, Mapping):
            limitations.append(
                {
                    "id": f"unbound_bridge:{bridge.get('bridge_id')}",
                    "summary": "cross-grain bridge unbound with a receipt; the lower rows bind as published.",
                }
            )
    holdout = _mapping(
        _mapping(manifest.get("fit"), "manifest.fit").get("rotated_holdout"),
        "manifest.fit.rotated_holdout",
    )
    limitations.append(
        {
            "id": "rotated_holdout",
            "summary": "capped relative-error loss on local rows the solve never saw.",
            "mean_holdout_loss": holdout.get("mean_holdout_loss"),
            "worst_holdout_loss": holdout.get("worst_holdout_loss"),
        }
    )
    return limitations


def _assemble(args: argparse.Namespace) -> dict[str, object]:
    candidate_dir: Path = args.candidate_dir
    manifest = _load_json(
        candidate_dir / "rowwise_candidate_manifest.json",
        label="rowwise_candidate_manifest.json",
    )
    outputs = _mapping(manifest.get("outputs"), "manifest.outputs")
    identity = _mapping(manifest.get("identity"), "manifest.identity")

    def output_path(key: str) -> Path:
        entry = _mapping(outputs.get(key), f"manifest.outputs.{key}")
        path = Path(str(entry.get("path")))
        return path if path.is_absolute() else candidate_dir / path

    candidate_h5 = output_path("dataset")
    diagnostics_path = output_path("calibration_diagnostics")
    report_path = output_path("local_gate_report")
    score_path = candidate_dir / "score_vs_incumbent.json"
    inputs = {
        "candidate": candidate_h5,
        "diagnostics": diagnostics_path,
        "gate report": report_path,
        "score receipt": score_path,
        "spine H5": args.spine_h5,
    }
    for label, path in inputs.items():
        if not path.is_file():
            raise SystemExit(f"error: {label} missing: {path}")
    measured = {label: _sha256(path) for label, path in inputs.items()}
    for key, label in (
        ("dataset", "candidate"),
        ("calibration_diagnostics", "diagnostics"),
        ("local_gate_report", "gate report"),
    ):
        _require_equal(
            f"{label} bytes vs manifest.outputs.{key}.sha256",
            measured[label],
            _mapping(outputs.get(key), key).get("sha256"),
        )
    spine_identity = _mapping(identity.get("spine"), "manifest.identity.spine")
    _require_equal(
        "spine H5 vs manifest.identity.spine.sha256",
        measured["spine H5"],
        spine_identity.get("sha256"),
    )
    for pinned in ("spine", "ladder"):
        if _mapping(identity.get(pinned), pinned).get("pin_verified") is not True:
            raise SystemExit(
                f"error: manifest.identity.{pinned}.pin_verified is not true"
            )
    report = _load_json(report_path, label="local gate report")
    rows = sorted(glob.glob(str(candidate_dir / "logbook-spool" / "*.json")))
    if not rows:
        raise SystemExit("error: the candidate directory carries no Logbook row")
    logbook_row = _load_json(Path(rows[-1]), label="Logbook row")
    attempt_id = str(logbook_row.get("build_id") or "")
    _require_equal(
        "gate report release_id vs Logbook build_id",
        report.get("release_id"),
        attempt_id,
    )
    preflight = _preflight_module()
    failures = preflight.check_candidate_dir(candidate_dir)
    if failures:
        raise SystemExit(
            "error: release-candidate pre-flight failed:\n"
            + "\n".join(f"  - {f}" for f in failures)
        )
    diagnostics = _load_json(diagnostics_path, label="calibration diagnostics")
    score = _load_json(score_path, label="score receipt")
    incumbent = _load_json(args.incumbent_manifest, label="--incumbent-manifest")
    runtime_block = _mapping(identity.get("runtime"), "manifest.identity.runtime")
    runtime = {}
    for package in _RUNTIME_PACKAGES:
        value = runtime_block.get(package)
        if not isinstance(value, str) or not value:
            raise SystemExit(f"error: manifest.identity.runtime.{package} is missing")
        runtime[package] = value
    code_pin = str(
        _mapping(identity.get("code"), "identity.code").get("git_commit") or ""
    )
    if not _GIT_COMMIT.fullmatch(code_pin):
        raise SystemExit(
            "error: manifest.identity.code.git_commit must be a 40-hex commit"
        )
    cut_tag = _cut_tag(attempt_id, args.cut_tag)
    parameters = _mapping(manifest.get("parameters"), "manifest.parameters")
    if parameters.get("release_candidate") is not True:
        raise SystemExit("error: the run was not a --release-candidate run")

    destination = args.out_dir / UK_DENSE_RELEASE_ID
    if destination.exists() or destination.is_symlink():
        raise SystemExit(
            f"error: {destination} already exists; remove the previous assembly first"
        )
    artifact_root: Path = args.artifact_root or candidate_h5.parent
    published_h5 = artifact_root / _DATASET_FILENAME
    if published_h5.exists():
        if _sha256(published_h5) != measured["candidate"]:
            raise SystemExit(
                f"error: {published_h5} exists with different bytes; remove it first"
            )
    else:
        _clone_file(candidate_h5, published_h5)
    staging_parent = args.out_dir / f".assemble-{uuid.uuid4().hex}"
    staging_parent.mkdir(parents=True)
    try:
        return _stage_and_finalize(
            staging_parent=staging_parent,
            destination=destination,
            artifact_root=artifact_root,
            manifest=manifest,
            report=report,
            diagnostics=diagnostics,
            score=score,
            incumbent=incumbent,
            measured=measured,
            attempt_id=attempt_id,
            cut_tag=cut_tag,
            runtime=runtime,
            code_pin=code_pin,
            report_path=report_path,
            score_path=score_path,
            logbook_row=logbook_row,
        )
    finally:
        shutil.rmtree(staging_parent, ignore_errors=True)


def _stage_and_finalize(
    *,
    staging_parent: Path,
    destination: Path,
    artifact_root: Path,
    manifest: Mapping[str, object],
    report: Mapping[str, object],
    diagnostics: Mapping[str, object],
    score: Mapping[str, object],
    incumbent: Mapping[str, object],
    measured: Mapping[str, str],
    attempt_id: str,
    cut_tag: str,
    runtime: Mapping[str, str],
    code_pin: str,
    report_path: Path,
    score_path: Path,
    logbook_row: Mapping[str, object],
) -> dict[str, object]:
    created_at = datetime.now(UTC).isoformat()
    release_dir = staging_parent / UK_DENSE_RELEASE_ID
    release_dir.mkdir(parents=True)
    identity = _mapping(manifest.get("identity"), "identity")
    parameters = _mapping(manifest.get("parameters"), "parameters")
    solve = _mapping(manifest.get("solve"), "solve")
    fit = _mapping(manifest.get("fit"), "fit")
    weights = _mapping(manifest.get("weights"), "weights")
    n_targets = len(list(diagnostics.get("targets") or []))
    households = int(solve.get("n_households") or diagnostics.get("n_records") or 0)
    shipped_diagnostics = {
        **diagnostics,
        # The local-area contract reads these two; the driver's schema-6
        # diagnostics carry the same facts under n_records / len(targets).
        "households": households,
        "n_targets": n_targets,
        "source_diagnostics_sha256": measured["diagnostics"],
    }
    _write_json(release_dir / "calibration_diagnostics.json", shipped_diagnostics)
    gates = _mapping(report.get("gates"), "report.gates")
    gate_summary = {
        "schema_version": 1,
        "source": _DATASET_KEY,
        "signed_report": "uk_local_gates.json",
        "gates": {
            gate_id: {
                "passed": entry.get("status") == "passed",
                "status": entry.get("status"),
                "failures": list(entry.get("failures") or []),
            }
            for gate_id, entry in gates.items()
            if isinstance(entry, Mapping)
            and entry.get("criticality") == "release_blocking"
        },
        "diagnostic_gates": {
            gate_id: {
                "status": entry.get("status"),
                "failures": list(entry.get("failures") or [])[:10],
            }
            for gate_id, entry in gates.items()
            if isinstance(entry, Mapping)
            and entry.get("criticality") != "release_blocking"
        },
        "reviewed_limitations": _reviewed_limitations(manifest, report),
    }
    _write_json(release_dir / "gate_summary.json", gate_summary)
    ladder = _mapping(identity.get("ladder"), "identity.ladder")
    ledger = _mapping(identity.get("ledger"), "identity.ledger")
    spine = _mapping(identity.get("spine"), "identity.spine")
    coverage = {
        "schema_version": 1,
        "spine": {
            "path_name": Path(str(spine.get("path"))).name,
            "sha256": spine.get("sha256"),
            "bytes": spine.get("bytes"),
            "provenance": {
                k: v
                for k, v in _mapping(
                    spine.get("spine_provenance"), "spine_provenance"
                ).items()
                if k
                in (
                    "rules_engine",
                    "source_vintages",
                    "stages",
                    "stochastic_contract_sha256",
                )
            },
        },
        "ledger_artifact": {
            "path_name": ledger.get("path_name"),
            "facts_sha256": ledger.get("facts_sha256"),
            "manifest_sha256": ledger.get("manifest_sha256"),
            "fact_row_count": ledger.get("fact_row_count"),
            "schema_version": ledger.get("schema_version"),
        },
        "geography_ladder": {
            "sha256": ladder.get("sha256"),
            "bytes": ladder.get("bytes"),
            "layer_vintages": ladder.get("layer_vintages"),
        },
        "incumbent": {
            "snapshot": incumbent.get("inputs"),
            "period": incumbent.get("period"),
            "households": incumbent.get("households"),
            "score_receipt_sha256": measured["score receipt"],
            "rows_compared": score.get("rows_compared"),
            "incumbent_missing_areas": score.get("incumbent_missing_areas"),
        },
        "doctrine": dict(_mapping(parameters.get("doctrine"), "parameters.doctrine")),
        "measure_exclusions": {
            name: {"expires_on": rec.get("expires_on"), "tracking": rec.get("tracking")}
            for name, rec in (manifest.get("measure_exclusions") or {}).items()
            if isinstance(rec, Mapping)
        },
        "signed_deferrals": {
            "binding_adjudications": _mapping(
                solve.get("binding_adjudications"), "binding_adjudications"
            ).get("stood_on"),
            "area_support_exclusions": _mapping(
                solve.get("area_support_exclusions"), "area_support_exclusions"
            ).get("entries_stood_on"),
            "unbound_bridges": [
                b.get("bridge_id")
                for b in (
                    _mapping(manifest.get("cross_grain"), "cross_grain").get(
                        "unbound_bridges"
                    )
                    or []
                )
                if isinstance(b, Mapping)
            ],
            "licensed_empty_legs": len(
                _mapping(manifest.get("cross_grain"), "cross_grain").get(
                    "empty_legs_licensed"
                )
                or []
            ),
        },
        "holdout": dict(_mapping(fit.get("rotated_holdout"), "fit.rotated_holdout")),
        "uprating": {
            k: v
            for k, v in _mapping(
                manifest.get("ladder_household_uprating"), "ladder_household_uprating"
            ).items()
            if k != "reason"
        },
        "weights": {
            "calibration_mass_change": weights.get("calibration_mass_change"),
            "realized_max_weight_ratio_vs_design": weights.get(
                "realized_max_weight_ratio_vs_design"
            ),
        },
        "logbook_row": {
            "build_id": logbook_row.get("build_id"),
            "row_digest": logbook_row.get("row_digest"),
            "prev_row_digest": logbook_row.get("prev_row_digest"),
        },
    }
    coverage["holdout"].pop("folds", None)
    _write_json(release_dir / "uk_source_coverage.json", coverage)
    shutil.copyfile(report_path, release_dir / "uk_local_gates.json")
    shutil.copyfile(score_path, release_dir / "score_vs_incumbent.json")
    build_manifest = {
        "build_id": UK_DENSE_RELEASE_ID,
        "build_sha": code_pin[:7],
        "code": {
            "repository": "PolicyEngine/microcosm",
            "git_commit": code_pin,
            "git_dirty": False,
        },
        "runtime": dict(runtime),
        "dataset": {
            "filename": _DATASET_FILENAME,
            "sha256": measured["candidate"],
            "households": households,
            "clone_count": parameters.get("n_clones"),
        },
        "calibration": {
            "doctrine": dict(_mapping(parameters.get("doctrine"), "doctrine")),
            "epochs": parameters.get("epochs"),
            "n_targets": n_targets,
            "n_targets_by_kind": solve.get("n_targets_by_kind"),
            "final_loss": solve.get("final_loss"),
        },
        "gates": {
            gate_id: entry.get("status")
            for gate_id, entry in gates.items()
            if isinstance(entry, Mapping)
        },
        "attempt_id": attempt_id,
        "cut_tag": cut_tag,
        "created_at": created_at,
    }
    _write_json(release_dir / "build_manifest.json", build_manifest)

    def artifact(kind: str, path: str, sha256: str) -> dict[str, str]:
        return {
            "kind": kind,
            "path": path,
            "repo_id": _REPO_ID,
            "revision": cut_tag,
            "sha256": sha256,
        }

    evidence_files = (
        "calibration_diagnostics.json",
        "gate_summary.json",
        "uk_source_coverage.json",
        "uk_local_gates.json",
        "score_vs_incumbent.json",
        "build_manifest.json",
    )
    evidence_sha = {name: _sha256(release_dir / name) for name in evidence_files}
    release_manifest = {
        "schema_version": 1,
        "data_package": {
            "name": "microcosm-data",
            "version": runtime["microcosm-data"],
        },
        "dataset_role": "non_default_local_area",
        "is_default": False,
        "default_datasets": {},
        "namespace": _NAMESPACE,
        "build": {
            "build_id": UK_DENSE_RELEASE_ID,
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
            {"name": "policyengine-uk", "specifier": f"=={runtime['policyengine-uk']}"}
        ],
        "artifacts": {
            _DATASET_KEY: artifact(
                "microdata", _DATASET_FILENAME, measured["candidate"]
            ),
            **{
                Path(name).stem: artifact("diagnostics", name, evidence_sha[name])
                for name in evidence_files
                if name != "build_manifest.json"
            },
        },
        "reviewed_limitations": gate_summary["reviewed_limitations"],
        "description": f"UK dense joint national + local line assembled from attempt {attempt_id} at immutable cut {cut_tag}; inspect lane, never the default slot.",
        "publisher_labels": {
            "obr": "Office for Budget Responsibility",
            "hmrc": "HM Revenue and Customs",
            "ons": "Office for National Statistics",
            "dwp": "Department for Work and Pensions",
            "voa": "Valuation Office Agency",
            "slc": "Student Loans Company",
        },
    }
    _write_json(release_dir / "release_manifest.json", release_manifest)
    sums = {
        name: _sha256(release_dir / name)
        for name in sorted(p.name for p in release_dir.glob("*.json"))
    }
    sums[_DATASET_FILENAME] = measured["candidate"]
    (release_dir / "sha256sums.txt").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(sums.items()))
    )
    validate_release_dir(release_dir)
    release_dir.rename(destination)
    publish_command = shlex.join(
        [
            "uv",
            "run",
            "python",
            "-m",
            "microcosm.data.publish_cli",
            str(destination),
            "--repo-id",
            _REPO_ID,
            "--artifact-root",
            str(artifact_root),
            "--no-latest",
            "--tag-name",
            cut_tag,
        ]
    )
    return {
        "release_id": UK_DENSE_RELEASE_ID,
        "release_dir": str(destination),
        "cut_tag": cut_tag,
        "dataset": {
            "path": str(artifact_root / _DATASET_FILENAME),
            "sha256": measured["candidate"],
        },
        "evidence": {name: _sha256(destination / name) for name in evidence_files},
        "publish_command": publish_command,
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-dir",
        required=True,
        type=Path,
        help="a finished --release-candidate run directory",
    )
    parser.add_argument("--spine-h5", required=True, type=Path)
    parser.add_argument(
        "--incumbent-manifest",
        required=True,
        type=Path,
        help="incumbent_local_surface_manifest.json from the extractor",
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help="where the published H5 is cloned (default: beside the candidate)",
    )
    parser.add_argument("--cut-tag")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    summary = _assemble(_parse_args(argv))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
