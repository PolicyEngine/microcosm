"""Assemble the dense joint UK line's release directory (microcosm#762 A18).

Mirrors ``assemble_uk_release_dir.py`` for the ``microcosm-uk-2024-25-dense``
line: verify the complete hash join over a finished ``--release-candidate``
run of ``build_uk_rowwise_candidate.py`` (its manifest, signed gate report,
full gate report and diagnostics, plus the spine, ladder and Ledger artifact
it stood on), run the release-candidate pre-flight, mint the per-cut tag from
the run's attempt id, stage the bundle under the non-default local-area role
with UK evidence, validate it with the release contract, and only then rename
it into place. The candidate H5 is cloned beside itself under the published
filename; nothing is uploaded — the summary prints the exact ``publish_cli``
command for the human step (``--no-latest --tag-only``, per-cut tag, private
repo).

The rotated holdout and the incumbent comparison are opt-in diagnostics
(microcosm#1301). The manifest's reviewed limitations say what the release
does not claim: no out-of-sample fit unless the holdout ran, no incumbent
comparison unless ``--incumbent-manifest`` attaches the score and the
surface evaluation (recorded, not enforced), how the engine measured the
file, the size record, and the reviewed exclusions the gates stood on, by
name with their register reasons copied verbatim.
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
from typing import NamedTuple

from pydantic import ValidationError

from microcosm.build.staging_v2 import (
    StagingContractError,
    disabled_staging_delivery,
    validate_staging_delivery,
)
from microcosm.build.uk_runtime.release_identity import UK_DENSE_RELEASE_ID
from microcosm.data.contract import (
    _UK_DENSE_SIGNED_EXCLUSION_GATES,
    _check_uk_incumbent_surface_evaluation,
    validate_release_dir,
)
from microcosm.diagnostics import (
    HIERARCHY_CALIBRATION_DIAGNOSTICS_SCHEMA_VERSIONS,
    parse_calibration_diagnostics,
)

_REPO_ID = "policyengine/populace-uk-private"
_NAMESPACE = "uk_dense"
# The published dense artifact's key and filename (microcosm#823: every
# 2024-25 line carries the FRS release vintage in its name). The contract
# mirrors the filename as ``_UK_DENSE_DATASET_FILENAME``; a lockstep test
# pins the two.
_DATASET_KEY = "microcosm_uk_2024_25_dense"
_DATASET_FILENAME = f"{_DATASET_KEY}.h5"
_ATTEMPT_SUFFIX = re.compile(r"[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}")
_RUNTIME_PACKAGES = ("policyengine-core", "policyengine-uk", "microcosm-data")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
#: The gates whose reviewed exclusions the manifest names, each with its
#: register reason copied verbatim (see ``_signed_exclusion_limitations``).
#: The contract owns the tuple, since it checks the names (microcosm#1301).
_SIGNED_EXCLUSION_GATES = _UK_DENSE_SIGNED_EXCLUSION_GATES


class _IncumbentAttachment(NamedTuple):
    """The incumbent comparison an operator attached with --incumbent-manifest.

    A named tuple rather than a dataclass: the tests load this tool by path,
    without registering it in ``sys.modules``.
    """

    manifest_path: Path
    manifest: Mapping[str, object]
    score_path: Path
    score: Mapping[str, object]
    surface_path: Path
    measured: Mapping[str, str]
    assessment: Mapping[str, object]


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


def _write_release_diagnostics(
    path: Path,
    *,
    source: Path,
    diagnostics: Mapping[str, object],
    measured_source_sha256: str,
    households: int,
) -> None:
    """Copy current diagnostics exactly; adapt only legacy candidate evidence."""

    # Schema 8 and schema 9 (labels in catalogs) are both current documents.
    if diagnostics.get("schema_version") in (
        HIERARCHY_CALIBRATION_DIAGNOSTICS_SCHEMA_VERSIONS
    ):
        try:
            parse_calibration_diagnostics(diagnostics)
        except ValidationError as error:
            raise SystemExit(
                "error: candidate calibration diagnostics do not satisfy the "
                f"canonical schema: {error}"
            ) from error
        shutil.copyfile(source, path)
        return

    shipped_diagnostics = {
        **diagnostics,
        "households": households,
        "n_targets": len(list(diagnostics.get("targets") or [])),
        "source_diagnostics_sha256": measured_source_sha256,
    }
    _write_json(path, shipped_diagnostics)


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


def _holdout_limitation(fit: Mapping[str, object]) -> dict[str, object]:
    """What the release claims about fit on rows the solve never saw."""

    holdout = _mapping(fit.get("rotated_holdout"), "manifest.fit.rotated_holdout")
    if holdout.get("skipped") is True:
        return {
            "id": "no_out_of_sample_claim",
            "summary": (
                "the rotated holdout was not requested (an opt-in diagnostic); "
                "the release makes no out-of-sample fit claim."
            ),
        }
    return {
        "id": "rotated_holdout",
        "claim": "report_only",
        "summary": "capped relative-error loss on local rows the solve never saw.",
        "mean_holdout_loss": holdout.get("mean_holdout_loss"),
        "worst_holdout_loss": holdout.get("worst_holdout_loss"),
    }


def _holdout_coverage(fit: Mapping[str, object]) -> dict[str, object]:
    """The coverage receipt's holdout block, with its claim discriminator."""

    holdout = _mapping(fit.get("rotated_holdout"), "fit.rotated_holdout")
    if holdout.get("skipped") is True:
        return {"run": False, "claim": "none"}
    return {
        **{
            key: value
            for key, value in holdout.items()
            if key not in ("folds", "graph_binding")
        },
        "run": True,
        "claim": "report_only",
    }


def _signed_exclusion_names(full_report: Mapping[str, object]) -> dict[str, list]:
    """The coverage's names of the reviewed exclusions each fit gate stood on.

    Every gate of ``_SIGNED_EXCLUSION_GATES`` is keyed, empty when it stood
    on none: the names the ``reviewed_exclusion`` limitations restate, which
    the contract checks (microcosm#1301).
    """

    names: dict[str, list] = {gate_id: [] for gate_id in _SIGNED_EXCLUSION_GATES}
    for limitation in _signed_exclusion_limitations(full_report):
        _, gate_id, name = str(limitation["id"]).split(":", 2)
        names[gate_id].append(name)
    return names


def _engine_measurement_limitation(solve: Mapping[str, object]) -> dict[str, object]:
    """How the engine measured the released file (no single-engine readback)."""

    resolution = _mapping(
        solve.get("measure_resolution") or {}, "solve.measure_resolution"
    )
    representation = resolution.get("engine_population_representation")
    representation = representation if isinstance(representation, Mapping) else {}
    return {
        "id": "engine_measurement",
        "summary": (
            "the fit gates read the measures the build resolved; the released "
            "file is not re-measured on a single engine."
        ),
        "blocks": resolution.get("blocks"),
        "representation_mode": representation.get("mode"),
        "representation_exact": representation.get("exact"),
        "representation_exact_scope": representation.get("exact_scope"),
    }


def _signed_exclusion_limitations(
    full_report: Mapping[str, object],
) -> list[dict[str, object]]:
    """The reviewed exclusions the release's fit gates stood on.

    Each carries its name and its register entry's reason (cut at 300
    characters), approver, adjudication and expiry, copied verbatim from the
    full gate report. The gate's own measurement of the row, such as its
    relative error, is not copied. That is not a no-figures guarantee: a
    reason is reviewer prose and can cite figures of its own, as the
    input-mass register's reasons cite national totals and percentages.
    """

    report = _mapping(full_report.get("report"), "full gate report.report")
    outcomes = report.get("outcomes")
    if not isinstance(outcomes, list):
        raise SystemExit("error: full gate report.report.outcomes must be a list")
    limitations: list[dict[str, object]] = []
    for outcome in outcomes:
        if not isinstance(outcome, Mapping):
            continue
        gate_id = outcome.get("id")
        if gate_id not in _SIGNED_EXCLUSION_GATES:
            continue
        details = outcome.get("details")
        details = details if isinstance(details, Mapping) else {}
        excluded = details.get("reviewed_exclusions")
        if not isinstance(excluded, Mapping):
            continue
        dormant = set(details.get("dormant_exclusions") or ())
        for name, record in sorted(excluded.items()):
            if name in dormant:
                continue
            record = record if isinstance(record, Mapping) else {"reason": record}
            limitations.append(
                {
                    "id": f"reviewed_exclusion:{gate_id}:{name}",
                    "summary": str(record.get("reason") or "")[:300],
                    "approved_by": record.get("approved_by"),
                    "adjudication": record.get("adjudication"),
                    "expires_on": record.get("expires_on"),
                }
            )
    return limitations


def _area_support_exclusion_limitations(
    solve: Mapping[str, object],
) -> list[dict[str, object]]:
    """The reviewed area-support exclusions the gate stood on, named by grain.

    An exclusion is gate-only since microcosm#1301 and may excuse a
    constituency as well as a local authority, so the summary names the
    entry's own grain.
    """

    exclusions = solve.get("area_support_exclusions")
    if not isinstance(exclusions, Mapping):
        return []
    limitations: list[dict[str, object]] = []
    for area in exclusions.get("entries_stood_on") or []:
        grain = str(area).partition("/")[0].replace("_", "-") or "local-area"
        limitations.append(
            {
                "id": f"area_support_exclusion:{area}",
                "summary": f"reviewed exclusion of the {grain} support floor.",
            }
        )
    return limitations


def _size_doctrine_record(
    parameters: Mapping[str, object], solve: Mapping[str, object]
) -> dict[str, object]:
    """The coverage receipt's size block: whether and how the file was sized."""

    requested = parameters.get("dataset_households")
    size = solve.get("dataset_size")
    size = size if isinstance(size, Mapping) else {}
    return {
        "sized": requested is not None,
        "clone_count": parameters.get("n_clones"),
        "dataset_households": requested,
        "realized_households": size.get("realized_households"),
        "selection_initial_lambda": parameters.get("selection_initial_lambda"),
        "selection_l0_lambda": size.get("selection_l0_lambda"),
        "search_bracket": size.get("selection_search_bracket"),
    }


def _reviewed_limitations(
    manifest: Mapping[str, object],
    report: Mapping[str, object],
    *,
    full_report: Mapping[str, object],
    incumbent: _IncumbentAttachment | None,
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
    limitations.extend(_area_support_exclusion_limitations(solve))
    cross_grain = _mapping(manifest.get("cross_grain"), "manifest.cross_grain")
    for bridge in cross_grain.get("unbound_bridges") or []:
        if isinstance(bridge, Mapping):
            limitations.append(
                {
                    "id": f"unbound_bridge:{bridge.get('bridge_id')}",
                    "summary": "cross-grain bridge unbound with a receipt; the lower rows bind as published.",
                }
            )
    limitations.extend(_signed_exclusion_limitations(full_report))
    limitations.append(
        _holdout_limitation(_mapping(manifest.get("fit"), "manifest.fit"))
    )
    if incumbent is None:
        limitations.append(
            {
                "id": "no_incumbent_comparison",
                "summary": (
                    "no incumbent comparison is attached (an opt-in "
                    "diagnostic); the release makes no claim against the "
                    "incumbent."
                ),
            }
        )
    else:
        failures = list(incumbent.assessment.get("limit_failures") or [])
        limitations.append(
            {
                "id": "incumbent_surface_diagnostic",
                "summary": (
                    "the incumbent-surface evaluation is attached as a "
                    "diagnostic; its measured limits are recorded, not enforced."
                ),
                "passed": not failures,
                "failure_count": len(failures),
            }
        )
    limitations.append(_engine_measurement_limitation(solve))
    parameters = _mapping(manifest.get("parameters"), "manifest.parameters")
    if parameters.get("dataset_households") is not None:
        size = _size_doctrine_record(parameters, solve)
        limitations.append(
            {
                "id": "size_doctrine",
                "summary": (
                    "the release ships the exact-k selection under the local "
                    "size doctrine, not the whole calibrated pool."
                ),
                **{key: value for key, value in size.items() if key != "sized"},
            }
        )
    return limitations


def _incumbent_attachment(
    args: argparse.Namespace,
    candidate_dir: Path,
    *,
    measured: Mapping[str, str],
    identity: Mapping[str, object],
) -> _IncumbentAttachment | None:
    """Verify the opt-in incumbent comparison, when --incumbent-manifest asks.

    Without the flag nothing is attached, even when the candidate directory
    holds a score or a surface evaluation. With it, both are required and
    hash-joined to the candidate and to the incumbent extraction; identity
    failures refuse, while the measured limits are recorded for the manifest.
    """

    score_path = candidate_dir / "score_vs_incumbent.json"
    surface_path = candidate_dir / "incumbent_surface_evaluation.json"
    if args.incumbent_manifest is None:
        stray = [path.name for path in (score_path, surface_path) if path.is_file()]
        if stray:
            print(
                "note: no --incumbent-manifest, so the incumbent comparison is "
                f"not attached; ignoring {', '.join(stray)}",
                file=sys.stderr,
            )
        return None
    paths = {
        "score receipt": score_path,
        "incumbent surface": surface_path,
        "incumbent manifest": args.incumbent_manifest,
    }
    for label, path in paths.items():
        if not path.is_file():
            raise SystemExit(f"error: {label} missing: {path}")
    attachment_measured = {label: _sha256(path) for label, path in paths.items()}
    score = _load_json(score_path, label="score receipt")
    incumbent = _load_json(args.incumbent_manifest, label="--incumbent-manifest")
    # Hash-join the score to the candidate and to the incumbent extraction it
    # names: a score computed on other bytes cannot ride into the release.
    score_artifacts = _mapping(score.get("artifacts"), "score receipt artifacts")
    _require_equal(
        "score receipt artifacts.candidate_diagnostics.sha256 vs diagnostics bytes",
        _mapping(
            score_artifacts.get("candidate_diagnostics"),
            "score receipt artifacts.candidate_diagnostics",
        ).get("sha256"),
        measured["diagnostics"],
    )
    incumbent_outputs = _mapping(
        incumbent.get("outputs"), "--incumbent-manifest outputs"
    )
    for score_key, output_key in (
        ("incumbent_household_metrics", "metrics"),
        ("incumbent_wide_weights", "weights"),
    ):
        _require_equal(
            f"score receipt artifacts.{score_key}.sha256 vs "
            f"--incumbent-manifest outputs.{output_key}.sha256",
            _mapping(
                score_artifacts.get(score_key), f"score receipt artifacts.{score_key}"
            ).get("sha256"),
            _mapping(
                incumbent_outputs.get(output_key),
                f"--incumbent-manifest outputs.{output_key}",
            ).get("sha256"),
        )
    evaluation = _load_json(surface_path, label="incumbent surface evaluation")
    chronicle = _mapping(
        _mapping(identity.get("targets"), "manifest.identity.targets").get("chronicle"),
        "manifest.identity.targets.chronicle",
    )
    expected_surface_identity = {
        "candidate_dataset_sha256": measured["candidate"],
        "candidate_manifest_sha256": measured["candidate manifest"],
        "candidate_diagnostics_sha256": measured["diagnostics"],
        "ledger_facts_sha256": chronicle.get("facts_sha256"),
        "ledger_manifest_sha256": chronicle.get("manifest_sha256"),
        "incumbent_manifest_sha256": attachment_measured["incumbent manifest"],
        "incumbent_metrics_sha256": _mapping(
            incumbent_outputs.get("metrics"), "incumbent metrics"
        ).get("sha256"),
        "incumbent_weights_sha256": _mapping(
            incumbent_outputs.get("weights"), "incumbent weights"
        ).get("sha256"),
    }
    surface_failures: list[str] = []
    assessment = _check_uk_incumbent_surface_evaluation(
        evaluation,
        surface_failures,
        expected_identity=expected_surface_identity,
        enforce_limits=False,
    )
    if surface_failures:
        raise SystemExit(
            "error: incumbent surface evaluation failed:\n"
            + "\n".join(surface_failures)
        )
    return _IncumbentAttachment(
        manifest_path=args.incumbent_manifest,
        manifest=incumbent,
        score_path=score_path,
        score=score,
        surface_path=surface_path,
        measured=attachment_measured,
        assessment=assessment,
    )


def _incumbent_coverage(incumbent: _IncumbentAttachment | None) -> dict[str, object]:
    """The coverage receipt's incumbent block, with its compared discriminator."""

    if incumbent is None:
        return {"compared": False}
    failures = list(incumbent.assessment.get("limit_failures") or [])
    return {
        "compared": True,
        "snapshot": incumbent.manifest.get("inputs"),
        "period": incumbent.manifest.get("period"),
        "households": incumbent.manifest.get("households"),
        "score_receipt_sha256": incumbent.measured["score receipt"],
        "rows_compared": incumbent.score.get("rows_compared"),
        "incumbent_missing_areas": incumbent.score.get("incumbent_missing_areas"),
        "assessment_passed": not failures,
        "assessment_failure_count": len(failures),
    }


def _area_support_block(solve: Mapping[str, object]) -> dict[str, object]:
    """The areas below the local support floor, as codes only (microcosm#1278).

    The support gate blocks release, so an area of a released file sits below
    a floor only under a reviewed exclusion the gate stood on, and those
    entries name the areas. The areas' support figures stay in the run
    directory's area-support table, which no release ships (microcosm#1301).
    """
    from microcosm.build.uk_runtime.battery_bindings import uk_area_support_floors

    exclusions = _mapping(
        solve.get("area_support_exclusions"), "area_support_exclusions"
    )
    below: dict[str, list[str]] = {}
    for key in exclusions.get("entries_stood_on") or []:
        level, separator, code = str(key).partition("/")
        if not separator or not level or not code:
            raise SystemExit(
                f"error: area support exclusion {key!r} is not '<level>/<code>'"
            )
        below.setdefault(level, []).append(code)
    return {
        "gate": "uk_local_area_support",
        "floors": uk_area_support_floors(),
        "below_floor": {level: sorted(codes) for level, codes in sorted(below.items())},
    }


def _assemble(args: argparse.Namespace) -> dict[str, object]:
    candidate_dir: Path = args.candidate_dir
    manifest = _load_json(
        candidate_dir / "rowwise_candidate_manifest.json",
        label="rowwise_candidate_manifest.json",
    )
    if manifest.get("release_role") != "dense":
        raise SystemExit(
            f"error: manifest.release_role is {manifest.get('release_role')!r}, "
            "not 'dense': this assembler covers the dense line only (a "
            "candidate built before the release role existed is rebuilt)"
        )
    outputs = _mapping(manifest.get("outputs"), "manifest.outputs")
    identity = _mapping(manifest.get("identity"), "manifest.identity")
    # The national assembler's rule: a release carries the staging telemetry
    # receipt of the run it came from, and publication refuses a release
    # whose run intended to stage and delivered nothing.
    raw_staging_delivery = manifest.get("staging_delivery")
    if not isinstance(raw_staging_delivery, Mapping):
        if not args.allow_missing_staging:
            raise SystemExit(
                "error: build record is missing valid staging-delivery evidence "
                "(a run built before the staging lane needs --allow-missing-staging)"
            )
        # An explicit, recorded opt-out mirrors publication's override: the
        # build manifest then says why no telemetry exists, instead of
        # nothing at all. Invalid evidence is still refused.
        manifest = {
            **manifest,
            "staging_delivery": disabled_staging_delivery(
                "assembled with --allow-missing-staging: the run predates "
                "staging telemetry"
            ),
        }
        raw_staging_delivery = manifest["staging_delivery"]
    try:
        validate_staging_delivery(raw_staging_delivery)
    except StagingContractError as error:
        raise SystemExit(
            f"error: invalid staging-delivery evidence: {error}"
        ) from error

    def output_path(key: str) -> Path:
        entry = _mapping(outputs.get(key), f"manifest.outputs.{key}")
        path = Path(str(entry.get("path")))
        return path if path.is_absolute() else candidate_dir / path

    candidate_h5 = output_path("dataset")
    diagnostics_path = output_path("calibration_diagnostics")
    report_path = output_path("local_gate_report")
    full_report_path = output_path("full_gate_report")
    inputs = {
        "candidate": candidate_h5,
        "diagnostics": diagnostics_path,
        "gate report": report_path,
        "full gate report": full_report_path,
        "spine H5": args.spine_h5,
        "candidate manifest": candidate_dir / "rowwise_candidate_manifest.json",
    }
    for label, path in inputs.items():
        if not path.is_file():
            raise SystemExit(f"error: {label} missing: {path}")
    measured = {label: _sha256(path) for label, path in inputs.items()}
    for key, label in (
        ("dataset", "candidate"),
        ("calibration_diagnostics", "diagnostics"),
        ("local_gate_report", "gate report"),
        ("full_gate_report", "full gate report"),
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
    full_report = _load_json(full_report_path, label="full gate report")
    if full_report.get("kind") != "uk_full_gate_report":
        raise SystemExit(
            "error: manifest.outputs.full_gate_report is not a uk_full_gate_report"
        )
    incumbent = _incumbent_attachment(
        args, candidate_dir, measured=measured, identity=identity
    )
    runtime_block = _mapping(identity.get("runtime"), "manifest.identity.runtime")
    runtime = {}
    for package in _RUNTIME_PACKAGES:
        value = runtime_block.get(package)
        if not isinstance(value, str) or not value:
            raise SystemExit(f"error: manifest.identity.runtime.{package} is missing")
        runtime[package] = value
    if _mapping(identity.get("code"), "identity.code").get("git_dirty") is not False:
        raise SystemExit(
            "error: manifest.identity.code.git_dirty must be false for release"
        )
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
            full_report=full_report,
            diagnostics=diagnostics,
            incumbent=incumbent,
            measured=measured,
            attempt_id=attempt_id,
            cut_tag=cut_tag,
            runtime=runtime,
            code_pin=code_pin,
            report_path=report_path,
            logbook_row=logbook_row,
            source_paths={
                "rowwise_candidate_manifest.json": candidate_dir
                / "rowwise_candidate_manifest.json",
                "source_calibration_diagnostics.json": diagnostics_path,
                **(
                    {}
                    if incumbent is None
                    else {
                        "incumbent_surface_evaluation.json": incumbent.surface_path,
                        "incumbent_manifest.json": incumbent.manifest_path,
                        "score_vs_incumbent.json": incumbent.score_path,
                    }
                ),
            },
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
    full_report: Mapping[str, object],
    diagnostics: Mapping[str, object],
    incumbent: _IncumbentAttachment | None,
    measured: Mapping[str, str],
    attempt_id: str,
    cut_tag: str,
    runtime: Mapping[str, str],
    code_pin: str,
    report_path: Path,
    logbook_row: Mapping[str, object],
    source_paths: Mapping[str, Path],
) -> dict[str, object]:
    created_at = datetime.now(UTC).isoformat()
    release_dir = staging_parent / UK_DENSE_RELEASE_ID
    release_dir.mkdir(parents=True)
    for name, source in source_paths.items():
        shutil.copyfile(source, release_dir / name)
    identity = _mapping(manifest.get("identity"), "identity")
    parameters = _mapping(manifest.get("parameters"), "parameters")
    # Validated once more here so the copy into build_manifest.json is the
    # normalized version 2 object, whatever the caller handed over.
    staging_delivery = validate_staging_delivery(
        _mapping(manifest.get("staging_delivery"), "staging_delivery")
    )
    solve = _mapping(manifest.get("solve"), "solve")
    fit = _mapping(manifest.get("fit"), "fit")
    weights = _mapping(manifest.get("weights"), "weights")
    n_targets = len(list(diagnostics.get("targets") or []))
    households = int(solve.get("n_households") or diagnostics.get("n_records") or 0)
    _write_release_diagnostics(
        release_dir / "calibration_diagnostics.json",
        source=source_paths["source_calibration_diagnostics.json"],
        diagnostics=diagnostics,
        measured_source_sha256=measured["diagnostics"],
        households=households,
    )
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
        "reviewed_limitations": _reviewed_limitations(
            manifest, report, full_report=full_report, incumbent=incumbent
        ),
    }
    _write_json(release_dir / "gate_summary.json", gate_summary)
    ladder = _mapping(identity.get("ladder"), "identity.ladder")
    chronicle = _mapping(
        _mapping(identity.get("targets"), "identity.targets").get("chronicle"),
        "identity.targets.chronicle",
    )
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
            "path_name": chronicle.get("path_name"),
            "facts_sha256": chronicle.get("facts_sha256"),
            "manifest_sha256": chronicle.get("manifest_sha256"),
            "fact_row_count": chronicle.get("fact_row_count"),
            "schema_version": chronicle.get("schema_version"),
        },
        "geography_ladder": {
            "sha256": ladder.get("sha256"),
            "bytes": ladder.get("bytes"),
            "layer_vintages": ladder.get("layer_vintages"),
        },
        "incumbent": _incumbent_coverage(incumbent),
        "doctrine": dict(_mapping(parameters.get("doctrine"), "parameters.doctrine")),
        "measure_exclusions": {
            name: dict(rec)
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
            # The names the manifest's reviewed_exclusion limitations restate
            # (microcosm#1301); the contract checks the local target-fit names
            # against the signed report.
            "reviewed_exclusions": _signed_exclusion_names(full_report),
        },
        "holdout": _holdout_coverage(fit),
        "size": _size_doctrine_record(parameters, solve),
        "uprating": dict(
            _mapping(
                manifest.get("census_household_uprating"),
                "census_household_uprating",
            )
        ),
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
    _write_json(release_dir / "uk_source_coverage.json", coverage)
    shutil.copyfile(report_path, release_dir / "uk_local_gates.json")
    # Measured by the candidate build when it recorded it; otherwise unmeasured
    # (None), never asserted clean.
    recorded_dirty = _mapping(identity.get("code"), "manifest.identity.code").get(
        "git_dirty"
    )
    git_dirty = recorded_dirty if isinstance(recorded_dirty, bool) else None
    build_manifest = {
        "build_id": UK_DENSE_RELEASE_ID,
        "build_sha": code_pin[:7],
        "code": {
            "repository": "PolicyEngine/microcosm",
            "git_commit": code_pin,
            "git_dirty": git_dirty,
            "git_dirty_measured": git_dirty is not None,
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
        "staging": dict(staging_delivery),
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
        "build_manifest.json",
    )
    evidence_files = (*evidence_files, *source_paths)
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
        "area_support": _area_support_block(solve),
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
    # The contract warns, without refusing, on a per-area support figure in a
    # shipped document (microcosm#1301); each warning is weighed before
    # publication.
    contract_warnings = validate_release_dir(release_dir)
    for contract_warning in contract_warnings:
        print(f"warning: {contract_warning}", file=sys.stderr)
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
            "--tag-only",
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
        "warnings": contract_warnings,
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
        type=Path,
        help=(
            "optional diagnostic (microcosm#1301): the extractor's "
            "incumbent_local_surface_manifest.json; attaches the candidate "
            "directory's score_vs_incumbent.json and "
            "incumbent_surface_evaluation.json, whose measured limits are "
            "recorded, not enforced"
        ),
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help="where the published H5 is cloned (default: beside the candidate)",
    )
    parser.add_argument("--cut-tag")
    parser.add_argument(
        "--allow-missing-staging",
        action="store_true",
        help=(
            "assemble a run built before staging telemetry existed, recording a "
            "disabled-staging opt-out with this reason in build_manifest.json"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    summary = _assemble(_parse_args(argv))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
