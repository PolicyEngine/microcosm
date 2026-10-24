"""Pre-flight for the UK local release-candidate run (microcosm#762).

Two postures, both fail-closed and both printing every failure by name:

``--env``  before the run: the signing key is present and well-formed, the
           four pins are present and the pinned files exist with matching
           digests, and the doctrine constants are the ruled ones.
``--candidate-dir``  after the run: the manifest and the signed gate report
           say what a publishable release candidate must say — release
           posture attested, shippable, every release-blocking gate passed,
           an exact engine population (one block, or one block per clone
           whose identical copies are scaled to the pool), doctrine values
           (and, for a sized run, the size doctrine: 150,000 households under
           the doctrine's draw, floor and refit penalty, selected by a size
           search in the doctrine's L0 bracket), the A15 census household
           uprating and A17 tenure application and the
           measure exclusions receipted, the rotated holdout either not
           requested (an opt-in diagnostic, microcosm#1301) or measured
           with 5 folds, Logbook row present, artifact digest recorded and
           matching.

Exit 0 only when nothing failed. This is the check that stops a 3-hour run
from ending in a manifest that cannot be released for a missing key or flag.
"""

from __future__ import annotations

import argparse
import base64
import glob
import hashlib
import json
import math
import os
import sys
from datetime import date
from pathlib import Path

from microcosm.data.gate_signing_keys import unregistered_gate_signing_key_problem

SIGNING_KEY_ENV = "MICROCOSM_UK_TERMINAL_GATE_SIGNING_KEY"
PIN_NAMES = ("spine", "ladder", "facts", "manifest")
RELEASE_BLOCKING = "release_blocking"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_env(
    *,
    pins_path: Path,
    spine_h5: Path,
    ladder_npz: Path,
    ledger_dir: Path,
    environ: dict[str, str] | None = None,
    verify_digests: bool = True,
) -> list[str]:
    env = os.environ if environ is None else environ
    failures: list[str] = []
    key = env.get(SIGNING_KEY_ENV, "")
    if not key.strip():
        failures.append(
            f"{SIGNING_KEY_ENV} is not set: the gate report cannot be signed."
        )
    else:
        try:
            raw = base64.b64decode(key.strip(), validate=True)
            if len(raw) != 32:
                failures.append(
                    f"{SIGNING_KEY_ENV} decodes to {len(raw)} bytes; the gate "
                    "battery signs only with exactly 32."
                )
            else:
                problem = unregistered_gate_signing_key_problem(
                    "uk", raw, purpose="sign"
                )
                if problem is not None:
                    failures.append(f"{SIGNING_KEY_ENV}: {problem}")
        except Exception:
            failures.append(f"{SIGNING_KEY_ENV} is not valid base64.")
    if not pins_path.exists():
        failures.append(f"pins file missing: {pins_path}")
        return failures
    pins: dict[str, str] = {}
    for token in pins_path.read_text().split():
        if "=" in token:
            name, value = token.split("=", 1)
            pins[name] = value.strip()
    for name in PIN_NAMES:
        if not pins.get(name):
            failures.append(f"pin {name!r} missing from {pins_path.name}")
    targets = {
        "spine": spine_h5,
        "ladder": ladder_npz,
        "facts": ledger_dir / "consumer_facts.jsonl",
        "manifest": ledger_dir / "manifest.json",
    }
    for name, path in targets.items():
        if not path.exists():
            failures.append(f"pinned input missing on disk: {name} -> {path}")
        elif verify_digests and pins.get(name) and _sha256(path) != pins[name]:
            failures.append(f"pinned input digest mismatch: {name} -> {path}")
    try:
        from microcosm.build.uk_runtime.local_doctrine import (
            UK_LOCAL_CLONE_COUNT,
            UK_LOCAL_DATASET_HOUSEHOLDS,
            UK_LOCAL_L0_SEARCH_FLOOR,
            UK_LOCAL_MAX_WEIGHT_RATIO,
            UK_LOCAL_SOLVE_DOCTRINE,
            UK_LOCAL_SOLVE_EPOCHS,
        )

        ruled = (
            10.0,
            "nation_grain_family_equal_sqrt_count",
            15,
            2000,
            SIZE_DOCTRINE["dataset_households"],
            SIZE_SEARCH_BRACKET["lower"],
        )
        found = (
            float(UK_LOCAL_MAX_WEIGHT_RATIO),
            UK_LOCAL_SOLVE_DOCTRINE.target_weight_rule,
            int(UK_LOCAL_CLONE_COUNT),
            int(UK_LOCAL_SOLVE_EPOCHS),
            int(UK_LOCAL_DATASET_HOUSEHOLDS),
            float(UK_LOCAL_L0_SEARCH_FLOOR),
        )
        if found != ruled:
            failures.append(f"doctrine constants {found} are not the ruled {ruled}.")
    except ImportError as error:
        failures.append(f"doctrine module not importable: {error}")
    return failures


#: The local size doctrine (microcosm#1124, ruling of 2026-10-13, as amended
#: for microcosm#1301): a sized release candidate ships this many households,
#: drawn and refit under these settings, with no selection L2 penalty
#: (refused for every run above), selected by a size search over this L0
#: bracket. Restated here, like the solve doctrine, so the pre-flight checks
#: the run against the ruling rather than against the code that ran it.
SIZE_DOCTRINE = {
    "dataset_households": 150_000,
    "selection_pi_hi": 0.5,
    "baseline_pi_floor": 0.5,
}
REFIT_L2_DOCTRINE = {
    "lambda": 0.0215748772773509,
    "anchor": "uniform",
    "basis": "chi_square",
}
SIZE_SEARCH_BRACKET = {"lower": 1e-10, "upper": 10.0}


def _size_failures(manifest: dict, parameters: dict) -> list[str]:
    """A sized run must carry the size doctrine; an unsized one, no refit L2."""

    requested = parameters.get("dataset_households")
    if requested is None:
        if parameters.get("refit_l2") is not None:
            return ["a refit L2 penalty was applied without a size request."]
        return []
    failures = [
        f"parameters.{key} = {parameters.get(key)!r}, expected the size "
        f"doctrine's {value!r}."
        for key, value in SIZE_DOCTRINE.items()
        if parameters.get(key) != value
    ]
    if parameters.get("refit_l2") != REFIT_L2_DOCTRINE:
        failures.append(
            f"parameters.refit_l2 = {parameters.get('refit_l2')!r}, expected the "
            f"size doctrine's {REFIT_L2_DOCTRINE!r}."
        )
    if parameters.get("resume_size_checkpoint") is not None:
        failures.append(
            "the selection was resumed from a size checkpoint; a release draws its own."
        )
    size = manifest.get("solve", {}).get("dataset_size")
    if not isinstance(size, dict):
        failures.append("solve.dataset_size is missing from a sized run.")
    elif not (
        size.get("requested_households") == size.get("realized_households") == requested
    ):
        failures.append(
            "solve.dataset_size requested "
            f"{size.get('requested_households')!r} / realized "
            f"{size.get('realized_households')!r}, expected {requested!r}."
        )
    if isinstance(size, dict) and (
        size.get("selection_search_bracket") != SIZE_SEARCH_BRACKET
    ):
        failures.append(
            "solve.dataset_size.selection_search_bracket = "
            f"{size.get('selection_search_bracket')!r}, expected the size "
            f"doctrine's {SIZE_SEARCH_BRACKET!r}."
        )
    if manifest.get("release_posture", {}).get("size_doctrine") is not True:
        failures.append("release_posture.size_doctrine is not true.")
    return failures


def _holdout_failures(manifest: dict, parameters: dict) -> list[str]:
    """The rotated holdout is an opt-in diagnostic (microcosm#1301).

    A run that did not request it records the skip marker and claims no
    out-of-sample fit; a run that did must carry the five measured folds and
    a finite mean. The record must agree with the request either way.
    """

    holdout = (manifest.get("fit") or {}).get("rotated_holdout")
    if not isinstance(holdout, dict):
        return ["fit.rotated_holdout is missing: neither measured nor skipped."]
    requested_skip = parameters.get("skip_holdout")
    if requested_skip not in (True, False):
        return [
            f"parameters.skip_holdout = {requested_skip!r}; the run must record "
            "whether the holdout was requested."
        ]
    if holdout.get("skipped") is True:
        if requested_skip is not True:
            return [
                "the rotated holdout was requested (--rotated-holdout) but the "
                "run recorded it as skipped."
            ]
        return []
    if requested_skip is True:
        return [
            "the rotated holdout was not requested but the run recorded a "
            "measured holdout."
        ]
    mean = holdout.get("mean_holdout_loss")
    if (
        holdout.get("n_folds") != 5
        or isinstance(mean, bool)
        or not isinstance(mean, (int, float))
        or not math.isfinite(mean)
    ):
        return ["rotated holdout is not measured with 5 folds and a finite mean."]
    return []


def check_candidate_dir(candidate_dir: Path, *, today: date | None = None) -> list[str]:
    failures: list[str] = []
    today = today or date.today()
    manifest_path = candidate_dir / "rowwise_candidate_manifest.json"
    if not manifest_path.exists():
        return [f"manifest missing: {manifest_path}"]
    manifest = json.loads(manifest_path.read_text())
    # The manifest declares its release role (microcosm#823); this pre-flight
    # is the dense line's and refuses every other role, including a
    # pre-role candidate, which is rebuilt rather than grandfathered.
    if manifest.get("release_role") != "dense":
        return [
            f"manifest.release_role is {manifest.get('release_role')!r}, not "
            "'dense': this pre-flight covers the dense line only."
        ]
    parameters = manifest.get("parameters", {})
    if parameters.get("release_candidate") is not True:
        failures.append("parameters.release_candidate is not true.")
    if manifest.get("releasable") is not True:
        failures.append("manifest.releasable is not true.")
    if manifest.get("blocked_at_f100"):
        failures.append(f"blocked at f100: {manifest.get('blocking_failures')}")
    doctrine = parameters.get("doctrine", {})
    expected = {
        "max_weight_ratio": 10.0,
        "target_weight_rule": "nation_grain_family_equal_sqrt_count",
        "solve_epochs": 2000,
        "clone_count": 15,
    }
    for key, value in expected.items():
        if doctrine.get(key) != value:
            failures.append(
                f"doctrine.{key} = {doctrine.get(key)!r}, expected {value!r}."
            )
    if parameters.get("epochs") != 2000 or parameters.get("n_clones") != 15:
        failures.append(
            f"run parameters epochs={parameters.get('epochs')} n_clones={parameters.get('n_clones')} are not the doctrine's 2000 / 15."
        )
    if parameters.get("selection_l2") is not None:
        failures.append("a selection L2 penalty was applied (candidate-only).")
    failures.extend(_size_failures(manifest, parameters))
    solve = manifest.get("solve", {})
    resolution = solve.get("measure_resolution", {})
    blocks = resolution.get("blocks")
    representation = resolution.get("engine_population_representation") or {}
    # One block per clone measures the pool exactly when the blocks are
    # identical copies scaled to the pool (rowwise_cli.release_verdict); the
    # build records that check, and nothing else stands in for it.
    per_clone_exact = (
        type(blocks) is int
        and blocks > 1
        and blocks == parameters.get("n_clones")
        and representation.get("blocks") == blocks
        and representation.get("mode") == "block_weights_scaled_to_pool"
        and representation.get("exact") is True
    )
    if blocks != 1 and not per_clone_exact:
        failures.append(
            f"engine resolved in {blocks} blocks without an exact pool "
            "representation; release posture is one block, or one block per "
            "clone recorded exact."
        )
    if solve.get("target_weight_rule_override"):
        failures.append("a target-weight-rule override was applied.")
    failures.extend(_holdout_failures(manifest, parameters))
    uprating = manifest.get("census_household_uprating", {})
    if uprating.get("applied") is not True:
        failures.append("A15 census household uprating not applied.")
    grains = uprating.get("grains")
    if not isinstance(grains, dict):
        failures.append("A15 census household uprating grains are missing.")
    else:
        for grain in ("constituency", "local_authority"):
            factor = (grains.get(grain) or {}).get("factor")
            if (
                not isinstance(factor, (int, float))
                or isinstance(factor, bool)
                or not math.isfinite(factor)
                or factor <= 0
            ):
                failures.append(
                    "A15 census household uprating "
                    f"grains.{grain}.factor is not positive and finite."
                )
    for label, cells in (
        ("A15 household", uprating.get("household_cells", {})),
        ("A17 tenure", uprating.get("tenure_cells", {})),
    ):
        if (
            cells.get("applied") is not True
            or not isinstance(cells.get("cells"), int)
            or cells.get("cells", 0) <= 0
        ):
            failures.append(f"{label}-cell uprating not applied or unmeasured.")
        holds = cells.get("holds")
        if not isinstance(holds, list) or not holds:
            failures.append(f"{label}-cell hold receipt is missing.")
            continue
        for key, field in (
            ("cells", "applied"),
            ("attempted_cells", "attempted"),
            ("eligible_cells", "eligible"),
            ("skipped_cells", "skipped"),
        ):
            if cells.get(key) != sum(r.get(field) is True for r in holds):
                failures.append(f"{label}-cell {key} does not close over its holds.")
        if cells.get("total_cells") != len(holds):
            failures.append(f"{label}-cell total_cells does not close over its holds.")
        if any(
            r.get("attempted") is True and r.get("applied") is not True for r in holds
        ):
            failures.append(
                f"{label} contains skipped attempted holds; "
                "review vintage/reason evidence."
            )
    from microcosm.data.contract import _check_uk_measure_exclusions

    exclusions = manifest.get("measure_exclusions") or {}
    _check_uk_measure_exclusions(exclusions, failures, today=today)
    for name in ("spine", "ladder"):
        if manifest.get("identity", {}).get(name, {}).get("pin_verified") is not True:
            failures.append(f"identity.{name}.pin_verified is not true.")
    # The release line assigns geography with the identity-keyed atomic law
    # on the registered supports (microcosm#932): the manifest's binding must
    # say so, and its support pins must be the sources.yaml rows, not merely
    # the digests the operator pinned.
    assignment = (manifest.get("geography") or {}).get("assignment") or {}
    if assignment.get("assignment") != "atomic":
        failures.append(
            f"geography.assignment.assignment is {assignment.get('assignment')!r}, "
            "not 'atomic'."
        )
    else:
        from microcosm.build.uk_runtime import country_adapter

        register = country_adapter.uk_atomic_support_register()
        support_pins = assignment.get("support_pins") or {}
        for system, expected in register.items():
            pin = support_pins.get(system) or {}
            if (
                pin.get("sha256") != expected["sha256"]
                or pin.get("size_bytes") != expected["size_bytes"]
            ):
                failures.append(
                    f"geography.assignment.support_pins.{system} is not the "
                    "registered support (sources.yaml)."
                )
    dataset = manifest.get("outputs", {}).get("dataset", {})
    h5_path = Path(str(dataset.get("path", "")))
    if not h5_path.is_absolute():
        h5_path = candidate_dir / h5_path
    if not h5_path.exists():
        failures.append(f"artifact missing: {h5_path}")
    elif dataset.get("sha256") and _sha256(h5_path) != dataset["sha256"]:
        failures.append("artifact digest does not match outputs.dataset.sha256.")
    report_paths = glob.glob(str(candidate_dir / "*local_gates.json"))
    if not report_paths:
        failures.append("gate report missing.")
    else:
        report = json.loads(Path(report_paths[0]).read_text())
        if report.get("release_candidate") is not True:
            failures.append(
                "gate report release_candidate is not true (battery ran in dev posture)."
            )
        if report.get("shippable") is not True:
            failures.append("gate report shippable is not true.")
        if report.get("blocked_at_phase") is not None:
            failures.append(
                f"gate report blocked at phase {report.get('blocked_at_phase')!r}."
            )
        attestation = report.get("attestation") or {}
        if not any(
            key in report or key in attestation
            for key in ("signature", "hmac", "signed_by", "signature_sha256")
        ):
            failures.append("gate report carries no signature field.")
        for gate_id, entry in (report.get("gates") or {}).items():
            if (
                entry.get("criticality") == RELEASE_BLOCKING
                and entry.get("status") != "passed"
            ):
                failures.append(
                    f"release-blocking gate {gate_id} status {entry.get('status')!r}."
                )
        # Authenticate the report exactly as the release contract will, so a
        # signing defect surfaces right after the run rather than at assembly.
        try:
            from microcosm.data.contract import _check_uk_dense_gate_report

            contract_failures: list[str] = []
            _check_uk_dense_gate_report(report, failures=contract_failures)
            failures += [f"contract: {line}" for line in contract_failures]
        except ImportError:
            failures.append(
                "microcosm.data.contract is not importable; the report cannot be verified."
            )
    if not glob.glob(str(candidate_dir / "logbook-spool" / "*.json")):
        failures.append("no Logbook row in logbook-spool/.")
    return failures


#: The opt-in incumbent comparison's files in a run directory, which the
#: assembler ships byte for byte when ``--incumbent-manifest`` attaches them.
INCUMBENT_COMPARISON_FILES = (
    "score_vs_incumbent.json",
    "incumbent_surface_evaluation.json",
)


def candidate_disclosure_warnings(
    candidate_dir: Path, *, incumbent_manifest: Path | None = None
) -> list[str]:
    """The release's disclosure walk over a run's to-be-shipped JSON (microcosm#1301).

    The assembler ships the candidate manifest, the original diagnostics and
    the signed local gate report byte for byte, and, when the incumbent
    comparison is attached, the score, the surface evaluation and the
    extraction manifest (``incumbent_manifest``). So a per-area support figure
    in one of them, or a false match, shows here after the build rather than
    first at assembly. The comparison's files are walked whenever the run
    directory holds them. Each file is read and walked as the contract does
    (:func:`microcosm.data.contract._uk_dense_disclosure_file_warnings`), so a
    file with NaN or a nesting too deep to walk is reported rather than
    skipped or crashing the pre-flight. Like the contract, it warns and never
    fails.
    """

    from microcosm.data.contract import _uk_dense_disclosure_file_warnings

    manifest_path = candidate_dir / "rowwise_candidate_manifest.json"
    try:
        outputs = json.loads(manifest_path.read_text()).get("outputs") or {}
    except (OSError, ValueError, AttributeError, RecursionError):
        # check_candidate_dir reports a missing or unreadable manifest, and
        # the walk below reports one it cannot walk.
        outputs = {}
    if not isinstance(outputs, dict):
        outputs = {}

    def shipped(key: str, pattern: str) -> Path | None:
        # The manifest's own output path, as the assembler reads it.
        entry = outputs.get(key)
        if isinstance(entry, dict) and entry.get("path"):
            path = Path(str(entry["path"]))
            return path if path.is_absolute() else candidate_dir / path
        matches = sorted(glob.glob(str(candidate_dir / pattern)))
        return Path(matches[0]) if matches else None

    paths = [
        manifest_path,
        shipped("calibration_diagnostics", "*.diagnostics.json"),
        shipped("local_gate_report", "*local_gates.json"),
        *(candidate_dir / name for name in INCUMBENT_COMPARISON_FILES),
        incumbent_manifest,
    ]
    warnings: list[str] = []
    for path in paths:
        if path is not None and path.is_file():
            warnings += _uk_dense_disclosure_file_warnings(path)
    return warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--env", action="store_true", help="check the launch environment"
    )
    parser.add_argument("--pins", type=Path)
    parser.add_argument("--spine-h5", type=Path)
    parser.add_argument("--ladder", type=Path)
    parser.add_argument("--ledger-facts", type=Path)
    parser.add_argument("--skip-digests", action="store_true")
    parser.add_argument("--candidate-dir", type=Path, help="check a finished run")
    parser.add_argument(
        "--incumbent-manifest",
        type=Path,
        help=(
            "with --candidate-dir, the incumbent extraction manifest assembly "
            "will attach; the disclosure walk reads it too"
        ),
    )
    parser.add_argument(
        "--release-dir", type=Path, help="validate an assembled release directory"
    )
    args = parser.parse_args(argv)
    if args.incumbent_manifest is not None and args.candidate_dir is None:
        parser.error("--incumbent-manifest needs --candidate-dir")
    failures: list[str] = []
    # The contract's warnings (the disclosure walk, microcosm#1301) are
    # printed but never fail the pre-flight.
    warnings: list[str] = []
    if args.release_dir is not None:
        from microcosm.data.contract import ReleaseContractError, validate_release_dir

        try:
            warnings += validate_release_dir(args.release_dir)
        except ReleaseContractError as error:
            failures += [str(line) for line in getattr(error, "failures", [str(error)])]
    if args.env:
        missing = [
            n
            for n, v in (
                ("--pins", args.pins),
                ("--spine-h5", args.spine_h5),
                ("--ladder", args.ladder),
                ("--ledger-facts", args.ledger_facts),
            )
            if v is None
        ]
        if missing:
            parser.error(f"--env needs {', '.join(missing)}")
        failures += check_env(
            pins_path=args.pins,
            spine_h5=args.spine_h5,
            ladder_npz=args.ladder,
            ledger_dir=args.ledger_facts,
            verify_digests=not args.skip_digests,
        )
    if args.candidate_dir is not None:
        failures += check_candidate_dir(args.candidate_dir)
        if (
            args.incumbent_manifest is not None
            and not args.incumbent_manifest.is_file()
        ):
            failures.append(f"incumbent manifest missing: {args.incumbent_manifest}")
        warnings += candidate_disclosure_warnings(
            args.candidate_dir, incumbent_manifest=args.incumbent_manifest
        )
    if not args.env and args.candidate_dir is None and args.release_dir is None:
        parser.error(
            "nothing to check: pass --env, --candidate-dir and/or --release-dir"
        )
    for warning in warnings:
        print(f"WARN {warning}", file=sys.stderr)
    for failure in failures:
        print(f"FAIL {failure}", file=sys.stderr)
    print(
        "preflight: "
        + ("OK" if not failures else f"{len(failures)} failure(s)")
        + (f", {len(warnings)} warning(s)" if warnings else ""),
        file=sys.stderr,
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
