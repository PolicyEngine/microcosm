"""Pre-flight for the UK local release-candidate run (microcosm#762).

Two postures, both fail-closed and both printing every failure by name:

``--env``  before the run: the signing key is present and well-formed, the
           four pins are present and the pinned files exist with matching
           digests, and the doctrine constants are the ruled ones.
``--candidate-dir``  after the run: the manifest and the signed gate report
           say what a publishable release candidate must say — release
           posture attested, shippable, every release-blocking gate passed,
           single-block engine, doctrine values, the A15/A17 uprating and the
           measure exclusions receipted, holdout measured, Logbook row
           present, artifact digest recorded and matching.

Exit 0 only when nothing failed. This is the check that stops a 3-hour run
from ending in a manifest that cannot be released for a missing key or flag.
"""

from __future__ import annotations

import argparse
import base64
import glob
import hashlib
import json
import os
import sys
from datetime import date
from pathlib import Path

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
            if len(raw) < 32:
                failures.append(
                    f"{SIGNING_KEY_ENV} decodes to {len(raw)} bytes; need ≥ 32."
                )
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
            UK_LOCAL_MAX_WEIGHT_RATIO,
            UK_LOCAL_SOLVE_DOCTRINE,
            UK_LOCAL_SOLVE_EPOCHS,
        )

        ruled = (10.0, "grain_equal", 15, 1500)
        found = (
            float(UK_LOCAL_MAX_WEIGHT_RATIO),
            UK_LOCAL_SOLVE_DOCTRINE.target_weight_rule,
            int(UK_LOCAL_CLONE_COUNT),
            int(UK_LOCAL_SOLVE_EPOCHS),
        )
        if found != ruled:
            failures.append(f"doctrine constants {found} are not the ruled {ruled}.")
    except ImportError as error:
        failures.append(f"doctrine module not importable: {error}")
    return failures


def check_candidate_dir(candidate_dir: Path, *, today: date | None = None) -> list[str]:
    failures: list[str] = []
    today = today or date.today()
    manifest_path = candidate_dir / "rowwise_candidate_manifest.json"
    if not manifest_path.exists():
        return [f"manifest missing: {manifest_path}"]
    manifest = json.loads(manifest_path.read_text())
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
        "target_weight_rule": "grain_equal",
        "solve_epochs": 1500,
        "clone_count": 15,
    }
    for key, value in expected.items():
        if doctrine.get(key) != value:
            failures.append(
                f"doctrine.{key} = {doctrine.get(key)!r}, expected {value!r}."
            )
    if parameters.get("epochs") != 1500 or parameters.get("n_clones") != 15:
        failures.append(
            f"run parameters epochs={parameters.get('epochs')} n_clones={parameters.get('n_clones')} are not the doctrine's 1500 / 15."
        )
    if parameters.get("skip_holdout"):
        failures.append("holdout was skipped.")
    solve = manifest.get("solve", {})
    resolution = solve.get("measure_resolution", {})
    if resolution.get("blocks") != 1:
        failures.append(
            f"engine resolved in {resolution.get('blocks')} blocks; release posture is 1."
        )
    if solve.get("target_weight_rule_override"):
        failures.append("a target-weight-rule override was applied.")
    holdout = manifest.get("fit", {}).get("rotated_holdout", {})
    if (
        holdout.get("skipped")
        or holdout.get("n_folds") != 5
        or holdout.get("mean_holdout_loss") is None
    ):
        failures.append("rotated holdout is not measured with 5 folds.")
    uprating = manifest.get("ladder_household_uprating", {})
    if uprating.get("applied") is not True:
        failures.append("A15 ladder household uprating not applied.")
    if uprating.get("tenure_cells", {}).get("applied") is not True:
        failures.append("A17 tenure-cell uprating not applied.")
    exclusions = manifest.get("measure_exclusions") or {}
    if not exclusions:
        failures.append("manifest carries no measure_exclusions receipt.")
    for name, record in exclusions.items():
        expires = record.get("expires_on")
        try:
            if expires and (date.fromisoformat(expires) - today).days < 0:
                failures.append(f"measure exclusion {name!r} expired {expires}.")
        except ValueError:
            failures.append(
                f"measure exclusion {name!r} has an unreadable expiry {expires!r}."
            )
    for name in ("spine", "ladder"):
        if manifest.get("identity", {}).get(name, {}).get("pin_verified") is not True:
            failures.append(f"identity.{name}.pin_verified is not true.")
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
        "--release-dir", type=Path, help="validate an assembled release directory"
    )
    args = parser.parse_args(argv)
    failures: list[str] = []
    if args.release_dir is not None:
        from microcosm.data.contract import ReleaseContractError, validate_release_dir

        try:
            validate_release_dir(args.release_dir)
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
    if not args.env and args.candidate_dir is None and args.release_dir is None:
        parser.error(
            "nothing to check: pass --env, --candidate-dir and/or --release-dir"
        )
    for failure in failures:
        print(f"FAIL {failure}", file=sys.stderr)
    print(
        "preflight: " + ("OK" if not failures else f"{len(failures)} failure(s)"),
        file=sys.stderr,
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
