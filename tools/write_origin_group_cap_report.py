"""Assemble source-only and invented-test evidence for this bounded local build."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = Path("/Users/maxghenis/architecture-reviews/microcosm-20260904")
GATE = Path(
    "/Users/maxghenis/chief-of-staff/state/subfleet/gates/20260906-200530-plan-61907ef5"
)
PLAN_SHA = "986752556045f743b18ef95f5bb70f4db623eb1a2d4e64ae34cf3397384e00ab"
INTEGRATION = "756566551ca931c9c34dfc0a12156bcad8882600"


def git(*args, cwd=ROOT):
    return subprocess.check_output(["git", "-C", str(cwd), *args], text=True).strip()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    plan = REPORTS / "us-clone-parent-calibration-cap-decision-REPORT.md"
    gate = json.loads((GATE / "gate.json").read_text())
    certificate = json.loads((GATE / "certificate.json").read_text())
    assert digest(plan) == PLAN_SHA == gate["subject"]["sha256"]
    assert gate["status"] == "completed" and gate["action"]["status"] == "authorized"
    assert certificate["artifact_revision"]["sha256"] == PLAN_SHA
    assert certificate["peer_verdict"]["verdict"] == "approve"
    assert certificate["main_approval"]["approved"] is True
    head = git("rev-parse", "HEAD")
    files = {}
    for name in git("diff", "--name-only", INTEGRATION, head).splitlines():
        path = ROOT / name
        if path.is_file():
            files[name] = {
                "sha256": digest(path),
                "git_blob": git("rev-parse", f"{head}:{name}"),
            }
    pins = [
        (
            "age_adam",
            Path(
                "/Users/maxghenis/PolicyEngine/_worktrees/microcosm-us-national-age-root-20260906"
            ),
            "a4e69c0a1e7b5e953922d6d0a0d0283890c3c329",
        ),
        (
            "native_origins",
            Path(
                "/Users/maxghenis/PolicyEngine/_worktrees/microcosm-us-native-origin-lineage-20260906"
            ),
            "2c092b69ca2253ca39718bca97beaf5e84c8d4fd",
        ),
        (
            "clone",
            Path(
                "/Users/maxghenis/PolicyEngine/_worktrees/microcosm-us-combined-clone-review-polish-20260906"
            ),
            "1f6ffa7b441701cb2b1254e4bb48241e1e2895de",
        ),
        (
            "integrated_factory",
            REPORTS / "us-survey-demographic-lineage-integration",
            "74f711382029619cdd9f2e8b01330628da0c228b",
        ),
    ]
    accepted = []
    for name, path, expected in pins:
        actual = git("rev-parse", "HEAD", cwd=path)
        status = git("status", "--short", "--untracked-files=no", cwd=path)
        assert actual == expected and not status
        subprocess.run(
            ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", expected, head],
            check=True,
        )
        accepted.append(
            {
                "name": name,
                "path": str(path),
                "head": actual,
                "clean": True,
                "actual_ancestor": True,
            }
        )
    # This exact allowlist contains only evidence created by this local task.
    selections = [
        "us-origin-group-cap-final-core.json",
        "us-origin-group-cap-final-graph.json",
        "us-origin-group-cap-final-admission.json",
        "us-origin-group-cap-final-preflight.json",
        "us-origin-group-cap-final-inventory.json",
        "us-origin-group-cap-final-lint.json",
        "us-origin-group-cap-final-format.json",
        "us-origin-group-cap-spine-001.json",
        "us-origin-group-cap-solver-legacy.json",
        "us-origin-group-cap-solver-benchmark.json",
        "us-origin-group-cap-solver-summary.json",
        "us-origin-group-cap-store-summary.json",
        "us-origin-group-cap-budget-REPORT.json",
        "us-origin-group-cap-observed-replay.json",
    ]
    evidence = {}
    for name in selections:
        path = REPORTS / name
        assert path.is_file(), name
        body = json.loads(path.read_text())
        evidence[name] = {"path": str(path), "sha256": digest(path)}
        log = path.with_suffix(path.suffix + ".log")
        if log.is_file():
            evidence[name]["log_sha256"] = digest(log)
        if name.startswith("us-origin-group-cap-final-"):
            assert body["returncode"] == 0, name
            assert body["child"]["source_integrity_passed"], name
            evidence[name]["head_before"] = body["child"]["head_before"]
            evidence[name]["head_after"] = body["child"]["head_after"]
            evidence[name]["loaded_source_count"] = len(body["child"]["loaded_sources"])
    observed = json.loads(
        (REPORTS / "us-origin-group-cap-observed-replay.json").read_text()
    )
    assert observed["warm"]["calls"] == []
    assert observed["cold"]["keys"] == observed["warm"]["keys"]
    notes = [
        "GitHub fetch was attempted and failed: Could not resolve host github.com. The observed origin/main c1b8324183819cf91511c002bf724a2676524e87 was used; current upstream main is not claimed freshly verified.",
        "Frame-store format v2 was necessary because v1 silently omitted source Frame metadata. Existing v1 Frame artifacts now refuse; no frozen or genuine store was migrated or opened.",
        "No genuine calibration, heldout including B19001/B25003, uncertainty, prior-wage work, source publication, push, deployment, package installation or package update was performed. This is not completed launch calibration.",
        "The first runner admits the complete initial two-role clone only; successor refits/pruning, active zero optimization, partial cloning, conserved-mass and L0/exact-k modes remain unsupported. The lineage checker can verify weight-only successors against the same frozen cap root.",
        "The matching prepatch byte regression uses one Torch intra-op/BLAS thread and the recorded 18 inter-op threads. Other bounded runs use one intra-op/inter-op thread. CPU and wall limits are enforced; macOS RSS limits are advisory.",
        "Preflight scope exception: an initial review-directory listing enumerated entry sizes/timestamps before discovery was narrowed. No data payload was opened. A strict no-metadata-stat claim for that initial listing is therefore not made.",
        "Local checks are the explicit invented test and source inventories listed here; no remote PR CI, wheel build/install lane, or genuine release certification was run.",
    ]
    result = {
        "status": "bounded_local_implementation",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "worktree": str(ROOT),
        "branch": git("branch", "--show-current"),
        "exact_head": head,
        "integration_head": INTEGRATION,
        "accepted_ancestry": accepted,
        "gate": {
            "id": gate["id"],
            "status": gate["status"],
            "round": gate["rounds"][-1]["number"],
            "plan_sha256": PLAN_SHA,
            "gate_sha256": digest(GATE / "gate.json"),
            "certificate_sha256": digest(GATE / "certificate.json"),
        },
        "preserved_plan_revisions": {
            p.name: digest(p)
            for p in sorted(
                REPORTS.glob("us-clone-parent-calibration-cap-decision-reviewed-v*.md")
            )
        },
        "files": files,
        "evidence": evidence,
        "observed_replay": observed,
        "actual_warm_kernel_call_count": len(observed["warm"]["calls"]),
        "typed_identity": [
            "native and expanded original-origin binding artifact types/keys/payloads",
            "full verified population plus live prepared DESIGN anchors",
            "frozen budget source and typed budget artifact",
            "ordered household IDs, exhaustive group map, absolute bounds and constraint digest",
            "typed diagnostic payload and solver options checked against exact final vector",
            "distinct runner-authored successor type/key/payload with persisted byte equality",
        ],
        "negative_tests": [
            "unsupported grouped scalar/prox/L0/exact-k/conserved modes",
            "initial w0 and effective prepared warm over-budget/zero/nonfinite refusal before optimizer",
            "missing/extra/duplicate/stale/fabricated roots/maps/budget/source/member/design/population",
            "cold and REQUIRED-warm solver bypass of each shared inequality",
            "map-only artifact rekeys actual budget/calibration nodes and fails materialized authority",
            "stale typed diagnostic anchors/options on cold and REQUIRED-warm paths",
            "fabricated persisted successor under its expected key refuses REQUIRED admission",
            "nonweight/metadata/membership/source/anchor/independent entity-weight/ownership changes",
            "no repeated-refit incoming-anchor ratchet",
        ],
        "required_ci_inventory": [
            "changes",
            "lint",
            "fast",
            "engine-shared",
            "engine-us",
            "engine-uk",
            "wheels",
            "ci-ok",
        ],
        "remaining_limitations": notes,
    }
    (REPORTS / "us-origin-group-cap-implementation-REPORT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    cold, warm = observed["cold"], observed["warm"]
    text = (
        f"""# Original-origin group cap implementation

Implemented the agreed bounded local invented-data build at `{head}` on `{result["branch"]}`.
Worktree: `{ROOT}`. No launch-calibration completion is claimed.

## Gate and accepted ancestry

The entire canonical plan matches SHA-256 `{PLAN_SHA}`. Durable gate `{gate["id"]}`
is completed with root approval and Fable approval in round 8; its certificate
authorizes proceed on that exact revision. All six saved reviewed revisions and
the four frozen source checkouts remain preserved. The four accepted heads are
actual ancestors of this implementation. Source integration was committed before
solver changes; the companion JSON records every file and evidence hash.

## Implemented behavior

- One immutable original-v4-origin capacity: `min(rD * prepared DESIGN, rI * frozen incoming IMPORTANCE)`, shared across both roles.
- Typed full-materialized origin/budget admission, complete zero-row lineage, strict pre-optimizer input and effective warm-start checks.
- Opt-in positive free-mass Adam with float64 log weights and materialization, differentiable float32 dense/sparse loss casts, stable-ID `math.fsum`, and at most 64 downward common-factor corrections.
- Exact accepted observer bytes become stored weights after mandatory byte-preserving no-op projection, with no final re-exponentiation.
- Mandatory cold/REQUIRED-warm checks of both inequalities, actual typed diagnostic bytes and options, plus distinct persisted weight-only successor lineage.
- All nonweight cells, IDs, memberships, roles, source metadata, strata and original anchors survive. Real REWEIGHT computational ownership is recorded as its declared transition. Only household weights are stored, with final enum CALIBRATED; other entity weights derive from them.

## Observed evidence

The real integrated factory produced {cold["households"]} descendant households
and {cold["persons"]} persons. The cold run executed grouped Adam once and had
{cold["corrected_updates"]} updates requiring a group correction. Actual unequal
role redistribution exceeded the old incoming-row allowance.

A fresh isolated REQUIRED replay executed **{len(warm["calls"])} kernels** with
raising and counting stubs. Every node key, manifest key, budget bytes, diagnostic
counts, final weight bytes and successor bytes/key matched cold execution.
Weights SHA-256: `{cold["weights_sha256"]}`. Budget SHA-256:
`{cold["budget_sha256"]}`. Successor SHA-256: `{cold["successor_sha256"]}`.
The persisted witness contains the loaded-source Git proofs from that fresh process.

The prepatch a4e69c0 invented legacy weights and trajectory were captured before
editing the solver. The matching fixed-runtime regression executed and reproduced
both stored SHA-256 digests exactly; no tolerance or same-process substitute was
used. The single-thread benchmark checked 20,000 invented households across
10,000 groups in a median **5.06 ms** (three measured checks).

The JSON enumerates the final passing test/format/inventory audits and their exact
source/evidence hashes, including negative tests for stale/fabricated artifacts,
both cold/warm solver-bypass inequalities, stale diagnostics, persisted successor
bytes, zero/infeasible starts and budget ratcheting. Source-spine guards passed
498 tests. Full remote CI and publication remain outside this local scope.

## Necessary integration fixes

The real cache lifecycle exposed omitted Frame metadata in the old store. Frame
format v2 preserves it, and v1 frames refuse reuse. Source kernel capabilities
were not relabelled. The successor recognizes the real executor's all-column
ownership transition for REWEIGHT while preserving source evidence and cells.

## Limitations and scope record

"""
        + "\n\n".join(notes)
        + "\n"
    )
    (REPORTS / "us-origin-group-cap-implementation-REPORT.md").write_text(text)
    print(
        json.dumps(
            {
                "exact_head": head,
                "files": len(files),
                "evidence": len(evidence),
                "actual_warm_kernel_call_count": len(warm["calls"]),
            }
        )
    )


if __name__ == "__main__":
    main()
