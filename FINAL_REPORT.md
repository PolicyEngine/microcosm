# Final report: 25% candidate runbook, round 3

Date: 2026-08-23

Branch: `candidate-25pct-runbook`

Required label: **one-surface + pkg3, legacy release arm, not exact-k
certified**.

Outcome: **STOPPED before launcher construction because a mandatory full-SCF
input is absent. No pool or release build ran.**

## Unified target surface: verified

The legacy arm on current main compiles the same fiscal target surface as
exact-k. PR #741 removed the former membership flags, and current tests require
the parser to reject them
(`packages/microcosm-build/tests/test_us_fiscal_refresh_builder.py:52-93`).
Exact-k and legacy differ while authenticating the base, then converge on the
same unconditional Ledger load and sole
`compile_us_fiscal_target_registry` call
(`tools/build_us_fiscal_refresh_release.py:8213-8258,8358-8371`). They pass the
same `target_specs` through the common materializer
(`tools/build_us_fiscal_refresh_release.py:10070-10108`) before exact-k calls
`calibrate_exact_k_ladder` and legacy dense calls `calibrate` with the same
registry target set (`tools/build_us_fiscal_refresh_release.py:10207-10246,
10300-10316`).

`--dense-default-dataset` does not narrow target membership. It selects the
full-pool `dense_no_l0` calibration/output identity and is incompatible with
exact-k (`tools/build_us_fiscal_refresh_release.py:1568-1571,10132-10140,
10300-10325`). One qualification is required: it also activates dense-arm SSI
delivery fences after calibration
(`tools/build_us_fiscal_refresh_release.py:10448-10462`). The target-surface
gate therefore passed, with the owner-required legacy/non-certified label.

## Ordered input stop

The July 28 invocation was traced from
`/Users/maxghenis/PolicyEngine/_buildo-runtime/scripts/buildp2_dense.sh:63-80`
through every current-main parser and loader. The six stage-1 raw inputs and
all present explicit/default stage-2 inputs were independently hashed. The
full inventory, exact host paths, measured SHA-256 values, and requiring code
lines are committed in
`experiments/candidate_25pct/input_audit_r3.md`.

The first unrecoverable input is:

```text
/Users/maxghenis/.cache/microcosm/scf/p22i6.dta
```

It is absent. The adjacent `scf2022s.zip` is absent too. Current main exposes
`--scf-full-extract` at
`tools/build_us_fiscal_refresh_release.py:1248-1256` and unconditionally
resolves and reads the full extract at
`tools/build_us_fiscal_refresh_release.py:9572-9587`. The default provisioning
loader has neither an archive nor member SHA-256 pin
(`packages/microcosm-build/src/microcosm/build/us_runtime/scf_auto_loans.py:67-77,162-216`).
A download would introduce unreviewed bytes, so none was attempted. No path,
pin, or substitute was fabricated.

The reviewed bare v9.4 Ledger feed is present at
`/Users/maxghenis/PolicyEngine/_buildh-runtime/inputs/consumer_facts_buildn_v9_4.jsonl`
and matches the owner pin
`b3c0835631a446eb96aa84d86f3ee962d15ca356174c7114db52974f1cacc080`.
That resolves rounds 1–2's exact-k artifact-envelope blocker only for this
owner-ruled legacy arm; it does not make this run exact-k certified.

## Zero-waiver disposition

The July QRF register contains nine waivers and was not authorized or reused.
The candidate must omit `--qrf-tail-concentration-exclusions`; the optional
loader maps omission to `{}`
(`tools/build_us_fiscal_refresh_release.py:7790-7803`). No
`--selection-mass-protection` or former membership/exclusion flag is allowed.

This guarantees zero per-run/operator waivers. Current main still applies
checked-in target-parity, release-input-coverage, eCPS-gap, and hard-coded
reviewed exclusions as source-commit semantics; the parser cannot replace
those with an empty register. If the owner's ruling means zero code-owned
exclusions too, current main is independently incompatible. The audit records
the exact package assets, hashes, loaders, and register counts.

## Dense/sparse output mismatch

One owner-ruled `--dense-default-dataset` invocation produces exactly one
`dense_no_l0`, `sparse=False` result and writes only
`<out>/artifacts/populace_us_2024.h5`
(`tools/build_us_fiscal_refresh_release.py:10300-10325,11060-11066`). The sparse
L0/refit H5 requires the separate non-dense branch
(`tools/build_us_fiscal_refresh_release.py:10326-10367`). Therefore the same
legacy release cannot truthfully print and score both dense and sparse
artifacts. A second artifact path or scorer command was not invented.

The scorer also cannot consume
`experiments/replacement_scorecard/incumbent_48b9d479.json` as
`--incumbent`; JSON is treated as a pool manifest. That committed file is
evidence. The actual pinned incumbent H5 exists at
`/Users/maxghenis/.cache/huggingface/hub/datasets--policyengine--populace-us/snapshots/26dcad66867687f15735dc4926523e3741920836/populace_us_2024.h5`
with measured SHA-256
`48b9d479fb4fd1c3537f9383ce4697d130b6f618658409d74f6233c43b994c7e`.
No candidate H5 exists to score.

## Runtime and memory evidence

Same-week f025 pool receipts support about 2.8 hours but contradict a 22 GiB
peak estimate:

- order arm: 2.819 h, 75.46 GiB maximum RSS, nonzero exit;
- investment arm: 2.799 h, 85.85 GiB maximum RSS, nonzero exit.

The receipts are
`/Users/maxghenis/PolicyEngine/_worktrees/microcosm-arm-split/_buildo-runtime/out/stacked-f025-arm-order-r1/build.status.json:44-48`
and the corresponding
`stacked-f025-arm-investment-r1/build.status.json:44-48`.

The July dense legacy attempt ran 1 h 39 m 07 s and exited nonzero
(`/Users/maxghenis/PolicyEngine/_buildo-runtime/logs/buildo-run/chain_densep2.log:16-21`).
Its pressure log peaked at 99,149 MiB (96.83 GiB)
(`/Users/maxghenis/PolicyEngine/_buildo-runtime/logs/buildo-run/pressure_densep2_20260728T065518Z.log:70`),
with measured log SHA-256
`4579df39aed1f4599a9fb242e057715d5e92a191e0328aba208558ddd1ac46f2`.
These historical failed-run figures are evidence, not predictions for the
current unified 32,842-target release. They make 22 GiB indefensible as the
documented peak expectation.

## Ordered disposition and validation

The explicit missing-input rule stopped the round before deliverables 3–4.
Accordingly this round did not:

- create
  `/Users/maxghenis/PolicyEngine/_buildo-runtime/out/candidate-25/run-candidate.sh`
  or the external `candidate-25` directory;
- fabricate partial stage-1, stage-2, release-ID, artifact, or scorer commands;
- run `bash -n` or claim a script `--dry-run` result;
- start either builder;
- publish, promote, push, tune, consume a logbook predecessor, or touch
  `logbook-pending-chain.txt`.

The honest non-run receipt is
`experiments/candidate_25pct/dry_run_r3.md`. `git diff --check` passed for the
audit receipt and final-report changes. No Python tests or linters were run
because this round changed Markdown journals only and the ordered stop forbade
builder validation.

Round-3 commits before this report:

- `7408d579` — start the round-3 legacy-arm audit.
- `6a6c4efe` — verify the unified legacy/exact-k target surface.
- `0f8799c0` — record the complete legacy input stop and non-run receipt.

No push was made.

## Required handoff

Before work may resume, the owner must:

1. supply and pin the exact full-SCF `p22i6.dta` bytes accepted for this legacy
   release arm; and
2. clarify whether the deliverable is one dense legacy candidate, consistent
   with the ruling, or separately authorize a second sparse build.

If “zero waivers” includes code-owned checked-in exclusions, the owner must
also provide a code-level ruling before any launcher is constructed. After the
decisions are recorded, restart input preflight, then construct and commit the
guarded off-chain launcher, run `bash -n`, and execute its real `--dry-run`.
