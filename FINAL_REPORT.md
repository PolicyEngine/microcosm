# Final report: 25% replacement-candidate host runbook

Date: 2026-08-23

Branch: `candidate-25pct-runbook`

Outcome: **blocked at the owner's required-input stop**

## Outcome

No host script was written to
`/Users/maxghenis/PolicyEngine/_buildo-runtime/out/candidate-25/run-candidate.sh`.
Doing so would require invented stage-2 values and pins, contrary to the
binding no-tuning/no-fabrication instruction. No pool build, release build,
publication, promotion, push, scorer run, or logbook-chain write occurred.

The first missing required host input is the manifest for the v9.4 PolicyEngine
Ledger consumer artifact. Current exact-k requires both facts and manifest
SHA-256 pins (`tools/build_us_fiscal_refresh_release.py:1563-1566`) and passes
them to the artifact loader before target compilation
(`tools/build_us_fiscal_refresh_release.py:8358-8371`). The available facts are:

- Path:
  `/Users/maxghenis/PolicyEngine/_buildh-runtime/inputs/consumer_facts_buildn_v9_4.jsonl`
- Size: `131852600` bytes
- SHA-256:
  `b3c0835631a446eb96aa84d86f3ee962d15ca356174c7114db52974f1cacc080`

They are a bare JSONL. No matching
`policyengine_ledger.consumer_artifact.v1/manifest.json` exists in the targeted
host locations, so no exact manifest path or SHA-256 can be supplied. The
loader requires `manifest.json` plus `consumer_facts.jsonl` in an artifact
directory and explicitly rejects a manifest pin for a bare feed
(`packages/microcosm-build/src/microcosm/build/ledger_artifact.py:88-158`).

The complete evidence is in
`experiments/candidate_25pct/input_audit.md`; the committed non-run receipt is
`experiments/candidate_25pct/dry_run.md`.

## What was verified before the stop

All six stage-1 raw inputs exist and match the exact pins copied from the
current 1% host queue. Their exact paths, measured SHA-256 values, and requiring
source lines are tabulated in
`experiments/candidate_25pct/input_audit.md`. The pool builder requires those
six file-and-pin pairs at `tools/build_us_multispine_pool.py:441-520`; it writes
an input-only H5 and leaves calibration downstream
(`tools/build_us_multispine_pool.py:2-16`).

The requested dense and sparse artifacts are two current exact-k stage-2 runs
against the same authenticated pool: `N` resolves to the pool size and takes
the full-pool refit path, while `57240` takes exact-k Sampford selection and
refit (`tools/build_us_fiscal_refresh_release.py:8227-8249,10126-10140,10207-10285`).

Three independent blockers were also established:

1. Exact-k needs explicit stage-2 `seed` and `pi_hi` values
   (`tools/build_us_fiscal_refresh_release.py:1500-1528`), while current-main
   says they have no default and must not be minted
   (`tools/us_bundle_generation/contracts.py:1-12`). The supplied `578` values
   govern stage-1 sampling and clone attachment only.
2. The only located incumbent `calibration_diagnostics.json` has SHA-256
   `870449b44e86b13b25bcea1a57f0e7af37f4d4db18be815eea3acdf9fe6eb40e`,
   but carries the July 5659-target surface
   `49bb0fe3dfd4c399e7b3f900b0e5ba29d9d72413d9170dfc155a9fa5e91c6f6f`
   and no current target-loss basis. The 32842-row committed incumbent JSON is
   a scorecard, not builder diagnostics. Current exact-k requires top-level
   diagnostic rows, exact target-surface equality, and an equal loss basis
   (`tools/build_us_fiscal_refresh_release.py:2676-2736,5862-5885,6666-6803`).
3. Stage 2 always consumes the full SCF donor for auto-loan inputs
   (`tools/build_us_fiscal_refresh_release.py:9572-9587`), but
   `/Users/maxghenis/.cache/microcosm/scf/p22i6.dta` is absent. Current-main has
   no archive or member SHA pin for its fallback fetch
   (`packages/microcosm-build/src/microcosm/build/us_runtime/scf_auto_loans.py:67-77,162-216`).

Per the owner's rule, the exhaustive stage-2 inventory stopped at the first
missing artifact. SSI prior/delivery bases, per-run QRF exclusions, remaining
donors/references, and vintage crosswalks must be audited after the four inputs
above are provisioned; none has been guessed from the July buildp script.

## Timing and memory evidence

Two recent f025 full-duration attempts took 10150.03 seconds (2.819 h) and
10075.75 seconds (2.799 h), but both exited rc=1 and are not green candidate
receipts. A roughly 22-GiB reading describes only the top-level process during
part of a run: the complete supervisors recorded 75.46 and 85.83 GiB. Evidence:

- `/Users/maxghenis/PolicyEngine/_worktrees/microcosm-arm-split/_buildo-runtime/out/stacked-f025-arm-order-r1/build.status.json`
- `/Users/maxghenis/PolicyEngine/_worktrees/microcosm-arm-split/_buildo-runtime/out/stacked-f025-arm-investment-r1/build.status.json`

F1's cold f001 primary-QRF worker profiles recorded 78.91, 84.15, 96.28, and
96.95 GiB. Stage 1 calls the primary-QRF chain
(`tools/build_us_multispine_pool.py:2413-2440`), which launches separately
profiled worker subprocesses
(`packages/microcosm-build/src/microcosm/build/us_runtime/puf_qrf_chain.py:270-349`;
RSS receipts at
`packages/microcosm-build/src/microcosm/build/stage_profile.py:20-45,100-217`).
Therefore the host's 90-GiB reclaimable gate protects **stage 1's QRF
subprocesses**, not stage 2's calibration solve. It is a scheduling threshold,
not proof of margin above the observed 96.95-GiB maximum. There is no current
exact-input f025 stage-2 timing/RSS receipt, so its expectation remains
unmeasured rather than inferred from the pre-migration buildp invocation.

## Scoring commands after inputs and builds exist

These are the audited scorer invocations, but they are not runnable until the
two calibrated stage-2 H5 paths exist. Calibrated H5s are scored directly; the
scorer rejects `--candidate-manifest-sha256` with an H5 and rejects a naked
stage-1 pool H5 (`tools/score_us_release_head_to_head.py:448-515`).

```bash
INCUMBENT_H5=/Users/maxghenis/.cache/huggingface/hub/datasets--policyengine--populace-us/snapshots/26dcad66867687f15735dc4926523e3741920836/populace_us_2024.h5
LEDGER_FACTS=/Users/maxghenis/PolicyEngine/_buildh-runtime/inputs/consumer_facts_buildn_v9_4.jsonl
DENSE_H5=<calibrated-stage2-dense-h5>
SPARSE_H5=<calibrated-stage2-sparse57k-h5>
DENSE_SHA8="$(shasum -a 256 "$DENSE_H5" | awk '{print substr($1,1,8)}')"
SPARSE_SHA8="$(shasum -a 256 "$SPARSE_H5" | awk '{print substr($1,1,8)}')"

.venv/bin/python tools/score_us_release_head_to_head.py \
  --incumbent "$INCUMBENT_H5" \
  --candidate "$DENSE_H5" \
  --ledger-facts "$LEDGER_FACTS" \
  --out-prefix "experiments/replacement_scorecard/head_to_head_dense_48b9d479_${DENSE_SHA8}" \
  --maximum-microsim-batch-size 5000

.venv/bin/python tools/score_us_release_head_to_head.py \
  --incumbent "$INCUMBENT_H5" \
  --candidate "$SPARSE_H5" \
  --ledger-facts "$LEDGER_FACTS" \
  --out-prefix "experiments/replacement_scorecard/head_to_head_sparse_48b9d479_${SPARSE_SHA8}" \
  --maximum-microsim-batch-size 5000
```

The owner reads those two scorecards against
`experiments/replacement_scorecard/incumbent_48b9d479.md`: candidate-minus-
incumbent weighted loss, lower/equal/higher absolute-error counts, family and
basis rollups, worst rows, and candidate battery legs. The scorer intentionally
issues no automated replacement verdict
(`tools/score_us_release_head_to_head.py:1957-1963,2016-2029`).

## Validation and repository state

- `uv sync --all-packages --extra us`: completed offline with the lockfile
  unchanged after recovering from the sandbox-denied default cache and disabled
  DNS; all five workspace packages were relinked and imported.
- `ruff check .`: passed.
- `ruff format --check .`: baseline red on 64 inherited files; this lane changed
  no Python or notebook source and did not reformat unrelated user files.
- `git diff --check`: passed.
- Focused pytest: not completed. Collection remained in the repository's
  schema-registry validation and was interrupted rather than allowing duplicate
  diagnostics to consume host resources. No code was changed by this lane.
- `bash -n`, script `--dry-run`, and release preflight: intentionally not run.
  No honest script exists, the release builder has no native validate-args mode
  (`tools/build_us_fiscal_refresh_release.py:829-1471`), and the separate
  preflight requires legacy base/selection inputs that exact-k rejects
  (`tools/preflight_us_release_gates.py:42-117`;
  `tools/build_us_fiscal_refresh_release.py:1500-1519,1573-1579`).

Commits made before this report:

- `a5d22341` — start the committed lane journals.
- `6000e36c` — record the offline US environment recovery.
- `47c3d00d` — commit the verified input audit and required-input stop.

No push was made.

## Required handoff

Before this lane can resume, provide:

1. The exact v9.4 Ledger consumer-artifact directory and both reviewed hashes.
2. Ratified stage-2 exact-k `seed` and `pi_hi`.
3. Current-32842-surface builder-compatible incumbent calibration diagnostics,
   including the current loss basis, diagnostics SHA, and frozen-surface SHA.
4. A local full SCF 2022 `p22i6.dta` with measured SHA-256.

Then resume the input inventory at the Ledger artifact. Only after every
remaining current-main input is present and pinned should the serial,
off-chain, launchd-safe script and its real dry-run receipt be created.
