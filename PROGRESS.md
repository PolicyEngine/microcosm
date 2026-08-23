# Progress: 25% replacement-candidate host runbook

## State

**Blocked at the owner's required-input stop.** Current exact-k stage 2 requires
a manifest-pinned Ledger consumer-artifact directory, but the host has only the
bare v9.4 facts JSONL. No matching `manifest.json` or manifest SHA-256 exists.
The lane has deliberately not written a partial `run-candidate.sh`, represented
one as dry-run-valid, or run either builder.

## Done

- Read `CLAUDE.md`, both builders, README's release/alert boundary,
  `tools/preflight_us_release_gates.py`, the exact-k launcher, the current
  scorer, and all argument/loader paths reached before the missing-input stop.
- Read the GitNexus exploration skill selected for the requested release-flow
  audit. This session exposes no GitNexus repository resources or query tools,
  so its documented direct-source fallback was used.
- Completed the prescribed US-extra environment sync offline without changing
  `uv.lock`; all five workspace packages point at this worktree and import.
- Verified all six stage-1 paths and SHA-256 pins copied from the current 1%
  host queue.
- Measured the v9.4 Ledger facts file and established that no corresponding
  schema-v1 artifact manifest is locally available. Recorded the exact stop in
  `experiments/candidate_25pct/input_audit.md` and the non-run receipt in
  `experiments/candidate_25pct/dry_run.md`.
- Independently confirmed three remaining blockers already visible: no
  owner-ratified exact-k seed/`pi_hi`, no current-surface builder-compatible
  incumbent calibration diagnostics, and no local full SCF `p22i6.dta`.
- Preserved the no-build, no-publish, no-promote, no-push, no-tuning,
  under-15-GiB lane boundary. The pending logbook chain was not touched.

## Next

The owner or host-input producer must provide, without changing the current
contract:

1. The exact v9.4 Ledger consumer-artifact directory containing its reviewed
   `manifest.json` and `consumer_facts.jsonl`, with both measured SHA-256 pins.
2. Ratified stage-2 exact-k `seed` and `pi_hi` run-request values.
3. Builder-compatible incumbent `calibration_diagnostics.json` scored on the
   current 32842-row surface and current loss basis, with its SHA-256 and frozen
   target-surface SHA-256.
4. The local full SCF 2022 `p22i6.dta` and measured SHA-256.

After those inputs exist, resume the stage-2 inventory at the Ledger artifact,
continue through SSI bases, reviewed QRF exclusions, all donors/references and
crosswalks, then write and dry-run the host script. Do not infer pins or release
parameters from the July buildp invocation, examples, or tests.
