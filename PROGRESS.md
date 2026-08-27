# Progress: historical scorer columns

## State

Final lane state as of 2026-08-26: implementation and complete workspace
verification are green. The real acceptance invocation cleared the original
incumbent formula-column defect and completed all incumbent scoring, then
stopped at a separate authenticated candidate-manifest failure:
late primary-QRF worker binding changed. No scorecard files were written.

The goal is to make head-to-head scoring deterministically remove and receipt
formula-owned columns from loaded historical incumbent or candidate H5
artifacts, while keeping the fresh-release leaf-only export gate strict.

## Done

- Read the repository operating instructions and confirmed the worktree starts
  clean on `scorer-legacy-incumbent-columns`.
- Confirmed the requested defect boundary: historical scorer inputs only; no
  pool build, release build, publishing, push, or validation bypass is in scope.
- Started this committed lane journal before implementation work.
- Reused the release gate's cached, period-sensitive
  `PolicyEngineUSVariableMetadataIndex` classification at the scorer loading
  boundary without changing the builder gate.
- Added fail-closed dependency-closure checks on each dropped output, including
  entity-correct leaf presence, before removing any artifact column.
- Added schema-3 JSON and Markdown receipts with deterministic total count and
  sorted column names per entity; clean artifacts seal an explicit empty
  receipt.
- Preserved tables, weights, strata, mass log, and frame metadata across the
  scorer-only normalization.
- Added the changelog fragment and three requested H5 tests: drop-and-score,
  missing-leaf refusal, and clean empty receipt. The full targeted scorer test
  file passes (16 passed). Its first explicit receipt run exposed and fixed an
  empty-entity pandas drop edge before full-shard verification.
- Passed repository-wide Ruff through the required `uv run --no-sync` command
  using a writable offline cache, and passed CI test-inventory verification for
  all 309 tracked test files.
- Passed all five full pytest shards in separate processes: build 6,516 passed /
  45 skipped; frame 295 / 36; calibrate 203 / 0; data 318 / 2; fit 93 / 0.
  Aggregate: 7,425 passed, 83 skipped, 0 failed.
- Ran the exact acceptance scorer from this worktree. It loaded the live
  incumbent, dropped and receipted one person column
  (has_marketplace_health_coverage), scored all five incumbent chunks and 12
  household slices per chunk, and released that state at 19.20 GiB peak RSS.
- Captured the complete subsequent traceback in out.md. Candidate
  authentication refused the pool manifest before its H5 loaded, so the run
  correctly stopped without bypassing the late-producer binding or emitting a
  partial JSON/Markdown scorecard.

## Next

1. Resolve or rebuild candidate-25 so its sealed primary-PUF-QRF worker binding
   matches the current authenticated late-producer contract.
2. Rerun the unchanged acceptance scorer command; do not weaken or bypass pool
   authentication.

## Historical prior lane

The stacked-pool CD-vintage provenance journal previously in this file is
historical. Its final state remains available at commit `19854a9f`.
# Gate-failed base-pool release lane

## State

Complete on 2026-08-26. Containment, opt-in carriage, and preflight surfacing
are implemented and fully verified. The implementation rejects both redundant
green-pool waivers and any release-manifest receipt that does not exactly match
the pool authenticated by preflight; the preflight's historically required
base/selection inputs and exit semantics are unchanged. No pool or release was
built, no artifact was published, and nothing was pushed.

## Done

- Confirmed the assigned branch and worktree.
- Recorded the v2 charter: close the legacy bare-H5 multispine bypass, add an
  explicit release-build opt-in, carry the authenticated red verdict, and
  surface it in publication preflight without making it an automatic
  publication failure.
- Confirmed that no network, artifact builds, publishing, or pushes are in
  scope.
- Traced the strict manifest loader, current stacked-only terminal-failure
  exception, H5 identity stamp, pool sidecar naming, legacy release arm,
  release manifests, and both preflight output modes.
- Reviewed the salvage branch's final source and test diff line by line. Its
  shared classifier/path-binding/receipt approach closes the bypass without
  changing either loader's contract and was retained with the subsequent
  coherence corrections recorded below.
- Completed the `simulation_ready` / `gate_failed` / loader consumer audit.
  Exact-k remains deliberately strict and head-to-head scoring remains the
  existing authenticated evidence exception.
- Identified report-only downstream caveats: stacked producer metadata still
  names only the k-ladder readiness consumer; red pool publication returns
  status 1 and stops shell chains; ACS-local derivatives keep a donor revision
  but do not project the nested red verdict; generic release consumers tolerate
  and ignore the additive receipt.
- Added a release/preflight-specific authenticated pool loader over the shared
  `require_simulation_ready` seam. The strict simulation-ready and existing
  scoring-only loader contracts remain unchanged.
- Closed the bare-H5 path by detecting either the canonical sibling manifest
  or the H5's stamped pool identity, requiring the sidecar, authenticating the
  publication triple, and binding it to the exact requested H5 path.
- Added `--allow-gate-failed-base-pool` only to the legacy `--base-h5` arm.
  It admits only a current authenticated stacked `gate_failed` pool, rejects a
  green or non-pool use, and never affects the exact-k arm.
- Added the self-contained `base_pool` receipt to both manifests, including
  status/readiness, immutable pool identities, flag use, gates JSON SHA-256,
  failure count/list, and the complete terminal verdict.
- Kept the static preflight inputs mandatory, authenticated its base identically,
  displayed red evidence prominently without changing its exit calculation,
  and required optional release-manifest carriage to match the authenticated
  receipt exactly.
- Hardened carried verdict normalization so nested pass/failure pairs and the
  aggregate verdict must be coherent and a red battery cannot report zero
  failures.
- Passed focused Ruff and the complete builder/H5/preflight test files after
  the containment changes.
- Passed the broader exact-k, launcher, release-contract, and publish-guard
  regression suites.
- Passed repository-wide Ruff and the CI test-group inventory verifier.
- Passed every pytest shard in its own process: build 6,545 passed / 45
  skipped; calibrate 203 passed; data 318 passed / 2 skipped; fit 93 passed;
  frame 295 passed / 36 skipped. Aggregate: 7,454 passed, 83 skipped.
- Confirmed `git diff --check` is clean and that no battery bounds,
  tolerances, plans, or terminal gate logic changed.
- Wrote the complete handoff, consumer audit, manifest schema, verification
  receipts, and commit inventory to `out.md`.

## Next

- Human review and merge of `release-from-gate-failed-pool`.
- Any later artifact operation remains separate: an operator must deliberately
  choose the red-pool flag, then run publication preflight and make the human
  publication decision. This lane performed none of those operations.

## Historical prior lane

The stacked-pool-to-release CD-vintage provenance lane previously maintained
this journal and completed before this work began. It authenticated and
applied household geography after source assembly, carried that authority
through checkpoint and publication identities, published verified CD-vintage
H5 attributes, and reached the unchanged release guard through the shared
fixed/table-aware reader. Its final verification was 7,241 passed, 77 skipped,
with repository-wide Ruff and anti-rot checks green. Full details remain at
commit `2263df36` (the parent of this lane's first journal commit).

The still-earlier PolicyEngine-US 1.819.0 lock-bump lane merged into
`origin/main` at `7b90bb18` on 2026-08-24; its final state remains at commit
`05d254aa` and its detailed receipts remain in the historical section of
`_LANE-NOTES.md`.
