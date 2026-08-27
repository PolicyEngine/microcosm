# Weeksgate: stacked release gates and integer-week provenance

## State

In progress on 2026-08-27. Real-pool provenance has refuted the proposed
post-transfer amount-mapping mechanism: every fractional week is an ACS-origin
non-native clone prediction outside the calibration's clone-0 recipient scope.
The source codec and release-gate architecture fixes are now being designed.
No network access, artifact build, publication, push, pool build, or release
build is in scope.

## Done

- Read `CLAUDE.md` and the GitNexus debugging workflow.
- Confirmed branch `stacked-release-gate-alignment` is clean at `4f453746`.
- Confirmed the local GitNexus CLI is installed but the repository is not yet
  indexed. Its offline analyzer parsed the repository but could not register
  the index because the sandbox forbids writes to `~/.gitnexus`; the generated
  local index was moved out of the worktree to `/private/tmp`.
- Recorded the four requested workstreams: fractional-week provenance and PUF
  misclassification; integer-support calibration repair; stacked/legacy weeks
  gate alignment; and the full release-side gate archaeology sweep.
- Recorded the required verification boundary: repository Ruff plus one pytest
  process per shard, with no pool/release builds.
- Read the fixed-format HDF5 blocks directly and classified all 369 noninteger
  `weeks_unemployed` rows: 360 are ACS clone 1 (355 UC=0, 5 UC>0) and 9 are ACS
  clone 2 (all UC=0); all are positive, all 369 values are distinct, and the
  exact range is 1.0003521955067698--37.796501228614694.
- Confirmed zero nonintegers on ASEC rows and ACS clone 0. The receipted
  calibration covers exactly the 856,626 ACS clone-0 rows, maps 8,419 carrier
  amounts onto observed ASEC support with zero donor-support violations, and
  records QED 0.5882352941176471 to 0.0.
- Reproduced the 5,218-row false "PUF" classification: the legacy role helper
  calls every clone index above zero `puf_tax_detail`, regardless of raw source
  channel. The rows are all ACS-origin clones: 4,733 integer clone-1 rows, 355
  fractional clone-1 rows, 121 integer clone-2 rows, and 9 fractional clone-2
  rows with nonzero weeks while UC is nonpositive.
- Traced the actual fractional mechanism to the ACS transfer target codec:
  PolicyEngine-US declares `weeks_unemployed` as physical `float`, so the
  generic QRF path treats it as continuous even though its reviewed source
  contract is integer-supported. The later calibration repairs clone 0 only.

## Next

- Bind `weeks_unemployed` to observed integer donor support in the actual ACS
  transfer codec and ensure the authority/receipt contract records that policy.
- Modernize the weeks gate to use raw assembled channels when present, scope
  LKWEEKS reconciliation to ASEC-source rows, and scope UC consistency to the
  rows for which the source constraint is defined.
- Complete the full release-gate roster audit and implement only unambiguous
  stacked-awareness repairs.
- Audit all release-side gates, run the complete prescribed verification, and
  write the final evidence and judgment calls to `out.md`.

# Historical: gate-failed base-pool release lane

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
