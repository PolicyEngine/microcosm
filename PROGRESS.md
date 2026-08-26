# Gate-failed base-pool release lane

## State

Implementation has started from `origin/main` at `2263df36`. Repository and
agent instructions have been read. The working tree was clean at kickoff.

## Done

- Confirmed the assigned branch and worktree.
- Recorded the v2 charter: close the legacy bare-H5 multispine bypass, add an
  explicit release-build opt-in, carry the authenticated red verdict, and
  surface it in publication preflight without making it an automatic
  publication failure.
- Confirmed that no network, artifact builds, publishing, or pushes are in
  scope.

## Next

- Trace the current release builder, H5 loader seam, pool stamping, preflight,
  and relevant tests.
- Review the untrusted salvage branch line by line as a design reference.
- Implement and test the containment and opt-in behavior.
- Audit other pool-manifest and `simulation_ready` consumers.
- Run Ruff and every pytest shard in its own process, then write `out.md`.

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
