# Gate-failed base-pool release lane

## State

Design and consumer review are complete. The final salvage tree is a useful
implementation base, but it needs two additional fail-closed checks before it
is trustworthy: reject a redundant gate-failure opt-in on a green pool, and
bind mixed preflight inputs so a release manifest cannot display one pool's
red verdict while static checks authenticate another pool.

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
  changing either loader's contract and is suitable to retain with the two
  coherence corrections above.
- Completed the `simulation_ready` / `gate_failed` / loader consumer audit.
  Exact-k remains deliberately strict and head-to-head scoring remains the
  existing authenticated evidence exception.
- Identified report-only downstream caveats: stacked producer metadata still
  names only the k-ladder readiness consumer; red pool publication returns
  status 1 and stops shell chains; ACS-local derivatives keep a donor revision
  but do not project the nested red verdict; generic release consumers tolerate
  and ignore the additive receipt.

## Next

- Import the re-reviewed salvage implementation as coherent local commits.
- Add redundant-opt-in and mixed-preflight receipt-binding regressions and
  enforce both contracts.
- Run the focused containment, carriage, preflight, exact-k, and data-contract
  suites.
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
