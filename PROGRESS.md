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
