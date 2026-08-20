# Progress: US reviewed-exclusion machinery

## State

Implementation discovery is in progress on `battery-excl-machinery`, starting
from `origin/main` at `164027e2`. The assigned scope is machinery only: port
the UK schema-2 reviewed-exclusion pattern to the US stacked battery and ship
an empty register. No band, threshold, seed, fold, comparator, or exclusion
approval is in scope.

## Done

- Confirmed the worktree started clean on the named branch.
- Read `CLAUDE.md`, the required adjudication report, and the machine-readable
  adjudication rows for the two reviewed-exclusion candidates.
- Started line-by-line inspection of every cited US mechanism and UK template
  module, plus read-only integration and verification audits.
- Attempted the required `uv sync --all-packages --extra us`. The default uv
  cache is not writable in the managed sandbox; a retry with a private cache
  reached resolution but cannot download `hf-xet` because sandbox DNS is
  disabled. Local-cache recovery is in progress.
- Attempted the GitNexus exploration workflow; no GitNexus MCP resources or
  tools are configured in this session, so discovery is using direct source
  and test call sites.

## Next

- Finish the cited-source and existing-test audit.
- Record the pre-change 1% behavior for both assigned checks.
- Implement and test schema-2 loading, exact leg-plus-criterion matching,
  lifecycle/anti-rot reporting, partial-configuration refusal, and sealed
  terminal-manifest receipts with the committed register empty.
- Run the affected suites, lint, and the required 1% after measurement.

## Historical note

The previous contents described the completed mortgage donor outlier-screen
lane. That merged-session journal was historical on this branch as of
2026-08-19; repository history retains it at parent commit `164027e2`.
