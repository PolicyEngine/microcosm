# UK diagnosis lane notes

## 2026-08-22 — lane start and environment

- Branch: `uk-caseload-counciltax-diagnosis`, starting clean at `2aa96795`
  (`origin/main`), which contains the post-#735 target-surface union and #733
  FRS 2024-25 retarget merge.
- Scope: diagnose the upstream UK data issues #452 and #448 using committed facts,
  receipts, fixtures, parity instruments, and code. This lane will not build a
  pool, alter a gate/band/ceiling/fold/seed, change an owner-only exclusion,
  push, publish, or post the draft comments.
- Environment: `uv sync --all-packages --extra us` could not complete because
  the sandbox denied the global uv cache and outbound DNS. The
  `microcosm-spec-engine` sibling has an identical `uv.lock` and all locked
  dependencies; validation uses its environment with `--no-sync` and this
  worktree's package sources first on `PYTHONPATH`.
- GitNexus: local analysis indexed 626 files / 12,541 nodes / 33,290 edges,
  then registration failed on the sandboxed global registry. Mechanism tracing
  therefore uses direct source and call-site inspection, with every final
  mechanism claim cited to a repository module and line.
- Initial validation: the build shard reached 4,661 passing tests before one
  root-tree provenance guard rejected a retired-package name in this journal.
  The wording now names the upstream issues without naming the archived
  package, and the isolated guard passes (1 passed in 60.68s). The interrupted
  red run is not a commit gate; a clean full rerun follows.
- Clean-suite validation then exposed two unrelated host-latency failures. On
  the first attempt, 6,032 build tests passed and 38 skipped before an atomic
  US-publication crash probe's child import exceeded its fixed 60-second
  timeout. A direct timing showed that CLI import alone took 55.00 seconds, and
  the focused test passed after warm-up. On the second attempt, the same 6,032
  build tests passed and 38 skipped, but a different US import-entry CLI child
  exceeded its 300-second timeout; that focused test then passed in 55 seconds.
  The calibration (203 passed), data (275 passed, 1 skipped), fit (93 passed),
  and frame (294 passed, 36 skipped) shards and Ruff all pass. These are
  environment/process-startup incidents, not UK diagnosis failures, but this
  lane still requires a single green exact full-suite exit before committing.
- Public-source verification: current Scottish, Welsh, and English 2025-26
  council-tax releases and the UC Regulations pension-capital disregard were
  checked on their official sites. GitHub issue/PR pages could not be fetched
  through either the web cache or the sandboxed CLI, so issue-state claims use
  the task brief and committed repository evidence.
- No pool build has run. A read-only, aggregate-only 2025 council-tax
  diagnostic used the locally cached SHA-pinned enhanced-FRS artifact; it did
  not write an artifact, calibrate weights, or expose record-level values.
