# Progress: historical scorer columns

## State

The scorer-legacy lane is in progress on 2026-08-26. The branch starts from
`origin/main` at `10bfa17e`; no implementation code has changed yet.

The goal is to make head-to-head scoring deterministically remove and receipt
formula-owned columns from loaded historical incumbent or candidate H5
artifacts, while keeping the fresh-release leaf-only export gate strict.

## Done

- Read the repository operating instructions and confirmed the worktree starts
  clean on `scorer-legacy-incumbent-columns`.
- Confirmed the requested defect boundary: historical scorer inputs only; no
  pool build, release build, publishing, push, or validation bypass is in scope.
- Started this committed lane journal before implementation work.

## Next

1. Trace the scorer load/export flow and the existing metadata-index gate.
2. Add deterministic historical-artifact sanitization, missing-leaf refusal,
   JSON/Markdown receipts, focused tests, and a changelog fragment.
3. Run targeted tests, all test shards, Ruff, and the real acceptance scorer.
4. Record exact evidence in `out.md`, finalize this journal, and leave a clean
   committed worktree.

## Historical prior lane

The stacked-pool CD-vintage provenance journal previously in this file is
historical. Its final state remains available at commit `19854a9f`.
