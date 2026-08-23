# QBI ownership lane progress

## State

This commit lands the recovered realized-QRF-regime persistence prerequisite.
It recomputes regimes from frozen donor support, persists the pattern-to-regime
map in transfer/bank/stacked receipts, and validates the origin envelope
fail-closed. Two incomplete recovered fixture repins were repaired, and a fresh
five-shard, Ruff, and whitespace gate is green; exact receipts are recorded in
`_LANE-NOTES.md`.

The four recovered `experiments/qbi_ownership/` evidence files remain
untracked and under audit. No ownership conclusion or terminal-role fix is
included in the prerequisite step. The audited decomposition distinguishes
terminal value provenance from the first failing criterion: all eight terminal
clone-0 cells have `qrf_transfer` provenance; all four incidence checks and the
UBIA QED first fail in transfer, while the BDC, REIT/PTP, and W2 QEDs are
already red at the clone-1 producer and worsen downstream.

No gate, band, ceiling, fold, seed, exclusion, build artifact, or logbook chain
has been changed. This lane has started no pool build; the headless order assigns
all builds to the host queue.

## Done

- Inspected salvage commits `03e23e42`, `21f95b71`, `8942ef97`, and newest
  `321b3185`. The newest snapshot supersedes the others; recovered source,
  tests, and docs match it byte-for-byte. Its four evidence assets are deferred
  from the prerequisite commit.
- Ran the mandated `uv sync --all-packages --extra us` successfully with a
  sandbox-writable uv cache.
- Audited the adjudication's workstream-5, remediation-order, reviewed-
  exclusion, and failed-attempt identity citations before forming an ownership
  view.
- Repaired two incomplete recovered fixtures exposed by the full diagnostic
  gate: the frozen legacy checkpoint materializer binding and the mandatory
  `unmodeled_rows` count in canonical stacked HDF5 receipts.
- Reran the affected tests successfully: exact-k ladder 3/3, stacked HDF5
  loader 38/38, and deterministic trade-entry build 1/1.
- Completed the prerequisite gate: frame 294 passed/36 skipped; fit 93 passed;
  calibrate 201 passed; data 275 passed/1 skipped; build 5,977 passed/39
  skipped; Ruff and `git diff --check` clean. Native/thread pools were bounded
  to one for the final build-shard run after an unbounded run starved four
  unchanged subprocess tests; those four also pass unchanged in isolation.

## Next

1. Make the recovered evidence extractor fail closed and atomic; bind all
   adjudication identities; reproduce every target-by-pattern donor identity;
   mirror the nine production invariants exactly; add the closed attribution
   regression; regenerate the SHA-verified canonical evidence; and commit it.
2. Implement the separate exact-two SSTB terminal-role authority fix only on
   that evidence, regenerate its battery contract, repin authority receipts,
   and commit it behind another full gate.
3. Finalize the coupled-surface refit plan, clearly separating safe lane work
   from the instrumented 1% and 25% host runs, then write `FINAL_REPORT.md` and
   leave both journals current.

The former root entry described the merged mortgage-donor lane and is retained
in Git history through commit `2c7a7218`; it is not current state for this lane.
