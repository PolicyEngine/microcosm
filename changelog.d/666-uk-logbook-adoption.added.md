Adopted the #628 append-only Logbook seam across the three UK build drivers
for #665/#666. `tools/build_uk_national_dataset.py`,
`tools/build_uk_rowwise_dataset.py`, and
`tools/build_uk_rowwise_candidate.py` now append one validated, hash-chained
row for every non-dry terminal invocation, with rows spooled beside the build
outputs under `logbook-spool/` and receipts under build-scoped
`logbook-receipts/<build_id>/` directories. Successes record `iterating`,
candidate refusals and battery blocks record `failed`, unexpected exceptions
record `failed` after writing a pipeline-error receipt, and the named
dev-rung SPI singleton abort keeps its existing `uk_rung_abort_receipt`
payload while recording the Logbook disposition as `discarded`. Dry-runs are
intentionally excluded from Logbook on both success and failure paths.

The UK rows anchor every terminal verdict into a durable local receipt rather
than host-absolute paths: national terminal-gate rows point into the schema-4
gate report, rung aborts point at the named-edge receipt, rowwise crosswalk
and ladder builds point into their manifests, and calibrated candidates point
at the published manifest's post-calibration gate, target-fit, and support
evidence. Candidate post-calibration gate refusals now write a dedicated
`candidate-refusal.json` receipt before re-raising the same `ValueError`, so
the refused artifact is visible without changing the command's exit behavior.

Added `microcosm.build.logbook_adoption`, a driver-agnostic extraction of the
generic US stacked-driver Logbook helpers: attempt state, predecessor
resolution, git code pins, exportable local artifact references, normalized
role-pin digests, atomic JSON receipts, pipeline-error verdict attachment, and
the record-once terminal write wrapper. The US stacked driver deliberately
keeps its local copy in this PR because its large monkeypatch-heavy contract
suite pins those seams; migrating it onto the shared module is the named
follow-up. The Logbook module docstring now points ownership at #665/#666
rather than the stale #616 adoption wording, while the Logbook core behavior
stays unchanged. The UK national pin surface also names the future
`ledger_facts` slot for #622/#623 without inventing a placeholder pin.
Logbook chain configuration (`--logbook-prev-row-digest` and
`POPULACE_LOGBOOK_PREV_ROW_DIGEST`) is validated before any side effect, so a
malformed or conflicting head refuses the run with no row and no destroyed
sidecars, and the candidate driver's recording envelope opens before input
verification so setup failures spool a failed row too.
