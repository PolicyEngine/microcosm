# SIPP equity evidence persistence — completion journal

Session journal for branch `us-sipp-equity-persistence-completion-20260906`,
a fresh dependent child of `b19c287a6e20c589c3619d0e3cc942c93ec6deeb`
(`us-sipp-equity-review-fixes-20260906`). Local only. Nothing pushed, no PR,
no merge, no deploy, no genuine fit, no genuine source open.

This journal is history, not state; check git for current truth.

## Why this branch exists

`us-sipp-nominal-equity-first-fit-fable51-review.md` (REQUEST_CHANGES, narrow)
arrived while the previous corrective task was answering the **root** review.
That task's report, `us-sipp-equity-review-fixes-REPORT.md`, closes the root
findings only. This branch closes Fable's, on top of that work, without
touching it.

## Fable's required changes

1. Coverage, exact `Decimal` evidence and lineage are never persisted: they
   live only in Frame metadata, and `store.py`'s frame writer drops metadata.
   Put coverage and admission in the donor receipt, and lineage plus exact
   weight/target strings in a second typed artifact.
2. The permuted-id cache test is vacuous (fresh store, different directory) and
   changes person-to-household content rather than id order; the changed-source
   test asserts only a refusal, never a key change.
3. Fresh-process warm equivalence, and the module-hash test that asserts only
   digest length.

Non-blocking: dead `selected_ssuid_digest` pin, inert expected counts, contract
identity not in node keys (already closed by the parent), scratch equal to the
store root (already closed), supplied size cast before comparison (already
closed), scratch side effect, dead code in `_copy_object`, stale "sanitized"
error wording, undocumented recipient-column constraint, plus a list of missing
tests worth adding.

## State

- [x] Read the Fable review, the root plan, the v2/v3 addenda, the first-fit
      report and the parent's review-fixes report in full.
- [x] `git status` / `git fetch origin --prune` / upstream inspection; new
      worktree created from `b19c287a` (the requested path is occupied by the
      original, which is left untouched).
- [x] Persist coverage + admission in the donor receipt, identifier-free.
- [x] Second typed artifact: exact `Decimal` target/weight strings and
      household/SSUID lineage, bound in the production graph path.
- [x] Recompute `selected_ssuid_digest` with the producer's own algorithm.
- [x] Expected-versus-observed admission reconciliation that refuses.
- [x] Meaningful same-store permutation and second-valid-source tests.
- [x] Documented recipient-column constraint and honest error wording.
- [x] Full test run, ruff, CI bucket verification, module manifests.

## What changed

`sipp_equity_graph.py` gained `admission_summary`, `reconcile_admission`,
`evidence_document`, `_encode_evidence` and the `donor_evidence`
`ArtifactOutput` on the donor CREATE node in the production builder. The donor
receipt carries the coverage report and an identifier-free admission summary
plus the evidence digest; the private artifact carries the identifiers, the
exact `Decimal` strings and the lineage. The bounded-copy scratch is a
temporary directory rather than a sibling of the source store.

`sipp_december_readback.py` recomputes `selected_ssuid_digest` with the
producer's own `_selected_ssuids` selector and `_key` formula, compares it to
both the packaged pin and the receipt's own field, and refuses a selection
other than all-SSUID. Both refusal types now say honestly what they do and do
not wrap.

`test_us_sipp_equity_graph_run.py`'s two vacuous cache tests are meaningful:
both permutation arms and the second-valid-source arm run against one warm
store. `test_us_sipp_equity_persistence.py` is new. The blindness suite's
module-hash test compares loader bytes to disk instead of a digest length.

## Not done here

No genuine fit, no genuine source open, nothing pushed. `17,588 / 20 / 38`
remain contract pins asserted by test, never claimed as observed. The
`weight_unknown` exclusion count is not separately pinned: the three pinned
counts already exhaust the pinned roster.
