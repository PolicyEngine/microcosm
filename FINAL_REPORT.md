# Final report: package 3 ACS QRF receipt scoping

## Outcome

Fixed the host 1% verification failure at
`person/puf_tax_itemization/taxable_interest_income` without opting that
unassigned target into the package 3 calibration. The final runtime tree is
identical to the complete `microcosm-build` suite checkpoint, and all 529 tests
in the five directly affected files plus repository lint pass on the final
tree. This continuation made no duplicate executable edit: independent source,
history, and regression audits confirmed that the complete correction and its
failing-target coverage were already committed on the branch.

No after artifact is claimed or accepted. An externally owned exact 1% retry
previously rebuilt all 47 survey targets plus the one housing target without
the supplied traceback and then entered the late DAG. That mutable log was
truncated to zero bytes during the final local verification, while the external
guard continued to report a resource wait and the directory still contained no
pool, manifest, gates artifact, or terminal marker. Terminal stacked receipt
validation and frozen-battery success therefore are not claimed. The frozen
sample/clone seed remains 578, and no battery band, threshold, comparator,
fold, publication boundary, or pending-chain state was changed.

The current 2026-08-20 owner continuation independently re-traced the old raise
site, current producer/receipt/validator selectors, failing-to-fixed history,
and exact regression surface. It found no remaining canonical path that can
opt `taxable_interest_income` into QRF regime work or evidence. The final test
tree also rejects a fully rehashed, plausible in-range `__batch_1` alias on an
assigned target, preventing the unsafe permissive workaround briefly present
in the first correction. All 14 decisive focused cases and all 529 tests in the
five directly affected files pass in this continuation. Independent source,
history, and regression audits agree on the scoped-fix verdict.

## Root cause and correction

Commit `33bf52fe` enabled QRF regime detection, verification, and receipt
provenance for every ordinary and banked ACS transfer target. That widened the
behavioral and receipt surface beyond the nine owner-declared calibration
targets. The 15-target `puf_tax_itemization` family is split at the certified
eight-target fit width; its real transfer record therefore used a bounded
`__batch_1` family while terminal validation tried to bind it to the canonical
unsplit family. The resulting mismatch raised the reported “ACS QRF pattern
record binding is invalid” error on an unassigned target.

The correction keeps regime work explicitly opt-in:

- `transfer_acs_inputs` defaults `regime_evidence_targets` to empty and scopes
  regime detection, fitted-result verification, bank-chain verification, and
  pattern provenance to exact `(entity, target)` selections.
- The stacked early and late owners derive those selections from the nine
  immutable calibration specifications on the current transfer surface: two
  early and seven late. The 15-target `puf_tax_itemization` family selects
  none.
- Unassigned records carry no regimes, unassigned receipts carry no QRF
  evidence, and generic serializers omit the empty opt-in field so their
  legacy JSON shape is unchanged.
- Validators reject QRF evidence on undeclared targets and require exact
  selected record-family binding. All canonical target receipts—selected or
  not—must also carry the complete, internally consistent four-field legacy
  transfer count block.
- Complete selected siblings are retained only when the whole bounded family
  is selected. A mixed selected/unselected family neither expands the selected
  fit nor changes the unassigned draw.
- Canonical production accepts only the certified eight-target fit width,
  matching the existing late-DAG boundary; narrower widths remain a
  non-production test seam.

The generic `transfer_acs_inputs` library API deliberately permits an explicit
caller to request regime auditing for any target on that caller's requested
surface. The canonical stacked entry points do not expose that choice: they
derive it internally from the nine immutable specs. “Assigned-only behavior”
therefore describes canonical generated values, regime work, provenance, and
receipts. The shared four-count transfer accounting invariant still validates
every canonical target so moving QRF validation behind the assignment branch
does not weaken legacy receipt checks. The in-memory `AcsTransferPattern` type
has an empty `target_regimes` field for default/unselected calls; repository
serializers deliberately omit that empty opt-in field, so persisted generic
provenance and canonical receipts retain their legacy shape.

## Regression coverage

The regression surface includes:

- the exact host target in a real wide, banked `puf_tax_itemization` family,
  proving its `__batch_1` record has no regimes or QRF receipt while selected
  unemployment compensation retains both;
- exact selected-family binding—including a plausible in-range `__batch_1`
  alias—regime tampering, and missing or inconsistent early/late transfer
  counts;
- mixed selected/unselected output equivalence and selected-sibling behavior;
- default ordinary and banked transfer behavior plus both generic serializers;
- rejection of non-default canonical fit widths; and
- canonical pool/H5 receipt fixtures using the strict four-count schema.

The real banked producer case and the canonical terminal-validator case are
separate fixtures: the first proves the runtime emits no evidence for the real
bounded record, while the second proves canonical validation rejects forged
evidence and accepts the evidence-free legacy receipt. Together they cover the
reported producer/validator boundary without granting a test authority receipt
canonical production authority.

The full-suite run found one stale synthetic H5 fixture that supplied only
`residual_null_rows`. The fixture—not production validation—was corrected to
four consistent zero counts, its complete test file reran green, and an
independent fixture scan found no other canonical partial-count fixtures. A
separate final scope audit found no residual behavior or receipt leak; it also
confirmed that the exact host family selects no regime-evidence targets.

## Verification

Verification used the already-synced `microcosm-707` exact-lock environment
with `PYTHONPATH` pinned to this worktree. Both worktrees have the same
`uv.lock` SHA-1. The default uv cache remains sandbox-inaccessible, and the
earlier writable-cache sync could not reach PyPI because sandbox DNS is
unavailable.

- All 225 `packages/microcosm-build/tests/test_*.py` files passed across fresh
  pytest processes at `a5be536f`. No runtime source, tool, spec, project, or
  lockfile changed between that commit and the final audit tree.
- The five directly affected transfer, multispine, stacked, pool-tool, and H5
  files were rerun together on the final tree: all 529 passed. The run used the
  owner-provided 12 GiB/20 ms guard and peaked at 1.594 GiB observed
  per-process RSS. The exact synthetic host-target test, real banked
  wide-family test, and all 12 QRF structure mutations also passed
  independently (14 cases total), peaking at 0.568 GiB.
- `ruff check .`: passed on the final audit tree.
- `ruff format --check` on all 15 Python files changed since `33bf52fe^`:
  passed.
- `git diff --check 33bf52fe^..HEAD`: passed.
- Exact Git tree-object comparisons for both `us_runtime` and all
  `microcosm-build/src` against `a5be536f`: passed. The only nonjournal tracked
  change after that all-suite checkpoint is the strict in-range-family
  regression in `test_us_stacked_spine.py`.

The GitNexus debugging workflow was attempted against a freshly generated
local graph. The sandbox denied registration in `~/.gitnexus/registry.json`,
and the graph parser omitted the two very large suspect runtime modules, so its
query/context results could not trace the raise-site symbols. Direct source and
commit tracing established the exact path instead. The generated 112 MiB
repo-local index was removed after the audit.

The originally cited `battery-verify/pkg3/build.log` was overwritten by a later
retry. The owner-preserved `_BUILD-FAILURE-1PCT.txt` retains the old traceback.
During this continuation the reused log reached survey target 47/47, housing
target 1/1, and the late DAG—including the bounded taxable-interest record—
without that traceback. At the final read-only check the same mutable log had
been truncated to zero bytes, `guard.log` still reported a resource wait, and
the directory had no runner exit marker, `pool.h5`, pool manifest, or gates
artifact. The earlier checkpoint passage is progress evidence only; it is not
revision-bound or a terminal certification result.

The sibling package suites were green before this continuation and their code
was not changed: `microcosm-fit` 93 passed, `microcosm-calibrate` 201 passed,
`microcosm-frame` 294 passed/36 skipped, and `microcosm-data` 275 passed/one
skipped.

## Continuation commits

- `e967bc5d` — record the package 3 host verification failure.
- `3a58c60f` — diagnose the ACS QRF evidence scope leak.
- `22b2c6bc` — add the failing-first scoped binding regression.
- `176c60fc` — scope ACS QRF evidence to calibration targets.
- `0b4339d1` — add failing-first unassigned count regressions.
- `887df056` — harden scoped transfer receipt validation.
- `94b7aecb` — close selected-family, count-stripping, and fit-width audit gaps.
- `5a91d9e6` — align the late pool fixture with strict counts.
- `943e33cf` — align the canonical stacked H5 fixture with strict counts.
- `40b76f6b` — reopen the scoped-binding verification journal.
- `1aed5a31` — independently confirm the scoped-binding diagnosis.
- `19ac8a49` — record the current-tree affected-suite and lint verification.
- `dbe47560` — reopen the current owner continuation.
- `8ebfeb08` — record the current independent scoped-binding diagnosis.
- `bc3d73ba` — record the guarded 528-test and lint verification.
- `bbe3634c` — reopen the current scoped-binding audit journal.
- `a1748679` — independently confirm the current assigned-only diagnosis.
- `dd10904c` — record eight focused scope and binding regressions.
- `e8c91b39` — record the current guarded 528-test affected suite.
- `ceda1c47` — record lint, format, diff, and executable-tree verification.
- `4f85040f` — reopen the final owner continuation journal.
- `31f26ea1` — confirm the current scoped binding diagnosis.
- `21a48ba5` — cover a plausible in-range ACS record-family forgery.
- `8bcaf867` — record the guarded 529-test affected-suite verification.
- `7c67dac7` — finalize the prior scoped-binding continuation report.
- `fdb49ffb` — reopen the current host-binding continuation journal.
- `a4a28c7f` — confirm the current assigned-only binding diagnosis.
- `212ab9a5` — record the current guarded 529-test affected suite.
- `8627b17c` — record current lint, format, whitespace, and drift checks.

## Remaining host step

Run or identify a revision-bound off-chain 1% retry that reaches a terminal
runner verdict and emits the expected pool, manifest, and gates artifacts.
Accept and record the 16 after measurements only if the stacked receipt
invariant, source-preservation proofs, and frozen battery checks all pass. Do not
publish or mutate the pending logbook chain during that run.
