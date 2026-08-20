# Final report: package 3 ACS QRF receipt scoping

## Outcome

Fixed the host 1% verification failure at
`person/puf_tax_itemization/taxable_interest_income` without opting that
unassigned target into the package 3 calibration. The final local tree passes
the complete `microcosm-build` suite and lint surface.

No after artifact is claimed or accepted. An externally owned exact 1% retry
started during the final audit and rebuilt all 47 survey targets plus the one
housing target without a traceback, including the exact bounded
`taxable_interest_income` record. Its log then became quiet while its guard
continued to report live build processes and no runner exit marker existed, so
terminal stacked receipt validation and frozen-battery success are not
claimed. The frozen sample/clone seed remains 578, and no battery band,
threshold, comparator, fold, publication boundary, or pending-chain state was
changed.

The 2026-08-20 owner continuation independently re-traced the old raise site,
the current runtime/receipt selector, the failing-to-fixed history, and the
exact regression surface. It found no remaining canonical path that can opt
`taxable_interest_income` into QRF regime work or evidence and therefore made
no further executable change. Eight narrow regressions and all 528 tests in the
five directly affected files pass on the current executable tree. Three
independent read-only audits of source, history, and regression coverage reached
the same scoped-fix verdict.

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
does not weaken legacy receipt checks.

## Regression coverage

The regression surface includes:

- the exact host target in a real wide, banked `puf_tax_itemization` family,
  proving its `__batch_1` record has no regimes or QRF receipt while selected
  unemployment compensation retains both;
- selected-family binding, forged batch aliases, regime tampering, and missing
  or inconsistent early/late transfer counts;
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

Verification used the already-synced exact-lock environment with
`UV_NO_SYNC=1` and a writable temporary uv cache because sandbox DNS and the
default uv cache are unavailable.

- All 225 `packages/microcosm-build/tests/test_*.py` files passed across fresh
  pytest processes at `a5be536f`. No executable file changed between that
  commit and the final audit tree.
- The five directly affected transfer, multispine, stacked, pool-tool, and H5
  files were rerun together on the final executable tree: all 528 passed. The
  run used the owner-provided 12 GiB/20 ms guard and peaked at 1.494 GiB
  observed per-process RSS. Eight narrow scope/binding regressions also passed
  independently, peaking at 0.559 GiB.
- `ruff check .`: passed on the final audit tree.
- `ruff format --check` on all nine continuation-touched Python files: passed.
- `git diff --check 33bf52fe^..HEAD`: passed.
- `git diff --name-only a5be536f..HEAD` lists only `FINAL_REPORT.md` and
  `PROGRESS.md`, confirming that the current executable tree is identical to
  the prior complete 225-file build-suite pass.

The originally cited `battery-verify/pkg3/build.log` was overwritten by the
later retry. The owner-preserved `_BUILD-FAILURE-1PCT.txt` retains the old
traceback. At the final read-only check, the reused log had reached survey
target 47/47, housing target 1/1, and the late DAG, including
`puf_tax_itemization__batch_1/taxable_interest_income`, without a new traceback.
The directory still had no runner exit marker, `pool.h5`, pool manifest, or
gates artifact, and the log did not bind a revision SHA; the external guard
continued to report other live work. That passage through transfer checkpoints
is useful progress evidence, but it is neither revision-bound nor a terminal
certification result.

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

## Remaining host step

Run or identify a revision-bound off-chain 1% retry that reaches a terminal
runner verdict and emits the expected pool, manifest, and gates artifacts.
Accept and record the 16 after measurements only if the stacked receipt
invariant, source-preservation proofs, and frozen battery checks all pass. Do not
publish or mutate the pending logbook chain during that run.
