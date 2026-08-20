# Final report: package 3 ACS QRF receipt scoping

## Outcome

Fixed the host 1% verification failure at
`person/puf_tax_itemization/taxable_interest_income` without opting that
unassigned target into the package 3 calibration. The final local tree passes
the complete `microcosm-build` suite and lint surface.

No after artifact is claimed or accepted. The exact 1% host rebuild remains
the only outstanding step because the pinned host data is unavailable in this
sandbox. The frozen sample/clone seed remains 578, and no battery band,
threshold, comparator, fold, publication boundary, or pending-chain state was
changed.

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

- All 225 `packages/microcosm-build/tests/test_*.py` files pass on the final
  tree across fresh pytest processes, including the five directly affected
  transfer/stacked/pool files and the repaired H5 file.
- `uv run ruff check .`: passed.
- `ruff format --check` on all nine continuation-touched Python files: passed.
- `git diff --check 33bf52fe..HEAD`: passed.

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

## Remaining host step

Rebuild off-chain at exactly 1% with sample and clone seed 578 under the host
memory guard. Accept and record the 16 after measurements only if the stacked
receipt invariant, source-preservation proofs, and frozen battery checks all
pass. Do not publish or mutate the pending logbook chain during that run.
