# Final report: package 3 ACS QRF receipt scoping

## Outcome

The shared stacked gap-fill leak is fixed and locally verified. Canonical
production now limits the added calibration-audit regime detection and
verification, per-target regime provenance, QRF receipt evidence, and two-part
post-transfer calibration to the nine assigned model-required,
source-operator, and adult-care targets. The QRF's ordinary operational regime
logic remains unchanged for every fitted target. The unassigned
`person/puf_tax_itemization/taxable_interest_income` target retains ordinary
transfer behavior: its physical fit record remains
`puf_tax_itemization__batch_1`, its audit regimes are empty, its receipt has the
four legacy transfer counts and no QRF pattern evidence, and it receives no
post-transfer calibration write.

The supplied traceback fingerprints historical executable commit `33bf52fe`.
The current branch already contained the assigned-only executable repair, the
exact synthetic binding regression, and the strengthened real ordinary/banked
producer-to-validator regression when this continuation began, so no duplicate
runtime or test edit was made. This pass independently reconstructed the
failure, traced every canonical caller and consumer, reconciled three
independent audits, and reran the focused, directly affected, and complete
repository surfaces at the current tip. Both wide-family cases are proven red
at the supplied invariant on the historical runtime and green on the current
runtime. The 16-check focused matrix, all 530 directly affected tests, the full
6,609-item repository suite, static checks, and exact production/test-tree
bindings are green. A live restricted-host retry has also rebuilt taxable
interest and progressed well beyond it without the reported binding error, but
that retry remains nonterminal and is not a certification result.

## Root cause

At `33bf52fe`, `validate_stacked_gap_fill_receipt` invoked
`_validate_acs_imputed_pattern_evidence` for every transferred target before it
looked up the target in the early calibration registry. The transfer runtime
also performed the newly added calibration-audit regime detection and
verification, retained its provenance, and attached QRF regime evidence
globally. The target's ordinary QRF fitting was not itself the leak.

The canonical `puf_tax_itemization` family has 15 targets and is physically
split at the certified maximum of eight targets per fit. Taxable interest
therefore carried a record whose physical family was
`puf_tax_itemization__batch_1`, while the old validator compared it with the
public family `puf_tax_itemization`. The supplied traceback maps exactly to the
old runtime:

- `stacked_spine.py:4512` at `33bf52fe`: unconditional evidence validation for
  every target; and
- `stacked_spine.py:4310` at `33bf52fe`: strict record-family binding failure.

Accepting a batch alias would not have repaired the leak. Unassigned targets
would still have performed calibration-specific regime work and emitted
calibration-specific provenance, and the evidence's regime-target surface
would still have been overbroad. The correct boundary is assignment scoping,
not weakened record binding.

## Correction and assigned surface

The committed correction has four matching fences:

1. `transfer_acs_inputs` defaults `regime_evidence_targets` to empty. The two
   stacked owners explicitly derive their selections from the immutable
   post-transfer calibration registry.
2. Ordinary and banked fits perform the additional audit detection and
   verification only for selected model targets. Per-target provenance strips
   audit regimes from unselected sibling records without changing their
   ordinary QRF draws.
3. Early and late receipt builders attach QRF evidence only to the same
   registry-derived selection, and calibration application writes only the
   selected target column and selected rows.
4. Both terminal validators first validate the four legacy row counts for
   every target. For an unassigned target they reject any QRF or calibration
   evidence and continue; strict QRF record binding remains exact and runs only
   for a declared target.

The immutable policy contains exactly two early and seven late targets:

- early: `unemployment_compensation` and
  `self_employment_income_last_year`;
- late adult care: `pre_subsidy_care_expenses`;
- late child support: `child_support_expense` and
  `child_support_received`;
- late source operators: `disability_benefits`, `weeks_unemployed`,
  `workers_compensation`, and `spm_unit_energy_subsidy`.

Taxable interest is absent. All four production `transfer_acs_inputs` caller
classes were traced. Only the early and late stacked owners opt into regime
evidence; generic multispine and pool-tool callers retain the empty default.
No alternate canonical producer, ordinary/banked fit, target-bank resume,
serializer, receipt builder, terminal validator, or calibration-write path
broadens the nine-target policy.

One API qualification is intentional: a noncanonical library caller may
explicitly request regime provenance for any target already on its requested
transfer surface. No production caller exposes that choice, and canonical
stacked validators reject evidence on unassigned targets.

One branch-scope qualification is independent of this correction. The original
lane also changed the two pinned SIPP vehicle and voluntary-filing readers to
streaming type inference after a full-donor parser exceeded the memory ceiling.
Those loaders output `household_vehicles_owned`, `household_vehicles_value`, and
`would_file_taxes_voluntarily`; none overlaps the nine calibration targets or
any calibration/evidence call path. `_LANE-NOTES.md` records the downstream
coercion, locked-fact coverage, and guarded memory results. This continuation
preserves those already verified operational mitigations; the assigned-only
claim here is specifically about calibration, QRF audit, and receipt behavior.

## Regression evidence

`test_gap_fill_qrf_binding_excludes_unassigned_batched_targets` constructs the
exact unassigned taxable-interest receipt with realistic
`puf_tax_itemization__batch_1` evidence. It requires the canonical validator to
reject that evidence as undeclared, then removes it and proves that the
unchanged four-count legacy receipt validates.

The test is genuinely failing-first. Commit `22b2c6bc` contains the regression
while its runtime object is byte-identical to failing `33bf52fe`; that runtime
reaches the supplied line-4512-to-line-4310 error instead of the corrected
undeclared-evidence boundary. A prior detached execution recorded in the
committed journal reproduced the exact taxable-interest failure.

`test_gap_fill_scopes_qrf_evidence_off_wide_unassigned_family` exercises the
real 15-target family through both ordinary and banked transfer. Both cases
prove that taxable interest retains the physical `__batch_1` record with empty
regimes and no QRF receipt, while selected unemployment compensation in the
same transfer retains regimes and evidence. Commit `ad2a44c1` then closes the
target-receipt-to-validator gap: each case copies the actual generated taxable
receipt into a canonical receipt and calls `validate_stacked_gap_fill_receipt`.
The splice is deliberately limited to that target because test-authority
execution does not create the unrelated canonical calibration-owner receipts.

The bridge is demonstrated failing-first, not inferred. In an isolated
temporary worktree, replacing only `acs_multispine.py`, `acs_transfer.py`, and
`stacked_spine.py` with their exact `33bf52fe` objects made both ordinary and
banked cases fail at lines 4512 and 4310 with the supplied taxable-interest
record-binding error. Restoring the current three objects made the identical
two cases pass. The temporary worktree was removed without changing the shared
tree.

## Verification

This continuation reran the strengthened boundary matrix under the
owner-provided 12 GiB/20 ms guard:

- exact taxable-interest validator regression;
- real ordinary and banked 15-target transfer;
- all 12 fully rehashed strict-binding structure mutations; and
- mixed-family selected-regime draw preservation.

All 16 checks passed with exit zero. The guard reported 0.029 GiB maximum
observed per-process RSS; the only warning was joblib's physical-core fallback.
The first launcher attempt never reached collection because the sandbox denied
`uv`'s default user cache, so every successful run used a task-local cache under
`/private/tmp`.

All five directly affected files then ran together under the owner-provided
12 GiB/20 ms guard:

- ordinary ACS transfer: 64 tests;
- multispine serialization: 5 tests;
- stacked spine: 259 tests;
- multispine pool tooling: 164 tests; and
- H5 receipt I/O: 38 tests.

All 530 completed together at the current executable/test tree with exit zero.
The guard reported 0.029 GiB maximum observed per-process RSS; warning display
was disabled for the broad matrix.

The complete repository suite then ran in one guarded process. Fresh collection
was 6,609 items. Pytest reached 100% with expected skips and exit zero, and the
guard again reported 0.029 GiB maximum observed per-process RSS. This is the
current-tip suite result requested by the continuation, not an inference from a
prior checkpoint.

Static verification also passed:

- repository-wide `ruff check .`;
- `ruff format --check` on all 15 Python files changed since `33bf52fe^`;
- `git diff --check 33bf52fe^..HEAD`; and
- index/worktree whitespace and final tracked-tree cleanliness checks.

A diagnostic repository-wide format check identified 49 pre-existing files
outside the changed range that would be reformatted. They span unrelated
experiments, UK runtime/tests, and other US/tool files; no out-of-scope bulk
reformat was made. The repository's prescribed lint gate and every changed-file
format check are green.

The current `microcosm-build/src` tree is `7234ac19`, identical to complete-suite
checkpoint `d29a8705` and reviewed regression checkpoint `ad2a44c1`. The current
build-tests tree is `0c5d7816`, identical to `ad2a44c1`. Relative to
`d29a8705`, the only test change is the strengthened ordinary/banked terminal
validator regression in `test_us_stacked_spine.py`; relative to `ad2a44c1`,
only `PROGRESS.md` and `FINAL_REPORT.md` differ. No production/configuration
file has drifted from either checkpoint.

The GitNexus debugging workflow guided the raise-site, history, caller, and
consumer trace. The normal graph-query tools were unavailable and the repository
was not registered in the CLI index. The skill-directed local analysis parsed
far enough to create a partial index, but sandbox policy blocked registration at
`/Users/maxghenis/.gitnexus/registry.json`; the generated 100 MiB index was moved
out of the worktree to `/private/tmp`. Direct source, exact Git-object, and
history tracing supplied the documented fallback. Three independent read-only
audits of invariant flow, regression strength, and branch scope agreed with the
result.

## Host verification boundary

Host certification is not claimed. The retry workflow deleted the original
checkpoints and repeatedly truncated the mutable `build.log`; the supplied
traceback now survives only in the owner-provided `_BUILD-FAILURE-1PCT.txt` and
committed journals. The offending executable blob is unambiguously the one
introduced by `33bf52fe` and retained through `22b2c6bc`, but the deleted
artifacts did not embed a Microcosm SHA. The exact process-launch journal commit
therefore cannot be recovered from the host log.

At the final read-only snapshot, `2026-08-21 11:42:58Z`, a new external retry
was active. Its mutable `build.log` was 211,849 bytes and contained no traceback,
`ValueError`, or binding failure. It had written the exact taxable-interest
checkpoint as target 22/47 with physical family
`puf_tax_itemization__batch_1`, then progressed through target 39/47. The
checkpoint tree contained 41 files, including the assembled checkpoint and
manifest. This is direct evidence that the reported boundary did not recur in
the live retry up to that snapshot.

The same snapshot had no runner exit marker and no final `pool.h5`,
`pool.manifest.json`, or `pool.gates.json`. The mutable log also does not bind
the process launch to a recoverable Microcosm SHA. Progress beyond the former
failure is therefore not a terminal, revision-bound host verdict, and no host
or certification success is inferred.

Completion of the external boundary requires a durable, terminal,
revision-bound 1% result with passing final pool, manifest, and gates artifacts.
Publication and release-chain mutation remain outside this task.

## Commit lineage

The executable/regression correction is carried by:

- `22b2c6bc` — add the failing-first scoped-binding regression;
- `176c60fc` — scope regime work, provenance, receipts, and validation to the
  registry-derived selection;
- `0b4339d1`, `887df056`, `94b7aecb`, and `21a48ba5` — harden legacy counts,
  exact family/width binding, mixed-family behavior, and rehashed forgery
  rejection;
- `f3246728` — exercise the real wide-family boundary through ordinary and
  banked transfer; and
- `ad2a44c1` — pass the real generated taxable-interest receipts through the
  canonical terminal validator in both modes.

This continuation audit is recorded by:

- `2210eb43` — reopen the required progress journal;
- `8880eec2` — record the independent raise-site, caller, history, and
  regression diagnosis;
- `522d64f8` — record the fresh 16-check focused matrix;
- `1bf2519f` — record the guarded 530-test affected matrix;
- `ef7e2e63` — record the fresh 6,609-item full suite, static checks, and exact
  prior-checkpoint object bindings; and
- this commit — refresh the required output report with the current result and
  nonterminal host snapshot.
