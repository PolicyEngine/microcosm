# Final report: package 3 ACS QRF receipt scoping

## Outcome

The shared stacked gap-fill leak is fixed and locally verified. Canonical
production now limits QRF regime detection, verification, receipt evidence,
and two-part post-transfer calibration to the nine assigned model-required,
source-operator, and adult-care targets. The unassigned
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
independent audits, and reran the focused and directly affected surfaces at the
current tip. Both wide-family cases are proven red at the supplied invariant on
the historical runtime and green on the current runtime. All 530 directly
affected tests, static checks, and exact production-tree binding are green.

## Root cause

At `33bf52fe`, `validate_stacked_gap_fill_receipt` invoked
`_validate_acs_imputed_pattern_evidence` for every transferred target before it
looked up the target in the early calibration registry. The transfer runtime
also detected, verified, retained, and attached QRF regime evidence globally.

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
2. Ordinary and banked fits detect and verify regimes only for the selected
   model targets. Per-target provenance strips regimes from unselected sibling
   records.
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

An independent focused audit reran the strengthened boundary matrix:

- exact taxable-interest validator regression;
- real ordinary 15-target transfer; and
- real banked 15-target transfer.

All three passed with exit zero; the ordinary case emitted only the known
joblib physical-core fallback warning. A separate run exercised all 12
fully-rehashed strict-binding mutation variants; all passed. These independent
focused runs used the project environment without a memory guard.

All five directly affected files then ran together under the owner-provided
12 GiB/20 ms guard:

- ordinary ACS transfer: 64 tests;
- multispine serialization: 5 tests;
- stacked spine: 259 tests;
- multispine pool tooling: 164 tests; and
- H5 receipt I/O: 38 tests.

All 530 passed together at the current executable/test tree with exit zero and
maximum observed per-process RSS of 1.596 GiB. This latest run supersedes the
earlier affected-matrix results. Warning display was disabled for the broad
matrix.

Static verification also passed:

- repository-wide `ruff check .`;
- `ruff format --check` on all 15 Python files changed since `33bf52fe^`;
- `git diff --check 33bf52fe^..HEAD`; and
- worktree whitespace checks.

Every current production source, tool, spec, project, and lock Git object is
identical to complete-suite checkpoint `d29a8705`. At that checkpoint, guarded
fresh-process shards covered all 6,608 collected repository items without a
failed shard. Current production and test objects are also exact matches to
reviewed regression checkpoint `ad2a44c1`; relative to the complete-suite
checkpoint, the only package difference is the strengthened ordinary/banked
regression in `test_us_stacked_spine.py`, whose complete 259-test file passed in
the fresh matrix.

The GitNexus debugging workflow guided the raise-site, history, caller, and
consumer trace. The normal registry-backed tools were unavailable, but the
preserved graph indexed at `42819d5` was opened through GitNexus's read-only
local backend. Error-text query and context resolved the regression surface;
custom Cypher confirmed that the oversized production module was absent from
the graph, so direct source and Git-object tracing supplied that portion of the
documented fallback. Three independent read-only audits of runtime reachability,
regression coverage, and branch scope agreed with the result.

## Host verification boundary

Host certification is not claimed. The retry script explicitly deleted the
original checkpoints and truncated the existing `build.log`; the traceback now
survives only in the owner-provided `_BUILD-FAILURE-1PCT.txt` and committed
journals. Timestamp and reflog evidence places the original run at journal-only
commit `f7ecac75`, whose complete `microcosm-build/src` tree and
`stacked_spine.py` object are identical to `33bf52fe`. The deleted artifacts did
not embed a Microcosm SHA, so this is a strong Git-object inference rather than
an artifact-contained revision receipt.

The active retry began at launch-window commit `3194df71`, correcting an
earlier journal inference of `8920193e`. Its complete production source tree,
build-tool object, and `stacked_spine.py` object are identical at both commits
and at current `HEAD`. Its editable environment and worker metadata bind it to
this worktree path, but its artifacts likewise embed no Microcosm revision.

At the final read-only snapshot, `2026-08-21 09:44:11Z`, the retry remained
live and nonterminal. It had rebuilt taxable interest, completed all eight late
`puf_tax_itemization__batch_1` targets, passed several later bounded families,
and reached target 3/8 `farm_income` in
`person/puf_tax_itemization__batch_3`, without recurrence of the historical
exception. The sole taxable-interest log entry recorded successful checkpoint
creation. However, final stacked gap-fill validation had not observably
returned, so fit completion is not claimed as passage of the supplied receipt
invariant. `build.log` changed at `09:43:45Z` and the external guard emitted a
fresh wait heartbeat at `09:44:09Z`. There was no traceback, `ValueError`,
binding text, terminal exit marker, transferred or simulated checkpoint, final
`pool.h5`, `pool.manifest.json`, or `pool.gates.json`. This is progress only,
not a terminal pass or certification verdict.

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

This continuation is recorded by:

- `d9355679` — reopen the required progress journal;
- `00eb041d` — record the raise-site, caller, and regression diagnosis;
- `ad2a44c1` — commit the strengthened producer/validator regression;
- `2a80261e` — record red/green, affected-suite, static, object, and host
  verification; and
- `42819d53`, `a3f30485`, `d67df37c`, and `13db9a36` — reopen the latest audit
  and record its independent diagnosis, focused tests, 530-test affected
  matrix, static checks, production-object comparison, and corrected host
  boundary; and
- `2d330316`, `405b7ee1`, `90fa567f`, and `7a070439` — reopen this owner
  recheck and record the focused matrix, independent audits, 530-test affected
  matrix, static checks, production-object comparison, and current host
  boundary; and
- `77581afb` — refresh the required final report with the current local
  verification; and
- `2d5e4540` — reopen the current required progress journal;
- `c2448d7f` — record the independent raise-site, caller, history, and
  regression diagnosis;
- `e2e0d5d5` — record the fresh guarded 530-test matrix, static checks, and
  exact object bindings; and
- this commit — refresh the required report with the current verification and
  nonterminal host boundary.
