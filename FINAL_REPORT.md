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
The current branch already contained the assigned-only repair and the exact
failing-first regression when this continuation began, so no duplicate
executable edit was made. This continuation independently reconstructed the
failure, traced every canonical caller and consumer, reran the exact validator
and real ordinary/banked regressions, passed all 530 directly affected tests,
passed static checks, and confirmed exact production-tree identity with the
branch's complete-suite checkpoint.

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
same transfer retains regimes and evidence. The producer cases use test
authority and the exact validator case separately uses canonical receipt
authority; together they cover both sides of the boundary without forging a
production authority.

## Verification

Fresh focused verification ran under the owner-provided 12 GiB/20 ms guard:

- exact taxable-interest validator regression;
- real ordinary 15-target transfer; and
- real banked 15-target transfer.

All 3 passed with exit zero and maximum observed per-process RSS of 0.572 GiB.

All five directly affected files then ran together under the same guard:

- ordinary ACS transfer: 64 tests;
- multispine serialization: 5 tests;
- stacked spine: 259 tests;
- multispine pool tooling: 164 tests; and
- H5 receipt I/O: 38 tests.

All 530 passed with exit zero and maximum observed per-process RSS of 1.658 GiB.
The only output beyond test progress was the known joblib physical-core fallback
and 2,313 pandas fixture-fragmentation warnings.

Static verification also passed:

- repository-wide `ruff check .`;
- `ruff format --check` on all 15 Python files changed since `33bf52fe^`;
- `git diff --check 33bf52fe^..HEAD`; and
- worktree whitespace checks.

Every current production source, tool, spec, project, and lock Git object is
identical to complete-suite checkpoint `d29a8705`. At that checkpoint, guarded
fresh-process shards covered all 6,608 collected repository items without a
failed shard. The only current package difference is the ordinary/banked
parametrization in `test_us_stacked_spine.py`; the complete 259-test file passed
inside the affected run above.

The GitNexus debugging workflow guided the raise-site, history, caller, and
consumer trace. Indexed query/context tools were unavailable in this session,
so direct source and Git-object tracing supplied the documented fallback. Three
independent read-only audits of runtime reachability, regression coverage, and
host/history binding agreed with the result.

## Host verification boundary

Host certification is not claimed. The original failed `build.log` and
checkpoints were overwritten by the retry script; the traceback survives in
the owner-provided `_BUILD-FAILURE-1PCT.txt` and committed journals. Timestamp
evidence strongly binds the original executable to `33bf52fe` (with a likely
journal-only child as run HEAD), but the old artifacts did not embed a Git SHA.

At the final read-only snapshot, `2026-08-21 07:57:18Z`, the newer retry had
completed all 47 survey targets, the one housing target, and all 10 primary-QRF
target checkpoints without the old taxable-interest traceback. In particular,
it rebuilt `puf_tax_itemization__batch_1/taxable_interest_income` as target
22/47 and continued through the remaining early targets.

The retry was still active: `guard.log` reported an active peer at
`2026-08-21 07:54:05Z`. The top-level host directory contained only
`build.log`, `guard.log`, and `pool.checkpoints`; it had no final `pool.h5`,
`pool.manifest.json`, `pool.gates.json`, terminal exit marker, or logbook
receipt. The run began during launch-window commit `8920193e`, whose production
`stacked_spine.py` object is identical to the current one, but neither logs nor
checkpoint metadata embed a Microcosm revision binding. This is encouraging
progress, not a terminal pass or certification verdict.

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
  rejection; and
- `f3246728` — exercise the real wide-family boundary through ordinary and
  banked transfer.

This continuation is recorded by:

- `60026619` — reopen the required progress journal;
- `b8d223dd` — record the independent diagnosis and focused green matrix;
- `798c0189` — record the reconciled audits and green 530-test affected suite;
- `39b8425e` — record static and exact-tree verification; and
- this commit — refresh the required final report.
