# Final report: package 3 ACS QRF receipt scoping

## Outcome

The package 3 shared-path leak is fixed and locally verified. Only the nine
assigned model-required, source-operator, and adult-care targets opt into QRF
regime detection, regime verification, pattern evidence, and two-part
post-transfer calibration. The unassigned
`person/puf_tax_itemization/taxable_interest_income` target retains ordinary
transfer behavior, the four legacy transfer counts, and no QRF pattern
evidence.

The supplied traceback fingerprints historical commit `33bf52fe`. The current
branch already contained the scoped implementation and exact failing-first
regression, so this continuation did not add duplicate executable or test code.
It independently reconstructed the failure and correction, reconciled three
read-only audits of every canonical producer/consumer path, reran the decisive
regression matrix and all 529 directly affected tests, and reran static and
executable-drift checks. Every completed local verdict is green.

## Root cause

At `33bf52fe`, `validate_stacked_gap_fill_receipt` called
`_validate_acs_imputed_pattern_evidence` for every transferred target before it
looked up the target in the assigned calibration registry. The transfer runtime
also detected and attached QRF regime evidence globally.

The canonical `puf_tax_itemization` family has 15 targets and is split at the
certified maximum of eight targets per fit. Taxable interest therefore had the
physical record family `puf_tax_itemization__batch_1`, while the old validator
required the public family `puf_tax_itemization`. That mismatch produced the
exact historical traceback:

- `stacked_spine.py:4512` at `33bf52fe`: unconditional evidence validation;
- `stacked_spine.py:4310` at `33bf52fe`: pattern-record binding failure; and
- `ValueError: ... taxable_interest_income: ACS QRF pattern record binding is invalid.`

Permitting a batch alias would not be a sound repair. It would next expose an
eight-recorded-regimes versus fifteen-expected-targets mismatch and would leave
unassigned targets doing calibration-specific runtime work. Value calibration
itself was already registry-filtered; the leak was in regime detection and
verification, provenance, receipt attachment, and terminal validation.

## Correction

The correction is carried by these committed steps:

- `22b2c6bc` adds the failing-first regression for the exact taxable-interest
  boundary.
- `176c60fc` introduces default-empty `regime_evidence_targets`, scopes ordinary
  and banked regime work and provenance to that explicit selection, makes both
  stacked owners derive the selection from the calibration registry, attaches
  evidence only for selected targets, and performs assignment lookup before
  strict record validation.
- `0b4339d1` covers unassigned legacy transfer-count tampering.
- `887df056` restores strict exact family binding, validates the four legacy
  counts independently for assigned and unassigned targets, and adds a real
  banked wide-family integration regression.
- `94b7aecb` requires the complete count block, pins the canonical fit width,
  and proves mixed selected/unselected families preserve the unassigned draw.
- `21a48ba5` rejects a fully rehashed, plausible in-range `__batch_1` forgery.

The current canonical reachability audit confirms:

- the immutable policy contains exactly two early and seven late assigned
  targets;
- only the early and late stacked owners opt into regime evidence, and both use
  that registry-derived selection;
- ordinary and banked fits compute and verify regimes only for selected model
  targets, while unselected target records receive empty regime tuples;
- warm target banks persist raw draws and chain state, not stale regime
  evidence;
- receipt builders omit QRF evidence for unassigned targets;
- early and late validators reject forged unassigned evidence before invoking
  strict pattern-record binding;
- generic multispine and pool-tool callers retain the empty default; and
- calibration writes only the selected target column and selected rows.

No alternate unassigned draw or write path was found. A noncanonical library
caller may explicitly request regime evidence for any target on its requested
surface; no production caller opts taxable interest into that API.

## Regression evidence

The principal regression is
`test_gap_fill_qrf_binding_excludes_unassigned_batched_targets`. It constructs
the exact `person/puf_tax_itemization/taxable_interest_income` receipt with
realistic `puf_tax_itemization__batch_1` evidence, requires the canonical
validator to reject the evidence as undeclared, then removes it and proves the
unchanged four-count legacy receipt validates.

Complementary coverage proves:

- a real banked 15-target PUF itemization run gives taxable interest the
  physical `__batch_1` record but empty regimes and no QRF receipt evidence;
- selected unemployment compensation in the same run retains regimes and
  evidence;
- selecting a sibling for evidence does not change an unselected target's draw;
- default wide-family transfer behavior has no regimes; and
- all fully rehashed structure mutations, including an in-range family alias,
  fail strict assigned-target binding.

The exact regression is demonstrably failing-first: at test-only commit
`22b2c6bc`, whose runtime matches `33bf52fe`, it reaches the supplied
line-4512-to-line-4310 record-binding error rather than the corrected
undeclared-evidence boundary.

## Verification

Fresh focused verification ran under the owner-provided 12 GiB/20 ms guard:

- exact taxable-interest validator regression;
- real banked 15-target integration regression; and
- all 12 rehashed QRF structure mutations.

Result: 14 passed in 15.47 seconds, exit zero, maximum observed per-process RSS
0.568 GiB. The only warning was joblib falling back to logical-core detection.

All five directly affected files then ran together under the same guard:

- ordinary ACS transfer: 64 tests;
- multispine serialization: 5 tests;
- stacked spine: 258 tests;
- multispine pool tool: 164 tests; and
- multispine H5 I/O: 38 tests.

Result: all 529 tests passed in 12:12 with exit zero; maximum observed
per-process RSS was 1.643 GiB.

The complete `packages`, `tools`, `specs`, `pyproject.toml`, and `uv.lock` Git
objects are identical to checkpoint `d29a8705`. At that checkpoint, guarded
fresh-process package/file shards covered all 6,608 collected repository items
without a failed shard. Later commits changed only root journals. Static
verification on the current tree also passes:

- repository-wide `ruff check .`;
- `ruff format --check` on all 15 Python files changed since `33bf52fe^`;
- `git diff --check 33bf52fe^..HEAD`; and
- worktree whitespace checks.

The GitNexus debugging workflow was selected. Its graph query/context tools
were unavailable in this session. Direct raise-site, Git-history, caller,
producer, serializer, validator, warm-bank, and regression tracing provided the
prescribed fallback and was reconciled by three independent read-only audits.

## Host verification boundary

Host certification is not claimed. At the final read-only snapshot taken
2026-08-21 07:08:01Z, the external 1% retry had rebuilt the assembled checkpoint
plus targets 1 through 34 of 47. It rebuilt the exact
`puf_tax_itemization__batch_1/taxable_interest_income` target as 22/47 and
continued without the old traceback, but had not reached terminal stacked
receipt validation or a runner verdict.

The host directory contained `build.log`, `guard.log`, and a checkpoint tree.
It did not contain final `pool.h5`, manifest, gates, or terminal-exit artifacts,
and the mutable log did not bind the run to a Microcosm Git revision. Progress
through target 13 is therefore neither a pass nor a certification result.

Completion of the external boundary requires a revision-bound off-chain 1%
retry with a durable terminal exit and passing final pool, manifest, and gates
artifacts. Publication and pending-logbook mutation remain outside this task.

## Continuation commits

- `bb47e5a3` — reopen this owner continuation in `PROGRESS.md`;
- `4f65ba59` — record the independently reconstructed diagnosis;
- `423e6906` — record the fresh failing-first and current-green proof;
- `f4926ea3` — record the green 529-test directly affected suite;
- `15e3881d` — record static checks, complete-suite object binding, and the
  final host snapshot; and
- this report commit — refresh the required output file from current evidence.
