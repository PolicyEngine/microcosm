# Final report: package 3 ACS QRF receipt scoping

## Outcome

The package 3 ACS QRF scope leak is corrected and locally verified. Only the
nine assigned model-required, source-operator, and adult-care targets opt into
QRF regime work, evidence, and two-part post-transfer calibration. The
unassigned `person/puf_tax_itemization/taxable_interest_income` target retains
ordinary transfer behavior, strict four-count receipt accounting, and no QRF
evidence.

The supplied traceback fingerprints historical commit `33bf52fe`. The current
branch already contained the complete correction and exact failing-first
regression, so this continuation made no duplicate executable or test edit. It
independently reconstructed the failure, reproduced the regression red on the
leaking runtime, verified it green on current `HEAD`, ran all 14 decisive cases
and all 529 directly affected cases, collected the exact 6,608-item repository
surface, and reran lint, formatting, whitespace, and Git-object drift checks.
All completed local verdicts are green.

The restricted host build is not claimed certified. At the final
2026-08-21 02:21:24 EDT read-only snapshot, its process still held the empty
live `build.log` open, the guard continued emitting resource-wait heartbeats,
and no final pool, manifest, gates, terminal exit, or revision-bound artifact
existed.

## Root cause

Commit `33bf52fe` widened ACS QRF regime detection, fitted-result verification,
provenance, receipt attachment, and terminal receipt validation to every
transferred target. That exceeded the nine owner-declared calibration targets.

The canonical `puf_tax_itemization` family has 15 targets and is split at the
certified eight-target fit width. The real taxable-interest record therefore
uses `puf_tax_itemization__batch_1`, while the public receipt surface and old
validator expected `puf_tax_itemization`. The old validator validated every
target before consulting the calibration registry, producing the exact
historical path:

- `validate_stacked_gap_fill_receipt`, line 4512 at `33bf52fe`;
- `_validate_acs_imputed_pattern_evidence`, line 4310 at `33bf52fe`; and
- `ValueError: ... taxable_interest_income: ACS QRF pattern record binding is invalid.`

Accepting arbitrary `__batch_*` aliases would weaken assigned-target binding
and merely expose the next 8-versus-15 target-order mismatch. Assignment
scoping, not permissive family matching, is the correct fix.

## Correction

The immutable registry contains exactly two early and seven late targets:

- early: `unemployment_compensation` and
  `self_employment_income_last_year`;
- late adult care: `pre_subsidy_care_expenses`;
- late child support: `child_support_expense` and
  `child_support_received`;
- late disability and weeks: `disability_benefits` and `weeks_unemployed`;
- late workers compensation: `workers_compensation`; and
- late energy subsidy: `spm_unit_energy_subsidy`.

The committed correction enforces that surface at independent boundaries:

- `transfer_acs_inputs` has an explicit, default-empty
  `regime_evidence_targets` selection. Both stacked owners derive their
  selection from the immutable registry.
- Ordinary and banked fits detect, verify, and retain regimes only for selected
  model targets. Unselected sibling records explicitly carry empty regimes.
- Early and late receipt builders independently attach QRF evidence only for
  selected targets.
- Terminal validators require the complete legacy four-count block for every
  target, reject forged evidence on unassigned targets before record binding,
  and retain exact family binding for assigned targets.
- Selection does not alter unassigned draws, post-transfer calibration writes
  only selected target columns, and warm banks persist raw draws/state while
  evidence is recomputed from the current selection.
- The canonical gap-fill owner remains pinned to the certified width eight.

The primary correction and hardening commits are:

- `22b2c6bc` — add the exact failing-first taxable-interest regression;
- `176c60fc` — scope ACS QRF evidence to calibration targets;
- `887df056` — restore strict family binding and independent legacy counts;
- `94b7aecb` — close mixed-family, count, and fit-width gaps; and
- `21a48ba5` — reject a plausible rehashed assigned `__batch_1` alias.

## Regression evidence

`test_gap_fill_qrf_binding_excludes_unassigned_batched_targets` is the exact
validator regression. In a detached temporary worktree at test-only commit
`22b2c6bc`, whose runtime is byte-identical to `33bf52fe`, it failed through
the supplied line 4512-to-4310 path with the same taxable-interest binding
message. Peak observed RSS was 0.452 GiB. The temporary worktree was removed.

On current `HEAD`, 14 decisive cases passed under the owner-provided 12 GiB/
20 ms guard with a 0.570 GiB maximum observed per-process RSS:

- the exact unassigned taxable-interest validator boundary;
- the real banked 15-target producer, proving taxable interest has empty
  regimes and no receipt evidence while assigned unemployment compensation
  retains both; and
- all 12 fully rehashed QRF structure mutations, including exact, plausible
  in-range, and out-of-range record-family forgeries.

Three independent read-only source, history, regression, bank-resume, and host
audits found no canonical alternate leakage path or material missing regression
for the supplied failure.

## Verification

The five directly affected ordinary-transfer, multispine-serialization,
stacked-spine, pool-tool, and H5 files collected 64, 5, 258, 164, and 38 cases:
529 total. All 529 passed together with exit zero under the owner guard; maximum
observed per-process RSS was 1.589 GiB.

The current repository collects exactly 6,608 items across 260 files, with a
1.163 GiB collection peak. `packages`, `tools`, `specs`, `pyproject.toml`, and
`uv.lock` are byte-for-byte Git-identical to complete-suite checkpoint
`d29a8705`; only `PROGRESS.md` and `FINAL_REPORT.md` differ. Fresh-process
shards at that exact checkpoint covered the complete collection without a
failure:

| Shard | Result | Peak RSS |
|---|---:|---:|
| `microcosm-fit` | 93 passed | 0.724 GiB |
| `microcosm-calibrate` | 201 passed | 0.436 GiB |
| `microcosm-data` | 275 passed, 1 skipped | 11.049 GiB |
| `microcosm-frame` | 294 passed, 36 skipped | 6.488 GiB |
| build core + UK | 1,856 passed, 33 skipped | 4.182 GiB |
| build US a-r | 2,406 passed, 3 skipped | 8.844 GiB |
| build US s-z | 1,411 passed, 1 skipped | 10.579 GiB |

The core+UK summary contains two additional passing subtest outcomes beyond
its 1,887 collected items; the shard collection union, rather than naive
summary addition, is the authoritative 6,608-item count.

Static verification passed:

- repository-wide `ruff check .`;
- `ruff format --check` on all 15 Python files changed since `33bf52fe^`;
- `git diff --check 33bf52fe^..HEAD` and worktree whitespace checks; and
- exact package/config Git-object equality with `d29a8705`.

The first format invocation supplied a newline-separated zsh value as one file
argument and exited before checking files. The null-delimited rerun checked all
15 files successfully and made no changes.

The GitNexus debugging workflow was selected. Graph query/context endpoints and
resources were unavailable, so direct raise-site, caller, producer, validator,
test, runtime-enumeration, warm-bank, and Git-history tracing provided the
documented fallback.

## Current continuation commits

- `3194df71` — reopen the owner continuation;
- `7e55e5c5` — confirm the current scoped diagnosis and red/green proof;
- `4c1f3fea` — record the guarded 529-test affected suite; and
- `b5706f6f` — record lint, format, whitespace, collection, drift, and host
  checks.

## Remaining host boundary

At 2026-08-21 02:21:24 EDT, PID 28857 still had the worktree as its current
directory and held stdout/stderr open to `pkg3/build.log`. The live state was:

- `build.log`: zero bytes, unchanged since 02:00:52 EDT;
- `guard.log`: 8,256 bytes, with a 02:19 EDT resource-wait heartbeat;
- directory contents: only `build.log` and `guard.log`; and
- absent: checkpoint tree, `pool.h5`, `pool.manifest.json`, `pool.gates.json`,
  logbook spool/receipts, and a terminal exit marker.

The preceding attempt had progressed through the transfer build without the
supplied traceback, then rolled over and deleted its checkpoints/truncated its
log. The retry script would have stopped before rollover if a gates artifact
existed, so that ended attempt was not certifying. The current live log also
contains no supplied binding traceback, but an empty self-truncating log is not
a success verdict.

Executable content is Git-identical to the complete local-suite checkpoint,
but neither launcher, log, nor artifact records a Microcosm Git SHA. Completion
of the external boundary requires a durable terminal exit and passing final
pool, manifest, and gates artifacts with explicit revision provenance. Do not
publish or mutate the release chain as a side effect of that verification.
