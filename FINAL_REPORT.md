# Final report: package 3 ACS QRF receipt scoping

## Outcome

The package 3 ACS QRF scope leak is corrected and locally verified. Only the
nine assigned model-required, source-operator, and adult-care targets opt into
QRF regime work and evidence. The unassigned
`person/puf_tax_itemization/taxable_interest_income` target retains ordinary
transfer behavior, four-count receipt accounting, and no QRF evidence.

The supplied traceback fingerprints historical commit `33bf52fe`; the current
branch already carried the complete correction and failing-first regression.
This continuation therefore made no duplicate executable edit. It independently
reconstructed the failure, proved the exact regression red on the bad runtime
and green on current `HEAD`, reran all 14 decisive cases, all 529 directly
affected cases, and the exact 6,608-item repository collection, and reran lint,
format, whitespace, and executable-drift checks. All completed local verdicts
are green.

The restricted host retry is not claimed as certified. At the final
2026-08-21 01:54:15 EDT read-only snapshot it was still running, had no current
traceback, and had progressed into the late source-operator transfers, but had
not emitted an exit marker, final pool, manifest, or gates artifact.

## Root cause

Commit `33bf52fe` widened QRF regime detection, fitted-result verification,
provenance, receipt attachment, and receipt validation to every ACS transfer
target. That exceeded the nine owner-declared calibration targets.

The canonical `puf_tax_itemization` family has 15 targets and is split at the
certified eight-target fit width. The real taxable-interest transfer record was
therefore bound to `puf_tax_itemization__batch_1`, while the public receipt
surface and validator expected `puf_tax_itemization`. Because the old validator
validated every target before checking calibration assignment, the unassigned
taxable-interest receipt reached the exact historical predicate:

- `validate_stacked_gap_fill_receipt`, line 4512 at `33bf52fe`;
- `_validate_acs_imputed_pattern_evidence`, line 4310 at `33bf52fe`; and
- `ValueError: ... taxable_interest_income: ACS QRF pattern record binding is invalid.`

Permitting arbitrary `__batch_*` aliases would have weakened the binding
invariant and merely exposed the next target-order mismatch. The correct fix is
assignment scoping while preserving exact binding for assigned targets.

## Correction

The committed correction has five parts:

- `transfer_acs_inputs` has an explicit, default-empty
  `regime_evidence_targets` selection. Only selected targets incur donor-regime
  detection, fitted-result verification, and pattern regime provenance.
- Both canonical stacked owners derive the selection internally from the
  immutable post-transfer calibration registry: two early targets and seven
  late targets. Taxable interest is absent.
- Ordinary and banked transfer records for unassigned targets carry no regimes;
  receipt builders omit QRF evidence for those targets.
- Early and late terminal validators require the complete four-field transfer
  count block for every target, reject forged QRF evidence on unassigned
  targets before record binding, and retain exact record-family binding for
  assigned targets.
- The canonical gap-fill path remains pinned to the certified eight-target fit
  width, and mixed selected/unselected families do not broaden the selected fit
  or alter unassigned draws.

The primary correction and hardening commits are:

- `22b2c6bc` — add the exact failing-first taxable-interest regression;
- `176c60fc` — scope ACS QRF evidence to calibration targets;
- `887df056` — restore strict family binding and independent legacy counts;
- `94b7aecb` — close mixed-family, count, and fit-width gaps; and
- `21a48ba5` — reject a plausible rehashed assigned `__batch_1` alias.

## Regression evidence

The exact validator regression is
`test_gap_fill_qrf_binding_excludes_unassigned_batched_targets`. In an isolated
temporary worktree at `22b2c6bc`, where that test exists but the runtime remains
unchanged from `33bf52fe`, it failed through the supplied line 4512-to-4310 path
with the same taxable-interest record-binding message. Peak observed RSS was
0.425 GiB. The temporary worktree was removed after the proof.

On current `HEAD`, the following 14 cases passed together under the owner
12 GiB/20 ms guard with a 0.564 GiB peak:

- the exact unassigned taxable-interest validator boundary;
- the real banked 15-target producer, proving the taxable record uses
  `__batch_1` but carries no regimes or QRF receipt evidence while assigned
  unemployment compensation retains both; and
- all 12 fully rehashed QRF structure mutations, including forged, plausible
  in-range, and out-of-range record-family bindings.

## Verification

The already-synced worktree virtualenv was used directly; no dependency or
lockfile changed.

The five directly affected ordinary-transfer, multispine-serialization,
stacked-spine, pool-tool, and H5 files collected 64, 5, 258, 164, and 38 cases.
All 529 passed together in one guarded process with a 1.658 GiB peak.

The repository contains exactly 6,608 collected items: 5,708 build and 900
across the four sibling packages. A monolithic run reached 83% with no test
failures before the guard correctly stopped cumulative single-process RSS at
12.170 GiB; that run is a resource non-verdict. Fresh-process shards then
covered the exact complete collection without failures:

| Shard | Result | Peak RSS |
|---|---:|---:|
| `microcosm-fit` | 93 passed | 0.724 GiB |
| `microcosm-calibrate` | 201 passed | 0.436 GiB |
| `microcosm-data` | 275 passed, 1 skipped | 11.049 GiB |
| `microcosm-frame` | 294 passed, 36 skipped | 6.488 GiB |
| build core + UK | 1,856 passed, 33 skipped | 4.182 GiB |
| build US a-r | 2,406 passed, 3 skipped | 8.844 GiB |
| build US s-z | 1,411 passed, 1 skipped | 10.579 GiB |

The shard collection union exactly equals all 6,608 repository items. The
core+UK outcome summary includes two additional passing subtest outcomes beyond
its 1,887 collected items; therefore naive addition of outcome counts exceeds
the unique collection by two.

Static verification also passed:

- repository-wide `ruff check .`;
- `ruff format --check` on all 15 Python files changed since `33bf52fe^`;
- `git diff --check 33bf52fe^..HEAD` and worktree whitespace checks; and
- exact Git-object equality for `microcosm-build/src`, `tools`, `specs`,
  `pyproject.toml`, and `uv.lock` against all-build-suite checkpoint
  `a5be536f`.

The GitNexus debugging workflow was selected. MCP graph endpoints were not
available; the local CLI built a 563-file graph but could not register it
because its hard-coded user-wide registry was outside the writable sandbox.
The generated cache was moved to the recoverable
`/private/tmp/microcosm-pkg3-two-part-gitnexus-23236043` location. Direct
raise-site, caller, producer, validator, test, and Git-history tracing plus two
independent audits reached the same diagnosis.

## Current continuation commits

- `23236043` — reopen the ACS binding continuation audit;
- `729ff466` — confirm the current scoped diagnosis;
- `b6dae1bc` — record the failing-first and current focused proof;
- `c52ccd7f` — record the guarded 529-test affected suite;
- `34de0ca7` — record lint, format, whitespace, and drift checks;
- `08900a30` — record the monolithic resource boundary;
- `014c3424` — record the green sibling-package shards;
- `f03b6477` — record the green build core+UK shard;
- `e18c3e49` — record the green build US a-r shard; and
- `d29a8705` — record the complete guarded repository suite.

## Remaining host boundary

At 2026-08-21 01:54:15 EDT the externally owned host process was still alive.
`build.log` was 577,161 bytes and `guard.log` was 8,016 bytes. The latest
completed checkpoint was the source-operator immigration status pair. The log
contained no traceback, `ValueError`, binding-invalid message, `ERROR`,
`FAILED`, or exception.

The host output still lacked:

- a `pkg3-r2 exit:` marker;
- `pool.h5`;
- `pool.manifest.json`; and
- `pool.gates.json`.

The runner began while executable content matched commit `6aeb7720`; only the
root journals differ from the current executable tree. The artifacts do not
explicitly record a Microcosm Git SHA, so this is not accepted as
revision-bound certification. Completion requires a terminal host verdict and
the final pool, manifest, and gates artifacts. Do not publish or mutate the
pending release chain as a side effect of that verification.
