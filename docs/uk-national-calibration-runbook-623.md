# UK national calibration runbook (#623)

This runbook documents the held licensed run for the first Ledger-calibrated UK
national candidate. It is not an instruction to run it in PR CI.

## Unblock conditions

Run only after one of these is true:

- WS-E spine is complete through E8 (#684) and E10 (#686).
- Maria explicitly names a different base spine for this run.

The input posture is a non-certified staging candidate at pre-clone grain,
matching the incumbent published national surface. The seam records that
posture itself and marks its artifact `shippable: false`: release certification
is the release-cut producer's job, not calibration's.

## Command shape

Calibration runs through the rowwise driver's national release role,
`tools/build_uk_rowwise_candidate.py --release-role national` (microcosm#823;
it replaced the retired `tools/calibrate_uk_national_dataset.py`). The role
delegates the build to the calibration seam library
(`uk_runtime.calibration_run.run_uk_calibration`): it is the only path that
builds the measure resolver from the input file and applies the committed
measure-exclusion register, and 187 of the activated references bind model
outputs that no frame carries — so it is the only path on which this target
surface materializes. No cloning, no ladder, national targets only. The June
builder (`tools/build_uk_national_dataset.py`) constructs the calibration
stage without either and aborts on the first unmaterializable reference; it
also rebuilds SPI income onto its input, which a spine artifact already
carries.

```bash
uv run --no-sync python tools/build_uk_rowwise_candidate.py --release-role national \
  --input-h5 data/ukds/acceptance/623-first-calibrated-candidate/input-spine.h5 \
  --input-sha256 <sha256-of-input-spine-h5> \
  --ledger-facts <ledger-consumer-artifact> \
  --ledger-facts-sha256 <sha256> \
  --ledger-manifest-sha256 <sha256> \
  --out data/ukds/acceptance/623-first-calibrated-candidate
```

The role names every output after the pinned FRS release vintage:
`microcosm_uk_2024_25.h5`, `calibration_diagnostics.json`,
`build_record.json`, `microcosm_uk_2024_25.terminal_gates.json`, the frozen
`national_target_registry.json` (the scorer's register) and
`rowwise_candidate_manifest.json`; `--dry-run` compiles the register and
prints the plan without solving. Staging telemetry and the staged bundle
follow the dense role's switches (`--staging-local-only`, `--no-staging`,
`--staging-read-back`, `--no-staged-dataset`); the run id is the attempt id.

The diagnostics digest is measured, not declared: the seam writes the
diagnostics file, hashes its actual bytes, and only then constructs and signs
the terminal gate evidence. There is no `--calibration-diagnostics-sha256` to
supply, and no way for the receipt to claim an identity the file does not have.

Solve parameters are per-run overrides of the declared doctrine. Since the
2026-09-20 ruling (microcosm#823) the doctrine is the campaign posture itself:
1,500 epochs, `family_equal`, learning rate 0.02, seed 0, so a doctrine run
passes no solve flags and records no overrides; `--target-weight-rule uniform`,
another `--epochs`, `--learning-rate` or `--target-loss-cap` is validated
through the doctrine dataclass and echoed as an explicit deviation in the
manifest, diagnostics and build record. The national role refuses
`--release-candidate` outright: its six-gate battery cannot sign
shippability, which comes only from `tools/certify_uk_release_cut.py`.

Signing the terminal gate report needs
`MICROCOSM_UK_TERMINAL_GATE_SIGNING_KEY` in the environment.

## Scoring

Rule 1 is decided by a separate tool run against the staged artifact, so run
identity never depends on the incumbent's bytes:

```bash
uv run --no-sync python tools/score_uk_national_candidate.py \
  --candidate-h5 data/ukds/acceptance/623-first-calibrated-candidate/microcosm_uk_2024_25.h5 \
  --candidate-sha256 <sha256-from-the-build-record> \
  --incumbent-h5 <enhanced-frs-2024-25-h5> \
  --incumbent-sha256 <sha256> \
  --registry-json data/ukds/acceptance/623-first-calibrated-candidate/national_target_registry.json \
  --output-json data/ukds/acceptance/623-first-calibrated-candidate/score_vs_enhanced_frs.json
```

Both artifacts are verified against the supplied digests before they are read,
and both sides are scored on the same frozen register.

A target whose measure the incumbent cannot materialize (the admin-basis UC
family measures, the CGT asset type, the ONS household type: inputs the
enhanced FRS never carried) is pruned from **both** arms and reported, never
refused: the scorer warns on stderr naming every absent measure and the rows
it removed, and the receipt carries them under
`incumbent_unresolvable_pruned` (`n_pruned`, `n_scored`, `n_surface`,
`pruned_targets` with each one's family and measure, `measures`, `families`). The score stands on the common surface
with band edges from the full register (#803). A measure the *candidate*
cannot materialize still refuses: that is a defect.
`--no-prune-incumbent-unresolvable` restores the refusal on the incumbent A measure may be pruned only under a signed, in-force entry of the reviewed incumbent-unresolvable register (`packages/microcosm-build/src/microcosm/build/uk/incumbent_unresolvable_measures.json`, keyed `entity.variable`, reviewed-exclusion schema); an unlisted measure refuses the evaluation, the receipt records the register digest and the entries used, and the certifier re-checks every pruned measure against the committed register, including each entry's window at the certification's own evaluation date. The candidate is validated on the full surface before any pruning, so an export missing a listed measure refuses rather than scoring on a reduced surface.
side too. The receipt's `evaluation` block decides rule 1 (#578) on that
surface: `verdict` is `passed` when the candidate's full loss is below the
incumbent's, `failed` otherwise, and the release-cut certifier refuses any
receipt whose verdict is not `passed` or whose surface does not close, so
publication never runs on an unpassed evaluation. The rowwise driver's
national role runs this evaluation at the end of every build it is given an
incumbent for (microcosm#965) and writes the same receipt as
`score_vs_incumbent.json`.

## Evidence directory

`data/ukds/acceptance/623-first-calibrated-candidate/` should contain:

- `microcosm_uk_2024_25.h5`
- `build_record.json`
- `microcosm_uk_2024_25.terminal_gates.json`
- `calibration_diagnostics.json`
- `national_target_registry.json`
- `national_contract_registry.json`
- `rowwise_candidate_manifest.json`
- `score_vs_incumbent.json` (the build's own when run with `--incumbent-h5`;
  the manifest's `evaluation` block records its verdict)
- `logbook-spool/` (one row for the attempt, whatever its disposition)

Acceptance follows #578: the candidate must not regress incumbent battery
observables, and the score block decides rule 1. A rule-1 loss is evidence for
#686/#736, not a threshold-edit instruction.

The calibration battery is scoped to the calibration-relevant gates; the
spine-construction and imputation gates are out of scope here and are listed in
the report as scope exclusions with their rationale. A publishable
certification combines this with the spine build's own battery, which is
release-cut work (#757).
