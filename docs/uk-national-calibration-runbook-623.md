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

Calibration runs through `tools/calibrate_uk_national_dataset.py`. That driver
is the only one that builds the measure resolver and applies the committed
measure-exclusion register, and 187 of the activated references bind model
outputs that no frame carries — so it is the only path on which this target
surface materializes. The June builder
(`tools/build_uk_national_dataset.py`) constructs the calibration stage without
either and aborts on the first unmaterializable reference; it also rebuilds SPI
income onto its input, which a spine artifact already carries.

```bash
uv run --no-sync python tools/calibrate_uk_national_dataset.py \
  --input-h5 data/ukds/acceptance/623-first-calibrated-candidate/input-spine.h5 \
  --input-sha256 <sha256-of-input-spine-h5> \
  --ledger-facts <ledger-consumer-artifact> \
  --ledger-facts-sha256 <sha256> \
  --ledger-manifest-sha256 <sha256> \
  --staging-h5 data/ukds/acceptance/623-first-calibrated-candidate/populace_uk_2023.h5 \
  --diagnostics-json data/ukds/acceptance/623-first-calibrated-candidate/calibration_diagnostics.json \
  --build-record-json data/ukds/acceptance/623-first-calibrated-candidate/build_record.json \
  --terminal-gate-json data/ukds/acceptance/623-first-calibrated-candidate/terminal_gates.json \
  --release-id dev-623-first-calibrated-candidate
```

The diagnostics digest is measured, not declared: the seam writes the
diagnostics file, hashes its actual bytes, and only then constructs and signs
the terminal gate evidence. There is no `--calibration-diagnostics-sha256` to
supply, and no way for the receipt to claim an identity the file does not have.

Solve parameters are per-run overrides, not defaults. The campaign settings are
`--epochs 1500 --target-weight-rule family_equal`; each override is validated
through the doctrine dataclass and echoed as an explicit deviation in the
manifest, diagnostics and build record. `--release-candidate` refuses every
override flag, and refuses `--measure-exclusions`, so a release candidate is
always solved under declared doctrine against the committed target surface.

Signing the terminal gate report needs
`MICROCOSM_UK_TERMINAL_GATE_SIGNING_KEY` in the environment.

## Scoring

Rule 1 is decided by a separate tool run against the staged artifact, so run
identity never depends on the incumbent's bytes:

```bash
uv run --no-sync python tools/score_uk_national_candidate.py \
  --candidate-h5 data/ukds/acceptance/623-first-calibrated-candidate/populace_uk_2023.h5 \
  --candidate-sha256 <sha256-from-the-build-record> \
  --incumbent-h5 <enhanced-frs-2024-25-h5> \
  --incumbent-sha256 <sha256> \
  --registry-json data/ukds/acceptance/623-first-calibrated-candidate/frozen-register.json \
  --output-json data/ukds/acceptance/623-first-calibrated-candidate/score_vs_enhanced_frs.json
```

Both artifacts are verified against the supplied digests before they are read,
and both sides are scored on the same frozen register.

## Evidence directory

`data/ukds/acceptance/623-first-calibrated-candidate/` should contain:

- `build_record.json`
- `terminal_gates.json`
- `calibration_diagnostics.json`
- `score_vs_enhanced_frs.json`
- `logbook-spool/` (one row for the attempt, whatever its disposition)

Acceptance follows #578: the candidate must not regress incumbent battery
observables, and the score block decides rule 1. A rule-1 loss is evidence for
#686/#736, not a threshold-edit instruction.

The calibration battery is scoped to the calibration-relevant gates; the
spine-construction and imputation gates are out of scope here and are listed in
the report as scope exclusions with their rationale. A publishable
certification combines this with the spine build's own battery, which is
release-cut work (#757).
