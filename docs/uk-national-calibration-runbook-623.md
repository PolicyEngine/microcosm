# UK national calibration runbook (#623)

This runbook documents the held licensed run for the first Ledger-calibrated UK
national candidate. It is not an instruction to run it in PR CI.

## Unblock conditions

Run only after one of these is true:

- WS-E spine is complete through E8 (#684) and E10 (#686).
- Maria explicitly names a different base spine for this run.

The input posture is a non-certified staging candidate at pre-clone grain,
matching the incumbent published national surface. The staging input must be
SHA-pinned with `--staging-candidate-input-sha256`; release-candidate posture
continues to require the certified input path.

## Command shape

```bash
uv run --no-sync python tools/build_uk_national_dataset.py \
  --input-h5 data/ukds/acceptance/623-first-calibrated-candidate/input-spine.h5 \
  --staging-h5 data/ukds/acceptance/623-first-calibrated-candidate/populace_uk_2023.h5 \
  --staging-candidate-input-sha256 <sha256> \
  --release-id populace-uk-2023-frs-k535080 \
  --calibration-diagnostics-sha256 <sha256-of-final-diagnostics-json> \
  --national-calibration-diagnostics-json data/ukds/acceptance/623-first-calibrated-candidate/calibration_diagnostics.json \
  --frs-raw-dir <licensed-frs-2024-25-dir> \
  --spi-tab <licensed-spi-tab> \
  --hmrc-ods <hmrc-personal-incomes-ods> \
  --cgt-ods <hmrc-cgt-table-3-ods> \
  --ledger-facts <ledger-consumer-artifact> \
  --ledger-facts-sha256 <sha256> \
  --ledger-manifest-sha256 <sha256> \
  --terminal-gates-json data/ukds/acceptance/623-first-calibrated-candidate/terminal_gates.json \
  --build-record-json data/ukds/acceptance/623-first-calibrated-candidate/build_record.json
```

## Evidence directory

`data/ukds/acceptance/623-first-calibrated-candidate/` should contain:

- `build_record.json`
- `terminal_gates.json`
- `calibration_diagnostics.json`
- calibration diagnostics SHA-256 receipt
- `score_vs_enhanced_frs.json`
- twin-build payload identity receipt

Acceptance follows #578: the candidate must not regress incumbent battery
observables, and the score block decides rule 1. A rule-1 loss is evidence for
#686/#736, not a threshold-edit instruction.
