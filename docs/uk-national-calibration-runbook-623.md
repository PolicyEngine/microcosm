# UK national calibration runbook (#623)

This runbook documents the held licensed run for the first Ledger-calibrated UK
national candidate. It is not an instruction to run it in PR CI.

## Unblock conditions

Run only after one of these is true:

- WS-E spine is complete through E8 (#684) and E10 (#686).
- Maria explicitly names a different base spine for this run.

The input posture is a non-certified staging candidate at pre-clone grain,
matching the published national surface. The seam records that
posture itself and marks its artifact `shippable: false`: release certification
is the release-cut producer's job, not calibration's.

## Command shape

Calibration runs through the UK build driver's national release role,
`microcosm-build-uk --release-role national` (`tools/build_uk_full.py`;
`tools/build_uk_rowwise_candidate.py` is a stub over the same driver). The
role arrived with microcosm#823, which retired
`tools/calibrate_uk_national_dataset.py`, moved into the graph driver with
microcosm#901 and onto the graph itself with the change that followed it.
The driver validates the request with its posture-aware validator and
composes the national graph (`uk_runtime.graph_national`): the bound
checkpoint, the national register compiled from the pinned Chronicle
artifact with the committed measure-exclusion register applied, the engine
measures resolved on the bound frame (187 of the activated references bind
model outputs that no frame carries, so the resolver is built from the input
frame, never skipped), the one ordered national problem, the shared solve
nodes under the national doctrine and the calibration-seam gate scope. The
seam-shaped evidence (the build record, the signed gate report, the seam
diagnostics, the registries, the manifest) is materialised from the stored
graph artifacts. No cloning, no ladder, national targets only; the role
builds from a bound `--input-h5` and refuses a `--spine-request`. The
retired June builder constructed the calibration stage without either and
aborted on the first unmaterializable reference; it also rebuilt SPI income
onto its input, which a spine artifact already carries.

```bash
uv run --no-sync python tools/build_uk_full.py --release-role national \
  --input-h5 data/ukds/acceptance/623-first-calibrated-candidate/input-spine.h5 \
  --input-sha256 <sha256-of-input-spine-h5> \
  --ledger-facts <ledger-consumer-artifact> \
  --ledger-facts-sha256 <sha256> \
  --ledger-manifest-sha256 <sha256> \
  --out data/ukds/acceptance/623-first-calibrated-candidate
```

`--out` must not already hold a candidate: the role refuses an occupied
output directory with the other argument refusals, before the attempt opens
(a `FileExistsError`; nothing is written into the directory, no Logbook row,
no staging run), so a rebuild goes into a fresh directory rather than over
the bytes an earlier row names. An
operator interrupt (Ctrl-C) records a `discarded` row and closes the staging
run as failed. `--review-date` (default today) dates the measure-exclusion
windows and the target-fit deferral register on this line, as it does on
the dense line; the retired seam used the run clock, so a back-dated value
keeps expired exclusions and deferrals in force and is only for replaying a
dated review. The date is a parameter of the `uk.full.national_targets` and
`uk.full.gates.calibrated` nodes (`build.graph.json`).

The role names every output after the pinned FRS release vintage:
`microcosm_uk_2024_25.h5`, `calibration_diagnostics.json`,
`build_record.json`, `microcosm_uk_2024_25.terminal_gates.json`, the frozen
`national_target_registry.json` (the scorer's register) and
`rowwise_candidate_manifest.json`; `--dry-run` compiles the register and
prints the plan without solving. Staging telemetry and the staged bundle
follow the dense role's switches (`--staging-local-only`, `--no-staging`,
`--staging-read-back`, `--no-staged-dataset`); the run id is the attempt id.

The diagnostics digest is measured, not declared: the driver writes the
diagnostics file through the seam writer, hashes its actual bytes, and only
then replays and signs the terminal gate evidence. There is no `--calibration-diagnostics-sha256` to
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

The candidate is compared with the current build of the national line, the
cut's reference (microcosm#1095; `tools/resolve_uk_reference.py` resolves it
and writes its record). The comparison is a separate tool run against the
staged artifact, so run identity never depends on the reference's bytes:

```bash
uv run --no-sync python tools/score_uk_national_candidate.py \
  --candidate-h5 data/ukds/acceptance/623-first-calibrated-candidate/microcosm_uk_2024_25.h5 \
  --candidate-sha256 <sha256-from-the-build-record> \
  --reference-h5 <reference-dir>/microcosm_uk_2024_25.h5 \
  --reference-record <reference-dir>/reference.json \
  --registry-json data/ukds/acceptance/623-first-calibrated-candidate/national_target_registry.json \
  --output-json data/ukds/acceptance/623-first-calibrated-candidate/score_vs_reference.json
```

Both artifacts are verified against their digests before they are read, and
both sides are scored on the same frozen register.

A target whose measure the reference cannot materialize (an input a newer
build added, for example) is pruned from **both** arms and reported, never
refused: the scorer warns on stderr naming every absent measure and the rows
it removed, and the receipt carries them under
`reference_unresolvable_pruned` (`n_pruned`, `n_scored`, `n_surface`,
`pruned_targets` with each one's family and measure, `measures`, `families`).
The comparison stands on the common surface with band edges from the full
register (#803). A measure the *candidate* cannot materialize still refuses:
that is a defect. `--no-prune-reference-unresolvable` restores the refusal on
the reference side too.

A measure may be pruned only under a signed, in-force entry for this
reference in the reviewed register
(`packages/microcosm-build/src/microcosm/build/uk/reference_unresolvable_measures.json`,
keyed by the reference's `<release>@<build>` label, then `entity.variable`,
reviewed-exclusion schema). An unlisted measure refuses the comparison; the
receipt records the register digest and the entries used, and the certifier
re-checks every pruned measure against the committed register, including
each entry's window at the certification's own evaluation date. The
candidate is validated on the full surface before any pruning, so an export
missing a listed measure refuses rather than scoring on a reduced surface.

The receipt is a report with no verdict: its `evaluation` block records each
side's loss and how many targets each fits more closely, with
`no_threshold_applied`. The release-cut certifier requires it to name the
candidate and the cut's reference and its surface to close; whether the
comparison is good enough is the owner's call. The build driver's national
role runs this comparison at the end of every build it is given a
`--reference-h5` and `--reference-record` for, and writes the same receipt
as `score_vs_reference.json`. Until microcosm#1095 the comparison was with
the enhanced FRS and decided rule 1 (#578): the candidate's full loss had to
be below the incumbent's.

## Evidence directory

`data/ukds/acceptance/623-first-calibrated-candidate/` should contain:

- `microcosm_uk_2024_25.h5`
- `build_record.json`
- `microcosm_uk_2024_25.terminal_gates.json`
- `calibration_diagnostics.json`
- `national_target_registry.json`
- `national_contract_registry.json`
- `rowwise_candidate_manifest.json`
- `score_vs_reference.json` (the build's own when run with `--reference-h5`;
  the manifest's `evaluation` block summarizes it)
- `logbook-spool/` (one row for the attempt, whatever its disposition)

The comparison with the reference informs acceptance and decides nothing
by itself (microcosm#1095). A candidate that fits worse than the reference is
evidence for #686/#736, not a threshold-edit instruction.

The calibration battery is scoped to the calibration-relevant gates; the
spine-construction and imputation gates are out of scope here and are listed in
the report as scope exclusions with their rationale. A publishable
certification combines this with the spine build's own battery, which is
release-cut work (#757).
