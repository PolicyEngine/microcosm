# UK national release assembly runbook (#806)

This runbook turns a green UK national calibration candidate into an
inspectable release without promoting it to `latest.json`. The release id is
constant across cuts (`microcosm-uk-2024-25-national`); each cut gets an
immutable tag derived from the calibration attempt id.

Do not run this sequence in PR CI. Calibration, certification, assembly, and
publication consume licensed data and operator credentials.

## Prerequisites

Before starting, export:

- `MICROCOSM_UK_TERMINAL_GATE_SIGNING_KEY`, containing the stable release key
  as base64-encoded 32 bytes. Certification signs with it and assembly uses it
  to re-verify the copied certification.
- `HF_TOKEN`, authenticated for write access to
  `policyengine/populace-uk-private`.
- `SLACK_WEBHOOK_POPULACE_UK`, for the eventual promoted-release alert. The
  inspect publication below uses `--no-latest`, so it does not announce a new
  latest release.

Keep the spine H5, its sibling `.build.json` and `.spine_gates.json`, the
Ledger consumer artifact and manifest digest, and the licensed input-mass
reference together through the run.

## 1. Calibrate the national candidate

Use the rowwise driver's national release role and record the input digest
rather than relying on a mutable path:

```bash
uv run --no-sync python tools/build_uk_rowwise_candidate.py --release-role national \
  --input-h5 <spine-h5> \
  --input-sha256 <spine-h5-sha256> \
  --ledger-facts <ledger-consumer-facts> \
  --ledger-facts-sha256 <ledger-facts-sha256> \
  --ledger-manifest-sha256 <ledger-manifest-sha256> \
  --incumbent-h5 <incumbent-h5> \
  --incumbent-sha256 <incumbent-h5-sha256> \
  --out <candidate-dir>
```

`--incumbent-h5` (with its digest; `--incumbent-label` names it, default
`enhanced_frs_2024_25`) makes the build evaluate the finished candidate
against the incumbent after the bundle is staged: `score_vs_incumbent.json`
lands beside the outputs with the rule-1 verdict, rows the incumbent cannot
materialize are pruned from both arms and named on stderr, the receipt rides
the staging telemetry as `artifacts/score_vs_incumbent.json`, and the manifest
records `evaluation` (`status`, `verdict`, `rule_1`, `scored_surface`,
`pruned_measures`). The evaluation never blocks the build: an error is
recorded as `status: error` and warned. Without the flags the manifest says
`not_requested`, and the certifier has no receipt to read until the candidate
is scored by hand.

The role's doctrine is the campaign posture (1,500 epochs, `family_equal`,
learning rate 0.02, seed 0): a certified cut passes no solve flags and records
no overrides. It writes `microcosm_uk_2024_25.h5`,
`calibration_diagnostics.json`, `build_record.json`,
`microcosm_uk_2024_25.terminal_gates.json`, `national_target_registry.json`,
`national_contract_registry.json` (the full compiled register the scoring
surface takes its band edges from) and `rowwise_candidate_manifest.json`
into `<candidate-dir>`. The build
record's id has the form
`uk-frs-calibration-attempt-<YYYYMMDDTHHMMSSZ>-<uuid8>`; assembly derives the
per-cut tag from that suffix.

## 2. Score and certify the cut

The rule-1 score receipt is the build's own when the national role was given
`--incumbent-h5` (`score_vs_incumbent.json` beside the candidate); otherwise
create it against the pinned incumbent following the scoring section of
`docs/uk-national-calibration-runbook-623.md`. Its `evaluation.verdict` must
be `passed`: the certifier refuses a receipt whose verdict is anything else,
whose scored surface does not close over its pruned rows, or which carries no
evaluation block. Then run the release-cut battery and compose the signed
certification:

```bash
uv run --no-sync python tools/certify_uk_release_cut.py \
  --candidate-h5 <candidate-dir>/microcosm_uk_2024_25.h5 \
  --candidate-sha256 <candidate-sha256-from-build-record> \
  --candidate-name microcosm_uk_2024_25 \
  --spine-h5 <spine-h5> \
  --spine-sha256 <spine-h5-sha256> \
  --diagnostics-json <candidate-dir>/calibration_diagnostics.json \
  --build-record-json <candidate-dir>/build_record.json \
  --seam-gate-report <candidate-dir>/microcosm_uk_2024_25.terminal_gates.json \
  --ledger-facts <ledger-consumer-facts> \
  --ledger-facts-sha256 <ledger-facts-sha256> \
  --ledger-manifest-sha256 <ledger-manifest-sha256> \
  --input-mass-reference <licensed-input-mass-reference> \
  --score-receipt <candidate-dir>/score_vs_incumbent.json \
  --release-id microcosm-uk-2024-25-national
```

The spine is stage evidence for the family build-state gates, so `--spine-sha256` must be the parent the calibration recorded (`build_record.input_posture.sha256`, `source_pins.input_h5.sha256`, the signed diagnostics' `build.input_posture`); a spine that pins correctly but is not that parent is refused before any gate runs, and the certification records it as `parent_spine`.

With the default paths, this writes
`microcosm_uk_2024_25.release_cut_gates.json` and
`microcosm_uk_2024_25.release_certification.json` next to the candidate. Continue
only when the certification says `shippable: true`.

## 3. Assemble the release directory

Assembly verifies the complete hash join before writing, mints the calibration
NPZ from the candidate and spine weights, copies signed evidence byte-for-byte,
and validates the finished directory:

```bash
uv run --no-sync python tools/assemble_uk_release_dir.py \
  --candidate-h5 <candidate-dir>/microcosm_uk_2024_25.h5 \
  --spine-h5 <spine-h5> \
  --certification-json <candidate-dir>/microcosm_uk_2024_25.release_certification.json \
  --build-record-json <candidate-dir>/build_record.json \
  --diagnostics-json <candidate-dir>/calibration_diagnostics.json \
  --seam-gate-report <candidate-dir>/microcosm_uk_2024_25.terminal_gates.json \
  --release-cut-gate-json <candidate-dir>/microcosm_uk_2024_25.release_cut_gates.json \
  --score-receipt <candidate-dir>/score_vs_incumbent.json \
  --out-dir releases
```

The output is
`releases/microcosm-uk-2024-25-national/`. The JSON summary records every
digest, the derived cut tag, and the exact publication command. Use
`--cut-tag microcosm-uk-2024-25-national-<YYYYMMDDTHHMMSSZ>-<uuid8>` only to
override the derived tag deliberately; the override must keep that grammar,
which the contract validates on every artifact revision.

Assembly stages into a private directory, validates there, and atomically
renames into empty destinations: re-assembling a cut requires removing the
previous `releases/microcosm-uk-2024-25-national/` directory and the
previously minted calibration NPZ first. Release identity — the attempt id,
spine digest, and every runtime pin — comes only from the signed diagnostics
build block; `--runtime-version PACKAGE=VERSION` may re-assert a signed value
as an operator cross-check but refuses to replace one.

## 4. Publish for inspection

Run the assembler's `publish_command` first. Its shape is:

```bash
uv run python -m microcosm.data.publish_cli \
  releases/microcosm-uk-2024-25-national \
  --repo-id policyengine/populace-uk-private \
  --artifact-root <candidate-dir> \
  --no-latest \
  --tag-name microcosm-uk-2024-25-national-<timestamp>-<uuid8>
```

The assembler also prints this `promote_command`; keep it for §6 and run it
only after the cut passes the review in §5:

```bash
uv run python -m microcosm.data.publish_cli \
  releases/microcosm-uk-2024-25-national \
  --repo-id policyengine/populace-uk-private \
  --artifact-root <candidate-dir> \
  --promote-line national \
  --tag-name microcosm-uk-2024-25-national-<timestamp>-<uuid8>
```

Do not omit `--tag-name`, and do not pass `--no-create-tag`: every artifact in
the manifest is pinned to that immutable per-cut tag. `--no-latest` is
mandatory for the inspect publication. The promotion command replaces it with
`--promote-line national`; the CLI requires that flag to accompany
`--tag-name` and refuses to combine it with `--no-latest`. A per-cut tag without
either flag cannot move the repository-global `latest.json`.

If tag creation returns HTTP 409 after the staging commit, publication can
leave the constant branch
`release-staging/microcosm-uk-2024-25-national` behind. Delete that branch
manually in the private Hugging Face repository before retrying the same cut.
Do not delete the immutable cut tag.

## 5. Inspect on the dashboard

Open the calibration-diagnostics dashboard with:

```text
?country=uk&release=microcosm-uk-2024-25-national
```

Adjudicate the release using the copied certification, scoped gate reports,
calibration diagnostics, and score receipt. The release remains inspect-only
until that review is complete.

## 6. Promote the reviewed cut

After completing the review in §5, run the assembler's `promote_command` shown
in §4. The promotion moves only `latest-national.json`, recording `national` as
the line and the reviewed cut tag as its `revision`. It never moves the
repository-global `latest.json`, which remains on the June 2023 release.

After promotion, `microcosm.data.resolve("uk")` lands on the 2025 registry
entry, whose certified loader follows `latest-national.json` to the reviewed
revision. An explicit `microcosm.data.load("uk", 2023)` continues to follow
`latest.json`.

With `SLACK_WEBHOOK_POPULACE_UK` configured, successful promotion sends an
alert naming the `national` line and the promoted revision.
