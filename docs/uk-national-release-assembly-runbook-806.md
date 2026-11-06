# UK national release assembly runbook (#806)

This runbook turns a green UK national calibration candidate into a
published build of the national release, `microcosm-uk-2024-25-national`,
and then, after review, makes it the release's current build. Each build is
published once into its own folder,
`releases/microcosm-uk-2024-25-national/<build>/`, through the emitter, and
the repository's `releases.json` records which build is current. The build is
the calibration attempt id; nothing is tagged.

Do not run this sequence in PR CI. Calibration, certification, assembly, and
publication consume licensed data and operator credentials.

## Prerequisites

Before starting, export:

- `MICROCOSM_UK_TERMINAL_GATE_SIGNING_KEY`, containing the stable release key
  as base64-encoded 32 bytes. Certification signs with it and assembly uses it
  to re-verify the copied certification.
- `HF_TOKEN` (or `HUGGINGFACE_TOKEN`, or a token saved by `hf auth login`),
  with write access to `policyengine/populace-uk-private`. The publication
  command starts the separately installed `microcosm-emitter` (0.11.9 or
  later) as its child process, and the emitter uses the token it inherits.

Publication sends no Slack alert. Alerts for new and current builds will come
from the dashboard, which compares `releases.json` before and after each
change; until that is in place, announce a build by hand.

Keep the spine H5, its sibling `.build.json` and `.spine_gates.json`, the
Ledger consumer artifact and manifest digest, and the licensed input-mass
reference together through the run.

## 1. Calibrate the national candidate

Use the UK build driver's national release role (`microcosm-build-uk`,
`tools/build_uk_full.py`; `tools/build_uk_rowwise_candidate.py` is a stub
over the same driver, and the role dispatches to the retained calibration
seam) and record the input digest rather than relying on a mutable path:

```bash
uv run --no-sync python tools/build_uk_full.py --release-role national \
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
`uk-frs-calibration-attempt-<YYYYMMDDTHHMMSSZ>-<uuid8>`; it is the build's
name in both release manifests and in its folder on the Hub.

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

**Interim check until microcosm#1276 lands.** microcosm#1281 removed the QRF
tail-concentration gate, and nothing release-blocking yet checks the SPI
stage-1 columns (`hmrc_spi_*` and the other stage-1 outputs). Before assembly,
measure the candidate and the currently promoted cut together:

```bash
uv run --no-sync python tools/measure_uk_weighted_integrity_baselines.py \
  --h5 candidate=<candidate-dir>/microcosm_uk_2024_25.h5 \
  --h5 previous=<promoted-cut-h5> \
  --output <candidate-dir>/stage1_comparison.json
```

Compare each stage-1 column's weighted total (`input_mass_totals`) and carrier
count (`qrf_tail`) between the two, and explain any column that moves by more
than half in either direction before continuing. microcosm#1256 cut
`hmrc_spi_incapacity_benefit_income` by about six-sevenths, and only this kind
of comparison would have shown it. The file is licensed output: keep it
local, out of the release directory and out of any public text. The tool
suppresses carrier counts below 10.

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
`releases/microcosm-uk-2024-25-national/`. Both manifests name the release
and the build (`release` and `build`), the release manifest records
`built_at`, `built_with_model_package` and `built_with_core_package` at its
top level and each file's `role` (`dataset`, `calibration_weights`,
`calibration_diagnostics`, and each evidence file's key), and no artifact
pins a revision. The JSON summary records every digest, the release and the
build, and the exact publication and promotion commands.

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
uv run microcosm-publish-release-build \
  releases/microcosm-uk-2024-25-national \
  --repo-id policyengine/populace-uk-private \
  --artifact-root <candidate-dir>
```

It refuses at once, before it starts an emitter, when `HF_HUB_OFFLINE` or
`TRANSFORMERS_OFFLINE` is set, since the emitter inherits them and could not
reach the Hub. It checks the build against the release contract again,
starts the emitter, and hands nothing over unless the emitter's write-access
check finds a credential that can write the repository. It then hands every
file to the emitter with its role and SHA-256. The emitter writes the build's
folder and `releases.json` in one commit, listing the build without making
it current. The command waits for the emitter's answer and prints its
receipt: `status` is `published`, `pending` (the emitter is still trying, and
keeps the change queued for an hour), `refused` with an `error_code`, or
`unavailable` (no emitter answered).
Running the same command again is safe: a build already published with the
same release manifest is reported `published`, and one published with a
different manifest is refused (`BUILD_FOLDER_EXISTS`), since a build is never
changed after publishing. `--preflight-only` runs the checks and starts no
emitter.

The assembler also prints this `promote_command`; keep it for §6 and run it
only after the build passes the review in §5:

```bash
uv run microcosm-publish-release-build \
  releases/microcosm-uk-2024-25-national \
  --repo-id policyengine/populace-uk-private \
  --promote
```

## 5. Inspect on the dashboard

Open the calibration-diagnostics dashboard with:

```text
?country=uk&release=microcosm-uk-2024-25-national
```

Adjudicate the release using the copied certification, scoped gate reports,
calibration diagnostics, and score receipt. The release remains inspect-only
until that review is complete.

The dashboard shows a build published into its own folder only once it reads
`releases.json`; until then, review the assembled directory's files.

## 6. Make the reviewed build current

After completing the review in §5, run the assembler's `promote_command` shown
in §4. The emitter sets the national release's `current` in `releases.json`
to this build. The command succeeds only when the receipt is `published` and
names this build as `current`.

While the repository still holds the older `latest*.json` pointer files, the
emitter refuses every change to a release's current build
(`POINTER_FILES_PRESENT`), so that the pointer files and `releases.json`
cannot disagree. Until those files are deleted, a build can be published but
not made current: the promotion is refused at once, changes nothing and says
to run it again, and the published build stays listed. Run `promote_command`
again once they are deleted.

The 2025 national release is the default UK load (microcosm#823):
`microcosm.data.load("uk")` and `microcosm.data.load("uk", 2025)` read its
current build from `releases.json`, and `load(..., build=<build>)` loads a
listed build instead. An explicit `microcosm.data.load("uk", 2023)`
continues to follow `latest.json`.
