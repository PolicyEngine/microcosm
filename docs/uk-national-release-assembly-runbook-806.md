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

Use the national calibration driver and record the input digest rather than
relying on a mutable path:

```bash
uv run --no-sync python tools/calibrate_uk_national_dataset.py \
  --input-h5 <spine-h5> \
  --input-sha256 <spine-h5-sha256> \
  --ledger-facts <ledger-consumer-facts> \
  --ledger-facts-sha256 <ledger-facts-sha256> \
  --ledger-manifest-sha256 <ledger-manifest-sha256> \
  --staging-h5 <candidate-dir>/microcosm_uk_2024.h5 \
  --diagnostics-json <candidate-dir>/calibration_diagnostics.json \
  --build-record-json <candidate-dir>/build_record.json \
  --terminal-gate-json <candidate-dir>/terminal_gates.json \
  --release-id dev-uk-national-calibration
```

Use only the campaign doctrine overrides that were separately adjudicated.
The build record's id has the form
`uk-frs-calibration-attempt-<YYYYMMDDTHHMMSSZ>-<uuid8>`; assembly derives the
per-cut tag from that suffix.

## 2. Score and certify the cut

First create the rule-1 score receipt against the pinned incumbent, following
the scoring section of
`docs/uk-national-calibration-runbook-623.md`. Then run the release-cut battery
and compose the signed certification:

```bash
uv run --no-sync python tools/certify_uk_release_cut.py \
  --candidate-h5 <candidate-dir>/microcosm_uk_2024.h5 \
  --candidate-sha256 <candidate-sha256-from-build-record> \
  --candidate-name microcosm_uk_2024 \
  --spine-h5 <spine-h5> \
  --diagnostics-json <candidate-dir>/calibration_diagnostics.json \
  --build-record-json <candidate-dir>/build_record.json \
  --seam-gate-report <candidate-dir>/terminal_gates.json \
  --ledger-facts <ledger-consumer-facts> \
  --ledger-facts-sha256 <ledger-facts-sha256> \
  --ledger-manifest-sha256 <ledger-manifest-sha256> \
  --input-mass-reference <licensed-input-mass-reference> \
  --score-receipt <candidate-dir>/score_vs_enhanced_frs.json \
  --release-id microcosm-uk-2024-25-national
```

With the default paths, this writes
`microcosm_uk_2024.release_cut_gates.json` and
`microcosm_uk_2024.release_certification.json` next to the candidate. Continue
only when the certification says `shippable: true`.

## 3. Assemble the release directory

Assembly verifies the complete hash join before writing, mints the calibration
NPZ from the candidate and spine weights, copies signed evidence byte-for-byte,
and validates the finished directory:

```bash
uv run --no-sync python tools/assemble_uk_release_dir.py \
  --candidate-h5 <candidate-dir>/microcosm_uk_2024.h5 \
  --spine-h5 <spine-h5> \
  --certification-json <candidate-dir>/microcosm_uk_2024.release_certification.json \
  --build-record-json <candidate-dir>/build_record.json \
  --diagnostics-json <candidate-dir>/calibration_diagnostics.json \
  --seam-gate-report <candidate-dir>/terminal_gates.json \
  --release-cut-gate-json <candidate-dir>/microcosm_uk_2024.release_cut_gates.json \
  --score-receipt <candidate-dir>/score_vs_enhanced_frs.json \
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

Run the command printed by the assembler. Its shape is:

```bash
uv run python -m microcosm.data.publish_cli \
  releases/microcosm-uk-2024-25-national \
  --repo-id policyengine/populace-uk-private \
  --artifact-root <candidate-dir> \
  --no-latest \
  --tag-name microcosm-uk-2024-25-national-<timestamp>-<uuid8>
```

Do not omit `--tag-name`, and do not pass `--no-create-tag`: every artifact in
the manifest is pinned to that immutable per-cut tag. `--no-latest` is
mandatory for this inspect lane — and enforced: publication refuses to move
`latest.json` for any tag that is not the release id itself, so omitting the
flag fails closed instead of promoting an inspect cut.

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

## Promotion is a separate change

Do not write `latest.json` for this line yet. The current pointer cannot name a
per-cut tag, while the certified loader expects the artifact revision to equal
the release id and fetches the manifest from a tag named by that id. Promotion
needs the loader/pointer design tracked in microcosm#823 before a reviewed cut
can become the default; publication enforces this by refusing a pointer move
for any per-cut tag. Until then, publish every national cut with `--no-latest`
and its explicit per-cut tag.
