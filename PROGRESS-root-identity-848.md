# PROGRESS — microcosm#848 root identity (branch `root-identity-848`)

Lane journal for hash-pinning every raw microdata input and referencing its
Chronicle registration. (The root `PROGRESS.md` belongs to the ACS
predictor-release-join lane; this file follows the `PROGRESS-<lane>.md`
convention already used by `PROGRESS-graph-acceptance.md`.) Historical once
merged — see CLAUDE.md "Root journals are history, not state".

## State

Started 2026-09-02 from origin/main `d2b31496`. Investigation phase.

## Done

- Read CLAUDE.md, DESIGN.md, issue #848, chronicle#221, and the Chronicle
  raw-microdata identity ADR (chronicle `origin/adr-raw-microdata-identity`).
- Inventoried every `*_microdata` artifact entry across
  `build/{am,be,uk,us}/source_stages.json` and
  `build/uk/hmrc_income_source_stages.json`.
- Confirmed Chronicle's live naming from `~/PolicyEngine/chronicle/db/data`:
  `source_id` is a publisher slug, `package_id` is kebab-case and
  publisher-prefixed (e.g. `hmrc` / `hmrc-spi-income-bands-2023-24`).

## Next

- Add `chronicle_artifact` validation to `source_manifest.py`.
- Add the fail-closed sha256 gate to the source runtime.
- Populate pins; write the pending allowlists; contract tests.
