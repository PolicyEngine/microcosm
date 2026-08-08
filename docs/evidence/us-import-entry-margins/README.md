# US import-entry margins — committed build evidence (#615 P1)

Immutable evidence from the P1 ingest run over the full 18-month window
(2025-01 … 2026-06), committed so the PR's reconciliation and provenance
claims are reviewable and recomputable in-branch rather than prose:

- `build_report.json` — the atomic publication's report: window, row and
  dimension counts, per-artifact sha256s (margins/totals/district/detail
  parquets, consumer facts + manifest, reconciliation evidence files,
  source manifest), zero reconciliation failures.
- `source_manifest.jsonl` — one retrieval row per source byte-stream: the
  18 IMDB archives (URL, sha256, size, download-manifest retrieval
  timestamps) and the archived CBP statistics page.
- `reconciliation/period=YYYY-MM.json` — the machine-readable record of
  every reconciliation comparison run for each month: per axis
  (country / commodity / district-of-entry) key-set sizes, duplicate-key
  verdicts, and per-measure compared/matched cell counts with both sides'
  integer totals.
- `crosscheck_api_report.json` — the independent Census International
  Trade API cross-check over the stratified (month, chapter) sample,
  including both parquet inputs' sha256s and every pair's fetched API
  retrieval manifest.

Chain of custody: the archive sha256s here equal the sha256s in the
committed golden pack manifest
(`packages/microcosm-build/tests/golden/us_trade/imdb/golden_manifest.json`)
for the two golden months, and the `build_report.json` sha256s equal the
ones quoted in PR #620's evidence section. The heavyweight artifacts
themselves (3.3 GB of archives, ~1 GB of parquet) live outside the repo;
every byte of them is pinned here.
